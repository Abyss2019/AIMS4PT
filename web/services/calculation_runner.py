"""Service-layer calculation runner for the four AIMS4PT web actions."""

from __future__ import annotations

import asyncio
import gc
import logging
import re
import threading
import time
from dataclasses import dataclass

from aims4pt.reporting.excel import build_report_payload
from web.config import settings
from web.services.input_validation import CPX_OXIDE_COLUMNS, LIQUID_OXIDE_COLUMNS
from web.services.memory import log_memory
from web.services.report_builder import write_payload_to_bytes
from web.services.session_store import CalculationResult, SessionData

logger = logging.getLogger("web.calculation")


@dataclass(frozen=True)
class CalculationSpec:
    key: str
    label: str
    target: str
    method_type: str
    requires_liquid: bool
    memory_suffix: str


CALCULATION_SPECS = {
    "cpx_only_P": CalculationSpec(
        key="cpx_only_P",
        label="Cpx-only barometers",
        target="P",
        method_type="cpx_only",
        requires_liquid=False,
        memory_suffix="cpx-only-P",
    ),
    "cpx_only_T": CalculationSpec(
        key="cpx_only_T",
        label="Cpx-only thermometers",
        target="T",
        method_type="cpx_only",
        requires_liquid=False,
        memory_suffix="cpx-only-T",
    ),
    "cpx_liq_P": CalculationSpec(
        key="cpx_liq_P",
        label="Cpx-liquid barometers",
        target="P",
        method_type="cpx_liq",
        requires_liquid=True,
        memory_suffix="cpx-liquid-P",
    ),
    "cpx_liq_T": CalculationSpec(
        key="cpx_liq_T",
        label="Cpx-liquid thermometers",
        target="T",
        method_type="cpx_liq",
        requires_liquid=True,
        memory_suffix="cpx-liquid-T",
    ),
}

calculation_semaphore = asyncio.Semaphore(settings.max_concurrent_calculations)
_registry_loaded = False
_registry_lock = threading.Lock()
_r_backend_available: bool | None = None
_r_backend_lock = threading.Lock()
R_BACKED_MODEL_MODULE_MARKERS = ("Higgins21", "Jorgenson22")
TENSORFLOW_MODEL_MODULE_MARKERS = ("Chicchi23",)


def ensure_model_registry_loaded() -> dict[str, str]:
    """Import model modules once so the registry can instantiate model pools."""
    global _registry_loaded
    with _registry_lock:
        if _registry_loaded:
            return {}

        from aims4pt.model_tools.model_registry import import_all_models

        imported, failed = import_all_models(skip_failed=True)
        _registry_loaded = True
        logger.info(
            "model_registry_loaded imported=%s failed=%s", len(imported), len(failed)
        )
        if failed:
            logger.warning("model_registry_import_failures modules=%s", list(failed))
        return failed


def parse_melt_tas_fields(raw_value: str | None) -> list[str] | None:
    """Parse comma, semicolon, or newline separated TAS fields."""
    if raw_value is None:
        return None
    fields = [
        item.strip()
        for item in re.split(r"[,;\n\r]+", raw_value)
        if item.strip()
    ]
    return list(dict.fromkeys(fields)) or None


def get_calculation_spec(calculation_key: str) -> CalculationSpec:
    """Return the supported calculation spec or raise a clear error."""
    try:
        return CALCULATION_SPECS[calculation_key]
    except KeyError as exc:
        raise ValueError("Unsupported calculation type.") from exc


def initialize_model_pools_for_session(session: SessionData) -> None:
    """Create and cache the model objects needed by a validated session."""
    if not session.validation.is_valid:
        return

    started_at = time.perf_counter()
    ensure_model_registry_loaded()

    from aims4pt.model_tools.model_registry import get_models_initial_pools

    specs = [
        spec
        for spec in CALCULATION_SPECS.values()
        if not spec.requires_liquid or session.has_liquid
    ]
    log_memory("before-model-pool-initialization")
    for spec in specs:
        if spec.key in session.model_pools:
            continue
        try:
            model_pool = get_models_initial_pools(
                spec.target, spec.method_type, if_hydrous=False
            )
            model_pool, environment_skipped_models = _filter_default_disabled_models(
                model_pool
            )
            session.model_pools[spec.key] = model_pool
            session.environment_skipped_models[spec.key] = environment_skipped_models
            logger.info(
                "model_pool_initialized session_id=%s calculation=%s models=%s environment_skipped=%s",
                session.session_id,
                spec.key,
                len(model_pool),
                len(environment_skipped_models),
            )
        except Exception as exc:
            session.model_pools[spec.key] = []
            session.model_pool_errors[spec.key] = str(exc)
            logger.exception(
                "model_pool_initialization_failed session_id=%s calculation=%s error_type=%s",
                session.session_id,
                spec.key,
                type(exc).__name__,
            )
    elapsed_seconds = time.perf_counter() - started_at
    log_memory("after-model-pool-initialization")
    logger.info(
        "model_pool_initialization_completed session_id=%s calculations=%s elapsed_seconds=%.2f",
        session.session_id,
        len(specs),
        elapsed_seconds,
    )


async def run_calculation_async(
    session: SessionData,
    calculation_key: str,
    melt_tas_fields: str | None = None,
) -> CalculationResult:
    """Run one calculation under the configured concurrency limit."""
    async with calculation_semaphore:
        if settings.enable_r_models:
            return run_calculation(session, calculation_key, melt_tas_fields)
        return await asyncio.to_thread(
            run_calculation, session, calculation_key, melt_tas_fields
        )


def run_calculation(
    session: SessionData,
    calculation_key: str,
    melt_tas_fields: str | None = None,
) -> CalculationResult:
    """Run one completed AIMS4PT workflow and build its report payload."""
    spec = get_calculation_spec(calculation_key)
    started_at = time.perf_counter()

    if not session.validation.is_valid or session.cleaned_df is None:
        raise ValueError("Input validation has not passed.")
    if spec.requires_liquid and not session.has_liquid:
        raise ValueError(
            "Clinopyroxene-liquid calculations are unavailable because no liquid composition was provided."
        )

    df = session.cleaned_df
    x_cpx = df[[column for column in CPX_OXIDE_COLUMNS if column in df.columns]].copy()
    x_liq = (
        df[[column for column in LIQUID_OXIDE_COLUMNS if column in df.columns]].copy()
        if session.has_liquid
        else None
    )
    parsed_tas_fields = parse_melt_tas_fields(melt_tas_fields)

    log_memory(f"before-{spec.memory_suffix}")
    try:
        ensure_model_registry_loaded()

        from aims4pt.model_tools.CpxTBSelect import workflow_thermobarometry

        if not session.model_pools:
            initialize_model_pools_for_session(session)

        model_pool = session.model_pools.get(spec.key, [])
        environment_skipped_models = session.environment_skipped_models.get(spec.key, [])
        model_pool_error = session.model_pool_errors.get(spec.key)
        if model_pool_error:
            raise RuntimeError(
                f"Model initialization failed for {spec.label}: {model_pool_error}"
            )
        if not model_pool:
            raise RuntimeError(f"No models are available for {spec.label}.")

        workflow, model_pool, skipped_models = _predict_with_environment_fallback(
            workflow_thermobarometry,
            model_pool,
            x_cpx,
            x_liq,
            spec,
            parsed_tas_fields,
        )
        session.model_pools[spec.key] = model_pool

        payload = build_report_payload(
            workflow_obj=workflow,
            model_list=model_pool,
            original_data=df,
        )
        model_summary_df = payload.model_summary_df.copy()
        model_votes_df = payload.model_votes_df.copy()
        report_output = write_payload_to_bytes(payload)
        report_bytes = report_output.getvalue()
        violin_plot_png_missing = payload.violin_plot_png is None
        del payload
        del workflow
        gc.collect()
    finally:
        log_memory(f"after-{spec.memory_suffix}")

    validation_warnings = [message.message for message in session.validation.warnings]
    calculation_warnings = []
    if environment_skipped_models:
        calculation_warnings.append(
            "Some optional-backend models were skipped in the current server environment: "
            + ", ".join(environment_skipped_models)
            + ". These models require a working R/rpy2 or TensorFlow runtime; set AIMS4PT_WEB_ENABLE_R_MODELS=false or AIMS4PT_WEB_ENABLE_TENSORFLOW_MODELS=false to disable them explicitly."
        )
    if skipped_models:
        calculation_warnings.append(
            "Some models were skipped because they failed during this calculation run: "
            + ", ".join(skipped_models)
            + "."
        )
    if violin_plot_png_missing and len(df) < 10:
        calculation_warnings.append(
            "The violin plot was not added to the report because at least 10 input samples are needed."
        )
    summary = (
        f"{spec.label} completed. Samples: {len(df)} | "
        f"Models: {len(model_pool)} | Report: ready"
    )
    logger.info(
        "calculation_completed session_id=%s calculation=%s rows=%s elapsed_seconds=%.2f",
        session.session_id,
        calculation_key,
        len(df),
        time.perf_counter() - started_at,
    )
    return CalculationResult(
        key=calculation_key,
        label=spec.label,
        summary=summary,
        model_summary_df=model_summary_df,
        model_votes_df=model_votes_df,
        report_bytes=report_bytes,
        warnings=list(dict.fromkeys([*validation_warnings, *calculation_warnings])),
    )


def _filter_default_disabled_models(model_pool):
    """Skip optional-backend models when disabled or unavailable."""
    model_pool, skipped_tf = _filter_tensorflow_models(model_pool)
    model_pool, skipped_r = _filter_environment_unavailable_models(model_pool)
    return model_pool, [*skipped_tf, *skipped_r]


def _filter_tensorflow_models(model_pool):
    """Skip TensorFlow-backed models unless explicitly enabled."""
    tf_models = [model for model in model_pool if _is_tensorflow_backed_model(model)]
    if not tf_models or settings.enable_tensorflow_models:
        return model_pool, []

    skipped = [getattr(model, "model_name", type(model).__name__) for model in tf_models]
    filtered = [model for model in model_pool if model not in tf_models]
    logger.warning("tensorflow_backend_disabled skipped_models=%s", skipped)
    return filtered, skipped


def _filter_environment_unavailable_models(model_pool):
    """Skip known R-backed models unless explicitly enabled and available."""
    r_models = [model for model in model_pool if _is_r_backed_model(model)]
    if not r_models:
        return model_pool, []

    if settings.enable_r_models and _can_use_r_backend():
        return model_pool, []

    skipped = [getattr(model, "model_name", type(model).__name__) for model in r_models]
    filtered = [model for model in model_pool if model not in r_models]
    logger.warning("r_backend_disabled_or_unavailable skipped_models=%s", skipped)
    return filtered, skipped


def _is_r_backed_model(model) -> bool:
    module_name = getattr(type(model), "__module__", "")
    return any(marker in module_name for marker in R_BACKED_MODEL_MODULE_MARKERS)


def _is_tensorflow_backed_model(model) -> bool:
    module_name = getattr(type(model), "__module__", "")
    return any(marker in module_name for marker in TENSORFLOW_MODEL_MODULE_MARKERS)


def _can_use_r_backend() -> bool:
    global _r_backend_available
    with _r_backend_lock:
        if _r_backend_available is not None:
            return _r_backend_available
        try:
            import rpy2.robjects  # noqa: F401

            _r_backend_available = True
        except Exception as exc:
            logger.warning("r_backend_check_failed error_type=%s", type(exc).__name__)
            _r_backend_available = False
        return _r_backend_available


def _predict_with_environment_fallback(
    workflow_cls,
    model_pool,
    x_cpx,
    x_liq,
    spec: CalculationSpec,
    parsed_tas_fields: list[str] | None,
):
    """Run a workflow and skip models that cannot run in this environment."""

    def predict_pool(pool):
        workflow = workflow_cls(pool)
        if spec.requires_liquid:
            workflow.predict(x_cpx, x_liq, plot=False)
        elif x_liq is None:
            workflow.predict(
                x_cpx,
                None,
                input_melt_TAS=parsed_tas_fields,
                melt_TAS_source="input_melt_TAS",
                plot=False,
            )
        else:
            workflow.predict(x_cpx, x_liq, plot=False)
        return workflow

    try:
        return predict_pool(model_pool), model_pool, []
    except Exception as full_exc:
        logger.warning(
            "calculation_pool_failed calculation=%s error_type=%s",
            spec.key,
            type(full_exc).__name__,
        )

    working_models = []
    skipped_models = []
    for model in model_pool:
        model_name = getattr(model, "model_name", type(model).__name__)
        try:
            predict_pool([model])
            working_models.append(model)
        except Exception as exc:
            skipped_models.append(model_name)
            logger.warning(
                "calculation_model_skipped calculation=%s model=%s error_type=%s error=%s",
                spec.key,
                model_name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )

    if not working_models:
        raise RuntimeError(f"No models could run for {spec.label}.")

    workflow = predict_pool(working_models)
    return workflow, working_models, skipped_models
