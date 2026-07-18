"""Validate an installed AIMS4PT macOS runtime."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from importlib.metadata import version as distribution_version
from importlib.resources import files
from pathlib import Path


REQUIRED_MODULES = [
    "aims4pt_web",
    "aims4pt",
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "tensorflow",
    "keras",
    "rpy2",
    "onnxruntime",
]

REQUIRED_RESOURCES = [
    ("aims4pt.reporting.data", "independent_composition_kde.npz"),
    ("aims4pt.reporting.data", "independent_data_final.xlsx"),
]

REQUIRED_R_PACKAGES = [
    "PerformanceAnalytics",
    "rJava",
    "extraTrees",
    "readxl",
    "EnvStats",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-prefix", required=True)
    return parser.parse_args()


def _assert_runtime_location(expected_prefix: Path) -> None:
    import aims4pt

    package_path = Path(aims4pt.__file__).resolve()
    prefix = expected_prefix.resolve()
    print(f"package_path {package_path}")
    if not package_path.is_relative_to(prefix):
        raise RuntimeError(
            f"AIMS4PT was imported from {package_path}, outside {prefix}."
        )

    if Path(sys.prefix).resolve() != prefix:
        raise RuntimeError(f"sys.prefix is {sys.prefix}, expected {prefix}.")


def _check_resources() -> None:
    for package_name, resource_name in REQUIRED_RESOURCES:
        resource = files(package_name).joinpath(resource_name)
        exists = resource.is_file()
        print(f"resource_ok {package_name} {resource_name} {exists}")
        if not exists:
            raise RuntimeError(
                f"Missing packaged resource: {package_name}/{resource_name}"
            )


def _check_tensorflow() -> None:
    import tensorflow as tf

    result = tf.reduce_sum(tf.constant([1.0, 2.0])).numpy().item()
    print(f"tensorflow_operation_ok {result}")
    if result != 3.0:
        raise RuntimeError(f"Unexpected TensorFlow result: {result}")


def _check_r_runtime() -> None:
    from aims4pt.model_tools.rpy_tools.r_env import configure_r_environment

    r_home = configure_r_environment()
    print(f"r_home {r_home}")
    if not r_home:
        raise RuntimeError("The bundled R runtime could not be discovered.")

    import rpy2.robjects as ro

    version_text = str(
        ro.r('paste(R.version$major, R.version$minor, sep=".")')[0]
    )
    print(f"r_version {version_text}")
    if version_text != "4.4.3":
        raise RuntimeError(f"Expected R 4.4.3, found {version_text}.")

    expression = (
        "c("
        + ",".join(
            f'"{package}"=requireNamespace("{package}", quietly=TRUE)'
            for package in REQUIRED_R_PACKAGES
        )
        + ")"
    )
    availability = ro.r(expression)
    missing = [
        package
        for package, available in zip(REQUIRED_R_PACKAGES, availability)
        if not bool(available)
    ]
    print(f"r_packages_ok {len(REQUIRED_R_PACKAGES) - len(missing)}")
    if missing:
        raise RuntimeError(f"Missing required R packages: {', '.join(missing)}")


def main() -> None:
    """Run version, import, resource, and native-runtime checks."""
    args = _parse_args()
    expected_prefix = Path(args.expected_prefix)
    os.environ.setdefault("MPLBACKEND", "Agg")

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            f"Expected Darwin arm64, found {platform.system()} {platform.machine()}."
        )

    installed_version = distribution_version("AIMS4PT")
    print(f"package_version {installed_version}")
    if installed_version != args.expected_version:
        raise RuntimeError(
            f"Expected AIMS4PT {args.expected_version}, found {installed_version}."
        )

    _assert_runtime_location(expected_prefix)

    for module_name in REQUIRED_MODULES:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", "<built-in>")
        print(f"import_ok {module_name} {module_file}")

    _check_resources()
    _check_tensorflow()
    _check_r_runtime()

    from aims4pt.model_tools.model_registry import import_all_models
    from aims4pt_web.main import app

    imported, failed = import_all_models(skip_failed=True)
    print(f"web_app_ok {app.title} {app.version}")
    print(f"model_registry_imported {len(imported)}")
    print(f"model_registry_failed {len(failed)}")
    if len(imported) != 8 or failed:
        raise RuntimeError(
            f"Expected all 8 model modules to import; imported={imported}, failed={failed}."
        )
    if app.version != args.expected_version:
        raise RuntimeError(
            f"Expected web app version {args.expected_version}, found {app.version}."
        )


if __name__ == "__main__":
    main()
