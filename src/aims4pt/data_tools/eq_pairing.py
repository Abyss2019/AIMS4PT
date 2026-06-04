# aims4pt/data_tools/pairing.py
#
# Mineral–liquid pairing utilities for AIMS4PT_cpx.
#
# Some liquid-mixing routines in this module were adapted in part from Thermobar:
#
# Wieser, P., Petrelli, M., Lubbers, J., Wieser, E., Ozaydin, S., Kent, A.,
# & Till, C. (2022). Thermobar: An open-source Python3 tool for
# thermobarometry and hygrometry. Volcanica, 5(2), 349–384.
# https://doi.org/10.30909/vol.05.02.349384
"""
Mineral–liquid pairing utilities.

Implements the same pairing logic as the Marapi notebook:

For each mineral row:
  1) Duplicate that row to match the number of liquids
  2) Run kdEquilibrium_test(mineral_repeated, liquids)
  3) Compute kd_error = |Kd - kd_target|
  4) If one_to_one=True: pick the liquid with minimal kd_error (record pass/fail)
     If one_to_one=False: keep all combinations (record pass/fail)

Notes
-----
- This module intentionally does NOT do suffix stripping or extra renaming beyond what
  kdEquilibrium_test already does (normalize_column_names inside equilibrium.py).
- This module is mode-agnostic: it passes mode through to kdEquilibrium_test.
- Optional liquid interpolation (linear mixing) can expand the liquid candidate pool:
      x_mix = f*x_A + (1-f)*x_B,  f in [0, 1]
  where (A, B) are sampled either within the liquid pool ("between") or from two
  provided endmember pools ("endmembers").

IMPORTANT
---------
- This module assumes inputs are already normalized by your normalize_column_names()
  upstream, so 'SiO2' and 'MgO' must exist if plot=True.
- Do NOT re-normalize inside this function (per your request).
- `cleaned_liquids` when one_to_one=True is defined as "liquid rows aligned one-to-one
  with minerals" (same length & order as minerals/pairs; duplicates kept). This matches
  your request: cpx–liq aligned table, with synthetic/original flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from aims4pt.constants import OXIDES_MOLE_MASS
from aims4pt.data_tools.equilibrium import kdEquilibrium_test

LiqInterpMode = Literal["between", "endmembers"]
FStrategy = Literal["uniform", "beta"]


@dataclass(frozen=True)
class PairingResult:
    """Container for pairing outputs."""
    pairs: pd.DataFrame
    pairs_pass_mask: pd.Series
    cleaned_minerals: pd.DataFrame
    cleaned_liquids: pd.DataFrame


def _plot_liquid_pool_diagnostics(
    X_liq_original: pd.DataFrame,
    X_liq_pool: pd.DataFrame,
    cleaned_liquids: pd.DataFrame,
    *,
    fig_dpi: int = 150,
    fig_size: tuple[float, float] = (7.6, 5.6),
    # marker sizes
    s_original: float = 70.0,
    s_synth: float = 26.0,
    s_selected: float = 140.0,
    alpha_original: float = 0.80,
    alpha_synth: float = 0.14,
    alpha_selected: float = 0.98,
) -> None:
    """
    SiO2-Mg# bivariate diagnostic plot.

    Strict assumptions (no guessing, no renaming):
      - Inputs already normalized upstream.
      - Columns 'SiO2', 'MgO', and 'FeO' exist and are value columns (wt%).
      - No column-name guessing is performed.
      - No repeated normalization is performed.

    Layers:
      1) Synthetic liquids (background, less emphasis)
      2) Original liquids (more emphasis)
      3) Selected equilibrium liquids (most emphasis)
    """
    import matplotlib.pyplot as plt

    for df_name, df in [
        ("X_liq_original", X_liq_original),
        ("X_liq_pool", X_liq_pool),
        ("cleaned_liquids", cleaned_liquids),
    ]:
        required_cols = ["SiO2", "MgO", "FeO"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"{df_name} must contain columns {required_cols} after normalization. "
                f"Missing columns: {missing_cols}. "
                f"Got columns head: {list(df.columns)[:30]}"
            )

    def add_liq_mg_number(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquid Mg# as molar Mg / (Mg + Fe)."""
        out = df[["SiO2", "MgO", "FeO"]].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        mg_mol = out["MgO"] / OXIDES_MOLE_MASS["MgO"]
        fe_mol = out["FeO"] / OXIDES_MOLE_MASS["FeO"]
        out["Mg#"] = mg_mol / (mg_mol + fe_mol)
        return out[["SiO2", "Mg#"]].dropna()

    # Prepare layers
    orig = add_liq_mg_number(X_liq_original)

    synth = None
    if "liq__is_synthetic" in X_liq_pool.columns:
        synth = add_liq_mg_number(X_liq_pool.loc[X_liq_pool["liq__is_synthetic"].astype(bool)])
        if len(synth) == 0:
            synth = None

    selected = add_liq_mg_number(cleaned_liquids)

    # Plot
    plt.figure(figsize=fig_size, dpi=fig_dpi)
    ax = plt.gca()

    # (1) synthetic (least emphasis)
    if synth is not None:
        ax.scatter(
            synth["SiO2"], synth["Mg#"],
            s=s_synth,
            alpha=alpha_synth,
            marker="o",
            linewidths=0,
            label="Synthetic liquids",
            zorder=1,
        )

    # (2) original (more emphasis)
    ax.scatter(
        orig["SiO2"], orig["Mg#"],
        s=s_original,
        alpha=alpha_original,
        marker="o",
        linewidths=0.5,
        label="Original liquids",
        zorder=2,
    )

    # (3) selected equilibrium (most emphasis)
    ax.scatter(
        selected["SiO2"], selected["Mg#"],
        s=s_selected,
        alpha=alpha_selected,
        marker="x",
        linewidths=1.6,
        label="Selected equilibrium liquids",
        zorder=3,
    )

    ax.set_xlabel("SiO2 (wt%)")
    ax.set_ylabel("Mg#")
    ax.grid(True, alpha=0.22)
    ax.legend(
        loc="best",
        fontsize=10,
        markerscale=0.9,
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.3,
        handletextpad=0.5,
    )

    # no title (per your request)
    plt.tight_layout()
    plt.show()


def _generate_synthetic_liquids_between(
    X_liq: pd.DataFrame,
    n_synth: int,
    *,
    f_strategy: FStrategy = "uniform",
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    random_state: Optional[int] = None,
    allow_self_pair: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic liquids by linear mixing between sampled liquid compositions.

    For each synthetic liquid, two compositions are sampled from ``X_liq`` and
    mixed as:

        x_mix = f * x_A + (1 - f) * x_B

    where ``f`` is sampled from either a uniform or beta distribution.

    This implementation was adapted in part from Thermobar
    (Wieser et al., 2022).
    
    """
    if n_synth <= 0:
        return X_liq.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)
    n = len(X_liq)
    if n == 0:
        raise ValueError("X_liq is empty, cannot interpolate.")
    if n == 1 and not allow_self_pair:
        raise ValueError("X_liq has only 1 row; set allow_self_pair=True or provide more liquids.")

    cols = list(X_liq.columns)

    idx_A = rng.integers(0, n, size=n_synth)
    if allow_self_pair:
        idx_B = rng.integers(0, n, size=n_synth)
    else:
        idx_B = (idx_A + rng.integers(1, n, size=n_synth)) % n

    A = X_liq.iloc[idx_A].to_numpy(dtype=float)
    B = X_liq.iloc[idx_B].to_numpy(dtype=float)

    if f_strategy == "uniform":
        f = rng.random(n_synth)
    elif f_strategy == "beta":
        f = rng.beta(beta_a, beta_b, size=n_synth)
    else:
        raise ValueError(f"Unknown f_strategy: {f_strategy}")

    f_col = f.reshape(-1, 1)
    mix = f_col * A + (1.0 - f_col) * B

    out = pd.DataFrame(mix, columns=cols)
    out["liq__is_synthetic"] = True
    out["liq__mix_f"] = f
    out["liq__endmemberA_idx"] = idx_A
    out["liq__endmemberB_idx"] = idx_B
    return out


def _generate_synthetic_liquids_endmembers(
    endmember1: pd.DataFrame,
    endmember2: pd.DataFrame,
    n_synth: int,
    *,
    f_strategy: FStrategy = "uniform",
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic liquids by linear mixing between two endmember pools.

    For each synthetic liquid, one composition is sampled from ``endmember1`` and
    one from ``endmember2``. The mixed composition is calculated as:

        x_mix = f * x1 + (1 - f) * x2

    where ``f`` is sampled from either a uniform or beta distribution.

    This implementation was adapted in part from Thermobar
    (Wieser et al., 2022)."""
    if n_synth <= 0:
        return endmember1.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)

    if len(endmember1) == 0 or len(endmember2) == 0:
        raise ValueError("endmember1 and endmember2 must both be non-empty.")

    if list(endmember1.columns) != list(endmember2.columns):
        raise ValueError("endmember1 and endmember2 must have identical columns (same order).")

    cols = list(endmember1.columns)
    idx_1 = rng.integers(0, len(endmember1), size=n_synth)
    idx_2 = rng.integers(0, len(endmember2), size=n_synth)

    X1 = endmember1.iloc[idx_1].to_numpy(dtype=float)
    X2 = endmember2.iloc[idx_2].to_numpy(dtype=float)

    if f_strategy == "uniform":
        f = rng.random(n_synth)
    elif f_strategy == "beta":
        f = rng.beta(beta_a, beta_b, size=n_synth)
    else:
        raise ValueError(f"Unknown f_strategy: {f_strategy}")

    f_col = f.reshape(-1, 1)
    mix = f_col * X1 + (1.0 - f_col) * X2

    out = pd.DataFrame(mix, columns=cols)
    out["liq__is_synthetic"] = True
    out["liq__mix_f"] = f
    out["liq__endmember1_idx"] = idx_1
    out["liq__endmember2_idx"] = idx_2
    return out


def _maybe_interpolate_liquids(
    X_liq_in: pd.DataFrame,
    *,
    interpolate_liq: bool,
    interp_mode: LiqInterpMode = "between",
    n_synth: int = 0,
    include_original_liq: bool = True,
    random_state: Optional[int] = None,
    f_strategy: FStrategy = "uniform",
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    endmember1: Optional[pd.DataFrame] = None,
    endmember2: Optional[pd.DataFrame] = None,
    allow_self_pair: bool = True,
) -> pd.DataFrame:
    """Return a liquid candidate pool: original X_liq plus optional synthetic liquids."""
    base = X_liq_in.copy()
    base["liq__is_synthetic"] = False

    if (not interpolate_liq) or n_synth <= 0:
        return base

    if interp_mode == "between":
        synth = _generate_synthetic_liquids_between(
            X_liq_in,
            n_synth,
            f_strategy=f_strategy,
            beta_a=beta_a,
            beta_b=beta_b,
            random_state=random_state,
            allow_self_pair=allow_self_pair,
        )
    elif interp_mode == "endmembers":
        if endmember1 is None or endmember2 is None:
            raise ValueError("interp_mode='endmembers' requires endmember1 and endmember2.")

        cols = list(X_liq_in.columns)
        em1 = endmember1.copy().reindex(columns=cols, fill_value=0.0)
        em2 = endmember2.copy().reindex(columns=cols, fill_value=0.0)

        synth = _generate_synthetic_liquids_endmembers(
            em1,
            em2,
            n_synth,
            f_strategy=f_strategy,
            beta_a=beta_a,
            beta_b=beta_b,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown interp_mode: {interp_mode}")

    if include_original_liq:
        pool = pd.concat([base, synth], ignore_index=True)
    else:
        pool = synth.reset_index(drop=True)

    return pool


def _select_liquids_aligned_to_pairs(
    X_liq_pool: pd.DataFrame,
    liq_row_idx_series: pd.Series,
) -> pd.DataFrame:
    """
    Build a liquid DataFrame aligned row-by-row to `pairs` (one-to-one mapping).

    - Keeps duplicates (no unique()).
    - Preserves order.
    - If liq_row_idx is NaN (no match), returns a row of NaNs with correct columns.
    """
    cols = list(X_liq_pool.columns)
    rows = []

    nan_row = pd.DataFrame([{c: np.nan for c in cols}])

    for v in liq_row_idx_series.to_numpy():
        if pd.isna(v):
            rows.append(nan_row)
        else:
            rows.append(X_liq_pool.iloc[[int(v)]].reset_index(drop=True))

    out = pd.concat(rows, ignore_index=True)
    return out


def pair_mineral_liquid_by_kd(
    X_mine: pd.DataFrame,
    X_liq: pd.DataFrame,
    *,
    kd: float = 0.28,
    error: float = 0.08,
    mode: str = "Fe-Mg",
    one_to_one: bool = True,
    require_pass: bool = False,
    mine_id_col: Optional[str] = None,
    liq_id_col: Optional[str] = None,
    keep_original_columns: bool = True,
    # --- interpolation / mixing options ---
    interpolate_liq: bool = False,
    interp_mode: LiqInterpMode = "between",
    n_synth: int = 0,
    include_original_liq: bool = True,
    random_state: Optional[int] = None,
    f_strategy: FStrategy = "uniform",
    beta_a: float = 1.0,
    beta_b: float = 1.0,
    endmember1: Optional[pd.DataFrame] = None,
    endmember2: Optional[pd.DataFrame] = None,
    allow_self_pair: bool = True,
    # --- plotting option ---
    plot: bool = True,
) -> PairingResult:
    """
    Pair mineral and liquid compositions using a Kd equilibrium criterion.

    Parameters
    ----------
    X_mine : pd.DataFrame
        Mineral composition table. Must already be normalized to the column names
        expected by ``kdEquilibrium_test``.
    X_liq : pd.DataFrame
        Liquid composition table. Must already be normalized to the column names
        expected by ``kdEquilibrium_test``.
    kd : float, default 0.28
        Target Kd value used to evaluate equilibrium.
    error : float, default 0.08
        Absolute tolerance around ``kd`` used by ``kdEquilibrium_test`` to define pass/fail.
    mode : str, default "Fe-Mg"
        Kd mode forwarded directly to ``kdEquilibrium_test``.
    one_to_one : bool, default True
        Pairing strategy.
        - ``True``: select one liquid per mineral (best match by ``kd_error``).
        - ``False``: keep all mineral-liquid combinations.
    require_pass : bool, default False
        Only used when ``one_to_one=True``.
        - ``True``: chosen liquid must pass the equilibrium test; if no candidate passes,
          that mineral receives no match (NaN liquid fields).
        - ``False``: choose the minimum ``kd_error`` candidate regardless of pass/fail.
    mine_id_col : Optional[str], default None
        Optional column in ``X_mine`` to use as mineral IDs in output metadata.
        If missing or ``None``, mineral index is used.
    liq_id_col : Optional[str], default None
        Optional column in liquid pool to use as liquid IDs in output metadata.
        If missing or ``None``, liquid index is used.
    keep_original_columns : bool, default True
        If ``True``, output ``pairs`` includes original mineral/liquid chemistry columns
        prefixed as ``mine__`` and ``liq__``. If ``False``, output includes only pairing
        metadata and Kd metrics.
    interpolate_liq : bool, default False
        If ``True``, expand candidate liquid pool with synthetic liquids by linear mixing.
    interp_mode : {"between", "endmembers"}, default "between"
        Synthetic liquid generation mode.
        - ``between``: sample both endmembers from ``X_liq``.
        - ``endmembers``: sample from ``endmember1`` and ``endmember2``.
    n_synth : int, default 0
        Number of synthetic liquids to generate when interpolation is enabled.
    include_original_liq : bool, default True
        If ``True``, candidate pool contains original + synthetic liquids;
        otherwise only synthetic liquids are used.
    random_state : Optional[int], default None
        Random seed for synthetic-liquid generation.
    f_strategy : {"uniform", "beta"}, default "uniform"
        Distribution used for mixing coefficient ``f`` in ``x_mix = f*x_A + (1-f)*x_B``.
    beta_a : float, default 1.0
        Alpha parameter of beta distribution when ``f_strategy='beta'``.
    beta_b : float, default 1.0
        Beta parameter of beta distribution when ``f_strategy='beta'``.
    endmember1 : Optional[pd.DataFrame], default None
        First endmember pool used when ``interp_mode='endmembers'``.
    endmember2 : Optional[pd.DataFrame], default None
        Second endmember pool used when ``interp_mode='endmembers'``.
    allow_self_pair : bool, default True
        Only used in ``interp_mode='between'``. If ``False``, prevents selecting the same
        liquid row as both endmembers for one synthetic sample.
    plot : bool, default True
        If ``True``, draw diagnostic SiO2-MgO scatter plot for original liquids,
        synthetic liquids (if any), and selected equilibrium liquids.

    Returns
    -------
    PairingResult
        Dataclass containing:

        ``pairs`` : pd.DataFrame
            Pairing results table. Always includes metadata and Kd metrics:
            ``mine__pair_mine_id``, ``liq__pair_liq_id``, ``mine__pair_row_idx``,
            ``liq__pair_row_idx``, ``kd_value``, ``kd_error``, ``if_pass_test``.
            If ``keep_original_columns=True``, also includes prefixed chemistry columns.

        ``pairs_pass_mask`` : pd.Series
            Boolean mask equal to ``pairs['if_pass_test']``.

        ``cleaned_minerals`` : pd.DataFrame
            Copy of input minerals used in pairing (no re-normalization applied).

        ``cleaned_liquids`` : pd.DataFrame
            Liquid table aligned to output rows, with pairing IDs prepended.
            - ``one_to_one=True``: same length/order as minerals and ``pairs``;
              duplicates preserved; unmatched minerals produce NaN liquid rows.
            - ``one_to_one=False``: same length/order as all rows in ``pairs``
              (all mineral-liquid combinations).
            Includes synthetic metadata columns when interpolation is used
            (for example ``liq__is_synthetic``, ``liq__mix_f``, and sampled endmember indices).

    Raises
    ------
    TypeError
        If ``X_mine`` or ``X_liq`` is not a pandas DataFrame.
    ValueError
        If either input table is empty, if interpolation settings are invalid,
        or if required endmember inputs are missing for ``interp_mode='endmembers'``.
    KeyError
        If ``plot=True`` and required plotting columns (``SiO2``, ``MgO``) are missing.

    Notes
    -----
    This function assumes chemistry column normalization is already done upstream.
    It does not rename or normalize columns internally.
    """
    if not isinstance(X_mine, pd.DataFrame) or not isinstance(X_liq, pd.DataFrame):
        raise TypeError("X_mine and X_liq must be pandas DataFrames.")

    # Per your request: do NOT re-normalize here
    X_mine_in = X_mine.copy()
    X_liq_in0 = X_liq.copy()

    n_mine = len(X_mine_in)
    n_liq0 = len(X_liq_in0)
    if n_mine == 0 or n_liq0 == 0:
        raise ValueError("X_mine and X_liq must both have at least one row.")

    em1_in = endmember1.copy() if endmember1 is not None else None
    em2_in = endmember2.copy() if endmember2 is not None else None

    X_liq_pool = _maybe_interpolate_liquids(
        X_liq_in0,
        interpolate_liq=interpolate_liq,
        interp_mode=interp_mode,
        n_synth=n_synth,
        include_original_liq=include_original_liq,
        random_state=random_state,
        f_strategy=f_strategy,
        beta_a=beta_a,
        beta_b=beta_b,
        endmember1=em1_in,
        endmember2=em2_in,
        allow_self_pair=allow_self_pair,
    )
    n_liq = len(X_liq_pool)

    mine_ids = (
        X_mine_in[mine_id_col].astype(str).to_numpy()
        if mine_id_col is not None and mine_id_col in X_mine_in.columns
        else X_mine_in.index.astype(str).to_numpy()
    )
    liq_ids = (
        X_liq_pool[liq_id_col].astype(str).to_numpy()
        if liq_id_col is not None and liq_id_col in X_liq_pool.columns
        else X_liq_pool.index.astype(str).to_numpy()
    )

    def _assemble_one_row(i: int, j: Optional[int], kd_val: float, kd_err: float, passed: bool) -> pd.DataFrame:
        if keep_original_columns:
            mine_part = X_mine_in.iloc[[i]].add_prefix("mine__").reset_index(drop=True)
            liq_part = (
                pd.DataFrame(index=[0])
                if j is None
                else X_liq_pool.iloc[[j]].add_prefix("liq__").reset_index(drop=True)
            )
            out = pd.concat([mine_part, liq_part], axis=1)
        else:
            out = pd.DataFrame(index=[0])

        out["mine__pair_mine_id"] = mine_ids[i]
        out["liq__pair_liq_id"] = (np.nan if j is None else liq_ids[j])
        out["mine__pair_row_idx"] = int(i)
        out["liq__pair_row_idx"] = (np.nan if j is None else int(j))
        out["kd_value"] = kd_val
        out["kd_error"] = kd_err
        out["if_pass_test"] = bool(passed)
        return out

    # --------------------------
    # one-to-one pairing
    # --------------------------
    if one_to_one:
        rows = []
        for i in range(n_mine):
            mine_rep = pd.DataFrame(
                np.tile(X_mine_in.iloc[i].to_numpy(), (n_liq, 1)),
                columns=X_mine_in.columns,
            )

            mask, kd_cond = kdEquilibrium_test(mine_rep, X_liq_pool, kd=kd, error=error, mode=mode)
            kd_cond = np.asarray(kd_cond, dtype=float)
            mask = np.asarray(mask, dtype=bool)

            kd_err = np.abs(kd_cond - kd)

            if require_pass:
                if np.any(mask):
                    cand = np.where(mask)[0]
                    j = int(cand[np.argmin(kd_err[cand])])
                else:
                    j = None
            else:
                j = int(np.nanargmin(kd_err))

            if j is None:
                rows.append(_assemble_one_row(i, None, float(np.nan), float(np.nan), False))
            else:
                rows.append(_assemble_one_row(i, j, float(kd_cond[j]), float(kd_err[j]), bool(mask[j])))

        pairs = pd.concat(rows, ignore_index=True)
        pairs_pass_mask = pairs["if_pass_test"].astype(bool)

        # cleaned_liquids aligned 1:1 with minerals (same length & order; duplicates kept)
        cleaned_liquids = _select_liquids_aligned_to_pairs(X_liq_pool, pairs["liq__pair_row_idx"])

        # add pairing ids for easy alignment
        cleaned_liquids = pd.concat(
            [
                pairs[["mine__pair_mine_id", "liq__pair_liq_id", "liq__pair_row_idx"]].reset_index(drop=True),
                cleaned_liquids.reset_index(drop=True),
            ],
            axis=1,
        )

        if plot:
            _plot_liquid_pool_diagnostics(
                X_liq_original=X_liq_in0,
                X_liq_pool=X_liq_pool,
                cleaned_liquids=cleaned_liquids,
            )

        return PairingResult(
            pairs=pairs,
            pairs_pass_mask=pairs_pass_mask,
            cleaned_minerals=X_mine_in,
            cleaned_liquids=cleaned_liquids,
        )

    # --------------------------
    # all-to-all pairing
    # --------------------------
    mine_idx = np.repeat(np.arange(n_mine), n_liq)
    liq_idx = np.tile(np.arange(n_liq), n_mine)

    kd_all = np.empty(n_mine * n_liq, dtype=float)
    mask_all = np.empty(n_mine * n_liq, dtype=bool)

    cursor = 0
    for i in range(n_mine):
        mine_rep = pd.DataFrame(
            np.tile(X_mine_in.iloc[i].to_numpy(), (n_liq, 1)),
            columns=X_mine_in.columns,
        )
        mask, kd_cond = kdEquilibrium_test(mine_rep, X_liq_pool, kd=kd, error=error, mode=mode)
        kd_all[cursor : cursor + n_liq] = np.asarray(kd_cond, dtype=float)
        mask_all[cursor : cursor + n_liq] = np.asarray(mask, dtype=bool)
        cursor += n_liq

    kd_err_all = np.abs(kd_all - kd)

    if keep_original_columns:
        mine_block = X_mine_in.iloc[mine_idx].reset_index(drop=True).add_prefix("mine__")
        liq_block = X_liq_pool.iloc[liq_idx].reset_index(drop=True).add_prefix("liq__")
        pairs = pd.concat([mine_block, liq_block], axis=1)
    else:
        pairs = pd.DataFrame(index=np.arange(n_mine * n_liq))

    pairs["mine__pair_mine_id"] = mine_ids[mine_idx]
    pairs["liq__pair_liq_id"] = liq_ids[liq_idx]
    pairs["mine__pair_row_idx"] = mine_idx
    pairs["liq__pair_row_idx"] = liq_idx
    pairs["kd_value"] = kd_all
    pairs["kd_error"] = kd_err_all
    pairs["if_pass_test"] = mask_all

    pairs_pass_mask = pairs["if_pass_test"].astype(bool)

    # all-to-all: cleaned_liquids aligned to pairs (same length/order as pairs)
    cleaned_liquids = X_liq_pool.iloc[liq_idx].reset_index(drop=True)
    cleaned_liquids = pd.concat(
        [
            pairs[["mine__pair_mine_id", "liq__pair_liq_id", "liq__pair_row_idx"]].reset_index(drop=True),
            cleaned_liquids.reset_index(drop=True),
        ],
        axis=1,
    )
    cleaned_minerals = X_mine_in.iloc[mine_idx].reset_index(drop=True)

    if plot:
        _plot_liquid_pool_diagnostics(
            X_liq_original=X_liq_in0,
            X_liq_pool=X_liq_pool,
            cleaned_liquids=cleaned_liquids,
        )

    return PairingResult(
        pairs=pairs,
        pairs_pass_mask=pairs_pass_mask,
        cleaned_minerals=cleaned_minerals,
        cleaned_liquids=cleaned_liquids,
    )
