"""Helper functions shared by the Merapi application notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Helpers extracted from 04_apply_AIMS4PT_cpx_to_Merapi.ipynb cell 9.
def print_df_for_md_format__nb04_c09(df: pd.DataFrame):
    """
    Print a DataFrame in a markdown table format suitable for GPT.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to print.
    """
    from io import StringIO

    buffer = StringIO()
    df.to_markdown(buf=buffer, index=False)
    markdown_table = buffer.getvalue()
    print(markdown_table)


# Helpers extracted from 04_apply_AIMS4PT_cpx_to_Merapi.ipynb cell 77.
import numpy as np
import pandas as pd

def _safe_positive__nb04_c77(X: np.ndarray, eps: float) -> np.ndarray:
    """
    Replace non-positive values with eps to make log-ratio valid.
    """
    X = np.asarray(X, dtype=float)
    X[X <= 0] = eps
    return X

def clr_transform__nb04_c77(df: pd.DataFrame, comp_cols: list[str], eps: float = 1e-4) -> pd.DataFrame:
    """
    CLR transform for compositional data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    comp_cols : list[str]
        Columns treated as compositional parts (must be positive after replacement).
    eps : float
        Small value to replace zeros/negatives.
    
    Returns
    -------
    pd.DataFrame
        CLR-transformed dataframe (same index, columns named 'clr_{col}').
    """
    X = df[comp_cols].to_numpy(dtype=float)
    X = _safe_positive__nb04_c77(X, eps=eps)

    # geometric mean along parts
    g = np.exp(np.mean(np.log(X), axis=1))  # shape (n,)
    clr = np.log(X / g[:, None])            # shape (n, D)

    out_cols = [f"clr_{c}" for c in comp_cols]
    return pd.DataFrame(clr, index=df.index, columns=out_cols)

def pairwise_euclidean__nb04_c77(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of A (n x d) and B (m x d).
    Returns dist matrix (n x m).
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    A2 = np.sum(A * A, axis=1)[:, None]   # (n,1)
    B2 = np.sum(B * B, axis=1)[None, :]   # (1,m)
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    D2 = np.maximum(D2, 0.0)              # numerical safety
    return np.sqrt(D2)

def topk_neighbors_clr__nb04_c77(
    S_eq: pd.DataFrame,
    S_na: pd.DataFrame,
    comp_cols: list[str],
    k: int = 5,
    eps: float = 1e-4,
    eq_meta_cols: list[str] = None,
    na_meta_cols: list[str] = None,
) -> dict:
    """
    For each natural sample, find top-k nearest experimental samples
    using Euclidean distance in CLR space (Aitchison distance).
    
    Returns
    -------
    dict: {na_id: pd.DataFrame_of_topk_neighbors}
    """
    if eq_meta_cols is None:
        eq_meta_cols = []
    if na_meta_cols is None:
        na_meta_cols = []

    # Ensure required columns exist
    missing_eq = [c for c in comp_cols if c not in S_eq.columns]
    missing_na = [c for c in comp_cols if c not in S_na.columns]
    if missing_eq:
        raise ValueError(f"S_eq missing compositional columns: {missing_eq}")
    if missing_na:
        raise ValueError(f"S_na missing compositional columns: {missing_na}")
    # fill na 
    S_eq.fillna(0, inplace=True)
    S_na.fillna(0, inplace=True)

    # CLR transform
    eq_clr = clr_transform__nb04_c77(S_eq, comp_cols=comp_cols, eps=eps)
    na_clr = clr_transform__nb04_c77(S_na, comp_cols=comp_cols, eps=eps)

    # Distances (na x eq)
    dist = pairwise_euclidean__nb04_c77(na_clr.to_numpy(), eq_clr.to_numpy())
    # dist = pairwise_euclidean__nb04_c77(S_na[comp_cols].to_numpy(), S_eq[comp_cols].to_numpy())

    results = {}
    eq_index = np.array(S_eq.index)

    for i, na_id in enumerate(S_na.index):
        drow = dist[i, :]  # distances to all eq points

        # argsort distances; take top-k
        nn_idx = np.argsort(drow)[:k]
        nn_ids = eq_index[nn_idx]
        nn_dist = drow[nn_idx]

        # Build output table
        out = pd.DataFrame({
            "eq_id": nn_ids,
            "distance": nn_dist,
        })

        # attach eq metadata
        for c in eq_meta_cols:
            if c not in S_eq.columns:
                raise ValueError(f"eq_meta_cols contains missing column in S_eq: {c}")
            out[c] = S_eq.loc[nn_ids, c].to_numpy()

        # attach na metadata (repeat same value)
        for c in na_meta_cols:
            if c not in S_na.columns:
                raise ValueError(f"na_meta_cols contains missing column in S_na: {c}")
            out[f"na_{c}"] = S_na.loc[na_id, c]

        out = out.sort_values("distance", ascending=True).reset_index(drop=True)
        results[na_id] = out

    return results



# Helpers extracted from 04_apply_AIMS4PT_cpx_to_Merapi.ipynb cell 78.
import numpy as np
import pandas as pd

def _safe_positive__nb04_c78(X: np.ndarray, eps: float) -> np.ndarray:
    """
    Replace non-positive values with eps to make log-ratio valid.
    """
    X = np.asarray(X, dtype=float)
    X[X <= 0] = eps
    return X

def clr_transform__nb04_c78(df: pd.DataFrame, comp_cols: list[str], eps: float = 1e-4) -> pd.DataFrame:
    """
    CLR transform for compositional data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    comp_cols : list[str]
        Columns treated as compositional parts (must be positive after replacement).
    eps : float
        Small value to replace zeros/negatives.
    
    Returns
    -------
    pd.DataFrame
        CLR-transformed dataframe (same index, columns named 'clr_{col}').
    """
    X = df[comp_cols].to_numpy(dtype=float)
    X = _safe_positive__nb04_c78(X, eps=eps)

    # geometric mean along parts
    g = np.exp(np.mean(np.log(X), axis=1))  # shape (n,)
    clr = np.log(X / g[:, None])            # shape (n, D)

    out_cols = [f"clr_{c}" for c in comp_cols]
    return pd.DataFrame(clr, index=df.index, columns=out_cols)

def pairwise_euclidean__nb04_c78(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of A (n x d) and B (m x d).
    Returns dist matrix (n x m).
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    A2 = np.sum(A * A, axis=1)[:, None]   # (n,1)
    B2 = np.sum(B * B, axis=1)[None, :]   # (1,m)
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    D2 = np.maximum(D2, 0.0)              # numerical safety
    return np.sqrt(D2)

def topk_neighbors_clr__nb04_c78(
    S_eq: pd.DataFrame,
    S_na: pd.DataFrame,
    comp_cols: list[str],
    k: int = 5,
    eps: float = 1e-4,
    eq_meta_cols: list[str] = None,
    na_meta_cols: list[str] = None,
) -> dict:
    """
    For each natural sample, find top-k nearest experimental samples
    using Euclidean distance in CLR space (Aitchison distance).
    
    Returns
    -------
    dict: {na_id: pd.DataFrame_of_topk_neighbors}
    """
    if eq_meta_cols is None:
        eq_meta_cols = []
    if na_meta_cols is None:
        na_meta_cols = []

    # Ensure required columns exist
    missing_eq = [c for c in comp_cols if c not in S_eq.columns]
    missing_na = [c for c in comp_cols if c not in S_na.columns]
    if missing_eq:
        raise ValueError(f"S_eq missing compositional columns: {missing_eq}")
    if missing_na:
        raise ValueError(f"S_na missing compositional columns: {missing_na}")
    # fill na 
    S_eq.fillna(0, inplace=True)
    S_na.fillna(0, inplace=True)

    # CLR transform
    # eq_clr = clr_transform__nb04_c78(S_eq, comp_cols=comp_cols, eps=eps)
    # na_clr = clr_transform__nb04_c78(S_na, comp_cols=comp_cols, eps=eps)

    # Distances (na x eq)
    # dist = pairwise_euclidean__nb04_c78(na_clr.to_numpy(), eq_clr.to_numpy())
    dist = pairwise_euclidean__nb04_c78(S_na[comp_cols].to_numpy(), S_eq[comp_cols].to_numpy())

    results = {}
    eq_index = np.array(S_eq.index)

    for i, na_id in enumerate(S_na.index):
        drow = dist[i, :]  # distances to all eq points

        # argsort distances; take top-k
        nn_idx = np.argsort(drow)[:k]
        nn_ids = eq_index[nn_idx]
        nn_dist = drow[nn_idx]

        # Build output table
        out = pd.DataFrame({
            "eq_id": nn_ids,
            "distance": nn_dist,
        })

        # attach eq metadata
        for c in eq_meta_cols:
            if c not in S_eq.columns:
                raise ValueError(f"eq_meta_cols contains missing column in S_eq: {c}")
            out[c] = S_eq.loc[nn_ids, c].to_numpy()

        # attach na metadata (repeat same value)
        for c in na_meta_cols:
            if c not in S_na.columns:
                raise ValueError(f"na_meta_cols contains missing column in S_na: {c}")
            out[f"na_{c}"] = S_na.loc[na_id, c]

        out = out.sort_values("distance", ascending=True).reset_index(drop=True)
        results[na_id] = out

    return results

