"""Plotting helpers for subplot layout and robust axis limits."""

from typing import Tuple

import math
import pandas as pd


def get_subplot_shape(n_images: int) -> Tuple[int, int]:
    """
    Determine a near-square subplot layout for ``n_images`` figures.

    Parameters
    ----------
    n_images : int
        Number of images or plots to display.

    Returns
    -------
    tuple[int, int]
        ``(rows, cols)`` describing the layout.
    """

    if n_images <= 0:
        return (0, 0)

    best_rows, best_cols = n_images, 1
    min_diff = n_images

    for cols in range(1, n_images + 1):
        rows = math.ceil(n_images / cols)
        if rows * cols < n_images:
            continue
        diff = abs(rows - cols)
        if diff < min_diff or (diff == min_diff and rows * cols < best_rows * best_cols):
            best_rows, best_cols = rows, cols
            min_diff = diff

    if n_images <= 4:
        return (1, n_images)
    return best_rows, best_cols


def get_robust_xlim(
    main_series: pd.Series,
    comp_series: pd.Series | None = None,
    main_lower_q: float = 0.005,
    main_upper_q: float = 0.995,
    comp_lower_q: float = 0.0015,
    comp_upper_q: float = 0.9985,
    similarity_threshold1: float = 0.3,
    similarity_threshold2: float = 0.1,
    expand_main: float = 0.1,
    pad: float = 0.05,
    allow_negative: bool = True,
    negative_fraction: float = 0.01,
) -> Tuple[float, float]:
    """
    Calculate a robust x-axis range from quantiles of the main (and optional comparison) data.

    Parameters
    ----------
    main_series : pandas.Series
        Series representing the main data.
    comp_series : pandas.Series, optional
        Comparison data to consider when computing limits.
    main_lower_q, main_upper_q : float
        Quantiles used to bound the main data.
    comp_lower_q, comp_upper_q : float
        Quantiles used to bound the comparison data.
    similarity_threshold1 : float
        Threshold (as a proportion of the combined range) for considering ranges similar.
    similarity_threshold2 : float
        Stricter similarity threshold used for narrowing the range further.
    expand_main : float
        Proportion used to expand the main data range when distributions overlap.
    pad : float
        Proportion used to pad the final range to avoid clipping.
    allow_negative : bool
        Whether negative lower limits are permitted.
    negative_fraction : float
        Fraction of the range to extend below zero when negatives are allowed.

    Returns
    -------
    tuple[float, float]
        Lower and upper x-axis limits.
    """

    q_min_main = main_series.quantile(main_lower_q)
    q_max_main = main_series.quantile(main_upper_q)

    if comp_series is not None:
        q_min_comp = comp_series.quantile(comp_lower_q)
        q_max_comp = comp_series.quantile(comp_upper_q)

        q_min_union = min(q_min_main, q_min_comp)
        q_max_union = max(q_max_main, q_max_comp)
        overall_range = q_max_union - q_min_union

        diff1 = q_max_comp - q_min_main
        diff2 = q_max_main - q_min_comp

        if min(diff1, diff2) < similarity_threshold1 * overall_range:
            q_min = q_min_main - expand_main * overall_range
            q_max = q_max_main + expand_main * overall_range
            if min(diff1, diff2) < similarity_threshold2 * overall_range:
                q_min = max(q_min_main, q_min_comp) - expand_main * overall_range
                q_max = min(q_max_main, q_max_comp) + expand_main * overall_range
        else:
            q_min, q_max = q_min_union, q_max_union
    else:
        q_min, q_max = q_min_main, q_max_main

    if q_min < 0:
        q_min = -negative_fraction * (q_max - 0) if allow_negative else 0

    padding = pad * (q_max - q_min)
    return q_min - padding, q_max + padding
