"""Utilities to locate elbow (knee) points in 2D datasets."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from kneed import KneeLocator


def find_elbow_point(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    curve: str = "concave",
    direction: str = "increasing",
    plot: bool = True,
) -> Optional[Any]:
    """Identify the elbow point in a dataset using the kneed algorithm.

    Parameters
    ----------
    data : pandas.DataFrame
        Data points containing numeric ``x`` and ``y`` columns.
    x_col : str
        Name of the column representing the x-axis values.
    y_col : str
        Name of the column representing the y-axis values.
    curve : str
        Curve type ("concave" or "convex").
    direction : str
        Direction of the curve ("increasing" or "decreasing").
    plot : bool, default True
        Whether to draw the knee plot. The default keeps the previous behavior
        of always plotting the knee detection output.

    Returns
    -------
    Optional[Any]
        The x-value at the elbow point, or ``None`` if no knee is detected.
    """

    if x_col not in data.columns or y_col not in data.columns:
        missing_cols = [col for col in (x_col, y_col) if col not in data.columns]
        missing_str = ", ".join(missing_cols)
        raise KeyError(f"Columns not found in data: {missing_str}")

    cleaned = data[[x_col, y_col]].dropna()
    if cleaned.empty:
        return None

    x = cleaned[x_col].to_numpy()
    y = cleaned[y_col].to_numpy()

    kneedle = KneeLocator(x, y, S=1.0, curve=curve, direction=direction)
    elbow_x = kneedle.knee
    if plot:
        kneedle.plot_knee()
    return elbow_x
