"""Lightweight helpers for manual testing of column normalization."""

from __future__ import annotations

import pandas as pd

from aims4pt.utils import normalize_column_names


# Test Case 1: Exact matches
def test_case_exact_matches() -> None:
    print("Test Case 1: Exact matches")
    df = pd.DataFrame(
        {
            "SiO2_variant": [50.3, 51.2],
            "FeO_val": [10.2, 9.8],
            "Al2O3_something": [18.3, 18.6],
        }
    )

    try:
        df_normalized = normalize_column_names(df)
        print("After normalization:\n", df_normalized)
    except ValueError as exc:
        print(exc)


# Test Case 2: No match
def test_case_no_match() -> None:
    print("\nTest Case 2: No match")
    df = pd.DataFrame({"UnknownCol": [0.1, 0.2], "AnotherUnknown": [0.3, 0.4]})

    try:
        df_normalized = normalize_column_names(df)
        print("After normalization:\n", df_normalized)
    except ValueError as exc:
        print(exc)


# Test Case 3: Multiple matches
def test_case_multiple_matches() -> None:
    print("\nTest Case 3: Multiple matches")
    df = pd.DataFrame(
        {
            "FeO_F_mix": [10.2, 9.8],  # Should match both FeO and F
            "SiO2.Fe2O3_mix": [50.3, 51.2],  # Should match SiO2 and Fe2O3
        }
    )

    try:
        df_normalized = normalize_column_names(df)
        print("After normalization:\n", df_normalized)
    except ValueError as exc:
        print(exc)


# Test Case 4: Correct disambiguation (FeO vs F)
def test_case_disambiguation() -> None:
    print("\nTest Case 4: Correct disambiguation (FeO vs F)")
    df = pd.DataFrame({"FeO_content": [10.2, 9.8], "F_content": [0.1, 0.2]})

    try:
        df_normalized = normalize_column_names(df)
        print("After normalization:\n", df_normalized)
    except ValueError as exc:
        print(exc)


# Test Case 5: Mixed valid and invalid columns
def test_case_mixed_valid_invalid() -> None:
    print("\nTest Case 5: Mixed valid and invalid columns")
    df = pd.DataFrame(
        {
            "TiO2_value": [0.5, 0.6],
            "MnO_level": [1.1, 1.2],
            "UnknownCol": [0.1, 0.2],
            "Cr2O3_metric": [0.3, 0.4],
        }
    )

    try:
        df_normalized = normalize_column_names(df)
        print("After normalization:\n", df_normalized)
    except ValueError as exc:
        print(exc)

def get_test_cpx_liq ():

    # Liquid / glass composition, wt%
    X_liq = pd.DataFrame({
        "SiO2":  [51.1, 51.5, 59.1, 52.5, 56.2],
        "TiO2":  [0.93, 1.19, 0.54, 0.98, 0.34],
        "Al2O3": [17.5, 19.2, 19.1, 19.2, 20.4],
        "FeOt":  [8.91, 8.70, 5.22, 8.04, 5.88],
        "MnO":   [0.18, 0.19, 0.19, 0.20, 0.20],
        "MgO":   [6.09, 4.98, 3.25, 4.99, 2.58],
        "CaO":   [11.5, 10.0, 7.45, 9.64, 7.18],
        "Na2O":  [3.53, 3.72, 4.00, 4.15, 6.02],
        "K2O":   [0.17, 0.42, 0.88, 0.21, 1.02],
        "Cr2O3": [0.00, 0.00, 0.00, 0.00, 0.00],
        "P2O5":  [0.15, 0.14, 0.31, 0.14, 0.23],
        "H2O":   [3.8, 6.2, 6.2, 6.2, 6.2],
    })

    # Saturation H2O, wt%
    H2O_sat = pd.Series(
        [-0.22, 4.85, 1.04, 3.88, 5.53],
        name="H2O_sat"
    )

    # Clinopyroxene composition, wt%
    X_cpx = pd.DataFrame({
        "SiO2":  [51.5, 50.3, 47.3, 51.1, 51.0],
        "TiO2":  [0.50, 0.73, 1.75, 0.63, 0.56],
        "Al2O3": [3.70, 4.12, 7.85, 4.41, 3.00],
        "FeOt":  [5.18, 5.83, 6.51, 5.66, 5.00],
        "MnO":   [0.09, 0.00, 0.14, 0.13, 0.20],
        "MgO":   [15.8, 15.0, 13.1, 15.6, 13.0],
        "CaO":   [22.8, 22.7, 22.5, 22.6, 22.4],
        "Na2O":  [0.24, 0.24, 0.25, 0.23, 0.31],
        "K2O":   [0.00, 0.00, 0.00, 0.00, 0.00],
        "Cr2O3": [0.66, 0.28, 0.22, 0.27, 0.09],
    })  
    return X_liq, X_cpx


def run_all_tests() -> None:
    """Run all demonstration test cases sequentially."""

    test_case_exact_matches()
    test_case_no_match()
    test_case_multiple_matches()
    test_case_disambiguation()
    test_case_mixed_valid_invalid()


# Run all test cases
if __name__ == "__main__":
    run_all_tests()
