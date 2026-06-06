"""Input workbook validation for the AIMS4PT web app."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Iterable

import pandas as pd


SHEET_NAME = "compositions"

CPX_OXIDE_COLUMNS = [
    "SiO2_cpx",
    "TiO2_cpx",
    "Al2O3_cpx",
    "FeOt_cpx",
    "MnO_cpx",
    "MgO_cpx",
    "CaO_cpx",
    "Na2O_cpx",
    "K2O_cpx",
    "Cr2O3_cpx",
]

LIQUID_OXIDE_COLUMNS = [
    "SiO2_liq",
    "TiO2_liq",
    "Al2O3_liq",
    "FeOt_liq",
    "MnO_liq",
    "MgO_liq",
    "CaO_liq",
    "Na2O_liq",
    "K2O_liq",
    "Cr2O3_liq",
    "P2O5_liq",
    "H2O_liq",
]

CORE_CPX_COLUMNS = [
    "SiO2_cpx",
    "Al2O3_cpx",
    "FeOt_cpx",
    "MgO_cpx",
    "CaO_cpx",
    "Na2O_cpx",
]

EXPECTED_INPUT_COLUMNS = ["Sample_ID", *CPX_OXIDE_COLUMNS, *LIQUID_OXIDE_COLUMNS]
ALL_OXIDE_COLUMNS = [*CPX_OXIDE_COLUMNS, *LIQUID_OXIDE_COLUMNS]
UNSAFE_SAMPLE_PREFIXES = ("=", "+", "@")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class ValidationMessage:
    level: str
    message: str
    rows: str = ""
    columns: str = ""


@dataclass
class ValidationResult:
    parsed_df: pd.DataFrame | None = None
    cleaned_df: pd.DataFrame | None = None
    messages: list[ValidationMessage] = field(default_factory=list)
    has_liquid: bool = False

    @property
    def errors(self) -> list[ValidationMessage]:
        return [message for message in self.messages if message.level == "Error"]

    @property
    def warnings(self) -> list[ValidationMessage]:
        return [message for message in self.messages if message.level == "Warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors and self.cleaned_df is not None

    @property
    def sample_count(self) -> int:
        if self.cleaned_df is not None:
            return len(self.cleaned_df)
        if self.parsed_df is not None:
            return len(self.parsed_df)
        return 0


def _is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _blank_rows(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: all(_is_blank(value) for value in row), axis=1)


def _clean_column_name(column: object) -> str:
    if pd.isna(column):
        return ""
    return str(column).strip()


def _composition_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column != "Sample_ID" and (column.endswith("_cpx") or column.endswith("_liq"))
    ]


def _detect_liquid_available(df: pd.DataFrame) -> bool:
    if "SiO2_liq" not in df.columns:
        return False
    return not df["SiO2_liq"].apply(_is_blank).all()


def _format_excel_rows(indexes: Iterable[int]) -> str:
    return ", ".join(str(index + 3) for index in indexes)


def _add_message(
    messages: list[ValidationMessage],
    level: str,
    message: str,
    *,
    rows: str = "",
    columns: Iterable[str] | str = "",
) -> None:
    if isinstance(columns, str):
        column_text = columns
    else:
        column_text = ", ".join(columns)
    messages.append(
        ValidationMessage(level=level, message=message, rows=rows, columns=column_text)
    )


def _is_unsafe_sample_id(value: object) -> bool:
    text = str(value).strip()
    if text.startswith(UNSAFE_SAMPLE_PREFIXES):
        return True
    return CONTROL_CHAR_RE.search(text) is not None


def _read_compositions_sheet(file_bytes: bytes) -> tuple[pd.DataFrame | None, list[ValidationMessage]]:
    messages: list[ValidationMessage] = []
    try:
        workbook = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    except Exception:
        _add_message(messages, "Error", "The workbook cannot be opened.")
        return None, messages

    if SHEET_NAME not in workbook.sheet_names:
        _add_message(messages, "Error", 'The sheet "compositions" is missing.')
        return None, messages

    try:
        df = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=SHEET_NAME,
            skiprows=1,
            engine="openpyxl",
        )
    except Exception:
        _add_message(messages, "Error", "The actual column-header row cannot be read.")
        return None, messages

    df.columns = [_clean_column_name(column) for column in df.columns]
    unnamed_empty_columns = [
        column
        for column in df.columns
        if column.startswith("Unnamed:") and df[column].apply(_is_blank).all()
    ]
    if unnamed_empty_columns:
        df = df.drop(columns=unnamed_empty_columns)

    if df.empty and len(df.columns) == 0:
        _add_message(messages, "Error", "The actual column-header row cannot be read.")
    elif all(_is_blank(column) or str(column).startswith("Unnamed:") for column in df.columns):
        _add_message(messages, "Error", "The actual column-header row cannot be read.")

    return df, messages


def validate_workbook_bytes(file_bytes: bytes) -> ValidationResult:
    """Validate an uploaded workbook and return parsed and cleaned DataFrames."""
    df, messages = _read_compositions_sheet(file_bytes)
    if df is None or any(message.level == "Error" for message in messages):
        return ValidationResult(parsed_df=df, messages=messages)

    has_liquid = _detect_liquid_available(df)

    empty_row_mask = _blank_rows(df)
    if empty_row_mask.any():
        _add_message(
            messages,
            "Error",
            "There are completely empty rows inside the data table.",
            rows=_format_excel_rows(df.index[empty_row_mask]),
        )

    if "Sample_ID" not in df.columns:
        _add_message(messages, "Error", "The Sample_ID column is missing.")
    else:
        blank_sample_mask = df["Sample_ID"].apply(_is_blank)
        if blank_sample_mask.any():
            _add_message(
                messages,
                "Error",
                "Sample_ID contains blank values.",
                rows=_format_excel_rows(df.index[blank_sample_mask]),
                columns="Sample_ID",
            )

        unsafe_mask = df["Sample_ID"].apply(
            lambda value: False if _is_blank(value) else _is_unsafe_sample_id(value)
        )
        if unsafe_mask.any():
            _add_message(
                messages,
                "Error",
                "Sample_ID contains unsafe or illegal characters.",
                rows=_format_excel_rows(df.index[unsafe_mask]),
                columns="Sample_ID",
            )

        duplicate_mask = df["Sample_ID"].duplicated(keep=False) & ~blank_sample_mask
        if duplicate_mask.any():
            _add_message(
                messages,
                "Warning",
                "Duplicated Sample_ID values were found.",
                rows=_format_excel_rows(df.index[duplicate_mask]),
                columns="Sample_ID",
            )

    missing_core = [column for column in CORE_CPX_COLUMNS if column not in df.columns]
    if missing_core:
        _add_message(
            messages,
            "Error",
            "One or more clinopyroxene composition columns are missing.",
            columns=missing_core,
        )

    cleaned_df = df.copy()
    for column in _composition_columns(cleaned_df.columns):
        blank_mask = cleaned_df[column].apply(_is_blank)
        converted = pd.to_numeric(cleaned_df[column], errors="coerce")
        invalid_mask = converted.isna() & ~blank_mask
        if invalid_mask.any():
            _add_message(
                messages,
                "Error",
                "A composition column contains values that cannot be converted to float.",
                rows=_format_excel_rows(cleaned_df.index[invalid_mask]),
                columns=column,
            )
        cleaned_df[column] = converted

    if any(message.level == "Error" for message in messages):
        return ValidationResult(parsed_df=df, cleaned_df=None, messages=messages, has_liquid=has_liquid)

    if not has_liquid:
        _add_message(
            messages,
            "Warning",
            "Clinopyroxene-liquid calculations are unavailable because no liquid composition was provided.",
        )

    missing_oxide_rows: set[int] = set()
    missing_oxide_columns: set[str] = set()
    h2o_rows: set[int] = set()
    h2o_columns: set[str] = set()

    for column in CPX_OXIDE_COLUMNS:
        if column not in cleaned_df.columns:
            cleaned_df[column] = 0.0
            missing_oxide_columns.add(column)
        else:
            blank_mask = cleaned_df[column].isna()
            if blank_mask.any():
                cleaned_df.loc[blank_mask, column] = 0.0
                missing_oxide_rows.update(cleaned_df.index[blank_mask].tolist())
                missing_oxide_columns.add(column)

    for column in LIQUID_OXIDE_COLUMNS:
        if column not in cleaned_df.columns:
            cleaned_df[column] = 0.0
            if has_liquid:
                if column == "H2O_liq":
                    h2o_columns.add(column)
                else:
                    missing_oxide_columns.add(column)
        else:
            blank_mask = cleaned_df[column].isna()
            if blank_mask.any():
                cleaned_df.loc[blank_mask, column] = 0.0
                if has_liquid:
                    if column == "H2O_liq":
                        h2o_rows.update(cleaned_df.index[blank_mask].tolist())
                        h2o_columns.add(column)
                    else:
                        missing_oxide_rows.update(cleaned_df.index[blank_mask].tolist())
                        missing_oxide_columns.add(column)

    extra_composition_cols = [
        column
        for column in _composition_columns(cleaned_df.columns)
        if column not in ALL_OXIDE_COLUMNS
    ]
    for column in extra_composition_cols:
        blank_mask = cleaned_df[column].isna()
        if blank_mask.any():
            cleaned_df.loc[blank_mask, column] = 0.0
            missing_oxide_rows.update(cleaned_df.index[blank_mask].tolist())
            missing_oxide_columns.add(column)

    cleaned_df["Sample_ID"] = cleaned_df["Sample_ID"].astype(str).str.strip()
    ordered_columns = [
        column for column in EXPECTED_INPUT_COLUMNS if column in cleaned_df.columns
    ]
    remaining_columns = [
        column for column in cleaned_df.columns if column not in ordered_columns
    ]
    cleaned_df = cleaned_df[ordered_columns + remaining_columns]

    if missing_oxide_columns:
        _add_message(
            messages,
            "Warning",
            "Some missing oxide values were filled with 0.",
            rows=_format_excel_rows(sorted(missing_oxide_rows)),
            columns=sorted(missing_oxide_columns),
        )
    if h2o_columns:
        _add_message(
            messages,
            "Warning",
            "Missing H2O_liq values were filled with 0.",
            rows=_format_excel_rows(sorted(h2o_rows)),
            columns="H2O_liq",
        )

    return ValidationResult(
        parsed_df=df,
        cleaned_df=cleaned_df,
        messages=messages,
        has_liquid=has_liquid,
    )
