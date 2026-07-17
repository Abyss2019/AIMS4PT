"""Stable model abbreviations used by reports and report figures."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable


PRESSURE_MODEL_ABBREVIATIONS = {
    "Putirka, 2008 eq32d_T; eq32a_P": "Pu08_32a",
    "Putirka, 2008 eq32d_T; eq32b_P": "Pu08_32b",
    "Putirka, 2008 eq32d_T_hydrousVersion; eq32b_P": "Pu08_32b",
    "Putirka, 2008 eq33_T; eq31_P": "Pu08_31",
    "Putirka, 2008 eq33_T; eq31_P (cpx_liq)": "Pu08_31",
    "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P": "NP17",
    "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P (cpx_liq)": "NP17",
    "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P (cpx_liq) (cpx_liq)": "NP17",
}

TEMPERATURE_MODEL_ABBREVIATIONS = {
    "Putirka, 2008 eq32d_T; eq32a_P": "Pu08_32d",
    "Putirka, 2008 eq32d_T; eq32b_P": "Pu08_32d",
    "Putirka, 2008 eq32d_T_hydrousVersion; eq32b_P": "Pu08_32d",
    "Putirka, 2008 eq33_T; eq31_P": "Pu08_33",
    "Putirka, 2008 eq33_T; eq31_P (cpx_liq)": "Pu08_33",
}

FAMILY_ABBREVIATIONS = (
    ("petrelli", "Pet20"),
    ("wang", "Wan21"),
    ("higgins", "Hig21"),
    ("jorgenson", "Jor22"),
    ("chicchi", "Chi23"),
)


def model_abbreviation(model_name: object, target: str) -> str:
    """Return an official or deterministic compact model abbreviation."""
    text = str(model_name)
    target_key = str(target).upper()
    if target_key not in {"P", "T"}:
        raise ValueError("target must be 'P' or 'T'.")

    if text.strip().lower() == "no selected model":
        return "No selected model"

    mapping = (
        PRESSURE_MODEL_ABBREVIATIONS
        if target_key == "P"
        else TEMPERATURE_MODEL_ABBREVIATIONS
    )
    if text in mapping:
        return mapping[text]

    normalized = re.sub(r"\s+", "", text.lower())
    if "putirka" in normalized and "2008" in normalized:
        if target_key == "P":
            for token, abbreviation in (
                ("eq32a", "Pu08_32a"),
                ("eq32b", "Pu08_32b"),
                ("eq31", "Pu08_31"),
            ):
                if token in normalized:
                    return abbreviation
        else:
            if "eq32d" in normalized:
                return "Pu08_32d"
            if "eq33" in normalized:
                return "Pu08_33"
    if "neave" in normalized and "putirka" in normalized:
        return "NP17"
    if ("agreda" in normalized or "greda" in normalized) and "2024" in normalized:
        return "AgL24"
    for family, abbreviation in FAMILY_ABBREVIATIONS:
        if family in normalized:
            return abbreviation

    match = re.search(r"([A-Za-z]+).*?(\d{4})", text)
    if match:
        return f"{match.group(1)[:3].title()}{match.group(2)[-2:]}"
    compact = re.sub(r"[^A-Za-z0-9]+", "", text)
    return compact[:14] or "Model"


def _phase_suffix(model_name: object) -> str:
    normalized = str(model_name).lower()
    if "cpx_only" in normalized:
        return "cpx"
    if "cpx_liq" in normalized or "liquid" in normalized:
        return "liq"
    return ""


def model_abbreviation_lookup(
    model_names: Iterable[object],
    target: str,
    *,
    disambiguate: bool = False,
) -> dict[object, str]:
    """Build abbreviations and optionally disambiguate duplicate plot labels."""
    names = list(model_names)
    abbreviations = [model_abbreviation(name, target) for name in names]
    counts = Counter(abbreviations)
    lookup: dict[object, str] = {}
    collision_groups: dict[str, list[object]] = {}
    for name, abbreviation in zip(names, abbreviations):
        if counts[abbreviation] > 1:
            collision_groups.setdefault(abbreviation, []).append(name)

    collision_labels: dict[object, str] = {}
    if disambiguate:
        for abbreviation, group_names in collision_groups.items():
            phase_suffixes = [_phase_suffix(name) for name in group_names]
            phase_counts = Counter(phase_suffixes)
            if len(phase_counts) == 1:
                for index, name in enumerate(group_names, start=1):
                    collision_labels[name] = f"{abbreviation}-{index}"
                continue

            phase_seen: Counter[str] = Counter()
            for name, suffix in zip(group_names, phase_suffixes):
                phase_seen[suffix] += 1
                label = f"{abbreviation}-{suffix}" if suffix else abbreviation
                if phase_counts[suffix] > 1:
                    label = f"{label}-{phase_seen[suffix]}"
                collision_labels[name] = label

    for name, abbreviation in zip(names, abbreviations):
        lookup[name] = collision_labels.get(name, abbreviation)
    return lookup


__all__ = ["model_abbreviation", "model_abbreviation_lookup"]
