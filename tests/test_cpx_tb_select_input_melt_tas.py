from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aims4pt.model_tools.CpxTBSelect import workflow_thermobarometry


class DummyDeviation:
    def predict_deviation(self, input_cpx, input_liq):
        return np.ones(len(input_cpx))


class DummyModel:
    T_P = "P"
    y_min = 0.0
    y_max = 100.0
    deviation_function = DummyDeviation()

    def __init__(self, model_name, rock_types=None):
        self.model_name = model_name
        if rock_types is not None:
            self.rock_types = rock_types

    def predict(self, input_cpx, input_liq):
        return np.full(len(input_cpx), 10.0)


@pytest.fixture
def input_cpx():
    return pd.DataFrame({"MgO_Cpx": [10.0, 11.0]}, index=["s1", "s2"])


def test_backward_compatible_calls_work(input_cpx):
    wf = workflow_thermobarometry([DummyModel("model", ["Basalt"])])

    wf.calculation(input_cpx)
    pred = wf.predict(input_cpx)

    assert wf.petrological_check_df["model"].all()
    assert pred.index.equals(input_cpx.index)


def test_no_liquid_and_no_assumed_tas_skips_tas_check(input_cpx):
    wf = workflow_thermobarometry([
        DummyModel("basalt_model", ["Basalt"]),
        DummyModel("rhyolite_model", ["Rhyolite"]),
    ])

    wf.calculation(input_cpx)

    assert wf.petrological_check_df.all().all()


def test_assumed_tas_passes_and_fails_by_model_rock_types(input_cpx):
    wf = workflow_thermobarometry([
        DummyModel("basaltic_andesite_model", ["Basalt", "Basaltic Andesite"]),
        DummyModel("rhyolite_model", ["Rhyolite"]),
    ])

    wf.calculation(input_cpx, input_melt_TAS=["Basaltic Andesite"])
    wf.decision()

    assert wf.petrological_check_df["basaltic_andesite_model"].all()
    assert not wf.petrological_check_df["rhyolite_model"].any()
    assert (wf.failure_reason_df["rhyolite_model"] == "TAS ood").all()


def test_assumed_tas_multi_field_passes_if_any_field_overlaps(input_cpx):
    wf = workflow_thermobarometry([
        DummyModel("andesite_model", ["Andesite"]),
    ])

    wf.calculation(input_cpx, input_melt_TAS=["Basaltic Andesite", "Andesite"])

    assert wf.petrological_check_df["andesite_model"].all()


def test_repeated_assumed_tas_fields_are_deduplicated(input_cpx):
    wf = workflow_thermobarometry([
        DummyModel("basalt_model", ["Basalt"]),
    ])

    wf.calculation(input_cpx, input_melt_TAS=["Basalt", "Basalt", "Basalt", "Basalt"])

    assert wf.petrological_check_df["basalt_model"].all()


@pytest.mark.parametrize(
    "bad_input, error_type",
    [
        ("Basalt", TypeError),
        ([], ValueError),
        (["Basalt", "Andesite", "Dacite", "Rhyolite"], ValueError),
        (["Unknown"], ValueError),
    ],
)
def test_input_melt_tas_validation(input_cpx, bad_input, error_type):
    wf = workflow_thermobarometry([DummyModel("model", ["Basalt"])])

    with pytest.raises(error_type):
        wf.calculation(input_cpx, input_melt_TAS=bad_input)


def test_unknown_input_melt_tas_error_lists_allowed_options(input_cpx):
    wf = workflow_thermobarometry([DummyModel("model", ["Basalt"])])

    with pytest.raises(ValueError, match="Allowed options"):
        wf.calculation(input_cpx, input_melt_TAS=["Unknown"])
