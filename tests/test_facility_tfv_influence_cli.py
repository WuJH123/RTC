from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_facility_tfv_influence_current.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("facility_tfv_influence_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_facility_tables_keeps_untested_facilities_visible() -> None:
    cli = _load_script_module()
    by_rain = pd.DataFrame(
        [
            {
                "rainfall_group": "storm_a",
                "actuator_id": "A",
                "tested_pairs": 2,
                "tested_checkpoints": 1,
                "tested_events": 1,
                "beneficial_pairs": 1,
                "harmful_pairs": 0,
                "below_threshold_pairs": 1,
                "beneficial_fraction": 0.5,
                "harmful_fraction": 0.0,
                "delayed_beneficial_pairs": 1,
                "evidence_class": "BENEFICIAL_IN_SAMPLED_ACTIONS",
                "has_sampled_control_value": True,
            }
        ]
    )
    global_facility = by_rain.drop(columns=["rainfall_group"]).copy()
    rain, global_out = cli._complete_facility_tables(
        by_rain,
        global_facility,
        rainfalls=["storm_a"],
        actuator_ids=["A", "B"],
    )
    assert len(rain) == 2
    missing = rain[rain["actuator_id"] == "B"].iloc[0]
    assert int(missing["tested_pairs"]) == 0
    assert missing["evidence_class"] == "UNTESTED_SINGLE_ACTUATOR"
    assert bool(missing["has_sampled_control_value"]) is False
    assert len(global_out) == 2
