from __future__ import annotations

from scripts.audit_project7_matched_static_contract import (
    AUDIT_CONTRACT,
    _clock_contract,
    _clock_projection,
)


def test_clock_projection_excludes_path_only_identity() -> None:
    clock = {
        "inp_path": r"C:\event.inp",
        "simulation_start": "2022-08-10T22:00:00",
        "first_positive_rainfall": "2022-08-11T00:00:00",
        "last_positive_rainfall": "2022-08-11T01:40:00",
        "rainfall_interval_minutes": 5.0,
        "effective_warmup_minutes": 120.0,
    }
    assert _clock_projection(clock) == {
        "simulation_start": "2022-08-10T22:00:00",
        "first_positive_rainfall": "2022-08-11T00:00:00",
        "last_positive_rainfall": "2022-08-11T01:40:00",
        "rainfall_interval_minutes": 5.0,
        "effective_warmup_minutes": 120.0,
    }


def test_static_audit_has_versioned_contract() -> None:
    assert AUDIT_CONTRACT == "PROJECT7_MATCHED_BASELINE_STATIC_CONTRACT_AUDIT_V1"


def test_duration_is_diagnostic_not_common_clock_contract() -> None:
    short = {
        "simulation_start": "2022-08-10T22:00:00",
        "first_positive_rainfall": "2022-08-11T00:00:00",
        "last_positive_rainfall": "2022-08-11T01:40:00",
        "rainfall_interval_minutes": 5.0,
        "effective_warmup_minutes": 120.0,
    }
    long = {**short, "last_positive_rainfall": "2022-08-11T04:40:00"}
    assert _clock_projection(short) != _clock_projection(long)
    assert _clock_contract(short) == _clock_contract(long)
