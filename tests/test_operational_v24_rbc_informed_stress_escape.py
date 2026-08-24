from __future__ import annotations

from pathlib import Path

from rtc.direct_tfv_operational_v24_runtime import (
    OPERATIONAL_V24_RUNTIME_CONTRACT,
    V24_HYBRID_ADMISSION_CONTRACT,
    V24_STRESS_ESCAPE_BLEND_MIN,
    V24_STRESS_ESCAPE_Q75,
    hydraulic_stress_escape_active,
)


def test_v24_escape_uses_frozen_auto_rbc_high_fill_threshold() -> None:
    assert V24_STRESS_ESCAPE_Q75 == 0.75
    assert V24_STRESS_ESCAPE_BLEND_MIN == 0.40
    assert hydraulic_stress_escape_active(0.75, 0.40)
    assert hydraulic_stress_escape_active(0.90, 1.00)
    assert not hydraulic_stress_escape_active(0.749999, 1.00)
    assert not hydraulic_stress_escape_active(0.90, 0.399999)


def test_v24_escape_threshold_validation_is_fail_closed() -> None:
    try:
        hydraulic_stress_escape_active(0.8, 0.8, stress_threshold=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative stress threshold must fail closed")
    try:
        hydraulic_stress_escape_active(0.8, 0.8, blend_minimum=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("blend minimum above one must fail closed")


def test_v24_is_new_development_lane_not_a_v23_rewrite() -> None:
    assert "V24" in OPERATIONAL_V24_RUNTIME_CONTRACT
    assert "HYDRAULIC_STRESS_ESCAPE" in V24_HYBRID_ADMISSION_CONTRACT
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "src" / "rtc" / "direct_tfv_operational_v24_runtime.py").read_text(
        encoding="utf-8"
    )
    runner = (root / "scripts" / "run_policy_direct_tfv_operational_v24_development.py").read_text(
        encoding="utf-8"
    )
    benchmark = (root / "scripts" / "run_project7_operational_benchmark5_v24_development.py").read_text(
        encoding="utf-8"
    )
    assert "build_operational_v23_controller" in runtime
    assert "historical_v23_policy_mutated\": False" in runtime
    assert "learned_v21_boundary_passed=false" in runtime
    assert '"formal_evidence": False' in runner
    assert '"historical_v23_evidence_mutated": False' in runner
    assert "run_policy_direct_tfv_operational_v24_development.py" in benchmark


def test_v24_escape_does_not_claim_surrogate_benefit() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "src" / "rtc" / "direct_tfv_operational_v24_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "causal_hydraulic_stress_escape_not_surrogate_calibrated" in runtime
    assert "predicted_delta_tfv_m3=0.0" in runtime
    assert "future realized rainfall" in runtime
    assert "online SWMM candidate" in runtime
