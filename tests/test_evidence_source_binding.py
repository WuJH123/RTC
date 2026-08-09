from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtc.acceptance_gate import create_gate
from rtc.code_contract import rtc_source_tree_sha256


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_current_implementation_metrics_flow_into_acceptance_gate(tmp_path: Path) -> None:
    metrics = _write(
        tmp_path / "metrics.json",
        {
            "contract": "STEP1_HELDOUT_ACCEPTANCE_V3_GROUP_BALANCED_TIME_LOCKED",
            "rtc_source_tree_sha256": rtc_source_tree_sha256(),
            "aggregation": "equal_weight_per_rainfall_group",
            "model_sha256": "a" * 64,
            "metrics": {"unobserved_depth_nse": 0.9},
        },
    )
    contract = _write(
        tmp_path / "contract.json",
        {
            "step1": {
                "minimum": {"unobserved_depth_nse": 0.5},
                "maximum": {},
            }
        },
    )
    out = tmp_path / "gate.json"
    result = create_gate(
        metrics_path=metrics,
        contract_path=contract,
        section="step1",
        output_path=out,
    )
    assert result["passed"] is True
    assert result["rtc_source_tree_sha256"] == rtc_source_tree_sha256()
    assert result["source_metrics_contract"] == (
        "STEP1_HELDOUT_ACCEPTANCE_V3_GROUP_BALANCED_TIME_LOCKED"
    )
    assert result["source_metrics_aggregation"] == "equal_weight_per_rainfall_group"


def test_incompatible_implementation_metrics_fail_closed(tmp_path: Path) -> None:
    metrics = _write(
        tmp_path / "metrics.json",
        {
            "contract": "D2_SWMM_TFV_GRADIENT_METRICS_V4_BOUND_AWARE_GROUP_BALANCED",
            "rtc_source_tree_sha256": "0" * 64,
            "aggregation": "equal_weight_per_rainfall_group",
            "metrics": {"tfv_gradient_sign_accuracy": 1.0},
        },
    )
    contract = _write(
        tmp_path / "contract.json",
        {
            "gradient": {
                "minimum": {"tfv_gradient_sign_accuracy": 0.5},
                "maximum": {},
            }
        },
    )
    with pytest.raises(ValueError, match="different RTC source tree"):
        create_gate(
            metrics_path=metrics,
            contract_path=contract,
            section="gradient",
            output_path=tmp_path / "gate.json",
        )
