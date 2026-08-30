from __future__ import annotations

from scripts.audit_project7_runtime_truth_alignment import (
    _runtime_action_class,
    _runtime_action_hash,
)


def test_runtime_action_class_prefers_versioned_actual_class() -> None:
    diagnostics = {
        "calibrated_runtime_action_class": "HOLD",
        "v28_action_class": "ACTION",
    }
    assert _runtime_action_class(diagnostics) == "ACTION"


def test_runtime_action_hash_uses_selected_supported_target() -> None:
    expected = "a" * 64
    row = {
        "settings": {"A1": 0.0},
        "diagnostics": {
            "v28_candidate_telemetry": [
                {
                    "candidate_selected": True,
                    "candidate_source": "TYPE_AWARE_HYDRAULIC_PRESSURE",
                    "supported_target_sha256": expected,
                }
            ]
        },
    }
    assert _runtime_action_hash(row, ("A1",)) == expected
