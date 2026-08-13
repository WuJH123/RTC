"""Regression tests for evidence-bound Step2 decision assembly."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_step2_current_decision.py"
    spec = importlib.util.spec_from_file_location("step2_current_decision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ladder(decision: str) -> dict:
    keys = (
        "delta_depth_m_skill_vs_zero",
        "delta_flood_m3s_skill_vs_zero",
        "delta_storage_m3_skill_vs_zero",
        "delta_managed_flow_m3s_skill_vs_zero",
    )
    return {
        "decision": {"decision": decision},
        "ladder": {
            level: {"overall": {key: 0.1 for key in keys}}
            for level in ("A_BOUNDARY", "B_PREDICTED_REFERENCE_TRAJECTORY", "C_ORACLE_REFERENCE_TRAJECTORY")
        },
    }


def test_markov_label_does_not_overrule_positive_local_signal() -> None:
    module = _module()
    root, _, markdown = module.build_decision(
        {"reference_candidate_pairing": {}, "target_recomputation_from_raw_compacts": {}, "hydraulic_effect_identifiability": {}},
        {},
        {"baselines": {"local_mlp": {"holdout": {"depth": {"skill_vs_zero": 0.2}, "flood": {"skill_vs_zero": -0.1}, "storage": {"skill_vs_zero": 0.3}, "managed_flow": {"skill_vs_zero": -0.1}}}}},
        _ladder("MARKOV_INSUFFICIENCY_SUPPORTED"),
    )
    assert root["new_swmm_authorized"] is False
    assert "LOCAL_D2_SIGNAL_PREVENTS_A_GLOBAL_INSUFFICIENCY_CLAIM" in root["primary_remaining_bottleneck"]
    assert "MARKOV_INSUFFICIENCY_SUPPORTED" in markdown


def test_predicted_reference_success_authorizes_only_development_v9() -> None:
    module = _module()
    root, _, _ = module.build_decision(
        {"reference_candidate_pairing": {}, "target_recomputation_from_raw_compacts": {}, "hydraulic_effect_identifiability": {}},
        {},
        {"baselines": {"local_mlp": {"holdout": {}}}},
        _ladder("PREDICTED_REFERENCE_TRAJECTORY_SUFFICIENT"),
    )
    assert root["formal_v9_authorized"] is True
    assert root["new_swmm_authorized"] is False
