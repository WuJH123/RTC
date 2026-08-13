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


def test_current_local_baseline_schema_is_used_for_learnability() -> None:
    """The report must consume the emitted TrainInternalHoldout_D2 schema."""
    module = _module()
    root, _, _ = module.build_decision(
        {"reference_candidate_pairing": {}, "target_recomputation_from_raw_compacts": {}, "hydraulic_effect_identifiability": {}},
        {},
        {
            "baselines": {
                "local_mlp": {
                    "TrainInternalHoldout_D2": {
                        "channels": {
                            "delta_depth_m": {"skill_vs_zero": 0.2},
                            "delta_flood_m3s": {"skill_vs_zero": -0.1},
                            "delta_storage_m3": {"skill_vs_zero": 0.3},
                            "delta_managed_flow_m3s": {"skill_vs_zero": -0.1},
                        }
                    }
                }
            }
        },
        _ladder("MARKOV_INSUFFICIENCY_SUPPORTED"),
    )
    learnability = next(item for item in root["findings"] if item["root_cause"] == "ALL_ACTION_EFFECT_DATA_UNLEARNABLE")
    assert learnability["evidence"]["local_mlp_holdout_positive_channels"] == ["depth", "storage"]


def test_decision_includes_history_graph_and_physical_controls_without_stale_next_step() -> None:
    module = _module()
    root, before_after, markdown = module.build_decision(
        {"reference_candidate_pairing": {}, "target_recomputation_from_raw_compacts": {}, "hydraulic_effect_identifiability": {}},
        {},
        {"baselines": {"local_mlp": {"holdout": {"depth": {"skill_vs_zero": 0.2}}}}},
        _ladder("CURRENT_SNAPSHOT_MODEL_CONTRACT_UNSUPPORTED"),
        fair_local_control={"lineage": {"git_head": "fair"}, "baselines": {"event_scheduled_local_mlp": {}}},
        history_ladder={"lineage": {"git_head": "history"}, "arms": {"B0_CURRENT_SNAPSHOT": {}}},
        graph_audit={"lineage": {"git_head": "graph"}, "authoritative_absolute_effect_mass": {"h8": {}}},
        physical_edge={
            "lineage": {"git_head": "physical"},
            "metrics": {
                "TrainInternalHoldout_D2": {
                    "overall": {
                        "delta_depth_m_skill_vs_zero": -0.1,
                        "delta_flood_m3s_skill_vs_zero": -0.1,
                        "delta_storage_m3_skill_vs_zero": -0.1,
                        "delta_managed_flow_m3s_skill_vs_zero": -0.1,
                    }
                }
            },
        },
    )
    graph = next(item for item in root["findings"] if item["root_cause"] == "GRAPH_REPRESENTATION_FAILURE")
    assert graph["status"] == "SUPPORTED"
    assert root["artifact_provenance"]["physical_edge"]["git_head"] == "physical"
    physical = next(item for item in root["findings"] if item["root_cause"] == "STATIC_DIRECTED_EDGE_PHYSICS")
    assert physical["status"] == "INSUFFICIENT"
    assert before_after["post_ladder_controls"]["history_ladder"]["lineage"]["git_head"] == "history"
    assert "RUN_EXISTING_HISTORY_LADDER" not in markdown


def test_markdown_distinguishes_full_network_physical_control_from_local_evidence() -> None:
    """A concise decision must not overclaim endpoint-local evidence as a network win."""
    module = _module()
    _, _, markdown = module.build_decision(
        {"reference_candidate_pairing": {}, "target_recomputation_from_raw_compacts": {}, "hydraulic_effect_identifiability": {}},
        {},
        {"baselines": {"local_mlp": {"holdout": {"depth": {"skill_vs_zero": 0.2}}}}},
        _ladder("CURRENT_SNAPSHOT_MODEL_CONTRACT_UNSUPPORTED"),
        fair_local_control={"lineage": {"git_head": "fair"}, "baselines": {"event_scheduled_local_mlp": {}}},
        history_ladder={"lineage": {"git_head": "history"}, "arms": {"B0_CURRENT_SNAPSHOT": {}}},
        graph_audit={"lineage": {"git_head": "graph"}, "authoritative_absolute_effect_mass": {"h8": {}}},
        physical_edge={
            "lineage": {"git_head": "physical"},
            "metrics": {
                "TrainInternalHoldout_D2": {
                    "overall": {
                        "delta_depth_m_skill_vs_zero": -0.1,
                        "delta_flood_m3s_skill_vs_zero": -0.1,
                        "delta_storage_m3_skill_vs_zero": -0.1,
                        "delta_managed_flow_m3s_skill_vs_zero": -0.1,
                    }
                }
            },
        },
    )
    assert "endpoint-local evidence is not a full-network success" in markdown
    assert "static directed conduit control" in markdown
    assert "Artifact provenance" in markdown
