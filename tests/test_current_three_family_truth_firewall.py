from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "scripts" / "run_policy_direct_tfv_base_hybrid_parent_current.py"
CAPTURE = ROOT / "scripts" / "capture_direct_tfv_policy_return_context_current.py"
DESIGN = ROOT / "scripts" / "design_direct_tfv_policy_return_portfolio_current.py"
QUERY = ROOT / "scripts" / "run_direct_tfv_policy_return_query_current.py"


def test_parent_regression_fix_remains_gradient_off_three_family() -> None:
    text = PARENT.read_text(encoding="utf-8")
    assert 'lineage.get("projected_gradient_h10_enabled") is not False' in text
    assert 'lineage.get("projected_gradient_cli_knobs_affect_current_policy") is not False' in text
    assert 'lineage.get("candidate_portfolio_family_count_max", -1)) != 3' in text
    assert 'int(lineage.get("supervisory_control_dimension", -1)) != 82' in text
    assert 'int(lineage.get("model_action_channel_count", -1)) != 109' in text


def test_query_truth_firewall_rejects_fourth_family_and_has_zero_swmm_preflight() -> None:
    text = QUERY.read_text(encoding="utf-8")
    assert "not 1 <= len(rows) <= 3" in text
    assert "not 1 <= len(rows) <= 4" not in text
    assert "CURRENT_THREE_FAMILY_SOURCES" in text
    assert "SUPPORT_CONSTRAINED_GRADIENT_H10" not in text
    assert 'payload.get("projected_gradient_online") is not False' in text
    assert 'payload.get("lbfgsb_used") is not False' in text
    assert '"--preflight-only"' in text
    assert '"swmm_truth_started": False' in text
    assert "audit_target_write_readback_v127" in text
    assert '"policy_return_development_diagnostic"' in text
    assert '"eligible_for_learning_dataset"' in text


def test_capture_to_design_to_query_is_cryptographically_bound() -> None:
    capture = CAPTURE.read_text(encoding="utf-8")
    design = DESIGN.read_text(encoding="utf-8")
    query = QUERY.read_text(encoding="utf-8")
    required = (
        "source_inp_sha256",
        "parent_decisions_sha256",
        "asset_manifest_sha256",
        "graph_sha256",
        "step2_checkpoint_sha256",
        "sequence_support_sha256",
        "supervisory_control_sha256",
        "continuation_policy_sha256",
        "recorded_prefix_action_sha256",
    )
    for key in required:
        assert key in capture
        assert key in design
        assert key in query
    assert "context_npz_sha256" in design
    assert "context_npz_sha256" in query
    assert "step2_checkpoint_sha256=step2_sha" in design


def test_current_design_accepts_one_to_three_distinct_candidates_without_gradient() -> None:
    text = DESIGN.read_text(encoding="utf-8")
    assert "include_projected_gradient_ablation=False" in text
    assert "if not rows:" in text
    assert "if len(rows) > 3:" in text
    assert "fewer than two" not in text.lower()
    assert "candidate.source not in CURRENT_THREE_FAMILY_SOURCES" in text
