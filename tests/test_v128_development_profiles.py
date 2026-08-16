from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import rtc.v128_preflight as v128_preflight
from rtc.development_profile_v128 import (
    apply_profile_to_design,
    deterministic_subset,
    get_execution_profile,
    profile_groups,
)
from rtc.spatial_diagnostics_v128 import (
    action_effect_by_distance,
    actuator_node_hops,
    error_by_distance,
    nearest_source_hops,
)
from rtc.stage_checkpoint_v128 import (
    load_stage_checkpoint_v128,
    save_stage_checkpoint_v128,
)
from rtc.step2_gradient_audit_v128_dev import _candidate_direction_from_settings
from rtc.v128_control_profile import build_v128_control_training_design


def test_profiles_preserve_full_contract_and_make_debug_explicitly_nonfinal() -> None:
    base = build_v128_control_training_design(
        hydraulic_branch_chunk=4,
        rollout_candidates_per_group=2,
        objective_candidate_chunk=2,
        evaluation_branch_chunk=4,
    )
    smoke = get_execution_profile("smoke")
    dev = get_execution_profile("dev")
    full = get_execution_profile("full")
    assert not smoke.scientific_claim_allowed and not smoke.final_checkpoint_allowed
    assert not dev.scientific_claim_allowed and not dev.final_checkpoint_allowed
    assert full.scientific_claim_allowed and full.final_checkpoint_allowed

    full_design = apply_profile_to_design(base, full)
    assert full_design.hydraulic_epochs == 4
    assert full_design.teacher_stride == 4
    assert full_design.rollout_horizons == (12, 24)
    assert full_design.rollout_candidates_per_group == 2
    assert full_design.objective_epochs == 3
    assert full_design.informative_pair_reference_fraction == 0.0


def test_debug_subsets_are_deterministic_and_do_not_depend_on_input_order() -> None:
    names = [f"D2::group-{i}" for i in range(20)]
    a = deterministic_subset(names, 5, salt="smoke:d2")
    b = deterministic_subset(list(reversed(names)), 5, salt="smoke:d2")
    assert a == b
    assert len(a) == len(set(a)) == 5

    selected = profile_groups(
        get_execution_profile("smoke"),
        fit_d2=names,
        fit_d3=[x.replace("D2::", "D3::") for x in names],
        hold_d2=names,
        hold_d3=[x.replace("D2::", "D3::") for x in names],
        d4_fit=[x.replace("D2::", "D4::") for x in names],
        d4_audit=[x.replace("D2::", "D4A::") for x in names],
    )
    assert {key: len(value) for key, value in selected.items()} == {
        "fit_d2": 4,
        "fit_d3": 4,
        "hold_d2": 2,
        "hold_d3": 2,
        "d4_fit": 4,
        "d4_audit": 2,
    }


def test_nonfinal_stage_checkpoint_is_resumeable_but_contract_locked(tmp_path: Path) -> None:
    graph_file = tmp_path / "graph.npz"
    graph_file.write_bytes(b"graph")
    model = torch.nn.Linear(3, 2)
    lineage = {"data": "abc"}
    design = {"epochs": 1}
    path = save_stage_checkpoint_v128(
        tmp_path / "stage_a.pt",
        model=model,
        completed_stage="stage_a",
        profile="smoke",
        graph_path=graph_file,
        lineage=lineage,
        training_design=design,
        history={"hydraulic": [{"loss": 1.0}]},
    )
    restored = torch.nn.Linear(3, 2)
    payload = load_stage_checkpoint_v128(
        path,
        model=restored,
        expected_profile="smoke",
        graph_path=graph_file,
        expected_lineage=lineage,
        expected_training_design=design,
    )
    assert payload["completed_stage"] == "stage_a"
    assert payload["scientific_claim_allowed"] is False
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b)

    with pytest.raises(ValueError, match="execution profile"):
        load_stage_checkpoint_v128(
            path,
            model=restored,
            expected_profile="dev",
            graph_path=graph_file,
            expected_lineage=lineage,
            expected_training_design=design,
        )


def test_spatial_hop_diagnostics_detect_near_and_far_error() -> None:
    # chain 0-1-2-3-4-5
    edge = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    dist = nearest_source_hops(edge, 6, [0])
    np.testing.assert_array_equal(dist, np.arange(6, dtype=np.int32))

    truth = np.zeros((2, 6), dtype=np.float32)
    pred = np.zeros_like(truth)
    pred[:, 4:] = 2.0
    metrics = error_by_distance(truth, pred, dist)
    assert metrics["1-3"]["rmse"] == 0.0
    assert metrics["4-6"]["rmse"] > 0.0


def test_action_effect_distance_metrics_use_actuator_endpoints() -> None:
    edge = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    hops = actuator_node_hops(edge, 6, [0], [1])
    assert hops.shape == (1, 6)
    np.testing.assert_array_equal(hops[0], np.asarray([0, 0, 1, 2, 3, 4]))

    true_ref = np.zeros(6, dtype=np.float32)
    pred_ref = np.zeros(6, dtype=np.float32)
    true_cand = np.asarray([[0, 0, -2, -3, -4, -5]], dtype=np.float32)
    pred_cand = np.asarray([[0, 0, -2, -3, +4, +5]], dtype=np.float32)
    metrics = action_effect_by_distance(
        true_ref,
        true_cand,
        pred_ref,
        pred_cand,
        [0],
        hops,
        effect_floor_m3=1.0,
    )
    assert metrics["1-3"]["informative_sign_total"] >= 2
    assert metrics["1-3"]["effect_sign_accuracy"] < 1.0


def test_settings_derived_gradient_direction_preserves_exact_d2_pulse() -> None:
    reference = np.full((6, 3), 0.5, dtype=np.float32)
    candidate = reference.copy()
    candidate[2:5, 1] = np.asarray([0.4, 0.3, 0.4], dtype=np.float32)
    direction, peak_step, active_steps = _candidate_direction_from_settings(
        reference,
        candidate,
        expected_actuator_index=1,
    )
    assert peak_step == pytest.approx(0.2)
    assert active_steps == 3
    np.testing.assert_allclose(direction[:, 0], 0.0)
    np.testing.assert_allclose(direction[:, 2], 0.0)
    np.testing.assert_allclose(
        direction[:, 1],
        np.asarray([0.0, 0.0, -0.5, -1.0, -0.5, 0.0]),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_settings_derived_gradient_direction_rejects_multi_actuator_candidate() -> None:
    reference = np.zeros((4, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[1:3, 0] = 0.2
    candidate[2, 2] = -0.1
    with pytest.raises(ValueError, match="changes 2 actuators"):
        _candidate_direction_from_settings(
            reference,
            candidate,
            expected_actuator_index=0,
        )


def test_preflight_applies_requested_v128_matmul_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.npz"
    graph_path.write_bytes(b"graph")

    class Graph:
        actuator_ids = tuple(f"A{i}" for i in range(109))

    class Envelope:
        source = "IDEALIZED_DEFAULT_0P5_PER_10MIN"
        semantic_sha256 = "semantic"
        is_idealized_default = True

    monkeypatch.setattr(v128_preflight, "_load_graph", lambda _: Graph())
    monkeypatch.setattr(
        v128_preflight,
        "idealized_engineering_envelope_v128",
        lambda _: Envelope(),
    )
    monkeypatch.setenv("RTC_V128_MATMUL_PRECISION", "high")

    before = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        payload = v128_preflight.inspect_v128_preflight(
            graph_path=graph_path,
            device_text="cpu",
        )
        assert payload["hardware"]["float32_matmul_precision_before"] == "highest"
        assert payload["hardware"]["float32_matmul_precision"] == "high"
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(before)


def test_current_step2_entrypoint_rejects_unbounded_raw_torch_trace() -> None:
    text = Path("scripts/run_step2_current.py").read_text(encoding="utf-8")
    assert '"--torch-profiler" in sys.argv[1:]' in text
    assert "disables raw --torch-profiler trace export" in text
    assert "TRAINING_TELEMETRY.jsonl" in text


def test_dev_extra_installs_psutil_for_resource_telemetry() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"psutil>=5.9"' in text
