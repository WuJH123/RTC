from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.direct_tfv_v27_auto_rbc_shadow import build_auto_rbc_shadow_proposal
from rtc.direct_tfv_v27_decision_value import (
    FittedV27DecisionValueModel,
    fit_v27_decision_value_model,
    pairwise_rank_accuracy,
)
from rtc.project7_v26_historical_supervision import ContextResolver, HistoricalCandidateRecord, causal_context_sha256
from rtc.project7_v27_context_recovery import recover_missing_contexts_v27


def test_v27_runtime_ranking_is_not_destroyed_by_reporting_clip() -> None:
    model = FittedV27DecisionValueModel(
        feature_mean=np.zeros(1),
        feature_scale=np.ones(1),
        weight=np.asarray([-1.0]),
        intercept=-9.0,
        target_scale_m3=100.0,
        ridge=1.0,
        pairwise_weight=1.0,
    )
    x = np.asarray([[0.0], [1.0]])
    latent = model.latent_numpy(x)
    reported = model.predict_m3_numpy(x)
    assert latent.tolist() == [-9.0, -10.0]
    assert reported[0] == reported[1]  # both reporting values hit the same clip
    assert int(np.argmin(latent)) == 1  # deployment ranking still distinguishes them


def test_v27_pairwise_training_uses_group_cv_and_validation_without_test() -> None:
    train_x = np.asarray(
        [[-2.0], [-1.0], [1.0], [2.0], [-1.5], [1.5], [-0.5], [0.5]],
        dtype=np.float64,
    )
    train_y = np.asarray([-20.0, -5.0, 5.0, 20.0, -15.0, 15.0, -4.0, 4.0])
    units = ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"]
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]
    val_x = np.asarray([[-1.2], [1.2], [-0.7], [0.7]], dtype=np.float64)
    val_y = np.asarray([-10.0, 10.0, -6.0, 6.0])
    model, report = fit_v27_decision_value_model(
        train_x,
        train_y,
        units,
        groups,
        val_x,
        val_y,
        ["v1", "v1", "v2", "v2"],
        cv_folds=2,
        ridge_grid=(0.1, 1.0),
        pairwise_weight_grid=(0.0, 1.0),
        validation_shortlist_size=2,
    )
    latent = model.latent_numpy(val_x)
    pair = pairwise_rank_accuracy(latent, val_y, ["v1", "v1", "v2", "v2"])
    assert pair["pairwise_rank_accuracy"] >= 0.5
    assert report["test_used_for_training_or_model_selection"] is False
    assert report["cv_folds"] == 2


def _context(value: float) -> dict[str, np.ndarray]:
    return {
        "current_state": np.full((3, 4), value, dtype=np.float32),
        "rainfall_scenarios": np.full((2, 3, 3, 1), value, dtype=np.float32),
        "active_target": np.full(109, value, dtype=np.float32),
        "previous_actuator_flow": np.full(109, value, dtype=np.float32),
    }


def test_v27_context_recovery_uses_precise_prefix_to_resolve_ambiguous_query(tmp_path) -> None:
    c1 = _context(0.1)
    c2 = _context(0.2)
    target = np.zeros(109, dtype=np.float32)
    ref1 = HistoricalCandidateRecord(
        row={"query_set_id": "q", "prefix_sha256": "a" * 64, "candidate_target": target.tolist()},
        source_path=tmp_path / "ref1.jsonl",
        source_index=0,
        embedded_context=c1,
        embedded_target=target,
    )
    ref2 = HistoricalCandidateRecord(
        row={"query_set_id": "q", "prefix_sha256": "b" * 64, "candidate_target": target.tolist()},
        source_path=tmp_path / "ref2.jsonl",
        source_index=0,
        embedded_context=c2,
        embedded_target=target,
    )
    missing = HistoricalCandidateRecord(
        row={"query_set_id": "q", "prefix_sha256": "b" * 64, "candidate_target": target.tolist()},
        source_path=tmp_path / "missing.jsonl",
        source_index=0,
    )
    report = recover_missing_contexts_v27(
        [missing],
        resolver=ContextResolver(study_root=None),
        references=[ref1, ref2],
    )
    assert report["repaired"] == 1
    assert report["ambiguous"] == 0
    assert missing.embedded_context is not None
    assert causal_context_sha256(missing.embedded_context) == causal_context_sha256(c2)


def _fake_graph() -> SimpleNamespace:
    node_count = 110
    actuator_count = 109
    static_names = ("max_depth_m", "storage_capacity_m3")
    static = np.column_stack((np.ones(node_count), np.full(node_count, 100.0)))
    physics_names = ("is_pump", "is_orifice", "is_weir", "is_outlet", "min_setting", "max_setting")
    physics = np.zeros((actuator_count, len(physics_names)), dtype=np.float64)
    for index in range(actuator_count):
        physics[index, index % 4] = 1.0
        physics[index, 4] = 0.0
        physics[index, 5] = 1.0
    return SimpleNamespace(
        node_ids=tuple(f"n{i}" for i in range(node_count)),
        actuator_ids=tuple(f"a{i}" for i in range(actuator_count)),
        actuator_upstream=np.arange(actuator_count, dtype=np.int64),
        actuator_downstream=np.arange(1, actuator_count + 1, dtype=np.int64),
        static_node_feature_names=static_names,
        static_node_features=static,
        actuator_physics_feature_names=physics_names,
        actuator_physics=physics,
    )


def test_v27_auto_rbc_shadow_obeys_mask_radius_and_changed_ceiling() -> None:
    graph = _fake_graph()
    state = torch.zeros((110, 4), dtype=torch.float32)
    state[:, 0] = torch.linspace(0.0, 1.2, 110)
    active = torch.full((109,), 0.5, dtype=torch.float32)
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    radius = np.full(109, 0.2, dtype=np.float64)
    proposal = build_auto_rbc_shadow_proposal(
        graph=graph,
        current_state=state,
        active_target=active,
        supervisory_mask=mask,
        first_radius=radius,
        max_changed_facilities=8,
        max_delta_per_update=0.5,
    )
    delta = torch.abs(proposal.target - active)
    assert int(torch.count_nonzero(delta > 1.0e-7)) <= 8
    assert float(delta.max()) <= 0.200001
    assert torch.allclose(proposal.target[~torch.as_tensor(mask)], active[~torch.as_tensor(mask)])
