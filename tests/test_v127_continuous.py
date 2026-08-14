from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rtc.checkpoint_v127 import graph_semantic_sha256_v127
from rtc.d5_gradient_v127 import (
    D5GradientDesignV127,
    broad_center_fractions_v127,
    directional_fractions_v127,
    symmetric_probe_v127,
)
from rtc.rule_baselines import StorageGeometry
from rtc.step2_differentiable_v127 import ControlOrientedDifferentiableSurrogateV127
from rtc.step2_state_store_v127 import CausalStateStoreV127
from rtc.step2_train_v127 import _spearman, _truth_node_volume
from rtc.step3_mpc_v127 import (
    ContinuousMPCDesignV127,
    Step2GradientEvidenceV127,
    decode_fractional_targets_v127,
)


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def test_v127_smooth_tfv_has_finite_action_gradient() -> None:
    torch.manual_seed(3)
    model = ControlOrientedDifferentiableSurrogateV127(
        state_dim=3,
        rainfall_dim=1,
        node_static_dim=2,
        actuator_physics_dim=2,
        actuator_count=1,
        hidden_dim=16,
        actuator_embedding_dim=4,
        delta_state_scale=torch.tensor([0.1, 0.1, 0.05]),
        delta_flow_scale=torch.tensor([0.1]),
    )
    initial = torch.zeros(1, 2, 3)
    rainfall = torch.zeros(1, 4, 2, 1)
    settings = torch.full((1, 4, 1), 0.5, requires_grad=True)
    result = model.objective_rollout(
        initial_state=initial,
        rainfall=rainfall,
        settings=settings,
        previous_actuator_flow=torch.zeros(1, 1),
        actuator_upstream=torch.tensor([0]),
        actuator_downstream=torch.tensor([1]),
        actuator_physics=torch.zeros(1, 2),
        static_node_features=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        flood_rate_index=2,
        priority_indices=torch.tensor([0]),
        dt_seconds=300.0,
    )
    assert torch.isfinite(result.tfv_m3).all()
    assert torch.isfinite(result.optimization_tfv_m3).all()
    gradient = torch.autograd.grad(result.optimization_tfv_m3.sum(), settings)[0]
    assert gradient.shape == settings.shape
    assert torch.isfinite(gradient).all()


def test_v127_fraction_decoder_is_1308_variable_bounded_rate_feasible() -> None:
    design = ContinuousMPCDesignV127()
    fractions = torch.linspace(0.0, 1.0, 12 * 109).reshape(12, 109)
    active = torch.full((109,), 0.4)
    lo = torch.zeros(109)
    hi = torch.ones(109)
    sequence = decode_fractional_targets_v127(
        fractions,
        active_target=active,
        min_setting=lo,
        max_setting=hi,
        design=design,
    )
    assert design.variable_count == 1308
    assert sequence.shape == (72, 109)
    torch.testing.assert_close(sequence[0], sequence[1])
    blocks = sequence[::2]
    assert bool(torch.all((blocks >= 0.0) & (blocks <= 1.0)))
    first_delta = torch.max(torch.abs(blocks[0] - active))
    later_delta = torch.max(torch.abs(blocks[1:] - blocks[:-1]))
    assert float(first_delta) <= 0.500001
    assert float(later_delta) <= 0.500001
    torch.testing.assert_close(blocks[12:], blocks[11:12].expand_as(blocks[12:]))


def test_v127_quality_scores_are_evidence_not_arbitrary_runtime_switch() -> None:
    # Weak-but-finite model quality remains visible evidence and does not redefine whether
    # continuous MPC exists.  Causality and numerical validity remain hard contracts.
    weak = Step2GradientEvidenceV127(
        holdout_rank=0.21,
        holdout_top1=0.10,
        d2_gradient_sign_accuracy=0.35,
        d2_gradient_cosine_similarity=-0.20,
        d5_gradient_sign_accuracy=0.42,
        d5_gradient_cosine_similarity=0.05,
        causal_step1_state_verified=True,
        causal_rainfall_verified=True,
    )
    weak.validate()
    noncausal = Step2GradientEvidenceV127(
        holdout_rank=0.9,
        holdout_top1=0.9,
        d2_gradient_sign_accuracy=0.9,
        d2_gradient_cosine_similarity=0.9,
        d5_gradient_sign_accuracy=0.9,
        d5_gradient_cosine_similarity=0.9,
        causal_step1_state_verified=False,
        causal_rainfall_verified=True,
    )
    with pytest.raises(ValueError, match="causal Step1"):
        noncausal.validate()
    nonfinite = Step2GradientEvidenceV127(
        holdout_rank=float("nan"),
        holdout_top1=0.9,
        d2_gradient_sign_accuracy=0.9,
        d2_gradient_cosine_similarity=0.9,
        d5_gradient_sign_accuracy=0.9,
        d5_gradient_cosine_similarity=0.9,
        causal_step1_state_verified=True,
        causal_rainfall_verified=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        nonfinite.validate()


def test_v127_causal_state_store_content_hash_and_unique_identity() -> None:
    state = np.zeros((2, 3, 4), dtype=np.float32)
    store = CausalStateStoreV127(
        event_ids=("e1", "e2"),
        checkpoint_ids=("c1", "c2"),
        elapsed_seconds=np.asarray([600, 1200], dtype=np.int64),
        state_si=state,
        current_setting=np.full((2, 109), 0.5, dtype=np.float32),
        state_sha256=(_sha(state[0]), _sha(state[1])),
        step1_sha256="a" * 64,
        sensor_sha256="b" * 64,
        graph_sha256="c" * 64,
    )
    store.validate()
    broken = CausalStateStoreV127(
        **{**store.__dict__, "state_sha256": ("0" * 64, _sha(state[1]))}
    )
    with pytest.raises(ValueError, match="content hash"):
        broken.validate()


def test_v127_authoritative_node_volume_labels_follow_reference_first_branch_order() -> None:
    # Original shard order is [candidate0, reference, candidate1].
    volume = np.asarray(
        [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]], dtype=np.float32
    )
    entry = SimpleNamespace(
        reference_index=1,
        indices=(0, 1, 2),
        arrays={"exact_node_flood_volume_m3": volume},
    )
    cache = SimpleNamespace(entry=lambda _: entry)
    ordered = _truth_node_volume(cache, "g")
    np.testing.assert_array_equal(ordered, volume[[1, 0, 2]])


def test_v127_spearman_uses_average_tie_ranks() -> None:
    # Two equal predictions must receive the same rank; deterministic argsort order must
    # not create artificial correlation.
    assert _spearman(np.asarray([1.0, 1.0, 3.0]), np.asarray([2.0, 2.0, 4.0])) == pytest.approx(1.0)
    assert _spearman(np.asarray([1.0, 1.0]), np.asarray([1.0, 2.0])) != _spearman(
        np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0])
    )


def test_v127_d5_probe_lives_in_exact_online_fraction_space() -> None:
    design = D5GradientDesignV127(
        max_checkpoints=1,
        directions_per_center=8,
    )
    center = broad_center_fractions_v127(
        109,
        checkpoint_identity="rain|event|checkpoint",
        free_control_blocks=12,
    )
    direction, family = directional_fractions_v127(
        109,
        checkpoint_identity="rain|event|checkpoint",
        checkpoint_rank=0,
        center_index=2,
        direction_index=0,
        free_control_blocks=12,
    )
    assert center.shape == direction.shape == (12, 109)
    assert family == "first_move_single_actuator"
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=5e-6)
    probe = symmetric_probe_v127(
        active_target=np.full(109, 0.5, dtype=np.float32),
        min_setting=np.zeros(109, dtype=np.float32),
        max_setting=np.ones(109, dtype=np.float32),
        center_fractions=center,
        direction=direction,
        design=design,
    )
    assert probe is not None
    plus = np.asarray(probe["plus_fractions"])
    minus = np.asarray(probe["minus_fractions"])
    np.testing.assert_allclose(0.5 * (plus + minus), center, atol=2e-6, rtol=0)
    assert np.asarray(probe["plus_sequence"]).shape == (72, 109)


def test_v127_graph_fingerprint_detects_same_shape_semantic_change() -> None:
    base = SimpleNamespace(
        node_ids=("n0", "n1"),
        static_node_feature_names=("invert",),
        actuator_ids=("a0",),
        actuator_physics_feature_names=("min_setting", "max_setting"),
        system_units="SI",
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        static_node_features=np.asarray([[0.0], [1.0]], dtype=np.float32),
        actuator_upstream=np.asarray([0], dtype=np.int64),
        actuator_downstream=np.asarray([1], dtype=np.int64),
        actuator_physics=np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    changed = SimpleNamespace(**{**base.__dict__, "actuator_downstream": np.asarray([0], dtype=np.int64)})
    assert graph_semantic_sha256_v127(base) != graph_semantic_sha256_v127(changed)


def test_efd_functional_storage_uses_volume_not_depth_fraction() -> None:
    # Area = depth, so V(d)=d^2/2. At half maximum depth the volume filling is 0.25,
    # explicitly different from normalized depth 0.5.
    geometry = StorageGeometry(
        shape="FUNCTIONAL",
        max_depth_native=2.0,
        functional_a1=1.0,
        functional_a2=1.0,
        functional_a0=0.0,
        system_units="SI",
    )
    assert geometry.capacity_m3 == pytest.approx(2.0)
    assert geometry.volume_m3(1.0) / geometry.capacity_m3 == pytest.approx(0.25)
