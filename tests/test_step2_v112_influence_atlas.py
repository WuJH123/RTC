import numpy as np
import pytest

from rtc.step2_d2_lineage_v112 import (
    classify_d2_population_v112,
    require_full_d2_source_claim_v112,
)
from rtc.step2_graph_distance_v112 import undirected_endpoint_hops_v112
from rtc.step2_influence_prior_v112 import (
    InfluencePriorAccumulatorV112,
    combine_support_proposals_noisy_or_v112,
)
from rtc.step2_influence_support_v112 import (
    build_influence_support_targets_v112,
    infer_single_changed_actuator_v112,
    source_flow_effective_v112,
)


def test_full_4800_source_is_not_conflated_with_derived_3600_cache():
    full = classify_d2_population_v112(
        authoritative_branches=4800, checkpoint_states=192, actuator_count=109, event_count=24
    )
    derived = classify_d2_population_v112(
        authoritative_branches=3600, checkpoint_states=144, actuator_count=109, group_count=144
    )
    assert full["population_view"] == "FULL_D2_SOURCE"
    assert derived["population_view"] == "DERIVED_D2_VIEW"
    with pytest.raises(RuntimeError):
        require_full_d2_source_claim_v112(derived)


def test_single_actuator_probe_is_fail_closed_for_joint_actions():
    ref = np.zeros((4, 3))
    cand = ref.copy()
    cand[:, 1] = 0.4
    assert infer_single_changed_actuator_v112(ref, cand) == 1
    cand[:, 2] = 0.2
    with pytest.raises(ValueError):
        infer_single_changed_actuator_v112(ref, cand)


def _targets():
    ref_s = np.zeros((4, 2, 6), dtype=float)
    cand_s = ref_s.copy()
    cand_s[1:, 0, 0] = -0.2
    cand_s[2:, 1, 2] = 0.5
    ref_f = np.zeros((4, 2), dtype=float)
    cand_f = ref_f.copy()
    cand_f[1:, 0] = -0.3
    return build_influence_support_targets_v112(
        reference_states=ref_s,
        candidate_states=cand_s,
        reference_flows=ref_f,
        candidate_flows=cand_f,
        state_active_threshold=np.full((2, 5), 0.05),
        flow_active_threshold=np.full(2, 0.05),
        retained_indices=(0, 1, 2, 3),
    )


def test_support_targets_preserve_negative_effects_and_lag():
    t = _targets()
    assert t.delta_state[1, 0, 0] == pytest.approx(-0.2)
    assert bool(t.state_active[1, 0, 0])
    assert not bool(t.state_active[1, 1, 1])
    assert bool(t.state_active[2, 1, 1])
    assert t.delta_flow[1, 0] == pytest.approx(-0.3)
    assert source_flow_effective_v112(t, 0)


def test_soft_prior_keeps_global_escape_and_no_action_is_exact_zero():
    t = _targets()
    acc = InfluencePriorAccumulatorV112(actuator_count=2, retained_count=4, node_count=2)
    acc.update(0, t)
    acc.update(1, t)
    prior = acc.finalize()["state_support_probability"]
    assert np.nanmin(prior) > 0.0
    active = np.zeros((1, 4, 2), dtype=bool)
    proposal = combine_support_proposals_noisy_or_v112(prior, active)
    assert np.array_equal(proposal, np.zeros_like(proposal))
    active[:, 1:, 0] = True
    proposal = combine_support_proposals_noisy_or_v112(prior, active)
    assert float(proposal[:, 1:].max()) > 0.0
    assert float(proposal.max()) <= 1.0


def test_joint_support_union_is_not_additive_magnitude_superposition():
    prior = np.full((2, 1, 1, 1), 0.6, dtype=np.float32)
    active = np.ones((1, 1, 2), dtype=bool)
    proposal = combine_support_proposals_noisy_or_v112(prior, active)
    assert proposal.item() == pytest.approx(0.84, abs=1e-6)


def test_endpoint_hops_are_diagnostic_without_hard_cutoff():
    edge_index = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    hops = undirected_endpoint_hops_v112(
        edge_index, node_count=5, upstream=1, downstream=2
    )
    assert hops.tolist() == [1, 0, 0, 1, 2]
    assert hops[4] == 2
