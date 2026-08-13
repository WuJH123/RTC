from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_control_response_v113 import SupportConditionedHydraulicSurrogateV113
from run_step2_v113_mechanism import _preflight_gate_v113


def _prepared(nodes=4, actuators=2):
    return SimpleNamespace(
        node_static=torch.zeros(nodes, 3),
        actuator_physics=torch.zeros(actuators, 4),
        actuator_upstream=torch.tensor([0, 2], dtype=torch.long),
        actuator_downstream=torch.tensor([1, 3], dtype=torch.long),
        max_depth_m=torch.ones(nodes),
        storage_mask=torch.tensor([False, False, True, True]),
    )


def _model(mode="phase"):
    a, t, n, k = 2, 4, 4, 5
    prior = np.full((a, t, n, k), 0.25, dtype=np.float32)
    flow = np.full((a, t, a), 0.25, dtype=np.float32)
    return SupportConditionedHydraulicSurrogateV113(
        node_static_dim=3, physics_dim=4, rainfall_dim=1,
        actuator_count=a, node_count=n,
        state_scale=np.ones((n, k), dtype=np.float32),
        flow_scale=np.ones(a, dtype=np.float32),
        overall_state_prior=prior, overall_flow_prior=flow,
        phase_state_priors=np.stack([prior * 0.5, prior, prior * 1.5]),
        phase_flow_priors=np.stack([flow * 0.5, flow, flow * 1.5]),
        phase_boundaries=(0.25, 0.75), retained_indices=(0, 1, 2, 3),
        prior_mode=mode, hidden_dim=8,
    )


def _inputs():
    initial = torch.zeros(1, 4, 6)
    rain = torch.zeros(1, 4, 4, 1)
    ref = torch.zeros(1, 4, 2)
    prev = torch.zeros(1, 2)
    return initial, rain, ref, prev


def test_v113_exact_zero_is_structural_for_state_and_flow():
    model = _model().eval()
    initial, rain, ref, prev = _inputs()
    with torch.no_grad():
        out = model(initial, rain, ref, ref[:, None], prev, _prepared())
    assert torch.equal(out.raw_delta_states_physical, torch.zeros_like(out.raw_delta_states_physical))
    assert torch.equal(out.raw_delta_flows_physical, torch.zeros_like(out.raw_delta_flows_physical))


def test_v113_future_action_does_not_change_earlier_retained_effect():
    model = _model().eval()
    initial, rain, ref, prev = _inputs()
    cand = ref[:, None].clone()
    cand[:, :, 2:, 0] = 1.0
    later = cand.clone()
    later[:, :, 3:, 1] = 1.0
    with torch.no_grad():
        a = model(initial, rain, ref, cand, prev, _prepared())
        b = model(initial, rain, ref, later, prev, _prepared())
    assert torch.equal(a.raw_delta_states_physical[:, :, :2], b.raw_delta_states_physical[:, :, :2])
    assert torch.equal(a.raw_delta_flows_physical[:, :, :2], b.raw_delta_flows_physical[:, :, :2])


def test_v113_storage_is_zero_outside_real_storage_domain():
    model = _model().eval()
    initial, rain, ref, prev = _inputs()
    cand = ref[:, None].clone(); cand[:, :, :, 0] = 1.0
    with torch.no_grad():
        out = model(initial, rain, ref, cand, prev, _prepared())
    assert torch.equal(out.raw_delta_states_physical[..., :2, 3], torch.zeros_like(out.raw_delta_states_physical[..., :2, 3]))


def test_v113_candidate_gradient_is_finite_and_nonzero():
    model = _model("overall")
    initial, rain, ref, prev = _inputs()
    cand = ref[:, None].clone().requires_grad_(True)
    cand.data[:, :, :, 0] = 1.0
    out = model(initial, rain, ref, cand, prev, _prepared())
    grad = torch.autograd.grad(out.raw_delta_states_physical.square().mean() + out.raw_delta_flows_physical.square().mean(), cand)[0]
    assert torch.isfinite(grad).all()
    assert int(torch.count_nonzero(grad)) > 0


def test_v113_multiple_candidates_do_not_create_extra_candidate_axis():
    model = _model("overall").eval()
    initial, rain, ref, prev = _inputs()
    candidate = ref[:, None].expand(1, 3, 4, 2).clone()
    candidate[:, 1, :, 0] = 1.0
    candidate[:, 2, :, 1] = -1.0
    with torch.no_grad():
        out = model(initial, rain, ref, candidate, prev, _prepared())
    assert out.raw_delta_states_physical.shape == (1, 3, 4, 4, 6)
    assert out.raw_delta_flows_physical.shape == (1, 3, 4, 2)


def test_v113_support_override_retains_per_actuator_prior_axis():
    model = _model("overall").eval()
    initial, rain, ref, prev = _inputs()
    candidate = ref[:, None].clone()
    candidate[:, :, :, 0] = 1.0
    override = torch.zeros(1, 1, 2, 4, 4, 5)
    override[:, :, 0, :, 0, 0] = 1.0
    with torch.no_grad():
        out = model(initial, rain, ref, candidate, prev, _prepared(), support_override=override)
    assert out.state_support_context.shape == (1, 1, 4, 4, 5)


def test_v113_preflight_gate_treats_prohibitions_as_negative_flags():
    assert _preflight_gate_v113({
        "exact_zero": True, "action_gradient_finite": True,
        "action_gradient_nonzero": True, "storage_domain_only": True,
        "future_truth_input": False, "hard_prior_mask": False,
    })
    assert not _preflight_gate_v113({
        "exact_zero": True, "action_gradient_finite": True,
        "action_gradient_nonzero": True, "storage_domain_only": True,
        "future_truth_input": True, "hard_prior_mask": False,
    })
