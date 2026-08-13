from __future__ import annotations

import math

import torch


def _old_multiplicative_response(
    active_probability: torch.Tensor,
    sign_logit: torch.Tensor,
    magnitude_logit: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The V110 physical decoder, isolated for the regression evidence."""
    return (
        torch.sigmoid(active_probability)
        * torch.tanh(sign_logit)
        * torch.nn.functional.softplus(magnitude_logit)
        * scale
    )


def test_old_multiplicative_decoder_can_overproduce_sparse_zero_domain_energy():
    nodes = 932
    truth = torch.zeros(nodes)
    truth[17] = 0.01
    # A low active probability and a small per-cell prediction still leaves
    # non-trivial pooled energy when replicated over the zero domain.
    pred = _old_multiplicative_response(
        torch.full((nodes,), -2.2),
        torch.full((nodes,), 0.15),
        torch.full((nodes,), -0.25),
        1.0,
    )
    pred[17] = truth[17]
    assert float(pred.square().sum()) > float(truth.square().sum())


def test_v111_direct_decoder_is_zero_anchored_and_has_changed_action_gradient():
    from rtc.step2_control_response_v111 import zero_anchored_direct_head

    head = torch.nn.Linear(4, 1)
    zero_anchored_direct_head(head, weight_value=1.0e-4)
    assert torch.equal(head.bias.detach(), torch.zeros_like(head.bias))
    assert float(head.weight.detach().abs().max()) <= 1.0e-4 + 1.0e-12
    x = torch.ones(1, 4, requires_grad=True)
    y = head(x)
    assert float(y.detach().abs().max()) < 1.0e-3
    grad = torch.autograd.grad(y.sum(), x)[0]
    assert bool(torch.isfinite(grad).all())
    assert int(torch.count_nonzero(grad)) > 0


def test_v111_event_pooled_skill_is_stable_with_near_zero_group():
    from rtc.step2_hydraulic_eval_v111 import event_pooled_skill_vs_zero

    # Group A has an almost-zero denominator; group B carries the actual
    # event signal.  Pooling SSE before taking the ratio must remain finite.
    records = [
        {"event_id": "e", "pred": torch.tensor([1.0e-5]), "truth": torch.tensor([0.0])},
        {"event_id": "e", "pred": torch.tensor([0.5, 0.0]), "truth": torch.tensor([1.0, -1.0])},
    ]
    value = event_pooled_skill_vs_zero(records)
    assert math.isfinite(value)
    assert -1.0 < value < 1.0
