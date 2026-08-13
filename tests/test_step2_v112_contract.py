import pytest

from rtc.step2_v112_contract import Step2V112Contract


def test_v112_gate_order_keeps_influence_atlas_before_new_hydraulic_training():
    contract = Step2V112Contract()
    contract.validate()
    order = contract.development_order
    assert order.index("build_trainfit_state_conditioned_influence_atlas") < order.index(
        "train_support_conditioned_direct_signed_hydraulic_tiny_micro"
    )
    assert order.index("canonical_trainfit_d2_gate") < order.index(
        "internal_holdout_once_after_freeze"
    )
    assert order.index("internal_holdout_once_after_freeze") < order.index(
        "direct_authoritative_d3_joint_action_training"
    )


def test_v112_refuses_prior_project_failures():
    with pytest.raises(ValueError):
        Step2V112Contract(sum_d2_magnitudes_for_d3=True).validate()
    with pytest.raises(ValueError):
        Step2V112Contract(hard_hop_support_mask=True).validate()
    with pytest.raises(ValueError):
        Step2V112Contract(validation_outcomes_allowed=True).validate()
    with pytest.raises(ValueError):
        Step2V112Contract(new_swmm_authorized=True).validate()
