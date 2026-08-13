from __future__ import annotations

import torch

from rtc.step2_control_response_v100 import (
    ActuatorNodeInfluenceAssetsV100,
    RegulatorAwareNonlocalOperatorV100,
)
from rtc.step2_v100_contract import (
    NonlocalHydraulicEffectLossContractV100,
    V100_INFLUENCE_ASSET_CONTRACT,
)


def _assets(*, connected: bool = True) -> ActuatorNodeInfluenceAssetsV100:
    pair = torch.zeros(2, 5, 15, dtype=torch.float32)
    pair[..., 12] = 1.0 if connected else 0.0
    pair[:, 0, 0] = 1.0
    pair[:, 1, 1] = 1.0
    pair[..., 2:8] = 0.25 if connected else 0.0
    mask = torch.full((2, 5), connected, dtype=torch.bool)
    if not connected:
        mask[:, :2] = True
        pair[:, :2, 12] = 1.0
    return ActuatorNodeInfluenceAssetsV100(
        contract=V100_INFLUENCE_ASSET_CONTRACT,
        inp_path="synthetic.inp",
        inp_sha256="0" * 64,
        node_count=5,
        actuator_count=2,
        physical_link_count=6,
        conduit_count=4,
        regulator_count=2,
        outlet_count=0,
        pair_feature_names=tuple(f"f{i}" for i in range(15)),
        pair_features=pair,
        same_component_mask=mask,
        actuator_ids=("A0", "A1"),
    )


def _operator(*, connected: bool = True) -> RegulatorAwareNonlocalOperatorV100:
    torch.manual_seed(4)
    contract = NonlocalHydraulicEffectLossContractV100()
    return RegulatorAwareNonlocalOperatorV100(
        hidden_dim=8,
        node_query_dim=6,
        actuator_count=2,
        node_count=5,
        assets=_assets(connected=connected),
        contract=contract,
    )


def test_zero_action_is_exact_zero() -> None:
    model = _operator()
    source = torch.zeros(1, 3, 2, 2, 8)
    query = torch.randn(1, 2, 5, 6)
    local = torch.zeros(1, 3, 2, 5, 8)
    output = model(source, query, local)
    assert torch.equal(output, torch.zeros_like(output))


def test_nonlocal_operator_reaches_remote_node_without_hop_diffusion() -> None:
    model = _operator()
    source = torch.zeros(1, 1, 1, 2, 8)
    source[..., 0, 0] = 1.0
    query = torch.zeros(1, 1, 5, 6)
    local = torch.zeros(1, 1, 1, 5, 8)
    output = model(source, query, local)
    assert float(output[..., 4, :].abs().max()) > 0.0


def test_disconnected_component_cannot_receive_action_effect() -> None:
    model = _operator(connected=False)
    source = torch.zeros(1, 1, 1, 2, 8)
    source[..., 0, 0] = 1.0
    query = torch.randn(1, 1, 5, 6)
    local = torch.zeros(1, 1, 1, 5, 8)
    output = model(source, query, local)
    assert torch.equal(output[..., 4, :], torch.zeros_like(output[..., 4, :]))


def test_joint_multi_actuator_response_is_not_sum_of_predicted_d2_outputs() -> None:
    model = _operator()
    query = torch.randn(1, 1, 5, 6)
    local = torch.zeros(1, 1, 1, 5, 8)
    a = torch.zeros(1, 1, 1, 2, 8)
    b = torch.zeros_like(a)
    a[..., 0, 0] = 0.8
    b[..., 1, 1] = -0.6
    joint = model(a + b, query, local)
    separate = model(a, query, local) + model(b, query, local)
    assert float((joint - separate).abs().max()) > 1e-7


def test_pair_geometry_has_no_oracle_or_link_flow_dependency() -> None:
    assets = _assets()
    assert assets.uses_future_truth is False
    assert assets.uses_online_link_flow is False
    assert assets.pair_features.shape == (2, 5, 15)
    assert assets.reachable_pair_fraction == 1.0
