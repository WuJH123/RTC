from __future__ import annotations

import numpy as np

from rtc.step2_control_basis_v60 import ControlBasisV60, ControlGroupingV60
from rtc.step2_manifold_semantics_v110 import candidate_manifold_descriptors_v110
from rtc.step2_v60_contract import ControlBasisContractV60, MultiResolutionHorizonV60


def _basis() -> ControlBasisV60:
    horizon = MultiResolutionHorizonV60()
    grouping = ControlGroupingV60(
        zone_id_by_actuator=np.asarray([0, 0], dtype=np.int64),
        type_id_by_actuator=np.asarray([0, 0], dtype=np.int64),
        group_id_by_actuator=np.asarray([0, 0], dtype=np.int64),
        group_labels=("zone00:pump",),
        medoid_actuator_indices=(0,),
    )
    temporal = np.zeros((horizon.control_blocks, 6), dtype=np.float32)
    centers = np.linspace(0, horizon.control_blocks - 1, 6)
    width = centers[1] - centers[0]
    x = np.arange(horizon.control_blocks, dtype=np.float64)
    temporal = np.maximum(1.0 - np.abs(x[:, None] - centers[None]) / width, 0.0)
    temporal /= temporal.sum(axis=1, keepdims=True)
    return ControlBasisV60(
        grouping=grouping,
        temporal_basis=temporal.astype(np.float32),
        min_setting=np.zeros(2, dtype=np.float32),
        max_setting=np.ones(2, dtype=np.float32),
        horizon=horizon,
        contract=ControlBasisContractV60(spatial_zone_count=1),
    )


def test_group_coherent_action_is_close_to_frozen_manifold() -> None:
    basis = _basis()
    reference = np.full((basis.horizon.horizon_steps, 2), 0.5, dtype=np.float32)
    coefficients = np.zeros((1, basis.temporal_basis_count, basis.group_count), dtype=np.float32)
    coefficients[0, 2, 0] = 0.2
    # project_actions_to_coefficients works in setting-delta units. Build an exact
    # linear-basis delta rather than using bounded decoder projection.
    group_delta = basis.temporal_basis @ coefficients[0]
    block_delta = np.repeat(group_delta, 2, axis=1)
    delta = np.repeat(block_delta, basis.horizon.control_block_steps, axis=0)
    candidate = reference[None] + delta[None]

    result = candidate_manifold_descriptors_v110(reference, candidate, basis)

    assert result.projection_residual_ratio[0] < 1.0e-5
    assert result.active_control_group_count.tolist() == [1]
    assert result.active_temporal_basis_count.tolist() == [1]


def test_single_actuator_action_is_identified_as_off_group_manifold() -> None:
    basis = _basis()
    reference = np.full((basis.horizon.horizon_steps, 2), 0.5, dtype=np.float32)
    candidate = reference[None].copy()
    candidate[0, :, 0] += 0.1

    result = candidate_manifold_descriptors_v110(reference, candidate, basis)

    assert result.projection_residual_ratio[0] > 0.1
