from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rtc.direct_tfv_sequence_support import (
    DIRECT_TFV_SEQUENCE_SUPPORT_CONTRACT,
    derive_direct_tfv_sequence_support,
    direct_tfv_sequence_geometry,
)
from rtc.step2_tfv_support import DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v6 import DirectTFVRecedingMPCV6


@dataclass
class _Entry:
    arrays: dict[str, np.ndarray]
    indices: tuple[int, ...]
    reference_index: int
    rainfall_group: str = "rain-a"


class _Cache:
    def __init__(self, entry: _Entry) -> None:
        self._entry = entry

    def entry(self, name: str) -> _Entry:
        assert name.startswith("D3::")
        return self._entry


def _sequence_support(limit: float = 2.0) -> dict:
    payload = {
        "contract": DIRECT_TFV_SEQUENCE_SUPPORT_CONTRACT,
        "development_only": True,
        "label_independent": True,
        "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
        "actuator_ids": [f"a{i}" for i in range(109)],
        "control_block_steps": 2,
        "free_control_blocks": 12,
        "joint_branch_count": 10,
    }
    for metric in ("first_block_l1", "h120_l1", "h120_total_variation_l1"):
        payload[f"{metric}_q90"] = limit
        payload[f"{metric}_q95"] = limit
        payload[f"{metric}_q99"] = limit
        payload[f"{metric}_max"] = limit * 2.0
    return payload


def _bare_v6(sequence_support: dict) -> DirectTFVRecedingMPCV6:
    mpc = object.__new__(DirectTFVRecedingMPCV6)
    mpc.design = DirectTFVMPCDesignV4(active_support_quantile="q95")
    mpc.sequence_support = sequence_support
    mpc.action_support = {
        "joint_density_extension_contract": DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT,
        "joint_changed_facility_count_q95": 23.0,
        "joint_changed_facility_count_max": 46,
    }
    return mpc


def test_sequence_geometry_counts_first_mass_h120_mass_and_temporal_variation() -> None:
    reference = np.zeros((72, 109), dtype=np.float32)
    candidate = reference.copy()
    candidate[:2, 0] = 0.5
    candidate[:2, 1] = 0.25
    candidate[2:4, 0] = 0.25
    geometry = direct_tfv_sequence_geometry(candidate, reference)
    assert np.isclose(geometry["first_block_l1"], 0.75)
    assert np.isclose(geometry["h120_l1"], 1.0)
    assert np.isclose(geometry["h120_total_variation_l1"], 1.5)


def test_derive_sequence_support_uses_only_multi_facility_h120_branch() -> None:
    settings = np.zeros((3, 72, 109), dtype=np.float32)
    settings[1, :24, 0] = 0.2
    settings[1, :24, 1] = 0.3
    settings[2, :24, 0] = 0.4
    cache = _Cache(_Entry(arrays={"settings": settings}, indices=(0, 1, 2), reference_index=0))
    payload = derive_direct_tfv_sequence_support(
        cache,
        ["D3::rain-a::event-a::checkpoint-a"],
        actuator_ids=[f"a{i}" for i in range(109)],
    )
    assert payload["joint_branch_count"] == 1
    assert payload["label_independent"] is True
    assert np.isclose(payload["first_block_l1_q95"], 0.5)
    assert np.isclose(payload["h120_l1_q95"], 6.0)


def test_v6_radial_contraction_is_differentiable_and_inside_joint_support() -> None:
    mpc = _bare_v6(_sequence_support(limit=1.0))
    values = torch.zeros((72, 109), dtype=torch.float32)
    values[:24, :4] = 0.5
    sequence = values.requires_grad_()
    active_target = torch.zeros(109, dtype=torch.float32)
    contracted = mpc._contract_to_joint_sequence_support(sequence, active_target)
    diagnostics = mpc.joint_sequence_support_diagnostics(contracted, active_target)
    assert float(diagnostics["max_ratio"]) <= 1.00001
    assert torch.all(contracted >= 0.0)
    assert torch.all(contracted <= 0.5)
    contracted.sum().backward()
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()


def test_v6_leaves_already_supported_sequence_unchanged() -> None:
    mpc = _bare_v6(_sequence_support(limit=100.0))
    active_target = torch.full((109,), 0.5, dtype=torch.float32)
    sequence = active_target[None].repeat(72, 1)
    sequence[:2, :2] += 0.1
    contracted = mpc._contract_to_joint_sequence_support(sequence, active_target)
    assert torch.allclose(contracted, sequence)
