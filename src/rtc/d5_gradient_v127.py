"""High-value D5 central-difference design for Project7 V127.

D5 validates the variable space that the online optimiser actually sees: twelve free
10-minute target-fraction blocks for every writable actuator.  It does not use the legacy
V60 grouped control basis.  All centres and +/- probes are defined in that exact fraction
space and are decoded with the same differentiable V127 MPC decoder used online.

Every authoritative label is therefore a directional derivative with respect to the
1308-dimensional L-BFGS-B decision vector.  A pair is accepted only when the decoded
physical targets remain centrally symmetric after bounds/rate handling; otherwise epsilon
is reduced or the direction is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from .step3_mpc_v127 import ContinuousMPCDesignV127, decode_fractional_targets_v127

V127_D5_CONTRACT = "PROJECT7_V127_D5_1308VAR_SYMMETRIC_DIRECTIONAL_GRADIENT_V2"


@dataclass(frozen=True)
class D5GradientDesignV127:
    max_checkpoints: int = 48
    centers_per_checkpoint: int = 3
    directions_per_center: int = 8
    initial_epsilon: float = 0.20
    minimum_epsilon: float = 0.0125
    audit_fraction: float = 0.25
    free_control_blocks: int = 12
    symmetry_atol: float = 2.5e-5

    def validate(self) -> None:
        if self.max_checkpoints <= 0 or self.centers_per_checkpoint != 3:
            raise ValueError("V127 D5 checkpoint/centre design is invalid")
        if self.directions_per_center < 4:
            raise ValueError("V127 D5 requires enough directions to cover temporal/spatial control modes")
        if not 0.0 < self.minimum_epsilon <= self.initial_epsilon <= 0.5:
            raise ValueError("V127 D5 epsilon contract is invalid")
        if not 0.0 < self.audit_fraction < 0.5:
            raise ValueError("V127 D5 audit fraction must lie in (0,0.5)")
        if self.free_control_blocks != 12:
            raise ValueError("V127 D5 must match H120 / 12 free 10-min MPC blocks")
        if self.symmetry_atol <= 0.0:
            raise ValueError("V127 D5 symmetry tolerance must be positive")

    @property
    def branches_per_checkpoint(self) -> int:
        return self.centers_per_checkpoint * (1 + 2 * self.directions_per_center)

    @property
    def planned_branches(self) -> int:
        return self.max_checkpoints * self.branches_per_checkpoint


def deterministic_rainfall_roles_v127(
    rainfall_groups: list[str] | tuple[str, ...], *, audit_fraction: float = 0.25
) -> dict[str, str]:
    groups = sorted(set(str(x) for x in rainfall_groups))
    if len(groups) < 4:
        raise ValueError("V127 D5 needs at least four rainfall groups")
    scored = sorted(
        (
            int(hashlib.sha256(f"V127_D5|{group}".encode()).hexdigest()[:16], 16),
            group,
        )
        for group in groups
    )
    count = max(1, int(round(float(audit_fraction) * len(groups))))
    count = min(count, len(groups) - 1)
    audit = {group for _, group in scored[-count:]}
    return {group: ("audit" if group in audit else "fit") for group in groups}


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def broad_center_fractions_v127(
    actuator_count: int,
    *,
    checkpoint_identity: str,
    free_control_blocks: int = 12,
) -> np.ndarray:
    """Deterministic interior centre spanning actuator and temporal variation.

    The construction is deliberately smooth enough to remain engineering-relevant while
    not being restricted to the RBC neighbourhood.  It is outcome-blind.
    """
    if actuator_count <= 0 or free_control_blocks <= 0:
        raise ValueError("V127 D5 broad centre dimensions are invalid")
    rng = np.random.default_rng(_seed(f"V127_D5_CENTER|{checkpoint_identity}"))
    spatial = rng.uniform(-0.26, 0.26, size=(1, actuator_count))
    temporal_knots = rng.uniform(-0.18, 0.18, size=(3, actuator_count))
    x = np.linspace(0.0, 2.0, free_control_blocks)
    temporal = np.empty((free_control_blocks, actuator_count), dtype=np.float64)
    for actuator in range(actuator_count):
        temporal[:, actuator] = np.interp(x, [0.0, 1.0, 2.0], temporal_knots[:, actuator])
    interaction = rng.uniform(-0.06, 0.06, size=(free_control_blocks, actuator_count))
    values = 0.5 + spatial + temporal + interaction
    return np.clip(values, 0.10, 0.90).astype(np.float32)


def directional_fractions_v127(
    actuator_count: int,
    *,
    checkpoint_identity: str,
    checkpoint_rank: int,
    center_index: int,
    direction_index: int,
    free_control_blocks: int = 12,
) -> tuple[np.ndarray, str]:
    """Deterministic unit-L2 direction in the exact 12 x actuator MPC variable space.

    Direction families deliberately mix interpretable local probes and coordinated sparse
    probes.  Across checkpoints the deterministic index rotation spreads coverage over all
    actuators and all free control blocks without using any SWMM outcome.
    """
    blocks = int(free_control_blocks)
    actuators = int(actuator_count)
    dim = blocks * actuators
    if min(blocks, actuators) <= 0:
        raise ValueError("V127 D5 fraction direction dimensions are invalid")
    flat = np.zeros(dim, dtype=np.float64)
    family_index = int(direction_index) % 8
    sweep = (
        int(checkpoint_rank) * 3 * max(8, int(direction_index) + 1)
        + int(center_index) * 8
        + int(direction_index)
    )
    rng = np.random.default_rng(
        _seed(
            f"V127_D5_DIR|{checkpoint_identity}|{checkpoint_rank}|{center_index}|{direction_index}"
        )
    )

    if family_index == 0:
        # First-move actuator derivative: directly validates the command that is executed.
        actuator = sweep % actuators
        flat[actuator] = 1.0
        family = "first_move_single_actuator"
    elif family_index == 1:
        # One actuator changed persistently across the complete free H120 horizon.
        actuator = (sweep * 7 + 3) % actuators
        flat[np.arange(blocks) * actuators + actuator] = 1.0
        family = "persistent_single_actuator"
    elif family_index == 2:
        # Late-horizon local derivative verifies temporal credit beyond the executed block.
        actuator = (sweep * 11 + 5) % actuators
        block = blocks - 1 - (checkpoint_rank % min(3, blocks))
        flat[block * actuators + actuator] = 1.0
        family = "late_horizon_single_actuator"
    elif family_index == 3:
        # A first-block coordinated action over a deterministic subset of facilities.
        count = min(12, actuators)
        chosen = rng.choice(actuators, size=count, replace=False)
        flat[chosen] = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        family = "first_move_multi_actuator"
    elif family_index == 4:
        # Temporal signed pattern on one actuator.
        actuator = (sweep * 13 + 1) % actuators
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=blocks)
        flat[np.arange(blocks) * actuators + actuator] = signs
        family = "temporal_single_actuator"
    elif family_index == 5:
        # One free block with broader spatial coupling.
        block = (checkpoint_rank + center_index + direction_index) % blocks
        count = min(24, actuators)
        chosen = rng.choice(actuators, size=count, replace=False)
        flat[block * actuators + chosen] = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        family = "single_block_spatial_interaction"
    elif family_index == 6:
        # Sparse spatiotemporal interaction in the same direct decision tensor.
        count = min(max(24, dim // 40), dim)
        chosen = rng.choice(dim, size=count, replace=False)
        flat[chosen] = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        family = "sparse_spatiotemporal_interaction"
    else:
        # Denser Rademacher direction gives a global Jacobian projection without expanding
        # D5 into one branch pair per one of the 1308 coordinates.
        count = min(max(64, dim // 12), dim)
        chosen = rng.choice(dim, size=count, replace=False)
        flat[chosen] = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        family = "broad_spatiotemporal_interaction"

    norm = float(np.linalg.norm(flat))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("V127 D5 produced a zero/non-finite direct-MPC direction")
    return (flat / norm).reshape(blocks, actuators).astype(np.float32), family


def decode_fractions_v127(
    *,
    fractions: np.ndarray,
    active_target: np.ndarray,
    min_setting: np.ndarray,
    max_setting: np.ndarray,
    design: D5GradientDesignV127 = D5GradientDesignV127(),
) -> np.ndarray:
    design.validate()
    mpc_design = ContinuousMPCDesignV127(
        free_control_blocks=design.free_control_blocks,
        min_improvement_vs_rbc_m3=0.0,
        movement_penalty_m3=0.0,
    )
    with torch.no_grad():
        sequence = decode_fractional_targets_v127(
            torch.as_tensor(fractions, dtype=torch.float32),
            active_target=torch.as_tensor(active_target, dtype=torch.float32),
            min_setting=torch.as_tensor(min_setting, dtype=torch.float32),
            max_setting=torch.as_tensor(max_setting, dtype=torch.float32),
            design=mpc_design,
        )
    return sequence.cpu().numpy().astype(np.float32)


def symmetric_probe_v127(
    *,
    active_target: np.ndarray,
    min_setting: np.ndarray,
    max_setting: np.ndarray,
    center_fractions: np.ndarray,
    direction: np.ndarray,
    design: D5GradientDesignV127 = D5GradientDesignV127(),
) -> dict[str, object] | None:
    """Decode a symmetric fraction-space +/- probe using the exact online MPC decoder."""
    design.validate()
    center_fraction = np.asarray(center_fractions, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    expected_shape = (design.free_control_blocks, len(np.asarray(active_target).reshape(-1)))
    if center_fraction.shape != expected_shape or direction.shape != expected_shape:
        raise ValueError("V127 D5 centre/direction does not match the direct MPC variable tensor")
    if not np.isfinite(center_fraction).all() or not np.isfinite(direction).all():
        raise ValueError("V127 D5 centre/direction contains non-finite values")
    if np.any((center_fraction < 0.0) | (center_fraction > 1.0)):
        raise ValueError("V127 D5 centre leaves L-BFGS-B fraction bounds")

    center = decode_fractions_v127(
        fractions=center_fraction,
        active_target=active_target,
        min_setting=min_setting,
        max_setting=max_setting,
        design=design,
    )
    epsilon = float(design.initial_epsilon)
    while epsilon >= float(design.minimum_epsilon) - 1e-12:
        plus_fraction = center_fraction + epsilon * direction
        minus_fraction = center_fraction - epsilon * direction
        # Do not clip the optimisation variables: clipping would destroy the central
        # derivative.  Reduce epsilon until the pair is genuinely interior.
        if np.any(plus_fraction > 1.0) or np.any(plus_fraction < 0.0) or np.any(minus_fraction > 1.0) or np.any(minus_fraction < 0.0):
            epsilon *= 0.5
            continue
        plus = decode_fractions_v127(
            fractions=plus_fraction,
            active_target=active_target,
            min_setting=min_setting,
            max_setting=max_setting,
            design=design,
        )
        minus = decode_fractions_v127(
            fractions=minus_fraction,
            active_target=active_target,
            min_setting=min_setting,
            max_setting=max_setting,
            design=design,
        )
        midpoint_error = float(np.max(np.abs(0.5 * (plus + minus) - center)))
        displacement_plus = plus - center
        displacement_minus = center - minus
        displacement_error = float(np.max(np.abs(displacement_plus - displacement_minus)))
        magnitude = float(np.max(np.abs(displacement_plus)))
        if (
            midpoint_error <= design.symmetry_atol
            and displacement_error <= 2.0 * design.symmetry_atol
            and magnitude > 1.0e-7
        ):
            return {
                "epsilon": epsilon,
                "center_sequence": center,
                "plus_sequence": plus,
                "minus_sequence": minus,
                "center_fractions": center_fraction.copy(),
                "plus_fractions": plus_fraction.astype(np.float32),
                "minus_fractions": minus_fraction.astype(np.float32),
                "direction_fractions": direction.copy(),
                "midpoint_error": midpoint_error,
                "displacement_symmetry_error": displacement_error,
                "max_setting_displacement": magnitude,
            }
        epsilon *= 0.5
    return None


def sequence_sha256_v127(sequence: np.ndarray) -> str:
    values = np.asarray(sequence, dtype="<f4")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def json_matrix_v127(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value, dtype=float).tolist(), separators=(",", ":"))


__all__ = [
    "D5GradientDesignV127",
    "V127_D5_CONTRACT",
    "broad_center_fractions_v127",
    "decode_fractions_v127",
    "deterministic_rainfall_roles_v127",
    "directional_fractions_v127",
    "json_matrix_v127",
    "sequence_sha256_v127",
    "symmetric_probe_v127",
]
