"""Information-efficient D5 central-difference design for Project7 V127.

D5 validates the exact variable space that the online optimiser sees: twelve free
10-minute target-fraction blocks for every writable actuator.  It is intentionally a
small, high-information gradient data set, not another broad random candidate bank.

Centres stay close to operating regions the rolling controller can plausibly visit
(HOLD, Sparse-RBC and a local non-RBC exploration point between/around them).  Directions
probe interpretable first-move, persistent, temporal and coordinated effects in the same
12 x 109 fraction tensor used by online L-BFGS-B.  Every authoritative label is an
antithetic central difference.  A pair is accepted only when the exact online decoder
preserves central symmetry; otherwise epsilon is reduced or the pair is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from .step3_mpc_v127 import ContinuousMPCDesignV127, decode_fractional_targets_v127

V127_D5_CONTRACT = "PROJECT7_V127_D5_INFORMATION_EFFICIENT_1308VAR_GRADIENT_V3"


@dataclass(frozen=True)
class D5GradientDesignV127:
    # D5 complements D2/D3; it is not intended to reproduce the whole action space by SWMM.
    max_checkpoints: int = 24
    centers_per_checkpoint: int = 3
    directions_per_center: int = 6
    initial_epsilon: float = 0.20
    minimum_epsilon: float = 0.0125
    audit_fraction: float = 0.25
    free_control_blocks: int = 12
    symmetry_atol: float = 2.5e-5

    def validate(self) -> None:
        if self.max_checkpoints <= 0 or self.centers_per_checkpoint != 3:
            raise ValueError("V127 D5 checkpoint/centre design is invalid")
        if self.directions_per_center < 4:
            raise ValueError("V127 D5 requires multiple temporal/spatial gradient directions")
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


def local_exploration_center_fractions_v127(
    hold_fractions: np.ndarray,
    rbc_fractions: np.ndarray,
    *,
    checkpoint_identity: str,
) -> np.ndarray:
    """Build a deterministic non-RBC centre close to the online HOLD/RBC corridor.

    A globally random 1308-D centre is expensive to label and is unlikely to be visited by
    a receding-horizon controller.  This centre starts halfway between HOLD and RBC, then
    perturbs a small deterministic set of coordinates.  It therefore adds non-RBC local
    curvature information without turning D5 into an out-of-distribution action bank.
    """
    hold = np.asarray(hold_fractions, dtype=np.float32)
    rbc = np.asarray(rbc_fractions, dtype=np.float32)
    if hold.shape != rbc.shape or hold.ndim != 2:
        raise ValueError("V127 D5 HOLD/RBC fraction centres are misaligned")
    base = 0.5 * (hold + rbc)
    rng = np.random.default_rng(_seed(f"V127_D5_LOCAL_CENTER|{checkpoint_identity}"))
    flat = base.reshape(-1).astype(np.float64)
    dim = flat.size
    count = min(36, max(12, dim // 40))
    # Guarantee that part of the exploration touches the executed first 10-min block.
    first_block = np.arange(hold.shape[1], dtype=int)
    first_count = min(max(4, count // 3), len(first_block))
    first = rng.choice(first_block, size=first_count, replace=False)
    remaining_pool = np.setdiff1d(np.arange(dim, dtype=int), first, assume_unique=False)
    rest = rng.choice(remaining_pool, size=count - first_count, replace=False)
    chosen = np.concatenate((first, rest))
    flat[chosen] += rng.choice(np.asarray([-0.12, 0.12]), size=count)
    return np.clip(flat.reshape(base.shape), 0.05, 0.95).astype(np.float32)


def directional_fractions_v127(
    actuator_count: int,
    *,
    checkpoint_identity: str,
    checkpoint_rank: int,
    center_index: int,
    direction_index: int,
    free_control_blocks: int = 12,
) -> tuple[np.ndarray, str]:
    """Deterministic unit-L2 direction in the exact online MPC variable space.

    Six families are deliberately retained because D5 is a gradient complement to the
    existing D2/D3 data, not a 1308-coordinate finite-difference sweep.  The set focuses on
    what can change the executed first move while retaining enough temporal/spatial probes
    to test H120 credit assignment and joint-action interactions.
    """
    blocks = int(free_control_blocks)
    actuators = int(actuator_count)
    dim = blocks * actuators
    if min(blocks, actuators) <= 0:
        raise ValueError("V127 D5 fraction direction dimensions are invalid")
    flat = np.zeros(dim, dtype=np.float64)
    family_index = int(direction_index) % 6
    sweep = (
        int(checkpoint_rank) * 3 * max(6, int(direction_index) + 1)
        + int(center_index) * 6
        + int(direction_index)
    )
    rng = np.random.default_rng(
        _seed(
            f"V127_D5_DIR|{checkpoint_identity}|{checkpoint_rank}|{center_index}|{direction_index}"
        )
    )

    if family_index == 0:
        actuator = sweep % actuators
        flat[actuator] = 1.0
        family = "first_move_single_actuator"
    elif family_index == 1:
        actuator = (sweep * 7 + 3) % actuators
        flat[np.arange(blocks) * actuators + actuator] = 1.0
        family = "persistent_single_actuator"
    elif family_index == 2:
        actuator = (sweep * 13 + 1) % actuators
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=blocks)
        flat[np.arange(blocks) * actuators + actuator] = signs
        family = "temporal_single_actuator"
    elif family_index == 3:
        count = min(12, actuators)
        chosen = rng.choice(actuators, size=count, replace=False)
        flat[chosen] = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        family = "first_move_multi_actuator"
    elif family_index == 4:
        block = (checkpoint_rank + center_index + direction_index) % blocks
        count = min(20, actuators)
        chosen = rng.choice(actuators, size=count, replace=False)
        flat[block * actuators + chosen] = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        family = "single_block_spatial_interaction"
    else:
        count = min(max(24, dim // 40), dim)
        chosen = rng.choice(dim, size=count, replace=False)
        flat[chosen] = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        # Ensure every sparse spatiotemporal direction contains some executed-first-block
        # information, rather than spending a SWMM pair entirely in a remote tail.
        first = rng.choice(actuators, size=min(4, actuators), replace=False)
        flat[first] = rng.choice(np.asarray([-1.0, 1.0]), size=len(first))
        family = "sparse_spatiotemporal_interaction"

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
        # Never clip antithetic variables: clipping would invalidate the central derivative.
        if (
            np.any(plus_fraction > 1.0)
            or np.any(plus_fraction < 0.0)
            or np.any(minus_fraction > 1.0)
            or np.any(minus_fraction < 0.0)
        ):
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
        physical_l2 = float(np.linalg.norm(displacement_plus))
        first_move_l2 = float(np.linalg.norm(displacement_plus[:2]))
        if (
            midpoint_error <= design.symmetry_atol
            and displacement_error <= 2.0 * design.symmetry_atol
            and physical_l2 > 1.0e-6
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
                "physical_displacement_l2": physical_l2,
                "first_move_displacement_l2": first_move_l2,
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
    "decode_fractions_v127",
    "deterministic_rainfall_roles_v127",
    "directional_fractions_v127",
    "json_matrix_v127",
    "local_exploration_center_fractions_v127",
    "sequence_sha256_v127",
    "symmetric_probe_v127",
]
