"""High-value D5 central-difference design for Project7 V127.

D5 is not another random candidate bank.  It exists to supervise and independently audit
continuous MPC gradients in the same 102-dimensional topology/time control basis used to
cover all 109 writable actuators.  Every label is an antithetic central difference around
an outcome-blind action centre.  A pair is accepted only when both sides remain symmetric
after the exact engineering-feasible decoder; otherwise epsilon is reduced or the pair is
rejected.  This prevents clipped/boundary probes from masquerading as gradient truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from .step2_control_basis_v60 import ControlBasisV60

V127_D5_CONTRACT = "PROJECT7_V127_D5_SYMMETRIC_DIRECTIONAL_GRADIENT_V1"


@dataclass(frozen=True)
class D5GradientDesignV127:
    max_checkpoints: int = 48
    centers_per_checkpoint: int = 3
    directions_per_center: int = 4
    initial_epsilon: float = 0.20
    minimum_epsilon: float = 0.025
    audit_fraction: float = 0.25
    free_control_blocks: int = 12
    symmetry_atol: float = 2.0e-5

    def validate(self) -> None:
        if self.max_checkpoints <= 0 or self.centers_per_checkpoint != 3:
            raise ValueError("V127 D5 checkpoint/centre design is invalid")
        if self.directions_per_center <= 0:
            raise ValueError("V127 D5 requires directional probes")
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


def broad_center_coefficients_v127(
    basis: ControlBasisV60, *, checkpoint_identity: str
) -> np.ndarray:
    """Deterministic interior broad-manifold centre; independent of SWMM outcomes."""
    rng = np.random.default_rng(_seed(f"V127_D5_CENTER|{checkpoint_identity}"))
    values = rng.uniform(-0.45, 0.45, size=(basis.temporal_basis_count, basis.group_count))
    return values.astype(np.float32)


def directional_coefficients_v127(
    basis: ControlBasisV60,
    *,
    checkpoint_identity: str,
    center_index: int,
    direction_index: int,
) -> np.ndarray:
    """One unit-L2 coefficient-space direction with deterministic coordinate coverage."""
    k, g = basis.temporal_basis_count, basis.group_count
    dim = k * g
    if dim <= 0:
        raise ValueError("V127 D5 basis is empty")
    flat = np.zeros(dim, dtype=np.float64)
    if direction_index == 0:
        # Round-robin one exact basis coordinate so every group/time mode eventually gets
        # an interpretable central finite difference.
        coordinate = _seed(
            f"V127_D5_COORD|{checkpoint_identity}|{center_index}"
        ) % dim
        flat[coordinate] = 1.0
    else:
        rng = np.random.default_rng(
            _seed(f"V127_D5_DIR|{checkpoint_identity}|{center_index}|{direction_index}")
        )
        flat = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
        # Sparse Rademacher directions reduce cancellation while probing interactions.
        keep = rng.random(dim) < min(0.25, 12.0 / max(dim, 1))
        if not keep.any():
            keep[rng.integers(0, dim)] = True
        flat *= keep
    norm = float(np.linalg.norm(flat))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("V127 D5 produced a zero/non-finite direction")
    return (flat / norm).reshape(k, g).astype(np.float32)


def _hold_tail_v127(sequence: np.ndarray, *, free_blocks: int, block_steps: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[0] % block_steps:
        raise ValueError("V127 D5 sequence is not aligned to 10-min blocks")
    blocks = values[::block_steps].copy()
    if not 1 <= int(free_blocks) <= len(blocks):
        raise ValueError("V127 D5 free-control block count is invalid")
    blocks[int(free_blocks):] = blocks[int(free_blocks) - 1]
    return np.repeat(blocks, block_steps, axis=0).astype(np.float32)


def decode_coefficients_v127(
    basis: ControlBasisV60,
    *,
    reference_sequence: np.ndarray,
    coefficients: np.ndarray,
    free_control_blocks: int = 12,
) -> np.ndarray:
    ref = torch.as_tensor(reference_sequence, dtype=torch.float32)[None]
    coeff = torch.as_tensor(coefficients, dtype=torch.float32)[None]
    with torch.no_grad():
        sequence = basis.decode(ref, coeff)[0].cpu().numpy().astype(np.float32)
    return _hold_tail_v127(
        sequence,
        free_blocks=int(free_control_blocks),
        block_steps=int(basis.horizon.control_block_steps),
    )


def symmetric_probe_v127(
    basis: ControlBasisV60,
    *,
    reference_sequence: np.ndarray,
    center_coefficients: np.ndarray,
    direction: np.ndarray,
    design: D5GradientDesignV127 = D5GradientDesignV127(),
) -> dict[str, object] | None:
    """Decode a symmetric +/- probe, reducing epsilon before ever accepting distortion."""
    design.validate()
    center = decode_coefficients_v127(
        basis,
        reference_sequence=reference_sequence,
        coefficients=center_coefficients,
        free_control_blocks=design.free_control_blocks,
    )
    epsilon = float(design.initial_epsilon)
    while epsilon >= float(design.minimum_epsilon) - 1e-12:
        plus = decode_coefficients_v127(
            basis,
            reference_sequence=reference_sequence,
            coefficients=np.asarray(center_coefficients) + epsilon * np.asarray(direction),
            free_control_blocks=design.free_control_blocks,
        )
        minus = decode_coefficients_v127(
            basis,
            reference_sequence=reference_sequence,
            coefficients=np.asarray(center_coefficients) - epsilon * np.asarray(direction),
            free_control_blocks=design.free_control_blocks,
        )
        midpoint_error = float(np.max(np.abs(0.5 * (plus + minus) - center)))
        displacement_plus = plus - center
        displacement_minus = center - minus
        displacement_error = float(np.max(np.abs(displacement_plus - displacement_minus)))
        magnitude = float(np.max(np.abs(displacement_plus)))
        if (
            midpoint_error <= design.symmetry_atol
            and displacement_error <= 2.0 * design.symmetry_atol
            and magnitude > 1.0e-6
        ):
            return {
                "epsilon": epsilon,
                "center_sequence": center,
                "plus_sequence": plus,
                "minus_sequence": minus,
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
    "broad_center_coefficients_v127",
    "decode_coefficients_v127",
    "deterministic_rainfall_roles_v127",
    "directional_coefficients_v127",
    "json_matrix_v127",
    "sequence_sha256_v127",
    "symmetric_probe_v127",
]
