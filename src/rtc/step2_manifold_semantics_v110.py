"""Control-manifold descriptors for the frozen D2/D3 counterfactual data.

D2 is valuable mechanism supervision even when a one-actuator perturbation is
not exactly representable by the frozen MPC basis. Targeted D3-v2 was generated
on that basis and should therefore be close to it. Quantifying this distinction
prevents the trainer/evaluator from treating all 7,200 branches as exchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .step2_control_basis_v60 import ControlBasisV60

V110_MANIFOLD_SEMANTICS_CONTRACT = "PROJECT7_STEP2_V110_CONTROL_MANIFOLD_SEMANTICS_V1"


@dataclass(frozen=True)
class ManifoldDescriptorsV110:
    coefficient_l2: np.ndarray
    coefficient_linf: np.ndarray
    active_coefficient_count: np.ndarray
    active_control_group_count: np.ndarray
    active_temporal_basis_count: np.ndarray
    projection_rmse_setting: np.ndarray
    projection_residual_ratio: np.ndarray


def candidate_manifold_descriptors_v110(
    reference_settings: np.ndarray,
    candidate_settings: np.ndarray,
    basis: ControlBasisV60,
    *,
    epsilon: float = 1.0e-7,
) -> ManifoldDescriptorsV110:
    """Project [C,H,A] action differences onto the frozen 102-D MPC basis."""
    basis.validate()
    reference = np.asarray(reference_settings, dtype=np.float64)
    candidate = np.asarray(candidate_settings, dtype=np.float64)
    expected = (basis.horizon.horizon_steps, basis.grouping.actuator_count)
    if reference.shape != expected or candidate.ndim != 3 or candidate.shape[1:] != expected:
        raise ValueError("V11 manifold descriptors require [H,A] reference and [C,H,A] candidates")
    delta = candidate - reference[None]
    block_delta = delta[:, :: basis.horizon.control_block_steps, :]
    coefficients = basis.project_actions_to_coefficients(block_delta)
    flat_coeff = coefficients.reshape(candidate.shape[0], -1)

    matrix = basis.design_matrix()
    reconstructed = (matrix @ flat_coeff.T).T.reshape(block_delta.shape)
    residual = block_delta - reconstructed
    rmse = np.sqrt(np.mean(np.square(residual), axis=(1, 2)))
    signal = np.sqrt(np.mean(np.square(block_delta), axis=(1, 2)))
    ratio = np.divide(rmse, signal, out=np.zeros_like(rmse), where=signal > 1.0e-12)

    active = np.abs(coefficients) > float(epsilon)
    return ManifoldDescriptorsV110(
        coefficient_l2=np.linalg.norm(flat_coeff, axis=1),
        coefficient_linf=np.max(np.abs(flat_coeff), axis=1, initial=0.0),
        active_coefficient_count=active.sum(axis=(1, 2)).astype(np.int64),
        active_control_group_count=active.any(axis=1).sum(axis=1).astype(np.int64),
        active_temporal_basis_count=active.any(axis=2).sum(axis=1).astype(np.int64),
        projection_rmse_setting=rmse,
        projection_residual_ratio=ratio,
    )


def summarize_manifold_descriptors_v110(
    descriptors: dict[str, ManifoldDescriptorsV110],
) -> dict[str, Any]:
    def stats(values: np.ndarray) -> dict[str, float]:
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
        return {
            "median": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
            "p95": float(np.quantile(x, 0.95)),
            "max": float(x.max()),
        }

    return {
        "contract": V110_MANIFOLD_SEMANTICS_CONTRACT,
        "interpretation": {
            "D2": "single-actuator mechanism supervision; may legitimately lie away from grouped MPC manifold",
            "D3": "targeted joint-action supervision generated on/near the frozen MPC manifold",
            "use": "stratification/coverage/routing diagnostics; never an outcome label",
        },
        "sources": {
            source: {
                "candidate_count": int(desc.coefficient_l2.size),
                "coefficient_l2": stats(desc.coefficient_l2),
                "active_coefficient_count": stats(desc.active_coefficient_count),
                "active_control_group_count": stats(desc.active_control_group_count),
                "active_temporal_basis_count": stats(desc.active_temporal_basis_count),
                "projection_rmse_setting": stats(desc.projection_rmse_setting),
                "projection_residual_ratio": stats(desc.projection_residual_ratio),
            }
            for source, desc in descriptors.items()
        },
    }


__all__ = [
    "ManifoldDescriptorsV110",
    "V110_MANIFOLD_SEMANTICS_CONTRACT",
    "candidate_manifold_descriptors_v110",
    "summarize_manifold_descriptors_v110",
]
