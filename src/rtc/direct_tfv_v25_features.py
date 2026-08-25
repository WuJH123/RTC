"""Shared, deployment-parity feature extraction for the V25 value gate.

The V25 calibrator is deliberately not allowed to invent a second candidate generator.  This
module executes the existing V23 portfolio once, applies the frozen V15 rank branch, and builds
the already-defined V21 selected-query feature for the rank-selected candidate.  It is used by
both the Development trainer and the online V25 runtime so the value model sees the same feature
contract in training and deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch

from .direct_tfv_operational_v23_runtime import _sha
from .direct_tfv_policy_return_facility_boundary_v20 import (
    build_facility_boundary_parts_v20,
)
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
)
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_policy_return_portfolio_v23 import (
    V23_HYDRAULIC_CANDIDATE_CONTRACT,
    V23_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio_v23,
)
from .direct_tfv_policy_return_query_margin_v2 import build_query_margin_v2_features
from .direct_tfv_policy_return_selected_boundary_v21 import (
    build_selected_portfolio_feature_v21,
)
from .direct_tfv_sequence_support import changed_facility_support_limit


V25_FEATURE_CONTRACT = (
    "PROJECT7_STEP3_V25_SELECTED_V23_PORTFOLIO_V15_RANK_V21_FEATURE_V1"
)


def first_target_sha256(target: torch.Tensor) -> str:
    """Hash the exact float64 first-action vector used by the truth audit."""

    value = np.ascontiguousarray(target.detach().cpu().to(torch.float64).numpy())
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class V25SelectedQuery:
    """Deployment-parity selected candidate and its causal value features."""

    feature: torch.Tensor
    selected_source: str
    selected_target: torch.Tensor
    selected_sequence: torch.Tensor
    selected_target_sha256: str
    selected_changed_facility_count: int
    selected_support: dict[str, Any]
    candidate_sources: tuple[str, ...]
    raw_rank_scores_m3: tuple[float, ...]
    relative_rank_scores: tuple[float, ...]
    base_step2_scores_m3: tuple[float, ...]
    network_stress_q75: float
    strong_storm_blend: float
    candidate_count: int


def _as_context_state(current_state: torch.Tensor) -> torch.Tensor:
    if current_state.ndim == 3 and int(current_state.shape[0]) == 1:
        return current_state[0]
    return current_state


def build_v25_selected_query(
    *,
    mpc: Any,
    current_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_actuator_flow: torch.Tensor,
    active_target: torch.Tensor,
) -> V25SelectedQuery:
    """Build one selected-query feature using the frozen current V23 ranking path.

    The function intentionally mirrors ``DirectTFVOperationalV23MPC.optimize``.  In particular,
    it scores every V23 candidate, computes the V15 relative rank, and only then constructs the
    single V21 query feature.  No truth, stress threshold, or V21 HOLD head is consulted here.
    """

    if tuple(active_target.shape) != (109,):
        raise ValueError("V25 active_target must be [109]")
    if tuple(previous_actuator_flow.shape) != (1, 109):
        raise ValueError("V25 previous_actuator_flow must be [1,109]")
    ceiling = changed_facility_support_limit(mpc.sequence_support, "q95")
    hybrid = build_hybrid_policy_return_portfolio_v23(
        model=mpc.model,
        normalization=mpc.normalization,
        graph=mpc.graph,
        current_state=current_state,
        rainfall_scenarios=rainfall,
        previous_actuator_flow=previous_actuator_flow,
        active_target=active_target,
        first_radius=mpc.first_radius,
        max_changed_facilities=int(ceiling),
        max_delta_per_update=float(mpc.design.max_setting_delta_per_update),
        probe_chunk_size=mpc.proposal_probe_chunk_size,
        supervisory_mask=mpc.supervisory_mask,
    )
    passive = torch.as_tensor(
        ~mpc.supervisory_mask,
        dtype=torch.bool,
        device=active_target.device,
    )
    evaluated: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    with torch.inference_mode():
        for proposal in hybrid.candidates:
            target, sequence, changed, support = mpc._h10_supported_target(
                proposal.target, active_target
            )
            if int(changed) <= 0:
                continue
            if bool(torch.any(torch.abs(target[passive] - active_target[passive]) > 1.0e-7)):
                raise RuntimeError("V25 candidate changed a passive channel")
            key = target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen:
                continue
            seen.add(key)
            raw_rank = float(
                mpc._score_policy_return_target(
                    current_state=current_state,
                    rainfall=rainfall,
                    previous_actuator_flow=previous_actuator_flow,
                    active_target=active_target,
                    candidate_target=target,
                ).detach().cpu()
            )
            base_score = float(
                score_h10_first_action_targets(
                    model=mpc.model,
                    normalization=mpc.normalization,
                    graph=mpc.graph,
                    current_state=current_state,
                    rainfall_scenarios=rainfall,
                    previous_actuator_flow=previous_actuator_flow,
                    active_target=active_target,
                    candidate_targets=target.reshape(1, 109),
                    probe_chunk_size=1,
                )[0].detach().cpu()
            )
            facility = build_facility_boundary_parts_v20(
                step2_model=mpc.model,
                normalization=mpc.normalization,
                graph=mpc.graph,
                current_state=_as_context_state(current_state),
                rainfall_scenarios=rainfall,
                previous_actuator_flow=previous_actuator_flow[0],
                active_target=active_target,
                candidate_target=target,
                candidate_source=str(proposal.source),
                supervisory_mask=mpc.supervisory_mask,
                target_scale_m3=float(mpc.model.target_scale_m3.detach().cpu()),
            ).feature
            evaluated.append(
                {
                    "source": str(proposal.source),
                    "target": target,
                    "sequence": sequence,
                    "changed": int(changed),
                    "raw_rank": raw_rank,
                    "base_score": base_score,
                    "support": dict(support),
                    "facility": facility,
                }
            )
    if not evaluated:
        raise RuntimeError("V25 V23 portfolio produced no non-empty candidate")

    targets = torch.stack([row["target"] for row in evaluated])
    raw_rank = torch.as_tensor(
        [row["raw_rank"] for row in evaluated],
        dtype=targets.dtype,
        device=targets.device,
    )
    base_scores = torch.as_tensor(
        [row["base_score"] for row in evaluated],
        dtype=targets.dtype,
        device=targets.device,
    )
    context_features, candidate_features = build_query_margin_v2_features(
        step2_model=mpc.model,
        normalization=mpc.normalization,
        graph=mpc.graph,
        current_state=_as_context_state(current_state),
        rainfall_scenarios=rainfall,
        previous_actuator_flow=previous_actuator_flow[0],
        active_target=active_target,
        candidate_targets=targets,
        base_step2_scores_m3=base_scores,
        candidate_sources=[row["source"] for row in evaluated],
        supervisory_mask=mpc.supervisory_mask,
        target_scale_m3=float(mpc.model.target_scale_m3.detach().cpu()),
    )
    with torch.inference_mode():
        joint = mpc.rank_adapter(
            raw_rank_scores_m3=raw_rank,
            context_features=context_features,
            candidate_features=candidate_features,
        )
    relative = joint.relative_rank_normalized
    selected_index = int(torch.argmin(relative).item())
    selected = evaluated[selected_index]
    facility_features = torch.stack([row["facility"] for row in evaluated])
    mask = torch.as_tensor(mpc.supervisory_mask, dtype=torch.bool, device=active_target.device)
    action_mass = torch.sqrt(
        torch.mean(torch.square((selected["target"] - active_target)[mask]))
    )
    parts = build_selected_portfolio_feature_v21(
        candidate_features=facility_features,
        rank_scores=relative,
        selected_index=selected_index,
        selected_action_mass=action_mass,
    )
    hydraulic = hybrid.hydraulic_diagnostics
    return V25SelectedQuery(
        feature=parts.feature.detach(),
        selected_source=str(selected["source"]),
        selected_target=selected["target"].detach(),
        selected_sequence=selected["sequence"].detach(),
        selected_target_sha256=first_target_sha256(selected["target"]),
        selected_changed_facility_count=int(selected["changed"]),
        selected_support=dict(selected["support"]),
        candidate_sources=tuple(row["source"] for row in evaluated),
        raw_rank_scores_m3=tuple(float(row["raw_rank"]) for row in evaluated),
        relative_rank_scores=tuple(float(value) for value in relative.detach().cpu().tolist()),
        base_step2_scores_m3=tuple(float(row["base_score"]) for row in evaluated),
        network_stress_q75=float(hydraulic.network_stress_q75),
        strong_storm_blend=float(hydraulic.strong_storm_blend),
        candidate_count=len(evaluated),
    )


__all__ = [
    "V23_HYDRAULIC_CANDIDATE_CONTRACT",
    "V23_PORTFOLIO_CONTRACT",
    "V25_FEATURE_CONTRACT",
    "V25SelectedQuery",
    "build_v25_selected_query",
    "first_target_sha256",
]
