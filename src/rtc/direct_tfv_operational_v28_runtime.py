"""Project7 V28 q95-matched residual-value Development runtime.

The V28 execution path is deliberately narrow: V23 generates the existing engineering portfolio,
every raw proposal is projected into the frozen q95 support manifold, equivalent supported targets
are deduplicated, frozen V27 (Q27) scores are corrected by a small residual model, and HOLD=0 is
included in the final argmin.  Raw proposals are never executable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch

from .closed_loop import CausalObservation, ControllerAction
from .controller_direct_tfv_portfolio import PortfolioMemorySafeDirectTFVAuthoritativeController
from .direct_tfv_operational_v23_runtime import (
    DirectTFVOperationalV23MPC,
    build_operational_v23_controller,
)
from .direct_tfv_operational_v27_runtime import (
    V27_AUTO_RBC_SHADOW_SOURCE,
    V27_PORTFOLIO_CONTRACT,
    DirectTFVOperationalV27MPC,
    build_operational_v27_controller,
)
from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    encode_policy_return_action_token,
)
from .direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT
from .direct_tfv_policy_return_portfolio_v23 import (
    V23_HYDRAULIC_CANDIDATE_CONTRACT,
    V23_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio_v23,
)
from .direct_tfv_sequence_support import changed_facility_support_limit
from .direct_tfv_v28_residual_value import (
    V28_RESIDUAL_FEATURE_NAMES,
    V28ResidualValueModel,
    build_v28_residual_features,
    load_v28_residual_value_model,
    q28_score_m3,
)
from .runtime_controller_guard import ContinuityGuardController
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12


V28_OPERATIONAL_RUNTIME_CONTRACT = (
    "PROJECT7_OPERATIONAL_DEVELOPMENT_V28_Q95_MATCHED_RESIDUAL_VALUE_RTC_V1"
)
V28_SELECTION_CONTRACT = (
    "PROJECT7_V28_ARGMIN_Q28_UNCLIPPED_SUPPORTED_VALUE_PLUS_HOLD_ZERO_V1"
)
V28_PORTFOLIO_CONTRACT = "PROJECT7_V28_V23_PORTFOLIO_Q95_SUPPORTED_RESIDUAL_VALUE_V1"
V28_TFV_VALUE_ESTIMAND = (
    "SYSTEM_WIDE_CUMULATIVE_TFV_CANDIDATE_H10_PLUS_FROZEN_CAUSAL_CONTINUATION_MINUS_"
    "HOLD_H10_PLUS_IDENTICAL_CONTINUATION_V1"
)


def _v23_strong_storm_blend(network_stress_q75: float) -> float:
    """Reconstruct the frozen V23 diagnostic blend from its causal stress feature."""

    return float(np.clip((float(network_stress_q75) - 0.65) / 0.25, 0.0, 1.0))


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _target_key(target: Any) -> bytes:
    if isinstance(target, torch.Tensor):
        return target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    return np.ascontiguousarray(np.asarray(target, dtype=np.float32).reshape(-1)).tobytes()


def _target_hash(target: Any) -> str:
    return hashlib.sha256(_target_key(target)).hexdigest()


@dataclass
class V28Candidate:
    """One canonical post-q95 candidate plus its raw proposal metadata."""

    source: str
    target: Any
    supported_target: Any
    supported_sequence: Any
    q27_score_m3: float
    residual_m3: float
    q28_score_m3: float
    q95_scale: float
    q95_max_ratio: float
    q95_binding: bool
    raw_first_move_l1: float
    supported_first_move_l1: float
    raw_to_supported_first_move_l1: float
    raw_to_supported_h120_l1: float
    raw_to_supported_tv_l1: float
    changed_facility_count: int
    network_stress_q75: float
    rain_level: float
    strong_storm_blend: float
    candidate_selected: bool
    contributing_sources: tuple[str, ...]
    raw_target: Any | None = None
    raw_target_sha256: str = ""
    supported_target_sha256: str = ""
    q27_latent: float = 0.0
    q27_reported_m3: float = 0.0
    q27_clip_hit: bool = False
    raw_q27_latent: float | None = None
    raw_q27_reported_m3: float | None = None
    raw_q27_clip_hit: bool | None = None

    @property
    def executed_target(self) -> Any:
        """Only the q95-supported target can be sent to the actuator writer."""

        return self.supported_target

    def telemetry(self) -> dict[str, Any]:
        return {
            "candidate_source": self.source,
            "contributing_sources": list(self.contributing_sources),
            "raw_target_sha256": self.raw_target_sha256 or _target_hash(self.target),
            "supported_target_sha256": self.supported_target_sha256 or _target_hash(self.supported_target),
            "q95_scale": float(self.q95_scale),
            "q95_max_ratio": float(self.q95_max_ratio),
            "q95_binding": bool(self.q95_binding),
            "raw_first_move_l1": float(self.raw_first_move_l1),
            "supported_first_move_l1": float(self.supported_first_move_l1),
            "raw_to_supported_first_move_l1": float(self.raw_to_supported_first_move_l1),
            "raw_to_supported_h120_l1": float(self.raw_to_supported_h120_l1),
            "raw_to_supported_total_variation_l1": float(self.raw_to_supported_tv_l1),
            "changed_facility_count": int(self.changed_facility_count),
            "q27_score_m3": float(self.q27_score_m3),
            "q27_residual_m3": float(self.residual_m3),
            "q28_score_m3": float(self.q28_score_m3),
            "margin_to_hold_m3": float(-self.q28_score_m3),
            "network_stress_q75": float(self.network_stress_q75),
            "rain_level": float(self.rain_level),
            "strong_storm_blend": float(self.strong_storm_blend),
            "candidate_selected": bool(self.candidate_selected),
            "candidate_executed": False,
            "q27_clip_hit": bool(self.q27_clip_hit),
        }


def post_q95_deduplicate(candidates: Iterable[V28Candidate]) -> tuple[list[V28Candidate], int]:
    """Deduplicate by the float32 supported target and preserve source contributors."""

    unique: list[V28Candidate] = []
    positions: dict[bytes, int] = {}
    duplicate_count = 0
    for candidate in candidates:
        key = _target_key(candidate.supported_target)
        position = positions.get(key)
        if position is None:
            candidate.contributing_sources = (str(candidate.source),)
            unique.append(candidate)
            positions[key] = len(unique) - 1
            continue
        duplicate_count += 1
        existing = unique[position]
        sources = list(existing.contributing_sources)
        for source in (str(candidate.source), *candidate.contributing_sources):
            if source not in sources:
                sources.append(source)
        existing.contributing_sources = tuple(sources)
    return unique, int(duplicate_count)


def select_v28_candidate(
    candidates: list[V28Candidate],
    *,
    residuals: Iterable[float] | None = None,
) -> V28Candidate:
    """Select the minimum Q28 candidate without source-priority or stress authority."""

    if not candidates:
        raise ValueError("V28 cannot select from an empty candidate list")
    if residuals is not None:
        values = list(residuals)
        if len(values) != len(candidates):
            raise ValueError("V28 residual count does not match candidate count")
        for candidate, residual in zip(candidates, values, strict=True):
            candidate.residual_m3 = float(residual)
            candidate.q28_score_m3 = q28_score_m3(candidate.q27_score_m3, candidate.residual_m3)
    selected = min(candidates, key=lambda candidate: (float(candidate.q28_score_m3), str(candidate.source)))
    for candidate in candidates:
        candidate.candidate_selected = candidate is selected
    return selected


def validate_v28_executable_candidate(
    supported_target: Any,
    active_target: Any,
    supervisory_mask: Any,
) -> bool:
    target = np.asarray(supported_target, dtype=np.float64).reshape(-1)
    active = np.asarray(active_target, dtype=np.float64).reshape(-1)
    mask = np.asarray(supervisory_mask, dtype=bool).reshape(-1)
    if target.shape != active.shape or mask.shape != active.shape or not np.isfinite(target).all():
        raise ValueError("V28 executable candidate shape/content is invalid")
    if np.any(target < -1.0e-8) or np.any(target > 1.0 + 1.0e-8):
        raise ValueError("V28 executable candidate violates actuator bounds")
    if np.any(np.abs(target[~mask] - active[~mask]) > 1.0e-7):
        raise ValueError("V28 executable candidate changed a passive channel")
    return True


def build_v28_lineage(
    *,
    q27_checkpoint_sha256: str,
    residual_checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    asset_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "contract": V28_OPERATIONAL_RUNTIME_CONTRACT,
        "development_only": True,
        "formal_evidence": False,
        "ready_for_policy_lock": False,
        "q95_mandatory": True,
        "raw_action_executable": False,
        "q27_frozen": True,
        "q27_checkpoint_sha256": str(q27_checkpoint_sha256).lower(),
        "residual_checkpoint_sha256": str(residual_checkpoint_sha256).lower(),
        "dataset_manifest_sha256": str(dataset_manifest_sha256).lower(),
        "asset_manifest_sha256": str(asset_manifest_sha256).lower(),
        "residual_feature_contract": "|".join(V28_RESIDUAL_FEATURE_NAMES),
        "event_id_feature": False,
        "future_realized_rainfall_used_as_model_input": False,
        "v15_rank_used_for_v28_candidate_selection": False,
        "v21_boundary_used_for_v28_action_admission": False,
        "v25_ucb_used_for_v28_action_admission": False,
        "post_q95_dedup": True,
        "auto_rbc_shadow_is_candidate_only": True,
        "v27_q95_execution_inherited": True,
        "v23_portfolio_contract": V23_PORTFOLIO_CONTRACT,
        "v23_hydraulic_candidate_contract": V23_HYDRAULIC_CANDIDATE_CONTRACT,
        "selection_contract": V28_SELECTION_CONTRACT,
        "tfv_value_estimand": V28_TFV_VALUE_ESTIMAND,
        "scientific_metrics_block_runtime": False,
    }


def _as_state(current_state: torch.Tensor) -> torch.Tensor:
    return current_state[0] if current_state.ndim == 3 and int(current_state.shape[0]) == 1 else current_state


def _q27_m3_from_latent(module: torch.nn.Module, latent: float) -> float:
    scale = float(getattr(module, "target_scale_m3").detach().cpu())
    value = float(np.sinh(float(latent)) * scale)
    if not np.isfinite(value):
        raise ValueError("V27 unclipped score is non-finite")
    return value


class DirectTFVOperationalV28MPC(DirectTFVOperationalV27MPC):
    """V28 q95-supported argmin with frozen Q27 plus a residual correction."""

    policy_mode = "operational_development_v28_q95_matched_residual_value"
    policy_mode_contract = V28_OPERATIONAL_RUNTIME_CONTRACT

    def __init__(
        self,
        *,
        residual_value_model: V28ResidualValueModel,
        v28_lineage: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.residual_value_model = residual_value_model
        self.v28_lineage = dict(v28_lineage)
        self.last_v28_telemetry: dict[str, Any] = {}

    def _raw_supported_geometry(
        self,
        *,
        raw_sequence: torch.Tensor,
        supported_sequence: torch.Tensor,
        active_target: torch.Tensor,
        raw_target: torch.Tensor,
        supported_target: torch.Tensor,
    ) -> dict[str, float]:
        raw_geometry = self._joint_sequence_geometry_torch(raw_sequence, active_target)
        supported_geometry = self._joint_sequence_geometry_torch(supported_sequence, active_target)
        raw_first = float(raw_geometry["first_block_l1"].detach().cpu())
        supported_first = float(supported_geometry["first_block_l1"].detach().cpu())
        scale = supported_first / raw_first if raw_first > 1.0e-12 else 1.0
        raw_delta = raw_sequence[1:] - raw_sequence[:-1]
        supported_delta = supported_sequence[1:] - supported_sequence[:-1]
        return {
            "q95_scale": float(np.clip(scale, 0.0, 1.0)),
            "raw_first_move_l1": raw_first,
            "supported_first_move_l1": supported_first,
            "raw_to_supported_first_move_l1": float(torch.abs(raw_target - supported_target).sum().detach().cpu()),
            "raw_to_supported_h120_l1": float(torch.abs(raw_sequence - supported_sequence).sum().detach().cpu()),
            "raw_to_supported_tv_l1": float(torch.abs(raw_delta - supported_delta).sum().detach().cpu()),
        }

    def _make_v28_candidate(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        flow: torch.Tensor,
        active_target: torch.Tensor,
        source: str,
        raw_target: torch.Tensor,
    ) -> V28Candidate | None:
        passive = torch.as_tensor(~self.supervisory_mask, dtype=torch.bool, device=active_target.device)
        if bool(torch.any(torch.abs(raw_target[passive] - active_target[passive]) > 1.0e-7)):
            raise RuntimeError("V28 raw candidate changed a passive channel")
        if not bool(torch.isfinite(raw_target).all()):
            raise RuntimeError("V28 raw candidate contains non-finite settings")
        if bool(torch.any(raw_target < -1.0e-8)) or bool(torch.any(raw_target > 1.0 + 1.0e-8)):
            raise RuntimeError("V28 raw candidate violates actuator bounds")
        _, raw_batch = encode_policy_return_action_token(
            active_target,
            raw_target,
            horizon_steps=int(self.design.prediction_horizon_steps),
            first_action_steps=int(self.design.control_block_steps),
        )
        raw_sequence = raw_batch[0].detach()
        supported_target, supported_sequence, changed, support = self._h10_supported_target(raw_target, active_target)
        if int(changed) <= 0:
            return None
        validate_v28_executable_candidate(
            supported_target.detach().cpu().numpy(),
            active_target.detach().cpu().numpy(),
            self.supervisory_mask,
        )
        geometry = self._raw_supported_geometry(
            raw_sequence=raw_sequence,
            supported_sequence=supported_sequence,
            active_target=active_target,
            raw_target=raw_target,
            supported_target=supported_target,
        )
        with torch.inference_mode():
            built, q27_latent_tensor, q27_reported_tensor, q27_clip_hit = self._feature_and_scores(
                current_state=current_state,
                rainfall=rainfall,
                flow=flow,
                active_target=active_target,
                target=supported_target,
                source=source,
            )
        q27_latent = float(q27_latent_tensor)
        q27_score = _q27_m3_from_latent(self.v27_value_model, q27_latent)
        residual_features = build_v28_residual_features(
            q27_score_m3=q27_score,
            q95_scale=geometry["q95_scale"],
            q95_max_ratio=float(support["max_ratio"]),
            q95_binding=bool(support["binding"]),
            raw_first_move_l1=geometry["raw_first_move_l1"],
            supported_first_move_l1=geometry["supported_first_move_l1"],
            raw_to_supported_first_move_l1=geometry["raw_to_supported_first_move_l1"],
            raw_to_supported_h120_l1=geometry["raw_to_supported_h120_l1"],
            raw_to_supported_tv_l1=geometry["raw_to_supported_tv_l1"],
            changed_facility_count=int(changed),
            network_stress_q75=float(built.base.network_stress_q75),
            rain_level=float(built.base.rain_level),
            strong_storm_blend=_v23_strong_storm_blend(built.base.network_stress_q75),
            candidate_source=source,
        )
        residual = self.residual_value_model.predict_m3(residual_features)
        q28 = q28_score_m3(q27_score, residual)
        return V28Candidate(
            source=str(source),
            target=raw_target.detach(),
            supported_target=supported_target.detach(),
            supported_sequence=supported_sequence.detach(),
            q27_score_m3=q27_score,
            residual_m3=residual,
            q28_score_m3=q28,
            q95_scale=geometry["q95_scale"],
            q95_max_ratio=float(support["max_ratio"]),
            q95_binding=bool(support["binding"]),
            raw_first_move_l1=geometry["raw_first_move_l1"],
            supported_first_move_l1=geometry["supported_first_move_l1"],
            raw_to_supported_first_move_l1=geometry["raw_to_supported_first_move_l1"],
            raw_to_supported_h120_l1=geometry["raw_to_supported_h120_l1"],
            raw_to_supported_tv_l1=geometry["raw_to_supported_tv_l1"],
            changed_facility_count=int(changed),
            network_stress_q75=float(built.base.network_stress_q75),
            rain_level=float(built.base.rain_level),
            strong_storm_blend=_v23_strong_storm_blend(built.base.network_stress_q75),
            candidate_selected=False,
            contributing_sources=(str(source),),
            raw_target=raw_target.detach(),
            raw_target_sha256=_target_hash(raw_target),
            supported_target_sha256=_target_hash(supported_target),
            q27_latent=q27_latent,
            q27_reported_m3=float(q27_reported_tensor),
            q27_clip_hit=bool(q27_clip_hit),
        )

    def _hold_result(
        self,
        *,
        started: float,
        active_target: torch.Tensor,
        learned: Any,
    ) -> DirectTFVMPCResultV12:
        result = super()._hold_result(started=started, active_target=active_target, learned=learned)
        self.last_v28_telemetry = {
            "contract": V28_OPERATIONAL_RUNTIME_CONTRACT,
            "raw_candidate_count": 0,
            "q95_supported_candidate_count": 0,
            "post_q95_unique_candidate_count": 0,
            "post_q95_duplicate_count": 0,
            "selected_source": "HOLD",
            "action_class": "HOLD",
            "candidates": [],
            "selected_q27_score_m3": 0.0,
            "selected_residual_m3": 0.0,
            "selected_q28_score_m3": 0.0,
        }
        return result.__class__(
            **{
                **result.__dict__,
                "policy_mode": self.policy_mode,
                "policy_mode_contract": self.policy_mode_contract,
                "selected_source": "LATCH_PREVIOUS_TARGET_OPERATIONAL_V28_NO_CANDIDATE",
                "calibrated_admission_contract": V28_SELECTION_CONTRACT,
                "policy_return_admission_contract": V28_SELECTION_CONTRACT,
                "policy_return_portfolio_contract": V28_PORTFOLIO_CONTRACT,
                "policy_return_estimand": V28_TFV_VALUE_ESTIMAND,
                "scipy_message": "V28_STRUCTURED_TELEMETRY_IN_CONTROLLER_DIAGNOSTICS",
            }
        )

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("operational V28 requires active_target [109]")
        current_state = kwargs["current_state"]
        rainfall = kwargs["rainfall"]
        flow = kwargs["previous_actuator_flow"]
        if tuple(flow.shape) != (1, 109):
            raise ValueError("operational V28 requires previous_actuator_flow [1,109]")

        ceiling = int(changed_facility_support_limit(self.sequence_support, "q95"))
        hybrid = build_hybrid_policy_return_portfolio_v23(
            model=self.model,
            normalization=self.normalization,
            graph=self.graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active_target,
            first_radius=self.first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
            probe_chunk_size=self.proposal_probe_chunk_size,
            supervisory_mask=self.supervisory_mask,
        )
        learned = hybrid.learned_probe
        proposals: list[tuple[str, torch.Tensor]] = [
            (str(proposal.source), proposal.target.detach()) for proposal in hybrid.candidates
        ]
        shadow = None
        from .direct_tfv_v27_auto_rbc_shadow import build_auto_rbc_shadow_proposal

        shadow = build_auto_rbc_shadow_proposal(
            graph=self.graph,
            current_state=current_state,
            active_target=active_target,
            supervisory_mask=self.supervisory_mask,
            first_radius=self.first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
        )
        if int(shadow.retained_changed_facility_count) > 0:
            proposals.append((V27_AUTO_RBC_SHADOW_SOURCE, shadow.target.detach()))

        candidates: list[V28Candidate] = []
        for source, raw_target in proposals:
            candidate = self._make_v28_candidate(
                current_state=current_state,
                rainfall=rainfall,
                flow=flow,
                active_target=active_target,
                source=source,
                raw_target=raw_target,
            )
            if candidate is not None:
                candidates.append(candidate)
        unique, duplicate_count = post_q95_deduplicate(candidates)
        if not unique:
            return self._hold_result(started=started, active_target=active_target, learned=learned)
        selected = select_v28_candidate(unique)
        action = float(selected.q28_score_m3) < 0.0
        hold = active_target.reshape(1, 109).expand(int(self.design.prediction_horizon_steps), -1).detach()
        executed = selected.supported_sequence if action else hold
        changed_indices = torch.nonzero(
            torch.abs(selected.supported_target - active_target) > 1.0e-7
        ).reshape(-1).tolist()
        changed_ids = tuple(str(self.graph.actuator_ids[int(index)]) for index in changed_indices)
        for candidate in unique:
            payload = candidate.telemetry()
            payload["candidate_executed"] = bool(action and candidate is selected)
            candidate._v28_payload = payload  # type: ignore[attr-defined]
        selected_payload = selected.telemetry()
        selected_payload["candidate_executed"] = bool(action)
        self.last_v28_telemetry = {
            "contract": V28_OPERATIONAL_RUNTIME_CONTRACT,
            "raw_candidate_count": int(len(proposals)),
            "q95_supported_candidate_count": int(len(candidates)),
            "post_q95_unique_candidate_count": int(len(unique)),
            "post_q95_duplicate_count": int(duplicate_count),
            "selected_source": str(selected.source) if action else "HOLD",
            "selected_candidate_source": str(selected.source),
            "selected_q27_score_m3": float(selected.q27_score_m3),
            "selected_residual_m3": float(selected.residual_m3),
            "selected_q28_score_m3": float(selected.q28_score_m3),
            "action_class": "ACTION" if action else "HOLD",
            "q95_mandatory": True,
            "raw_action_executable": False,
            "auto_rbc_shadow_candidate_only": True,
            "candidates": [getattr(candidate, "_v28_payload", candidate.telemetry()) for candidate in unique],
        }
        support = {
            "quantile": "q95",
            "first_block_l1": selected.supported_first_move_l1,
            "h120_l1": float(self._joint_sequence_geometry_torch(selected.supported_sequence, active_target)["h120_l1"].detach().cpu()),
            "h120_total_variation_l1": float(self._joint_sequence_geometry_torch(selected.supported_sequence, active_target)["h120_total_variation_l1"].detach().cpu()),
            "max_ratio": selected.q95_max_ratio,
            "binding": selected.q95_binding,
        }
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=selected.supported_sequence,
            predicted_delta_tfv_m3=float(selected.q28_score_m3) if action else 0.0,
            raw_optimized_predicted_delta_tfv_m3=float(selected.q28_score_m3),
            selected_source=(f"OPERATIONAL_V28::{selected.source}" if action else "LATCH_PREVIOUS_TARGET_OPERATIONAL_V28_Q28_NONNEGATIVE"),
            candidate_valid=bool(action),
            admission_margin_m3=0.0,
            admission_upper_bound_m3=float(selected.q28_score_m3),
            admission_margin_kind="v28_q28_unclipped_hold_zero_reference",
            admission_passed=bool(action),
            calibrated_admission_contract=V28_SELECTION_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=82,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=int(selected.changed_facility_count),
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=int(selected.changed_facility_count) if action else 0,
            maximum_support_ratio=self._first_move_support_ratio(selected.supported_target, active_target),
            joint_sequence_support_quantile="q95",
            joint_sequence_first_block_l1=float(support["first_block_l1"]),
            joint_sequence_h120_l1=float(support["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(support["h120_total_variation_l1"]),
            joint_sequence_support_max_ratio=float(support["max_ratio"]),
            joint_sequence_support_binding=bool(support["binding"]),
            policy_return_predicted_delta_tfv_m3=float(selected.q28_score_m3),
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=float(selected.q28_score_m3),
            policy_return_admission_passed=bool(action),
            policy_return_admission_contract=V28_SELECTION_CONTRACT,
            policy_return_estimand=V28_TFV_VALUE_ESTIMAND,
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=V28_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(unique),
            policy_return_portfolio_selected_source=str(selected.source) if action else "HOLD",
            policy_return_portfolio_sources=tuple(candidate.source for candidate in unique),
            policy_return_portfolio_scores_m3=tuple(float(candidate.q28_score_m3) for candidate in unique),
            policy_return_portfolio_upper_bounds_m3=tuple(float(candidate.q28_score_m3) for candidate in unique),
            policy_return_portfolio_base_step2_scores_m3=tuple(float(candidate.q27_score_m3) for candidate in unique),
            h10_probe_generator_contract=DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
            h10_probe_count=int(learned.probe_count),
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics=DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            refined_first_move_changed_facility_count=int(selected.changed_facility_count),
            refined_first_move_changed_facility_ids=changed_ids,
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            gradient_norm=0.0,
            scipy_message="V28_STRUCTURED_TELEMETRY_IN_CONTROLLER_DIAGNOSTICS",
            tfv_value_available=True,
            tfv_value_prediction_m3=float(selected.q28_score_m3),
            tfv_value_upper_bound_m3=float(selected.q28_score_m3),
            tfv_value_admission_passed=bool(action),
            engineering_feasible=True,
            passive_channels_unchanged=True,
            sequence_support_valid=True,
            tfv_value_gate_reason="V28_Q28_NEGATIVE" if action else "V28_Q28_NONNEGATIVE_HOLD",
        )


class DirectTFVAuthoritativeV28Controller(PortfolioMemorySafeDirectTFVAuthoritativeController):
    """Portfolio controller that writes V28 structured telemetry into the decision JSONL."""

    def decide(
        self,
        obs: CausalObservation,
        *,
        observation_already_recorded: bool = False,
    ) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        inner = self._direct_mpc_adapter.inner
        telemetry = dict(getattr(inner, "last_v28_telemetry", {}))
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "v28_runtime_contract": V28_OPERATIONAL_RUNTIME_CONTRACT,
                "v28_selection_contract": V28_SELECTION_CONTRACT,
                "v28_q95_mandatory": True,
                "v28_raw_action_executable": False,
                "v28_q27_frozen": True,
                "v28_candidate_telemetry": list(telemetry.get("candidates", [])),
                "v28_raw_candidate_count": int(telemetry.get("raw_candidate_count", 0)),
                "v28_q95_supported_candidate_count": int(telemetry.get("q95_supported_candidate_count", 0)),
                "v28_post_q95_unique_candidate_count": int(telemetry.get("post_q95_unique_candidate_count", 0)),
                "v28_post_q95_duplicate_count": int(telemetry.get("post_q95_duplicate_count", 0)),
                "v28_selected_source": str(telemetry.get("selected_source", "HOLD")),
                "v28_selected_q27_score_m3": float(telemetry.get("selected_q27_score_m3", 0.0)),
                "v28_selected_residual_m3": float(telemetry.get("selected_residual_m3", 0.0)),
                "v28_selected_q28_score_m3": float(telemetry.get("selected_q28_score_m3", 0.0)),
                "v28_action_class": str(telemetry.get("action_class", "HOLD")),
                "v28_auto_rbc_shadow_candidate_only": True,
                "v28_event_id_feature": False,
                "v28_development_only": True,
            }
        )
        source = "MPC_DIRECT_TFV_V28" if telemetry.get("action_class") == "ACTION" else "LATCH_PREVIOUS_TARGET_DIRECT_TFV_V28"
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


def build_operational_v28_controller(
    *,
    graph_path: str | Path,
    sensors_path: str | Path,
    config_path: str | Path,
    step1_path: str | Path,
    step2_path: str | Path,
    supervisory_control_path: str | Path,
    sequence_support_path: str | Path,
    v15_rank_checkpoint_path: str | Path,
    v21_boundary_checkpoint_path: str | Path,
    v27_value_checkpoint_path: str | Path,
    v28_residual_checkpoint_path: str | Path,
    dataset_manifest_path: str | Path,
    v27_dataset_manifest_path: str | Path | None = None,
    asset_manifest_path: str | Path,
    device: torch.device,
    decision_runtime_budget_seconds: float = 180.0,
    proposal_probe_chunk_size: int = 24,
) -> tuple[object, object, tuple[str, ...], dict[str, Any]]:
    v27_dataset_path = Path(v27_dataset_manifest_path or dataset_manifest_path).resolve()
    parent_controller, graph, sensors, parent_lineage = build_operational_v27_controller(
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=supervisory_control_path,
        sequence_support_path=sequence_support_path,
        v15_rank_checkpoint_path=v15_rank_checkpoint_path,
        v21_boundary_checkpoint_path=v21_boundary_checkpoint_path,
        v27_value_checkpoint_path=v27_value_checkpoint_path,
        # V27 checkpoint lineage is bound to the immutable V27 dataset.  V28
        # may use an augmented dataset for residual fitting, so these two
        # identities must remain explicit and separate.
        dataset_manifest_path=v27_dataset_path,
        asset_manifest_path=asset_manifest_path,
        device=device,
        decision_runtime_budget_seconds=decision_runtime_budget_seconds,
        proposal_probe_chunk_size=proposal_probe_chunk_size,
    )
    parent_inner = parent_controller.controller
    old_mpc = parent_inner._direct_mpc_adapter.inner
    q27_sha = _sha(v27_value_checkpoint_path)
    residual, residual_payload = load_v28_residual_value_model(
        str(v28_residual_checkpoint_path),
        expected_q27_checkpoint_sha256=q27_sha,
        expected_dataset_manifest_sha256=_sha(dataset_manifest_path),
    )
    lineage = build_v28_lineage(
        q27_checkpoint_sha256=q27_sha,
        residual_checkpoint_sha256=_sha(v28_residual_checkpoint_path),
        dataset_manifest_sha256=_sha(dataset_manifest_path),
        asset_manifest_sha256=_sha(asset_manifest_path),
    )
    lineage.update(
        {
            "v27_value_checkpoint_sha256": q27_sha,
            "v27_parent_dataset_manifest_sha256": _sha(v27_dataset_path),
            "v27_value_lineage": dict(getattr(old_mpc, "v27_lineage", {})),
            "residual_checkpoint_contract": residual_payload.get("contract"),
            "residual_checkpoint_ridge": float(residual.ridge),
            "residual_feature_width": int(residual.feature_width),
        }
    )
    v28 = DirectTFVOperationalV28MPC(
        model=old_mpc.model,
        graph=old_mpc.graph,
        normalization=old_mpc.normalization,
        action_support=old_mpc.action_support,
        sequence_support=old_mpc.sequence_support,
        supervisory_mask=old_mpc.supervisory_mask,
        policy_return_model=old_mpc.policy_return_model,
        policy_return_normalization=old_mpc.policy_return_normalization,
        rank_adapter=old_mpc.rank_adapter,
        boundary_calibrator=old_mpc.boundary_calibrator,
        continuation_policy_sha256=old_mpc.policy_return_parent_continuation_sha256,
        design=old_mpc.design,
        proposal_probe_chunk_size=old_mpc.proposal_probe_chunk_size,
        v27_value_model=old_mpc.v27_value_model,
        v27_lineage=dict(getattr(old_mpc, "v27_lineage", {})),
        residual_value_model=residual,
        v28_lineage=lineage,
    )
    controller_cfg = parent_inner.config
    inner = DirectTFVAuthoritativeV28Controller(
        step1=parent_inner.step1,
        mpc=v28,
        graph=graph,
        sensor_nodes=sensors,
        forecast=parent_inner.forecast,
        config=controller_cfg,
        device=device,
    )
    controller = ContinuityGuardController(
        inner,
        max_delta_per_update=0.5,
        allow_projection=False,
        enforce_current_delta=False,
    )
    full_lineage = dict(parent_lineage)
    full_lineage.update(lineage)
    return controller, graph, sensors, full_lineage


__all__ = [
    "DirectTFVAuthoritativeV28Controller",
    "DirectTFVOperationalV28MPC",
    "V28Candidate",
    "V28_OPERATIONAL_RUNTIME_CONTRACT",
    "V28_PORTFOLIO_CONTRACT",
    "V28_SELECTION_CONTRACT",
    "V28_TFV_VALUE_ESTIMAND",
    "build_operational_v28_controller",
    "build_v28_lineage",
    "post_q95_deduplicate",
    "select_v28_candidate",
    "validate_v28_executable_candidate",
]
