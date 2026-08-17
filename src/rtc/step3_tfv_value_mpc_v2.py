"""Trust-region Direct-TFV MPC for Project7.

Every one of the frozen 109 writable facilities remains eligible at each 10-minute decision.  Step3
first screens all facilities with small first-move perturbations inside TrainFit support, then
optimises only the most promising dynamic active set.  The continuous search is constrained to the
per-facility action magnitudes observed in authoritative TrainFit counterfactuals.

This prevents the 1308-dimensional L-BFGS-B solver from exploiting unsupported combinations while
preserving the core research claim: learn 109 facility-to-TFV effects and use those learned effects
for receding-horizon control.  The first 10-minute target is the only target executed before the
system is observed again.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
import torch
from scipy.optimize import minimize

from .step2_tfv_support import DIRECT_TFV_ACTION_SUPPORT_CONTRACT
from .step2_tfv_value import DirectFacilityTFVValueModel
from .step2_train_response_v60 import InputNormalizationV60
from .step2_v60_contract import require_feature


DIRECT_TFV_STEP3_CONTRACT = "PROJECT7_DIRECT_TFV_109ACT_SCREENED_TRUST_REGION_MPC_V2"


@dataclass(frozen=True)
class DirectTFVMPCDesignV2:
    model_step_seconds: int = 300
    control_update_seconds: int = 600
    prediction_horizon_steps: int = 72
    free_control_blocks: int = 12
    max_setting_delta_per_update: float = 0.50
    maxiter: int = 30
    ftol: float = 1.0e-7
    gtol: float = 1.0e-5
    deadline_seconds: float = 120.0
    minimum_predicted_improvement_m3: float = 0.0
    active_facility_count: int = 0

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    def validate(self) -> None:
        if (self.model_step_seconds, self.control_update_seconds) != (300, 600):
            raise ValueError("Direct-TFV Step3 requires 300-s model / 600-s control")
        if (self.prediction_horizon_steps, self.free_control_blocks) != (72, 12):
            raise ValueError("Direct-TFV Step3 requires H360 prediction / H120 free control")
        if not 0.0 < float(self.max_setting_delta_per_update) <= 0.5:
            raise ValueError("max setting movement must lie in (0,0.5]")
        if self.maxiter <= 0 or min(self.ftol, self.gtol) <= 0.0:
            raise ValueError("invalid L-BFGS-B configuration")
        if not 0.0 < float(self.deadline_seconds) < float(self.control_update_seconds):
            raise ValueError("Step3 deadline must be inside one control interval")
        if not math.isfinite(float(self.minimum_predicted_improvement_m3)) or float(
            self.minimum_predicted_improvement_m3
        ) < 0.0:
            raise ValueError("minimum predicted improvement must be finite and non-negative")
        if self.active_facility_count < 0 or self.active_facility_count > 109:
            raise ValueError("active_facility_count must lie in [0,109]")


@dataclass(frozen=True)
class DirectTFVMPCResultV2:
    settings: torch.Tensor
    predicted_delta_tfv_m3: float
    selected_source: str
    optimizer_success: bool
    optimizer_steps: int
    optimizer_starts: int
    gradient_norm: float
    scipy_message: str
    elapsed_seconds: float
    screened_facility_count: int
    active_facility_count: int
    active_facility_ids: tuple[str, ...]
    active_facility_screening_scores_m3: tuple[float, ...]
    first_move_changed_facility_count: int
    maximum_support_ratio: float
    minimum_predicted_improvement_m3: float
    training_joint_changed_facility_q50: float
    policy_mode: str = "direct_tfv_screened_trust_region_mpc"
    policy_mode_contract: str = DIRECT_TFV_STEP3_CONTRACT


class DirectTFVTrustRegionMPC:
    """Screen all 109 facilities, then optimise a TrainFit-supported dynamic active set."""

    accepts_previous_requested_settings = True
    policy_mode = "direct_tfv_screened_trust_region_mpc"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: DirectFacilityTFVValueModel,
        graph: Any,
        normalization: InputNormalizationV60,
        action_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV2 = DirectTFVMPCDesignV2(),
    ) -> None:
        design.validate()
        if len(graph.actuator_ids) != 109:
            raise ValueError("Direct-TFV trust-region MPC requires exactly 109 actuators")
        if str(action_support.get("contract")) != DIRECT_TFV_ACTION_SUPPORT_CONTRACT:
            raise ValueError("Step3 requires current Direct-TFV action-support evidence")
        support_ids = tuple(str(value) for value in action_support.get("actuator_ids", ()))
        graph_ids = tuple(str(value) for value in graph.actuator_ids)
        if support_ids != graph_ids:
            raise ValueError("Step3 action-support actuator order differs from graph order")
        if int(action_support.get("single_facility_coverage_count", -1)) != 109:
            raise ValueError("Step3 requires 109/109 exact single-facility TrainFit coverage")
        self.model = model
        self.graph = graph
        self.normalization = normalization
        self.action_support = dict(action_support)
        self.design = design
        names = tuple(graph.actuator_physics_feature_names)
        physics = np.asarray(graph.actuator_physics, dtype=np.float32)
        self.min_setting = physics[:, require_feature(names, "min_setting")].astype(np.float32)
        self.max_setting = physics[:, require_feature(names, "max_setting")].astype(np.float32)
        self.first_radius = np.asarray(
            action_support["first_move_abs_q95_per_facility"], dtype=np.float32
        ).reshape(-1)
        self.sequence_radius = np.asarray(
            action_support["sequence_abs_q95_per_facility"], dtype=np.float32
        ).reshape(-1)
        if self.first_radius.shape != (109,) or self.sequence_radius.shape != (109,):
            raise ValueError("Step3 action-support radii must contain 109 values")
        if np.any(~np.isfinite(self.first_radius)) or np.any(~np.isfinite(self.sequence_radius)):
            raise ValueError("Step3 action-support radii must be finite")
        if np.any(self.first_radius < 0.0) or np.any(self.sequence_radius < 0.0):
            raise ValueError("Step3 action-support radii must be non-negative")

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.normalization.state_mean, dtype=state.dtype, device=state.device)
        std = torch.as_tensor(self.normalization.state_std, dtype=state.dtype, device=state.device).clamp_min(1.0e-6)
        return (state - mean) / std

    def _normalize_rainfall(self, rainfall: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.normalization.rainfall_mean, dtype=rainfall.dtype, device=rainfall.device)
        std = torch.as_tensor(self.normalization.rainfall_std, dtype=rainfall.dtype, device=rainfall.device).clamp_min(1.0e-6)
        return (rainfall - mean) / std

    def _normalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.normalization.flow_mean, dtype=flow.dtype, device=flow.device)
        std = torch.as_tensor(self.normalization.flow_std, dtype=flow.dtype, device=flow.device).clamp_min(1.0e-6)
        return (flow - mean) / std

    def _hold_sequence(self, active_target: torch.Tensor) -> torch.Tensor:
        return active_target.reshape(1, -1).expand(self.design.prediction_horizon_steps, -1)

    def _score_sequences(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        sequences: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        if current_state.shape[0] != 1 or previous_actuator_flow.shape[0] != 1:
            raise ValueError("Direct-TFV Step3 expects one current state/flow vector")
        if rainfall.ndim != 4 or rainfall.shape[0] != 1:
            raise ValueError("current Direct-TFV Step3 requires exactly one causal rainfall forecast scenario")
        if sequences.ndim != 3 or tuple(sequences.shape[1:]) != (
            self.design.prediction_horizon_steps,
            109,
        ):
            raise ValueError("Step3 sequences must be [candidate,H72,109]")
        count = int(sequences.shape[0])
        state = self._normalize_state(current_state).expand(count, -1, -1)
        rain = self._normalize_rainfall(rainfall).expand(count, -1, -1, -1)
        flow = self._normalize_flow(previous_actuator_flow).expand(count, -1)
        reference = self._hold_sequence(active_target)[None].expand(count, -1, -1)
        output = self.model(
            current_state=state,
            rainfall=rain,
            reference_settings=reference,
            candidate_settings=sequences,
            previous_actuator_flow=flow,
            actuator_upstream=torch.as_tensor(
                self.graph.actuator_upstream, dtype=torch.long, device=state.device
            ),
            actuator_downstream=torch.as_tensor(
                self.graph.actuator_downstream, dtype=torch.long, device=state.device
            ),
            actuator_physics=torch.as_tensor(
                self.graph.actuator_physics, dtype=state.dtype, device=state.device
            ),
        )
        return output.total_delta_tfv_m3

    def score_sequence(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        sequence: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        return self._score_sequences(
            current_state=current_state,
            rainfall=rainfall,
            sequences=sequence[None],
            previous_actuator_flow=previous_actuator_flow,
            active_target=active_target,
        )[0]

    def _screen_all_facilities(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        device, dtype = active_target.device, active_target.dtype
        hold = self._hold_sequence(active_target)
        sequences: list[torch.Tensor] = []
        mapping: list[tuple[int, int]] = []
        for index in range(109):
            radius = float(self.first_radius[index])
            if radius <= 1.0e-7:
                continue
            for direction in (-1, 1):
                target = float(active_target[index].detach().cpu()) + direction * radius
                target = max(float(self.min_setting[index]), min(float(self.max_setting[index]), target))
                if abs(target - float(active_target[index].detach().cpu())) <= 1.0e-7:
                    continue
                sequence = hold.clone()
                sequence[: self.design.control_block_steps, index] = torch.as_tensor(
                    target, dtype=dtype, device=device
                )
                sequences.append(sequence)
                mapping.append((index, direction))
        best_score = np.full(109, np.inf, dtype=np.float64)
        best_direction = np.zeros(109, dtype=np.int64)
        if not sequences:
            return np.arange(109, dtype=np.int64), best_score, best_direction
        batch = torch.stack(sequences)
        with torch.no_grad():
            scores = self._score_sequences(
                current_state=current_state,
                rainfall=rainfall,
                sequences=batch,
                previous_actuator_flow=previous_actuator_flow,
                active_target=active_target,
            ).detach().cpu().numpy()
        for score, (index, direction) in zip(scores.tolist(), mapping, strict=True):
            if float(score) < best_score[index]:
                best_score[index] = float(score)
                best_direction[index] = int(direction)
        eligible = np.flatnonzero(np.isfinite(best_score) & (self.first_radius > 1.0e-7))
        order = eligible[np.argsort(best_score[eligible], kind="mergesort")]
        return order.astype(np.int64), best_score, best_direction

    def _active_count(self, eligible_count: int) -> int:
        if eligible_count <= 0:
            return 0
        if self.design.active_facility_count > 0:
            requested = int(self.design.active_facility_count)
        else:
            requested = int(
                max(1, math.ceil(float(self.action_support["joint_changed_facility_count_q50"])))
            )
        return int(min(eligible_count, max(1, min(109, requested))))

    def _decode_active_fractions(
        self,
        fractions: torch.Tensor,
        *,
        active_indices: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        if fractions.ndim != 2 or fractions.shape != (
            self.design.free_control_blocks,
            int(active_indices.numel()),
        ):
            raise ValueError("active fractions must be [12,K]")
        device, dtype = active_target.device, active_target.dtype
        lo = torch.as_tensor(self.min_setting, dtype=dtype, device=device)
        hi = torch.as_tensor(self.max_setting, dtype=dtype, device=device)
        first_radius = torch.as_tensor(self.first_radius, dtype=dtype, device=device)
        sequence_radius = torch.as_tensor(self.sequence_radius, dtype=dtype, device=device)
        previous = active_target
        blocks: list[torch.Tensor] = []
        for block in range(self.design.free_control_blocks):
            radius = first_radius if block == 0 else sequence_radius
            lower = torch.maximum(lo, previous - float(self.design.max_setting_delta_per_update))
            upper = torch.minimum(hi, previous + float(self.design.max_setting_delta_per_update))
            lower = torch.maximum(lower, active_target - radius)
            upper = torch.minimum(upper, active_target + radius)
            target = active_target.clone()
            ai = active_indices.long()
            width = (upper.index_select(0, ai) - lower.index_select(0, ai)).clamp_min(0.0)
            target_active = lower.index_select(0, ai) + fractions[block] * width
            target = target.scatter(0, ai, target_active)
            blocks.append(target)
            previous = target
        free_blocks = torch.stack(blocks)
        total_blocks = self.design.prediction_horizon_steps // self.design.control_block_steps
        if total_blocks > self.design.free_control_blocks:
            free_blocks = torch.cat(
                (
                    free_blocks,
                    free_blocks[-1:].expand(total_blocks - self.design.free_control_blocks, -1),
                ),
                dim=0,
            )
        return free_blocks.repeat_interleave(self.design.control_block_steps, dim=0)

    def _hold_start(self, *, active_indices: torch.Tensor, active_target: torch.Tensor) -> torch.Tensor:
        device, dtype = active_target.device, active_target.dtype
        lo = torch.as_tensor(self.min_setting, dtype=dtype, device=device)
        hi = torch.as_tensor(self.max_setting, dtype=dtype, device=device)
        first_radius = torch.as_tensor(self.first_radius, dtype=dtype, device=device)
        sequence_radius = torch.as_tensor(self.sequence_radius, dtype=dtype, device=device)
        previous = active_target
        rows: list[torch.Tensor] = []
        ai = active_indices.long()
        for block in range(self.design.free_control_blocks):
            radius = first_radius if block == 0 else sequence_radius
            lower = torch.maximum(lo, previous - float(self.design.max_setting_delta_per_update))
            upper = torch.minimum(hi, previous + float(self.design.max_setting_delta_per_update))
            lower = torch.maximum(lower, active_target - radius)
            upper = torch.minimum(upper, active_target + radius)
            low_a = lower.index_select(0, ai)
            width = (upper.index_select(0, ai) - low_a).clamp_min(1.0e-12)
            fraction = ((active_target.index_select(0, ai) - low_a) / width).clamp(0.0, 1.0)
            rows.append(fraction)
            previous = active_target
        return torch.stack(rows)

    def _direction_start(
        self,
        *,
        hold_start: torch.Tensor,
        active_indices_np: np.ndarray,
        best_direction: np.ndarray,
    ) -> torch.Tensor:
        seed = hold_start.clone()
        for column, actuator_index in enumerate(active_indices_np.tolist()):
            direction = int(best_direction[actuator_index])
            if direction > 0:
                seed[:, column] = 1.0
            elif direction < 0:
                seed[:, column] = 0.0
        return seed

    def _support_ratio(self, sequence: torch.Tensor, active_target: torch.Tensor) -> float:
        blocks = sequence[:: self.design.control_block_steps]
        delta = torch.abs(blocks - active_target[None])
        first = torch.as_tensor(self.first_radius, dtype=delta.dtype, device=delta.device).clamp_min(1.0e-12)
        later = torch.as_tensor(self.sequence_radius, dtype=delta.dtype, device=delta.device).clamp_min(1.0e-12)
        ratio_first = delta[0] / first
        ratio_later = delta[1:] / later[None] if blocks.shape[0] > 1 else ratio_first[None]
        return float(torch.maximum(ratio_first.max(), ratio_later.max()).detach().cpu())

    def optimize(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        current_settings: torch.Tensor,
        active_target: torch.Tensor,
    ) -> DirectTFVMPCResultV2:
        del current_settings  # kept in the public controller interface for closed-loop compatibility.
        started = time.perf_counter()
        device, dtype = current_state.device, current_state.dtype
        hold = self._hold_sequence(active_target)
        order, screening_scores, best_direction = self._screen_all_facilities(
            current_state=current_state,
            rainfall=rainfall,
            previous_actuator_flow=previous_actuator_flow,
            active_target=active_target,
        )
        active_count = self._active_count(len(order))
        if active_count <= 0:
            return DirectTFVMPCResultV2(
                settings=hold,
                predicted_delta_tfv_m3=0.0,
                selected_source="HOLD_NO_SUPPORTED_FACILITY",
                optimizer_success=True,
                optimizer_steps=0,
                optimizer_starts=0,
                gradient_norm=0.0,
                scipy_message="no facility has non-zero TrainFit first-move support",
                elapsed_seconds=float(time.perf_counter() - started),
                screened_facility_count=109,
                active_facility_count=0,
                active_facility_ids=(),
                active_facility_screening_scores_m3=(),
                first_move_changed_facility_count=0,
                maximum_support_ratio=0.0,
                minimum_predicted_improvement_m3=float(self.design.minimum_predicted_improvement_m3),
                training_joint_changed_facility_q50=float(self.action_support["joint_changed_facility_count_q50"]),
            )

        active_indices_np = order[:active_count]
        active_indices = torch.as_tensor(active_indices_np, dtype=torch.long, device=device)
        hold_start = self._hold_start(active_indices=active_indices, active_target=active_target)
        starts = [
            hold_start,
            self._direction_start(
                hold_start=hold_start,
                active_indices_np=active_indices_np,
                best_direction=best_direction,
            ),
        ]
        best_result = None
        best_sequence = hold
        best_score = 0.0
        last_gradient_norm = 0.0
        messages: list[str] = []
        total_steps = 0

        for start in starts:
            if time.perf_counter() - started >= float(self.design.deadline_seconds):
                messages.append("deadline reached before next start")
                break

            def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
                nonlocal last_gradient_norm
                fractions = torch.as_tensor(flat, dtype=dtype, device=device).reshape(
                    self.design.free_control_blocks, active_count
                ).requires_grad_(True)
                sequence = self._decode_active_fractions(
                    fractions,
                    active_indices=active_indices,
                    active_target=active_target,
                )
                score = self.score_sequence(
                    current_state=current_state,
                    rainfall=rainfall,
                    sequence=sequence,
                    previous_actuator_flow=previous_actuator_flow,
                    active_target=active_target,
                )
                gradient = torch.autograd.grad(score, fractions, allow_unused=False)[0]
                if not bool(torch.isfinite(score)) or not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("Direct-TFV Step3 produced non-finite objective/gradient")
                last_gradient_norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
                return float(score.detach().cpu()), gradient.detach().cpu().numpy().reshape(-1).astype(np.float64)

            result = minimize(
                objective,
                start.detach().cpu().numpy().reshape(-1).astype(np.float64),
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * int(start.numel()),
                options={
                    "maxiter": int(self.design.maxiter),
                    "ftol": float(self.design.ftol),
                    "gtol": float(self.design.gtol),
                },
            )
            total_steps += int(getattr(result, "nit", 0))
            messages.append(str(result.message))
            candidate_fraction = torch.as_tensor(
                np.asarray(result.x, dtype=np.float32), dtype=dtype, device=device
            ).reshape(self.design.free_control_blocks, active_count)
            candidate = self._decode_active_fractions(
                candidate_fraction,
                active_indices=active_indices,
                active_target=active_target,
            )
            with torch.no_grad():
                score = float(
                    self.score_sequence(
                        current_state=current_state,
                        rainfall=rainfall,
                        sequence=candidate,
                        previous_actuator_flow=previous_actuator_flow,
                        active_target=active_target,
                    ).detach().cpu()
                )
            if best_result is None or score < best_score:
                best_result = result
                best_score = score
                best_sequence = candidate

        threshold = float(self.design.minimum_predicted_improvement_m3)
        if best_result is None or not math.isfinite(best_score) or best_score >= -threshold:
            selected = hold
            selected_score = 0.0
            source = "HOLD_NO_CONFIDENT_TFV_IMPROVEMENT"
        else:
            selected = best_sequence
            selected_score = float(best_score)
            source = "DIRECT_TFV_TRUST_REGION_LBFGSB"

        first_move = selected[: self.design.control_block_steps].mean(dim=0)
        changed = int(torch.sum(torch.abs(first_move - active_target) > 1.0e-7).detach().cpu())
        support_ratio = 0.0 if source.startswith("HOLD") else self._support_ratio(selected, active_target)
        active_ids = tuple(str(self.graph.actuator_ids[index]) for index in active_indices_np.tolist())
        active_scores = tuple(float(screening_scores[index]) for index in active_indices_np.tolist())
        success = bool(best_result.success) if best_result is not None else True
        return DirectTFVMPCResultV2(
            settings=selected,
            predicted_delta_tfv_m3=float(selected_score),
            selected_source=source,
            optimizer_success=success,
            optimizer_steps=int(total_steps),
            optimizer_starts=int(len(starts)),
            gradient_norm=float(last_gradient_norm),
            scipy_message=" | ".join(messages),
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=109,
            active_facility_count=int(active_count),
            active_facility_ids=active_ids,
            active_facility_screening_scores_m3=active_scores,
            first_move_changed_facility_count=int(changed),
            maximum_support_ratio=float(support_ratio),
            minimum_predicted_improvement_m3=threshold,
            training_joint_changed_facility_q50=float(self.action_support["joint_changed_facility_count_q50"]),
        )


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "DirectTFVMPCDesignV2",
    "DirectTFVMPCResultV2",
    "DirectTFVTrustRegionMPC",
]
