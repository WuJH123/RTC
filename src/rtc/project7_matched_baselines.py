"""Development-only information/action-authority matched Project7 baselines.

The matched baselines are routed through the same outer Project7 controller as Proposed: frozen
sparse sensors, causal Step1 reconstruction, causal rainfall forecast, 109-channel actuator order and
target-latch semantics. Baseline rules therefore never receive extra SWMM node observations.

Raw rule targets are projected through the same execution authority as Proposed: native 82-channel
supervisory mask, first-move support radius, q95 changed-facility ceiling, 0.5 target slew and q95
joint-sequence support. Native Internal SWMM rules are deliberately not relabelled as matched because
they read simulator-internal states; they remain an external operational reference until a validated
reconstructed-state rule interpreter exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

import numpy as np
import torch

from .actuator_release_semantics import release_fraction_to_setting
from .direct_tfv_operational_v23_runtime import (
    DirectTFVOperationalV23MPC,
    build_operational_v23_controller,
)
from .direct_tfv_policy_return_portfolio import _bounded_supported_target, _node_feature
from .direct_tfv_sequence_support import changed_facility_support_limit
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12

MATCHED_BASELINE_CONTRACT = "PROJECT7_SPARSE_STEP1_ACTION_AUTHORITY_MATCHED_BASELINES_V1"
MATCHED_AUTO_RBC = "matched_auto_rbc"
MATCHED_EFD = "matched_efd"
MATCHED_ACTIVE_BASELINES = (MATCHED_AUTO_RBC, MATCHED_EFD)


@dataclass(frozen=True)
class MatchedRuleDiagnostics:
    strategy: str
    raw_changed_facility_count: int
    projected_changed_facility_count: int
    raw_first_move_l1: float
    projected_first_move_l1: float
    retained_l1_fraction: float
    mean_upstream_fill: float = 0.0
    max_downstream_fill: float = 0.0
    mean_storage_fill: float = 0.0
    storage_fill_std: float = 0.0


def _state2(current_state: torch.Tensor) -> torch.Tensor:
    state = current_state
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state = state[0]
    if state.ndim != 2:
        raise ValueError("matched baseline requires reconstructed current_state [node,state]")
    return state


def _actuator_kinds(graph: Any) -> tuple[str, ...]:
    names = tuple(str(value) for value in getattr(graph, "actuator_physics_feature_names", ()))
    physics = np.asarray(getattr(graph, "actuator_physics", ()), dtype=np.float64)
    ids = tuple(str(value) for value in getattr(graph, "actuator_ids", ()))
    columns = {
        kind: names.index(f"is_{kind}")
        for kind in ("pump", "orifice", "weir", "outlet")
        if f"is_{kind}" in names
    }
    if physics.ndim != 2 or physics.shape[0] != len(ids) or len(columns) != 4:
        raise ValueError("matched baseline requires complete actuator physics/type indicators")
    out: list[str] = []
    for row in range(len(ids)):
        kinds = [kind for kind, column in columns.items() if physics[row, column] > 0.5]
        if len(kinds) != 1:
            raise ValueError(f"actuator {ids[row]!r} has ambiguous type")
        out.append(kinds[0])
    return tuple(out)


def _raw_auto_rbc_target(
    *,
    graph: Any,
    current_state: torch.Tensor,
    active_target: torch.Tensor,
    low_fill: float = 0.25,
    high_fill: float = 0.75,
    downstream_congestion_fill: float = 0.90,
    response: float = 0.60,
) -> tuple[torch.Tensor, float, float]:
    state = _state2(current_state)
    dtype, device = active_target.dtype, active_target.device
    max_depth = torch.as_tensor(
        np.maximum(_node_feature(graph, "max_depth_m", 1.0), 1.0e-6),
        dtype=dtype,
        device=device,
    )
    upstream = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    downstream = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    depth = torch.clamp(state[:, 0].to(dtype=dtype, device=device), min=0.0)
    fill = torch.clamp(depth / max_depth, 0.0, 1.5)
    up, down = fill[upstream], fill[downstream]
    drive = torch.clamp((up - low_fill) / (high_fill - low_fill), 0.0, 1.0)
    penalty = torch.clamp(
        (down - downstream_congestion_fill) / (1.0 - downstream_congestion_fill),
        0.0,
        1.0,
    )
    release = drive * (1.0 - penalty)
    desired = torch.as_tensor(
        [
            release_fraction_to_setting(kind, float(value))
            for kind, value in zip(
                _actuator_kinds(graph), release.detach().cpu(), strict=True
            )
        ],
        dtype=dtype,
        device=device,
    )
    raw = active_target + float(response) * (desired - active_target)
    return raw, float(up.mean().detach().cpu()), float(down.max().detach().cpu())


def _raw_efd_target(
    *,
    graph: Any,
    current_state: torch.Tensor,
    active_target: torch.Tensor,
    equalization_gain: float = 1.0,
    response: float = 0.60,
) -> tuple[torch.Tensor, float, float]:
    state = _state2(current_state)
    dtype, device = active_target.dtype, active_target.device
    capacity_np = np.asarray(_node_feature(graph, "storage_capacity_m3", 0.0), dtype=np.float64)
    capacity = torch.as_tensor(np.maximum(capacity_np, 1.0e-9), dtype=dtype, device=device)
    volume = torch.clamp(state[:, 3].to(dtype=dtype, device=device), min=0.0)
    fill = torch.clamp(volume / capacity, 0.0, 1.5)
    upstream_np = np.asarray(graph.actuator_upstream, dtype=np.int64)
    storage_nodes = sorted(
        {int(node) for node in upstream_np if capacity_np[int(node)] > 1.0e-9}
    )
    if not storage_nodes:
        return active_target.clone(), 0.0, 0.0
    storage_fill = fill[torch.as_tensor(storage_nodes, dtype=torch.long, device=device)]
    mean_fill = float(storage_fill.mean().detach().cpu())
    std_fill = float(storage_fill.std(unbiased=False).detach().cpu())
    raw = active_target.clone()
    kinds = _actuator_kinds(graph)
    for node in storage_nodes:
        filling = float(fill[node].detach().cpu())
        release = float(
            np.clip(filling + equalization_gain * (filling - mean_fill), 0.0, 1.0)
        )
        for index in np.flatnonzero(upstream_np == node):
            desired = release_fraction_to_setting(kinds[int(index)], release)
            raw[int(index)] = active_target[int(index)] + float(response) * (
                float(desired) - active_target[int(index)]
            )
    return raw, mean_fill, std_fill


class MatchedInformationBaselineMPC(DirectTFVOperationalV23MPC):
    policy_mode = "matched_information_baseline"
    policy_mode_contract = MATCHED_BASELINE_CONTRACT

    def __init__(
        self,
        *,
        matched_strategy: Literal["matched_auto_rbc", "matched_efd"],
        **kwargs: Any,
    ) -> None:
        if matched_strategy not in MATCHED_ACTIVE_BASELINES:
            raise ValueError(f"unsupported matched baseline: {matched_strategy}")
        super().__init__(**kwargs)
        self.matched_strategy = matched_strategy

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        current_state = kwargs.get("current_state")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("matched baseline requires active_target [109]")
        if not isinstance(current_state, torch.Tensor):
            raise ValueError("matched baseline requires reconstructed current_state")

        if self.matched_strategy == MATCHED_AUTO_RBC:
            raw, mean_up, max_down = _raw_auto_rbc_target(
                graph=self.graph,
                current_state=current_state,
                active_target=active_target,
            )
            mean_storage = storage_std = 0.0
        else:
            raw, mean_storage, storage_std = _raw_efd_target(
                graph=self.graph,
                current_state=current_state,
                active_target=active_target,
            )
            mean_up = max_down = 0.0

        ceiling = int(changed_facility_support_limit(self.sequence_support, "q95"))
        raw_delta = (raw - active_target).detach().cpu().numpy()
        raw_changed = int(np.count_nonzero(np.abs(raw_delta) > 1.0e-7))
        first_radius = np.asarray(
            torch.as_tensor(self.first_radius).detach().cpu(), dtype=np.float64
        ).reshape(-1)
        first_projected = _bounded_supported_target(
            active_target=active_target.detach().cpu().numpy(),
            raw_delta=raw_delta,
            graph=self.graph,
            first_radius=first_radius,
            max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
            supervisory_mask=self.supervisory_mask,
        )
        projected = torch.as_tensor(
            first_projected, dtype=active_target.dtype, device=active_target.device
        )
        target, sequence, changed, support = self._h10_supported_target(
            projected, active_target
        )
        raw_l1 = float(torch.sum(torch.abs(raw - active_target)).detach().cpu())
        projected_l1 = float(torch.sum(torch.abs(target - active_target)).detach().cpu())
        retained = 1.0 if raw_l1 <= 1.0e-12 else projected_l1 / raw_l1
        rule_diag = MatchedRuleDiagnostics(
            strategy=self.matched_strategy,
            raw_changed_facility_count=raw_changed,
            projected_changed_facility_count=int(changed),
            raw_first_move_l1=raw_l1,
            projected_first_move_l1=projected_l1,
            retained_l1_fraction=float(retained),
            mean_upstream_fill=mean_up,
            max_downstream_fill=max_down,
            mean_storage_fill=mean_storage,
            storage_fill_std=storage_std,
        )

        hold = active_target.reshape(1, 109).expand(
            int(self.design.prediction_horizon_steps), -1
        ).detach()
        action = int(changed) > 0
        executed = sequence if action else hold
        changed_indices = torch.nonzero(
            torch.abs(target - active_target) > 1.0e-7
        ).reshape(-1).tolist()
        changed_ids = tuple(
            str(self.graph.actuator_ids[int(index)]) for index in changed_indices
        )
        source = (
            self.matched_strategy.upper()
            if action
            else f"HOLD::{self.matched_strategy.upper()}"
        )
        diagnostics_text = f"MATCHED_RULE|{rule_diag!r}"
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=0.0,
            raw_optimized_predicted_delta_tfv_m3=0.0,
            selected_source=source,
            candidate_valid=action,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=0.0,
            admission_margin_kind="not_applicable_matched_rule_baseline",
            admission_passed=action,
            calibrated_admission_contract=MATCHED_BASELINE_CONTRACT,
            elapsed_seconds=float(time.perf_counter() - started),
            screened_facility_count=int(np.sum(self.supervisory_mask)),
            predicted_beneficial_facility_count=0,
            active_facility_count=int(changed),
            active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=(),
            first_move_changed_facility_count=int(changed) if action else 0,
            maximum_support_ratio=self._first_move_support_ratio(target, active_target),
            joint_sequence_support_quantile=str(support["quantile"]),
            joint_sequence_first_block_l1=float(support["first_block_l1"]),
            joint_sequence_h120_l1=float(support["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(
                support["h120_total_variation_l1"]
            ),
            joint_sequence_support_max_ratio=float(support["max_ratio"]),
            joint_sequence_support_binding=bool(support["binding"]),
            policy_return_predicted_delta_tfv_m3=0.0,
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=0.0,
            policy_return_admission_passed=action,
            policy_return_admission_contract=MATCHED_BASELINE_CONTRACT,
            policy_return_estimand="NOT_USED_MATCHED_RULE_BASELINE",
            policy_return_parent_continuation_sha256=(
                self.policy_return_parent_continuation_sha256
            ),
            policy_return_portfolio_contract=MATCHED_BASELINE_CONTRACT,
            policy_return_portfolio_candidate_count=1,
            policy_return_portfolio_selected_source=(
                self.matched_strategy if action else "HOLD"
            ),
            policy_return_portfolio_sources=(self.matched_strategy,),
            policy_return_portfolio_scores_m3=(0.0,),
            policy_return_portfolio_upper_bounds_m3=(0.0,),
            policy_return_portfolio_base_step2_scores_m3=(0.0,),
            h10_probe_generator_contract=MATCHED_BASELINE_CONTRACT,
            h10_probe_count=0,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics="MATCHED_RULE_H10_TARGET_LATCH",
            refined_first_move_changed_facility_count=int(changed),
            refined_first_move_changed_facility_ids=changed_ids,
            optimizer_success=True,
            optimizer_steps=0,
            optimizer_starts=0,
            gradient_norm=0.0,
            scipy_message=diagnostics_text,
            tfv_value_available=False,
            engineering_feasible=True,
            passive_channels_unchanged=True,
            sequence_support_valid=True,
            tfv_value_gate_reason=diagnostics_text,
        )


def build_matched_information_baseline_controller(*, matched_strategy: str, **kwargs: Any):
    """Reuse Proposed sparse-Step1 outer runtime and replace only the decision rule."""
    controller, graph, sensors, parent_lineage = build_operational_v23_controller(**kwargs)
    old = controller.controller._direct_mpc_adapter.inner
    matched = MatchedInformationBaselineMPC(
        matched_strategy=matched_strategy,
        model=old.model,
        graph=old.graph,
        normalization=old.normalization,
        action_support=old.action_support,
        sequence_support=old.sequence_support,
        supervisory_mask=old.supervisory_mask,
        policy_return_model=old.policy_return_model,
        policy_return_normalization=old.policy_return_normalization,
        rank_adapter=old.rank_adapter,
        boundary_calibrator=old.boundary_calibrator,
        continuation_policy_sha256=old.policy_return_parent_continuation_sha256,
        design=old.design,
        proposal_probe_chunk_size=old.proposal_probe_chunk_size,
    )
    controller.controller._direct_mpc_adapter.inner = matched
    lineage = dict(parent_lineage)
    lineage.update(
        {
            "contract": MATCHED_BASELINE_CONTRACT,
            "matched_strategy": matched_strategy,
            "same_sparse_sensor_set_as_proposed": True,
            "same_frozen_step1_reconstruction_as_proposed": True,
            "same_rainfall_forecast_as_proposed": True,
            "same_109_channel_representation": True,
            "same_82_channel_supervisory_mask": True,
            "same_passive_27_channels": True,
            "same_q95_changed_facility_ceiling": True,
            "same_q95_joint_sequence_support": True,
            "same_max_setting_delta_per_update": 0.5,
            "same_target_latch_semantics": True,
            "native_internal_rtc_is_not_a_matched_comparator": True,
            "development_only": True,
            "formal_evidence": False,
            "ready_for_policy_lock": False,
        }
    )
    return controller, graph, sensors, lineage


__all__ = [
    "MATCHED_ACTIVE_BASELINES",
    "MATCHED_AUTO_RBC",
    "MATCHED_BASELINE_CONTRACT",
    "MATCHED_EFD",
    "MatchedInformationBaselineMPC",
    "MatchedRuleDiagnostics",
    "build_matched_information_baseline_controller",
]
