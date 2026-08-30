"""Project7 information-matched baseline V2.

This module preserves the sparse-sensor/Step1/action-authority matching introduced by V1 and repairs
one scientific defect: EFD no longer assumes that every controlled storage is the direct upstream
node of a supervisory actuator.  Instead, static INP topology is used to find the first supervisory
outlet(s) downstream of each storage through passive links.  Invalid/degenerate EFD topology fails
closed rather than silently collapsing to HOLD/No-control.

The locked Q27 baseline implementation is intentionally left unchanged for provenance.  V2 is a new
Development-only comparator contract and requires a new Policy Lock before any publication use.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .actuator_release_semantics import release_fraction_to_setting
from .direct_tfv_operational_v23_runtime import build_operational_v23_controller
from .direct_tfv_policy_return_portfolio import _bounded_supported_target, _node_feature
from .direct_tfv_sequence_support import changed_facility_support_limit
from .project7_efd_topology import EFD_TOPOLOGY_CONTRACT, EFDTopologyMap, build_efd_topology_map
from .project7_matched_baselines import (
    MATCHED_ACTIVE_BASELINES,
    MATCHED_AUTO_RBC,
    MATCHED_EFD,
    MATCHED_INTERNAL_RTC,
    MatchedInformationBaselineMPC,
    MatchedRuleDiagnostics,
)
from .project7_matched_internal import MATCHED_INTERNAL_RULE_CONTRACT
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12


MATCHED_BASELINE_V2_CONTRACT = "PROJECT7_SPARSE_STEP1_TOPOLOGY_VALID_ACTION_AUTHORITY_MATCHED_BASELINES_V2"
MATCHED_EFD_V2_CONTRACT = "PROJECT7_TOPOLOGY_AWARE_STORAGE_EFD_MATCHED_V2"


def _state2(current_state: torch.Tensor) -> torch.Tensor:
    state = current_state
    if state.ndim == 3 and int(state.shape[0]) == 1:
        state = state[0]
    if state.ndim != 2:
        raise ValueError("matched baseline V2 requires reconstructed current_state [node,state]")
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
        raise ValueError("matched baseline V2 requires complete actuator type indicators")
    result: list[str] = []
    for row, actuator_id in enumerate(ids):
        kinds = [kind for kind, column in columns.items() if physics[row, column] > 0.5]
        if len(kinds) != 1:
            raise ValueError(f"actuator {actuator_id!r} has ambiguous type")
        result.append(kinds[0])
    return tuple(result)


class TopologyMatchedInformationBaselineMPC(MatchedInformationBaselineMPC):
    """V1 matched baselines with a topology-valid EFD branch."""

    policy_mode = "matched_information_baseline_v2"
    policy_mode_contract = MATCHED_BASELINE_V2_CONTRACT

    def __init__(
        self,
        *,
        matched_strategy: str,
        source_inp_path: str | None = None,
        native_controls_inp_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            matched_strategy=matched_strategy,
            native_controls_inp_path=native_controls_inp_path,
            **kwargs,
        )
        self.source_inp_path = source_inp_path
        self.efd_topology: EFDTopologyMap | None = None
        self.efd_storage_node_indices: dict[str, int] = {}
        self.efd_actuator_indices: dict[str, int] = {}
        if matched_strategy == MATCHED_EFD:
            if not source_inp_path:
                raise ValueError("matched EFD V2 requires source_inp_path")
            self.efd_topology = build_efd_topology_map(
                source_inp_path,
                actuator_ids=tuple(str(value) for value in self.graph.actuator_ids),
                supervisory_mask=np.asarray(self.supervisory_mask, dtype=bool),
            )
            node_index = {str(node_id): i for i, node_id in enumerate(self.graph.node_ids)}
            actuator_index = {
                str(actuator_id): i for i, actuator_id in enumerate(self.graph.actuator_ids)
            }
            missing_nodes = [
                storage_id
                for storage_id in self.efd_topology.mapped_storage_ids
                if storage_id not in node_index
            ]
            if missing_nodes:
                raise ValueError(
                    "EFD topology storages are absent from graph schema: " + ",".join(missing_nodes[:8])
                )
            capacity = np.asarray(_node_feature(self.graph, "storage_capacity_m3", 0.0), dtype=np.float64)
            invalid_capacity = [
                storage_id
                for storage_id in self.efd_topology.mapped_storage_ids
                if capacity[node_index[storage_id]] <= 1.0e-9
            ]
            if invalid_capacity:
                raise ValueError(
                    "topology-mapped EFD storages require positive graph storage_capacity_m3: "
                    + ",".join(invalid_capacity[:8])
                )
            self.efd_storage_node_indices = {
                storage_id: node_index[storage_id]
                for storage_id in self.efd_topology.mapped_storage_ids
            }
            self.efd_actuator_indices = {
                actuator_id: actuator_index[actuator_id]
                for actuator_id in self.efd_topology.mapped_actuator_ids
            }

    def _raw_topology_efd_target(
        self,
        *,
        current_state: torch.Tensor,
        active_target: torch.Tensor,
        equalization_gain: float = 1.0,
        response: float = 0.60,
    ) -> tuple[torch.Tensor, float, float]:
        topology = self.efd_topology
        if topology is None:
            raise RuntimeError("EFD V2 topology was not initialized")
        state = _state2(current_state)
        dtype, device = active_target.dtype, active_target.device
        capacity_np = np.asarray(_node_feature(self.graph, "storage_capacity_m3", 0.0), dtype=np.float64)
        capacity = torch.as_tensor(capacity_np, dtype=dtype, device=device)
        volume = torch.clamp(state[:, 3].to(dtype=dtype, device=device), min=0.0)

        fill: dict[str, float] = {}
        for storage_id, node_index in self.efd_storage_node_indices.items():
            cap = float(capacity_np[node_index])
            if cap <= 1.0e-9:
                raise RuntimeError("EFD V2 encountered a non-positive mapped storage capacity")
            fill[storage_id] = float(
                torch.clamp(volume[node_index] / capacity[node_index], 0.0, 1.5).detach().cpu()
            )
        values = np.asarray([fill[sid] for sid in topology.mapped_storage_ids], dtype=np.float64)
        if values.size < 2 or not np.isfinite(values).all():
            raise RuntimeError("EFD V2 requires >=2 finite mapped storage filling degrees")
        mean_fill = float(values.mean())
        std_fill = float(values.std())

        raw = active_target.clone()
        kinds = _actuator_kinds(self.graph)
        passive = np.asarray(~np.asarray(self.supervisory_mask, dtype=bool), dtype=bool)
        for storage_id in topology.mapped_storage_ids:
            filling = fill[storage_id]
            # Fixed, pre-specified EFD-like equalization law: fuller-than-average storages are
            # biased toward greater release, under-filled storages toward retention.  No Final
            # outcome or event-specific parameter enters this rule.
            release = float(
                np.clip(
                    filling + float(equalization_gain) * (filling - mean_fill),
                    0.0,
                    1.0,
                )
            )
            for actuator_id in topology.storage_to_actuators[storage_id]:
                actuator_index = self.efd_actuator_indices[actuator_id]
                desired = release_fraction_to_setting(kinds[actuator_index], release)
                raw[actuator_index] = active_target[actuator_index] + float(response) * (
                    float(desired) - active_target[actuator_index]
                )
        if bool(torch.any(torch.abs(raw[torch.as_tensor(passive, device=device)] - active_target[torch.as_tensor(passive, device=device)]) > 1.0e-7)):
            raise RuntimeError("EFD V2 raw rule changed a passive actuator")
        return raw, mean_fill, std_fill

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        if self.matched_strategy != MATCHED_EFD:
            return super().optimize(**kwargs)

        started = time.perf_counter()
        active_target = kwargs.get("active_target")
        current_state = kwargs.get("current_state")
        if not isinstance(active_target, torch.Tensor) or tuple(active_target.shape) != (109,):
            raise ValueError("matched EFD V2 requires active_target [109]")
        if not isinstance(current_state, torch.Tensor):
            raise ValueError("matched EFD V2 requires reconstructed current_state")
        raw, mean_storage, storage_std = self._raw_topology_efd_target(
            current_state=current_state,
            active_target=active_target,
        )

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
        target, sequence, changed, support = self._h10_supported_target(projected, active_target)
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
        changed_ids = tuple(str(self.graph.actuator_ids[int(index)]) for index in changed_indices)
        topology = self.efd_topology
        assert topology is not None
        diagnostics_payload = {
            "rule": repr(rule_diag),
            "efd_contract": MATCHED_EFD_V2_CONTRACT,
            "topology_contract": EFD_TOPOLOGY_CONTRACT,
            "source_inp_path": str(Path(str(self.source_inp_path)).resolve()),
            "all_storage_count": len(topology.all_storage_ids),
            "mapped_storage_count": len(topology.mapped_storage_ids),
            "mapped_actuator_count": len(topology.mapped_actuator_ids),
            "unmapped_storage_count": len(topology.unmapped_storage_ids),
            "ambiguous_actuator_count": len(topology.ambiguous_actuator_ids),
            "mapped_storage_ids": list(topology.mapped_storage_ids),
            "mapped_actuator_ids": list(topology.mapped_actuator_ids),
        }
        diagnostics_text = "MATCHED_EFD_V2|" + json.dumps(
            diagnostics_payload, sort_keys=True, separators=(",", ":")
        )
        return DirectTFVMPCResultV12(
            settings=executed,
            optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=0.0,
            raw_optimized_predicted_delta_tfv_m3=0.0,
            selected_source=("MATCHED_EFD_V2" if action else "HOLD::MATCHED_EFD_V2"),
            candidate_valid=action,
            admission_margin_m3=0.0,
            admission_upper_bound_m3=0.0,
            admission_margin_kind="not_applicable_matched_rule_baseline_v2",
            admission_passed=action,
            calibrated_admission_contract=MATCHED_BASELINE_V2_CONTRACT,
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
            joint_sequence_h120_total_variation_l1=float(support["h120_total_variation_l1"]),
            joint_sequence_support_max_ratio=float(support["max_ratio"]),
            joint_sequence_support_binding=bool(support["binding"]),
            policy_return_predicted_delta_tfv_m3=0.0,
            policy_return_margin_m3=0.0,
            policy_return_upper_bound_m3=0.0,
            policy_return_admission_passed=action,
            policy_return_admission_contract=MATCHED_BASELINE_V2_CONTRACT,
            policy_return_estimand="NOT_USED_MATCHED_RULE_BASELINE_V2",
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=MATCHED_BASELINE_V2_CONTRACT,
            policy_return_portfolio_candidate_count=1,
            policy_return_portfolio_selected_source=(MATCHED_EFD if action else "HOLD"),
            policy_return_portfolio_sources=(MATCHED_EFD,),
            policy_return_portfolio_scores_m3=(0.0,),
            policy_return_portfolio_upper_bounds_m3=(0.0,),
            policy_return_portfolio_base_step2_scores_m3=(0.0,),
            h10_probe_generator_contract=MATCHED_BASELINE_V2_CONTRACT,
            h10_probe_count=0,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics="MATCHED_RULE_H10_TARGET_LATCH_V2",
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


def build_matched_information_baseline_v2_controller(
    *,
    matched_strategy: str,
    source_inp_path: str,
    native_controls_inp_path: str | None = None,
    **kwargs: Any,
):
    if matched_strategy not in MATCHED_ACTIVE_BASELINES:
        raise ValueError(f"unsupported matched baseline V2: {matched_strategy}")
    controller, graph, sensors, parent_lineage = build_operational_v23_controller(**kwargs)
    old = controller.controller._direct_mpc_adapter.inner
    matched = TopologyMatchedInformationBaselineMPC(
        matched_strategy=matched_strategy,
        source_inp_path=source_inp_path,
        native_controls_inp_path=native_controls_inp_path,
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
            "contract": MATCHED_BASELINE_V2_CONTRACT,
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
            "static_inp_topology_is_online_information": False,
            "source_inp_path": str(Path(source_inp_path).resolve()),
            "efd_topology_contract": (
                EFD_TOPOLOGY_CONTRACT if matched_strategy == MATCHED_EFD else None
            ),
            "matched_efd_contract": (
                MATCHED_EFD_V2_CONTRACT if matched_strategy == MATCHED_EFD else None
            ),
            "matched_internal_rule_contract": (
                MATCHED_INTERNAL_RULE_CONTRACT if matched_strategy == MATCHED_INTERNAL_RTC else None
            ),
            "native_controls_inp_path": native_controls_inp_path,
            "development_only": True,
            "formal_evidence": False,
            "requires_new_policy_lock": True,
            "ready_for_policy_lock": False,
        }
    )
    if matched.efd_topology is not None:
        lineage["efd_mapped_storage_count"] = len(matched.efd_topology.mapped_storage_ids)
        lineage["efd_mapped_actuator_count"] = len(matched.efd_topology.mapped_actuator_ids)
        lineage["efd_unmapped_storage_count"] = len(matched.efd_topology.unmapped_storage_ids)
        lineage["efd_ambiguous_actuator_count"] = len(matched.efd_topology.ambiguous_actuator_ids)
    return controller, graph, sensors, lineage


__all__ = [
    "MATCHED_BASELINE_V2_CONTRACT",
    "MATCHED_EFD_V2_CONTRACT",
    "TopologyMatchedInformationBaselineMPC",
    "build_matched_information_baseline_v2_controller",
]
