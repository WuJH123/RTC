"""V12.5 anchor-relative finite RTC policy.

The Sparse-RBC engineering action is the online Value reference. Each learned candidate
differs only in the executable first 10-minute block and then shares the exact anchor
continuation. Step2 predicts candidate-minus-anchor TFV/PFV directly. The anchor remains
exact zero even when PFV model uncertainty is nonzero; that uncertainty is applied only
to candidates that actually deviate from the anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .step2_control_basis_v60 import ControlBasisV60
from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_value_v123 import DualVolumeValueV123
from .step2_d4_action_support_v125 import (
    D4ActionSupportContractV125,
    common_anchor_continuation_sequence_v125,
    knowledge_neighbourhood_first_moves_v125,
)
from .step2_policy_v120 import RuntimeNormalizationV120, _project_executable_sequences_v120
from .step2_policy_v123 import anchor_base_settings_v123, safe_runtime_delta_v123
from .step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123
from .step3_objective_v123 import TFVPFVObjectiveV123
from .step3_objective_v125 import tfv_pfv_score_v125

V125_POLICY_CONTRACT = "PROJECT7_V125_ANCHOR_DEFAULT_EVIDENCE_GATED_OVERRIDE_V2_DIRECT_ADVANTAGE"
V125_OVERRIDE_CALIBRATION_CONTRACT = "PROJECT7_V125_ANCHOR_RELATIVE_TFV_FALSE_BENEFIT_V2"


@dataclass(frozen=True)
class AnchorOverrideResultV125:
    settings: torch.Tensor
    candidate_valid: bool
    selected_candidate_index: int
    raw_candidate_count: int
    first_move_group_count: int
    tail_only_noop_candidate_count: int
    scenario_count: int
    predicted_delta_tfv_m3: float
    predicted_delta_pfv_m3: float
    tfv_risk_m3: float
    pfv_risk_m3: float
    pfv_soft_excess_m3: float
    pfv_penalty_m3_equivalent: float
    objective_score_m3_equivalent: float
    selected_group_score_m3: float
    false_benefit_margin_m3: float
    scoring_projection_max: float
    knowledge_anchor_candidate_index: int
    knowledge_anchor_selected: bool
    knowledge_anchor_fallback_used: bool
    knowledge_anchor_first_move_delta_max: float
    anchor_tfv_risk_m3: float
    anchor_pfv_risk_m3: float
    anchor_objective_score_m3: float
    learned_tfv_risk_m3: float
    learned_pfv_risk_m3: float
    learned_objective_score_m3: float
    predicted_override_advantage_tfv_m3: float
    anchor_override_margin_m3: float
    learned_override_admitted: bool
    selected_source: str
    selected_candidate_family: str
    policy_mode: str = "anchor_override"
    policy_mode_contract: str = V125_POLICY_CONTRACT

    @property
    def candidate_count(self) -> int:
        return int(self.raw_candidate_count)

    @property
    def controller_source(self) -> str:
        return "MPC_V125"


class AnchorOverridePolicyV125:
    """Causal first-move finite shooting around a Sparse-RBC reference action."""

    accepts_previous_requested_settings = True
    policy_mode = "anchor_override"
    policy_mode_contract = V125_POLICY_CONTRACT
    use_sparse_rbc_anchor = True
    knowledge_anchor_fallback = True

    def __init__(
        self,
        *,
        model: DualVolumeValueV123,
        basis: ControlBasisV60,
        prepared: PreparedStaticV60,
        normalization: RuntimeNormalizationV120,
        objective: TFVPFVObjectiveV123,
        anchor_override_margin_m3: float,
        graph,
        max_active_groups: int = 3,
        local_fraction: float = 0.25,
        first_move_group_atol: float = 1.0e-7,
    ) -> None:
        basis.validate()
        normalization.validate()
        objective.validate()
        margin = float(anchor_override_margin_m3)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("V125 anchor override margin must be finite and non-negative")
        if graph is None:
            raise ValueError("V125 requires the frozen graph for causal Sparse-RBC anchor")
        if max_active_groups <= 0 or not 0.0 < float(local_fraction) <= 0.5:
            raise ValueError("V125 local candidate design is invalid")
        if not math.isfinite(float(first_move_group_atol)) or first_move_group_atol <= 0.0:
            raise ValueError("V125 first-move grouping tolerance must be positive")
        self.model = model
        self.basis = basis
        self.prepared = prepared
        self.normalization = normalization
        self.objective = objective
        self.anchor_override_margin_m3 = margin
        self.false_benefit_margin_m3 = margin
        self.graph = graph
        self.max_active_groups = int(max_active_groups)
        self.local_fraction = float(local_fraction)
        self.first_move_group_atol = float(first_move_group_atol)

    def _anchor(
        self,
        *,
        initial_state: torch.Tensor,
        command_base: torch.Tensor,
        fallback_settings: torch.Tensor,
        runtime_delta: float,
    ) -> torch.Tensor:
        anchor = build_sparse_state_auto_rbc_anchor_v123(
            initial_state.detach().cpu().numpy(),
            command_base.detach().cpu().numpy(),
            fallback_settings[0].detach().cpu().numpy(),
            self.graph,
            control_block_steps=int(self.basis.horizon.control_block_steps),
            max_delta_per_update=float(runtime_delta),
        )
        return torch.as_tensor(
            anchor,
            dtype=fallback_settings.dtype,
            device=fallback_settings.device,
        )

    def optimize(
        self,
        initial_state: torch.Tensor,
        rainfall_scenarios: torch.Tensor,
        fallback_settings: torch.Tensor,
        *,
        current_settings: torch.Tensor | None = None,
        previous_requested_settings: torch.Tensor | None = None,
        previous_actuator_flow: torch.Tensor | None = None,
        max_delta_per_update: float | torch.Tensor | None = None,
        **_controller_compatibility: object,
    ) -> AnchorOverrideResultV125:
        if initial_state.ndim == 3 and initial_state.shape[0] == 1:
            initial_state = initial_state[0]
        if initial_state.ndim != 2 or rainfall_scenarios.ndim != 4:
            raise ValueError("V125 policy received incompatible state/rainfall shape")
        if fallback_settings.ndim != 3 or fallback_settings.shape[0] != 1:
            raise ValueError("V125 fallback settings must be [1,H,A]")
        if fallback_settings.shape[1] != self.basis.horizon.horizon_steps:
            raise ValueError("V125 fallback/value horizons differ")
        if current_settings is None:
            raise ValueError("V125 policy requires current-setting readback")

        actuator_count = self.basis.grouping.actuator_count
        current = current_settings.reshape(-1)
        previous_target = (
            None if previous_requested_settings is None
            else previous_requested_settings.reshape(-1)
        )
        if current.numel() != actuator_count or (
            previous_target is not None and previous_target.numel() != actuator_count
        ):
            raise ValueError("V125 actuator readback count mismatch")
        if previous_actuator_flow is None:
            previous_actuator_flow = torch.zeros(
                actuator_count, dtype=initial_state.dtype, device=initial_state.device
            )
        flow0 = previous_actuator_flow.reshape(-1)
        if flow0.numel() != actuator_count:
            raise ValueError("V125 previous actuator-flow count mismatch")

        frozen_delta = float(self.basis.contract.max_setting_delta_per_update)
        runtime_delta = frozen_delta
        if max_delta_per_update is not None:
            raw = torch.as_tensor(max_delta_per_update, dtype=torch.float32).reshape(-1)
            if not bool(torch.isfinite(raw).all()):
                raise ValueError("V125 runtime max delta is non-finite")
            runtime_delta = float(raw.max())
            if runtime_delta > frozen_delta + 1.0e-9:
                raise ValueError("V125 runtime max delta is looser than frozen basis")
        effective_delta = safe_runtime_delta_v123(runtime_delta)
        if effective_delta <= 0.0:
            raise ValueError("V125 runtime delta leaves no executable control range")

        command_base = anchor_base_settings_v123(current, previous_target)
        anchor = self._anchor(
            initial_state=initial_state,
            command_base=command_base,
            fallback_settings=fallback_settings,
            runtime_delta=effective_delta,
        )
        if anchor.shape != fallback_settings[0].shape:
            raise ValueError("V125 Sparse-RBC anchor/value horizon mismatch")
        block = int(self.basis.horizon.control_block_steps)
        anchor_np = anchor.detach().cpu().numpy().astype(np.float32)
        anchor_target = anchor_np[:block].mean(axis=0).astype(np.float32)
        contract = D4ActionSupportContractV125(
            max_checkpoints=1,
            local_fraction=self.local_fraction,
            max_active_groups=self.max_active_groups,
            max_delta_per_update=effective_delta,
        )
        plans = knowledge_neighbourhood_first_moves_v125(
            command_base.detach().cpu().numpy(),
            anchor_target,
            self.basis.grouping.group_id_by_actuator,
            self.basis.min_setting,
            self.basis.max_setting,
            contract=contract,
        )
        sequences: list[np.ndarray] = []
        families: list[str] = []
        for family, target in plans:
            sequences.append(
                common_anchor_continuation_sequence_v125(
                    target, anchor_np, control_block_steps=block
                )
            )
            families.append(str(family))
        if not sequences:
            raise RuntimeError("V125 local candidate generator returned no actions")
        candidate_one = torch.as_tensor(
            np.stack(sequences), dtype=anchor.dtype, device=anchor.device
        )
        candidate_one, projection_max = _project_executable_sequences_v120(
            candidate_one,
            current_settings=current,
            previous_requested_settings=previous_target,
            min_settings=torch.as_tensor(
                self.basis.min_setting, dtype=anchor.dtype, device=anchor.device
            ),
            max_settings=torch.as_tensor(
                self.basis.max_setting, dtype=anchor.dtype, device=anchor.device
            ),
            max_delta_per_update=effective_delta,
            control_block_steps=block,
        )
        if float(projection_max) > 1.0e-7:
            raise RuntimeError(
                "V125 local candidate generator produced a non-executable action "
                f"(projection={projection_max})"
            )
        anchor_matches = torch.all(
            torch.isclose(candidate_one, anchor[None], rtol=0.0, atol=1.0e-7),
            dim=(1, 2),
        )
        if int(anchor_matches.sum().item()) != 1:
            if torch.allclose(candidate_one[0], anchor, rtol=0.0, atol=1.0e-7):
                anchor_index = 0
            else:
                raise RuntimeError("V125 candidate family lacks a unique exact anchor reference")
        else:
            anchor_index = int(torch.nonzero(anchor_matches, as_tuple=False)[0, 0].item())

        scenarios = int(rainfall_scenarios.shape[0])
        state = initial_state[None].expand(scenarios, -1, -1)
        reference = anchor[None].expand(scenarios, -1, -1)
        candidate = candidate_one[None].expand(scenarios, -1, -1, -1)
        flow = flow0[None].expand(scenarios, -1)
        output = self.model(
            self.normalization.state(state),
            self.normalization.rainfall(rainfall_scenarios),
            reference,
            candidate,
            self.normalization.flow(flow),
            self.prepared,
        )
        expected = (scenarios, candidate_one.shape[0])
        if output.delta_tfv_m3.shape != expected or output.delta_pfv_m3.shape != expected:
            raise RuntimeError("V125 Value output shape drift")
        first = candidate_one[:, :block].mean(dim=1)
        anchor_first = anchor[:block].mean(dim=0)
        movement = torch.mean(torch.abs(first - anchor_first[None]), dim=1)
        scored = tfv_pfv_score_v125(
            output.delta_tfv_m3,
            output.delta_pfv_m3,
            movement_from_anchor=movement,
            contract=self.objective,
        )
        score = scored["score_m3_equivalent"]
        tfv_risk = scored["tfv_risk_m3"]
        pfv_risk = scored["pfv_risk_m3"]
        if (
            abs(float(tfv_risk[anchor_index].detach())) > 1.0e-5
            or abs(float(pfv_risk[anchor_index].detach())) > 1.0e-5
            or abs(float(score[anchor_index].detach())) > 1.0e-5
        ):
            raise RuntimeError("V125 anchor reference lost exact-zero TFV/PFV/objective value")

        rounded = torch.round(first / self.first_move_group_atol) * self.first_move_group_atol
        grouped: dict[bytes, list[int]] = {}
        for i, row in enumerate(rounded.detach().cpu().numpy()):
            grouped.setdefault(row.astype("float64").tobytes(), []).append(i)
        records: list[dict[str, object]] = []
        anchor_record: dict[str, object] | None = None
        for indices in grouped.values():
            contains_anchor = anchor_index in indices
            values = score[indices]
            median_score = values.median()
            nearest = torch.argmin(torch.abs(values - median_score))
            representative = indices[int(nearest.item())]
            record: dict[str, object] = {
                "representative": representative,
                "score": float(median_score.detach()),
                "tfv_risk": float(tfv_risk[indices].median().detach()),
                "pfv_risk": float(pfv_risk[indices].median().detach()),
                "contains_anchor": contains_anchor,
            }
            records.append(record)
            if contains_anchor:
                anchor_record = record
        if anchor_record is None:
            raise RuntimeError("V125 exact anchor first-move group disappeared")

        learned_records = [r for r in records if not bool(r["contains_anchor"])]
        best_learned = (
            min(learned_records, key=lambda r: float(r["score"]))
            if learned_records else None
        )
        eligible = [
            r for r in learned_records
            if float(r["score"]) < 0.0
            and float(r["tfv_risk"]) < -float(self.anchor_override_margin_m3)
        ]
        if eligible:
            selected_record = min(eligible, key=lambda r: float(r["score"]))
            selected = int(selected_record["representative"])
            override = True
            selected_source = "learned_override"
        else:
            selected_record = anchor_record
            selected = anchor_index
            override = False
            selected_source = "anchor_default"

        active_anchor = not torch.allclose(
            anchor_first, command_base, rtol=0.0, atol=1.0e-7
        )
        candidate_valid = bool(override or active_anchor)
        if not candidate_valid:
            selected_source = "passive_anchor"
        best = best_learned or anchor_record

        def scalar(name: str, index: int) -> float:
            return float(scored[name][index].detach())

        selected_score = float(selected_record["score"])
        selected_tfv = float(selected_record["tfv_risk"])
        selected_pfv = float(selected_record["pfv_risk"])
        anchor_delta = float(torch.abs(anchor_first - command_base).max().detach())
        return AnchorOverrideResultV125(
            settings=candidate_one[selected].detach(),
            candidate_valid=candidate_valid,
            selected_candidate_index=selected,
            raw_candidate_count=int(candidate_one.shape[0]),
            first_move_group_count=int(len(grouped)),
            tail_only_noop_candidate_count=0,
            scenario_count=scenarios,
            predicted_delta_tfv_m3=scalar("delta_tfv_mean_m3", selected),
            predicted_delta_pfv_m3=scalar("delta_pfv_mean_m3", selected),
            tfv_risk_m3=selected_tfv,
            pfv_risk_m3=selected_pfv,
            pfv_soft_excess_m3=scalar("pfv_soft_excess_m3", selected),
            pfv_penalty_m3_equivalent=scalar("pfv_penalty_m3_equivalent", selected),
            objective_score_m3_equivalent=selected_score,
            selected_group_score_m3=selected_score,
            false_benefit_margin_m3=float(self.anchor_override_margin_m3),
            scoring_projection_max=float(projection_max),
            knowledge_anchor_candidate_index=anchor_index,
            knowledge_anchor_selected=bool(selected == anchor_index),
            knowledge_anchor_fallback_used=bool(not override and active_anchor),
            knowledge_anchor_first_move_delta_max=anchor_delta,
            anchor_tfv_risk_m3=0.0,
            anchor_pfv_risk_m3=0.0,
            anchor_objective_score_m3=0.0,
            learned_tfv_risk_m3=float(best["tfv_risk"]),
            learned_pfv_risk_m3=float(best["pfv_risk"]),
            learned_objective_score_m3=float(best["score"]),
            predicted_override_advantage_tfv_m3=float(best["tfv_risk"]),
            anchor_override_margin_m3=float(self.anchor_override_margin_m3),
            learned_override_admitted=override,
            selected_source=selected_source,
            selected_candidate_family=families[selected],
        )


__all__ = [
    "AnchorOverridePolicyV125",
    "AnchorOverrideResultV125",
    "V125_OVERRIDE_CALIBRATION_CONTRACT",
    "V125_POLICY_CONTRACT",
]
