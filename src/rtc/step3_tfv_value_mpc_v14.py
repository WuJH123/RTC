"""Project7 Step3 V14: jointly rank the three-family query and separately admit versus HOLD.

V13 scored each candidate independently with one absolute scalar and then used that same scalar both
for ordering and crossing HOLD=0.  V14 preserves the exact H10 portfolio/support geometry but evaluates
the whole 1--3 candidate set jointly.  Relative rank chooses the candidate; a separately supervised
query-level best-candidate margin decides ACTION/HOLD.  One-sided conformal uncertainty is applied
only after the best candidate is selected, so uncertainty does not silently become an optimization
objective.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import numpy as np
import torch

from .direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT, policy_return_margin_m3
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio,
)
from .direct_tfv_policy_return_portfolio import score_h10_first_action_targets
from .direct_tfv_policy_return_query_margin import (
    DIRECT_TFV_QUERY_MARGIN_CONTRACT,
    QueryConditionedPolicyReturnAdapter,
    build_query_margin_features,
)
from .direct_tfv_sequence_support import changed_facility_support_limit
from .step3_tfv_value_mpc_v12 import DirectTFVMPCResultV12
from .step3_tfv_value_mpc_v13 import DirectTFVHybridPolicyReturnMPCV13


DIRECT_TFV_QUERY_MARGIN_STEP3_CONTRACT = (
    "PROJECT7_PRACTICAL_RTC_QUERY_CONDITIONED_POLICY_RETURN_V16_82CONTROL_109REP"
)


class DirectTFVQueryMarginMPCV14(DirectTFVHybridPolicyReturnMPCV13):
    policy_mode = "practical_direct_tfv_query_conditioned_policy_return"
    policy_mode_contract = DIRECT_TFV_QUERY_MARGIN_STEP3_CONTRACT

    def __init__(self, *, query_margin_adapter: QueryConditionedPolicyReturnAdapter, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.query_margin_adapter = query_margin_adapter
        self.query_margin_adapter.eval()

    def _hold_result(self, *, started: float, active_target: torch.Tensor, learned: Any) -> DirectTFVMPCResultV12:
        result = super()._hold_result(started=started, active_target=active_target, learned=learned)
        return replace(result, policy_mode=self.policy_mode, policy_mode_contract=self.policy_mode_contract)

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV12:
        started=time.perf_counter()
        active_target=kwargs.get("active_target")
        if not isinstance(active_target,torch.Tensor) or tuple(active_target.shape)!=(109,):
            raise ValueError("query-conditioned Step3 requires active_target [109]")
        current_state=kwargs["current_state"]; rainfall=kwargs["rainfall"]
        flow=kwargs["previous_actuator_flow"]
        if tuple(flow.shape)!=(1,109): raise ValueError("query-conditioned Step3 requires flow [1,109]")
        ceiling=changed_facility_support_limit(self.sequence_support,"q95")
        hybrid=build_hybrid_policy_return_portfolio(
            model=self.model,normalization=self.normalization,graph=self.graph,
            current_state=current_state,rainfall_scenarios=rainfall,previous_actuator_flow=flow,
            active_target=active_target,first_radius=self.first_radius,max_changed_facilities=ceiling,
            max_delta_per_update=float(self.design.max_setting_delta_per_update),
            probe_chunk_size=self.proposal_probe_chunk_size,supervisory_mask=self.supervisory_mask,
            include_projected_gradient_ablation=False,
        )
        learned=hybrid.learned_probe
        evaluated=[]; seen:set[bytes]=set()
        passive=torch.as_tensor(~self.supervisory_mask,dtype=torch.bool,device=active_target.device)
        for proposal in hybrid.candidates:
            target,sequence,changed,diagnostics=self._h10_supported_target(proposal.target,active_target)
            if changed<=0: continue
            if bool(torch.any(torch.abs(target[passive]-active_target[passive])>1e-7)):
                raise RuntimeError("query-conditioned candidate changed a passive channel")
            key=target.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            if key in seen: continue
            seen.add(key)
            raw_rank=float(self._score_policy_return_target(
                current_state=current_state,rainfall=rainfall,previous_actuator_flow=flow,
                active_target=active_target,candidate_target=target).detach().cpu())
            base_score=float(score_h10_first_action_targets(
                model=self.model,normalization=self.normalization,graph=self.graph,
                current_state=current_state,rainfall_scenarios=rainfall,previous_actuator_flow=flow,
                active_target=active_target,candidate_targets=target.reshape(1,109),probe_chunk_size=1
            )[0].detach().cpu())
            evaluated.append([proposal.source,target,sequence,changed,raw_rank,base_score,diagnostics])
        if not evaluated:
            return self._hold_result(started=started,active_target=active_target,learned=learned)

        targets=torch.stack([row[1] for row in evaluated])
        raw_rank=torch.as_tensor([row[4] for row in evaluated],dtype=targets.dtype,device=targets.device)
        base_scores=torch.as_tensor([row[5] for row in evaluated],dtype=targets.dtype,device=targets.device)
        context_features,candidate_features=build_query_margin_features(
            current_state=current_state[0] if current_state.ndim==3 and int(current_state.shape[0])==1 else current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow[0],active_target=active_target,candidate_targets=targets,
            base_step2_scores_m3=base_scores,candidate_sources=[str(row[0]) for row in evaluated],
            supervisory_mask=self.supervisory_mask,
            target_scale_m3=float(self.policy_return_model.target_scale_m3.detach().cpu()),
        )
        with torch.no_grad():
            joint=self.query_margin_adapter(
                raw_rank_scores_m3=raw_rank,context_features=context_features,candidate_features=candidate_features
            )
        predicted=joint.predicted_returns_m3.detach().cpu().numpy().astype(float)
        relative=joint.relative_rank_normalized.detach().cpu().numpy().astype(float)
        # Ranking is prediction-only. Conformal uncertainty is deliberately not used to rerank actions.
        selected_index=int(np.argmin(relative))
        source,target,sequence,changed,raw_score,base_score,diagnostics=evaluated[selected_index]
        score=float(predicted[selected_index])
        query_margin=float(joint.query_best_margin_m3.detach().cpu())
        if abs(score-query_margin)>max(1e-4,1e-6*max(1.0,abs(query_margin))):
            raise RuntimeError("selected query-margin candidate is not anchored at the query best margin")
        margin=float(policy_return_margin_m3(self.policy_return_admission,changed))
        upper=score+margin
        passed=bool(upper<0.0)
        hold=active_target.reshape(1,109).expand(int(self.design.prediction_horizon_steps),-1).detach()
        executed=sequence if passed else hold
        changed_indices=torch.nonzero(torch.abs(target-active_target)>1e-7).reshape(-1).tolist()
        changed_ids=tuple(str(self.graph.actuator_ids[int(i)]) for i in changed_indices)
        uppers=[]
        for row,pred in zip(evaluated,predicted.tolist()):
            uppers.append(float(pred)+float(policy_return_margin_m3(self.policy_return_admission,int(row[3]))))
        return DirectTFVMPCResultV12(
            settings=executed,optimized_candidate_settings=sequence,
            predicted_delta_tfv_m3=score if passed else 0.0,
            raw_optimized_predicted_delta_tfv_m3=score,
            selected_source=(f"DIRECT_TFV_QUERY_MARGIN::{source}" if passed else "LATCH_PREVIOUS_TARGET_QUERY_MARGIN_UCB_NONNEGATIVE"),
            candidate_valid=passed,admission_margin_m3=margin,admission_upper_bound_m3=upper,
            admission_margin_kind="query_conditioned_best_candidate_policy_return",
            admission_passed=passed,calibrated_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
            elapsed_seconds=float(time.perf_counter()-started),screened_facility_count=self.supervisory_control_dimension,
            predicted_beneficial_facility_count=int(learned.predicted_beneficial_facility_count),
            active_facility_count=changed,active_facility_ids=changed_ids,
            active_facility_screening_scores_m3=learned.facility_best_scores_m3,
            first_move_changed_facility_count=changed if passed else 0,
            maximum_support_ratio=self._first_move_support_ratio(target,active_target),
            joint_sequence_support_quantile=str(diagnostics["quantile"]),
            joint_sequence_first_block_l1=float(diagnostics["first_block_l1"]),
            joint_sequence_h120_l1=float(diagnostics["h120_l1"]),
            joint_sequence_h120_total_variation_l1=float(diagnostics["h120_total_variation_l1"]),
            joint_sequence_support_max_ratio=float(diagnostics["max_ratio"]),
            joint_sequence_support_binding=bool(diagnostics["binding"]),
            policy_return_predicted_delta_tfv_m3=score,policy_return_margin_m3=margin,
            policy_return_upper_bound_m3=upper,policy_return_admission_passed=passed,
            policy_return_admission_contract=DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
            policy_return_estimand=self.policy_return_admission["estimand"],
            policy_return_parent_continuation_sha256=self.policy_return_parent_continuation_sha256,
            policy_return_portfolio_contract=DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            policy_return_portfolio_candidate_count=len(evaluated),
            policy_return_portfolio_selected_source=source if passed else "HOLD",
            policy_return_portfolio_sources=tuple(str(row[0]) for row in evaluated),
            policy_return_portfolio_scores_m3=tuple(float(x) for x in predicted.tolist()),
            policy_return_portfolio_upper_bounds_m3=tuple(uppers),
            policy_return_portfolio_base_step2_scores_m3=tuple(float(row[5]) for row in evaluated),
            h10_probe_generator_contract=DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
            h10_probe_count=int(learned.probe_count),policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
            refined_first_move_semantics=self.policy_return_admission["action_encoding_contract"],
            refined_first_move_changed_facility_count=changed,
            refined_first_move_changed_facility_ids=changed_ids,
            optimizer_success=True,optimizer_steps=0,optimizer_starts=0,gradient_norm=0.0,
            scipy_message=("NOT_USED_QUERY_CONDITIONED_THREE_FAMILY_FINITE_PORTFOLIO::"+DIRECT_TFV_QUERY_MARGIN_CONTRACT),
            first_move_refiner_steps=0,first_move_refinement_gain_m3=0.0,
        )


__all__=["DIRECT_TFV_QUERY_MARGIN_STEP3_CONTRACT","DirectTFVQueryMarginMPCV14"]
