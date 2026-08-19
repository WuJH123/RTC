"""Experimental scenario-mean Direct-TFV MPC for Development-only V12 work.

Why this exists
---------------
The current V11 policy scores exactly one persistence-decay rainfall forecast.  Long-duration and
high-intensity Development probes exposed first-move sign/magnitude failures even though target-latch
execution, engineering checks and score==execute were correct.  This module adds a *label-independent*
robustness dimension: every candidate is scored under several causal rainfall scenarios and the online
objective is the mean predicted system-wide delta TFV across those scenarios.

The TFV objective is unchanged; no PFV/peak/action penalty or baseline imitation is introduced.  The
class is deliberately fail-closed on a scenario-matched first-move admission artifact.  Therefore it
cannot silently reuse V11 single-scenario calibration.  It is additive Development code and is not
current/Policy-Locked until fresh exact-query SWMM calibration and independent probes pass.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from .direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS
from .direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
)
from .direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from .step3_tfv_value_mpc_v7 import DirectTFVRecedingMPCV7
from .step3_tfv_value_mpc_v9 import DirectTFVRecedingMPCV9


DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT = (
    "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V10_CAUSAL_RAINFALL_SCENARIO_MEAN"
)
DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT = (
    "PROJECT7_CAUSAL_RAINFALL_PERSISTENCE_DECAY_MULTIPLIER_ENSEMBLE_V1"
)


class DirectTFVScenarioMeanMPCV10(DirectTFVRecedingMPCV9):
    """V11 target-latch first-move policy with differentiable mean-over-scenarios scoring.

    ``rainfall`` is ``[scenario,H,node,1]``.  Candidate and rainfall dimensions are crossed into one
    model batch, then reshaped back to ``[candidate,scenario]`` and averaged.  The operation remains
    differentiable, so the existing L-BFGS-B full-plan optimizer, shrink-only first-move refiner and
    TFV-consistent backward pruning all optimize the same scenario-mean objective.
    """

    policy_mode = "direct_tfv_all109_receding_mpc_v10_scenario_mean"
    policy_mode_contract = DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        policy_admission_calibration: Mapping[str, Any],
        first_move_admission_calibration: Mapping[str, Any],
        sequence_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
        first_move_maxiter: int = 12,
        first_move_deadline_seconds: float = 30.0,
        minimum_rainfall_scenarios: int = 3,
    ) -> None:
        # Replicate V9 admission validation, but require a V10/scenario-matched exact-query artifact.
        policy = dict(policy_admission_calibration)
        if str(policy.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
            raise ValueError("scenario-mean Direct-TFV requires accepted V2 policy admission lineage")
        first = dict(first_move_admission_calibration)
        if str(first.get("contract", "")) != DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT:
            raise ValueError("scenario-mean Direct-TFV requires a first-move admission artifact")
        if first.get("development_only") is not True:
            raise ValueError("scenario-mean first-move admission must be Development-only")
        if str(first.get("execution_estimand", "")) != DIRECT_TFV_FIRST_MOVE_SEMANTICS:
            raise ValueError("scenario-mean first-move admission has the wrong execution estimand")
        if str(first.get("query_step3_contract", "")) != DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT:
            raise ValueError(
                "scenario-mean policy cannot reuse a single-scenario V11 first-move calibration"
            )
        if str(first.get("rainfall_scenario_contract", "")) != DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT:
            raise ValueError("scenario-mean first-move admission has the wrong rainfall scenario contract")
        if first.get("generic_d3_floor_controls_execution") is not False:
            raise ValueError("generic D3 margin must not control scenario-mean first-move execution")
        if int(first.get("calibration_rainfall_group_count", 0)) < int(
            first.get("minimum_calibration_rainfall_groups", 24)
        ):
            raise ValueError("scenario-mean first-move calibration has insufficient rainfall groups")
        if minimum_rainfall_scenarios < 2:
            raise ValueError("scenario-mean Direct-TFV requires at least two rainfall scenarios")
        if first_move_maxiter <= 0 or not 0.0 < float(first_move_deadline_seconds) < 600.0:
            raise ValueError("invalid scenario-mean first-move refinement budget")

        # Call the V7 constructor directly because V9 hard-codes its own single-scenario query
        # contract.  All V9 optimize/refinement behavior is inherited below; only the exact-query
        # calibration contract and scenario scoring differ.
        DirectTFVRecedingMPCV7.__init__(
            self,
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            policy_admission_calibration=policy,
            sequence_support=sequence_support,
            design=design,
        )
        self.first_move_admission_calibration = first
        self.first_move_maxiter = int(first_move_maxiter)
        self.first_move_deadline_seconds = float(first_move_deadline_seconds)
        self.minimum_rainfall_scenarios = int(minimum_rainfall_scenarios)

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
            raise ValueError("scenario-mean Direct-TFV expects one current state/flow vector")
        if rainfall.ndim != 4:
            raise ValueError("rainfall scenarios must be [scenario,H,node,1]")
        scenario_count = int(rainfall.shape[0])
        if scenario_count < self.minimum_rainfall_scenarios:
            raise ValueError(
                f"scenario-mean Direct-TFV requires >= {self.minimum_rainfall_scenarios} scenarios; "
                f"got {scenario_count}"
            )
        if sequences.ndim != 3 or tuple(sequences.shape[1:]) != (
            self.design.prediction_horizon_steps,
            109,
        ):
            raise ValueError("scenario-mean sequences must be [candidate,H72,109]")
        if rainfall.shape[1] != self.design.prediction_horizon_steps or rainfall.shape[-1] != 1:
            raise ValueError("rainfall scenario horizon/channel differs from Direct-TFV design")

        candidate_count = int(sequences.shape[0])
        batch_count = candidate_count * scenario_count
        state = self._normalize_state(current_state).expand(batch_count, -1, -1)
        rain = self._normalize_rainfall(rainfall)
        rain = rain.unsqueeze(0).expand(candidate_count, -1, -1, -1, -1).reshape(
            batch_count, *rainfall.shape[1:]
        )
        flow = self._normalize_flow(previous_actuator_flow).expand(batch_count, -1)
        candidate = sequences.unsqueeze(1).expand(-1, scenario_count, -1, -1).reshape(
            batch_count, self.design.prediction_horizon_steps, 109
        )
        reference = self._hold_sequence(active_target)[None, None].expand(
            candidate_count, scenario_count, -1, -1
        ).reshape(batch_count, self.design.prediction_horizon_steps, 109)
        output = self.model(
            current_state=state,
            rainfall=rain,
            reference_settings=reference,
            candidate_settings=candidate,
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
        scenario_scores = output.total_delta_tfv_m3.reshape(candidate_count, scenario_count)
        if not bool(torch.isfinite(scenario_scores).all()):
            raise RuntimeError("scenario-mean Direct-TFV produced non-finite scenario scores")
        return scenario_scores.mean(dim=1)


__all__ = [
    "DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT",
    "DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT",
    "DirectTFVScenarioMeanMPCV10",
]
