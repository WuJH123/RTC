"""V9 objective: signed unclipped counterfactual effects are the primary targets."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .step2_hydraulic_objective_v80 import (
    derive_onset_sqrt_positive_weight_v80,
    hydraulic_effect_loss_v80,
    initial_flood_physical_v80,
    retained_onset_targets_v80,
)
from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch
from .step2_train_response_v70 import TargetScalesV70
from .step2_v90_contract import DirectHydraulicEffectLossContractV90


def hydraulic_effect_loss_v90(
    output: Any,
    batch: V60GroupBatch,
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    *,
    onset_positive_weight: float,
    contract: DirectHydraulicEffectLossContractV90 = DirectHydraulicEffectLossContractV90(),
):
    """Reuse the audited V8 loss terms but bind them explicitly to V9 raw deltas.

    Projected absolute candidate trajectories are intentionally absent from this proxy,
    so physical clipping cannot alter the signed training target or prediction.
    """
    proxy = SimpleNamespace(
        horizon_indices=output.horizon_indices,
        reference_states_physical=output.reference_states_physical,
        delta_states_physical=output.raw_delta_states_physical,
        reference_flows_physical=output.reference_flows_physical,
        delta_flows_physical=output.raw_delta_flows_physical,
        reference_flood_onset_logits=output.reference_flood_onset_logits,
        candidate_flood_onset_logits=output.candidate_flood_onset_logits,
    )
    return hydraulic_effect_loss_v80(
        proxy,
        batch,
        normalization,
        scales,
        onset_positive_weight=onset_positive_weight,
        contract=contract,
    )


__all__ = [
    "derive_onset_sqrt_positive_weight_v80",
    "hydraulic_effect_loss_v90",
    "initial_flood_physical_v80",
    "retained_onset_targets_v80",
]
