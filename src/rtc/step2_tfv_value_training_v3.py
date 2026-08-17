"""Selection-aware Direct-TFV training for Project7.

V2 established that direct exact-delta-TFV learning gives useful D2/D3/D4 ordering.  The remaining
failure was more specific: candidate ordering could be useful while the model still crossed the
HOLD=0 decision boundary too aggressively.  V3 therefore keeps the same V(candidate)-V(reference)
architecture and adds losses that explicitly teach the control decision the runtime actually makes.

Training stages:
1. MAIN: exact single-facility counterfactuals learn 109 facility main-value differences.
2. JOINT: freeze the main representation and fit the interaction value on multi-facility branches.
3. SELECTION: low-learning-rate head/late-encoder fine-tuning on all TrainFit branches with explicit
   HOLD-vs-action sign and oracle-choice losses.

The primary regression target remains authoritative exact SWMM delta TFV.  No future hydraulic
trajectory or gradient label is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_tfv_value import DirectFacilityTFVValueModel
from .step2_tfv_value_training import (
    _branch_indices,
    _epoch_group_order,
    _forward_candidates,
    _graph_tensors,
    _pairwise_ranking_loss,
    _with_hold_zero,
    derive_direct_tfv_target_scale_m3,
    evaluate_direct_tfv_value_model,
)
from .step2_train_response_v60 import InputNormalizationV60


DIRECT_TFV_TRAINING_CONTRACT = "PROJECT7_DIRECT_TFV_SELECTION_AWARE_TRAINING_V3"


@dataclass(frozen=True)
class DirectTFVTrainingDesignV3:
    main_epochs: int = 6
    interaction_epochs: int = 4
    selection_epochs: int = 2
    learning_rate: float = 5.0e-4
    interaction_learning_rate: float = 5.0e-4
    selection_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    rank_weight: float = 0.35
    hold_action_weight: float = 0.35
    oracle_choice_weight: float = 0.15
    harmful_false_benefit_weight: float = 2.0
    interaction_l1_weight: float = 0.01
    practical_zero_m3: float = 1.0
    grad_clip: float = 5.0
    seed: int = 42

    def validate(self) -> None:
        if min(self.main_epochs, self.interaction_epochs, self.selection_epochs) <= 0:
            raise ValueError("Direct-TFV V3 training epochs must be positive")
        for value in (
            self.learning_rate,
            self.interaction_learning_rate,
            self.selection_learning_rate,
            self.weight_decay,
            self.rank_weight,
            self.hold_action_weight,
            self.oracle_choice_weight,
            self.interaction_l1_weight,
            self.practical_zero_m3,
            self.grad_clip,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("Direct-TFV V3 training design contains an invalid value")
        if not math.isfinite(float(self.harmful_false_benefit_weight)) or float(
            self.harmful_false_benefit_weight
        ) < 1.0:
            raise ValueError("harmful_false_benefit_weight must be finite and >= 1")
        if self.seed < 0:
            raise ValueError("training seed must be non-negative")


def _hold_action_loss(
    prediction_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: torch.Tensor,
    harmful_false_benefit_weight: float,
    practical_zero_m3: float,
) -> torch.Tensor:
    """Teach the sign of candidate advantage relative to exact HOLD=0 explicitly.

    Positive truth means the action is worse than HOLD and must receive a positive predicted delta.
    Negative truth means the action is beneficial and must receive a negative predicted delta.
    Harmful actions receive extra weight because a false-beneficial prediction can trigger a real
    control move, while a missed benefit merely falls back to HOLD.
    """

    informative = torch.abs(truth_m3) > float(practical_zero_m3)
    if not bool(informative.any()):
        return prediction_m3.new_zeros(())
    prediction = prediction_m3[informative]
    truth = truth_m3[informative]
    desired_sign = torch.sign(truth)
    normalized = (prediction / scale_m3.clamp_min(1.0)) * desired_sign
    weights = torch.where(
        truth > float(practical_zero_m3),
        prediction.new_full(prediction.shape, float(harmful_false_benefit_weight)),
        prediction.new_ones(prediction.shape),
    )
    return torch.mean(weights * F.softplus(-normalized))


def _oracle_choice_loss(
    prediction_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: torch.Tensor,
) -> torch.Tensor:
    """Listwise control-choice loss with HOLD included as the exact zero option."""

    prediction = _with_hold_zero(prediction_m3)
    truth = _with_hold_zero(truth_m3)
    oracle = torch.argmin(truth.detach()).reshape(1)
    logits = (-prediction / scale_m3.clamp_min(1.0)).reshape(1, -1)
    return F.cross_entropy(logits, oracle)


def _group_loss_v3(
    model: DirectFacilityTFVValueModel,
    batch: Any,
    *,
    mode: str,
    graph_tensors: Mapping[str, torch.Tensor],
    design: DirectTFVTrainingDesignV3,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    indices = _branch_indices(batch, mode=mode)
    if int(indices.numel()) == 0:
        return None
    output = _forward_candidates(model, batch, indices, graph_tensors=graph_tensors)
    truth = batch.true_delta_tfv_m3[0].index_select(0, indices)
    scale = model.target_scale_m3.to(truth)
    regression = F.smooth_l1_loss(output.total_delta_tfv_m3 / scale, truth / scale)
    ranking = _pairwise_ranking_loss(
        _with_hold_zero(output.total_delta_tfv_m3),
        _with_hold_zero(truth),
        scale_m3=scale,
    )
    hold_action = _hold_action_loss(
        output.total_delta_tfv_m3,
        truth,
        scale_m3=scale,
        harmful_false_benefit_weight=design.harmful_false_benefit_weight,
        practical_zero_m3=design.practical_zero_m3,
    )
    oracle_choice = _oracle_choice_loss(
        output.total_delta_tfv_m3,
        truth,
        scale_m3=scale,
    )
    interaction_reg = torch.mean(torch.abs(output.interaction_residual_m3 / scale))
    loss = (
        regression
        + float(design.rank_weight) * ranking
        + float(design.hold_action_weight) * hold_action
        + float(design.oracle_choice_weight) * oracle_choice
    )
    if mode in {"joint", "all"}:
        loss = loss + float(design.interaction_l1_weight) * interaction_reg
    return loss, {
        "loss": float(loss.detach().cpu()),
        "regression": float(regression.detach().cpu()),
        "ranking": float(ranking.detach().cpu()),
        "hold_action": float(hold_action.detach().cpu()),
        "oracle_choice": float(oracle_choice.detach().cpu()),
        "interaction_l1": float(interaction_reg.detach().cpu()),
        "branches": float(indices.numel()),
    }


def _set_trainable(model: DirectFacilityTFVValueModel, *, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "main":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    if stage == "joint":
        for parameter in model.interaction_head.parameters():
            parameter.requires_grad_(True)
        return
    if stage == "selection":
        # Preserve the causal/global representation learned in MAIN.  The late facility encoder and
        # value heads are sufficient to move the absolute HOLD decision boundary without rewriting
        # the entire representation at the end of training.
        for module in (model.facility_encoder, model.facility_head, model.interaction_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return
    raise ValueError(f"unknown Direct-TFV V3 stage: {stage}")


def _train_stage(
    model: DirectFacilityTFVValueModel,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    stage: str,
    mode: str,
    epochs: int,
    learning_rate: float,
    design: DirectTFVTrainingDesignV3,
) -> list[dict[str, float | int | str]]:
    _set_trainable(model, stage=stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"Direct-TFV V3 stage {stage!r} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(design.weight_decay),
    )
    static = _graph_tensors(graph, device)
    history: list[dict[str, float | int | str]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        totals: dict[str, list[float]] = {
            "loss": [],
            "regression": [],
            "ranking": [],
            "hold_action": [],
            "oracle_choice": [],
            "interaction_l1": [],
        }
        groups_used = branches_used = 0
        for source, name in _epoch_group_order(source_groups, epoch=epoch, seed=design.seed):
            batch = source_caches[source].batch(name, normalization, device)
            result = _group_loss_v3(
                model,
                batch,
                mode=mode,
                graph_tensors=static,
                design=design,
            )
            if result is None:
                continue
            loss, metrics = result
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(design.grad_clip))
            optimizer.step()
            groups_used += 1
            branches_used += int(metrics["branches"])
            for key in totals:
                totals[key].append(float(metrics[key]))
        if not totals["loss"]:
            raise RuntimeError(f"Direct-TFV V3 stage {stage!r} found no usable branches")
        history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "groups": int(groups_used),
                "branches": int(branches_used),
                **{key: float(np.mean(values)) for key, values in totals.items()},
            }
        )
    return history


def train_direct_tfv_value_model_v3(
    model: DirectFacilityTFVValueModel,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: DirectTFVTrainingDesignV3 = DirectTFVTrainingDesignV3(),
) -> dict[str, list[dict[str, float | int | str]]]:
    design.validate()
    main = _train_stage(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="main",
        mode="single",
        epochs=design.main_epochs,
        learning_rate=design.learning_rate,
        design=design,
    )
    joint = _train_stage(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="joint",
        mode="joint",
        epochs=design.interaction_epochs,
        learning_rate=design.interaction_learning_rate,
        design=design,
    )
    selection = _train_stage(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="selection",
        mode="all",
        epochs=design.selection_epochs,
        learning_rate=design.selection_learning_rate,
        design=design,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {"main": main, "joint": joint, "selection": selection}


__all__ = [
    "DIRECT_TFV_TRAINING_CONTRACT",
    "DirectTFVTrainingDesignV3",
    "derive_direct_tfv_target_scale_m3",
    "evaluate_direct_tfv_value_model",
    "train_direct_tfv_value_model_v3",
    "_hold_action_loss",
    "_oracle_choice_loss",
]
