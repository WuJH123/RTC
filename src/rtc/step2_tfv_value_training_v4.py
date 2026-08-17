"""Core Direct-TFV training for Project7.

V5 keeps the validated pairwise value architecture and TFV-only target, but fixes a joint-action
learning bottleneck exposed by the Development baseline/counterfactual audit:

1. MAIN: exact single-facility branches learn the 109 facility effects with facility-balanced
   regression.
2. JOINT: multi-facility branches are balanced by changed-facility count and update the shared
   facility representation as well as the interaction head. V4 trained only ``interaction_head``;
   that prevented dense coordinated examples from adapting the representation used by Step3.
3. CONTROL: only D3 HOLD-reference TrainFit groups fine-tune the late value representation for the
   online sign decision: predicted delta TFV < 0 means act, otherwise HOLD.

D4 remains reference-shift stress evaluation only. No top-1 gate, uncertainty gate, improvement
threshold, PFV objective or baseline imitation is introduced.
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


DIRECT_TFV_TRAINING_CONTRACT = "PROJECT7_DIRECT_TFV_CORE_TRAINING_V5"
DIRECT_TFV_JOINT_DENSITY_TRAINING_CONTRACT = (
    "PROJECT7_DIRECT_TFV_JOINT_DENSITY_BALANCED_SHARED_REPRESENTATION_V1"
)


@dataclass(frozen=True)
class DirectTFVTrainingDesignV4:
    main_epochs: int = 6
    interaction_epochs: int = 4
    control_epochs: int = 2
    learning_rate: float = 5.0e-4
    interaction_learning_rate: float = 3.0e-4
    control_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    rank_weight: float = 0.35
    control_sign_weight: float = 0.25
    interaction_l1_weight: float = 0.01
    practical_zero_m3: float = 1.0
    grad_clip: float = 5.0
    seed: int = 42

    def validate(self) -> None:
        if min(self.main_epochs, self.interaction_epochs, self.control_epochs) <= 0:
            raise ValueError("Direct-TFV V5 training epochs must be positive")
        for value in (
            self.learning_rate,
            self.interaction_learning_rate,
            self.control_learning_rate,
            self.weight_decay,
            self.rank_weight,
            self.control_sign_weight,
            self.interaction_l1_weight,
            self.practical_zero_m3,
            self.grad_clip,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("Direct-TFV V5 training design contains an invalid value")
        if self.seed < 0:
            raise ValueError("training seed must be non-negative")


def _single_facility_ids(batch: Any, indices: torch.Tensor) -> torch.Tensor:
    candidates = batch.candidate_settings[0].index_select(0, indices)
    reference = batch.reference_settings[0][None]
    changed = torch.any(torch.abs(candidates - reference) > 1.0e-7, dim=1)
    counts = changed.sum(dim=1)
    if not bool(torch.all(counts == 1)):
        raise ValueError("facility-balanced MAIN received a non-single branch")
    return torch.argmax(changed.to(torch.int64), dim=1)


def _changed_facility_counts(batch: Any, indices: torch.Tensor) -> torch.Tensor:
    candidates = batch.candidate_settings[0].index_select(0, indices)
    reference = batch.reference_settings[0][None]
    changed = torch.any(torch.abs(candidates - reference) > 1.0e-7, dim=1)
    return changed.sum(dim=1).to(torch.int64)


def _balanced_smooth_l1(
    prediction_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: torch.Tensor,
    balance_ids: torch.Tensor | None,
) -> torch.Tensor:
    per_branch = F.smooth_l1_loss(
        prediction_m3 / scale_m3.clamp_min(1.0),
        truth_m3 / scale_m3.clamp_min(1.0),
        reduction="none",
    )
    if balance_ids is None:
        return per_branch.mean()
    means: list[torch.Tensor] = []
    for balance_id in torch.unique(balance_ids):
        mask = balance_ids == balance_id
        means.append(per_branch[mask].mean())
    return torch.stack(means).mean() if means else per_branch.mean()


def _facility_balanced_regression(
    prediction_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: torch.Tensor,
    facility_ids: torch.Tensor | None,
) -> torch.Tensor:
    return _balanced_smooth_l1(
        prediction_m3,
        truth_m3,
        scale_m3=scale_m3,
        balance_ids=facility_ids,
    )


def _joint_density_balanced_regression(
    prediction_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: torch.Tensor,
    changed_facility_counts: torch.Tensor,
) -> torch.Tensor:
    """Give every represented joint-action density equal regression influence within a group.

    This is not a performance weight. It prevents numerous sparse joint branches from drowning out
    the rarer dense coordinated branches that define the upper active-set support used by Step3.
    """

    if changed_facility_counts.shape != prediction_m3.shape:
        raise ValueError("joint changed-facility counts must match branch predictions")
    if bool(torch.any(changed_facility_counts <= 1)):
        raise ValueError("joint-density regression received a single/HOLD branch")
    return _balanced_smooth_l1(
        prediction_m3,
        truth_m3,
        scale_m3=scale_m3,
        balance_ids=changed_facility_counts,
    )


def _control_sign_loss(
    prediction_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: torch.Tensor,
    practical_zero_m3: float,
) -> torch.Tensor:
    """Symmetric D3 HOLD-relative sign loss; no extra safety bias or threshold is introduced."""
    informative = torch.abs(truth_m3) > float(practical_zero_m3)
    if not bool(informative.any()):
        return prediction_m3.new_zeros(())
    prediction = prediction_m3[informative]
    truth = truth_m3[informative]
    margin = (prediction / scale_m3.clamp_min(1.0)) * torch.sign(truth)
    return F.softplus(-margin).mean()


def _group_loss_v4(
    model: DirectFacilityTFVValueModel,
    batch: Any,
    *,
    mode: str,
    graph_tensors: Mapping[str, torch.Tensor],
    design: DirectTFVTrainingDesignV4,
    control_decision_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    indices = _branch_indices(batch, mode=mode)
    if int(indices.numel()) == 0:
        return None
    output = _forward_candidates(model, batch, indices, graph_tensors=graph_tensors)
    truth = batch.true_delta_tfv_m3[0].index_select(0, indices)
    scale = model.target_scale_m3.to(truth)
    facility_ids = _single_facility_ids(batch, indices) if mode == "single" else None
    changed_counts = _changed_facility_counts(batch, indices) if mode == "joint" else None
    if changed_counts is not None:
        regression = _joint_density_balanced_regression(
            output.total_delta_tfv_m3,
            truth,
            scale_m3=scale,
            changed_facility_counts=changed_counts,
        )
    else:
        regression = _facility_balanced_regression(
            output.total_delta_tfv_m3,
            truth,
            scale_m3=scale,
            facility_ids=facility_ids,
        )
    ranking = _pairwise_ranking_loss(
        _with_hold_zero(output.total_delta_tfv_m3),
        _with_hold_zero(truth),
        scale_m3=scale,
    )
    interaction_reg = torch.mean(torch.abs(output.interaction_residual_m3 / scale))
    control_sign = (
        _control_sign_loss(
            output.total_delta_tfv_m3,
            truth,
            scale_m3=scale,
            practical_zero_m3=design.practical_zero_m3,
        )
        if control_decision_loss
        else output.total_delta_tfv_m3.new_zeros(())
    )
    loss = regression + float(design.rank_weight) * ranking
    if mode == "joint":
        loss = loss + float(design.interaction_l1_weight) * interaction_reg
    if control_decision_loss:
        loss = loss + float(design.control_sign_weight) * control_sign
    return loss, {
        "loss": float(loss.detach().cpu()),
        "regression": float(regression.detach().cpu()),
        "ranking": float(ranking.detach().cpu()),
        "control_sign": float(control_sign.detach().cpu()),
        "interaction_l1": float(interaction_reg.detach().cpu()),
        "branches": float(indices.numel()),
        "joint_density_count": float(torch.unique(changed_counts).numel()) if changed_counts is not None else 0.0,
    }


def _set_trainable(model: DirectFacilityTFVValueModel, *, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "main":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    if stage == "joint":
        # V5 P1 fix: dense joint examples must adapt the shared facility representation used by
        # both main effects and the interaction residual; V4 updated only interaction_head.
        for module in (model.facility_encoder, model.facility_head, model.interaction_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return
    if stage == "control":
        for module in (model.facility_encoder, model.facility_head, model.interaction_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return
    raise ValueError(f"unknown Direct-TFV V5 stage: {stage}")


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
    design: DirectTFVTrainingDesignV4,
    control_decision_loss: bool,
) -> list[dict[str, float | int | str]]:
    _set_trainable(model, stage=stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"Direct-TFV V5 stage {stage!r} has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(design.weight_decay))
    static = _graph_tensors(graph, device)
    history: list[dict[str, float | int | str]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        totals: dict[str, list[float]] = {
            "loss": [], "regression": [], "ranking": [], "control_sign": [],
            "interaction_l1": [], "joint_density_count": [],
        }
        groups_used = branches_used = 0
        for source, name in _epoch_group_order(source_groups, epoch=epoch, seed=design.seed):
            batch = source_caches[source].batch(name, normalization, device)
            result = _group_loss_v4(
                model,
                batch,
                mode=mode,
                graph_tensors=static,
                design=design,
                control_decision_loss=control_decision_loss,
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
            raise RuntimeError(f"Direct-TFV V5 stage {stage!r} found no usable branches")
        history.append({
            "stage": stage,
            "epoch": epoch,
            "groups": int(groups_used),
            "branches": int(branches_used),
            **{key: float(np.mean(values)) for key, values in totals.items()},
        })
    return history


def train_direct_tfv_value_model_v4(
    model: DirectFacilityTFVValueModel,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: DirectTFVTrainingDesignV4 = DirectTFVTrainingDesignV4(),
) -> dict[str, list[dict[str, float | int | str]]]:
    """Train the V5 contract through the stable historical function name used by the current CLI."""
    design.validate()
    main = _train_stage(
        model, source_caches=source_caches, source_groups=source_groups, normalization=normalization,
        graph=graph, device=device, stage="main", mode="single", epochs=design.main_epochs,
        learning_rate=design.learning_rate, design=design, control_decision_loss=False,
    )
    joint = _train_stage(
        model, source_caches=source_caches, source_groups=source_groups, normalization=normalization,
        graph=graph, device=device, stage="joint", mode="joint", epochs=design.interaction_epochs,
        learning_rate=design.interaction_learning_rate, design=design, control_decision_loss=False,
    )
    if "D3" not in source_groups or not source_groups["D3"]:
        raise RuntimeError("Direct-TFV V5 CONTROL stage requires D3 HOLD-reference TrainFit groups")
    control = _train_stage(
        model, source_caches=source_caches, source_groups={"D3": source_groups["D3"]},
        normalization=normalization, graph=graph, device=device, stage="control", mode="all",
        epochs=design.control_epochs, learning_rate=design.control_learning_rate, design=design,
        control_decision_loss=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {"main": main, "joint": joint, "control": control}


__all__ = [
    "DIRECT_TFV_TRAINING_CONTRACT",
    "DIRECT_TFV_JOINT_DENSITY_TRAINING_CONTRACT",
    "DirectTFVTrainingDesignV4",
    "derive_direct_tfv_target_scale_m3",
    "evaluate_direct_tfv_value_model",
    "train_direct_tfv_value_model_v4",
    "_control_sign_loss",
    "_facility_balanced_regression",
    "_joint_density_balanced_regression",
]
