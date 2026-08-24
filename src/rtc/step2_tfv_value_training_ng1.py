"""Training utilities for the isolated NG1-R3 Step2 Development candidate.

R3 deliberately restores the R1 loss/weighting semantics so that the only
scientific change relative to R1 is the adaptive-connectivity interaction
representation.  This avoids confounding representation changes with R2's
selection-temperature, best-margin, and global-density revisions.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_tfv_value_ng1 import (
    NG1ProcessAwareDirectTFVValueModel,
    d2_magnitude_strata,
    d2_magnitude_weights,
)
from .step2_tfv_value_training import (
    _branch_indices,
    _epoch_group_order,
    _forward_candidates,
    _graph_tensors,
    _pairwise_ranking_loss,
    _with_hold_zero,
)
from .step2_tfv_value_training_v4 import _single_facility_ids


NG1_TRAINING_CONTRACT = "PROJECT7_STEP2_NG1_ADAPTIVE_CONNECTIVITY_R1_LOSS_TRAINING_V3"


@dataclass(frozen=True)
class NG1TrainingDesign:
    main_epochs: int = 8
    interaction_epochs: int = 5
    control_epochs: int = 3
    learning_rate: float = 5.0e-4
    interaction_learning_rate: float = 5.0e-4
    control_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    rank_weight: float = 0.35
    sign_weight: float = 0.25
    regret_weight: float = 0.25
    interaction_l1_weight: float = 0.01
    grad_clip: float = 5.0
    practical_zero_m3: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if min(self.main_epochs, self.interaction_epochs, self.control_epochs) <= 0:
            raise ValueError("NG1 epochs must be positive")
        for value in (
            self.learning_rate,
            self.interaction_learning_rate,
            self.control_learning_rate,
            self.weight_decay,
            self.rank_weight,
            self.sign_weight,
            self.regret_weight,
            self.interaction_l1_weight,
            self.grad_clip,
            self.practical_zero_m3,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("NG1 training design contains an invalid value")
        if self.seed < 0:
            raise ValueError("NG1 seed must be non-negative")


def _sha_update(digest: hashlib._Hash, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))


def ng1_main_parameter_sha256(model: NG1ProcessAwareDirectTFVValueModel) -> str:
    digest = hashlib.sha256()
    names = set(model.main_parameter_names())
    for name, value in sorted(model.named_parameters()):
        if name in names:
            _sha_update(digest, name, value)
    return digest.hexdigest()


def ng1_interaction_parameter_sha256(model: NG1ProcessAwareDirectTFVValueModel) -> str:
    digest = hashlib.sha256()
    names = set(model.interaction_parameter_names())
    for name, value in sorted(model.named_parameters()):
        if name in names:
            _sha_update(digest, name, value)
    return digest.hexdigest()


def freeze_ng1_main(model: NG1ProcessAwareDirectTFVValueModel) -> None:
    interaction = set(model.interaction_parameter_names())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in interaction)


def _sign_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    scale: torch.Tensor,
    practical_zero: float,
) -> torch.Tensor:
    informative = torch.abs(truth) > float(practical_zero)
    if not bool(informative.any()):
        return prediction.new_zeros(())
    margin = prediction[informative] / scale.clamp_min(1.0) * torch.sign(truth[informative])
    return F.softplus(-margin).mean()


def _soft_selection_regret(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    tau_m3: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(-prediction / tau_m3.clamp_min(1.0), dim=0)
    regret = truth - truth.min()
    return (probabilities * regret / scale.clamp_min(1.0)).sum()


def ng1_d2_group_loss(
    model: NG1ProcessAwareDirectTFVValueModel,
    batch: Any,
    *,
    indices: torch.Tensor,
    graph_tensors: Mapping[str, torch.Tensor],
    scale_m3: torch.Tensor,
    branch_weights: torch.Tensor | None = None,
    rank_weight: float = 0.35,
    sign_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = _forward_candidates(model, batch, indices, graph_tensors=graph_tensors)
    truth = batch.true_delta_tfv_m3[0].index_select(0, indices)
    facility_ids = _single_facility_ids(batch, indices)
    per_branch = F.smooth_l1_loss(
        output.total_delta_tfv_m3 / scale_m3.clamp_min(1.0),
        truth / scale_m3.clamp_min(1.0),
        reduction="none",
    )
    if branch_weights is not None:
        per_branch = per_branch * branch_weights.to(per_branch)
    facility_losses = [
        per_branch[facility_ids == value].mean()
        for value in torch.unique(facility_ids)
    ]
    regression = torch.stack(facility_losses).mean()
    ranking = _pairwise_ranking_loss(
        _with_hold_zero(output.total_delta_tfv_m3),
        _with_hold_zero(truth),
        scale_m3=scale_m3,
    )
    sign = _sign_loss(output.total_delta_tfv_m3, truth, scale_m3, 1.0)
    loss = regression + float(rank_weight) * ranking + float(sign_weight) * sign
    return loss, {
        "loss": float(loss.detach().cpu()),
        "regression": float(regression.detach().cpu()),
        "ranking": float(ranking.detach().cpu()),
        "sign": float(sign.detach().cpu()),
        "branches": float(indices.numel()),
    }


def ng1_d3_group_loss(
    model: NG1ProcessAwareDirectTFVValueModel,
    batch: Any,
    *,
    indices: torch.Tensor,
    graph_tensors: Mapping[str, torch.Tensor],
    scale_m3: torch.Tensor,
    tau_m3: torch.Tensor,
    branch_weights: torch.Tensor | None = None,
    rank_weight: float = 0.35,
    sign_weight: float = 0.25,
    regret_weight: float = 0.25,
    interaction_l1_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = _forward_candidates(model, batch, indices, graph_tensors=graph_tensors)
    truth_candidates = batch.true_delta_tfv_m3[0].index_select(0, indices)
    scale = scale_m3.clamp_min(1.0)
    final_prediction = output.total_delta_tfv_m3
    residual_target = truth_candidates - output.facility_main_effect_m3.sum(dim=-1).detach()
    residual_regression = F.smooth_l1_loss(
        output.interaction_residual_m3 / scale,
        residual_target / scale,
        reduction="none",
    )
    final_regression = F.smooth_l1_loss(
        final_prediction / scale,
        truth_candidates / scale,
        reduction="none",
    )
    if branch_weights is not None:
        weights = branch_weights.to(final_regression)
        final_regression = final_regression * weights
        residual_regression = residual_regression * weights
    regression = final_regression.mean() + residual_regression.mean()
    prediction = _with_hold_zero(final_prediction)
    truth = _with_hold_zero(truth_candidates)
    ranking = _pairwise_ranking_loss(prediction, truth, scale_m3=scale_m3)
    sign = _sign_loss(final_prediction, truth_candidates, scale_m3, 1.0)
    regret = _soft_selection_regret(prediction, truth, tau_m3, scale_m3)
    interaction_l1 = torch.mean(torch.abs(output.interaction_residual_m3 / scale))
    loss = (
        regression
        + float(rank_weight) * ranking
        + float(sign_weight) * sign
        + float(regret_weight) * regret
        + float(interaction_l1_weight) * interaction_l1
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "regression": float(regression.detach().cpu()),
        "residual_regression": float(residual_regression.mean().detach().cpu()),
        "ranking": float(ranking.detach().cpu()),
        "sign": float(sign.detach().cpu()),
        "selection_regret": float(regret.detach().cpu()),
        "interaction_l1": float(interaction_l1.detach().cpu()),
        "branches": float(indices.numel()),
    }


def _extract_values(cache: Any, names: Sequence[str]) -> np.ndarray:
    values: list[float] = []
    for name in names:
        entry = cache.entry(name)
        ref = int(entry.reference_index)
        candidates = [int(index) for index in entry.indices if int(index) != ref]
        reference = float(
            np.asarray(entry.arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum()
        )
        candidate = np.asarray(
            entry.arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64
        ).sum(axis=1)
        values.extend((candidate - reference).tolist())
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("NG1 cannot derive labels from empty/non-finite cache")
    return result


def _density_contract(cache: Any, names: Sequence[str]) -> dict[str, Any]:
    counts: list[int] = []
    for name in names:
        entry = cache.entry(name)
        ref = int(entry.reference_index)
        reference = np.asarray(entry.arrays["settings"][ref], dtype=np.float64)
        for index in entry.indices:
            if int(index) == ref:
                continue
            candidate = np.asarray(entry.arrays["settings"][int(index)], dtype=np.float64)
            counts.append(
                int(np.any(np.abs(candidate - reference) > 1.0e-7, axis=0).sum())
            )
    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0:
        raise ValueError("NG1 D3 density contract requires non-empty candidates")
    quantiles = [float(np.quantile(values, q)) for q in (0.50, 0.75, 0.90)]
    return {
        "contract": "PROJECT7_STEP2_NG1_D3_DENSITY_STRATA_TRAIN_FIT_V1",
        "q50": quantiles[0],
        "q75": quantiles[1],
        "q90": quantiles[2],
        "count": int(values.size),
    }


def _density_weights(counts: torch.Tensor, contract: Mapping[str, Any]) -> torch.Tensor:
    q50, q75, q90 = (
        float(contract[key]) for key in ("q50", "q75", "q90")
    )
    bins = torch.where(
        counts.to(torch.float32) <= q50,
        0,
        torch.where(counts <= q75, 1, torch.where(counts <= q90, 2, 3)),
    )
    weights = torch.ones_like(counts, dtype=torch.float32)
    for value in torch.unique(bins):
        mask = bins == value
        weights[mask] = 1.0 / float(mask.sum())
    return weights * (float(weights.numel()) / weights.sum().clamp_min(1.0e-12))


def _batch_weights(
    batch: Any, indices: torch.Tensor, contract: Mapping[str, Any]
) -> torch.Tensor:
    candidate = batch.candidate_settings[0].index_select(0, indices)
    reference = batch.reference_settings[0][None]
    counts = torch.any(
        torch.abs(candidate - reference) > 1.0e-7, dim=1
    ).sum(dim=1)
    return _density_weights(counts, contract)


def _run_stage(
    model: NG1ProcessAwareDirectTFVValueModel,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: Any,
    graph: Any,
    device: torch.device,
    stage: str,
    epochs: int,
    learning_rate: float,
    design: NG1TrainingDesign,
    scale_m3: torch.Tensor,
    tau_m3: torch.Tensor,
    density_contract: Mapping[str, Any],
    strata: Mapping[str, Any] | None = None,
) -> list[dict[str, float | int | str]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"NG1 {stage} stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(design.weight_decay),
    )
    static = _graph_tensors(graph, device)
    history: list[dict[str, float | int | str]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        records: list[dict[str, float]] = []
        for name in _epoch_group_order(
            {"D2" if stage == "main" else "D3": names},
            epoch=epoch,
            seed=design.seed,
        ):
            _, group_name = name
            batch = cache.batch(group_name, normalization, device)
            mode = "single" if stage == "main" else (
                "joint" if stage == "joint" else "all"
            )
            indices = _branch_indices(batch, mode=mode)
            if int(indices.numel()) == 0:
                continue
            if stage == "main":
                truth = (
                    batch.true_delta_tfv_m3[0]
                    .index_select(0, indices)
                    .detach()
                    .cpu()
                    .numpy()
                )
                weights = d2_magnitude_weights(truth, strata or {}).copy()
                result = ng1_d2_group_loss(
                    model,
                    batch,
                    indices=indices,
                    graph_tensors=static,
                    scale_m3=scale_m3,
                    branch_weights=torch.as_tensor(weights, device=device),
                    rank_weight=design.rank_weight,
                    sign_weight=design.sign_weight,
                )
            else:
                weights = _batch_weights(batch, indices, density_contract)
                result = ng1_d3_group_loss(
                    model,
                    batch,
                    indices=indices,
                    graph_tensors=static,
                    scale_m3=scale_m3,
                    tau_m3=tau_m3,
                    branch_weights=weights,
                    rank_weight=design.rank_weight,
                    sign_weight=design.sign_weight,
                    regret_weight=design.regret_weight,
                    interaction_l1_weight=design.interaction_l1_weight,
                )
            loss, metrics = result
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(design.grad_clip))
            optimizer.step()
            records.append(metrics)
        if not records:
            raise RuntimeError(f"NG1 {stage} stage found no usable branches")
        keys = sorted(records[0])
        history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "groups": len(records),
                **{
                    key: float(np.mean([row[key] for row in records]))
                    for key in keys
                },
            }
        )
    return history


def train_ng1_model(
    model: NG1ProcessAwareDirectTFVValueModel,
    *,
    source_cache: Any,
    fit_d2_names: Sequence[str],
    fit_d3_names: Sequence[str],
    normalization: Any,
    graph: Any,
    device: torch.device,
    target_scale_m3: float,
    design: NG1TrainingDesign = NG1TrainingDesign(),
) -> dict[str, Any]:
    design.validate()
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("NG1 requires non-empty D2 and D3 TrainFit groups")
    d2_values = _extract_values(source_cache, fit_d2_names)
    strata = d2_magnitude_strata(d2_values)
    density = _density_contract(source_cache, fit_d3_names)
    tau = max(
        1.0,
        float(np.quantile(np.abs(_extract_values(source_cache, fit_d3_names)), 0.75)),
    )
    scale = torch.as_tensor(
        float(target_scale_m3), dtype=torch.float32, device=device
    )
    tau_tensor = torch.as_tensor(tau, dtype=torch.float32, device=device)

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    main_history = _run_stage(
        model,
        cache=source_cache,
        names=fit_d2_names,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="main",
        epochs=design.main_epochs,
        learning_rate=design.learning_rate,
        design=design,
        scale_m3=scale,
        tau_m3=tau_tensor,
        density_contract=density,
        strata=strata,
    )
    main_state_dict = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    main_hash_before_d3 = ng1_main_parameter_sha256(model)
    freeze_ng1_main(model)
    joint_history = _run_stage(
        model,
        cache=source_cache,
        names=fit_d3_names,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="joint",
        epochs=design.interaction_epochs,
        learning_rate=design.interaction_learning_rate,
        design=design,
        scale_m3=scale,
        tau_m3=tau_tensor,
        density_contract=density,
    )
    control_history = _run_stage(
        model,
        cache=source_cache,
        names=fit_d3_names,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="control",
        epochs=design.control_epochs,
        learning_rate=design.control_learning_rate,
        design=design,
        scale_m3=scale,
        tau_m3=tau_tensor,
        density_contract=density,
    )
    main_hash_after_d3 = ng1_main_parameter_sha256(model)
    if main_hash_before_d3 != main_hash_after_d3:
        raise RuntimeError("NG1 main parameters changed during D3 interaction training")
    return {
        "main": main_history,
        "main_state_dict": main_state_dict,
        "joint": joint_history,
        "control": control_history,
        "d2_magnitude_strata": {
            key: value for key, value in strata.items() if key != "masks"
        },
        "d3_density_strata": density,
        "d3_tau_m3": tau,
        "d3_tau_definition": "TrainFit q75 absolute delta TFV (R1 semantics)",
        "main_parameter_sha256_before_d3": main_hash_before_d3,
        "main_parameter_sha256_after_d3": main_hash_after_d3,
        "main_frozen_after_main": True,
    }


__all__ = [
    "NG1_TRAINING_CONTRACT",
    "NG1TrainingDesign",
    "freeze_ng1_main",
    "ng1_d2_group_loss",
    "ng1_d3_group_loss",
    "ng1_interaction_parameter_sha256",
    "ng1_main_parameter_sha256",
    "train_ng1_model",
]
