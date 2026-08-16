"""Training utilities for the direct Project7 facility-action -> delta-TFV model.

The curriculum mirrors the information content of the existing SWMM data instead of the old
hydraulic-surrogate implementation stages:

1. MAIN: exact same-prefix single-actuator branches teach the 109 facility main-value differences.
2. JOINT: main/context parameters are frozen and multi-actuator branches teach only the residual
   interaction value difference.

The authoritative target is always exact SWMM delta TFV. The historical 1%-of-reference
"meaningful" threshold is a reporting convention and is intentionally not used to erase or
binarize continuous training labels. HOLD/reference is an explicit zero-delta option in ranking
losses and selection metrics because real-time control must learn when doing nothing is better.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_tfv_value import DirectFacilityTFVValueModel
from .step2_train_response_v60 import InputNormalizationV60


DIRECT_TFV_TRAINING_CONTRACT = "PROJECT7_DIRECT_TFV_MAIN_THEN_INTERACTION_TRAINING_V2"


@dataclass(frozen=True)
class DirectTFVTrainingDesign:
    main_epochs: int = 6
    interaction_epochs: int = 4
    learning_rate: float = 5.0e-4
    interaction_learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-5
    rank_weight: float = 0.35
    interaction_l1_weight: float = 0.01
    grad_clip: float = 5.0
    seed: int = 42

    def validate(self) -> None:
        if self.main_epochs <= 0 or self.interaction_epochs <= 0:
            raise ValueError("direct TFV training epochs must be positive")
        for value in (
            self.learning_rate,
            self.interaction_learning_rate,
            self.weight_decay,
            self.rank_weight,
            self.interaction_l1_weight,
            self.grad_clip,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("direct TFV training design contains an invalid value")
        if self.seed < 0:
            raise ValueError("direct TFV training seed must be non-negative")


def _stable_rng(text: str, seed: int) -> np.random.Generator:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng((value + int(seed)) % (2**32))


def _candidate_change_counts(batch: Any) -> torch.Tensor:
    reference = batch.reference_settings[:, None]
    delta = torch.abs(batch.candidate_settings - reference)
    return torch.any(delta > 1.0e-7, dim=2).sum(dim=-1)[0]


def _branch_indices(batch: Any, *, mode: str) -> torch.Tensor:
    counts = _candidate_change_counts(batch)
    if mode == "single":
        mask = counts == 1
    elif mode == "joint":
        mask = counts > 1
    elif mode == "all":
        mask = counts > 0
    else:
        raise ValueError(f"unknown direct TFV branch mode: {mode}")
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _graph_tensors(graph: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "up": torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        "down": torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        "physics": torch.as_tensor(graph.actuator_physics, dtype=torch.float32, device=device),
    }


def _forward_candidates(
    model: DirectFacilityTFVValueModel,
    batch: Any,
    indices: torch.Tensor,
    *,
    graph_tensors: Mapping[str, torch.Tensor],
) -> Any:
    count = int(indices.numel())
    if count <= 0:
        raise ValueError("cannot score an empty direct TFV candidate subset")
    candidates = batch.candidate_settings[0].index_select(0, indices)
    return model(
        current_state=batch.initial_state.expand(count, -1, -1),
        rainfall=batch.rainfall.expand(count, -1, -1, -1),
        reference_settings=batch.reference_settings.expand(count, -1, -1),
        candidate_settings=candidates,
        previous_actuator_flow=batch.previous_actuator_flow.expand(count, -1),
        actuator_upstream=graph_tensors["up"],
        actuator_downstream=graph_tensors["down"],
        actuator_physics=graph_tensors["physics"],
    )


def _pairwise_ranking_loss(
    prediction_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: torch.Tensor,
) -> torch.Tensor:
    if prediction_m3.numel() < 2:
        return prediction_m3.new_zeros(())
    p = prediction_m3[:, None] - prediction_m3[None, :]
    y = truth_m3[:, None] - truth_m3[None, :]
    upper = torch.triu(torch.ones_like(y, dtype=torch.bool), diagonal=1)
    informative = upper & (torch.abs(y) > 1.0)
    if not bool(informative.any()):
        return prediction_m3.new_zeros(())
    sign = torch.sign(y[informative])
    margin = (p[informative] / scale_m3.clamp_min(1.0)) * sign
    return F.softplus(-margin).mean()


def _with_hold_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value.new_zeros((1,)), value.reshape(-1)))


def _group_loss(
    model: DirectFacilityTFVValueModel,
    batch: Any,
    *,
    mode: str,
    graph_tensors: Mapping[str, torch.Tensor],
    rank_weight: float,
    interaction_l1_weight: float,
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
    interaction_reg = torch.mean(torch.abs(output.interaction_residual_m3 / scale))
    loss = regression + float(rank_weight) * ranking
    if mode == "joint":
        loss = loss + float(interaction_l1_weight) * interaction_reg
    return loss, {
        "loss": float(loss.detach().cpu()),
        "regression": float(regression.detach().cpu()),
        "ranking": float(ranking.detach().cpu()),
        "branches": float(indices.numel()),
    }


def derive_direct_tfv_target_scale_m3(
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
) -> float:
    """Derive a robust target scale from TrainFit exact delta-TFV labels only."""
    values: list[float] = []
    for source, names in source_groups.items():
        cache = source_caches[source]
        for name in names:
            entry = cache.entry(name)
            arrays = entry.arrays
            ref = int(entry.reference_index)
            candidates = [int(i) for i in entry.indices if int(i) != ref]
            if not candidates:
                continue
            reference = float(
                np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum()
            )
            candidate = np.asarray(
                arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64
            ).sum(axis=1)
            delta = np.abs(candidate - reference)
            values.extend(delta[np.isfinite(delta)].tolist())
    if not values:
        raise ValueError("cannot derive direct TFV target scale from empty TrainFit labels")
    absolute = np.asarray(values, dtype=np.float64)
    positive = absolute[absolute > 1.0e-6]
    if positive.size == 0:
        raise ValueError("TrainFit exact delta TFV is identically zero")
    return max(100.0, float(np.quantile(positive, 0.75)))


def _set_trainable(model: DirectFacilityTFVValueModel, *, interaction_only: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(not interaction_only)
    if interaction_only:
        for parameter in model.interaction_head.parameters():
            parameter.requires_grad_(True)


def _epoch_group_order(
    source_groups: Mapping[str, Sequence[str]], *, epoch: int, seed: int
) -> list[tuple[str, str]]:
    """Interleave sources so the numerous D2 branches do not erase D3/D4 diversity."""
    shuffled: dict[str, list[str]] = {}
    for source, names in source_groups.items():
        order = list(names)
        rng = _stable_rng(f"{source}|{epoch}", seed)
        if order:
            order = [order[int(i)] for i in rng.permutation(len(order))]
        shuffled[source] = order
    result: list[tuple[str, str]] = []
    maximum = max((len(names) for names in shuffled.values()), default=0)
    for index in range(maximum):
        for source in sorted(shuffled):
            names = shuffled[source]
            if index < len(names):
                result.append((source, names[index]))
    return result


def _train_stage(
    model: DirectFacilityTFVValueModel,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    mode: str,
    epochs: int,
    learning_rate: float,
    design: DirectTFVTrainingDesign,
) -> list[dict[str, float | int | str]]:
    interaction_only = mode == "joint"
    _set_trainable(model, interaction_only=interaction_only)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(design.weight_decay),
    )
    static = _graph_tensors(graph, device)
    history: list[dict[str, float | int | str]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        losses: list[float] = []
        regressions: list[float] = []
        rankings: list[float] = []
        groups_used = branches_used = 0
        for source, name in _epoch_group_order(source_groups, epoch=epoch, seed=design.seed):
            batch = source_caches[source].batch(name, normalization, device)
            result = _group_loss(
                model,
                batch,
                mode=mode,
                graph_tensors=static,
                rank_weight=design.rank_weight,
                interaction_l1_weight=design.interaction_l1_weight,
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
            losses.append(metrics["loss"])
            regressions.append(metrics["regression"])
            rankings.append(metrics["ranking"])
        if not losses:
            raise RuntimeError(f"direct TFV {mode} stage found no usable branches")
        history.append(
            {
                "stage": mode,
                "epoch": epoch,
                "groups": groups_used,
                "branches": branches_used,
                "loss": float(np.mean(losses)),
                "regression": float(np.mean(regressions)),
                "ranking": float(np.mean(rankings)),
            }
        )
    return history


def train_direct_tfv_value_model(
    model: DirectFacilityTFVValueModel,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: DirectTFVTrainingDesign = DirectTFVTrainingDesign(),
) -> dict[str, list[dict[str, float | int | str]]]:
    design.validate()
    main = _train_stage(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
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
        mode="joint",
        epochs=design.interaction_epochs,
        learning_rate=design.interaction_learning_rate,
        design=design,
    )
    _set_trainable(model, interaction_only=False)
    return {"main": main, "joint": joint}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    a, b = _rankdata(left), _rankdata(right)
    if float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _pairwise_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    hit = total = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            difference = float(truth[i] - truth[j])
            if abs(difference) <= 1.0:
                continue
            total += 1
            hit += int(np.sign(difference) == np.sign(float(prediction[i] - prediction[j])))
    return float(hit / total) if total else float("nan")


def evaluate_direct_tfv_value_model(
    model: DirectFacilityTFVValueModel,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate ranking *and* whether the selected action is actually better than HOLD."""
    static = _graph_tensors(graph, device)
    ranks: list[float] = []
    pairwise: list[float] = []
    mae: list[float] = []
    sign: list[float] = []
    regrets: list[float] = []
    selected_truth: list[float] = []
    top1 = hold_selected = groups = branches = 0
    oracle_hold = false_action_when_hold = selected_beneficial = selected_harmful = 0
    model.eval()
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, device)
            indices = _branch_indices(batch, mode="all")
            if int(indices.numel()) == 0:
                continue
            output = _forward_candidates(model, batch, indices, graph_tensors=static)
            candidate_truth = batch.true_delta_tfv_m3[0].index_select(0, indices).detach().cpu().numpy()
            candidate_prediction = output.total_delta_tfv_m3.detach().cpu().numpy()
            truth = np.concatenate((np.zeros(1, dtype=np.float64), candidate_truth.astype(np.float64)))
            prediction = np.concatenate(
                (np.zeros(1, dtype=np.float64), candidate_prediction.astype(np.float64))
            )
            groups += 1
            branches += len(candidate_truth)
            ranks.append(_spearman(truth, prediction))
            pairwise.append(_pairwise_accuracy(truth, prediction))
            mae.append(float(np.mean(np.abs(candidate_prediction - candidate_truth))))
            informative = np.abs(candidate_truth) > 1.0
            if np.any(informative):
                sign.append(
                    float(
                        np.mean(
                            np.sign(candidate_prediction[informative])
                            == np.sign(candidate_truth[informative])
                        )
                    )
                )
            selected = int(np.argmin(prediction))
            oracle = int(np.argmin(truth))
            selected_delta = float(truth[selected])
            hold_selected += int(selected == 0)
            oracle_hold += int(oracle == 0)
            false_action_when_hold += int(oracle == 0 and selected != 0)
            selected_beneficial += int(selected_delta < -1.0)
            selected_harmful += int(selected_delta > 1.0)
            top1 += int(selected == oracle)
            selected_truth.append(selected_delta)
            regrets.append(float(selected_delta - truth[oracle]))

    def finite_mean(values: Sequence[float]) -> float:
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    if groups <= 0:
        raise ValueError("direct TFV evaluation has no usable groups")
    return {
        "groups": groups,
        "branches": branches,
        "rank": finite_mean(ranks),
        "pairwise": finite_mean(pairwise),
        "sign": finite_mean(sign),
        "top1_fraction": float(top1 / groups),
        "hold_selected_fraction": float(hold_selected / groups),
        "oracle_hold_groups": int(oracle_hold),
        "false_action_when_hold_oracle_count": int(false_action_when_hold),
        "false_action_when_hold_oracle_fraction": (
            float(false_action_when_hold / oracle_hold) if oracle_hold else 0.0
        ),
        "selected_beneficial_fraction": float(selected_beneficial / groups),
        "selected_harmful_fraction": float(selected_harmful / groups),
        "selected_true_delta_tfv_m3": finite_mean(selected_truth),
        "delta_tfv_mae_m3": finite_mean(mae),
        "selected_regret_m3": finite_mean(regrets),
    }


__all__ = [
    "DIRECT_TFV_TRAINING_CONTRACT",
    "DirectTFVTrainingDesign",
    "derive_direct_tfv_target_scale_m3",
    "evaluate_direct_tfv_value_model",
    "train_direct_tfv_value_model",
]
