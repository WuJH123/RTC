"""Project7 V126 source-aware Step2 curriculum.

V125 proved that simply appending D4 groups to the generic D3 stage does not make the
Value model decision-reliable. V126 keeps the accepted causal V124 representation but
separates broad representation learning from the final decision task:

1. D2 + targeted D3 train the broad state/action representation using the frozen V124
   objective and rainfall-group split.
2. D4-FIT then fine-tunes the model on direct candidate-minus-Sparse-RBC-anchor TFV
   advantage with a local decision loss. D4-AUDIT is never fitted.

The D4 loss is deliberately decision-centric: physical magnitude, pairwise ordering,
benefit sign, listwise ordering and best-action margin. It does not re-introduce the
historical full-hydraulic multi-task route whose gradients conflicted with direct TFV.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v60 import PreparedStaticV60
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache

V126_CURRICULUM_CONTRACT = "PROJECT7_V126_SOURCE_AWARE_ANCHOR_ADVANTAGE_CURRICULUM_V1"


@dataclass(frozen=True)
class AnchorFineTuneContractV126:
    """One frozen first-pass D4 decision curriculum; no hyperparameter sweep."""

    epochs: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 5.0
    physical_weight: float = 1.0
    pairwise_magnitude_weight: float = 1.0
    pairwise_sign_weight: float = 0.50
    benefit_sign_weight: float = 1.0
    listwise_weight: float = 0.50
    best_action_weight: float = 0.50
    informative_effect_fraction: float = 0.02

    def validate(self) -> None:
        values = tuple(float(v) for v in self.__dict__.values() if not isinstance(v, int))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("V126 fine-tune contract contains non-finite values")
        if self.epochs <= 0 or self.learning_rate <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("V126 fine-tune schedule is invalid")
        if self.weight_decay < 0.0:
            raise ValueError("V126 weight decay cannot be negative")
        weights = (
            self.physical_weight,
            self.pairwise_magnitude_weight,
            self.pairwise_sign_weight,
            self.benefit_sign_weight,
            self.listwise_weight,
            self.best_action_weight,
        )
        if any(float(v) < 0.0 for v in weights):
            raise ValueError("V126 loss weights cannot be negative")
        if not 0.0 < float(self.informative_effect_fraction) < 1.0:
            raise ValueError("V126 informative effect fraction must lie in (0,1)")


def derive_anchor_tfv_scale_v126(cache: V60TrainCache, names: Sequence[str]) -> float:
    """Freeze a D4-local physical scale from FIT truth only."""
    values: list[float] = []
    for name in names:
        entry = cache.entry(name)
        arrays = entry.arrays
        ref = int(entry.reference_index)
        candidates = [int(i) for i in entry.indices if int(i) != ref]
        if not candidates:
            continue
        truth = np.asarray(arrays["exact_node_flood_volume_m3"], dtype=np.float64)
        ref_tfv = float(truth[ref].sum())
        candidate_tfv = truth[candidates].sum(axis=1)
        values.extend(np.abs(candidate_tfv - ref_tfv).tolist())
    absolute = np.asarray(values, dtype=np.float64)
    if absolute.size == 0 or not np.isfinite(absolute).all():
        raise ValueError("V126 cannot derive D4-local TFV scale from empty/non-finite FIT truth")
    positive = absolute[absolute > 1.0e-9]
    source = positive if positive.size else absolute
    return max(float(np.quantile(source, 0.75)), 100.0)


def anchor_advantage_loss_v126(
    predicted_m3: torch.Tensor,
    truth_m3: torch.Tensor,
    *,
    scale_m3: float,
    contract: AnchorFineTuneContractV126 = AnchorFineTuneContractV126(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Decision-centric loss for direct candidate-minus-anchor TFV advantage."""
    contract.validate()
    if predicted_m3.ndim != 2 or predicted_m3.shape != truth_m3.shape:
        raise ValueError("V126 advantage loss expects aligned [batch,candidate] tensors")
    if predicted_m3.shape[1] < 2:
        raise ValueError("V126 advantage loss requires at least two candidates per group")
    if not bool(torch.isfinite(predicted_m3).all()) or not bool(torch.isfinite(truth_m3).all()):
        raise ValueError("V126 advantage loss received non-finite values")
    scale = torch.as_tensor(
        max(float(scale_m3), 1.0), dtype=predicted_m3.dtype, device=predicted_m3.device
    )
    pred = predicted_m3 / scale
    truth = truth_m3.detach() / scale

    physical = F.smooth_l1_loss(pred, truth, beta=0.5)

    count = predicted_m3.shape[1]
    upper = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=predicted_m3.device), diagonal=1
    )[None].expand(predicted_m3.shape[0], -1, -1)
    truth_diff = truth[:, :, None] - truth[:, None, :]
    pred_diff = pred[:, :, None] - pred[:, None, :]
    informative = upper & (
        truth_diff.abs() >= float(contract.informative_effect_fraction)
    )
    if not bool(informative.any()):
        informative = upper
    pair_magnitude = F.smooth_l1_loss(
        pred_diff[informative], truth_diff[informative], beta=0.5
    )
    pair_sign = F.softplus(
        -torch.sign(truth_diff[informative]) * pred_diff[informative]
    ).mean()

    meaningful = truth.abs() >= float(contract.informative_effect_fraction)
    if not bool(meaningful.any()):
        meaningful = torch.ones_like(truth, dtype=torch.bool)
    benefit_target = (truth < 0.0).to(pred.dtype)
    # Lower TFV is better, so -pred acts as the benefit logit.
    benefit_sign = F.binary_cross_entropy_with_logits(
        -pred[meaningful], benefit_target[meaningful]
    )

    target_probability = torch.softmax(-truth, dim=1)
    listwise = -(
        target_probability * F.log_softmax(-pred, dim=1)
    ).sum(dim=1).mean()

    best_index = torch.argmin(truth, dim=1)
    best_pred = pred.gather(1, best_index[:, None])
    best_truth = truth.gather(1, best_index[:, None])
    true_gap = torch.clamp(truth - best_truth, min=0.0, max=1.0)
    best_mask = torch.ones_like(truth, dtype=torch.bool)
    best_mask.scatter_(1, best_index[:, None], False)
    # The predicted best should stay below alternatives in proportion to the true gap.
    best_action = torch.relu(
        best_pred - pred + 0.25 * true_gap
    )[best_mask].mean()

    total = (
        float(contract.physical_weight) * physical
        + float(contract.pairwise_magnitude_weight) * pair_magnitude
        + float(contract.pairwise_sign_weight) * pair_sign
        + float(contract.benefit_sign_weight) * benefit_sign
        + float(contract.listwise_weight) * listwise
        + float(contract.best_action_weight) * best_action
    )
    with torch.no_grad():
        benefit_accuracy = (
            (predicted_m3[meaningful] < 0.0) == (truth_m3[meaningful] < 0.0)
        ).float().mean()
    return total, {
        "loss_v126": float(total.detach()),
        "physical": float(physical.detach()),
        "pairwise_magnitude": float(pair_magnitude.detach()),
        "pairwise_sign": float(pair_sign.detach()),
        "benefit_sign": float(benefit_sign.detach()),
        "benefit_accuracy": float(benefit_accuracy.detach()),
        "listwise": float(listwise.detach()),
        "best_action": float(best_action.detach()),
    }


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        entry = cache.entry(name)
        key = f"{entry.rainfall_group}::{entry.event_id}"
        grouped.setdefault(key, []).append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def train_anchor_advantage_finetune_v126(
    model: torch.nn.Module,
    cache: V60TrainCache,
    *,
    fit_d4_names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    device: str | torch.device,
    scale_m3: float,
    seed: int = 42,
    contract: AnchorFineTuneContractV126 = AnchorFineTuneContractV126(),
) -> list[dict[str, Any]]:
    """Fine-tune only on D4-FIT after broad D2/D3 pretraining."""
    contract.validate()
    if int(seed) != 42:
        raise ValueError("V126 first curriculum run is frozen at seed 42")
    if not fit_d4_names:
        raise ValueError("V126 requires non-empty D4-FIT groups")
    target = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model.to(target).float().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract.learning_rate),
        weight_decay=float(contract.weight_decay),
    )
    events = _events(cache, fit_d4_names)
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(contract.epochs) + 1):
        keys = list(events)
        rng.shuffle(keys)
        records: list[dict[str, float]] = []
        for key in keys:
            optimizer.zero_grad(set_to_none=True)
            names = events[key]
            for name in names:
                batch = cache.batch(name, normalization, target)
                output = model(
                    batch.initial_state,
                    batch.rainfall,
                    batch.reference_settings,
                    batch.candidate_settings,
                    batch.previous_actuator_flow,
                    prepared,
                )
                loss, metrics = anchor_advantage_loss_v126(
                    output.delta_tfv_m3,
                    batch.true_delta_tfv_m3,
                    scale_m3=scale_m3,
                    contract=contract,
                )
                (loss / len(names)).backward()
                records.append(metrics)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(contract.grad_clip))
            optimizer.step()
        row: dict[str, Any] = {"stage": "D4_anchor_advantage_finetune", "epoch": epoch}
        for metric in sorted({name for record in records for name in record}):
            values = [float(record[metric]) for record in records]
            row[metric] = float(np.mean(values)) if values else float("nan")
        history.append(row)
        fields = " ".join(
            f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items()
        )
        print(f"[V126_VALUE] {fields}", flush=True)
    return history


__all__ = [
    "AnchorFineTuneContractV126",
    "V126_CURRICULUM_CONTRACT",
    "anchor_advantage_loss_v126",
    "derive_anchor_tfv_scale_v126",
    "train_anchor_advantage_finetune_v126",
]
