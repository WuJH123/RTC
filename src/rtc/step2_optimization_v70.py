"""Canonical event-balanced optimizers for Project7 Step2 V7.0."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v60 import prepare_static_v60
from .step2_control_response_v70 import (
    ControlValueSurrogateV70,
    HydraulicResponseSurrogateV70,
)
from .step2_hydraulic_effect_v70 import hydraulic_effect_loss_v70
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70, value_loss_v70
from .step2_v70_contract import (
    DirectValueLossContractV70,
    HydraulicEffectLossContractV70,
)


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        entry = cache.entry(name)
        grouped[f"{entry.rainfall_group}::{entry.event_id}"].append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def train_value_event_balanced_v70(
    model: ControlValueSurrogateV70,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    fit_d3_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    graph: Any,
    device: str = "cuda",
    seed: int = 42,
    contract: DirectValueLossContractV70 = DirectValueLossContractV70(),
) -> list[dict[str, Any]]:
    """D2 warm start, then D3-primary event-balanced direct objective learning."""
    contract.validate()
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("V7 value training requires D2 and targeted D3")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(target).float().train()
    prepared = prepare_static_v60(graph, target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )
    rng = np.random.default_rng(seed)
    d2_events = _events(cache, fit_d2_names)
    d3_events = _events(cache, fit_d3_names)
    history: list[dict[str, Any]] = []

    def backward_group(name: str, weight: float) -> dict[str, float]:
        batch = cache.batch(name, normalization, target)
        output = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings,
            batch.previous_actuator_flow,
            prepared,
        )
        loss, metrics = value_loss_v70(
            output,
            batch.true_delta_tfv_m3,
            scale_m3=scales.direct_tfv_scale_m3,
            contract=contract,
        )
        (weight * loss).backward()
        return metrics

    def summarize(stage: str, epoch: int, records: list[dict[str, float]]) -> dict[str, Any]:
        keys = sorted({k for record in records for k in record})
        result: dict[str, Any] = {"stage": stage, "epoch": int(epoch)}
        for key in keys:
            values = [float(record[key]) for record in records if np.isfinite(float(record[key]))]
            result[key] = float(np.mean(values)) if values else float("nan")
        return result

    for epoch in range(1, contract.d2_pretrain_epochs + 1):
        keys = list(d2_events)
        rng.shuffle(keys)
        records: list[dict[str, float]] = []
        for key in keys:
            optimizer.zero_grad(set_to_none=True)
            groups = d2_events[key]
            records.extend(
                backward_group(name, 1.0 / len(groups)) for name in groups
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), contract.grad_clip)
            optimizer.step()
        history.append(summarize("D2_direct_sensitivity", epoch, records))

    d2_keys = list(d2_events)
    for epoch in range(1, contract.joint_epochs + 1):
        d3_keys = list(d3_events)
        rng.shuffle(d3_keys)
        rng.shuffle(d2_keys)
        records = []
        for index, key in enumerate(d3_keys):
            optimizer.zero_grad(set_to_none=True)
            d3_groups = d3_events[key]
            d2_groups = d2_events[d2_keys[index % len(d2_keys)]]
            records.extend(
                backward_group(
                    name, contract.joint_d3_weight / len(d3_groups)
                )
                for name in d3_groups
            )
            records.extend(
                backward_group(
                    name, contract.joint_d2_weight / len(d2_groups)
                )
                for name in d2_groups
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), contract.grad_clip)
            optimizer.step()
        history.append(summarize("D3_primary_with_D2_anchor", epoch, records))
    return history


def train_hydraulic_event_balanced_v70(
    model: HydraulicResponseSurrogateV70,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    fit_d3_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    onset_positive_weight: float,
    graph: Any,
    device: str = "cuda",
    seed: int = 42,
    contract: HydraulicEffectLossContractV70 = HydraulicEffectLossContractV70(),
) -> list[dict[str, Any]]:
    """D3-primary hydraulic effect learning with D2 anchor events."""
    contract.validate()
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("V7 hydraulic training requires D2 and targeted D3")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + 17)
    np.random.seed(seed + 17)
    model.to(target).float().train()
    prepared = prepare_static_v60(graph, target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )
    rng = np.random.default_rng(seed + 17)
    d2_events = _events(cache, fit_d2_names)
    d3_events = _events(cache, fit_d3_names)
    d2_keys = list(d2_events)
    history: list[dict[str, Any]] = []

    def backward_group(name: str, weight: float) -> dict[str, float]:
        batch = cache.batch(name, normalization, target)
        output = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings,
            prepared,
        )
        loss, metrics = hydraulic_effect_loss_v70(
            output,
            batch,
            prepared,
            normalization,
            scales,
            onset_positive_weight=onset_positive_weight,
            horizon=model.horizon_contract,
            contract=contract,
        )
        (weight * loss).backward()
        return metrics

    for epoch in range(1, contract.epochs + 1):
        d3_keys = list(d3_events)
        rng.shuffle(d3_keys)
        rng.shuffle(d2_keys)
        records: list[dict[str, float]] = []
        for index, key in enumerate(d3_keys):
            optimizer.zero_grad(set_to_none=True)
            d3_groups = d3_events[key]
            d2_groups = d2_events[d2_keys[index % len(d2_keys)]]
            records.extend(
                backward_group(name, 0.75 / len(d3_groups)) for name in d3_groups
            )
            records.extend(
                backward_group(name, 0.25 / len(d2_groups)) for name in d2_groups
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), contract.grad_clip)
            optimizer.step()
        summary: dict[str, Any] = {
            "stage": "D3_hydraulic_effect_with_D2_anchor",
            "epoch": int(epoch),
        }
        for metric in sorted({k for row in records for k in row}):
            values = [float(row[metric]) for row in records if np.isfinite(float(row[metric]))]
            summary[metric] = float(np.mean(values)) if values else float("nan")
        history.append(summary)
    return history


__all__ = [
    "train_hydraulic_event_balanced_v70",
    "train_value_event_balanced_v70",
]
