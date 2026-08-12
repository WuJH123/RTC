"""Authoritative V6 optimizers with explicit event/source balancing."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v60 import (
    ControlValueSurrogateV60,
    HydraulicResponseSurrogateV60,
    prepare_static_v60,
)
from .step2_train_response_v60 import (
    InputNormalizationV60,
    TargetScalesV60,
    V60TrainCache,
    hydraulic_loss_v60,
    value_loss_v60,
)


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        entry = cache.entry(name)
        grouped[f"{entry.rainfall_group}::{entry.event_id}"].append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def event_balance_summary_v60(cache: V60TrainCache, names: Sequence[str]) -> dict[str, Any]:
    grouped = _events(cache, names)
    counts = [len(values) for values in grouped.values()]
    return {
        "events": len(grouped),
        "groups": len(names),
        "groups_per_event": {
            "min": min(counts) if counts else 0,
            "median": float(np.median(counts)) if counts else 0.0,
            "max": max(counts) if counts else 0,
        },
        "optimizer_step_unit": "event",
        "group_gradient_weight_within_event": "1/groups_in_event",
    }


def train_value_event_balanced_v60(
    model: ControlValueSurrogateV60,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    fit_d3_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV60,
    graph: Any,
    device: str = "cuda",
    d2_pretrain_epochs: int = 4,
    joint_epochs: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("V6 value training needs D2 and targeted D3")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(target).float().train()
    prepared = prepare_static_v60(graph, target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    d2_events, d3_events = _events(cache, fit_d2_names), _events(cache, fit_d3_names)
    history: list[dict[str, Any]] = []

    def backward_group(name: str, weight: float) -> float:
        batch = cache.batch(name, normalization, target)
        output = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings,
            prepared,
            batch.elapsed_seconds,
        )
        loss, _ = value_loss_v60(
            output.delta_tfv_m3,
            batch.true_delta_tfv_m3,
            scale_m3=scales.tfv_scale(batch.source_kind),
        )
        (weight * loss).backward()
        return float(loss.detach())

    for epoch in range(1, d2_pretrain_epochs + 1):
        event_keys = list(d2_events)
        rng.shuffle(event_keys)
        losses: list[float] = []
        for key in event_keys:
            optimizer.zero_grad(set_to_none=True)
            groups = d2_events[key]
            losses.extend(backward_group(name, 1.0 / len(groups)) for name in groups)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        history.append(
            {
                "stage": "D2_event_balanced_sensitivity",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "events": len(event_keys),
            }
        )

    d2_keys = list(d2_events)
    for epoch in range(1, joint_epochs + 1):
        d3_keys = list(d3_events)
        rng.shuffle(d3_keys)
        rng.shuffle(d2_keys)
        losses = []
        for index, key in enumerate(d3_keys):
            optimizer.zero_grad(set_to_none=True)
            d3_groups = d3_events[key]
            d2_groups = d2_events[d2_keys[index % len(d2_keys)]]
            losses.extend(
                backward_group(name, 0.5 / len(d3_groups)) for name in d3_groups
            )
            losses.extend(
                backward_group(name, 0.5 / len(d2_groups)) for name in d2_groups
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        history.append(
            {
                "stage": "D3_event_balanced_with_D2_anchor",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "d3_events": len(d3_keys),
            }
        )
    return history


def train_hydraulic_event_balanced_v60(
    model: HydraulicResponseSurrogateV60,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    fit_d3_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV60,
    graph: Any,
    device: str = "cuda",
    epochs: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("V6 hydraulic training needs D2 and targeted D3")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + 17)
    np.random.seed(seed + 17)
    model.to(target).float().train()
    prepared = prepare_static_v60(graph, target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 17)
    d2_events, d3_events = _events(cache, fit_d2_names), _events(cache, fit_d3_names)
    d2_keys = list(d2_events)
    history: list[dict[str, Any]] = []

    def backward_group(name: str, weight: float) -> float:
        batch = cache.batch(name, normalization, target)
        output = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings,
            prepared,
        )
        loss, _ = hydraulic_loss_v60(
            output,
            batch,
            prepared,
            scales,
            horizon=model.horizon_contract,
        )
        (weight * loss).backward()
        return float(loss.detach())

    for epoch in range(1, epochs + 1):
        d3_keys = list(d3_events)
        rng.shuffle(d3_keys)
        rng.shuffle(d2_keys)
        losses: list[float] = []
        for index, key in enumerate(d3_keys):
            optimizer.zero_grad(set_to_none=True)
            d3_groups = d3_events[key]
            d2_groups = d2_events[d2_keys[index % len(d2_keys)]]
            losses.extend(
                backward_group(name, 0.5 / len(d3_groups)) for name in d3_groups
            )
            losses.extend(
                backward_group(name, 0.5 / len(d2_groups)) for name in d2_groups
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        history.append(
            {
                "stage": "hydraulic_event_source_balanced",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "d3_events": len(d3_keys),
            }
        )
    return history


__all__ = [
    "event_balance_summary_v60",
    "train_hydraulic_event_balanced_v60",
    "train_value_event_balanced_v60",
]
