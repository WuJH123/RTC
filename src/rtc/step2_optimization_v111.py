"""Deterministic V11.1 training helpers for development diagnostics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import random
from typing import Sequence

import numpy as np
import torch

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v111 import ActuatorSetHydraulicResponseV111
from .step2_hydraulic_objective_v111 import EffectScalesV111, hydraulic_effect_loss_v111
from .step2_train_response_v60 import V60GroupBatch, V60TrainCache
from .step2_v111_contract import V111LossContract


def _slice_candidates(batch: V60GroupBatch, start: int, end: int) -> V60GroupBatch:
    return replace(
        batch,
        candidate_settings=batch.candidate_settings[:, start:end],
        true_candidate_states=batch.true_candidate_states[:, start:end],
        true_candidate_flows=batch.true_candidate_flows[:, start:end],
        true_delta_tfv_m3=batch.true_delta_tfv_m3[:, start:end],
    )


def deterministic_event_interleave_v111(cache: V60TrainCache, names: Sequence[str], *, seed: int = 42) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[cache.entry(name).event_id].append(name)
    rng = random.Random(int(seed))
    for values in groups.values():
        values.sort()
        rng.shuffle(values)
    events = sorted(groups)
    rng.shuffle(events)
    result: list[str] = []
    while any(groups[event] for event in events):
        for event in events:
            if groups[event]:
                result.append(groups[event].pop())
    return result


def backward_group_v111(
    model: ActuatorSetHydraulicResponseV111,
    batch: V60GroupBatch,
    prepared: PreparedStaticV60,
    scales: EffectScalesV111,
    *,
    loss_contract: V111LossContract = V111LossContract(),
) -> dict[str, float]:
    candidates = int(batch.candidate_settings.shape[1])
    total: dict[str, float] = defaultdict(float)
    for start in range(0, candidates, 4):
        end = min(start + 4, candidates)
        output = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.candidate_settings[:, start:end],
            batch.previous_actuator_flow,
            prepared,
        )
        chunk = _slice_candidates(batch, start, end)
        loss, metrics = hydraulic_effect_loss_v111(output, chunk, scales, contract=loss_contract)
        fraction = (end - start) / float(candidates)
        (fraction * loss).backward()
        for key, value in metrics.items():
            total[key] += fraction * float(value)
    return dict(total)


def _optimizer(model: ActuatorSetHydraulicResponseV111, contract: V111LossContract):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=contract.learning_rate, weight_decay=contract.weight_decay)


def train_d2_v111(
    model: ActuatorSetHydraulicResponseV111,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization,
    prepared: PreparedStaticV60,
    scales: EffectScalesV111,
    *,
    device: torch.device,
    epochs: int,
    loss_contract: V111LossContract = V111LossContract(),
    log_prefix: str = "V111",
) -> list[dict[str, float]]:
    loss_contract.validate()
    if epochs <= 0 or epochs > loss_contract.canonical_max_epochs:
        raise ValueError("V111 epoch count outside frozen development range")
    ordered = deterministic_event_interleave_v111(cache, names, seed=loss_contract.seed)
    optimizer = _optimizer(model, loss_contract)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        random.seed(loss_contract.seed + epoch)
        np.random.seed(loss_contract.seed + epoch)
        sums: dict[str, float] = defaultdict(float)
        for number, name in enumerate(ordered, 1):
            optimizer.zero_grad(set_to_none=True)
            batch = cache.batch(name, normalization, device)
            metrics = backward_group_v111(model, batch, prepared, scales, loss_contract=loss_contract)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], loss_contract.grad_clip
            )
            optimizer.step()
            for key, value in metrics.items():
                sums[key] += float(value)
            if number == len(ordered) or number % 16 == 0:
                print(
                    f"[{log_prefix}] stage=d2 epoch={epoch} groups={number}/{len(ordered)} "
                    f"loss={metrics.get('loss', float('nan')):.6g} "
                    f"direct_active={metrics.get('direct_active', float('nan')):.6g} "
                    f"direct_inactive={metrics.get('direct_inactive', float('nan')):.6g}",
                    flush=True,
                )
        history.append({"epoch": float(epoch), **{key: value / len(ordered) for key, value in sums.items()}})
    return history


__all__ = ["backward_group_v111", "deterministic_event_interleave_v111", "train_d2_v111"]
