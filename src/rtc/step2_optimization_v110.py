"""Fixed event-level training schedules for Project7 Step2 V11."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import random
from typing import Sequence

import numpy as np
import torch

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v110 import ActuatorSetHydraulicResponseV110
from .step2_hydraulic_objective_v110 import EffectScalesV110, hydraulic_effect_loss_v110
from .step2_train_response_v60 import InputNormalizationV60, V60GroupBatch, V60TrainCache
from .step2_v110_contract import ActuatorSetHydraulicContractV110


def _slice_candidates(batch: V60GroupBatch, start: int, end: int) -> V60GroupBatch:
    return replace(
        batch,
        candidate_settings=batch.candidate_settings[:, start:end],
        true_candidate_states=batch.true_candidate_states[:, start:end],
        true_candidate_flows=batch.true_candidate_flows[:, start:end],
        true_delta_tfv_m3=batch.true_delta_tfv_m3[:, start:end],
    )


def _event_interleave(
    cache: V60TrainCache, names: Sequence[str], *, seed: int
) -> list[str]:
    by_event: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_event[cache.entry(name).event_id].append(name)
    rng = random.Random(int(seed))
    for values in by_event.values():
        rng.shuffle(values)
    events = sorted(by_event)
    rng.shuffle(events)
    result: list[str] = []
    while any(by_event[event] for event in events):
        for event in events:
            if by_event[event]:
                result.append(by_event[event].pop())
    return result


def _backward_group(
    model: ActuatorSetHydraulicResponseV110,
    batch: V60GroupBatch,
    prepared: PreparedStaticV60,
    scales: EffectScalesV110,
    *,
    weight: float,
    contract: ActuatorSetHydraulicContractV110,
) -> dict[str, float]:
    candidates = batch.candidate_settings.shape[1]
    totals: dict[str, float] = defaultdict(float)
    for start in range(0, candidates, contract.candidate_chunk_size):
        end = min(start + contract.candidate_chunk_size, candidates)
        chunk = _slice_candidates(batch, start, end)
        output = model(
            chunk.initial_state,
            chunk.rainfall,
            chunk.reference_settings,
            chunk.candidate_settings,
            chunk.previous_actuator_flow,
            prepared,
        )
        loss, metrics = hydraulic_effect_loss_v110(output, chunk, scales, contract=contract)
        fraction = (end - start) / float(candidates)
        (float(weight) * fraction * loss).backward()
        for key, value in metrics.items():
            totals[key] += fraction * float(value)
    return dict(totals)


def _optimizer(model: ActuatorSetHydraulicResponseV110, contract: ActuatorSetHydraulicContractV110):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("V11 has no trainable hydraulic parameters")
    return torch.optim.AdamW(
        parameters,
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_d2_mechanism_v110(
    model: ActuatorSetHydraulicResponseV110,
    cache: V60TrainCache,
    fit_d2_names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    scales: EffectScalesV110,
    *,
    device: torch.device | str,
    contract: ActuatorSetHydraulicContractV110 = ActuatorSetHydraulicContractV110(),
) -> list[dict[str, float]]:
    """Four-epoch D2 local/mechanism learning; one optimizer step per counterfactual group."""
    contract.validate()
    if not fit_d2_names or any(not name.startswith("D2::") for name in fit_d2_names):
        raise ValueError("V11 D2 stage requires TrainFit D2 groups only")
    _seed(contract.seed)
    target = torch.device(device)
    model.to(target).train()
    optimizer = _optimizer(model, contract)
    history: list[dict[str, float]] = []
    for epoch in range(contract.d2_epochs):
        names = _event_interleave(cache, fit_d2_names, seed=contract.seed + epoch)
        epoch_metrics: dict[str, list[float]] = defaultdict(list)
        grad_norms: list[float] = []
        for name in names:
            optimizer.zero_grad(set_to_none=True)
            batch = cache.batch(name, normalization, target)
            metrics = _backward_group(
                model, batch, prepared, scales, weight=1.0, contract=contract
            )
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], contract.grad_clip
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("V11 D2 non-finite gradient norm")
            optimizer.step()
            grad_norms.append(float(norm.detach().cpu()))
            for key, value in metrics.items():
                epoch_metrics[key].append(float(value))
        row = {"epoch": float(epoch + 1), "gradient_norm": float(np.mean(grad_norms))}
        row.update({key: float(np.mean(values)) for key, values in epoch_metrics.items()})
        history.append(row)
    return history


def train_d3_interaction_v110(
    model: ActuatorSetHydraulicResponseV110,
    cache: V60TrainCache,
    fit_d3_names: Sequence[str],
    anchor_d2_names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    scales: EffectScalesV110,
    *,
    device: torch.device | str,
    contract: ActuatorSetHydraulicContractV110 = ActuatorSetHydraulicContractV110(),
) -> list[dict[str, float]]:
    """Ten-epoch targeted D3 interaction stage with a fixed D2 mechanism anchor.

    D2 predicted effects are never summed to create a D3 target.  D2 only
    regularizes the shared mechanism encoder through separate authoritative D2
    labels, while D3 authoritative branches supervise the joint actuator set.
    """
    contract.validate()
    if not fit_d3_names or any(not name.startswith("D3::") for name in fit_d3_names):
        raise ValueError("V11 D3 stage requires targeted TrainFit D3 groups")
    if not anchor_d2_names or any(not name.startswith("D2::") for name in anchor_d2_names):
        raise ValueError("V11 D3 stage requires a D2 anchor set")
    _seed(contract.seed)
    target = torch.device(device)
    model.to(target).train()
    optimizer = _optimizer(model, contract)
    history: list[dict[str, float]] = []
    for epoch in range(contract.d3_epochs):
        d3_names = _event_interleave(cache, fit_d3_names, seed=contract.seed + 100 + epoch)
        d2_names = _event_interleave(cache, anchor_d2_names, seed=contract.seed + 200 + epoch)
        epoch_metrics: dict[str, list[float]] = defaultdict(list)
        grad_norms: list[float] = []
        for step, d3_name in enumerate(d3_names):
            optimizer.zero_grad(set_to_none=True)
            d3_batch = cache.batch(d3_name, normalization, target)
            d3_metrics = _backward_group(
                model,
                d3_batch,
                prepared,
                scales,
                weight=contract.d3_weight,
                contract=contract,
            )
            d2_name = d2_names[step % len(d2_names)]
            d2_batch = cache.batch(d2_name, normalization, target)
            d2_metrics = _backward_group(
                model,
                d2_batch,
                prepared,
                scales,
                weight=contract.d2_anchor_weight,
                contract=contract,
            )
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], contract.grad_clip
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("V11 D3 non-finite gradient norm")
            optimizer.step()
            grad_norms.append(float(norm.detach().cpu()))
            for key, value in d3_metrics.items():
                epoch_metrics[f"d3_{key}"].append(value)
            for key, value in d2_metrics.items():
                epoch_metrics[f"d2_anchor_{key}"].append(value)
        row = {"epoch": float(epoch + 1), "gradient_norm": float(np.mean(grad_norms))}
        row.update({key: float(np.mean(values)) for key, values in epoch_metrics.items()})
        history.append(row)
    return history


__all__ = ["train_d2_mechanism_v110", "train_d3_interaction_v110"]
