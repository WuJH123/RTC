"""Fixed, no-sweep optimizer for Project7 Step2 V9 development."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v90 import DirectHydraulicEffectSurrogateV90, prepare_static_v80
from .step2_hydraulic_objective_v90 import hydraulic_effect_loss_v90
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v90_contract import DirectHydraulicEffectLossContractV90, LEVEL_C


def candidate_batch_chunks_v90(batch, *, candidate_chunk_size: int | None):
    """Yield a loss-equivalent partition over candidate rows only.

    This is an execution-memory guard, not a model or scientific-training
    hyperparameter: initial state, causal rainfall, reference action and all
    target semantics are retained exactly.  A caller weights each chunk by its
    candidate count before one event-level optimizer step.
    """
    candidate_count = int(batch.candidate_settings.shape[1])
    if candidate_count <= 0:
        raise ValueError("V9 candidate batch must be nonempty")
    if candidate_chunk_size is None:
        yield batch
        return
    size = int(candidate_chunk_size)
    if size <= 0:
        raise ValueError("V9 candidate chunk size must be positive")
    for start in range(0, candidate_count, size):
        stop = min(start + size, candidate_count)
        yield replace(
            batch,
            candidate_settings=batch.candidate_settings[:, start:stop],
            true_candidate_states=batch.true_candidate_states[:, start:stop],
            true_candidate_flows=batch.true_candidate_flows[:, start:stop],
            true_delta_tfv_m3=batch.true_delta_tfv_m3[:, start:stop],
        )


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        entry = cache.entry(name)
        grouped[f"{entry.rainfall_group}::{entry.event_id}"].append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def _forward(model, batch, prepared):
    kwargs = {}
    if model.conditioning_level == LEVEL_C:
        # Oracle path is intentionally confined to this diagnostic branch.
        kwargs = {
            "oracle_reference_states_physical": batch.true_reference_states,
            "oracle_reference_flows_physical": batch.true_reference_flows,
        }
    return model(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        batch.candidate_settings,
        batch.previous_actuator_flow,
        prepared,
        **kwargs,
    )


def _summary(stage: str, epoch: int, rows: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"stage": stage, "epoch": int(epoch)}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def train_d2_mechanism_v90(
    model: DirectHydraulicEffectSurrogateV90,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    onset_positive_weight: float,
    graph: Any,
    device: str = "cuda",
    seed: int = 42,
    contract: DirectHydraulicEffectLossContractV90 = DirectHydraulicEffectLossContractV90(),
    candidate_chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    """Four-epoch D2-only mechanism test used by the A/B/C sufficiency ladder."""
    contract.validate()
    if not fit_d2_names:
        raise ValueError("V9 D2 mechanism test requires TrainFit D2 groups")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(target).train()
    prepared = prepare_static_v80(graph, target)
    trainable = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )
    rng = np.random.default_rng(seed)
    events = _events(cache, fit_d2_names)
    history: list[dict[str, Any]] = []
    for epoch in range(1, contract.d2_pretrain_epochs + 1):
        keys = list(events)
        rng.shuffle(keys)
        rows: list[dict[str, float]] = []
        norms: list[float] = []
        for key in keys:
            optimizer.zero_grad(set_to_none=True)
            groups = events[key]
            for name in groups:
                batch = cache.batch(name, normalization, target)
                chunks = tuple(
                    candidate_batch_chunks_v90(
                        batch, candidate_chunk_size=candidate_chunk_size
                    )
                )
                candidates = int(batch.candidate_settings.shape[1])
                for chunk in chunks:
                    output = _forward(model, chunk, prepared)
                    loss, metrics = hydraulic_effect_loss_v90(
                        output,
                        chunk,
                        normalization,
                        scales,
                        onset_positive_weight=onset_positive_weight,
                        contract=contract,
                    )
                    weight = float(chunk.candidate_settings.shape[1]) / float(candidates)
                    (loss * weight / len(groups)).backward()
                    # Preserve candidate-weighted diagnostic summaries even
                    # when the final execution chunk is smaller.
                    rows.extend([metrics] * int(chunk.candidate_settings.shape[1]))
            norm = torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            norms.append(float(norm.detach()))
            optimizer.step()
        row = _summary(f"D2_MECHANISM_{model.conditioning_level}", epoch, rows)
        row["gradient_norm"] = float(np.mean(norms)) if norms else float("nan")
        history.append(row)
        print(
            "[V9_D2] " + " ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            ),
            flush=True,
        )
    return history


def train_hydraulic_effect_v90(
    model: DirectHydraulicEffectSurrogateV90,
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
    contract: DirectHydraulicEffectLossContractV90 = DirectHydraulicEffectLossContractV90(),
) -> list[dict[str, Any]]:
    """Formal V9 development schedule: D2=4, then D3=10 with 0.75/0.25 anchor.

    This function is provided only for the branch selected by the ladder.  The ladder
    runner itself does not invoke D3 training.
    """
    contract.validate()
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("V9 joint training requires both D2 and targeted D3")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(target).train()
    prepared = prepare_static_v80(graph, target)
    trainable = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=contract.learning_rate, weight_decay=contract.weight_decay
    )
    rng = np.random.default_rng(seed)
    d2_events = _events(cache, fit_d2_names)
    d3_events = _events(cache, fit_d3_names)
    history: list[dict[str, Any]] = []

    def backward_name(name: str, weight: float) -> dict[str, float]:
        batch = cache.batch(name, normalization, target)
        output = _forward(model, batch, prepared)
        loss, metrics = hydraulic_effect_loss_v90(
            output,
            batch,
            normalization,
            scales,
            onset_positive_weight=onset_positive_weight,
            contract=contract,
        )
        (weight * loss).backward()
        return metrics

    # Fixed D2 pretrain.
    for epoch in range(1, contract.d2_pretrain_epochs + 1):
        keys = list(d2_events)
        rng.shuffle(keys)
        rows: list[dict[str, float]] = []
        for key in keys:
            optimizer.zero_grad(set_to_none=True)
            names = d2_events[key]
            rows.extend(backward_name(n, 1.0 / len(names)) for n in names)
            torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            optimizer.step()
        history.append(_summary("D2_DIRECT_SIGNED_EFFECT", epoch, rows))

    # Fixed D3-primary / D2-anchor training.
    d2_keys = list(d2_events)
    for epoch in range(1, contract.d3_joint_epochs + 1):
        d3_keys = list(d3_events)
        rng.shuffle(d3_keys)
        rng.shuffle(d2_keys)
        rows = []
        for i, key in enumerate(d3_keys):
            optimizer.zero_grad(set_to_none=True)
            d3_names = d3_events[key]
            d2_names = d2_events[d2_keys[i % len(d2_keys)]]
            rows.extend(
                backward_name(n, contract.d3_weight / len(d3_names)) for n in d3_names
            )
            rows.extend(
                backward_name(n, contract.d2_anchor_weight / len(d2_names)) for n in d2_names
            )
            torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            optimizer.step()
        history.append(_summary("D3_PRIMARY_WITH_D2_ANCHOR", epoch, rows))
    return history


__all__ = [
    "candidate_batch_chunks_v90",
    "train_d2_mechanism_v90",
    "train_hydraulic_effect_v90",
]
