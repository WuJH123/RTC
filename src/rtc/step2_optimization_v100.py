"""Fixed D2 mechanism optimizer for the Project7 V10 nonlocal operator."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v80 import prepare_static_v80
from .step2_control_response_v100 import DirectHydraulicEffectSurrogateV100
from .step2_hydraulic_objective_v90 import hydraulic_effect_loss_v90
from .step2_optimization_v90 import candidate_batch_chunks_v90
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v100_contract import NonlocalHydraulicEffectLossContractV100


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        entry = cache.entry(name)
        grouped[f"{entry.rainfall_group}::{entry.event_id}"].append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def train_d2_nonlocal_mechanism_v100(
    model: DirectHydraulicEffectSurrogateV100,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    onset_positive_weight: float,
    graph: Any,
    device: str = "cuda",
    seed: int = 42,
    contract: NonlocalHydraulicEffectLossContractV100 = NonlocalHydraulicEffectLossContractV100(),
) -> list[dict[str, Any]]:
    """Train the single pre-registered V10 D2 mechanism test.

    Candidate chunking is an execution-memory guard only: chunk losses are
    weighted back to the original full-group mean before the one event-level
    optimizer step. The canonical schedule remains 14 updates/epoch x 4 epochs.
    """
    contract.validate()
    if not fit_d2_names:
        raise ValueError("V10 D2 mechanism test requires TrainFit D2 groups")
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
    events = _events(cache, fit_d2_names)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []

    for epoch in range(1, contract.d2_pretrain_epochs + 1):
        event_keys = list(events)
        rng.shuffle(event_keys)
        epoch_loss: list[float] = []
        grad_norms: list[float] = []
        for event_key in event_keys:
            optimizer.zero_grad(set_to_none=True)
            group_names = events[event_key]
            event_loss = 0.0
            for name in group_names:
                batch = cache.batch(name, normalization, target)
                candidate_count = int(batch.candidate_settings.shape[1])
                if candidate_count <= 0:
                    raise RuntimeError("V10 encountered a D2 group without candidates")
                for chunk in candidate_batch_chunks_v90(
                    batch,
                    candidate_chunk_size=contract.candidate_chunk_size,
                ):
                    chunk_count = int(chunk.candidate_settings.shape[1])
                    output = model(
                        chunk.initial_state,
                        chunk.rainfall,
                        chunk.reference_settings,
                        chunk.candidate_settings,
                        chunk.previous_actuator_flow,
                        prepared,
                    )
                    loss, _ = hydraulic_effect_loss_v90(
                        output,
                        chunk,
                        normalization,
                        scales,
                        onset_positive_weight=onset_positive_weight,
                        contract=contract,
                    )
                    weight = (chunk_count / candidate_count) / len(group_names)
                    (loss * weight).backward()
                    event_loss += float(loss.detach().cpu()) * weight
            norm = torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            norm_tensor = torch.as_tensor(norm)
            if not torch.isfinite(norm_tensor):
                raise RuntimeError("V10 D2 mechanism produced a non-finite gradient norm")
            optimizer.step()
            epoch_loss.append(event_loss)
            grad_norms.append(float(norm_tensor.detach().cpu()))
        row = {
            "stage": "D2_V100_NONLOCAL_OPERATOR",
            "epoch": int(epoch),
            "events": int(len(event_keys)),
            "optimizer_updates": int(len(event_keys)),
            "mean_loss": float(np.mean(epoch_loss)),
            "mean_gradient_norm": float(np.mean(grad_norms)),
            "candidate_chunk_size": int(contract.candidate_chunk_size),
            "finite_gradients": True,
        }
        history.append(row)
        print("[V100_D2] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


__all__ = ["train_d2_nonlocal_mechanism_v100"]
