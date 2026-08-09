from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import DifferentiableHydraulicWorldModel, SparseStateEstimator


@dataclass(frozen=True)
class TrainingHistory:
    epoch_loss: tuple[float, ...]


def _device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _tensor(value: np.ndarray | torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=dtype)


def train_step1(
    model: SparseStateEstimator,
    *,
    observed_history: np.ndarray | torch.Tensor,
    observation_mask: np.ndarray | torch.Tensor,
    target_state: np.ndarray | torch.Tensor,
    static_node_features: np.ndarray | torch.Tensor,
    edge_index: np.ndarray | torch.Tensor,
    context_history: np.ndarray | torch.Tensor | None = None,
    state_weights: Sequence[float] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
) -> TrainingHistory:
    """Train Step1 on development-only causal windows.

    Split construction is deliberately external: callers must pass only the development
    rainfall partition here. Calibration, safety-audit and final samples must never enter
    this function.
    """

    dev = _device(device)
    obs = _tensor(observed_history)
    mask = _tensor(observation_mask)
    target = _tensor(target_state)
    if obs.shape != mask.shape or obs.shape[0] != target.shape[0]:
        raise ValueError("Step1 sample dimensions do not align")
    if context_history is None:
        dataset = TensorDataset(obs, mask, target)
    else:
        context = _tensor(context_history)
        if context.shape[0] != obs.shape[0]:
            raise ValueError("context sample dimension mismatch")
        dataset = TensorDataset(obs, mask, target, context)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    static = _tensor(static_node_features).to(dev)
    edges = _tensor(edge_index, dtype=torch.long).to(dev)
    weights = None if state_weights is None else torch.as_tensor(state_weights, dtype=torch.float32, device=dev)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        running = 0.0
        count = 0
        for batch in loader:
            if context_history is None:
                b_obs, b_mask, b_target = batch
                b_context = None
            else:
                b_obs, b_mask, b_target, b_context = batch
                b_context = b_context.to(dev)
            b_obs, b_mask, b_target = b_obs.to(dev), b_mask.to(dev), b_target.to(dev)
            pred = model(b_obs, b_mask, static, edges, b_context)
            error = (pred - b_target).square()
            if weights is not None:
                error = error * weights.view(1, 1, -1)
            loss = error.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * b_obs.shape[0]
            count += b_obs.shape[0]
        losses.append(running / max(count, 1))
    return TrainingHistory(tuple(losses))


def train_step2(
    model: DifferentiableHydraulicWorldModel,
    *,
    initial_state: np.ndarray | torch.Tensor,
    rainfall: np.ndarray | torch.Tensor,
    settings: np.ndarray | torch.Tensor,
    previous_actuator_flow: np.ndarray | torch.Tensor,
    actuator_physics: np.ndarray | torch.Tensor,
    target_states: np.ndarray | torch.Tensor,
    target_actuator_flows: np.ndarray | torch.Tensor,
    actuator_upstream: np.ndarray | torch.Tensor,
    actuator_downstream: np.ndarray | torch.Tensor,
    static_node_features: np.ndarray | torch.Tensor,
    edge_index: np.ndarray | torch.Tensor,
    state_weights: Sequence[float] | None = None,
    flow_loss_weight: float = 1.0,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
) -> TrainingHistory:
    """Train the coupled setting->flow->hydraulic free-rollout world model.

    D2 same-checkpoint probes should be mixed with D0/D1 base dynamics and D3 interaction
    rollouts by the caller. The loss supervises full hydraulic trajectories and actuator
    flows rather than only PFV/TFV scores.
    """

    dev = _device(device)
    tensors = [_tensor(x) for x in (initial_state, rainfall, settings, previous_actuator_flow, actuator_physics, target_states, target_actuator_flows)]
    sample_count = tensors[0].shape[0]
    if any(x.shape[0] != sample_count for x in tensors):
        raise ValueError("Step2 sample dimensions do not align")
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    up = _tensor(actuator_upstream, dtype=torch.long).to(dev)
    down = _tensor(actuator_downstream, dtype=torch.long).to(dev)
    static = _tensor(static_node_features).to(dev)
    edges = _tensor(edge_index, dtype=torch.long).to(dev)
    weights = None if state_weights is None else torch.as_tensor(state_weights, dtype=torch.float32, device=dev)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        running = 0.0
        count = 0
        for batch in loader:
            b_initial, b_rain, b_settings, b_prev, b_phys, b_states, b_flows = [x.to(dev) for x in batch]
            rollout = model.rollout(
                b_initial,
                b_rain,
                b_settings,
                b_prev,
                up,
                down,
                b_phys,
                static,
                edges,
            )
            state_error = (rollout.states - b_states).square()
            if weights is not None:
                state_error = state_error * weights.view(1, 1, 1, -1)
            state_loss = state_error.mean()
            flow_loss = (rollout.actuator_flows - b_flows).square().mean()
            loss = state_loss + float(flow_loss_weight) * flow_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * b_initial.shape[0]
            count += b_initial.shape[0]
        losses.append(running / max(count, 1))
    return TrainingHistory(tuple(losses))


def save_torch_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    model_config: dict[str, object],
    training_manifest_sha256: str,
    scientific_split: str = "development",
) -> dict[str, str]:
    if scientific_split != "development":
        raise ValueError("trainable checkpoints may only be fit on the development split")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "training_manifest_sha256": training_manifest_sha256,
            "scientific_split": scientific_split,
        },
        out,
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "checkpoint": str(out),
        "sha256": digest,
        "training_manifest_sha256": training_manifest_sha256,
        "scientific_split": scientific_split,
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta
