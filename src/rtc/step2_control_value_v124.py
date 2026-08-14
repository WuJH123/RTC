"""Interaction-aware direct control Value model for Project7 V12.4 development.

V7/V123 compresses 109 actuator effect tokens with fixed sum/mean/max/algebraic-pair
statistics.  That representation is inexpensive but can lose which *specific* actuator
combinations are active.  V124 keeps the same causal inputs, direct signed Delta-TFV
supervision and exact-zero reference contract, while adding masked self-attention across
active actuator effect tokens before pooling.

This module is development-only until it beats the frozen causal V123 Value on the same
InternalHoldout split.  It does not authorize continuous MPC by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .step2_control_response_v60 import PreparedStaticV60, prepare_static_v60
from .step2_control_response_v70 import DirectValueOutputV70, TemporalActionProjectorV70
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70, value_loss_v70
from .step2_v70_contract import DirectValueLossContractV70

V124_VALUE_CONTRACT = "PROJECT7_V124_INTERACTION_AWARE_DIRECT_TFV_VALUE_V1"


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _bias_free_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=False),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1, bias=False),
    )


class ControlValueSurrogateV124(nn.Module):
    """State-conditioned actuator-set attention -> direct signed Delta-TFV."""

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        physics_dim: int,
        actuator_count: int,
        temporal_basis: np.ndarray,
        control_block_steps: int,
        tfv_scale_m3: float,
        hidden_dim: int = 96,
        actuator_embedding_dim: int = 16,
        attention_heads: int = 4,
        contract: DirectValueLossContractV70 = DirectValueLossContractV70(),
    ) -> None:
        super().__init__()
        contract.validate()
        if hidden_dim % int(attention_heads):
            raise ValueError("V124 hidden_dim must be divisible by attention_heads")
        self.state_dim = int(state_dim)
        self.rainfall_dim = int(rainfall_dim)
        self.actuator_count = int(actuator_count)
        self.hidden_dim = int(hidden_dim)
        self.contract = contract
        self.temporal = TemporalActionProjectorV70(
            temporal_basis, control_block_steps=control_block_steps
        )
        k = self.temporal.feature_count
        global_dim = 3 * self.state_dim + k * self.rainfall_dim
        self.global_context = _mlp(global_dim, hidden_dim, hidden_dim)
        self.actuator_embedding = nn.Embedding(self.actuator_count, actuator_embedding_dim)
        base_dim = (
            2 * self.state_dim
            + 1
            + int(physics_dim)
            + k
            + hidden_dim
            + actuator_embedding_dim
        )
        self.effect_encoder = _mlp(base_dim + k, hidden_dim, hidden_dim)
        self.interaction_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=int(attention_heads),
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.interaction_ff = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.linear_action_head = nn.Linear(self.actuator_count * k, 1, bias=False)
        # active-sum, active-mean, active-max, global context
        self.direct_head = _bias_free_head(4 * hidden_dim, hidden_dim)
        scale = max(float(tfv_scale_m3), 1.0)
        self.register_buffer("tfv_scale_m3", torch.tensor(scale, dtype=torch.float32))
        nn.init.zeros_(self.linear_action_head.weight)
        for module in self.direct_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.35)

    def _global_boundary(
        self, initial_state: torch.Tensor, rainfall: torch.Tensor
    ) -> torch.Tensor:
        mean = initial_state.mean(dim=1)
        maximum = initial_state.amax(dim=1)
        std = initial_state.std(dim=1, unbiased=False)
        rain = self.temporal.rainfall_features(rainfall)
        return self.global_context(torch.cat((mean, maximum, std, rain), dim=-1))

    def _base_actuator_features(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV60,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_actuator_flow.ndim != 2:
            raise ValueError("V124 previous_actuator_flow must be [B,A]")
        batch = initial_state.shape[0]
        if previous_actuator_flow.shape != (batch, self.actuator_count):
            raise ValueError("V124 current actuator-flow vector is misaligned")
        up = initial_state[:, prepared.actuator_upstream]
        down = initial_state[:, prepared.actuator_downstream]
        ref = self.temporal.settings_features(reference_settings)
        physics = prepared.actuator_physics[None].expand(batch, -1, -1)
        global_context = self._global_boundary(initial_state, rainfall)
        global_by_actuator = global_context[:, None].expand(
            batch, self.actuator_count, -1
        )
        ids = torch.arange(self.actuator_count, device=initial_state.device)
        identity = self.actuator_embedding(ids)[None].expand(batch, -1, -1)
        base = torch.cat(
            (
                up,
                down,
                previous_actuator_flow[..., None],
                physics,
                ref,
                global_by_actuator,
                identity,
            ),
            dim=-1,
        )
        return base, global_context

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV60,
    ) -> DirectValueOutputV70:
        if candidate_settings.ndim != 4:
            raise ValueError("V124 candidate settings must be [B,C,H,A]")
        batch, candidates, _, actuators = candidate_settings.shape
        if actuators != self.actuator_count:
            raise ValueError("V124 actuator count mismatch")
        base, global_context = self._base_actuator_features(
            initial_state,
            rainfall,
            reference_settings,
            previous_actuator_flow,
            prepared,
        )
        reference_expanded = reference_settings[:, None].expand_as(candidate_settings)
        delta = candidate_settings - reference_expanded
        delta_features = self.temporal.settings_features(delta)
        base_expanded = base[:, None].expand(batch, candidates, -1, -1)
        zeros = torch.zeros_like(delta_features)
        effect_input = torch.cat((base_expanded, delta_features), dim=-1)
        zero_input = torch.cat((base_expanded, zeros), dim=-1)
        effect = self.effect_encoder(effect_input) - self.effect_encoder(zero_input)

        active = torch.linalg.vector_norm(delta_features, dim=-1) > 1.0e-8
        flat_effect = effect.reshape(batch * candidates, self.actuator_count, self.hidden_dim)
        flat_active = active.reshape(batch * candidates, self.actuator_count)
        # MultiheadAttention cannot accept a row with every key masked.  Temporarily keep
        # token zero visible for exact-HOLD rows; exact-zero is restored structurally at
        # the output below.
        safe_active = flat_active.clone()
        empty = ~safe_active.any(dim=1)
        if bool(empty.any()):
            safe_active[empty, 0] = True
        attended, _ = self.interaction_attention(
            flat_effect,
            flat_effect,
            flat_effect,
            key_padding_mask=~safe_active,
            need_weights=False,
        )
        mixed = self.attention_norm(flat_effect + attended)
        mixed = self.ff_norm(mixed + self.interaction_ff(mixed))
        mixed = mixed * flat_active[..., None].to(mixed.dtype)
        mixed = mixed.reshape(batch, candidates, self.actuator_count, self.hidden_dim)

        active_float = active[..., None].to(mixed.dtype)
        count = active_float.sum(dim=2).clamp_min(1.0)
        active_sum = mixed.sum(dim=2) / torch.sqrt(count)
        active_mean = mixed.sum(dim=2) / count
        masked = mixed.masked_fill(~active[..., None], float("-inf"))
        active_max = masked.amax(dim=2)
        active_max = torch.where(
            active.any(dim=2, keepdim=True),
            active_max,
            torch.zeros_like(active_max),
        )
        global_by_candidate = global_context[:, None].expand(batch, candidates, -1)
        pooled = torch.cat(
            (active_sum, active_mean, active_max, global_by_candidate), dim=-1
        )

        linear = self.linear_action_head(
            delta_features.reshape(batch, candidates, -1)
        ).squeeze(-1)
        nonlinear = self.direct_head(pooled).squeeze(-1)
        normalized = linear + nonlinear
        limit = float(self.contract.transformed_limit)
        normalized = limit * torch.tanh(normalized / limit)
        delta_tfv = self.tfv_scale_m3.to(normalized) * torch.sinh(normalized)

        same_action = torch.all(candidate_settings == reference_expanded, dim=(2, 3))
        normalized = torch.where(same_action, torch.zeros_like(normalized), normalized)
        delta_tfv = torch.where(same_action, torch.zeros_like(delta_tfv), delta_tfv)
        effect = torch.where(
            same_action[..., None, None], torch.zeros_like(effect), effect
        )
        pooled = torch.where(same_action[..., None], torch.zeros_like(pooled), pooled)
        return DirectValueOutputV70(
            delta_tfv_m3=delta_tfv,
            normalized_delta_tfv=normalized,
            actuator_effect_tokens=effect,
            pooled_interaction=pooled,
        )


@dataclass(frozen=True)
class ValueLossContractV124:
    base: DirectValueLossContractV70 = DirectValueLossContractV70()
    listwise_weight: float = 0.30
    listwise_temperature: float = 1.0

    def validate(self) -> None:
        self.base.validate()
        if not 0.0 <= float(self.listwise_weight) <= 2.0:
            raise ValueError("V124 listwise weight is invalid")
        if not 0.05 <= float(self.listwise_temperature) <= 10.0:
            raise ValueError("V124 listwise temperature is invalid")


def value_loss_v124(
    output: DirectValueOutputV70,
    truth: torch.Tensor,
    *,
    scale_m3: float,
    contract: ValueLossContractV124 = ValueLossContractV124(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Retain V7 magnitude/pair losses and add top-of-list distribution matching."""
    contract.validate()
    base_loss, metrics = value_loss_v70(
        output,
        truth,
        scale_m3=scale_m3,
        contract=contract.base,
    )
    scale = max(float(scale_m3), 1.0) * float(contract.listwise_temperature)
    truth_logits = -truth.detach() / scale
    predicted_logits = -output.delta_tfv_m3 / scale
    target_probability = torch.softmax(truth_logits, dim=1)
    listwise = -(target_probability * F.log_softmax(predicted_logits, dim=1)).sum(dim=1).mean()
    total = base_loss + float(contract.listwise_weight) * listwise
    return total, {
        **metrics,
        "listwise": float(listwise.detach()),
        "loss_v124": float(total.detach()),
    }


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        entry = cache.entry(name)
        key = f"{entry.rainfall_group}::{entry.event_id}"
        grouped.setdefault(key, []).append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def train_value_event_balanced_v124(
    model: ControlValueSurrogateV124,
    cache: V60TrainCache,
    *,
    fit_d2_names: Sequence[str],
    fit_d3_names: Sequence[str],
    normalization: InputNormalizationV60,
    scales: TargetScalesV70,
    graph: Any,
    device: str = "cuda",
    seed: int = 42,
    contract: ValueLossContractV124 = ValueLossContractV124(),
) -> list[dict[str, Any]]:
    """Same event-balanced D2->D3 schedule as V7, with V124 ranking loss."""
    contract.validate()
    if not fit_d2_names or not fit_d3_names:
        raise ValueError("V124 training requires D2 and targeted D3")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(target).float().train()
    prepared = prepare_static_v60(graph, target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.base.learning_rate,
        weight_decay=contract.base.weight_decay,
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
        loss, metrics = value_loss_v124(
            output,
            batch.true_delta_tfv_m3,
            scale_m3=scales.direct_tfv_scale_m3,
            contract=contract,
        )
        (weight * loss).backward()
        return metrics

    def run_stage(
        stage: str,
        epochs: int,
        primary_events: dict[str, list[str]],
        secondary_events: dict[str, list[str]] | None,
        primary_weight: float,
        secondary_weight: float,
    ) -> None:
        secondary_keys = list(secondary_events or {})
        for epoch in range(1, int(epochs) + 1):
            keys = list(primary_events)
            rng.shuffle(keys)
            if secondary_keys:
                rng.shuffle(secondary_keys)
            records: list[dict[str, float]] = []
            for index, key in enumerate(keys):
                optimizer.zero_grad(set_to_none=True)
                groups = primary_events[key]
                records.extend(
                    backward_group(name, primary_weight / len(groups)) for name in groups
                )
                if secondary_keys and secondary_events is not None:
                    other = secondary_events[secondary_keys[index % len(secondary_keys)]]
                    records.extend(
                        backward_group(name, secondary_weight / len(other)) for name in other
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), contract.base.grad_clip)
                optimizer.step()
            row: dict[str, Any] = {"stage": stage, "epoch": epoch}
            for metric in sorted({k for record in records for k in record}):
                values = [
                    float(record[metric])
                    for record in records
                    if np.isfinite(float(record[metric]))
                ]
                row[metric] = float(np.mean(values)) if values else float("nan")
            history.append(row)
            fields = " ".join(
                f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in row.items()
            )
            print(f"[V124_VALUE] {fields}", flush=True)

    run_stage(
        "D2_direct_sensitivity",
        contract.base.d2_pretrain_epochs,
        d2_events,
        None,
        1.0,
        0.0,
    )
    run_stage(
        "D3_primary_with_D2_anchor",
        contract.base.joint_epochs,
        d3_events,
        d2_events,
        float(contract.base.joint_d3_weight),
        float(contract.base.joint_d2_weight),
    )
    return history


__all__ = [
    "ControlValueSurrogateV124",
    "V124_VALUE_CONTRACT",
    "ValueLossContractV124",
    "train_value_event_balanced_v124",
    "value_loss_v124",
]
