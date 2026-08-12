"""Project7 Step2 V7.0 control-oriented surrogates.

The MPC-facing value model predicts signed counterfactual Delta-TFV directly in physical
volume units. It does not route the objective through a tiny flooding-rate difference
and a 72-step integration. The action encoder is state-conditioned at each actuator,
uses the six frozen temporal basis functions, current actuator flow, actuator physics,
and then performs set-level multi-actuator interaction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .step2_control_basis_v60 import ControlBasisV60
from .step2_control_response_v60 import (
    HydraulicResponseSurrogateV60,
    PreparedStaticV60,
    prepare_static_v60,
)
from .step2_v70_contract import DirectValueLossContractV70


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


@dataclass(frozen=True)
class DirectValueOutputV70:
    delta_tfv_m3: torch.Tensor
    normalized_delta_tfv: torch.Tensor
    actuator_effect_tokens: torch.Tensor
    pooled_interaction: torch.Tensor


class TemporalActionProjectorV70(nn.Module):
    """Compress H72 actions to the frozen six temporal control coordinates per actuator."""

    def __init__(self, temporal_basis: np.ndarray, *, control_block_steps: int = 2) -> None:
        super().__init__()
        basis = torch.as_tensor(np.asarray(temporal_basis), dtype=torch.float32)
        if basis.ndim != 2:
            raise ValueError("V7 temporal basis must be [control_blocks,K]")
        if control_block_steps <= 0:
            raise ValueError("control_block_steps must be positive")
        weights = basis / basis.sum(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("weights", weights)
        self.control_block_steps = int(control_block_steps)

    @property
    def feature_count(self) -> int:
        return int(self.weights.shape[1])

    def _blocks(self, values: torch.Tensor) -> torch.Tensor:
        expected_horizon = int(self.weights.shape[0]) * self.control_block_steps
        if values.shape[-2] != expected_horizon:
            raise ValueError(
                f"V7 action horizon {values.shape[-2]} != {expected_horizon}"
            )
        return values[..., :: self.control_block_steps, :]

    def settings_features(self, values: torch.Tensor) -> torch.Tensor:
        blocks = self._blocks(values)
        if blocks.ndim == 3:
            return torch.einsum("tk,bta->bak", self.weights.to(blocks), blocks)
        if blocks.ndim == 4:
            return torch.einsum("tk,bcta->bcak", self.weights.to(blocks), blocks)
        raise ValueError("V7 settings must be [B,H,A] or [B,C,H,A]")

    def rainfall_features(self, rainfall: torch.Tensor) -> torch.Tensor:
        if rainfall.ndim != 4:
            raise ValueError("V7 rainfall must be [B,H,N,R]")
        batch, horizon, _, rain_dim = rainfall.shape
        expected_horizon = int(self.weights.shape[0]) * self.control_block_steps
        if horizon != expected_horizon:
            raise ValueError("V7 rainfall/action horizons differ")
        rain = rainfall.mean(dim=2)
        rain = rain.reshape(
            batch,
            int(self.weights.shape[0]),
            self.control_block_steps,
            rain_dim,
        ).mean(dim=2)
        projected = torch.einsum("tk,btr->bkr", self.weights.to(rain), rain)
        return projected.reshape(batch, -1)


class ControlValueSurrogateV70(nn.Module):
    """Direct signed Delta-TFV model for candidate scoring and differentiable MPC."""

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
        contract: DirectValueLossContractV70 = DirectValueLossContractV70(),
    ) -> None:
        super().__init__()
        contract.validate()
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
        self.direct_head = _bias_free_head(4 * hidden_dim, hidden_dim)
        scale = max(float(tfv_scale_m3), 1.0)
        self.register_buffer("tfv_scale_m3", torch.tensor(scale, dtype=torch.float32))
        for module in self.direct_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.35)

    def _global_boundary(self, initial_state: torch.Tensor, rainfall: torch.Tensor) -> torch.Tensor:
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
    ) -> torch.Tensor:
        if previous_actuator_flow.ndim != 2:
            raise ValueError("V7 previous_actuator_flow must be [B,A]")
        batch = initial_state.shape[0]
        if previous_actuator_flow.shape != (batch, self.actuator_count):
            raise ValueError("V7 current actuator-flow vector is misaligned")
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
        return torch.cat(
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
            raise ValueError("V7 candidate settings must be [B,C,H,A]")
        batch, candidates, _, actuators = candidate_settings.shape
        if actuators != self.actuator_count:
            raise ValueError("V7 actuator count mismatch")
        base = self._base_actuator_features(
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

        summed = effect.sum(dim=2) / max(float(self.actuator_count) ** 0.5, 1.0)
        mean = effect.mean(dim=2)
        maximum = effect.amax(dim=2)
        sum_raw = effect.sum(dim=2)
        sum_sq = torch.square(effect).sum(dim=2)
        pair = (torch.square(sum_raw) - sum_sq) / max(
            float(self.actuator_count * max(self.actuator_count - 1, 1)) ** 0.5,
            1.0,
        )
        pooled = torch.cat((summed, mean, maximum, pair), dim=-1)
        normalized = self.direct_head(pooled).squeeze(-1)
        limit = float(self.contract.transformed_limit)
        normalized = limit * torch.tanh(normalized / limit)
        delta_tfv = self.tfv_scale_m3.to(normalized) * torch.sinh(normalized)

        # Fail closed on the exact counterfactual identity. This is a structural
        # contract, not a learned shortcut; all non-identical candidate gradients
        # remain unchanged.
        same_action = torch.all(
            candidate_settings == reference_expanded, dim=(2, 3)
        )
        normalized = torch.where(same_action, torch.zeros_like(normalized), normalized)
        delta_tfv = torch.where(same_action, torch.zeros_like(delta_tfv), delta_tfv)
        effect = torch.where(
            same_action[..., None, None], torch.zeros_like(effect), effect
        )
        pooled = torch.where(
            same_action[..., None], torch.zeros_like(pooled), pooled
        )
        return DirectValueOutputV70(
            delta_tfv_m3=delta_tfv,
            normalized_delta_tfv=normalized,
            actuator_effect_tokens=effect,
            pooled_interaction=pooled,
        )

    def forward_coefficients(
        self,
        *,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        coefficients: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV60,
        basis: ControlBasisV60,
    ) -> DirectValueOutputV70:
        reference = reference_settings[:, None].expand(
            coefficients.shape[0], coefficients.shape[1], -1, -1
        )
        candidate = basis.decode(reference, coefficients)
        return self(
            initial_state,
            rainfall,
            reference_settings,
            candidate,
            previous_actuator_flow,
            prepared,
        )


class HydraulicResponseSurrogateV70(HydraulicResponseSurrogateV60):
    """V7 keeps the validated decoder but changes supervision to hydraulic effects."""


class DualStep2SurrogateV70(nn.Module):
    def __init__(
        self,
        value_model: ControlValueSurrogateV70,
        hydraulic_model: HydraulicResponseSurrogateV70,
    ) -> None:
        super().__init__()
        self.value = value_model
        self.hydraulic = hydraulic_model

    def assert_disjoint_parameters(self) -> None:
        if {id(p) for p in self.value.parameters()} & {
            id(p) for p in self.hydraulic.parameters()
        }:
            raise RuntimeError("V7 value/hydraulic parameter sets must be disjoint")


__all__ = [
    "ControlValueSurrogateV70",
    "DirectValueOutputV70",
    "DualStep2SurrogateV70",
    "HydraulicResponseSurrogateV70",
    "TemporalActionProjectorV70",
    "prepare_static_v60",
]
