"""Bounded causal-history endpoint diagnostic for V9 hydraulic effects.

This is deliberately *not* a replacement Step2 model.  It is a fixed-capacity,
TrainFit-D2-only control that asks whether a 60-minute causal history changes
learnability of the signed response at a changed actuator's two endpoints.  Its
only permissible history sources are explicitly labelled ``none``,
``frozen_step1_reconstruction`` (online-eligible), and ``oracle_past_swmm``
(diagnostic-only).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch import nn


HISTORY_LADDER_CONTRACT_V90 = "PROJECT7_STEP2_V90_CAUSAL_HISTORY_LADDER_V1"
HISTORY_FRAMES_V90 = 13
HISTORY_ENDPOINT_FEATURE_DIM_V90 = 13  # upstream state 6 + downstream state 6 + q_actuator

HistorySourceV90 = Literal["none", "frozen_step1_reconstruction", "oracle_past_swmm"]


def history_source_contract_v90(source: HistorySourceV90 | str) -> dict[str, bool | str]:
    """Return the immutable eligibility semantics for one diagnostic source."""
    value = str(source)
    contracts = {
        "none": {
            "source_type": "none",
            "online_eligible": True,
            "oracle_diagnostic_only": False,
        },
        "frozen_step1_reconstruction": {
            "source_type": "frozen_step1_reconstruction",
            "online_eligible": True,
            "oracle_diagnostic_only": False,
        },
        "oracle_past_swmm": {
            "source_type": "oracle_past_swmm",
            "online_eligible": False,
            "oracle_diagnostic_only": True,
        },
    }
    if value not in contracts:
        raise ValueError(f"unknown V9 history source: {source}")
    return dict(contracts[value])


def endpoint_history_features_v90(
    states_physical: np.ndarray,
    actuator_flows_physical: np.ndarray,
    *,
    actuator_index: int,
    upstream_index: int,
    downstream_index: int,
) -> np.ndarray:
    """Select one actuator's causal endpoint history without any future frame.

    ``states_physical`` must already be the fixed ``[t-3600,...,t]`` history
    returned by :mod:`rtc.step2_history_v90`.  This helper cannot choose future
    timestamps; it only projects an admitted history to the same local fields
    used by the bounded diagnostic.
    """
    states = np.asarray(states_physical, dtype=np.float32)
    flows = np.asarray(actuator_flows_physical, dtype=np.float32)
    if states.shape != (HISTORY_FRAMES_V90, states.shape[1], 6):
        raise ValueError("history states must be [13,node,6]")
    if flows.ndim != 2 or flows.shape[0] != HISTORY_FRAMES_V90:
        raise ValueError("history actuator flows must be [13,actuator]")
    nodes, actuators = int(states.shape[1]), int(flows.shape[1])
    if not 0 <= int(actuator_index) < actuators:
        raise ValueError("actuator index is outside history actuator dimension")
    if not 0 <= int(upstream_index) < nodes or not 0 <= int(downstream_index) < nodes:
        raise ValueError("endpoint index is outside history node dimension")
    out = np.concatenate(
        (
            states[:, int(upstream_index), :],
            states[:, int(downstream_index), :],
            flows[:, int(actuator_index) : int(actuator_index) + 1],
        ),
        axis=-1,
    )
    if out.shape != (HISTORY_FRAMES_V90, HISTORY_ENDPOINT_FEATURE_DIM_V90):
        raise RuntimeError("endpoint history feature dimension is not fixed")
    if not np.isfinite(out).all():
        raise ValueError("endpoint history contains non-finite values")
    return out.astype(np.float32, copy=False)


class CausalEndpointHistoryEncoderV90(nn.Module):
    """A small fixed causal temporal encoder shared by endpoint histories.

    The convolution is left-padded then cropped, hence no output at a history
    frame depends on a later history frame.  Subtracting the all-zero response
    makes absence of history structurally exact-zero despite trainable biases.
    """

    output_dim = 40

    def __init__(self) -> None:
        super().__init__()
        self.state_conv = nn.Conv1d(6, 8, kernel_size=3, padding=2)
        self.flow_conv = nn.Conv1d(1, 4, kernel_size=3, padding=2)

    @staticmethod
    def _causal_crop(value: torch.Tensor) -> torch.Tensor:
        return value[..., :HISTORY_FRAMES_V90]

    def _encode_raw(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (
            HISTORY_FRAMES_V90,
            HISTORY_ENDPOINT_FEATURE_DIM_V90,
        ):
            raise ValueError("history encoder expects [row,13,13]")
        upstream = history[..., :6].transpose(1, 2)
        downstream = history[..., 6:12].transpose(1, 2)
        flow = history[..., 12:].transpose(1, 2)

        def state_features(value: torch.Tensor) -> torch.Tensor:
            encoded = torch.nn.functional.silu(self._causal_crop(self.state_conv(value)))
            return torch.cat((encoded[..., -1], encoded.mean(dim=-1)), dim=-1)

        encoded_flow = torch.nn.functional.silu(self._causal_crop(self.flow_conv(flow)))
        flow_features = torch.cat((encoded_flow[..., -1], encoded_flow.mean(dim=-1)), dim=-1)
        return torch.cat((state_features(upstream), state_features(downstream), flow_features), dim=-1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_raw(history)
        # Keep the zero-history source semantically absent, not merely small.
        zero = self._encode_raw(torch.zeros_like(history))
        return encoded - zero


class LocalHistoryEffectModelV90(nn.Module):
    """Fixed endpoint-local direct signed-effect diagnostic model.

    ``forward`` uses structural subtraction at zero action.  It does not contain
    a graph path, future target, candidate ID, recurrent free-run, or any SWMM
    output.  Every A/B1/C1 arm instantiates exactly this same capacity.
    """

    def __init__(
        self,
        *,
        base_feature_dim: int,
        action_feature_indices: tuple[int, int],
        action_zero_values: tuple[float, float] = (0.0, 0.0),
        output_dim: int,
    ) -> None:
        super().__init__()
        if base_feature_dim <= 0 or output_dim <= 0:
            raise ValueError("local history diagnostic dimensions must be positive")
        if any(index < 0 or index >= base_feature_dim for index in action_feature_indices):
            raise ValueError("action feature indices are outside the base feature schema")
        self.base_feature_dim = int(base_feature_dim)
        self.action_feature_indices = tuple(int(index) for index in action_feature_indices)
        self.register_buffer(
            "action_zero_values",
            torch.as_tensor(action_zero_values, dtype=torch.float32).reshape(2),
            persistent=False,
        )
        self.history_encoder = CausalEndpointHistoryEncoderV90()
        self.network = nn.Sequential(
            nn.Linear(self.base_feature_dim + self.history_encoder.output_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, int(output_dim)),
        )

    def forward(self, base_features: torch.Tensor, history_features: torch.Tensor) -> torch.Tensor:
        if base_features.ndim != 2 or base_features.shape[1] != self.base_feature_dim:
            raise ValueError("base local features have an incompatible shape")
        encoded = self.history_encoder(history_features)
        full = torch.cat((base_features, encoded), dim=-1)
        zero_action = base_features.clone()
        # Base features are normalized by the caller.  Use the normalized values
        # corresponding to *physical* delta-u=0 rather than assuming zero happens
        # to be the empirical mean of a nonzero candidate corpus.
        zero_action[:, list(self.action_feature_indices)] = self.action_zero_values.to(base_features)
        baseline = torch.cat((zero_action, encoded), dim=-1)
        # Exact shared-operator subtraction establishes zero effect at zero delta-u.
        return self.network(full) - self.network(baseline)


def _rmse(predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    if mask is not None:
        error = error[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan")


def history_reconstruction_metrics_v90(
    predicted_states: np.ndarray,
    oracle_states: np.ndarray,
    *,
    max_depth_m: np.ndarray,
    priority_indices: np.ndarray,
) -> dict[str, float | int]:
    """Compare frozen-Step1 history with oracle past truth; no online substitution."""
    predicted = np.asarray(predicted_states, dtype=np.float32)
    oracle = np.asarray(oracle_states, dtype=np.float32)
    if predicted.shape != oracle.shape or predicted.ndim != 4 or predicted.shape[-1] != 6:
        raise ValueError("history reconstruction requires [sample,13,node,6] arrays")
    if predicted.shape[1] != HISTORY_FRAMES_V90:
        raise ValueError("history reconstruction requires exactly 13 causal frames")
    depths = oracle[..., 0]
    max_depth = np.asarray(max_depth_m, dtype=np.float32).reshape(1, 1, -1)
    if max_depth.shape[-1] != oracle.shape[2]:
        raise ValueError("max_depth schema differs from history node dimension")
    normalized_depth = depths / np.maximum(max_depth, 1e-6)
    wet = normalized_depth >= 0.25
    high = normalized_depth >= 0.75
    priority = np.asarray(priority_indices, dtype=np.int64).reshape(-1)
    if priority.size and (priority.min() < 0 or priority.max() >= oracle.shape[2]):
        raise ValueError("priority index lies outside history node dimension")
    result: dict[str, float | int] = {
        "sample_count": int(predicted.shape[0]),
        "node_count": int(predicted.shape[2]),
        "depth_rmse_m": _rmse(predicted[..., 0], oracle[..., 0]),
        "wet_depth_rmse_m": _rmse(predicted[..., 0], oracle[..., 0], wet),
        "high_depth_rmse_m": _rmse(predicted[..., 0], oracle[..., 0], high),
        "priority_depth_rmse_m": _rmse(
            predicted[:, :, priority, 0], oracle[:, :, priority, 0]
        ) if priority.size else float("nan"),
        "flooding_rate_rmse_m3s": _rmse(predicted[..., 2], oracle[..., 2]),
        "storage_volume_rmse_m3": _rmse(predicted[..., 3], oracle[..., 3]),
        "wet_node_count": int(wet.sum()),
        "high_depth_node_count": int(high.sum()),
        "priority_node_count": int(priority.size),
    }
    return result


__all__ = [
    "CausalEndpointHistoryEncoderV90",
    "HISTORY_ENDPOINT_FEATURE_DIM_V90",
    "HISTORY_FRAMES_V90",
    "HISTORY_LADDER_CONTRACT_V90",
    "LocalHistoryEffectModelV90",
    "endpoint_history_features_v90",
    "history_reconstruction_metrics_v90",
    "history_source_contract_v90",
]
