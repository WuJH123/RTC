from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .closed_loop import CausalObservation, ControllerAction
from .forecast import PersistenceDecayForecast
from .graph import GraphSchema
from .models import SparseStateEstimator
from .mpc import ContinuousSafetyMPC
from .runtime import choose_first_move, verify_setting_readback


FallbackSequenceProvider = Callable[[CausalObservation, int], np.ndarray]


@dataclass(frozen=True)
class ControllerConfig:
    history_steps: int = 13
    horizon_steps: int = 12
    control_block_steps: int = 1
    optimizer_iterations: int = 120
    optimizer_learning_rate: float = 0.04
    max_setting_delta_per_update: float | None = None
    readback_target_tolerance: float = 1e-6
    readback_current_tolerance: float = 0.05
    fallback_policy_id: str = "CAUSAL_HOLD_CURRENT_V1"

    def validate(self) -> None:
        if self.history_steps < 2 or self.horizon_steps <= 0:
            raise ValueError("history_steps must be >=2 and horizon_steps positive")
        if self.control_block_steps <= 0:
            raise ValueError("control_block_steps must be positive")
        if self.optimizer_iterations <= 0 or self.optimizer_learning_rate <= 0:
            raise ValueError("optimizer settings must be positive")
        if self.max_setting_delta_per_update is not None and self.max_setting_delta_per_update < 0:
            raise ValueError("max_setting_delta_per_update must be non-negative")


def hold_current_fallback(observation: CausalObservation, horizon_steps: int) -> np.ndarray:
    current = np.asarray(observation.actuator_current_setting, dtype=float).reshape(-1)
    return np.repeat(current[None, :], int(horizon_steps), axis=0)


class TorchMPCController:
    """Production adapter for Step1 -> causal forecast -> Step2/MPC -> first move.

    No full-network SWMM state or future realised forcing enters this class. The only
    runtime object accepted is ``CausalObservation`` from ``run_authoritative_closed_loop``.
    Model failures, history warm-up, safety rejection and readback failures all fail closed
    to the configured causal fallback sequence. ``control_block_steps`` is frozen so the
    surrogate assumes the same action-hold duration that the runtime actually executes.
    """

    def __init__(
        self,
        *,
        step1: SparseStateEstimator,
        mpc: ContinuousSafetyMPC,
        graph: GraphSchema,
        sensor_nodes: tuple[str, ...],
        forecast: PersistenceDecayForecast | None = None,
        fallback_sequence_provider: FallbackSequenceProvider | None = None,
        config: ControllerConfig = ControllerConfig(),
        device: str | torch.device | None = None,
    ):
        config.validate()
        self.step1 = step1
        self.mpc = mpc
        self.graph = graph
        self.sensor_nodes = tuple(sensor_nodes)
        self.forecast = forecast or PersistenceDecayForecast()
        self.fallback_sequence_provider = fallback_sequence_provider or hold_current_fallback
        self.config = config
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        missing = sorted(set(sensor_nodes) - set(graph.node_ids))
        if missing:
            raise ValueError(f"controller sensor nodes absent from graph: {missing}")
        self.node_index = {nid: i for i, nid in enumerate(graph.node_ids)}
        self.sensor_index = np.array([self.node_index[n] for n in sensor_nodes], dtype=int)
        self.observed_history: deque[np.ndarray] = deque(maxlen=config.history_steps)
        self.mask_history: deque[np.ndarray] = deque(maxlen=config.history_steps)
        self.context_history: deque[np.ndarray] = deque(maxlen=config.history_steps)
        self.rainfall_history: deque[np.ndarray] = deque(maxlen=config.history_steps)
        self.last_requested: np.ndarray | None = None
        self.step1.to(self.device).eval()
        self.mpc.model.to(self.device).eval()

    def _validate_observation(self, obs: CausalObservation) -> None:
        if tuple(obs.sensor_ids) != self.sensor_nodes:
            raise ValueError("runtime sensor ordering differs from frozen controller schema")
        if tuple(obs.actuator_ids) != self.graph.actuator_ids:
            raise ValueError("runtime actuator ordering differs from frozen graph schema")
        if tuple(obs.rainfall_node_ids) != self.graph.node_ids:
            raise ValueError("runtime rainfall node ordering differs from frozen graph schema")
        if len(obs.sensor_depth_m) != len(self.sensor_nodes) or len(obs.sensor_head_m) != len(self.sensor_nodes):
            raise ValueError("sensor observation length mismatch")
        if len(obs.observed_rainfall_mmhr) != len(self.graph.node_ids):
            raise ValueError("rainfall observation length mismatch")

    def _append_history(self, obs: CausalObservation) -> None:
        n = len(self.graph.node_ids)
        observed = np.zeros((n, 2), dtype=np.float32)
        mask = np.zeros((n, 2), dtype=np.float32)
        observed[self.sensor_index, 0] = np.asarray(obs.sensor_depth_m, dtype=np.float32)
        observed[self.sensor_index, 1] = np.asarray(obs.sensor_head_m, dtype=np.float32)
        mask[self.sensor_index, :] = 1.0
        rain = np.asarray(obs.observed_rainfall_mmhr, dtype=np.float32).reshape(n, 1)
        context = np.concatenate(
            [
                np.array([float(rain.mean()), float(rain.max())], dtype=np.float32),
                np.asarray(obs.actuator_target_setting, dtype=np.float32),
                np.asarray(obs.actuator_current_setting, dtype=np.float32),
                np.asarray(obs.actuator_flow_m3s, dtype=np.float32),
            ]
        )
        self.observed_history.append(observed)
        self.mask_history.append(mask)
        self.context_history.append(context)
        self.rainfall_history.append(rain)

    def _fallback(self, obs: CausalObservation) -> np.ndarray:
        sequence = np.asarray(
            self.fallback_sequence_provider(obs, self.config.horizon_steps), dtype=float
        )
        expected = (self.config.horizon_steps, len(self.graph.actuator_ids))
        if sequence.shape != expected:
            raise ValueError(f"fallback sequence must have shape {expected}, got {sequence.shape}")
        if not np.isfinite(sequence).all() or np.any((sequence < 0.0) | (sequence > 1.0)):
            raise ValueError("fallback sequence contains invalid settings")
        return sequence

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        self._validate_observation(obs)
        self._append_history(obs)
        fallback = self._fallback(obs)
        current = np.asarray(obs.actuator_current_setting, dtype=float).reshape(-1)

        if self.last_requested is not None:
            readback = verify_setting_readback(
                self.last_requested,
                np.asarray(obs.actuator_target_setting, dtype=float),
                current,
                target_tolerance=self.config.readback_target_tolerance,
                current_tolerance=self.config.readback_current_tolerance,
            )
            if not readback.passed:
                self.last_requested = fallback[0].copy()
                return ControllerAction(
                    settings=dict(zip(self.graph.actuator_ids, fallback[0], strict=True)),
                    source="FALLBACK_READBACK",
                    diagnostics={
                        "fallback_policy": self.config.fallback_policy_id,
                        "readback_max_target_error": readback.max_target_error,
                        "readback_max_current_error": readback.max_current_error,
                        "readback_failed_actuators": len(readback.failed_indices),
                    },
                )

        if len(self.observed_history) < self.config.history_steps:
            self.last_requested = fallback[0].copy()
            return ControllerAction(
                settings=dict(zip(self.graph.actuator_ids, fallback[0], strict=True)),
                source="FALLBACK_HISTORY_WARMUP",
                diagnostics={
                    "fallback_policy": self.config.fallback_policy_id,
                    "history_frames": len(self.observed_history),
                    "required_history_frames": self.config.history_steps,
                },
            )

        try:
            observed = torch.as_tensor(
                np.stack(self.observed_history)[None], dtype=torch.float32, device=self.device
            )
            mask = torch.as_tensor(
                np.stack(self.mask_history)[None], dtype=torch.float32, device=self.device
            )
            context = torch.as_tensor(
                np.stack(self.context_history)[None], dtype=torch.float32, device=self.device
            )
            static = torch.as_tensor(
                self.graph.static_node_features, dtype=torch.float32, device=self.device
            )
            edges = torch.as_tensor(self.graph.edge_index, dtype=torch.long, device=self.device)
            with torch.no_grad():
                initial_state = self.step1(observed, mask, static, edges, context)

            rain_history = np.stack(self.rainfall_history)
            rainfall_scenarios = self.forecast.forecast(
                rain_history, horizon_steps=self.config.horizon_steps
            )
            rainfall_tensor = torch.as_tensor(
                rainfall_scenarios, dtype=torch.float32, device=self.device
            )
            current_tensor = torch.as_tensor(current, dtype=torch.float32, device=self.device)
            fallback_tensor = torch.as_tensor(
                fallback[None], dtype=torch.float32, device=self.device
            )
            previous_flow = torch.as_tensor(
                np.asarray(obs.actuator_flow_m3s, dtype=float)[None],
                dtype=torch.float32,
                device=self.device,
            )
            up = torch.as_tensor(
                self.graph.actuator_upstream, dtype=torch.long, device=self.device
            )
            down = torch.as_tensor(
                self.graph.actuator_downstream, dtype=torch.long, device=self.device
            )
            physics = torch.as_tensor(
                self.graph.actuator_physics[None], dtype=torch.float32, device=self.device
            )
            result = self.mpc.optimize(
                initial_state=initial_state,
                rainfall_scenarios=rainfall_tensor,
                current_settings=current_tensor,
                fallback_settings=fallback_tensor,
                previous_actuator_flow=previous_flow,
                actuator_upstream=up,
                actuator_downstream=down,
                actuator_physics=physics,
                static_node_features=static,
                edge_index=edges,
                iterations=self.config.optimizer_iterations,
                learning_rate=self.config.optimizer_learning_rate,
                control_block_steps=self.config.control_block_steps,
            )
            decision = choose_first_move(
                optimized_sequence=result.settings.detach().cpu().numpy(),
                surrogate_admissible=result.admissible,
                fallback_first_move=fallback[0],
                current_settings=current,
                min_settings=0.0,
                max_settings=1.0,
                max_delta_per_update=self.config.max_setting_delta_per_update,
            )
            self.last_requested = decision.requested.copy()
            return ControllerAction(
                settings=dict(zip(self.graph.actuator_ids, decision.requested, strict=True)),
                source=decision.source,
                diagnostics={
                    "fallback_policy": self.config.fallback_policy_id,
                    "control_block_steps": self.config.control_block_steps,
                    "surrogate_admissible": result.admissible,
                    "tfv_risk_m3": result.tfv_risk_m3,
                    "worst_site_flood_deterioration_m3": result.worst_site_flood_deterioration_m3,
                    "worst_site_depth_deterioration_m": result.worst_site_depth_deterioration_m,
                    "max_site_flood_margin_m3": result.max_site_flood_margin_m3,
                    "max_site_depth_margin_m": result.max_site_depth_margin_m,
                    "projected_first_move": decision.projected,
                },
            )
        except Exception as exc:
            self.last_requested = fallback[0].copy()
            return ControllerAction(
                settings=dict(zip(self.graph.actuator_ids, fallback[0], strict=True)),
                source="FALLBACK_RUNTIME_ERROR",
                diagnostics={
                    "fallback_policy": self.config.fallback_policy_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )
