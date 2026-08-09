from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch

from .closed_loop import CausalObservation, ControllerAction
from .context_features import build_node_context
from .forecast import PersistenceDecayForecast
from .graph import GraphSchema
from .models import SparseStateEstimator
from .runtime import choose_first_move, verify_setting_readback


class MPCProtocol(Protocol):
    model: torch.nn.Module

    def optimize(self, **kwargs): ...


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
    decision_runtime_budget_seconds: float | None = None
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
        if self.decision_runtime_budget_seconds is not None and self.decision_runtime_budget_seconds <= 0:
            raise ValueError("decision_runtime_budget_seconds must be positive when supplied")


def hold_current_fallback(observation: CausalObservation, horizon_steps: int) -> np.ndarray:
    current = np.asarray(observation.actuator_current_setting, dtype=float).reshape(-1)
    return np.repeat(current[None, :], int(horizon_steps), axis=0)


class TorchMPCController:
    """Step1 -> causal rainfall forecast -> Step2/MPC -> executable first move.

    Step1 receives node-local rainfall/actuator context rather than broadcasting the full
    actuator vector to every node. Priority PFV remains a soft optimizer preference. Runtime
    fallback is reserved for history/readback/numerical/deadline failures. The optional wall-
    clock budget turns the simulation controller into a meaningful real-time implementation
    contract: a stale optimization result is never executed after its frozen deadline.
    """

    def __init__(
        self,
        *,
        step1: SparseStateEstimator,
        mpc: MPCProtocol,
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
        self.last_observed_elapsed_seconds: int | None = None
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

    def observe(self, obs: CausalObservation) -> None:
        self._validate_observation(obs)
        if self.last_observed_elapsed_seconds == obs.elapsed_seconds:
            return
        if self.last_observed_elapsed_seconds is not None and obs.elapsed_seconds <= self.last_observed_elapsed_seconds:
            raise ValueError("controller observations must be strictly increasing in time")
        n = len(self.graph.node_ids)
        observed = np.zeros((n, 2), dtype=np.float32)
        mask = np.zeros((n, 2), dtype=np.float32)
        observed[self.sensor_index, 0] = np.asarray(obs.sensor_depth_m, dtype=np.float32)
        observed[self.sensor_index, 1] = np.asarray(obs.sensor_head_m, dtype=np.float32)
        mask[self.sensor_index, :] = 1.0
        rain = np.asarray(obs.observed_rainfall_mmhr, dtype=np.float32).reshape(n, 1)
        context = build_node_context(
            rainfall_mmhr=rain,
            actuator_setting=np.asarray(obs.actuator_current_setting, dtype=np.float32),
            actuator_flow_m3s=np.asarray(obs.actuator_flow_m3s, dtype=np.float32),
            actuator_upstream=self.graph.actuator_upstream,
            actuator_downstream=self.graph.actuator_downstream,
            node_count=n,
        )
        self.observed_history.append(observed)
        self.mask_history.append(mask)
        self.context_history.append(context)
        self.rainfall_history.append(rain)
        self.last_observed_elapsed_seconds = obs.elapsed_seconds

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

    def decide(self, obs: CausalObservation, *, observation_already_recorded: bool = False) -> ControllerAction:
        if not observation_already_recorded:
            self.observe(obs)
        else:
            self._validate_observation(obs)
            if self.last_observed_elapsed_seconds != obs.elapsed_seconds:
                raise ValueError("decision observation was not recorded at this model step")
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

        started = time.perf_counter()
        try:
            static = torch.as_tensor(
                self.graph.static_node_features, dtype=torch.float32, device=self.device
            )
            edges = torch.as_tensor(self.graph.edge_index, dtype=torch.long, device=self.device)
            with torch.no_grad():
                initial_state = self.step1(
                    torch.as_tensor(np.stack(self.observed_history)[None], dtype=torch.float32, device=self.device),
                    torch.as_tensor(np.stack(self.mask_history)[None], dtype=torch.float32, device=self.device),
                    static,
                    edges,
                    torch.as_tensor(np.stack(self.context_history)[None], dtype=torch.float32, device=self.device),
                )
            rainfall_scenarios = self.forecast.forecast(
                np.stack(self.rainfall_history), horizon_steps=self.config.horizon_steps
            )
            result = self.mpc.optimize(
                initial_state=initial_state,
                rainfall_scenarios=torch.as_tensor(
                    rainfall_scenarios, dtype=torch.float32, device=self.device
                ),
                current_settings=torch.as_tensor(current, dtype=torch.float32, device=self.device),
                fallback_settings=torch.as_tensor(fallback[None], dtype=torch.float32, device=self.device),
                previous_actuator_flow=torch.as_tensor(
                    obs.actuator_flow_m3s[None], dtype=torch.float32, device=self.device
                ),
                actuator_upstream=torch.as_tensor(
                    self.graph.actuator_upstream, dtype=torch.long, device=self.device
                ),
                actuator_downstream=torch.as_tensor(
                    self.graph.actuator_downstream, dtype=torch.long, device=self.device
                ),
                actuator_physics=torch.as_tensor(
                    self.graph.actuator_physics[None], dtype=torch.float32, device=self.device
                ),
                static_node_features=static,
                edge_index=edges,
                iterations=self.config.optimizer_iterations,
                learning_rate=self.config.optimizer_learning_rate,
                control_block_steps=self.config.control_block_steps,
            )
            candidate_valid = bool(
                getattr(result, "candidate_valid", getattr(result, "admissible", False))
            )
            decision = choose_first_move(
                optimized_sequence=result.settings.detach().cpu().numpy(),
                surrogate_admissible=candidate_valid,
                fallback_first_move=fallback[0],
                current_settings=current,
                min_settings=0.0,
                max_settings=1.0,
                max_delta_per_update=self.config.max_setting_delta_per_update,
            )
            runtime_seconds = float(time.perf_counter() - started)
            budget = self.config.decision_runtime_budget_seconds
            if budget is not None and runtime_seconds > budget:
                self.last_requested = fallback[0].copy()
                return ControllerAction(
                    settings=dict(zip(self.graph.actuator_ids, fallback[0], strict=True)),
                    source="FALLBACK_COMPUTE_DEADLINE",
                    diagnostics={
                        "fallback_policy": self.config.fallback_policy_id,
                        "decision_runtime_seconds": runtime_seconds,
                        "decision_runtime_budget_seconds": float(budget),
                    },
                )
            self.last_requested = decision.requested.copy()
            diagnostics: dict[str, float | int | bool | str] = {
                "fallback_policy": self.config.fallback_policy_id,
                "control_block_steps": self.config.control_block_steps,
                "candidate_valid": candidate_valid,
                "tfv_risk_m3": float(result.tfv_risk_m3),
                "projected_first_move": decision.projected,
                "decision_runtime_seconds": runtime_seconds,
                "decision_runtime_budget_seconds": float(budget) if budget is not None else -1.0,
            }
            for name in (
                "primary_tfv_reference_m3",
                "priority_positive_flood_deterioration_m3",
                "worst_site_flood_deterioration_m3",
                "worst_site_depth_deterioration_m",
                "tfv_near_opt_excess_m3",
            ):
                if hasattr(result, name):
                    diagnostics[name] = float(getattr(result, name))
            return ControllerAction(
                settings=dict(zip(self.graph.actuator_ids, decision.requested, strict=True)),
                source=decision.source,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            runtime_seconds = float(time.perf_counter() - started)
            self.last_requested = fallback[0].copy()
            return ControllerAction(
                settings=dict(zip(self.graph.actuator_ids, fallback[0], strict=True)),
                source="FALLBACK_RUNTIME_ERROR",
                diagnostics={
                    "fallback_policy": self.config.fallback_policy_id,
                    "decision_runtime_seconds": runtime_seconds,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        return self.decide(obs)
