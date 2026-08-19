"""Offline controller wrapper for paired receding-policy return SWMM branches."""
from __future__ import annotations

import gc
from typing import Mapping

import numpy as np
import torch

from .closed_loop import CausalObservation, ControllerAction


POLICY_RETURN_REPLAY_CONTRACT = "PROJECT7_RECEDING_POLICY_RETURN_EXACT_PREFIX_REPLAY_V2_CAUSAL_CONTEXT"
POLICY_RETURN_BRANCH_RELEASE_CONTRACT = (
    "PROJECT7_POLICY_RETURN_BRANCH_GPU_LIFECYCLE_RELEASE_V1"
)


def _sync_supervisory_latch(controller: object, target: np.ndarray) -> None:
    """Synchronize known controller wrappers after an externally replayed target command."""
    value = np.asarray(target, dtype=float).copy()
    if hasattr(controller, "last_requested"):
        setattr(controller, "last_requested", value.copy())
    if hasattr(controller, "previous_requested"):
        setattr(controller, "previous_requested", value.copy())
    inner = getattr(controller, "controller", None)
    if inner is not None and inner is not controller:
        _sync_supervisory_latch(inner, value)


def _causal_torch_controller(controller: object) -> object:
    """Find the inner Torch controller without depending on a particular wrapper depth."""
    current = controller
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if all(
            hasattr(current, name)
            for name in (
                "step1",
                "graph",
                "forecast",
                "observed_history",
                "mask_history",
                "context_history",
                "rainfall_history",
                "device",
                "config",
            )
        ):
            return current
        next_controller = getattr(current, "controller", None)
        if next_controller is None:
            break
        current = next_controller
    raise TypeError("could not locate causal Torch controller inside replay delegate")


def _capture_online_context(controller: object, obs: CausalObservation) -> dict[str, np.ndarray]:
    """Reproduce exactly the causal Step1/forecast inputs available to the online controller."""
    inner = _causal_torch_controller(controller)
    history_steps = int(inner.config.history_steps)
    if len(inner.observed_history) < history_steps:
        raise RuntimeError("policy-return branch point lacks complete causal Step1 history")
    static = torch.as_tensor(
        inner.graph.static_node_features, dtype=torch.float32, device=inner.device
    )
    edges = torch.as_tensor(inner.graph.edge_index, dtype=torch.long, device=inner.device)
    with torch.no_grad():
        state = inner.step1(
            torch.as_tensor(
                np.stack(inner.observed_history)[None], dtype=torch.float32, device=inner.device
            ),
            torch.as_tensor(
                np.stack(inner.mask_history)[None], dtype=torch.float32, device=inner.device
            ),
            static,
            edges,
            torch.as_tensor(
                np.stack(inner.context_history)[None], dtype=torch.float32, device=inner.device
            ),
        )
    rainfall = inner.forecast.forecast(
        np.stack(inner.rainfall_history), horizon_steps=int(inner.config.horizon_steps)
    )
    return {
        "current_state": state.detach().cpu().numpy()[0].astype(np.float32),
        "rainfall_scenarios": np.asarray(rainfall, dtype=np.float32),
        "active_target": np.asarray(obs.actuator_target_setting, dtype=np.float32).reshape(109),
        "previous_actuator_flow": np.asarray(obs.actuator_flow_m3s, dtype=np.float32).reshape(109),
    }


def snapshot_and_release_policy_return_branch(
    wrapper: "ExactPrefixThenFrozenPolicyController",
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, int | str]]:
    """Copy the causal branch context and release the completed branch controller.

    A replay wrapper owns its complete frozen continuation delegate, including Step1/Step2 models on
    CUDA.  Paired CANDIDATE/HOLD replay is sequential, so retaining the first wrapper while loading the
    second creates two model stacks in GPU memory.  The scientific information needed after a branch
    is only its CPU causal context and lineage/result paths.  Copy that context, sever the delegate
    reference, collect Python cycles, and finally release *unoccupied* allocator cache before the next
    branch is constructed.
    """
    if wrapper.branch_context is None:
        raise RuntimeError("cannot release policy-return branch before causal context capture")
    context = {
        str(key): np.asarray(value).copy()
        for key, value in wrapper.branch_context.items()
    }
    telemetry: dict[str, int | str] = {
        "contract": POLICY_RETURN_BRANCH_RELEASE_CONTRACT,
        "cuda_device_type": str(device.type),
        "allocated_before_bytes": 0,
        "reserved_before_bytes": 0,
        "allocated_after_bytes": 0,
        "reserved_after_bytes": 0,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        telemetry["allocated_before_bytes"] = int(torch.cuda.memory_allocated(device))
        telemetry["reserved_before_bytes"] = int(torch.cuda.memory_reserved(device))

    # This reference is the critical ownership edge: delegate -> wrapped runtime -> CUDA models.
    wrapper.delegate = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        telemetry["allocated_after_bytes"] = int(torch.cuda.memory_allocated(device))
        telemetry["reserved_after_bytes"] = int(torch.cuda.memory_reserved(device))
    return context, telemetry


class ExactPrefixThenFrozenPolicyController:
    """Replay an exact recorded prefix, inject one first action, then resume a frozen policy."""

    def __init__(
        self,
        *,
        delegate: object,
        actuator_ids: tuple[str, ...],
        prefix_actions: Mapping[int, Mapping[str, float]],
        branch_elapsed_seconds: int,
        branch_target: Mapping[str, float],
        branch_kind: str,
    ) -> None:
        if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
            raise ValueError("policy-return replay requires 109 unique actuator IDs")
        self.delegate = delegate
        self.actuator_ids = tuple(actuator_ids)
        self.prefix_actions = {
            int(elapsed): {aid: float(settings[aid]) for aid in self.actuator_ids}
            for elapsed, settings in prefix_actions.items()
        }
        self.branch_elapsed_seconds = int(branch_elapsed_seconds)
        if self.branch_elapsed_seconds < 0:
            raise ValueError("branch elapsed time must be non-negative")
        self.branch_target = {aid: float(branch_target[aid]) for aid in self.actuator_ids}
        if branch_kind not in {"CANDIDATE", "HOLD"}:
            raise ValueError("branch_kind must be CANDIDATE or HOLD")
        self.branch_kind = branch_kind
        self.continuation_started = False
        self.branch_context: dict[str, np.ndarray] | None = None

    def observe(self, obs: CausalObservation) -> None:
        if hasattr(self.delegate, "observe"):
            self.delegate.observe(obs)

    def _external_action(self, target: Mapping[str, float], source: str) -> ControllerAction:
        vector = np.asarray([float(target[aid]) for aid in self.actuator_ids], dtype=float)
        if not np.isfinite(vector).all() or np.any((vector < 0.0) | (vector > 1.0)):
            raise ValueError("policy-return replay target is outside [0,1]")
        _sync_supervisory_latch(self.delegate, vector)
        return ControllerAction(
            settings={aid: float(target[aid]) for aid in self.actuator_ids},
            source=source,
            diagnostics={
                "policy_return_replay_contract": POLICY_RETURN_REPLAY_CONTRACT,
                "policy_return_branch_kind": self.branch_kind,
                "policy_return_external_replay": True,
                "policy_return_continuation_started": False,
            },
        )

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        elapsed = int(obs.elapsed_seconds)
        if elapsed < self.branch_elapsed_seconds:
            if elapsed not in self.prefix_actions:
                raise RuntimeError(f"policy-return replay lacks recorded prefix action at {elapsed}s")
            return self._external_action(
                self.prefix_actions[elapsed], "POLICY_RETURN_EXACT_PREFIX_REPLAY"
            )
        if elapsed == self.branch_elapsed_seconds:
            self.branch_context = _capture_online_context(self.delegate, obs)
            return self._external_action(
                self.branch_target,
                f"POLICY_RETURN_{self.branch_kind}_FIRST_ACTION",
            )
        self.continuation_started = True
        if not hasattr(self.delegate, "decide"):
            raise TypeError("policy-return continuation delegate lacks decide()")
        action = self.delegate.decide(
            obs, observation_already_recorded=observation_already_recorded
        )
        if not isinstance(action, ControllerAction):
            raise TypeError("policy-return continuation delegate must return ControllerAction")
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(
            {
                "policy_return_replay_contract": POLICY_RETURN_REPLAY_CONTRACT,
                "policy_return_branch_kind": self.branch_kind,
                "policy_return_external_replay": False,
                "policy_return_continuation_started": True,
            }
        )
        return ControllerAction(settings=action.settings, source=action.source, diagnostics=diagnostics)


__all__ = [
    "POLICY_RETURN_BRANCH_RELEASE_CONTRACT",
    "POLICY_RETURN_REPLAY_CONTRACT",
    "ExactPrefixThenFrozenPolicyController",
    "snapshot_and_release_policy_return_branch",
]
