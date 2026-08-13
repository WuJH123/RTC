"""V120-only controller adapter that binds scoring to SWMM target readback.

The shared :class:`TorchMPCController` deliberately keeps its historical MPC
signature.  V120 needs one additional causal quantity before candidate scoring:
the active SWMM ``target_setting``.  This adapter injects that readback into the
value policy without changing legacy MPC, Auto-RBC or EFD behavior.
"""
from __future__ import annotations

import numpy as np
import torch

from .closed_loop import CausalObservation
from .controller import TorchMPCController


class PreviousTargetBoundPolicyV120:
    """Proxy a V120 policy and inject the latest causal target readback."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.model = policy.model
        self._previous_target: np.ndarray | None = None

    def bind_previous_target(self, value: np.ndarray) -> None:
        target = np.asarray(value, dtype=np.float32).reshape(-1)
        if not target.size or not np.isfinite(target).all():
            raise ValueError("V120 active target readback is empty or non-finite")
        self._previous_target = target.copy()

    def optimize(self, **kwargs):
        if self._previous_target is None:
            raise RuntimeError("V120 candidate scoring was called before target readback binding")
        reference = kwargs.get("current_settings")
        if not isinstance(reference, torch.Tensor):
            raise ValueError("V120 controller did not supply current_settings tensor")
        kwargs["previous_requested_settings"] = torch.as_tensor(
            self._previous_target,
            dtype=reference.dtype,
            device=reference.device,
        )
        return self.policy.optimize(**kwargs)


class V120TorchMPCController(TorchMPCController):
    """Bind the decision-epoch target readback before the shared rolling pipeline."""

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ):
        if not isinstance(self.mpc, PreviousTargetBoundPolicyV120):
            raise TypeError("V120TorchMPCController requires PreviousTargetBoundPolicyV120")
        self.mpc.bind_previous_target(obs.actuator_target_setting)
        return super().decide(
            obs, observation_already_recorded=observation_already_recorded
        )


__all__ = ["PreviousTargetBoundPolicyV120", "V120TorchMPCController"]
