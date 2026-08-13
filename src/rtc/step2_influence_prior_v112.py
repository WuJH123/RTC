"""Soft D2 influence prior for Project7 Step2 V11.2.

The prior stores only response *support* frequencies. It never stores or sums
D2 hydraulic magnitudes, so multi-actuator truth must still be learned directly
from authoritative D3 joint branches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .step2_influence_support_v112 import InfluenceSupportTargetsV112, source_flow_effective_v112

V112_INFLUENCE_PRIOR_CONTRACT = "PROJECT7_STEP2_V112_SOFT_INFLUENCE_PRIOR_V1"


@dataclass
class InfluencePriorAccumulatorV112:
    actuator_count: int
    retained_count: int
    node_count: int
    channel_count: int = 5

    def __post_init__(self) -> None:
        self.exposure = np.zeros(self.actuator_count, dtype=np.int64)
        self.state_support_count = np.zeros(
            (self.actuator_count, self.retained_count, self.node_count, self.channel_count),
            dtype=np.uint32,
        )
        self.flow_support_count = np.zeros(
            (self.actuator_count, self.retained_count, self.actuator_count), dtype=np.uint32
        )
        self.source_flow_effective_count = np.zeros(self.actuator_count, dtype=np.uint32)

    def update(self, source_actuator: int, targets: InfluenceSupportTargetsV112) -> None:
        a = int(source_actuator)
        if a < 0 or a >= self.actuator_count:
            raise IndexError("V112 source actuator outside prior accumulator")
        if targets.state_active.shape != (self.retained_count, self.node_count, self.channel_count):
            raise ValueError("V112 state-support shape mismatch")
        if targets.flow_active.shape != (self.retained_count, self.actuator_count):
            raise ValueError("V112 flow-support shape mismatch")
        self.exposure[a] += 1
        self.state_support_count[a] += targets.state_active.astype(np.uint32)
        self.flow_support_count[a] += targets.flow_active.astype(np.uint32)
        self.source_flow_effective_count[a] += np.uint32(source_flow_effective_v112(targets, a))

    def finalize(self, *, alpha: float = 0.5) -> dict[str, np.ndarray]:
        """Jeffreys-smoothed probabilities for observed source actuators.

        Non-zero smoothing is a soft global escape: an observed actuator-node
        pair is never declared physically impossible. Unobserved actuators stay
        NaN so downstream code must fail closed instead of inventing evidence.
        """
        if alpha <= 0:
            raise ValueError("V112 smoothing alpha must be positive")
        state = np.full(self.state_support_count.shape, np.nan, dtype=np.float32)
        flow = np.full(self.flow_support_count.shape, np.nan, dtype=np.float32)
        source = np.full(self.exposure.shape, np.nan, dtype=np.float32)
        for a, n in enumerate(self.exposure.tolist()):
            if n <= 0:
                continue
            denominator = float(n + 2.0 * alpha)
            state[a] = (self.state_support_count[a].astype(np.float64) + alpha) / denominator
            flow[a] = (self.flow_support_count[a].astype(np.float64) + alpha) / denominator
            source[a] = float(self.source_flow_effective_count[a] + alpha) / denominator
        return {
            "contract": np.asarray(V112_INFLUENCE_PRIOR_CONTRACT),
            "state_support_probability": state,
            "flow_support_probability": flow,
            "source_flow_effective_probability": source,
            "exposure": self.exposure.copy(),
        }


def combine_support_proposals_noisy_or_v112(
    support_probability: np.ndarray,
    active_source_mask: np.ndarray,
) -> np.ndarray:
    """Union single-actuator D2 support for a joint action, without magnitude addition."""
    prior = np.asarray(support_probability, dtype=np.float64)
    active = np.asarray(active_source_mask, dtype=bool)
    if prior.ndim != 4:
        raise ValueError("V112 support prior must be [A,T,N,K]")
    if active.ndim < 2 or active.shape[-2:] != (prior.shape[1], prior.shape[0]):
        raise ValueError("V112 active-source mask must end in [T,A]")
    if np.isnan(prior).any():
        raise ValueError("V112 support prior contains unobserved actuator entries")
    prior = np.clip(prior, 1.0e-6, 1.0 - 1.0e-6)
    out = np.zeros(active.shape[:-2] + (prior.shape[1], prior.shape[2], prior.shape[3]), dtype=np.float64)
    for t in range(prior.shape[1]):
        log_not = np.log1p(-prior[:, t])
        summed = np.einsum("...a,ank->...nk", active[..., t, :].astype(np.float64), log_not)
        proposal = 1.0 - np.exp(summed)
        no_action = ~active[..., t, :].any(axis=-1)
        out[..., t, :, :] = np.where(no_action[..., None, None], 0.0, proposal)
    return out.astype(np.float32)


__all__ = [
    "InfluencePriorAccumulatorV112", "V112_INFLUENCE_PRIOR_CONTRACT",
    "combine_support_proposals_noisy_or_v112",
]
