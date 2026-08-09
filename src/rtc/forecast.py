from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersistenceDecayForecast:
    """A deliberately causal rainfall forecaster for RTC development.

    It uses only rainfall observed up to the decision time. Event IDs and future realised
    rainfall are never inputs. More sophisticated operational forecasts can replace this
    class later as long as they obey the same interface and are frozen before Policy Lock.
    """

    decay_per_step: float = 0.92
    scenario_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25)
    history_steps_for_level: int = 3

    def forecast(self, observed_history: np.ndarray, *, horizon_steps: int) -> np.ndarray:
        history = np.asarray(observed_history, dtype=float)
        if history.ndim < 1 or history.shape[0] < 1:
            raise ValueError("observed rainfall history is required")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if not 0.0 <= self.decay_per_step <= 1.0:
            raise ValueError("decay_per_step must lie in [0,1]")
        tail = history[-min(self.history_steps_for_level, history.shape[0]) :]
        level = np.clip(tail, 0.0, None).mean(axis=0)
        steps = np.arange(horizon_steps, dtype=float)
        base = np.stack([level * (self.decay_per_step**k) for k in steps], axis=0)
        scenarios = [base * float(multiplier) for multiplier in self.scenario_multipliers]
        return np.stack(scenarios, axis=0)
