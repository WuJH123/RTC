from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TimeScaleConfig:
    """MPC time-scale parameters.

    These are experiment outputs/configuration, not immutable scientific constants.
    """

    model_step_minutes: int = 10
    control_update_minutes: int = 10
    prediction_horizon_minutes: int = 120
    control_block_minutes: tuple[int, ...] = (10, 10, 10, 30, 60)

    def validate(self) -> None:
        if min(self.model_step_minutes, self.control_update_minutes) <= 0:
            raise ValueError("time steps must be positive")
        if self.prediction_horizon_minutes <= 0:
            raise ValueError("prediction horizon must be positive")
        if sum(self.control_block_minutes) != self.prediction_horizon_minutes:
            raise ValueError("control blocks must sum to the prediction horizon")
        if any(x <= 0 for x in self.control_block_minutes):
            raise ValueError("control blocks must be positive")


@dataclass(frozen=True)
class PrioritySafetyContract:
    """Safety configuration for the eight observed ponding sites.

    Budgets are deliberately supplied from an independent calibration artefact rather
    than embedded as scientific constants in model code.
    """

    priority_nodes: tuple[str, ...]
    priority_flood_budget_m3: float
    priority_depth_budget_m: float
    nonpriority_new_flood_budget_m3: float | None = None
    quantile: float = 0.95

    def validate(self) -> None:
        if not self.priority_nodes:
            raise ValueError("at least one priority node is required")
        if len(set(self.priority_nodes)) != len(self.priority_nodes):
            raise ValueError("priority node IDs must be unique")
        if self.priority_flood_budget_m3 < 0 or self.priority_depth_budget_m < 0:
            raise ValueError("calibrated safety budgets must be non-negative")
        if self.nonpriority_new_flood_budget_m3 is not None:
            if self.nonpriority_new_flood_budget_m3 < 0:
                raise ValueError("new-flood budget must be non-negative")
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("one-sided safety quantile must lie in (0.5, 1)")


def load_priority_nodes(path: str | Path) -> tuple[str, ...]:
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    values = [v for v in values if v and not v.startswith("#")]
    if not values:
        raise ValueError(f"no priority nodes found in {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate priority nodes found in {path}")
    return tuple(values)


def require_nodes_exist(priority_nodes: Sequence[str], node_ids: Sequence[str]) -> None:
    missing = sorted(set(priority_nodes) - set(node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from INP: {missing}")
