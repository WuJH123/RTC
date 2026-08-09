from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class SafetyAuditReport:
    decisions: int
    admitted_decisions: int
    fallback_decisions: int
    false_safe_decisions: int
    false_safe_rate_given_admitted: float
    event_balanced_false_safe_rate: float
    sitewise_flood_coverage: tuple[float, ...]
    sitewise_depth_coverage: tuple[float, ...]
    passed: bool

    def to_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_selected_action_safety(
    *,
    event_ids: np.ndarray,
    admitted: np.ndarray,
    fallback_used: np.ndarray,
    predicted_flood_ucb_m3: np.ndarray,
    true_flood_deterioration_m3: np.ndarray,
    flood_budget_m3: np.ndarray,
    predicted_depth_ucb_m: np.ndarray,
    true_depth_deterioration_m: np.ndarray,
    depth_budget_m: np.ndarray,
    maximum_false_safe_rate: float = 0.01,
    minimum_sitewise_coverage: float = 0.95,
) -> SafetyAuditReport:
    """Audit exact selected decisions on a rainfall partition not used for calibration.

    A false-safe decision is one admitted by the online safety rule whose authoritative
    SWMM outcome violates at least one site's frozen engineering deterioration budget.
    This is deliberately separate from fitting the calibration quantiles.
    """

    events = np.asarray(event_ids).astype(str).reshape(-1)
    admitted_arr = np.asarray(admitted, dtype=bool).reshape(-1)
    fallback = np.asarray(fallback_used, dtype=bool).reshape(-1)
    pred_v = np.asarray(predicted_flood_ucb_m3, dtype=float)
    true_v = np.asarray(true_flood_deterioration_m3, dtype=float)
    pred_h = np.asarray(predicted_depth_ucb_m, dtype=float)
    true_h = np.asarray(true_depth_deterioration_m, dtype=float)
    if pred_v.shape != true_v.shape or pred_h.shape != true_h.shape or pred_v.shape != pred_h.shape:
        raise ValueError("safety arrays must have identical [decision,site] shape")
    if pred_v.ndim != 2 or pred_v.shape[0] != events.size:
        raise ValueError("safety arrays must align with event_ids")
    if admitted_arr.size != events.size or fallback.size != events.size:
        raise ValueError("decision flags must align with event_ids")
    flood_budget = np.broadcast_to(np.asarray(flood_budget_m3, dtype=float).reshape(-1), (pred_v.shape[1],))
    depth_budget = np.broadcast_to(np.asarray(depth_budget_m, dtype=float).reshape(-1), (pred_v.shape[1],))

    # Check that an action labeled admitted actually passed the *predicted* online rule.
    predicted_pass = (pred_v <= flood_budget).all(axis=1) & (pred_h <= depth_budget).all(axis=1)
    if np.any(admitted_arr & ~predicted_pass):
        raise ValueError("admitted flag disagrees with recorded online safety UCB")

    truth_pass = (true_v <= flood_budget).all(axis=1) & (true_h <= depth_budget).all(axis=1)
    false_safe = admitted_arr & ~truth_pass
    admitted_count = int(admitted_arr.sum())
    false_safe_rate = float(false_safe.sum() / admitted_count) if admitted_count else 0.0

    event_rates: list[float] = []
    for event in sorted(set(events.tolist())):
        mask = (events == event) & admitted_arr
        if mask.any():
            event_rates.append(float(false_safe[mask].mean()))
    event_balanced = float(np.mean(event_rates)) if event_rates else 0.0

    # Empirical coverage of the independently calibrated model-error UCB itself.
    flood_coverage = tuple(float(x) for x in np.mean(true_v <= pred_v, axis=0))
    depth_coverage = tuple(float(x) for x in np.mean(true_h <= pred_h, axis=0))
    passed = bool(
        false_safe_rate <= maximum_false_safe_rate
        and event_balanced <= maximum_false_safe_rate
        and min(flood_coverage, default=1.0) >= minimum_sitewise_coverage
        and min(depth_coverage, default=1.0) >= minimum_sitewise_coverage
    )
    return SafetyAuditReport(
        decisions=int(events.size),
        admitted_decisions=admitted_count,
        fallback_decisions=int(fallback.sum()),
        false_safe_decisions=int(false_safe.sum()),
        false_safe_rate_given_admitted=false_safe_rate,
        event_balanced_false_safe_rate=event_balanced,
        sitewise_flood_coverage=flood_coverage,
        sitewise_depth_coverage=depth_coverage,
        passed=passed,
    )
