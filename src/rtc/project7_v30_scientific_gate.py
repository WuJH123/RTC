"""Fail-closed scientific promotion gate for Project7 V30 and baseline V2.

A controller that compiles or passes engineering checks has not demonstrated the research objective.
This gate requires a *fresh Development-validation* authoritative SWMM panel before a V30 policy may
be considered for a new Policy Lock.  Locked/Final outcomes are explicitly rejected as gate input.

The gate is intentionally performance-agnostic with respect to tuning: thresholds are the frozen
research contract itself (PFV non-inferiority and TFV no worse than every competitive comparator),
not values selected after looking at the evaluation outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


V30_SCIENTIFIC_GATE_CONTRACT = "PROJECT7_V30_PRELOCK_SCIENTIFIC_GATE_V1"
V30_DEVELOPMENT_PARTITION = "development_validation"
V30_PROPOSED_STRATEGY = "proposed_v30_development"
V30_COMPETITIVE_BASELINES = (
    "no_control",
    "matched_internal_rtc",
    "matched_auto_rbc",
    "matched_efd",
)


@dataclass(frozen=True)
class ScientificPanelRow:
    event_id: str
    strategy: str
    partition: str
    source_inp_sha256: str
    tfv_m3: float
    pfv_m3: float
    engineering_pass: bool
    action_decisions: int = 0
    decision_count: int = 0
    ever_changed_actuator_count: int = 0


@dataclass(frozen=True)
class ScientificGateResult:
    contract: str
    passed: bool
    event_count: int
    issues: tuple[str, ...]
    diagnostics: dict[str, object]


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) >= 0.0


def _relative_delta(proposed: float, baseline: float, eps: float = 1.0e-9) -> float:
    p = float(proposed)
    b = float(baseline)
    if b <= eps:
        return 0.0 if p <= eps else float("inf")
    return (p - b) / b


def evaluate_v30_scientific_gate(
    rows: Iterable[ScientificPanelRow],
    *,
    required_baselines: tuple[str, ...] = V30_COMPETITIVE_BASELINES,
) -> ScientificGateResult:
    material = tuple(rows)
    issues: list[str] = []
    if not material:
        return ScientificGateResult(
            contract=V30_SCIENTIFIC_GATE_CONTRACT,
            passed=False,
            event_count=0,
            issues=("empty scientific panel",),
            diagnostics={},
        )

    allowed = {V30_PROPOSED_STRATEGY, *required_baselines}
    by_event: dict[str, dict[str, ScientificPanelRow]] = {}
    for row in material:
        if row.partition != V30_DEVELOPMENT_PARTITION:
            issues.append(
                f"{row.event_id}/{row.strategy}: partition {row.partition!r} is not fresh Development-validation"
            )
        if row.strategy not in allowed:
            issues.append(f"{row.event_id}: unexpected strategy {row.strategy!r}")
        if not row.event_id.strip():
            issues.append("panel contains blank event_id")
        if len(row.source_inp_sha256) != 64:
            issues.append(f"{row.event_id}/{row.strategy}: invalid source INP SHA256")
        if not _finite_nonnegative(row.tfv_m3) or not _finite_nonnegative(row.pfv_m3):
            issues.append(f"{row.event_id}/{row.strategy}: non-finite/negative PFV or TFV")
        if int(row.action_decisions) < 0 or int(row.decision_count) < 0:
            issues.append(f"{row.event_id}/{row.strategy}: negative decision count")
        bucket = by_event.setdefault(row.event_id, {})
        if row.strategy in bucket:
            issues.append(f"{row.event_id}: duplicate strategy row {row.strategy}")
        bucket[row.strategy] = row

    required = (V30_PROPOSED_STRATEGY, *required_baselines)
    complete_events: list[str] = []
    for event_id, bucket in sorted(by_event.items()):
        missing = [strategy for strategy in required if strategy not in bucket]
        if missing:
            issues.append(f"{event_id}: missing strategies {','.join(missing)}")
            continue
        shas = {bucket[strategy].source_inp_sha256.lower() for strategy in required}
        if len(shas) != 1:
            issues.append(f"{event_id}: strategies do not share the same source INP SHA256")
        complete_events.append(event_id)

    diagnostics: dict[str, object] = {
        "required_strategies": list(required),
        "complete_event_count": len(complete_events),
        "partition": V30_DEVELOPMENT_PARTITION,
        "final_or_locked_outcomes_allowed": False,
        "pfv_contract": "PFV_proposed <= 100 + 1.05 * PFV_no_control for every event",
        "tfv_contract": "aggregate and event-balanced relative TFV <= every competitive baseline",
    }

    if complete_events:
        pfv_failures: list[str] = []
        engineering_failures: list[str] = []
        for event_id in complete_events:
            bucket = by_event[event_id]
            proposed = bucket[V30_PROPOSED_STRATEGY]
            no_control = bucket["no_control"]
            pfv_limit = 100.0 + 1.05 * float(no_control.pfv_m3)
            if float(proposed.pfv_m3) > pfv_limit + 1.0e-9:
                pfv_failures.append(event_id)
            for strategy in required:
                if not bool(bucket[strategy].engineering_pass):
                    engineering_failures.append(f"{event_id}/{strategy}")
        if pfv_failures:
            issues.append("PFV safety failed: " + ",".join(pfv_failures))
        if engineering_failures:
            issues.append("engineering execution failed: " + ",".join(engineering_failures))
        diagnostics["pfv_safety_failures"] = pfv_failures
        diagnostics["engineering_failures"] = engineering_failures

        tfv_comparison: dict[str, dict[str, float | int | bool]] = {}
        for baseline in required_baselines:
            proposed_values = [
                float(by_event[event_id][V30_PROPOSED_STRATEGY].tfv_m3)
                for event_id in complete_events
            ]
            baseline_values = [
                float(by_event[event_id][baseline].tfv_m3) for event_id in complete_events
            ]
            aggregate_proposed = float(sum(proposed_values))
            aggregate_baseline = float(sum(baseline_values))
            relative = [
                _relative_delta(p, b) for p, b in zip(proposed_values, baseline_values, strict=True)
            ]
            event_balanced = float(sum(relative) / len(relative))
            noninferior_events = sum(
                p <= b + 1.0e-9
                for p, b in zip(proposed_values, baseline_values, strict=True)
            )
            majority_required = int(math.ceil(len(complete_events) / 2.0))
            passed = (
                aggregate_proposed <= aggregate_baseline + 1.0e-9
                and event_balanced <= 1.0e-12
                and noninferior_events >= majority_required
            )
            tfv_comparison[baseline] = {
                "aggregate_proposed_m3": aggregate_proposed,
                "aggregate_baseline_m3": aggregate_baseline,
                "event_balanced_mean_relative_delta": event_balanced,
                "noninferior_event_count": int(noninferior_events),
                "majority_required": majority_required,
                "passed": passed,
            }
            if not passed:
                issues.append(f"TFV objective not achieved versus {baseline}")
        diagnostics["tfv_comparison"] = tfv_comparison

        # A rule comparator that never sends a non-HOLD action across the entire fresh panel is not
        # evidence of an effective active strategy.  EFD additionally must exercise at least two
        # distinct mapped actuators, consistent with the equal-filling topology contract.
        active_baseline_diagnostics: dict[str, dict[str, int | bool]] = {}
        for baseline in ("matched_auto_rbc", "matched_efd"):
            action_total = sum(
                int(by_event[event_id][baseline].action_decisions)
                for event_id in complete_events
            )
            changed_max = max(
                int(by_event[event_id][baseline].ever_changed_actuator_count)
                for event_id in complete_events
            )
            valid = action_total > 0 and (
                baseline != "matched_efd" or changed_max >= 2
            )
            active_baseline_diagnostics[baseline] = {
                "action_decisions": action_total,
                "ever_changed_actuator_count_max": changed_max,
                "valid_active_comparator": valid,
            }
            if not valid:
                issues.append(f"{baseline} is degenerate/inactive on the Development panel")
        diagnostics["active_baseline_diagnostics"] = active_baseline_diagnostics

    passed = len(issues) == 0 and len(complete_events) == len(by_event)
    diagnostics["ready_for_new_policy_lock"] = passed
    return ScientificGateResult(
        contract=V30_SCIENTIFIC_GATE_CONTRACT,
        passed=passed,
        event_count=len(complete_events),
        issues=tuple(issues),
        diagnostics=diagnostics,
    )


__all__ = [
    "ScientificGateResult",
    "ScientificPanelRow",
    "V30_COMPETITIVE_BASELINES",
    "V30_DEVELOPMENT_PARTITION",
    "V30_PROPOSED_STRATEGY",
    "V30_SCIENTIFIC_GATE_CONTRACT",
    "evaluate_v30_scientific_gate",
]
