"""D2 source-lineage contract for Project7 Step2 V11.2.

The original v0.6.9 intervention census contains 9216 requested probes that
resolve to 4800 unique authoritative SWMM branches over 192 checkpoint states
with 109/109 actuator coverage. Later V60/V11 caches are derived training views
and must not be relabelled as the complete D2 source experiment.

Source membership is not training authorization: current TrainFit/holdout rules
still apply and Validation/Final outcomes remain forbidden for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


V112_D2_LINEAGE_CONTRACT = "PROJECT7_STEP2_V112_D2_SOURCE_LINEAGE_V1"


@dataclass(frozen=True)
class FullD2SourceContractV112:
    requested_branches: int = 9216
    unique_authoritative_branches: int = 4800
    development_events_at_generation: int = 24
    checkpoint_states: int = 192
    actuator_count: int = 109
    horizon_minutes: int = 360
    model_stride_seconds: int = 300

    def validate(self) -> None:
        expected = (9216, 4800, 24, 192, 109, 360, 300)
        actual = (
            self.requested_branches,
            self.unique_authoritative_branches,
            self.development_events_at_generation,
            self.checkpoint_states,
            self.actuator_count,
            self.horizon_minutes,
            self.model_stride_seconds,
        )
        if actual != expected:
            raise ValueError(f"V112 full-D2 lineage drift: {actual} != {expected}")


FULL_D2_SOURCE_V112 = FullD2SourceContractV112()


def classify_d2_population_v112(
    *,
    authoritative_branches: int,
    checkpoint_states: int | None = None,
    actuator_count: int | None = None,
    event_count: int | None = None,
    group_count: int | None = None,
) -> dict[str, Any]:
    """Classify source D2 versus a later derived cache without conflation."""
    FULL_D2_SOURCE_V112.validate()
    branches = int(authoritative_branches)
    is_full = branches == FULL_D2_SOURCE_V112.unique_authoritative_branches
    if checkpoint_states is not None:
        is_full &= int(checkpoint_states) == FULL_D2_SOURCE_V112.checkpoint_states
    if actuator_count is not None:
        is_full &= int(actuator_count) == FULL_D2_SOURCE_V112.actuator_count
    return {
        "contract": V112_D2_LINEAGE_CONTRACT,
        "population_view": "FULL_D2_SOURCE" if is_full else "DERIVED_D2_VIEW",
        "authoritative_branches": branches,
        "group_count": None if group_count is None else int(group_count),
        "event_count": None if event_count is None else int(event_count),
        "checkpoint_states": None if checkpoint_states is None else int(checkpoint_states),
        "actuator_count": None if actuator_count is None else int(actuator_count),
        "full_source_expected": {
            "requested_branches": 9216,
            "unique_authoritative_branches": 4800,
            "development_events_at_generation": 24,
            "checkpoint_states": 192,
            "actuator_count": 109,
            "horizon_minutes": 360,
            "model_stride_seconds": 300,
        },
        "training_authorized_by_source_membership": False,
        "validation_final_outcomes_allowed": False,
    }


def require_full_d2_source_claim_v112(summary: Mapping[str, Any]) -> None:
    """Reject reports that call a 3600-branch/144-group cache 'complete D2'."""
    result = classify_d2_population_v112(
        authoritative_branches=int(summary.get("authoritative_branches", -1)),
        checkpoint_states=summary.get("checkpoint_states"),
        actuator_count=summary.get("actuator_count"),
        event_count=summary.get("event_count"),
        group_count=summary.get("group_count"),
    )
    if result["population_view"] != "FULL_D2_SOURCE":
        raise RuntimeError(
            "complete-D2 claim rejected: this population is a derived training view, "
            "not the frozen 4800-branch source census"
        )


__all__ = [
    "FULL_D2_SOURCE_V112",
    "FullD2SourceContractV112",
    "V112_D2_LINEAGE_CONTRACT",
    "classify_d2_population_v112",
    "require_full_d2_source_claim_v112",
]
