from __future__ import annotations

from types import SimpleNamespace

import pytest

from rtc.direct_tfv_operational_v30_runtime import _v27_best_supported_source


def test_v30_uses_executed_v27_source_when_parent_action_exists() -> None:
    parent = SimpleNamespace(
        policy_return_portfolio_selected_source="STEP2_H10_PROBE_SCALE_0.50",
        policy_return_portfolio_sources=("STEP2_H10_PROBE_SCALE_0.50",),
        scipy_message="unused",
    )
    assert _v27_best_supported_source(parent) == "STEP2_H10_PROBE_SCALE_0.50"


def test_v30_recovers_unexecuted_v27_supported_best_source_from_diagnostics() -> None:
    parent = SimpleNamespace(
        policy_return_portfolio_selected_source="HOLD",
        policy_return_portfolio_sources=(
            "STEP2_H10_PROBE_SCALE_0.50",
            "TYPE_AWARE_HYDRAULIC_PRESSURE",
        ),
        scipy_message=(
            "V27_DECISION_AWARE|latent_min=1.2|"
            "supported_best=TYPE_AWARE_HYDRAULIC_PRESSURE|shadow_present=true"
        ),
    )
    assert _v27_best_supported_source(parent) == "TYPE_AWARE_HYDRAULIC_PRESSURE"


def test_v30_fails_closed_instead_of_inventing_candidate_family() -> None:
    parent = SimpleNamespace(
        policy_return_portfolio_selected_source="HOLD",
        policy_return_portfolio_sources=("STEP2_H10_PROBE_SCALE_0.50",),
        scipy_message="V27_DECISION_AWARE|supported_best=UNKNOWN_FAMILY",
    )
    with pytest.raises(RuntimeError, match="absent from its portfolio lineage"):
        _v27_best_supported_source(parent)
