from __future__ import annotations

import inspect

from rtc.direct_tfv_operational_v30_runtime import (
    DirectTFVOperationalV30MPC,
    OPERATIONAL_V30_RUNTIME_CONTRACT,
    V30_SELECTION_CONTRACT,
)


def test_v30_is_a_thin_v27_objective_selector_without_baseline_veto() -> None:
    source = inspect.getsource(DirectTFVOperationalV30MPC.optimize)
    assert "super().optimize" in source
    assert "select_conservative" not in source
    assert "anchor" not in source.lower()
    assert "dual_estimator" not in source


def test_v30_contract_describes_objective_argmin_not_baseline_dominance() -> None:
    assert "OBJECTIVE_DRIVEN" in OPERATIONAL_V30_RUNTIME_CONTRACT
    assert "ARGMIN" in V30_SELECTION_CONTRACT
    assert "RBC_ANCHORED" not in OPERATIONAL_V30_RUNTIME_CONTRACT
    assert "DOMINANCE" not in V30_SELECTION_CONTRACT
