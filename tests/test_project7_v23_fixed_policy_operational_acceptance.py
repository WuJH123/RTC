from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from rtc.project7_publication_statistics import (
    exact_bootstrap_mean_ci,
    exact_two_sided_sign_test_pvalue,
)


REPO = Path(__file__).resolve().parents[1]


def _load_policy_lock_module():
    path = REPO / "scripts" / "create_project7_v23_policy_lock_current.py"
    spec = importlib.util.spec_from_file_location("project7_policy_lock_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operational_acceptance_retains_failed_step2_but_allows_fixed_policy_validation(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "formal_mode": "FIXED_POLICY_NO_RETRAIN",
                "final_opened": False,
                "validation_can_update_model_parameters": False,
                "final_can_tune_any_model_threshold_or_candidate": False,
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "step2_evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "acceptance.json"
    command = [
        sys.executable,
        str(REPO / "scripts" / "compile_project7_v23_fixed_policy_operational_acceptance_current.py"),
        "--operational-acceptance-contract",
        str(REPO / "configs" / "project7_v23_fixed_policy_operational_acceptance_contract_v1.json"),
        "--legacy-model-acceptance-contract",
        str(REPO / "configs" / "model_acceptance_contract_v4.json"),
        "--formal-protocol",
        str(protocol),
        "--step1-unobserved-depth-nse",
        "0.9243753",
        "--step2-tfv-exact-truth-rank-correlation",
        "-0.272494",
        "--step2-query-balanced-top1",
        "0.311111",
        "--step2-mean-selected-regret-m3",
        "56366.11",
        "--source-evidence",
        str(evidence),
        "--out",
        str(out),
    ]
    subprocess.run(command, check=True, cwd=REPO)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["step1_accepted"] is True
    assert payload["step2_legacy_component_accepted"] is False
    assert payload["step2_required_for_policy_lock"] is False
    assert payload["step2_failure_retained_as_publication_limitation"] is True
    assert payload["step2_standalone_surrogate_claim_allowed"] is False
    assert payload["accepted_for_operational_validation"] is True
    assert payload["step3_disposition"] == "FROZEN_V15_V21_FIXED_POLICY_NO_RETRAIN"

    policy_lock = _load_policy_lock_module()
    basis, disposition, step2_pass, step2_gate, restrictions = policy_lock._validate_acceptance(
        payload,
        protocol_mode="FIXED_POLICY_NO_RETRAIN",
    )
    assert basis == "FIXED_POLICY_END_TO_END_OPERATIONAL"
    assert disposition == "FROZEN_V15_V21_FIXED_POLICY_NO_RETRAIN"
    assert step2_pass is False
    assert step2_gate is False
    assert "DO_NOT_CLAIM_STEP2_STANDALONE_TFV_RANKING_ACCEPTANCE" in restrictions


def test_exact_final6_bootstrap_is_deterministic_and_sign_test_is_exact() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    first = exact_bootstrap_mean_ci(values)
    second = exact_bootstrap_mean_ci(values)
    assert first == second
    assert first[0] < sum(values) / 6.0 < first[1]
    assert exact_two_sided_sign_test_pvalue(wins=6, losses=0) == pytest.approx(0.03125)
    assert exact_two_sided_sign_test_pvalue(wins=3, losses=3) == pytest.approx(1.0)
    assert exact_two_sided_sign_test_pvalue(wins=0, losses=0) is None
