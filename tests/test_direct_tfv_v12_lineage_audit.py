from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rtc.direct_tfv_v12_lineage_audit import (
    V12_REFRESH_RECOMMENDATION,
    V12_REUSE_RECOMMENDATION,
    audit_v12_admission_lineage,
    direct_tfv_v12_behavioral_manifest,
)
from rtc.step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]


def _admission(*, behavior: str, step2: str = "step2", support: str = "support") -> dict:
    return {
        "query_step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        "rainfall_scenario_contract": DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
        "v12_behavioral_source_sha256": behavior,
        "lineage": {
            "v12_behavioral_source_sha256": behavior,
            "step2_checkpoint_sha256": step2,
            "sequence_support_sha256": support,
        },
    }


def test_current_v12_admission_is_reusable_only_on_exact_lineage() -> None:
    current = direct_tfv_v12_behavioral_manifest()["v12_behavioral_source_sha256"]
    result = audit_v12_admission_lineage(
        _admission(behavior=current),
        step2_checkpoint_sha256="step2",
        sequence_support_sha256="support",
    )
    assert result["safe_to_reuse_admission"] is True
    assert result["recommended_action"] == V12_REUSE_RECOMMENDATION
    assert result["checks"]["behavioral_fingerprint_match"] is True


def test_stale_v12_admission_requires_role_pure_refresh_without_new_rainfall() -> None:
    result = audit_v12_admission_lineage(
        _admission(behavior="stale-behavior"),
        step2_checkpoint_sha256="step2",
        sequence_support_sha256="support",
    )
    assert result["safe_to_reuse_admission"] is False
    assert result["recommended_action"] == V12_REFRESH_RECOMMENDATION
    assert result["checks"]["behavioral_fingerprint_match"] is False
    assert result["component_behavioral_equivalence_proven"] is False
    assert result["compatibility_whitelist_allowed"] is False
    assert result["new_rainfall_required_for_lineage_refresh"] is False
    assert result["generic_d3_required_for_lineage_refresh"] is False
    assert result["minimum_role_pure_calibration_rainfall_groups"] == 24
    assert result["authoritative_refresh_branch_count"] == 48


def test_step2_or_support_drift_also_fails_closed() -> None:
    current = direct_tfv_v12_behavioral_manifest()["v12_behavioral_source_sha256"]
    result = audit_v12_admission_lineage(
        _admission(behavior=current),
        step2_checkpoint_sha256="different-step2",
        sequence_support_sha256="support",
    )
    assert result["safe_to_reuse_admission"] is False
    assert result["checks"]["step2_checkpoint_match"] is False
    assert result["recommended_action"] == V12_REFRESH_RECOMMENDATION


def test_v12_lineage_audit_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/audit_direct_tfv_v12_admission_lineage_current.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--require-compatible" in completed.stdout
