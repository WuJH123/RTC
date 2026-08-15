from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict[str, object]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_lf_sha256(path: Path) -> str:
    """Hash text bytes after normalizing only platform EOL representation to LF.

    Git may check a text file out as CRLF on Windows when core.autocrlf is enabled.
    The scientific split must not change identity solely because of that transport-level
    representation. All non-EOL bytes, row order, values and final-newline presence remain
    part of the frozen identity.
    """

    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def test_active_split_hash_and_18_6_6_contract() -> None:
    registry_path = ROOT / "configs/project7_v069_events_with_splits.csv"
    contract = _json("configs/project7_v069_split_contract.json")
    assert contract["portable_registry_hash_semantics"] == (
        "sha256 of registry bytes after CRLF/CR to LF normalization; all other bytes remain frozen"
    )
    assert _canonical_lf_sha256(registry_path) == contract["portable_registry_sha256"]
    frame = pd.read_csv(registry_path, keep_default_na=False)
    train = frame[
        (frame["scientific_split"] == "development")
        & (frame["development_fold"] == "train")
    ]
    validation = frame[
        (frame["scientific_split"] == "development")
        & (frame["development_fold"] == "validation")
    ]
    final = frame[frame["scientific_split"] == "final"]
    assert (len(train), len(validation), len(final)) == (18, 6, 6)
    assert set(frame["scientific_split"]) == {"development", "final"}
    assert set(validation["duration_minutes"].astype(int)) == {60, 120, 180, 240, 300, 360}
    assert set(final["duration_minutes"].astype(int)) == {60, 120, 180, 240, 300, 360}
    assert set(validation["return_period_year"].astype(int)) == {5, 10, 20, 50, 100}
    assert set(final["return_period_year"].astype(int)) == {5, 10, 20, 50, 100}
    assert train.groupby("duration_minutes")["event_id"].count().to_dict() == {
        60: 3,
        120: 3,
        180: 3,
        240: 3,
        300: 3,
        360: 3,
    }


def test_split_registry_hash_is_identical_for_lf_and_crlf_worktrees(tmp_path: Path) -> None:
    source = ROOT / "configs/project7_v069_events_with_splits.csv"
    contract = _json("configs/project7_v069_split_contract.json")
    canonical = source.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == contract["portable_registry_sha256"]

    crlf_path = tmp_path / "events_crlf.csv"
    crlf_path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    assert _canonical_lf_sha256(crlf_path) == contract["portable_registry_sha256"]


def test_controller_is_fully_resolved_and_frozen() -> None:
    path = ROOT / "configs/formal_controller_v5.json"
    text = path.read_text(encoding="utf-8")
    assert "REPLACE_" not in text
    cfg = json.loads(text)
    assert cfg["model_step_seconds"] == 300
    assert cfg["control_update_seconds"] == 600
    assert cfg["control_start_minutes"] == 60
    assert cfg["objective"] == {
        "primary": "minimize rainfall-ensemble risk-adjusted cumulative TFV",
        "priority_role": "soft lexicographic secondary preference within TFV near-optimal set",
        "forecast_quantile": 0.95,
        "tfv_cvar_alpha": 0.90,
        "tfv_near_opt_relative": 0.01,
        "tfv_near_opt_absolute_m3": 1.0,
        "near_opt_penalty": 10000.0,
        "movement_tiebreak": 1e-06,
        "min_predicted_tfv_improvement_m3": 0.0,
        "min_predicted_tfv_improvement_relative": 0.0,
    }
    assert cfg["forecast"] == {
        "decay_per_step": 0.92,
        "scenario_multipliers": [0.75, 1.0, 1.25],
        "history_steps_for_level": 3,
    }
    controller = cfg["controller"]
    assert controller["history_steps"] == 13
    assert controller["horizon_steps"] == 72
    assert controller["optimizer_iterations"] == 120
    assert controller["optimizer_learning_rate"] == 0.04
    assert controller["readback_target_tolerance"] == 1e-6
    assert controller["readback_current_tolerance"] == 0.05
    assert controller["decision_runtime_budget_seconds"] == 300.0


def test_dimensionless_acceptance_is_preregistered() -> None:
    contract = _json("configs/model_acceptance_contract_v4.json")
    assert contract["step1"]["minimum"] == {"unobserved_depth_nse": 0.70}
    assert contract["step1"]["maximum"] == {}
    assert contract["step2"]["minimum"] == {"tfv_exact_truth_rank_correlation": 0.70}
    assert contract["step2"]["maximum"] == {}
    assert contract["gradient"]["minimum"] == {
        "tfv_gradient_sign_accuracy": 0.70,
        "tfv_gradient_cosine_similarity": 0.60,
    }
    assert contract["gradient"]["maximum"] == {}
    assert contract["candidate_ranking"]["minimum"] == {
        "d2_tfv_rank_correlation": 0.70,
        "d2_tfv_top1_hit_rate": 0.50,
        "d3_tfv_rank_correlation": 0.70,
        "d3_tfv_top1_hit_rate": 0.50,
    }
    assert contract["candidate_ranking"]["maximum"] == {}


def test_phase0_budget_has_dedicated_cli_scope() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'rtc-design-phase0-checkpoints = "rtc.execution_defaults_guard:checkpoint_design_main"' in pyproject
    assert 'rtc-design-phase0-probes = "rtc.execution_defaults_guard:efficient_probe_design_main"' in pyproject
    assert 'rtc-design-checkpoints = "rtc.checkpoint_design:main"' in pyproject
    assert 'rtc-design-probes-efficient = "rtc.efficient_probe_design:main"' in pyproject
    assert 'rtc-design-checkpoints = "rtc.execution_defaults_guard:checkpoint_design_main"' not in pyproject
    assert 'rtc-design-probes-efficient = "rtc.execution_defaults_guard:efficient_probe_design_main"' not in pyproject


def test_step0_prepares_effective120_before_fresh_workspace() -> None:
    script = (ROOT / "scripts/adopt_and_step0_project7_v067.ps1").read_text(encoding="utf-8")
    prepare = script.index("rtc-prepare-event-suite")
    initialize = script.index("rtc-init-fresh-workspace")
    assert prepare < initialize
    assert "--events $PreparedRegistry" in script[initialize : initialize + 300]
    assert 'Join-Path $Study "preflight\\inp_audit.json"' in script
    assert 'Join-Path $Study "contracts\\study_readiness.json"' in script
    assert "--target-effective-warmup-minutes 120" in script
    assert "--post-rain-tail-minutes 360" in script


def test_old_active_looking_contracts_are_absent() -> None:
    obsolete = [
        "configs/formal_controller_v5.template.json",
        "configs/model_acceptance_contract_v3.template.json",
        "data/method_testbed_v067/contracts/events_with_splits.csv",
    ]
    assert all(not (ROOT / path).exists() for path in obsolete)
