from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _report(path: Path, *, d2: dict, d3: dict) -> None:
    lineage = {
        "graph_sha256": "g",
        "base_cache_sha256": "b",
        "d4_fit_cache_sha256": "f",
        "d4_audit_cache_sha256": "a",
        "causal_rainfall_sha256": "r",
        "causal_state_store_sha256": "s",
        "step1_model_semantic_sha256": "m",
        "sensor_layout_semantic_sha256": "l",
    }
    payload = {
        "profile": "dev",
        "seed": 42,
        "selected_group_counts": {"fit_d2": 112, "fit_d3": 112},
        "lineage": lineage,
        "evaluations": {
            "internal_holdout_d2": d2,
            "internal_holdout_d3": d3,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(*, rank: float, pairwise: float, sign: float, regret: float, harmful: float) -> dict:
    return {
        "rank": rank,
        "pairwise": pairwise,
        "sign": sign,
        "top1_fraction": 0.5,
        "selected_harmful_fraction": harmful,
        "selected_true_delta_tfv_m3": -100.0,
        "delta_tfv_mae_m3": 50.0,
        "selected_regret_m3": regret,
    }


def test_comparator_passes_only_when_d2_preserved_and_d3_improves(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    _report(
        baseline,
        d2=_metrics(rank=0.60, pairwise=0.70, sign=0.75, regret=100.0, harmful=0.10),
        d3=_metrics(rank=0.50, pairwise=0.60, sign=0.70, regret=200.0, harmful=0.20),
    )
    _report(
        candidate,
        d2=_metrics(rank=0.61, pairwise=0.71, sign=0.75, regret=90.0, harmful=0.10),
        d3=_metrics(rank=0.72, pairwise=0.65, sign=0.71, regret=150.0, harmful=0.15),
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_step2_historical_retrain_v6.py",
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--out",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["pass_existing_data_offline_gate"] is True
    assert result["formal_authorized"] is False


def test_comparator_rejects_d2_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    _report(
        baseline,
        d2=_metrics(rank=0.60, pairwise=0.70, sign=0.75, regret=100.0, harmful=0.10),
        d3=_metrics(rank=0.50, pairwise=0.60, sign=0.70, regret=200.0, harmful=0.20),
    )
    _report(
        candidate,
        d2=_metrics(rank=0.59, pairwise=0.71, sign=0.75, regret=90.0, harmful=0.10),
        d3=_metrics(rank=0.72, pairwise=0.65, sign=0.71, regret=150.0, harmful=0.15),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_step2_historical_retrain_v6.py",
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--out",
            str(output),
        ],
        check=False,
    )
    assert completed.returncode == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["pass_existing_data_offline_gate"] is False
