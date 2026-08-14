from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from scripts.compare_step2_v123_t5_strategies import _all_node_tfv, build_report
from scripts.build_step2_v123_current_decision import _decision_summary


def _stats(path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "delta_flooding_volume_m3"])
        writer.writeheader()
        writer.writerow({"node_id": "P1", "delta_flooding_volume_m3": "2.0"})
        writer.writerow({"node_id": "P2", "delta_flooding_volume_m3": "3.0"})
        for i in range(3, 9):
            writer.writerow({"node_id": f"P{i}", "delta_flooding_volume_m3": "0.0"})


def _metadata(path: Path, strategy: str, stats: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "event_id": "T5_D180_chicago",
                "global_peak_flood_rate_m3s": 4.0,
                "flow_routing_error_pct": 0.0,
                "decisions": 0,
                "decision_file": "decisions.jsonl",
                "strategy": strategy,
            }
        ),
        encoding="utf-8",
    )
    path.with_name("decisions.jsonl").write_text("", encoding="utf-8")


def test_all_node_tfv_uses_authoritative_volume(tmp_path: Path) -> None:
    stats = tmp_path / "stats.csv.gz"
    _stats(stats)
    assert _all_node_tfv(stats) == (5.0, 8)


def test_build_report_adds_priority8_and_reductions(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    priority = tmp_path / "priority.txt"
    priority.write_text("P1\nP2\nP3\nP4\nP5\nP6\nP7\nP8\n", encoding="utf-8")
    rows = []
    for strategy in ("no_control", "internal_rtc", "auto_rbc", "efd", "all_open", "all_closed"):
        folder = baseline / strategy
        folder.mkdir()
        stats = folder / f"{strategy}.node_statistics.csv.gz"
        meta = folder / f"{strategy}.json"
        _stats(stats)
        _metadata(meta, strategy, stats)
        rows.append({"strategy": strategy, "metadata_path": str(meta), "node_statistics_path": str(stats)})
    (baseline / "BASELINE_COMPARISON_V122.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    proposed = tmp_path / "proposed"
    proposed.mkdir()
    stats = proposed / "proposed.node_statistics.csv.gz"
    meta = proposed / "proposed.json"
    _stats(stats)
    _metadata(meta, "proposed_v123", stats)
    payload = build_report(baseline_root=baseline, proposed_dir=proposed, priority_path=priority, out=tmp_path / "out.json")
    assert payload["rows"][-1]["strategy"] == "proposed_v123"
    assert payload["rows"][0]["pfv_priority8_m3"] == 5.0
    assert payload["rows"][-1]["tfv_reduction_vs_no_control_pct"] == 0.0


def test_runtime_summary_reads_nested_diagnostics(tmp_path: Path) -> None:
    metadata = tmp_path / "run.json"
    metadata.write_text(
        json.dumps({
            "decision_file": "run.decisions.jsonl",
            "future_realized_rainfall_used_as_model_input": False,
            "v123_runtime_causal_rainfall": True,
            "score_only_executable_sequences": True,
        }),
        encoding="utf-8",
    )
    (tmp_path / "run.decisions.jsonl").write_text(
        json.dumps({"source": "MPC_V123", "diagnostics": {"predicted_delta_tfv_m3": -10.0}}) + "\n"
        + json.dumps({"source": "PASSIVE_MPC_NO_PREDICTED_BENEFIT", "diagnostics": {"predicted_delta_tfv_m3": 0.0}}) + "\n",
        encoding="utf-8",
    )
    summary = _decision_summary(tmp_path, 100.0)
    assert summary["nonhold_predicted_tfv_improvement_m3"] == [10.0]
    assert summary["small_effect_selection_risk"] is True
