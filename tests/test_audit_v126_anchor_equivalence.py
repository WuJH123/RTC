from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from scripts.audit_v126_anchor_equivalence import build_audit


def _write_decisions(path: Path, offset: float = 0.0, override: bool = False) -> None:
    rows = []
    for elapsed in (3600, 4200):
        rows.append(
            {
                "elapsed_seconds": elapsed,
                "settings": {"A": 0.2 + offset, "B": 0.6},
                "diagnostics": {"learned_override_admitted": override},
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_stats(path: Path, first: float = 10.0) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "delta_flooding_volume_m3"])
        writer.writeheader()
        writer.writerow({"node_id": "P1", "delta_flooding_volume_m3": first})
        writer.writerow({"node_id": "N2", "delta_flooding_volume_m3": 5.0})


def test_anchor_equivalence_passes_only_for_same_no_override_trace(tmp_path: Path) -> None:
    anchor_decisions = tmp_path / "anchor.jsonl"
    proposed_decisions = tmp_path / "proposed.jsonl"
    anchor_stats = tmp_path / "anchor.csv.gz"
    proposed_stats = tmp_path / "proposed.csv.gz"
    priority = tmp_path / "priority.txt"
    _write_decisions(anchor_decisions)
    _write_decisions(proposed_decisions)
    _write_stats(anchor_stats)
    _write_stats(proposed_stats)
    priority.write_text("P1\nP2\nP3\nP4\nP5\nP6\nP7\nP8\n", encoding="utf-8")
    result = build_audit(
        anchor_decisions=anchor_decisions,
        proposed_decisions=proposed_decisions,
        anchor_stats=anchor_stats,
        proposed_stats=proposed_stats,
        priority_nodes=priority,
    )
    assert result["passed"]
    assert result["verdict"] == "ANCHOR_EQUIVALENT"


def test_anchor_equivalence_blocks_same_claim_with_different_settings(tmp_path: Path) -> None:
    anchor_decisions = tmp_path / "anchor.jsonl"
    proposed_decisions = tmp_path / "proposed.jsonl"
    anchor_stats = tmp_path / "anchor.csv.gz"
    proposed_stats = tmp_path / "proposed.csv.gz"
    priority = tmp_path / "priority.txt"
    _write_decisions(anchor_decisions)
    _write_decisions(proposed_decisions, offset=1.0e-3)
    _write_stats(anchor_stats)
    _write_stats(proposed_stats)
    priority.write_text("P1\nP2\nP3\nP4\nP5\nP6\nP7\nP8\n", encoding="utf-8")
    result = build_audit(
        anchor_decisions=anchor_decisions,
        proposed_decisions=proposed_decisions,
        anchor_stats=anchor_stats,
        proposed_stats=proposed_stats,
        priority_nodes=priority,
    )
    assert not result["passed"]
    assert result["verdict"] == "ANCHOR_EQUIVALENCE_BLOCKED"
