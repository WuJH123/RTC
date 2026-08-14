from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from rtc.step3_metrics_v123 import priority_flood_volume_from_node_statistics_v123


def _write(path: Path, rows: list[tuple[str, float]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node_id", "delta_flooding_volume_m3"])
        writer.writerows(rows)


def test_priority_pfv_sums_exact_swmm_node_statistic_volume(tmp_path: Path) -> None:
    path = tmp_path / "stats.csv.gz"
    _write(path, [("P1", 10.0), ("other", 999.0), ("P2", 2.5)])
    value = priority_flood_volume_from_node_statistics_v123(
        path, priority_nodes=("P1", "P2")
    )
    assert value == 12.5


def test_priority_pfv_fails_closed_when_priority_node_missing(tmp_path: Path) -> None:
    path = tmp_path / "stats.csv.gz"
    _write(path, [("P1", 10.0)])
    with pytest.raises(ValueError, match="missing"):
        priority_flood_volume_from_node_statistics_v123(
            path, priority_nodes=("P1", "P2")
        )
