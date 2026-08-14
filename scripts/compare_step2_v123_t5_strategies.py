"""Build the V123 seven-strategy T5 development comparison.

This report is deliberately read-only: it consumes existing SWMM node-statistics
artifacts and never re-integrates sampled flooding rates.  TFV is the sum of the
authoritative ``delta_flooding_volume_m3`` column over all nodes; PFV is the same
column restricted to the frozen Priority8 set.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rtc.step3_metrics_v123 import priority_flood_volume_from_node_statistics_v123


STRATEGIES = ("no_control", "internal_rtc", "auto_rbc", "efd", "all_open", "all_closed")
CONTRACT = "PROJECT7_V123_T5_SEVEN_STRATEGY_TFV_PFV_COMPARISON_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _priority_nodes(path: Path) -> tuple[str, ...]:
    nodes = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(nodes) != 8 or len(set(nodes)) != 8:
        raise ValueError("V123 T5 comparison requires exactly eight unique Priority8 nodes")
    return nodes


def _all_node_tfv(path: Path) -> tuple[float, int]:
    total = 0.0
    node_ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"node_id", "delta_flooding_volume_m3"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"missing authoritative TFV columns in {path}")
        for row in reader:
            node = str(row["node_id"])
            if node in node_ids:
                raise ValueError(f"duplicate node in statistics: {node}")
            node_ids.add(node)
            value = float(row["delta_flooding_volume_m3"])
            if value < -1.0e-8 or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"invalid flooding volume for {node}: {value}")
            total += max(value, 0.0)
    if not node_ids:
        raise ValueError(f"empty node statistics: {path}")
    return float(total), len(node_ids)


def _metadata(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("global_peak_flood_rate_m3s", "flow_routing_error_pct"):
        if key not in payload:
            raise ValueError(f"metadata missing {key}: {path}")
    return payload


def _find_single(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(paths)}")
    return paths[0]


def _baseline_paths(root: Path, strategy: str, comparison: Mapping[str, Any]) -> tuple[Path, Path]:
    rows = [row for row in comparison.get("rows", []) if row.get("strategy") == strategy]
    if len(rows) != 1:
        raise ValueError(f"baseline comparison lacks unique row for {strategy}")
    row = rows[0]
    metadata = Path(str(row["metadata_path"]))
    stats = Path(str(row["node_statistics_path"]))
    if not metadata.is_file() or not stats.is_file():
        # Keep the report usable when the frozen comparison was moved as a directory.
        directory = root / strategy
        metadata = _find_single(directory, "*.json")
        stats = _find_single(directory, "*.node_statistics.csv.gz")
    return metadata, stats


def _decision_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source = str(record.get("source", "UNKNOWN"))
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _row(strategy: str, metadata_path: Path, stats_path: Path, priority: Sequence[str]) -> dict[str, Any]:
    metadata = _metadata(metadata_path)
    tfv, node_count = _all_node_tfv(stats_path)
    pfv = priority_flood_volume_from_node_statistics_v123(stats_path, priority_nodes=priority)
    decisions_path = metadata_path.with_name(str(metadata.get("decision_file", "")))
    return {
        "strategy": strategy,
        "event_id": str(metadata.get("event_id", "T5_D180_chicago")),
        "metadata_path": str(metadata_path),
        "node_statistics_path": str(stats_path),
        "metadata_sha256": _sha256(metadata_path),
        "node_statistics_sha256": _sha256(stats_path),
        "node_count": node_count,
        "tfv_m3": tfv,
        "pfv_priority8_m3": pfv,
        "global_peak_flood_rate_m3s": float(metadata["global_peak_flood_rate_m3s"]),
        "flow_routing_error_pct": float(metadata["flow_routing_error_pct"]),
        "decisions": int(metadata.get("decisions", 0)),
        "decision_source_counts": _decision_counts(decisions_path),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    ref = next(row for row in payload["rows"] if row["strategy"] == "no_control")
    lines = [
        "# V123 T5 seven-strategy TFV/PFV comparison",
        "",
        "Read-only development comparison from authoritative SWMM node statistics.",
        "TFV/PFV use `delta_flooding_volume_m3`; sampled-rate re-integration is not used.",
        "",
        "| Strategy | TFV (m3) | TFV reduction (%) | Priority8 PFV (m3) | PFV change (%) | Global peak (m3/s) | Decisions | Routing error (%) |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['strategy']} | {row['tfv_m3']:.3f} | {row['tfv_reduction_vs_no_control_pct']:.3f} | "
            f"{row['pfv_priority8_m3']:.3f} | {row['pfv_change_vs_no_control_pct']:.3f} | "
            f"{row['global_peak_flood_rate_m3s']:.3f} | {row['decisions']} | {row['flow_routing_error_pct']:.6f} |"
        )
    lines += [
        "",
        f"No-control TFV: {ref['tfv_m3']:.3f} m3; Priority8 PFV: {ref['pfv_priority8_m3']:.3f} m3.",
        "This is a single development event and is not Formal/Validation evidence.",
    ]
    return "\n".join(lines) + "\n"


def build_report(*, baseline_root: Path, proposed_dir: Path, priority_path: Path, out: Path) -> dict[str, Any]:
    comparison_path = baseline_root / "BASELINE_COMPARISON_V122.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    priority = _priority_nodes(priority_path)
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        metadata, stats = _baseline_paths(baseline_root, strategy, comparison)
        rows.append(_row(strategy, metadata, stats, priority))
    proposed_metadata = _find_single(proposed_dir, "*.json")
    proposed_stats = _find_single(proposed_dir, "*.node_statistics.csv.gz")
    rows.append(_row("proposed_v123", proposed_metadata, proposed_stats, priority))
    ref = next(row for row in rows if row["strategy"] == "no_control")
    for row in rows:
        row["tfv_reduction_vs_no_control_pct"] = 100.0 * (ref["tfv_m3"] - row["tfv_m3"]) / ref["tfv_m3"]
        row["pfv_change_vs_no_control_pct"] = 100.0 * (ref["pfv_priority8_m3"] - row["pfv_priority8_m3"]) / ref["pfv_priority8_m3"] if ref["pfv_priority8_m3"] else None
    payload: dict[str, Any] = {
        "contract": CONTRACT,
        "event_id": "T5_D180_chicago",
        "priority_nodes": list(priority),
        "rows": rows,
        "strategies": [row["strategy"] for row in rows],
        "boundary": {
            "new_swmm": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "source": "existing_authoritative_node_statistics",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--proposed-dir", required=True, type=Path)
    parser.add_argument("--priority-nodes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = build_report(
        baseline_root=args.baseline_root,
        proposed_dir=args.proposed_dir,
        priority_path=args.priority_nodes,
        out=args.out,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
