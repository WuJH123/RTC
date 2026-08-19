"""Add Priority8 PFV as a report-only metric to an existing Direct-TFV baseline comparison.

TFV remains the sole optimization objective and the existing TFV classification is preserved.  This
post-processor only sums authoritative SWMM node flooding volume over a frozen Priority8 node list so
PFV can be reported alongside TFV and Global Peak for every strategy.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def _priority_nodes(path: str | Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(values) != 8 or len(set(values)) != 8:
        raise ValueError("PFV reporting requires exactly eight unique frozen Priority8 node IDs")
    return values


def _pfv_m3(path: str | Path, priority_nodes: tuple[str, ...]) -> float:
    wanted = set(priority_nodes)
    seen: set[str] = set()
    total = 0.0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            node = str(row["node_id"])
            if node in wanted:
                seen.add(node)
                total += float(row["delta_flooding_volume_m3"])
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"Priority8 nodes absent from authoritative node statistics: {missing}")
    return float(total)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--priority-nodes", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    payload = json.loads(Path(args.comparison_json).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("input comparison JSON lacks rows")
    priority = _priority_nodes(args.priority_nodes)
    rows = [dict(row) for row in payload["rows"]]
    for row in rows:
        statistics = row.get("node_statistics_path")
        if not statistics:
            raise ValueError(f"comparison row lacks node_statistics_path: {row.get('strategy')}")
        row["pfv_m3"] = _pfv_m3(statistics, priority)

    by_strategy = {str(row["strategy"]): row for row in rows}
    if "no_control" not in by_strategy or "proposed" not in by_strategy:
        raise ValueError("comparison must contain proposed and no_control rows")
    nc = float(by_strategy["no_control"]["pfv_m3"])
    proposed = float(by_strategy["proposed"]["pfv_m3"])
    for row in rows:
        pfv = float(row["pfv_m3"])
        row["delta_pfv_vs_no_control_m3"] = pfv - nc
        row["pfv_reduction_vs_no_control_pct"] = (
            100.0 * (nc - pfv) / nc if nc > 0.0 else None
        )
        if row["strategy"] != "proposed":
            row["proposed_minus_strategy_pfv_m3"] = proposed - pfv

    payload["rows"] = rows
    payload["priority_nodes"] = list(priority)
    payload["pfv_role"] = "report_only_secondary_not_optimization_objective_or_gate"
    payload["tfv_remains_primary_objective"] = True
    payload["global_peak_role"] = "report_only"
    payload["classification_unchanged_by_pfv"] = True

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
