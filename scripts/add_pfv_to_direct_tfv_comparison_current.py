"""Add Priority8 PFV and its one-sided safety envelope to a Direct-TFV comparison.

Project7 keeps system-wide TFV as the sole online optimization objective. Priority8 PFV is evaluated
from authoritative SWMM node flooding volumes as a secondary safety requirement: the Proposed policy
must not materially worsen priority-area flooding relative to No-control. This avoids adding another
uncertain online surrogate while preventing a TFV-only policy from buying global volume reduction by
concentrating flooding at priority nodes.

The default envelope reuses the project's earlier engineering convention:

    PFV_proposed <= 100 m3 + 1.05 * PFV_no_control

The 5%/100 m3 values are Project7/Project6 study tolerances, not universal regulatory thresholds, and
are explicit CLI parameters so the frozen study contract is visible in every report.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


PFV_SAFETY_CONTRACT = "PROJECT7_PRIORITY8_PFV_SECONDARY_SAFETY_V1"


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
    p.add_argument("--pfv-relative-tolerance", type=float, default=0.05)
    p.add_argument("--pfv-absolute-tolerance-m3", type=float, default=100.0)
    args = p.parse_args()
    if not 0.0 <= float(args.pfv_relative_tolerance) <= 1.0:
        raise ValueError("PFV relative tolerance must lie in [0,1]")
    if float(args.pfv_absolute_tolerance_m3) < 0.0:
        raise ValueError("PFV absolute tolerance must be non-negative")

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
    limit = (
        (1.0 + float(args.pfv_relative_tolerance)) * nc
        + float(args.pfv_absolute_tolerance_m3)
    )
    for row in rows:
        pfv = float(row["pfv_m3"])
        row["delta_pfv_vs_no_control_m3"] = pfv - nc
        row["pfv_reduction_vs_no_control_pct"] = (
            100.0 * (nc - pfv) / nc if nc > 0.0 else None
        )
        row["within_pfv_no_control_safety_envelope"] = bool(pfv <= limit + 1.0e-9)
        if row["strategy"] != "proposed":
            row["proposed_minus_strategy_pfv_m3"] = proposed - pfv

    proposal_pass = bool(proposed <= limit + 1.0e-9)
    payload["rows"] = rows
    payload["priority_nodes"] = list(priority)
    payload["pfv_safety_contract"] = PFV_SAFETY_CONTRACT
    payload["pfv_role"] = "secondary_authoritative_safety_gate_not_online_optimization_objective"
    payload["pfv_no_control_m3"] = nc
    payload["pfv_relative_tolerance"] = float(args.pfv_relative_tolerance)
    payload["pfv_absolute_tolerance_m3"] = float(args.pfv_absolute_tolerance_m3)
    payload["pfv_safety_limit_m3"] = float(limit)
    payload["proposed_pfv_m3"] = proposed
    payload["proposed_pfv_safety_pass"] = proposal_pass
    payload["tfv_remains_primary_objective"] = True
    payload["global_peak_role"] = "report_only"
    payload["performance_claim_requires_pfv_safety_pass"] = True
    payload["pfv_safety_threshold_is_project_specific_not_universal_regulation"] = True

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
