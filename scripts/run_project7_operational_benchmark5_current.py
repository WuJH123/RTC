"""Run only the current Proposed controller on the frozen five-event benchmark and compare cached baselines.

Fixed No-control/Internal/Auto-RBC/EFD evidence is never recomputed here. This is the fast operational
feedback loop for Project7 development: authoritative SWMM TFV first, Priority8 PFV safety second,
Global Peak report-only, engineering/runtime diagnostics retained. Formal/Policy-Lock claims remain
closed regardless of this Development result.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from rtc.baseline_panel import tfv_m3
from rtc.operational_benchmark_v1 import (
    OPERATIONAL_BASELINE_CACHE_CONTRACT,
    OPERATIONAL_COMPARATORS,
    OPERATIONAL_COMPARISON_CONTRACT,
    load_benchmark_manifest,
    sha256_file,
)


def _priority_nodes(path: str | Path) -> tuple[str, ...]:
    values = tuple(
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(values) != 8 or len(set(values)) != 8:
        raise ValueError("operational benchmark requires exactly eight frozen Priority8 nodes")
    return values


def _pfv(path: str | Path, priority: tuple[str, ...]) -> float:
    wanted = set(priority)
    seen: set[str] = set()
    total = 0.0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            node = str(row["node_id"])
            if node in wanted:
                seen.add(node)
                total += float(row["delta_flooding_volume_m3"])
    if seen != wanted:
        raise ValueError(f"Priority8 nodes missing from node statistics: {sorted(wanted-seen)}")
    return float(total)


def _load_baseline_master(path: Path, benchmark_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != OPERATIONAL_BASELINE_CACHE_CONTRACT:
        raise ValueError("wrong operational baseline cache contract")
    if str(payload.get("benchmark_manifest_sha256", "")).lower() != sha256_file(benchmark_path).lower():
        raise ValueError("baseline cache was frozen for another five-event benchmark")
    if tuple(payload.get("competitive_baselines", ())) != OPERATIONAL_COMPARATORS:
        raise ValueError("operational baseline cache comparator set drifted")
    return payload


def _baseline_rows(cache_path: Path, event_id: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("contract") != OPERATIONAL_BASELINE_CACHE_CONTRACT or str(payload.get("event_id")) != event_id:
        raise ValueError(f"invalid baseline cache for {event_id}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("baseline event cache lacks rows")
    by = {str(row["strategy"]): dict(row) for row in rows}
    if tuple(by) != OPERATIONAL_COMPARATORS:
        raise ValueError("baseline event cache comparator set/order drifted")
    return by


def _metadata_and_stats(metadata_path: Path) -> tuple[dict[str, Any], Path]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Proposed metadata must be an object")
    stats = metadata_path.with_name(metadata_path.stem + ".node_statistics.csv.gz")
    if not stats.is_file():
        candidate = metadata.get("node_statistics_path") or metadata.get("node_statistics_file")
        if candidate:
            candidate_path = Path(str(candidate))
            stats = candidate_path if candidate_path.is_absolute() else metadata_path.parent / candidate_path
    if not stats.is_file():
        raise FileNotFoundError(f"Proposed node statistics not found for {metadata_path}")
    return metadata, stats.resolve()


def _run_proposed(
    *, event: dict[str, Any], asset_manifest: Path, v15_rank: Path, v21_boundary: Path,
    root: Path, device: str, budget: float, probe_chunk: int,
) -> tuple[dict[str, Any], Path, Path]:
    event_id = str(event["event_id"])
    event_root = root / event_id
    run_id = f"{event_id}__proposed"
    metadata_path = event_root / f"{run_id}.json"
    if metadata_path.exists():
        raise FileExistsError(
            f"Proposed result already exists for {event_id}; use a fresh proposal-version --out-dir"
        )
    event_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_policy_direct_tfv_operational_v21_development.py"),
        "--asset-manifest", str(asset_manifest),
        "--inp", str(event["inp_path"]),
        "--out-dir", str(event_root),
        "--run-id", run_id,
        "--v15-rank-checkpoint", str(v15_rank),
        "--v21-boundary-checkpoint", str(v21_boundary),
        "--device", device,
        "--decision-runtime-budget-seconds", str(float(budget)),
        "--probe-chunk-size", str(int(probe_chunk)),
    ]
    subprocess.run(command, check=True)
    metadata, stats = _metadata_and_stats(metadata_path)
    if metadata.get("operational_steering_only") is not True or metadata.get("formal_evidence") is not False:
        raise RuntimeError("Proposed operational run lost Development-only firewall")
    if str(metadata.get("source_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError("Proposed run used another event INP")
    return metadata, metadata_path.resolve(), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_manifest).resolve()
    benchmark = load_benchmark_manifest(benchmark_path)
    baseline_master = _load_baseline_master(Path(args.baseline_cache).resolve(), benchmark_path)
    cache_by_event = {str(row["event_id"]): Path(str(row["cache_path"])).resolve() for row in baseline_master["events"]}
    if set(cache_by_event) != {str(event["event_id"]) for event in benchmark["events"]}:
        raise ValueError("baseline cache and benchmark event IDs differ")
    priority = _priority_nodes(args.priority_nodes)
    asset_manifest = Path(args.asset_manifest).resolve()
    v15_rank = Path(args.v15_rank_checkpoint).resolve()
    v21_boundary = Path(args.v21_boundary_checkpoint).resolve()
    for required in (asset_manifest, v15_rank, v21_boundary):
        if not required.is_file():
            raise FileNotFoundError(required)
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    event_results: list[dict[str, Any]] = []
    for event in benchmark["events"]:
        event_id = str(event["event_id"])
        baselines = _baseline_rows(cache_by_event[event_id], event_id)
        proposed_meta, proposed_meta_path, proposed_stats = _run_proposed(
            event=event,
            asset_manifest=asset_manifest,
            v15_rank=v15_rank,
            v21_boundary=v21_boundary,
            root=root,
            device=args.device,
            budget=float(args.decision_runtime_budget_seconds),
            probe_chunk=int(args.probe_chunk_size),
        )
        proposed_tfv = tfv_m3(proposed_stats)
        proposed_pfv = _pfv(proposed_stats, priority)
        no_control_pfv = _pfv(baselines["no_control"]["node_statistics_path"], priority)
        pfv_limit = 100.0 + 1.05 * no_control_pfv
        comparisons: dict[str, Any] = {}
        for strategy in OPERATIONAL_COMPARATORS:
            row = baselines[strategy]
            base_tfv = float(row["tfv_m3"])
            base_pfv = _pfv(row["node_statistics_path"], priority)
            comparisons[strategy] = {
                "baseline_tfv_m3": base_tfv,
                "proposed_minus_baseline_tfv_m3": proposed_tfv - base_tfv,
                "tfv_reduction_pct": (100.0 * (base_tfv - proposed_tfv) / base_tfv if base_tfv > 0.0 else None),
                "proposed_better_tfv": bool(proposed_tfv < base_tfv - 1.0e-9),
                "baseline_pfv_m3": base_pfv,
                "proposed_minus_baseline_pfv_m3": proposed_pfv - base_pfv,
                "baseline_global_peak_flood_rate_m3s": float(row.get("global_peak_flood_rate_m3s", 0.0)),
            }
        event_results.append({
            "event_id": event_id,
            "proposed_metadata_path": str(proposed_meta_path),
            "proposed_node_statistics_path": str(proposed_stats),
            "proposed_tfv_m3": proposed_tfv,
            "proposed_pfv_m3": proposed_pfv,
            "pfv_no_control_m3": no_control_pfv,
            "pfv_safety_limit_m3": pfv_limit,
            "proposed_pfv_safety_pass": bool(proposed_pfv <= pfv_limit + 1.0e-9),
            "proposed_global_peak_flood_rate_m3s": float(proposed_meta.get("global_peak_flood_rate_m3s", 0.0)),
            "flow_routing_error_pct": float(proposed_meta.get("flow_routing_error_pct", 0.0)),
            "decisions": int(proposed_meta.get("decisions", 0)),
            "comparisons": comparisons,
        })

    aggregate: dict[str, Any] = {}
    for strategy in OPERATIONAL_COMPARATORS:
        reductions = [float(event["comparisons"][strategy]["tfv_reduction_pct"]) for event in event_results if event["comparisons"][strategy]["tfv_reduction_pct"] is not None]
        deltas = [float(event["comparisons"][strategy]["proposed_minus_baseline_tfv_m3"]) for event in event_results]
        proposal_sum = sum(float(event["proposed_tfv_m3"]) for event in event_results)
        baseline_sum = sum(float(event["comparisons"][strategy]["baseline_tfv_m3"]) for event in event_results)
        aggregate[strategy] = {
            "events_proposed_better_tfv": sum(bool(event["comparisons"][strategy]["proposed_better_tfv"]) for event in event_results),
            "event_count": 5,
            "event_balanced_mean_tfv_reduction_pct": statistics.mean(reductions) if reductions else None,
            "event_balanced_median_tfv_reduction_pct": statistics.median(reductions) if reductions else None,
            "mean_proposed_minus_baseline_tfv_m3": statistics.mean(deltas),
            "aggregate_volume_tfv_reduction_pct": (100.0 * (baseline_sum - proposal_sum) / baseline_sum if baseline_sum > 0.0 else None),
        }

    pfv_pass_count = sum(bool(event["proposed_pfv_safety_pass"]) for event in event_results)
    payload = {
        "contract": OPERATIONAL_COMPARISON_CONTRACT,
        "development_only": True,
        "operational_steering_only": True,
        "formal_evidence": False,
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "baseline_cache_sha256": sha256_file(args.baseline_cache),
        "v15_rank_checkpoint_sha256": sha256_file(v15_rank),
        "proposal_boundary_checkpoint_sha256": sha256_file(v21_boundary),
        "event_count": 5,
        "competitive_baselines": list(OPERATIONAL_COMPARATORS),
        "tfv_primary": True,
        "pfv_role": "secondary_authoritative_no_control_noninferiority_safety",
        "global_peak_role": "report_only",
        "event_results": event_results,
        "aggregate": aggregate,
        "pfv_safety_pass_count": pfv_pass_count,
        "pfv_safety_all_events_pass": bool(pfv_pass_count == 5),
        "operational_signal": {
            "beats_no_control_event_count": aggregate["no_control"]["events_proposed_better_tfv"],
            "beats_internal_rtc_event_count": aggregate["internal_rtc"]["events_proposed_better_tfv"],
            "beats_auto_rbc_event_count": aggregate["auto_rbc"]["events_proposed_better_tfv"],
            "beats_efd_event_count": aggregate["efd"]["events_proposed_better_tfv"],
            "all_four_event_balanced_mean_tfv_reductions_positive": all(
                aggregate[strategy]["event_balanced_mean_tfv_reduction_pct"] is not None
                and float(aggregate[strategy]["event_balanced_mean_tfv_reduction_pct"]) > 0.0
                for strategy in OPERATIONAL_COMPARATORS
            ),
        },
        "ready_for_policy_lock": False,
    }
    out_json = root / "OPERATIONAL_BENCHMARK5_COMPARISON.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_csv = root / "OPERATIONAL_BENCHMARK5_EVENT_TABLE.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = ["event_id", "proposed_tfv_m3", "proposed_pfv_m3", "proposed_pfv_safety_pass"]
        for strategy in OPERATIONAL_COMPARATORS:
            fields += [f"{strategy}_tfv_m3", f"tfv_reduction_vs_{strategy}_pct", f"proposed_better_than_{strategy}"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in event_results:
            row: dict[str, Any] = {
                "event_id": event["event_id"],
                "proposed_tfv_m3": event["proposed_tfv_m3"],
                "proposed_pfv_m3": event["proposed_pfv_m3"],
                "proposed_pfv_safety_pass": event["proposed_pfv_safety_pass"],
            }
            for strategy in OPERATIONAL_COMPARATORS:
                comp = event["comparisons"][strategy]
                row[f"{strategy}_tfv_m3"] = comp["baseline_tfv_m3"]
                row[f"tfv_reduction_vs_{strategy}_pct"] = comp["tfv_reduction_pct"]
                row[f"proposed_better_than_{strategy}"] = comp["proposed_better_tfv"]
            writer.writerow(row)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
