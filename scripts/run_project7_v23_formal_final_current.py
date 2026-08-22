"""Run the complete Policy-Locked Project7 V23 Final6 comparison once.

For each preregistered Final event this command runs locked Proposed plus the four competitive
comparators (No-control, Internal RTC, Auto-RBC, EFD) sequentially in authoritative SWMM.  It refuses
overwrite/re-entry, preserves the Final firewall, computes TFV as the primary metric, Priority8 PFV
non-inferiority as the safety metric, and reports Global Peak without gating.
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
from rtc.baselines import COMPETITIVE_BASELINE_IDS
from rtc.direct_tfv_policy_return import sha256_file
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


FINAL_COMPARISON_CONTRACT = "PROJECT7_V23_POLICY_LOCKED_FINAL6_COMPARISON_V1"
POLICY_LOCK_CONTRACT = "PROJECT7_V23_POLICY_LOCK_V1"
FINAL_EVENT_MANIFEST_CONTRACT = "PROJECT7_V23_FROZEN_FINAL6_EVENT_MANIFEST_V1"


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _priority_nodes(path: str | Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(values) != 8 or len(set(values)) != 8:
        raise ValueError("Formal Final requires exactly eight frozen Priority8 nodes")
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
    return total


def _metadata_stats(metadata_path: Path) -> tuple[dict[str, Any], Path]:
    metadata = _json(metadata_path)
    stats = metadata_path.with_name(metadata_path.stem + ".node_statistics.csv.gz")
    if not stats.is_file():
        candidate = metadata.get("node_statistics_path") or metadata.get("node_statistics_file")
        if candidate:
            raw = Path(str(candidate))
            stats = raw if raw.is_absolute() else metadata_path.parent / raw
    if not stats.is_file():
        raise FileNotFoundError(stats)
    return metadata, stats.resolve()


def _run_baseline(
    *,
    strategy: str,
    event: dict[str, Any],
    root: Path,
    sensors: Path,
    config: Path,
    native_controls_template: Path,
    policy_lock_sha256: str,
) -> tuple[dict[str, Any], Path, Path]:
    event_id = str(event["event_id"])
    out = root / event_id / strategy
    run_id = f"{event_id}__{strategy}"
    metadata_path = out / f"{run_id}.json"
    if metadata_path.exists():
        raise FileExistsError(
            f"Final comparator result already exists; Final is single-use: {metadata_path}"
        )
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "rtc.production_guard",
        "--strategy",
        strategy,
        "--inp",
        str(event["inp_path"]),
        "--out-dir",
        str(out),
        "--run-id",
        run_id,
        "--sensors",
        str(sensors),
        "--config",
        str(config),
        "--runtime-inp-cache-dir",
        str(root / "_baseline_runtime_inp"),
    ]
    if strategy == "internal_rtc":
        command.extend(["--native-controls-template", str(native_controls_template)])
    subprocess.run(command, check=True)
    metadata, stats = _metadata_stats(metadata_path)
    if str(metadata.get("strategy", "")) != strategy:
        raise RuntimeError(f"Final comparator strategy mismatch: {strategy}")
    if str(metadata.get("source_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError(f"Final comparator used another forcing event: {strategy}")
    if strategy in {"auto_rbc", "efd"}:
        audit = audit_target_write_readback_v127(metadata_path=metadata_path)
        if audit.get("passed") is not True:
            raise RuntimeError(f"Final comparator write/readback failed: {strategy}")
        metadata["target_write_readback_audit"] = audit
    metadata.update(
        {
            "formal_final_comparator": True,
            "formal_evidence": True,
            "development_only": False,
            "policy_lock_sha256": policy_lock_sha256,
            "final_result_used_for_training": False,
            "final_result_used_for_tuning": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata, metadata_path.resolve(), stats


def _run_proposed(
    *,
    event: dict[str, Any],
    root: Path,
    policy_lock: Path,
    final_manifest: Path,
    asset_manifest: Path,
    rank_checkpoint: Path,
    boundary_checkpoint: Path,
    device: str,
    budget: float,
    probe_chunk: int,
) -> tuple[dict[str, Any], Path, Path]:
    event_id = str(event["event_id"])
    out = root / event_id / "proposed"
    run_id = f"{event_id}__proposed"
    metadata_path = out / f"{run_id}.json"
    if metadata_path.exists():
        raise FileExistsError(f"Final Proposed result already exists: {metadata_path}")
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_policy_direct_tfv_v23_locked_current.py"),
        "--policy-lock",
        str(policy_lock),
        "--final-event-manifest",
        str(final_manifest),
        "--event-id",
        event_id,
        "--asset-manifest",
        str(asset_manifest),
        "--v15-rank-checkpoint",
        str(rank_checkpoint),
        "--v21-boundary-checkpoint",
        str(boundary_checkpoint),
        "--out-dir",
        str(out),
        "--run-id",
        run_id,
        "--device",
        device,
        "--decision-runtime-budget-seconds",
        str(float(budget)),
        "--probe-chunk-size",
        str(int(probe_chunk)),
    ]
    subprocess.run(command, check=True)
    metadata, stats = _metadata_stats(metadata_path)
    if metadata.get("formal_evidence") is not True or metadata.get("policy_locked") is not True:
        raise RuntimeError("Proposed Final run was not Policy-Locked Formal evidence")
    return metadata, metadata_path.resolve(), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--final-event-manifest", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--native-controls-template", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()

    lock_path = Path(args.policy_lock).resolve()
    manifest_path = Path(args.final_event_manifest).resolve()
    lock = _json(lock_path)
    manifest = _json(manifest_path)
    if lock.get("contract") != POLICY_LOCK_CONTRACT or lock.get("ready_for_final") is not True:
        raise RuntimeError("Final6 comparison requires a ready V23 Policy Lock")
    if manifest.get("contract") != FINAL_EVENT_MANIFEST_CONTRACT:
        raise ValueError("wrong frozen Final6 manifest")
    if set(manifest.get("event_ids", ())) != set(lock.get("final_event_ids", ())):
        raise RuntimeError("Policy Lock and Final6 manifest event IDs differ")

    asset_manifest = Path(args.asset_manifest).resolve()
    rank_checkpoint = Path(args.v15_rank_checkpoint).resolve()
    boundary_checkpoint = Path(args.v21_boundary_checkpoint).resolve()
    native = Path(args.native_controls_template).resolve()
    for path in (asset_manifest, rank_checkpoint, boundary_checkpoint, native):
        if not path.is_file():
            raise FileNotFoundError(path)
    assets = load_practical_rtc_asset_manifest(asset_manifest)
    sensors = practical_asset_path(assets, "sensors")
    config = practical_asset_path(assets, "config")
    priority = _priority_nodes(args.priority_nodes)
    root = Path(args.out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Formal Final output directory must be new and empty")
    root.mkdir(parents=True, exist_ok=True)
    lock_sha = sha256_file(lock_path)

    event_results: list[dict[str, Any]] = []
    for event in manifest["events"]:
        event_id = str(event["event_id"])
        proposed_meta, proposed_meta_path, proposed_stats = _run_proposed(
            event=event,
            root=root,
            policy_lock=lock_path,
            final_manifest=manifest_path,
            asset_manifest=asset_manifest,
            rank_checkpoint=rank_checkpoint,
            boundary_checkpoint=boundary_checkpoint,
            device=args.device,
            budget=float(args.decision_runtime_budget_seconds),
            probe_chunk=int(args.probe_chunk_size),
        )
        baseline_rows: dict[str, dict[str, Any]] = {}
        for strategy in COMPETITIVE_BASELINE_IDS:
            meta, meta_path, stats = _run_baseline(
                strategy=strategy,
                event=event,
                root=root,
                sensors=sensors,
                config=config,
                native_controls_template=native,
                policy_lock_sha256=lock_sha,
            )
            baseline_rows[strategy] = {
                "metadata": meta,
                "metadata_path": str(meta_path),
                "stats": stats,
                "tfv_m3": tfv_m3(stats),
                "pfv_m3": _pfv(stats, priority),
            }
        proposed_tfv = tfv_m3(proposed_stats)
        proposed_pfv = _pfv(proposed_stats, priority)
        no_control_pfv = float(baseline_rows["no_control"]["pfv_m3"])
        pfv_limit = 100.0 + 1.05 * no_control_pfv
        comparisons: dict[str, Any] = {}
        for strategy in COMPETITIVE_BASELINE_IDS:
            baseline = baseline_rows[strategy]
            baseline_tfv = float(baseline["tfv_m3"])
            baseline_meta = baseline["metadata"]
            comparisons[strategy] = {
                "baseline_tfv_m3": baseline_tfv,
                "baseline_pfv_m3": float(baseline["pfv_m3"]),
                "proposed_minus_baseline_tfv_m3": proposed_tfv - baseline_tfv,
                "tfv_reduction_pct": (
                    100.0 * (baseline_tfv - proposed_tfv) / baseline_tfv
                    if baseline_tfv > 0.0
                    else None
                ),
                "proposed_better_tfv": bool(proposed_tfv < baseline_tfv - 1.0e-9),
                "baseline_global_peak_flood_rate_m3s": float(
                    baseline_meta.get("global_peak_flood_rate_m3s", 0.0)
                ),
            }
        event_results.append(
            {
                "event_id": event_id,
                "proposed_metadata_path": str(proposed_meta_path),
                "proposed_tfv_m3": proposed_tfv,
                "proposed_pfv_m3": proposed_pfv,
                "pfv_no_control_m3": no_control_pfv,
                "pfv_safety_limit_m3": pfv_limit,
                "proposed_pfv_safety_pass": bool(proposed_pfv <= pfv_limit + 1.0e-9),
                "proposed_global_peak_flood_rate_m3s": float(
                    proposed_meta.get("global_peak_flood_rate_m3s", 0.0)
                ),
                "flow_routing_error_pct": float(proposed_meta.get("flow_routing_error_pct", 0.0)),
                "decisions": int(proposed_meta.get("decisions", 0)),
                "target_write_readback_passed": bool(
                    proposed_meta.get("target_write_readback_audit", {}).get("passed") is True
                ),
                "comparisons": comparisons,
            }
        )

    aggregate: dict[str, Any] = {}
    for strategy in COMPETITIVE_BASELINE_IDS:
        reductions = [
            float(event["comparisons"][strategy]["tfv_reduction_pct"])
            for event in event_results
            if event["comparisons"][strategy]["tfv_reduction_pct"] is not None
        ]
        proposed_sum = sum(float(event["proposed_tfv_m3"]) for event in event_results)
        baseline_sum = sum(
            float(event["comparisons"][strategy]["baseline_tfv_m3"])
            for event in event_results
        )
        aggregate[strategy] = {
            "events_proposed_better_tfv": sum(
                bool(event["comparisons"][strategy]["proposed_better_tfv"])
                for event in event_results
            ),
            "event_count": 6,
            "event_balanced_mean_tfv_reduction_pct": statistics.mean(reductions),
            "event_balanced_median_tfv_reduction_pct": statistics.median(reductions),
            "aggregate_volume_tfv_reduction_pct": (
                100.0 * (baseline_sum - proposed_sum) / baseline_sum if baseline_sum > 0.0 else None
            ),
        }

    pfv_pass_count = sum(bool(event["proposed_pfv_safety_pass"]) for event in event_results)
    engineering_pass = all(
        abs(float(event["flow_routing_error_pct"])) < 1.0e-9
        and bool(event["target_write_readback_passed"])
        for event in event_results
    )
    payload = {
        "contract": FINAL_COMPARISON_CONTRACT,
        "formal_evidence": True,
        "policy_locked": True,
        "policy_lock_sha256": lock_sha,
        "final_event_manifest_sha256": sha256_file(manifest_path),
        "event_count": 6,
        "competitive_baselines": list(COMPETITIVE_BASELINE_IDS),
        "tfv_primary": True,
        "pfv_role": "secondary_authoritative_no_control_noninferiority_safety",
        "global_peak_role": "report_only",
        "final_results_used_for_training": False,
        "final_results_used_for_tuning": False,
        "event_results": event_results,
        "aggregate": aggregate,
        "pfv_safety_pass_count": pfv_pass_count,
        "pfv_safety_all_events_pass": bool(pfv_pass_count == 6),
        "engineering_all_events_pass": engineering_pass,
        "ready_for_paper_reporting": bool(pfv_pass_count == 6 and engineering_pass),
    }
    out_json = root / "PROJECT7_V23_FORMAL_FINAL6_COMPARISON.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_csv = root / "PROJECT7_V23_FORMAL_FINAL6_EVENT_TABLE.csv"
    fields = ["event_id", "proposed_tfv_m3", "proposed_pfv_m3", "pfv_safety_pass"]
    for strategy in COMPETITIVE_BASELINE_IDS:
        fields.extend(
            [
                f"{strategy}_tfv_m3",
                f"tfv_reduction_vs_{strategy}_pct",
                f"proposed_better_than_{strategy}",
            ]
        )
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in event_results:
            row: dict[str, Any] = {
                "event_id": event["event_id"],
                "proposed_tfv_m3": event["proposed_tfv_m3"],
                "proposed_pfv_m3": event["proposed_pfv_m3"],
                "pfv_safety_pass": event["proposed_pfv_safety_pass"],
            }
            for strategy in COMPETITIVE_BASELINE_IDS:
                comp = event["comparisons"][strategy]
                row[f"{strategy}_tfv_m3"] = comp["baseline_tfv_m3"]
                row[f"tfv_reduction_vs_{strategy}_pct"] = comp["tfv_reduction_pct"]
                row[f"proposed_better_than_{strategy}"] = comp["proposed_better_tfv"]
            writer.writerow(row)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
