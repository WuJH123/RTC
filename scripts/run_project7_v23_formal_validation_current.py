"""Run the frozen six-event Project7 V23 Formal Development-Validation panel.

This pre-lock evaluator uses current frozen V23 Proposed plus the four competitive comparators on the
preregistered Validation events. Results may decide whether the already-fixed policy is safe/engineered
enough to lock, but must never update model parameters, thresholds or candidate rules. Final remains
sealed throughout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from rtc.baseline_panel import tfv_m3
from rtc.baselines import COMPETITIVE_BASELINE_IDS
from rtc.direct_tfv_policy_return import sha256_file
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path

import run_project7_v23_formal_final_current as common


VALIDATION_MANIFEST_CONTRACT = "PROJECT7_V23_FROZEN_DEVELOPMENT_VALIDATION6_MANIFEST_V1"
VALIDATION_EVIDENCE_CONTRACT = "PROJECT7_V23_FORMAL_DEVELOPMENT_VALIDATION_EVIDENCE_V1"


def _run_proposed(
    *,
    event: dict[str, Any],
    root: Path,
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
        raise FileExistsError(f"Validation Proposed result already exists: {metadata_path}")
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_policy_direct_tfv_operational_v23_development.py"),
        "--asset-manifest",
        str(asset_manifest),
        "--inp",
        str(event["inp_path"]),
        "--out-dir",
        str(out),
        "--run-id",
        run_id,
        "--v15-rank-checkpoint",
        str(rank_checkpoint),
        "--v21-boundary-checkpoint",
        str(boundary_checkpoint),
        "--device",
        device,
        "--decision-runtime-budget-seconds",
        str(float(budget)),
        "--probe-chunk-size",
        str(int(probe_chunk)),
    ]
    subprocess.run(command, check=True)
    metadata, stats = common._metadata_stats(metadata_path)
    if metadata.get("development_only") is not True or metadata.get("formal_evidence") is not False:
        raise RuntimeError("pre-lock Validation Proposed lost its Development firewall")
    if str(metadata.get("source_inp_sha256", "")).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError("Validation Proposed used another forcing event")
    return metadata, metadata_path.resolve(), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-event-manifest", required=True)
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

    manifest_path = Path(args.validation_event_manifest).resolve()
    manifest = common._json(manifest_path)
    if manifest.get("contract") != VALIDATION_MANIFEST_CONTRACT:
        raise ValueError("wrong frozen Development-Validation manifest")
    if manifest.get("final_truth_opened") is not False:
        raise RuntimeError("Validation manifest indicates Final truth exposure")
    if len(manifest.get("events", ())) != 6:
        raise ValueError("Formal Development-Validation requires exactly six events")

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
    priority = common._priority_nodes(args.priority_nodes)
    root = Path(args.out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Formal Validation output directory must be new and empty")
    root.mkdir(parents=True, exist_ok=True)

    event_results: list[dict[str, Any]] = []
    for event in manifest["events"]:
        proposed_meta, _proposed_meta_path, proposed_stats = _run_proposed(
            event=event,
            root=root,
            asset_manifest=asset_manifest,
            rank_checkpoint=rank_checkpoint,
            boundary_checkpoint=boundary_checkpoint,
            device=args.device,
            budget=float(args.decision_runtime_budget_seconds),
            probe_chunk=int(args.probe_chunk_size),
        )
        baselines: dict[str, dict[str, Any]] = {}
        for strategy in COMPETITIVE_BASELINE_IDS:
            meta, meta_path, stats = common._run_baseline(
                strategy=strategy,
                event=event,
                root=root,
                sensors=sensors,
                config=config,
                native_controls_template=native,
                policy_lock_sha256="PRELOCK_VALIDATION_NO_POLICY_LOCK",
            )
            baselines[strategy] = {
                "metadata": meta,
                "metadata_path": str(meta_path),
                "stats": stats,
                "tfv_m3": tfv_m3(stats),
                "pfv_m3": common._pfv(stats, priority),
            }
        proposed_tfv = tfv_m3(proposed_stats)
        proposed_pfv = common._pfv(proposed_stats, priority)
        no_control_pfv = float(baselines["no_control"]["pfv_m3"])
        comparisons: dict[str, Any] = {}
        for strategy in COMPETITIVE_BASELINE_IDS:
            base_tfv = float(baselines[strategy]["tfv_m3"])
            comparisons[strategy] = {
                "baseline_tfv_m3": base_tfv,
                "proposed_minus_baseline_tfv_m3": proposed_tfv - base_tfv,
                "tfv_reduction_pct": (
                    100.0 * (base_tfv - proposed_tfv) / base_tfv if base_tfv > 0.0 else None
                ),
                "proposed_better_tfv": bool(proposed_tfv < base_tfv - 1.0e-9),
            }
        event_results.append(
            {
                "event_id": str(event["event_id"]),
                "proposed_tfv_m3": proposed_tfv,
                "proposed_pfv_m3": proposed_pfv,
                "pfv_no_control_m3": no_control_pfv,
                "pfv_safety_limit_m3": 100.0 + 1.05 * no_control_pfv,
                "proposed_pfv_safety_pass": bool(
                    proposed_pfv <= 100.0 + 1.05 * no_control_pfv + 1.0e-9
                ),
                "flow_routing_error_pct": float(proposed_meta.get("flow_routing_error_pct", 0.0)),
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
        aggregate[strategy] = {
            "events_proposed_better_tfv": sum(
                bool(event["comparisons"][strategy]["proposed_better_tfv"])
                for event in event_results
            ),
            "event_balanced_mean_tfv_reduction_pct": statistics.mean(reductions),
            "event_balanced_median_tfv_reduction_pct": statistics.median(reductions),
        }
    pfv_all = all(bool(event["proposed_pfv_safety_pass"]) for event in event_results)
    engineering_all = all(
        abs(float(event["flow_routing_error_pct"])) < 1.0e-9
        and bool(event["target_write_readback_passed"])
        for event in event_results
    )
    payload = {
        "contract": VALIDATION_EVIDENCE_CONTRACT,
        "formal_development_validation": True,
        "event_ids": [str(event["event_id"]) for event in event_results],
        "event_count": 6,
        "validation_event_manifest_sha256": sha256_file(manifest_path),
        "v15_rank_checkpoint_sha256": sha256_file(rank_checkpoint),
        "v21_boundary_checkpoint_sha256": sha256_file(boundary_checkpoint),
        "candidate_generator_contract": "PROJECT7_OPERATIONAL_V23_STRESS_ADAPTIVE_ABSOLUTE_FILL_RELEASE_TARGET_V1",
        "tfv_primary": True,
        "pfv_safety_all_events_pass": pfv_all,
        "engineering_all_events_pass": engineering_all,
        "policy_changed_after_validation_started": False,
        "validation_used_for_training": False,
        "final_truth_opened": False,
        "event_results": event_results,
        "aggregate": aggregate,
        "eligible_for_policy_lock": bool(pfv_all and engineering_all),
    }
    out = root / "PROJECT7_V23_FORMAL_DEVELOPMENT_VALIDATION_EVIDENCE.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["eligible_for_policy_lock"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
