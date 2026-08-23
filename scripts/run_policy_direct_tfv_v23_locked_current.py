"""Run one Policy-Locked Project7 V23 Final event in authoritative SWMM.

This is the publication-facing Proposed runner.  It refuses to run without an immutable Policy Lock
and the frozen Final6 manifest, verifies checkpoint/asset/event hashes, then executes exactly the same
causal V23 controller that was locked. Final output is evaluation evidence only and can never feed
back into training or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.direct_tfv_operational_v23_runtime import (
    OPERATIONAL_V23_RUNTIME_CONTRACT,
    build_operational_v23_controller,
)
from rtc.event_clock import inspect_prepared_event_clock
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES, validate_project7_runtime_config


POLICY_LOCK_CONTRACT = "PROJECT7_V23_POLICY_LOCK_V1"
FINAL_EVENT_MANIFEST_CONTRACT = "PROJECT7_V23_FROZEN_FINAL6_EVENT_MANIFEST_V1"
LOCKED_FINAL_RUN_CONTRACT = "PROJECT7_V23_POLICY_LOCKED_FINAL_PROPOSED_RUN_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--final-event-manifest", required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()

    lock_path = Path(args.policy_lock).resolve()
    final_manifest_path = Path(args.final_event_manifest).resolve()
    lock = _json(lock_path)
    manifest = _json(final_manifest_path)
    if lock.get("contract") != POLICY_LOCK_CONTRACT or lock.get("locked") is not True:
        raise RuntimeError("Final Proposed requires an immutable V23 Policy Lock")
    if lock.get("ready_for_final") is not True or lock.get("final_opened_at_lock") is not False:
        raise RuntimeError("Policy Lock is not in the sealed-ready-for-Final state")
    if manifest.get("contract") != FINAL_EVENT_MANIFEST_CONTRACT:
        raise ValueError("wrong frozen Final event manifest contract")
    if manifest.get("hydraulic_outcomes_opened") is not False:
        raise RuntimeError("Final manifest was not frozen outcome-blind")
    event_id = str(args.event_id)
    if event_id not in set(str(value) for value in lock.get("final_event_ids", ())):
        raise ValueError(f"event is not in locked Final cohort: {event_id}")
    rows = manifest.get("events")
    if not isinstance(rows, list):
        raise ValueError("Final event manifest lacks events")
    matches = [row for row in rows if isinstance(row, dict) and str(row.get("event_id")) == event_id]
    if len(matches) != 1:
        raise ValueError(f"Final manifest does not uniquely resolve {event_id}")
    event = matches[0]
    source_inp = Path(str(event["inp_path"])).resolve()
    if not source_inp.is_file() or _sha(source_inp).lower() != str(event["inp_sha256"]).lower():
        raise RuntimeError("Final event INP changed after manifest freeze")

    asset_path = Path(args.asset_manifest).resolve()
    rank_path = Path(args.v15_rank_checkpoint).resolve()
    boundary_path = Path(args.v21_boundary_checkpoint).resolve()
    expected = {
        "asset_manifest_sha256": _sha(asset_path),
        "v15_rank_checkpoint_sha256": _sha(rank_path),
        "v21_boundary_checkpoint_sha256": _sha(boundary_path),
    }
    for key, actual in expected.items():
        if str(lock.get(key, "")).lower() != actual.lower():
            raise RuntimeError(f"Policy-Lock hash mismatch: {key}")

    if not 0.0 < float(args.decision_runtime_budget_seconds) < 600.0:
        raise ValueError("controller runtime budget must fit inside the 600-s update")
    assets = load_practical_rtc_asset_manifest(asset_path)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")

    graph_path = practical_asset_path(assets, "graph")
    sensors_path = practical_asset_path(assets, "sensors")
    config_path = practical_asset_path(assets, "config")
    step1_path = practical_asset_path(assets, "step1")
    step2_path = practical_asset_path(assets, "step2")
    supervisory_control_path = practical_asset_path(assets, "supervisory_control")
    sequence_support_path = practical_asset_path(assets, "sequence_support")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    clock = inspect_prepared_event_clock(source_inp)
    if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1.0e-6:
        raise ValueError("Final event violates common 120-min warm-up clock")

    controller, _graph, sensors, lineage = build_operational_v23_controller(
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=supervisory_control_path,
        sequence_support_path=sequence_support_path,
        v15_rank_checkpoint_path=rank_path,
        v21_boundary_checkpoint_path=boundary_path,
        device=device,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )
    out_dir = Path(args.out_dir).resolve()
    metadata_path = out_dir / f"{args.run_id}.json"
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    runtime_inp = _controls_disabled_runtime(
        source_inp=source_inp,
        cache_dir=out_dir / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=out_dir,
        run_id=args.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=int(cfg["control_start_minutes"]),
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=False,
    )
    write_audit = audit_target_write_readback_v127(metadata_path=result.metadata_path)
    if write_audit.get("passed") is not True:
        raise RuntimeError("locked V23 Final run failed target write/readback audit")

    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "contract": LOCKED_FINAL_RUN_CONTRACT,
            "strategy": "proposed",
            "formal_evidence": True,
            "development_only": False,
            "operational_steering_only": False,
            "policy_locked": True,
            "policy_lock_path": str(lock_path),
            "policy_lock_sha256": _sha(lock_path),
            "formal_mode": lock.get("formal_mode"),
            "event_id": event_id,
            "final_event_manifest_sha256": _sha(final_manifest_path),
            "final_result_used_for_training": False,
            "final_result_used_for_tuning": False,
            "runtime_contract": OPERATIONAL_V23_RUNTIME_CONTRACT,
            "tfv_primary": True,
            "pfv_role": "secondary_authoritative_no_control_noninferiority_safety",
            "global_peak_role": "report_only",
            "future_realized_rainfall_used_as_model_input": False,
            "online_swmm_candidate_search": False,
            "online_lbfgsb_used": False,
            "projected_gradient_h10_enabled": False,
            "conformal_admission_used": False,
            "v15_rank_checkpoint_sha256": _sha(rank_path),
            "v21_boundary_checkpoint_sha256": _sha(boundary_path),
            "asset_manifest_sha256": _sha(asset_path),
            "source_inp_path": str(source_inp),
            "source_inp_sha256": _sha(source_inp),
            "runtime_inp_sha256": _sha(runtime_inp),
            "controller_config_sha256": _sha(config_path),
            "prepared_event_clock": clock,
            "target_write_readback_audit": write_audit,
            "project7_runtime_contract": project_contract,
            "runtime_factory_lineage": lineage,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "contract": LOCKED_FINAL_RUN_CONTRACT,
                "event_id": event_id,
                "metadata_path": str(metadata_path),
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "flow_routing_error_pct": result.flow_routing_error_pct,
                "target_write_readback_passed": True,
                "formal_evidence": True,
                "policy_locked": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
