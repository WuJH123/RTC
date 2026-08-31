"""Run one information-matched Project7 baseline V2 event.

Auto-RBC, topology-aware EFD, and matched Internal RTC use the same sparse sensors, frozen Step1
reconstruction, update interval, supervisory authority, and engineering projection as Proposed.
They are independent comparators: their performance never gates Proposed action selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.event_clock import inspect_prepared_event_clock
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES, validate_project7_runtime_config
from rtc.project7_matched_baselines import MATCHED_ACTIVE_BASELINES, MATCHED_INTERNAL_RTC
from rtc.project7_matched_baselines_v2 import (
    MATCHED_BASELINE_V2_CONTRACT,
    build_matched_information_baseline_v2_controller,
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _publication_lineage(lineage: dict) -> dict:
    cleaned = dict(lineage)
    for obsolete_flag in (
        "development_only",
        "formal_evidence",
        "requires_new_policy_lock",
        "ready_for_policy_lock",
    ):
        cleaned.pop(obsolete_flag, None)
    cleaned.update(
        {
            "publication_candidate": True,
            "baseline_performance_gates_proposed": False,
            "historical_completed_outcomes_may_inform_development": True,
            "training_evaluation_split_must_remain_disjoint": True,
        }
    )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=MATCHED_ACTIVE_BASELINES)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--native-controls-inp")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    device = torch.device(args.device)
    graph_path = practical_asset_path(assets, "graph")
    sensors_path = practical_asset_path(assets, "sensors")
    config_path = practical_asset_path(assets, "config")
    step1_path = practical_asset_path(assets, "step1")
    step2_path = practical_asset_path(assets, "step2")
    control_path = practical_asset_path(assets, "supervisory_control")
    support_path = practical_asset_path(assets, "sequence_support")
    source_inp = Path(args.inp).resolve()
    if not source_inp.is_file():
        raise FileNotFoundError(source_inp)
    native_controls_inp = Path(args.native_controls_inp).resolve() if args.native_controls_inp else None
    if args.strategy == MATCHED_INTERNAL_RTC and native_controls_inp is None:
        raise ValueError("--native-controls-inp is required for matched_internal_rtc")
    if native_controls_inp is not None and not native_controls_inp.is_file():
        raise FileNotFoundError(native_controls_inp)

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    clock = inspect_prepared_event_clock(source_inp)
    if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1.0e-6:
        raise ValueError("event violates common warm-up clock")

    controller, _, sensors, lineage = build_matched_information_baseline_v2_controller(
        matched_strategy=args.strategy,
        source_inp_path=str(source_inp),
        native_controls_inp_path=str(native_controls_inp) if native_controls_inp is not None else None,
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=control_path,
        sequence_support_path=support_path,
        v15_rank_checkpoint_path=args.v15_rank_checkpoint,
        v21_boundary_checkpoint_path=args.v21_boundary_checkpoint,
        device=device,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )
    lineage = _publication_lineage(lineage)
    out_dir = Path(args.out_dir).resolve()
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
        raise RuntimeError(f"{args.strategy} V2 failed target write/readback audit")

    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy": args.strategy,
            "matched_baseline_contract": MATCHED_BASELINE_V2_CONTRACT,
            "fair_comparator_claim_eligible": True,
            "baseline_performance_gates_proposed": False,
            "historical_completed_outcomes_may_inform_development": True,
            "training_evaluation_split_must_remain_disjoint": True,
            "same_sparse_sensor_set_as_proposed": True,
            "same_frozen_step1_reconstruction_as_proposed": True,
            "same_rainfall_forecast_as_proposed": True,
            "same_82_channel_supervisory_mask": True,
            "same_passive_27_channels": True,
            "same_q95_changed_facility_ceiling": True,
            "same_q95_joint_sequence_support": True,
            "same_max_setting_delta_per_update": 0.5,
            "same_target_latch_semantics": True,
            "source_inp_path": str(source_inp),
            "source_inp_sha256": _sha(source_inp),
            "native_controls_inp_path": str(native_controls_inp) if native_controls_inp is not None else None,
            "native_controls_inp_sha256": _sha(native_controls_inp) if native_controls_inp is not None else None,
            # Hashes below are provenance records, not comparison/admission gates.
            "asset_manifest_sha256": _sha(args.asset_manifest),
            "step1_model_sha256": _sha(step1_path),
            "step2_model_sha256": _sha(step2_path),
            "sensor_set_sha256": _sha(sensors_path),
            "supervisory_control_sha256": _sha(control_path),
            "sequence_support_sha256": _sha(support_path),
            "runtime_inp_path": str(Path(runtime_inp).resolve()),
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
                "strategy": args.strategy,
                "matched_baseline_contract": MATCHED_BASELINE_V2_CONTRACT,
                "metadata_path": str(metadata_path),
                "decision_path": result.decision_path,
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "flow_routing_error_pct": result.flow_routing_error_pct,
                "target_write_readback_passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
