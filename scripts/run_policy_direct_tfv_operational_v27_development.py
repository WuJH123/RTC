"""Run one Development-only Project7 V27 closed-loop event."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.direct_tfv_operational_v27_runtime import (
    OPERATIONAL_V27_RUNTIME_CONTRACT,
    build_operational_v27_controller,
)
from rtc.event_clock import inspect_prepared_event_clock
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES, validate_project7_runtime_config


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--v27-value-checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()
    if not 0.0 < float(args.decision_runtime_budget_seconds) < 600.0:
        raise ValueError("controller runtime budget must fit inside the 600-s update")
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
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    clock = inspect_prepared_event_clock(source_inp)
    if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1.0e-6:
        raise ValueError("event violates common warm-up clock")

    controller, _, sensors, lineage = build_operational_v27_controller(
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=control_path,
        sequence_support_path=support_path,
        v15_rank_checkpoint_path=args.v15_rank_checkpoint,
        v21_boundary_checkpoint_path=args.v21_boundary_checkpoint,
        v27_value_checkpoint_path=args.v27_value_checkpoint,
        dataset_manifest_path=args.dataset_manifest,
        asset_manifest_path=args.asset_manifest,
        device=device,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )
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
        raise RuntimeError("operational V27 failed target write/readback audit")

    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy": "proposed",
            "operational_development_runtime_contract": OPERATIONAL_V27_RUNTIME_CONTRACT,
            "development_only": True,
            "operational_steering_only": True,
            "formal_evidence": False,
            "tfv_primary": True,
            "pfv_role": "secondary_authoritative_no_control_noninferiority_safety",
            "global_peak_role": "report_only",
            "v27_value_estimand": lineage["v27_value_estimand"],
            "v27_selection_contract": lineage["v27_selection_contract"],
            "v27_portfolio_contract": lineage["v27_portfolio_contract"],
            "v27_auto_rbc_shadow_contract": lineage["v27_auto_rbc_shadow_contract"],
            "v27_auto_rbc_shadow_is_candidate_only": True,
            "v27_runtime_ranking_uses_unclipped_latent": True,
            "v27_q95_support_execution": True,
            "v27_q95_precontraction_counterfactual_scoring": True,
            "scientific_metrics_block_runtime": False,
            "v15_rank_used_for_v27_candidate_selection": False,
            "v21_boundary_used_for_v27_action_admission": False,
            "v25_ucb_used_for_v27_action_admission": False,
            "future_realized_rainfall_used_as_model_input": False,
            "online_swmm_candidate_search": False,
            "online_lbfgsb_used": False,
            "projected_gradient_h10_enabled": False,
            "supervisory_control_dimension": 82,
            "model_action_channel_count": 109,
            "passive_setting_channel_count": 27,
            "asset_manifest_sha256": _sha(args.asset_manifest),
            "v27_value_checkpoint_sha256": _sha(args.v27_value_checkpoint),
            "v27_dataset_manifest_sha256": _sha(args.dataset_manifest),
            "source_inp_path": str(source_inp),
            "source_inp_sha256": _sha(source_inp),
            "runtime_inp_path": str(Path(runtime_inp).resolve()),
            "runtime_inp_sha256": _sha(runtime_inp),
            "controller_config_sha256": _sha(config_path),
            "prepared_event_clock": clock,
            "target_write_readback_audit": write_audit,
            "project7_runtime_contract": project_contract,
            "runtime_factory_lineage": lineage,
            "ready_for_policy_lock": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "strategy": "proposed",
                "development_only": True,
                "formal_evidence": False,
                "metadata_path": str(metadata_path),
                "decision_path": result.decision_path,
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "flow_routing_error_pct": result.flow_routing_error_pct,
                "target_write_readback_passed": True,
                "scientific_metrics_block_runtime": False,
                "ready_for_policy_lock": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
