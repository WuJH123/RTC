"""Run the current Practical H10 policy-return controller in authoritative Development SWMM.

This entrypoint is intentionally path-safe and optimizer-agnostic. All frozen base artifacts come from
one SHA-verified absolute asset manifest. The policy-return checkpoint/admission must select the
current Practical H10-probe portfolio; legacy V12/L-BFGS-B policy-return bridges are rejected here and
remain available only to explicit offline parent-label tools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    load_policy_return_checkpoint,
)
from rtc.direct_tfv_policy_return_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
from rtc.direct_tfv_policy_return_runtime_factory import build_frozen_policy_return_continuation_controller
from rtc.event_clock import inspect_prepared_event_clock
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.production_cli import _controls_disabled_runtime, _load_graph
from rtc.project7_contract import EFFECTIVE_WARMUP_MINUTES, validate_project7_runtime_config
from rtc.step2_state_store_v127 import semantic_model_state_dict_sha256, semantic_sensor_layout_sha256
from rtc.step1_runtime_v127 import load_frozen_step1_v127


DIRECT_TFV_POLICY_RETURN_RUNTIME_CONTRACT = (
    "PROJECT7_PRACTICAL_H10_POLICY_RETURN_AUTHORITATIVE_DEVELOPMENT_RTC_V1"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_step1_lineage(*, step1_path: str, step2_path: str, sensors_path: str, graph_path: str, device: torch.device) -> None:
    graph = _load_graph(graph_path)
    step1 = load_frozen_step1_v127(step1_path, device)
    from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
    from rtc.production_cli import _load_lines

    _, _, checkpoint = load_direct_tfv_runtime_checkpoint(step2_path, graph=graph, device=device)
    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("base Direct-TFV checkpoint lacks Step1/sensor lineage")
    sensors = _load_lines(sensors_path)
    expected = {
        "step1_model_semantic_sha256": semantic_model_state_dict_sha256(step1),
        "sensor_layout_semantic_sha256": semantic_sensor_layout_sha256(sensors),
    }
    for key, actual in expected.items():
        if str(lineage.get(key, "")).lower() != actual.lower():
            raise ValueError(f"runtime Step1/sensor semantics differ from base Step2 training: {key}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-manifest", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--policy-return-checkpoint", required=True)
    p.add_argument("--policy-return-admission", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    args = p.parse_args()
    if not 0.0 < float(args.decision_runtime_budget_seconds) < 600.0:
        raise ValueError("Practical controller runtime budget must fit inside one 600-s update")

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("Practical RTC requested CUDA but CUDA is unavailable")
    graph_path = practical_asset_path(assets, "graph")
    sensors_path = practical_asset_path(assets, "sensors")
    config_path = practical_asset_path(assets, "config")
    step1_path = practical_asset_path(assets, "step1")
    step2_path = practical_asset_path(assets, "step2")
    sequence_support_path = practical_asset_path(assets, "sequence_support")

    _require_step1_lineage(
        step1_path=step1_path,
        step2_path=step2_path,
        sensors_path=sensors_path,
        graph_path=graph_path,
        device=device,
    )
    graph = _load_graph(graph_path)
    return_model, _, return_checkpoint = load_policy_return_checkpoint(
        args.policy_return_checkpoint,
        graph=graph,
        device=device,
        expected_base_step2_sha256=_sha(step2_path),
    )
    del return_model
    if str(return_checkpoint.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
        raise ValueError("current Development runtime accepts only the Practical H10 candidate portfolio")
    if str(return_checkpoint.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
        raise ValueError("current Development runtime critic has the wrong H10 action encoding")
    admission = json.loads(Path(args.policy_return_admission).read_text(encoding="utf-8"))
    if str(admission.get("contract", "")) != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT:
        raise ValueError("current Development runtime requires current policy-return admission")
    if str(admission.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
        raise ValueError("current Development runtime admission uses another candidate portfolio")
    if str(admission.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
        raise ValueError("current Development runtime admission uses another action encoding")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    clock = inspect_prepared_event_clock(args.inp)
    if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1e-6:
        raise ValueError("Practical event violates the common warm-up clock")

    controller, _, sensors, lineage = build_frozen_policy_return_continuation_controller(
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        policy_admission_path=practical_asset_path(assets, "policy_admission"),
        v12_first_move_admission_path=practical_asset_path(assets, "v12_first_move_admission"),
        sequence_support_path=sequence_support_path,
        policy_return_checkpoint_path=args.policy_return_checkpoint,
        policy_return_admission_path=args.policy_return_admission,
        device=device,
        lbfgsb_maxiter=1,
        optimizer_deadline_seconds=30.0,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        first_move_maxiter=1,
        first_move_deadline_seconds=1.0,
    )
    if lineage.get("portfolio_mode") is not True or lineage.get("online_lbfgsb_used") is not False:
        raise RuntimeError("current Development runtime unexpectedly resolved to a legacy optimizer path")
    if lineage.get("legacy_v12_admission_required_online") is not False:
        raise RuntimeError("Practical runtime must not depend on legacy V12 online admission")

    runtime_inp = _controls_disabled_runtime(
        source_inp=Path(args.inp).resolve(),
        cache_dir=Path(args.out_dir).resolve() / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=args.out_dir,
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
        raise RuntimeError("Practical runtime failed target write/readback audit")
    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy": "proposed_practical_h10_policy_return",
            "direct_tfv_development_runtime_contract": DIRECT_TFV_POLICY_RETURN_RUNTIME_CONTRACT,
            "direct_tfv_step3_contract": lineage["policy_return_step3_contract"],
            "development_only": True,
            "tfv_primary": True,
            "pfv_role": "secondary_authoritative_no_control_noninferiority_safety",
            "pfv_default_safety_contract": "PFV_PROPOSED_LE_100_M3_PLUS_1P05_TIMES_PFV_NO_CONTROL",
            "global_peak_role": "report_only",
            "future_realized_rainfall_used_as_model_input": False,
            "online_swmm_candidate_search": False,
            "online_lbfgsb_used": False,
            "policy_return_action_encoding": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            "policy_return_checkpoint_sha256": _sha(args.policy_return_checkpoint),
            "policy_return_admission_sha256": _sha(args.policy_return_admission),
            "policy_return_parent_continuation_sha256": admission["continuation_policy_sha256"],
            "base_step2_sha256": _sha(step2_path),
            "sequence_support_sha256": _sha(sequence_support_path),
            "asset_manifest_path": str(Path(args.asset_manifest).resolve()),
            "asset_manifest_sha256": _sha(args.asset_manifest),
            "legacy_v12_admission_required_online": False,
            "generic_d3_floor_controls_execution": False,
            "all_109_facilities_screened_each_decision": True,
            "target_latch_semantics": "LAST_COMMANDED_TARGET_PERSISTS_UNTIL_EXPLICITLY_CHANGED",
            "runtime_telemetry_graph_release": True,
            "target_write_readback_audit": write_audit,
            "project7_runtime_contract": project_contract,
            "prepared_event_clock": clock,
            "policy_return_training_validation_metrics": return_checkpoint.get("validation_metrics", {}),
            "runtime_factory_lineage": lineage,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "strategy": metadata["strategy"],
                "step3_contract": metadata["direct_tfv_step3_contract"],
                "metadata_path": result.metadata_path,
                "decision_path": result.decision_path,
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "online_lbfgsb_used": False,
                "target_write_readback_passed": True,
                "flow_routing_error_pct": result.flow_routing_error_pct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
