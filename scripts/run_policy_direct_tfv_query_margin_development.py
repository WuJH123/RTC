"""Run the query-conditioned three-family Project7 controller in authoritative Development SWMM.

This is the first SWMM stage permitted after the fresh-validation and matched-calibration gates pass.
It preserves frozen Step1/base Step2, 82/109 control geometry, q95 support and exact H10 receding-policy
return lineage. Proposed metadata is compatible with the existing TFV/Global-Peak/PFV comparison
scripts; Policy Lock remains closed.
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
)
from rtc.direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
)
from rtc.direct_tfv_policy_return_query_margin import (
    DIRECT_TFV_QUERY_MARGIN_CONTRACT,
    load_query_margin_checkpoint,
)
from rtc.direct_tfv_policy_return_query_margin_runtime_factory import (
    build_frozen_query_margin_controller,
)
from rtc.event_clock import inspect_prepared_event_clock
from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.practical_rtc_assets import (
    load_practical_rtc_asset_manifest,
    practical_asset_path,
)
from rtc.production_cli import _controls_disabled_runtime, _load_graph
from rtc.project7_contract import (
    EFFECTIVE_WARMUP_MINUTES,
    validate_project7_runtime_config,
)
from rtc.step1_runtime_v127 import load_frozen_step1_v127
from rtc.step2_state_store_v127 import (
    semantic_model_state_dict_sha256,
    semantic_sensor_layout_sha256,
)


QUERY_MARGIN_DEVELOPMENT_RUNTIME_CONTRACT = (
    "PROJECT7_QUERY_CONDITIONED_POLICY_RETURN_AUTHORITATIVE_DEVELOPMENT_RTC_V1"
)
CURRENT_THREE_FAMILY_PORTFOLIO = (
    "STEP2_H10_PROBE_SCALE_0.50",
    "STEP2_H10_PROBE_SCALE_1.00",
    "TYPE_AWARE_HYDRAULIC_PRESSURE",
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_step1_lineage(
    *,
    step1_path: str,
    step2_path: str,
    sensors_path: str,
    graph_path: str,
    device: torch.device,
) -> None:
    graph = _load_graph(graph_path)
    step1 = load_frozen_step1_v127(step1_path, device)
    from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
    from rtc.production_cli import _load_lines

    _, _, checkpoint = load_direct_tfv_runtime_checkpoint(
        step2_path,
        graph=graph,
        device=device,
    )
    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("base Step2 lacks Step1/sensor lineage")
    expected = {
        "step1_model_semantic_sha256": semantic_model_state_dict_sha256(step1),
        "sensor_layout_semantic_sha256": semantic_sensor_layout_sha256(
            _load_lines(sensors_path)
        ),
    }
    for key, actual in expected.items():
        if str(lineage.get(key, "")).lower() != actual.lower():
            raise ValueError(
                f"runtime Step1/sensor semantics differ from Step2 training: {key}"
            )


def _require_lineage(lineage: dict, mask) -> None:
    required_true = (
        "portfolio_mode",
        "fresh_validation_verified",
        "candidate_selection_uses_relative_rank_only",
        "hold_decision_uses_query_best_margin_only",
    )
    for key in required_true:
        if lineage.get(key) is not True:
            raise RuntimeError(f"query-margin runtime lost {key}")
    if lineage.get("conformal_uncertainty_reranks_candidates") is not False:
        raise RuntimeError("conformal uncertainty must not rerank current candidates")
    if (
        lineage.get("online_lbfgsb_used") is not False
        or lineage.get("projected_gradient_h10_enabled") is not False
    ):
        raise RuntimeError("query-margin runtime enabled a forbidden online optimizer")
    if (
        tuple(str(x) for x in lineage.get("candidate_portfolio_families", ()))
        != CURRENT_THREE_FAMILY_PORTFOLIO
    ):
        raise RuntimeError("query-margin runtime has unexpected candidate families")
    if int(lineage.get("supervisory_control_dimension", -1)) != 82 or int(mask.sum()) != 82:
        raise RuntimeError("query-margin runtime lost the 82-control mask")
    if int(lineage.get("model_action_channel_count", -1)) != 109:
        raise RuntimeError("query-margin runtime lost the 109-channel representation")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-manifest", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--policy-return-checkpoint", required=True)
    p.add_argument("--policy-return-admission", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--decision-runtime-budget-seconds",
        type=float,
        default=180.0,
    )
    p.add_argument("--probe-chunk-size", type=int, default=24)
    p.add_argument("--projected-gradient-steps", type=int, default=6)
    p.add_argument("--projected-gradient-step-fraction", type=float, default=0.25)
    args = p.parse_args()

    if not 0.0 < float(args.decision_runtime_budget_seconds) < 600.0:
        raise ValueError("controller runtime budget must fit inside a 600-s update")

    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")

    graph_path = practical_asset_path(assets, "graph")
    sensors_path = practical_asset_path(assets, "sensors")
    config_path = practical_asset_path(assets, "config")
    step1_path = practical_asset_path(assets, "step1")
    step2_path = practical_asset_path(assets, "step2")
    supervisory_control_path = practical_asset_path(assets, "supervisory_control")
    sequence_support_path = practical_asset_path(assets, "sequence_support")
    source_inp_path = Path(args.inp).resolve()
    controller_config_path = Path(config_path).resolve()

    _require_step1_lineage(
        step1_path=step1_path,
        step2_path=step2_path,
        sensors_path=sensors_path,
        graph_path=graph_path,
        device=device,
    )

    graph = _load_graph(graph_path)
    control, mask = load_native_supervisory_control(
        supervisory_control_path,
        actuator_ids=graph.actuator_ids,
    )
    _, _, _, checkpoint = load_query_margin_checkpoint(
        args.policy_return_checkpoint,
        graph=graph,
        base_step2_path=step2_path,
        device=device,
    )
    if checkpoint.get("fresh_validation_verified") is not True:
        raise ValueError("critic did not pass fresh validation")
    if (
        str(checkpoint.get("candidate_portfolio_contract", ""))
        != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
    ):
        raise ValueError("critic uses another candidate portfolio")
    if (
        str(checkpoint.get("action_encoding_contract", ""))
        != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
    ):
        raise ValueError("critic uses another H10 action encoding")
    if (
        str(checkpoint.get("supervisory_mask_sha256", "")).lower()
        != str(control["supervisory_mask_sha256"]).lower()
    ):
        raise ValueError("critic uses another supervisory mask")

    admission = json.loads(
        Path(args.policy_return_admission).read_text(encoding="utf-8")
    )
    if (
        str(admission.get("contract", ""))
        != DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT
    ):
        raise ValueError("wrong admission contract")
    if (
        str(admission.get("candidate_portfolio_contract", ""))
        != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
    ):
        raise ValueError("wrong admission portfolio")
    if (
        str(admission.get("supervisory_mask_sha256", "")).lower()
        != str(control["supervisory_mask_sha256"]).lower()
    ):
        raise ValueError("admission uses another mask")
    if (
        str(admission.get("policy_return_checkpoint_sha256", "")).lower()
        != _sha(args.policy_return_checkpoint)
    ):
        raise ValueError("admission/checkpoint mismatch")

    cfg = json.loads(controller_config_path.read_text(encoding="utf-8"))
    project_contract = validate_project7_runtime_config(cfg)
    clock = inspect_prepared_event_clock(source_inp_path)
    if (
        abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES)
        > 1e-6
    ):
        raise ValueError("event violates common warm-up clock")

    controller, _, sensors, lineage = build_frozen_query_margin_controller(
        graph_path=graph_path,
        sensors_path=sensors_path,
        config_path=config_path,
        step1_path=step1_path,
        step2_path=step2_path,
        supervisory_control_path=supervisory_control_path,
        sequence_support_path=sequence_support_path,
        policy_return_checkpoint_path=args.policy_return_checkpoint,
        policy_return_admission_path=args.policy_return_admission,
        device=device,
        decision_runtime_budget_seconds=float(args.decision_runtime_budget_seconds),
        proposal_probe_chunk_size=int(args.probe_chunk_size),
        projected_gradient_steps=int(args.projected_gradient_steps),
        projected_gradient_step_fraction=float(args.projected_gradient_step_fraction),
    )
    _require_lineage(lineage, mask)

    runtime_inp = _controls_disabled_runtime(
        source_inp=source_inp_path,
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
        raise RuntimeError("query-margin runtime failed target write/readback audit")

    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy": "proposed_practical_h10_query_conditioned_policy_return",
            "direct_tfv_development_runtime_contract": QUERY_MARGIN_DEVELOPMENT_RUNTIME_CONTRACT,
            "direct_tfv_step3_contract": lineage["policy_return_step3_contract"],
            "development_only": True,
            "tfv_primary": True,
            "pfv_role": "secondary_authoritative_no_control_noninferiority_safety",
            "pfv_default_safety_contract": "PFV_PROPOSED_LE_100_M3_PLUS_1P05_TIMES_PFV_NO_CONTROL",
            "global_peak_role": "report_only",
            "future_realized_rainfall_used_as_model_input": False,
            "online_swmm_candidate_search": False,
            "online_lbfgsb_used": False,
            "supervisory_control_dimension": 82,
            "model_action_channel_count": 109,
            "passive_setting_channel_count": 27,
            "supervisory_mask_sha256": control["supervisory_mask_sha256"],
            "projected_gradient_h10_enabled": False,
            "candidate_portfolio_family_count_max": 3,
            "candidate_portfolio_families": list(CURRENT_THREE_FAMILY_PORTFOLIO),
            "query_margin_contract": DIRECT_TFV_QUERY_MARGIN_CONTRACT,
            "candidate_selection_uses_relative_rank_only": True,
            "hold_decision_uses_query_best_margin_only": True,
            "conformal_uncertainty_reranks_candidates": False,
            "fresh_validation_verified": True,
            "policy_return_action_encoding": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            "policy_return_checkpoint_sha256": _sha(args.policy_return_checkpoint),
            "policy_return_admission_sha256": _sha(args.policy_return_admission),
            "policy_return_parent_continuation_sha256": admission[
                "continuation_policy_sha256"
            ],
            "base_step2_sha256": _sha(step2_path),
            "sequence_support_sha256": _sha(sequence_support_path),
            "asset_manifest_path": str(Path(args.asset_manifest).resolve()),
            "asset_manifest_sha256": _sha(args.asset_manifest),
            "source_inp_path": str(source_inp_path),
            "source_inp_sha256": _sha(source_inp_path),
            "controller_config_path": str(controller_config_path),
            "controller_config_sha256": _sha(controller_config_path),
            "runtime_inp_path": str(Path(runtime_inp).resolve()),
            "runtime_inp_sha256": _sha(runtime_inp),
            "legacy_v12_admission_required_online": False,
            "generic_d3_floor_controls_execution": False,
            "step1_retrained_for_control_mask": False,
            "base_step2_retrained_for_control_mask": False,
            "target_latch_semantics": "LAST_COMMANDED_TARGET_PERSISTS_UNTIL_EXPLICITLY_CHANGED",
            "target_write_readback_audit": write_audit,
            "project7_runtime_contract": project_contract,
            "prepared_event_clock": clock,
            "policy_return_fresh_validation_metrics": checkpoint.get(
                "fresh_validation_metrics", {}
            ),
            "runtime_factory_lineage": lineage,
            "ready_for_policy_lock": False,
        }
    )
    for key in (
        "source_inp_sha256",
        "controller_config_sha256",
        "swmm_engine_version",
        "prepared_event_clock",
    ):
        if metadata.get(key) in (None, ""):
            raise RuntimeError(f"Proposed metadata lacks comparison lineage {key}")

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "strategy": metadata["strategy"],
                "step3_contract": metadata["direct_tfv_step3_contract"],
                "metadata_path": result.metadata_path,
                "decision_path": result.decision_path,
                "node_statistics_path": result.node_statistics_path,
                "decisions": result.decisions,
                "supervisory_control_dimension": 82,
                "model_action_channel_count": 109,
                "query_margin_contract": DIRECT_TFV_QUERY_MARGIN_CONTRACT,
                "fresh_validation_verified": True,
                "target_write_readback_passed": True,
                "flow_routing_error_pct": result.flow_routing_error_pct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
