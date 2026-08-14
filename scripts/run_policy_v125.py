"""Run the V125 10-minute rolling sparse-state anchor/learned-override RTC policy.

V125 is finite-only.  It reuses the validated V123 causal Value/runtime assets but changes
the decision semantics: Sparse-RBC is the default action and a learned candidate can
replace it only if candidate-vs-anchor TFV improvement clears a separately calibrated
D4-FIT one-sided error budget.  PFV remains a one-sided soft penalty after that gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rtc.closed_loop import run_authoritative_closed_loop
from rtc.controller_v125 import V125TorchMPCController
from rtc.forecast import PersistenceDecayForecast
from rtc.production_cli import (
    _controller_config,
    _controls_disabled_runtime,
    _load_graph,
    _load_lines,
    _load_step1,
)
from rtc.runtime_controller_guard import ContinuityGuardController
from rtc.step2_policy_v123 import FirstMoveTFVPFVPolicyV123
from rtc.step2_policy_v125 import (
    AnchorOverridePolicyV125,
    V125_OVERRIDE_CALIBRATION_CONTRACT,
    V125_POLICY_CONTRACT,
)

# run_policy_v123.py is deliberately retained as the frozen asset loader so V125 does not
# create another subtly different model/normalisation lineage.
from run_policy_v123 import _load_policy


def _load_override_margin(path: str | Path) -> tuple[float, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = payload.get("calibration", {})
    if calibration.get("contract") != V125_OVERRIDE_CALIBRATION_CONTRACT:
        raise ValueError("V125 runtime requires the anchor-relative D4 calibration contract")
    margin = float(calibration.get("anchor_override_margin_m3", float("nan")))
    if not (margin >= 0.0 and margin < float("inf")):
        raise ValueError("V125 anchor override margin is invalid")
    boundary = payload.get("boundary", {})
    if boundary and not bool(boundary.get("calibration_uses_fit_only", False)):
        raise ValueError("V125 runtime rejects calibration not restricted to D4 fit")
    if boundary and bool(boundary.get("audit_used_for_calibration", True)):
        raise ValueError("V125 runtime rejects D4 audit leakage into calibration")
    return margin, payload


def _v125_policy(base: FirstMoveTFVPFVPolicyV123, *, margin_m3: float) -> AnchorOverridePolicyV125:
    common = dict(
        model=base.model,
        basis=base.basis,
        prepared=base.prepared,
        normalization=base.normalization,
        objective=base.objective,
        false_benefit_margin_m3=base.false_benefit_margin_m3,
    )
    anchor = FirstMoveTFVPFVPolicyV123(
        **common,
        graph=base.graph,
        use_sparse_rbc_anchor=True,
        knowledge_anchor_fallback=True,
        policy_mode="anchor_only",
    )
    learned = FirstMoveTFVPFVPolicyV123(
        **common,
        graph=None,
        use_sparse_rbc_anchor=False,
        knowledge_anchor_fallback=False,
        policy_mode="learned_only",
    )
    return AnchorOverridePolicyV125(
        anchor_policy=anchor,
        learned_policy=learned,
        anchor_override_margin_m3=float(margin_m3),
        require_objective_improvement=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Project7 V125 finite anchor-default rolling RTC")
    p.add_argument("--inp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--tfv-checkpoint", required=True)
    p.add_argument("--pfv-checkpoint", required=True)
    p.add_argument("--tfv-report", required=True)
    p.add_argument("--pfv-report", required=True)
    p.add_argument("--objective-report", required=True)
    p.add_argument("--calibration-report", required=True, help="V123 passive/PFV calibration")
    p.add_argument("--anchor-override-calibration", required=True, help="V125 D4-FIT calibration")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("v120_contract") != "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1":
        raise ValueError("V125 preserves the frozen V120 timing/config contract")
    forecast_cfg = cfg.get("forecast", {})
    if tuple(float(x) for x in forecast_cfg.get("scenario_multipliers", [])) != (1.0,):
        raise ValueError("V125 requires the frozen single central rainfall forecast")
    if abs(float(forecast_cfg.get("decay_per_step", -1.0)) - 0.92) > 1e-12:
        raise ValueError("V125 rainfall decay differs from frozen causal contract")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    sensors = _load_lines(args.sensors)
    step1 = _load_step1(args.step1, device)
    base, lineage = _load_policy(
        graph=graph,
        cache_manifest=args.cache_manifest,
        causal_store_path=args.causal_store,
        tfv_checkpoint=args.tfv_checkpoint,
        pfv_checkpoint=args.pfv_checkpoint,
        objective_report=args.objective_report,
        calibration_report=args.calibration_report,
        tfv_report=args.tfv_report,
        pfv_report=args.pfv_report,
        policy_mode="hybrid",
        device=device,
    )
    override_margin, override_calibration = _load_override_margin(args.anchor_override_calibration)
    policy = _v125_policy(base, margin_m3=override_margin)

    controller_cfg = cfg["controller"]
    controller = V125TorchMPCController(
        step1=step1,
        mpc=policy,
        graph=graph,
        sensor_nodes=sensors,
        forecast=PersistenceDecayForecast(
            decay_per_step=0.92,
            scenario_multipliers=(1.0,),
            history_steps_for_level=1,
        ),
        config=_controller_config(controller_cfg, control_block_steps=2),
        device=device,
    )
    controller = ContinuityGuardController(controller, max_delta_per_update=0.5, allow_projection=False)
    runtime_inp = _controls_disabled_runtime(
        source_inp=Path(args.inp),
        cache_dir=Path(args.out_dir) / "_runtime_inp",
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=args.out_dir,
        run_id=args.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=60,
        control_update_seconds=600,
        observation_update_seconds=300,
        record_stride_seconds=300,
        exact_global_peak=bool(cfg.get("exact_global_peak", False)),
    )
    metadata_path = Path(result.metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "strategy": "proposed_v125_anchor_override",
        "v125_policy_contract": V125_POLICY_CONTRACT,
        "control_update_seconds": 600,
        "all_writable_actuators_commanded_each_decision": True,
        "tfv_primary": True,
        "pfv_one_sided_soft_protection": True,
        "pfv_can_buy_worse_anchor_relative_tfv": False,
        "anchor_default": True,
        "learned_override_requires_anchor_relative_error_budget": True,
        "continuous_gradient_search": False,
        "future_realized_rainfall_used_as_model_input": False,
        "v123_asset_lineage": lineage,
        "v125_anchor_override_calibration": override_calibration,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "strategy": "proposed_v125_anchor_override",
        "policy_contract": V125_POLICY_CONTRACT,
        "anchor_override_margin_m3": override_margin,
        "metadata_path": result.metadata_path,
        "decision_path": result.decision_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
