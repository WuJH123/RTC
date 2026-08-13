from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import baseline_sensor_nodes, canonical_baseline_id, fixed_baseline_controller
from .closed_loop import run_authoritative_closed_loop
from .production_cli import _controls_disabled_runtime, run_policy_main as legacy_run_policy_main
from .production_v120_router import is_v120_bundle
from .runtime_controller_guard import ContinuityGuardController


def run_policy_main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--step2")
    parser.add_argument("--runtime-inp-cache-dir")
    known, _ = parser.parse_known_args()
    strategy = canonical_baseline_id(known.strategy)
    if strategy == "proposed" and is_v120_bundle(known.step2):
        from .production_v120_bound import run_policy_v120_bound_main
        run_policy_v120_bound_main()
        return
    if strategy not in {"auto_rbc", "efd"}:
        legacy_run_policy_main()
        return

    cfg = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("controller config must be a JSON object")
    model_step_seconds = int(cfg["model_step_seconds"])
    control_update_seconds = int(cfg["control_update_seconds"])
    record_stride_seconds = int(cfg.get("record_stride_seconds", model_step_seconds))
    control_start_minutes = int(cfg.get("control_start_minutes", 0))
    if model_step_seconds <= 0 or control_update_seconds % model_step_seconds:
        raise ValueError("control_update_seconds must be a positive multiple of model_step_seconds")
    if record_stride_seconds != model_step_seconds:
        raise ValueError("production record stride must equal model step")

    controller_cfg = cfg.get("controller", {})
    if not isinstance(controller_cfg, dict):
        controller_cfg = {}
    raw_delta = controller_cfg.get("max_setting_delta_per_update")
    if raw_delta is None:
        raise ValueError("Formal rule baselines require max_setting_delta_per_update")
    max_delta = float(raw_delta)

    source_inp = Path(known.inp)
    cache_dir = Path(known.runtime_inp_cache_dir) if known.runtime_inp_cache_dir else Path(known.out_dir) / "_runtime_inp"
    runtime_inp = _controls_disabled_runtime(
        source_inp=source_inp,
        cache_dir=cache_dir,
        swmm_threads=int(cfg.get("swmm_threads", 1)),
    )
    sensors = baseline_sensor_nodes(strategy, source_inp)
    raw_controller = fixed_baseline_controller(strategy, inp_path=source_inp, max_delta_per_update=max_delta)
    controller = ContinuityGuardController(raw_controller, max_delta_per_update=max_delta, allow_projection=True)
    result = run_authoritative_closed_loop(
        inp_path=runtime_inp,
        output_dir=known.out_dir,
        run_id=known.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=control_start_minutes,
        control_update_seconds=control_update_seconds,
        observation_update_seconds=model_step_seconds,
        record_stride_seconds=record_stride_seconds,
        exact_global_peak=bool(cfg.get("exact_global_peak", False)),
    )
    print(json.dumps({
        "strategy": strategy,
        "source_inp": str(source_inp.resolve()),
        "runtime_inp": str(runtime_inp.resolve()),
        "native_controls_enabled": False,
        "rule_sensor_nodes": list(sensors),
        "metadata_path": result.metadata_path,
        "node_statistics_path": result.node_statistics_path,
        "decisions": result.decisions,
        "global_peak_flood_rate_m3s": result.global_peak_flood_rate_m3s,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }, indent=2))
