"""Audit calibrated Direct-TFV Step3 V5 on D3 HOLD-reference Development states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_admission import DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v5 import DirectTFVRecedingMPCV5


CURRENT_STEP3_CALIBRATED_AUDIT_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_STEP3_CALIBRATED_AUDIT_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _physical_inputs(batch, normalization):
    sm = torch.as_tensor(normalization.state_mean, dtype=batch.initial_state.dtype, device=batch.initial_state.device)
    ss = torch.as_tensor(normalization.state_std, dtype=batch.initial_state.dtype, device=batch.initial_state.device)
    rm = torch.as_tensor(normalization.rainfall_mean, dtype=batch.rainfall.dtype, device=batch.rainfall.device)
    rs = torch.as_tensor(normalization.rainfall_std, dtype=batch.rainfall.dtype, device=batch.rainfall.device)
    fm = torch.as_tensor(normalization.flow_mean, dtype=batch.previous_actuator_flow.dtype, device=batch.previous_actuator_flow.device)
    fs = torch.as_tensor(normalization.flow_std, dtype=batch.previous_actuator_flow.dtype, device=batch.previous_actuator_flow.device)
    return batch.initial_state * ss + sm, batch.rainfall * rs + rm, batch.previous_actuator_flow * fs + fm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--admission-calibration", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--active-support-quantile", choices=("q90", "q95", "q99"), default="q95")
    p.add_argument("--max-groups", type=int, default=0)
    args = p.parse_args()

    started = time.perf_counter()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(args.checkpoint, graph=graph, device=device)
    admission = json.loads(Path(args.admission_calibration).read_text(encoding="utf-8"))
    if str(admission.get("contract")) != DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT:
        raise ValueError("wrong Direct-TFV admission calibration contract")
    if str(admission.get("lineage", {}).get("step2_checkpoint_sha256", "")).lower() != _sha(args.checkpoint).lower():
        raise ValueError("admission calibration/checkpoint lineage mismatch")

    base = V60TrainCache(args.cache_manifest)
    rain = load_causal_forecast_store_v123(args.causal_store)
    state = load_causal_state_store_v127(args.causal_state_store)
    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain), state)
    _, holdout = deterministic_rainfall_split_v60(
        base, names=sorted(base.names("D2") + base.targeted_d3_names()), holdout_fraction=0.20
    )
    audit_names = set(str(value) for value in admission.get("audit_names", ()))
    names = sorted(name for name in holdout if name.startswith("D3::") and name in audit_names)
    if int(args.max_groups) > 0:
        names = names[: int(args.max_groups)]
    if not names:
        raise ValueError("calibrated Step3 audit has no rainfall-disjoint D3 audit groups")

    design = DirectTFVMPCDesignV4(active_support_quantile=str(args.active_support_quantile))
    controller = DirectTFVRecedingMPCV5(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        admission_calibration=admission,
        design=design,
    )
    records = []
    action_count = hold_count = support_violations = engineering_violations = 0
    for name in names:
        batch = online.batch(name, normalization, device)
        reference = batch.reference_settings[0]
        active_target = reference[0]
        state_raw, rain_raw, flow_raw = _physical_inputs(batch, normalization)
        result = controller.optimize(
            current_state=state_raw,
            rainfall=rain_raw,
            previous_actuator_flow=flow_raw,
            current_settings=active_target,
            active_target=active_target,
        )
        action = result.selected_source == "DIRECT_TFV_RECEDING_LBFGSB"
        action_count += int(action)
        hold_count += int(not action)
        support_violations += int(float(result.maximum_support_ratio) > 1.0001)
        engineering_violations += int(result.first_move_changed_facility_count > result.active_facility_count)
        if action and not (result.admission_passed and float(result.admission_upper_bound_m3) < 0.0 and result.first_move_changed_facility_count > 0):
            engineering_violations += 1
        records.append(
            {
                "group": name,
                "selected_source": result.selected_source,
                "raw_optimized_predicted_delta_tfv_m3": float(result.raw_optimized_predicted_delta_tfv_m3),
                "admission_margin_m3": float(result.admission_margin_m3),
                "admission_upper_bound_m3": float(result.admission_upper_bound_m3),
                "admission_margin_kind": result.admission_margin_kind,
                "admission_passed": bool(result.admission_passed),
                "active_facility_count": int(result.active_facility_count),
                "first_move_changed_facility_count": int(result.first_move_changed_facility_count),
                "maximum_support_ratio": float(result.maximum_support_ratio),
                "solver_elapsed_seconds": float(result.elapsed_seconds),
            }
        )
    payload = {
        "contract": CURRENT_STEP3_CALIBRATED_AUDIT_CONTRACT,
        "development_only": True,
        "step3_contract": controller.policy_mode_contract,
        "groups": len(records),
        "action_count": int(action_count),
        "hold_count": int(hold_count),
        "support_violation_count": int(support_violations),
        "engineering_violation_count": int(engineering_violations),
        "admission_global_margin_m3": float(admission["global_margin_m3"]),
        "admission_dense_margin_m3": float(admission["dense_margin_m3"]),
        "active_support_quantile_effective": controller.active_support_quantile_effective(),
        "active_support_ceiling": controller.active_support_ceiling(),
        "records": records,
        "ready_for_authoritative_swmm_probe": bool(action_count > 0 and support_violations == 0 and engineering_violations == 0),
        "lineage": {
            "checkpoint_sha256": _sha(args.checkpoint),
            "admission_calibration_sha256": _sha(args.admission_calibration),
        },
        "wall_seconds": float(time.perf_counter() - started),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ready_for_authoritative_swmm_probe"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
