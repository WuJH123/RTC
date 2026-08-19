"""Design a small authoritative calibration panel from the current V6 raw optimizer queries.

The script does not run SWMM.  For each fresh admission rainfall group it evaluates the frozen V6
optimizer at the same D3-HOLD checkpoint and writes exactly one support-contracted raw optimizer
candidate plus its HOLD reference in the existing D3 sequence-manifest format.  The resulting
20-row panel (10 HOLD + 10 candidate in the current study) can be executed by the existing D3 SWMM
pipeline and compiled independently from Step2 training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.data_design import canonical_sequence_sha
from rtc.direct_tfv_admission import DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
from rtc.direct_tfv_policy_admission import (
    DIRECT_TFV_POLICY_PANEL_CONTRACT,
    DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
)
from rtc.direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d3_design_v60 import D3_V60_HOLD_ROLE
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v6 import DirectTFVRecedingMPCV6


POLICY_CANDIDATE_ROLE = "D3_V6_POLICY_CALIBRATION_CANDIDATE"
CURRENT_POLICY_PANEL_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_V6_POLICY_PANEL_DESIGN_V1"


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


def _sequence(ids: tuple[str, ...], values: np.ndarray) -> list[dict[str, float]]:
    return [
        {aid: float(step[index]) for index, aid in enumerate(ids)}
        for step in values
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-admission", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--fresh-cache-manifest", required=True)
    p.add_argument("--fresh-causal-store", required=True)
    p.add_argument("--fresh-causal-state-store", required=True)
    p.add_argument("--template-d3-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--active-support-quantile", choices=("q90", "q95", "q99"), default="q95")
    args = p.parse_args()

    if str(args.active_support_quantile) != "q95":
        raise ValueError("current policy calibration panel is preregistered on canonical q95 only")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(args.checkpoint, graph=graph, device=device)
    base_admission = json.loads(Path(args.base_admission).read_text(encoding="utf-8"))
    if str(base_admission.get("contract", "")) != DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT:
        raise ValueError("policy panel requires the accepted V1 fresh-D3 admission artifact")
    partition = base_admission.get("partition")
    if not isinstance(partition, dict) or partition.get("ready_for_admission_calibration") is not True:
        raise ValueError("base admission partition is not ready")
    expected_groups = {str(x) for x in partition.get("fresh_calibration_rainfall_groups", ())}
    if len(expected_groups) < 9:
        raise ValueError("policy panel requires at least nine fresh calibration rainfall groups")

    sequence_support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        sequence_support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=_sha(args.checkpoint),
    )
    fresh = V60TrainCache(args.fresh_cache_manifest)
    rain = load_causal_forecast_store_v123(args.fresh_causal_store)
    state = load_causal_state_store_v127(args.fresh_causal_state_store)
    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(fresh, rain), state)
    template = pd.read_csv(args.template_d3_manifest)
    if template.empty or "checkpoint_id" not in template or "data_role" not in template:
        raise ValueError("template D3 manifest is empty or incomplete")

    controller = DirectTFVRecedingMPCV6(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        admission_calibration=base_admission,
        sequence_support=sequence_support,
        design=DirectTFVMPCDesignV4(active_support_quantile="q95"),
    )
    actuator_ids = tuple(str(x) for x in graph.actuator_ids)
    output_rows: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    seen_rainfall: set[str] = set()

    for name in sorted(fresh.targeted_d3_names()):
        entry = fresh.entry(name)
        rainfall_group = str(entry.rainfall_group)
        if rainfall_group not in expected_groups:
            raise ValueError(f"fresh cache contains non-calibration rainfall group: {rainfall_group}")
        if rainfall_group in seen_rainfall:
            raise ValueError(
                "policy panel requires exactly one D3 checkpoint per fresh rainfall group; "
                f"duplicate={rainfall_group}"
            )
        seen_rainfall.add(rainfall_group)
        hold_rows = template[
            (template["checkpoint_id"].astype(str) == str(entry.checkpoint_id))
            & (template["data_role"].astype(str).str.lower() == D3_V60_HOLD_ROLE.lower())
        ]
        if len(hold_rows) != 1:
            raise ValueError(f"{name}: template manifest does not contain exactly one HOLD row")
        hold_row = hold_rows.iloc[0].to_dict()
        if str(hold_row.get("rainfall_group", "")) != rainfall_group:
            raise ValueError(f"{name}: template rainfall identity mismatch")
        if str(hold_row.get("event_id", "")) != str(entry.event_id):
            raise ValueError(f"{name}: template event identity mismatch")

        batch = online.batch(name, normalization, device)
        active_target = batch.reference_settings[0, 0]
        state_raw, rain_raw, flow_raw = _physical_inputs(batch, normalization)
        result = controller.optimize(
            current_state=state_raw,
            rainfall=rain_raw,
            previous_actuator_flow=flow_raw,
            current_settings=active_target,
            active_target=active_target,
        )
        candidate = result.optimized_candidate_settings
        if candidate is None:
            raise RuntimeError(f"{name}: V6 did not preserve a raw optimized candidate")
        if tuple(candidate.shape) != (72, 109):
            raise RuntimeError(f"{name}: V6 candidate is not [H72,109]")
        block_values = (
            candidate.detach().cpu().numpy().astype(np.float64).reshape(36, 2, 109).mean(axis=1)
        )
        candidate_sequence = _sequence(actuator_ids, block_values)
        candidate_sha = canonical_sequence_sha(candidate_sequence)
        hold_sequence = json.loads(str(hold_row["settings_sequence_json"]))
        if len(hold_sequence) != 36:
            raise ValueError(f"{name}: HOLD template does not contain H360/36 control blocks")

        hold_output = dict(hold_row)
        hold_output.update(
            {
                "policy_panel_contract": DIRECT_TFV_POLICY_PANEL_CONTRACT,
                "policy_query_step3_contract": DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
                "policy_calibration_role": "HOLD_REFERENCE",
            }
        )
        output_rows.append(hold_output)

        candidate_output = dict(hold_row)
        candidate_output.update(
            {
                "data_role": POLICY_CANDIDATE_ROLE,
                "sequence_index": 1,
                "candidate_family": "v6_q95_raw_optimizer_query",
                "v60_coefficients_json": json.dumps([]),
                "settings_sequence_json": json.dumps(candidate_sequence, sort_keys=True),
                "sequence_sha256": candidate_sha,
                "active_control_groups": -1,
                "active_actuators": int(result.active_facility_count),
                "sequence_rate_feasible": True,
                "policy_panel_contract": DIRECT_TFV_POLICY_PANEL_CONTRACT,
                "policy_query_step3_contract": DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
                "policy_calibration_role": "CURRENT_RAW_OPTIMIZER_QUERY",
                "predicted_delta_tfv_m3": float(result.raw_optimized_predicted_delta_tfv_m3),
                "active_facility_count": int(result.active_facility_count),
                "first_move_changed_facility_count": int(result.first_move_changed_facility_count),
                "joint_sequence_support_quantile": str(result.joint_sequence_support_quantile),
                "joint_sequence_support_max_ratio": float(result.joint_sequence_support_max_ratio),
                "joint_sequence_support_binding": bool(result.joint_sequence_support_binding),
                "base_v1_admission_passed": bool(result.admission_passed),
                "base_v1_admission_margin_m3": float(result.admission_margin_m3),
            }
        )
        output_rows.append(candidate_output)
        summary_records.append(
            {
                "group": name,
                "rainfall_group": rainfall_group,
                "event_id": str(entry.event_id),
                "checkpoint_id": str(entry.checkpoint_id),
                "sequence_sha256": candidate_sha,
                "predicted_delta_tfv_m3": float(result.raw_optimized_predicted_delta_tfv_m3),
                "active_facility_count": int(result.active_facility_count),
                "support_max_ratio": float(result.joint_sequence_support_max_ratio),
                "support_binding": bool(result.joint_sequence_support_binding),
                "base_v1_admission_passed": bool(result.admission_passed),
            }
        )

    if seen_rainfall != expected_groups:
        raise ValueError(
            "policy panel does not cover exactly the fresh calibration rainfall groups; "
            f"missing={sorted(expected_groups - seen_rainfall)}"
        )
    frame = pd.DataFrame.from_records(output_rows)
    if len(frame) != 2 * len(expected_groups):
        raise RuntimeError("policy panel must contain exactly one HOLD and one optimizer query per rainfall group")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    payload = {
        "contract": CURRENT_POLICY_PANEL_RUN_CONTRACT,
        "development_only": True,
        "policy_panel_contract": DIRECT_TFV_POLICY_PANEL_CONTRACT,
        "policy_query_step3_contract": DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
        "active_support_quantile": "q95",
        "rainfall_group_count": len(expected_groups),
        "rows": len(frame),
        "hold_rows": len(expected_groups),
        "candidate_rows": len(expected_groups),
        "records": summary_records,
        "lineage": {
            "step2_checkpoint_sha256": _sha(args.checkpoint),
            "base_admission_sha256": _sha(args.base_admission),
            "sequence_support_sha256": _sha(args.sequence_support),
            "fresh_cache_sha256": _sha(args.fresh_cache_manifest),
            "fresh_causal_rainfall_sha256": _sha(args.fresh_causal_store),
            "fresh_causal_state_sha256": _sha(args.fresh_causal_state_store),
            "template_d3_manifest_sha256": _sha(args.template_d3_manifest),
        },
        "online_swmm_called": False,
    }
    summary_path = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
