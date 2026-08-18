"""Design fresh Development calibration queries for target-latched Direct-TFV first moves.

The script does not run SWMM. For every selected fresh D3-HOLD rainfall group it obtains the frozen
V6/V7 q95-supported H120 optimizer query, refines only the target written at the next 10-minute
control instant by shrink-only L-BFGS-B, and emits exactly one HOLD reference plus one candidate whose
new target remains latched through H360 if no later command is issued. Deterministic modulo sharding
allows up to four independent GPU processes to design disjoint rainfall groups.
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
from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS, refine_supported_first_move
from rtc.direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS,
    DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
)
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d3_design_v60 import D3_V60_HOLD_ROLE
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v7 import DirectTFVRecedingMPCV7


FIRST_MOVE_CANDIDATE_ROLE = "D3_V9_REFINED_FIRST_MOVE_CALIBRATION_CANDIDATE"
CURRENT_FIRST_MOVE_PANEL_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_FIRST_MOVE_PANEL_DESIGN_V1"


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


def _sequence(ids: tuple[str, ...], tensor: torch.Tensor) -> list[dict[str, float]]:
    blocks = tensor.reshape(36, 2, 109).mean(dim=1).detach().cpu().numpy().astype(np.float64)
    return [{aid: float(row[index]) for index, aid in enumerate(ids)} for row in blocks]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy-admission", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--fresh-cache-manifest", required=True)
    p.add_argument("--fresh-causal-store", required=True)
    p.add_argument("--fresh-causal-state-store", required=True)
    p.add_argument("--template-d3-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--first-move-maxiter", type=int, default=12)
    p.add_argument("--first-move-deadline-seconds", type=float, default=30.0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()

    if args.shard_count <= 0 or args.shard_count > 4:
        raise ValueError("first-move panel shard-count must lie in [1,4]")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("first-move panel shard-index must lie in [0, shard-count)")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(args.checkpoint, graph=graph, device=device)
    policy = json.loads(Path(args.policy_admission).read_text(encoding="utf-8"))
    if str(policy.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("first-move panel requires accepted V2 policy admission lineage")
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

    all_names = sorted(fresh.targeted_d3_names())
    all_rainfall_groups = {str(fresh.entry(name).rainfall_group) for name in all_names}
    if len(all_rainfall_groups) < DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS:
        raise ValueError(
            "first-move panel needs at least "
            f"{DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS} fresh rainfall groups; got {len(all_rainfall_groups)}"
        )
    if len(all_names) != len(all_rainfall_groups):
        raise ValueError("first-move panel requires exactly one D3 checkpoint per fresh rainfall group")
    names = [
        name for position, name in enumerate(all_names)
        if position % int(args.shard_count) == int(args.shard_index)
    ]
    if not names:
        raise ValueError("first-move panel shard contains no rainfall groups")

    mpc = DirectTFVRecedingMPCV7(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        policy_admission_calibration=policy,
        sequence_support=sequence_support,
        design=DirectTFVMPCDesignV4(active_support_quantile="q95"),
    )
    actuator_ids = tuple(str(value) for value in graph.actuator_ids)
    output_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    seen: set[str] = set()

    for name in names:
        entry = fresh.entry(name)
        group = str(entry.rainfall_group)
        if group in seen:
            raise ValueError(f"duplicate rainfall group in first-move panel shard: {group}")
        seen.add(group)
        hold_rows = template[
            (template["checkpoint_id"].astype(str) == str(entry.checkpoint_id))
            & (template["data_role"].astype(str).str.lower() == D3_V60_HOLD_ROLE.lower())
        ]
        if len(hold_rows) != 1:
            raise ValueError(f"{name}: template manifest does not contain exactly one HOLD row")
        hold_row = hold_rows.iloc[0].to_dict()
        if str(hold_row.get("rainfall_group", "")) != group or str(hold_row.get("event_id", "")) != str(entry.event_id):
            raise ValueError(f"{name}: template/fresh identity mismatch")

        batch = online.batch(name, normalization, device)
        active_target = batch.reference_settings[0, 0]
        state_raw, rain_raw, flow_raw = _physical_inputs(batch, normalization)
        upstream = mpc.optimize(
            current_state=state_raw,
            rainfall=rain_raw,
            previous_actuator_flow=flow_raw,
            current_settings=active_target,
            active_target=active_target,
        )
        base_candidate = upstream.optimized_candidate_settings
        if base_candidate is None:
            raise RuntimeError(f"{name}: V7 did not preserve the raw q95 optimizer candidate")
        refined = refine_supported_first_move(
            mpc=mpc,
            base_candidate=base_candidate,
            current_state=state_raw,
            rainfall=rain_raw,
            previous_actuator_flow=flow_raw,
            active_target=active_target,
            maxiter=int(args.first_move_maxiter),
            deadline_seconds=float(args.first_move_deadline_seconds),
        )
        candidate_sequence = _sequence(actuator_ids, refined.sequence)
        candidate_sha = canonical_sequence_sha(candidate_sequence)

        hold_output = dict(hold_row)
        hold_output.update(
            {
                "first_move_panel_contract": DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
                "first_move_query_step3_contract": DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
                "first_move_role": "LATCH_PREVIOUS_TARGET_REFERENCE",
                "first_move_semantics": "LATCH_PREVIOUS_TARGET_H360",
            }
        )
        output_rows.append(hold_output)

        candidate_output = dict(hold_row)
        candidate_output.update(
            {
                "data_role": FIRST_MOVE_CANDIDATE_ROLE,
                "sequence_index": 1,
                "candidate_family": "v9_refined_target_latch_first_move",
                "v60_coefficients_json": json.dumps([]),
                "settings_sequence_json": json.dumps(candidate_sequence, sort_keys=True),
                "sequence_sha256": candidate_sha,
                "active_control_groups": -1,
                "active_actuators": int(refined.changed_facility_count),
                "sequence_rate_feasible": True,
                "first_move_panel_contract": DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
                "first_move_query_step3_contract": DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
                "first_move_role": "REFINED_TARGET_LATCH_QUERY",
                "first_move_semantics": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
                "predicted_refined_delta_tfv_m3": float(refined.predicted_delta_tfv_m3),
                "base_prefix_predicted_delta_tfv_m3": float(refined.base_prefix_predicted_delta_tfv_m3),
                "refinement_gain_m3": float(refined.gain_vs_base_prefix_m3),
                "first_move_changed_facility_count": int(refined.changed_facility_count),
                "full_plan_predicted_delta_tfv_m3": float(upstream.raw_optimized_predicted_delta_tfv_m3),
            }
        )
        output_rows.append(candidate_output)
        summaries.append(
            {
                "group": name,
                "rainfall_group": group,
                "event_id": str(entry.event_id),
                "checkpoint_id": str(entry.checkpoint_id),
                "sequence_sha256": candidate_sha,
                "predicted_refined_delta_tfv_m3": float(refined.predicted_delta_tfv_m3),
                "base_prefix_predicted_delta_tfv_m3": float(refined.base_prefix_predicted_delta_tfv_m3),
                "refinement_gain_m3": float(refined.gain_vs_base_prefix_m3),
                "first_move_changed_facility_count": int(refined.changed_facility_count),
            }
        )

    frame = pd.DataFrame.from_records(output_rows)
    if len(frame) != 2 * len(seen):
        raise RuntimeError("first-move panel shard must contain exactly one reference and one candidate per rainfall group")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    payload = {
        "contract": CURRENT_FIRST_MOVE_PANEL_RUN_CONTRACT,
        "development_only": True,
        "first_move_panel_contract": DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
        "first_move_query_step3_contract": DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
        "execution_estimand": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
        "reference_semantics": "LATCH_PREVIOUS_TARGET_H360",
        "active_support_quantile": "q95",
        "global_rainfall_group_count": len(all_rainfall_groups),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "rainfall_group_count": len(seen),
        "rainfall_groups": sorted(seen),
        "rows": len(frame),
        "hold_rows": len(seen),
        "candidate_rows": len(seen),
        "records": summaries,
        "lineage": {
            "step2_checkpoint_sha256": _sha(args.checkpoint),
            "policy_admission_sha256": _sha(args.policy_admission),
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
