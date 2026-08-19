"""Design exact target-latched first-move calibration queries from candidate-free contexts.

No generic D3 candidate row is loaded or required. Each Development rainfall group contributes one
previous-target HOLD reference and one q95-supported refined/pruned first-move candidate. The script
does not run SWMM. Deterministic modulo sharding supports up to four GPU design processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from rtc.checkpoint_direct_tfv import (
    direct_tfv_first_move_behavioral_source_sha256,
    direct_tfv_first_move_source_sha256,
    load_direct_tfv_runtime_checkpoint,
)
from rtc.data_design import canonical_sequence_sha
from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS, refine_supported_first_move
from rtc.direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS,
    DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
)
from rtc.direct_tfv_first_move_context import load_first_move_context_store
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from rtc.production_cli import _load_graph
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v7 import DirectTFVRecedingMPCV7


HOLD_ROLE = "D3_V60_HOLD_REFERENCE"
FIRST_MOVE_CANDIDATE_ROLE = "D3_V9_REFINED_FIRST_MOVE_CALIBRATION_CANDIDATE"
CURRENT_FIRST_MOVE_PANEL_RUN_CONTRACT = (
    "PROJECT7_CURRENT_DIRECT_TFV_FIRST_MOVE_PANEL_DESIGN_V2_CANDIDATE_FREE_CONTEXT"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sequence(ids: tuple[str, ...], tensor: torch.Tensor) -> list[dict[str, float]]:
    blocks = tensor.reshape(36, 2, 109).mean(dim=1).detach().cpu().to(torch.float64).numpy()
    return [{aid: float(row[index]) for index, aid in enumerate(ids)} for row in blocks]


def _hold_sequence(ids: tuple[str, ...], active_target: torch.Tensor) -> list[dict[str, float]]:
    values = active_target.detach().cpu().to(torch.float64).tolist()
    row = {aid: float(values[index]) for index, aid in enumerate(ids)}
    return [dict(row) for _ in range(36)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy-admission", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--context-store", required=True)
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
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.checkpoint, graph=graph, device=device
    )
    policy = json.loads(Path(args.policy_admission).read_text(encoding="utf-8"))
    if str(policy.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("first-move panel requires accepted V2 policy admission lineage")
    support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support, actuator_ids=graph.actuator_ids, step2_checkpoint_sha256=_sha(args.checkpoint)
    )
    context = load_first_move_context_store(args.context_store)
    if context.graph_sha256.lower() != _sha(args.graph).lower():
        raise ValueError("first-move context graph differs from panel graph")
    all_groups = sorted(context.rainfall_groups)
    if len(all_groups) < DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS:
        raise ValueError(
            "first-move panel needs at least "
            f"{DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS} rainfall groups"
        )
    groups = [
        group for position, group in enumerate(all_groups)
        if position % int(args.shard_count) == int(args.shard_index)
    ]
    if not groups:
        raise ValueError("first-move panel shard contains no rainfall groups")

    mpc = DirectTFVRecedingMPCV7(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        policy_admission_calibration=policy,
        sequence_support=support,
        design=DirectTFVMPCDesignV4(active_support_quantile="q95"),
    )
    actuator_ids = tuple(str(value) for value in graph.actuator_ids)
    full_sha = direct_tfv_first_move_source_sha256()
    behavioral_sha = direct_tfv_first_move_behavioral_source_sha256()
    output_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for group in groups:
        entry = context.entry(group)
        active_target = torch.as_tensor(entry["active_target"], dtype=torch.float32, device=device)
        current_state = torch.as_tensor(entry["current_state"], dtype=torch.float32, device=device)[None]
        rainfall = torch.as_tensor(
            entry["default_rainfall_forecast"], dtype=torch.float32, device=device
        )
        previous_flow = torch.as_tensor(
            entry["previous_actuator_flow"], dtype=torch.float32, device=device
        )[None]
        upstream = mpc.optimize(
            current_state=current_state,
            rainfall=rainfall,
            previous_actuator_flow=previous_flow,
            current_settings=active_target,
            active_target=active_target,
        )
        base_candidate = upstream.optimized_candidate_settings
        if base_candidate is None:
            raise RuntimeError(f"{group}: V7 did not preserve a q95 optimizer candidate")
        refined = refine_supported_first_move(
            mpc=mpc,
            base_candidate=base_candidate,
            current_state=current_state,
            rainfall=rainfall,
            previous_actuator_flow=previous_flow,
            active_target=active_target,
            maxiter=int(args.first_move_maxiter),
            deadline_seconds=float(args.first_move_deadline_seconds),
        )
        hold_sequence = _hold_sequence(actuator_ids, active_target)
        candidate_sequence = _sequence(actuator_ids, refined.sequence)
        hold_sha = canonical_sequence_sha(hold_sequence)
        candidate_sha = canonical_sequence_sha(candidate_sequence)
        common = {
            "event_id": entry["event_id"],
            "rainfall_group": group,
            "scientific_split": entry["scientific_split"],
            "development_fold": entry["development_fold"],
            "checkpoint_id": entry["checkpoint_id"],
            "checkpoint_minutes": int(entry["elapsed_seconds"]) // 60,
            "inp_path": entry["inp_path"],
            "trajectory_metadata_path": entry["trajectory_metadata_path"],
            "first_move_panel_contract": DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
            "first_move_query_step3_contract": DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
            "first_move_source_sha256": full_sha,
            "first_move_behavioral_source_sha256": behavioral_sha,
            "first_move_context_sha256": entry["context_sha256"],
            "prefix_sha256": entry["prefix_sha256"],
            "candidate_rows_used": False,
            "generic_d3_candidate_dependency": False,
            "sequence_rate_feasible": True,
        }
        output_rows.append(
            {
                **common,
                "data_role": HOLD_ROLE,
                "sequence_index": 0,
                "candidate_family": "previous_target_latch_hold",
                "settings_sequence_json": json.dumps(hold_sequence, sort_keys=True),
                "sequence_sha256": hold_sha,
                "first_move_role": "LATCH_PREVIOUS_TARGET_REFERENCE",
                "first_move_semantics": "LATCH_PREVIOUS_TARGET_H360",
                "first_move_changed_facility_count": 0,
                "predicted_refined_delta_tfv_m3": 0.0,
            }
        )
        output_rows.append(
            {
                **common,
                "data_role": FIRST_MOVE_CANDIDATE_ROLE,
                "sequence_index": 1,
                "candidate_family": "v9_refined_target_latch_first_move",
                "settings_sequence_json": json.dumps(candidate_sequence, sort_keys=True),
                "sequence_sha256": candidate_sha,
                "first_move_role": "REFINED_TARGET_LATCH_QUERY",
                "first_move_semantics": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
                "predicted_refined_delta_tfv_m3": float(refined.predicted_delta_tfv_m3),
                "base_prefix_predicted_delta_tfv_m3": float(
                    refined.base_prefix_predicted_delta_tfv_m3
                ),
                "refinement_gain_m3": float(refined.gain_vs_base_prefix_m3),
                "first_move_changed_facility_count": int(refined.changed_facility_count),
                "pre_prune_changed_facility_count": int(refined.pre_prune_changed_facility_count),
                "pruned_facility_count": int(refined.pruned_facility_count),
                "full_plan_predicted_delta_tfv_m3": float(
                    upstream.raw_optimized_predicted_delta_tfv_m3
                ),
            }
        )
        summaries.append(
            {
                "rainfall_group": group,
                "event_id": entry["event_id"],
                "checkpoint_id": entry["checkpoint_id"],
                "sequence_sha256": candidate_sha,
                "predicted_refined_delta_tfv_m3": float(refined.predicted_delta_tfv_m3),
                "first_move_changed_facility_count": int(refined.changed_facility_count),
                "pre_prune_changed_facility_count": int(refined.pre_prune_changed_facility_count),
                "pruned_facility_count": int(refined.pruned_facility_count),
            }
        )

    frame = pd.DataFrame.from_records(output_rows)
    if len(frame) != 2 * len(groups):
        raise RuntimeError("first-move panel must have exactly HOLD+candidate per rainfall group")
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
        "global_rainfall_group_count": len(all_groups),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "rainfall_group_count": len(groups),
        "rainfall_groups": groups,
        "rows": len(frame),
        "hold_rows": len(groups),
        "candidate_rows": len(groups),
        "candidate_rows_used_for_context": False,
        "generic_d3_candidate_dependency": False,
        "records": summaries,
        "lineage": {
            "first_move_source_sha256": full_sha,
            "first_move_behavioral_source_sha256": behavioral_sha,
            "step2_checkpoint_sha256": _sha(args.checkpoint),
            "policy_admission_sha256": _sha(args.policy_admission),
            "sequence_support_sha256": _sha(args.sequence_support),
            "context_store_sha256": _sha(args.context_store),
        },
        "online_swmm_called": False,
    }
    summary_path = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
