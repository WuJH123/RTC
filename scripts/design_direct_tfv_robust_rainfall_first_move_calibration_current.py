"""Design candidate-free V12 scenario-mean target-latch first-move calibration queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.data_design import canonical_sequence_sha
from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS, refine_supported_first_move
from rtc.direct_tfv_first_move_admission import DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT
from rtc.direct_tfv_first_move_context import load_first_move_context_store
from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.direct_tfv_sequence_support import validate_direct_tfv_sequence_support
from rtc.direct_tfv_v12_lineage import (
    V12_RAINFALL_DECAY_PER_STEP,
    V12_RAINFALL_HISTORY_STEPS,
    V12_RAINFALL_MULTIPLIERS,
    direct_tfv_v12_behavioral_sha256,
)
from rtc.forecast import PersistenceDecayForecast
from rtc.production_cli import _load_graph
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4
from rtc.step3_tfv_value_mpc_v7 import DirectTFVRecedingMPCV7
from rtc.step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
)


HOLD_ROLE = "D3_V60_HOLD_REFERENCE"
CANDIDATE_ROLE = "D3_V12_SCENARIO_MEAN_REFINED_FIRST_MOVE_CALIBRATION_CANDIDATE"
RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_V12_SCENARIO_MEAN_FIRST_MOVE_PANEL_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _ScenarioMeanQueryMPC(DirectTFVRecedingMPCV7):
    minimum_rainfall_scenarios = 3

    def _score_sequences(self, *, current_state, rainfall, sequences, previous_actuator_flow, active_target):
        if current_state.shape[0] != 1 or previous_actuator_flow.shape[0] != 1:
            raise ValueError("V12 query expects one state/flow vector")
        if rainfall.ndim != 4 or int(rainfall.shape[0]) < self.minimum_rainfall_scenarios:
            raise ValueError("V12 query requires causal rainfall scenarios [S,H,node,1]")
        if sequences.ndim != 3 or tuple(sequences.shape[1:]) != (
            self.design.prediction_horizon_steps,
            109,
        ):
            raise ValueError("V12 query sequences must be [candidate,H72,109]")
        candidate_count = int(sequences.shape[0])
        scenario_count = int(rainfall.shape[0])
        batch = candidate_count * scenario_count
        state = self._normalize_state(current_state).expand(batch, -1, -1)
        rain = self._normalize_rainfall(rainfall).unsqueeze(0).expand(
            candidate_count, -1, -1, -1, -1
        ).reshape(batch, *rainfall.shape[1:])
        flow = self._normalize_flow(previous_actuator_flow).expand(batch, -1)
        candidate = sequences.unsqueeze(1).expand(-1, scenario_count, -1, -1).reshape(
            batch, self.design.prediction_horizon_steps, 109
        )
        reference = self._hold_sequence(active_target)[None, None].expand(
            candidate_count, scenario_count, -1, -1
        ).reshape(batch, self.design.prediction_horizon_steps, 109)
        output = self.model(
            current_state=state,
            rainfall=rain,
            reference_settings=reference,
            candidate_settings=candidate,
            previous_actuator_flow=flow,
            actuator_upstream=torch.as_tensor(
                self.graph.actuator_upstream, dtype=torch.long, device=state.device
            ),
            actuator_downstream=torch.as_tensor(
                self.graph.actuator_downstream, dtype=torch.long, device=state.device
            ),
            actuator_physics=torch.as_tensor(
                self.graph.actuator_physics, dtype=state.dtype, device=state.device
            ),
        )
        scores = output.total_delta_tfv_m3.reshape(candidate_count, scenario_count)
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("V12 scenario-mean query produced non-finite scores")
        return scores.mean(dim=1)


def _sequence(ids, tensor):
    blocks = tensor.reshape(36, 2, 109).mean(dim=1).detach().cpu().to(torch.float64).numpy()
    return [{aid: float(row[i]) for i, aid in enumerate(ids)} for row in blocks]


def _hold(ids, active_target):
    values = active_target.detach().cpu().to(torch.float64).tolist()
    row = {aid: float(values[i]) for i, aid in enumerate(ids)}
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
    if not 1 <= args.shard_count <= 4 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("V12 panel requires shard-count 1..4 and valid shard-index")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.checkpoint, graph=graph, device=device
    )
    policy = json.loads(Path(args.policy_admission).read_text(encoding="utf-8"))
    if str(policy.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("V12 panel requires frozen V2 policy admission lineage")
    support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support, actuator_ids=graph.actuator_ids, step2_checkpoint_sha256=_sha(args.checkpoint)
    )
    context = load_first_move_context_store(args.context_store)
    groups_all = sorted(context.rainfall_groups)
    if len(groups_all) < 24:
        raise ValueError("V12 panel requires >=24 rainfall groups")
    groups = [g for i, g in enumerate(groups_all) if i % args.shard_count == args.shard_index]
    if not groups:
        raise ValueError("V12 panel shard is empty")
    mpc = _ScenarioMeanQueryMPC(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        policy_admission_calibration=policy,
        sequence_support=support,
        design=DirectTFVMPCDesignV4(active_support_quantile="q95"),
    )
    ids = tuple(str(x) for x in graph.actuator_ids)
    forecast = PersistenceDecayForecast(
        decay_per_step=V12_RAINFALL_DECAY_PER_STEP,
        scenario_multipliers=V12_RAINFALL_MULTIPLIERS,
        history_steps_for_level=V12_RAINFALL_HISTORY_STEPS,
    )
    behavior_sha = direct_tfv_v12_behavioral_sha256()
    rows = []
    summaries = []
    for group in groups:
        entry = context.entry(group)
        active = torch.as_tensor(entry["active_target"], dtype=torch.float32, device=device)
        state = torch.as_tensor(entry["current_state"], dtype=torch.float32, device=device)[None]
        rain_np = forecast.forecast(
            entry["rainfall_history"], horizon_steps=72
        ).astype("float32")
        rain = torch.as_tensor(rain_np, dtype=torch.float32, device=device)
        flow = torch.as_tensor(
            entry["previous_actuator_flow"], dtype=torch.float32, device=device
        )[None]
        upstream = mpc.optimize(
            current_state=state,
            rainfall=rain,
            previous_actuator_flow=flow,
            current_settings=active,
            active_target=active,
        )
        if upstream.optimized_candidate_settings is None:
            raise RuntimeError(f"{group}: V12 query generator produced no raw candidate")
        refined = refine_supported_first_move(
            mpc=mpc,
            base_candidate=upstream.optimized_candidate_settings,
            current_state=state,
            rainfall=rain,
            previous_actuator_flow=flow,
            active_target=active,
            maxiter=args.first_move_maxiter,
            deadline_seconds=args.first_move_deadline_seconds,
        )
        hold = _hold(ids, active)
        candidate = _sequence(ids, refined.sequence)
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
            "first_move_query_step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
            "rainfall_scenario_contract": DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
            "v12_behavioral_source_sha256": behavior_sha,
            "first_move_context_sha256": entry["context_sha256"],
            "prefix_sha256": entry["prefix_sha256"],
            "candidate_rows_used": False,
            "generic_d3_candidate_dependency": False,
            "sequence_rate_feasible": True,
        }
        hold_sha = canonical_sequence_sha(hold)
        candidate_sha = canonical_sequence_sha(candidate)
        rows.append(
            {
                **common,
                "data_role": HOLD_ROLE,
                "sequence_index": 0,
                "settings_sequence_json": json.dumps(hold, sort_keys=True),
                "sequence_sha256": hold_sha,
                "first_move_changed_facility_count": 0,
                "predicted_refined_delta_tfv_m3": 0.0,
            }
        )
        rows.append(
            {
                **common,
                "data_role": CANDIDATE_ROLE,
                "sequence_index": 1,
                "settings_sequence_json": json.dumps(candidate, sort_keys=True),
                "sequence_sha256": candidate_sha,
                "first_move_semantics": DIRECT_TFV_FIRST_MOVE_SEMANTICS,
                "predicted_refined_delta_tfv_m3": float(refined.predicted_delta_tfv_m3),
                "first_move_changed_facility_count": int(refined.changed_facility_count),
                "pre_prune_changed_facility_count": int(refined.pre_prune_changed_facility_count),
                "pruned_facility_count": int(refined.pruned_facility_count),
            }
        )
        summaries.append(
            {
                "rainfall_group": group,
                "candidate_sequence_sha256": candidate_sha,
                "predicted_refined_delta_tfv_m3": float(refined.predicted_delta_tfv_m3),
                "changed_facility_count": int(refined.changed_facility_count),
            }
        )
    frame = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(out, index=False)
    summary = {
        "contract": RUN_CONTRACT,
        "development_only": True,
        "query_step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        "rainfall_scenario_contract": DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
        "v12_behavioral_source_sha256": behavior_sha,
        "rainfall_multipliers": list(V12_RAINFALL_MULTIPLIERS),
        "rainfall_history_steps": V12_RAINFALL_HISTORY_STEPS,
        "rainfall_decay_per_step": V12_RAINFALL_DECAY_PER_STEP,
        "global_rainfall_group_count": len(groups_all),
        "rainfall_group_count": len(groups),
        "rainfall_groups": groups,
        "rows": len(frame),
        "records": summaries,
        "candidate_rows_used_for_context": False,
        "generic_d3_candidate_dependency": False,
        "lineage": {
            "step2_checkpoint_sha256": _sha(args.checkpoint),
            "policy_admission_sha256": _sha(args.policy_admission),
            "sequence_support_sha256": _sha(args.sequence_support),
            "context_store_sha256": _sha(args.context_store),
        },
        "online_swmm_called": False,
    }
    path = Path(args.summary_out) if args.summary_out else out.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
