"""Freeze deterministic V125 D4 first-move counterfactuals; never runs SWMM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rtc.lazy_step1 import CausalStep1TrajectoryDataset
from rtc.production_cli import _load_graph, _load_step1
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_d4_action_support_v125 import (
    D4_ACTION_SUPPORT_CONTRACT_V125,
    D4ActionSupportContractV125,
    action_sequence_sha256_v125,
    action_support_gap_v125,
    common_anchor_continuation_sequence_v125,
    deterministic_d4_rainfall_roles_v125,
    first_move_family_summary_v125,
    knowledge_neighbourhood_first_moves_v125,
    select_gap_balanced_checkpoints_v125,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step3_knowledge_seeds_v123 import build_sparse_state_auto_rbc_anchor_v123


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _samples_by_checkpoint(dataset: CausalStep1TrajectoryDataset) -> dict[tuple[str, str, int], int]:
    result: dict[tuple[str, str, int], int] = {}
    for i, ref in enumerate(dataset.samples):
        key = (str(ref.event_id), str(ref.rainfall_group), int(ref.end_index * 300))
        if key in result:
            raise ValueError(f"duplicate causal no-control Step1 key: {key}")
        result[key] = i
    return result


def _train_no_control_index(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"scientific_split", "development_fold", "strategy", "event_id", "rainfall_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"D4 train index missing {missing}")
    frame = frame[
        (frame["scientific_split"].astype(str) == "development")
        & (frame["development_fold"].astype(str) == "train")
        & (frame["strategy"].astype(str) == "no_control")
    ].copy()
    if frame.empty:
        raise ValueError("D4 requires development/train/no_control Step1 lineage")
    if bool(frame.duplicated(["event_id", "rainfall_group"], keep=False).any()):
        raise ValueError("D4 no_control Step1 lineage is duplicated")
    return frame


def _candidate_first_moves(settings: np.ndarray, *, control_block_steps: int = 2) -> np.ndarray:
    values = np.asarray(settings, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] < control_block_steps:
        raise ValueError("D4 cache settings must be [candidate,time,actuator]")
    return np.mean(values[:, : int(control_block_steps), :], axis=1, dtype=np.float64).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = D4ActionSupportContractV125(
        max_checkpoints=int(args.max_checkpoints),
        max_active_groups=int(args.max_active_groups),
        audit_fraction=float(args.audit_fraction),
    )
    contract.validate()
    graph = _load_graph(args.graph)
    basis = build_control_basis_v60(graph)
    cache = V60TrainCache(args.cache_manifest)
    names = sorted(cache.names("D2"))
    fit, holdout = deterministic_rainfall_split_v60(cache, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    if len(fit_d2) != 112 or len(holdout_d2) != 32:
        raise ValueError("D4 requires the frozen 112/32 D2 split")

    train_index = _train_no_control_index(args.train_index)
    sensors = tuple(
        line.strip() for line in Path(args.sensors).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    dataset = CausalStep1TrajectoryDataset(
        train_index,
        graph=graph,
        sensor_nodes=sensors,
        history_steps=13,
        model_step_seconds=300,
        scientific_split="development",
        development_fold="train",
    )
    sample_map = _samples_by_checkpoint(dataset)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    step1 = _load_step1(args.step1, device)

    geometry_records: list[dict[str, Any]] = []
    payload_by_group: dict[str, dict[str, np.ndarray]] = {}
    for name in fit_d2:
        entry = cache.entry(name)
        ref_index = int(entry.reference_index)
        elapsed = int(np.asarray(entry.arrays["elapsed_seconds"][ref_index]).reshape(-1)[0])
        key = (str(entry.event_id), str(entry.rainfall_group), elapsed)
        sample_index = sample_map.get(key)
        if sample_index is None:
            raise ValueError(f"D4 missing no-control causal Step1 sample for {name}")
        sample_ref = dataset.samples[sample_index]
        compact = dataset._load(sample_ref.trajectory_index)
        current_setting = np.asarray(compact["setting"][sample_ref.end_index], dtype=np.float32)
        observed, mask, context, _ = dataset[sample_index]
        with torch.no_grad():
            sparse_state = step1(
                observed[None].to(device),
                mask[None].to(device),
                torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
                torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
                context[None].to(device),
            )[0].detach().cpu().numpy().astype(np.float32)

        reference_settings = np.asarray(entry.arrays["settings"][ref_index], dtype=np.float32)
        anchor_sequence = np.asarray(
            build_sparse_state_auto_rbc_anchor_v123(
                sparse_state,
                current_setting,
                reference_settings,
                graph,
                control_block_steps=2,
                max_delta_per_update=contract.max_delta_per_update,
            ),
            dtype=np.float32,
        )
        anchor_target = np.asarray(anchor_sequence[0], dtype=np.float32)
        first_moves = _candidate_first_moves(np.asarray(entry.arrays["settings"], dtype=np.float32))
        gap = action_support_gap_v125(
            current_setting,
            anchor_target,
            first_moves,
            max_delta_per_update=contract.max_delta_per_update,
        )
        record = {
            "group": name,
            "event_id": str(entry.event_id),
            "rainfall_group": str(entry.rainfall_group),
            "checkpoint_id": str(entry.checkpoint_id),
            "elapsed_seconds": elapsed,
            "step1_sample_index": int(sample_index),
            **gap,
        }
        geometry_records.append(record)
        payload_by_group[name] = {
            "current_setting": current_setting,
            "anchor_target": anchor_target,
            "anchor_sequence": anchor_sequence,
        }

    # Freeze D4 fit/audit roles before any D4 outcome exists.  The role is rainfall-level,
    # not branch-level, so no event/checkpoint sibling can leak across the calibration gate.
    rainfall_roles = deterministic_d4_rainfall_roles_v125(
        [str(x["rainfall_group"]) for x in geometry_records],
        audit_fraction=contract.audit_fraction,
    )
    for record in geometry_records:
        record["d4_split_role"] = rainfall_roles[str(record["rainfall_group"])]

    selected = select_gap_balanced_checkpoints_v125(
        geometry_records, max_checkpoints=contract.max_checkpoints
    )
    plan_rows: list[dict[str, Any]] = []
    family_pairs: list[tuple[str, np.ndarray]] = []
    block = int(basis.horizon.control_block_steps)
    for selected_rank, record in enumerate(selected):
        group = str(record["group"])
        payload = payload_by_group[group]
        current = payload["current_setting"]
        anchor = payload["anchor_target"]
        anchor_sequence = payload["anchor_sequence"]
        plans = knowledge_neighbourhood_first_moves_v125(
            current,
            anchor,
            basis.grouping.group_id_by_actuator,
            basis.min_setting,
            basis.max_setting,
            contract=contract,
        )
        for family, target in plans:
            family_pairs.append((family, target))
            sequence = common_anchor_continuation_sequence_v125(
                target, anchor_sequence, control_block_steps=block
            )
            seq_sha = action_sequence_sha256_v125(sequence)
            row_id = hashlib.sha256(
                f"{D4_ACTION_SUPPORT_CONTRACT_V125}|{group}|{family}|{seq_sha}".encode("utf-8")
            ).hexdigest()
            plan_rows.append({
                "contract": D4_ACTION_SUPPORT_CONTRACT_V125,
                "plan_row_id": row_id,
                "selected_rank": int(selected_rank),
                "split_role": str(record["d4_split_role"]),
                "group": group,
                "event_id": str(record["event_id"]),
                "rainfall_group": str(record["rainfall_group"]),
                "checkpoint_id": str(record["checkpoint_id"]),
                "elapsed_seconds": int(record["elapsed_seconds"]),
                "support_gap_l1_normalized": float(record["nearest_anchor_l1_normalized"]),
                "candidate_family": family,
                "control_block_steps": block,
                "horizon_steps": int(sequence.shape[0]),
                "first_move_target_json": json.dumps(target.astype(float).tolist(), separators=(",", ":")),
                "first_move_delta_json": json.dumps((target - current).astype(float).tolist(), separators=(",", ":")),
                "anchor_first_move_target_json": json.dumps(anchor.astype(float).tolist(), separators=(",", ":")),
                "action_sequence_sha256": seq_sha,
                # Preserve the exact SWMM supervision sequence; this plan is a local
                # artifact, not committed to Git, and must contain enough information to
                # reproduce every branch without re-running Step1/RBC logic.
                "action_sequence_json": json.dumps(sequence.astype(float).tolist(), separators=(",", ":")),
                "future_action_rule": "candidate_first_600s_then_exact_common_sparse_rbc_anchor_continuation",
                "swmm_generated": False,
            })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_frame = pd.DataFrame.from_records(plan_rows)
    plan_path = out / "STEP2_V125_D4_ACTION_SUPPORT_PLAN.csv"
    plan_frame.to_csv(plan_path, index=False)
    geometry_path = out / "STEP2_V125_D4_SUPPORT_GEOMETRY.csv"
    pd.DataFrame.from_records(geometry_records).to_csv(geometry_path, index=False)

    gap_values = np.asarray([float(x["nearest_anchor_l1_normalized"]) for x in geometry_records])
    selected_gap = np.asarray([float(x["nearest_anchor_l1_normalized"]) for x in selected])
    fit_groups = sorted(group for group, role in rainfall_roles.items() if role == "fit")
    audit_groups = sorted(group for group, role in rainfall_roles.items() if role == "audit")
    report: dict[str, Any] = {
        "contract": D4_ACTION_SUPPORT_CONTRACT_V125,
        "verdict": "D4_V2_PLAN_FROZEN_REVIEW_REQUIRED_BEFORE_SWMM",
        "boundary": {
            "new_swmm": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "holdout_outcomes_accessed": False,
            "selection_uses_outcome_labels": False,
            "split_uses_outcome_labels": False,
            "continuous_mpc_unblocked": False,
        },
        "lineage": {
            "graph_sha256": _sha(args.graph),
            "cache_manifest_sha256": _sha(args.cache_manifest),
            "step1_sha256": _sha(args.step1),
            "sensor_layout_sha256": _sha(args.sensors),
            "train_index_sha256": _sha(args.train_index),
            "step1_source_role": "development/train/no_control_only",
            "fit_d2_groups": len(fit_d2),
            "holdout_d2_groups_not_used": len(holdout_d2),
        },
        "design": {
            "max_checkpoints": contract.max_checkpoints,
            "selected_checkpoints": len(selected),
            "anchor_scales": list(contract.anchor_scales),
            "local_fraction": contract.local_fraction,
            "max_active_groups": contract.max_active_groups,
            "max_delta_per_update": contract.max_delta_per_update,
            "selection": "rainfall-group round-robin by largest normalized nearest-anchor L1 support gap",
            "legacy_support_first_move": "mean of first two 300-s frames = executable 600-s move",
            "counterfactual_credit": "only first executable 600-s block differs; all candidates share exact sparse-RBC anchor continuation",
            "families": first_move_family_summary_v125(family_pairs),
            "planned_branches": len(plan_rows),
        },
        "d4_split": {
            "unit": "rainfall_group",
            "audit_fraction": contract.audit_fraction,
            "fit_rainfall_groups": fit_groups,
            "audit_rainfall_groups": audit_groups,
            "overlap": sorted(set(fit_groups) & set(audit_groups)),
        },
        "support_geometry": {
            "fit_d2_count": len(geometry_records),
            "all_gap_median": float(np.median(gap_values)),
            "all_gap_p90": float(np.quantile(gap_values, 0.90)),
            "selected_gap_median": float(np.median(selected_gap)) if selected_gap.size else None,
            "selected_gap_min": float(np.min(selected_gap)) if selected_gap.size else None,
            "selected_rainfall_groups": len({str(x["rainfall_group"]) for x in selected}),
            "selected_events": len({str(x["event_id"]) for x in selected}),
        },
        "control_basis": basis_manifest_v60(basis),
        "artifacts": {"plan_csv": str(plan_path.resolve()), "geometry_csv": str(geometry_path.resolve())},
    }
    if report["d4_split"]["overlap"]:
        raise RuntimeError("D4 V2 fit/audit rainfall groups overlap")
    report_path = out / "STEP2_V125_D4_ACTION_SUPPORT_PLAN.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# STEP2 V125 D4 action-support plan V2",
        "",
        "**Plan only. No SWMM branch has been generated.**",
        "",
        f"- verdict: `{report['verdict']}`",
        f"- selected checkpoints: {len(selected)} / {contract.max_checkpoints}",
        f"- planned branches: {len(plan_rows)}",
        f"- D4 fit/audit rainfall groups: {len(fit_groups)}/{len(audit_groups)}",
        f"- selected support-gap median: {report['support_geometry']['selected_gap_median']:.6g}",
        "- counterfactual credit: candidate differs only in first 600 s; continuation is identical sparse-RBC anchor",
        "",
        "The split and selection are causal/action-geometric only and were frozen before D4 outcomes.",
    ]
    (out / "STEP2_V125_D4_ACTION_SUPPORT_PLAN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--train-index", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-checkpoints", type=int, default=48)
    p.add_argument("--max-active-groups", type=int, default=3)
    p.add_argument("--audit-fraction", type=float, default=0.25)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
