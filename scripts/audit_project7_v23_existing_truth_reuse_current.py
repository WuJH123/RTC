"""Audit whether completed Project7 exact-return truth can be reused by V23 without new SWMM.

For every frozen learning query this script regenerates the current V23 three-family portfolio from
the saved causal context, applies the same q95 H10 joint-sequence contraction, and compares the
resulting first target with already recorded authoritative truth. Formal retraining is allowed only
when every non-excluded supplied query is fully exact-matched; partial-match cherry-picking is not
allowed. Original train/validation/calibration roles are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    encode_policy_return_action_token,
    sha256_file,
)
from rtc.direct_tfv_policy_return_portfolio_admission import validate_policy_return_learning_record
from rtc.direct_tfv_policy_return_portfolio_v23 import (
    V23_PORTFOLIO_CONTRACT,
    build_hybrid_policy_return_portfolio_v23,
)
from rtc.direct_tfv_sequence_support import (
    SEQUENCE_SUPPORT_METRICS,
    changed_facility_support_limit,
    sequence_support_limit,
    validate_direct_tfv_sequence_support,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph
from rtc.project7_v23_formal_reuse import (
    LEARNING_ROLES,
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    compare_candidate_targets,
    learning_groups_by_role,
)
from rtc.step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4, DirectTFVRecedingMPCV4


def _read_records(paths: list[str]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            validate_policy_return_learning_record(row)
            key = (
                str(row["query_set_id"]).lower(),
                str(row["context_npz_sha256"]).lower(),
                str(row["candidate_source"]),
            )
            previous = unique.get(key)
            if previous is not None:
                if json.dumps(previous, sort_keys=True) != json.dumps(row, sort_keys=True):
                    raise ValueError(f"conflicting duplicate exact-return record: {key}")
                continue
            unique[key] = row
    if not unique:
        raise ValueError("no validator-pure existing learning records found")
    records = list(unique.values())
    learning_groups_by_role(records)
    return records


def _load_query_context(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    reference: dict[str, np.ndarray] | None = None
    shared_keys = (
        "current_state",
        "rainfall_scenarios",
        "active_target",
        "previous_actuator_flow",
    )
    for row in rows:
        path = Path(str(row["context_npz"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path).lower() != str(row["context_npz_sha256"]).lower():
            raise ValueError(f"context SHA mismatch: {path}")
        data = np.load(path, allow_pickle=False)
        current = {key: np.asarray(data[key]).copy() for key in shared_keys}
        if reference is None:
            reference = current
        else:
            for key in shared_keys:
                if reference[key].shape != current[key].shape or not np.array_equal(
                    reference[key], current[key]
                ):
                    raise ValueError(
                        f"query {row['query_set_id']} candidate contexts do not share exact causal {key}"
                    )
    assert reference is not None
    return reference


def _recorded_target(row: dict[str, Any]) -> np.ndarray:
    data = np.load(Path(str(row["context_npz"])).resolve(), allow_pickle=False)
    target = np.asarray(data["candidate_target"], dtype=np.float32)
    if target.ndim == 2 and target.shape[0] == 1:
        target = target[0]
    target = target.reshape(-1)
    if target.shape != (109,):
        raise ValueError("existing truth candidate_target must contain 109 settings")
    return target


def _contract_q95(
    target: torch.Tensor,
    active: torch.Tensor,
    *,
    support: dict[str, Any],
    horizon_steps: int,
    control_block_steps: int,
    free_control_blocks: int,
) -> torch.Tensor:
    _, candidate = encode_policy_return_action_token(
        active,
        target,
        horizon_steps=horizon_steps,
        first_action_steps=control_block_steps,
    )
    sequence = candidate[0]
    blocks = sequence.reshape(-1, control_block_steps, 109).mean(dim=1)
    free = blocks[:free_control_blocks]
    delta = free - active[None]
    previous = torch.cat((torch.zeros_like(delta[:1]), delta[:-1]), dim=0)
    geometry = {
        "first_block_l1": torch.sum(torch.abs(delta[0])),
        "h120_l1": torch.sum(torch.abs(delta)),
        "h120_total_variation_l1": torch.sum(torch.abs(delta - previous)),
    }
    scales: list[torch.Tensor] = []
    for metric in SEQUENCE_SUPPORT_METRICS:
        limit = torch.as_tensor(
            sequence_support_limit(support, metric, "q95"),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        mass = geometry[metric]
        scales.append(
            torch.where(
                mass > 1.0e-12,
                limit / mass.clamp_min(1.0e-12),
                torch.ones_like(mass),
            )
        )
    scale = torch.clamp(torch.min(torch.stack(scales)), min=0.0, max=1.0)
    hold = active[None].expand_as(sequence)
    return (hold + scale * (sequence - hold))[0].detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", action="append", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--supervisory-control", required=True)
    parser.add_argument("--sequence-support", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--exclude-event-id", action="append", default=[])
    args = parser.parse_args()

    device = torch.device(args.device)
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.step2,
        graph=graph,
        device=device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    control, supervisory_mask = load_native_supervisory_control(
        args.supervisory_control,
        actuator_ids=graph.actuator_ids,
    )
    support = json.loads(Path(args.sequence_support).read_text(encoding="utf-8"))
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=sha256_file(args.step2),
        supervisory_mask=supervisory_mask,
        supervisory_control_contract=str(control["contract"]),
    )
    design = DirectTFVMPCDesignV4(
        maxiter=1,
        deadline_seconds=30.0,
        active_facility_count=0,
        active_support_quantile="q95",
    )
    geometry = DirectTFVRecedingMPCV4(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=checkpoint["action_support"],
        design=design,
    )
    ceiling = changed_facility_support_limit(support, "q95")

    records = _read_records(args.records_jsonl)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_query[str(row["query_set_id"])].append(row)

    excluded_events = {str(value) for value in args.exclude_event_id}
    query_reports: list[dict[str, Any]] = []
    exact_rows_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_matched: Counter[str] = Counter()
    source_generated: Counter[str] = Counter()

    for query_id, rows in sorted(by_query.items()):
        rainfall_groups = {str(row["rainfall_group"]) for row in rows}
        event_ids = {str(row["event_id"]) for row in rows}
        roles = {str(row["data_role"]) for row in rows}
        if len(rainfall_groups) != 1 or len(roles) != 1:
            raise ValueError(f"query {query_id} crosses rainfall groups or original learning roles")
        rainfall_group = next(iter(rainfall_groups))
        source_role = next(iter(roles))
        excluded = bool(event_ids & excluded_events)

        context = _load_query_context(rows)
        current_state = torch.as_tensor(context["current_state"], dtype=torch.float32, device=device)
        if current_state.ndim == 2:
            current_state = current_state[None]
        rainfall = torch.as_tensor(context["rainfall_scenarios"], dtype=torch.float32, device=device)
        if rainfall.ndim == 5 and rainfall.shape[0] == 1:
            rainfall = rainfall[0]
        active = torch.as_tensor(context["active_target"], dtype=torch.float32, device=device)
        if active.ndim == 2 and active.shape[0] == 1:
            active = active[0]
        flow = torch.as_tensor(context["previous_actuator_flow"], dtype=torch.float32, device=device)
        if flow.ndim == 1:
            flow = flow[None]

        portfolio = build_hybrid_policy_return_portfolio_v23(
            model=model,
            normalization=normalization,
            graph=graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active,
            first_radius=geometry.first_radius,
            max_changed_facilities=int(ceiling),
            max_delta_per_update=0.5,
            probe_chunk_size=int(args.probe_chunk_size),
            supervisory_mask=supervisory_mask,
        )
        generated: dict[str, torch.Tensor] = {}
        for proposal in portfolio.candidates:
            generated[str(proposal.source)] = _contract_q95(
                proposal.target,
                active,
                support=support,
                horizon_steps=int(design.prediction_horizon_steps),
                control_block_steps=int(design.control_block_steps),
                free_control_blocks=int(design.free_control_blocks),
            )
            source_generated[str(proposal.source)] += 1

        existing_by_source = {str(row["candidate_source"]): row for row in rows}
        detail: dict[str, Any] = {}
        all_generated_matched = bool(generated)
        for source, target in generated.items():
            row = existing_by_source.get(source)
            if row is None:
                detail[source] = {"existing_truth_present": False, "exact_match": False}
                all_generated_matched = False
                continue
            match = compare_candidate_targets(
                target.detach().cpu().numpy(),
                _recorded_target(row),
            )
            detail[source] = {
                "existing_truth_present": True,
                "exact_match": match.matched,
                "maximum_absolute_difference": match.maximum_absolute_difference,
                "v23_target_float32_sha256": match.exact_float32_sha256,
                "recorded_candidate_first_target_sha256": row.get("candidate_first_target_sha256"),
            }
            if match.matched:
                source_matched[source] += 1
                exact_rows_by_query[query_id].append(row)
            else:
                all_generated_matched = False
        stale_sources = sorted(set(existing_by_source) - set(generated))
        if stale_sources:
            all_generated_matched = False
        eligible = bool(all_generated_matched and not excluded)
        query_reports.append(
            {
                "query_set_id": query_id,
                "rainfall_group": rainfall_group,
                "source_data_role": source_role,
                "event_ids": sorted(event_ids),
                "excluded_development_event": excluded,
                "generated_sources": sorted(generated),
                "existing_sources": sorted(existing_by_source),
                "stale_existing_sources": stale_sources,
                "all_generated_candidates_exactly_matched": all_generated_matched,
                "formal_reuse_eligible": eligible,
                "source_match": detail,
                "v23_network_stress_q75": portfolio.hydraulic_diagnostics.network_stress_q75,
                "v23_strong_storm_blend": portfolio.hydraulic_diagnostics.strong_storm_blend,
            }
        )

    nonexcluded = [row for row in query_reports if not row["excluded_development_event"]]
    all_nonexcluded_matched = bool(nonexcluded) and all(
        row["all_generated_candidates_exactly_matched"] for row in nonexcluded
    )
    eligible_query_ids = {
        str(row["query_set_id"]) for row in query_reports if row["formal_reuse_eligible"]
    }
    matched_records = [
        row
        for query_id in sorted(eligible_query_ids)
        for row in exact_rows_by_query[query_id]
    ]
    role_groups = learning_groups_by_role(records)
    eligible_role_groups: dict[str, set[str]] = {role: set() for role in LEARNING_ROLES}
    for row in query_reports:
        if row["formal_reuse_eligible"]:
            eligible_role_groups[str(row["source_data_role"])].add(str(row["rainfall_group"]))
    role_complete = all(
        eligible_role_groups[role] == set(role_groups[role]) for role in LEARNING_ROLES
    )
    formal_allowed = bool(all_nonexcluded_matched and role_complete)

    report = {
        "contract": V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
        "v23_portfolio_contract": V23_PORTFOLIO_CONTRACT,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "no_new_empirical_data": True,
        "new_swmm_truth_generated": False,
        "new_rainfall_generated": False,
        "new_training_data_generated": False,
        "nearest_neighbor_matching_used": False,
        "interpolation_used": False,
        "pseudo_labels_used": False,
        "partial_match_cherry_picking_allowed": False,
        "original_learning_roles_preserved": True,
        "source_records": [str(Path(value).resolve()) for value in args.records_jsonl],
        "source_records_sha256": [sha256_file(value) for value in args.records_jsonl],
        "graph_sha256": sha256_file(args.graph),
        "step2_sha256": sha256_file(args.step2),
        "supervisory_control_sha256": sha256_file(args.supervisory_control),
        "sequence_support_sha256": sha256_file(args.sequence_support),
        "excluded_development_event_ids": sorted(excluded_events),
        "validated_existing_record_count": len(records),
        "query_count": len(by_query),
        "nonexcluded_query_count": len(nonexcluded),
        "formal_reuse_eligible_query_count": len(eligible_query_ids),
        "all_nonexcluded_queries_fully_exact_matched": all_nonexcluded_matched,
        "all_original_learning_role_groups_retained": role_complete,
        "source_learning_group_counts": {
            role: len(groups) for role, groups in role_groups.items()
        },
        "eligible_learning_group_counts": {
            role: len(groups) for role, groups in eligible_role_groups.items()
        },
        "generated_candidate_source_counts": dict(sorted(source_generated.items())),
        "exact_matched_candidate_source_counts": dict(sorted(source_matched.items())),
        "formal_training_allowed": formal_allowed,
        "formal_training_fail_closed_reason": (
            None
            if formal_allowed
            else "V23 candidate truth is not fully exact-matched across the complete preserved existing learning roles"
        ),
        "ready_for_policy_lock": False,
        "query_reports": query_reports,
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "V23_EXISTING_TRUTH_EXACT_MATCH_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matched_path = out_dir / "V23_EXISTING_TRUTH_EXACT_MATCHED_RECORDS.jsonl"
    with matched_path.open("w", encoding="utf-8") as handle:
        for row in matched_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                **{key: value for key, value in report.items() if key != "query_reports"},
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "matched_records_path": str(matched_path),
                "matched_records_sha256": sha256_file(matched_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
