"""Audit reuse of existing exact-return truth for Project7 V23 without new empirical data.

The frozen v0.6.9 18/6/6 forcing-only split is the sole authority for scientific roles. Historical
``data_role`` values (including old calibration rows) are provenance only. Final rows are never opened
pre-lock. Development Train/Validation rows are regenerated under the current V23 portfolio and are
reusable for retraining only when the complete frozen roles are covered and every generated candidate
matches existing authoritative truth exactly. Otherwise the scientifically valid path is a fixed-policy
Formal evaluation with no retraining, not selective/cherry-picked reuse.
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
    ARCHIVAL_ROLE,
    FORMAL_LEARNING_ROLES,
    V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
    compare_candidate_targets,
    scientific_role_for_record,
    validate_frozen_split,
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
            if previous is not None and json.dumps(previous, sort_keys=True) != json.dumps(
                row, sort_keys=True
            ):
                raise ValueError(f"conflicting duplicate exact-return record: {key}")
            unique[key] = row
    if not unique:
        raise ValueError("no validator-pure existing policy-return records found")
    return list(unique.values())


def _identity(row: dict[str, Any]) -> str:
    for key in ("event_id", "rainfall_group"):
        text = Path(str(row.get(key, ""))).name.removesuffix(".inp")
        if text:
            return text
    return ""


def _load_query_context(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    reference: dict[str, np.ndarray] | None = None
    keys = ("current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow")
    for row in rows:
        path = Path(str(row["context_npz"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path).lower() != str(row["context_npz_sha256"]).lower():
            raise ValueError(f"context SHA mismatch: {path}")
        data = np.load(path, allow_pickle=False)
        current = {key: np.asarray(data[key]).copy() for key in keys}
        if reference is None:
            reference = current
        else:
            for key in keys:
                if reference[key].shape != current[key].shape or not np.array_equal(
                    reference[key], current[key]
                ):
                    raise ValueError(
                        f"query {row['query_set_id']} candidates do not share exact causal {key}"
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
        raise ValueError("existing candidate_target must contain 109 settings")
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
    ratios: list[torch.Tensor] = []
    for metric in SEQUENCE_SUPPORT_METRICS:
        limit = torch.as_tensor(
            sequence_support_limit(support, metric, "q95"),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        mass = geometry[metric]
        ratios.append(
            torch.where(
                mass > 1.0e-12,
                limit / mass.clamp_min(1.0e-12),
                torch.ones_like(mass),
            )
        )
    scale = torch.clamp(torch.min(torch.stack(ratios)), min=0.0, max=1.0)
    return (active[None].expand_as(sequence) + scale * (sequence - active[None]))[0].detach()


def _blend_bin(value: float) -> str:
    if value <= 0.0:
        return "blend_eq_0"
    if value <= 0.5:
        return "blend_0_to_0p5"
    return "blend_gt_0p5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", action="append", required=True)
    parser.add_argument("--formal-split-contract", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--supervisory-control", required=True)
    parser.add_argument("--sequence-support", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--exclude-event-id", action="append", default=[])
    args = parser.parse_args()

    split_path = Path(args.formal_split_contract).resolve()
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    frozen_roles = validate_frozen_split(split_payload)
    excluded_events = {str(value) for value in args.exclude_event_id}
    records = _read_records(args.records_jsonl)

    role_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    historical_data_roles: Counter[str] = Counter()
    for row in records:
        historical_data_roles[str(row.get("data_role", ""))] += 1
        role_records[scientific_role_for_record(row, split_payload)].append(row)

    # Final truth is sealed pre-lock. Merely finding a historical record mapped to Final is a
    # contamination signal; its context/label is intentionally never opened by this audit.
    final_record_count = len(role_records.get("final", ()))
    final_truth_contamination = final_record_count > 0

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

    query_reports: list[dict[str, Any]] = []
    authorised_rows: list[dict[str, Any]] = []
    source_generated: Counter[str] = Counter()
    source_matched: Counter[str] = Counter()
    hydraulic_blend_total: Counter[str] = Counter()
    hydraulic_blend_matched: Counter[str] = Counter()
    covered_events: dict[str, set[str]] = {role: set() for role in FORMAL_LEARNING_ROLES}
    all_queries_matched: dict[str, bool] = {role: True for role in FORMAL_LEARNING_ROLES}

    for role in FORMAL_LEARNING_ROLES:
        by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in role_records.get(role, ()):
            by_query[str(row["query_set_id"])].append(row)
        if not by_query:
            all_queries_matched[role] = False
            continue
        for query_id, rows in sorted(by_query.items()):
            event_ids = {str(row.get("event_id", "")) for row in rows}
            excluded = bool(event_ids & excluded_events)
            event_identity = _identity(rows[0])
            if event_identity in set(frozen_roles[role]):
                covered_events[role].add(event_identity)

            context = _load_query_context(rows)
            current_state = torch.as_tensor(
                context["current_state"], dtype=torch.float32, device=device
            )
            if current_state.ndim == 2:
                current_state = current_state[None]
            rainfall = torch.as_tensor(
                context["rainfall_scenarios"], dtype=torch.float32, device=device
            )
            if rainfall.ndim == 5 and rainfall.shape[0] == 1:
                rainfall = rainfall[0]
            active = torch.as_tensor(context["active_target"], dtype=torch.float32, device=device)
            if active.ndim == 2 and active.shape[0] == 1:
                active = active[0]
            flow = torch.as_tensor(
                context["previous_actuator_flow"], dtype=torch.float32, device=device
            )
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

            existing = {str(row["candidate_source"]): row for row in rows}
            details: dict[str, Any] = {}
            query_exact = bool(generated) and not excluded
            exact_rows: list[dict[str, Any]] = []
            blend_bin = _blend_bin(float(portfolio.hydraulic_diagnostics.strong_storm_blend))
            for source, target in generated.items():
                row = existing.get(source)
                if row is None:
                    details[source] = {"existing_truth_present": False, "exact_match": False}
                    query_exact = False
                    continue
                match = compare_candidate_targets(
                    target.detach().cpu().numpy(),
                    _recorded_target(row),
                )
                details[source] = {
                    "existing_truth_present": True,
                    "exact_match": match.matched,
                    "maximum_absolute_difference": match.maximum_absolute_difference,
                    "v23_target_float32_sha256": match.exact_float32_sha256,
                }
                if source == "TYPE_AWARE_HYDRAULIC_PRESSURE":
                    hydraulic_blend_total[blend_bin] += 1
                    if match.matched:
                        hydraulic_blend_matched[blend_bin] += 1
                if match.matched:
                    source_matched[source] += 1
                    exact_rows.append(row)
                else:
                    query_exact = False
            stale = sorted(set(existing) - set(generated))
            if stale:
                query_exact = False
            if not query_exact:
                all_queries_matched[role] = False
            else:
                authorised_rows.extend(exact_rows)
            query_reports.append(
                {
                    "query_set_id": query_id,
                    "scientific_role": role,
                    "event_identity": event_identity,
                    "historical_data_roles": sorted({str(row.get("data_role", "")) for row in rows}),
                    "excluded_development_steering_event": excluded,
                    "all_generated_candidates_exactly_matched": query_exact,
                    "generated_sources": sorted(generated),
                    "existing_sources": sorted(existing),
                    "stale_existing_sources": stale,
                    "v23_network_stress_q75": portfolio.hydraulic_diagnostics.network_stress_q75,
                    "v23_strong_storm_blend": portfolio.hydraulic_diagnostics.strong_storm_blend,
                    "source_match": details,
                }
            )

    full_coverage = {
        role: covered_events[role] == set(frozen_roles[role]) for role in FORMAL_LEARNING_ROLES
    }
    formal_retraining_allowed = bool(
        not final_truth_contamination
        and all(full_coverage.values())
        and all(all_queries_matched.values())
    )
    fixed_policy_formal_evaluation_allowed = bool(not final_truth_contamination)
    if formal_retraining_allowed:
        formal_mode = "EXACT_MATCH_RETRAIN_ALLOWED"
    elif fixed_policy_formal_evaluation_allowed:
        formal_mode = "FIXED_POLICY_NO_RETRAIN"
    else:
        formal_mode = "BLOCKED_FINAL_CONTAMINATION"

    if not formal_retraining_allowed:
        # Prevent accidental matched-subset training when the complete frozen learning roles are not
        # distribution-matched. The detailed per-query audit remains available for diagnosis.
        authorised_rows = []

    blend_summary = {}
    for name in ("blend_eq_0", "blend_0_to_0p5", "blend_gt_0p5"):
        total = int(hydraulic_blend_total[name])
        matched = int(hydraulic_blend_matched[name])
        blend_summary[name] = {
            "total": total,
            "exact_matched": matched,
            "exact_match_fraction": (matched / total if total else None),
        }

    report = {
        "contract": V23_EXISTING_TRUTH_REUSE_AUDIT_CONTRACT,
        "v23_portfolio_contract": V23_PORTFOLIO_CONTRACT,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "formal_split_contract_path": str(split_path),
        "formal_split_contract_sha256": sha256_file(split_path),
        "scientific_role_authority": "FROZEN_V069_SPLIT_CONTRACT_NOT_HISTORICAL_DATA_ROLE",
        "historical_policy_return_calibration_role_is_archival_only": True,
        "calibration_role_used_for_formal_learning": False,
        "no_new_empirical_data": True,
        "new_swmm_truth_generated": False,
        "new_rainfall_generated": False,
        "new_training_data_generated": False,
        "nearest_neighbor_matching_used": False,
        "interpolation_used": False,
        "pseudo_labels_used": False,
        "partial_match_cherry_picking_allowed": False,
        "source_records": [str(Path(value).resolve()) for value in args.records_jsonl],
        "source_records_sha256": [sha256_file(value) for value in args.records_jsonl],
        "historical_data_role_counts": dict(sorted(historical_data_roles.items())),
        "scientific_role_record_counts": {
            role: len(role_records.get(role, ()))
            for role in (*FORMAL_LEARNING_ROLES, "final", ARCHIVAL_ROLE)
        },
        "final_truth_record_count_detected_prelock": final_record_count,
        "final_truth_contamination": final_truth_contamination,
        "frozen_role_event_counts": {role: len(values) for role, values in frozen_roles.items()},
        "covered_learning_events": {role: sorted(values) for role, values in covered_events.items()},
        "complete_frozen_role_coverage": full_coverage,
        "all_role_queries_exact_matched": all_queries_matched,
        "generated_candidate_source_counts": dict(sorted(source_generated.items())),
        "exact_matched_candidate_source_counts": dict(sorted(source_matched.items())),
        "hydraulic_exact_match_by_v23_blend": blend_summary,
        "formal_retraining_allowed": formal_retraining_allowed,
        "fixed_policy_formal_evaluation_allowed": fixed_policy_formal_evaluation_allowed,
        "recommended_formal_mode": formal_mode,
        "ready_for_policy_lock": False,
        "query_reports": query_reports,
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "V23_EXISTING_TRUTH_EXACT_MATCH_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authorised_path = out_dir / "V23_FORMAL_RETRAIN_AUTHORISED_EXISTING_RECORDS.jsonl"
    with authorised_path.open("w", encoding="utf-8") as handle:
        for row in authorised_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                **{key: value for key, value in report.items() if key != "query_reports"},
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "authorised_records_path": str(authorised_path),
                "authorised_records_sha256": sha256_file(authorised_path),
                "authorised_record_count": len(authorised_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
