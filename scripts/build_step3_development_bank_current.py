"""Build one validator-pure Step3 development bank from all completed exact-return truth.

This is deliberately a DEVELOPMENT repartition, not a new formal evidence split. It reuses completed
train/validation/calibration truth plus any later completed learning rows, validates every row through
the current authoritative firewall, deduplicates exact records, and repartitions complete rainfall
groups 8:1:1. Partial contexts or parent-only artifacts never enter because they have no valid learning
record. The split is deterministic and stratified on oracle HOLD-vs-ACTION so the small validation and
calibration panels are not accidentally single-class.

No SWMM is called and no source record/context is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    sha256_file,
)
from rtc.direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
from rtc.direct_tfv_policy_return_portfolio_admission import validate_policy_return_learning_record


STEP3_DEVELOPMENT_BANK_CONTRACT = "PROJECT7_STEP3_ACCURACY_FIRST_DEVELOPMENT_BANK_811_V1"
_ALLOWED_SOURCE_ROLES = {
    "policy_return_train",
    "policy_return_validation",
    "policy_return_calibration",
}
_SPLIT_ROLES = {
    "train": "policy_return_train",
    "validation": "policy_return_validation",
    "calibration": "policy_return_calibration",
}


def _hash_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _read_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path_like in paths:
        path = Path(path_like)
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            validate_policy_return_learning_record(row)
            role = str(row.get("data_role", ""))
            if role not in _ALLOWED_SOURCE_ROLES:
                raise ValueError(f"unexpected source role after firewall validation: {role}")
            query = str(row.get("query_set_id", "")).strip().lower()
            context_sha = str(row.get("context_npz_sha256", "")).strip().lower()
            source = str(row.get("candidate_source", "")).strip()
            key = (query, context_sha, source)
            previous = unique.get(key)
            if previous is not None:
                if json.dumps(previous, sort_keys=True) != json.dumps(row, sort_keys=True):
                    raise ValueError(f"conflicting duplicate exact-return record: {key}")
                continue
            unique[key] = row
    if not unique:
        raise ValueError("no complete validator-pure exact-return records were found")
    return list(unique.values())


def _group_queries(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[str(row["rainfall_group"])].append(row)
    for group, rows in by_group.items():
        query_ids = {str(row["query_set_id"]) for row in rows}
        if len(query_ids) != 1:
            raise ValueError(f"development bank expects one frozen query per rainfall group: {group}")
        if len(rows) < 2:
            raise ValueError(f"development bank requires >=2 candidates per query: {group}")
    return dict(by_group)


def _oracle_hold(rows: list[dict[str, Any]]) -> bool:
    return min(float(row["true_policy_return_delta_tfv_m3"]) for row in rows) >= 0.0


def _class_eval_count(class_count: int, eval_total: int, total: int) -> int:
    if class_count <= 0:
        return 0
    raw = int(round(class_count * eval_total / max(1, total)))
    if class_count >= 3 and eval_total > 0:
        raw = max(1, raw)
    return min(raw, class_count)


def _split_groups(
    by_group: dict[str, list[dict[str, Any]]], *, seed: int
) -> dict[str, list[str]]:
    groups = sorted(by_group)
    total = len(groups)
    if total < 20:
        raise ValueError("Step3 8:1:1 development repartition requires at least 20 complete groups")
    eval_count = max(1, int(round(total * 0.10)))
    if total - 2 * eval_count <= 0:
        raise ValueError("not enough groups for 8:1:1 development repartition")

    hold = sorted(
        [group for group in groups if _oracle_hold(by_group[group])],
        key=lambda value: _hash_key(seed, value),
    )
    action = sorted(
        [group for group in groups if not _oracle_hold(by_group[group])],
        key=lambda value: _hash_key(seed, value),
    )
    hold_eval = _class_eval_count(len(hold), eval_count, total)
    action_eval = eval_count - hold_eval
    if action_eval > len(action):
        deficit = action_eval - len(action)
        action_eval = len(action)
        hold_eval = min(len(hold) // 2, hold_eval + deficit)
    if 2 * hold_eval > len(hold):
        hold_eval = len(hold) // 2
        action_eval = eval_count - hold_eval
    if 2 * action_eval > len(action):
        action_eval = len(action) // 2
        hold_eval = eval_count - action_eval
    if hold_eval + action_eval != eval_count:
        raise ValueError("cannot create class-stratified 8:1:1 development split")

    validation = hold[:hold_eval] + action[:action_eval]
    calibration = (
        hold[hold_eval : 2 * hold_eval]
        + action[action_eval : 2 * action_eval]
    )
    used = set(validation) | set(calibration)
    train = [group for group in groups if group not in used]
    return {
        "train": sorted(train),
        "validation": sorted(validation),
        "calibration": sorted(calibration),
    }


def _compile(
    records: list[dict[str, Any]],
    *,
    role: str,
    split_name: str,
    out: Path,
) -> dict[str, Any]:
    current_states: list[np.ndarray] = []
    rainfall: list[np.ndarray] = []
    active_targets: list[np.ndarray] = []
    candidate_targets: list[np.ndarray] = []
    flows: list[np.ndarray] = []
    truth: list[float] = []
    base_scores: list[float] = []
    groups: list[str] = []
    event_ids: list[str] = []
    decision_indices: list[int] = []
    changed_counts: list[int] = []
    context_shas: list[str] = []
    query_ids: list[str] = []
    sources: list[str] = []
    source_roles: list[str] = []

    continuations = {str(row["continuation_policy_sha256"]).lower() for row in records}
    masks = {str(row["supervisory_mask_sha256"]).lower() for row in records}
    portfolios = {str(row["candidate_portfolio_contract"]) for row in records}
    if len(continuations) != 1 or len(masks) != 1:
        raise ValueError("development repartition mixes continuation or supervisory-mask lineage")
    if portfolios != {DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT}:
        raise ValueError("development repartition mixes candidate-portfolio lineage")

    for row in records:
        context_path = Path(str(row["context_npz"]))
        if not context_path.is_file():
            raise FileNotFoundError(context_path)
        expected_sha = str(row["context_npz_sha256"]).lower()
        if sha256_file(context_path).lower() != expected_sha:
            raise ValueError(f"context SHA mismatch: {context_path}")
        data = np.load(context_path, allow_pickle=False)
        if str(np.asarray(data["contract"]).reshape(-1)[0]) != DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT:
            raise ValueError("context dataset contract mismatch")
        if str(np.asarray(data["estimand"]).reshape(-1)[0]) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
            raise ValueError("context estimand mismatch")
        if str(np.asarray(data["action_encoding_contract"]).reshape(-1)[0]) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
            raise ValueError("context action encoding mismatch")
        source_role = str(np.asarray(data["data_role"]).reshape(-1)[0])
        if source_role != str(row["data_role"]):
            raise ValueError("source record/context role mismatch")
        current_states.append(np.asarray(data["current_state"])[0].astype(np.float32))
        rainfall.append(np.asarray(data["rainfall_scenarios"])[0].astype(np.float32))
        active_targets.append(np.asarray(data["active_target"])[0].astype(np.float32))
        candidate_targets.append(np.asarray(data["candidate_target"])[0].astype(np.float32))
        flows.append(np.asarray(data["previous_actuator_flow"])[0].astype(np.float32))
        truth.append(float(row["true_policy_return_delta_tfv_m3"]))
        base_scores.append(float(row["base_step2_h10_score_m3"]))
        groups.append(str(row["rainfall_group"]))
        event_ids.append(str(row["event_id"]))
        decision_indices.append(int(row["decision_index"]))
        changed_counts.append(int(row["first_move_changed_facility_count"]))
        context_shas.append(expected_sha)
        query_ids.append(str(row["query_set_id"]))
        sources.append(str(row["candidate_source"]))
        source_roles.append(str(row["data_role"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        contract=np.asarray(DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT),
        estimand=np.asarray(DIRECT_TFV_POLICY_RETURN_ESTIMAND),
        action_encoding_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING),
        data_role=np.asarray(role),
        development_bank_contract=np.asarray(STEP3_DEVELOPMENT_BANK_CONTRACT),
        development_split=np.asarray(split_name),
        continuation_policy_sha256=np.asarray(next(iter(continuations))),
        candidate_portfolio_contract=np.asarray(DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT),
        supervisory_control_dimension=np.asarray(82, dtype=np.int64),
        model_action_channel_count=np.asarray(109, dtype=np.int64),
        supervisory_mask_sha256=np.asarray(next(iter(masks))),
        current_state=np.stack(current_states),
        rainfall_scenarios=np.stack(rainfall),
        active_target=np.stack(active_targets),
        candidate_target=np.stack(candidate_targets),
        previous_actuator_flow=np.stack(flows),
        true_policy_return_delta_tfv_m3=np.asarray(truth, dtype=np.float64),
        base_step2_h10_score_m3=np.asarray(base_scores, dtype=np.float64),
        rainfall_group=np.asarray(groups),
        event_id=np.asarray(event_ids),
        decision_index=np.asarray(decision_indices, dtype=np.int64),
        first_move_changed_facility_count=np.asarray(changed_counts, dtype=np.int64),
        context_npz_sha256=np.asarray(context_shas),
        query_set_id=np.asarray(query_ids),
        candidate_source=np.asarray(sources),
        source_data_role=np.asarray(source_roles),
    )
    return {
        "split": split_name,
        "data_role": role,
        "rainfall_group_count": len(set(groups)),
        "query_set_count": len(set(query_ids)),
        "record_count": len(records),
        "oracle_hold_group_count": len(
            {
                group
                for group in set(groups)
                if min(
                    truth[index]
                    for index, value in enumerate(groups)
                    if value == group
                )
                >= 0.0
            }
        ),
        "candidate_source_counts": dict(sorted(Counter(sources).items())),
        "source_role_counts": dict(sorted(Counter(source_roles).items())),
        "output": str(out.resolve()),
        "output_sha256": sha256_file(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _read_records(args.records_jsonl)
    by_group = _group_queries(records)
    splits = _split_groups(by_group, seed=args.seed)
    split_sets = {name: set(values) for name, values in splits.items()}
    if split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["calibration"] or split_sets["validation"] & split_sets["calibration"]:
        raise RuntimeError("development split groups overlap")

    out_dir = Path(args.out_dir)
    summaries: dict[str, Any] = {}
    for name, role in _SPLIT_ROLES.items():
        rows = [row for row in records if str(row["rainfall_group"]) in split_sets[name]]
        summaries[name] = _compile(
            rows,
            role=role,
            split_name=name,
            out=out_dir / f"STEP3_DEVELOPMENT_{name.upper()}.npz",
        )

    oracle_hold = {
        group: _oracle_hold(rows)
        for group, rows in by_group.items()
    }
    report = {
        "contract": STEP3_DEVELOPMENT_BANK_CONTRACT,
        "development_only": True,
        "no_new_swmm_truth": True,
        "source_records_jsonl": [str(Path(value).resolve()) for value in args.records_jsonl],
        "source_records_sha256": [sha256_file(value) for value in args.records_jsonl],
        "validated_record_count": len(records),
        "complete_rainfall_group_count": len(by_group),
        "split_ratio_target": [0.8, 0.1, 0.1],
        "split_unit": "RAINFALL_GROUP",
        "split_seed": args.seed,
        "stratification": "ORACLE_HOLD_VS_ACTION_DEVELOPMENT_ONLY",
        "partial_parent_or_context_artifacts_included": False,
        "train_groups": splits["train"],
        "validation_groups": splits["validation"],
        "calibration_groups": splits["calibration"],
        "oracle_hold_counts": {
            name: sum(oracle_hold[group] for group in groups)
            for name, groups in splits.items()
        },
        "splits": summaries,
        "ready_for_policy_lock": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "STEP3_DEVELOPMENT_BANK_811.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256_file(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
