"""Compile policy-return replay contexts into a role-pure, query-set-aware training dataset.

Rainfall groups remain the independent scientific split unit.  Candidate ranking, however, is only
well-defined among actions evaluated from the *same hydraulic prefix*.  Earlier code grouped the
ranking loss by rainfall_group, which can compare candidates from different states when more than
one decision is sampled from an event.  The compiler therefore carries a stable query_set_id and
candidate source into the dataset while remaining backward-compatible with legacy single-candidate
records.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS,
    DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS,
    DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS,
    sha256_file,
    validate_policy_return_record,
)
from rtc.direct_tfv_policy_return_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT


_MIN_GROUPS = {
    "policy_return_train": DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS,
    "policy_return_validation": DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS,
    "policy_return_calibration": DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS,
}


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"record line {line_number} is not an object")
        validate_policy_return_record(row)
        records.append(row)
    if not records:
        raise ValueError("policy-return record JSONL is empty")
    return records


def _query_set_id(row: dict) -> str:
    supplied = str(row.get("query_set_id", "")).strip().lower()
    if supplied:
        if len(supplied) != 64 or any(ch not in "0123456789abcdef" for ch in supplied):
            raise ValueError("policy-return query_set_id must be a canonical sha256")
        return supplied
    # Legacy one-candidate records did not carry an explicit query-set ID.  Derive one from fields
    # that uniquely identify the same authoritative decision prefix; never use candidate identity.
    return _canonical_sha(
        {
            "event_id": str(row["event_id"]),
            "rainfall_group": str(row["rainfall_group"]),
            "decision_index": int(row["decision_index"]),
            "decision_elapsed_seconds": int(row.get("decision_elapsed_seconds", -1)),
            "prefix_sha256": str(row["prefix_sha256"]).lower(),
            "hold_first_target_sha256": str(row["hold_first_target_sha256"]).lower(),
            "continuation_policy_sha256": str(row["continuation_policy_sha256"]).lower(),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records-jsonl", required=True)
    p.add_argument(
        "--data-role",
        choices=("policy_return_train", "policy_return_validation", "policy_return_calibration"),
        required=True,
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()
    records = _read_records(args.records_jsonl)
    if any(str(row["data_role"]) != args.data_role for row in records):
        raise ValueError("dataset compiler received mixed or wrong policy-return roles")
    groups = {str(row["rainfall_group"]) for row in records}
    if len(groups) < _MIN_GROUPS[args.data_role]:
        raise ValueError(
            f"{args.data_role} requires >= {_MIN_GROUPS[args.data_role]} independent rainfall groups"
        )
    continuation = {str(row["continuation_policy_sha256"]).lower() for row in records}
    if len(continuation) != 1:
        raise ValueError("policy-return dataset mixes continuation-policy lineages")

    portfolio_contracts = {
        str(row.get("candidate_portfolio_contract", "")).strip()
        for row in records
        if str(row.get("candidate_portfolio_contract", "")).strip()
    }
    if portfolio_contracts and portfolio_contracts != {DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT}:
        raise ValueError("policy-return dataset mixes or uses an unknown candidate portfolio contract")
    if portfolio_contracts and any(
        str(row.get("candidate_portfolio_contract", "")).strip() != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
        for row in records
    ):
        raise ValueError("portfolio dataset must not mix legacy and portfolio candidate records")
    portfolio_contract = next(iter(portfolio_contracts), "")

    current_states = []
    rainfall = []
    active_targets = []
    candidate_targets = []
    flows = []
    truth = []
    rainfall_groups = []
    event_ids = []
    decision_indices = []
    changed_counts = []
    context_shas = []
    query_set_ids = []
    candidate_sources = []
    for row in records:
        context_path = Path(str(row.get("context_npz", "")))
        if not context_path.is_file():
            raise FileNotFoundError(f"policy-return context missing: {context_path}")
        expected_sha = str(row.get("context_npz_sha256", "")).lower()
        if sha256_file(context_path).lower() != expected_sha:
            raise ValueError(f"policy-return context SHA mismatch: {context_path}")
        data = np.load(context_path, allow_pickle=False)
        if str(np.asarray(data["contract"]).reshape(-1)[0]) != DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT:
            raise ValueError("policy-return context has the wrong contract")
        if str(np.asarray(data["estimand"]).reshape(-1)[0]) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
            raise ValueError("policy-return context has the wrong estimand")
        if str(np.asarray(data["data_role"]).reshape(-1)[0]) != args.data_role:
            raise ValueError("policy-return context role differs from compiler role")
        current_states.append(np.asarray(data["current_state"])[0].astype(np.float32))
        rainfall.append(np.asarray(data["rainfall_scenarios"])[0].astype(np.float32))
        active_targets.append(np.asarray(data["active_target"])[0].astype(np.float32))
        candidate_targets.append(np.asarray(data["candidate_target"])[0].astype(np.float32))
        flows.append(np.asarray(data["previous_actuator_flow"])[0].astype(np.float32))
        truth.append(float(row["true_policy_return_delta_tfv_m3"]))
        rainfall_groups.append(str(row["rainfall_group"]))
        event_ids.append(str(row["event_id"]))
        decision_indices.append(int(row["decision_index"]))
        changed_counts.append(int(row["first_move_changed_facility_count"]))
        context_shas.append(expected_sha)
        query_set_ids.append(_query_set_id(row))
        candidate_sources.append(str(row.get("candidate_source", "LEGACY_SINGLE_CANDIDATE")))

    query_counts = Counter(query_set_ids)
    source_counts = Counter(candidate_sources)
    multi_query_sets = sum(count >= 2 for count in query_counts.values())
    if portfolio_contract and multi_query_sets <= 0:
        raise ValueError("portfolio dataset requires at least one same-prefix multi-candidate query set")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        contract=np.asarray(DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT),
        estimand=np.asarray(DIRECT_TFV_POLICY_RETURN_ESTIMAND),
        data_role=np.asarray(args.data_role),
        continuation_policy_sha256=np.asarray(next(iter(continuation))),
        candidate_portfolio_contract=np.asarray(portfolio_contract),
        current_state=np.stack(current_states),
        rainfall_scenarios=np.stack(rainfall),
        active_target=np.stack(active_targets),
        candidate_target=np.stack(candidate_targets),
        previous_actuator_flow=np.stack(flows),
        true_policy_return_delta_tfv_m3=np.asarray(truth, dtype=np.float64),
        rainfall_group=np.asarray(rainfall_groups),
        event_id=np.asarray(event_ids),
        decision_index=np.asarray(decision_indices, dtype=np.int64),
        first_move_changed_facility_count=np.asarray(changed_counts, dtype=np.int64),
        context_npz_sha256=np.asarray(context_shas),
        query_set_id=np.asarray(query_set_ids),
        candidate_source=np.asarray(candidate_sources),
    )
    summary = {
        "contract": DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "data_role": args.data_role,
        "sample_count": len(records),
        "rainfall_group_count": len(groups),
        "rainfall_groups": sorted(groups),
        "query_set_count": len(query_counts),
        "multi_candidate_query_set_count": int(multi_query_sets),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "candidate_portfolio_contract": portfolio_contract,
        "ranking_unit": "SAME_AUTHORITATIVE_PREFIX_QUERY_SET",
        "scientific_split_unit": "RAINFALL_GROUP",
        "continuation_policy_sha256": next(iter(continuation)),
        "output_sha256": sha256_file(out),
        "records_jsonl_sha256": sha256_file(args.records_jsonl),
    }
    out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
