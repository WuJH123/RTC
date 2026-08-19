"""Compile one-sample policy-return replay contexts into one role-pure training dataset."""
from __future__ import annotations

import argparse
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


_MIN_GROUPS = {
    "policy_return_train": DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS,
    "policy_return_validation": DIRECT_TFV_POLICY_RETURN_MIN_VALIDATION_GROUPS,
    "policy_return_calibration": DIRECT_TFV_POLICY_RETURN_MIN_CALIBRATION_GROUPS,
}


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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        contract=np.asarray(DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT),
        estimand=np.asarray(DIRECT_TFV_POLICY_RETURN_ESTIMAND),
        data_role=np.asarray(args.data_role),
        continuation_policy_sha256=np.asarray(next(iter(continuation))),
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
    )
    summary = {
        "contract": DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "data_role": args.data_role,
        "sample_count": len(records),
        "rainfall_group_count": len(groups),
        "rainfall_groups": sorted(groups),
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
