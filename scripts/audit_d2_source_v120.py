"""Audit the upstream D2=4,800 authoritative source census for V120."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

CONTRACT = "PROJECT7_V120_D2_SOURCE_CENSUS_AUDIT_V1"
SPLIT_CONTRACT = "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"
EXPECTED_SOURCE_BRANCHES = 4800
EXPECTED_TRAIN_ELIGIBLE_BRANCHES = 3600
EXPECTED_BRANCHES_PER_GROUP = 25
EXPECTED_TRAIN_GROUPS = 144
EXPECTED_VALIDATION_GROUPS = 48


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _branch_identity(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Use simulation identity when present, otherwise an event/checkpoint/action composite."""
    if "simulation_identity_sha256" in frame.columns:
        value = frame["simulation_identity_sha256"].astype(str)
        if (value != "").all():
            return value, "simulation_identity_sha256"
    if not {"event_id", "checkpoint_id"}.issubset(frame.columns):
        raise ValueError("D2 source index requires event_id/checkpoint_id for composite identity")
    action_column = next(
        (
            name
            for name in ("candidate_action_sha256", "action_or_sequence_sha256", "metadata_path")
            if name in frame.columns
        ),
        None,
    )
    if action_column is None:
        raise ValueError("D2 source index lacks action/metadata identity")
    parts = frame[["event_id", "checkpoint_id", action_column]].astype(str)
    if (parts == "").any().any():
        raise ValueError("D2 source composite identity contains empty fields")
    return parts.agg("::".join, axis=1), f"event_id+checkpoint_id+{action_column}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Project7 D2=4800 source census")
    parser.add_argument("--index", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_contract).read_text(encoding="utf-8"))
    if not isinstance(split, dict) or split.get("contract") != SPLIT_CONTRACT:
        raise ValueError("D2 source audit requires frozen Project7 18/6/6 split")
    train_events = {str(x) for x in split.get("development_train", [])}
    validation_events = {str(x) for x in split.get("development_validation", [])}
    final_events = {str(x) for x in split.get("final", [])}
    if (len(train_events), len(validation_events), len(final_events)) != (18, 6, 6):
        raise ValueError("frozen split is not 18/6/6")

    frame = pd.read_csv(args.index, keep_default_na=False)
    if "event_id" not in frame.columns or "checkpoint_id" not in frame.columns:
        raise ValueError("D2 source index lacks event_id/checkpoint_id")
    if "source_kind" in frame.columns:
        frame = frame[frame["source_kind"].astype(str).str.upper() == "D2"].copy()
    if frame.empty:
        raise ValueError("D2 source index contains no D2 rows")

    identity, identity_contract = _branch_identity(frame)
    if identity.duplicated().any():
        raise ValueError("D2 source index is not a unique authoritative branch census")
    branches = int(len(frame))
    if branches != EXPECTED_SOURCE_BRANCHES:
        raise ValueError(f"D2 source census must be 4800 unique branches, got {branches}")

    events = frame["event_id"].astype(str)
    unknown_events = sorted(set(events) - train_events - validation_events - final_events)
    if unknown_events:
        raise ValueError(f"D2 source census contains events outside frozen split: {unknown_events}")
    if events.isin(final_events).any():
        raise ValueError("D2 source census must not contain untouched Final events")
    train_rows = frame[events.isin(train_events)].copy()
    validation_rows = frame[events.isin(validation_events)].copy()
    if int(len(train_rows)) != EXPECTED_TRAIN_ELIGIBLE_BRANCHES:
        raise ValueError(f"D2 frozen-Train view must be 3600 branches; got {len(train_rows)}")
    if int(len(validation_rows)) != EXPECTED_SOURCE_BRANCHES - EXPECTED_TRAIN_ELIGIBLE_BRANCHES:
        raise ValueError("D2 source Train/Validation partition does not sum to 4800")

    group_sizes = frame.groupby(["event_id", "checkpoint_id"], sort=False).size()
    if not (group_sizes == EXPECTED_BRANCHES_PER_GROUP).all():
        bad = group_sizes[group_sizes != EXPECTED_BRANCHES_PER_GROUP]
        raise ValueError(f"D2 source groups must contain 25 branches; bad groups: {bad.head().to_dict()}")
    train_groups = int(train_rows.groupby(["event_id", "checkpoint_id"]).ngroups)
    validation_groups = int(validation_rows.groupby(["event_id", "checkpoint_id"]).ngroups)
    if (train_groups, validation_groups) != (EXPECTED_TRAIN_GROUPS, EXPECTED_VALIDATION_GROUPS):
        raise ValueError(
            f"D2 source group census must be 144 Train + 48 Validation; got {train_groups}+{validation_groups}"
        )

    payload = {
        "contract": CONTRACT,
        "source_index": str(Path(args.index).resolve()),
        "source_index_sha256": _sha(args.index),
        "split_contract": str(Path(args.split_contract).resolve()),
        "split_contract_sha256": _sha(args.split_contract),
        "identity_contract": identity_contract,
        "unique_authoritative_branches": branches,
        "branches_per_group": EXPECTED_BRANCHES_PER_GROUP,
        "source_group_count": int(group_sizes.size),
        "frozen_train_eligible_branches": int(len(train_rows)),
        "frozen_train_group_count": train_groups,
        "development_validation_branches": int(len(validation_rows)),
        "development_validation_group_count": validation_groups,
        "final_branches": 0,
        "frozen_train_event_count": int(train_rows["event_id"].astype(str).nunique()),
        "development_validation_event_count": int(validation_rows["event_id"].astype(str).nunique()),
        "source_census_is_not_iid_training_population": True,
        "validation_branches_excluded_from_training": True,
        "final_untouched": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
