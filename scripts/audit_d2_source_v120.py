"""Audit the upstream D2=4,800 authoritative source census for V120.

This script never trains and never runs SWMM.  It proves which already-generated
D2 branches belong to frozen Train18 versus Development-Validation, preventing the
4,800-source census from being mistaken for 4,800 IID training rows.
"""
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


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    if "event_id" not in frame.columns:
        raise ValueError("D2 source index lacks event_id")
    if "source_kind" in frame.columns:
        frame = frame[frame["source_kind"].astype(str).str.upper() == "D2"].copy()
    if frame.empty:
        raise ValueError("D2 source index contains no D2 rows")

    identity_column = None
    for candidate in (
        "simulation_identity_sha256",
        "candidate_action_sha256",
        "action_or_sequence_sha256",
        "metadata_path",
    ):
        if candidate in frame.columns:
            identity_column = candidate
            break
    if identity_column is None:
        raise ValueError("D2 source index lacks a branch identity column")
    identity = frame[identity_column].astype(str)
    if (identity == "").any():
        raise ValueError(f"D2 source index has empty {identity_column}")
    if identity.duplicated().any():
        raise ValueError("D2 source index is not a unique authoritative branch census")

    branches = int(len(frame))
    if branches != EXPECTED_SOURCE_BRANCHES:
        raise ValueError(f"D2 source census must be 4800 unique branches, got {branches}")
    events = frame["event_id"].astype(str)
    unknown_events = sorted(set(events) - train_events - validation_events - final_events)
    if unknown_events:
        raise ValueError(f"D2 source census contains events outside frozen split: {unknown_events}")
    final_rows = frame[events.isin(final_events)]
    if not final_rows.empty:
        raise ValueError("D2 source census must not contain untouched Final events")
    train_rows = frame[events.isin(train_events)]
    validation_rows = frame[events.isin(validation_events)]
    if int(len(train_rows)) != EXPECTED_TRAIN_ELIGIBLE_BRANCHES:
        raise ValueError(
            "D2 frozen-Train eligible source view must be 3600 branches; "
            f"got {len(train_rows)}"
        )
    if int(len(validation_rows)) != EXPECTED_SOURCE_BRANCHES - EXPECTED_TRAIN_ELIGIBLE_BRANCHES:
        raise ValueError("D2 source Train/Validation partition does not sum to 4800")

    payload = {
        "contract": CONTRACT,
        "source_index": str(Path(args.index).resolve()),
        "source_index_sha256": _sha(args.index),
        "split_contract": str(Path(args.split_contract).resolve()),
        "split_contract_sha256": _sha(args.split_contract),
        "identity_column": identity_column,
        "unique_authoritative_branches": branches,
        "frozen_train_eligible_branches": int(len(train_rows)),
        "development_validation_branches": int(len(validation_rows)),
        "final_branches": 0,
        "frozen_train_event_count": int(train_rows["event_id"].astype(str).nunique()),
        "development_validation_event_count": int(
            validation_rows["event_id"].astype(str).nunique()
        ),
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
