from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .contracts import load_priority_nodes
from .final_eval import compile_closed_loop_run_index, event_balanced_summary, paired_strategy_comparison
from .pipeline import sha256_file


def _load_verified_policy_lock(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != "WUHAN_RTC_POLICY_LOCK_V1":
        raise ValueError("not a WUHAN_RTC_POLICY_LOCK_V1 file")
    artefacts = payload.get("artefacts")
    hashes = payload.get("sha256")
    if not isinstance(artefacts, dict) or not isinstance(hashes, dict):
        raise ValueError("policy lock is missing artefact/hash maps")
    for name, raw_path in artefacts.items():
        artifact = Path(str(raw_path))
        if not artifact.is_file():
            raise RuntimeError(f"locked artefact disappeared before Final: {name}: {artifact}")
        current = sha256_file(artifact)
        expected = str(hashes.get(name, ""))
        if current != expected:
            raise RuntimeError(f"locked artefact changed before Final: {name}: {artifact}")
    return payload


def _validate_final_matrix(index: pd.DataFrame, strategies: list[str]) -> None:
    required = {"event_id", "rainfall_group", "strategy", "metadata_path"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Final run index missing columns: {missing}")
    expected = set(strategies)
    if "proposed" not in expected:
        raise ValueError("baseline plan must include proposed")
    for event, group in index.groupby("event_id", sort=False):
        present = set(group["strategy"].astype(str))
        if present != expected:
            raise ValueError(
                f"incomplete/extra Final strategy matrix for {event}: "
                f"missing={sorted(expected-present)}, extra={sorted(present-expected)}"
            )
        if group["rainfall_group"].astype(str).nunique() != 1:
            raise ValueError(f"event {event} maps to multiple rainfall groups")


def compile_final_main() -> None:
    parser = argparse.ArgumentParser(description="Compile untouched policy-locked Final SWMM evidence")
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--pairwise-dir")
    args = parser.parse_args()

    lock = _load_verified_policy_lock(args.policy_lock)
    artefacts = lock["artefacts"]
    if not isinstance(artefacts, dict):
        raise ValueError("invalid policy-lock artefact map")
    for required_name in ("split_registry", "baseline_plan", "priority_nodes"):
        if required_name not in artefacts:
            raise ValueError(f"policy lock is missing required Final artefact: {required_name}")

    split_registry = pd.read_csv(str(artefacts["split_registry"]))
    if not {"rainfall_group", "scientific_split"}.issubset(split_registry.columns):
        raise ValueError("locked split_registry requires rainfall_group and scientific_split")
    group_role = (
        split_registry[["rainfall_group", "scientific_split"]]
        .drop_duplicates()
        .assign(rainfall_group=lambda x: x["rainfall_group"].astype(str))
        .set_index("rainfall_group")["scientific_split"]
        .astype(str)
        .to_dict()
    )
    plan = json.loads(Path(str(artefacts["baseline_plan"])).read_text(encoding="utf-8"))
    strategies = [str(x) for x in plan.get("strategies", [])]
    if not strategies:
        raise ValueError("locked baseline_plan has no strategies")

    index = pd.read_csv(args.run_index)
    _validate_final_matrix(index, strategies)
    run_groups = set(index["rainfall_group"].astype(str))
    wrong = sorted(group for group in run_groups if group_role.get(group) != "final")
    if wrong:
        raise ValueError(f"Final run index contains non-final/unknown rainfall groups: {wrong[:20]}")

    detail = compile_closed_loop_run_index(
        index,
        priority_nodes=load_priority_nodes(str(artefacts["priority_nodes"])),
    )
    summary = event_balanced_summary(detail)
    detail_path = Path(args.detail_out)
    summary_path = Path(args.summary_out)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    pairwise_outputs: dict[str, str] = {}
    if args.pairwise_dir:
        pair_dir = Path(args.pairwise_dir)
        pair_dir.mkdir(parents=True, exist_ok=True)
        for reference in strategies:
            if reference == "proposed":
                continue
            paired = paired_strategy_comparison(detail, proposed="proposed", reference=reference)
            path = pair_dir / f"proposed_vs_{reference}.csv"
            paired.to_csv(path, index=False)
            pairwise_outputs[reference] = str(path)

    print(json.dumps({
        "policy_sha256": lock["policy_sha256"],
        "final_events": int(index["event_id"].nunique()),
        "strategies": strategies,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "pairwise": pairwise_outputs,
    }, indent=2))
