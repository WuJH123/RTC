"""Merge 1..4 disjoint V12 scenario-mean first-move panel shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from rtc.step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
)


EXPECTED_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_V12_SCENARIO_MEAN_FIRST_MOVE_PANEL_V1"
MERGE_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_V12_SCENARIO_MEAN_PANEL_MERGE_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--summary", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out", required=True)
    args = p.parse_args()
    if len(args.manifest) != len(args.summary) or not 1 <= len(args.manifest) <= 4:
        raise ValueError("V12 merge requires 1..4 manifest/summary pairs")
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summary]
    first = summaries[0]
    for summary in summaries:
        if str(summary.get("contract", "")) != EXPECTED_CONTRACT:
            raise ValueError("V12 shard has the wrong summary contract")
        if str(summary.get("query_step3_contract", "")) != DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT:
            raise ValueError("V12 shard has the wrong Step3 query contract")
        if str(summary.get("rainfall_scenario_contract", "")) != DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT:
            raise ValueError("V12 shard has the wrong rainfall scenario contract")
        for key in (
            "v12_behavioral_source_sha256", "global_rainfall_group_count",
            "rainfall_multipliers", "rainfall_history_steps", "rainfall_decay_per_step", "lineage",
        ):
            if summary.get(key) != first.get(key):
                raise ValueError(f"V12 panel shards differ in lineage/contract field {key}")
    seen: set[str] = set()
    frames = []
    for manifest_path, summary in zip(args.manifest, summaries, strict=True):
        groups = {str(x) for x in summary.get("rainfall_groups", [])}
        overlap = seen & groups
        if overlap:
            raise ValueError(f"V12 panel shards overlap rainfall groups: {sorted(overlap)}")
        seen |= groups
        frame = pd.read_csv(manifest_path)
        if set(frame["rainfall_group"].astype(str)) != groups:
            raise ValueError("V12 manifest/summary rainfall groups differ")
        frames.append(frame)
    if len(seen) < 24:
        raise ValueError("merged V12 panel requires at least 24 rainfall groups")
    expected_global = int(first["global_rainfall_group_count"])
    if len(seen) != expected_global:
        raise ValueError(
            f"V12 shard union incomplete: got {len(seen)}, expected {expected_global}"
        )
    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["rainfall_group", "sequence_index"], kind="mergesort"
    )
    counts = merged.groupby("rainfall_group").size()
    if not bool((counts == 2).all()):
        raise ValueError("merged V12 panel must contain exactly HOLD+candidate per rainfall group")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); merged.to_csv(out, index=False)
    payload = {
        "contract": MERGE_CONTRACT,
        "development_only": True,
        "query_step3_contract": DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        "rainfall_scenario_contract": DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
        "v12_behavioral_source_sha256": first["v12_behavioral_source_sha256"],
        "rainfall_group_count": len(seen),
        "rainfall_groups": sorted(seen),
        "rows": len(merged),
        "merged_manifest_sha256": _sha(out),
        "source_manifest_sha256": [_sha(path) for path in args.manifest],
        "source_summary_sha256": [_sha(path) for path in args.summary],
        "lineage": first["lineage"],
        "rainfall_multipliers": first["rainfall_multipliers"],
        "rainfall_history_steps": first["rainfall_history_steps"],
        "rainfall_decay_per_step": first["rainfall_decay_per_step"],
    }
    summary_out = Path(args.summary_out); summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
