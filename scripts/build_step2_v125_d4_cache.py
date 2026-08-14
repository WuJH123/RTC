"""Build an isolated V125 D4 FIT or AUDIT cache from authoritative SWMM outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rtc.step2_d4_cache_v125 import D4_CACHE_CONTRACT_V125, build_d4_run_index_v125
from rtc.step2_shards import compile_step2_shards, sha256_file
from rtc.step2_training_cache import build_step2_training_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--execution-manifest", required=True)
    p.add_argument("--run-summary", required=True)
    p.add_argument("--split-role", required=True, choices=("fit", "audit"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-size", type=int, default=128)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    execution = pd.read_csv(args.execution_manifest)
    runs = pd.read_csv(args.run_summary)
    index = build_d4_run_index_v125(execution, runs, split_role=args.split_role)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / f"STEP2_V125_D4_{args.split_role.upper()}_RUN_INDEX.csv"
    index.to_csv(index_path, index=False)
    shard_manifest = compile_step2_shards(
        index,
        output_dir=out / "shards",
        shard_size=int(args.shard_size),
        expected_model_step_seconds=300,
        expected_horizon_steps=72,
    )
    cache_manifest = build_step2_training_cache(
        shard_manifest,
        out / "training_cache",
        force=bool(args.force),
    )
    rain = sorted(index["rainfall_group"].astype(str).unique().tolist())
    payload = {
        "contract": D4_CACHE_CONTRACT_V125,
        "split_role": args.split_role,
        "rows": int(len(index)),
        "groups": int(index["checkpoint_id"].nunique()),
        "rainfall_groups": rain,
        "execution_manifest_sha256": sha256_file(args.execution_manifest),
        "run_summary_sha256": sha256_file(args.run_summary),
        "run_index": str(index_path.resolve()),
        "shard_manifest": str(shard_manifest.resolve()),
        "cache_manifest": str(cache_manifest.resolve()),
        "audit_used_for_training": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    report = out / f"STEP2_V125_D4_{args.split_role.upper()}_CACHE.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
