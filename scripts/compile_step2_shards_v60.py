"""Compile the dedicated V6 D2 + targeted-D3 run index into lineage-bound Step2 shards."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from rtc.step2_shards_v60 import compile_step2_shards_v60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--model-step-seconds", type=int, default=300)
    parser.add_argument("--horizon-steps", type=int, default=72)
    args = parser.parse_args()
    frame = pd.read_csv(args.run_index)
    manifest = compile_step2_shards_v60(
        frame,
        output_dir=args.out_dir,
        shard_size=args.shard_size,
        expected_model_step_seconds=args.model_step_seconds,
        expected_horizon_steps=args.horizon_steps,
    )
    print(json.dumps({"contract": "PROJECT7_STEP2_V60_SHARD_COMPILE_V1", "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
