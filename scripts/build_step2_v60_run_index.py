"""Build the dedicated V6 D2 + targeted D3-v2 run index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rtc.step2_data_index_v60 import build_step2_v60_run_index, v60_run_index_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2-manifest", required=True)
    parser.add_argument("--d2-run-summary", required=True)
    parser.add_argument("--d3-v60-manifest", required=True)
    parser.add_argument("--d3-v60-run-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = build_step2_v60_run_index(
        d2_manifest=pd.read_csv(args.d2_manifest),
        d2_run_summary=pd.read_csv(args.d2_run_summary),
        d3_v60_manifest=pd.read_csv(args.d3_v60_manifest),
        d3_v60_run_summary=pd.read_csv(args.d3_v60_run_summary),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary = v60_run_index_summary(frame)
    summary["out"] = str(out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
