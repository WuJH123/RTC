from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .baseline_cache_cli import locked_final_contract, validate_final_event_registry
from .baseline_cache_v3 import CACHE_CONTRACT, build_baseline_cache, parse_strategies, write_views
from .causal_timing import timing_from_controller_config
from .inp_runtime import sha256_file


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audited fixed baselines including Auto-RBC and EFD"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", help="resolved controller/runtime config; required for prelock")
    parser.add_argument(
        "--strategies",
        help=(
            "comma-separated baselines; default: "
            "no_control,internal_rtc,auto_rbc,efd,all_open,all_closed"
        ),
    )
    parser.add_argument("--stage", choices=["prelock", "final"], default="prelock")
    parser.add_argument("--policy-lock")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    if args.stage == "final":
        if not args.policy_lock:
            raise ValueError("--policy-lock is required for final baseline generation")
        config, strategies, physical_sha, final_groups = locked_final_contract(args.policy_lock)
        if args.config and sha256_file(args.config) != sha256_file(config):
            raise ValueError("--config differs from the Policy-Locked controller config")
        events = validate_final_event_registry(
            events,
            locked_physical_sha256=physical_sha,
            locked_final_groups=final_groups,
        )
        formalize_final = True
    else:
        if not args.config:
            raise ValueError("--config is required for prelock baseline generation")
        config = Path(args.config)
        timing_from_controller_config(_json(config)).validate(
            require_full_history_before_first_control=True
        )
        strategies = parse_strategies(args.strategies)
        formalize_final = False

    out = Path(args.out_dir)
    frame = build_baseline_cache(
        event_registry=events,
        output_dir=out,
        config_path=config,
        strategies=strategies,
        stage=args.stage,
        workers=args.workers,
        swmm_threads_per_process=args.swmm_threads_per_process,
        force=args.force,
        formalize_final=formalize_final,
    )
    views = write_views(frame, out)
    print(json.dumps({
        "contract": CACHE_CONTRACT,
        "stage": args.stage,
        "rows": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "rainfall_groups": int(frame["rainfall_group"].nunique()),
        "strategies": sorted(frame["strategy"].unique().tolist()),
        "computed": int((frame["status"] == "completed").sum()),
        "resumed": int((frame["status"] == "resumed").sum()),
        **views,
    }, indent=2))


if __name__ == "__main__":
    main()
