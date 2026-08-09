from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .baseline_cache import (
    CACHE_CONTRACT,
    _parse_strategies,
    _write_views,
    build_baseline_cache,
)
from .baselines import FIXED_BASELINE_IDS
from .causal_timing import timing_from_controller_config
from .code_contract import rtc_source_tree_sha256
from .inp_lineage import physical_contract_sha256
from .inp_runtime import sha256_file


POLICY_LOCK_CONTRACT = "WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND"


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _locked_artifact(
    *, artefacts: dict[str, object], hashes: dict[str, object], name: str
) -> Path:
    path = Path(str(artefacts.get(name, "")))
    if not path.is_file():
        raise ValueError(f"Policy-Locked artifact is missing: {name}: {path}")
    expected = str(hashes.get(name, ""))
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"Policy-Locked artifact changed: {name}: {path}")
    return path


def locked_final_contract(
    policy_lock_path: str | Path,
) -> tuple[Path, tuple[str, ...], str, frozenset[str]]:
    """Resolve immutable code/runtime/baseline/split/physical Final contracts."""

    lock = _json(policy_lock_path)
    if lock.get("contract") != POLICY_LOCK_CONTRACT:
        raise ValueError("Final baseline generation requires code/time/data-bound Policy Lock V4")
    if lock.get("rtc_source_tree_sha256") != rtc_source_tree_sha256():
        raise ValueError(
            "current RTC source tree differs from Policy Lock; Final generation is forbidden"
        )
    artefacts_raw = lock.get("artefacts")
    hashes_raw = lock.get("sha256")
    if not isinstance(artefacts_raw, dict) or not isinstance(hashes_raw, dict):
        raise ValueError("Policy Lock lacks artefact/hash maps")
    artefacts = {str(k): v for k, v in artefacts_raw.items()}
    hashes = {str(k): v for k, v in hashes_raw.items()}

    config = _locked_artifact(
        artefacts=artefacts, hashes=hashes, name="controller_config"
    )
    plan_path = _locked_artifact(
        artefacts=artefacts, hashes=hashes, name="baseline_plan"
    )
    split_path = _locked_artifact(
        artefacts=artefacts, hashes=hashes, name="split_registry"
    )
    timing_from_controller_config(_json(config)).validate(
        require_full_history_before_first_control=True
    )

    plan = _json(plan_path)
    strategies = tuple(
        str(x) for x in plan.get("strategies", []) if str(x) != "proposed"
    )
    if strategies != FIXED_BASELINE_IDS:
        raise ValueError(
            "locked fixed-baseline matrix must be exactly "
            f"{list(FIXED_BASELINE_IDS)}; got {list(strategies)}"
        )

    split = pd.read_csv(split_path)
    required = {"rainfall_group", "scientific_split"}
    if not required.issubset(split.columns):
        raise ValueError("locked split registry lacks rainfall_group/scientific_split")
    split = split.copy()
    split["rainfall_group"] = split["rainfall_group"].astype(str)
    split["scientific_split"] = split["scientific_split"].astype(str)
    cross = split.groupby("rainfall_group")["scientific_split"].nunique()
    if (cross != 1).any():
        raise ValueError("locked split registry contains rainfall-group leakage")
    final_groups = frozenset(
        split.loc[split["scientific_split"] == "final", "rainfall_group"].tolist()
    )
    if len(final_groups) < 24:
        raise ValueError("locked Final requires at least 24 independent rainfall groups")

    physical_sha = str(lock.get("physical_network_sha256", ""))
    if len(physical_sha) != 64:
        raise ValueError("Policy Lock lacks a valid frozen physical-network SHA-256")
    return config, strategies, physical_sha, final_groups


def validate_final_event_registry(
    event_registry: pd.DataFrame,
    *,
    locked_physical_sha256: str,
    locked_final_groups: frozenset[str],
) -> pd.DataFrame:
    """Fail before expensive SWMM if Final events differ from the locked design."""

    required = {"event_id", "rainfall_group", "inp_path", "scientific_split"}
    missing = sorted(required - set(event_registry.columns))
    if missing:
        raise ValueError(f"event registry missing columns: {missing}")
    frame = event_registry.copy()
    frame["scientific_split"] = frame["scientific_split"].astype(str)
    frame["rainfall_group"] = frame["rainfall_group"].astype(str)
    final = frame[frame["scientific_split"] == "final"].copy()
    if final.empty:
        raise ValueError("event registry contains no Final rows")
    present_groups = frozenset(final["rainfall_group"].tolist())
    if present_groups != locked_final_groups:
        raise ValueError(
            "event-registry Final rainfall groups differ from Policy Lock: "
            f"missing={sorted(locked_final_groups-present_groups)}, "
            f"extra={sorted(present_groups-locked_final_groups)}"
        )
    if final["event_id"].astype(str).duplicated().any():
        raise ValueError("Final event registry contains duplicate event_id values")
    for _, row in final.iterrows():
        source = Path(str(row["inp_path"]))
        if not source.is_file():
            raise ValueError(f"Final event INP missing: {source}")
        actual = physical_contract_sha256(source)
        if actual != locked_physical_sha256:
            raise ValueError(
                f"Final event {row['event_id']} physical network differs from Policy Lock"
            )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate every fixed baseline once per rainfall event and reuse code-bound evidence"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--config", help="resolved controller/runtime config; required for prelock"
    )
    parser.add_argument(
        "--strategies",
        help="comma-separated Formal fixed baselines; default is no_control,internal_rtc,all_open,all_closed",
    )
    parser.add_argument("--stage", choices=["prelock", "final"], default="prelock")
    parser.add_argument(
        "--policy-lock",
        help="required for final; supplies locked code/config/baseline/split/physical contract",
    )
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1)
    )
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    if args.stage == "final":
        if not args.policy_lock:
            raise ValueError("--policy-lock is required for final baseline generation")
        config, strategies, physical_sha, final_groups = locked_final_contract(
            args.policy_lock
        )
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
        strategies = _parse_strategies(args.strategies)
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
    views = _write_views(frame, out)
    print(
        json.dumps(
            {
                "contract": CACHE_CONTRACT,
                "stage": args.stage,
                "rows": int(len(frame)),
                "events": int(frame["event_id"].nunique()),
                "strategies": sorted(frame["strategy"].unique().tolist()),
                "computed": int((frame["status"] == "completed").sum()),
                "resumed": int((frame["status"] == "resumed").sum()),
                "workers": min(
                    args.workers,
                    max(1, int((frame["status"] == "completed").sum())),
                ),
                **views,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
