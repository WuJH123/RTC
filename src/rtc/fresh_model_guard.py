from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from .code_contract import rtc_source_tree_sha256
from .fresh_workspace import load_fresh_workspace, validate_fresh_run_index
from .step2_shards import compile_step2_shards, load_shard_manifest


def _strip_option(name: str) -> None:
    while name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 >= len(sys.argv):
            raise ValueError(f"{name} requires a value")
        del sys.argv[idx : idx + 2]


def _stamp_metrics(path: str | Path) -> None:
    out = Path(path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("acceptance metrics must be a JSON object")
    payload["rtc_source_tree_sha256"] = rtc_source_tree_sha256()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(out)


def _validate_shard_manifest(
    manifest_path: str, workspace_manifest: str
) -> dict[str, object]:
    load_fresh_workspace(workspace_manifest)
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = load_shard_manifest(manifest_file)
    for item in manifest["shards"]:
        shard = Path(str(item["path"])).expanduser()
        if not shard.is_absolute():
            shard = manifest_file.parent / shard
        if not shard.resolve().is_file():
            raise ValueError(f"Step2 shard is missing: {shard.resolve()}")
    return manifest


def _guard_step1(delegate: Callable[[], None], *, acceptance: bool) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume-state")
    if acceptance:
        parser.add_argument("--model", required=True)
    known, _ = parser.parse_known_args()
    validate_fresh_run_index(
        run_index_path=known.run_index,
        workspace_manifest_path=known.workspace_manifest,
        reject_final=True,
    )
    if acceptance and not Path(known.model).is_file():
        raise ValueError("Step1 model is missing: " + str(known.model))
    _strip_option("--workspace-manifest")
    delegate()
    if acceptance:
        _stamp_metrics(known.out)


def train_step1_main() -> None:
    from .step1_train_v2 import train_step1_large_v2_main

    _guard_step1(train_step1_large_v2_main, acceptance=False)


def accept_step1_main() -> None:
    from .step1_accept_v3 import accept_step1_large_v3_main

    _guard_step1(accept_step1_large_v3_main, acceptance=True)


def compile_step2_shards_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile lineage-valid D2/D3 branches into one frozen-time, "
            "counterfactual-group-preserving Step2 shard set"
        )
    )
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument(
        "--development-fold", choices=["train", "validation", "all"], default="train"
    )
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--horizon-steps", type=int, required=True)
    args = parser.parse_args()
    validate_fresh_run_index(
        run_index_path=args.run_index,
        workspace_manifest_path=args.workspace_manifest,
        reject_final=True,
    )
    frame = pd.read_csv(args.run_index)
    if "scientific_split" not in frame.columns:
        raise ValueError("Step2 run index requires scientific_split")
    frame = frame[frame["scientific_split"].astype(str) == args.split].copy()
    if args.split == "development" and args.development_fold != "all":
        frame = frame[
            frame["development_fold"].astype(str) == args.development_fold
        ].copy()
    if frame.empty:
        raise ValueError("no Step2 branches remain after split/fold filtering")
    if "source_kind" not in frame.columns:
        raise ValueError(
            "action-sensitive Step2 compile requires source_kind to distinguish D2 and D3"
        )
    manifest = compile_step2_shards(
        frame,
        output_dir=args.out_dir,
        shard_size=args.shard_size,
        expected_model_step_seconds=args.model_step_seconds,
        expected_horizon_steps=args.horizon_steps,
    )
    print(
        json.dumps(
            {
                "contract": "STEP2_SHARD_COMPILE_COUNTERFACTUAL_GROUPED_V2",
                "manifest": str(manifest),
                "branches": len(frame),
                "model_step_seconds": args.model_step_seconds,
                "horizon_steps": args.horizon_steps,
                "counterfactual_groups_preserved": True,
            },
            indent=2,
        )
    )


def _guard_step2(delegate: Callable[[], None], *, acceptance: bool) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume-state")
    if acceptance:
        parser.add_argument("--model", required=True)
    known, _ = parser.parse_known_args()
    _validate_shard_manifest(known.manifest, known.workspace_manifest)
    if acceptance and not Path(known.model).is_file():
        raise ValueError(f"Step2 model is missing: {known.model}")
    _strip_option("--workspace-manifest")
    delegate()
    if acceptance:
        _stamp_metrics(known.out)


def train_step2_main() -> None:
    from .step2_train_v3 import train_step2_large_v3_main

    _guard_step2(train_step2_large_v3_main, acceptance=False)


def accept_step2_main() -> None:
    from .step2_accept_v2 import accept_step2_large_v2_main

    _guard_step2(accept_step2_large_v2_main, acceptance=True)


def validate_index_main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate run-index scientific lineage against the study workspace contract"
    )
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    payload = validate_fresh_run_index(
        run_index_path=args.run_index,
        workspace_manifest_path=args.workspace_manifest,
        reject_final=True,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2))
