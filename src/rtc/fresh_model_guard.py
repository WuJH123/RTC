from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from .fresh_workspace import (
    load_fresh_workspace,
    require_path_inside_workspace,
    validate_fresh_run_index,
)
from .step2_shards import compile_step2_shards, load_shard_manifest


def _strip_option(name: str) -> None:
    while name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 >= len(sys.argv):
            raise ValueError(f"{name} requires a value")
        del sys.argv[idx : idx + 2]


def _workspace_root(manifest_path: str) -> Path:
    workspace = load_fresh_workspace(manifest_path)
    return Path(str(workspace["output_root"])).resolve()


def _require_output_inside(path: str, root: Path) -> None:
    require_path_inside_workspace(Path(path).resolve(), root)


def _validate_shard_manifest(manifest_path: str, workspace_manifest: str) -> dict[str, object]:
    root = _workspace_root(workspace_manifest)
    manifest_file = Path(manifest_path).expanduser().resolve()
    require_path_inside_workspace(manifest_file, root)
    manifest = load_shard_manifest(manifest_file)
    for item in manifest["shards"]:
        shard = Path(str(item["path"])).expanduser()
        if not shard.is_absolute():
            shard = manifest_file.parent / shard
        shard = shard.resolve()
        require_path_inside_workspace(shard, root)
        if not shard.is_file():
            raise ValueError(f"fresh Step2 shard is missing: {shard}")
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
    root = _workspace_root(known.workspace_manifest)
    _require_output_inside(known.out, root)
    if known.resume_state:
        _require_output_inside(known.resume_state, root)
    if acceptance:
        require_path_inside_workspace(known.model, root)
    _strip_option("--workspace-manifest")
    delegate()


def train_step1_main() -> None:
    from .step1_train_v2 import train_step1_large_v2_main

    _guard_step1(train_step1_large_v2_main, acceptance=False)


def accept_step1_main() -> None:
    from .step1_accept_v3 import accept_step1_large_v3_main

    _guard_step1(accept_step1_large_v3_main, acceptance=True)


def compile_step2_shards_main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile current-code D2/D3 branches into one frozen-time Step2 shard set"
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
    root = _workspace_root(args.workspace_manifest)
    _require_output_inside(args.out_dir, root)
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
                "contract": "STEP2_SHARD_COMPILE_TIME_LOCKED_V1",
                "manifest": str(manifest),
                "branches": len(frame),
                "model_step_seconds": args.model_step_seconds,
                "horizon_steps": args.horizon_steps,
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
    root = _workspace_root(known.workspace_manifest)
    _require_output_inside(known.out, root)
    if known.resume_state:
        _require_output_inside(known.resume_state, root)
    if acceptance:
        require_path_inside_workspace(known.model, root)
    _strip_option("--workspace-manifest")
    delegate()


def train_step2_main() -> None:
    from .step2_train_v2 import train_step2_large_v2_main

    _guard_step2(train_step2_large_v2_main, acceptance=False)


def accept_step2_main() -> None:
    from .step2_accept_v2 import accept_step2_large_v2_main

    _guard_step2(accept_step2_large_v2_main, acceptance=True)


def validate_index_main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate every branch in a run index against current Fresh Workspace code"
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
        root = _workspace_root(args.workspace_manifest)
        require_path_inside_workspace(out.resolve(), root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2))
