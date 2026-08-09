from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from .fresh_workspace import (
    load_fresh_workspace,
    require_path_inside_workspace,
    validate_fresh_run_index,
)
from .step2_shards import load_shard_manifest


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


def _validate_shard_manifest(manifest_path: str, workspace_manifest: str) -> None:
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


def _guard_step1(delegate: Callable[[], None], *, acceptance: bool) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out", required=True)
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
    if acceptance:
        require_path_inside_workspace(known.model, root)
    _strip_option("--workspace-manifest")
    delegate()


def train_step1_main() -> None:
    from .large_model_cli import train_step1_large_main

    _guard_step1(train_step1_large_main, acceptance=False)


def accept_step1_main() -> None:
    from .large_model_cli import accept_step1_large_main

    _guard_step1(accept_step1_large_main, acceptance=True)


def compile_step2_shards_main() -> None:
    from .large_model_cli import compile_step2_shards_main as delegate

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    known, _ = parser.parse_known_args()
    validate_fresh_run_index(
        run_index_path=known.run_index,
        workspace_manifest_path=known.workspace_manifest,
        reject_final=True,
    )
    root = _workspace_root(known.workspace_manifest)
    _require_output_inside(known.out_dir, root)
    _strip_option("--workspace-manifest")
    delegate()


def _guard_step2(delegate: Callable[[], None], *, acceptance: bool) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-manifest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    if acceptance:
        parser.add_argument("--model", required=True)
    known, _ = parser.parse_known_args()
    _validate_shard_manifest(known.manifest, known.workspace_manifest)
    root = _workspace_root(known.workspace_manifest)
    _require_output_inside(known.out, root)
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
        description="Validate that every branch in a run index belongs to the bound fresh workspace"
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
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
