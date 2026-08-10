from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .simulation_assets import (
    DIAGNOSTIC_ONLY,
    VALID_REUSABLE,
    SimulationAssetRegistry,
    index_d2_metadata_paths,
)


def _metadata_paths_from_summary(path: str | Path) -> list[str]:
    frame = pd.read_csv(path)
    if "metadata_path" not in frame.columns:
        raise ValueError(f"run summary lacks metadata_path: {path}")
    values = [str(value).strip() for value in frame["metadata_path"] if str(value).strip()]
    return list(dict.fromkeys(values))


def index_existing_d2_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Index already-generated exact-prefix D2 branches into the local simulation asset "
            "registry without copying large trajectory files"
        )
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--run-summary", action="append", required=True)
    parser.add_argument(
        "--qualification",
        choices=[VALID_REUSABLE, DIAGNOSTIC_ONLY],
        default=VALID_REUSABLE,
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths: list[str] = []
    for summary in args.run_summary:
        paths.extend(_metadata_paths_from_summary(summary))
    paths = list(dict.fromkeys(paths))
    result = index_d2_metadata_paths(
        args.asset_root,
        paths,
        qualification=args.qualification,
        qualification_reason=(
            "existing branch explicitly indexed after exact-prefix/artifact audit"
        ),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed"]:
        raise SystemExit(2)


def asset_audit_main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the local simulation asset registry and referenced large files"
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    registry = SimulationAssetRegistry(args.asset_root)
    payload = registry.audit()
    rows = pd.DataFrame(registry.rows())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows.to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["metadata_missing_or_changed"]:
        raise SystemExit(2)
