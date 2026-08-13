"""Fail-closed entrypoint for Project7 Step2 V8.0 development."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_v70_contract import V70_CONTRACT
from run_step2_v80 import main as run_v80_main


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(path: str | Path, kind: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint payload is not a dictionary")
    if payload.get("contract") != V70_CONTRACT or payload.get("kind") != kind:
        raise ValueError(f"{path}: V7 checkpoint contract/kind mismatch")
    return payload


def _require_equal(label: str, values: list[str]) -> None:
    cleaned = [str(value) for value in values]
    if len(set(cleaned)) != 1:
        raise ValueError(f"V8 lineage mismatch for {label}: {cleaned}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--v70-report", required=True)
    known, _ = parser.parse_known_args()

    cache_lineage = validate_v60_cache_lineage(known.cache_manifest)
    value = _checkpoint(known.v70_value_checkpoint, "control_value")
    hydraulic = _checkpoint(known.v70_hydraulic_checkpoint, "hydraulic_response")
    report = json.loads(Path(known.v70_report).read_text(encoding="utf-8"))
    if report.get("contract") != V70_CONTRACT:
        raise ValueError("V8 requires the canonical V7 development report")
    if bool(report.get("catastrophic_value_collapse", True)):
        raise ValueError("V8 refuses a collapsed V7 Value lineage")

    graph_sha = _sha256(known.graph)
    cache_sha = _sha256(known.cache_manifest)
    value_lineage = value.get("lineage", {})
    hydraulic_lineage = hydraulic.get("lineage", {})
    report_lineage = report.get("lineage", {})
    _require_equal(
        "graph_sha256",
        [graph_sha, value_lineage.get("graph_sha256", ""), hydraulic_lineage.get("graph_sha256", ""), report_lineage.get("graph_sha256", "")],
    )
    _require_equal(
        "cache_manifest_sha256",
        [cache_sha, value_lineage.get("cache_manifest_sha256", ""), hydraulic_lineage.get("cache_manifest_sha256", ""), report_lineage.get("cache_manifest_sha256", "")],
    )
    _require_equal(
        "basis_sha256",
        [
            cache_lineage["v60_control_basis_sha256"],
            value_lineage.get("basis_sha256_from_cache_lineage", ""),
            hydraulic_lineage.get("basis_sha256_from_cache_lineage", ""),
            report_lineage.get("basis_sha256_from_cache_lineage", ""),
        ],
    )
    _require_equal(
        "design_sha256",
        [
            cache_lineage["v60_design_contract_sha256"],
            value_lineage.get("design_sha256_from_cache_lineage", ""),
            hydraulic_lineage.get("design_sha256_from_cache_lineage", ""),
            report_lineage.get("design_sha256_from_cache_lineage", ""),
        ],
    )
    _require_equal(
        "split_manifest_sha256",
        [value.get("split_manifest_sha256", ""), hydraulic.get("split_manifest_sha256", "")],
    )

    print(
        json.dumps(
            {
                "V8_LINEAGE_PREFLIGHT": {
                    "status": "PASS",
                    "graph_sha256": graph_sha,
                    "cache_manifest_sha256": cache_sha,
                    "basis_sha256": cache_lineage["v60_control_basis_sha256"],
                    "design_sha256": cache_lineage["v60_design_contract_sha256"],
                    "v70_value_checkpoint_sha256": _sha256(known.v70_value_checkpoint),
                    "v70_hydraulic_checkpoint_sha256": _sha256(known.v70_hydraulic_checkpoint),
                    "v70_report_sha256": _sha256(known.v70_report),
                }
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    run_v80_main()


if __name__ == "__main__":
    main()
