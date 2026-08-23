"""Audit that Formal uses exactly the Step2/V15/V21 lineage that produced V23.

No model is trained and SWMM is never started.  The Practical asset manifest is the runtime authority;
this command verifies that its Step2 is the frozen Direct-TFV V5 checkpoint and that V15/V21 point back
to that same Step2.  It prevents a historical V4/V41/smoke report from being substituted into Formal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from rtc.direct_tfv_policy_return import sha256_file
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path
from rtc.project7_v23_step2_lineage import (
    V23_STEP2_CHECKPOINT_BASENAME,
    V23_STEP2_CHECKPOINT_SHA256,
    V23_STEP2_LINEAGE_EVIDENCE_CONTRACT,
    V23_STEP2_RUN_DIRECTORY,
    V23_V15_CHECKPOINT_SHA256,
    V23_V21_CHECKPOINT_SHA256,
    validate_v23_step2_lineage_evidence,
)


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _sha(value: object) -> str:
    return str(value or "").strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.asset_manifest).resolve()
    manifest = load_practical_rtc_asset_manifest(manifest_path)
    step2_path = Path(practical_asset_path(manifest, "step2")).resolve()
    v15_path = Path(args.v15_rank_checkpoint).resolve()
    v21_path = Path(args.v21_boundary_checkpoint).resolve()
    for path in (step2_path, v15_path, v21_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    step2_sha = sha256_file(step2_path).lower()
    manifest_step2_sha = _sha(manifest["assets"]["step2"]["sha256"])
    v15_sha = sha256_file(v15_path).lower()
    v21_sha = sha256_file(v21_path).lower()
    if step2_sha != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Practical V23 asset manifest resolves to another Step2 checkpoint; "
            f"expected={V23_STEP2_CHECKPOINT_SHA256}, actual={step2_sha}, path={step2_path}"
        )
    if manifest_step2_sha != step2_sha:
        raise RuntimeError("Practical manifest Step2 SHA differs from the actual checkpoint")
    if step2_path.name != V23_STEP2_CHECKPOINT_BASENAME:
        raise RuntimeError(f"V23 Step2 checkpoint basename drifted: {step2_path.name}")
    if V23_STEP2_RUN_DIRECTORY not in step2_path.parts:
        raise RuntimeError(f"V23 Step2 checkpoint is outside the frozen V5 run directory: {step2_path}")
    if v15_sha != V23_V15_CHECKPOINT_SHA256:
        raise RuntimeError(f"wrong V15 checkpoint for V23 Formal: {v15_sha}")
    if v21_sha != V23_V21_CHECKPOINT_SHA256:
        raise RuntimeError(f"wrong V21 checkpoint for V23 Formal: {v21_sha}")

    v15 = _load_torch_mapping(v15_path)
    v21 = _load_torch_mapping(v21_path)
    v15_base = _sha(v15.get("base_step2_sha256"))
    v21_base = _sha(v21.get("base_step2_sha256"))
    v21_rank_source = _sha(v21.get("rank_source_checkpoint_sha256"))
    if v15_base != step2_sha:
        raise RuntimeError("V15/base-Step2 lineage mismatch")
    if v21_base != step2_sha:
        raise RuntimeError("V21/base-Step2 lineage mismatch")
    if v21_rank_source != v15_sha:
        raise RuntimeError("V21/rank-source lineage mismatch")

    payload = {
        "contract": V23_STEP2_LINEAGE_EVIDENCE_CONTRACT,
        "lineage_pass": True,
        "runtime_policy": "V23_STRONG_STORM_HYDRAULIC_CANDIDATE_WITH_V15_RANK_V21_BOUNDARY",
        "asset_manifest_path": str(manifest_path),
        "asset_manifest_sha256": sha256_file(manifest_path),
        "step2_checkpoint_path": str(step2_path),
        "step2_checkpoint_sha256": step2_sha,
        "asset_manifest_step2_sha256": manifest_step2_sha,
        "step2_run_directory": V23_STEP2_RUN_DIRECTORY,
        "step2_checkpoint_basename": step2_path.name,
        "v15_checkpoint_path": str(v15_path),
        "v15_checkpoint_sha256": v15_sha,
        "v15_base_step2_sha256": v15_base,
        "v21_checkpoint_path": str(v21_path),
        "v21_checkpoint_sha256": v21_sha,
        "v21_base_step2_sha256": v21_base,
        "v21_rank_source_checkpoint_sha256": v21_rank_source,
        "step2_retrained_for_formal": False,
        "new_training_data_generated": False,
        "new_policy_return_truth_generated": False,
        "swmm_started": False,
    }
    validate_v23_step2_lineage_evidence(payload)
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
