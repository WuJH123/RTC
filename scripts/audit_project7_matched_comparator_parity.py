"""Zero-SWMM parity audit for Project7 Proposed vs matched active baselines.

The audit reads existing metadata/decision JSONL only. It does not execute SWMM, train a model, alter
evidence, or infer missing provenance. A matched baseline is fair-comparator eligible only when its
sparse sensor set, Step1, source event, controller config, 82/27 authority, q95 support contract,
0.5 target slew and target-latch semantics are explicitly aligned with Proposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rtc.project7_matched_baselines import MATCHED_ACTIVE_BASELINES, MATCHED_BASELINE_CONTRACT

AUDIT_CONTRACT = "PROJECT7_MATCHED_COMPARATOR_PARITY_AUDIT_V1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be an object: {path}")
    return value


def _sensor_nodes(metadata: dict[str, Any]) -> tuple[str, ...]:
    value = metadata.get("sensor_nodes")
    if not isinstance(value, list) or not value:
        raise ValueError("metadata lacks explicit sensor_nodes")
    return tuple(str(item) for item in value)


def _decision_summary(metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    name = metadata.get("decision_file")
    if not name:
        raise ValueError(f"metadata lacks decision_file: {metadata_path}")
    path = metadata_path.parent / str(name)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    maximum_k = 0
    maximum_target_delta = 0.0
    q95_non_q95 = 0
    passive_failures = 0
    action_rows = 0
    for row in rows:
        diagnostics = row.get("diagnostics") if isinstance(row, dict) else None
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        k = int(diagnostics.get("first_move_changed_facility_count", diagnostics.get("active_facility_count", 0)) or 0)
        maximum_k = max(maximum_k, k)
        action_rows += int(k > 0)
        delta = diagnostics.get("target_change_max")
        if delta is not None:
            maximum_target_delta = max(maximum_target_delta, float(delta))
        quantile = str(diagnostics.get("joint_sequence_support_quantile", "")).strip().lower()
        if k > 0 and quantile and quantile != "q95":
            q95_non_q95 += 1
        if diagnostics.get("passive_channels_unchanged") is False:
            passive_failures += 1
    return {
        "decision_path": str(path.resolve()),
        "decision_sha256": _sha(path),
        "decision_rows": len(rows),
        "action_rows": action_rows,
        "maximum_changed_facilities": maximum_k,
        "maximum_target_change": maximum_target_delta,
        "non_q95_action_rows": q95_non_q95,
        "passive_channel_failure_rows": passive_failures,
    }


def _compare(proposed_path: Path, matched_path: Path) -> dict[str, Any]:
    proposed = _metadata(proposed_path)
    matched = _metadata(matched_path)
    failures: list[str] = []
    strategy = str(matched.get("strategy", ""))
    if strategy not in MATCHED_ACTIVE_BASELINES:
        failures.append(f"strategy {strategy!r} is not a matched active baseline")
    if str(matched.get("matched_baseline_contract", "")) != MATCHED_BASELINE_CONTRACT:
        failures.append("matched baseline contract mismatch")
    for key in (
        "same_sparse_sensor_set_as_proposed",
        "same_frozen_step1_reconstruction_as_proposed",
        "same_rainfall_forecast_as_proposed",
        "same_82_channel_supervisory_mask",
        "same_passive_27_channels",
        "same_q95_changed_facility_ceiling",
        "same_q95_joint_sequence_support",
        "same_target_latch_semantics",
    ):
        if matched.get(key) is not True:
            failures.append(f"{key} is not explicitly true")
    if abs(float(matched.get("same_max_setting_delta_per_update", -1.0)) - 0.5) > 1.0e-9:
        failures.append("matched baseline does not declare 0.5 target slew")
    if _sensor_nodes(proposed) != _sensor_nodes(matched):
        failures.append("sensor_nodes differ from Proposed")
    for key in ("source_inp_sha256", "controller_config_sha256"):
        if not proposed.get(key) or not matched.get(key) or proposed.get(key) != matched.get(key):
            failures.append(f"{key} mismatch or missing")
    if proposed.get("step1_model_sha256") and matched.get("step1_model_sha256"):
        if proposed["step1_model_sha256"] != matched["step1_model_sha256"]:
            failures.append("Step1 checkpoint mismatch")
    elif proposed.get("asset_manifest_sha256") and matched.get("asset_manifest_sha256"):
        if proposed["asset_manifest_sha256"] != matched["asset_manifest_sha256"]:
            failures.append("asset manifest mismatch; Step1 identity cannot be inherited")
    else:
        failures.append("cannot prove common Step1 identity from metadata")

    decisions = _decision_summary(matched_path, matched)
    if decisions["maximum_changed_facilities"] > 20:
        failures.append("matched baseline exceeded q95 K=20")
    if decisions["maximum_target_change"] > 0.5000001:
        failures.append("matched baseline exceeded 0.5 target slew")
    if decisions["non_q95_action_rows"]:
        failures.append("matched baseline executed non-q95 action rows")
    if decisions["passive_channel_failure_rows"]:
        failures.append("matched baseline changed passive channels")
    return {
        "strategy": strategy,
        "matched_metadata": str(matched_path.resolve()),
        "matched_metadata_sha256": _sha(matched_path),
        "decision_summary": decisions,
        "failures": failures,
        "passed": not failures,
        "fair_comparator_claim_eligible": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposed-metadata", required=True)
    parser.add_argument("--matched-metadata", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    proposed_path = Path(args.proposed_metadata).resolve()
    matched_paths = [Path(value).resolve() for value in args.matched_metadata]
    results = [_compare(proposed_path, path) for path in matched_paths]
    payload = {
        "contract": AUDIT_CONTRACT,
        "proposed_metadata": str(proposed_path),
        "proposed_metadata_sha256": _sha(proposed_path),
        "matched_results": results,
        "passed": bool(results) and all(row["passed"] for row in results),
        "native_internal_rtc_role": "EXTERNAL_OPERATIONAL_REFERENCE_NOT_FAIR_MATCHED_COMPARATOR",
        "no_control_role": "PASSIVE_REFERENCE_NO_INFORMATION_OR_ACTION_AUTHORITY",
        "new_swmm_runs": 0,
        "training_performed": False,
        "historical_evidence_mutated": False,
        "development_only": True,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
