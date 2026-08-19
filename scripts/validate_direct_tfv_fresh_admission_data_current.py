"""Fail-closed preflight for fresh Direct-TFV admission-calibration Development data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.direct_tfv_admission import DIRECT_TFV_ADMISSION_COVERAGE
from rtc.direct_tfv_admission_split import (
    DEFAULT_RESERVED_POSTCALIBRATION_EVENT_TOKENS,
    optimizer_replay_event_ids,
    validate_fresh_admission_partition,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


CURRENT_FRESH_ADMISSION_PREFLIGHT_CONTRACT = "PROJECT7_CURRENT_FRESH_ADMISSION_PREFLIGHT_V1"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-cache-manifest", required=True)
    p.add_argument("--fresh-calibration-cache-manifest", required=True)
    p.add_argument("--optimizer-replay-report", required=True)
    p.add_argument("--coverage", type=float, default=DIRECT_TFV_ADMISSION_COVERAGE)
    p.add_argument("--reserved-event-id", action="append", default=[])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = V60TrainCache(args.base_cache_manifest)
    fresh = V60TrainCache(args.fresh_calibration_cache_manifest)
    replay = json.loads(Path(args.optimizer_replay_report).read_text(encoding="utf-8"))
    if not isinstance(replay, dict):
        raise ValueError("optimizer replay report must be a JSON object")

    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d3 = sorted(name for name in fit if name.startswith("D3::"))
    audit_d3 = sorted(name for name in holdout if name.startswith("D3::"))
    fresh_d3 = sorted(fresh.targeted_d3_names())
    reserved = tuple(
        dict.fromkeys(
            (*DEFAULT_RESERVED_POSTCALIBRATION_EVENT_TOKENS, *(str(value) for value in args.reserved_event_id))
        )
    )
    partition = validate_fresh_admission_partition(
        base_cache=base,
        step2_trainfit_d3_names=fit_d3,
        original_audit_d3_names=audit_d3,
        fresh_cache=fresh,
        fresh_calibration_d3_names=fresh_d3,
        coverage=float(args.coverage),
        optimizer_replay_events=sorted(optimizer_replay_event_ids(replay)),
        reserved_postcalibration_event_tokens=reserved,
    )
    payload = {
        "contract": CURRENT_FRESH_ADMISSION_PREFLIGHT_CONTRACT,
        **partition,
        "lineage": {
            "base_cache_sha256": _sha(args.base_cache_manifest),
            "fresh_calibration_cache_sha256": _sha(args.fresh_calibration_cache_manifest),
            "optimizer_replay_report_sha256": _sha(args.optimizer_replay_report),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
