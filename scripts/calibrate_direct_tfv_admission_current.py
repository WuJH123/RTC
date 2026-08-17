"""Calibrate optimizer-aware Direct-TFV Step3 admission from disjoint Development evidence.

The accepted Step2 V5 TrainFit rainfall groups are model-fitting data and are never reused for
split-conformal calibration.  A dedicated fresh Development D3-HOLD cache supplies the one-sided
rainfall-group residual bound, while every original Step2 D3 holdout group remains an independent
cached admission audit.  Exact optimizer-selected H360 replay still contributes only its empirical
maximum optimism residual and carries no formal coverage claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_admission import (
    DIRECT_TFV_ADMISSION_COVERAGE,
    derive_direct_tfv_admission_calibration,
    evaluate_direct_tfv_admission,
)
from rtc.direct_tfv_admission_split import (
    DEFAULT_RESERVED_POSTCALIBRATION_EVENT_TOKENS,
    DIRECT_TFV_FRESH_ADMISSION_SOURCE,
    optimizer_replay_event_ids,
    validate_fresh_admission_partition,
)
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


CURRENT_DIRECT_TFV_ADMISSION_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_ADMISSION_CALIBRATION_V4"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--optimizer-replay-report", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True, help="accepted Step2 V5 base cache")
    p.add_argument("--causal-store", required=True, help="accepted Step2 causal rainfall store")
    p.add_argument("--causal-state-store", required=True, help="accepted Step2 causal state store")
    p.add_argument(
        "--fresh-calibration-cache-manifest",
        required=True,
        help="fresh Development-only D3-HOLD cache never used for Step2 fitting",
    )
    p.add_argument(
        "--fresh-calibration-causal-store",
        required=True,
        help="causal rainfall store covering the fresh calibration groups",
    )
    p.add_argument(
        "--fresh-calibration-causal-state-store",
        required=True,
        help="causal Step1-state store covering the fresh calibration groups",
    )
    p.add_argument(
        "--reserved-event-id",
        action="append",
        default=[],
        help=(
            "event token forbidden from fresh calibration; repeatable. T10_D180/T20_D300 are "
            "always reserved in the current Project7 Development contract"
        ),
    )
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--coverage", type=float, default=DIRECT_TFV_ADMISSION_COVERAGE)
    return p


def main() -> None:
    args = _parser().parse_args()
    coverage = float(args.coverage)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.checkpoint, graph=graph, device=device
    )
    replay = json.loads(Path(args.optimizer_replay_report).read_text(encoding="utf-8"))
    if not isinstance(replay, dict):
        raise ValueError("optimizer replay report must be a JSON object")

    base = V60TrainCache(args.cache_manifest)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)

    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d3 = sorted(name for name in fit if name.startswith("D3::"))
    audit_names = sorted(name for name in holdout if name.startswith("D3::"))
    if len(fit_d3) != 112 or len(audit_names) != 32:
        raise ValueError(
            "current Direct-TFV admission expects the accepted Step2 split "
            f"112 D3 TrainFit / 32 D3 holdout; got {len(fit_d3)} / {len(audit_names)}"
        )

    fresh = V60TrainCache(args.fresh_calibration_cache_manifest)
    fresh_names = sorted(fresh.targeted_d3_names())
    if not fresh_names:
        raise ValueError("fresh calibration cache contains no targeted D3 HOLD-reference groups")
    fresh_rain = load_causal_forecast_store_v123(args.fresh_calibration_causal_store)
    fresh_state = load_causal_state_store_v127(args.fresh_calibration_causal_state_store)
    fresh_online = CausalStep1StateCacheV127(
        CausalForecastValueCacheV123(fresh, fresh_rain), fresh_state
    )

    reserved = tuple(
        dict.fromkeys(
            (*DEFAULT_RESERVED_POSTCALIBRATION_EVENT_TOKENS, *(str(value) for value in args.reserved_event_id))
        )
    )
    partition = validate_fresh_admission_partition(
        base_cache=base,
        step2_trainfit_d3_names=fit_d3,
        original_audit_d3_names=audit_names,
        fresh_cache=fresh,
        fresh_calibration_d3_names=fresh_names,
        coverage=coverage,
        optimizer_replay_events=sorted(optimizer_replay_event_ids(replay)),
        reserved_postcalibration_event_tokens=reserved,
    )

    calibration = derive_direct_tfv_admission_calibration(
        model,
        cache=fresh_online,
        names=fresh_names,
        normalization=normalization,
        graph=graph,
        device=device,
        action_support=checkpoint["action_support"],
        optimizer_replay_report=replay,
        coverage=coverage,
    )
    audit = evaluate_direct_tfv_admission(
        model,
        cache=base_online,
        names=audit_names,
        normalization=normalization,
        graph=graph,
        device=device,
        calibration=calibration,
    )
    payload = {
        **calibration,
        "run_contract": CURRENT_DIRECT_TFV_ADMISSION_RUN_CONTRACT,
        "fresh_admission_data_contract": partition["contract"],
        "calibration_source_semantics": DIRECT_TFV_FRESH_ADMISSION_SOURCE,
        "partition": partition,
        "d3_calibration_names": fresh_names,
        "audit_names": audit_names,
        "audit": audit,
        "lineage": {
            "step2_checkpoint_sha256": _sha(args.checkpoint),
            "optimizer_replay_report_sha256": _sha(args.optimizer_replay_report),
            "graph_sha256": _sha(args.graph),
            "base_cache_sha256": _sha(args.cache_manifest),
            "causal_rainfall_sha256": _sha(args.causal_store),
            "causal_state_store_sha256": _sha(args.causal_state_store),
            "fresh_calibration_cache_sha256": _sha(args.fresh_calibration_cache_manifest),
            "fresh_calibration_causal_rainfall_sha256": _sha(args.fresh_calibration_causal_store),
            "fresh_calibration_causal_state_store_sha256": _sha(
                args.fresh_calibration_causal_state_store
            ),
            "step2_training_contract": str(checkpoint.get("training_contract", "")),
        },
        "online_swmm_called": False,
        "coverage_claim_scope": (
            "90% one-sided finite-sample claim applies only to fresh rainfall-group calibration "
            "residual maxima; optimizer replay remains empirical-maximum evidence only"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
