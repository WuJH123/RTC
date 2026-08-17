"""Calibrate optimizer-aware Direct-TFV Step3 admission from Development evidence.

The final margin takes the conservative maximum of a rainfall-group-balanced D3 HOLD residual bound
and exact same-prefix H360 SWMM residuals from optimizer-selected plans. The optimizer replay event(s)
are calibration evidence and cannot later be described as independent post-calibration validation.
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
    split_d3_holdout_for_admission,
)
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


CURRENT_DIRECT_TFV_ADMISSION_RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_ADMISSION_CALIBRATION_V3"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--optimizer-replay-report", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--coverage", type=float, default=DIRECT_TFV_ADMISSION_COVERAGE)
    args = p.parse_args()

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
    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)

    _, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    hold_d3 = sorted(name for name in holdout if name.startswith("D3::"))
    if len(hold_d3) != 32:
        raise ValueError(f"current Direct-TFV admission expects 32 D3 holdout groups, got {len(hold_d3)}")
    calibration_names, audit_names, split = split_d3_holdout_for_admission(
        base,
        hold_d3,
        coverage=float(args.coverage),
    )
    calibration = derive_direct_tfv_admission_calibration(
        model,
        cache=online,
        names=calibration_names,
        normalization=normalization,
        graph=graph,
        device=device,
        action_support=checkpoint["action_support"],
        optimizer_replay_report=replay,
        coverage=float(args.coverage),
    )
    audit = evaluate_direct_tfv_admission(
        model,
        cache=online,
        names=audit_names,
        normalization=normalization,
        graph=graph,
        device=device,
        calibration=calibration,
    )
    payload = {
        **calibration,
        "run_contract": CURRENT_DIRECT_TFV_ADMISSION_RUN_CONTRACT,
        "split": split,
        "d3_calibration_names": calibration_names,
        "audit_names": audit_names,
        "audit": audit,
        "lineage": {
            "step2_checkpoint_sha256": _sha(args.checkpoint),
            "optimizer_replay_report_sha256": _sha(args.optimizer_replay_report),
            "graph_sha256": _sha(args.graph),
            "base_cache_sha256": _sha(args.cache_manifest),
            "causal_rainfall_sha256": _sha(args.causal_store),
            "causal_state_store_sha256": _sha(args.causal_state_store),
            "step2_training_contract": str(checkpoint.get("training_contract", "")),
        },
        "online_swmm_called": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
