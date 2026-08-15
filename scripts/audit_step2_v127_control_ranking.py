"""Evaluate a final V127 checkpoint on frozen development ranking splits.

Reports legacy hard-TFV metrics plus smooth optimization ranking, informative pairwise
accuracy and delta-TFV errors. Holdout/AUDIT outcomes are read-only and never train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.checkpoint_v127 import input_normalization_from_v127_checkpoint, load_step2_v127
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127_control import V127ControlTrainingDesign, evaluate_objective_groups_control_v127

V127_CONTROL_RANKING_AUDIT_CONTRACT = (
    "PROJECT7_V127_CONTROL_ORIENTED_RANKING_AUDIT_V1_CHECKPOINT_BOUND"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--evaluation-branch-chunk", type=int, default=8)
    args = p.parse_args()

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    graph = _load_graph(args.graph)
    model, payload = load_step2_v127(args.step2, graph=graph, device=device)
    normalization = input_normalization_from_v127_checkpoint(payload)
    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain = load_causal_forecast_store_v123(args.causal_store)
    state = load_causal_state_store_v127(args.causal_state_store)
    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain), state)
    d4_fit = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rain), state)
    d4_audit = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rain), state)

    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    hold_d3 = [name for name in holdout if name.startswith("D3::")]
    fit_d4 = d4_fit_raw.names(D4_SOURCE_KIND)
    audit_d4 = d4_audit_raw.names(D4_SOURCE_KIND)
    design = V127ControlTrainingDesign(evaluation_branch_chunk=int(args.evaluation_branch_chunk))

    def evaluate(cache, names):
        return evaluate_objective_groups_control_v127(
            model,
            cache=cache,
            names=names,
            normalization=normalization,
            graph=graph,
            device=device,
            flood_rate_index=int(args.flood_rate_index),
            design=design,
        )

    result = {
        "contract": V127_CONTROL_RANKING_AUDIT_CONTRACT,
        "step2_sha256": _sha(args.step2),
        "trainfit_d2": evaluate(base_online, fit_d2),
        "trainfit_d3": evaluate(base_online, fit_d3),
        "internal_holdout_d2": evaluate(base_online, hold_d2),
        "internal_holdout_d3": evaluate(base_online, hold_d3),
        "d4_fit": evaluate(d4_fit, fit_d4),
        "d4_audit": evaluate(d4_audit, audit_d4),
        "lineage": {
            "graph_sha256": _sha(args.graph),
            "cache_manifest_sha256": _sha(args.cache_manifest),
            "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
            "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
            "causal_rainfall_sha256": _sha(args.causal_store),
            "causal_state_store_sha256": _sha(args.causal_state_store),
        },
        "boundary": {
            "internal_holdout_used_for_training": False,
            "d4_audit_used_for_training": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
