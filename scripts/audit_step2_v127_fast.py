"""Fused V127 ranking + H30-H360 hydraulic audit for the same frozen checkpoint.

The old pipeline ran two independent H360 scans over the same D2/D3/D4 branches. This
entrypoint evaluates each branch once and derives both evidence families from that rollout.
It is read-only and does not alter any model parameter or scientific split.
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
from rtc.step2_eval_v127_fast import V127_FAST_EVAL_CONTRACT, evaluate_groups_fast_v127
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60

V127_FAST_AUDIT_CONTRACT = "PROJECT7_V127_FAST_FUSED_DEVELOPMENT_EVIDENCE_V1"


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
    p.add_argument("--ranking-out", required=True)
    p.add_argument("--horizon-out", required=True)
    p.add_argument("--telemetry-out", required=True)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--branch-chunk", type=int, default=12)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.branch_chunk <= 0:
        raise ValueError("V127 fast audit branch chunk must be positive")
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
    specs = {
        "trainfit_d2": (base_online, [x for x in fit if x.startswith("D2::")]),
        "trainfit_d3": (base_online, [x for x in fit if x.startswith("D3::")]),
        "internal_holdout_d2": (base_online, [x for x in holdout if x.startswith("D2::")]),
        "internal_holdout_d3": (base_online, [x for x in holdout if x.startswith("D3::")]),
        "d4_fit": (d4_fit, d4_fit_raw.names(D4_SOURCE_KIND)),
        "d4_audit": (d4_audit, d4_audit_raw.names(D4_SOURCE_KIND)),
    }
    ranking: dict[str, object] = {
        "contract": "PROJECT7_V127_CONTROL_ORIENTED_RANKING_AUDIT_V2_FAST_FUSED",
        "step2_sha256": _sha(args.step2),
    }
    horizon: dict[str, object] = {
        "contract": "PROJECT7_V127_HYDRAULIC_ROLLOUT_HORIZON_AUDIT_V2_FAST_FUSED",
        "step2_sha256": _sha(args.step2),
        "results": {},
        "metric_semantics": {
            "hydraulic_truth": "authoritative SWMM target state/managed-flow trajectories sampled at 300 s",
            "initial_input": "causal Step1 reconstruction",
            "purpose": "diagnose autoregressive error growth; not a replacement for exact SWMM TFV",
        },
    }
    telemetry: dict[str, object] = {
        "contract": V127_FAST_AUDIT_CONTRACT,
        "execution_contract": V127_FAST_EVAL_CONTRACT,
        "step2_sha256": _sha(args.step2),
        "splits": {},
    }
    for key, (cache, names) in specs.items():
        r, h, t = evaluate_groups_fast_v127(
            model,
            cache=cache,
            names=names,
            normalization=normalization,
            graph=graph,
            device=device,
            depth_index=int(args.depth_index),
            flood_rate_index=int(args.flood_rate_index),
            requested_branch_chunk=int(args.branch_chunk),
        )
        ranking[key] = r
        horizon["results"][key] = h
        telemetry["splits"][key] = t

    boundary = {
        "read_only": True,
        "internal_holdout_used_for_training": False,
        "d4_audit_used_for_training": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    ranking["lineage"] = {
        "graph_sha256": _sha(args.graph),
        "cache_manifest_sha256": _sha(args.cache_manifest),
        "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
        "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
        "causal_rainfall_sha256": _sha(args.causal_store),
        "causal_state_store_sha256": _sha(args.causal_state_store),
    }
    ranking["boundary"] = boundary
    horizon["boundary"] = boundary
    telemetry["boundary"] = boundary

    for path_text, payload_out in (
        (args.ranking_out, ranking),
        (args.horizon_out, horizon),
        (args.telemetry_out, telemetry),
    ):
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload_out, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(telemetry, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
