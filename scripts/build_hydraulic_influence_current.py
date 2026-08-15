"""Build a frozen sparse hydraulic-influence shortcut artifact from Development D2 only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rtc.hydraulic_influence_v128 import HYDRAULIC_INFLUENCE_CONTRACT, build_hydraulic_influence_edges_v128, influence_edges_to_arrays
from rtc.production_cli import _load_graph
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--min-hops", type=int, default=7)
    p.add_argument("--min-effect-m3", type=float, default=1.0)
    p.add_argument("--rate-effect-floor-m3s", type=float, default=1.0e-4)
    p.add_argument("--top-k-per-actuator", type=int, default=8)
    args = p.parse_args()

    graph = _load_graph(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    fit, _ = deterministic_rainfall_split_v60(
        cache,
        names=sorted(cache.names("D2") + cache.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    if len(fit_d2) != 112:
        raise ValueError("hydraulic influence builder requires canonical 112 Development TrainFit D2 groups")
    edges = build_hydraulic_influence_edges_v128(
        cache,
        fit_d2,
        graph=graph,
        min_graph_hops=int(args.min_hops),
        min_mean_abs_effect_m3=float(args.min_effect_m3),
        flood_rate_effect_floor_m3s=float(args.rate_effect_floor_m3s),
        top_k_per_actuator=int(args.top_k_per_actuator),
    )
    arrays = influence_edges_to_arrays(edges)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        contract=np.asarray(HYDRAULIC_INFLUENCE_CONTRACT),
        scientific_split=np.asarray("development"),
        graph_sha256=np.asarray(_sha(args.graph)),
        cache_manifest_sha256=np.asarray(_sha(args.cache_manifest)),
        **arrays,
    )
    payload = {
        "contract": HYDRAULIC_INFLUENCE_CONTRACT,
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "full_model_enabled": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "graph_sha256": _sha(args.graph),
        "cache_manifest_sha256": _sha(args.cache_manifest),
        "source_group_count": len(fit_d2),
        "shortcut_edges": len(edges),
        "thresholds": {
            "min_graph_hops": int(args.min_hops),
            "min_mean_abs_effect_m3": float(args.min_effect_m3),
            "flood_rate_effect_floor_m3s": float(args.rate_effect_floor_m3s),
            "top_k_per_actuator": int(args.top_k_per_actuator),
        },
        "npz": str(out_npz.resolve()),
        "promotion_rule": (
            "This artifact is a Development-only candidate influence graph. It may be used in "
            "smoke/dev architecture ablations only; promote/freeze before full only if held-out "
            "spatial action-effect and ranking evidence improves."
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
