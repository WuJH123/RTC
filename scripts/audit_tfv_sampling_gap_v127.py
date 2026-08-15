"""Quantify the representational gap between 300-s sampled SWMM flooding rates and exact SWMM TFV.

V127 predicts hydraulics at 300 s, while authoritative exact_node_flood_volume_m3 is
accumulated by SWMM at routing time steps. This read-only audit establishes whether coarse
trapezoidal integration itself preserves the within-group action ordering before model
accuracy is blamed for a low D3 rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127 import _branch_indices, _spearman

V127_TFV_SAMPLING_GAP_CONTRACT = "PROJECT7_V127_SWMM_300S_TO_EXACT_TFV_REPRESENTATION_AUDIT_V1"


def _trapezoid(initial_rate: np.ndarray, future_rate: np.ndarray, dt_seconds: float = 300.0) -> np.ndarray:
    initial = np.maximum(np.asarray(initial_rate, dtype=np.float64), 0.0)
    future = np.maximum(np.asarray(future_rate, dtype=np.float64), 0.0)
    rates = np.concatenate((initial[:, None, :], future), axis=1)
    return (0.5 * (rates[:, :-1] + rates[:, 1:]) * float(dt_seconds)).sum(axis=1)


def _pairwise(truth: np.ndarray, pred: np.ndarray, threshold: float = 0.0) -> float:
    hits = pairs = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            delta = float(truth[i] - truth[j])
            if abs(delta) <= float(threshold):
                continue
            pairs += 1
            hits += int(np.sign(delta) == np.sign(float(pred[i] - pred[j])))
    return float(hits / pairs) if pairs else float("nan")


def _evaluate(cache: V60TrainCache, names: Sequence[str], flood_rate_index: int) -> dict[str, float]:
    rank: list[float] = []
    pair: list[float] = []
    informative: list[float] = []
    mae: list[float] = []
    delta_mae: list[float] = []
    for name in names:
        entry = cache.entry(name)
        indices = _branch_indices(entry)
        arrays = entry.arrays
        exact = np.asarray(arrays["exact_node_flood_volume_m3"][indices], dtype=np.float64).sum(axis=1)
        initial = np.asarray(arrays["initial_state"][indices], dtype=np.float64)[..., int(flood_rate_index)]
        future = np.asarray(arrays["target_states"][indices], dtype=np.float64)[..., int(flood_rate_index)]
        sampled = _trapezoid(initial, future).sum(axis=1)
        if not np.isfinite(exact).all() or not np.isfinite(sampled).all():
            raise RuntimeError(f"{name}: non-finite sampled/exact TFV")
        rank.append(_spearman(sampled, exact))
        pair.append(_pairwise(exact, sampled))
        threshold = max(1.0, 0.001 * abs(float(exact[0])))
        informative.append(_pairwise(exact, sampled, threshold=threshold))
        mae.append(float(np.mean(np.abs(sampled - exact))))
        delta_mae.append(
            float(np.mean(np.abs((sampled[1:] - sampled[0]) - (exact[1:] - exact[0]))))
        )

    def mean(values: Sequence[float]) -> float:
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    return {
        "groups": float(len(names)),
        "sampled_vs_exact_rank": mean(rank),
        "sampled_vs_exact_pairwise": mean(pair),
        "sampled_vs_exact_informative_pairwise": mean(informative),
        "sampled_vs_exact_tfv_mae_m3": mean(mae),
        "sampled_vs_exact_delta_tfv_mae_m3": mean(delta_mae),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--flood-rate-index", type=int, default=2)
    args = p.parse_args()

    cache = V60TrainCache(args.cache_manifest)
    fit, holdout = deterministic_rainfall_split_v60(
        cache,
        names=sorted(cache.names("D2") + cache.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    result = {
        "contract": V127_TFV_SAMPLING_GAP_CONTRACT,
        "trainfit_d2": _evaluate(cache, [x for x in fit if x.startswith("D2::")], args.flood_rate_index),
        "trainfit_d3": _evaluate(cache, [x for x in fit if x.startswith("D3::")], args.flood_rate_index),
        "internal_holdout_d2": _evaluate(cache, [x for x in holdout if x.startswith("D2::")], args.flood_rate_index),
        "internal_holdout_d3": _evaluate(cache, [x for x in holdout if x.startswith("D3::")], args.flood_rate_index),
        "semantics": {
            "sampled": "trapezoidal integration of authoritative SWMM flooding-rate states sampled at 300 s",
            "exact": "authoritative SWMM routing-step cumulative exact_node_flood_volume_m3",
            "purpose": "estimate TFV representation ceiling imposed by the 300-s state interface",
        },
        "boundary": {
            "read_only": True,
            "used_for_training": False,
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
