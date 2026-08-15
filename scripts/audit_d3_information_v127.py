"""Train-only D3 action-support and hard-TFV information audit (no training/SWMM)."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairwise(values: np.ndarray, threshold: float = 0.0) -> float:
    n = len(values); total = correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(values[i] - values[j])) <= threshold:
                continue
            total += 1
            correct += int(values[i] != values[j])
    return float(correct / total) if total else float("nan")


def _group(cache: V60TrainCache, name: str) -> dict[str, object]:
    e = cache.entry(name); ref = int(e.reference_index)
    branches = [int(i) for i in e.indices]
    tfv_all = np.asarray(e.arrays["exact_node_flood_volume_m3"][branches], dtype=np.float64).sum(axis=1)
    ref_tfv = float(tfv_all[branches.index(ref)])
    cand_pos = [i for i, b in enumerate(branches) if b != ref]
    cand = tfv_all[cand_pos]
    delta = cand - ref_tfv
    threshold = max(1.0, 0.001 * abs(ref_tfv))
    # D3 executable first move is the mean of the two 300-s settings.
    settings = np.asarray(e.arrays["settings"][branches], dtype=np.float64)
    first = settings[:, :2].mean(axis=1)
    seq = np.asarray(e.arrays.get("action_or_sequence_sha256", np.arange(len(branches))), dtype=object)
    return {
        "name": name, "event_id": e.event_id, "rainfall_group": e.rainfall_group, "checkpoint_id": e.checkpoint_id,
        "branch_count": len(branches), "unique_action_sequence_count": int(len({str(seq[i]) for i in cand_pos})),
        "unique_first_move_count": int(len({first[i].tobytes() for i in cand_pos})),
        "reference_tfv_m3": ref_tfv, "tfv_min_m3": float(np.min(tfv_all)), "tfv_max_m3": float(np.max(tfv_all)),
        "tfv_range_m3": float(np.ptp(tfv_all)), "tfv_std_m3": float(np.std(tfv_all)),
        "delta_tfv_min_m3": float(np.min(delta)), "delta_tfv_median_m3": float(np.median(delta)),
        "delta_tfv_p75_abs_m3": float(np.quantile(np.abs(delta), .75)), "delta_tfv_p90_abs_m3": float(np.quantile(np.abs(delta), .90)),
        "delta_tfv_max_abs_m3": float(np.max(np.abs(delta))), "near_tie_fraction": float(np.mean(np.abs(delta) <= threshold)),
        "exact_pair_count": int(sum(abs(float(cand[i]-cand[j])) <= 1e-9 for i in range(len(cand)) for j in range(i+1,len(cand)))),
        "informative_pair_count": int(sum(abs(float(cand[i]-cand[j])) > threshold for i in range(len(cand)) for j in range(i+1,len(cand)))),
        "exact_pairwise": _pairwise(cand), "informative_pairwise": _pairwise(cand, threshold),
        "informative_threshold_m3": threshold,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-manifest", required=True); p.add_argument("--out", required=True)
    args = p.parse_args()
    cache = V60TrainCache(args.cache_manifest)
    fit, hold = deterministic_rainfall_split_v60(cache, names=sorted(cache.names("D2") + cache.targeted_d3_names()), holdout_fraction=.20)
    names = sorted(n for n in fit if n.startswith("D3::"))
    groups = [_group(cache, n) for n in names]
    out = {"contract": "PROJECT7_V127_D3_INFORMATION_CONTENT_AUDIT_TRAIN_ONLY_V1", "cache_sha256": _sha(args.cache_manifest), "groups": groups,
           "group_count": len(groups), "branch_count": int(sum(g["branch_count"] for g in groups)), "outcome_holdout_accessed": False,
           "informative_pair_threshold": "max(1.0 m3, 0.001 * reference_TFV_m3)", "validation_accessed": False, "final_accessed": False, "formal_accessed": False}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("group_count", "branch_count", "outcome_holdout_accessed")}, indent=2))


if __name__ == "__main__": main()
