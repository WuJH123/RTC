from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .d2_eval import exact_node_volumes, join_manifest_runs, model_metrics
from .gradient_truth import compare_gradient_vectors
from .production_cli import _load_graph, _load_step2


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_gradient_truth(
    *,
    manifest_path: str | Path,
    run_summary_path: str | Path,
    graph_path: str | Path,
    step2_path: str | Path,
    split: str = "development",
    development_fold: str = "validation",
    device_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare Step2 autograd with authoritative SWMM finite differences.

    Interior settings use a central finite difference. At 0/1 bounds the exact design only
    has a feasible one-sided perturbation, so forward/backward differences are valid Formal
    truth instead of silently discarding those actuators (especially pumps that are OFF in
    No-control prefixes).
    """

    merged = join_manifest_runs(pd.read_csv(manifest_path), pd.read_csv(run_summary_path))
    if "scientific_split" in merged.columns:
        merged = merged[merged["scientific_split"].astype(str) == split]
    if split == "development" and "development_fold" in merged.columns:
        merged = merged[merged["development_fold"].astype(str) == development_fold]
    if merged.empty:
        raise ValueError("no held-out D2 branches remain after split filtering")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(graph_path)
    model = _load_step2(step2_path, device)
    actuator_index = {aid: i for i, aid in enumerate(graph.actuator_ids)}
    exact_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    group_cols = ["checkpoint_id", "actuator_id"]
    if "event_id" in merged.columns:
        group_cols.insert(0, "event_id")

    for _, group in merged.groupby(group_cols, sort=False):
        group = group.copy()
        group["requested_setting"] = group["requested_setting"].astype(float)
        base_setting = float(group["base_setting"].iloc[0])
        center = group[np.isclose(group["requested_setting"], base_setting)]
        if center.empty:
            continue
        mid = center.iloc[0]
        below = group[group["requested_setting"] < base_setting].sort_values("requested_setting")
        above = group[group["requested_setting"] > base_setting].sort_values("requested_setting")

        def exact(row: pd.Series) -> np.ndarray:
            path = str(row["metadata_path"])
            if path not in exact_cache:
                exact_cache[path] = exact_node_volumes(path, graph.node_ids)
            return exact_cache[path]

        method: str
        if not below.empty and not above.empty:
            lo, hi = below.iloc[-1], above.iloc[0]
            du = float(hi["requested_setting"] - lo["requested_setting"])
            true_grad = float((exact(hi).sum() - exact(lo).sum()) / du)
            method = "central"
        elif not above.empty:
            hi = above.iloc[0]
            du = float(hi["requested_setting"] - base_setting)
            true_grad = float((exact(hi).sum() - exact(mid).sum()) / du)
            method = "forward_bound"
        elif not below.empty:
            lo = below.iloc[-1]
            du = float(base_setting - lo["requested_setting"])
            true_grad = float((exact(mid).sum() - exact(lo).sum()) / du)
            method = "backward_bound"
        else:
            continue
        if du <= 1e-12:
            continue
        aid = str(mid["actuator_id"])
        if aid not in actuator_index:
            raise ValueError(f"D2 actuator absent from frozen graph: {aid}")
        _tfv, _pfv, pred_grad, _pred_pfv_grad = model_metrics(
            model=model,
            graph=graph,
            metadata_path=str(mid["metadata_path"]),
            priority_indices=np.asarray([], dtype=int),
            device=device,
            gradient_actuator_index=actuator_index[aid],
        )
        if pred_grad is None:
            raise RuntimeError("autograd helper did not return a TFV gradient")
        rows.append({
            "event_id": str(mid.get("event_id", "")),
            "checkpoint_id": str(mid["checkpoint_id"]),
            "actuator_id": aid,
            "base_setting": base_setting,
            "finite_difference_method": method,
            "pred_tfv_gradient_m3_per_setting": float(pred_grad),
            "true_tfv_gradient_m3_per_setting": true_grad,
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("no held-out D2 finite-difference cases were available")
    agreement = compare_gradient_vectors(
        detail["pred_tfv_gradient_m3_per_setting"],
        detail["true_tfv_gradient_m3_per_setting"],
    )
    counts = detail["finite_difference_method"].value_counts().to_dict()
    metrics = {
        "tfv_gradient_sign_accuracy": agreement.sign_accuracy,
        "tfv_gradient_cosine_similarity": agreement.cosine_similarity,
        "tfv_gradient_mae": agreement.magnitude_mae,
        "gradient_cases": float(len(detail)),
        "gradient_central_cases": float(counts.get("central", 0)),
        "gradient_forward_bound_cases": float(counts.get("forward_bound", 0)),
        "gradient_backward_bound_cases": float(counts.get("backward_bound", 0)),
    }
    return detail, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out SWMM TFV gradient truth with bound-aware finite differences")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--development-fold", default="validation")
    parser.add_argument("--device")
    args = parser.parse_args()
    detail, metrics = build_gradient_truth(
        manifest_path=args.manifest,
        run_summary_path=args.run_summary,
        graph_path=args.graph,
        step2_path=args.step2,
        split=args.split,
        development_fold=args.development_fold,
        device_name=args.device,
    )
    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    payload = {
        "contract": "D2_SWMM_TFV_GRADIENT_METRICS_V3_BOUND_AWARE",
        "metrics": metrics,
        "step2_sha256": _sha(args.step2),
        "manifest_sha256": _sha(args.manifest),
        "run_summary_sha256": _sha(args.run_summary),
        "detail_csv": str(Path(args.detail_out)),
        "prediction_volume_integration": "trapezoid_current_plus_future_flooding_rate",
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
