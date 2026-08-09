from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .acceptance import apply_metric_thresholds
from .contracts import load_priority_nodes
from .gradient_truth import compare_gradient_vectors
from .production_cli import _load_graph, _load_step2
from .validation_cli import _exact_node_volumes, _join_manifest_runs, _predict_metrics_and_gradients


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _thresholds(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        {str(k): float(v) for k, v in payload.get("minimum", {}).items()},
        {str(k): float(v) for k, v in payload.get("maximum", {}).items()},
    )


def run_gradient_gate(
    *,
    manifest_path: str | Path,
    run_summary_path: str | Path,
    graph_path: str | Path,
    step2_path: str | Path,
    priority_path: str | Path,
    thresholds_path: str | Path,
    output_path: str | Path,
    split: str = "development",
    development_fold: str = "validation",
    device: str | None = None,
) -> dict[str, object]:
    merged = _join_manifest_runs(pd.read_csv(manifest_path), pd.read_csv(run_summary_path))
    if "scientific_split" in merged.columns:
        merged = merged[merged["scientific_split"].astype(str) == split]
    if split == "development" and "development_fold" in merged.columns:
        merged = merged[merged["development_fold"].astype(str) == development_fold]
    if merged.empty:
        raise ValueError("no held-out D2 branches for gradient gate")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(graph_path)
    model = _load_step2(step2_path, dev)
    priority = load_priority_nodes(priority_path)
    missing = sorted(set(priority) - set(graph.node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from graph: {missing}")
    pidx = np.asarray([graph.node_ids.index(n) for n in priority], dtype=int)
    actuator_index = {aid: i for i, aid in enumerate(graph.actuator_ids)}

    exact_cache: dict[str, np.ndarray] = {}
    pred_cache: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    detail: list[dict[str, object]] = []
    group_cols = ["checkpoint_id", "actuator_id"]
    if "event_id" in merged.columns:
        group_cols.insert(0, "event_id")
    for _, group in merged.groupby(group_cols, sort=False):
        group = group.sort_values("requested_setting")
        base = float(group["base_setting"].iloc[0])
        below = group[group["requested_setting"].astype(float) < base]
        center = group[np.isclose(group["requested_setting"].astype(float), base)]
        above = group[group["requested_setting"].astype(float) > base]
        if below.empty or center.empty or above.empty:
            continue
        lo, mid, hi = below.iloc[-1], center.iloc[0], above.iloc[0]
        aid = str(mid["actuator_id"])
        if aid not in actuator_index:
            raise ValueError(f"D2 actuator absent from graph: {aid}")

        def exact(row: pd.Series) -> np.ndarray:
            path = str(row["metadata_path"])
            if path not in exact_cache:
                exact_cache[path] = _exact_node_volumes(path, graph.node_ids)
            return exact_cache[path]

        lo_v, hi_v = exact(lo), exact(hi)
        du = float(hi["requested_setting"]) - float(lo["requested_setting"])
        if abs(du) <= 1e-12:
            continue
        true_tfv = float((hi_v.sum() - lo_v.sum()) / du)
        true_pfv = float((hi_v[pidx].sum() - lo_v[pidx].sum()) / du)
        center_path = str(mid["metadata_path"])
        cache_key = (center_path, aid)
        if cache_key not in pred_cache:
            pred_cache[cache_key] = _predict_metrics_and_gradients(
                model=model,
                graph=graph,
                metadata_path=center_path,
                priority_indices=pidx,
                actuator_index=actuator_index[aid],
                device=dev,
            )
        _, _, pred_tfv, pred_pfv = pred_cache[cache_key]
        detail.append(
            {
                "event_id": str(mid.get("event_id", "")),
                "checkpoint_id": str(mid["checkpoint_id"]),
                "actuator_id": aid,
                "pred_tfv_gradient": pred_tfv,
                "true_tfv_gradient": true_tfv,
                "pred_pfv_gradient": pred_pfv,
                "true_pfv_gradient": true_pfv,
            }
        )
    frame = pd.DataFrame(detail)
    if frame.empty:
        raise ValueError("no complete lower/center/upper D2 triplets for gradient gate")
    tfv = compare_gradient_vectors(frame["pred_tfv_gradient"], frame["true_tfv_gradient"])
    pfv = compare_gradient_vectors(frame["pred_pfv_gradient"], frame["true_pfv_gradient"])
    metrics = {
        "tfv_gradient_sign_accuracy": tfv.sign_accuracy,
        "tfv_gradient_cosine_similarity": tfv.cosine_similarity,
        "tfv_gradient_mae": tfv.magnitude_mae,
        "pfv_gradient_sign_accuracy": pfv.sign_accuracy,
        "pfv_gradient_cosine_similarity": pfv.cosine_similarity,
        "pfv_gradient_mae": pfv.magnitude_mae,
        "gradient_cases": float(len(frame)),
    }
    minimum, maximum = _thresholds(thresholds_path)
    result = apply_metric_thresholds(metrics, minimum=minimum, maximum=maximum)
    detail_path = Path(output_path).with_suffix(".detail.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(detail_path, index=False)
    payload: dict[str, object] = {
        "contract": "D2_SWMM_GRADIENT_TRUTH_ACCEPTANCE_V2",
        "passed": result.passed,
        "failed_metrics": list(result.failed_metrics),
        "metrics": result.metrics,
        "thresholds": {"minimum": minimum, "maximum": maximum},
        "split": split,
        "development_fold": development_fold if split == "development" else "",
        "step2_sha256": _sha(step2_path),
        "manifest_sha256": _sha(manifest_path),
        "run_summary_sha256": _sha(run_summary_path),
        "detail_csv": str(detail_path),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out authoritative SWMM finite-difference gradient gate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--development-fold", default="validation")
    parser.add_argument("--device")
    args = parser.parse_args()
    payload = run_gradient_gate(
        manifest_path=args.manifest,
        run_summary_path=args.run_summary,
        graph_path=args.graph,
        step2_path=args.step2,
        priority_path=args.priority,
        thresholds_path=args.thresholds,
        output_path=args.out,
        split=args.split,
        development_fold=args.development_fold,
        device=args.device,
    )
    print(json.dumps(payload, indent=2))
    if payload["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
