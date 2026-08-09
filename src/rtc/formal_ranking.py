from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .acceptance import apply_metric_thresholds, rank_correlation
from .contracts import load_priority_nodes
from .d2_eval import exact_node_volumes, join_manifest_runs, model_metrics
from .production_cli import _load_graph, _load_step2


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _thresholds(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        {str(k): float(v) for k, v in payload.get("minimum", {}).items()},
        {str(k): float(v) for k, v in payload.get("maximum", {}).items()},
    )


def run_ranking_gate(
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
    merged = join_manifest_runs(pd.read_csv(manifest_path), pd.read_csv(run_summary_path))
    if "scientific_split" in merged.columns:
        merged = merged[merged["scientific_split"].astype(str) == split]
    if split == "development" and "development_fold" in merged.columns:
        merged = merged[merged["development_fold"].astype(str) == development_fold]
    if merged.empty:
        raise ValueError("no held-out D2 branches for ranking gate")

    # The D2 manifest repeats the same center/base physical action once per probed actuator.
    # Ranking must operate on unique physical actions so the base action is not overweighted.
    action_keys = ["checkpoint_id", "candidate_action_sha256"]
    group_keys = ["checkpoint_id"]
    if "event_id" in merged.columns:
        action_keys.insert(0, "event_id")
        group_keys.insert(0, "event_id")
    unique = merged.drop_duplicates(action_keys).copy()

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(graph_path)
    model = _load_step2(step2_path, dev)
    priority = load_priority_nodes(priority_path)
    missing = sorted(set(priority) - set(graph.node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from graph: {missing}")
    pidx = np.asarray([graph.node_ids.index(n) for n in priority], dtype=int)

    exact_cache: dict[str, np.ndarray] = {}
    detail: list[dict[str, object]] = []
    for _, group in unique.groupby(group_keys, sort=False):
        if len(group) < 3:
            continue
        pred_tfv: list[float] = []
        pred_pfv: list[float] = []
        true_tfv: list[float] = []
        true_pfv: list[float] = []
        shas: list[str] = []
        for _, row in group.iterrows():
            path = str(row["metadata_path"])
            predicted_tfv, predicted_pfv, _, _ = model_metrics(
                model=model,
                graph=graph,
                metadata_path=path,
                priority_indices=pidx,
                device=dev,
            )
            if path not in exact_cache:
                exact_cache[path] = exact_node_volumes(path, graph.node_ids)
            exact = exact_cache[path]
            pred_tfv.append(predicted_tfv)
            pred_pfv.append(predicted_pfv)
            true_tfv.append(float(exact.sum()))
            true_pfv.append(float(exact[pidx].sum()))
            shas.append(str(row["candidate_action_sha256"]))
        best_pred = int(np.argmin(pred_tfv))
        best_true = int(np.argmin(true_tfv))
        detail.append({
            "event_id": str(group["event_id"].iloc[0]) if "event_id" in group.columns else "",
            "checkpoint_id": str(group["checkpoint_id"].iloc[0]),
            "candidate_count": len(group),
            "tfv_rank_correlation": rank_correlation(np.asarray(pred_tfv), np.asarray(true_tfv)),
            "pfv_rank_correlation": rank_correlation(np.asarray(pred_pfv), np.asarray(true_pfv)),
            "tfv_top1_hit": float(best_pred == best_true),
            "tfv_selected_regret_m3": float(true_tfv[best_pred] - true_tfv[best_true]),
            "predicted_best_action_sha256": shas[best_pred],
            "true_best_action_sha256": shas[best_true],
        })
    frame = pd.DataFrame(detail)
    if frame.empty:
        raise ValueError("no held-out checkpoint has at least three unique D2 actions")
    metrics = {
        "tfv_rank_correlation": float(frame["tfv_rank_correlation"].mean()),
        "pfv_rank_correlation": float(frame["pfv_rank_correlation"].mean()),
        "tfv_top1_hit_rate": float(frame["tfv_top1_hit"].mean()),
        "tfv_selected_regret_m3": float(frame["tfv_selected_regret_m3"].mean()),
        "ranking_checkpoints": float(len(frame)),
    }
    minimum, maximum = _thresholds(thresholds_path)
    result = apply_metric_thresholds(metrics, minimum=minimum, maximum=maximum)
    detail_path = Path(output_path).with_suffix(".detail.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(detail_path, index=False)
    payload: dict[str, object] = {
        "contract": "D2_SWMM_CANDIDATE_RANKING_ACCEPTANCE_V3_TRAPEZOID",
        "passed": result.passed,
        "failed_metrics": list(result.failed_metrics),
        "metrics": result.metrics,
        "thresholds": {"minimum": minimum, "maximum": maximum},
        "unique_action_dedup": action_keys,
        "split": split,
        "development_fold": development_fold if split == "development" else "",
        "step2_sha256": _sha(step2_path),
        "manifest_sha256": _sha(manifest_path),
        "run_summary_sha256": _sha(run_summary_path),
        "prediction_volume_integration": "trapezoid_current_plus_future_flooding_rate",
        "truth_source_tfv_pfv": "SWMM_NODE_STATISTICS_CUMULATIVE_EXACT_HORIZON",
        "detail_csv": str(detail_path),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out exact-SWMM candidate-ranking gate")
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
    payload = run_ranking_gate(
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
