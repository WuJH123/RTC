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
    raw = payload.get("candidate_ranking", payload)
    if not isinstance(raw, dict):
        raise ValueError("ranking threshold payload must be an object")
    return (
        {str(k): float(v) for k, v in raw.get("minimum", {}).items()},
        {str(k): float(v) for k, v in raw.get("maximum", {}).items()},
    )


def _heldout(frame: pd.DataFrame, *, split: str, development_fold: str) -> pd.DataFrame:
    if "scientific_split" not in frame.columns:
        raise ValueError("ranking evidence requires scientific_split")
    out = frame[frame["scientific_split"].astype(str) == split].copy()
    if split == "development":
        if "development_fold" not in out.columns:
            raise ValueError("development ranking evidence requires development_fold")
        out = out[out["development_fold"].astype(str) == development_fold].copy()
    if out.empty:
        raise ValueError("no held-out branches remain after ranking split filtering")
    return out


def _score_groups(
    frame: pd.DataFrame,
    *,
    action_sha_column: str,
    source_kind: str,
    model,
    graph,
    pidx: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    group_keys = ["checkpoint_id"]
    if "event_id" in frame.columns:
        group_keys.insert(0, "event_id")
    exact_cache: dict[str, np.ndarray] = {}
    detail: list[dict[str, object]] = []
    for _, group in frame.groupby(group_keys, sort=False):
        group = group.drop_duplicates(action_sha_column).copy()
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
                device=device,
            )
            if path not in exact_cache:
                exact_cache[path] = exact_node_volumes(path, graph.node_ids)
            exact = exact_cache[path]
            pred_tfv.append(predicted_tfv)
            pred_pfv.append(predicted_pfv)
            true_tfv.append(float(exact.sum()))
            true_pfv.append(float(exact[pidx].sum()))
            shas.append(str(row[action_sha_column]))
        best_pred = int(np.argmin(pred_tfv))
        best_true = int(np.argmin(true_tfv))
        detail.append(
            {
                "source_kind": source_kind,
                "event_id": str(group["event_id"].iloc[0]) if "event_id" in group.columns else "",
                "checkpoint_id": str(group["checkpoint_id"].iloc[0]),
                "candidate_count": len(group),
                "tfv_rank_correlation": rank_correlation(
                    np.asarray(pred_tfv), np.asarray(true_tfv)
                ),
                "pfv_rank_correlation": rank_correlation(
                    np.asarray(pred_pfv), np.asarray(true_pfv)
                ),
                "tfv_top1_hit": float(best_pred == best_true),
                "tfv_selected_regret_m3": float(
                    true_tfv[best_pred] - true_tfv[best_true]
                ),
                "predicted_best_action_sha256": shas[best_pred],
                "true_best_action_sha256": shas[best_true],
            }
        )
    result = pd.DataFrame(detail)
    if result.empty:
        raise ValueError(
            f"no held-out {source_kind} checkpoint has at least three unique candidate actions"
        )
    return result


def run_ranking_gate(
    *,
    manifest_path: str | Path,
    run_summary_path: str | Path,
    d3_run_summary_path: str | Path,
    graph_path: str | Path,
    step2_path: str | Path,
    priority_path: str | Path,
    thresholds_path: str | Path,
    output_path: str | Path,
    split: str = "development",
    development_fold: str = "validation",
    device: str | None = None,
) -> dict[str, object]:
    d2 = join_manifest_runs(pd.read_csv(manifest_path), pd.read_csv(run_summary_path))
    d2 = _heldout(d2, split=split, development_fold=development_fold)
    action_keys = ["checkpoint_id", "candidate_action_sha256"]
    if "event_id" in d2.columns:
        action_keys.insert(0, "event_id")
    d2 = d2.drop_duplicates(action_keys).copy()

    d3 = _heldout(
        pd.read_csv(d3_run_summary_path),
        split=split,
        development_fold=development_fold,
    )
    required_d3 = {"checkpoint_id", "sequence_sha256", "metadata_path"}
    missing = sorted(required_d3 - set(d3.columns))
    if missing:
        raise ValueError(f"D3 ranking summary lacks columns: {missing}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(graph_path)
    model = _load_step2(step2_path, dev)
    priority = load_priority_nodes(priority_path)
    missing_priority = sorted(set(priority) - set(graph.node_ids))
    if missing_priority:
        raise ValueError(f"priority nodes absent from graph: {missing_priority}")
    pidx = np.asarray([graph.node_ids.index(n) for n in priority], dtype=int)

    d2_detail = _score_groups(
        d2,
        action_sha_column="candidate_action_sha256",
        source_kind="D2_SINGLE_ACTUATOR",
        model=model,
        graph=graph,
        pidx=pidx,
        device=dev,
    )
    d3_detail = _score_groups(
        d3,
        action_sha_column="sequence_sha256",
        source_kind="D3_JOINT_SEQUENCE",
        model=model,
        graph=graph,
        pidx=pidx,
        device=dev,
    )
    detail = pd.concat([d2_detail, d3_detail], ignore_index=True)

    def metrics_for(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_tfv_rank_correlation": float(frame["tfv_rank_correlation"].mean()),
            f"{prefix}_pfv_rank_correlation": float(frame["pfv_rank_correlation"].mean()),
            f"{prefix}_tfv_top1_hit_rate": float(frame["tfv_top1_hit"].mean()),
            f"{prefix}_tfv_selected_regret_m3": float(
                frame["tfv_selected_regret_m3"].mean()
            ),
            f"{prefix}_ranking_checkpoints": float(len(frame)),
        }

    metrics = {
        **metrics_for(d2_detail, "d2"),
        **metrics_for(d3_detail, "d3"),
    }
    # Conservative combined diagnostics retained for compact reporting; Formal thresholds
    # should target the explicit D2/D3 metrics above.
    metrics["tfv_rank_correlation"] = min(
        metrics["d2_tfv_rank_correlation"], metrics["d3_tfv_rank_correlation"]
    )
    metrics["tfv_top1_hit_rate"] = min(
        metrics["d2_tfv_top1_hit_rate"], metrics["d3_tfv_top1_hit_rate"]
    )
    metrics["tfv_selected_regret_m3"] = max(
        metrics["d2_tfv_selected_regret_m3"],
        metrics["d3_tfv_selected_regret_m3"],
    )

    minimum, maximum = _thresholds(thresholds_path)
    result = apply_metric_thresholds(metrics, minimum=minimum, maximum=maximum)
    detail_path = Path(output_path).with_suffix(".detail.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    payload: dict[str, object] = {
        "contract": "SWMM_JOINT_ACTION_RANKING_ACCEPTANCE_V4_D2_D3",
        "passed": result.passed,
        "failed_metrics": list(result.failed_metrics),
        "metrics": result.metrics,
        "thresholds": {"minimum": minimum, "maximum": maximum},
        "split": split,
        "development_fold": development_fold if split == "development" else "",
        "step2_sha256": _sha(step2_path),
        "d2_manifest_sha256": _sha(manifest_path),
        "d2_run_summary_sha256": _sha(run_summary_path),
        "d3_run_summary_sha256": _sha(d3_run_summary_path),
        "prediction_volume_integration": "trapezoid_current_plus_future_flooding_rate",
        "truth_source_tfv_pfv": "SWMM_NODE_STATISTICS_CUMULATIVE_EXACT_HORIZON",
        "priority_metrics_diagnostic_only": True,
        "detail_csv": str(detail_path),
        "interpretation": "D2 validates local/single-actuator ordering; D3 validates the joint multi-actuator multi-step sequence ordering actually required by MPC.",
    }
    Path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out exact-SWMM ranking gate for both D2 and D3 action spaces"
    )
    parser.add_argument("--manifest", required=True, help="D2 design manifest")
    parser.add_argument("--run-summary", required=True, help="D2 run summary")
    parser.add_argument("--d3-run-summary", required=True)
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
        d3_run_summary_path=args.d3_run_summary,
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
