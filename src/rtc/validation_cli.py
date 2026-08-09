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
from .dataset_compile import compile_branch_tensors
from .gradient_truth import compare_gradient_vectors
from .production_cli import _load_graph, _load_step2


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _thresholds(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        {str(k): float(v) for k, v in payload.get("minimum", {}).items()},
        {str(k): float(v) for k, v in payload.get("maximum", {}).items()},
    )


def _exact_node_volumes(metadata_path: str | Path, node_ids: tuple[str, ...]) -> np.ndarray:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats_name = meta.get("node_statistics_file")
    if not stats_name:
        raise ValueError(f"D2 branch lacks exact node_statistics_file: {metadata_path}")
    stats = pd.read_csv(meta_path.parent / str(stats_name), compression="infer")
    if not {"node_id", "delta_flooding_volume_m3"}.issubset(stats.columns):
        raise ValueError("node statistics file lacks exact flooding-volume columns")
    values = stats.set_index(stats["node_id"].astype(str))["delta_flooding_volume_m3"]
    missing = [node for node in node_ids if node not in values.index]
    if missing:
        raise ValueError(f"node statistics missing nodes: {missing[:20]}")
    return values.reindex(node_ids).to_numpy(dtype=float)


def _join_manifest_runs(manifest: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_action_sha256"]
    for candidate in ("event_id", "checkpoint_id", "checkpoint_minutes"):
        if candidate in manifest.columns and candidate in runs.columns:
            keys.append(candidate)
    merged = manifest.merge(runs, on=keys, how="inner", suffixes=("", "_run"))
    if merged.empty:
        raise ValueError(f"manifest and run summary have no matching D2 branches using keys {keys}")
    if merged["metadata_path"].isna().any():
        raise ValueError("joined D2 rows contain missing metadata paths")
    return merged


def _predict_metrics_and_gradients(
    *,
    model,
    graph,
    metadata_path: str,
    priority_indices: np.ndarray,
    actuator_index: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    branch = compile_branch_tensors(metadata_path)
    if branch.node_ids != graph.node_ids or branch.actuator_ids != graph.actuator_ids:
        raise ValueError("D2 branch schema differs from locked graph schema")
    dt = np.diff(branch.elapsed_seconds).astype(np.float32)
    if np.any(dt <= 0):
        raise ValueError("D2 branch time grid is not strictly increasing")
    base = torch.as_tensor(branch.settings[0], dtype=torch.float32, device=device).clone().detach()
    base.requires_grad_(True)
    settings = base.view(1, 1, -1).expand(1, branch.settings.shape[0], -1)
    rollout = model.rollout(
        torch.as_tensor(branch.initial_state[None], dtype=torch.float32, device=device),
        torch.as_tensor(branch.rainfall[None], dtype=torch.float32, device=device),
        settings,
        torch.as_tensor(branch.previous_actuator_flow[None], dtype=torch.float32, device=device),
        torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        torch.as_tensor(graph.actuator_physics[None], dtype=torch.float32, device=device),
        torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
        torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
    )
    rates = rollout.states[0, ..., 2].clamp_min(0.0)
    dt_tensor = torch.as_tensor(dt, dtype=torch.float32, device=device).view(-1, 1)
    node_volumes = (rates * dt_tensor).sum(dim=0)
    tfv = node_volumes.sum()
    pfv = node_volumes[torch.as_tensor(priority_indices, dtype=torch.long, device=device)].sum()
    tfv_grad = torch.autograd.grad(tfv, base, retain_graph=True)[0][actuator_index]
    pfv_grad = torch.autograd.grad(pfv, base)[0][actuator_index]
    return float(tfv.detach()), float(pfv.detach()), float(tfv_grad.detach()), float(pfv_grad.detach())


def validate_d2_main() -> None:
    parser = argparse.ArgumentParser(description="Validate Step2 gradient direction and candidate ranking against exact D2 SWMM truth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--step2", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--gradient-thresholds", required=True)
    parser.add_argument("--ranking-thresholds", required=True)
    parser.add_argument("--gradient-out", required=True)
    parser.add_argument("--ranking-out", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--development-fold", default="validation")
    parser.add_argument("--device")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    runs = pd.read_csv(args.run_summary)
    merged = _join_manifest_runs(manifest, runs)
    if "scientific_split" in merged.columns:
        merged = merged[merged["scientific_split"].astype(str) == args.split]
    if args.split == "development" and "development_fold" in merged.columns:
        merged = merged[merged["development_fold"].astype(str) == args.development_fold]
    if merged.empty:
        raise ValueError("no held-out D2 branches remain after split filtering")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = _load_graph(args.graph)
    model = _load_step2(args.step2, device)
    priority = load_priority_nodes(args.priority)
    missing = sorted(set(priority) - set(graph.node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from graph: {missing}")
    priority_indices = np.asarray([graph.node_ids.index(node) for node in priority], dtype=int)
    actuator_index = {aid: i for i, aid in enumerate(graph.actuator_ids)}

    exact_cache: dict[str, np.ndarray] = {}
    prediction_cache: dict[str, tuple[float, float, float, float]] = {}
    detail_rows: list[dict[str, object]] = []
    group_cols = ["checkpoint_id", "actuator_id"]
    if "event_id" in merged.columns:
        group_cols.insert(0, "event_id")

    for keys, group in merged.groupby(group_cols, sort=False):
        group = group.sort_values("requested_setting")
        base_setting = float(group["base_setting"].iloc[0])
        below = group[group["requested_setting"].astype(float) < base_setting]
        above = group[group["requested_setting"].astype(float) > base_setting]
        center = group[np.isclose(group["requested_setting"].astype(float), base_setting)]
        if below.empty or above.empty or center.empty:
            continue
        lo, hi, mid = below.iloc[-1], above.iloc[0], center.iloc[0]
        aid = str(mid["actuator_id"])
        if aid not in actuator_index:
            raise ValueError(f"D2 actuator absent from graph: {aid}")

        def exact(row) -> np.ndarray:
            path = str(row["metadata_path"])
            if path not in exact_cache:
                exact_cache[path] = _exact_node_volumes(path, graph.node_ids)
            return exact_cache[path]

        lo_v, hi_v = exact(lo), exact(hi)
        du = float(hi["requested_setting"] - lo["requested_setting"])
        if abs(du) <= 1e-12:
            continue
        true_tfv_grad = float((hi_v.sum() - lo_v.sum()) / du)
        true_pfv_grad = float((hi_v[priority_indices].sum() - lo_v[priority_indices].sum()) / du)
        center_path = str(mid["metadata_path"])
        if center_path not in prediction_cache:
            prediction_cache[center_path] = _predict_metrics_and_gradients(
                model=model,
                graph=graph,
                metadata_path=center_path,
                priority_indices=priority_indices,
                actuator_index=actuator_index[aid],
                device=device,
            )
        pred_tfv, pred_pfv, pred_tfv_grad, pred_pfv_grad = prediction_cache[center_path]
        detail_rows.append(
            {
                "event_id": str(mid.get("event_id", "")),
                "checkpoint_id": str(mid["checkpoint_id"]),
                "actuator_id": aid,
                "pred_tfv_gradient": pred_tfv_grad,
                "true_tfv_gradient": true_tfv_grad,
                "pred_pfv_gradient": pred_pfv_grad,
                "true_pfv_gradient": true_pfv_grad,
                "pred_center_tfv_m3": pred_tfv,
                "pred_center_pfv_m3": pred_pfv,
            }
        )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError("no complete lower/center/upper D2 finite-difference triplets")
    tfv_agreement = compare_gradient_vectors(detail["pred_tfv_gradient"], detail["true_tfv_gradient"])
    pfv_agreement = compare_gradient_vectors(detail["pred_pfv_gradient"], detail["true_pfv_gradient"])
    gradient_metrics = {
        "tfv_gradient_sign_accuracy": tfv_agreement.sign_accuracy,
        "tfv_gradient_cosine_similarity": tfv_agreement.cosine_similarity,
        "tfv_gradient_mae": tfv_agreement.magnitude_mae,
        "pfv_gradient_sign_accuracy": pfv_agreement.sign_accuracy,
        "pfv_gradient_cosine_similarity": pfv_agreement.cosine_similarity,
        "pfv_gradient_mae": pfv_agreement.magnitude_mae,
        "gradient_cases": float(len(detail)),
    }

    ranking_rows: list[dict[str, object]] = []
    rank_group_cols = ["checkpoint_id"]
    if "event_id" in merged.columns:
        rank_group_cols.insert(0, "event_id")
    for keys, group in merged.drop_duplicates(group_cols + ["candidate_action_sha256"]).groupby(rank_group_cols, sort=False):
        predicted_tfv: list[float] = []
        predicted_pfv: list[float] = []
        true_tfv: list[float] = []
        true_pfv: list[float] = []
        shas: list[str] = []
        for _, row in group.iterrows():
            path = str(row["metadata_path"])
            branch = compile_branch_tensors(path)
            if branch.node_ids != graph.node_ids or branch.actuator_ids != graph.actuator_ids:
                raise ValueError("candidate branch schema differs from graph")
            dt = np.diff(branch.elapsed_seconds).astype(np.float32)
            with torch.no_grad():
                rollout = model.rollout(
                    torch.as_tensor(branch.initial_state[None], dtype=torch.float32, device=device),
                    torch.as_tensor(branch.rainfall[None], dtype=torch.float32, device=device),
                    torch.as_tensor(branch.settings[None], dtype=torch.float32, device=device),
                    torch.as_tensor(branch.previous_actuator_flow[None], dtype=torch.float32, device=device),
                    torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
                    torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
                    torch.as_tensor(graph.actuator_physics[None], dtype=torch.float32, device=device),
                    torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
                    torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
                )
                rates = rollout.states[0, ..., 2].clamp_min(0.0)
                node_vol = (rates * torch.as_tensor(dt, device=device).view(-1, 1)).sum(dim=0).cpu().numpy()
            exact = exact_cache.get(path)
            if exact is None:
                exact = _exact_node_volumes(path, graph.node_ids)
                exact_cache[path] = exact
            predicted_tfv.append(float(node_vol.sum()))
            predicted_pfv.append(float(node_vol[priority_indices].sum()))
            true_tfv.append(float(exact.sum()))
            true_pfv.append(float(exact[priority_indices].sum()))
            shas.append(str(row["candidate_action_sha256"]))
        if len(shas) < 3:
            continue
        best_pred = int(np.argmin(predicted_tfv))
        best_true = int(np.argmin(true_tfv))
        ranking_rows.append(
            {
                "event_id": str(group["event_id"].iloc[0]) if "event_id" in group.columns else "",
                "checkpoint_id": str(group["checkpoint_id"].iloc[0]),
                "tfv_rank_correlation": rank_correlation(np.asarray(predicted_tfv), np.asarray(true_tfv)),
                "pfv_rank_correlation": rank_correlation(np.asarray(predicted_pfv), np.asarray(true_pfv)),
                "tfv_top1_hit": float(best_pred == best_true),
                "tfv_selected_regret_m3": float(true_tfv[best_pred] - true_tfv[best_true]),
            }
        )
    rank_detail = pd.DataFrame(ranking_rows)
    if rank_detail.empty:
        raise ValueError("no held-out checkpoint has enough candidates for ranking acceptance")
    ranking_metrics = {
        "tfv_rank_correlation": float(rank_detail["tfv_rank_correlation"].mean()),
        "pfv_rank_correlation": float(rank_detail["pfv_rank_correlation"].mean()),
        "tfv_top1_hit_rate": float(rank_detail["tfv_top1_hit"].mean()),
        "tfv_selected_regret_m3": float(rank_detail["tfv_selected_regret_m3"].mean()),
        "ranking_checkpoints": float(len(rank_detail)),
    }

    def write_result(path: str, contract: str, metrics: dict[str, float], threshold_path: str, details: pd.DataFrame) -> None:
        minimum, maximum = _thresholds(threshold_path)
        result = apply_metric_thresholds(metrics, minimum=minimum, maximum=maximum)
        detail_path = str(Path(path).with_suffix(".detail.csv"))
        details.to_csv(detail_path, index=False)
        payload = {
            "contract": contract,
            "passed": result.passed,
            "failed_metrics": list(result.failed_metrics),
            "metrics": result.metrics,
            "thresholds": {"minimum": minimum, "maximum": maximum},
            "step2_sha256": _sha(args.step2),
            "manifest_sha256": _sha(args.manifest),
            "run_summary_sha256": _sha(args.run_summary),
            "detail_csv": detail_path,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not result.passed:
            print(json.dumps(payload, indent=2))

    write_result(args.gradient_out, "D2_SWMM_GRADIENT_TRUTH_ACCEPTANCE_V1", gradient_metrics, args.gradient_thresholds, detail)
    write_result(args.ranking_out, "D2_SWMM_CANDIDATE_RANKING_ACCEPTANCE_V1", ranking_metrics, args.ranking_thresholds, rank_detail)
    grad_pass = json.loads(Path(args.gradient_out).read_text(encoding="utf-8"))["passed"]
    rank_pass = json.loads(Path(args.ranking_out).read_text(encoding="utf-8"))["passed"]
    print(json.dumps({"gradient": gradient_metrics, "ranking": ranking_metrics, "passed": bool(grad_pass and rank_pass)}, indent=2))
    if not (grad_pass and rank_pass):
        raise SystemExit(2)


def main() -> None:
    validate_d2_main()


if __name__ == "__main__":
    main()
