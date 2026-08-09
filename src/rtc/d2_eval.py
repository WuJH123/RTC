from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset_compile import compile_branch_tensors
from .flood_volume import trapezoid_node_flood_volume


def exact_node_volumes(metadata_path: str | Path, node_ids: tuple[str, ...]) -> np.ndarray:
    """Load authoritative exact-horizon cumulative flooding volume for every node."""

    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats_name = meta.get("node_statistics_file")
    if not stats_name:
        raise ValueError(f"branch lacks exact node_statistics_file: {metadata_path}")
    stats = pd.read_csv(meta_path.parent / str(stats_name), compression="infer")
    required = {"node_id", "delta_flooding_volume_m3"}
    if not required.issubset(stats.columns):
        raise ValueError("node statistics file lacks exact flooding-volume columns")
    table = stats.copy()
    table["node_id"] = table["node_id"].astype(str)
    values = table.set_index("node_id")["delta_flooding_volume_m3"]
    missing = [node for node in node_ids if node not in values.index]
    if missing:
        raise ValueError(f"node statistics missing nodes: {missing[:20]}")
    return values.reindex(node_ids).to_numpy(dtype=float)


def join_manifest_runs(manifest: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Join D2 design provenance to one executed branch record."""

    keys = ["candidate_action_sha256"]
    for candidate in ("event_id", "checkpoint_id", "checkpoint_minutes"):
        if candidate in manifest.columns and candidate in runs.columns:
            keys.append(candidate)
    merged = manifest.merge(runs, on=keys, how="inner", suffixes=("", "_run"))
    if merged.empty:
        raise ValueError(
            f"manifest and run summary have no matching D2 branches using keys {keys}"
        )
    if "metadata_path" not in merged.columns or merged["metadata_path"].isna().any():
        raise ValueError("joined D2 rows contain missing metadata paths")
    return merged


def _verify_model_branch_time_contract(model, branch) -> None:
    runtime = dict(getattr(model, "runtime_metadata", {}))
    if runtime:
        step = int(runtime.get("model_step_seconds", -1))
        horizon = int(runtime.get("horizon_steps", -1))
        if step != branch.model_step_seconds:
            raise ValueError(
                f"action-effect branch step {branch.model_step_seconds}s differs from Step2 {step}s"
            )
        if horizon != branch.horizon_steps:
            raise ValueError(
                f"action-effect branch horizon {branch.horizon_steps} differs from Step2 {horizon}"
            )


def model_metrics(
    *,
    model,
    graph,
    metadata_path: str | Path,
    priority_indices: np.ndarray,
    device: torch.device,
    gradient_actuator_index: int | None = None,
) -> tuple[float, float, float | None, float | None]:
    """Predict cumulative TFV/PFV and optional setting gradients in physical SI units."""

    branch = compile_branch_tensors(metadata_path)
    if branch.node_ids != graph.node_ids or branch.actuator_ids != graph.actuator_ids:
        raise ValueError("action-effect branch schema differs from locked graph schema")
    _verify_model_branch_time_contract(model, branch)
    dt = np.diff(branch.elapsed_seconds).astype(np.float32)
    if np.any(dt <= 0):
        raise ValueError("action-effect branch time grid is not strictly increasing")

    initial = torch.as_tensor(branch.initial_state[None], dtype=torch.float32, device=device)
    if gradient_actuator_index is None:
        settings = torch.as_tensor(branch.settings[None], dtype=torch.float32, device=device)
        base = None
    else:
        base = torch.as_tensor(
            branch.settings[0], dtype=torch.float32, device=device
        ).clone().detach()
        base.requires_grad_(True)
        settings = base.view(1, 1, -1).expand(1, branch.settings.shape[0], -1)

    rollout = model.rollout(
        initial,
        torch.as_tensor(branch.rainfall[None], dtype=torch.float32, device=device),
        settings,
        torch.as_tensor(
            branch.previous_actuator_flow[None], dtype=torch.float32, device=device
        ),
        torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        torch.as_tensor(graph.actuator_physics[None], dtype=torch.float32, device=device),
        torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
        torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
    )
    node_volume = trapezoid_node_flood_volume(
        initial,
        rollout.states,
        flood_rate_index=2,
        dt_seconds=torch.as_tensor(dt, dtype=torch.float32, device=device),
    )[0]
    tfv = node_volume.sum()
    pidx = torch.as_tensor(priority_indices, dtype=torch.long, device=device)
    pfv = node_volume[pidx].sum() if pidx.numel() else tfv.new_zeros(())

    if gradient_actuator_index is None:
        return float(tfv.detach()), float(pfv.detach()), None, None
    if base is None:
        raise RuntimeError("gradient base setting tensor was not created")
    tfv_grad = torch.autograd.grad(tfv, base, retain_graph=pidx.numel() > 0)[0][
        int(gradient_actuator_index)
    ]
    pfv_grad: float | None = None
    if pidx.numel() > 0:
        grad = torch.autograd.grad(pfv, base)[0][int(gradient_actuator_index)]
        pfv_grad = float(grad.detach())
    return float(tfv.detach()), float(pfv.detach()), float(tfv_grad.detach()), pfv_grad
