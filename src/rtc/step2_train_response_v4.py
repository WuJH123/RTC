"""Train-only diagnostics for the isolated V4 response model.

This module is deliberately not connected to the public Step2 trainer.  It consumes
already-compiled Train-only V6/cache arrays and supports bounded tiny-overfit and micro
mechanism experiments without any SWMM execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .flood_volume import trapezoid_node_flood_volume
from .large_model_cli import _step2_stats
from .production_cli import _load_graph
from .step2_action_pathway_audit_v4 import direct_pair_delta_tfv
from .step2_control_response_v4 import CounterfactualResponseV4, DifferentiableCounterfactualResponseModelV4
from .step2_counterfactual import counterfactual_groups, reference_index
from .step2_stability import derive_train_only_delta_scales
from .step2_training_cache import load_step2_training_cache


@dataclass(frozen=True)
class ResponseNormalizationV4:
    state_mean: np.ndarray
    state_std: np.ndarray
    rain_mean: np.ndarray
    rain_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    physics_mean: np.ndarray
    physics_std: np.ndarray
    flow_std: np.ndarray
    state_delta_scale: np.ndarray
    flow_delta_scale: np.ndarray
    source_manifest_sha256: str


@dataclass(frozen=True)
class ResponsePairV4:
    source_kind: str
    group: str
    reference: dict[str, np.ndarray]
    candidate: dict[str, np.ndarray]


def build_full_train_normalization(full_train_manifest: str | Path, graph) -> ResponseNormalizationV4:
    """Derive all scales from the complete Train18 V6 manifest only."""

    from .step2_shards import load_shard_manifest, sha256_file

    manifest = load_shard_manifest(full_train_manifest)
    (state_mean, state_std), (rain_mean, rain_std), (physics_mean, physics_std), (static_mean, static_std), flow_std = _step2_stats(manifest, graph)
    state_delta, flow_delta, _ = derive_train_only_delta_scales(
        str(full_train_manifest), state_std=state_std, flow_std=flow_std
    )
    return ResponseNormalizationV4(
        state_mean=np.asarray(state_mean, dtype=np.float32), state_std=np.asarray(state_std, dtype=np.float32),
        rain_mean=np.asarray(rain_mean, dtype=np.float32), rain_std=np.asarray(rain_std, dtype=np.float32),
        static_mean=np.asarray(static_mean, dtype=np.float32), static_std=np.asarray(static_std, dtype=np.float32),
        physics_mean=np.asarray(physics_mean, dtype=np.float32), physics_std=np.asarray(physics_std, dtype=np.float32),
        flow_std=np.asarray(flow_std, dtype=np.float32), state_delta_scale=np.asarray(state_delta, dtype=np.float32),
        flow_delta_scale=np.asarray(flow_delta, dtype=np.float32), source_manifest_sha256=sha256_file(full_train_manifest),
    )


def build_full_train_normalization_from_checkpoint(
    v3_checkpoint: str | Path,
    delta_scale_json: str | Path,
) -> ResponseNormalizationV4:
    """Recover already-stamped full-Train normalization without rescanning V6 shards.

    The V3 checkpoint is used only as a stamped normalization container; its weights are
    never loaded into a V4 model.  The delta scales are read from the immutable Train-only
    scale artifact and remain bound to its source manifest SHA.
    """

    from .production_cli import _load_step2

    stamped = _load_step2(str(v3_checkpoint), torch.device("cpu")).eval()
    details = json.loads(Path(delta_scale_json).read_text(encoding="utf-8"))
    if "details" in details:
        details = details["details"] | {"delta_state_scale": details["delta_state_scale"], "delta_flow_scale": details["delta_flow_scale"]}
    return ResponseNormalizationV4(
        state_mean=stamped.transition.state_mean.detach().numpy().astype(np.float32),
        state_std=stamped.transition.state_std.detach().numpy().astype(np.float32),
        rain_mean=stamped.transition.rain_mean.detach().numpy().astype(np.float32),
        rain_std=stamped.transition.rain_std.detach().numpy().astype(np.float32),
        static_mean=stamped.transition.static_mean.detach().numpy().astype(np.float32),
        static_std=stamped.transition.static_std.detach().numpy().astype(np.float32),
        physics_mean=stamped.actuator.physics_mean.detach().numpy().astype(np.float32),
        physics_std=stamped.actuator.physics_std.detach().numpy().astype(np.float32),
        flow_std=stamped.actuator.flow_std.detach().numpy().astype(np.float32),
        state_delta_scale=np.asarray(details["delta_state_scale"], dtype=np.float32),
        flow_delta_scale=np.asarray(details["delta_flow_scale"], dtype=np.float32),
        source_manifest_sha256=str(details["source_manifest_sha256"]),
    )


def _norm(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean) / np.maximum(std, 1e-6)).astype(np.float32, copy=False)


def _normalise_row(row: dict[str, np.ndarray], norm: ResponseNormalizationV4) -> dict[str, np.ndarray]:
    return {
        "initial_state": _norm(row["initial_state"], norm.state_mean, norm.state_std),
        "rainfall": _norm(row["rainfall"], norm.rain_mean, norm.rain_std),
        "settings": np.ascontiguousarray(row["settings"], dtype=np.float32),
        "previous_actuator_flow": _norm(row["previous_actuator_flow"], np.zeros_like(norm.flow_std), norm.flow_std),
        "target_states": _norm(row["target_states"], norm.state_mean, norm.state_std),
        "target_actuator_flows": _norm(row["target_actuator_flows"], np.zeros_like(norm.flow_std), norm.flow_std),
        "elapsed_seconds": np.asarray(row["elapsed_seconds"], dtype=np.float32),
        "target_states_physical": np.asarray(row["target_states"], dtype=np.float32),
        "initial_state_physical": np.asarray(row["initial_state"], dtype=np.float32),
        "target_actuator_flows_physical": np.asarray(row["target_actuator_flows"], dtype=np.float32),
        "exact_node_flood_volume_m3": np.asarray(row["exact_node_flood_volume_m3"], dtype=np.float32),
    }


def load_train_groups(cache_dir: str | Path, norm: ResponseNormalizationV4, selected_groups: list[str] | None = None) -> dict[str, list[ResponsePairV4]]:
    cache = load_step2_training_cache(Path(cache_dir) / "CACHE_MANIFEST.json")
    all_pairs: dict[str, list[ResponsePairV4]] = {}
    for item in cache["shards"]:
        arrays = item["arrays"]
        groups = counterfactual_groups(arrays)
        for group, indices in groups.items():
            if selected_groups is not None and group not in selected_groups:
                continue
            ref = reference_index(arrays, indices)
            source = group.split("::", 1)[0]
            pairs = all_pairs.setdefault(group, [])
            for candidate in sorted(indices):
                if candidate == ref:
                    continue
                def row(index: int) -> dict[str, np.ndarray]:
                    return {name: np.asarray(value[index]) for name, value in arrays.items() if name in {
                        "initial_state", "rainfall", "settings", "previous_actuator_flow", "target_states",
                        "target_actuator_flows", "elapsed_seconds", "exact_node_flood_volume_m3"
                    }}
                pairs.append(ResponsePairV4(source, group, _normalise_row(row(ref), norm), _normalise_row(row(candidate), norm)))
    if selected_groups is not None:
        missing = sorted(set(selected_groups) - set(all_pairs))
        if missing:
            raise ValueError(f"selected Train-only groups missing from cache: {missing}")
    return all_pairs


def _stack_pairs(pairs: list[ResponsePairV4], device: torch.device, norm: ResponseNormalizationV4) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    def stack(name: str, side: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(pair, side)[name] for pair in pairs], axis=0)).to(device)
    ref = {name: stack(name, "reference") for name in pairs[0].reference}
    cand = {name: stack(name, "candidate") for name in pairs[0].candidate}
    def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([a, b], dim=1).reshape(2 * len(pairs), *a.shape[1:])
    batch = {
        "initial_state": interleave(ref["initial_state"], cand["initial_state"]),
        "rainfall": interleave(ref["rainfall"], cand["rainfall"]),
        "settings": interleave(ref["settings"], cand["settings"]),
        "previous_actuator_flow": interleave(ref["previous_actuator_flow"], cand["previous_actuator_flow"]),
        "target_states": interleave(ref["target_states"], cand["target_states"]),
        "target_actuator_flows": interleave(ref["target_actuator_flows"], cand["target_actuator_flows"]),
        "elapsed_seconds": interleave(ref["elapsed_seconds"], cand["elapsed_seconds"]),
        "target_states_physical": interleave(ref["target_states_physical"], cand["target_states_physical"]),
        "initial_state_physical": interleave(ref["initial_state_physical"], cand["initial_state_physical"]),
        "target_actuator_flows_physical": interleave(ref["target_actuator_flows_physical"], cand["target_actuator_flows_physical"]),
        "exact_node_flood_volume_m3": interleave(ref["exact_node_flood_volume_m3"], cand["exact_node_flood_volume_m3"]),
    }
    reference_settings = interleave(ref["settings"], ref["settings"])
    candidate_settings = batch["settings"]
    return batch, reference_settings, candidate_settings


def _loss_and_metrics(
    output: CounterfactualResponseV4,
    batch: dict[str, torch.Tensor],
    reference_settings: torch.Tensor,
    candidate_settings: torch.Tensor,
    norm: ResponseNormalizationV4,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    w = {"absolute_state": 0.5, "absolute_flow": 0.5, "delta_state": 1.0, "delta_flow": 1.0, "delta_tfv": 2.0, "ranking": 1.0, "physical": 0.1}
    if weights:
        w.update(weights)
    ref = slice(0, None, 2); cand = slice(1, None, 2)
    pred_ref_state = output.reference_states[ref]
    pred_cand_state = output.candidate_states[cand]
    pred_ref_flow = output.reference_flows[ref]
    pred_cand_flow = output.candidate_flows[cand]
    true_ref_state = batch["target_states"][ref]
    true_cand_state = batch["target_states"][cand]
    true_ref_flow = batch["target_actuator_flows"][ref]
    true_cand_flow = batch["target_actuator_flows"][cand]
    absolute_state = F.mse_loss(pred_ref_state, true_ref_state) + F.mse_loss(pred_cand_state, true_cand_state)
    absolute_flow = F.mse_loss(pred_ref_flow, true_ref_flow) + F.mse_loss(pred_cand_flow, true_cand_flow)
    delta_state = F.mse_loss(pred_cand_state - pred_ref_state, true_cand_state - true_ref_state)
    delta_flow = F.mse_loss(pred_cand_flow - pred_ref_flow, true_cand_flow - true_ref_flow)
    state_mean = torch.as_tensor(norm.state_mean, device=pred_ref_state.device, dtype=pred_ref_state.dtype)
    state_std = torch.as_tensor(norm.state_std, device=pred_ref_state.device, dtype=pred_ref_state.dtype)
    initial_phys = batch["initial_state_physical"][ref]
    pred_ref_phys = pred_ref_state * state_std + state_mean
    pred_cand_phys = pred_cand_state * state_std + state_mean
    true_ref_phys = batch["target_states_physical"][ref]
    true_cand_phys = batch["target_states_physical"][cand]
    dt = batch["elapsed_seconds"][ref][:, 1:] - batch["elapsed_seconds"][ref][:, :-1]
    smooth_scale = max(float(0.01 * norm.state_std[2]), 1e-4)
    pred_delta_tfv = direct_pair_delta_tfv(pred_cand_phys[..., 2], pred_ref_phys[..., 2], dt_seconds=dt, smooth=True, softplus_scale=smooth_scale)
    true_delta_tfv = direct_pair_delta_tfv(true_cand_phys[..., 2], true_ref_phys[..., 2], dt_seconds=dt, smooth=False)
    scale = true_delta_tfv.detach().abs().clamp_min(1.0)
    delta_tfv = F.smooth_l1_loss((pred_delta_tfv - true_delta_tfv.detach()) / scale, torch.zeros_like(scale))
    meaningful = true_delta_tfv.detach().abs() >= 1.0
    if bool(meaningful.any()):
        sign = true_delta_tfv.detach().sign()
        ranking = F.softplus(-sign * pred_delta_tfv / scale)[meaningful].mean()
        sign_accuracy = (pred_delta_tfv.detach().sign()[meaningful] == sign[meaningful]).float().mean()
    else:
        ranking = pred_delta_tfv.sum() * 0.0
        sign_accuracy = torch.ones((), device=pred_delta_tfv.device)
    physical = F.relu(-pred_cand_phys[..., :4]).square().mean() + F.relu(-pred_ref_phys[..., :4]).square().mean()
    total = w["absolute_state"] * absolute_state + w["absolute_flow"] * absolute_flow + w["delta_state"] * delta_state + w["delta_flow"] * delta_flow + w["delta_tfv"] * delta_tfv + w["ranking"] * ranking + w["physical"] * physical
    metrics = {
        "total_loss": float(total.detach()), "absolute_state_loss": float(absolute_state.detach()), "absolute_flow_loss": float(absolute_flow.detach()),
        "delta_state_loss": float(delta_state.detach()), "delta_flow_loss": float(delta_flow.detach()), "delta_tfv_loss": float(delta_tfv.detach()),
        "ranking_loss": float(ranking.detach()), "physical_loss": float(physical.detach()), "sensitivity_ratio": float((pred_delta_tfv.detach().abs() / scale).mean()),
        "sign_accuracy": float(sign_accuracy.detach()), "tfv_mae_m3": float((pred_delta_tfv.detach() - true_delta_tfv.detach()).abs().mean()),
    }
    return total, metrics


def train_response_v4(
    *,
    model: DifferentiableCounterfactualResponseModelV4,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    norm: ResponseNormalizationV4,
    graph,
    out_path: str | Path,
    epochs: int,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one fixed FP32 Train-only diagnostic fit and save history/checkpoint."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(target_device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pairs = [pair for group in sorted(grouped_pairs) for pair in grouped_pairs[group]]
    if not pairs:
        raise ValueError("response diagnostic has no Train-only pairs")
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, int(epochs) + 1):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(pairs)).tolist()
        model.train()
        totals: dict[str, float] = {}
        grad_values: list[float] = []
        for start in range(0, len(order), batch_size):
            selected = [pairs[i] for i in order[start:start + batch_size]]
            batch, reference_settings, candidate_settings = _stack_pairs(selected, target_device, norm)
            static = torch.as_tensor(_norm(graph.static_node_features, norm.static_mean, norm.static_std), device=target_device)
            physics = torch.as_tensor(_norm(graph.actuator_physics, norm.physics_mean, norm.physics_std), device=target_device)
            rainfall = batch["rainfall"]
            output = model(batch["initial_state"], rainfall, reference_settings, candidate_settings, batch["previous_actuator_flow"], static, physics, torch.as_tensor(graph.actuator_upstream, device=target_device), torch.as_tensor(graph.actuator_downstream, device=target_device), torch.as_tensor(graph.edge_index, device=target_device))
            loss, metrics = _loss_and_metrics(output, batch, reference_settings, candidate_settings, norm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V4 response loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite V4 response gradient at epoch {epoch}")
            optimizer.step()
            grad_values.append(grad_norm)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        denom = max(1, int(np.ceil(len(order) / batch_size)))
        row = {"epoch": epoch, "pairs": len(order), "batch_size": batch_size, "gradient_norm_max": max(grad_values), "gradient_norm_mean": float(np.mean(grad_values))}
        row.update({key: value / denom for key, value in totals.items()})
        history.append(row)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "training_contract": "STEP2_CONTROL_RESPONSE_V4_TRAIN_ONLY_DIAGNOSTIC", "seed": seed, "full_train_manifest_sha256": norm.source_manifest_sha256}, out)
    meta = {"checkpoint": str(out), "epochs": epochs, "pairs": len(pairs), "seed": seed, "device": str(target_device), "precision": "fp32", "full_train_manifest_sha256": norm.source_manifest_sha256, "history": history}
    out.with_suffix(out.suffix + ".history.json").write_text(__import__("json").dumps(meta, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return meta


def evaluate_response_groups(
    *, model: DifferentiableCounterfactualResponseModelV4, grouped_pairs: dict[str, list[ResponsePairV4]], norm: ResponseNormalizationV4, graph, device: str = "cuda"
) -> list[dict[str, Any]]:
    target_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.eval().to(target_device).float()
    static = torch.as_tensor(_norm(graph.static_node_features, norm.static_mean, norm.static_std), device=target_device)
    physics = torch.as_tensor(_norm(graph.actuator_physics, norm.physics_mean, norm.physics_std), device=target_device)
    rows: list[dict[str, Any]] = []
    for group, pairs in sorted(grouped_pairs.items()):
        preds: list[float] = []
        truths: list[float] = []
        for start in range(0, len(pairs), 8):
            selected = pairs[start:start + 8]
            batch, ref_settings, cand_settings = _stack_pairs(selected, target_device, norm)
            with torch.no_grad():
                out = model(batch["initial_state"], batch["rainfall"], ref_settings, cand_settings, batch["previous_actuator_flow"], static, physics, torch.as_tensor(graph.actuator_upstream, device=target_device), torch.as_tensor(graph.actuator_downstream, device=target_device), torch.as_tensor(graph.edge_index, device=target_device))
                state_mean = torch.as_tensor(norm.state_mean, device=target_device)
                state_std = torch.as_tensor(norm.state_std, device=target_device)
                pred_ref = out.reference_states[0::2] * state_std + state_mean
                pred_cand = out.candidate_states[1::2] * state_std + state_mean
                dt = batch["elapsed_seconds"][0::2, 1:] - batch["elapsed_seconds"][0::2, :-1]
                pred_delta = direct_pair_delta_tfv(pred_cand[..., 2], pred_ref[..., 2], dt_seconds=dt, smooth=True, softplus_scale=max(float(.01 * norm.state_std[2]), 1e-4))
            pred_values = pred_delta.detach().cpu().numpy().tolist()
            for pair, value in zip(selected, pred_values):
                true_ref = pair.reference["target_states_physical"][..., 2]
                true_cand = pair.candidate["target_states_physical"][..., 2]
                dt_np = pair.reference["elapsed_seconds"][1:] - pair.reference["elapsed_seconds"][:-1]
                true = direct_pair_delta_tfv(torch.from_numpy(true_cand[None]), torch.from_numpy(true_ref[None]), dt_seconds=torch.from_numpy(dt_np[None]), smooth=False).item()
                preds.append(float(value)); truths.append(float(true))
        pred = np.asarray(preds); truth = np.asarray(truths)
        sign = np.sign(pred) == np.sign(truth)
        rank = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(truth)))[0, 1]) if np.ptp(pred) > 0 and np.ptp(truth) > 0 else float("nan")
        rows.append({"group": group, "source_kind": group.split("::", 1)[0], "candidate_count": len(pred), "predicted_delta_tfv_spread_m3": float(np.ptp(pred)), "true_delta_tfv_spread_m3": float(np.ptp(truth)), "spread_ratio": float(np.ptp(pred) / max(np.ptp(truth), 1e-6)), "rank": rank, "sign": float(np.mean(sign)), "top1": bool(np.argmin(pred) == np.argmin(truth))})
    return rows


__all__ = [
    "ResponseNormalizationV4", "ResponsePairV4", "build_full_train_normalization", "build_full_train_normalization_from_checkpoint", "load_train_groups", "train_response_v4", "evaluate_response_groups",
]
