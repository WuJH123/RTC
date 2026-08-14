"""D5 directional-gradient supervision/audit for the V127 smooth MPC objective."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60

V127_GRADIENT_TRAINING_CONTRACT = "PROJECT7_V127_D5_DIRECTIONAL_GRADIENT_SUPERVISION_V2_SMOOTH_OBJECTIVE"
V127_GRADIENT_AUDIT_CONTRACT = "PROJECT7_V127_D5_DIRECTIONAL_GRADIENT_AUDIT_V2_SMOOTH_OBJECTIVE"


@dataclass(frozen=True)
class V127GradientTrainingDesign:
    epochs: int = 2
    learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-6
    grad_clip: float = 5.0
    magnitude_weight: float = 1.0
    sign_weight: float = 0.50
    hard_center_tfv_weight: float = 0.10

    def validate(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("V127 gradient-training schedule is invalid")
        for value in (
            self.weight_decay,
            self.magnitude_weight,
            self.sign_weight,
            self.hard_center_tfv_weight,
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("V127 gradient-training weights are invalid")


def _sequence(row: pd.Series, actuator_ids: Sequence[str]) -> np.ndarray:
    blocks = json.loads(str(row["settings_sequence_json"]))
    if not isinstance(blocks, list) or len(blocks) != 36:
        raise ValueError("V127 D5 execution row requires 36 control blocks")
    values = np.asarray([[float(block[aid]) for aid in actuator_ids] for block in blocks], dtype=np.float32)
    return np.repeat(values, 2, axis=0)


def build_direction_cases_v127(
    labels: pd.DataFrame,
    execution_manifest: pd.DataFrame,
    *,
    actuator_ids: Sequence[str],
    split_role: str,
) -> list[dict[str, Any]]:
    role = str(split_role).lower()
    if role not in {"fit", "audit"}:
        raise ValueError("V127 gradient case role must be fit/audit")
    labels = labels[labels["split_role"].astype(str) == role]
    manifest = execution_manifest[execution_manifest["d5_split_role"].astype(str) == role]
    cases: list[dict[str, Any]] = []
    for _, label in labels.iterrows():
        center_id = str(label["center_id"])
        direction_id = str(label["direction_id"])
        center = manifest[(manifest["center_id"].astype(str) == center_id) & (manifest["probe_role"] == "center")]
        plus = manifest[(manifest["direction_id"].astype(str) == direction_id) & (manifest["probe_role"] == "plus")]
        minus = manifest[(manifest["direction_id"].astype(str) == direction_id) & (manifest["probe_role"] == "minus")]
        if len(center) != 1 or len(plus) != 1 or len(minus) != 1:
            raise RuntimeError(f"V127 D5 {direction_id} lacks exactly one center/+/- branch")
        eps = float(label["epsilon"])
        if eps <= 0:
            raise RuntimeError("V127 D5 epsilon must be positive")
        center_seq = _sequence(center.iloc[0], actuator_ids)
        plus_seq = _sequence(plus.iloc[0], actuator_ids)
        minus_seq = _sequence(minus.iloc[0], actuator_ids)
        direction = (plus_seq - minus_seq) / (2.0 * eps)
        if not np.isfinite(direction).all() or float(np.max(np.abs(direction))) <= 1e-8:
            raise RuntimeError("V127 D5 action-space direction is zero/non-finite")
        cases.append({
            "rainfall_group": str(label["rainfall_group"]),
            "event_id": str(label["event_id"]),
            "checkpoint_id": str(label["checkpoint_id"]),
            "center_id": center_id,
            "direction_id": direction_id,
            "center_sequence": center_seq,
            "direction_sequence_per_coeff": direction.astype(np.float32),
            "true_tfv_gradient": float(label["true_tfv_directional_gradient_m3_per_coeff"]),
            "center_tfv_m3": float(label["center_tfv_m3"]),
        })
    if not cases:
        raise ValueError(f"V127 D5 has no {role} gradient cases")
    return cases


def _lookup(cache: Any) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for name in cache.names():
        entry = cache.entry(name)
        key = (str(entry.event_id), str(entry.checkpoint_id))
        previous = result.get(key)
        if previous is None or (name.startswith("D2::") and not previous.startswith("D2::")):
            result[key] = name
    return result


def _physical(cache: Any, name: str, normalization: InputNormalizationV60, device: torch.device):
    batch = cache.batch(name, normalization, device)
    dtype = batch.initial_state.dtype
    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device)
    return (
        batch.initial_state * tensor(normalization.state_std).clamp_min(1e-6) + tensor(normalization.state_mean),
        batch.rainfall * tensor(normalization.rainfall_std).clamp_min(1e-6) + tensor(normalization.rainfall_mean),
        batch.previous_actuator_flow * tensor(normalization.flow_std).clamp_min(1e-6) + tensor(normalization.flow_mean),
    )


def predicted_directional_gradient_v127(
    model,
    *,
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    center_sequence: np.ndarray,
    direction_sequence_per_coeff: np.ndarray,
    graph: Any,
    flood_rate_index: int,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = initial_state.device
    center = torch.as_tensor(center_sequence, dtype=initial_state.dtype, device=device)[None].requires_grad_(True)
    direction = torch.as_tensor(direction_sequence_per_coeff, dtype=initial_state.dtype, device=device)[None]
    output = model.objective_rollout(
        initial_state=initial_state,
        rainfall=rainfall,
        settings=center,
        previous_actuator_flow=previous_flow,
        actuator_upstream=torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        actuator_physics=torch.as_tensor(graph.actuator_physics, dtype=initial_state.dtype, device=device),
        static_node_features=torch.as_tensor(graph.static_node_features, dtype=initial_state.dtype, device=device),
        edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
        flood_rate_index=int(flood_rate_index),
        priority_indices=None,
        dt_seconds=300.0,
    )
    smooth_tfv = output.optimization_tfv_m3.sum()
    action_gradient = torch.autograd.grad(
        smooth_tfv, center, create_graph=create_graph, retain_graph=create_graph
    )[0]
    directional = torch.sum(action_gradient * direction)
    return directional, output.tfv_m3.sum()


def _scale(cases: Sequence[dict[str, Any]]) -> float:
    values = np.abs(np.asarray([float(case["true_tfv_gradient"]) for case in cases], dtype=float))
    values = values[np.isfinite(values) & (values > 1e-8)]
    return max(float(np.quantile(values, 0.75)), 100.0) if values.size else 100.0


def train_d5_gradient_v127(
    model,
    *,
    cache: Any,
    cases: Sequence[dict[str, Any]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127GradientTrainingDesign = V127GradientTrainingDesign(),
) -> list[dict[str, float | int | str]]:
    design.validate()
    scale = _scale(cases)
    lookup = _lookup(cache)
    optimizer = torch.optim.AdamW(model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay)
    model.train().to(device)
    history = []
    for epoch in range(1, design.epochs + 1):
        order = list(range(len(cases)))
        np.random.default_rng(42 + epoch).shuffle(order)
        losses, hits = [], []
        for index in order:
            case = cases[index]
            name = lookup.get((str(case["event_id"]), str(case["checkpoint_id"])))
            if name is None:
                raise KeyError("V127 D5 gradient case lacks base causal input group")
            initial, rainfall, flow = _physical(cache, name, normalization, device)
            optimizer.zero_grad(set_to_none=True)
            predicted, hard_center_tfv = predicted_directional_gradient_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                center_sequence=np.asarray(case["center_sequence"]),
                direction_sequence_per_coeff=np.asarray(case["direction_sequence_per_coeff"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
                create_graph=True,
            )
            truth = torch.as_tensor(float(case["true_tfv_gradient"]), dtype=predicted.dtype, device=device)
            magnitude = F.smooth_l1_loss(predicted / scale, truth / scale, beta=0.5)
            sign_loss = F.softplus(-torch.sign(truth) * predicted / scale)
            hard_truth = torch.as_tensor(float(case["center_tfv_m3"]), dtype=hard_center_tfv.dtype, device=device)
            hard_scale = max(abs(float(case["center_tfv_m3"])), 1000.0)
            center_loss = F.smooth_l1_loss(
                (hard_center_tfv - hard_truth) / hard_scale,
                torch.zeros_like(hard_center_tfv),
                beta=0.5,
            )
            loss = (
                design.magnitude_weight * magnitude
                + design.sign_weight * sign_loss
                + design.hard_center_tfv_weight * center_loss
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V127 D5 gradient loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            hits.append(float(torch.sign(predicted.detach()) == torch.sign(truth)))
        row = {
            "stage": "D5_gradient_finetune",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_sign_accuracy": float(np.mean(hits)),
            "gradient_scale_m3_per_coeff": scale,
        }
        history.append(row)
        print("[V127_D5_GRADIENT] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


def evaluate_d5_gradients_v127(
    model,
    *,
    cache: Any,
    cases: Sequence[dict[str, Any]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    lookup = _lookup(cache)
    rows = []
    model.eval().to(device)
    for case in cases:
        name = lookup.get((str(case["event_id"]), str(case["checkpoint_id"])))
        if name is None:
            raise KeyError("V127 D5 audit case lacks base causal input group")
        initial, rainfall, flow = _physical(cache, name, normalization, device)
        with torch.enable_grad():
            predicted, _ = predicted_directional_gradient_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                center_sequence=np.asarray(case["center_sequence"]),
                direction_sequence_per_coeff=np.asarray(case["direction_sequence_per_coeff"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
                create_graph=False,
            )
        rows.append({
            "rainfall_group": str(case["rainfall_group"]),
            "event_id": str(case["event_id"]),
            "checkpoint_id": str(case["checkpoint_id"]),
            "center_id": str(case["center_id"]),
            "direction_id": str(case["direction_id"]),
            "true_tfv_gradient_m3_per_coeff": float(case["true_tfv_gradient"]),
            "predicted_tfv_gradient_m3_per_coeff": float(predicted.detach()),
        })
    detail = pd.DataFrame(rows)
    per_rain = []
    for _, group in detail.groupby("rainfall_group", sort=True):
        truth = group["true_tfv_gradient_m3_per_coeff"].to_numpy(float)
        pred = group["predicted_tfv_gradient_m3_per_coeff"].to_numpy(float)
        mask = np.abs(truth) > 1e-8
        sign = float(np.mean(np.sign(pred[mask]) == np.sign(truth[mask]))) if mask.any() else float("nan")
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        cosine = float(np.dot(truth, pred) / denom) if denom > 1e-12 else float("nan")
        per_rain.append((sign, cosine, float(np.mean(np.abs(pred - truth)))))
    array = np.asarray(per_rain, dtype=float)
    if array.size == 0 or not np.isfinite(array[:, :2]).any(axis=0).all():
        raise RuntimeError("V127 D5 gradient audit cannot produce finite sign/cosine metrics")
    metrics = {
        "gradient_cases": float(len(detail)),
        "gradient_rainfall_groups": float(detail["rainfall_group"].nunique()),
        "tfv_gradient_sign_accuracy": float(np.nanmean(array[:, 0])),
        "tfv_gradient_cosine_similarity": float(np.nanmean(array[:, 1])),
        "tfv_gradient_mae": float(np.nanmean(array[:, 2])),
    }
    return detail, metrics


__all__ = [
    "V127GradientTrainingDesign",
    "V127_GRADIENT_AUDIT_CONTRACT",
    "V127_GRADIENT_TRAINING_CONTRACT",
    "build_direction_cases_v127",
    "evaluate_d5_gradients_v127",
    "predicted_directional_gradient_v127",
    "train_d5_gradient_v127",
]
