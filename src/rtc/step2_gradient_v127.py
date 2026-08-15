"""D5 directional-gradient supervision/audit for the V127 smooth MPC objective.

D5-FIT training uses authoritative symmetric +/- SWMM pairs and therefore can match the
model's symmetric smooth-TFV finite difference without second-order parameter gradients.
D5-AUDIT evaluates the exact quantity the online solver consumes: autograd with respect to
the 12 x 109 bounded fraction tensor, through the same sequential target decoder used by
L-BFGS-B, projected onto the frozen D5 fraction-space direction.
"""
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
from .step2_v60_contract import require_feature
from .step3_mpc_v127 import ContinuousMPCDesignV127, decode_fractional_targets_v127

V127_GRADIENT_TRAINING_CONTRACT = "PROJECT7_V127_D5_1308VAR_SYMMETRIC_FIRST_ORDER_GRADIENT_SUPERVISION_V4"
V127_GRADIENT_AUDIT_CONTRACT = "PROJECT7_V127_D5_1308VAR_AUTOGRAD_AUDIT_V4_SMOOTH_OBJECTIVE"


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
    values = np.asarray(
        [[float(block[aid]) for aid in actuator_ids] for block in blocks], dtype=np.float32
    )
    sequence = np.repeat(values, 2, axis=0)
    if sequence.shape != (72, len(actuator_ids)) or not np.isfinite(sequence).all():
        raise ValueError("V127 D5 expanded action sequence is invalid")
    return sequence


def _matrix(raw: object, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    value = np.asarray(json.loads(str(raw)), dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"V127 D5 {label} shape/content is invalid")
    return value


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
    manifest = execution_manifest[
        execution_manifest["d5_split_role"].astype(str) == role
    ]
    actuator_count = len(actuator_ids)
    if actuator_count != 109:
        raise ValueError("V127 D5 gradient cases require the frozen 109 actuators")
    cases: list[dict[str, Any]] = []
    for _, label in labels.iterrows():
        center_id = str(label["center_id"])
        direction_id = str(label["direction_id"])
        center = manifest[
            (manifest["center_id"].astype(str) == center_id)
            & (manifest["probe_role"].astype(str) == "center")
        ]
        plus = manifest[
            (manifest["direction_id"].astype(str) == direction_id)
            & (manifest["probe_role"].astype(str) == "plus")
        ]
        minus = manifest[
            (manifest["direction_id"].astype(str) == direction_id)
            & (manifest["probe_role"].astype(str) == "minus")
        ]
        if len(center) != 1 or len(plus) != 1 or len(minus) != 1:
            raise RuntimeError(
                f"V127 D5 {direction_id} lacks exactly one center/+/- branch"
            )
        eps = float(label["epsilon"])
        if not math.isfinite(eps) or eps <= 0:
            raise RuntimeError("V127 D5 epsilon must be finite and positive")
        center_row, plus_row, minus_row = center.iloc[0], plus.iloc[0], minus.iloc[0]
        center_seq = _sequence(center_row, actuator_ids)
        plus_seq = _sequence(plus_row, actuator_ids)
        minus_seq = _sequence(minus_row, actuator_ids)
        center_fraction = _matrix(
            center_row["center_fractions_json"],
            (12, actuator_count),
            label="center fractions",
        )
        plus_fraction = _matrix(
            plus_row["probe_fractions_json"],
            (12, actuator_count),
            label="plus fractions",
        )
        minus_fraction = _matrix(
            minus_row["probe_fractions_json"],
            (12, actuator_count),
            label="minus fractions",
        )
        direction_fraction = _matrix(
            plus_row["direction_fractions_json"],
            (12, actuator_count),
            label="direction fractions",
        )
        if not np.allclose(
            direction_fraction,
            _matrix(
                minus_row["direction_fractions_json"],
                (12, actuator_count),
                label="minus direction fractions",
            ),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("V127 D5 +/- rows do not share the same fraction direction")
        if abs(float(np.linalg.norm(direction_fraction)) - 1.0) > 5e-5:
            raise RuntimeError("V127 D5 fraction direction is not unit-L2")
        if float(np.max(np.abs((plus_fraction - minus_fraction) / (2.0 * eps) - direction_fraction))) > 2.0e-5:
            raise RuntimeError("V127 D5 fraction pair does not encode the frozen direction")
        if float(np.max(np.abs(0.5 * (plus_fraction + minus_fraction) - center_fraction))) > 2.0e-6:
            raise RuntimeError("V127 D5 fraction pair lost central symmetry")
        midpoint_error = float(
            np.max(np.abs(0.5 * (plus_seq + minus_seq) - center_seq))
        )
        if midpoint_error > 2.5e-5:
            raise RuntimeError(
                f"V127 D5 {direction_id} lost decoded-target central symmetry ({midpoint_error})"
            )
        active_target = _matrix(
            center_row["active_target_json"],
            (actuator_count,),
            label="active target",
        )
        cases.append(
            {
                "rainfall_group": str(label["rainfall_group"]),
                "event_id": str(label["event_id"]),
                "checkpoint_id": str(label["checkpoint_id"]),
                "center_id": center_id,
                "center_family": str(label.get("center_family", "")),
                "direction_id": direction_id,
                "direction_family": str(label.get("direction_family", "")),
                "epsilon": eps,
                "active_target": active_target,
                "center_fractions": center_fraction,
                "plus_fractions": plus_fraction,
                "minus_fractions": minus_fraction,
                "direction_fractions": direction_fraction,
                "center_sequence": center_seq,
                "plus_sequence": plus_seq,
                "minus_sequence": minus_seq,
                "true_tfv_gradient": float(
                    label["true_tfv_directional_gradient_m3_per_fraction"]
                ),
                "center_tfv_m3": float(label["center_tfv_m3"]),
            }
        )
    if not cases:
        raise ValueError(f"V127 D5 has no {role} gradient cases")
    return cases


def _lookup(cache: Any) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for name in cache.names():
        entry = cache.entry(name)
        key = (str(entry.event_id), str(entry.checkpoint_id))
        previous = result.get(key)
        if previous is None or (
            name.startswith("D2::") and not previous.startswith("D2::")
        ):
            result[key] = name
    return result


def _physical(
    cache: Any,
    name: str,
    normalization: InputNormalizationV60,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = cache.batch(name, normalization, device)
    dtype = batch.initial_state.dtype

    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device)

    return (
        batch.initial_state * tensor(normalization.state_std).clamp_min(1e-6)
        + tensor(normalization.state_mean),
        batch.rainfall * tensor(normalization.rainfall_std).clamp_min(1e-6)
        + tensor(normalization.rainfall_mean),
        batch.previous_actuator_flow * tensor(normalization.flow_std).clamp_min(1e-6)
        + tensor(normalization.flow_mean),
    )


def _objective_output(
    model,
    *,
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    settings: torch.Tensor,
    graph: Any,
    flood_rate_index: int,
):
    device = initial_state.device
    return model.objective_rollout(
        initial_state=initial_state,
        rainfall=rainfall,
        settings=settings,
        previous_actuator_flow=previous_flow,
        actuator_upstream=torch.as_tensor(
            graph.actuator_upstream, dtype=torch.long, device=device
        ),
        actuator_downstream=torch.as_tensor(
            graph.actuator_downstream, dtype=torch.long, device=device
        ),
        actuator_physics=torch.as_tensor(
            graph.actuator_physics, dtype=initial_state.dtype, device=device
        ),
        static_node_features=torch.as_tensor(
            graph.static_node_features, dtype=initial_state.dtype, device=device
        ),
        edge_index=torch.as_tensor(
            graph.edge_index, dtype=torch.long, device=device
        ),
        flood_rate_index=int(flood_rate_index),
        priority_indices=None,
        dt_seconds=300.0,
    )


def _setting_bounds(graph: Any, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    names = tuple(graph.actuator_physics_feature_names)
    physics = np.asarray(graph.actuator_physics, dtype=np.float32)
    lo = physics[:, require_feature(names, "min_setting")]
    hi = physics[:, require_feature(names, "max_setting")]
    return (
        torch.as_tensor(lo, dtype=dtype, device=device),
        torch.as_tensor(hi, dtype=dtype, device=device),
    )


def predicted_directional_gradient_v127(
    model,
    *,
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    center_fractions: np.ndarray,
    direction_fractions: np.ndarray,
    active_target: np.ndarray,
    graph: Any,
    flood_rate_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact autograd quantity consumed by the online L-BFGS-B objective."""
    device = initial_state.device
    lo, hi = _setting_bounds(graph, dtype=initial_state.dtype, device=device)
    fractions = torch.as_tensor(
        center_fractions, dtype=initial_state.dtype, device=device
    ).requires_grad_(True)
    direction = torch.as_tensor(
        direction_fractions, dtype=initial_state.dtype, device=device
    )
    mpc_design = ContinuousMPCDesignV127(
        min_improvement_vs_rbc_m3=0.0,
        movement_penalty_m3=0.0,
    )
    sequence = decode_fractional_targets_v127(
        fractions,
        active_target=torch.as_tensor(
            active_target, dtype=initial_state.dtype, device=device
        ),
        min_setting=lo,
        max_setting=hi,
        design=mpc_design,
    )
    output = _objective_output(
        model,
        initial_state=initial_state,
        rainfall=rainfall,
        previous_flow=previous_flow,
        settings=sequence[None],
        graph=graph,
        flood_rate_index=flood_rate_index,
    )
    fraction_gradient = torch.autograd.grad(
        output.optimization_tfv_m3.sum(), fractions
    )[0]
    return torch.sum(fraction_gradient * direction), output.tfv_m3.sum()


def _predicted_symmetric_difference_v127(
    model,
    *,
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    plus_sequence: np.ndarray,
    minus_sequence: np.ndarray,
    epsilon: float,
    graph: Any,
    flood_rate_index: int,
) -> torch.Tensor:
    """First-order-in-parameters D5 FIT target; avoids H360 second-order autograd."""
    device = initial_state.device
    settings = torch.as_tensor(
        np.stack((plus_sequence, minus_sequence)),
        dtype=initial_state.dtype,
        device=device,
    )
    state = initial_state.expand(2, -1, -1)
    rain = rainfall.expand(2, -1, -1, -1)
    flow = previous_flow.expand(2, -1)
    output = _objective_output(
        model,
        initial_state=state,
        rainfall=rain,
        previous_flow=flow,
        settings=settings,
        graph=graph,
        flood_rate_index=flood_rate_index,
    )
    return (
        output.optimization_tfv_m3[0] - output.optimization_tfv_m3[1]
    ) / (2.0 * epsilon)


def _hard_center_tfv_v127(
    model,
    *,
    initial_state: torch.Tensor,
    rainfall: torch.Tensor,
    previous_flow: torch.Tensor,
    center_sequence: np.ndarray,
    graph: Any,
    flood_rate_index: int,
) -> torch.Tensor:
    settings = torch.as_tensor(
        center_sequence, dtype=initial_state.dtype, device=initial_state.device
    )[None]
    return _objective_output(
        model,
        initial_state=initial_state,
        rainfall=rainfall,
        previous_flow=previous_flow,
        settings=settings,
        graph=graph,
        flood_rate_index=flood_rate_index,
    ).tfv_m3.sum()


def _scale(cases: Sequence[dict[str, Any]]) -> float:
    values = np.abs(
        np.asarray(
            [float(case["true_tfv_gradient"]) for case in cases], dtype=float
        )
    )
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay
    )
    model.train().to(device)
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, design.epochs + 1):
        order = list(range(len(cases)))
        np.random.default_rng(42 + epoch).shuffle(order)
        losses: list[float] = []
        hits: list[float] = []
        for index in order:
            case = cases[index]
            name = lookup.get(
                (str(case["event_id"]), str(case["checkpoint_id"]))
            )
            if name is None:
                raise KeyError("V127 D5 gradient case lacks base causal input group")
            initial, rainfall, flow = _physical(
                cache, name, normalization, device
            )
            optimizer.zero_grad(set_to_none=True)
            predicted = _predicted_symmetric_difference_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                plus_sequence=np.asarray(case["plus_sequence"]),
                minus_sequence=np.asarray(case["minus_sequence"]),
                epsilon=float(case["epsilon"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
            )
            hard_center_tfv = _hard_center_tfv_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                center_sequence=np.asarray(case["center_sequence"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
            )
            truth = torch.as_tensor(
                float(case["true_tfv_gradient"]),
                dtype=predicted.dtype,
                device=device,
            )
            magnitude = F.smooth_l1_loss(
                predicted / scale, truth / scale, beta=0.5
            )
            # A zero physical gradient carries no directional sign information; in that
            # case the magnitude term alone teaches the local flatness.
            if abs(float(case["true_tfv_gradient"])) > 1e-8:
                sign_loss = F.softplus(-torch.sign(truth) * predicted / scale)
            else:
                sign_loss = predicted.new_zeros(())
            hard_truth = torch.as_tensor(
                float(case["center_tfv_m3"]),
                dtype=hard_center_tfv.dtype,
                device=device,
            )
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
            if abs(float(case["true_tfv_gradient"])) > 1e-8:
                hits.append(
                    float(torch.sign(predicted.detach()) == torch.sign(truth))
                )
        row: dict[str, float | int | str] = {
            "stage": "D5_1308var_symmetric_gradient_finetune",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_sign_accuracy": float(np.mean(hits)) if hits else float("nan"),
            "gradient_scale_m3_per_fraction": scale,
            "second_order_parameter_backprop": "false",
        }
        history.append(row)
        print(
            "[V127_D5_GRADIENT] "
            + " ".join(f"{k}={v}" for k, v in row.items()),
            flush=True,
        )
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
    rows: list[dict[str, Any]] = []
    model.eval().to(device)
    for case in cases:
        name = lookup.get(
            (str(case["event_id"]), str(case["checkpoint_id"]))
        )
        if name is None:
            raise KeyError("V127 D5 audit case lacks base causal input group")
        initial, rainfall, flow = _physical(cache, name, normalization, device)
        with torch.enable_grad():
            predicted, _ = predicted_directional_gradient_v127(
                model,
                initial_state=initial,
                rainfall=rainfall,
                previous_flow=flow,
                center_fractions=np.asarray(case["center_fractions"]),
                direction_fractions=np.asarray(case["direction_fractions"]),
                active_target=np.asarray(case["active_target"]),
                graph=graph,
                flood_rate_index=flood_rate_index,
            )
        rows.append(
            {
                "rainfall_group": str(case["rainfall_group"]),
                "event_id": str(case["event_id"]),
                "checkpoint_id": str(case["checkpoint_id"]),
                "center_id": str(case["center_id"]),
                "center_family": str(case.get("center_family", "")),
                "direction_id": str(case["direction_id"]),
                "direction_family": str(case.get("direction_family", "")),
                "true_tfv_gradient_m3_per_fraction": float(
                    case["true_tfv_gradient"]
                ),
                "predicted_tfv_gradient_m3_per_fraction": float(
                    predicted.detach()
                ),
            }
        )
    detail = pd.DataFrame(rows)
    per_rain: list[tuple[float, float, float]] = []
    for _, group in detail.groupby("rainfall_group", sort=True):
        truth = group["true_tfv_gradient_m3_per_fraction"].to_numpy(dtype=float)
        pred = group["predicted_tfv_gradient_m3_per_fraction"].to_numpy(dtype=float)
        mask = np.abs(truth) > 1e-8
        sign = (
            float(np.mean(np.sign(pred[mask]) == np.sign(truth[mask])))
            if mask.any()
            else float("nan")
        )
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        cosine = (
            float(np.dot(truth, pred) / denom)
            if denom > 1e-12
            else float("nan")
        )
        per_rain.append(
            (sign, cosine, float(np.mean(np.abs(pred - truth))))
        )
    array = np.asarray(per_rain, dtype=float)
    if array.size == 0 or not np.isfinite(array[:, :2]).any(axis=0).all():
        raise RuntimeError(
            "V127 D5 gradient audit cannot produce finite sign/cosine metrics"
        )
    metrics = {
        "gradient_cases": float(len(detail)),
        "gradient_rainfall_groups": float(detail["rainfall_group"].nunique()),
        "tfv_gradient_sign_accuracy": float(np.nanmean(array[:, 0])),
        "tfv_gradient_cosine_similarity": float(np.nanmean(array[:, 1])),
        "tfv_gradient_mae_m3_per_fraction": float(np.nanmean(array[:, 2])),
        "gradient_variable_space": "exact online 12x109 L-BFGS-B fraction tensor",
        "audit_prediction_semantics": "smooth-TFV autograd through online V127 decoder",
        "training_prediction_semantics": "smooth-TFV symmetric finite difference on D5-FIT only",
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
