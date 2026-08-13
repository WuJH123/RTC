"""Train-only D2 endpoint-local hydraulic-effect learnability baselines.

This is deliberately a diagnostic, not a replacement Step2 surrogate.  It asks a
minimal question before any graph or history change: can the frozen D2 cache support
signed action-effect learning at the changed actuator's two endpoints?  The four
fixed baselines are zero, per-actuator signed mean, action-only ridge, and a small
local MLP.  They never open Validation/Final/Formal assets or invoke SWMM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from rtc.graph import GraphSchema
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v80_contract import DirectHydraulicEffectLossContractV80


BASELINE_CONTRACT = "PROJECT7_STEP2_D2_ENDPOINT_LOCAL_LEARNABILITY_BASELINES_V1"
STATE_CHANNELS = (0, 2, 3)  # depth, flooding rate, storage volume
STATE_LABELS = ("delta_depth_m", "delta_flood_m3s", "delta_storage_m3")
FLOW_LABEL = "delta_managed_flow_m3s"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def _finite_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def event_balanced_effect_metrics(
    event_arrays: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    active_fraction: float,
) -> dict[str, float | int]:
    """Return event-balanced signed-effect metrics with elementwise scales.

    ``scale`` must broadcast against each event's truth.  This is important for
    managed flow: its sparse-active mask is defined using the changed actuator's
    own TrainFit RMS scale, never a global median across 109 actuators.
    """
    per_event: list[dict[str, float]] = []
    for event, payload in sorted(event_arrays.items()):
        predicted, truth, scale = (np.asarray(value, dtype=np.float64) for value in payload)
        try:
            scale = np.broadcast_to(scale, truth.shape)
            predicted = np.broadcast_to(predicted, truth.shape)
        except ValueError as exc:
            raise ValueError(f"{event}: prediction/truth/scale are not broadcast-compatible") from exc
        finite = np.isfinite(predicted) & np.isfinite(truth) & np.isfinite(scale)
        if not finite.any():
            continue
        predicted, truth, scale = predicted[finite], truth[finite], scale[finite]
        scale = np.maximum(np.abs(scale), 1e-12)
        error_mse = float(np.mean(np.square(predicted - truth)))
        zero_mse = float(np.mean(np.square(truth)))
        active = np.abs(truth) / scale >= float(active_fraction)
        active_error_mse = float(np.mean(np.square(predicted[active] - truth[active]))) if active.any() else float("nan")
        active_zero_mse = float(np.mean(np.square(truth[active]))) if active.any() else float("nan")
        per_event.append(
            {
                "skill_vs_zero": 1.0 - error_mse / max(zero_mse, 1e-12),
                "response_ratio": float(np.mean(np.abs(predicted))) / max(float(np.mean(np.abs(truth))), 1e-12),
                "rmse": float(np.sqrt(error_mse)),
                "active_skill_vs_zero": (
                    1.0 - active_error_mse / max(active_zero_mse, 1e-12)
                    if np.isfinite(active_error_mse)
                    else float("nan")
                ),
                "active_sign_accuracy": (
                    float(np.mean(np.sign(predicted[active]) == np.sign(truth[active])))
                    if active.any()
                    else float("nan")
                ),
                "active_fraction": float(np.mean(active)),
                "sample_count": float(truth.size),
            }
        )
    if not per_event:
        return {"events": 0, "sample_count": 0, "skill_vs_zero": float("nan")}
    result: dict[str, float | int] = {
        "events": int(len(per_event)),
        "sample_count": int(sum(row["sample_count"] for row in per_event)),
    }
    for key in (
        "skill_vs_zero",
        "response_ratio",
        "rmse",
        "active_skill_vs_zero",
        "active_sign_accuracy",
        "active_fraction",
    ):
        values = np.asarray([row[key] for row in per_event], dtype=np.float64)
        result[key] = _finite_float(float(np.mean(values[np.isfinite(values)]))) if np.isfinite(values).any() else float("nan")
    return result


def local_feature_row(
    *,
    actuator_index: int,
    actuator_count: int,
    current_delta: float,
    prefix_delta: float,
    up_state: np.ndarray,
    down_state: np.ndarray,
    previous_flow: float,
    rainfall_up: float,
    rainfall_down: float,
    rainfall_mean: float,
    retained_minutes: float,
    physics: np.ndarray,
) -> np.ndarray:
    """One causal endpoint-local feature row; identity occupies the leading block."""
    if not 0 <= int(actuator_index) < int(actuator_count):
        raise ValueError("actuator index is outside the frozen actuator catalogue")
    up = np.asarray(up_state, dtype=np.float64).reshape(-1)
    down = np.asarray(down_state, dtype=np.float64).reshape(-1)
    physical = np.asarray(physics, dtype=np.float64).reshape(-1)
    if up.size != 6 or down.size != 6:
        raise ValueError("local baseline requires six-channel upstream/downstream state")
    identity = np.zeros(int(actuator_count), dtype=np.float64)
    identity[int(actuator_index)] = 1.0
    # Retained horizon timestamps are prediction end-times in minutes; this does
    # not read future hydraulic truth and the action prefix is already causal.
    values = np.concatenate(
        (
            identity,
            np.asarray(
                (
                    float(current_delta),
                    float(prefix_delta),
                ),
                dtype=np.float64,
            ),
            up,
            down,
            np.asarray(
                (
                    float(previous_flow),
                    float(rainfall_up),
                    float(rainfall_down),
                    float(rainfall_mean),
                    float(retained_minutes) / 360.0,
                ),
                dtype=np.float64,
            ),
            physical,
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("local baseline input contains non-finite causal values")
    return values


def _single_changed_actuator(
    candidate_settings: np.ndarray,
    reference_settings: np.ndarray,
) -> int:
    changed = np.flatnonzero(
        np.any(np.abs(candidate_settings - reference_settings) > 1e-8, axis=0)
    )
    if changed.size != 1:
        raise ValueError(
            "D2 local baseline requires exactly one changed actuator; "
            f"observed {changed.size}"
        )
    return int(changed.item())


def _actuator_kind(graph: GraphSchema, actuator_index: int) -> str:
    physics = np.asarray(graph.actuator_physics[actuator_index], dtype=np.float64)
    for index, label in ((0, "pump"), (1, "orifice"), (2, "weir")):
        if index < physics.size and physics[index] > 0.5:
            return label
    return "other"


def build_d2_endpoint_dataset(
    cache: V60TrainCache,
    names: Sequence[str],
    graph: GraphSchema,
    *,
    horizon: MultiResolutionHorizonV60 | None = None,
) -> dict[str, np.ndarray]:
    """Materialize the fixed Train-only D2 endpoint-local diagnostic table.

    Every target is raw signed candidate-minus-reference physical output at a
    retained future time.  No candidate ID, graph message, model prediction, or
    future truth enters the input features.
    """
    if not names:
        raise ValueError("cannot materialize a dataset from no groups")
    horizon = horizon or MultiResolutionHorizonV60()
    retained = np.asarray(horizon.indices(), dtype=np.int64)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    events: list[str] = []
    actuators: list[int] = []
    times: list[int] = []
    types: list[str] = []
    expected_feature_dim: int | None = None

    for name in sorted(names):
        entry = cache.entry(name)
        if entry.source_kind != "D2":
            raise ValueError(f"{name}: local baseline is D2-only")
        arrays, ref = entry.arrays, entry.reference_index
        initial = np.asarray(arrays["initial_state"][ref], dtype=np.float64)
        rainfall = np.asarray(arrays["rainfall"][ref], dtype=np.float64)[..., 0]
        previous_flow = np.asarray(arrays["previous_actuator_flow"][ref], dtype=np.float64)
        reference_settings = np.asarray(arrays["settings"][ref], dtype=np.float64)
        reference_state = np.asarray(arrays["target_states"][ref], dtype=np.float64)
        reference_flow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)
        event = f"{entry.rainfall_group}::{entry.event_id}"

        for candidate in entry.indices:
            if candidate == ref:
                continue
            candidate_settings = np.asarray(arrays["settings"][candidate], dtype=np.float64)
            actuator = _single_changed_actuator(candidate_settings, reference_settings)
            delta_action = candidate_settings[:, actuator] - reference_settings[:, actuator]
            candidate_state = np.asarray(arrays["target_states"][candidate], dtype=np.float64)
            candidate_flow = np.asarray(arrays["target_actuator_flows"][candidate], dtype=np.float64)
            upstream, downstream = (
                int(graph.actuator_upstream[actuator]),
                int(graph.actuator_downstream[actuator]),
            )
            kind = _actuator_kind(graph, actuator)
            for slot, raw_index in enumerate(retained.tolist()):
                prefix = float(np.mean(delta_action[: raw_index + 1]))
                feature = local_feature_row(
                    actuator_index=actuator,
                    actuator_count=len(graph.actuator_ids),
                    current_delta=float(delta_action[raw_index]),
                    prefix_delta=prefix,
                    up_state=initial[upstream],
                    down_state=initial[downstream],
                    previous_flow=float(previous_flow[actuator]),
                    rainfall_up=float(np.mean(rainfall[: raw_index + 1, upstream])),
                    rainfall_down=float(np.mean(rainfall[: raw_index + 1, downstream])),
                    rainfall_mean=float(np.mean(rainfall[: raw_index + 1])),
                    retained_minutes=float((raw_index + 1) * horizon.model_step_seconds / 60.0),
                    physics=np.asarray(graph.actuator_physics[actuator], dtype=np.float64),
                )
                target = np.asarray(
                    (
                        candidate_state[raw_index, upstream, 0] - reference_state[raw_index, upstream, 0],
                        candidate_state[raw_index, upstream, 2] - reference_state[raw_index, upstream, 2],
                        candidate_state[raw_index, upstream, 3] - reference_state[raw_index, upstream, 3],
                        candidate_state[raw_index, downstream, 0] - reference_state[raw_index, downstream, 0],
                        candidate_state[raw_index, downstream, 2] - reference_state[raw_index, downstream, 2],
                        candidate_state[raw_index, downstream, 3] - reference_state[raw_index, downstream, 3],
                        candidate_flow[raw_index, actuator] - reference_flow[raw_index, actuator],
                    ),
                    dtype=np.float64,
                )
                if not np.isfinite(target).all():
                    raise ValueError(f"{name}: non-finite raw signed endpoint target")
                if expected_feature_dim is None:
                    expected_feature_dim = int(feature.size)
                elif feature.size != expected_feature_dim:
                    raise ValueError("local feature dimension changed across D2 candidates")
                feature_rows.append(feature)
                target_rows.append(target)
                events.append(event)
                actuators.append(actuator)
                times.append(slot)
                types.append(kind)
    if not feature_rows:
        raise ValueError("D2 endpoint-local materialization produced no candidates")
    return {
        "features": np.stack(feature_rows).astype(np.float32),
        "targets": np.stack(target_rows).astype(np.float64),
        "event": np.asarray(events, dtype=object),
        "actuator": np.asarray(actuators, dtype=np.int64),
        "time_slot": np.asarray(times, dtype=np.int64),
        "actuator_type": np.asarray(types, dtype=object),
        "retained_indices": retained,
    }


def _mean_by_actuator_time(
    targets: np.ndarray,
    actuators: np.ndarray,
    time_slots: np.ndarray,
    *,
    actuator_count: int,
    retained_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((actuator_count, retained_count, targets.shape[1]), dtype=np.float64)
    counts = np.zeros((actuator_count, retained_count), dtype=np.int64)
    for row, actuator, slot in zip(targets, actuators, time_slots, strict=True):
        sums[int(actuator), int(slot)] += row
        counts[int(actuator), int(slot)] += 1
    global_time = np.zeros((retained_count, targets.shape[1]), dtype=np.float64)
    for slot in range(retained_count):
        mask = time_slots == slot
        if mask.any():
            global_time[slot] = np.mean(targets[mask], axis=0)
    means = np.divide(
        sums,
        counts[..., None],
        out=np.broadcast_to(global_time[None], sums.shape).copy(),
        where=counts[..., None] > 0,
    )
    return means, counts


def _action_ridge_features(
    actuators: np.ndarray,
    features: np.ndarray,
    *,
    actuator_count: int,
) -> np.ndarray:
    """Fixed action-only ridge features: identity plus identity × current/prefix delta."""
    if features.shape[1] < actuator_count + 2:
        raise ValueError("local feature table has no action terms after actuator identity")
    rows = features.shape[0]
    out = np.zeros((rows, actuator_count * 3), dtype=np.float64)
    row = np.arange(rows)
    current, prefix = features[:, actuator_count], features[:, actuator_count + 1]
    out[row, actuators] = 1.0
    out[row, actuator_count + actuators] = current
    out[row, 2 * actuator_count + actuators] = prefix
    return out


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    """Closed-form fixed ridge with no data-dependent tuning."""
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge features/targets have incompatible shapes")
    if ridge_lambda <= 0:
        raise ValueError("ridge lambda must be strictly positive")
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += float(ridge_lambda)
    return np.linalg.solve(gram, x.T @ y)


class _LocalMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def fit_local_mlp(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    prediction_feature_sets: Sequence[np.ndarray],
    *,
    seed: int,
    epochs: int,
    device: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Fixed 4-epoch small local MLP, fit only on TrainFit endpoint samples."""
    if epochs != 4:
        raise ValueError("local learnability MLP epoch budget is frozen at four")
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_mean = np.mean(fit_features, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(fit_features, axis=0, dtype=np.float64), 1e-6)
    y_scale = np.maximum(
        np.sqrt(np.mean(np.square(fit_targets), axis=0, dtype=np.float64)), 1e-6
    )
    fit_x = torch.as_tensor((fit_features - x_mean) / x_std, dtype=torch.float32, device=target)
    fit_y = torch.as_tensor(fit_targets / y_scale, dtype=torch.float32, device=target)
    model = _LocalMLP(fit_x.shape[1], fit_y.shape[1]).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float | int]] = []
    batch_size = 256
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(fit_x.shape[0], generator=generator)
        losses: list[float] = []
        model.train()
        for begin in range(0, fit_x.shape[0], batch_size):
            indices = permutation[begin : begin + batch_size].to(target)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(fit_x.index_select(0, indices))
            loss = torch.nn.functional.smooth_l1_loss(prediction, fit_y.index_select(0, indices), beta=0.5)
            if not torch.isfinite(loss):
                raise FloatingPointError("local MLP baseline reached a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "normalized_smooth_l1": float(np.mean(losses))})
    model.eval()
    with torch.no_grad():
        predictions = [
            model(
                torch.as_tensor(
                    (features - x_mean) / x_std, dtype=torch.float32, device=target
                )
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            * y_scale
            for features in prediction_feature_sets
        ]
    return predictions, {
        "architecture": "identity+action-prefix+endpoint-state+previous-flow+rainfall+time+physics -> 64 -> 64 -> 7",
        "epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": batch_size,
        "device": str(target),
        "history": history,
        "feature_normalization": "TrainFit only, per feature mean/std",
        "target_normalization": "TrainFit only, per output RMS",
    }


def _event_arrays_for_channel(
    dataset: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    *,
    target_indices: Sequence[int],
    scale: np.ndarray | float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    events = np.asarray(dataset["event"], dtype=object)
    targets = np.asarray(dataset["targets"], dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    for event in sorted(set(events.tolist())):
        rows = events == event
        truth = targets[rows][:, list(target_indices)]
        values = predicted[rows][:, list(target_indices)]
        if np.isscalar(scale):
            scales = np.full(truth.shape, float(scale), dtype=np.float64)
        else:
            base = np.asarray(scale, dtype=np.float64)
            if base.shape == truth.shape:
                scales = base
            elif base.ndim == 1 and base.shape[0] == targets.shape[0]:
                scales = base[rows, None]
            else:
                raise ValueError("channel scale cannot be aligned to endpoint rows")
        output[str(event)] = (values.reshape(-1), truth.reshape(-1), scales.reshape(-1))
    return output


def evaluate_endpoint_predictions(
    dataset: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    *,
    state_delta_scale: np.ndarray,
    flow_delta_scale: np.ndarray,
    active_fraction: float,
) -> dict[str, Any]:
    """Event-balanced endpoint-local D2 scores for all four primary channels."""
    state_scale = np.asarray(state_delta_scale, dtype=np.float64)
    flow_scale = np.asarray(flow_delta_scale, dtype=np.float64)
    result: dict[str, Any] = {
        "scientific_primary": "event_balanced_endpoint_local_raw_signed_candidate_minus_reference_effect",
        "active_effect_fraction": float(active_fraction),
        "channels": {},
    }
    for label, state_channel, output_indices in (
        (STATE_LABELS[0], STATE_CHANNELS[0], (0, 3)),
        (STATE_LABELS[1], STATE_CHANNELS[1], (1, 4)),
        (STATE_LABELS[2], STATE_CHANNELS[2], (2, 5)),
    ):
        result["channels"][label] = event_balanced_effect_metrics(
            _event_arrays_for_channel(
                dataset,
                prediction,
                target_indices=output_indices,
                scale=float(state_scale[state_channel]),
            ),
            active_fraction=active_fraction,
        )
    actuator = np.asarray(dataset["actuator"], dtype=np.int64)
    result["channels"][FLOW_LABEL] = event_balanced_effect_metrics(
        _event_arrays_for_channel(
            dataset,
            prediction,
            target_indices=(6,),
            scale=flow_scale[actuator],
        ),
        active_fraction=active_fraction,
    )
    result["events"] = int(len(set(np.asarray(dataset["event"], dtype=object).tolist())))
    result["rows"] = int(np.asarray(dataset["targets"]).shape[0])
    return result


def _summarize_dataset(dataset: Mapping[str, np.ndarray]) -> dict[str, Any]:
    types = np.asarray(dataset["actuator_type"], dtype=object)
    return {
        "rows": int(np.asarray(dataset["targets"]).shape[0]),
        "events": int(len(set(np.asarray(dataset["event"], dtype=object).tolist()))),
        "unique_actuators": int(len(set(np.asarray(dataset["actuator"], dtype=np.int64).tolist()))),
        "retained_horizon_points": int(len(np.asarray(dataset["retained_indices"]))),
        "actuator_type_rows": {str(kind): int(np.sum(types == kind)) for kind in sorted(set(types.tolist()))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 Train-only D2 local-effect learnability baselines")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("local learnability baseline seed is frozen at 42")
    if abs(float(args.holdout_fraction) - 0.20) > 1e-12:
        raise ValueError("local learnability baseline holdout fraction is frozen at 0.20")

    started = time.perf_counter()
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError(f"expected frozen 109-actuator graph, got {len(graph.actuator_ids)}")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("diagnostic refuses a cache containing legacy dense D3")
    selected = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    if not fit_d2 or not holdout_d2:
        raise ValueError("deterministic split yielded an empty D2 side")
    fit_events = {f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}" for name in fit_d2}
    holdout_events = {f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}" for name in holdout_d2}
    if fit_events & holdout_events:
        raise ValueError("D2 local baseline detected event leakage across deterministic split")

    # Target scales deliberately include the frozen TrainFit D2+D3 corpus, matching
    # the V7/V9 TrainFit-only scale contract.  No holdout target participates.
    scales = derive_target_scales_v70(cache, fit)
    active_fraction = float(DirectHydraulicEffectLossContractV80().active_effect_fraction)
    fit_data = build_d2_endpoint_dataset(cache, fit_d2, graph)
    holdout_data = build_d2_endpoint_dataset(cache, holdout_d2, graph)
    output_dim = int(fit_data["targets"].shape[1])

    zero_fit = np.zeros_like(fit_data["targets"], dtype=np.float64)
    zero_holdout = np.zeros_like(holdout_data["targets"], dtype=np.float64)

    mean_table, mean_counts = _mean_by_actuator_time(
        fit_data["targets"],
        fit_data["actuator"],
        fit_data["time_slot"],
        actuator_count=len(graph.actuator_ids),
        retained_count=len(fit_data["retained_indices"]),
    )
    def mean_prediction(data: Mapping[str, np.ndarray]) -> np.ndarray:
        return mean_table[
            np.asarray(data["actuator"], dtype=np.int64),
            np.asarray(data["time_slot"], dtype=np.int64),
        ]
    mean_fit, mean_holdout = mean_prediction(fit_data), mean_prediction(holdout_data)

    ridge_lambda = 1e-3
    ridge_weights = fit_ridge(
        _action_ridge_features(fit_data["actuator"], fit_data["features"], actuator_count=len(graph.actuator_ids)),
        fit_data["targets"],
        ridge_lambda=ridge_lambda,
    )
    def ridge_prediction(data: Mapping[str, np.ndarray]) -> np.ndarray:
        return _action_ridge_features(
            data["actuator"], data["features"], actuator_count=len(graph.actuator_ids)
        ) @ ridge_weights
    ridge_fit, ridge_holdout = ridge_prediction(fit_data), ridge_prediction(holdout_data)

    (mlp_fit, mlp_holdout), mlp_fit_meta = fit_local_mlp(
        fit_data["features"],
        fit_data["targets"],
        (fit_data["features"], holdout_data["features"]),
        seed=args.seed,
        epochs=4,
        device=args.device,
    )

    def evaluate(fit_prediction: np.ndarray, holdout_prediction: np.ndarray) -> dict[str, Any]:
        return {
            "TrainFit_D2": evaluate_endpoint_predictions(
                fit_data,
                fit_prediction,
                state_delta_scale=scales.state_delta_scale,
                flow_delta_scale=scales.flow_delta_scale,
                active_fraction=active_fraction,
            ),
            "TrainInternalHoldout_D2": evaluate_endpoint_predictions(
                holdout_data,
                holdout_prediction,
                state_delta_scale=scales.state_delta_scale,
                flow_delta_scale=scales.flow_delta_scale,
                active_fraction=active_fraction,
            ),
        }

    payload = {
        "contract": BASELINE_CONTRACT,
        "development_only": True,
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "production_compatible": False,
        "lineage": {
            "git_head": _git_head(),
            "graph_path": str(Path(args.graph).resolve()),
            "graph_sha256": _sha256(args.graph),
            "cache_manifest_path": str(Path(args.cache_manifest).resolve()),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "fit_d2_group_count": len(fit_d2),
            "holdout_d2_group_count": len(holdout_d2),
            "fit_d2_group_digest": hashlib.sha256("\n".join(sorted(fit_d2)).encode()).hexdigest(),
            "holdout_d2_group_digest": hashlib.sha256("\n".join(sorted(holdout_d2)).encode()).hexdigest(),
            "fit_events": sorted(fit_events),
            "holdout_events": sorted(holdout_events),
        },
        "fixed_setup": {
            "seed": args.seed,
            "fp32": True,
            "holdout_fraction": args.holdout_fraction,
            "split": "deterministic V60 rainfall-group SHA256 split",
            "source": "D2 single-actuator candidates only",
            "target": "raw signed candidate-minus-reference endpoint-local physical effect",
            "retained_horizon": MultiResolutionHorizonV60().indices(),
            "active_effect_fraction": active_fraction,
            "managed_flow_active_scale": "per changed-actuator TrainFit RMS scale",
            "no_graph_message_passing": True,
        },
        "data": {
            "fit": _summarize_dataset(fit_data),
            "holdout": _summarize_dataset(holdout_data),
            "state_delta_scale": np.asarray(scales.state_delta_scale, dtype=np.float64).tolist(),
            "flow_delta_scale": np.asarray(scales.flow_delta_scale, dtype=np.float64).tolist(),
        },
        "baselines": {
            "zero": evaluate(zero_fit, zero_holdout),
            "per_actuator_signed_mean": {
                **evaluate(mean_fit, mean_holdout),
                "fit_observed_actuator_time_cells": int(np.sum(mean_counts > 0)),
                "total_actuator_time_cells": int(mean_counts.size),
            },
            "action_magnitude_ridge": {
                **evaluate(ridge_fit, ridge_holdout),
                "features": "actuator identity + identity*current_delta + identity*causal_prefix_mean_delta",
                "ridge_lambda": ridge_lambda,
            },
            "local_mlp": {
                **evaluate(mlp_fit, mlp_holdout),
                **mlp_fit_meta,
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(out),
                "fit_d2_groups": len(fit_d2),
                "holdout_d2_groups": len(holdout_d2),
                "fit_events": len(fit_events),
                "holdout_events": len(holdout_events),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
