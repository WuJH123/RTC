"""Run the bounded Train-only V9 causal-history D2 mechanism diagnostic.

This runner intentionally does *not* alter V9's full-network graph model.  It
uses a single fixed endpoint-local direct signed-effect control on the exact
same eligible D2 TrainFit groups for three causal-information arms:

``B0`` current snapshot/action only;
``B1`` B0 plus frozen-Step1 reconstructed 13-frame history (online eligible);
``C1`` B0 plus authoritative past SWMM history (oracle diagnostic only).

It never invokes SWMM and never opens Validation, Final, or Formal assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rtc.graph import GraphSchema
from rtc.models import SparseStateEstimator
from rtc.step2_history_ladder_v90 import (
    HISTORY_LADDER_CONTRACT_V90,
    LocalHistoryEffectModelV90,
    endpoint_history_features_v90,
    history_reconstruction_metrics_v90,
    history_source_contract_v90,
)
from rtc.step2_history_v90 import (
    HistoryAssetV90,
    build_online_step1_history_v90,
    load_oracle_history_v90,
)
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v80_contract import DirectHydraulicEffectLossContractV80


LEVEL_B0 = "B0_CURRENT_SNAPSHOT"
LEVEL_B1 = "B1_FROZEN_STEP1_RECONSTRUCTED_HISTORY"
LEVEL_C1 = "C1_ORACLE_PAST_SWMM_HISTORY"
FIXED_EPOCHS = 4
FIXED_SEED = 42


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


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _development_train_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("scientific_split") == "development"
        and row.get("development_fold") == "train"
    ]


def _baseline_module() -> ModuleType:
    path = Path(__file__).with_name("audit_step2_hydraulic_learnability_baselines.py")
    spec = importlib.util.spec_from_file_location("v90_history_baseline_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen endpoint-baseline helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_history_eligible_d2_groups_v90(
    cache: V60TrainCache,
    fit_d2_names: Sequence[str],
    checkpoint_rows: Sequence[dict[str, str]],
) -> tuple[list[str], dict[str, str]]:
    """Select only checkpoints with enough pre-action sensor context.

    A frozen 13-window Step1 reconstruction ending at ``t-3600`` itself needs
    causal observations back to ``t-7200``.  This selection cannot pad, round,
    borrow a nearby checkpoint, or use a future frame.
    """
    by_checkpoint = {str(row["checkpoint_id"]): row for row in _development_train_rows(checkpoint_rows)}
    eligible: list[str] = []
    rejected: dict[str, str] = {}
    for name in sorted(fit_d2_names):
        entry = cache.entry(name)
        row = by_checkpoint.get(entry.checkpoint_id)
        if row is None:
            rejected[name] = "checkpoint_absent_from_development_train_table"
            continue
        if str(row.get("event_id")) != entry.event_id or str(row.get("rainfall_group")) != entry.rainfall_group:
            rejected[name] = "cache_checkpoint_event_or_rainfall_lineage_mismatch"
            continue
        try:
            elapsed = int(row["checkpoint_elapsed_seconds"])
        except (KeyError, ValueError):
            rejected[name] = "invalid_checkpoint_elapsed_seconds"
            continue
        if elapsed < 7200:
            rejected[name] = "insufficient_pre_action_history_for_13_frozen_step1_windows"
            continue
        eligible.append(name)
    return eligible, rejected


def _load_frozen_step1(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[SparseStateEstimator, dict[str, Any]]:
    """Load a frozen Step1 solely for inference, preserving its historical code SHA."""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("frozen Step1 checkpoint is not a state-dict payload")
    if payload.get("scientific_split") != "development":
        raise ValueError("frozen Step1 checkpoint is not development-lineage")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("frozen Step1 checkpoint has no model_config")
    cfg = dict(config)
    history_steps = int(cfg.get("history_steps", -1))
    model_step_seconds = int(cfg.get("model_step_seconds", -1))
    if history_steps != 13 or model_step_seconds != 300:
        raise ValueError("frozen Step1 checkpoint does not satisfy the 13x300s history contract")
    model = SparseStateEstimator(**cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lineage = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_contract": str(payload.get("checkpoint_contract", "")),
        "checkpoint_source_tree_sha256": str(payload.get("rtc_source_tree_sha256", "")),
        "scientific_split": str(payload.get("scientific_split", "")),
        "model_config": cfg,
        "frozen": True,
        "requires_grad_any": any(parameter.requires_grad for parameter in model.parameters()),
    }
    return model, lineage


def _load_sensor_nodes(path: str | Path, graph: GraphSchema) -> np.ndarray:
    names = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError("frozen Step1 sensor layout is empty")
    missing = sorted(set(names) - set(graph.node_ids))
    if missing:
        raise ValueError(f"frozen Step1 sensor nodes absent from frozen graph: {missing[:3]}")
    return np.asarray([graph.node_ids.index(name) for name in names], dtype=np.int64)


def _priority_indices(path: str | Path, graph: GraphSchema) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [str(value) for value in payload.get("priority_nodes", [])]
    missing = sorted(set(names) - set(graph.node_ids))
    if missing:
        raise ValueError(f"priority nodes absent from frozen graph: {missing[:3]}")
    return np.asarray([graph.node_ids.index(name) for name in names], dtype=np.int64)


def _max_depth(graph: GraphSchema) -> np.ndarray:
    try:
        column = graph.static_node_feature_names.index("max_depth_m")
    except ValueError as exc:
        raise ValueError("frozen graph lacks max_depth_m for pre-registered wet/high-depth audit") from exc
    return np.maximum(np.asarray(graph.static_node_features[:, column], dtype=np.float32), 1e-6)


def _checkpoint_contexts(
    cache: V60TrainCache,
    names: Sequence[str],
    *,
    checkpoint_rows: Sequence[dict[str, str]],
    train_index_rows: Sequence[dict[str, str]],
    graph: GraphSchema,
    step1_model: SparseStateEstimator,
    step1_path: Path,
    sensor_path: Path,
    sensor_indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, tuple[HistoryAssetV90, HistoryAssetV90]], dict[str, Any]]:
    """Build each retained group history once and record exact causal lineage."""
    checkpoints = {str(row["checkpoint_id"]): row for row in _development_train_rows(checkpoint_rows)}
    no_control = {
        str(row["event_id"]): Path(str(row["compact_path"]))
        for row in _development_train_rows(train_index_rows)
        if str(row.get("strategy")) == "no_control"
    }
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)

    def predictor(obs: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return step1_model(
                obs.to(device), mask.to(device), static, edges, context.to(device)
            ).detach().cpu()

    contexts: dict[str, tuple[HistoryAssetV90, HistoryAssetV90]] = {}
    source_rows: list[dict[str, Any]] = []
    for name in sorted(names):
        entry = cache.entry(name)
        checkpoint = checkpoints.get(entry.checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"{name}: missing development/train checkpoint lineage")
        compact_path = no_control.get(entry.event_id)
        if compact_path is None or not compact_path.is_file():
            raise ValueError(f"{name}: missing development/train no-control compact")
        elapsed = int(checkpoint["checkpoint_elapsed_seconds"])
        initial = np.asarray(entry.arrays["initial_state"][entry.reference_index], dtype=np.float32)
        oracle = load_oracle_history_v90(
            compact_path, checkpoint_elapsed_seconds=elapsed, cache_initial_state=initial
        )
        online = build_online_step1_history_v90(
            compact_path,
            checkpoint_elapsed_seconds=elapsed,
            cache_initial_state=initial,
            sensor_indices=sensor_indices,
            actuator_upstream=graph.actuator_upstream,
            actuator_downstream=graph.actuator_downstream,
            step1_checkpoint_path=step1_path,
            sensor_layout_path=sensor_path,
            predict_window=predictor,
        )
        if online.lineage["future_leakage"] != "NONE" or oracle.lineage["future_leakage"] != "NONE":
            raise RuntimeError(f"{name}: causal history builder reported future leakage")
        contexts[name] = (online, oracle)
        source_rows.append(
            {
                "group": name,
                "checkpoint_id": entry.checkpoint_id,
                "event_id": entry.event_id,
                "rainfall_group": entry.rainfall_group,
                "checkpoint_elapsed_seconds": elapsed,
                "online_current_sensor_max_abs_difference": online.lineage[
                    "current_sensor_observation_max_abs_difference"
                ],
                "oracle_current_state_max_abs_difference": oracle.lineage[
                    "current_frame_max_abs_difference"
                ],
                "online_input_elapsed_start_seconds": online.lineage["input_elapsed_start_seconds"],
                "online_input_elapsed_end_seconds": online.lineage["input_elapsed_end_seconds"],
            }
        )
    return contexts, {
        "groups": len(contexts),
        "source_rows": source_rows,
        "online_current_sensor_max_abs_difference": max(
            float(row["online_current_sensor_max_abs_difference"]) for row in source_rows
        ) if source_rows else float("nan"),
        "oracle_current_state_max_abs_difference": max(
            float(row["oracle_current_state_max_abs_difference"]) for row in source_rows
        ) if source_rows else float("nan"),
        "future_leakage": "NONE",
        "pre_action_only": True,
    }


def _history_rows_for_dataset(
    cache: V60TrainCache,
    names: Sequence[str],
    graph: GraphSchema,
    contexts: dict[str, tuple[HistoryAssetV90, HistoryAssetV90]],
    base: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Align group-level histories to every pre-existing endpoint-target row."""
    retained_count = int(len(base["retained_indices"]))
    online_rows: list[np.ndarray] = []
    oracle_rows: list[np.ndarray] = []
    cursor = 0
    for name in sorted(names):
        entry = cache.entry(name)
        online, oracle = contexts[name]
        reference_settings = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float32)
        for candidate in entry.indices:
            if candidate == entry.reference_index:
                continue
            candidate_settings = np.asarray(entry.arrays["settings"][candidate], dtype=np.float32)
            changed = np.flatnonzero(
                np.any(np.abs(candidate_settings - reference_settings) > 1e-8, axis=0)
            )
            if changed.size != 1:
                raise ValueError(f"{name}: D2 history diagnostic requires exactly one changed actuator")
            actuator = int(changed.item())
            expected = np.full(retained_count, actuator, dtype=np.int64)
            if not np.array_equal(base["actuator"][cursor : cursor + retained_count], expected):
                raise RuntimeError("endpoint baseline/history row order differs; refusing misaligned history")
            upstream, downstream = int(graph.actuator_upstream[actuator]), int(graph.actuator_downstream[actuator])
            online_feature = endpoint_history_features_v90(
                online.states_physical,
                online.actuator_flows_physical,
                actuator_index=actuator,
                upstream_index=upstream,
                downstream_index=downstream,
            )
            oracle_feature = endpoint_history_features_v90(
                oracle.states_physical,
                oracle.actuator_flows_physical,
                actuator_index=actuator,
                upstream_index=upstream,
                downstream_index=downstream,
            )
            online_rows.append(np.repeat(online_feature[None], retained_count, axis=0))
            oracle_rows.append(np.repeat(oracle_feature[None], retained_count, axis=0))
            cursor += retained_count
    if cursor != int(base["targets"].shape[0]):
        raise RuntimeError("history rows do not cover every endpoint target")
    return np.concatenate(online_rows, axis=0), np.concatenate(oracle_rows, axis=0)


def _normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.maximum(np.std(values, axis=0, dtype=np.float64), 1e-6)
    return ((values - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _dataset_subset(dataset: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    chosen = np.asarray(mask, dtype=bool)
    return {
        key: value if key == "retained_indices" else np.asarray(value)[chosen]
        for key, value in dataset.items()
    }


def _bucket_mask(dataset: dict[str, np.ndarray], lower: float, upper: float) -> np.ndarray:
    retained = np.asarray(dataset["retained_indices"], dtype=np.int64)
    minutes = (retained[np.asarray(dataset["time_slot"], dtype=np.int64)] + 1) * 5.0
    if lower == 0.0:
        return (minutes >= lower) & (minutes <= upper)
    return (minutes > lower) & (minutes <= upper)


def _evaluate(
    baseline: ModuleType,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    *,
    state_delta_scale: np.ndarray,
    flow_delta_scale: np.ndarray,
    active_fraction: float,
) -> dict[str, Any]:
    def metrics(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"events": 0, "rows": 0, "channels": {}}
        return baseline.evaluate_endpoint_predictions(
            _dataset_subset(dataset, mask),
            prediction[mask],
            state_delta_scale=state_delta_scale,
            flow_delta_scale=flow_delta_scale,
            active_fraction=active_fraction,
        )

    all_rows = np.ones(dataset["targets"].shape[0], dtype=bool)
    return {
        "overall": metrics(all_rows),
        "horizon_buckets": {
            "0_30_min": metrics(_bucket_mask(dataset, 0.0, 30.0)),
            "30_120_min": metrics(_bucket_mask(dataset, 30.0, 120.0)),
            "120_360_min": metrics(_bucket_mask(dataset, 120.0, 360.0)),
        },
    }


def _train_arm(
    *,
    base_features: np.ndarray,
    history_features: np.ndarray,
    targets: np.ndarray,
    event: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed 4-epoch / one-equal-event-update direct signed-effect control."""
    if seed != FIXED_SEED:
        raise ValueError("history ladder seed is frozen at 42")
    if history_features.shape[:2] != (base_features.shape[0], 13):
        raise ValueError("history features do not align with base endpoint rows")
    if targets.shape[0] != base_features.shape[0]:
        raise ValueError("history ladder target/base row count mismatch")
    torch.manual_seed(seed)
    np.random.seed(seed)
    base_norm, base_mean, base_std = _normalize(base_features)
    target_scale = np.maximum(
        np.sqrt(np.mean(np.square(targets), axis=0, dtype=np.float64)), 1e-6
    ).astype(np.float32)
    # Source-specific input standardization is derived exclusively from this
    # common TrainFit corpus.  B0 uses literal zero context after normalization.
    history_norm, history_mean, history_std = _normalize(history_features)
    model = LocalHistoryEffectModelV90(
        base_feature_dim=base_norm.shape[1],
        action_feature_indices=(109, 110),
        action_zero_values=(
            float(-base_mean[109] / base_std[109]),
            float(-base_mean[110] / base_std[110]),
        ),
        output_dim=targets.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    tensor_base = torch.as_tensor(base_norm, dtype=torch.float32, device=device)
    tensor_history = torch.as_tensor(history_norm, dtype=torch.float32, device=device)
    tensor_target = torch.as_tensor(targets / target_scale, dtype=torch.float32, device=device)
    event_labels = np.asarray(event, dtype=object)
    event_rows = {
        key: np.flatnonzero(event_labels == key).astype(np.int64)
        for key in sorted(set(event_labels.tolist()))
    }
    if len(event_rows) != 14:
        raise ValueError(f"history ladder requires 14 common TrainFit events, got {len(event_rows)}")
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, FIXED_EPOCHS + 1):
        keys = list(event_rows)
        rng.shuffle(keys)
        losses: list[float] = []
        gradients: list[float] = []
        model.train()
        for key in keys:
            index = torch.as_tensor(event_rows[key], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                tensor_base.index_select(0, index), tensor_history.index_select(0, index)
            )
            loss = F.smooth_l1_loss(prediction, tensor_target.index_select(0, index), beta=0.5)
            if not torch.isfinite(loss):
                raise FloatingPointError("history ladder received a non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("history ladder received non-finite gradients")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradients.append(float(norm.detach().cpu()))
        row = {
            "epoch": epoch,
            "event_updates": len(keys),
            "normalized_smooth_l1": float(np.mean(losses)),
            "gradient_norm": float(np.mean(gradients)),
            "finite_gradients": True,
        }
        history.append(row)
        print("[V9_HISTORY] " + " ".join(f"{key}={value}" for key, value in row.items()), flush=True)
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_base, tensor_history).detach().cpu().numpy() * target_scale
        zero_base = tensor_base.clone()
        zero_base[:, 109] = -torch.as_tensor(base_mean[109] / base_std[109], device=device)
        zero_base[:, 110] = -torch.as_tensor(base_mean[110] / base_std[110], device=device)
        exact_zero = torch.equal(
            model(zero_base, tensor_history), torch.zeros_like(model(zero_base, tensor_history))
        )
    return prediction.astype(np.float64), {
        "architecture": "same fixed local endpoint direct signed-effect MLP; shared 13-frame causal Conv1D history encoder zero-centered",
        "epochs": FIXED_EPOCHS,
        "event_updates_per_epoch": len(event_rows),
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "fp32": True,
        "seed": seed,
        "base_feature_normalization": {"mean": base_mean.tolist(), "std": base_std.tolist()},
        "history_feature_normalization": {"mean": history_mean.tolist(), "std": history_std.tolist()},
        "target_rms_scale": target_scale.tolist(),
        "action_zero_exact": bool(exact_zero),
        "training_history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 Train-only V9 causal history ladder")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--step1-checkpoint", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--formal-asset-audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.seed != FIXED_SEED:
        raise ValueError("history ladder seed is frozen at 42")
    if abs(float(args.holdout_fraction) - 0.20) > 1e-12:
        raise ValueError("history ladder must use the frozen 0.20 deterministic split")

    started = time.perf_counter()
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109 or len(graph.node_ids) != 932:
        raise ValueError("history ladder requires the frozen 932-node/109-actuator graph")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("history ladder refuses legacy dense D3 cache entries")
    selected = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    checkpoint_rows = _read_csv(args.checkpoints)
    eligible, rejected = select_history_eligible_d2_groups_v90(cache, fit_d2, checkpoint_rows)
    if len(eligible) != 88:
        raise ValueError(
            "frozen history ladder expected 88 eligible TrainFit D2 groups; "
            f"got {len(eligible)}"
        )
    if len({f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}" for name in eligible}) != 14:
        raise ValueError("history ladder common eligible corpus does not span 14 TrainFit events")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    step1_path, sensor_path = Path(args.step1_checkpoint), Path(args.sensors)
    step1, step1_lineage = _load_frozen_step1(step1_path, device=device)
    sensor_indices = _load_sensor_nodes(sensor_path, graph)
    contexts, history_lineage = _checkpoint_contexts(
        cache,
        eligible,
        checkpoint_rows=checkpoint_rows,
        train_index_rows=_read_csv(args.train_index),
        graph=graph,
        step1_model=step1,
        step1_path=step1_path,
        sensor_path=sensor_path,
        sensor_indices=sensor_indices,
        device=device,
    )
    online_stack = np.stack([contexts[name][0].states_physical for name in sorted(eligible)])
    oracle_stack = np.stack([contexts[name][1].states_physical for name in sorted(eligible)])
    reconstruction = history_reconstruction_metrics_v90(
        online_stack,
        oracle_stack,
        max_depth_m=_max_depth(graph),
        priority_indices=_priority_indices(args.formal_asset_audit, graph),
    )

    baseline = _baseline_module()
    base = baseline.build_d2_endpoint_dataset(cache, eligible, graph)
    online_history, oracle_history = _history_rows_for_dataset(cache, eligible, graph, contexts, base)
    if online_history.shape != oracle_history.shape or online_history.shape[0] != base["targets"].shape[0]:
        raise RuntimeError("history arms do not align with the common endpoint target table")
    b0_history = np.zeros_like(online_history, dtype=np.float32)
    # Use the frozen canonical TrainFit target scales for active masks, including
    # per-actuator managed-flow scales.  Holdout never participates.
    scales = derive_target_scales_v70(cache, fit)
    active_fraction = float(DirectHydraulicEffectLossContractV80().active_effect_fraction)

    arm_features = {
        LEVEL_B0: (b0_history, "none"),
        LEVEL_B1: (online_history, "frozen_step1_reconstruction"),
        LEVEL_C1: (oracle_history, "oracle_past_swmm"),
    }
    arms: dict[str, Any] = {}
    for label, (history_features, source) in arm_features.items():
        prediction, training = _train_arm(
            base_features=np.asarray(base["features"], dtype=np.float32),
            history_features=np.asarray(history_features, dtype=np.float32),
            targets=np.asarray(base["targets"], dtype=np.float64),
            event=np.asarray(base["event"], dtype=object),
            seed=args.seed,
            device=device,
        )
        arms[label] = {
            "history_source": history_source_contract_v90(source),
            "training": training,
            "metrics": _evaluate(
                baseline,
                base,
                prediction,
                state_delta_scale=scales.state_delta_scale,
                flow_delta_scale=scales.flow_delta_scale,
                active_fraction=active_fraction,
            ),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": HISTORY_LADDER_CONTRACT_V90,
        "development_only": True,
        "production_compatible": False,
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "oracle_history_forbidden_online": True,
        "lineage": {
            "git_head": _git_head(),
            "graph_path": str(Path(args.graph).resolve()),
            "graph_sha256": _sha256(args.graph),
            "cache_manifest_path": str(Path(args.cache_manifest).resolve()),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "checkpoint_table_path": str(Path(args.checkpoints).resolve()),
            "checkpoint_table_sha256": _sha256(args.checkpoints),
            "train_index_path": str(Path(args.train_index).resolve()),
            "train_index_sha256": _sha256(args.train_index),
            "sensor_layout_path": str(sensor_path.resolve()),
            "sensor_layout_sha256": _sha256(sensor_path),
            "step1": step1_lineage,
            "fit_d2_group_count_before_history_gate": len(fit_d2),
            "eligible_d2_group_count": len(eligible),
            "eligible_d2_group_digest": hashlib.sha256("\n".join(sorted(eligible)).encode()).hexdigest(),
            "ineligible_group_reasons": rejected,
        },
        "fixed_contract": {
            "common_source": "matched eligible TrainFit D2 only",
            "candidate_target": "raw signed candidate-minus-reference endpoint-local physical hydraulic effect",
            "seed": args.seed,
            "epochs": FIXED_EPOCHS,
            "event_updates_per_epoch": 14,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 5.0,
            "fp32": True,
            "no_graph_message_passing": True,
            "no_future_state_input": True,
            "history_frames": 13,
            "history_frame_seconds": 300,
            "history_window": "t-3600,...,t inclusive",
            "retained_horizon_indices": MultiResolutionHorizonV60().indices(),
            "active_effect_fraction": active_fraction,
            "managed_flow_active_scale": "per changed actuator frozen TrainFit RMS scale",
        },
        "history_lineage": history_lineage,
        "step1_reconstruction_vs_oracle_past": reconstruction,
        "dataset": {
            "endpoint_rows": int(base["targets"].shape[0]),
            "events": int(len(set(np.asarray(base["event"], dtype=object).tolist()))),
            "retained_horizon_points": int(len(base["retained_indices"])),
            "state_delta_scale": np.asarray(scales.state_delta_scale, dtype=np.float64).tolist(),
            "flow_delta_scale": np.asarray(scales.flow_delta_scale, dtype=np.float64).tolist(),
        },
        "arms": arms,
        "elapsed_seconds": time.perf_counter() - started,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(out),
                "eligible_d2_groups": len(eligible),
                "events": payload["dataset"]["events"],
                "reconstruction": reconstruction,
                "overall": {
                    label: arm["metrics"]["overall"]["channels"]
                    for label, arm in arms.items()
                },
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
