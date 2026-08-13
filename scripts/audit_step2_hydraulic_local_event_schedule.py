"""Fair endpoint-local D2 control using the exact V9 event-update exposure.

The earlier local learnability diagnostic updates a small MLP over 256-row
minibatches (about 968 optimizer steps in four epochs).  V9's D2 mechanism
ladder instead takes one optimizer step per rainfall/event after equally
averaging its checkpoint-group losses (14 event updates per epoch; 56 total).
This development-only control changes *only* that optimization exposure.  It
uses the same causal endpoint features, raw signed targets, deterministic split,
TrainFit-only scales, and all seven local outputs as the existing baseline.

It is deliberately not a Step2 model, does not read Validation/Final/Formal
assets, and never invokes SWMM.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v80_contract import DirectHydraulicEffectLossContractV80


CONTRACT = "PROJECT7_STEP2_D2_ENDPOINT_LOCAL_EVENT_SCHEDULE_CONTROL_V1"
OUTPUT_NAMES = (
    "upstream_delta_depth_m",
    "upstream_delta_flood_m3s",
    "upstream_delta_storage_m3",
    "downstream_delta_depth_m",
    "downstream_delta_flood_m3s",
    "downstream_delta_storage_m3",
    "delta_managed_flow_m3s",
)


def _baseline_module():
    """Load the frozen local-baseline helpers without making scripts a package."""
    path = Path(__file__).with_name("audit_step2_hydraulic_learnability_baselines.py")
    spec = importlib.util.spec_from_file_location("step2_hydraulic_learnability_baselines", path)
    if spec is None or spec.loader is None:  # pragma: no cover - filesystem fail-closed
        raise RuntimeError(f"cannot load local baseline helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASELINE = _baseline_module()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def canonical_event_schedule(*, fit_d2_groups: int, fit_events: int, epochs: int) -> dict[str, int]:
    """Validate and describe the frozen V9 D2 event-balanced exposure."""
    if int(fit_d2_groups) != 112:
        raise ValueError("fair local control requires exactly 112 D2 groups")
    if int(fit_events) != 14:
        raise ValueError("fair local control requires exactly 14 TrainFit events")
    if int(epochs) != 4:
        raise ValueError("fair local control requires exactly four epochs")
    if fit_d2_groups % fit_events:
        raise ValueError("D2 groups must divide evenly across TrainFit events")
    return {
        "fit_d2_groups": int(fit_d2_groups),
        "fit_events": int(fit_events),
        "groups_per_event": int(fit_d2_groups // fit_events),
        "epochs": int(epochs),
        "optimizer_updates_per_epoch": int(fit_events),
        "optimizer_updates_total": int(fit_events * epochs),
    }


def event_group_rows(dataset: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Return deterministic rows for each rainfall/event then checkpoint group."""
    events = np.asarray(dataset["event"], dtype=object)
    groups = np.asarray(dataset["group"], dtype=object)
    if events.ndim != 1 or groups.ndim != 1 or events.shape != groups.shape:
        raise ValueError("endpoint dataset event/group labels must be aligned one-dimensional arrays")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for event in sorted(set(events.tolist())):
        event_rows = np.flatnonzero(events == event)
        by_group: dict[str, np.ndarray] = {}
        for group in sorted(set(groups[event_rows].tolist())):
            rows = event_rows[groups[event_rows] == group]
            if rows.size == 0:
                raise ValueError(f"{event}/{group}: empty endpoint group")
            by_group[str(group)] = rows.astype(np.int64, copy=False)
        grouped[str(event)] = by_group
    return grouped


def event_group_mean_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_rows: Sequence[torch.Tensor],
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    """Equal-mean checkpoint-group loss for one optimizer event update."""
    if not group_rows:
        raise ValueError("an event update requires at least one checkpoint group")
    losses = []
    for rows in group_rows:
        if rows.numel() == 0:
            raise ValueError("event update received an empty checkpoint group")
        losses.append(F.smooth_l1_loss(prediction.index_select(0, rows), target.index_select(0, rows), beta=beta))
    return torch.stack(losses).mean()


def build_grouped_d2_endpoint_dataset(cache: V60TrainCache, names, graph) -> dict[str, np.ndarray]:
    """Materialize exactly the established endpoint rows, preserving each D2 group."""
    pieces: list[dict[str, np.ndarray]] = []
    retained: np.ndarray | None = None
    for name in sorted(names):
        part = _BASELINE.build_d2_endpoint_dataset(cache, [name], graph)
        part_retained = np.asarray(part["retained_indices"], dtype=np.int64)
        if retained is None:
            retained = part_retained
        elif not np.array_equal(retained, part_retained):
            raise ValueError("retained horizon differs across D2 groups")
        part["group"] = np.full(part["targets"].shape[0], str(name), dtype=object)
        pieces.append(part)
    if not pieces or retained is None:
        raise ValueError("fair local control requires at least one D2 group")
    fields = ("features", "targets", "event", "actuator", "time_slot", "actuator_type", "group")
    return {
        field: np.concatenate([np.asarray(piece[field]) for piece in pieces], axis=0)
        for field in fields
    } | {"retained_indices": retained}


class _EventScheduledLocalMLP(nn.Module):
    """The exact bounded 64→64 endpoint-local architecture used by the prior control."""

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


def fit_event_scheduled_local_mlp(
    fit_data: Mapping[str, np.ndarray],
    prediction_feature_sets: Sequence[np.ndarray],
    *,
    seed: int,
    epochs: int,
    device: str,
    expected_schedule: Mapping[str, int],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Train the bounded local MLP using V9's event/group optimizer contract."""
    grouped = event_group_rows(fit_data)
    total_groups = sum(len(groups) for groups in grouped.values())
    observed = canonical_event_schedule(
        fit_d2_groups=total_groups,
        fit_events=len(grouped),
        epochs=epochs,
    )
    if dict(expected_schedule) != observed:
        raise ValueError("caller schedule does not match the observed frozen V9 event exposure")
    if any(len(groups) != observed["groups_per_event"] for groups in grouped.values()):
        raise ValueError("V9 fair local control requires eight D2 groups per TrainFit event")

    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature = np.asarray(fit_data["features"], dtype=np.float64)
    truth = np.asarray(fit_data["targets"], dtype=np.float64)
    x_mean = np.mean(feature, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(feature, axis=0, dtype=np.float64), 1e-6)
    y_scale = np.maximum(np.sqrt(np.mean(np.square(truth), axis=0, dtype=np.float64)), 1e-6)
    fit_x = torch.as_tensor((feature - x_mean) / x_std, dtype=torch.float32, device=target)
    fit_y = torch.as_tensor(truth / y_scale, dtype=torch.float32, device=target)
    model = _EventScheduledLocalMLP(int(fit_x.shape[1]), int(fit_y.shape[1])).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    update_count = 0

    for epoch in range(1, epochs + 1):
        event_keys = list(grouped)
        rng.shuffle(event_keys)
        losses: list[float] = []
        norms: list[float] = []
        model.train()
        for event in event_keys:
            optimizer.zero_grad(set_to_none=True)
            row_tensors = [
                torch.as_tensor(rows, dtype=torch.long, device=target)
                for _, rows in grouped[event].items()
            ]
            prediction = model(fit_x)
            loss = event_group_mean_smooth_l1(prediction, fit_y, row_tensors)
            if not torch.isfinite(loss):
                raise FloatingPointError("event-scheduled local MLP reached a non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            update_count += 1
            losses.append(float(loss.detach().cpu()))
            norms.append(float(norm.detach().cpu()))
        history.append(
            {
                "epoch": int(epoch),
                "event_updates": int(len(event_keys)),
                "groups_per_event": int(observed["groups_per_event"]),
                "mean_equal_group_smooth_l1": float(np.mean(losses)),
                "mean_gradient_norm": float(np.mean(norms)),
            }
        )

    if update_count != observed["optimizer_updates_total"]:
        raise RuntimeError("event-scheduled local MLP optimizer exposure diverged from V9")
    model.eval()
    with torch.no_grad():
        predictions = [
            model(torch.as_tensor((np.asarray(features, dtype=np.float64) - x_mean) / x_std, dtype=torch.float32, device=target))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            * y_scale
            for features in prediction_feature_sets
        ]
    return predictions, {
        "architecture": "identity+action-prefix+endpoint-state+previous-flow+rainfall+time+physics -> 64 -> 64 -> 7",
        "loss": "equal mean over checkpoint-group smooth-L1 losses within each event update",
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "device": str(target),
        "feature_normalization": "TrainFit only, per feature mean/std",
        "target_normalization": "TrainFit only, per output RMS",
        "schedule": observed,
        "training_history": history,
    }


def _per_output_metrics(
    dataset: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    *,
    state_delta_scale: np.ndarray,
    flow_delta_scale: np.ndarray,
    active_fraction: float,
) -> dict[str, Any]:
    """Expose every one of the seven local signed outputs, not only channel aggregates."""
    outputs: dict[str, Any] = {}
    state_channel_for_output = (0, 2, 3, 0, 2, 3)
    for index, name in enumerate(OUTPUT_NAMES):
        if index < 6:
            scale: np.ndarray | float = float(np.asarray(state_delta_scale)[state_channel_for_output[index]])
        else:
            scale = np.asarray(flow_delta_scale, dtype=np.float64)[np.asarray(dataset["actuator"], dtype=np.int64)]
        outputs[name] = _BASELINE.event_balanced_effect_metrics(
            _BASELINE._event_arrays_for_channel(
                dataset,
                prediction,
                target_indices=(index,),
                scale=scale,
            ),
            active_fraction=active_fraction,
        )
    return outputs


def _evaluate(
    dataset: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    *,
    state_delta_scale: np.ndarray,
    flow_delta_scale: np.ndarray,
    active_fraction: float,
) -> dict[str, Any]:
    return {
        "channel_aggregate": _BASELINE.evaluate_endpoint_predictions(
            dataset,
            prediction,
            state_delta_scale=state_delta_scale,
            flow_delta_scale=flow_delta_scale,
            active_fraction=active_fraction,
        ),
        "all_seven_endpoint_outputs": _per_output_metrics(
            dataset,
            prediction,
            state_delta_scale=state_delta_scale,
            flow_delta_scale=flow_delta_scale,
            active_fraction=active_fraction,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 fair V9-event-schedule local D2 control")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("fair local control seed is frozen at 42")
    if abs(float(args.holdout_fraction) - 0.20) > 1e-12:
        raise ValueError("fair local control holdout fraction is frozen at 0.20")

    started = time.perf_counter()
    graph = _BASELINE._load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError(f"expected frozen 109-actuator graph, got {len(graph.actuator_ids)}")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("fair local control refuses legacy dense D3")
    selected = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(cache, names=selected, holdout_fraction=float(args.holdout_fraction))
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    fit_events = {f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}" for name in fit_d2}
    holdout_events = {f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}" for name in holdout_d2}
    if fit_events & holdout_events:
        raise ValueError("fair local control detected event leakage across deterministic split")
    schedule = canonical_event_schedule(fit_d2_groups=len(fit_d2), fit_events=len(fit_events), epochs=4)

    # The scales deliberately follow V7/V9's TrainFit-only D2+D3 contract.
    scales = derive_target_scales_v70(cache, fit)
    active_fraction = float(DirectHydraulicEffectLossContractV80().active_effect_fraction)
    fit_data = build_grouped_d2_endpoint_dataset(cache, fit_d2, graph)
    holdout_data = build_grouped_d2_endpoint_dataset(cache, holdout_d2, graph)
    (fit_prediction, holdout_prediction), fit_meta = fit_event_scheduled_local_mlp(
        fit_data,
        (fit_data["features"], holdout_data["features"]),
        seed=args.seed,
        epochs=4,
        device=args.device,
        expected_schedule=schedule,
    )
    zero_fit = np.zeros_like(fit_data["targets"], dtype=np.float64)
    zero_holdout = np.zeros_like(holdout_data["targets"], dtype=np.float64)

    payload = {
        "contract": CONTRACT,
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
            "active_effect_fraction": active_fraction,
            "managed_flow_active_scale": "per changed-actuator TrainFit RMS scale",
            "no_graph_message_passing": True,
            "purpose": "control optimizer-update exposure against V9; not a replacement surrogate",
        },
        "data": {
            "fit_rows": int(fit_data["targets"].shape[0]),
            "holdout_rows": int(holdout_data["targets"].shape[0]),
            "output_names": list(OUTPUT_NAMES),
            "state_delta_scale": np.asarray(scales.state_delta_scale, dtype=np.float64).tolist(),
            "flow_delta_scale": np.asarray(scales.flow_delta_scale, dtype=np.float64).tolist(),
        },
        "schedule_control": fit_meta,
        "baselines": {
            "zero": {
                "TrainFit_D2": _evaluate(fit_data, zero_fit, state_delta_scale=scales.state_delta_scale, flow_delta_scale=scales.flow_delta_scale, active_fraction=active_fraction),
                "TrainInternalHoldout_D2": _evaluate(holdout_data, zero_holdout, state_delta_scale=scales.state_delta_scale, flow_delta_scale=scales.flow_delta_scale, active_fraction=active_fraction),
            },
            "event_scheduled_local_mlp": {
                "TrainFit_D2": _evaluate(fit_data, fit_prediction, state_delta_scale=scales.state_delta_scale, flow_delta_scale=scales.flow_delta_scale, active_fraction=active_fraction),
                "TrainInternalHoldout_D2": _evaluate(holdout_data, holdout_prediction, state_delta_scale=scales.state_delta_scale, flow_delta_scale=scales.flow_delta_scale, active_fraction=active_fraction),
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), **schedule, "fit_rows": int(fit_data["targets"].shape[0])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
