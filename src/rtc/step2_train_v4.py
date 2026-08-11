from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .flood_volume import trapezoid_node_flood_volume
from .large_model_cli import _device, _step2_stats
from .models import DifferentiableHydraulicWorldModel
from .production_cli import _load_graph
from .step2_counterfactual import (
    CounterfactualLossWeights,
    counterfactual_action_loss,
    counterfactual_groups,
    rotated_reference_pairs,
    same_prefix_diagnostic,
)
from .step2_shards import load_shard_manifest, sha256_file
from .step2_stability import (
    STABILITY_AMENDMENT,
    STABILITY_MODEL_CONTRACT,
    derive_train_only_delta_scales,
)
from .step2_train_v3 import (
    _chunks,
    _history_row,
    _open_training_shard,
    _pair_numpy,
    _precision,
    _source_group_counts,
    _source_kind,
    _source_weight,
    _to_device,
)
from .step2_training_cache import build_step2_training_cache, load_step2_training_cache
from .train_state import restore_training_state, save_training_state, training_contract_sha
from .training import save_torch_checkpoint

TRAINING_CONTRACT = "STEP2_COUNTERFACTUAL_MULTISHOOTING_TRAINING_V4"
CURRICULUM_CONTRACT = "FLOW_H1_TO_H72_MULTISHOOTING_V3"
TRAINING_AMENDMENT = "PROJECT7_V069_STEP2_MULTISHOOTING_AMENDMENT_V3"
SEGMENT_STEPS = 12
MAX_PARALLEL_SEGMENTS = 3


def _stage(epoch: int, total_epochs: int) -> tuple[str, int, bool, bool, bool]:
    """Fixed Train-only schedule after the V2 H60 failure.

    The model first learns local H1/H2/H6/H12 dynamics. Full 360-min action sequences are
    then trained by multiple shooting: every 60-min segment starts from authoritative
    Train-only state/flow, so no backward path exceeds 12 recurrent steps. The final two
    epochs add authoritative H360 flood-volume calibration to the sum of the six segment
    predictions. Validation/Final are never consulted here.
    """

    if total_epochs != 18:
        raise ValueError("Step2 multishooting amendment V3 requires exactly 18 epochs")
    completed = int(epoch) + 1
    schedule = (
        (2, "flow_h1", 1, True, False, False),
        (4, "joint_h1", 1, False, False, False),
        (6, "h2", 2, False, False, False),
        (8, "h6", 6, False, False, False),
        (10, "h12", 12, False, False, False),
        (16, "h72_multishooting", 72, False, True, False),
        (18, "h72_multishooting_exact", 72, False, True, True),
    )
    for end_epoch, name, horizon, flow_only, use_d3, exact_calibration in schedule:
        if completed <= end_epoch:
            return name, horizon, flow_only, use_d3, exact_calibration
    raise AssertionError("multishooting schedule did not cover epoch")


def _segment_starts(horizon: int, segment_steps: int = SEGMENT_STEPS) -> list[int]:
    if horizon <= 0 or segment_steps <= 0:
        raise ValueError("horizon/segment_steps must be positive")
    if horizon % segment_steps:
        raise ValueError("multishooting horizon must be divisible by segment_steps")
    return list(range(0, horizon, segment_steps))


def _pack_segment_chunk(
    pair: dict[str, torch.Tensor],
    starts: list[int],
    *,
    segment_steps: int = SEGMENT_STEPS,
) -> dict[str, torch.Tensor]:
    """Pack several teacher-forced time segments into one parallel CUDA batch.

    Rows in every source pair are already [reference,candidate]. Concatenating whole
    branch batches segment-by-segment preserves that pair adjacency for the counterfactual
    loss while exposing more independent 60-min windows to the GPU in parallel.
    """

    if not starts:
        raise ValueError("segment chunk is empty")
    b = int(pair["initial_state"].shape[0])
    payload: dict[str, list[torch.Tensor]] = {
        "initial_state": [],
        "rainfall": [],
        "settings": [],
        "previous_actuator_flow": [],
        "target_states": [],
        "target_actuator_flows": [],
        "elapsed_seconds": [],
    }
    for start in starts:
        end = start + int(segment_steps)
        if start == 0:
            initial = pair["initial_state"]
            previous = pair["previous_actuator_flow"]
        else:
            initial = pair["target_states"][:, start - 1]
            previous = pair["target_actuator_flows"][:, start - 1]
        payload["initial_state"].append(initial)
        payload["rainfall"].append(pair["rainfall"][:, start:end])
        payload["settings"].append(pair["settings"][:, start:end])
        payload["previous_actuator_flow"].append(previous)
        payload["target_states"].append(pair["target_states"][:, start:end])
        payload["target_actuator_flows"].append(
            pair["target_actuator_flows"][:, start:end]
        )
        payload["elapsed_seconds"].append(pair["elapsed_seconds"][:, start : end + 1])
    result = {name: torch.cat(values, dim=0) for name, values in payload.items()}
    if result["initial_state"].shape[0] != b * len(starts):
        raise AssertionError("parallel segment packing changed branch count")
    return result


def _exact_segment_sum_loss(
    *,
    predicted_node_volume: torch.Tensor,
    exact_node_volume: torch.Tensor,
) -> torch.Tensor:
    exact = exact_node_volume.clamp_min(0.0)
    pred = predicted_node_volume.clamp_min(0.0)
    node_loss = torch.square(torch.log1p(pred) - torch.log1p(exact)).mean()
    total_loss = torch.square(
        torch.log1p(pred.sum(dim=-1)) - torch.log1p(exact.sum(dim=-1))
    ).mean()
    return node_loss + total_loss


def train_step2_large_v4_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train action-sensitive Step2 with bounded dynamics, smooth delta-TFV gradients, "
            "and 60-min parallel multiple-shooting windows"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--actuator-embedding-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exact-flood-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument(
        "--precision", choices=("auto", "bf16", "fp32", "fp16"), default="auto"
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-state")
    parser.add_argument("--no-resume-training", action="store_true")
    parser.add_argument("--d2-pairs-per-group", type=int, default=4)
    parser.add_argument("--d3-pairs-per-group", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    if args.epochs != 18:
        raise ValueError("PROJECT7_V069_STEP2_MULTISHOOTING_AMENDMENT_V3 requires 18 epochs")
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("counterfactual Step2 --batch-size must be a positive even number")
    if args.d2_pairs_per_group != 4 or args.d3_pairs_per_group != 2:
        raise ValueError("V3 fixes pair budgets at D2=4 and D3=2 per group/epoch")
    if args.grad_accum != 2:
        raise ValueError("V3 fixes grad_accum=2")
    if args.gradient_clip <= 0 or args.learning_rate <= 0:
        raise ValueError("learning-rate and gradient-clip must be positive")
    pair_batch_size = args.batch_size // 2

    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    manifest = load_shard_manifest(args.manifest)
    if manifest.get("counterfactual_groups_preserved") is not True:
        raise ValueError("Step2 V4 requires V6 counterfactual-group-preserving shards")
    if int(manifest["horizon_steps"]) != 72:
        raise ValueError("Step2 V4 requires the frozen 72-step/360-min trajectory")
    graph = _load_graph(args.graph)
    model_step_seconds = int(manifest["model_step_seconds"])
    full_horizon = int(manifest["horizon_steps"])
    swmm_engine_version = str(manifest["swmm_engine_version"])
    (
        (state_mean, state_std),
        (rain_mean, rain_std),
        (physics_mean, physics_std),
        (static_mean, static_std),
        flow_std,
    ) = _step2_stats(manifest, graph)
    delta_state_scale, delta_flow_scale, scale_details = derive_train_only_delta_scales(
        args.manifest, state_std=state_std, flow_std=flow_std
    )

    training_manifest = dict(manifest)
    if args.cache_dir:
        cache_manifest_path = build_step2_training_cache(args.manifest, args.cache_dir)
        cached = load_step2_training_cache(cache_manifest_path)
        if len(cached["shards"]) != len(manifest["shards"]):
            raise ValueError("training cache shard count differs from V6 manifest")
        training_manifest["shards"] = [
            {**source_item, "arrays": cached_item["arrays"]}
            for source_item, cached_item in zip(
                manifest["shards"], cached["shards"], strict=True
            )
        ]

    with np.load(str(manifest["shards"][0]["path"]), allow_pickle=False) as first:
        core_config = {
            "state_dim": int(first["initial_state"].shape[-1]),
            "rainfall_dim": int(first["rainfall"].shape[-1]),
            "node_static_dim": int(graph.static_node_features.shape[-1]),
            "actuator_physics_dim": int(graph.actuator_physics.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "actuator_count": len(graph.actuator_ids),
            "actuator_embedding_dim": args.actuator_embedding_dim,
            "direct_action_context": True,
            "bounded_state_residual": True,
            "bounded_flow_residual": True,
            "delta_state_scale": delta_state_scale.tolist(),
            "delta_flow_scale": delta_flow_scale.tolist(),
            "stability_amendment": STABILITY_AMENDMENT,
            "stability_model_contract": STABILITY_MODEL_CONTRACT,
        }

    model = DifferentiableHydraulicWorldModel(**core_config).to(device)
    model.set_normalization(
        state_mean=torch.as_tensor(state_mean, device=device),
        state_std=torch.as_tensor(state_std, device=device),
        rain_mean=torch.as_tensor(rain_mean, device=device),
        rain_std=torch.as_tensor(rain_std, device=device),
        static_mean=torch.as_tensor(static_mean, device=device),
        static_std=torch.as_tensor(static_std, device=device),
        physics_mean=torch.as_tensor(physics_mean, device=device),
        physics_std=torch.as_tensor(physics_std, device=device),
        flow_std=torch.as_tensor(flow_std, device=device),
    )
    up = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    down = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    physics = torch.as_tensor(graph.actuator_physics, dtype=torch.float32, device=device)

    precision_requested = "fp32" if args.no_amp else args.precision
    autocast_enabled, autocast_dtype, use_scaler, precision_mode = _precision(
        device, precision_requested
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    source_counts = _source_group_counts(training_manifest)
    weights = CounterfactualLossWeights(exact_flood=args.exact_flood_loss_weight)

    train_payload = {
        "manifest_sha256": sha256_file(args.manifest),
        "graph_sha256": sha256_file(args.graph),
        "training_contract": TRAINING_CONTRACT,
        "training_amendment": TRAINING_AMENDMENT,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "segment_steps": SEGMENT_STEPS,
        "max_parallel_segments": MAX_PARALLEL_SEGMENTS,
        "model_step_seconds": model_step_seconds,
        "horizon_steps": full_horizon,
        "swmm_engine_version": swmm_engine_version,
        "hidden_dim": args.hidden_dim,
        "actuator_embedding_dim": args.actuator_embedding_dim,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "precision": precision_mode,
        "gradient_clip": args.gradient_clip,
        "d2_pairs_per_group": args.d2_pairs_per_group,
        "d3_pairs_per_group": args.d3_pairs_per_group,
        "loss_weights": weights.__dict__,
        "seed": args.seed,
    }
    contract_sha, code_sha = training_contract_sha("step2_multishooting_v4", train_payload)
    resume_path = Path(args.resume_state or (str(args.out) + ".trainstate.pt"))
    start_epoch = 0
    history: list[dict[str, object]] = []
    if not args.no_resume_training and resume_path.is_file():
        start_epoch, extra = restore_training_state(
            resume_path,
            expected_contract_sha256=contract_sha,
            expected_code_sha256=code_sha,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        if "numpy_rng_state" in extra:
            rng.bit_generator.state = extra["numpy_rng_state"]
        history = [dict(x) for x in extra.get("history", [])]
    elif args.no_resume_training and resume_path.is_file():
        resume_path.unlink()

    for epoch in range(start_epoch, args.epochs):
        stage, horizon, flow_only, use_d3, exact_calibration = _stage(epoch, args.epochs)
        source_balance = use_d3 and source_counts["D2"] > 0 and source_counts["D3"] > 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        shard_order = list(training_manifest["shards"])
        rng.shuffle(shard_order)
        totals = {
            "total_loss": 0.0,
            "absolute_state_loss": 0.0,
            "absolute_flow_loss": 0.0,
            "delta_state_loss": 0.0,
            "delta_flow_loss": 0.0,
            "delta_tfv_loss": 0.0,
            "ranking_loss": 0.0,
            "exact_flood_loss": 0.0,
            "physical_loss": 0.0,
            "sensitivity_ratio": 0.0,
            "sign_accuracy": 0.0,
        }
        source_metric_sum = {
            "D2": {"sensitivity": 0.0, "sign": 0.0, "pairs": 0},
            "D3": {"sensitivity": 0.0, "sign": 0.0, "pairs": 0},
        }
        pairs_seen = micro_step = optimizer_steps = 0
        grad_norm_sum = grad_norm_max = 0.0

        for item in shard_order:
            with _open_training_shard(item) as ds:
                selected: dict[str, list[tuple[int, int]]] = {"D2": [], "D3": []}
                groups = list(counterfactual_groups(ds).items())
                rng.shuffle(groups)
                for group_key, indices in groups:
                    source = _source_kind(group_key)
                    if source == "D3" and not use_d3:
                        continue
                    budget = args.d3_pairs_per_group if source == "D3" else args.d2_pairs_per_group
                    selected[source].extend(
                        rotated_reference_pairs(ds, indices, epoch=epoch, budget=budget)
                    )

                for source in ("D2", "D3"):
                    source_pairs = selected[source]
                    if not source_pairs:
                        continue
                    rng.shuffle(source_pairs)
                    source_factor = _source_weight(
                        source, source_counts, enable_balance=source_balance
                    )
                    for pair_rows in _chunks(source_pairs, pair_batch_size):
                        pair_count = len(pair_rows)
                        np_pair = _pair_numpy(ds, pair_rows, horizon)
                        pair = _to_device(np_pair, device)
                        same_prefix_diagnostic(
                            pair["initial_state"], pair["rainfall"], pair["previous_actuator_flow"]
                        )
                        dt_all = pair["elapsed_seconds"][:, 1:] - pair["elapsed_seconds"][:, :-1]
                        if torch.any(dt_all <= 0) or torch.any(
                            torch.abs(dt_all - float(model_step_seconds)) > 1e-6
                        ):
                            raise ValueError("counterfactual pair violates frozen model step")

                        metric_acc = {name: 0.0 for name in totals}
                        exact_loss_tensor = pair["initial_state"].sum() * 0.0
                        if horizon <= SEGMENT_STEPS:
                            segment_chunks = [[0]]
                            segment_steps = horizon
                        else:
                            starts = _segment_starts(horizon)
                            segment_chunks = [
                                starts[i : i + MAX_PARALLEL_SEGMENTS]
                                for i in range(0, len(starts), MAX_PARALLEL_SEGMENTS)
                            ]
                            segment_steps = SEGMENT_STEPS

                        total_segments = sum(len(chunk) for chunk in segment_chunks)
                        pred_exact_node_total = None
                        for starts in segment_chunks:
                            if horizon <= SEGMENT_STEPS:
                                segment_pair = pair
                            else:
                                segment_pair = _pack_segment_chunk(
                                    pair, starts, segment_steps=segment_steps
                                )
                            dt = (
                                segment_pair["elapsed_seconds"][:, 1:]
                                - segment_pair["elapsed_seconds"][:, :-1]
                            )
                            with torch.autocast(
                                device_type=device.type,
                                dtype=autocast_dtype,
                                enabled=autocast_enabled,
                            ):
                                rollout = model.rollout(
                                    segment_pair["initial_state"],
                                    segment_pair["rainfall"],
                                    segment_pair["settings"],
                                    segment_pair["previous_actuator_flow"],
                                    up,
                                    down,
                                    physics,
                                    static,
                                    edges,
                                )
                                metrics = counterfactual_action_loss(
                                    initial_state=segment_pair["initial_state"],
                                    rollout_states=rollout.states,
                                    rollout_flows=rollout.actuator_flows,
                                    target_states=segment_pair["target_states"],
                                    target_flows=segment_pair["target_actuator_flows"],
                                    exact_node_flood_volume_m3=None,
                                    dt_seconds=dt,
                                    state_std=model.transition.state_std,
                                    flow_std=model.actuator.flow_std,
                                    full_horizon=False,
                                    weights=weights,
                                    flow_only=flow_only,
                                    use_authoritative_endpoint_truth=False,
                                    use_exact_flood_loss=False,
                                )
                                fraction = float(len(starts)) / float(total_segments)
                                segment_loss = metrics.total * source_factor * fraction

                                if exact_calibration and not flow_only:
                                    node_volume = trapezoid_node_flood_volume(
                                        segment_pair["initial_state"],
                                        rollout.states,
                                        flood_rate_index=2,
                                        dt_seconds=dt,
                                    )
                                    seg_count = len(starts)
                                    branch_n = int(pair["initial_state"].shape[0])
                                    node_volume = node_volume.reshape(
                                        seg_count, branch_n, node_volume.shape[-1]
                                    ).sum(dim=0)
                                    pred_exact_node_total = (
                                        node_volume
                                        if pred_exact_node_total is None
                                        else pred_exact_node_total + node_volume
                                    )

                                loss = segment_loss / args.grad_accum
                            if not torch.isfinite(segment_loss):
                                raise RuntimeError(
                                    f"non-finite Step2 segment loss epoch={epoch + 1} stage={stage}"
                                )
                            scaler.scale(loss).backward()
                            for name, metric_name in (
                                ("total_loss", "total"),
                                ("absolute_state_loss", "absolute_state"),
                                ("absolute_flow_loss", "absolute_flow"),
                                ("delta_state_loss", "delta_state"),
                                ("delta_flow_loss", "delta_flow"),
                                ("delta_tfv_loss", "delta_tfv"),
                                ("ranking_loss", "ranking"),
                                ("physical_loss", "physical"),
                            ):
                                metric_acc[name] += float(getattr(metrics, metric_name).detach()) * fraction
                            if not flow_only:
                                metric_acc["sensitivity_ratio"] += float(
                                    metrics.sensitivity_ratio.detach()
                                ) * fraction
                                metric_acc["sign_accuracy"] += float(
                                    metrics.sign_correct.detach()
                                ) * fraction

                        if exact_calibration and pred_exact_node_total is not None:
                            with torch.autocast(
                                device_type=device.type,
                                dtype=autocast_dtype,
                                enabled=autocast_enabled,
                            ):
                                exact_loss_tensor = _exact_segment_sum_loss(
                                    predicted_node_volume=pred_exact_node_total,
                                    exact_node_volume=pair["exact_node_flood_volume_m3"],
                                )
                                exact_weighted = (
                                    weights.exact_flood * exact_loss_tensor * source_factor
                                )
                            scaler.scale(exact_weighted / args.grad_accum).backward()
                            metric_acc["exact_flood_loss"] = float(exact_loss_tensor.detach())
                            metric_acc["total_loss"] += float(exact_weighted.detach())

                        micro_step += 1
                        pairs_seen += pair_count
                        if not flow_only:
                            source_metric_sum[source]["sensitivity"] += (
                                metric_acc["sensitivity_ratio"] * pair_count
                            )
                            source_metric_sum[source]["sign"] += (
                                metric_acc["sign_accuracy"] * pair_count
                            )
                            source_metric_sum[source]["pairs"] += pair_count

                        if micro_step % args.grad_accum == 0:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.gradient_clip
                            )
                            if not torch.isfinite(grad_norm):
                                raise RuntimeError(
                                    f"non-finite Step2 gradient norm at epoch={epoch + 1}"
                                )
                            grad_value = float(grad_norm.detach())
                            grad_norm_sum += grad_value
                            grad_norm_max = max(grad_norm_max, grad_value)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1

                        for name in totals:
                            totals[name] += metric_acc[name] * pair_count

        if micro_step % args.grad_accum:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite Step2 gradient norm at epoch={epoch + 1}")
            grad_value = float(grad_norm.detach())
            grad_norm_sum += grad_value
            grad_norm_max = max(grad_norm_max, grad_value)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if pairs_seen == 0:
            raise RuntimeError("no counterfactual pairs available")
        if flow_only:
            totals["sensitivity_ratio"] = float("nan") * pairs_seen
            totals["sign_accuracy"] = float("nan") * pairs_seen
        row = _history_row(
            epoch=epoch,
            stage=stage,
            horizon=horizon,
            precision=precision_mode,
            totals=totals,
            pairs=pairs_seen,
            micro_batches=micro_step,
            optimizer_steps=optimizer_steps,
            grad_norm_mean=grad_norm_sum / max(optimizer_steps, 1),
            grad_norm_max=grad_norm_max,
        )
        row["training_mode"] = "multishooting" if horizon > SEGMENT_STEPS else "recurrent"
        row["segment_steps"] = SEGMENT_STEPS if horizon > SEGMENT_STEPS else horizon
        row["parallel_segments"] = MAX_PARALLEL_SEGMENTS if horizon > SEGMENT_STEPS else 1
        for source in ("D2", "D3"):
            count = int(source_metric_sum[source]["pairs"])
            row[f"{source.lower()}_sensitivity_ratio"] = (
                source_metric_sum[source]["sensitivity"] / count if count else float("nan")
            )
            row[f"{source.lower()}_sign_accuracy"] = (
                source_metric_sum[source]["sign"] / count if count else float("nan")
            )
        history.append(row)
        save_training_state(
            resume_path,
            contract_sha256=contract_sha,
            rtc_source_tree_sha256=code_sha,
            completed_epochs=epoch + 1,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            extra_state={"numpy_rng_state": rng.bit_generator.state, "history": history},
        )

    checkpoint_config = {
        **core_config,
        "model_step_seconds": model_step_seconds,
        "horizon_steps": full_horizon,
        "swmm_engine_version": swmm_engine_version,
        "time_contract": "STEP2_FIXED_DISCRETE_TIME_ENGINE_V2",
        "training_contract_sha256": contract_sha,
        "step2_training_contract": TRAINING_CONTRACT,
        "training_amendment": TRAINING_AMENDMENT,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "training_mode": "parallel_multiple_shooting",
        "segment_steps": SEGMENT_STEPS,
        "max_parallel_segments": MAX_PARALLEL_SEGMENTS,
        "scale_derivation": scale_details,
        "bounded_state_residual": True,
        "bounded_flow_residual": True,
        "counterfactual_tfv_proxy": "softplus_train_only_scale",
    }
    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config=checkpoint_config,
        training_manifest_sha256=sha256_file(args.manifest),
        scientific_split="development",
    )
    history_path = Path(str(args.out) + ".history.json")
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "contract": TRAINING_CONTRACT,
                "training_amendment": TRAINING_AMENDMENT,
                "checkpoint": meta,
                "completed_epochs": args.epochs,
                "resumed_from_epoch": start_epoch,
                "final": history[-1] if history else None,
                "history": str(history_path),
                "precision": precision_mode,
                "source_group_counts": source_counts,
                "segment_steps": SEGMENT_STEPS,
                "max_parallel_segments": MAX_PARALLEL_SEGMENTS,
                "training_state": str(resume_path),
                "training_contract_sha256": contract_sha,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    train_step2_large_v4_main()
