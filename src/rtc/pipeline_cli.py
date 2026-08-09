from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import write_passive_no_rtc_inp
from .contracts import load_priority_nodes
from .data_design import design_multi_actuator_rollouts
from .dataset_compile import compile_branches_to_npz
from .final_eval import compile_closed_loop_run_index, event_balanced_summary
from .graph import build_graph_schema, save_graph_schema
from .hydraulic_trajectory import run_hydraulic_trajectory
from .inp import discover_actuators
from .models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from .splits import assign_rainfall_group_splits
from .step1_dataset import compile_step1_windows, save_step1_windows
from .swmm_sequence import run_control_sequence_branch
from .training import save_torch_checkpoint, train_step1, train_step2


def _load_lines(path: str | Path) -> tuple[str, ...]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return tuple(line for line in lines if line and not line.startswith("#"))


def build_graph_main() -> None:
    parser = argparse.ArgumentParser(description="Compile frozen SWMM INP into GNN graph/model schema")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--directed-only", action="store_true")
    args = parser.parse_args()
    schema = build_graph_schema(args.inp, bidirectional=not args.directed_only)
    save_graph_schema(schema, args.out)
    print(json.dumps({
        "graph": args.out,
        "nodes": len(schema.node_ids),
        "edges": int(schema.edge_index.shape[1]),
        "actuators": len(schema.actuator_ids),
        "system_units": schema.system_units,
    }, indent=2))


def make_passive_inp_main() -> None:
    parser = argparse.ArgumentParser(description="Create passive No-RTC INP by removing only [CONTROLS]")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = write_passive_no_rtc_inp(args.inp, args.out)
    print(str(out))


def split_groups_main() -> None:
    parser = argparse.ArgumentParser(description="Assign rainfall groups to leakage-safe scientific roles")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rainfall-group-col", default="rainfall_group")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    out = assign_rainfall_group_splits(
        frame, rainfall_group_col=args.rainfall_group_col, seed=args.seed
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    summary = out.groupby("scientific_split")[args.rainfall_group_col].nunique().to_dict()
    print(json.dumps({"out": args.out, "rainfall_groups": summary}, indent=2))


def run_trajectory_main() -> None:
    parser = argparse.ArgumentParser(description="Run D0/D1 authoritative full-event hydraulic trajectory")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stride-seconds", type=int, default=300)
    args = parser.parse_args()
    result = run_hydraulic_trajectory(
        inp_path=args.inp,
        output_dir=args.out_dir,
        run_id=args.run_id,
        record_stride_seconds=args.stride_seconds,
    )
    print(json.dumps({
        "metadata": result.metadata_path,
        "flow_routing_error_pct": result.flow_routing_error_pct,
    }, indent=2))


def design_d3_main() -> None:
    parser = argparse.ArgumentParser(description="Design D3 multi-actuator continuous rollout sequences")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--sequences-per-checkpoint", type=int, default=8)
    parser.add_argument("--perturbation-std", type=float, default=0.20)
    parser.add_argument("--change-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    catalog = discover_actuators(args.inp)
    checkpoints = pd.read_csv(args.checkpoints)
    manifest = design_multi_actuator_rollouts(
        checkpoints,
        catalog,
        horizon_steps=args.horizon_steps,
        sequences_per_checkpoint=args.sequences_per_checkpoint,
        perturbation_std=args.perturbation_std,
        change_probability=args.change_probability,
        seed=args.seed,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(json.dumps({
        "rows": len(manifest),
        "checkpoints": int(manifest["checkpoint_id"].nunique()),
        "all_actuators_eligible": bool(manifest["all_actuators_eligible"].all()),
        "out": args.out,
    }, indent=2))


def run_d3_main() -> None:
    parser = argparse.ArgumentParser(description="Run authoritative D3 multi-actuator SWMM sequence branches")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inp")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--control-block-seconds", type=int, required=True)
    parser.add_argument("--stride-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest).drop_duplicates("sequence_sha256")
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    results: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        inp = row.get("inp_path", args.inp)
        if pd.isna(inp) or not inp:
            inp = args.inp
        if not inp:
            raise ValueError("an INP is required via --inp or manifest inp_path")
        checkpoint = int(row["checkpoint_minutes"])
        event = str(row.get("event_id", "event"))
        sequence = json.loads(str(row["settings_sequence_json"]))
        branch_id = f"{event}__t{checkpoint:04d}__{str(row['sequence_sha256'])[:16]}"
        result = run_control_sequence_branch(
            inp_path=str(inp),
            checkpoint_minutes=checkpoint,
            settings_sequence=sequence,
            control_block_seconds=args.control_block_seconds,
            output_dir=args.out_dir,
            branch_id=branch_id,
            python_intervention_seconds=args.stride_seconds,
        )
        results.append({
            "branch_id": branch_id,
            "metadata_path": result.metadata_path,
            "flow_routing_error_pct": result.flow_routing_error_pct,
        })
    summary = Path(args.out_dir) / "RUN_SUMMARY.csv"
    pd.DataFrame(results).to_csv(summary, index=False)
    print(json.dumps({"branches": len(results), "summary": str(summary)}, indent=2))


def compile_step1_main() -> None:
    parser = argparse.ArgumentParser(description="Compile D0/D1 trajectories into causal sparse-sensor Step1 windows")
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-steps", type=int, default=13)
    parser.add_argument("--stride-steps", type=int, default=1)
    args = parser.parse_args()
    windows = compile_step1_windows(
        pd.read_csv(args.run_index),
        sensor_nodes=_load_lines(args.sensors),
        history_steps=args.history_steps,
        stride_steps=args.stride_steps,
    )
    save_step1_windows(windows, args.out)
    print(json.dumps({
        "windows": int(windows.target_state.shape[0]),
        "nodes": len(windows.node_ids),
        "context_dim": int(windows.context_history.shape[-1]),
        "out": args.out,
    }, indent=2))


def compile_step2_main() -> None:
    parser = argparse.ArgumentParser(description="Compile D2/D3 authoritative branches into Step2 rollout tensors")
    parser.add_argument("--run-index", required=True, help="CSV containing metadata_path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="development")
    args = parser.parse_args()
    index = pd.read_csv(args.run_index)
    if "metadata_path" not in index.columns:
        raise ValueError("run index requires metadata_path")
    if "scientific_split" in index.columns and args.split:
        index = index[index["scientific_split"].astype(str) == args.split]
    paths = index["metadata_path"].astype(str).tolist()
    compile_branches_to_npz(paths, args.out)
    print(json.dumps({"branches": len(paths), "split": args.split, "out": args.out}, indent=2))


def _channel_weights(target: np.ndarray) -> np.ndarray:
    axes = tuple(range(target.ndim - 1))
    variance = np.var(target, axis=axes)
    weights = 1.0 / np.maximum(variance, 1e-8)
    return weights / np.mean(weights)


def train_step1_main() -> None:
    parser = argparse.ArgumentParser(description="Train Step1 on development-only causal windows")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    args = parser.parse_args()
    ds = np.load(args.dataset, allow_pickle=False)
    graph = np.load(args.graph, allow_pickle=False)
    split = ds["scientific_splits"].astype(str)
    use = split == "development"
    if not np.any(use):
        raise ValueError("Step1 dataset has no development windows")
    model = SparseStateEstimator(
        observed_dim=int(ds["observed_history"].shape[-1]),
        static_dim=int(graph["static_node_features"].shape[-1]),
        state_dim=int(ds["target_state"].shape[-1]),
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        context_dim=int(ds["context_history"].shape[-1]),
    )
    weights = _channel_weights(ds["target_state"][use])
    history = train_step1(
        model,
        observed_history=ds["observed_history"][use],
        observation_mask=ds["observation_mask"][use],
        context_history=ds["context_history"][use],
        target_state=ds["target_state"][use],
        static_node_features=graph["static_node_features"],
        edge_index=graph["edge_index"],
        state_weights=weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    import hashlib
    digest = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config={
            "observed_dim": int(ds["observed_history"].shape[-1]),
            "static_dim": int(graph["static_node_features"].shape[-1]),
            "state_dim": int(ds["target_state"].shape[-1]),
            "hidden_dim": args.hidden_dim,
            "graph_layers": args.graph_layers,
            "context_dim": int(ds["context_history"].shape[-1]),
            "state_weights": weights.tolist(),
        },
        training_manifest_sha256=digest,
    )
    print(json.dumps({"checkpoint": meta, "final_loss": history.epoch_loss[-1]}, indent=2))


def train_step2_main() -> None:
    parser = argparse.ArgumentParser(description="Train coupled Step2 setting-to-flow-to-hydraulics world model")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=160)
    args = parser.parse_args()
    ds = np.load(args.dataset, allow_pickle=False)
    graph = np.load(args.graph, allow_pickle=False)
    model = DifferentiableHydraulicWorldModel(
        state_dim=int(ds["initial_state"].shape[-1]),
        rainfall_dim=int(ds["rainfall"].shape[-1]),
        node_static_dim=int(graph["static_node_features"].shape[-1]),
        actuator_physics_dim=int(graph["actuator_physics"].shape[-1]),
        hidden_dim=args.hidden_dim,
    )
    state_weights = _channel_weights(ds["target_states"])
    physics = np.repeat(graph["actuator_physics"][None], ds["initial_state"].shape[0], axis=0)
    flow_var = float(np.var(ds["target_actuator_flows"]))
    state_var = float(np.mean(np.var(ds["target_states"], axis=tuple(range(ds["target_states"].ndim - 1)))))
    flow_weight = state_var / max(flow_var, 1e-8)
    history = train_step2(
        model,
        initial_state=ds["initial_state"],
        rainfall=ds["rainfall"],
        settings=ds["settings"],
        previous_actuator_flow=ds["previous_actuator_flow"],
        actuator_physics=physics,
        target_states=ds["target_states"],
        target_actuator_flows=ds["target_actuator_flows"],
        actuator_upstream=graph["actuator_upstream"],
        actuator_downstream=graph["actuator_downstream"],
        static_node_features=graph["static_node_features"],
        edge_index=graph["edge_index"],
        state_weights=state_weights,
        flow_loss_weight=flow_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    import hashlib
    digest = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    meta = save_torch_checkpoint(
        model,
        args.out,
        model_config={
            "state_dim": int(ds["initial_state"].shape[-1]),
            "rainfall_dim": int(ds["rainfall"].shape[-1]),
            "node_static_dim": int(graph["static_node_features"].shape[-1]),
            "actuator_physics_dim": int(graph["actuator_physics"].shape[-1]),
            "hidden_dim": args.hidden_dim,
            "state_weights": state_weights.tolist(),
            "flow_loss_weight": flow_weight,
        },
        training_manifest_sha256=digest,
    )
    print(json.dumps({"checkpoint": meta, "final_loss": history.epoch_loss[-1]}, indent=2))


def compile_final_main() -> None:
    parser = argparse.ArgumentParser(description="Compile untouched closed-loop SWMM final runs")
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    detail = compile_closed_loop_run_index(
        pd.read_csv(args.run_index), priority_nodes=load_priority_nodes(args.priority)
    )
    summary = event_balanced_summary(detail)
    Path(args.detail_out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    print(json.dumps({"detail": args.detail_out, "summary": args.summary_out, "rows": len(detail)}, indent=2))
