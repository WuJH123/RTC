from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .acceptance import mae, nse, rank_correlation, rmse
from .context_features import build_node_context
from .contracts import load_priority_nodes
from .lazy_step1 import CausalStep1TrajectoryDataset, TrajectoryBatchSampler
from .models import DifferentiableHydraulicWorldModel, SparseStateEstimator
from .production_cli import _load_graph, _load_step1, _load_step2
from .step2_shards import compile_step2_shards, load_shard_manifest, sha256_file
from .training import save_torch_checkpoint


class StreamingStats:
    def __init__(self, dim: int):
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64).reshape(-1, self.sum.size)
        finite = np.isfinite(x).all(axis=1)
        x = x[finite]
        if x.size == 0:
            return
        self.count += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("normalization statistics received no finite samples")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 1e-12)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type == "cuda")


def _read_lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _filtered_index(path: str, *, split: str, fold: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["scientific_split"].astype(str) == split].copy()
    if split == "development" and fold is not None:
        if "development_fold" not in frame.columns:
            raise ValueError("development index lacks development_fold")
        frame = frame[frame["development_fold"].astype(str) == fold]
    if frame.empty:
        raise ValueError("no rows remain after scientific split/fold filtering")
    return frame


def _step1_normalization(index: pd.DataFrame, graph, sensors: tuple[str, ...]):
    state_stats = StreamingStats(6)
    obs_stats = StreamingStats(2)
    context_stats = StreamingStats(5)
    sensor_idx = np.asarray([graph.node_ids.index(n) for n in sensors], dtype=int)
    for _, row in index.iterrows():
        meta_path = Path(str(row["metadata_path"]))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(meta_path.parent / str(meta["compact_file"]), allow_pickle=False) as raw:
            state = raw["state_si"].astype(np.float32)
            rain = raw["rainfall_mmhr"].astype(np.float32)
            setting = raw["current_setting"].astype(np.float32)
            flow = raw["actuator_flow_m3s"].astype(np.float32)
        state_stats.update(state)
        obs_stats.update(state[:, sensor_idx, :2])
        context_stats.update(build_node_context(
            rainfall_mmhr=rain,
            actuator_setting=setting,
            actuator_flow_m3s=flow,
            actuator_upstream=graph.actuator_upstream,
            actuator_downstream=graph.actuator_downstream,
            node_count=len(graph.node_ids),
        ))
    static_stats = StreamingStats(graph.static_node_features.shape[-1])
    static_stats.update(graph.static_node_features)
    return obs_stats.finish(), context_stats.finish(), state_stats.finish(), static_stats.finish()


def train_step1_large_main() -> None:
    parser = argparse.ArgumentParser(description="Train Step1 lazily on compact large-network trajectories")
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-steps", type=int, required=True)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    graph = _load_graph(args.graph)
    sensors = _read_lines(args.sensors)
    index = _filtered_index(args.run_index, split="development", fold="train")
    (obs_mean, obs_std), (ctx_mean, ctx_std), (state_mean, state_std), (static_mean, static_std) = _step1_normalization(index, graph, sensors)
    dataset = CausalStep1TrajectoryDataset(
        index,
        graph=graph,
        sensor_nodes=sensors,
        history_steps=args.history_steps,
        model_step_seconds=args.model_step_seconds,
        scientific_split="development",
        development_fold="train",
        cache_trajectories=2,
    )
    sampler = TrajectoryBatchSampler(dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    model = SparseStateEstimator(
        observed_dim=2,
        static_dim=graph.static_node_features.shape[-1],
        state_dim=6,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        context_dim=5,
    ).to(device)
    model.set_normalization(
        observed_mean=torch.as_tensor(obs_mean, device=device), observed_std=torch.as_tensor(obs_std, device=device),
        static_mean=torch.as_tensor(static_mean, device=device), static_std=torch.as_tensor(static_std, device=device),
        context_mean=torch.as_tensor(ctx_mean, device=device), context_std=torch.as_tensor(ctx_std, device=device),
        state_mean=torch.as_tensor(state_mean, device=device), state_std=torch.as_tensor(state_std, device=device),
    )
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    amp = _amp_enabled(device, not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history: list[float] = []
    for _epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        samples = 0
        for step, (obs, mask, context, target) in enumerate(loader):
            obs, mask, context, target = [x.to(device, non_blocking=True) for x in (obs, mask, context, target)]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                pred = model(obs, mask, static, edges, context)
                loss = (((pred - target) / model.state_std) ** 2).mean() / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss.detach()) * args.grad_accum * obs.shape[0]
            samples += obs.shape[0]
        history.append(running / max(samples, 1))
    config = {
        "observed_dim": 2,
        "static_dim": int(graph.static_node_features.shape[-1]),
        "state_dim": 6,
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "context_dim": 5,
        "history_steps": args.history_steps,
        "model_step_seconds": args.model_step_seconds,
        "context_contract": "NODE_LOCAL_CAUSAL_CONTEXT_V1",
    }
    meta = save_torch_checkpoint(
        model, args.out, model_config=config,
        training_manifest_sha256=_sha(args.run_index), scientific_split="development",
    )
    print(json.dumps({
        "checkpoint": meta, "windows": len(dataset), "final_normalized_mse": history[-1],
        "device": str(device), "amp": amp, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
    }, indent=2))


def accept_step1_large_main() -> None:
    parser = argparse.ArgumentParser(description="Accept Step1 on lazy development/validation trajectories")
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--history-steps", type=int, required=True)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--priority")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device")
    args = parser.parse_args()
    device = _device(args.device)
    graph = _load_graph(args.graph)
    sensors = _read_lines(args.sensors)
    index = _filtered_index(args.run_index, split="development", fold="validation")
    dataset = CausalStep1TrajectoryDataset(
        index, graph=graph, sensor_nodes=sensors, history_steps=args.history_steps,
        model_step_seconds=args.model_step_seconds, scientific_split="development",
        development_fold="validation", cache_trajectories=2,
    )
    sampler = TrajectoryBatchSampler(dataset, batch_size=args.batch_size, seed=0, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    model = _load_step1(args.model, device)
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    pred_all, truth_all, events = [], [], []
    cursor = 0
    with torch.no_grad():
        for obs, mask, context, target in loader:
            pred = model(obs.to(device), mask.to(device), static, edges, context.to(device)).cpu().numpy()
            pred_all.append(pred)
            truth_all.append(target.numpy())
            for _ in range(len(target)):
                events.append(dataset.samples[cursor].event_id)
                cursor += 1
    pred, truth = np.concatenate(pred_all), np.concatenate(truth_all)
    sensor_idx = np.asarray([graph.node_ids.index(n) for n in sensors], dtype=int)
    unobserved = np.asarray(sorted(set(range(len(graph.node_ids))) - set(sensor_idx.tolist())), dtype=int)
    metrics: dict[str, float] = {
        "unobserved_depth_rmse_m": rmse(pred[:, unobserved, 0], truth[:, unobserved, 0]),
        "unobserved_depth_nse": nse(pred[:, unobserved, 0], truth[:, unobserved, 0]),
        "all_state_normalized_rmse": float(np.sqrt(np.mean(((pred-truth)/model.state_std.cpu().numpy())**2))),
    }
    if args.priority:
        priority = load_priority_nodes(args.priority)
        missing = sorted(set(priority) - set(graph.node_ids))
        if missing:
            raise ValueError(f"priority mapping incompatible with graph: {missing}")
        pidx = np.asarray([graph.node_ids.index(n) for n in priority], dtype=int)
        metrics["priority_depth_rmse_m"] = rmse(pred[:, pidx, 0], truth[:, pidx, 0])
        metrics["priority_depth_nse"] = nse(pred[:, pidx, 0], truth[:, pidx, 0])
    payload = {
        "contract": "STEP1_LARGE_HELDOUT_ACCEPTANCE_V2",
        "model_sha256": _sha(args.model), "metrics": metrics,
        "validation_windows": int(len(dataset)), "priority_diagnostic_only": True,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def compile_step2_shards_main() -> None:
    parser = argparse.ArgumentParser(description="Compile compact D2/D3 branches into bounded-memory shards")
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--development-fold", choices=["train", "validation", "all"], default="train")
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()
    frame = pd.read_csv(args.run_index)
    if "scientific_split" in frame.columns:
        frame = frame[frame["scientific_split"].astype(str) == args.split]
    if args.split == "development" and args.development_fold != "all":
        frame = frame[frame["development_fold"].astype(str) == args.development_fold]
    manifest = compile_step2_shards(frame, output_dir=args.out_dir, shard_size=args.shard_size)
    print(json.dumps({"manifest": str(manifest), "branches": len(frame)}, indent=2))


def _step2_stats(manifest, graph):
    state_stats = StreamingStats(6)
    rain_stats = None
    flow_sq_sum = 0.0
    flow_count = 0
    for item in manifest["shards"]:
        with np.load(str(item["path"]), allow_pickle=False) as ds:
            state_stats.update(ds["initial_state"])
            state_stats.update(ds["target_states"])
            if rain_stats is None:
                rain_stats = StreamingStats(ds["rainfall"].shape[-1])
            rain_stats.update(ds["rainfall"])
            for values in (ds["previous_actuator_flow"], ds["target_actuator_flows"]):
                x = values.astype(np.float64)
                flow_sq_sum += float(np.square(x).sum())
                flow_count += x.size
    if rain_stats is None or flow_count == 0:
        raise ValueError("empty Step2 shards")
    physics_stats = StreamingStats(graph.actuator_physics.shape[-1]); physics_stats.update(graph.actuator_physics)
    static_stats = StreamingStats(graph.static_node_features.shape[-1]); static_stats.update(graph.static_node_features)
    state = state_stats.finish(); rain = rain_stats.finish(); physics = physics_stats.finish(); static = static_stats.finish()
    flow_std = np.asarray([max(np.sqrt(flow_sq_sum / flow_count), 1e-6)], dtype=np.float32)
    return state, rain, physics, static, flow_std


def train_step2_large_main() -> None:
    parser = argparse.ArgumentParser(description="Train Step2 from bounded-memory shards with AMP")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--actuator-embedding-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = _device(args.device)
    graph = _load_graph(args.graph)
    manifest = load_shard_manifest(args.manifest)
    (state_mean, state_std), (rain_mean, rain_std), (physics_mean, physics_std), (static_mean, static_std), flow_std = _step2_stats(manifest, graph)
    first = np.load(str(manifest["shards"][0]["path"]), allow_pickle=False)
    config = {
        "state_dim": int(first["initial_state"].shape[-1]),
        "rainfall_dim": int(first["rainfall"].shape[-1]),
        "node_static_dim": int(graph.static_node_features.shape[-1]),
        "actuator_physics_dim": int(graph.actuator_physics.shape[-1]),
        "hidden_dim": args.hidden_dim,
        "actuator_count": len(graph.actuator_ids),
        "actuator_embedding_dim": args.actuator_embedding_dim,
    }
    first.close()
    model = DifferentiableHydraulicWorldModel(**config).to(device)
    model.set_normalization(
        state_mean=torch.as_tensor(state_mean, device=device), state_std=torch.as_tensor(state_std, device=device),
        rain_mean=torch.as_tensor(rain_mean, device=device), rain_std=torch.as_tensor(rain_std, device=device),
        static_mean=torch.as_tensor(static_mean, device=device), static_std=torch.as_tensor(static_std, device=device),
        physics_mean=torch.as_tensor(physics_mean, device=device), physics_std=torch.as_tensor(physics_std, device=device),
        flow_std=torch.as_tensor(flow_std, device=device),
    )
    up = torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device)
    down = torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device)
    static = torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device)
    edges = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    physics = torch.as_tensor(graph.actuator_physics, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    amp = _amp_enabled(device, not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rng = np.random.default_rng(args.seed)
    history = []
    for _epoch in range(args.epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        shard_order = list(manifest["shards"]); rng.shuffle(shard_order)
        running = 0.0; seen = 0; micro_step = 0
        for item in shard_order:
            with np.load(str(item["path"]), allow_pickle=False) as ds:
                tensors = [torch.from_numpy(ds[name].astype(np.float32)) for name in (
                    "initial_state", "rainfall", "settings", "previous_actuator_flow", "target_states", "target_actuator_flows"
                )]
            loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type=="cuda")
            for initial, rain, settings, prev, target_state, target_flow in loader:
                initial, rain, settings, prev, target_state, target_flow = [x.to(device, non_blocking=True) for x in (initial,rain,settings,prev,target_state,target_flow)]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    rollout = model.rollout(initial, rain, settings, prev, up, down, physics.unsqueeze(0).expand(initial.shape[0],-1,-1), static, edges)
                    state_loss = (((rollout.states-target_state)/model.transition.state_std)**2).mean()
                    flow_loss = (((rollout.actuator_flows-target_flow)/model.actuator.flow_std)**2).mean()
                    loss = (state_loss + flow_loss) / args.grad_accum
                scaler.scale(loss).backward(); micro_step += 1
                if micro_step % args.grad_accum == 0:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                running += float(loss.detach())*args.grad_accum*initial.shape[0]; seen += initial.shape[0]
        if micro_step % args.grad_accum:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        history.append(running/max(seen,1))
    meta = save_torch_checkpoint(
        model, args.out, model_config=config,
        training_manifest_sha256=sha256_file(args.manifest), scientific_split="development",
    )
    print(json.dumps({
        "checkpoint": meta, "final_normalized_loss": history[-1], "device": str(device),
        "amp": amp, "micro_batch": args.batch_size, "grad_accum": args.grad_accum,
    }, indent=2))


def accept_step2_large_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step2 shards using exact SWMM cumulative flood-volume truth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--priority")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device")
    args = parser.parse_args()
    device = _device(args.device); graph = _load_graph(args.graph); model = _load_step2(args.model, device)
    manifest = load_shard_manifest(args.manifest)
    up=torch.as_tensor(graph.actuator_upstream,dtype=torch.long,device=device); down=torch.as_tensor(graph.actuator_downstream,dtype=torch.long,device=device)
    static=torch.as_tensor(graph.static_node_features,dtype=torch.float32,device=device); edges=torch.as_tensor(graph.edge_index,dtype=torch.long,device=device)
    physics=torch.as_tensor(graph.actuator_physics,dtype=torch.float32,device=device)
    pred_depth=[]; true_depth=[]; pred_flow=[]; true_flow=[]; pred_tfv=[]; true_tfv=[]; pred_pfv=[]; true_pfv=[]
    pidx=None
    if args.priority:
        priority=load_priority_nodes(args.priority); missing=sorted(set(priority)-set(graph.node_ids))
        if missing: raise ValueError(f"priority mapping incompatible with graph: {missing}")
        pidx=np.asarray([graph.node_ids.index(n) for n in priority],dtype=int)
    with torch.no_grad():
        for item in manifest["shards"]:
            with np.load(str(item["path"]),allow_pickle=False) as ds:
                if "exact_node_flood_volume_m3" not in ds.files:
                    raise ValueError("Step2 acceptance requires exact SWMM node flooding-volume truth in every shard")
                count=ds["initial_state"].shape[0]
                for start in range(0,count,args.batch_size):
                    end=min(count,start+args.batch_size); b=end-start
                    initial=torch.as_tensor(ds["initial_state"][start:end],dtype=torch.float32,device=device)
                    rain=torch.as_tensor(ds["rainfall"][start:end],dtype=torch.float32,device=device)
                    settings=torch.as_tensor(ds["settings"][start:end],dtype=torch.float32,device=device)
                    prev=torch.as_tensor(ds["previous_actuator_flow"][start:end],dtype=torch.float32,device=device)
                    rollout=model.rollout(initial,rain,settings,prev,up,down,physics.unsqueeze(0).expand(b,-1,-1),static,edges)
                    ps=rollout.states.cpu().numpy(); pf=rollout.actuator_flows.cpu().numpy()
                    ts=ds["target_states"][start:end]; tf=ds["target_actuator_flows"][start:end]
                    dt=np.diff(ds["elapsed_seconds"][start:end].astype(float),axis=1)
                    pred_node=(np.clip(ps[...,2],0,None)*dt[:,:,None]).sum(axis=1)
                    exact=ds["exact_node_flood_volume_m3"][start:end].astype(float)
                    pred_depth.append(ps[...,0]); true_depth.append(ts[...,0]); pred_flow.append(pf); true_flow.append(tf)
                    pred_tfv.extend(pred_node.sum(axis=1)); true_tfv.extend(exact.sum(axis=1))
                    if pidx is not None:
                        pred_pfv.extend(pred_node[:,pidx].sum(axis=1)); true_pfv.extend(exact[:,pidx].sum(axis=1))
    metrics={
        "depth_rmse_m":rmse(np.concatenate(pred_depth),np.concatenate(true_depth)),
        "managed_flow_rmse_m3s":rmse(np.concatenate(pred_flow),np.concatenate(true_flow)),
        "tfv_exact_truth_mae_m3":mae(np.asarray(pred_tfv),np.asarray(true_tfv)),
        "tfv_exact_truth_rank_correlation":rank_correlation(np.asarray(pred_tfv),np.asarray(true_tfv)),
    }
    if pidx is not None:
        metrics["priority_flood_exact_truth_mae_m3"]=mae(np.asarray(pred_pfv),np.asarray(true_pfv))
        metrics["priority_flood_exact_truth_rank_correlation"]=rank_correlation(np.asarray(pred_pfv),np.asarray(true_pfv))
    payload={
        "contract":"STEP2_LARGE_EXACT_TRUTH_ACCEPTANCE_V2","model_sha256":_sha(args.model),
        "manifest_sha256":sha256_file(args.manifest),"metrics":metrics,"priority_diagnostic_only":True,
        "truth_source_tfv_pfv":"SWMM_NODE_STATISTICS_CUMULATIVE_EXACT_HORIZON",
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))
