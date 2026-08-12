"""V6 training/data contracts with group/event balancing and critical-hydraulic weighting."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v60 import (
    ControlValueSurrogateV60, HydraulicOutputV60, HydraulicResponseSurrogateV60,
    PreparedStaticV60, prepare_static_v60,
)
from .step2_training_cache import load_step2_training_cache
from .step2_v60_contract import HydraulicLossContractV60, MultiResolutionHorizonV60, ValueLossContractV60


@dataclass(frozen=True)
class InputNormalizationV60:
    state_mean: np.ndarray
    state_std: np.ndarray
    rainfall_mean: np.ndarray
    rainfall_std: np.ndarray
    flow_mean: np.ndarray
    flow_std: np.ndarray


@dataclass(frozen=True)
class TargetScalesV60:
    state_scale: np.ndarray
    flow_scale: np.ndarray
    d2_tfv_scale_m3: float
    d3_tfv_scale_m3: float
    tfv_rate_scale_m3s: float

    def tfv_scale(self, source_kind: str) -> float:
        source = str(source_kind).upper()
        if source == "D2":
            return float(self.d2_tfv_scale_m3)
        if source == "D3":
            return float(self.d3_tfv_scale_m3)
        raise ValueError(f"unsupported V6 source kind: {source_kind}")


@dataclass(frozen=True)
class V60GroupBatch:
    source_kind: str
    group_name: str
    initial_state: torch.Tensor
    rainfall: torch.Tensor
    reference_settings: torch.Tensor
    candidate_settings: torch.Tensor
    previous_actuator_flow: torch.Tensor
    elapsed_seconds: torch.Tensor
    true_reference_states: torch.Tensor
    true_candidate_states: torch.Tensor
    true_reference_flows: torch.Tensor
    true_candidate_flows: torch.Tensor
    true_delta_tfv_m3: torch.Tensor


@dataclass(frozen=True)
class _GroupEntry:
    arrays: dict[str, np.ndarray]
    indices: tuple[int, ...]
    reference_index: int
    source_kind: str
    rainfall_group: str
    event_id: str
    checkpoint_id: str
    candidate_roles: tuple[str, ...]


def _string_array(arrays: dict[str, np.ndarray], name: str, count: int, default: str = "") -> np.ndarray:
    return np.asarray(arrays[name]).astype(str) if name in arrays else np.asarray([default] * count)


def _group_index(arrays: dict[str, np.ndarray]) -> dict[str, _GroupEntry]:
    count = int(arrays["initial_state"].shape[0])
    source = _string_array(arrays, "source_kind", count, "D2")
    rain, event = _string_array(arrays, "rainfall_group", count), _string_array(arrays, "event_id", count)
    checkpoint, roles = _string_array(arrays, "checkpoint_id", count), _string_array(arrays, "data_role", count)
    action_sha, base_sha = _string_array(arrays, "action_or_sequence_sha256", count), _string_array(arrays, "base_action_sha256", count)
    raw: dict[str, list[int]] = {}
    for i in range(count):
        raw.setdefault("::".join((source[i], rain[i], event[i], checkpoint[i])), []).append(i)
    result: dict[str, _GroupEntry] = {}
    for name, indices in raw.items():
        src = str(source[indices[0]]).upper()
        if src == "D3":
            hold = [i for i in indices if str(roles[i]).strip().lower() == "d3_hold_reference"]
            if len(hold) != 1:
                raise ValueError(f"{name}: D3 requires exactly one D3_HOLD_REFERENCE")
            ref = int(hold[0])
        else:
            ref_candidates = [i for i in indices if base_sha[i] and base_sha[i] == action_sha[i]]
            named = [i for i in indices if str(roles[i]).strip().lower() in {"base", "reference", "hold", "center"}]
            ref = int(ref_candidates[0] if ref_candidates else named[0] if named else min(indices, key=lambda i: action_sha[i]))
        result[name] = _GroupEntry(
            arrays=arrays, indices=tuple(int(i) for i in indices), reference_index=ref, source_kind=src,
            rainfall_group=str(rain[ref]), event_id=str(event[ref]), checkpoint_id=str(checkpoint[ref]),
            candidate_roles=tuple(str(roles[i]) for i in indices if i != ref),
        )
    return result


class V60TrainCache:
    """Group-preserving cache view. Legacy dense D3 is diagnostic-only."""
    def __init__(self, manifest_path: str | Path) -> None:
        cache = load_step2_training_cache(manifest_path)
        index: dict[str, _GroupEntry] = {}
        for shard in cache["shards"]:
            for name, entry in _group_index(shard["arrays"]).items():
                if name in index:
                    raise ValueError(f"duplicate counterfactual group across V6 shards: {name}")
                index[name] = entry
        self._index = index
        self.manifest_path = str(Path(manifest_path))

    def names(self, source: str | None = None) -> list[str]:
        names = sorted(self._index)
        return names if source is None else [name for name in names if name.startswith(source.upper() + "::")]

    def entry(self, name: str) -> _GroupEntry:
        return self._index[name]

    def is_targeted_d3_v60(self, name: str) -> bool:
        entry = self.entry(name)
        if entry.source_kind != "D3":
            return False
        roles = {role.strip().upper() for role in entry.candidate_roles}
        return bool(roles) and roles <= {"D3_V60_MANIFOLD_CANDIDATE", "D3_V60_ACTIVE_LEARNING_CANDIDATE"}

    def targeted_d3_names(self) -> list[str]:
        return [name for name in self.names("D3") if self.is_targeted_d3_v60(name)]

    def legacy_d3_names(self) -> list[str]:
        return [name for name in self.names("D3") if not self.is_targeted_d3_v60(name)]

    def batch(self, name: str, normalization: InputNormalizationV60, device: torch.device | str) -> V60GroupBatch:
        entry, arrays, ref = self.entry(name), self.entry(name).arrays, self.entry(name).reference_index
        candidates = [i for i in entry.indices if i != ref]
        if not candidates:
            raise ValueError(f"{name}: group has no candidates")
        target = torch.device(device)
        def t(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).to(target)
        initial_raw = np.asarray(arrays["initial_state"][ref], dtype=np.float32)
        rain_raw = np.asarray(arrays["rainfall"][ref], dtype=np.float32)
        flow_raw = np.asarray(arrays["previous_actuator_flow"][ref], dtype=np.float32)
        initial = (initial_raw - normalization.state_mean) / np.maximum(normalization.state_std, 1e-6)
        rain = (rain_raw - normalization.rainfall_mean) / np.maximum(normalization.rainfall_std, 1e-6)
        previous_flow = (flow_raw - normalization.flow_mean) / np.maximum(normalization.flow_std, 1e-6)
        reference_tfv = float(np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum())
        candidate_tfv = np.asarray(arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64).sum(axis=1)
        return V60GroupBatch(
            source_kind=entry.source_kind, group_name=name,
            initial_state=t(initial)[None], rainfall=t(rain)[None],
            reference_settings=t(np.asarray(arrays["settings"][ref], dtype=np.float32))[None],
            candidate_settings=t(np.asarray(arrays["settings"][candidates], dtype=np.float32))[None],
            previous_actuator_flow=t(previous_flow)[None],
            elapsed_seconds=t(np.asarray(arrays["elapsed_seconds"][ref], dtype=np.float32))[None],
            true_reference_states=t(np.asarray(arrays["target_states"][ref], dtype=np.float32))[None],
            true_candidate_states=t(np.asarray(arrays["target_states"][candidates], dtype=np.float32))[None],
            true_reference_flows=t(np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float32))[None],
            true_candidate_flows=t(np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float32))[None],
            true_delta_tfv_m3=t((candidate_tfv - reference_tfv).astype(np.float32))[None],
        )


def deterministic_rainfall_split_v60(cache: V60TrainCache, *, names: Sequence[str] | None = None, holdout_fraction: float = 0.2) -> tuple[list[str], list[str]]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must lie in (0,0.5)")
    selected = list(cache.names() if names is None else names)
    if not selected:
        raise ValueError("cannot split an empty V6 group set")
    rain_groups = sorted({cache.entry(name).rainfall_group for name in selected})
    holdout_rain = {rain for rain in rain_groups if int(hashlib.sha256(rain.encode()).hexdigest()[:8], 16) % 10000 < int(round(holdout_fraction * 10000))}
    if not holdout_rain and rain_groups:
        holdout_rain.add(rain_groups[-1])
    fit = [name for name in selected if cache.entry(name).rainfall_group not in holdout_rain]
    holdout = [name for name in selected if cache.entry(name).rainfall_group in holdout_rain]
    if not fit or not holdout:
        raise ValueError("deterministic V6 rainfall split produced an empty side")
    return fit, holdout


def _channel_stats(values: Iterable[np.ndarray], channels: int) -> tuple[np.ndarray, np.ndarray]:
    total, square, count = np.zeros(channels), np.zeros(channels), 0
    for value in values:
        x = np.asarray(value, dtype=np.float64).reshape(-1, channels)
        total += x.sum(axis=0); square += np.square(x).sum(axis=0); count += x.shape[0]
    if count <= 0:
        raise ValueError("cannot derive normalization from no values")
    mean = total / count
    return mean.astype(np.float32), np.sqrt(np.maximum(square / count - mean * mean, 1e-12)).astype(np.float32)


def derive_input_normalization_v60(cache: V60TrainCache, fit_names: Sequence[str]) -> InputNormalizationV60:
    refs = [cache.entry(name) for name in fit_names]
    first = refs[0].arrays
    sm, ss = _channel_stats((e.arrays["initial_state"][e.reference_index] for e in refs), int(first["initial_state"].shape[-1]))
    rm, rs = _channel_stats((e.arrays["rainfall"][e.reference_index] for e in refs), int(first["rainfall"].shape[-1]))
    fm, fs = _channel_stats((e.arrays["previous_actuator_flow"][e.reference_index] for e in refs), int(first["previous_actuator_flow"].shape[-1]))
    return InputNormalizationV60(sm, ss, rm, rs, fm, fs)


def derive_target_scales_v60(cache: V60TrainCache, fit_names: Sequence[str], horizon: MultiResolutionHorizonV60 = MultiResolutionHorizonV60()) -> TargetScalesV60:
    indices = np.asarray(horizon.indices(), dtype=np.int64)
    first = cache.entry(fit_names[0]).arrays
    state_dim, actuator_count = int(first["target_states"].shape[-1]), int(first["target_actuator_flows"].shape[-1])
    state_square, flow_square, state_count, flow_count = np.zeros(state_dim), np.zeros(actuator_count), 0, 0
    tfv_values: dict[str, list[float]] = {"D2": [], "D3": []}
    for name in fit_names:
        entry, arrays, ref = cache.entry(name), cache.entry(name).arrays, cache.entry(name).reference_index
        candidates = [i for i in entry.indices if i != ref]
        states = np.asarray(arrays["target_states"][candidates], dtype=np.float64)[:, indices]
        flows = np.asarray(arrays["target_actuator_flows"][candidates], dtype=np.float64)[:, indices]
        state_square += np.square(states).reshape(-1, state_dim).sum(axis=0); state_count += int(np.prod(states.shape[:-1]))
        flow_square += np.square(flows).reshape(-1, actuator_count).sum(axis=0); flow_count += int(np.prod(flows.shape[:-1]))
        ref_tfv = float(np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum())
        cand = np.asarray(arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64).sum(axis=1)
        tfv_values[entry.source_kind.upper()].extend((cand - ref_tfv).tolist())
    def rms(values: list[float]) -> float:
        x = np.asarray(values, dtype=np.float64)
        return max(float(np.sqrt(np.mean(np.square(x)))) if x.size else 1.0, 1.0)
    d2, d3 = rms(tfv_values["D2"]), rms(tfv_values["D3"])
    return TargetScalesV60(
        state_scale=np.maximum(np.sqrt(state_square / max(state_count, 1)).astype(np.float32), 1e-6),
        flow_scale=np.maximum(np.sqrt(flow_square / max(flow_count, 1)).astype(np.float32), 1e-6),
        d2_tfv_scale_m3=d2, d3_tfv_scale_m3=d3,
        tfv_rate_scale_m3s=max(float(np.sqrt(d2 * d3)) / (horizon.horizon_steps * horizon.model_step_seconds), 1e-6),
    )


def listwise_loss_v60(predicted: torch.Tensor, truth: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("listwise loss expects [B,C]")
    temperature = torch.as_tensor(scale, dtype=predicted.dtype, device=predicted.device).clamp_min(1.0)
    score, order = -predicted / temperature, torch.argsort(truth.detach(), dim=1)
    mask, losses = torch.ones_like(score, dtype=torch.bool), []
    for position in range(score.shape[1]):
        log_probs = torch.log_softmax(score.masked_fill(~mask, float("-inf")), dim=1)
        chosen = order[:, position]
        losses.append(-log_probs.gather(1, chosen[:, None]).squeeze(1))
        mask = mask.scatter(1, chosen[:, None], False)
    return torch.stack(losses, dim=1).mean()


def value_loss_v60(predicted: torch.Tensor, truth: torch.Tensor, *, scale_m3: float, contract: ValueLossContractV60 = ValueLossContractV60()) -> tuple[torch.Tensor, dict[str, float]]:
    scale = torch.as_tensor(scale_m3, dtype=predicted.dtype, device=predicted.device).clamp_min(1.0)
    exact = F.smooth_l1_loss((predicted - truth.detach()) / scale, torch.zeros_like(predicted))
    spread = (truth.detach().amax(dim=1) - truth.detach().amin(dim=1)).clamp_min(0.05 * scale)
    centred_error = (predicted - predicted.mean(dim=1, keepdim=True) - truth.detach() + truth.detach().mean(dim=1, keepdim=True)) / spread[:, None]
    centred = F.smooth_l1_loss(centred_error, torch.zeros_like(centred_error))
    listwise = listwise_loss_v60(predicted, truth, scale)
    probability = torch.softmax(-predicted / scale, dim=1)
    regret = (((probability * truth.detach()).sum(dim=1) - truth.detach().amin(dim=1)) / scale).clamp_min(0.0).mean()
    total = contract.exact_delta_tfv_weight * exact + contract.group_centered_weight * centred + contract.listwise_rank_weight * listwise + contract.regret_weight * regret
    return total, {"loss": float(total.detach()), "exact": float(exact.detach()), "centered": float(centred.detach()), "listwise": float(listwise.detach()), "regret_surrogate": float(regret.detach())}


def hydraulic_critical_weights_v60(true_candidate_states: torch.Tensor, prepared: PreparedStaticV60, *, contract: HydraulicLossContractV60 = HydraulicLossContractV60()) -> torch.Tensor:
    """Node-balanced weights for wet, near-surcharge and storage-capacity states."""
    contract.validate()
    depth, volume = true_candidate_states[..., 0].clamp_min(0.0), true_candidate_states[..., 3].clamp_min(0.0)
    max_depth = prepared.max_depth_m.to(depth); surcharge = (prepared.max_depth_m + prepared.surcharge_depth_m).to(depth)
    capacity, storage_mask = prepared.storage_capacity_m3.to(depth), prepared.storage_mask.to(depth.device)
    while max_depth.ndim < depth.ndim:
        max_depth=max_depth.unsqueeze(0); surcharge=surcharge.unsqueeze(0); capacity=capacity.unsqueeze(0); storage_mask=storage_mask.unsqueeze(0)
    wet = torch.where(max_depth > 1e-3, (depth / max_depth.clamp_min(1e-3)).clamp(0, 2).sqrt(), torch.zeros_like(depth))
    near = torch.where(surcharge > 1e-3, torch.sigmoid(((depth / surcharge.clamp_min(1e-3)) - contract.near_surcharge_start_ratio) / 0.05), torch.zeros_like(depth))
    storage = torch.where(storage_mask, (volume / capacity.clamp_min(1e-3)).clamp(0, 2), torch.zeros_like(volume))
    weights = 1 + contract.wet_node_gain * wet + contract.near_surcharge_gain * near + contract.storage_proximity_gain * storage
    return weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)


def hydraulic_loss_v60(output: HydraulicOutputV60, batch: V60GroupBatch, prepared: PreparedStaticV60, scales: TargetScalesV60, *, horizon: MultiResolutionHorizonV60 = MultiResolutionHorizonV60(), contract: HydraulicLossContractV60 = HydraulicLossContractV60()) -> tuple[torch.Tensor, dict[str, float]]:
    idx = output.horizon_indices
    true_ref = batch.true_reference_states.index_select(1, idx)[:, None].expand_as(output.reference_states_physical)
    true_cand = batch.true_candidate_states.index_select(2, idx)
    true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None].expand_as(output.reference_flows_physical)
    true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
    weights = hydraulic_critical_weights_v60(true_cand, prepared, contract=contract)
    time_weight = torch.as_tensor(horizon.weights(), dtype=weights.dtype, device=weights.device).reshape(1, 1, -1, 1)
    weight = weights * time_weight
    state_scale = torch.as_tensor(scales.state_scale, dtype=weights.dtype, device=weights.device)
    flow_scale = torch.as_tensor(scales.flow_scale, dtype=weights.dtype, device=weights.device)
    def channel(pred, truth, c):
        err = F.smooth_l1_loss((pred[..., c] - truth[..., c]) / state_scale[c].clamp_min(1e-6), torch.zeros_like(pred[..., c]), reduction="none")
        return (err * weight).mean()
    depth = 0.5 * (channel(output.reference_states_physical, true_ref, 0) + channel(output.candidate_states_physical, true_cand, 0))
    flood = 0.5 * (channel(output.reference_states_physical, true_ref, 2) + channel(output.candidate_states_physical, true_cand, 2))
    storage = 0.5 * (channel(output.reference_states_physical, true_ref, 3) + channel(output.candidate_states_physical, true_cand, 3))
    flow = 0.5 * (F.smooth_l1_loss(output.reference_flows_physical / flow_scale, true_ref_flow / flow_scale) + F.smooth_l1_loss(output.candidate_flows_physical / flow_scale, true_cand_flow / flow_scale))
    ref_target = (true_ref[..., 2] > contract.onset_epsilon_m3s).to(weights.dtype); cand_target = (true_cand[..., 2] > contract.onset_epsilon_m3s).to(weights.dtype)
    onset = 0.5 * ((F.binary_cross_entropy_with_logits(output.reference_flood_onset_logits, ref_target, reduction="none") + F.binary_cross_entropy_with_logits(output.candidate_flood_onset_logits, cand_target, reduction="none")) * weight).mean()
    total = contract.depth_weight * depth + contract.flooding_weight * flood + contract.storage_weight * storage + contract.managed_flow_weight * flow + contract.flooding_onset_weight * onset
    return total, {"loss": float(total.detach()), "depth": float(depth.detach()), "flooding": float(flood.detach()), "storage": float(storage.detach()), "managed_flow": float(flow.detach()), "flooding_onset": float(onset.detach())}


def _spearman(predicted: np.ndarray, truth: np.ndarray) -> float:
    if predicted.size < 2 or np.allclose(predicted, predicted[0]) or np.allclose(truth, truth[0]):
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(predicted)).astype(float), np.argsort(np.argsort(truth)).astype(float))[0, 1])


def _pairwise_accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    total=correct=0
    for i in range(len(truth)):
        for j in range(i+1, len(truth)):
            if abs(float(truth[i]-truth[j])) <= 1e-9: continue
            total += 1; correct += int(np.sign(predicted[i]-predicted[j]) == np.sign(truth[i]-truth[j]))
    return float(correct/total) if total else float("nan")


def evaluate_value_v60(model: ControlValueSurrogateV60, cache: V60TrainCache, names: Sequence[str], normalization: InputNormalizationV60, prepared: PreparedStaticV60, *, device: torch.device | str) -> dict[str, Any]:
    model.eval(); ranks=[]; pairwise=[]; top1=0; regrets=[]; maes=[]
    with torch.no_grad():
        for name in names:
            batch=cache.batch(name,normalization,device); output=model(batch.initial_state,batch.rainfall,batch.reference_settings,batch.candidate_settings,prepared,batch.elapsed_seconds)
            pred=output.delta_tfv_m3[0].cpu().numpy(); truth=batch.true_delta_tfv_m3[0].cpu().numpy()
            ranks.append(_spearman(pred,truth)); pairwise.append(_pairwise_accuracy(pred,truth)); bp=int(np.argmin(pred)); bt=int(np.argmin(truth)); top1 += int(bp==bt); regrets.append(float(truth[bp]-truth[bt])); maes.append(float(np.mean(np.abs(pred-truth))))
    fr=[x for x in ranks if np.isfinite(x)]; fp=[x for x in pairwise if np.isfinite(x)]
    return {"groups":len(names),"rank":float(np.mean(fr)) if fr else float("nan"),"pairwise":float(np.mean(fp)) if fp else float("nan"),"top1":int(top1),"top1_denominator":len(names),"mean_regret_m3":float(np.mean(regrets)) if regrets else float("nan"),"max_regret_m3":float(np.max(regrets)) if regrets else float("nan"),"tfv_mae_m3":float(np.mean(maes)) if maes else float("nan")}


def evaluate_hydraulic_v60(model: HydraulicResponseSurrogateV60, cache: V60TrainCache, names: Sequence[str], normalization: InputNormalizationV60, graph: Any, *, device: torch.device | str) -> dict[str, Any]:
    target=torch.device(device); model.to(target).eval(); prepared=prepare_static_v60(graph,target); d=[]; f=[]; s=[]; q=[]; ba=[]; storage_mask=prepared.storage_mask.detach().cpu().numpy().astype(bool)
    with torch.no_grad():
        for name in names:
            batch=cache.batch(name,normalization,target); out=model(batch.initial_state,batch.rainfall,batch.reference_settings,batch.candidate_settings,prepared); idx=out.horizon_indices
            truth=batch.true_candidate_states.index_select(2,idx); flow=batch.true_candidate_flows.index_select(2,idx)
            d.append(float(torch.sqrt(torch.mean((out.candidate_states_physical[...,0]-truth[...,0])**2)))); f.append(float(torch.sqrt(torch.mean((out.candidate_states_physical[...,2]-truth[...,2])**2)))); q.append(float(torch.sqrt(torch.mean((out.candidate_flows_physical-flow)**2))))
            if storage_mask.any():
                mask=torch.as_tensor(storage_mask,device=target); s.append(float(torch.sqrt(torch.mean((out.candidate_states_physical[...,3][...,mask]-truth[...,3][...,mask])**2))))
            y=truth[...,2] > HydraulicLossContractV60().onset_epsilon_m3s; p=out.candidate_flood_onset_logits>0; tpr=(p&y).sum().float()/y.sum().clamp_min(1); tnr=((~p)&(~y)).sum().float()/(~y).sum().clamp_min(1); ba.append(float(0.5*(tpr+tnr)))
    return {"groups":len(names),"depth_rmse_m":float(np.mean(d)),"flooding_rmse_m3s":float(np.mean(f)),"storage_volume_rmse_m3":float(np.mean(s)) if s else float("nan"),"managed_flow_rmse_m3s":float(np.mean(q)),"flooding_onset_balanced_accuracy":float(np.mean(ba)),"event_group_balanced":True,"multi_resolution_points":len(model.horizon_contract.indices())}


def train_value_v60(model: ControlValueSurrogateV60, cache: V60TrainCache, *, fit_d2_names: Sequence[str], fit_d3_names: Sequence[str], normalization: InputNormalizationV60, scales: TargetScalesV60, graph: Any, device: str="cuda", d2_pretrain_epochs: int=4, joint_epochs: int=8, learning_rate: float=1e-3, seed: int=42) -> list[dict[str, float | int | str]]:
    if not fit_d3_names: raise ValueError("V6 joint training requires targeted D3_V60 groups; legacy dense D3 is diagnostic only")
    target=torch.device(device if device=="cuda" and torch.cuda.is_available() else "cpu"); torch.manual_seed(seed); np.random.seed(seed); model.to(target).float().train(); prepared=prepare_static_v60(graph,target); optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-5); history=[]
    def step(name):
        b=cache.batch(name,normalization,target); o=model(b.initial_state,b.rainfall,b.reference_settings,b.candidate_settings,prepared,b.elapsed_seconds); return value_loss_v60(o.delta_tfv_m3,b.true_delta_tfv_m3,scale_m3=scales.tfv_scale(b.source_kind))[0]
    rng=np.random.default_rng(seed)
    for epoch in range(1,d2_pretrain_epochs+1):
        order=list(fit_d2_names); rng.shuffle(order); losses=[]
        for name in order: optimizer.zero_grad(set_to_none=True); loss=step(name); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); losses.append(float(loss.detach()))
        history.append({"stage":"D2_sensitivity_pretrain","epoch":epoch,"loss":float(np.mean(losses))})
    d2_order=list(fit_d2_names)
    for epoch in range(1,joint_epochs+1):
        d3_order=list(fit_d3_names); rng.shuffle(d3_order); rng.shuffle(d2_order); losses=[]
        for i,d3_name in enumerate(d3_order): optimizer.zero_grad(set_to_none=True); loss=0.5*step(d3_name)+0.5*step(d2_order[i%len(d2_order)]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); losses.append(float(loss.detach()))
        history.append({"stage":"D3_joint_with_D2_anchor","epoch":epoch,"loss":float(np.mean(losses))})
    return history


def train_hydraulic_v60(model: HydraulicResponseSurrogateV60, cache: V60TrainCache, *, fit_d2_names: Sequence[str], fit_d3_names: Sequence[str], normalization: InputNormalizationV60, scales: TargetScalesV60, graph: Any, device: str="cuda", epochs: int=8, learning_rate: float=1e-3, seed: int=42) -> list[dict[str, float | int | str]]:
    if not fit_d3_names: raise ValueError("V6 hydraulic training requires targeted D3_V60 groups")
    target=torch.device(device if device=="cuda" and torch.cuda.is_available() else "cpu"); torch.manual_seed(seed+17); np.random.seed(seed+17); model.to(target).float().train(); prepared=prepare_static_v60(graph,target); optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-5); rng=np.random.default_rng(seed+17); history=[]
    for epoch in range(1,epochs+1):
        d3=list(fit_d3_names); d2=list(fit_d2_names); rng.shuffle(d3); rng.shuffle(d2); losses=[]
        for i,d3_name in enumerate(d3):
            optimizer.zero_grad(set_to_none=True); total=torch.zeros((),device=target)
            for name,w in ((d3_name,0.5),(d2[i%len(d2)],0.5)):
                b=cache.batch(name,normalization,target); o=model(b.initial_state,b.rainfall,b.reference_settings,b.candidate_settings,prepared); total=total+w*hydraulic_loss_v60(o,b,prepared,scales,horizon=model.horizon_contract)[0]
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); losses.append(float(total.detach()))
        history.append({"stage":"hydraulic_source_balanced","epoch":epoch,"loss":float(np.mean(losses))})
    return history


__all__ = ["InputNormalizationV60","TargetScalesV60","V60GroupBatch","V60TrainCache","derive_input_normalization_v60","derive_target_scales_v60","deterministic_rainfall_split_v60","evaluate_hydraulic_v60","evaluate_value_v60","hydraulic_critical_weights_v60","hydraulic_loss_v60","listwise_loss_v60","train_hydraulic_v60","train_value_v60","value_loss_v60"]
