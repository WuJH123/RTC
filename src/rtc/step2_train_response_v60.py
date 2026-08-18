"""V6 training/data contracts with group/event balancing and critical-hydraulic weighting."""
from __future__ import annotations

import hashlib
from collections import defaultdict
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


def derive_magnitude_strata_v60(
    cache: "V60TrainCache", fit_d3_names: Sequence[str]
) -> dict[str, Any]:
    """Freeze D3 magnitude thresholds from TrainFit only.

    The thresholds are deliberately derived from authoritative candidate TFV values
    in the targeted D3 TrainFit groups.  Holdout values never enter this helper.
    """
    if not fit_d3_names:
        raise ValueError("magnitude strata require at least one TrainFit D3 group")
    values: list[float] = []
    for name in fit_d3_names:
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        candidates = [i for i in entry.indices if i != ref]
        reference = float(
            np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum()
        )
        candidate = np.asarray(
            arrays["exact_node_flood_volume_m3"][candidates], dtype=np.float64
        ).sum(axis=1)
        values.extend(np.abs(candidate - reference).tolist())
    absolute = np.asarray(values, dtype=np.float64)
    if absolute.size == 0 or not np.isfinite(absolute).all():
        raise ValueError("TrainFit D3 magnitude values are empty or non-finite")
    q33, q67 = (float(np.quantile(absolute, q)) for q in (1.0 / 3.0, 2.0 / 3.0))
    if q67 < q33:
        raise ValueError("magnitude thresholds are not ordered")
    return {
        "contract": "PROJECT7_V60_D3_MAGNITUDE_STRATA_TRAIN_FIT_V1",
        "q33_m3": q33,
        "q67_m3": q67,
        "source": "TrainFit targeted D3 authoritative exact delta TFV",
        "source_group_count": int(len(fit_d3_names)),
        "source_candidate_count": int(absolute.size),
    }


def magnitude_strata_partition_v60(
    values_m3: np.ndarray | Sequence[float],
    strata: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return the mutually-exclusive/exhaustive small/medium/large masks."""
    values = np.asarray(values_m3, dtype=np.float64)
    absolute = np.abs(values)
    q33, q67 = float(strata["q33_m3"]), float(strata["q67_m3"])
    masks = {
        "small": absolute < q33,
        "medium": (absolute >= q33) & (absolute < q67),
        "large": absolute >= q67,
    }
    if not np.array_equal(
        masks["small"] | masks["medium"] | masks["large"],
        np.ones_like(absolute, dtype=bool),
    ):
        raise RuntimeError("V6 magnitude strata are not collectively exhaustive")
    if np.any(masks["small"] & masks["medium"]) or np.any(
        masks["medium"] & masks["large"]
    ):
        raise RuntimeError("V6 magnitude strata overlap")
    return masks


def event_balanced_mean_v60(
    records: Sequence[dict[str, Any]], key: str
) -> float:
    """Mean a group metric within event, then mean equally across events."""
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record.get(key, float("nan")))
        if np.isfinite(value):
            by_event[str(record["event_key"])].append(value)
    event_means = [float(np.mean(values)) for values in by_event.values() if values]
    return float(np.mean(event_means)) if event_means else float("nan")


def response_collapse_v60(
    predicted_spread_m3: float,
    truth_spread_m3: float,
    *,
    threshold: float = 1e-3,
) -> bool:
    """Diagnostic gate for the historical near-zero response-spread failure."""
    truth = max(abs(float(truth_spread_m3)), 1e-12)
    return abs(float(predicted_spread_m3)) / truth < float(threshold)


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
        return bool(roles) and roles <= {
            "D3_V60_MANIFOLD_CANDIDATE",
            "D3_V60_ACTIVE_LEARNING_CANDIDATE",
            "D3_V6_POLICY_CALIBRATION_CANDIDATE",
        }

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


def _value_group_record(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    event_key: str,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = np.ones(truth.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    pred, actual = np.asarray(predicted, dtype=np.float64)[selected], np.asarray(truth, dtype=np.float64)[selected]
    if pred.size == 0:
        return {"event_key": event_key, "count": 0}
    orderable = pred.size >= 2
    truth_spread = float(actual.max() - actual.min()) if orderable else 0.0
    predicted_spread = float(pred.max() - pred.min()) if orderable else 0.0
    nonzero = np.abs(actual) > 1e-9
    sign = float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero]))) if nonzero.any() else float("nan")
    if orderable:
        predicted_best, truth_best = int(np.argmin(pred)), int(np.argmin(actual))
        regret = float(actual[predicted_best] - actual[truth_best])
        top1 = float(predicted_best == truth_best)
    else:
        regret, top1 = float("nan"), float("nan")
    return {
        "event_key": event_key,
        "count": int(pred.size),
        "rank": _spearman(pred, actual) if orderable else float("nan"),
        "pairwise": _pairwise_accuracy(pred, actual) if orderable else float("nan"),
        "sign_accuracy": sign,
        "top1_rate": top1,
        "mean_regret_m3": regret,
        "tfv_mae_m3": float(np.mean(np.abs(pred - actual))),
        "tfv_bias_m3": float(np.mean(pred - actual)),
        "truth_spread_m3": truth_spread,
        "predicted_spread_m3": predicted_spread,
        "mean_abs_truth_m3": float(np.mean(np.abs(actual))),
        "mean_abs_prediction_m3": float(np.mean(np.abs(pred))),
    }


_VALUE_METRIC_KEYS = (
    "rank", "pairwise", "sign_accuracy", "top1_rate", "mean_regret_m3",
    "tfv_mae_m3", "tfv_bias_m3", "truth_spread_m3", "predicted_spread_m3",
    "mean_abs_truth_m3", "mean_abs_prediction_m3",
)


def _aggregate_value_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str, subset: Sequence[dict[str, Any]]) -> float:
        values = [float(r[key]) for r in subset if key in r and np.isfinite(float(r[key]))]
        return float(np.mean(values)) if values else float("nan")

    group_balanced = {key: mean(key, records) for key in _VALUE_METRIC_KEYS}
    event_keys = sorted({str(r["event_key"]) for r in records})
    event_balanced = {
        key: event_balanced_mean_v60(records, key) for key in _VALUE_METRIC_KEYS
    }
    for target in (group_balanced, event_balanced):
        target["response_ratio"] = target["mean_abs_prediction_m3"] / max(
            target["mean_abs_truth_m3"], 1e-6
        )
        target["spread_ratio"] = target["predicted_spread_m3"] / max(
            target["truth_spread_m3"], 1e-6
        )
    max_regret = max(
        (float(r["mean_regret_m3"]) for r in records if np.isfinite(float(r.get("mean_regret_m3", np.nan)))),
        default=float("nan"),
    )
    event_balanced["max_regret_m3"] = max_regret
    group_balanced["max_regret_m3"] = max_regret
    top1_count = sum(
        int(float(r.get("top1_rate", float("nan"))) == 1.0) for r in records
    )
    result = {
        "groups": int(len(records)),
        "events": int(len(event_keys)),
        "group_balanced": group_balanced,
        "event_balanced": event_balanced,
        "scientific_primary": "event_balanced",
        # Backward-compatible aliases now point to the scientific primary.
        "rank": event_balanced["rank"],
        "pairwise": event_balanced["pairwise"],
        "sign_accuracy": event_balanced["sign_accuracy"],
        "top1": event_balanced["top1_rate"],
        "top1_count": int(top1_count),
        "top1_denominator": int(len(records)),
        "mean_regret_m3": event_balanced["mean_regret_m3"],
        "max_regret_m3": max_regret,
        "tfv_mae_m3": event_balanced["tfv_mae_m3"],
        "truth_spread_m3": event_balanced["truth_spread_m3"],
        "predicted_spread_m3": event_balanced["predicted_spread_m3"],
        "spread_ratio": event_balanced["spread_ratio"],
        "response_ratio": event_balanced["response_ratio"],
        "mean_abs_truth_m3": event_balanced["mean_abs_truth_m3"],
        "mean_abs_prediction_m3": event_balanced["mean_abs_prediction_m3"],
        "response_collapse": response_collapse_v60(
            event_balanced["predicted_spread_m3"], event_balanced["truth_spread_m3"]
        ),
        "response_collapse_threshold": 1e-3,
    }
    return result


def evaluate_value_v60(
    model: ControlValueSurrogateV60,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV60,
    *,
    device: torch.device | str,
    magnitude_strata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, device)
            output = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                prepared,
                batch.elapsed_seconds,
            )
            pred = output.delta_tfv_m3[0].detach().cpu().numpy()
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
            entry = cache.entry(name)
            records.append(
                _value_group_record(
                    pred,
                    truth,
                    event_key=f"{entry.rainfall_group}::{entry.event_id}",
                )
            )
    result = _aggregate_value_records(records)
    if magnitude_strata is not None:
        strata_metrics: dict[str, Any] = {}
        for stratum, masks in (
            ("small", []),
            ("medium", []),
            ("large", []),
        ):
            del masks
            strata_records: list[dict[str, Any]] = []
            with torch.no_grad():
                for name in names:
                    batch = cache.batch(name, normalization, device)
                    output = model(
                        batch.initial_state,
                        batch.rainfall,
                        batch.reference_settings,
                        batch.candidate_settings,
                        prepared,
                        batch.elapsed_seconds,
                    )
                    pred = output.delta_tfv_m3[0].detach().cpu().numpy()
                    truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
                    mask = magnitude_strata_partition_v60(truth, magnitude_strata)[stratum]
                    entry = cache.entry(name)
                    strata_records.append(
                        _value_group_record(
                            pred,
                            truth,
                            event_key=f"{entry.rainfall_group}::{entry.event_id}",
                            mask=mask,
                        )
                    )
            strata_result = _aggregate_value_records(
                [record for record in strata_records if record.get("count", 0) > 0]
            )
            strata_result["count"] = int(sum(record.get("count", 0) for record in strata_records))
            strata_metrics[stratum] = strata_result
        result["magnitude_strata"] = {
            "thresholds": dict(magnitude_strata),
            "small": strata_metrics["small"],
            "medium": strata_metrics["medium"],
            "large": strata_metrics["large"],
            "large_effect_response_ratio": strata_metrics["large"].get("response_ratio"),
        }
    return result


def _hydraulic_scalar_summary(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> dict[str, Any]:
    group_balanced = {
        key: float(np.mean([r[key] for r in records if np.isfinite(r.get(key, np.nan))]))
        if any(np.isfinite(r.get(key, np.nan)) for r in records) else float("nan")
        for key in keys
    }
    event_balanced = {key: event_balanced_mean_v60(records, key) for key in keys}
    return {
        "group_balanced": group_balanced,
        "event_balanced": event_balanced,
        "scientific_primary": "event_balanced",
        **event_balanced,
    }


def _hydraulic_region_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    onset_epsilon: float,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    count = int(selected.sum())
    result: dict[str, Any] = {"sample_count": count}
    if count == 0:
        return result
    p, t = predicted[selected], truth[selected]
    result["depth_rmse_m"] = float(np.sqrt(np.mean(np.square(p[..., 0] - t[..., 0]))))
    result["flooding_rmse_m3s"] = float(np.sqrt(np.mean(np.square(p[..., 2] - t[..., 2]))))
    result["storage_volume_rmse_m3"] = float(np.sqrt(np.mean(np.square(p[..., 3] - t[..., 3]))))
    y = t[..., 2] > onset_epsilon
    pred_onset = logits[selected] > 0.0
    tp, fn = np.sum(pred_onset & y), np.sum((~pred_onset) & y)
    fp, tn = np.sum(pred_onset & (~y)), np.sum((~pred_onset) & (~y))
    result["onset_recall"] = float(tp / max(tp + fn, 1))
    result["onset_precision"] = float(tp / max(tp + fp, 1))
    result["onset_balanced_accuracy"] = float(
        0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    )
    return result


def evaluate_hydraulic_v60(
    model: HydraulicResponseSurrogateV60,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    target = torch.device(device)
    model.to(target).eval()
    prepared = prepare_static_v60(graph, target)
    storage_mask = prepared.storage_mask.detach().cpu().numpy().astype(bool)
    max_depth = prepared.max_depth_m.detach().cpu().numpy()
    surcharge = (prepared.max_depth_m + prepared.surcharge_depth_m).detach().cpu().numpy()
    capacity = prepared.storage_capacity_m3.detach().cpu().numpy()
    onset_epsilon = HydraulicLossContractV60().onset_epsilon_m3s
    records: list[dict[str, Any]] = []
    region_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target)
            out = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                prepared,
            )
            idx = out.horizon_indices
            predicted = out.candidate_states_physical.detach().cpu().numpy()
            truth = batch.true_candidate_states.index_select(2, idx).detach().cpu().numpy()
            flow_pred = out.candidate_flows_physical.detach().cpu().numpy()
            flow_truth = batch.true_candidate_flows.index_select(2, idx).detach().cpu().numpy()
            logits = out.candidate_flood_onset_logits.detach().cpu().numpy()
            depth_error = predicted[..., 0] - truth[..., 0]
            flood_error = predicted[..., 2] - truth[..., 2]
            storage_error = predicted[..., 3] - truth[..., 3]
            flow_error = flow_pred - flow_truth
            depth_rmse = float(np.sqrt(np.mean(np.square(depth_error))))
            flood_rmse = float(np.sqrt(np.mean(np.square(flood_error))))
            flow_rmse = float(np.sqrt(np.mean(np.square(flow_error))))
            if storage_mask.any():
                storage_diff = storage_error[..., storage_mask]
                storage_rmse = float(np.sqrt(np.mean(np.square(storage_diff))))
            else:
                storage_rmse = float("nan")
            baseline = truth[..., 0] - np.mean(truth[..., 0])
            depth_nse = float(1.0 - np.sum(np.square(depth_error)) / max(np.sum(np.square(baseline)), 1e-12))
            y = truth[..., 2] > onset_epsilon
            p = logits > 0.0
            tp, fn = np.sum(p & y), np.sum((~p) & y)
            fp, tn = np.sum(p & (~y)), np.sum((~p) & (~y))
            record = {
                "event_key": f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}",
                "depth_rmse_m": depth_rmse,
                "depth_nse": depth_nse,
                "flooding_rmse_m3s": flood_rmse,
                "storage_volume_rmse_m3": storage_rmse,
                "managed_flow_rmse_m3s": flow_rmse,
                "flooding_onset_balanced_accuracy": float(
                    0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
                ),
            }
            records.append(record)
            ratios = truth[..., 0] / np.maximum(max_depth[None, None, None, :], 1e-6)
            surcharge_ratios = truth[..., 0] / np.maximum(surcharge[None, None, None, :], 1e-6)
            storage_ratios = truth[..., 3] / np.maximum(capacity[None, None, None, :], 1e-6)
            masks = {
                "wet": ratios >= 0.5,
                "near_surcharge": surcharge_ratios >= 0.8,
                "storage_near_capacity": storage_ratios >= 0.8,
                "flooding_onset": truth[..., 2] > onset_epsilon,
            }
            for key, mask in masks.items():
                # Storage proximity is defined only on storage nodes.
                if key == "storage_near_capacity":
                    mask = mask & storage_mask[None, None, None, :]
                region = _hydraulic_region_metrics(
                    predicted,
                    truth,
                    logits,
                    mask,
                    onset_epsilon=onset_epsilon,
                )
                region["event_key"] = record["event_key"]
                region_records[key].append(region)
    keys = (
        "depth_rmse_m", "depth_nse", "flooding_rmse_m3s", "storage_volume_rmse_m3",
        "managed_flow_rmse_m3s", "flooding_onset_balanced_accuracy",
    )
    summary = _hydraulic_scalar_summary(records, keys)
    summary.update(
        {
            "groups": int(len(names)),
            "events": int(len({r["event_key"] for r in records})),
            "multi_resolution_points": len(model.horizon_contract.indices()),
            "critical_strata": {},
        }
    )
    region_keys = (
        "depth_rmse_m", "flooding_rmse_m3s", "storage_volume_rmse_m3",
        "onset_recall", "onset_precision", "onset_balanced_accuracy",
    )
    for name, region in region_records.items():
        valid = [r for r in region if int(r.get("sample_count", 0)) > 0]
        summary["critical_strata"][name] = {
            "sample_count": int(sum(int(r.get("sample_count", 0)) for r in region)),
            "group_balanced": {
                key: float(np.mean([r[key] for r in valid if key in r and np.isfinite(r[key])]))
                if any(key in r and np.isfinite(r[key]) for r in valid) else float("nan")
                for key in region_keys
            },
            "event_balanced": {
                key: event_balanced_mean_v60(valid, key)
                for key in region_keys
            },
        }
        summary["critical_strata"][name]["scientific_primary"] = "event_balanced"
        summary["critical_strata"][name].update(summary["critical_strata"][name]["event_balanced"])
    return summary


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


__all__ = [
    "InputNormalizationV60", "TargetScalesV60", "V60GroupBatch", "V60TrainCache",
    "derive_input_normalization_v60", "derive_target_scales_v60",
    "derive_magnitude_strata_v60", "magnitude_strata_partition_v60",
    "event_balanced_mean_v60", "response_collapse_v60",
    "deterministic_rainfall_split_v60", "evaluate_hydraulic_v60", "evaluate_value_v60",
    "hydraulic_critical_weights_v60", "hydraulic_loss_v60", "listwise_loss_v60",
    "train_hydraulic_v60", "train_value_v60", "value_loss_v60",
]
