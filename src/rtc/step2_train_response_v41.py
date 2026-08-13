"""Train-only grouped calibration utilities for the isolated Step2 V4.1 model."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v41 import (
    CounterfactualGroupResponseV41,
    DifferentiableCounterfactualResponseModelV41,
    PreparedStaticV41,
)
from .step2_counterfactual import counterfactual_groups, reference_index
from .step2_train_response_v4 import ResponseNormalizationV4, ResponsePairV4

SCALE_CONTRACT_V41 = "STEP2_COUNTERFACTUAL_DELTA_SCALES_V41_TRAIN_ONLY"
TRAINING_CONTRACT_V41 = "STEP2_RESPONSE_CALIBRATION_V41_TRAIN_ONLY_DIAGNOSTIC"


class _GpuUtilizationSampler:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.utilization: list[float] = []
        self.memory_mib: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                first = completed.stdout.strip().splitlines()[0]
                utilization, memory = (
                    float(value.strip()) for value in first.split(",")[:2]
                )
                self.utilization.append(utilization)
                self.memory_mib.append(memory)
            except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
                self.enabled = False
                return
            self._stop.wait(0.25)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self.utilization:
            return {"available": False, "samples": 0}
        values = np.asarray(self.utilization, dtype=np.float64)
        memory = np.asarray(self.memory_mib, dtype=np.float64)
        return {
            "available": True,
            "samples": int(values.size),
            "mean_percent": float(values.mean()),
            "p50_percent": float(np.quantile(values, 0.5)),
            "p90_percent": float(np.quantile(values, 0.9)),
            "max_percent": float(values.max()),
            "mean_memory_mib": float(memory.mean()),
            "max_memory_mib": float(memory.max()),
        }


@dataclass(frozen=True)
class ScaleDeltaBlockV41:
    source_kind: str
    delta_states: np.ndarray
    delta_flows: np.ndarray
    delta_tfv_m3: np.ndarray


@dataclass(frozen=True)
class SourceCounterfactualScalesV41:
    state_scale: np.ndarray
    flow_scale: np.ndarray
    tfv_scale_m3: float
    tfv_abs_quantiles_m3: dict[str, float]
    state_statistics: list[dict[str, float]]
    flow_statistics: list[dict[str, float]]
    tfv_statistics: dict[str, float]
    candidate_count: int
    group_count: int


@dataclass(frozen=True)
class CounterfactualDeltaScalesV41:
    source_manifest_sha256: str
    by_source: dict[str, SourceCounterfactualScalesV41]
    contract: str = SCALE_CONTRACT_V41

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "source_manifest_sha256": self.source_manifest_sha256,
            "derivation": {
                "split": "development",
                "fold": "train",
                "scale": "physical_counterfactual_delta_rms_with_robust_floor",
                "robust_floor": "max(1e-6, 0.01 * median_abs_nonzero)",
                "validation_used": False,
            },
            "by_source": {
                source: {
                    "state_scale": values.state_scale.tolist(),
                    "flow_scale": values.flow_scale.tolist(),
                    "tfv_scale_m3": values.tfv_scale_m3,
                    "tfv_abs_quantiles_m3": values.tfv_abs_quantiles_m3,
                    "state_statistics": values.state_statistics,
                    "flow_statistics": values.flow_statistics,
                    "tfv_statistics": values.tfv_statistics,
                    "candidate_count": values.candidate_count,
                    "group_count": values.group_count,
                }
                for source, values in sorted(self.by_source.items())
            },
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CounterfactualDeltaScalesV41:
        if payload.get("contract") != SCALE_CONTRACT_V41:
            raise ValueError("not a V4.1 Train-only counterfactual scale artifact")
        by_source = {
            source: SourceCounterfactualScalesV41(
                state_scale=np.asarray(values["state_scale"], dtype=np.float32),
                flow_scale=np.asarray(values["flow_scale"], dtype=np.float32),
                tfv_scale_m3=float(values["tfv_scale_m3"]),
                tfv_abs_quantiles_m3={
                    str(key): float(value)
                    for key, value in values["tfv_abs_quantiles_m3"].items()
                },
                state_statistics=list(values["state_statistics"]),
                flow_statistics=list(values["flow_statistics"]),
                tfv_statistics=dict(values["tfv_statistics"]),
                candidate_count=int(values["candidate_count"]),
                group_count=int(values["group_count"]),
            )
            for source, values in payload["by_source"].items()
        }
        return cls(source_manifest_sha256=str(payload["source_manifest_sha256"]), by_source=by_source)


class _RunningVectorStatistics:
    def __init__(self, channels: int, *, sample_limit: int = 250_000, seed: int = 0) -> None:
        self.channels = int(channels)
        self.sample_limit = int(sample_limit)
        self.sum_squares = np.zeros(self.channels, dtype=np.float64)
        self.count = np.zeros(self.channels, dtype=np.int64)
        self.zero_count = np.zeros(self.channels, dtype=np.int64)
        self.maximum = np.zeros(self.channels, dtype=np.float64)
        self.samples = [np.empty(0, dtype=np.float64) for _ in range(self.channels)]
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.channels)
        finite = np.isfinite(array)
        safe = np.where(finite, array, 0.0)
        absolute = np.abs(safe)
        self.sum_squares += np.square(safe).sum(axis=0)
        self.count += finite.sum(axis=0)
        self.zero_count += (finite & (absolute <= 1e-12)).sum(axis=0)
        self.maximum = np.maximum(self.maximum, absolute.max(axis=0, initial=0.0))
        for channel in range(self.channels):
            candidate = absolute[finite[:, channel], channel]
            if candidate.size > self.sample_limit:
                indices = self.rng.choice(candidate.size, self.sample_limit, replace=False)
                candidate = candidate[indices]
            merged = np.concatenate((self.samples[channel], candidate))
            if merged.size > self.sample_limit:
                indices = self.rng.choice(merged.size, self.sample_limit, replace=False)
                merged = merged[indices]
            self.samples[channel] = merged

    def finalize(self) -> tuple[np.ndarray, list[dict[str, float]]]:
        scales: list[float] = []
        rows: list[dict[str, float]] = []
        for channel in range(self.channels):
            count = max(1, int(self.count[channel]))
            rms = float(np.sqrt(self.sum_squares[channel] / count))
            sample = self.samples[channel]
            if sample.size:
                median = float(np.median(sample))
                q25, q75, p90, p95, p99 = np.quantile(
                    sample, [0.25, 0.75, 0.90, 0.95, 0.99]
                ).tolist()
                nonzero = sample[sample > 1e-12]
                median_nonzero = float(np.median(nonzero)) if nonzero.size else 0.0
            else:
                median = q25 = q75 = p90 = p95 = p99 = median_nonzero = 0.0
            robust_floor = max(1e-6, 0.01 * median_nonzero)
            scale = max(rms, robust_floor)
            scales.append(scale)
            rows.append(
                {
                    "rms": rms,
                    "median_abs": median,
                    "iqr_abs": float(q75 - q25),
                    "p90_abs": float(p90),
                    "p95_abs": float(p95),
                    "p99_abs": float(p99),
                    "max_abs": float(self.maximum[channel]),
                    "zero_fraction": float(self.zero_count[channel] / count),
                    "robust_floor": robust_floor,
                    "selected_scale": scale,
                    "finite_count": float(self.count[channel]),
                }
            )
        return np.asarray(scales, dtype=np.float32), rows


class _SourceScaleAccumulator:
    def __init__(self, state_channels: int, actuator_count: int, *, seed: int) -> None:
        self.state = _RunningVectorStatistics(state_channels, seed=seed)
        self.flow = _RunningVectorStatistics(actuator_count, seed=seed + 1)
        self.tfv = _RunningVectorStatistics(1, seed=seed + 2)
        self.candidate_count = 0
        self.group_count = 0

    def update(self, block: ScaleDeltaBlockV41) -> None:
        self.state.update(block.delta_states)
        self.flow.update(block.delta_flows)
        self.tfv.update(np.asarray(block.delta_tfv_m3).reshape(-1, 1))
        self.candidate_count += int(np.asarray(block.delta_tfv_m3).size)
        self.group_count += 1

    def finalize(self) -> SourceCounterfactualScalesV41:
        state_scale, state_statistics = self.state.finalize()
        flow_scale, flow_statistics = self.flow.finalize()
        tfv_scale, tfv_statistics = self.tfv.finalize()
        tfv_sample = self.tfv.samples[0]
        quantiles = (
            np.quantile(tfv_sample, [1 / 3, 2 / 3]).tolist()
            if tfv_sample.size
            else [0.0, 0.0]
        )
        return SourceCounterfactualScalesV41(
            state_scale=state_scale,
            flow_scale=flow_scale,
            tfv_scale_m3=float(tfv_scale[0]),
            tfv_abs_quantiles_m3={"q33": float(quantiles[0]), "q67": float(quantiles[1])},
            state_statistics=state_statistics,
            flow_statistics=flow_statistics,
            tfv_statistics=tfv_statistics[0],
            candidate_count=self.candidate_count,
            group_count=self.group_count,
        )


def derive_counterfactual_delta_scales_v41(
    blocks: Iterable[ScaleDeltaBlockV41], *, source_manifest_sha256: str
) -> CounterfactualDeltaScalesV41:
    """Derive source-separated physical delta scales without absolute-state statistics."""

    accumulators: dict[str, _SourceScaleAccumulator] = {}
    for block in blocks:
        source = block.source_kind.upper()
        state_channels = int(np.asarray(block.delta_states).shape[-1])
        actuator_count = int(np.asarray(block.delta_flows).shape[-1])
        accumulator = accumulators.setdefault(
            source,
            _SourceScaleAccumulator(
                state_channels, actuator_count, seed=20260811 + len(accumulators) * 10
            ),
        )
        if accumulator.state.channels != state_channels or accumulator.flow.channels != actuator_count:
            raise ValueError("counterfactual delta block dimensions changed during scale derivation")
        accumulator.update(block)
    if set(accumulators) != {"D2", "D3"}:
        raise ValueError("V4.1 requires both D2 and D3 Train-only scale blocks")
    return CounterfactualDeltaScalesV41(
        source_manifest_sha256=source_manifest_sha256,
        by_source={source: accumulator.finalize() for source, accumulator in accumulators.items()},
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_scale_blocks(manifest: dict[str, Any]) -> Iterable[ScaleDeltaBlockV41]:
    for shard in manifest["shards"]:
        with np.load(shard["path"], allow_pickle=False) as arrays:
            split = {str(value).lower() for value in arrays["scientific_split"].tolist()}
            fold = {str(value).lower() for value in arrays["development_fold"].tolist()}
            if split != {"development"} or fold != {"train"}:
                raise RuntimeError(
                    f"scale shard is not development/train only: {shard['path']} {split=} {fold=}"
                )
            for group, indices in counterfactual_groups(arrays).items():
                reference = reference_index(arrays, indices)
                candidates = [index for index in indices if index != reference]
                if not candidates:
                    continue
                source = group.split("::", 1)[0].upper()
                delta_state = arrays["target_states"][candidates] - arrays["target_states"][reference]
                delta_flow = (
                    arrays["target_actuator_flows"][candidates]
                    - arrays["target_actuator_flows"][reference]
                )
                candidate_volume = arrays["exact_node_flood_volume_m3"][candidates].sum(axis=1)
                reference_volume = float(arrays["exact_node_flood_volume_m3"][reference].sum())
                yield ScaleDeltaBlockV41(
                    source_kind=source,
                    delta_states=delta_state,
                    delta_flows=delta_flow,
                    delta_tfv_m3=candidate_volume - reference_volume,
                )


def load_or_derive_train_only_scales_v41(
    manifest_path: str | Path, cache_path: str | Path
) -> tuple[CounterfactualDeltaScalesV41, bool]:
    """Load a lineage-matched cache or scan the full development/train manifest once."""

    manifest_file = Path(manifest_path).resolve()
    cache_file = Path(cache_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("contract") != "STEP2_SHARDED_DATASET_V6_COUNTERFACTUAL_GROUP_PRESERVING":
        raise ValueError("V4.1 scales require the authoritative group-preserving V6 manifest")
    manifest_sha = _sha256_file(manifest_file)
    if cache_file.is_file():
        cached = CounterfactualDeltaScalesV41.from_json_dict(
            json.loads(cache_file.read_text(encoding="utf-8"))
        )
        if cached.source_manifest_sha256 == manifest_sha:
            return cached, True
    scales = derive_counterfactual_delta_scales_v41(
        _manifest_scale_blocks(manifest), source_manifest_sha256=manifest_sha
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(scales.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return scales, False


@dataclass(frozen=True)
class ResponseGroupBatchV41:
    source_kind: str
    group: str
    initial_state: torch.Tensor
    rainfall: torch.Tensor
    reference_settings: torch.Tensor
    candidate_settings: torch.Tensor
    previous_actuator_flow: torch.Tensor
    elapsed_seconds: torch.Tensor
    true_reference_states_physical: torch.Tensor
    true_candidate_states_physical: torch.Tensor
    true_delta_states_physical: torch.Tensor
    true_reference_flows_physical: torch.Tensor
    true_candidate_flows_physical: torch.Tensor
    true_delta_flows_physical: torch.Tensor
    true_delta_tfv_m3: torch.Tensor


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    writable = np.array(array, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(writable).to(device)


def stack_response_group_v41(
    pairs: list[ResponsePairV4], device: torch.device
) -> ResponseGroupBatchV41:
    """Build one group batch with one reference and every candidate."""

    if not pairs:
        raise ValueError("counterfactual group has no candidates")
    source = pairs[0].source_kind.upper()
    group = pairs[0].group
    reference = pairs[0].reference
    for pair in pairs:
        if pair.group != group or pair.source_kind.upper() != source:
            raise ValueError("mixed counterfactual groups cannot share a group batch")
        for name in ("initial_state", "rainfall", "settings", "previous_actuator_flow"):
            if not np.array_equal(pair.reference[name], reference[name]):
                raise ValueError(f"same-prefix reference differs within group for {name}")

    def candidates(name: str) -> np.ndarray:
        return np.stack([pair.candidate[name] for pair in pairs], axis=0)

    reference_state = np.asarray(reference["target_states_physical"], dtype=np.float32)
    candidate_state = candidates("target_states_physical")
    reference_flow = np.asarray(reference["target_actuator_flows_physical"], dtype=np.float32)
    candidate_flow = candidates("target_actuator_flows_physical")
    reference_tfv = float(np.asarray(reference["exact_node_flood_volume_m3"]).sum())
    candidate_tfv = np.asarray(
        [np.asarray(pair.candidate["exact_node_flood_volume_m3"]).sum() for pair in pairs],
        dtype=np.float32,
    )
    return ResponseGroupBatchV41(
        source_kind=source,
        group=group,
        initial_state=_tensor(np.asarray(reference["initial_state"])[None], device),
        rainfall=_tensor(np.asarray(reference["rainfall"])[None], device),
        reference_settings=_tensor(np.asarray(reference["settings"])[None], device),
        candidate_settings=_tensor(candidates("settings")[None], device),
        previous_actuator_flow=_tensor(
            np.asarray(reference["previous_actuator_flow"])[None], device
        ),
        elapsed_seconds=_tensor(np.asarray(reference["elapsed_seconds"])[None], device),
        true_reference_states_physical=_tensor(reference_state[None], device),
        true_candidate_states_physical=_tensor(candidate_state[None], device),
        true_delta_states_physical=_tensor((candidate_state - reference_state)[None], device),
        true_reference_flows_physical=_tensor(reference_flow[None], device),
        true_candidate_flows_physical=_tensor(candidate_flow[None], device),
        true_delta_flows_physical=_tensor((candidate_flow - reference_flow)[None], device),
        true_delta_tfv_m3=_tensor((candidate_tfv - reference_tfv)[None], device),
    )


def weighted_pairwise_ranking_loss(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    *,
    group_scale: torch.Tensor,
    minimum_normalized_gap: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Weighted all-pairs group ranking; negligible true gaps contribute nothing."""

    if predicted.shape != truth.shape or predicted.dim() != 2:
        raise ValueError("predicted and truth ranking tensors must both be [B,C]")
    candidates = predicted.shape[1]
    left, right = torch.triu_indices(candidates, candidates, offset=1, device=predicted.device)
    if left.numel() == 0:
        return predicted.sum() * 0.0, {"pair_count": 0, "meaningful_pair_count": 0}
    scale = torch.as_tensor(group_scale, device=predicted.device, dtype=predicted.dtype)
    scale = scale.reshape(-1, 1).clamp_min(1e-6)
    true_gap = truth[:, left] - truth[:, right]
    predicted_gap = predicted[:, left] - predicted[:, right]
    normalized_gap = true_gap.abs() / scale
    meaningful = normalized_gap >= float(minimum_normalized_gap)
    weights = normalized_gap.detach()
    losses = F.softplus(-true_gap.detach().sign() * predicted_gap / scale) * weights
    if bool(meaningful.any()):
        loss = losses[meaningful].sum() / weights[meaningful].sum().clamp_min(1e-12)
    else:
        loss = predicted.sum() * 0.0
    return loss, {
        "pair_count": int(true_gap.numel()),
        "meaningful_pair_count": int(meaningful.sum().item()),
    }


def _within_group_scale(truth: torch.Tensor, source_scale: torch.Tensor) -> torch.Tensor:
    spread = truth.amax(dim=1) - truth.amin(dim=1)
    floor = 0.05 * torch.as_tensor(source_scale, device=truth.device, dtype=truth.dtype)
    return torch.maximum(spread, floor).clamp_min(1e-6)


def balanced_magnitude_stratum_weights(
    authoritative_delta_tfv: torch.Tensor, *, q33: float, q67: float
) -> dict[str, torch.Tensor]:
    """Return per-group equal-stratum weights and one shared partition.

    ``authoritative_delta_tfv`` is ``[B, C]``.  Each present magnitude stratum
    receives the same total weight *within each group*, while each group's mean
    candidate weight remains one.  Missing strata contribute no weight and do
    not create a divide-by-zero path.
    """

    if authoritative_delta_tfv.ndim != 2:
        raise ValueError("authoritative_delta_tfv must be [B, C]")
    if not np.isfinite(float(q33)) or not np.isfinite(float(q67)) or not 0.0 <= q33 < q67:
        raise ValueError("magnitude thresholds must satisfy 0 <= q33 < q67")
    if authoritative_delta_tfv.shape[1] == 0:
        raise ValueError("magnitude weighting requires at least one candidate")
    absolute_truth = authoritative_delta_tfv.detach().abs()
    masks = {
        "small": absolute_truth < float(q33),
        "medium": (absolute_truth >= float(q33)) & (absolute_truth < float(q67)),
        "large": absolute_truth >= float(q67),
    }
    ordered_masks = (masks["small"], masks["medium"], masks["large"])
    mask_stack = torch.stack(ordered_masks, dim=1)  # [B, 3, C]
    counts = torch.stack(
        [mask.sum(dim=1) for mask in ordered_masks], dim=1
    )  # [B, 3]
    present = counts.gt(0)
    present_count = present.sum(dim=1, keepdim=True).to(authoritative_delta_tfv.dtype)
    raw_weights = torch.zeros_like(authoritative_delta_tfv)
    for index, mask in enumerate(ordered_masks):
        denominator = counts[:, index : index + 1].clamp_min(1).to(authoritative_delta_tfv.dtype)
        raw_weights = raw_weights + mask.to(authoritative_delta_tfv.dtype) / denominator
    candidate_count = float(authoritative_delta_tfv.shape[1])
    weights = raw_weights * (candidate_count / present_count.clamp_min(1.0))
    return {
        "weights": weights,
        "small_mask": masks["small"],
        "medium_mask": masks["medium"],
        "large_mask": masks["large"],
        "counts": counts,
        "small_count": counts[:, 0],
        "medium_count": counts[:, 1],
        "large_count": counts[:, 2],
        "partition_valid": mask_stack.sum(dim=1).eq(1).all(dim=1),
        "balanced_valid": torch.isfinite(weights).all(dim=1)
        & torch.isclose(weights.mean(dim=1), torch.ones_like(weights.mean(dim=1)), atol=1e-6, rtol=0.0),
    }


def tfv_loss_components_v41(
    predicted_direct: torch.Tensor,
    predicted_trajectory: torch.Tensor,
    authoritative_delta_tfv: torch.Tensor,
    *,
    source_scale: torch.Tensor,
    magnitude_calibration: bool = False,
    magnitude_q33: float | None = None,
    magnitude_q67: float | None = None,
) -> dict[str, torch.Tensor]:
    """Exact-TFV, within-group, ranking, and trajectory/direct consistency losses."""

    scale = torch.as_tensor(
        source_scale, device=predicted_direct.device, dtype=predicted_direct.dtype
    ).clamp_min(1.0)
    group_scale = _within_group_scale(authoritative_delta_tfv, scale)
    normalized_direct_error = (predicted_direct - authoritative_delta_tfv) / scale
    direct_centered = predicted_direct - predicted_direct.mean(dim=1, keepdim=True)
    truth_centered = authoritative_delta_tfv - authoritative_delta_tfv.mean(
        dim=1, keepdim=True
    )
    ranking, _ = weighted_pairwise_ranking_loss(
        predicted_direct, authoritative_delta_tfv, group_scale=group_scale
    )
    direct_error = F.smooth_l1_loss(
        normalized_direct_error, torch.zeros_like(normalized_direct_error), reduction="none"
    )
    trajectory_error = F.smooth_l1_loss(
        (predicted_trajectory - authoritative_delta_tfv) / scale,
        torch.zeros_like(predicted_trajectory),
        reduction="none",
    )
    if magnitude_calibration:
        if magnitude_q33 is None or magnitude_q67 is None:
            raise ValueError("D3 magnitude calibration requires fixed q33 and q67")
        absolute_truth = authoritative_delta_tfv.detach().abs()
        strata = balanced_magnitude_stratum_weights(
            authoritative_delta_tfv,
            q33=float(magnitude_q33),
            q67=float(magnitude_q67),
        )
        stratum_weights = strata["weights"].to(dtype=predicted_direct.dtype)
        direct_loss = (direct_error * stratum_weights).sum() / stratum_weights.sum().clamp_min(1e-6)
        trajectory_loss = (trajectory_error * stratum_weights).sum() / stratum_weights.sum().clamp_min(1e-6)
        normalized_true_magnitude = torch.log1p(absolute_truth / scale)
        normalized_predicted_magnitude = torch.log1p(predicted_direct.abs() / scale)
        log_magnitude = F.smooth_l1_loss(
            normalized_predicted_magnitude,
            normalized_true_magnitude,
            reduction="none",
        )
        log_magnitude = (log_magnitude * stratum_weights).sum() / stratum_weights.sum().clamp_min(1e-6)
        magnitude_weight_mean = stratum_weights.mean().detach()
    else:
        direct_loss = direct_error.mean()
        trajectory_loss = trajectory_error.mean()
        log_magnitude = predicted_direct.sum() * 0.0
        magnitude_weight_mean = predicted_direct.new_tensor(0.0)
    return {
        "absolute_direct": direct_loss,
        "group_centered_direct": F.smooth_l1_loss(
            (direct_centered - truth_centered) / group_scale[:, None],
            torch.zeros_like(direct_centered),
        ),
        "authoritative_trajectory": trajectory_loss,
        "direct_trajectory_consistency": F.smooth_l1_loss(
            (predicted_direct - predicted_trajectory) / scale,
            torch.zeros_like(predicted_direct),
        ),
        "ranking": ranking,
        "log_magnitude_calibration": log_magnitude,
        "magnitude_stratum_weight_mean": magnitude_weight_mean,
    }


def source_parameter_is_trainable(name: str, source_kind: str) -> bool:
    """Keep D2 single effects and D3 interactions from overwriting one another."""

    source = source_kind.upper()
    if source == "D2":
        return not name.startswith(("interaction_", "direct_interaction_tfv_head"))
    if source == "D3":
        return name.startswith(
            ("reference_", "interaction_", "direct_interaction_tfv_head")
        )
    raise ValueError("source_kind must be D2 or D3")


def clear_disallowed_source_gradients(
    model: DifferentiableCounterfactualResponseModelV41, source_kind: str
) -> None:
    for name, parameter in model.named_parameters():
        if not source_parameter_is_trainable(name, source_kind):
            parameter.grad = None


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def group_metrics_v41(
    *, predicted: np.ndarray, truth: np.ndarray, group: str, source_kind: str
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if predicted.shape != truth.shape or predicted.size < 2:
        raise ValueError("group metrics require matching candidate vectors of length >= 2")
    predicted_rank = _rank(predicted)
    truth_rank = _rank(truth)
    rank = (
        float(np.corrcoef(predicted_rank, truth_rank)[0, 1])
        if np.ptp(predicted_rank) > 0 and np.ptp(truth_rank) > 0
        else float("nan")
    )
    left, right = np.triu_indices(predicted.size, 1)
    true_gap = truth[left] - truth[right]
    pred_gap = predicted[left] - predicted[right]
    meaningful = np.abs(true_gap) > 1e-9
    pairwise = (
        float(np.mean(np.sign(true_gap[meaningful]) == np.sign(pred_gap[meaningful])))
        if meaningful.any()
        else float("nan")
    )
    meaningful_sign = np.abs(truth) >= 1.0
    sign = (
        float(np.mean(np.sign(predicted[meaningful_sign]) == np.sign(truth[meaningful_sign])))
        if meaningful_sign.any()
        else float("nan")
    )
    best_predicted = int(np.argmin(predicted))
    best_truth = int(np.argmin(truth))
    true_spread = float(np.ptp(truth))
    return {
        "group": group,
        "source_kind": source_kind.upper(),
        "candidate_count": int(predicted.size),
        "predicted_delta_tfv_spread_m3": float(np.ptp(predicted)),
        "true_delta_tfv_spread_m3": true_spread,
        "spread_ratio": float(np.ptp(predicted) / max(true_spread, 1e-12)),
        "mae_m3": float(np.mean(np.abs(predicted - truth))),
        "normalized_mae": float(
            np.mean(np.abs(predicted - truth)) / max(true_spread, 1e-12)
        ),
        "rank": rank,
        "pairwise": pairwise,
        "sign": sign,
        "top1": best_predicted == best_truth,
        "regret_m3": float(truth[best_predicted] - truth[best_truth]),
    }


def magnitude_strata_metrics_v41(
    contributions: list[dict[str, Any]], *, q33: float, q67: float
) -> dict[str, dict[str, float | int]]:
    """Summarise D3 response calibration separately for fixed Train-only strata."""

    result: dict[str, dict[str, float | int]] = {}
    d3_rows = [
        row for row in contributions if row.get("source_kind", "").upper() == "D3"
    ]
    if not d3_rows:
        return {
            stratum: {
                "count": 0,
                "mae_m3": float("nan"),
                "bias_m3": float("nan"),
                "response_ratio": float("nan"),
                "rank": float("nan"),
                "pairwise": float("nan"),
                "sign": float("nan"),
            }
            for stratum in ("small", "medium", "large")
        }
    truth_tensor = torch.tensor(
        [[float(row["true_delta_tfv_m3"]) for row in d3_rows]], dtype=torch.float64
    )
    partition = balanced_magnitude_stratum_weights(truth_tensor, q33=q33, q67=q67)
    partition_masks = {
        "small": partition["small_mask"][0].tolist(),
        "medium": partition["medium_mask"][0].tolist(),
        "large": partition["large_mask"][0].tolist(),
    }
    for stratum in ("small", "medium", "large"):
        rows = [
            row
            for row, selected in zip(d3_rows, partition_masks[stratum], strict=True)
            if selected
        ]
        truth = np.asarray([float(row["true_delta_tfv_m3"]) for row in rows], dtype=np.float64)
        predicted = np.asarray(
            [float(row["predicted_final_delta_tfv_m3"]) for row in rows], dtype=np.float64
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("group", "")), []).append(row)
        ranks = []
        pairs = []
        for group_rows in grouped.values():
            if len(group_rows) < 2:
                continue
            metrics = group_metrics_v41(
                predicted=np.asarray([float(row["predicted_final_delta_tfv_m3"]) for row in group_rows]),
                truth=np.asarray([float(row["true_delta_tfv_m3"]) for row in group_rows]),
                group="magnitude-stratum",
                source_kind="D3",
            )
            if np.isfinite(metrics["rank"]):
                ranks.append(float(metrics["rank"]))
            if np.isfinite(metrics["pairwise"]):
                pairs.append(float(metrics["pairwise"]))
        result[stratum] = {
            "count": int(truth.size),
            "mae_m3": float(np.mean(np.abs(predicted - truth))) if truth.size else float("nan"),
            "bias_m3": float(np.mean(predicted - truth)) if truth.size else float("nan"),
            "response_ratio": float(np.mean(np.abs(predicted)) / max(np.mean(np.abs(truth)), 1e-12)) if truth.size else float("nan"),
            "rank": float(np.mean(ranks)) if ranks else float("nan"),
            "pairwise": float(np.mean(pairs)) if pairs else float("nan"),
            "sign": float(np.mean(np.sign(predicted) == np.sign(truth))) if truth.size else float("nan"),
        }
    return result


def magnitude_weight_epoch_audit_v41(
    batches: dict[str, ResponseGroupBatchV41], *, q33: float, q67: float
) -> dict[str, Any]:
    """Record per-group weight totals so every magnitude-calibration epoch is auditable."""

    group_rows: list[dict[str, Any]] = []
    for group, batch in sorted(batches.items()):
        if batch.source_kind.upper() != "D3":
            continue
        partition = balanced_magnitude_stratum_weights(
            batch.true_delta_tfv_m3.detach(), q33=q33, q67=q67
        )
        for batch_index in range(batch.true_delta_tfv_m3.shape[0]):
            counts = {
                name: int(partition[f"{name}_count"][batch_index].item())
                for name in ("small", "medium", "large")
            }
            totals = {
                name: float(
                    (
                        partition["weights"][batch_index]
                        * partition[f"{name}_mask"][batch_index]
                    ).sum()
                )
                for name in ("small", "medium", "large")
            }
            present = [totals[name] for name in counts if counts[name] > 0]
            group_rows.append(
                {
                    "group": group if batch.true_delta_tfv_m3.shape[0] == 1 else f"{group}#{batch_index}",
                    "small_count": counts["small"],
                    "medium_count": counts["medium"],
                    "large_count": counts["large"],
                    "small_total_weight": totals["small"],
                    "medium_total_weight": totals["medium"],
                    "large_total_weight": totals["large"],
                    "partition_valid": bool(partition["partition_valid"][batch_index].item()),
                    "balanced_valid": bool(
                        partition["balanced_valid"][batch_index].item()
                    )
                    and max(present) - min(present) <= 1e-6,
                }
            )
    return {
        "d3_group_count": len(group_rows),
        "small_total_weight_mean": float(
            np.mean([row["small_total_weight"] for row in group_rows])
        )
        if group_rows
        else 0.0,
        "medium_total_weight_mean": float(
            np.mean([row["medium_total_weight"] for row in group_rows])
        )
        if group_rows
        else 0.0,
        "large_total_weight_mean": float(
            np.mean([row["large_total_weight"] for row in group_rows])
        )
        if group_rows
        else 0.0,
        "all_groups_partition_valid": all(row["partition_valid"] for row in group_rows),
        "all_groups_balanced_valid": all(row["balanced_valid"] for row in group_rows),
        "groups": group_rows,
    }


def calibration_selection_score_v41(
    metric: dict[str, Any],
    *,
    policy: str = "calibration",
    large_response_floor: float = 0.4113782853,
    d3_max_regret_reference_m3: float = 372115.0,
) -> float:
    """Return the deterministic checkpoint-selection score for one group summary."""

    spread = abs(np.log10(max(float(metric["spread_ratio"]), 1e-8)))
    rank = 1.0 - float(metric["rank"]) if np.isfinite(metric["rank"]) else 1.0
    pairwise = 1.0 - float(metric["pairwise"]) if np.isfinite(metric["pairwise"]) else 1.0
    sign = 1.0 - float(metric["sign"]) if np.isfinite(metric["sign"]) else 1.0
    top1 = 0.0 if bool(metric["top1"]) else 1.0
    normalized_mae = float(metric.get("normalized_mae", 0.0))
    if policy == "rank_first":
        return 4.0 * rank + 4.0 * pairwise + 2.0 * top1 + spread + 0.25 * sign + normalized_mae
    if policy == "d3_magnitude":
        strata = metric.get("d3_magnitude_strata", {})
        large = strata.get("large", {})
        large_ratio = float(large.get("response_ratio", 0.0))
        large_rank = float(large.get("rank", float("nan")))
        large_pairwise = float(large.get("pairwise", float("nan")))
        large_ratio_error = abs(np.log10(max(large_ratio, 1e-8)))
        large_rank_error = 1.0 - large_rank if np.isfinite(large_rank) else 1.0
        large_pairwise_error = 1.0 - large_pairwise if np.isfinite(large_pairwise) else 1.0
        d3_max_regret = float(metric.get("d3_max_regret_m3", d3_max_regret_reference_m3))
        regret_ratio = max(d3_max_regret, 0.0) / max(d3_max_regret_reference_m3, 1e-6)
        large_floor_penalty = (
            20.0 + 10.0 * (large_response_floor - large_ratio)
            if large_ratio < large_response_floor
            else 0.0
        )
        return (
            large_floor_penalty
            + 0.5 * large_ratio_error
            + 4.0 * large_rank_error
            + 4.0 * large_pairwise_error
            + 0.25 * rank
            + 0.25 * pairwise
            + 0.25 * sign
            + 0.5 * max(0.0, 1.0 - float(metric.get("top1_fraction", 1.0)))
            + regret_ratio
        )
    if policy != "calibration":
        raise ValueError("unknown V4.1 selection policy")
    return spread + rank + pairwise + sign + top1 + normalized_mae


@dataclass(frozen=True)
class ResponseLossWeightsV41:
    reference_state: float = 0.05
    reference_flow: float = 0.05
    delta_state: float = 1.0
    delta_flow: float = 1.0
    direct_tfv: float = 2.0
    centered_tfv: float = 5.0
    trajectory_tfv: float = 1.0
    consistency: float = 1.0
    ranking: float = 5.0
    interaction_energy: float = 0.01
    magnitude_calibration: float = 1.0


def response_group_loss_v41(
    output: CounterfactualGroupResponseV41,
    batch: ResponseGroupBatchV41,
    source_scales: SourceCounterfactualScalesV41,
    normalization: ResponseNormalizationV4,
    *,
    weights: ResponseLossWeightsV41 | None = None,
    magnitude_calibration: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One equally weighted group loss; the reference term is evaluated once."""

    w = weights or ResponseLossWeightsV41()
    device = output.direct_delta_tfv_m3.device
    dtype = output.direct_delta_tfv_m3.dtype
    absolute_state_scale = torch.as_tensor(
        normalization.state_std, device=device, dtype=dtype
    ).clamp_min(1e-6)
    absolute_flow_scale = torch.as_tensor(
        normalization.flow_std, device=device, dtype=dtype
    ).clamp_min(1e-6)
    delta_state_scale = torch.as_tensor(
        source_scales.state_scale, device=device, dtype=dtype
    ).clamp_min(1e-6)
    delta_flow_scale = torch.as_tensor(
        source_scales.flow_scale, device=device, dtype=dtype
    ).clamp_min(1e-6)
    reference_state = F.smooth_l1_loss(
        (output.reference_states_physical - batch.true_reference_states_physical)
        / absolute_state_scale,
        torch.zeros_like(output.reference_states_physical),
    )
    reference_flow = F.smooth_l1_loss(
        (output.reference_flows_physical - batch.true_reference_flows_physical)
        / absolute_flow_scale,
        torch.zeros_like(output.reference_flows_physical),
    )
    state_error = (
        output.delta_states_physical - batch.true_delta_states_physical
    ) / delta_state_scale
    flow_error = (
        output.delta_flows_physical - batch.true_delta_flows_physical
    ) / delta_flow_scale
    delta_state = F.smooth_l1_loss(state_error, torch.zeros_like(state_error))
    delta_flow = F.smooth_l1_loss(flow_error, torch.zeros_like(flow_error))
    tfv = tfv_loss_components_v41(
        output.direct_delta_tfv_m3,
        output.trajectory_delta_tfv_m3,
        batch.true_delta_tfv_m3,
        source_scale=torch.tensor(source_scales.tfv_scale_m3, device=device, dtype=dtype),
        magnitude_calibration=magnitude_calibration and batch.source_kind.upper() == "D3",
        magnitude_q33=source_scales.tfv_abs_quantiles_m3.get("q33"),
        magnitude_q67=source_scales.tfv_abs_quantiles_m3.get("q67"),
    )
    interaction_energy = (
        output.interaction_delta_states_physical.div(delta_state_scale).square().mean()
        + output.interaction_delta_flows_physical.div(delta_flow_scale).square().mean()
    )
    total = (
        w.reference_state * reference_state
        + w.reference_flow * reference_flow
        + w.delta_state * delta_state
        + w.delta_flow * delta_flow
        + w.direct_tfv * tfv["absolute_direct"]
        + w.centered_tfv * tfv["group_centered_direct"]
        + w.trajectory_tfv * tfv["authoritative_trajectory"]
        + w.consistency * tfv["direct_trajectory_consistency"]
        + w.ranking * tfv["ranking"]
        + w.interaction_energy * interaction_energy
        + w.magnitude_calibration * tfv["log_magnitude_calibration"]
    )
    components = {
        "loss": float(total.detach()),
        "reference_state_loss": float(reference_state.detach()),
        "reference_flow_loss": float(reference_flow.detach()),
        "delta_state_loss": float(delta_state.detach()),
        "delta_flow_loss": float(delta_flow.detach()),
        "direct_tfv_loss": float(tfv["absolute_direct"].detach()),
        "centered_tfv_loss": float(tfv["group_centered_direct"].detach()),
        "trajectory_tfv_loss": float(tfv["authoritative_trajectory"].detach()),
        "consistency_loss": float(tfv["direct_trajectory_consistency"].detach()),
        "ranking_loss": float(tfv["ranking"].detach()),
        "interaction_energy_loss": float(interaction_energy.detach()),
        "magnitude_calibration_loss": float(tfv["log_magnitude_calibration"].detach()),
    }
    if magnitude_calibration and batch.source_kind.upper() == "D3":
        weight_audit = balanced_magnitude_stratum_weights(
            batch.true_delta_tfv_m3.detach(),
            q33=float(source_scales.tfv_abs_quantiles_m3["q33"]),
            q67=float(source_scales.tfv_abs_quantiles_m3["q67"]),
        )
        for name in ("small", "medium", "large"):
            components[f"magnitude_{name}_weight_total"] = float(
                (weight_audit["weights"] * weight_audit[f"{name}_mask"]).sum(dim=1).mean()
            )
    return total, components


def prepare_graph_v41(
    model: DifferentiableCounterfactualResponseModelV41,
    graph: Any,
    normalization: ResponseNormalizationV4,
    device: torch.device,
) -> PreparedStaticV41:
    static = (
        np.asarray(graph.static_node_features, dtype=np.float32) - normalization.static_mean
    ) / np.maximum(normalization.static_std, 1e-6)
    physics = (
        np.asarray(graph.actuator_physics, dtype=np.float32) - normalization.physics_mean
    ) / np.maximum(normalization.physics_std, 1e-6)
    names = list(graph.static_node_feature_names)
    if "invert_elevation_m" not in names:
        raise ValueError("graph schema has no unambiguous invert_elevation_m")
    invert = np.asarray(graph.static_node_features, dtype=np.float32)[
        :, names.index("invert_elevation_m")
    ]
    return model.prepare_static(
        static_node_features=_tensor(static, device),
        actuator_physics=_tensor(physics, device),
        actuator_upstream=torch.as_tensor(graph.actuator_upstream, device=device),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream, device=device),
        edge_index=torch.as_tensor(graph.edge_index, device=device),
        invert_elevation_m=_tensor(invert, device),
    )


def evaluate_response_groups_v41(
    *,
    model: DifferentiableCounterfactualResponseModelV41,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    prepared: PreparedStaticV41,
    device: torch.device,
    batches: dict[str, ResponseGroupBatchV41] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    metrics: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    for group, pairs in sorted(grouped_pairs.items()):
        batch = batches[group] if batches is not None else stack_response_group_v41(pairs, device)
        with torch.no_grad():
            output = model.forward_group(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
                batch.elapsed_seconds,
                source_kind=batch.source_kind,
            )
        predicted = output.direct_delta_tfv_m3[0].detach().cpu().numpy()
        truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
        metrics.append(
            group_metrics_v41(
                predicted=predicted,
                truth=truth,
                group=group,
                source_kind=batch.source_kind,
            )
        )
        for candidate, values in enumerate(
            zip(
                truth,
                output.direct_single_delta_tfv_m3[0].detach().cpu().numpy(),
                output.direct_interaction_delta_tfv_m3[0].detach().cpu().numpy(),
                predicted,
                strict=True,
            )
        ):
            true_value, additive, interaction, final = values
            contributions.append(
                {
                    "group": group,
                    "source_kind": batch.source_kind,
                    "candidate_index": candidate,
                    "true_delta_tfv_m3": float(true_value),
                    "predicted_additive_single_delta_tfv_m3": float(additive),
                    "predicted_interaction_delta_tfv_m3": float(interaction),
                    "predicted_final_delta_tfv_m3": float(final),
                }
            )
    return metrics, contributions


def train_response_v41(
    *,
    model: DifferentiableCounterfactualResponseModelV41,
    grouped_pairs: dict[str, list[ResponsePairV4]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    out_path: str | Path,
    epochs: int,
    learning_rate: float = 2e-3,
    seed: int = 42,
    device: str = "cuda",
    early_stop_patience: int = 25,
    magnitude_calibration: bool = False,
    selection_policy: str = "calibration",
) -> dict[str, Any]:
    """Fit complete groups in FP32 and record calibration metrics every epoch."""

    if not grouped_pairs:
        raise ValueError("V4.1 training requires at least one complete group")
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target_device).float()
    prepared = prepare_graph_v41(model, graph, normalization, target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    stale = 0
    profile_totals = {
        "data_load_seconds": 0.0,
        "forward_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
    }
    stamp = time.perf_counter()
    batches = {
        group: stack_response_group_v41(pairs, target_device)
        for group, pairs in sorted(grouped_pairs.items())
    }
    profile_totals["data_load_seconds"] = time.perf_counter() - stamp
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    started = time.perf_counter()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    gpu_sampler = _GpuUtilizationSampler(target_device.type == "cuda")
    gpu_sampler.start()
    for epoch in range(1, int(epochs) + 1):
        order = sorted(grouped_pairs)
        np.random.default_rng(seed + epoch).shuffle(order)
        model.train()
        epoch_loss = 0.0
        gradient_norms: list[float] = []
        component_sums: dict[str, float] = {}
        for group in order:
            batch = batches[group]
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            stamp = time.perf_counter()
            output = model.forward_group(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
                batch.elapsed_seconds,
                source_kind=batch.source_kind,
            )
            loss, components = response_group_loss_v41(
                output,
                batch,
                scales.by_source[batch.source_kind],
                normalization,
                magnitude_calibration=magnitude_calibration,
            )
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            profile_totals["forward_seconds"] += time.perf_counter() - stamp
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V4.1 loss at epoch {epoch}, group {group}")
            optimizer.zero_grad(set_to_none=True)
            stamp = time.perf_counter()
            loss.backward()
            clear_disallowed_source_gradients(model, batch.source_kind)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            profile_totals["backward_seconds"] += time.perf_counter() - stamp
            if not np.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite V4.1 gradient at epoch {epoch}, group {group}"
                )
            stamp = time.perf_counter()
            optimizer.step()
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            profile_totals["optimizer_seconds"] += time.perf_counter() - stamp
            gradient_norms.append(gradient_norm)
            epoch_loss += float(loss.detach())
            for name, value in components.items():
                component_sums[name] = component_sums.get(name, 0.0) + value

        group_rows, epoch_contributions = evaluate_response_groups_v41(
            model=model,
            grouped_pairs=grouped_pairs,
            prepared=prepared,
            device=target_device,
            batches=batches,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "groups": len(order),
            "loss": epoch_loss / len(order),
            "spread_ratio": float(np.nanmean([item["spread_ratio"] for item in group_rows])),
            "rank": float(np.nanmean([item["rank"] for item in group_rows])),
            "pairwise": float(np.nanmean([item["pairwise"] for item in group_rows])),
            "sign": float(np.nanmean([item["sign"] for item in group_rows])),
            "top1": float(np.mean([item["top1"] for item in group_rows])),
            "top1_fraction": float(np.mean([item["top1"] for item in group_rows])),
            "gradient_norm": float(np.mean(gradient_norms)),
        }
        row.update({name: value / len(order) for name, value in component_sums.items()})
        d3_rows = [item for item in group_rows if item["source_kind"] == "D3"]
        row["d3_rank"] = float(np.nanmean([item["rank"] for item in d3_rows])) if d3_rows else float("nan")
        row["d3_pairwise"] = float(np.nanmean([item["pairwise"] for item in d3_rows])) if d3_rows else float("nan")
        row["d3_sign"] = float(np.nanmean([item["sign"] for item in d3_rows])) if d3_rows else float("nan")
        row["d3_top1_fraction"] = float(np.mean([item["top1"] for item in d3_rows])) if d3_rows else float("nan")
        row["d3_max_regret_m3"] = float(max((item["regret_m3"] for item in d3_rows), default=float("nan")))
        if magnitude_calibration:
            q = scales.by_source["D3"].tfv_abs_quantiles_m3
            row["d3_magnitude_strata"] = magnitude_strata_metrics_v41(
                epoch_contributions, q33=float(q["q33"]), q67=float(q["q67"])
            )
            row["magnitude_weight_audit"] = magnitude_weight_epoch_audit_v41(
                batches, q33=float(q["q33"]), q67=float(q["q67"])
            )
        history.append(row)
        if selection_policy == "d3_magnitude" and magnitude_calibration:
            calibration_score = calibration_selection_score_v41(row, policy=selection_policy)
        else:
            calibration_score = float(
                np.mean([calibration_selection_score_v41(item, policy=selection_policy) for item in group_rows])
            )
        if calibration_score + 1e-5 < best_score:
            best_score = calibration_score
            stale = 0
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(early_stop_patience):
            break
    gpu_utilization = gpu_sampler.stop()

    if best_state is not None:
        model.load_state_dict(best_state)
    group_rows, contributions = evaluate_response_groups_v41(
        model=model,
        grouped_pairs=grouped_pairs,
        prepared=prepared,
        device=target_device,
        batches=batches,
    )
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": TRAINING_CONTRACT_V41,
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "precision": "fp32",
            "full_train_manifest_sha256": scales.source_manifest_sha256,
        },
        out,
    )
    total_seconds = time.perf_counter() - started
    payload = {
        "contract": TRAINING_CONTRACT_V41,
        "checkpoint": str(out),
        "epochs_requested": int(epochs),
        "epochs_completed": len(history),
        "early_stopped": len(history) < int(epochs),
        "best_epoch": best_epoch,
        "groups": sorted(grouped_pairs),
        "device": str(target_device),
        "precision": "fp32",
        "full_train_manifest_sha256": scales.source_manifest_sha256,
        "history": history,
        "group_metrics": group_rows,
        "candidate_contributions": contributions,
        "profile_seconds": {**profile_totals, "wall_time_seconds": total_seconds},
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0
        ),
        "gpu_utilization": gpu_utilization,
        "loss_weights": asdict(ResponseLossWeightsV41()),
        "selection_policy": selection_policy,
    }
    out.with_suffix(out.suffix + ".history.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    return payload


__all__ = [
    "SCALE_CONTRACT_V41",
    "CounterfactualDeltaScalesV41",
    "ResponseGroupBatchV41",
    "ResponseLossWeightsV41",
    "ScaleDeltaBlockV41",
    "SourceCounterfactualScalesV41",
    "balanced_magnitude_stratum_weights",
    "clear_disallowed_source_gradients",
    "derive_counterfactual_delta_scales_v41",
    "evaluate_response_groups_v41",
    "group_metrics_v41",
    "magnitude_strata_metrics_v41",
    "magnitude_weight_epoch_audit_v41",
    "calibration_selection_score_v41",
    "load_or_derive_train_only_scales_v41",
    "prepare_graph_v41",
    "response_group_loss_v41",
    "source_parameter_is_trainable",
    "stack_response_group_v41",
    "tfv_loss_components_v41",
    "train_response_v41",
    "weighted_pairwise_ranking_loss",
]
