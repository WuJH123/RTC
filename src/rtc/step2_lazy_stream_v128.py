"""Lazy host-memory branch streaming for current V128 training.

The legacy streaming path avoided full-group CUDA materialization but still called
``cache.batch()``, which copied every candidate state/flow trajectory into host Torch tensors
before selecting a 2-4 branch microbatch. On a 16-GB workstation this can trigger Windows
paging while the GPU is under-utilised.

Current V128 can safely avoid that copy because its outer cache is contractually a
``CausalStep1StateCacheV127`` wrapping a causal rainfall cache. This module reads only the
small shared causal inputs eagerly (Step1 state, rainfall forecast, previous actuator flow and
settings) and keeps the large SWMM truth state/flow arrays as mmap-backed branch views until
``select_to_device_v128_lazy`` requests the active microbatch.

Scientific semantics are unchanged: the same ordered branches, causal initial state, causal
rainfall, settings and SWMM truth labels are returned. Unknown cache types fail back to the
audited legacy implementation rather than guessing their wrapper structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _branch_indices
from . import step2_train_v127_streaming as _legacy_streaming

V128_LAZY_STREAM_CONTRACT = "PROJECT7_V128_LAZY_MMAP_BRANCH_STREAM_V1"
_LEGACY_CPU_GROUP = _legacy_streaming._cpu_group
_LEGACY_SELECT_TO_DEVICE = _legacy_streaming._select_to_device


@dataclass(frozen=True)
class LazyBranchArrayV128:
    source: Any
    raw_branch_indices: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        source_shape = tuple(int(v) for v in np.asarray(self.source).shape)
        if len(source_shape) < 2:
            raise ValueError("lazy branch source must contain branch and time axes")
        return (int(self.raw_branch_indices.size), *source_shape[1:])

    def select(self, positions: np.ndarray, *, horizon: int) -> np.ndarray:
        logical = np.asarray(positions, dtype=np.int64).reshape(-1)
        if logical.size <= 0:
            raise ValueError("lazy branch selection is empty")
        if np.any(logical < 0) or np.any(logical >= self.raw_branch_indices.size):
            raise IndexError("lazy branch logical position outside group")
        h = int(horizon)
        if h <= 0 or h > self.shape[1]:
            raise ValueError("lazy branch horizon is invalid")
        raw = self.raw_branch_indices[logical]
        return np.asarray(self.source[raw, :h], dtype=np.float32)


def _current_causal_group(cache: Any, name: str) -> dict[str, Any] | None:
    """Return current causal components without calling ``cache.batch()``.

    The structural checks are intentionally strict. If any expected wrapper component is
    absent, the caller falls back to the legacy path so historical cache types keep working.
    """
    state_store = getattr(cache, "store", None)
    rain_cache = getattr(cache, "base", None)
    rain_store = getattr(rain_cache, "store", None)
    rain_index = getattr(rain_cache, "_index", None)
    raw_cache = getattr(rain_cache, "base", None)
    if state_store is None or rain_cache is None or rain_store is None or not isinstance(rain_index, dict) or raw_cache is None:
        return None
    if name not in rain_index:
        return None
    try:
        entry = raw_cache.entry(name)
    except Exception:
        return None
    if entry is not cache.entry(name):
        # Wrapper entry methods should preserve the exact frozen group object. Do not try to
        # infer ordering if a future wrapper changes that invariant.
        return None
    arrays = entry.arrays
    required = {
        "settings",
        "target_states",
        "target_actuator_flows",
        "previous_actuator_flow",
    }
    if not required.issubset(set(arrays)):
        return None
    order = np.asarray(_branch_indices(entry), dtype=np.int64)
    ref = int(order[0])
    initial = np.asarray(state_store.state_for(entry), dtype=np.float32)
    rainfall = np.asarray(rain_store.forecast_mmhr[int(rain_index[name])], dtype=np.float32)
    previous = np.asarray(arrays["previous_actuator_flow"][ref], dtype=np.float32)
    settings = np.asarray(arrays["settings"][order], dtype=np.float32)
    if settings.ndim != 3:
        raise ValueError(f"{name}: lazy V128 settings must be [branch,H,actuator]")
    if rainfall.shape[0] != settings.shape[1]:
        raise ValueError(f"{name}: lazy V128 causal rainfall horizon differs from settings")
    return {
        "initial": torch.as_tensor(initial, dtype=torch.float32, device="cpu")[None],
        "rainfall": torch.as_tensor(rainfall, dtype=torch.float32, device="cpu")[None],
        "previous_flow": torch.as_tensor(previous, dtype=torch.float32, device="cpu")[None],
        "settings": torch.as_tensor(settings, dtype=torch.float32, device="cpu"),
        "states": LazyBranchArrayV128(arrays["target_states"], order),
        "flows": LazyBranchArrayV128(arrays["target_actuator_flows"], order),
        "lazy_mmap_branch_stream": True,
        "lazy_stream_contract": V128_LAZY_STREAM_CONTRACT,
    }


def cpu_group_v128_lazy(
    cache: Any,
    name: str,
    normalization: InputNormalizationV60,
) -> dict[str, Any]:
    """Return physical causal inputs + lazy truth branch views for current caches."""
    del normalization  # current causal values and SWMM truth are already physical SI.
    current = _current_causal_group(cache, str(name))
    if current is not None:
        return current
    return _LEGACY_CPU_GROUP(cache, name, normalization)  # pragma: no cover - archival fallback


def select_to_device_v128_lazy(
    data: dict[str, Any],
    positions: Sequence[int] | np.ndarray,
    *,
    device: torch.device,
    horizon: int | None = None,
    include_truth: bool = True,
) -> dict[str, torch.Tensor]:
    if not bool(data.get("lazy_mmap_branch_stream", False)):
        return _LEGACY_SELECT_TO_DEVICE(
            data,
            positions,
            device=device,
            horizon=horizon,
            include_truth=include_truth,
        )
    logical = np.asarray(positions, dtype=np.int64).reshape(-1)
    if logical.size <= 0:
        raise ValueError("V128 lazy streaming selection is empty")
    settings = data["settings"]
    h = int(settings.shape[1] if horizon is None else horizon)
    if h <= 0 or h > int(settings.shape[1]):
        raise ValueError("V128 lazy streaming horizon is invalid")
    pos = torch.as_tensor(logical, dtype=torch.long, device="cpu")
    n = int(logical.size)
    result: dict[str, torch.Tensor] = {
        "initial": data["initial"].to(device).expand(n, -1, -1),
        "rainfall": data["rainfall"][:, :h].to(device).expand(n, -1, -1, -1),
        "previous_flow": data["previous_flow"].to(device).expand(n, -1),
        "settings": settings.index_select(0, pos)[:, :h].to(device),
    }
    if include_truth:
        dtype = result["initial"].dtype
        state_np = data["states"].select(logical, horizon=h)
        flow_np = data["flows"].select(logical, horizon=h)
        result["states"] = torch.as_tensor(state_np, dtype=dtype, device=device)
        result["flows"] = torch.as_tensor(flow_np, dtype=dtype, device=device)
    return result


def install_v128_lazy_streaming() -> None:
    """Patch shared streaming globals for the current process only.

    V128 exact/hydraulic modules import their helpers by name, so the current runner also
    replaces those module globals explicitly. Historical command lines are unaffected unless
    they intentionally call this installer.
    """
    from . import step2_train_v128_exact as exact
    from . import step2_train_v128_hydraulic as hydraulic

    _legacy_streaming._cpu_group = cpu_group_v128_lazy
    _legacy_streaming._select_to_device = select_to_device_v128_lazy
    exact._cpu_group = cpu_group_v128_lazy
    exact._select_to_device = select_to_device_v128_lazy
    hydraulic._cpu_group = cpu_group_v128_lazy
    hydraulic._select_to_device = select_to_device_v128_lazy


__all__ = [
    "LazyBranchArrayV128",
    "V128_LAZY_STREAM_CONTRACT",
    "cpu_group_v128_lazy",
    "install_v128_lazy_streaming",
    "select_to_device_v128_lazy",
]
