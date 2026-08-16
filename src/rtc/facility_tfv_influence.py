"""Model-independent facility influence analysis from authoritative SWMM trajectory caches.

This module answers a question that must be settled before changing the Step2 architecture:
which writable facilities measurably change future system-wide TFV under each sampled rainfall
and hydraulic checkpoint?

The strongest evidence is an exact same-prefix single-actuator counterfactual.  A candidate is
admitted to facility-level attribution only when reference and candidate share initial hydraulic
state, previous managed flow and causal rainfall, while exactly one actuator setting trajectory
changes.  Multi-actuator candidates are retained separately as joint/interaction evidence and are
never naively credited to individual facilities.

Full-horizon TFV uses ``exact_node_flood_volume_m3`` accumulated by authoritative SWMM routing
steps.  H30/H60/H120/H360 timing diagnostics use trapezoidal integration of authoritative SWMM
flooding-rate states sampled at the model step, so they are explicitly diagnostic rather than a
replacement for exact TFV.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .step2_train_response_v60 import V60TrainCache


FACILITY_TFV_INFLUENCE_CONTRACT = (
    "PROJECT7_AUTHORITATIVE_TRAJECTORY_FACILITY_TFV_INFLUENCE_V1"
)
DEFAULT_HORIZONS_MINUTES = (30, 60, 120, 360)
_FORBIDDEN_SPLIT_TOKENS = (
    "validation",
    "final",
    "formal",
    "policy_lock",
    "policy-lock",
    "locked",
    "blind",
    "challenge",
)


@dataclass(frozen=True)
class CacheSpec:
    label: str
    manifest_path: str


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_cache_spec(text: str) -> CacheSpec:
    raw = str(text).strip()
    if "=" not in raw:
        path = Path(raw)
        return CacheSpec(label=path.parent.name or path.stem, manifest_path=str(path))
    label, path = raw.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError("cache spec must be LABEL=PATH or PATH")
    return CacheSpec(label=label.strip(), manifest_path=path.strip())


def discover_compatible_cache_specs(roots: Sequence[str | Path]) -> tuple[list[CacheSpec], list[dict[str, str]]]:
    """Discover V60-compatible mmap caches without opening SWMM or mutating data.

    Discovery is intentionally tolerant: incompatible historical caches are reported as skipped.
    Duplicate cache copies are removed using their source-manifest SHA when available.
    """
    specs: list[CacheSpec] = []
    skipped: list[dict[str, str]] = []
    seen_source: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            skipped.append({"path": str(root), "reason": "root_missing"})
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("CACHE_MANIFEST.json"))
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                arrays = payload.get("shards", [{}])[0].get("arrays", {}) if payload.get("shards") else {}
                required = {
                    "initial_state",
                    "rainfall",
                    "settings",
                    "previous_actuator_flow",
                    "target_states",
                    "exact_node_flood_volume_m3",
                    "actuator_ids",
                    "scientific_split",
                }
                if not required <= set(arrays):
                    skipped.append({"path": str(path), "reason": "missing_required_arrays"})
                    continue
                source_sha = str(payload.get("source_manifest_sha256") or sha256_file(path))
                if source_sha in seen_source:
                    skipped.append({"path": str(path), "reason": "duplicate_source_manifest"})
                    continue
                # Validate the actual cache contract/index before accepting discovery.
                V60TrainCache(path)
                seen_source.add(source_sha)
                specs.append(
                    CacheSpec(
                        label=f"{root.name}:{path.parent.name}:{source_sha[:8]}",
                        manifest_path=str(path),
                    )
                )
            except Exception as exc:  # historical roots legitimately contain incompatible caches
                skipped.append({"path": str(path), "reason": f"incompatible:{type(exc).__name__}:{exc}"})
    return specs, skipped


def _row_text(arrays: dict[str, np.ndarray], name: str, index: int, default: str = "") -> str:
    if name not in arrays:
        return default
    values = np.asarray(arrays[name])
    if values.ndim == 0:
        return str(values.item())
    return str(values[index])


def _actuator_ids(arrays: dict[str, np.ndarray], index: int, count: int) -> list[str]:
    raw = np.asarray(arrays["actuator_ids"])
    if raw.ndim == 1 and raw.size == count:
        values = raw
    else:
        values = np.asarray(raw[index]).reshape(-1)
    if values.size != count:
        raise ValueError(f"actuator id count {values.size} differs from setting width {count}")
    return [str(value) for value in values]


def _exact_tfv_m3(arrays: dict[str, np.ndarray], index: int) -> float:
    value = np.asarray(arrays["exact_node_flood_volume_m3"][index], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("exact_node_flood_volume_m3 contains non-finite values")
    return float(np.clip(value, 0.0, None).sum())


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def same_prefix_diagnostics(
    arrays: dict[str, np.ndarray], reference: int, candidate: int
) -> dict[str, float | bool]:
    state = _max_abs_difference(arrays["initial_state"][reference], arrays["initial_state"][candidate])
    flow = _max_abs_difference(
        arrays["previous_actuator_flow"][reference], arrays["previous_actuator_flow"][candidate]
    )
    rain = _max_abs_difference(arrays["rainfall"][reference], arrays["rainfall"][candidate])
    tolerance = 1.0e-7
    return {
        "initial_state_max_abs_diff": state,
        "previous_flow_max_abs_diff": flow,
        "rainfall_max_abs_diff": rain,
        "same_prefix": bool(max(state, flow, rain) <= tolerance),
    }


def sampled_cumulative_tfv_m3(
    initial_state: np.ndarray,
    future_states: np.ndarray,
    *,
    flood_rate_index: int,
    dt_seconds: float,
) -> np.ndarray:
    """Cumulative sampled TFV using the same representation audit semantics as V127."""
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    initial = np.asarray(initial_state, dtype=np.float64)
    future = np.asarray(future_states, dtype=np.float64)
    if initial.ndim != 2 or future.ndim != 3 or initial.shape[0] != future.shape[1]:
        raise ValueError("invalid state shapes for sampled TFV integration")
    if not 0 <= int(flood_rate_index) < initial.shape[-1]:
        raise ValueError("flood_rate_index outside state feature dimension")
    initial_rate = np.maximum(initial[:, int(flood_rate_index)], 0.0)
    future_rate = np.maximum(future[..., int(flood_rate_index)], 0.0)
    rates = np.concatenate((initial_rate[None, :], future_rate), axis=0)
    step_volume = 0.5 * (rates[:-1] + rates[1:]) * float(dt_seconds)
    return np.cumsum(step_volume.sum(axis=1), axis=0)


def _horizon_values(
    arrays: dict[str, np.ndarray],
    index: int,
    *,
    flood_rate_index: int,
    dt_seconds: float,
    horizons_minutes: Sequence[int],
) -> tuple[dict[str, float], float]:
    cumulative = sampled_cumulative_tfv_m3(
        arrays["initial_state"][index],
        arrays["target_states"][index],
        flood_rate_index=flood_rate_index,
        dt_seconds=dt_seconds,
    )
    available_minutes = float(len(cumulative) * dt_seconds / 60.0)
    values: dict[str, float] = {}
    for minutes in horizons_minutes:
        steps = int(round(float(minutes) * 60.0 / float(dt_seconds)))
        key = f"sampled_tfv_h{int(minutes)}_m3"
        values[key] = float(cumulative[steps - 1]) if 0 < steps <= len(cumulative) else float("nan")
    return values, available_minutes


def _split_allowed(split: str) -> bool:
    text = str(split).strip().lower()
    return not any(token in text for token in _FORBIDDEN_SPLIT_TOKENS)


def _rainfall_selected(rainfall_group: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch(str(rainfall_group), pattern) for pattern in patterns)


def _effect_class(delta_tfv_m3: float, threshold_m3: float) -> str:
    if delta_tfv_m3 <= -threshold_m3:
        return "BENEFICIAL"
    if delta_tfv_m3 >= threshold_m3:
        return "HARMFUL"
    return "BELOW_PRACTICAL_THRESHOLD"


def _direction(values: np.ndarray, tolerance: float = 1.0e-8) -> str:
    x = np.asarray(values, dtype=np.float64)
    active = x[np.abs(x) > tolerance]
    if active.size == 0:
        return "UNCHANGED"
    if np.all(active > 0):
        return "INCREASE"
    if np.all(active < 0):
        return "DECREASE"
    return "MIXED"


def _candidate_identity(arrays: dict[str, np.ndarray], index: int) -> str:
    value = _row_text(arrays, "action_or_sequence_sha256", index)
    return value or f"row:{index}"


def extract_cache_influence_rows(
    cache: V60TrainCache,
    *,
    source_label: str,
    rainfall_patterns: Sequence[str] = (),
    flood_rate_index: int = 2,
    dt_seconds: float = 300.0,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
    meaningful_absolute_m3: float = 1.0,
    meaningful_relative: float = 0.01,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Extract exact single-facility and joint-candidate TFV evidence from one cache."""
    if meaningful_absolute_m3 < 0 or meaningful_relative < 0:
        raise ValueError("meaningful thresholds must be non-negative")
    exact_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    counters = {
        "groups_seen": 0,
        "groups_selected": 0,
        "groups_excluded_split": 0,
        "candidate_pairs_seen": 0,
        "candidate_pairs_prefix_mismatch": 0,
        "single_actuator_pairs": 0,
        "joint_actuator_pairs": 0,
        "zero_action_pairs": 0,
    }

    for name in cache.names():
        entry = cache.entry(name)
        arrays = entry.arrays
        ref = int(entry.reference_index)
        counters["groups_seen"] += 1
        split = _row_text(arrays, "scientific_split", ref)
        if not _split_allowed(split):
            counters["groups_excluded_split"] += 1
            continue
        if not _rainfall_selected(entry.rainfall_group, rainfall_patterns):
            continue
        counters["groups_selected"] += 1

        reference_settings = np.asarray(arrays["settings"][ref], dtype=np.float64)
        if reference_settings.ndim != 2:
            raise ValueError(f"{name}: settings must have [time, actuator] shape")
        actuator_ids = _actuator_ids(arrays, ref, int(reference_settings.shape[-1]))
        reference_exact = _exact_tfv_m3(arrays, ref)
        reference_horizon, available_minutes = _horizon_values(
            arrays,
            ref,
            flood_rate_index=flood_rate_index,
            dt_seconds=dt_seconds,
            horizons_minutes=horizons_minutes,
        )
        threshold = max(
            float(meaningful_absolute_m3),
            float(meaningful_relative) * max(reference_exact, 0.0),
        )
        base_sha = _row_text(arrays, "base_action_sha256", ref)

        for candidate in entry.indices:
            candidate = int(candidate)
            if candidate == ref:
                continue
            counters["candidate_pairs_seen"] += 1
            prefix = same_prefix_diagnostics(arrays, ref, candidate)
            if not bool(prefix["same_prefix"]):
                counters["candidate_pairs_prefix_mismatch"] += 1
                continue
            candidate_settings = np.asarray(arrays["settings"][candidate], dtype=np.float64)
            if candidate_settings.shape != reference_settings.shape:
                raise ValueError(f"{name}: candidate/reference setting shape differs")
            action_delta = candidate_settings - reference_settings
            changed = np.flatnonzero(np.any(np.abs(action_delta) > 1.0e-7, axis=0))
            if changed.size == 0:
                counters["zero_action_pairs"] += 1
                continue

            candidate_exact = _exact_tfv_m3(arrays, candidate)
            delta_exact = float(candidate_exact - reference_exact)
            reduction_pct = (
                float(-100.0 * delta_exact / reference_exact) if reference_exact > 1.0e-12 else float("nan")
            )
            candidate_horizon, candidate_available = _horizon_values(
                arrays,
                candidate,
                flood_rate_index=flood_rate_index,
                dt_seconds=dt_seconds,
                horizons_minutes=horizons_minutes,
            )
            common: dict[str, object] = {
                "source_label": str(source_label),
                "source_kind": str(entry.source_kind),
                "group_name": str(name),
                "rainfall_group": str(entry.rainfall_group),
                "event_id": str(entry.event_id),
                "checkpoint_id": str(entry.checkpoint_id),
                "scientific_split": split,
                "base_action_sha256": base_sha,
                "candidate_action_sha256": _candidate_identity(arrays, candidate),
                "reference_tfv_exact_m3": reference_exact,
                "candidate_tfv_exact_m3": candidate_exact,
                "delta_tfv_exact_m3": delta_exact,
                "tfv_reduction_pct": reduction_pct,
                "practical_threshold_m3": threshold,
                "effect_class": _effect_class(delta_exact, threshold),
                "changed_actuator_count": int(changed.size),
                "trajectory_horizon_minutes": min(available_minutes, candidate_available),
                **prefix,
            }
            for minutes in horizons_minutes:
                key = f"sampled_tfv_h{int(minutes)}_m3"
                left, right = reference_horizon[key], candidate_horizon[key]
                common[f"delta_tfv_h{int(minutes)}_sampled_m3"] = (
                    float(right - left) if np.isfinite(left) and np.isfinite(right) else float("nan")
                )

            if changed.size == 1:
                counters["single_actuator_pairs"] += 1
                actuator = int(changed[0])
                series = action_delta[:, actuator]
                active_blocks = np.flatnonzero(np.abs(series) > 1.0e-7)
                row = {
                    **common,
                    "actuator_index": actuator,
                    "actuator_id": actuator_ids[actuator],
                    "action_direction": _direction(series),
                    "changed_control_blocks": int(active_blocks.size),
                    "first_changed_block": int(active_blocks.min()) if active_blocks.size else -1,
                    "last_changed_block": int(active_blocks.max()) if active_blocks.size else -1,
                    "max_abs_setting_delta": float(np.max(np.abs(series))),
                    "mean_setting_delta_on_active": (
                        float(np.mean(series[active_blocks])) if active_blocks.size else 0.0
                    ),
                    "signed_setting_exposure_blocks": float(np.sum(series)),
                }
                h30 = float(row.get("delta_tfv_h30_sampled_m3", float("nan")))
                later = [
                    float(row.get(f"delta_tfv_h{int(minutes)}_sampled_m3", float("nan")))
                    for minutes in horizons_minutes
                    if int(minutes) > 30
                ]
                row["delayed_benefit_sampled"] = bool(
                    delta_exact <= -threshold
                    and (not np.isfinite(h30) or h30 > -threshold)
                    and any(np.isfinite(value) and value <= -threshold for value in later)
                )
                exact_rows.append(row)
            else:
                counters["joint_actuator_pairs"] += 1
                changed_ids = [actuator_ids[int(i)] for i in changed]
                active = np.abs(action_delta[:, changed]) > 1.0e-7
                joint_rows.append(
                    {
                        **common,
                        "changed_actuator_ids": "|".join(changed_ids),
                        "action_l1_setting_blocks": float(np.abs(action_delta[:, changed]).sum()),
                        "action_l2_setting_blocks": float(np.linalg.norm(action_delta[:, changed])),
                        "changed_setting_cells": int(active.sum()),
                        "attribution_semantics": (
                            "JOINT_EFFECT_ONLY; do not assign joint delta TFV to individual facilities"
                        ),
                    }
                )

    return exact_rows, joint_rows, counters


def _aggregate_exact(
    exact: pd.DataFrame,
    *,
    keys: Sequence[str],
    all_rainfalls: Sequence[str] | None = None,
    all_actuators: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = {
        tuple(str(value) for value in (key if isinstance(key, tuple) else (key,))): group
        for key, group in exact.groupby(list(keys), sort=True, dropna=False)
    } if not exact.empty else {}

    if list(keys) == ["rainfall_group", "actuator_id"] and all_rainfalls is not None and all_actuators is not None:
        identities: Iterable[tuple[str, ...]] = (
            (str(rain), str(actuator)) for rain in all_rainfalls for actuator in all_actuators
        )
    elif list(keys) == ["actuator_id"] and all_actuators is not None:
        identities = ((str(actuator),) for actuator in all_actuators)
    else:
        identities = sorted(grouped)

    for identity in identities:
        group = grouped.get(tuple(identity))
        base = dict(zip(keys, identity, strict=True))
        if group is None or group.empty:
            rows.append(
                {
                    **base,
                    "tested_pairs": 0,
                    "tested_checkpoints": 0,
                    "tested_events": 0,
                    "beneficial_pairs": 0,
                    "harmful_pairs": 0,
                    "below_threshold_pairs": 0,
                    "beneficial_fraction": 0.0,
                    "harmful_fraction": 0.0,
                    "median_delta_tfv_exact_m3": np.nan,
                    "median_abs_tfv_effect_m3": np.nan,
                    "max_abs_tfv_effect_m3": np.nan,
                    "best_tfv_reduction_m3": np.nan,
                    "worst_tfv_increase_m3": np.nan,
                    "max_tfv_reduction_pct": np.nan,
                    "delayed_beneficial_pairs": 0,
                    "evidence_class": "UNTESTED_SINGLE_ACTUATOR",
                    "has_sampled_control_value": False,
                }
            )
            continue

        delta = group["delta_tfv_exact_m3"].astype(float).to_numpy()
        beneficial = group["effect_class"].astype(str).eq("BENEFICIAL")
        harmful = group["effect_class"].astype(str).eq("HARMFUL")
        neutral = ~(beneficial | harmful)
        if beneficial.any() and harmful.any():
            evidence_class = "STATE_DEPENDENT_MIXED"
        elif beneficial.any():
            evidence_class = "BENEFICIAL_IN_SAMPLED_ACTIONS"
        elif harmful.any():
            evidence_class = "HARMFUL_ONLY_IN_SAMPLED_ACTIONS"
        else:
            evidence_class = "BELOW_PRACTICAL_THRESHOLD"
        reduction_pct = group["tfv_reduction_pct"].astype(float).to_numpy()
        reduction_pct = reduction_pct[np.isfinite(reduction_pct)]
        row: dict[str, object] = {
            **base,
            "tested_pairs": int(len(group)),
            "tested_checkpoints": int(group["checkpoint_id"].astype(str).nunique()),
            "tested_events": int(group["event_id"].astype(str).nunique()),
            "beneficial_pairs": int(beneficial.sum()),
            "harmful_pairs": int(harmful.sum()),
            "below_threshold_pairs": int(neutral.sum()),
            "beneficial_fraction": float(beneficial.mean()),
            "harmful_fraction": float(harmful.mean()),
            "median_delta_tfv_exact_m3": float(np.median(delta)),
            "median_abs_tfv_effect_m3": float(np.median(np.abs(delta))),
            "max_abs_tfv_effect_m3": float(np.max(np.abs(delta))),
            "best_tfv_reduction_m3": float(max(0.0, -float(np.min(delta)))),
            "worst_tfv_increase_m3": float(max(0.0, float(np.max(delta)))),
            "max_tfv_reduction_pct": float(np.max(reduction_pct)) if reduction_pct.size else np.nan,
            "delayed_beneficial_pairs": int(group["delayed_benefit_sampled"].astype(bool).sum()),
            "evidence_class": evidence_class,
            "has_sampled_control_value": bool(beneficial.any()),
        }
        for minutes in DEFAULT_HORIZONS_MINUTES:
            column = f"delta_tfv_h{int(minutes)}_sampled_m3"
            if column in group:
                values = group[column].astype(float).to_numpy()
                values = values[np.isfinite(values)]
                row[f"median_{column}"] = float(np.median(values)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_facility_tfv_influence(
    specs: Sequence[CacheSpec],
    *,
    rainfall_patterns: Sequence[str] = (),
    flood_rate_index: int = 2,
    dt_seconds: float = 300.0,
    meaningful_absolute_m3: float = 1.0,
    meaningful_relative: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Analyze one or more compatible trajectory caches and return four evidence tables + report."""
    if not specs:
        raise ValueError("facility influence analysis requires at least one compatible cache")
    exact_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    source_reports: list[dict[str, object]] = []
    all_actuators: set[str] = set()
    all_rainfalls: set[str] = set()

    for spec in specs:
        path = Path(spec.manifest_path)
        if not path.is_file():
            raise ValueError(f"cache manifest missing: {path}")
        cache = V60TrainCache(path)
        local_exact, local_joint, counters = extract_cache_influence_rows(
            cache,
            source_label=spec.label,
            rainfall_patterns=rainfall_patterns,
            flood_rate_index=flood_rate_index,
            dt_seconds=dt_seconds,
            meaningful_absolute_m3=meaningful_absolute_m3,
            meaningful_relative=meaningful_relative,
        )
        exact_rows.extend(local_exact)
        joint_rows.extend(local_joint)
        for row in local_exact:
            all_actuators.add(str(row["actuator_id"]))
            all_rainfalls.add(str(row["rainfall_group"]))
        for row in local_joint:
            all_rainfalls.add(str(row["rainfall_group"]))
            all_actuators.update(str(row["changed_actuator_ids"]).split("|"))
        source_reports.append(
            {
                "label": spec.label,
                "manifest_path": str(path),
                "manifest_sha256": sha256_file(path),
                **counters,
            }
        )

    exact = pd.DataFrame(exact_rows)
    joint = pd.DataFrame(joint_rows)
    exact_before = int(len(exact))
    joint_before = int(len(joint))
    identity = [
        "rainfall_group",
        "event_id",
        "checkpoint_id",
        "base_action_sha256",
        "candidate_action_sha256",
    ]
    if not exact.empty:
        exact = exact.sort_values(identity + ["source_label"]).drop_duplicates(identity, keep="first")
    if not joint.empty:
        joint = joint.sort_values(identity + ["source_label"]).drop_duplicates(identity, keep="first")

    rainfalls = sorted(all_rainfalls)
    actuators = sorted(value for value in all_actuators if value)
    by_rain = _aggregate_exact(
        exact,
        keys=("rainfall_group", "actuator_id"),
        all_rainfalls=rainfalls,
        all_actuators=actuators,
    )
    global_facility = _aggregate_exact(exact, keys=("actuator_id",), all_actuators=actuators)

    tested = by_rain[by_rain["tested_pairs"] > 0] if not by_rain.empty else by_rain
    valuable = by_rain[by_rain["has_sampled_control_value"]] if not by_rain.empty else by_rain
    top_by_rain: dict[str, list[dict[str, object]]] = {}
    for rain in rainfalls:
        subset = by_rain[
            (by_rain["rainfall_group"].astype(str) == rain)
            & (by_rain["tested_pairs"].astype(int) > 0)
        ].copy()
        subset = subset.sort_values(
            ["best_tfv_reduction_m3", "max_abs_tfv_effect_m3"], ascending=[False, False]
        ).head(15)
        top_by_rain[rain] = [
            {
                "actuator_id": str(row.actuator_id),
                "tested_pairs": int(row.tested_pairs),
                "best_tfv_reduction_m3": float(row.best_tfv_reduction_m3),
                "max_abs_tfv_effect_m3": float(row.max_abs_tfv_effect_m3),
                "beneficial_fraction": float(row.beneficial_fraction),
                "evidence_class": str(row.evidence_class),
            }
            for row in subset.itertuples(index=False)
        ]

    beneficial_actuators = (
        int(global_facility["has_sampled_control_value"].astype(bool).sum())
        if not global_facility.empty
        else 0
    )
    measured_actuators = (
        int((global_facility["tested_pairs"].astype(int) > 0).sum()) if not global_facility.empty else 0
    )
    report: dict[str, object] = {
        "contract": FACILITY_TFV_INFLUENCE_CONTRACT,
        "authoritative_truth": "SWMM exact_node_flood_volume_m3",
        "facility_attribution": "same-prefix exact single-actuator counterfactual only",
        "joint_attribution": "joint effect retained; no naive per-actuator credit assignment",
        "timing_diagnostic": (
            "H30/H60/H120/H360 use trapezoidal integration of authoritative 300-s flooding-rate states"
        ),
        "model_predictions_used": False,
        "gradient_labels_used": False,
        "source_reports": source_reports,
        "rainfall_patterns": list(rainfall_patterns),
        "rainfall_groups_analyzed": rainfalls,
        "actuator_ids_seen": len(actuators),
        "actuators_with_exact_single_evidence": measured_actuators,
        "actuators_with_at_least_one_beneficial_sampled_action": beneficial_actuators,
        "rainfall_actuator_cells_tested": int(len(tested)),
        "rainfall_actuator_cells_with_beneficial_action": int(len(valuable)),
        "exact_single_pair_rows": int(len(exact)),
        "joint_pair_rows": int(len(joint)),
        "duplicate_exact_rows_removed": exact_before - int(len(exact)),
        "duplicate_joint_rows_removed": joint_before - int(len(joint)),
        "physical_feasibility_interpretation": (
            "PHYSICAL_TFV_CONTROL_SIGNAL_PRESENT"
            if beneficial_actuators > 0
            else "NO_BENEFICIAL_SINGLE_ACTUATOR_SIGNAL_FOUND_IN_SELECTED_DATA"
        ),
        "collapse_interpretation": (
            "If exact SWMM single-actuator TFV effects are material while the surrogate response is near zero, "
            "the collapse is a learning/representation/training problem rather than proof that RTC is physically infeasible."
        ),
        "coverage_warning": (
            "UNTESTED_SINGLE_ACTUATOR means no admissible same-prefix single-facility evidence was found; "
            "it does not mean the facility is physically ineffective."
        ),
        "top_facilities_by_rainfall": top_by_rain,
        "boundary": {
            "read_only": True,
            "used_for_training": False,
            "swmm_launched": False,
            "nondevelopment_splits_excluded": True,
        },
    }
    return exact, by_rain, global_facility, joint, report


__all__ = [
    "CacheSpec",
    "DEFAULT_HORIZONS_MINUTES",
    "FACILITY_TFV_INFLUENCE_CONTRACT",
    "analyze_facility_tfv_influence",
    "discover_compatible_cache_specs",
    "extract_cache_influence_rows",
    "parse_cache_spec",
    "sampled_cumulative_tfv_m3",
]
