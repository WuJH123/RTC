"""TrainFit-only signed hydraulic-effect distribution audit for V11.3.

This is a read-only census.  It uses candidate-reference deltas directly and
never clips negative effects or reads InternalHoldout outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_graph_distance_v112 import all_actuator_endpoint_hops_v112
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v110_contract import HydraulicHorizonV110
from rtc.step2_v113_audit import signed_distribution_v113
from run_step2_v110 import _load_graph

STATE_NAMES = ("depth_m", "flooding_rate_m3s", "storage_volume_m3", "inflow_m3s", "outflow_m3s")
STATE_CHANNELS = (0, 2, 3, 4, 5)
FLOW_NAME = "managed_flow_m3s"
DISTANCE_BINS = ("endpoint", "1_4", "5_8", "gt8", "disconnected")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _type_labels(graph) -> list[str]:
    names = list(graph.actuator_physics_feature_names)
    physics = np.asarray(graph.actuator_physics)
    labels = []
    for row in physics:
        label = "other"
        for feature in ("is_pump", "is_orifice", "is_weir", "is_outlet"):
            if float(row[names.index(feature)]) > 0.5:
                label = feature[3:]
                break
        labels.append(label)
    return labels


def _phase_load(entry, max_depth: np.ndarray) -> float:
    state = np.asarray(entry.arrays["initial_state"][entry.reference_index], dtype=np.float64)
    valid = np.isfinite(state[:, 0]) & np.isfinite(max_depth) & (max_depth > 1e-6)
    if not valid.any():
        raise ValueError("V113 phase audit found no valid depth/capacity nodes")
    return float(np.quantile(state[valid, 0] / max_depth[valid], 0.90))


class _Stats:
    """Exact moments/counts plus a deterministic bounded quantile sample."""

    def __init__(self, cap: int = 2048) -> None:
        self.cap = int(cap)
        self.count = 0
        self.active_count = 0
        self.positive_active = 0
        self.negative_active = 0
        self.total = 0.0
        self.square = 0.0
        self.sample = np.empty(0, dtype=np.float64)

    def add(self, values: np.ndarray, active: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        a = np.asarray(active, dtype=bool).reshape(-1)
        if x.shape != a.shape:
            raise ValueError("V113 signed audit values and active mask misaligned")
        finite = np.isfinite(x)
        x, a = x[finite], a[finite]
        if x.size == 0:
            return
        self.count += int(x.size)
        self.active_count += int(a.sum())
        active_values = x[a]
        self.positive_active += int((active_values > 0).sum())
        self.negative_active += int((active_values < 0).sum())
        self.total += float(x.sum())
        self.square += float(np.square(x).sum())
        stride = max(1, int(np.ceil(x.size / self.cap)))
        sampled = x[::stride][: self.cap]
        self.sample = np.concatenate((self.sample, sampled))
        if self.sample.size > self.cap:
            keep_stride = max(1, int(np.ceil(self.sample.size / self.cap)))
            self.sample = self.sample[::keep_stride][: self.cap]

    def finalize(self) -> dict[str, Any]:
        if self.count == 0:
            return signed_distribution_v113(np.empty(0))
        sample = self.sample
        return {
            "count": int(self.count),
            "p50_abs": float(np.quantile(np.abs(sample), 0.50)),
            "p75_abs": float(np.quantile(np.abs(sample), 0.75)),
            "p90_abs": float(np.quantile(np.abs(sample), 0.90)),
            "p95_abs": float(np.quantile(np.abs(sample), 0.95)),
            "p99_abs": float(np.quantile(np.abs(sample), 0.99)),
            "max_abs": float(np.max(np.abs(sample))),
            "signed_mean": float(self.total / self.count),
            "positive_active_fraction": float(self.positive_active / self.active_count) if self.active_count else None,
            "negative_active_fraction": float(self.negative_active / self.active_count) if self.active_count else None,
            "active_fraction": float(self.active_count / self.count),
            "effect_rms": float(np.sqrt(self.square / self.count)),
            "quantiles_from_deterministic_sample_cap": int(self.cap),
        }


def _add(table: dict[str, _Stats], key: str, values: np.ndarray, active: np.ndarray) -> None:
    table.setdefault(key, _Stats()).add(values, active)


def _finalize_table(table: dict[str, _Stats]) -> dict[str, Any]:
    return {key: value.finalize() for key, value in sorted(table.items())}


def _distance_masks(distance: np.ndarray) -> dict[str, np.ndarray]:
    d = np.asarray(distance, dtype=np.int32)
    return {
        "endpoint": d == 0,
        "1_4": (d >= 1) & (d <= 4),
        "5_8": (d >= 5) & (d <= 8),
        "gt8": d > 8,
        "disconnected": d < 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = V60TrainCache(args.cache_manifest)
    graph = _load_graph(Path(args.graph))
    prepared = prepare_static_v60(graph, "cpu")
    d2 = cache.names("D2")
    fit, holdout = deterministic_rainfall_split_v60(cache, names=d2, holdout_fraction=0.20)
    # ``holdout`` is retained only for metadata counts; no arrays/outcomes are read.
    scales = derive_effect_scales_v111(cache, fit, prepared)
    fine = HydraulicHorizonV110()
    fine_idx = np.asarray(fine.indices(), dtype=np.int64)
    full = MultiResolutionHorizonV60()
    full_idx = np.asarray(full.indices(), dtype=np.int64)
    full_minutes = (full_idx + 1) * 5.0
    buckets = {
        "0_30": full_minutes <= 30.0,
        "30_120": (full_minutes > 30.0) & (full_minutes <= 120.0),
        "120_360": full_minutes > 120.0,
    }
    max_depth = prepared.max_depth_m.detach().cpu().numpy().astype(np.float64)
    load_values = np.asarray([_phase_load(cache.entry(name), max_depth) for name in fit], dtype=np.float64)
    q1, q2 = (float(np.quantile(load_values, q)) for q in (1 / 3, 2 / 3))
    phases = {name: ("low" if load <= q1 else "mid" if load <= q2 else "high") for name, load in zip(fit, load_values)}
    types = _type_labels(graph)
    hops = all_actuator_endpoint_hops_v112(graph)
    flow_distance = np.minimum(
        hops[:, graph.actuator_upstream], hops[:, graph.actuator_downstream]
    )
    storage_mask = prepared.storage_mask.detach().cpu().numpy().astype(bool)
    action_magnitudes: dict[str, float] = {}
    for name in fit:
        entry = cache.entry(name); ref = entry.reference_index
        ref_settings = np.asarray(entry.arrays["settings"][ref], dtype=np.float64)
        for candidate in (i for i in entry.indices if i != ref):
            action_magnitudes[f"{name}::{candidate}"] = float(
                np.max(np.abs(np.asarray(entry.arrays["settings"][candidate], dtype=np.float64) - ref_settings))
            )
    magnitude_values = np.asarray(list(action_magnitudes.values()), dtype=np.float64)
    aq1, aq2 = (float(np.quantile(magnitude_values, q)) for q in (1 / 3, 2 / 3))

    state_tables: dict[str, dict[str, _Stats]] = {
        key: {} for key in ("overall", "by_type", "by_phase", "by_horizon", "by_distance", "by_effectiveness", "by_action_magnitude", "by_actuator")
    }
    flow_tables = {key: {} for key in state_tables}
    event_state: dict[str, dict[str, _Stats]] = {}
    event_flow: dict[str, dict[str, _Stats]] = {}
    probes = 0

    for name in fit:
        entry = cache.entry(name); arrays = entry.arrays; ref = entry.reference_index
        ref_state = np.asarray(arrays["target_states"][ref], dtype=np.float64)[full_idx][:, :, list(STATE_CHANNELS)]
        ref_flow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)[full_idx]
        source = None
        for candidate in (i for i in entry.indices if i != ref):
            candidate_state = np.asarray(arrays["target_states"][candidate], dtype=np.float64)[full_idx][:, :, list(STATE_CHANNELS)]
            candidate_flow = np.asarray(arrays["target_actuator_flows"][candidate], dtype=np.float64)[full_idx]
            delta_state = candidate_state - ref_state
            delta_flow = candidate_flow - ref_flow
            settings_delta = np.asarray(arrays["settings"][candidate], dtype=np.float64) - np.asarray(arrays["settings"][ref], dtype=np.float64)
            changed = np.flatnonzero(np.any(np.abs(settings_delta) > 1e-7, axis=0))
            if changed.size != 1:
                raise ValueError(f"{name}: expected one changed D2 actuator, got {changed.size}")
            source = int(changed[0])
            flow_active = np.abs(delta_flow) >= scales.flow_active_threshold[None]
            effective = bool(flow_active[:, source].any())
            mag = float(np.max(np.abs(settings_delta)))
            magnitude_label = "small" if mag < aq1 else "medium" if mag < aq2 else "large"
            phase = phases[name]
            type_label = types[source]
            phase_label = phase
            event = str(entry.event_id)
            event_state.setdefault(event, {})
            event_flow.setdefault(event, {})
            state_active = np.abs(delta_state) >= scales.state_active_threshold[None]
            flow_active = np.abs(delta_flow) >= scales.flow_active_threshold[None]

            for ci, channel in enumerate(STATE_NAMES):
                domain = storage_mask if channel == "storage_volume_m3" else np.ones(len(graph.node_ids), dtype=bool)
                distance = hops[source, domain]
                distance_masks = _distance_masks(distance)
                values = delta_state[:, domain, ci]
                active = state_active[:, domain, ci]
                _add(state_tables["overall"], channel, values, active)
                _add(state_tables["by_type"], f"{type_label}::{channel}", values, active)
                _add(state_tables["by_phase"], f"{phase_label}::{channel}", values, active)
                _add(state_tables["by_effectiveness"], f"{'effective' if effective else 'ineffective'}::{channel}", values, active)
                _add(state_tables["by_action_magnitude"], f"{magnitude_label}::{channel}", values, active)
                _add(state_tables["by_actuator"], f"{graph.actuator_ids[source]}::{channel}", values, active)
                for horizon_name, time_mask in buckets.items():
                    _add(state_tables["by_horizon"], f"{horizon_name}::{channel}", values[time_mask], active[time_mask])
                for distance_name, node_mask in distance_masks.items():
                    _add(state_tables["by_distance"], f"{distance_name}::{channel}", values[:, node_mask], active[:, node_mask])
                event_state[event].setdefault(channel, _Stats()).add(values, active)
                for horizon_name, time_mask in buckets.items():
                    event_state[event].setdefault(f"{horizon_name}::{channel}", _Stats()).add(values[time_mask], active[time_mask])

            flow_values = delta_flow
            flow_active = np.abs(flow_values) >= scales.flow_active_threshold[None]
            _add(flow_tables["overall"], FLOW_NAME, flow_values, flow_active)
            _add(flow_tables["by_type"], f"{type_label}::{FLOW_NAME}", flow_values, flow_active)
            _add(flow_tables["by_phase"], f"{phase_label}::{FLOW_NAME}", flow_values, flow_active)
            _add(flow_tables["by_effectiveness"], f"{'effective' if effective else 'ineffective'}::{FLOW_NAME}", flow_values, flow_active)
            _add(flow_tables["by_action_magnitude"], f"{magnitude_label}::{FLOW_NAME}", flow_values, flow_active)
            _add(flow_tables["by_actuator"], f"{graph.actuator_ids[source]}::{FLOW_NAME}", flow_values[:, source], flow_active[:, source])
            for horizon_name, time_mask in buckets.items():
                _add(flow_tables["by_horizon"], f"{horizon_name}::{FLOW_NAME}", flow_values[time_mask], flow_active[time_mask])
            for distance_name, actuator_mask in _distance_masks(flow_distance[source]).items():
                _add(flow_tables["by_distance"], f"{distance_name}::{FLOW_NAME}", flow_values[:, actuator_mask], flow_active[:, actuator_mask])
            event_flow[event].setdefault(FLOW_NAME, _Stats()).add(flow_values, flow_active)
            for horizon_name, time_mask in buckets.items():
                event_flow[event].setdefault(f"{horizon_name}::{FLOW_NAME}", _Stats()).add(flow_values[time_mask], flow_active[time_mask])
            probes += 1
        if probes == 24 or probes % 480 == 0 or name == fit[-1]:
            print(f"[V113_MAGNITUDE] groups_done={fit.index(name)+1}/{len(fit)} probes={probes}", flush=True)

    def final_nested(tables: dict[str, dict[str, _Stats]]) -> dict[str, Any]:
        return {dimension: _finalize_table(table) for dimension, table in tables.items()}

    def event_balanced(events: dict[str, dict[str, _Stats]]) -> dict[str, Any]:
        keys = sorted({key for values in events.values() for key in values})
        output: dict[str, Any] = {}
        for key in keys:
            records = [events[event][key].finalize() for event in events if key in events[event] and events[event][key].count]
            if not records:
                continue
            output[key] = {
                metric: float(np.mean([record[metric] for record in records if record[metric] is not None]))
                if any(record[metric] is not None for record in records) else None
                for metric in ("signed_mean", "positive_active_fraction", "negative_active_fraction", "active_fraction", "effect_rms", "p50_abs", "p90_abs", "p95_abs", "max_abs")
            }
            output[key]["event_count"] = len(records)
        return output

    report = {
        "contract": "PROJECT7_STEP2_V113_SIGNED_MAGNITUDE_AUDIT_V1",
        "git_head": _git_head(),
        "development_only": True,
        "new_swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "internal_holdout_outcomes_accessed": False,
        "data": {"d2_groups": len(d2), "trainfit_groups": len(fit), "trainfit_events": len({cache.entry(n).event_id for n in fit}), "candidate_probes": probes, "holdout_groups_metadata_only": len(holdout), "actuator_count": len(graph.actuator_ids), "node_count": len(graph.node_ids)},
        "units": {"depth": "m", "flooding_rate": "m3/s", "storage_volume": "m3", "inflow": "m3/s", "outflow": "m3/s", "managed_flow": "m3/s", "action_setting": "normalized setting"},
        "target_definition": "candidate - reference; signed and unclipped",
        "threshold_contract": "V111 P90 x 0.25 with physical floors, derived from all 112 TrainFit D2 groups",
        "phase_contract": {"metric": "causal_checkpoint_p90_depth_over_max_depth", "boundaries": [q1, q2], "groups": {label: int(sum(value == label for value in phases.values())) for label in ("low", "mid", "high")}},
        "action_magnitude_contract": {"metric": "max_abs_setting_delta", "q33": aq1, "q67": aq2, "bins": {"small": "<q33", "medium": ">=q33 and <q67", "large": ">=q67"}},
        "overall": {"state": final_nested(state_tables), "managed_flow": final_nested(flow_tables)},
        "event_balanced": {"state": event_balanced(event_state), "managed_flow": event_balanced(event_flow)},
        "lineage": {"cache_manifest_sha256": _sha256(args.cache_manifest), "graph_sha256": _sha256(args.graph), "fit_group_digest": hashlib.sha256("\n".join(fit).encode()).hexdigest()},
        "quantile_note": "p50-p99 use deterministic bounded samples (cap=2048) per stratum; counts, means, RMS and active/sign fractions are exact.",
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "trainfit_groups": len(fit), "candidate_probes": probes}, indent=2))


if __name__ == "__main__":
    main()
