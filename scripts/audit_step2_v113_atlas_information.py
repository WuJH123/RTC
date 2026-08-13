"""Test whether the frozen V112 support prior contains signed-magnitude information."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v110_contract import HydraulicHorizonV110
from rtc.step2_v113_audit import _auprc, _auroc, _rankdata, atlas_binary_metrics_v113
from run_step2_v110 import _load_graph

STATE_NAMES = ("depth_m", "flooding_rate_m3s", "storage_volume_m3", "inflow_m3s", "outflow_m3s")
STATE_CHANNELS = (0, 2, 3, 4, 5)


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


def _phase_load(entry, max_depth: np.ndarray) -> float:
    state = np.asarray(entry.arrays["initial_state"][entry.reference_index], dtype=np.float64)
    valid = np.isfinite(state[:, 0]) & np.isfinite(max_depth) & (max_depth > 1e-6)
    return float(np.quantile(state[valid, 0] / max_depth[valid], 0.90))


class _Info:
    def __init__(self, cap: int = 100_000) -> None:
        self.cap = int(cap)
        self.count = 0
        self.active_count = 0
        self.magnitude_sum = 0.0
        self.thresholds = {float(t): {"predicted": 0, "overlap": 0, "mass": 0.0} for t in (0.1, 0.25, 0.5)}
        self.sample_support = np.empty(0, dtype=np.float64)
        self.sample_magnitude = np.empty(0, dtype=np.float64)
        self.sample_active = np.empty(0, dtype=bool)
        self.bin_count = np.zeros(10, dtype=np.int64)
        self.bin_active = np.zeros(10, dtype=np.int64)
        self.bin_mass = np.zeros(10, dtype=np.float64)

    def add(self, support: np.ndarray, magnitude: np.ndarray, active: np.ndarray) -> None:
        s = np.asarray(support, dtype=np.float64).reshape(-1)
        m = np.asarray(magnitude, dtype=np.float64).reshape(-1)
        a = np.asarray(active, dtype=bool).reshape(-1)
        if s.shape != m.shape or s.shape != a.shape:
            raise ValueError("V113 Atlas information arrays are misaligned")
        finite = np.isfinite(s) & np.isfinite(m)
        s, m, a = np.clip(s[finite], 0.0, 1.0), np.abs(m[finite]), a[finite]
        if s.size == 0:
            return
        self.count += int(s.size)
        self.active_count += int(a.sum())
        self.magnitude_sum += float(m.sum())
        for threshold, values in self.thresholds.items():
            predicted = s >= threshold
            values["predicted"] += int(predicted.sum())
            values["overlap"] += int((predicted & a).sum())
            values["mass"] += float(m[predicted].sum())
        bins = np.minimum((s * 10.0).astype(np.int64), 9)
        for b in range(10):
            mask = bins == b
            self.bin_count[b] += int(mask.sum())
            self.bin_active[b] += int(a[mask].sum())
            self.bin_mass[b] += float(m[mask].sum())
        stride = max(1, int(np.ceil(s.size / self.cap)))
        sample = slice(None, None, stride)
        self.sample_support = np.concatenate((self.sample_support, s[sample][: self.cap]))
        self.sample_magnitude = np.concatenate((self.sample_magnitude, m[sample][: self.cap]))
        self.sample_active = np.concatenate((self.sample_active, a[sample][: self.cap]))
        if self.sample_support.size > self.cap:
            keep = max(1, int(np.ceil(self.sample_support.size / self.cap)))
            self.sample_support = self.sample_support[::keep][: self.cap]
            self.sample_magnitude = self.sample_magnitude[::keep][: self.cap]
            self.sample_active = self.sample_active[::keep][: self.cap]

    def finalize(self) -> dict[str, Any]:
        sample = self.sample_support
        magnitude = self.sample_magnitude
        active = self.sample_active
        total_mass = self.magnitude_sum
        corr = float(np.corrcoef(_rankdata(sample), _rankdata(magnitude))[0, 1]) if sample.size > 1 else float("nan")
        auroc = _auroc(sample, active) if sample.size else float("nan")
        auprc = _auprc(sample, active) if sample.size else float("nan")
        metrics: dict[str, Any] = {
            "sample_count": int(self.count),
            "true_active_fraction": float(self.active_count / self.count) if self.count else None,
            "spearman_support_abs_effect": corr if np.isfinite(corr) else None,
            "auroc": auroc if np.isfinite(auroc) else None,
            "auprc": auprc if np.isfinite(auprc) else None,
            "sample_cap": int(self.cap),
            "support_bins": [],
        }
        for threshold, values in self.thresholds.items():
            metrics[f"threshold_{threshold:g}"] = {
                "support_count": int(values["predicted"]),
                "active_recall": float(values["overlap"] / self.active_count) if self.active_count else None,
                "active_precision": float(values["overlap"] / values["predicted"]) if values["predicted"] else None,
                "effect_mass_recall": float(values["mass"] / total_mass) if total_mass > 0 else None,
            }
        for b in range(10):
            metrics["support_bins"].append({
                "lo": b / 10.0, "hi": (b + 1) / 10.0,
                "count": int(self.bin_count[b]),
                "mean_abs_effect": float(self.bin_mass[b] / self.bin_count[b]) if self.bin_count[b] else None,
                "active_fraction": float(self.bin_active[b] / self.bin_count[b]) if self.bin_count[b] else None,
            })
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = V60TrainCache(args.cache_manifest)
    graph = _load_graph(Path(args.graph))
    prepared = prepare_static_v60(graph, "cpu")
    atlas = np.load(args.atlas, allow_pickle=False)
    fit, holdout = deterministic_rainfall_split_v60(cache, names=cache.names("D2"), holdout_fraction=0.20)
    scales = derive_effect_scales_v111(cache, fit, prepared)
    horizon = HydraulicHorizonV110()
    idx = np.asarray(horizon.indices(), dtype=np.int64)
    minutes = (idx + 1) * 5.0
    max_depth = prepared.max_depth_m.detach().cpu().numpy().astype(np.float64)
    load_values = np.asarray([_phase_load(cache.entry(name), max_depth) for name in fit], dtype=np.float64)
    q1, q2 = (float(np.quantile(load_values, q)) for q in (1 / 3, 2 / 3))
    phase_by_group = {name: ("low" if value <= q1 else "mid" if value <= q2 else "high") for name, value in zip(fit, load_values)}
    accum: dict[str, dict[str, _Info]] = {"overall": {}, "phase_conditioned": {}}
    probes = 0
    for name in fit:
        entry = cache.entry(name); arrays = entry.arrays; ref = entry.reference_index
        ref_state = np.asarray(arrays["target_states"][ref], dtype=np.float64)[idx][:, :, list(STATE_CHANNELS)]
        ref_flow = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float64)[idx]
        source_support_state = atlas["overall__state_support_probability"]
        source_support_flow = atlas["overall__flow_support_probability"]
        phase = phase_by_group[name]
        phase_support_state = atlas[f"{phase}__state_support_probability"]
        phase_support_flow = atlas[f"{phase}__flow_support_probability"]
        for candidate in (i for i in entry.indices if i != ref):
            settings_delta = np.asarray(arrays["settings"][candidate], dtype=np.float64) - np.asarray(arrays["settings"][ref], dtype=np.float64)
            changed = np.flatnonzero(np.any(np.abs(settings_delta) > 1e-7, axis=0))
            if changed.size != 1:
                raise ValueError(f"{name}: expected one changed D2 actuator")
            source = int(changed[0])
            delta_state = np.asarray(arrays["target_states"][candidate], dtype=np.float64)[idx][:, :, list(STATE_CHANNELS)] - ref_state
            delta_flow = np.asarray(arrays["target_actuator_flows"][candidate], dtype=np.float64)[idx] - ref_flow
            state_active = np.abs(delta_state) >= scales.state_active_threshold[None]
            flow_active = np.abs(delta_flow) >= scales.flow_active_threshold[None]
            for ci, channel in enumerate(STATE_NAMES):
                support_a = source_support_state[source, :, :, ci]
                phase_a = phase_support_state[source, :, :, ci]
                _domain = prepared.storage_mask.detach().cpu().numpy().astype(bool) if channel == "storage_volume_m3" else np.ones(len(graph.node_ids), dtype=bool)
                accum["overall"].setdefault(channel, _Info()).add(support_a[:, _domain], np.abs(delta_state[:, _domain, ci]), state_active[:, _domain, ci])
                accum["phase_conditioned"].setdefault(channel, _Info()).add(phase_a[:, _domain], np.abs(delta_state[:, _domain, ci]), state_active[:, _domain, ci])
            accum["overall"].setdefault("managed_flow_m3s", _Info()).add(source_support_flow[source], np.abs(delta_flow), flow_active)
            accum["phase_conditioned"].setdefault("managed_flow_m3s", _Info()).add(phase_support_flow[source], np.abs(delta_flow), flow_active)
            probes += 1
        if probes == 24 or probes % 480 == 0 or name == fit[-1]:
            print(f"[V113_ATLAS_INFO] groups_done={fit.index(name)+1}/{len(fit)} probes={probes}", flush=True)

    overall = {key: value.finalize() for key, value in accum["overall"].items()}
    phase = {key: value.finalize() for key, value in accum["phase_conditioned"].items()}
    comparison = {}
    for key in sorted(overall):
        comparison[key] = {}
        for metric in ("spearman_support_abs_effect", "auroc", "auprc"):
            left, right = phase[key].get(metric), overall[key].get(metric)
            comparison[key][metric] = float(left - right) if left is not None and right is not None and np.isfinite(left) and np.isfinite(right) else None
        for threshold in ("threshold_0.1", "threshold_0.25", "threshold_0.5"):
            comparison[key][threshold] = {}
            for metric in ("active_recall", "active_precision", "effect_mass_recall"):
                left, right = phase[key][threshold].get(metric), overall[key][threshold].get(metric)
                comparison[key][threshold][metric + "_delta"] = float(left - right) if left is not None and right is not None and np.isfinite(left) and np.isfinite(right) else None
    finite_deltas = [value["auprc"] for value in comparison.values() if value["auprc"] is not None]
    report = {
        "contract": "PROJECT7_STEP2_V113_ATLAS_INFORMATION_VALUE_V1",
        "git_head": _git_head(),
        "development_only": True,
        "new_swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "internal_holdout_outcomes_accessed": False,
        "fit_groups": len(fit),
        "fit_events": len({cache.entry(name).event_id for name in fit}),
        "holdout_groups_metadata_only": len(holdout),
        "candidate_probes": probes,
        "support_source": "frozen V112 TrainFit D2 prior; continuous probability, no hard mask",
        "phase_contract": {"metric": "causal_checkpoint_p90_depth_over_max_depth", "boundaries": [q1, q2]},
        "overall_prior": overall,
        "phase_conditioned_prior": phase,
        "phase_minus_overall": comparison,
        "does_phase_conditioned_atlas_add_information": bool(finite_deltas and float(np.mean(finite_deltas)) > 0.0),
        "interpretation": "in-sample TrainFit information audit only; not a generalization or acceptance result",
        "lineage": {"cache_manifest_sha256": _sha256(args.cache_manifest), "graph_sha256": _sha256(args.graph), "atlas_sha256": _sha256(args.atlas)},
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "candidate_probes": probes, "phase_adds_information": report["does_phase_conditioned_atlas_add_information"]}, indent=2))


if __name__ == "__main__":
    main()
