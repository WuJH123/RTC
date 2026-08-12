"""Read-only Train-only audit of the D2 additive superposition burden.

This script consumes the existing Train cache and the frozen V4.2-compatible
backbone.  It never launches SWMM, writes no model checkpoint, and does not
read Validation or Final outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtc.production_cli import _load_graph  # noqa: E402
from rtc.step2_control_response_v41 import DifferentiableCounterfactualResponseModelV41  # noqa: E402
from rtc.step2_train_response_v4 import build_full_train_normalization_from_checkpoint, load_train_groups  # noqa: E402
from rtc.step2_train_response_v41 import (  # noqa: E402
    CounterfactualDeltaScalesV41,
    group_metrics_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)


ROOT = Path(r"E:\RTC_sewer\Project7")
STUDY = ROOT / "study_v069"
OUT = STUDY / "step2_direct_multiactuator_v45"
FULL_TRAIN_CACHE = STUDY / "step2_counterfactual_stability_v2" / "step2_training_cache_v1"
MICRO_CACHE = STUDY / "step2_counterfactual_stability_v2" / "02_micro_smoke" / "cache"
GRAPH = STUDY / "formal_assets" / "graph_schema.npz"
V3_CHECKPOINT = STUDY / "step2_multishooting_v3" / "01_micro" / "model" / "step2_multishooting_v3_micro.pt"
SCALE_JSON = STUDY / "step2_response_calibration_v41" / "00_scales" / "counterfactual_delta_scales_train18.json"
OLD_NORMALIZATION_SCALE_JSON = STUDY / "step2_counterfactual_stability_v2" / "00_scales" / "train_only_delta_scales.json"
BEST_D2 = STUDY / "step2_d3_magnitude_calibration_v42" / "04_12_group_micro" / "v42_12_group_micro.pt"
DOCS = ROOT / "repo" / "docs"
EXPECTED_COHORT_SHA = "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"
EXPECTED_BEST_D2_SHA = "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(graph: Any, normalization: Any, scales: CounterfactualDeltaScalesV41) -> DifferentiableCounterfactualResponseModelV41:
    return DifferentiableCounterfactualResponseModelV41(
        state_dim=int(normalization.state_mean.size),
        rainfall_dim=int(normalization.rain_mean.size),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_physics_dim=int(graph.actuator_physics.shape[1]),
        hidden_dim=64,
        actuator_count=len(graph.actuator_ids),
        actuator_embedding_dim=16,
        temporal_embedding_dim=12,
        state_mean=torch.as_tensor(normalization.state_mean),
        state_std=torch.as_tensor(normalization.state_std),
        flow_std=torch.as_tensor(normalization.flow_std),
        d2_state_scale=torch.as_tensor(scales.by_source["D2"].state_scale),
        d3_state_scale=torch.as_tensor(scales.by_source["D3"].state_scale),
        d2_flow_scale=torch.as_tensor(scales.by_source["D2"].flow_scale),
        d3_flow_scale=torch.as_tensor(scales.by_source["D3"].flow_scale),
        d2_tfv_scale=scales.by_source["D2"].tfv_scale_m3,
        d3_tfv_scale=scales.by_source["D3"].tfv_scale_m3,
        max_horizon_steps=72,
        effect_rank=12,
        interaction_magnitude_features_enabled=True,
    )


def _load_best(model: torch.nn.Module) -> None:
    if _sha256(BEST_D2) != EXPECTED_BEST_D2_SHA:
        raise RuntimeError("best V4.2 D2 backbone SHA mismatch; fail closed")
    payload = torch.load(BEST_D2, map_location="cpu", weights_only=False)
    incoming = payload.get("model_state_dict", payload)
    current = model.state_dict()
    compatible = {
        name: value for name, value in incoming.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected or missing:
        raise RuntimeError(f"best backbone mismatch: {missing=} {unexpected=}")


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {name: float("nan") for name in ("median", "p75", "p90", "p95", "max")}
    median, p75, p90, p95 = np.percentile(values, [50, 75, 90, 95])
    return {
        "median": float(median),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "max": float(np.max(values)),
    }


def _burden(rows: list[dict[str, Any]], meaningful_scale: float) -> dict[str, Any]:
    required = np.asarray([row["required_interaction_m3"] for row in rows], dtype=np.float64)
    true = np.asarray([row["true_total_m3"] for row in rows], dtype=np.float64)
    additive = np.asarray([row["d2_additive_m3"] for row in rows], dtype=np.float64)
    denominator_scale = max(float(meaningful_scale), 1e-6)
    r_true = np.abs(required) / np.maximum(np.abs(true), denominator_scale)
    r_additive = np.abs(required) / np.maximum(np.abs(additive), denominator_scale)
    r_joint = np.abs(required) / np.maximum(np.abs(true) + np.abs(additive), denominator_scale)
    valid_sign = (np.abs(true) > 1e-9) & (np.abs(additive) > 1e-9)
    sign_reversal = valid_sign & (np.sign(true) != np.sign(additive))
    severe = sign_reversal & (np.abs(additive) > np.abs(true))
    return {
        "count": int(required.size),
        "meaningful_scale_m3": float(meaningful_scale),
        "R_true": _percentiles(r_true),
        "R_additive": _percentiles(r_additive),
        "R_joint": _percentiles(r_joint),
        "fraction_required_gt_true": float(np.mean(np.abs(required) > np.abs(true))) if required.size else float("nan"),
        "fraction_required_gt_additive": float(np.mean(np.abs(required) > np.abs(additive))) if required.size else float("nan"),
        "fraction_required_gt_half_true": float(np.mean(np.abs(required) > 0.5 * np.abs(true))) if required.size else float("nan"),
        "sign_reversal_fraction": float(np.mean(sign_reversal)) if required.size else float("nan"),
        "severe_cancellation_fraction": float(np.mean(severe)) if required.size else float("nan"),
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"groups": 0}
    return {
        "groups": len(rows),
        "rank": float(np.nanmean([row["rank"] for row in rows])),
        "pairwise": float(np.nanmean([row["pairwise"] for row in rows])),
        "top1": int(sum(bool(row["top1"]) for row in rows)),
        "mean_regret_m3": float(np.mean([row["regret_m3"] for row in rows])),
        "max_regret_m3": float(np.max([row["regret_m3"] for row in rows])),
    }


def run(device_name: str = "cuda") -> dict[str, Any]:
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(str(GRAPH))
    normalization = build_full_train_normalization_from_checkpoint(V3_CHECKPOINT, OLD_NORMALIZATION_SCALE_JSON)
    scales = CounterfactualDeltaScalesV41.from_json_dict(json.loads(SCALE_JSON.read_text(encoding="utf-8")))
    groups = load_train_groups(FULL_TRAIN_CACHE, normalization)
    d3_group_names = sorted(name for name, pairs in groups.items() if pairs and pairs[0].source_kind.upper() == "D3")
    cohort_digest = __import__("hashlib").sha256("\n".join(d3_group_names).encode("utf-8")).hexdigest()
    if cohort_digest == "":
        raise RuntimeError("empty D3 Train-only cohort")
    model = _model(graph, normalization, scales).to(device).float().eval()
    _load_best(model)
    prepared = prepare_graph_v41(model, graph, normalization, device)
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for group in d3_group_names:
            batch = __import__("rtc.step2_train_response_v41", fromlist=["stack_response_group_v41"]).stack_response_group_v41(groups[group], device)
            output = model.forward_group(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
                batch.elapsed_seconds,
                source_kind="D3",
            )
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy()
            additive = output.direct_single_delta_tfv_m3[0].detach().cpu().numpy()
            metric = group_metrics_v41(predicted=additive, truth=truth, group=group, source_kind="D3")
            metric_rows.append(metric)
            for index, (true, add) in enumerate(zip(truth, additive, strict=True)):
                candidate_rows.append({
                    "group": group,
                    "candidate_index": int(index),
                    "true_total_m3": float(true),
                    "d2_additive_m3": float(add),
                    "required_interaction_m3": float(true - add),
                })
    q33 = float(scales.by_source["D3"].tfv_abs_quantiles_m3["q33"])
    q67 = float(scales.by_source["D3"].tfv_abs_quantiles_m3["q67"])
    strata: dict[str, list[dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    for row in candidate_rows:
        absolute = abs(row["true_total_m3"])
        name = "small" if absolute < q33 else "medium" if absolute < q67 else "large"
        strata[name].append(row)
    medium_large = strata["medium"] + strata["large"]
    burden = _burden(medium_large, scales.by_source["D3"].tfv_scale_m3)
    additive_metrics = _aggregate_metrics(metric_rows)
    sign_reversal = sum(
        abs(row["true_total_m3"]) > 1e-9
        and abs(row["d2_additive_m3"]) > 1e-9
        and np.sign(row["true_total_m3"]) != np.sign(row["d2_additive_m3"])
        for row in medium_large
    ) / max(1, len(medium_large))
    superposition_invalid = bool(
        burden["R_true"]["median"] >= 0.5
        and (sign_reversal >= 0.1 or additive_metrics["rank"] < 0.5)
    )
    payload = {
        "contract": "STEP2_ADDITIVE_SUPERPOSITION_AUDIT_V45",
        "boundary": {
            "swmm_launched": False,
            "d2_regenerated": False,
            "d3_regenerated": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_run": False,
            "full_train_smoke_run": False,
            "production_wiring_modified": False,
        },
        "data": {
            "cache": str(FULL_TRAIN_CACHE),
            "micro_cache": str(MICRO_CACHE),
            "source_manifest_sha256": normalization.source_manifest_sha256,
            "d3_group_count": len(d3_group_names),
            "d3_candidate_count": len(candidate_rows),
            "d3_group_name_sha256": cohort_digest,
            "expected_micro_cohort_sha256": EXPECTED_COHORT_SHA,
            "meaningful_scale_m3": float(scales.by_source["D3"].tfv_scale_m3),
            "q33_m3": q33,
            "q67_m3": q67,
            "best_d2_checkpoint": str(BEST_D2),
            "best_d2_sha256": _sha256(BEST_D2),
        },
        "definition": {
            "true_total": "authoritative SWMM delta TFV",
            "d2_additive": "frozen D2 sum-of-single predicted delta TFV",
            "required_interaction": "true_total - d2_additive",
            "R_true": "abs(required_interaction)/max(abs(true_total), meaningful_scale)",
            "R_additive": "abs(required_interaction)/max(abs(d2_additive), meaningful_scale)",
            "R_joint": "abs(required_interaction)/max(abs(true_total)+abs(d2_additive), meaningful_scale)",
        },
        "overall_additive_candidate_metrics": additive_metrics,
        "medium_large_burden": burden,
        "strata": {name: {"count": len(rows), **_burden(rows, scales.by_source["D3"].tfv_scale_m3)} for name, rows in strata.items()},
        "candidate_rows": candidate_rows,
        "verdict": "SUPERPOSITION INVALID" if superposition_invalid else "AUDIT DOES NOT SUPPORT INVALIDITY",
        "direct_v45_authorized": superposition_invalid,
        "device": str(device),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STEP2_ADDITIVE_SUPERPOSITION_AUDIT_V45.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (DOCS / "STEP2_ADDITIVE_SUPERPOSITION_AUDIT_V45.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    markdown = "# STEP2 ADDITIVE SUPERPOSITION AUDIT V4.5\n\n```json\n" + json.dumps(payload, indent=2, allow_nan=True) + "\n```\n"
    (OUT / "STEP2_ADDITIVE_SUPERPOSITION_AUDIT_V45.md").write_text(markdown, encoding="utf-8")
    (DOCS / "STEP2_ADDITIVE_SUPERPOSITION_AUDIT_V45.md").write_text(markdown, encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2, allow_nan=True))
