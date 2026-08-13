"""Promote a passed development V120 bundle without retraining or retuning.

Promotion is byte-new but state-preserving: model weights and normalization are
unchanged.  It is allowed only after the exact bundle has produced passed
execution/continuity evidence under the frozen V120 controller config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .production_cli import _load_graph
from .step2_causal_forecast_v120 import V120_CAUSAL_RAINFALL_CONTRACT
from .step2_v120_contract import (
    V120_BUNDLE_CONTRACT,
    V120_CANDIDATE_POLICY_CONTRACT,
    V120_CONTRACT,
    v120_graph_semantic_sha256,
    v120_runtime_contract_sha256,
)

PROMOTION_CONTRACT = "PROJECT7_V120_EVIDENCE_ONLY_PRODUCTION_PROMOTION_V1"
RUNTIME_ACCEPTANCE_CONTRACT = "DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V2_TEMPORAL_CONTINUITY"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _verify_bundle(bundle_path: str | Path, graph_path: str | Path) -> dict[str, Any]:
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("V120 bundle must be a dictionary")
    if payload.get("bundle_contract") != V120_BUNDLE_CONTRACT:
        raise ValueError("promotion requires the execution-bound V120 bundle")
    if payload.get("step2_contract") != V120_CONTRACT:
        raise ValueError("V120 scientific contract mismatch")
    if payload.get("runtime_compatible") is not True:
        raise ValueError("V120 value gate did not authorize runtime development")
    if payload.get("production_compatible") is True:
        raise ValueError("V120 bundle is already production-compatible")
    gate = payload.get("value_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("V120 control-value gate did not pass")
    if str(payload.get("v120_runtime_contract_sha256", "")) != v120_runtime_contract_sha256():
        raise ValueError("V120 bundle implementation differs from current production code")

    graph = _load_graph(graph_path)
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V120 bundle lacks lineage")
    if str(lineage.get("graph_sha256", "")) != _sha(graph_path):
        raise ValueError("V120 bundle graph file differs from promotion graph")
    if str(lineage.get("graph_semantic_sha256", "")) != v120_graph_semantic_sha256(graph):
        raise ValueError("V120 bundle graph semantics differ from promotion graph")

    rainfall = payload.get("rainfall_input")
    if not isinstance(rainfall, dict) or rainfall.get("contract") != V120_CAUSAL_RAINFALL_CONTRACT:
        raise ValueError("V120 promotion requires causal-rainfall training")
    if rainfall.get("future_realized_rainfall_used_as_model_input") is not False:
        raise ValueError("V120 promotion refuses future-rainfall leakage")
    if int(rainfall.get("required_runtime_history_steps_for_level", -1)) != 1:
        raise ValueError("V120 rainfall information budget is not runtime-aligned")

    split = payload.get("split")
    if not isinstance(split, dict):
        raise ValueError("V120 bundle lacks split evidence")
    if len(split.get("fit_event_ids", [])) != 14 or len(split.get("holdout_event_ids", [])) != 4:
        raise ValueError("V120 bundle lacks frozen 14/4 Train-only split")
    if int(split.get("validation_events_loaded", -1)) != 0 or int(split.get("final_events_loaded", -1)) != 0:
        raise ValueError("V120 bundle accessed Validation or Final labels")

    census = payload.get("data_census")
    if not isinstance(census, dict):
        raise ValueError("V120 bundle lacks data census")
    if int(census.get("source_d2_authoritative_branch_census", -1)) != 4800:
        raise ValueError("V120 bundle does not record D2=4800 source census")
    if int(census.get("targeted_d3_branches", -1)) != 3600:
        raise ValueError("V120 bundle lacks complete D3=3600 Train view")
    if int(census.get("eligible_event_count", -1)) != 18:
        raise ValueError("V120 bundle is not frozen-Train18 only")

    candidate = payload.get("candidate_policy")
    if not isinstance(candidate, dict) or candidate.get("contract") != V120_CANDIDATE_POLICY_CONTRACT:
        raise ValueError("V120 candidate-policy contract mismatch")
    if candidate.get("score_only_executable_sequences") is not True:
        raise ValueError("V120 bundle did not score executable sequences")
    return payload


def promote_v120(
    *,
    bundle_path: str | Path,
    graph_path: str | Path,
    controller_config_path: str | Path,
    runtime_acceptance_path: str | Path,
    development_run_index_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    bundle = _verify_bundle(bundle_path, graph_path)
    config = _json(controller_config_path)
    if config.get("v120_contract") != "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1":
        raise ValueError("promotion requires the frozen causal V120 controller config")
    forecast = config.get("forecast")
    if not isinstance(forecast, dict):
        raise ValueError("V120 controller lacks forecast contract")
    if int(forecast.get("history_steps_for_level", -1)) != 1 or abs(float(forecast.get("decay_per_step", -1)) - 0.92) > 1e-12:
        raise ValueError("V120 runtime rainfall contract differs from training")

    acceptance = _json(runtime_acceptance_path)
    if acceptance.get("contract") != RUNTIME_ACCEPTANCE_CONTRACT or acceptance.get("passed") is not True:
        raise ValueError("V120 promotion requires passed runtime/continuity acceptance")
    if str(acceptance.get("controller_config_sha256", "")) != _sha(controller_config_path):
        raise ValueError("runtime acceptance used a different V120 controller config")
    metrics = acceptance.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("runtime acceptance lacks metrics")
    for key in (
        "control_grid_violations",
        "first_decision_violations",
        "missing_runtime_diagnostics",
        "continuity_evidence_missing",
        "cross_decision_continuity_violations",
        "planned_horizon_continuity_violations",
        "fatal_runtime_fallbacks",
    ):
        if int(metrics.get(key, -1)) != 0:
            raise ValueError(f"V120 promotion refuses runtime evidence with {key}")

    index = pd.read_csv(development_run_index_path)
    if not {"strategy", "metadata_path"}.issubset(index.columns):
        raise ValueError("development run index lacks strategy/metadata_path")
    proposed = index[index["strategy"].astype(str) == "proposed"]
    if proposed.empty:
        raise ValueError("V120 promotion requires at least one Proposed development run")
    bundle_sha = _sha(bundle_path)
    config_sha = _sha(controller_config_path)
    for _, row in proposed.iterrows():
        meta = _json(str(row["metadata_path"]))
        if meta.get("strategy") != "proposed":
            raise ValueError("development evidence is not stamped Proposed")
        if str(meta.get("step2_model_sha256", "")) != bundle_sha:
            raise ValueError("development run did not execute the exact V120 bundle being promoted")
        if str(meta.get("controller_config_sha256", "")) != config_sha:
            raise ValueError("development run used a different V120 controller config")

    promoted = dict(bundle)
    promoted["production_compatible"] = True
    promoted["promotion"] = {
        "contract": PROMOTION_CONTRACT,
        "source_bundle_sha256": bundle_sha,
        "runtime_acceptance_sha256": _sha(runtime_acceptance_path),
        "development_run_index_sha256": _sha(development_run_index_path),
        "controller_config_sha256": config_sha,
        "graph_sha256": _sha(graph_path),
        "v120_runtime_contract_sha256": v120_runtime_contract_sha256(),
        "model_parameters_changed": False,
        "normalization_changed": False,
        "thresholds_retuned": False,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(promoted, out)
    return {
        "contract": PROMOTION_CONTRACT,
        "production_compatible": True,
        "source_bundle_sha256": bundle_sha,
        "promoted_bundle_path": str(out.resolve()),
        "promoted_bundle_sha256": _sha(out),
        "model_parameters_changed": False,
        "thresholds_retuned": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote passed V120 development bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-acceptance", required=True)
    parser.add_argument("--development-run-index", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(promote_v120(
        bundle_path=args.bundle,
        graph_path=args.graph,
        controller_config_path=args.config,
        runtime_acceptance_path=args.runtime_acceptance,
        development_run_index_path=args.development_run_index,
        output_path=args.out,
    ), indent=2))


if __name__ == "__main__":
    main()
