"""Policy Lock for the simplified Project7 V120 research question.

Unlike legacy Policy Lock V6, this lock does not require a nodewise Hydraulic or
Hydraulic-gradient acceptance gate.  It locks exactly what the production claim
uses: Step1, the causal direct-Delta-TFV V120 bundle, its frozen joint-action
candidate policy, execution/continuity evidence, split lineage and baseline plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .code_contract import rtc_source_tree_sha256
from .control_lineage import section_payload_sha256
from .inp_lineage import physical_contract_sha256
from .project7_contract import validate_project7_runtime_config
from .step2_causal_forecast_v120 import V120_CAUSAL_RAINFALL_CONTRACT
from .step2_v120_contract import (
    V120_BUNDLE_CONTRACT,
    V120_CANDIDATE_POLICY_CONTRACT,
    V120_CONTRACT,
    v120_runtime_contract_sha256,
)

POLICY_LOCK_V120_CONTRACT = "WUHAN_RTC_V120_TFV_ONLY_POLICY_LOCK_V1"
SPLIT_CONTRACT = "PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1"
PROMOTION_CONTRACT = "PROJECT7_V120_EVIDENCE_ONLY_PRODUCTION_PROMOTION_V1"
RUNTIME_ACCEPTANCE_CONTRACT = "DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V2_TEMPORAL_CONTINUITY"
EXPECTED_STRATEGIES = (
    "proposed",
    "no_control",
    "internal_rtc",
    "auto_rbc",
    "efd",
    "all_open",
    "all_closed",
)
REQUIRED = {
    "inp_preflight",
    "frozen_inp",
    "priority_nodes",
    "sensor_layout",
    "graph_schema",
    "split_contract",
    "split_registry",
    "step1_model",
    "step1_acceptance",
    "step2_model",
    "controller_config",
    "baseline_plan",
    "runtime_acceptance",
}


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _lines(path: str | Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _verify_split(split_contract_path: str, split_registry_path: str) -> dict[str, Any]:
    contract = _json(split_contract_path)
    if contract.get("contract") != SPLIT_CONTRACT:
        raise ValueError("V120 Policy Lock requires frozen 18/6/6 split contract")
    train = {str(x) for x in contract.get("development_train", [])}
    validation = {str(x) for x in contract.get("development_validation", [])}
    final = {str(x) for x in contract.get("final", [])}
    if (len(train), len(validation), len(final)) != (18, 6, 6):
        raise ValueError("V120 Policy Lock split is not 18/6/6")
    if train & validation or train & final or validation & final:
        raise ValueError("V120 Policy Lock scientific splits overlap")

    registry = pd.read_csv(split_registry_path, keep_default_na=False)
    required = {"event_id", "scientific_split", "development_fold"}
    if not required.issubset(registry.columns):
        raise ValueError("V120 split registry lacks event/fold columns")
    event = registry["event_id"].astype(str)
    scientific = registry["scientific_split"].astype(str)
    fold = registry["development_fold"].astype(str)
    registry_train = set(event[(scientific == "development") & (fold == "train")])
    registry_validation = set(event[(scientific == "development") & (fold == "validation")])
    registry_final = set(event[scientific == "final"])
    if registry_train != train or registry_validation != validation or registry_final != final:
        raise ValueError("V120 split registry differs from frozen split contract")
    return {
        "contract": SPLIT_CONTRACT,
        "development_train": sorted(train),
        "development_validation": sorted(validation),
        "final": sorted(final),
    }


def _verify_step1(path: str, acceptance_path: str) -> tuple[dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Step1 checkpoint payload is invalid")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Step1 checkpoint lacks model_config")
    if int(config.get("model_step_seconds", -1)) != 300:
        raise ValueError("Step1 model step differs from V120")
    if int(config.get("history_steps", -1)) != 13:
        raise ValueError("Step1 history differs from V120")
    engine = str(config.get("swmm_engine_version", "")).strip()
    if not engine:
        raise ValueError("Step1 lacks SWMM engine lineage")
    acceptance = _json(acceptance_path)
    if acceptance.get("passed") is not True:
        raise ValueError("Step1 acceptance did not pass")
    if str(acceptance.get("model_sha256", "")) != _sha(path):
        raise ValueError("Step1 acceptance belongs to another checkpoint")
    return config, engine


def _verify_v120_bundle(path: str, graph_path: str, frozen_train: set[str]) -> tuple[dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("V120 bundle payload is invalid")
    if payload.get("bundle_contract") != V120_BUNDLE_CONTRACT or payload.get("step2_contract") != V120_CONTRACT:
        raise ValueError("Policy Lock requires execution-bound V120 bundle")
    if payload.get("runtime_compatible") is not True or payload.get("production_compatible") is not True:
        raise ValueError("Policy Lock requires promoted production-compatible V120")
    promotion = payload.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("contract") != PROMOTION_CONTRACT:
        raise ValueError("V120 bundle lacks evidence-only production promotion")
    if promotion.get("model_parameters_changed") is not False or promotion.get("thresholds_retuned") is not False:
        raise ValueError("V120 promotion modified trained policy semantics")
    if str(payload.get("v120_runtime_contract_sha256", "")) != v120_runtime_contract_sha256():
        raise ValueError("V120 implementation changed after training/promotion")

    lineage = payload.get("lineage")
    if not isinstance(lineage, dict) or str(lineage.get("graph_sha256", "")) != _sha(graph_path):
        raise ValueError("V120 graph lineage mismatch")
    split = payload.get("split")
    if not isinstance(split, dict):
        raise ValueError("V120 bundle lacks split evidence")
    trained = set(str(x) for x in split.get("fit_event_ids", [])) | set(
        str(x) for x in split.get("holdout_event_ids", [])
    )
    if trained != frozen_train:
        raise ValueError("V120 bundle was not trained/evaluated only on frozen Train18")
    if len(split.get("fit_event_ids", [])) != 14 or len(split.get("holdout_event_ids", [])) != 4:
        raise ValueError("V120 bundle does not preserve frozen 14/4 internal split")
    if int(split.get("validation_events_loaded", -1)) != 0 or int(split.get("final_events_loaded", -1)) != 0:
        raise ValueError("V120 bundle accessed Validation/Final labels")

    census = payload.get("data_census")
    if not isinstance(census, dict):
        raise ValueError("V120 bundle lacks data census")
    if int(census.get("source_d2_authoritative_branch_census", -1)) != 4800:
        raise ValueError("V120 bundle does not record D2 source census=4800")
    if int(census.get("targeted_d3_branches", -1)) != 3600:
        raise ValueError("V120 bundle lacks targeted D3=3600")
    if int(census.get("eligible_event_count", -1)) != 18:
        raise ValueError("V120 bundle training population is not Train18")

    rainfall = payload.get("rainfall_input")
    if not isinstance(rainfall, dict) or rainfall.get("contract") != V120_CAUSAL_RAINFALL_CONTRACT:
        raise ValueError("V120 bundle lacks causal rainfall input contract")
    if rainfall.get("future_realized_rainfall_used_as_model_input") is not False:
        raise ValueError("V120 bundle leaks future rainfall")
    if int(rainfall.get("required_runtime_history_steps_for_level", -1)) != 1:
        raise ValueError("V120 training/runtime rainfall information budgets differ")

    candidate = payload.get("candidate_policy")
    if not isinstance(candidate, dict) or candidate.get("contract") != V120_CANDIDATE_POLICY_CONTRACT:
        raise ValueError("V120 candidate-policy contract mismatch")
    if candidate.get("continuous_gradient_search") is not False or candidate.get("score_only_executable_sequences") is not True:
        raise ValueError("V120 candidate search is not the frozen execution-bound finite policy")
    gate = payload.get("value_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("V120 internal control-value gate did not pass")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("V120 bundle lacks model_config")
    return model_config, str(model_config.get("swmm_engine_version", "")).strip()


def create_policy_lock_v120(*, artefacts_path: str, output_path: str) -> dict[str, Any]:
    raw = _json(artefacts_path)
    artefacts = {str(key): str(value) for key, value in raw.items()}
    missing = sorted(REQUIRED - set(artefacts))
    if missing:
        raise ValueError(f"V120 Policy Lock missing artifacts: {missing}")
    for name, path in artefacts.items():
        if not Path(path).is_file():
            raise ValueError(f"V120 Policy Lock artifact missing: {name}: {path}")

    split = _verify_split(artefacts["split_contract"], artefacts["split_registry"])
    frozen_train = set(split["development_train"])
    preflight = _json(artefacts["inp_preflight"])
    if preflight.get("contract") != "LARGE_SWMM_INP_PREFLIGHT_V3_CAUSAL_RTC":
        raise ValueError("V120 Policy Lock requires causal large-INP preflight V3")
    no_control = preflight.get("no_control_contract")
    if not isinstance(no_control, dict) or no_control.get("id") != "NO_SUPERVISORY_RTC_V2":
        raise ValueError("V120 Policy Lock lacks passive No-control contract")

    priority = _lines(artefacts["priority_nodes"])
    sensors = _lines(artefacts["sensor_layout"])
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError("V120 reporting requires exactly eight priority nodes")
    if not sensors:
        raise ValueError("V120 sensor layout cannot be empty")
    with np.load(artefacts["graph_schema"], allow_pickle=False) as graph:
        node_ids = set(graph["node_ids"].astype(str).tolist())
    missing_nodes = sorted((set(priority) | set(sensors)) - node_ids)
    if missing_nodes:
        raise ValueError(f"V120 sensor/priority nodes absent from graph: {missing_nodes[:20]}")

    controller = _json(artefacts["controller_config"])
    if controller.get("v120_contract") != "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1":
        raise ValueError("V120 Policy Lock requires frozen causal V120 controller config")
    runtime_contract = validate_project7_runtime_config(controller)
    objective = controller.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("V120 controller lacks objective")
    if objective.get("priority_role") != "report_only" or objective.get("global_peak_role") != "report_only":
        raise ValueError("V120 priority/global peak must remain report-only")
    forecast = controller.get("forecast")
    if not isinstance(forecast, dict) or int(forecast.get("history_steps_for_level", -1)) != 1:
        raise ValueError("V120 online forecast must match causal training information budget")
    if abs(float(forecast.get("decay_per_step", -1)) - 0.92) > 1e-12:
        raise ValueError("V120 online rainfall decay differs from training")

    step1_config, step1_engine = _verify_step1(artefacts["step1_model"], artefacts["step1_acceptance"])
    step2_config, step2_engine = _verify_v120_bundle(
        artefacts["step2_model"], artefacts["graph_schema"], frozen_train
    )
    if not step2_engine or step2_engine != step1_engine:
        raise ValueError(f"Step1/V120 SWMM engine mismatch: {step1_engine} != {step2_engine}")
    if int(step2_config.get("state_dim", -1)) != 6:
        raise ValueError("V120 Value input state dimension must be six")

    runtime = _json(artefacts["runtime_acceptance"])
    if runtime.get("contract") != RUNTIME_ACCEPTANCE_CONTRACT or runtime.get("passed") is not True:
        raise ValueError("V120 Policy Lock requires passed runtime/continuity acceptance")
    if str(runtime.get("controller_config_sha256", "")) != _sha(artefacts["controller_config"]):
        raise ValueError("V120 runtime acceptance used another controller config")
    metrics = runtime.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("V120 runtime acceptance lacks metrics")
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
            raise ValueError(f"V120 Policy Lock refuses runtime {key}")

    plan = _json(artefacts["baseline_plan"])
    strategies = tuple(str(x) for x in plan.get("strategies", []))
    if strategies != EXPECTED_STRATEGIES:
        raise ValueError("V120 Final baseline matrix must contain exactly seven frozen strategies")

    hashes = {name: _sha(path) for name, path in sorted(artefacts.items())}
    canonical = json.dumps(
        {
            "contract": POLICY_LOCK_V120_CONTRACT,
            "v120_runtime_contract_sha256": v120_runtime_contract_sha256(),
            "artefacts": hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    policy_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload = {
        "contract": POLICY_LOCK_V120_CONTRACT,
        "policy_sha256": policy_sha,
        "rtc_source_tree_sha256": rtc_source_tree_sha256(),
        "v120_runtime_contract_sha256": v120_runtime_contract_sha256(),
        "physical_network_sha256": physical_contract_sha256(artefacts["frozen_inp"]),
        "native_controls_payload_sha256": section_payload_sha256(artefacts["frozen_inp"], "CONTROLS"),
        "artefacts": artefacts,
        "sha256": hashes,
        "scientific_split": split,
        "runtime_contract": runtime_contract,
        "model_contracts": {
            "step1_model_sha256": hashes["step1_model"],
            "step2_model_sha256": hashes["step2_model"],
            "model_step_seconds": 300,
            "history_steps": 13,
            "value_horizon_steps": 72,
            "swmm_engine_version": step1_engine,
            "step1_model_config": step1_config,
            "v120_model_config": step2_config,
        },
        "primary_objective": "whole_system_cumulative_TFV_m3",
        "priority_is_hard_constraint": False,
        "priority_role": "report_only",
        "global_peak_role": "report_only",
        "formal_metric_aggregation": "equal_weight_per_independent_rainfall_group",
        "hydraulic_surrogate_required": False,
        "hydraulic_gradient_gate_required": False,
        "candidate_policy": "finite_targeted_D3_distribution_matched_execution_bound",
        "strategies": list(EXPECTED_STRATEGIES),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simplified V120 TFV-only Policy Lock")
    parser.add_argument("--artefacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = create_policy_lock_v120(artefacts_path=args.artefacts, output_path=args.out)
    print(json.dumps({
        "contract": payload["contract"],
        "policy_sha256": payload["policy_sha256"],
        "primary_objective": payload["primary_objective"],
        "hydraulic_surrogate_required": False,
    }, indent=2))


if __name__ == "__main__":
    main()
