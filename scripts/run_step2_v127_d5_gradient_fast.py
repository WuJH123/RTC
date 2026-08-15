"""Fine-tune V127 with D5-FIT while reusing causal inputs and center gradients."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from rtc.checkpoint_v127 import (
    input_normalization_from_v127_checkpoint,
    load_step2_v127,
    save_step2_v127,
)
from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_gradient_v127 import V127GradientTrainingDesign, build_direction_cases_v127
from rtc.step2_gradient_v127_fast import (
    V127_GRADIENT_FAST_CONTRACT,
    evaluate_d5_gradients_fast_v127,
    train_d5_gradient_fast_v127,
)
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache

V127_D5_FAST_RUN_CONTRACT = "PROJECT7_V127_D5_GRADIENT_FINETUNE_FAST_V1_16GB_RAM_8GB_VRAM"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--base-cache-manifest", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--d5-execution-manifest", required=True)
    p.add_argument("--d5-gradient-labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        if total < 7.0 * 1024**3:
            raise RuntimeError("V127 D5 fast profile expects an approximately 8-GB CUDA GPU")
        if free < 4.0 * 1024**3:
            raise RuntimeError("less than 4 GB CUDA memory is free before D5 fine-tuning")

    graph = _load_graph(args.graph)
    model, checkpoint_payload = load_step2_v127(args.step2, graph=graph, device=device)
    normalization = input_normalization_from_v127_checkpoint(checkpoint_payload)
    base = V60TrainCache(args.base_cache_manifest)
    rain = load_causal_forecast_store_v123(args.causal_store)
    state = load_causal_state_store_v127(args.causal_state_store)
    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain), state)

    labels = pd.read_csv(args.d5_gradient_labels)
    manifest = pd.read_csv(args.d5_execution_manifest)
    fit_cases = build_direction_cases_v127(
        labels, manifest, actuator_ids=graph.actuator_ids, split_role="fit"
    )
    audit_cases = build_direction_cases_v127(
        labels, manifest, actuator_ids=graph.actuator_ids, split_role="audit"
    )
    fit_rain = {str(case["rainfall_group"]) for case in fit_cases}
    audit_rain = {str(case["rainfall_group"]) for case in audit_cases}
    if fit_rain & audit_rain:
        raise ValueError("V127 D5 FIT/AUDIT rainfall leakage")

    before_detail, before = evaluate_d5_gradients_fast_v127(
        model,
        cache=online,
        cases=audit_cases,
        normalization=normalization,
        graph=graph,
        device=device,
        flood_rate_index=int(args.flood_rate_index),
    )
    history = train_d5_gradient_fast_v127(
        model,
        cache=online,
        cases=fit_cases,
        normalization=normalization,
        graph=graph,
        device=device,
        flood_rate_index=int(args.flood_rate_index),
        design=V127GradientTrainingDesign(),
    )
    after_detail, after = evaluate_d5_gradients_fast_v127(
        model,
        cache=online,
        cases=audit_cases,
        normalization=normalization,
        graph=graph,
        device=device,
        flood_rate_index=int(args.flood_rate_index),
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before_detail.to_csv(out / "D5_AUDIT_GRADIENTS_BEFORE.csv", index=False)
    after_detail.to_csv(out / "D5_AUDIT_GRADIENTS_AFTER.csv", index=False)
    lineage = {
        "base_step2_sha256": _sha(args.step2),
        "graph_sha256": _sha(args.graph),
        "base_cache_sha256": _sha(args.base_cache_manifest),
        "causal_rainfall_sha256": _sha(args.causal_store),
        "causal_rainfall_forecast_contract": str(rain.forecast_contract),
        "causal_state_store_sha256": _sha(args.causal_state_store),
        "d5_execution_manifest_sha256": _sha(args.d5_execution_manifest),
        "d5_gradient_labels_sha256": _sha(args.d5_gradient_labels),
    }
    base_lineage = checkpoint_payload.get("lineage")
    if not isinstance(base_lineage, dict):
        raise ValueError("V127 base Step2 checkpoint lacks causal lineage")
    for key in (
        "swmm_engine_version",
        "causal_state_step1_model_semantic_sha256",
        "causal_state_sensor_layout_semantic_sha256",
    ):
        value = str(base_lineage.get(key, "")).strip()
        if not value:
            raise ValueError(f"V127 base Step2 checkpoint lacks {key}")
        lineage[key] = value
    if lineage["causal_state_step1_model_semantic_sha256"] != str(state.step1_model_semantic_sha256):
        raise ValueError("V127 D5 Step1 model semantics differ from base Step2 checkpoint")
    if lineage["causal_state_sensor_layout_semantic_sha256"] != str(state.sensor_layout_semantic_sha256):
        raise ValueError("V127 D5 sensor layout semantics differ from base Step2 checkpoint")

    report = {
        "contract": V127_D5_FAST_RUN_CONTRACT,
        "execution_contract": V127_GRADIENT_FAST_CONTRACT,
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "fit_directions": len(fit_cases),
        "audit_directions": len(audit_cases),
        "fit_rainfall_groups": sorted(fit_rain),
        "audit_rainfall_groups": sorted(audit_rain),
        "rainfall_overlap": sorted(fit_rain & audit_rain),
        "causal_rainfall_forecast_contract": str(rain.forecast_contract),
        "audit_before": before,
        "training_history": history,
        "audit_after": after,
        "lineage": lineage,
        "boundary": {
            "d5_audit_used_for_training": False,
            "internal_holdout_used_for_training": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    checkpoint = save_step2_v127(
        out / "step2_v127_d5_gradient.pt",
        model=model,
        graph=graph,
        input_normalization=normalization,
        training_report=report,
        lineage=lineage,
    )
    report["checkpoint"] = str(checkpoint.resolve())
    report["final_step2_sha256"] = _sha(checkpoint)
    report_path = out / "STEP2_V127_D5_GRADIENT_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
