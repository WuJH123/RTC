"""Loss-component gradient audit for one fixed TrainFit D2 group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111, hydraulic_effect_loss_v111
from rtc.step2_train_response_v60 import V60TrainCache, derive_input_normalization_v60
from rtc.step2_v110_contract import HydraulicHorizonV110
from rtc.step2_v111_contract import V111LossContract
from run_step2_v110 import _build_reference, _load_checkpoint, _load_graph, _sha256
from run_step2_v111 import _build_model, _git_head


def _norm(parameters):
    vals = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(vals).norm()) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True); parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True); parser.add_argument("--group", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    contract = V111LossContract(); contract.validate()
    graph = _load_graph(args.graph); cache = V60TrainCache(args.cache_manifest)
    fit_names = [n for n in cache.names("D2") if n.startswith("D2::")]
    normalization = derive_input_normalization_v60(cache, fit_names)
    prepared_cpu = prepare_static_v60(graph, "cpu")
    scales = derive_effect_scales_v111(cache, fit_names, prepared_cpu)
    target = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    prepared = prepare_static_v60(graph, target)
    model = _build_model(graph, cache, fit_names, normalization, _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response"), prepared_cpu, scales).to(target)
    batch = cache.batch(args.group, normalization, target)
    output = model(batch.initial_state, batch.rainfall, batch.reference_settings, batch.candidate_settings,
                   batch.previous_actuator_flow, prepared)
    _total, metrics, terms = hydraulic_effect_loss_v111(output, batch, scales, return_terms=True)
    buckets = {
        "shared_source_encoder": [p for n, p in model.named_parameters() if n.startswith("source_encoder.")],
        "node_decoder": [p for n, p in model.named_parameters() if n.startswith("node_decoder.")],
        "flow_decoder": [p for n, p in model.named_parameters() if n.startswith("flow_decoder.")],
        "direct_head": list(model.state_direct_effect_head.parameters()) + list(model.flow_direct_effect_head.parameters()),
        "active_head": list(model.state_active_head.parameters()) + list(model.flow_active_head.parameters()),
        "sign_head": list(model.state_sign_head.parameters()) + list(model.flow_sign_head.parameters()),
        "magnitude_head": list(model.state_magnitude_head.parameters()) + list(model.flow_magnitude_head.parameters()),
    }
    report = {"contract": "PROJECT7_STEP2_V111_ZERO_ANCHORED_SIGNED_EFFECT_V1", "git_head": _git_head(),
              "group": args.group, "lineage": {"graph_sha256": _sha256(args.graph),
              "cache_manifest_sha256": _sha256(args.cache_manifest),
              "v70_hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint)}, "terms": {}}
    for term_name, term in terms.items():
        model.zero_grad(set_to_none=True)
        term.backward(retain_graph=True)
        flat = {}
        for bucket, params in buckets.items():
            flat[bucket] = _norm(params)
        finite = all(np.isfinite(v) for v in flat.values())
        report["terms"][term_name] = {"loss": float(term.detach()), "gradient_norms": flat,
                                       "finite": finite, "nonzero": any(v > 0.0 for v in flat.values())}
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
