"""Audit V5 D2/D3 gradient interference without training or changing a checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, derive_v127_input_normalization, load_causal_state_store_v127
from rtc.step2_tfv_value import DirectFacilityTFVValueModel, DirectTFVValueDesign
from rtc.step2_tfv_value_training import _branch_indices, _graph_tensors
from rtc.step2_tfv_value_training_v4 import DirectTFVTrainingDesignV4, _group_loss_v4
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_gradients(model: torch.nn.Module, names: tuple[str, ...]) -> torch.Tensor:
    selected = set(names)
    values: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if name in selected:
            values.append(torch.zeros_like(parameter).reshape(-1) if parameter.grad is None else parameter.grad.detach().reshape(-1))
    return torch.cat(values) if values else torch.zeros(1, device=next(model.parameters()).device)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1.0e-12:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-checkpoint", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--causal-state-store", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    payload = torch.load(args.v5_checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("V5 checkpoint does not contain model_state_dict")
    graph = _load_graph(args.graph)
    base = V60TrainCache(args.cache_manifest)
    rainfall_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    fit, holdout = deterministic_rainfall_split_v60(base, names=sorted(base.names("D2") + base.targeted_d3_names()), holdout_fraction=0.20)
    fit_d2 = sorted(name for name in fit if name.startswith("D2::"))
    fit_d3 = sorted(name for name in fit if name.startswith("D3::"))
    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rainfall_store), state_store)
    normalization = derive_v127_input_normalization(base_cache=base, causal_rainfall=rainfall_store, causal_state=state_store, fit_names=fit_d2 + fit_d3)
    model_design = DirectTFVValueDesign(**dict(payload.get("model_design", {})))
    model = DirectFacilityTFVValueModel(
        state_dim=int(payload["state_dim"]),
        rainfall_dim=int(payload["rainfall_dim"]),
        actuator_physics_dim=int(payload["actuator_physics_dim"]),
        target_scale_m3=float(payload["target_scale_m3"]),
        design=model_design,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.train()
    static = _graph_tensors(graph, device)
    design = DirectTFVTrainingDesignV4(main_epochs=1, interaction_epochs=1, control_epochs=1, seed=int(args.seed))
    shared = tuple(name for name, _ in model.named_parameters() if name.startswith(("facility_encoder.", "facility_head.")))
    if not shared:
        raise RuntimeError("V5 checkpoint has no shared facility encoder/head parameters")

    def group_loss(name: str, *, d3: bool) -> torch.Tensor:
        batch = base_online.batch(name, normalization, device)
        mode = "all" if d3 else "single"
        result = _group_loss_v4(model, batch, mode=mode, graph_tensors=static, design=design, control_decision_loss=d3)
        if result is None:
            raise RuntimeError(f"{name}: no usable branches for gradient audit")
        return result[0]

    def aggregate(names: list[str], *, d3: bool, scope: tuple[str, ...]) -> tuple[torch.Tensor, float]:
        model.zero_grad(set_to_none=True)
        losses: list[float] = []
        for name in names:
            loss = group_loss(name, d3=d3) / float(len(names))
            loss.backward()
            losses.append(float(loss.detach().cpu()))
        return _flat_gradients(model, scope).detach().clone(), float(np.mean(losses))

    encoder_names = tuple(name for name in shared if name.startswith("facility_encoder."))
    head_names = tuple(name for name in shared if name.startswith("facility_head."))
    d2_global, d2_loss = aggregate(fit_d2, d3=False, scope=shared)
    d3_global, d3_loss = aggregate(fit_d3, d3=True, scope=shared)
    d2_encoder, _ = aggregate(fit_d2, d3=False, scope=encoder_names)
    d3_encoder, _ = aggregate(fit_d3, d3=True, scope=encoder_names)
    d2_head, _ = aggregate(fit_d2, d3=False, scope=head_names)
    d3_head, _ = aggregate(fit_d3, d3=True, scope=head_names)
    paired_cosines: list[float] = []
    for d2_name, d3_name in zip(fit_d2, fit_d3, strict=True):
        model.zero_grad(set_to_none=True)
        group_loss(d2_name, d3=False).backward()
        d2_grad = _flat_gradients(model, shared).detach().clone()
        model.zero_grad(set_to_none=True)
        group_loss(d3_name, d3=True).backward()
        d3_grad = _flat_gradients(model, shared).detach().clone()
        paired_cosines.append(_cosine(d2_grad, d3_grad))
    finite_cosines = np.asarray([value for value in paired_cosines if np.isfinite(value)], dtype=np.float64)
    result = {
        "contract": "PROJECT7_STEP2_V5_D2_D3_SHARED_GRADIENT_INTERFERENCE_AUDIT_V1",
        "development_only": True,
        "training_performed": False,
        "checkpoint_modified": False,
        "new_swmm_runs": 0,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_used_for_tuning": False,
        "v5_checkpoint": str(args.v5_checkpoint),
        "v5_checkpoint_sha256": _sha(args.v5_checkpoint),
        "graph_sha256": _sha(args.graph),
        "fit_d2_groups": len(fit_d2),
        "fit_d3_groups": len(fit_d3),
        "rainfall_group_overlap": 0,
        "shared_parameter_scope": list(shared),
        "d2_global_loss": d2_loss,
        "d3_global_loss": d3_loss,
        "d2_shared_gradient_norm": float(torch.linalg.vector_norm(d2_global).cpu()),
        "d3_shared_gradient_norm": float(torch.linalg.vector_norm(d3_global).cpu()),
        "global_shared_gradient_cosine": _cosine(d2_global, d3_global),
        "facility_encoder_gradient_cosine": _cosine(d2_encoder, d3_encoder),
        "facility_head_gradient_cosine": _cosine(d2_head, d3_head),
        "group_pair_count": len(paired_cosines),
        "negative_cosine_sampled_fraction": float(np.mean(finite_cosines < 0.0)) if finite_cosines.size else float("nan"),
        "sampled_cosine_min": float(np.min(finite_cosines)) if finite_cosines.size else float("nan"),
        "sampled_cosine_median": float(np.median(finite_cosines)) if finite_cosines.size else float("nan"),
        "sampled_cosine_max": float(np.max(finite_cosines)) if finite_cosines.size else float("nan"),
        "interpretation": "negative shared D2/D3 gradient cosine is direct evidence that D3 updates can interfere with the V5 D2 backbone; this audit is diagnostic only",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
