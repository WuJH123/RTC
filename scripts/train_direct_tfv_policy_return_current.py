"""Fine-tune Direct-TFV Step2 on paired receding-policy return labels.

This is Development-only policy evaluation.  It initializes from the frozen accepted Direct-TFV V5
checkpoint and changes the *supervision target*, not the network architecture.  Training, model-
selection validation and conformal calibration rainfall groups must be disjoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_CHECKPOINT_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS,
    sha256_file,
)
from rtc.production_cli import _load_graph


MIN_VALIDATION_GROUPS = 12


def _scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data:
        raise ValueError(f"policy-return dataset lacks {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"policy-return dataset field {key} must be scalar")
    return str(value.reshape(-1)[0])


def _load_dataset(path: str | Path, *, role: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    if _scalar_text(data, "contract") != DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT:
        raise ValueError("policy-return dataset has the wrong contract")
    if _scalar_text(data, "estimand") != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
        raise ValueError("policy-return dataset has the wrong estimand")
    if _scalar_text(data, "data_role") != role:
        raise ValueError(f"policy-return dataset role must be {role}")
    required = (
        "current_state", "rainfall_scenarios", "active_target", "candidate_target",
        "previous_actuator_flow", "true_policy_return_delta_tfv_m3", "rainfall_group",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"policy-return dataset lacks {key}")
    result = {key: np.asarray(data[key]) for key in required}
    n = int(result["current_state"].shape[0])
    if n <= 0 or any(int(result[key].shape[0]) != n for key in required):
        raise ValueError("policy-return dataset arrays have inconsistent sample counts")
    if result["rainfall_scenarios"].ndim != 6:
        raise ValueError("rainfall_scenarios must be [sample,scenario,H,node,rain_feature]")
    if int(result["rainfall_scenarios"].shape[1]) < 2:
        raise ValueError("policy-return training requires at least two causal rainfall scenarios")
    if tuple(result["active_target"].shape[1:]) != (109,) or tuple(result["candidate_target"].shape[1:]) != (109,):
        raise ValueError("policy-return target arrays must contain 109 actuators")
    if tuple(result["previous_actuator_flow"].shape[1:]) != (109,):
        raise ValueError("policy-return flow array must contain 109 actuators")
    for key in required[:-1]:
        if not np.isfinite(result[key].astype(np.float64)).all():
            raise ValueError(f"policy-return dataset {key} contains non-finite values")
    result["groups"] = np.asarray(result.pop("rainfall_group")).astype(str)
    result["path"] = str(Path(path).resolve())
    result["sha256"] = sha256_file(path)
    return result


def _normalization_tensors(normalization: Any, *, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.as_tensor(normalization.state_mean, dtype=dtype, device=device),
        "state_std": torch.as_tensor(normalization.state_std, dtype=dtype, device=device).clamp_min(1e-6),
        "rain_mean": torch.as_tensor(normalization.rainfall_mean, dtype=dtype, device=device),
        "rain_std": torch.as_tensor(normalization.rainfall_std, dtype=dtype, device=device).clamp_min(1e-6),
        "flow_mean": torch.as_tensor(normalization.flow_mean, dtype=dtype, device=device),
        "flow_std": torch.as_tensor(normalization.flow_std, dtype=dtype, device=device).clamp_min(1e-6),
    }


def _predict_group(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    graph: Any,
    normalization: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(dataset["current_state"][indices], dtype=torch.float32, device=device)
    rain = torch.as_tensor(dataset["rainfall_scenarios"][indices], dtype=torch.float32, device=device)
    active = torch.as_tensor(dataset["active_target"][indices], dtype=torch.float32, device=device)
    candidate_target = torch.as_tensor(dataset["candidate_target"][indices], dtype=torch.float32, device=device)
    flow = torch.as_tensor(dataset["previous_actuator_flow"][indices], dtype=torch.float32, device=device)
    truth = torch.as_tensor(dataset["true_policy_return_delta_tfv_m3"][indices], dtype=torch.float32, device=device)
    b, scenarios, horizon, nodes, rain_features = rain.shape
    norm = _normalization_tensors(normalization, dtype=state.dtype, device=device)
    state = (state - norm["state_mean"]) / norm["state_std"]
    rain = (rain - norm["rain_mean"]) / norm["rain_std"]
    flow = (flow - norm["flow_mean"]) / norm["flow_std"]
    state = state[:, None].expand(-1, scenarios, -1, -1).reshape(b * scenarios, *state.shape[1:])
    rain = rain.reshape(b * scenarios, horizon, nodes, rain_features)
    flow = flow[:, None].expand(-1, scenarios, -1).reshape(b * scenarios, 109)
    reference = active[:, None, None].expand(-1, scenarios, horizon, -1).reshape(b * scenarios, horizon, 109)
    candidate = candidate_target[:, None, None].expand(-1, scenarios, horizon, -1).reshape(b * scenarios, horizon, 109)
    output = model(
        current_state=state,
        rainfall=rain,
        reference_settings=reference,
        candidate_settings=candidate,
        previous_actuator_flow=flow,
        actuator_upstream=torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        actuator_physics=torch.as_tensor(graph.actuator_physics, dtype=state.dtype, device=device),
    )
    prediction = output.total_delta_tfv_m3.reshape(b, scenarios).mean(dim=1)
    return prediction, truth


def _group_loss(prediction: torch.Tensor, truth: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
    regression = F.smooth_l1_loss(prediction / scale, truth / scale)
    informative = torch.abs(truth) > 1.0
    sign_loss = prediction.new_zeros(())
    if bool(informative.any()):
        margin = (prediction[informative] / scale) * torch.sign(truth[informative])
        sign_loss = F.softplus(-margin).mean()
    ranking = prediction.new_zeros(())
    if int(prediction.numel()) >= 2:
        pi = prediction[:, None] - prediction[None, :]
        ti = truth[:, None] - truth[None, :]
        mask = torch.triu(torch.abs(ti) > 1.0, diagonal=1)
        if bool(mask.any()):
            ranking = F.softplus(-(pi[mask] / scale) * torch.sign(ti[mask])).mean()
    return regression + 0.25 * sign_loss + 0.35 * ranking


def _evaluate(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *, graph: Any, normalization: Any, device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[tuple[float, float]] = []
    with torch.no_grad():
        for group in sorted(set(dataset["groups"].tolist())):
            idx = np.flatnonzero(dataset["groups"] == group)
            pred, truth = _predict_group(model, dataset, idx, graph=graph, normalization=normalization, device=device)
            rows.extend(zip(pred.detach().cpu().tolist(), truth.detach().cpu().tolist(), strict=True))
    pred = np.asarray([x[0] for x in rows], dtype=float)
    truth = np.asarray([x[1] for x in rows], dtype=float)
    sign = np.sign(pred) == np.sign(truth)
    informative = np.abs(truth) > 1.0
    return {
        "mae_m3": float(np.mean(np.abs(pred - truth))),
        "sign_accuracy": float(np.mean(sign[informative])) if np.any(informative) else 1.0,
        "false_beneficial_rate": float(np.mean((pred < 0.0) & (truth >= 0.0))),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-step2", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--train-dataset", required=True)
    p.add_argument("--validation-dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2.0e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs <= 0 or not 0.0 < args.learning_rate < 1.0e-2:
        raise ValueError("invalid policy-return training hyperparameters")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, base = load_direct_tfv_runtime_checkpoint(args.base_step2, graph=graph, device=device)
    train = _load_dataset(args.train_dataset, role="policy_return_train")
    valid = _load_dataset(args.validation_dataset, role="policy_return_validation")
    train_groups = set(train["groups"].tolist()); valid_groups = set(valid["groups"].tolist())
    if len(train_groups) < DIRECT_TFV_POLICY_RETURN_MIN_TRAIN_GROUPS:
        raise ValueError("insufficient policy-return training rainfall groups")
    if len(valid_groups) < MIN_VALIDATION_GROUPS:
        raise ValueError("policy-return model selection requires >=12 validation rainfall groups")
    if train_groups & valid_groups:
        raise ValueError("policy-return train/validation rainfall groups overlap")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-5)
    scale = model.target_scale_m3.to(device).clamp_min(1.0)
    best_state = copy.deepcopy(model.state_dict()); best_metric = float("inf"); history = []
    groups = sorted(train_groups)
    for epoch in range(1, int(args.epochs) + 1):
        random.Random(args.seed + epoch).shuffle(groups)
        model.train(); losses = []
        for group in groups:
            idx = np.flatnonzero(train["groups"] == group)
            prediction, truth = _predict_group(model, train, idx, graph=graph, normalization=normalization, device=device)
            loss = _group_loss(prediction, truth, scale=scale)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _evaluate(model, valid, graph=graph, normalization=normalization, device=device)
        selection = metrics["mae_m3"] * (1.0 + metrics["false_beneficial_rate"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        if selection < best_metric:
            best_metric = selection; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); final_metrics = _evaluate(model, valid, graph=graph, normalization=normalization, device=device)
    payload = {
        "contract": DIRECT_TFV_POLICY_RETURN_CHECKPOINT_CONTRACT,
        "development_only": True,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "base_step2_sha256": sha256_file(args.base_step2),
        "graph_sha256": sha256_file(args.graph),
        "train_dataset_sha256": train["sha256"],
        "validation_dataset_sha256": valid["sha256"],
        "train_rainfall_group_count": len(train_groups),
        "validation_rainfall_group_count": len(valid_groups),
        "actuator_ids": [str(x) for x in graph.actuator_ids],
        "state_dim": int(base["state_dim"]),
        "rainfall_dim": int(base["rainfall_dim"]),
        "actuator_physics_dim": int(base["actuator_physics_dim"]),
        "target_scale_m3": float(base["target_scale_m3"]),
        "model_design": dict(base["model_design"]),
        "normalization": dict(base["normalization"]),
        "model_state_dict": model.state_dict(),
        "training_history": history,
        "validation_metrics": final_metrics,
        "selection_rule": "min_validation_MAE_times_one_plus_false_beneficial_rate",
        "future_realized_rainfall_used_as_model_input": False,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, out)
    report = {k: v for k, v in payload.items() if k not in {"model_state_dict", "normalization"}}
    report["checkpoint_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
