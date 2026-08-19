"""Score untouched policy-return calibration contexts with a frozen selected critic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    load_policy_return_checkpoint,
    sha256_file,
    validate_policy_return_record,
)
from rtc.production_cli import _load_graph


def _normalization_tensors(normalization, *, dtype, device):
    return {
        "state_mean": torch.as_tensor(normalization.state_mean, dtype=dtype, device=device),
        "state_std": torch.as_tensor(normalization.state_std, dtype=dtype, device=device).clamp_min(1e-6),
        "rain_mean": torch.as_tensor(normalization.rainfall_mean, dtype=dtype, device=device),
        "rain_std": torch.as_tensor(normalization.rainfall_std, dtype=dtype, device=device).clamp_min(1e-6),
        "flow_mean": torch.as_tensor(normalization.flow_mean, dtype=dtype, device=device),
        "flow_std": torch.as_tensor(normalization.flow_std, dtype=dtype, device=device).clamp_min(1e-6),
    }


def _score(model, normalization, graph, context_path: Path, device: torch.device) -> float:
    data = np.load(context_path, allow_pickle=False)
    if str(np.asarray(data["contract"]).reshape(-1)[0]) != DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT:
        raise ValueError("calibration context has the wrong dataset contract")
    if str(np.asarray(data["estimand"]).reshape(-1)[0]) != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
        raise ValueError("calibration context has the wrong estimand")
    if str(np.asarray(data["data_role"]).reshape(-1)[0]) != "policy_return_calibration":
        raise ValueError("critic scorer accepts only policy_return_calibration contexts")
    state = torch.as_tensor(data["current_state"], dtype=torch.float32, device=device)
    rain = torch.as_tensor(data["rainfall_scenarios"], dtype=torch.float32, device=device)
    active = torch.as_tensor(data["active_target"], dtype=torch.float32, device=device)
    candidate_target = torch.as_tensor(data["candidate_target"], dtype=torch.float32, device=device)
    flow = torch.as_tensor(data["previous_actuator_flow"], dtype=torch.float32, device=device)
    if rain.ndim != 5 or int(rain.shape[0]) != 1 or int(rain.shape[1]) < 2:
        raise ValueError("calibration rainfall must be [1,scenario,H,node,feature]")
    _, scenarios, horizon, nodes, features = rain.shape
    norm = _normalization_tensors(normalization, dtype=state.dtype, device=device)
    state = (state - norm["state_mean"]) / norm["state_std"]
    rain = (rain - norm["rain_mean"]) / norm["rain_std"]
    flow = (flow - norm["flow_mean"]) / norm["flow_std"]
    state = state.expand(scenarios, -1, -1)
    rain = rain.reshape(scenarios, horizon, nodes, features)
    flow = flow.expand(scenarios, -1)
    reference = active.reshape(1, 1, 109).expand(scenarios, horizon, 109)
    candidate = candidate_target.reshape(1, 1, 109).expand(scenarios, horizon, 109)
    with torch.no_grad():
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
    scores = output.total_delta_tfv_m3
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("policy-return calibration critic produced non-finite scores")
    return float(scores.mean().detach().cpu())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records-jsonl", required=True)
    p.add_argument("--policy-return-checkpoint", required=True)
    p.add_argument("--base-step2", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    model, normalization, checkpoint = load_policy_return_checkpoint(
        args.policy_return_checkpoint,
        graph=graph,
        device=device,
        expected_base_step2_sha256=sha256_file(args.base_step2),
    )
    parent_sha = str(checkpoint.get("continuation_policy_sha256", "")).lower()
    rows = []
    groups = set()
    for line_number, raw in enumerate(Path(args.records_jsonl).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"calibration record {line_number} is not an object")
        if str(row.get("data_role", "")) != "policy_return_calibration":
            raise ValueError("critic scorer received a non-calibration policy-return record")
        probe = dict(row)
        probe["predicted_policy_return_delta_tfv_m3"] = 0.0
        validate_policy_return_record(probe)
        if str(row["continuation_policy_sha256"]).lower() != parent_sha:
            raise ValueError("calibration record continuation policy differs from critic training")
        context = Path(str(row.get("context_npz", "")))
        if not context.is_file() or sha256_file(context).lower() != str(row.get("context_npz_sha256", "")).lower():
            raise ValueError(f"calibration context missing/SHA mismatch: {context}")
        prediction = _score(model, normalization, graph, context, device)
        scored = dict(row)
        scored["predicted_policy_return_delta_tfv_m3"] = prediction
        scored["policy_return_checkpoint_sha256"] = sha256_file(args.policy_return_checkpoint)
        validate_policy_return_record(scored)
        rows.append(scored)
        groups.add(str(scored["rainfall_group"]))
    if not rows:
        raise ValueError("no policy-return calibration records were scored")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "records": len(rows),
        "rainfall_group_count": len(groups),
        "rainfall_groups": sorted(groups),
        "policy_return_checkpoint_sha256": sha256_file(args.policy_return_checkpoint),
        "continuation_policy_sha256": parent_sha,
        "scored_records_sha256": sha256_file(out),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
