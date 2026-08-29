"""Plan a small V28 q95-matched truth augmentation from existing causal contexts.

The planner is deliberately conservative.  It scores only current causal-prefix and action
geometry diagnostics, never event identifiers or realized benchmark outcomes.  Existing exact
supported actions are marked for reuse; only missing supported action identities are eligible for a
future Development-only counterfactual job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_operational_v27_runtime import build_operational_v27_controller
from rtc.direct_tfv_operational_v28_runtime import _q27_m3_from_latent, _v23_strong_storm_blend
from rtc.direct_tfv_policy_return import encode_policy_return_action_token
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


V28_TRUTH_PLAN_CONTRACT = "PROJECT7_V28_Q95_TARGETED_EXACT_RETURN_TRUTH_PLAN_V1"
SOURCE_ORDER = (
    "STEP2_H10_PROBE_SCALE_0.50",
    "STEP2_H10_PROBE_SCALE_1.00",
    "TYPE_AWARE_HYDRAULIC_PRESSURE",
    "AUTO_RBC_SHADOW_TOPK",
)


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _present(value: Any) -> bool:
    """Treat zero-valued provenance fields (such as decision index 0) as present."""
    return value is not None and value != ""


def _load_context(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    try:
        return {
            "current_state": np.asarray(data["current_state"]).copy(),
            "rainfall_scenarios": np.asarray(data["rainfall_scenarios"]).copy(),
            "active_target": np.asarray(data["active_target"]).reshape(-1).copy(),
            "previous_actuator_flow": np.asarray(data["previous_actuator_flow"]).reshape(-1).copy(),
        }
    finally:
        data.close()


def _bin(value: float, edges: tuple[float, float, float]) -> str:
    if value < edges[0]:
        return "low"
    if value < edges[1]:
        return "medium"
    if value < edges[2]:
        return "high"
    return "very-high"


def _candidate_geometry(mpc: Any, context: dict[str, np.ndarray], row: dict[str, Any]) -> dict[str, Any] | None:
    device = next(mpc.model.parameters()).device
    active = torch.as_tensor(context["active_target"], dtype=torch.float32, device=device).reshape(-1)
    raw = torch.as_tensor(row["candidate_target"], dtype=torch.float32, device=device).reshape(-1)
    supported, supported_sequence, changed, support = mpc._h10_supported_target(raw, active)
    if changed <= 0:
        return None
    _, raw_batch = encode_policy_return_action_token(
        active,
        raw,
        horizon_steps=int(mpc.design.prediction_horizon_steps),
        first_action_steps=int(mpc.design.control_block_steps),
    )
    raw_sequence = raw_batch[0].detach()
    raw_geometry = mpc._joint_sequence_geometry_torch(raw_sequence, active)
    supported_geometry = mpc._joint_sequence_geometry_torch(supported_sequence, active)
    raw_first = float(raw_geometry["first_block_l1"].detach().cpu())
    supported_first = float(supported_geometry["first_block_l1"].detach().cpu())
    raw_delta = raw_sequence[1:] - raw_sequence[:-1]
    supported_delta = supported_sequence[1:] - supported_sequence[:-1]
    state = torch.as_tensor(context["current_state"], dtype=torch.float32, device=device)
    rain = torch.as_tensor(context["rainfall_scenarios"], dtype=torch.float32, device=device)
    flow = torch.as_tensor(context["previous_actuator_flow"], dtype=torch.float32, device=device)
    with torch.inference_mode():
        built, latent, _, _ = mpc._feature_and_scores(
            current_state=state,
            rainfall=rain,
            flow=flow.reshape(1, -1),
            active_target=active,
            target=supported,
            source=str(row.get("candidate_source", "")),
        )
    q27 = _q27_m3_from_latent(mpc.v27_value_model, float(latent))
    return {
        "supported_target": supported.detach().cpu().numpy().astype(np.float32),
        "supported_target_sha256": hashlib.sha256(supported.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest(),
        "q95_scale": float(np.clip(supported_first / raw_first if raw_first > 1.0e-12 else 1.0, 0.0, 1.0)),
        "q95_max_ratio": float(support["max_ratio"]),
        "q95_binding": bool(support["binding"]),
        "raw_to_supported_first_move_l1": float(torch.abs(raw - supported).sum().detach().cpu()),
        "raw_to_supported_h120_l1": float(torch.abs(raw_sequence - supported_sequence).sum().detach().cpu()),
        "raw_to_supported_tv_l1": float(torch.abs(raw_delta - supported_delta).sum().detach().cpu()),
        "changed_facility_count": int(changed),
        "network_stress_q75": float(built.base.network_stress_q75),
        "rain_level": float(built.base.rain_level),
        "strong_storm_blend": _v23_strong_storm_blend(built.base.network_stress_q75),
        "q27_score_m3": float(q27),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v27-root", required=True)
    parser.add_argument("--dataset-records", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--v27-value-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-contexts", type=int, default=32)
    parser.add_argument("--max-candidates-per-context", type=int, default=4)
    args = parser.parse_args()
    records_path = Path(args.dataset_records).resolve()
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("V28 truth planner received no dataset rows")
    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    parent, _, _, _ = build_operational_v27_controller(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=practical_asset_path(assets, "step2"),
        supervisory_control_path=practical_asset_path(assets, "supervisory_control"),
        sequence_support_path=practical_asset_path(assets, "sequence_support"),
        v15_rank_checkpoint_path=args.v15_rank_checkpoint,
        v21_boundary_checkpoint_path=args.v21_boundary_checkpoint,
        v27_value_checkpoint_path=args.v27_value_checkpoint,
        dataset_manifest_path=Path(args.v27_root).resolve() / "dataset" / "V27_EXACT_RETURN_DATASET_MANIFEST.json",
        asset_manifest_path=args.asset_manifest,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        decision_runtime_budget_seconds=180.0,
        proposal_probe_chunk_size=24,
    )
    mpc = parent.controller._direct_mpc_adapter.inner
    by_context: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        context_id = str(row.get("causal_context_fingerprint_sha256", "")).lower()
        context_path = Path(str(row.get("context_npz", "")))
        if not context_id or not context_path.is_file():
            continue
        geometry = _candidate_geometry(mpc, _load_context(context_path), row)
        if geometry is None:
            continue
        by_context.setdefault(context_id, []).append({"row": row, "geometry": geometry})
    existing_supported: set[tuple[str, str]] = set()
    for context_id, members in by_context.items():
        for member in members:
            geometry = member["geometry"]
            raw = np.asarray(member["row"]["candidate_target"], dtype=np.float32).reshape(-1)
            if np.array_equal(raw, geometry["supported_target"]):
                existing_supported.add((context_id, geometry["supported_target_sha256"]))
    candidates: list[dict[str, Any]] = []
    for context_id, members in by_context.items():
        first = members[0]
        geometry = first["geometry"]
        family_names = sorted({str(member["row"].get("candidate_source", "")) for member in members})
        disagreement = float(max(member["geometry"]["q27_score_m3"] for member in members) - min(member["geometry"]["q27_score_m3"] for member in members))
        equivalent_groups: dict[str, list[dict[str, Any]]] = {}
        for member in members:
            equivalent_groups.setdefault(str(member["geometry"]["supported_target_sha256"]), []).append(member)
        ranked_groups = sorted(
            equivalent_groups.values(),
            key=lambda group: (
                not bool(group[0]["geometry"]["q95_binding"]),
                abs(float(group[0]["geometry"]["q27_score_m3"])),
                -float(group[0]["geometry"]["raw_to_supported_h120_l1"]),
                min(str(item["row"].get("candidate_source", "")) for item in group),
                min(str(item["row"].get("query_set_id", "")) for item in group),
            ),
        )[: max(1, int(args.max_candidates_per_context))]
        for equivalent_members in ranked_groups:
            member = min(
                equivalent_members,
                key=lambda item: (
                    -sum(
                        _present(item["row"].get(field))
                        for field in (
                            "historical_input_source_path",
                            "hold_metadata_path",
                            "parent_decisions_sha256",
                            "source_inp_sha256",
                            "prefix_sha256",
                            "decision_elapsed_seconds",
                            "decision_index",
                        )
                    ),
                    SOURCE_ORDER.index(str(item["row"].get("candidate_source", "")))
                    if str(item["row"].get("candidate_source", "")) in SOURCE_ORDER
                    else len(SOURCE_ORDER),
                    str(item["row"].get("query_set_id", "")),
                ),
            )
            row = member["row"]
            geometry = member["geometry"]
            overlap = (context_id, geometry["supported_target_sha256"]) in existing_supported
            contributing_sources = sorted(
                {str(item["row"].get("candidate_source", "")) for item in equivalent_members},
                key=lambda value: SOURCE_ORDER.index(value) if value in SOURCE_ORDER else len(SOURCE_ORDER),
            )
            candidates.append(
                {
                    "causal_context_fingerprint_sha256": context_id,
                    "query_set_id": str(row.get("query_set_id", "")),
                    "event_id": str(row.get("event_id", "")),
                    "decision_index": row.get("decision_index"),
                    "decision_elapsed_seconds": row.get("decision_elapsed_seconds"),
                    "rainfall_group": str(row.get("rainfall_group", "")),
                    "state_hash": context_id,
                    "context_npz": str(row.get("context_npz", "")),
                    "context_npz_sha256": row.get("context_npz_sha256"),
                    "active_target_sha256": hashlib.sha256(np.asarray(_load_context(Path(str(row["context_npz"])))["active_target"], dtype=np.float32).tobytes()).hexdigest(),
                    "candidate_family_list": family_names,
                    "candidate_source": str(row.get("candidate_source", "")),
                    "contributing_sources": contributing_sources,
                    "candidate_target": np.asarray(row["candidate_target"], dtype=np.float32).reshape(-1).tolist(),
                    "q95_supported_target": np.asarray(geometry["supported_target"], dtype=np.float32).reshape(-1).tolist(),
                    "q95_supported_target_sha256": geometry["supported_target_sha256"],
                    "source_inp_sha256": row.get("source_inp_sha256"),
                    "parent_decisions_sha256": row.get("parent_decisions_sha256"),
                    "prefix_sha256": row.get("prefix_sha256"),
                    "continuation_policy_sha256": row.get("continuation_policy_sha256"),
                    "hold_metadata_path": row.get("hold_metadata_path"),
                    "hold_node_statistics_path": row.get("hold_node_statistics_path"),
                    "hold_compact_path": row.get("hold_compact_path"),
                    "historical_input_source_path": row.get("historical_input_source_path"),
                    "historical_origin_source_path": row.get("historical_origin_source_path"),
                    "selection_reason": {
                        "q95_binding_candidates": int(sum(bool(item["geometry"]["q95_binding"]) for item in members)),
                        "q27_abs_score_m3": abs(float(geometry["q27_score_m3"])),
                        "candidate_family_disagreement_m3": disagreement,
                        "raw_to_supported_h120_l1": float(geometry["raw_to_supported_h120_l1"]),
                    },
                    "stress_bin": _bin(float(geometry["network_stress_q75"]), (0.25, 0.50, 0.75)),
                    "rain_bin": _bin(float(geometry["rain_level"]), (0.25, 0.50, 0.75)),
                    "existing_truth_overlap": bool(overlap),
                    "new_truth_required": not bool(overlap),
                    "q95_diagnostics": {
                        key: geometry[key]
                        for key in ("q95_scale", "q95_max_ratio", "q95_binding", "raw_to_supported_first_move_l1", "raw_to_supported_h120_l1", "raw_to_supported_tv_l1", "changed_facility_count", "network_stress_q75", "rain_level", "strong_storm_blend")
                    },
                }
            )
    # Deterministic, non-event-based coverage: rank context groups by causal geometry, then hash.
    context_rank: dict[str, tuple[Any, ...]] = {}
    for item in candidates:
        context_rank.setdefault(
            item["causal_context_fingerprint_sha256"],
            (
                not bool(item["q95_diagnostics"]["q95_binding"]),
                float(item["selection_reason"]["q27_abs_score_m3"]),
                -float(item["selection_reason"]["raw_to_supported_h120_l1"]),
                hashlib.sha256(item["causal_context_fingerprint_sha256"].encode("utf-8")).hexdigest(),
            ),
        )
    selected_contexts = [value for value, _ in sorted(context_rank.items(), key=lambda pair: pair[1])[: max(1, int(args.max_contexts))]]
    selected = [item for item in candidates if item["causal_context_fingerprint_sha256"] in set(selected_contexts)]
    selected.sort(key=lambda item: (selected_contexts.index(item["causal_context_fingerprint_sha256"]), str(item["candidate_source"]), str(item["q95_supported_target_sha256"])))
    output = {
        "contract": V28_TRUTH_PLAN_CONTRACT,
        "development_only": True,
        "formal_evidence": False,
        "event_id_used_for_selection": False,
        "benchmark_outcome_used_for_selection": False,
        "v27_root": str(Path(args.v27_root).resolve()),
        "dataset_records": str(records_path),
        "dataset_records_sha256": _sha(records_path),
        "candidate_context_count": len(by_context),
        "selected_context_count": len(set(selected_contexts)),
        "selected_candidate_count": len(selected),
        "existing_truth_overlap_count": int(sum(not item["new_truth_required"] for item in selected)),
        "new_truth_required_count": int(sum(item["new_truth_required"] for item in selected)),
        "selection_policy": {
            "q95_binding_first": True,
            "near_hold_q27_score_second": True,
            "displacement_and_family_coverage_used": True,
            "stress_and_rain_bins_used_for_diagnostic_coverage": True,
            "event_id_or_benchmark_tfvs_used": False,
        },
        "contexts": selected,
    }
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
