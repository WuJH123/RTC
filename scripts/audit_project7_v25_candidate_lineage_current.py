"""Audit current V23 portfolio action hashes against existing Development truth.

This is a read-only, no-SWMM audit.  It regenerates the current V23 portfolio from each unique
Train query context and compares the exact first-target float64 hash for every candidate family.
Source-name equality is never treated as a match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_operational_v23_runtime import build_operational_v23_controller
from rtc.direct_tfv_policy_return_portfolio_v23 import build_hybrid_policy_return_portfolio_v23
from rtc.direct_tfv_sequence_support import changed_facility_support_limit
from rtc.practical_rtc_assets import load_practical_rtc_asset_manifest, practical_asset_path


def _sha(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    import hashlib

    raw = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("records JSONL is empty or contains a non-object")
    return rows


def _context(path: Path) -> dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=False)
    out = {key: np.asarray(data[key]).copy() for key in (
        "current_state", "rainfall_scenarios", "active_target", "previous_actuator_flow"
    )}
    if out["current_state"].ndim == 2:
        out["current_state"] = out["current_state"][None]
    if out["rainfall_scenarios"].ndim == 4:
        out["rainfall_scenarios"] = out["rainfall_scenarios"][None]
    if out["active_target"].ndim == 1:
        out["active_target"] = out["active_target"][None]
    if out["previous_actuator_flow"].ndim == 1:
        out["previous_actuator_flow"] = out["previous_actuator_flow"][None]
    return out


def _generated_candidates(controller: object, context: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    mpc = controller.controller._direct_mpc_adapter.inner
    device = next(mpc.model.parameters()).device
    current_state = torch.as_tensor(context["current_state"], dtype=torch.float32, device=device)
    rainfall = torch.as_tensor(context["rainfall_scenarios"], dtype=torch.float32, device=device)
    if rainfall.ndim == 5 and rainfall.shape[0] == 1:
        rainfall = rainfall[0]
    flow = torch.as_tensor(context["previous_actuator_flow"], dtype=torch.float32, device=device)
    if flow.ndim == 1:
        flow = flow[None]
    active = torch.as_tensor(context["active_target"], dtype=torch.float32, device=device)
    if active.ndim == 2 and active.shape[0] == 1:
        active = active[0]
    ceiling = changed_facility_support_limit(mpc.sequence_support, "q95")
    with torch.inference_mode():
        portfolio = build_hybrid_policy_return_portfolio_v23(
            model=mpc.model,
            normalization=mpc.normalization,
            graph=mpc.graph,
            current_state=current_state,
            rainfall_scenarios=rainfall,
            previous_actuator_flow=flow,
            active_target=active,
            first_radius=mpc.first_radius,
            max_changed_facilities=int(ceiling),
            max_delta_per_update=float(mpc.design.max_setting_delta_per_update),
            probe_chunk_size=mpc.proposal_probe_chunk_size,
            supervisory_mask=mpc.supervisory_mask,
        )
        rows: list[dict[str, Any]] = []
        for proposal in portfolio.candidates:
            target, sequence, changed, support = mpc._h10_supported_target(proposal.target, active)
            if changed <= 0:
                continue
            rows.append(
                {
                    "candidate_source": str(proposal.source),
                    "candidate_first_target_sha256": _array_sha(target.detach().cpu().numpy()),
                    "first_move_changed_facility_count": int(changed),
                    "network_stress_q75": float(portfolio.hydraulic_diagnostics.network_stress_q75),
                    "strong_storm_blend": float(portfolio.hydraulic_diagnostics.strong_storm_blend),
                    "support": dict(support),
                    "target": target.detach().cpu().numpy().astype(np.float32).tolist(),
                    "sequence_shape": list(sequence.shape),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V25 candidate audit requested CUDA but CUDA is unavailable")

    records_path = Path(args.records_jsonl).resolve()
    rows = _records(records_path)
    assets = load_practical_rtc_asset_manifest(args.asset_manifest)
    device = torch.device(args.device)
    controller, _, _, lineage = build_operational_v23_controller(
        graph_path=practical_asset_path(assets, "graph"),
        sensors_path=practical_asset_path(assets, "sensors"),
        config_path=practical_asset_path(assets, "config"),
        step1_path=practical_asset_path(assets, "step1"),
        step2_path=practical_asset_path(assets, "step2"),
        supervisory_control_path=practical_asset_path(assets, "supervisory_control"),
        sequence_support_path=practical_asset_path(assets, "sequence_support"),
        v15_rank_checkpoint_path=args.v15_rank_checkpoint,
        v21_boundary_checkpoint_path=args.v21_boundary_checkpoint,
        device=device,
        decision_runtime_budget_seconds=180.0,
        proposal_probe_chunk_size=int(args.probe_chunk_size),
    )

    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_set_id"]), []).append(row)
    query_reports: list[dict[str, Any]] = []
    source_summary: dict[str, dict[str, int]] = {}
    for query_id, query_rows in sorted(by_query.items()):
        generated: list[dict[str, Any]] = []
        generated_by_source: dict[str, dict[str, Any]] = {}
        for source_row in query_rows:
            context = _context(Path(str(source_row["context_npz"])).resolve())
            source_generated = _generated_candidates(controller, context)
            for generated_row in source_generated:
                row_copy = dict(generated_row)
                row_copy["context_npz"] = str(source_row["context_npz"])
                row_copy["context_npz_sha256"] = _sha(str(source_row["context_npz"]))
                generated.append(row_copy)
                generated_by_source[str(generated_row["candidate_source"])] = row_copy
        existing = {
            str(row.get("candidate_source")): str(row.get("candidate_first_target_sha256"))
            for row in query_rows
        }
        generated_map = generated_by_source
        matches = {}
        for source, generated_row in generated_map.items():
            expected = str(generated_row["candidate_first_target_sha256"]).lower()
            observed = existing.get(source)
            matched = bool(observed and observed.lower() == expected)
            matches[source] = matched
            info = source_summary.setdefault(source, {"generated_count": 0, "existing_truth_present_count": 0, "exact_matched_count": 0})
            info["generated_count"] += 1
            if observed:
                info["existing_truth_present_count"] += 1
            if matched:
                info["exact_matched_count"] += 1
        query_reports.append(
            {
                "query_set_id": query_id,
                "event_id": str(query_rows[0]["event_id"]),
                "rainfall_group": str(query_rows[0]["rainfall_group"]),
                "generated_candidates": generated,
                "existing_candidate_sources": sorted(existing),
                "exact_match_by_source": matches,
                "hydraulic_exact_match": bool(matches.get("TYPE_AWARE_HYDRAULIC_PRESSURE", False)),
            }
        )
    for info in source_summary.values():
        info["unmatched_count"] = info["generated_count"] - info["exact_matched_count"]
    hydraulic_missing = [
        row["query_set_id"] for row in query_reports if not row["hydraulic_exact_match"]
    ]
    report = {
        "contract": "PROJECT7_V25_CURRENT_V23_CANDIDATE_HASH_LINEAGE_AUDIT_V1",
        "development_only": True,
        "formal_evidence": False,
        "new_swmm_runs": 0,
        "records_path": str(records_path),
        "records_sha256": _sha(records_path),
        "record_count": len(rows),
        "query_count": len(query_reports),
        "source_summary": source_summary,
        "query_reports": query_reports,
        "missing_current_v23_hydraulic_query_count": len(hydraulic_missing),
        "missing_current_v23_hydraulic_query_set_ids": hydraulic_missing,
        "v23_lineage": lineage,
        "step2_checkpoint_sha256": _sha(practical_asset_path(assets, "step2")),
        "v15_rank_checkpoint_sha256": _sha(args.v15_rank_checkpoint),
        "v21_boundary_checkpoint_sha256": _sha(args.v21_boundary_checkpoint),
        "asset_manifest_sha256": _sha(args.asset_manifest),
        "matching_rule": "candidate_first_target_sha256 exact float64 contiguous action hash; source name alone is insufficient",
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "records": len(rows),
        "queries": len(query_reports),
        "source_summary": source_summary,
        "missing_current_v23_hydraulic_query_count": len(hydraulic_missing),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
