"""Build V125 D4 FIT/AUDIT direct candidate-minus-anchor TFV/PFV evidence.

No SWMM is run here.  The script reads the physically separate D4 caches and the accepted
V125 TFV/PFV checkpoints, reproduces the causal online Value inputs, and writes one row
per non-reference D4 candidate for one-sided calibration/audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_priority_value_v123 import PriorityValueCacheV123
from rtc.step2_train_response_v60 import V60TrainCache

try:  # pragma: no cover - invocation style
    from run_policy_v123 import _load_policy
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_policy_v123 import _load_policy

EVIDENCE_CONTRACT = "PROJECT7_V125_DIRECT_ANCHOR_ADVANTAGE_EVIDENCE_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _priority_nodes(path: str | Path) -> tuple[str, ...]:
    values = tuple(
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(values) != 8 or len(set(values)) != 8:
        raise ValueError("V125 evidence requires the frozen unique Priority8 node list")
    return values


def _rows_for_cache(
    *,
    split_role: str,
    base: V60TrainCache,
    causal: D4CausalForecastValueCacheV125,
    priority: PriorityValueCacheV123,
    policy,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in causal.names(D4_SOURCE_KIND):
        batch = causal.batch(name, policy.normalization, device)
        pfv_batch = priority.batch(name, policy.normalization, device)
        with torch.no_grad():
            output = policy.model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                policy.prepared,
            )
        pred_tfv = output.delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        pred_pfv = output.delta_pfv_m3[0].detach().cpu().numpy().astype(np.float64)
        truth_tfv = batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        truth_pfv = pfv_batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
        if not (pred_tfv.shape == pred_pfv.shape == truth_tfv.shape == truth_pfv.shape):
            raise RuntimeError(f"{name}: V125 evidence target shape drift")
        entry = base.entry(name)
        candidate_indices = list(entry.indices)
        if len(candidate_indices) != len(pred_tfv):
            raise RuntimeError(f"{name}: V125 cache candidate identity drift")
        shas = np.asarray(entry.arrays["action_or_sequence_sha256"])[candidate_indices]
        for i, raw_sha in enumerate(shas):
            sequence_sha = str(raw_sha)
            row_id = hashlib.sha256(
                f"{split_role}|{entry.event_id}|{entry.checkpoint_id}|{sequence_sha}".encode("utf-8")
            ).hexdigest()
            rows.append({
                "contract": EVIDENCE_CONTRACT,
                "split_role": split_role,
                "rainfall_group": str(entry.rainfall_group),
                "event_id": str(entry.event_id),
                "checkpoint_id": str(entry.checkpoint_id),
                "plan_row_id": row_id,
                "action_or_sequence_sha256": sequence_sha,
                "truth_tfv_advantage_m3": float(truth_tfv[i]),
                "predicted_tfv_advantage_m3": float(pred_tfv[i]),
                "truth_pfv_advantage_m3": float(truth_pfv[i]),
                "predicted_pfv_advantage_m3": float(pred_pfv[i]),
                "reference_semantics": "causal_sparse_rbc_anchor_exact_zero",
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--base-cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--priority-nodes", required=True)
    p.add_argument("--tfv-checkpoint", required=True)
    p.add_argument("--pfv-checkpoint", required=True)
    p.add_argument("--tfv-report", required=True)
    p.add_argument("--pfv-report", required=True)
    p.add_argument("--objective-report", required=True)
    p.add_argument("--v123-calibration-report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    policy, lineage = _load_policy(
        graph=graph,
        cache_manifest=args.base_cache_manifest,
        causal_store_path=args.causal_store,
        tfv_checkpoint=args.tfv_checkpoint,
        pfv_checkpoint=args.pfv_checkpoint,
        objective_report=args.objective_report,
        calibration_report=args.v123_calibration_report,
        tfv_report=args.tfv_report,
        pfv_report=args.pfv_report,
        policy_mode="hybrid",
        device=device,
    )
    store = load_causal_forecast_store_v123(args.causal_store)
    nodes = _priority_nodes(args.priority_nodes)
    fit_base = V60TrainCache(args.d4_fit_cache)
    audit_base = V60TrainCache(args.d4_audit_cache)
    fit = D4CausalForecastValueCacheV125(fit_base, store)
    audit = D4CausalForecastValueCacheV125(audit_base, store)
    fit_priority = PriorityValueCacheV123(fit, nodes)
    audit_priority = PriorityValueCacheV123(audit, nodes)
    rows = _rows_for_cache(
        split_role="fit", base=fit_base, causal=fit, priority=fit_priority,
        policy=policy, device=device,
    ) + _rows_for_cache(
        split_role="audit", base=audit_base, causal=audit, priority=audit_priority,
        policy=policy, device=device,
    )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty or frame["plan_row_id"].duplicated().any():
        raise RuntimeError("V125 anchor-relative evidence is empty or has duplicate identities")
    fit_rain = set(frame.loc[frame["split_role"] == "fit", "rainfall_group"].astype(str))
    audit_rain = set(frame.loc[frame["split_role"] == "audit", "rainfall_group"].astype(str))
    if fit_rain & audit_rain:
        raise RuntimeError("V125 evidence has D4 FIT/AUDIT rainfall leakage")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    report = {
        "contract": EVIDENCE_CONTRACT,
        "rows": int(len(frame)),
        "fit_rows": int((frame["split_role"] == "fit").sum()),
        "audit_rows": int((frame["split_role"] == "audit").sum()),
        "fit_rainfall_groups": sorted(fit_rain),
        "audit_rainfall_groups": sorted(audit_rain),
        "rainfall_overlap": sorted(fit_rain & audit_rain),
        "reference_semantics": "direct candidate-minus-causal-Sparse-RBC-anchor",
        "base_cache_sha256": _sha(args.base_cache_manifest),
        "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
        "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
        "tfv_checkpoint_sha256": _sha(args.tfv_checkpoint),
        "pfv_checkpoint_sha256": _sha(args.pfv_checkpoint),
        "runtime_loader_lineage": lineage,
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
