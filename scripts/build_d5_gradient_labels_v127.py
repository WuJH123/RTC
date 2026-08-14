"""Compile V127 D5 central-difference TFV/PFV gradient truth from authoritative SWMM.

The script never estimates gradients from a learned model.  It only joins the frozen D5
execution manifest to completed SWMM branches and computes (f_plus-f_minus)/(2*epsilon)
for each coefficient-space unit direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.d2_eval import exact_node_volumes
from rtc.d5_gradient_v127 import V127_D5_CONTRACT
from rtc.data_index import standardize_d3_run_index
from rtc.production_cli import _load_graph

V127_D5_LABEL_CONTRACT = "PROJECT7_V127_D5_AUTHORITATIVE_DIRECTIONAL_GRADIENT_LABELS_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _priority(path: str | Path, graph) -> np.ndarray:
    nodes = [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(nodes) != 8 or len(set(nodes)) != 8:
        raise ValueError("V127 D5 requires the frozen unique Priority8 list")
    index = {node: i for i, node in enumerate(graph.node_ids)}
    missing = [node for node in nodes if node not in index]
    if missing:
        raise ValueError(f"V127 D5 priority nodes absent from graph: {missing}")
    return np.asarray([index[node] for node in nodes], dtype=np.int64)


def build_labels(
    *, execution_manifest_path: str | Path, run_summary_path: str | Path,
    graph_path: str | Path, priority_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = pd.read_csv(execution_manifest_path)
    if set(manifest["v127_d5_contract"].astype(str)) != {V127_D5_CONTRACT}:
        raise ValueError("V127 D5 execution manifest contract mismatch")
    runs = standardize_d3_run_index(pd.read_csv(run_summary_path))
    provenance = manifest.rename(columns={"sequence_sha256": "action_or_sequence_sha256"})
    keys = ["checkpoint_id", "action_or_sequence_sha256"]
    joined = runs.merge(
        provenance,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_run", "_plan"),
    )
    if len(joined) != len(manifest):
        raise ValueError(f"V127 D5 SWMM summary matched {len(joined)} of {len(manifest)} rows")
    graph = _load_graph(graph_path)
    pidx = _priority(priority_path, graph)
    volume_cache: dict[str, np.ndarray] = {}

    def volumes(row: pd.Series) -> np.ndarray:
        path = str(row.get("metadata_path", row.get("metadata_path_run", "")))
        if not path:
            raise ValueError("V127 D5 run row lacks metadata_path")
        if path not in volume_cache:
            volume_cache[path] = exact_node_volumes(path, graph.node_ids)
        return volume_cache[path]

    rows: list[dict[str, object]] = []
    for center_id, center_group in joined.groupby("center_id", sort=False):
        center_rows = center_group[center_group["probe_role"] == "center"]
        if len(center_rows) != 1:
            raise RuntimeError(f"V127 D5 {center_id} lacks one authoritative center")
        center_row = center_rows.iloc[0]
        center_volume = volumes(center_row)
        center_tfv = float(center_volume.sum())
        center_pfv = float(center_volume[pidx].sum())
        probes = center_group[center_group["probe_role"].isin(["plus", "minus"])]
        for direction_id, pair in probes.groupby("direction_id", sort=False):
            plus = pair[pair["probe_role"] == "plus"]
            minus = pair[pair["probe_role"] == "minus"]
            if len(plus) != 1 or len(minus) != 1:
                raise RuntimeError(f"V127 D5 {direction_id} is not one completed +/- pair")
            plus_row, minus_row = plus.iloc[0], minus.iloc[0]
            eps = float(plus_row["epsilon"])
            if eps <= 0 or abs(float(minus_row["epsilon"]) - eps) > 1e-12:
                raise RuntimeError(f"V127 D5 {direction_id} epsilon drift")
            plus_volume = volumes(plus_row)
            minus_volume = volumes(minus_row)
            plus_tfv, minus_tfv = float(plus_volume.sum()), float(minus_volume.sum())
            plus_pfv, minus_pfv = float(plus_volume[pidx].sum()), float(minus_volume[pidx].sum())
            rows.append({
                "contract": V127_D5_LABEL_CONTRACT,
                "split_role": str(plus_row["d5_split_role"]),
                "rainfall_group": str(plus_row["rainfall_group_plan"] if "rainfall_group_plan" in plus_row else plus_row["rainfall_group"]),
                "event_id": str(plus_row["event_id_plan"] if "event_id_plan" in plus_row else plus_row["event_id"]),
                "checkpoint_id": str(plus_row["checkpoint_id"]),
                "center_id": str(center_id),
                "center_family": str(plus_row["center_family"]),
                "direction_id": str(direction_id),
                "direction_coefficients_json": str(plus_row["direction_coefficients_json"]),
                "epsilon": eps,
                "center_sequence_sha256": str(center_row["d5_scoring_sequence_sha256"]),
                "center_tfv_m3": center_tfv,
                "center_pfv_m3": center_pfv,
                "plus_tfv_m3": plus_tfv,
                "minus_tfv_m3": minus_tfv,
                "plus_pfv_m3": plus_pfv,
                "minus_pfv_m3": minus_pfv,
                "true_tfv_directional_gradient_m3_per_coeff": (plus_tfv - minus_tfv) / (2.0 * eps),
                "true_pfv_directional_gradient_m3_per_coeff": (plus_pfv - minus_pfv) / (2.0 * eps),
                "central_difference": True,
                "symmetric_pair_verified": True,
            })
    result = pd.DataFrame.from_records(rows)
    if result.empty or result["direction_id"].duplicated().any():
        raise RuntimeError("V127 D5 gradient label set is empty or duplicates directions")
    if set(result["split_role"].astype(str)) - {"fit", "audit"}:
        raise RuntimeError("V127 D5 labels have invalid FIT/AUDIT role")
    if (result.groupby("rainfall_group")["split_role"].nunique() != 1).any():
        raise RuntimeError("V127 D5 label rainfall leakage")
    summary = {
        "contract": V127_D5_LABEL_CONTRACT,
        "directions": len(result),
        "fit_directions": int((result["split_role"] == "fit").sum()),
        "audit_directions": int((result["split_role"] == "audit").sum()),
        "rainfall_groups": int(result["rainfall_group"].nunique()),
        "execution_manifest_sha256": _sha(execution_manifest_path),
        "run_summary_sha256": _sha(run_summary_path),
        "graph_sha256": _sha(graph_path),
        "priority_sha256": _sha(priority_path),
        "truth": "authoritative SWMM central finite difference in normalized coefficient directions",
        "audit_used_for_training": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    return result, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution-manifest", required=True)
    p.add_argument("--run-summary", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--priority-nodes", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    frame, summary = build_labels(
        execution_manifest_path=args.execution_manifest,
        run_summary_path=args.run_summary,
        graph_path=args.graph,
        priority_path=args.priority_nodes,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
