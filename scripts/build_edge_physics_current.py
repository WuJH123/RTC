"""Compile graph-edge-aligned SWMM physics for the current repaired smoke/dev Step2.

The artifact is derived only from the frozen INP and graph schema.  It contains no outcome
labels and can be reused across Development runs as long as its INP/graph hashes match.  The
current action-identifiable smoke/dev model requires this artifact; full remains blocked until
held-out Development evidence explicitly promotes the repaired model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.edge_physics_current_v128 import (
    EDGE_PHYSICS_CURRENT_CONTRACT,
    build_edge_physics_artifact_v128,
    save_edge_physics_artifact_v128,
)
from rtc.production_cli import _load_graph


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inp", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()
    graph = _load_graph(args.graph)
    artifact = build_edge_physics_artifact_v128(args.inp, graph)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_edge_physics_artifact_v128(artifact, str(out_npz))
    payload = {
        "contract": EDGE_PHYSICS_CURRENT_CONTRACT,
        "role": "REQUIRED_CURRENT_ACTION_IDENTIFIABLE_SMOKE_DEV_INPUT",
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "current_smoke_dev_required": True,
        "current_full_model_enabled": False,
        "inp_sha256": _sha(args.inp),
        "graph_sha256": _sha(args.graph),
        "edge_count": int(artifact.edge_index.shape[1]),
        "feature_count": int(artifact.edge_static_features.shape[1]),
        "feature_names": list(artifact.edge_static_feature_names),
        "parallel_edge_pairs_present": bool((artifact.physical_link_count > 1).any()),
        "npz": str(out_npz.resolve()),
        "contains_outcome_labels": False,
        "rebuild_rule": "rebuild only when frozen INP or graph schema hash changes",
        "promotion_rule": (
            "Required for current smoke/dev. It does not itself enable full; full remains "
            "blocked until held-out action-flow/gradient/ranking/spatial evidence improves."
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
