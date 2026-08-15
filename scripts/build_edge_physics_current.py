"""Compile graph-edge aligned SWMM physics for smoke/dev edge-aware ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.edge_physics_current_v128 import EDGE_PHYSICS_CURRENT_CONTRACT, build_edge_physics_artifact_v128, save_edge_physics_artifact_v128
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
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "current_full_model_enabled": False,
        "inp_sha256": _sha(args.inp),
        "graph_sha256": _sha(args.graph),
        "edge_count": int(artifact.edge_index.shape[1]),
        "feature_count": int(artifact.edge_static_features.shape[1]),
        "feature_names": list(artifact.edge_static_feature_names),
        "parallel_edge_pairs_present": bool((artifact.physical_link_count > 1).any()),
        "npz": str(out_npz.resolve()),
        "promotion_rule": (
            "Use in smoke/dev edge-aware ablations only. Do not change the full model until "
            "held-out spatial action-effect/ranking evidence improves."
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
