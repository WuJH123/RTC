"""Build the current Project7 native supervisory-control mask from one source SWMM INP.

The output keeps all 109 pretrained Step2 action channels but marks as online-controllable only
facilities that appear in explicit action clauses in ``[CONTROLS]``.  The current Wuhan testbed is
expected to resolve to 82 supervisory facilities. No SWMM simulation or performance label is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.native_supervisory_control import (
    PROJECT7_EXPECTED_SUPERVISORY_CONTROL_DIMENSION,
    derive_native_supervisory_control,
)
from rtc.production_cli import _load_graph


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inp", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--expected-control-dimension",
        type=int,
        default=PROJECT7_EXPECTED_SUPERVISORY_CONTROL_DIMENSION,
    )
    args = p.parse_args()

    graph = _load_graph(args.graph)
    payload = derive_native_supervisory_control(
        args.inp,
        actuator_ids=graph.actuator_ids,
        expected_control_dimension=int(args.expected_control_dimension),
    )
    payload["lineage"] = {
        "graph_path": str(Path(args.graph).resolve()),
        "graph_sha256": _sha(args.graph),
        "label_independent": True,
        "online_swmm_called": False,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
