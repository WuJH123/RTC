"""Build the Project7 82-DOF supervisory-control scope from the frozen native SWMM controls.

The output preserves the pretrained 109-channel Step2 representation but marks only links targeted by
native ``[CONTROLS]`` actions as online supervisory degrees of freedom.  No SWMM simulation is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.production_cli import _load_graph
from rtc.supervisory_control_scope import build_supervisory_control_scope


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--native-controls-inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    graph = _load_graph(args.graph)
    payload = build_supervisory_control_scope(
        actuator_ids=graph.actuator_ids,
        native_controls_inp=args.native_controls_inp,
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
