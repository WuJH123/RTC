"""Freeze one path-safe absolute artifact manifest for current Project7 Practical RTC.

Local Codex first discovers the intended existing study assets from manifests/reports, then calls this
script once. It never searches for replacements. Historical V12 admission files are deliberately not
part of the current manifest because the first paired-label parent is the Practical base-H10-probe
policy and the deployed policy uses only its H10 policy-return critic/admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.practical_rtc_assets import build_practical_rtc_asset_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "graph",
        "sensors",
        "config",
        "step1",
        "step2",
        "sequence-support",
        "priority8",
    ):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    payload = build_practical_rtc_asset_manifest(
        {
            "graph": args.graph,
            "sensors": args.sensors,
            "config": args.config,
            "step1": args.step1,
            "step2": args.step2,
            "sequence_support": args.sequence_support,
            "priority8": args.priority8,
        }
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["manifest_path"] = str(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
