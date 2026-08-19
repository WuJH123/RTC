"""Freeze one path-safe absolute artifact manifest for current Project7 Practical RTC.

Local Codex first discovers the intended existing frozen assets, then builds the label-independent
native supervisory-control artifact and masked q95 sequence support. The resulting manifest is the
only downstream path source. Historical V12 admissions remain excluded.
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
        "supervisory-control",
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
            "supervisory_control": args.supervisory_control,
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
