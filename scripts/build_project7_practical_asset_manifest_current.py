"""Freeze one path-safe absolute artifact manifest for Project7 Practical RTC.

The local supervisor is expected to discover the intended existing study assets from manifests and
prior reports, then call this script once. It never searches for replacements. Downstream commands
should reuse the generated manifest so a missing/stale path fails closed instead of falling back into
an older V* directory.
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
        "policy-admission",
        "v12-first-move-admission",
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
            "policy_admission": args.policy_admission,
            "v12_first_move_admission": args.v12_first_move_admission,
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
