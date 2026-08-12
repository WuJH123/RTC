"""Fail-closed entrypoint for V6 development training.

This wrapper validates cache→shard→basis/design lineage before delegating to the
bounded implementation runner. Use this entrypoint, not run_step2_v60.py directly.
"""
from __future__ import annotations

import argparse
import json

from rtc.step2_shards_v60 import validate_v60_cache_lineage
from run_step2_v60 import main as run_v60_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-manifest", required=True)
    known, _ = parser.parse_known_args()
    lineage = validate_v60_cache_lineage(known.cache_manifest)
    print(json.dumps({"V6_CACHE_LINEAGE": lineage}, indent=2, sort_keys=True))
    run_v60_main()


if __name__ == "__main__":
    main()
