"""Fail-closed entrypoint for Project7 Step2 V8.0 development."""
from __future__ import annotations

import argparse
import json
import sys

from rtc.step2_shards_v60 import validate_v60_cache_lineage
from run_step2_v80 import main as run_v80_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-manifest", required=True)
    known, _ = parser.parse_known_args()
    lineage = validate_v60_cache_lineage(known.cache_manifest)
    print(json.dumps({"V8_CACHE_LINEAGE": lineage}, indent=2, sort_keys=True), flush=True)
    run_v80_main()


if __name__ == "__main__":
    main()
