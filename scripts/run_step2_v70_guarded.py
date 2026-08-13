"""Fail-closed entrypoint for Project7 Step2 V7 development training."""
from __future__ import annotations

import argparse
import json

from rtc.step2_shards_v60 import validate_v60_cache_lineage
from run_step2_v70 import main as run_v70_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-manifest", required=True)
    known, _ = parser.parse_known_args()
    lineage = validate_v60_cache_lineage(known.cache_manifest)
    print(json.dumps({"V7_REUSED_V60_CACHE_LINEAGE": lineage}, indent=2, sort_keys=True))
    run_v70_main()


if __name__ == "__main__":
    main()
