"""Build the rebuildable mmap cache only from a V6 lineage-bound shard manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_training_cache import build_step2_training_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if source.get("v60_contract") is None:
        raise ValueError("refusing to build V6 cache from a non-V6 shard manifest")
    cache = build_step2_training_cache(args.manifest, args.out_dir, force=args.force)
    lineage = validate_v60_cache_lineage(cache)
    print(json.dumps({"contract": "PROJECT7_STEP2_V60_CACHE_BUILD_V1", "cache_manifest": str(cache), "lineage": lineage}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
