from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .formal_final_v120 import main as _legacy_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    known, _ = parser.parse_known_args()
    lock = json.loads(Path(known.policy_lock).read_text(encoding="utf-8"))
    artifacts = lock.get("artefacts") if isinstance(lock, dict) else None
    if not isinstance(artifacts, dict) or "split_registry" not in artifacts:
        raise ValueError("strict V120 Final requires locked split registry")
    registry = pd.read_csv(str(artifacts["split_registry"]), keep_default_na=False)
    final = registry[registry["scientific_split"].astype(str) == "final"]
    if int(final["event_id"].astype(str).nunique()) != 6:
        raise ValueError("V120 Final requires six unique Final events")
    if int(final["rainfall_group"].astype(str).nunique()) != 6:
        raise ValueError("V120 Final requires six independent rainfall groups")
    _legacy_main()


if __name__ == "__main__":
    main()
