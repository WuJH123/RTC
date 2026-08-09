from __future__ import annotations

import argparse
import json
from pathlib import Path

from .code_contract import rtc_source_tree_sha256
from .formal_ranking import main as ranking_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    known, _ = parser.parse_known_args()
    ranking_main()
    out = Path(known.out)
    if not out.is_file():
        raise RuntimeError(f"ranking gate completed without evidence: {out}")
    payload = json.loads(out.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ranking evidence must be a JSON object")
    payload["rtc_source_tree_sha256"] = rtc_source_tree_sha256()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(out)


if __name__ == "__main__":
    main()
