from __future__ import annotations

import argparse
import json
from pathlib import Path

from .code_contract import rtc_source_tree_sha256
from .formal_gradient_v2 import main as gradient_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics-out", required=True)
    known, _ = parser.parse_known_args()
    gradient_main()
    out = Path(known.metrics_out)
    if not out.is_file():
        raise RuntimeError(f"gradient metric generation completed without evidence: {out}")
    payload = json.loads(out.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gradient metrics must be a JSON object")
    payload["rtc_source_tree_sha256"] = rtc_source_tree_sha256()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(out)


if __name__ == "__main__":
    main()
