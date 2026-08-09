from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import canonical_baseline_id
from .causal_timing import timing_from_controller_config
from .production_cli import run_policy_main


def main() -> None:
    # Parse only the fields needed for fail-fast timing validation, then delegate the exact
    # original CLI so there is one authoritative execution implementation.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config", required=True)
    known, _ = parser.parse_known_args()
    raw = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("controller config must be a JSON object")
    timing = timing_from_controller_config(raw)
    timing.validate(
        require_full_history_before_first_control=(
            canonical_baseline_id(known.strategy) == "proposed"
        )
    )
    run_policy_main()


if __name__ == "__main__":
    main()
