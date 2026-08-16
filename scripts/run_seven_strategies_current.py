"""Stable current seven-strategy entrypoint; fail closed before production promotion."""
from __future__ import annotations

import argparse
import sys


CURRENT_SEVEN_STRATEGY_BLOCK_CONTRACT = (
    "PROJECT7_CURRENT_SEVEN_STRATEGY_BLOCKED_UNTIL_COUNTERFACTUAL_PRODUCTION_LOADER_V1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Current Project7 counterfactual-first Step2 is Development-only. The authoritative "
            "seven-strategy SWMM comparison is intentionally blocked until the current surrogate "
            "is explicitly promoted with a matching production checkpoint loader/runtime."
        )
    )
    parser.add_argument(
        "--promotion-status",
        action="store_true",
        help="print the fail-closed production-promotion state",
    )
    argv = list(sys.argv[1:])
    if any(value in {"-h", "--help"} for value in argv):
        parser.parse_args(argv)
        return
    known, _ = parser.parse_known_args(argv)
    if known.promotion_status:
        print(CURRENT_SEVEN_STRATEGY_BLOCK_CONTRACT)
        print("seven_strategy_enabled=false")
        return
    raise RuntimeError(
        f"{CURRENT_SEVEN_STRATEGY_BLOCK_CONTRACT}: current smoke/dev checkpoints are NONFINAL and "
        "cannot be compared as the Proposed production policy. Finish Development gates and add "
        "the matching production checkpoint loader/runtime before the authoritative seven-strategy run."
    )


if __name__ == "__main__":
    main()
