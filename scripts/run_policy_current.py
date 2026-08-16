"""Stable current Project7 policy entrypoint; fail closed before production promotion."""
from __future__ import annotations

import argparse
import sys


CURRENT_RUNTIME_BLOCK_CONTRACT = (
    "PROJECT7_CURRENT_RUNTIME_BLOCKED_UNTIL_COUNTERFACTUAL_PRODUCTION_LOADER_V1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Current Project7 counterfactual-first Step2 is Development-only. Authoritative "
            "closed-loop runtime is intentionally blocked until an explicit production model "
            "checkpoint factory/loader and matching controller contract are promoted."
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
        print(CURRENT_RUNTIME_BLOCK_CONTRACT)
        print("runtime_enabled=false")
        return
    raise RuntimeError(
        f"{CURRENT_RUNTIME_BLOCK_CONTRACT}: current smoke/dev checkpoints are NONFINAL and no "
        "matching production checkpoint loader exists. Do not route an old base-V128 checkpoint "
        "as the current Proposal. Finish Development gates, then implement an explicit production "
        "promotion before authoritative SWMM runtime."
    )


if __name__ == "__main__":
    main()
