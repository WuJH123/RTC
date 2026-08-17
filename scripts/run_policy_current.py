"""Stable current Project7 policy entrypoint; fail closed before production promotion."""
from __future__ import annotations

import argparse
import sys


CURRENT_RUNTIME_BLOCK_CONTRACT = (
    "PROJECT7_CURRENT_RUNTIME_BLOCKED_UNTIL_OPTIMIZER_CONSISTENT_SWMM_PROBE_V4"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Current Project7 Direct-TFV Step2, D3-HOLD joint-sequence support, optimizer-aware "
            "admission and Step3 V6 are Development-only. Production runtime remains blocked until "
            "fresh authoritative SWMM closed-loop and exact same-prefix replay evidence pass."
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
        f"{CURRENT_RUNTIME_BLOCK_CONTRACT}: production runtime is intentionally blocked. "
        "Build checkpoint-bound D3-HOLD joint-sequence support, pass Step3 V6 solver/audit gates, "
        "and show fresh Development SWMM/replay evidence before production promotion."
    )


if __name__ == "__main__":
    main()
