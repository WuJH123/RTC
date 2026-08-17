"""Stable current Project7 policy entrypoint; fail closed before production promotion."""
from __future__ import annotations

import argparse
import sys


CURRENT_RUNTIME_BLOCK_CONTRACT = "PROJECT7_CURRENT_RUNTIME_BLOCKED_UNTIL_DIRECT_TFV_SWMM_PROBE_V3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Current Project7 selection-aware Direct-TFV Step2 and screened trust-region Step3 are "
            "Development-only. Production runtime remains blocked until the solver audit and a "
            "small authoritative SWMM first-move/closed-loop probe pass."
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
        "Finish the selection-aware Step2 DEV gate, HOLD-threshold calibration, screened Step3 "
        "solver audit, and authoritative Development SWMM probe before production promotion."
    )


if __name__ == "__main__":
    main()
