"""Run the established group-balanced Step1 acceptance on a V122 checkpoint."""
from __future__ import annotations

import rtc.step1_accept_v3 as acceptance
from rtc.step1_runtime_v122 import load_step1_v122


def main() -> None:
    # The evaluator itself remains unchanged so V122 is measured on the same rainfall-group
    # balanced and dry/wet/flood-high evidence as the legacy architecture.
    acceptance._load_step1 = load_step1_v122
    acceptance.accept_step1_large_v3_main()


if __name__ == "__main__":
    main()
