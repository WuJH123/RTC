"""Compatibility entrypoint for the canonical execution-bound V120 trainer.

All V120 training now delegates to ``run_step2_v120_final.py`` so there is only
one data-split, bundle-lineage and acceptance implementation.
"""
from __future__ import annotations

from run_step2_v120_final import main


if __name__ == "__main__":
    main()
