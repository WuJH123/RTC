"""Stable current Project7 Step2 entrypoint.

The caller must explicitly choose ``--profile smoke|dev|full``. This prevents an accidental
multi-hour full training run during ordinary debugging. Versioned runners are internal or
archival; user/Codex workflows should call only this file.
"""
from run_step2_v128_current_profiles import main


if __name__ == "__main__":
    main()
