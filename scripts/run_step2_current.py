"""Stable current Project7 Step2 entrypoint.

User/Codex workflows should call this file rather than selecting a versioned trainer.
The selected implementation is pinned by configs/step2_current_contract.json and
configs/project7_execution_registry.json.
"""
from run_step2_v128_control_4060 import main


if __name__ == "__main__":
    main()
