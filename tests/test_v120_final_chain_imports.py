from __future__ import annotations


def test_v120_final_chain_imports() -> None:
    from rtc.formal_final_v120 import main as final_main
    from rtc.policy_lock_v120 import main as lock_main
    from rtc.production_v120_bound import run_policy_v120_bound_main
    from rtc.promote_v120 import main as promote_main

    assert callable(run_policy_v120_bound_main)
    assert callable(promote_main)
    assert callable(lock_main)
    assert callable(final_main)
