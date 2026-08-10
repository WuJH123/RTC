from __future__ import annotations

import math
import sys
from typing import Callable


def _set_or_verify(flag: str, expected: str, *, numeric: bool = False) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, expected])
        return
    pos = sys.argv.index(flag)
    if pos + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    actual = sys.argv[pos + 1]
    if numeric:
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Project7 v0.6.9 execution freeze requires {flag}={expected}, got {actual}"
            )
    elif actual != expected:
        raise ValueError(
            f"Project7 v0.6.9 execution freeze requires {flag}={expected}, got {actual}"
        )


def _reject_flag(flag: str, reason: str) -> None:
    if flag in sys.argv:
        raise ValueError(f"{flag} is forbidden by the Project7 execution freeze: {reason}")


def _delegate(main: Callable[[], None]) -> None:
    main()


def phase0_events_main() -> None:
    _set_or_verify("--groups", "6", numeric=True)
    _set_or_verify("--development-fold", "train")
    _set_or_verify("--seed", "42", numeric=True)
    _reject_flag("--all-events-per-group", "Phase-0 is exactly six events, one per rainfall group")
    from .phase0_design import main

    _delegate(main)


def checkpoint_design_main() -> None:
    _set_or_verify("--checkpoints-per-event", "4", numeric=True)
    _set_or_verify("--minimum-elapsed-minutes", "60", numeric=True)
    _set_or_verify("--minimum-tail-minutes", "360", numeric=True)
    _set_or_verify("--seed", "42", numeric=True)
    from .checkpoint_design import main

    _delegate(main)


def efficient_probe_design_main() -> None:
    _set_or_verify("--epsilon", "0.15", numeric=True)
    _set_or_verify("--actuators-per-checkpoint", "12", numeric=True)
    _set_or_verify("--seed", "42", numeric=True)
    _reject_flag("--no-center", "the frozen Phase-0 design includes the center/reference action")
    from .efficient_probe_design import main

    _delegate(main)


def _freeze_d1_arguments() -> None:
    _set_or_verify("--seed", "42", numeric=True)
    _set_or_verify("--control-start-minutes", "60", numeric=True)
    _set_or_verify("--perturbation-std", "0.12", numeric=True)
    _set_or_verify("--change-probability", "0.35", numeric=True)
    _set_or_verify("--max-delta", "0.20", numeric=True)


def d1_batch_main() -> None:
    _freeze_d1_arguments()
    from .large_data_cli import run_d1_batch_main

    _delegate(run_d1_batch_main)


def d1_exploration_main() -> None:
    _freeze_d1_arguments()
    from .d1_exploration import main

    _delegate(main)


def d3_design_main() -> None:
    _set_or_verify("--sequences-per-checkpoint", "8", numeric=True)
    _set_or_verify("--perturbation-std", "0.20", numeric=True)
    _set_or_verify("--change-probability", "0.25", numeric=True)
    _set_or_verify("--seed", "42", numeric=True)
    from .d3_design_cli import main

    _delegate(main)
