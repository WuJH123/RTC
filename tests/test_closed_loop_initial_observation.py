from __future__ import annotations

from datetime import datetime

from rtc.closed_loop import _observation_time


class _NotRunningSimulation:
    start_time = datetime(2022, 8, 10, 22, 0, 0)

    @property
    def current_time(self):
        raise RuntimeError("Simulation Not Running")


def test_initial_observation_uses_start_time_before_simulation_runs() -> None:
    sim = _NotRunningSimulation()
    assert _observation_time(sim, 0) == sim.start_time


def test_running_observation_uses_current_time() -> None:
    class RunningSimulation(_NotRunningSimulation):
        current_time = datetime(2022, 8, 10, 22, 5, 0)

    sim = RunningSimulation()
    assert _observation_time(sim, 300) == sim.current_time
