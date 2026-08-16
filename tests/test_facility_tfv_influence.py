from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rtc.facility_tfv_influence import (
    extract_cache_influence_rows,
    sampled_cumulative_tfv_m3,
)


class _FakeCache:
    def __init__(self, entry):
        self._entry = entry

    def names(self):
        return ["D2::T20_D120_chicago::event_a::t3600"]

    def entry(self, name):
        assert name == "D2::T20_D120_chicago::event_a::t3600"
        return self._entry


def _fixture_entry():
    rows = 3
    state_steps = 72
    nodes = 1
    features = 3
    control_steps = 4
    actuators = 2

    initial = np.zeros((rows, nodes, features), dtype=np.float32)
    rainfall = np.zeros((rows, state_steps, nodes, 1), dtype=np.float32)
    previous_flow = np.zeros((rows, actuators), dtype=np.float32)
    states = np.zeros((rows, state_steps, nodes, features), dtype=np.float32)
    states[0, :, 0, 2] = 1.0
    states[1, :6, 0, 2] = 1.0
    states[1, 6:, 0, 2] = 0.5
    states[2, :, 0, 2] = 0.4

    settings = np.zeros((rows, control_steps, actuators), dtype=np.float32)
    settings[1, :, 0] = 0.2
    settings[2, :, 0] = 0.1
    settings[2, :, 1] = -0.1

    arrays = {
        "initial_state": initial,
        "rainfall": rainfall,
        "settings": settings,
        "previous_actuator_flow": previous_flow,
        "target_states": states,
        "exact_node_flood_volume_m3": np.asarray(
            [[20000.0], [10000.0], [9000.0]], dtype=np.float64
        ),
        "actuator_ids": np.asarray(
            [["PUMP_A", "ORIFICE_B"]] * rows
        ),
        "scientific_split": np.asarray(["development"] * rows),
        "action_or_sequence_sha256": np.asarray(["base", "single_a", "joint_ab"]),
        "base_action_sha256": np.asarray(["base"] * rows),
    }
    return SimpleNamespace(
        arrays=arrays,
        indices=(0, 1, 2),
        reference_index=0,
        source_kind="D2",
        rainfall_group="T20_D120_chicago",
        event_id="event_a",
        checkpoint_id="t3600",
    )


def test_sampled_cumulative_tfv_is_trapezoidal_and_monotone() -> None:
    initial = np.zeros((1, 3), dtype=np.float32)
    future = np.zeros((2, 1, 3), dtype=np.float32)
    future[:, 0, 2] = 1.0
    cumulative = sampled_cumulative_tfv_m3(
        initial,
        future,
        flood_rate_index=2,
        dt_seconds=300.0,
    )
    np.testing.assert_allclose(cumulative, np.asarray([150.0, 450.0]))
    assert np.all(np.diff(cumulative) >= 0)


def test_exact_single_actuator_is_attributed_but_joint_candidate_is_not() -> None:
    exact, joint, counters = extract_cache_influence_rows(
        _FakeCache(_fixture_entry()),
        source_label="fixture",
        meaningful_absolute_m3=1.0,
        meaningful_relative=0.01,
    )
    assert counters["single_actuator_pairs"] == 1
    assert counters["joint_actuator_pairs"] == 1
    assert len(exact) == 1
    assert len(joint) == 1

    row = exact[0]
    assert row["actuator_id"] == "PUMP_A"
    assert row["changed_actuator_count"] == 1
    assert row["delta_tfv_exact_m3"] == -10000.0
    assert row["effect_class"] == "BENEFICIAL"
    assert row["delayed_benefit_sampled"] is True
    assert row["delta_tfv_h30_sampled_m3"] == 0.0
    assert row["delta_tfv_h60_sampled_m3"] < 0.0

    joint_row = joint[0]
    assert joint_row["changed_actuator_count"] == 2
    assert set(str(joint_row["changed_actuator_ids"]).split("|")) == {"PUMP_A", "ORIFICE_B"}
    assert str(joint_row["attribution_semantics"]).startswith("JOINT_EFFECT_ONLY")


def test_prefix_mismatch_is_rejected_from_facility_attribution() -> None:
    entry = _fixture_entry()
    entry.arrays["initial_state"][1, 0, 0] = 0.01
    exact, _, counters = extract_cache_influence_rows(
        _FakeCache(entry),
        source_label="fixture",
    )
    assert len(exact) == 0
    assert counters["candidate_pairs_prefix_mismatch"] == 1
