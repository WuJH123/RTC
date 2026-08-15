from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rtc.engineering_v128 import (
    V128_IDEALIZED_ENVELOPE_SOURCE,
    idealized_engineering_envelope_v128,
    load_engineering_envelope_v128,
    save_idealized_engineering_envelope_v128,
)
from rtc.runtime import command_continuity
from rtc.runtime_controller_guard import _delta_vector


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        actuator_ids=("pump_A", "orifice_B", "weir_C"),
        actuator_physics_feature_names=(
            "is_pump",
            "min_setting",
            "max_setting",
        ),
        actuator_physics=np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
                [0.0, 0.0, 0.8],
            ],
            dtype=np.float32,
        ),
    )


def test_v128_idealized_envelope_is_explicit_not_field_claim(tmp_path) -> None:
    graph = _graph()
    envelope = idealized_engineering_envelope_v128(graph)
    assert envelope.source == V128_IDEALIZED_ENVELOPE_SOURCE
    assert envelope.is_idealized_default is True
    np.testing.assert_allclose(envelope.max_delta_per_10min, 0.5)
    assert len(envelope.semantic_sha256) == 64

    path = save_idealized_engineering_envelope_v128(graph, tmp_path / "envelope.json")
    loaded = load_engineering_envelope_v128(path, graph=graph)
    assert loaded.actuator_ids == graph.actuator_ids
    assert loaded.is_idealized_default is True
    np.testing.assert_allclose(loaded.min_setting, [0.0, 0.1, 0.0])
    np.testing.assert_allclose(loaded.max_setting, [1.0, 0.9, 0.8])


def test_per_actuator_delta_vector_changes_continuity_by_device() -> None:
    delta = _delta_vector(np.asarray([0.1, 0.3, 0.5]), 3)
    requested = np.asarray([0.2, 0.2, 0.4])
    previous = np.zeros(3)
    result = command_continuity(
        requested,
        current_settings=np.zeros(3),
        previous_requested_settings=previous,
        max_delta_per_update=delta,
        enforce_current_delta=False,
    )
    assert result.passed is False
    assert result.failed_previous_indices == (0,)


def test_engineering_envelope_fails_closed_on_wrong_actuator_order(tmp_path) -> None:
    graph = _graph()
    path = save_idealized_engineering_envelope_v128(graph, tmp_path / "envelope.json")
    changed = SimpleNamespace(
        **{
            **graph.__dict__,
            "actuator_ids": ("orifice_B", "pump_A", "weir_C"),
        }
    )
    with pytest.raises(ValueError, match="order differs"):
        load_engineering_envelope_v128(path, graph=changed)
