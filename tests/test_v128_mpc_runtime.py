from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rtc.engineering_v128 import V128EngineeringEnvelope
from rtc.runtime_evidence_v128 import audit_v128_runtime_decisions
from rtc.step3_mpc_v128 import (
    ContinuousMPCDesignV128,
    decode_fractional_targets_v128,
    encode_sequence_to_fraction_v128,
)


def _envelope() -> V128EngineeringEnvelope:
    ids = tuple(f"a{i}" for i in range(109))
    delta = np.full(109, 0.5, dtype=np.float64)
    delta[0], delta[1] = 0.1, 0.25
    env = V128EngineeringEnvelope(
        actuator_ids=ids,
        min_setting=np.zeros(109, dtype=np.float64),
        max_setting=np.ones(109, dtype=np.float64),
        max_delta_per_10min=delta,
        source="test",
        source_sha256="a" * 64,
    )
    env.validate()
    return env


def test_v128_decoder_enforces_per_actuator_slew_before_scoring() -> None:
    design = ContinuousMPCDesignV128()
    envelope = _envelope()
    active = torch.full((109,), 0.5)
    fractions = torch.ones(12, 109)
    sequence = decode_fractional_targets_v128(
        fractions,
        active_target=active,
        envelope=envelope,
        design=design,
    )
    blocks = sequence[::2]
    assert blocks.shape == (36, 109)
    assert float(blocks[0, 0] - active[0]) == pytest.approx(0.1, abs=1e-6)
    assert float(blocks[0, 1] - active[1]) == pytest.approx(0.25, abs=1e-6)
    assert float(blocks[0, 2] - active[2]) == pytest.approx(0.5, abs=1e-6)
    np.testing.assert_allclose(
        blocks[12:].numpy(),
        np.repeat(blocks[11:12].numpy(), 24, axis=0),
        atol=1e-7,
    )
    encoded = encode_sequence_to_fraction_v128(
        sequence.numpy(),
        active_target=active.numpy(),
        envelope=envelope,
        design=design,
    )
    # Once a target reaches a hard bound the feasible interval can have zero width, so the
    # latent fraction is not unique. The correct invariant is physical sequence identity.
    round_trip = decode_fractional_targets_v128(
        torch.as_tensor(encoded, dtype=active.dtype),
        active_target=active,
        envelope=envelope,
        design=design,
    )
    torch.testing.assert_close(round_trip, sequence, rtol=0.0, atol=1e-6)


def _write_decisions(path: Path, runtimes: list[float]) -> None:
    rows = []
    for index, runtime in enumerate(runtimes):
        rows.append(
            {
                "elapsed_seconds": 3600 + 600 * index,
                "source": "MPC_V128_CONTINUOUS",
                "diagnostics": {
                    "decision_runtime_seconds": runtime,
                    "optimizer_elapsed_seconds": max(0.0, runtime - 1.0),
                    "optimizer_deadline_exceeded": False,
                    "score_equals_execute": True,
                    "score_equals_execute_under_engineering_envelope": True,
                    "continuity_guard_passed": True,
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_v128_runtime_acceptance_requires_every_decision_below_600s(tmp_path) -> None:
    good = tmp_path / "good.jsonl"
    _write_decisions(good, [12.0, 20.0, 30.0])
    audit = audit_v128_runtime_decisions(good)
    assert audit["passed"] is True
    assert audit["decision_interval_exact_600s"] is True
    assert audit["decision_runtime_seconds"]["max"] == pytest.approx(30.0)

    bad = tmp_path / "bad.jsonl"
    _write_decisions(bad, [12.0, 601.0, 30.0])
    failed = audit_v128_runtime_decisions(bad)
    assert failed["passed"] is False
    assert failed["hard_realtime_max_lt_600s"] is False
