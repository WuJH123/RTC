from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.build_step2_v125_d4_execution_manifest import _runner_blocks


def test_runner_blocks_require_identical_5min_pairs() -> None:
    ids = tuple(f"a{i}" for i in range(109))
    blocks = np.zeros((36, 109), dtype=np.float32)
    blocks[0] = 0.25
    blocks[1:] = 0.50
    sequence = np.repeat(blocks, 2, axis=0)
    settings, recovered = _runner_blocks(json.dumps(sequence.tolist()), ids)
    assert len(settings) == 36
    np.testing.assert_array_equal(recovered, blocks)
    assert set(settings[0]) == set(ids)
    assert settings[0]["a0"] == 0.25
    assert settings[1]["a0"] == 0.50


def test_runner_blocks_fail_closed_on_non_executable_5min_pair() -> None:
    ids = tuple(f"a{i}" for i in range(109))
    sequence = np.zeros((72, 109), dtype=np.float32)
    sequence[1, 0] = 0.1
    with pytest.raises(ValueError, match="not executable at 10-minute cadence"):
        _runner_blocks(json.dumps(sequence.tolist()), ids)
