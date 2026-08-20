from __future__ import annotations

import inspect

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.step1_runtime_v127 import (
    V127_STEP1_RUNTIME_LOADER_CONTRACT,
    load_frozen_step1_v127,
)


def test_frozen_step1_runtime_loader_keeps_semantic_contract_but_uses_cpu_mmap() -> None:
    source = inspect.getsource(load_frozen_step1_v127)
    assert V127_STEP1_RUNTIME_LOADER_CONTRACT == (
        "PROJECT7_V127_FROZEN_STEP1_SEMANTIC_LOADER_V1"
    )
    assert 'map_location="cpu"' in source
    assert "mmap=True" in source
    assert "del state, payload" in source
    assert '"runtime_checkpoint_mmap": True' in source
    assert '"runtime_checkpoint_staged_on_cpu": True' in source


def test_direct_tfv_runtime_loader_drops_duplicate_cpu_model_weights() -> None:
    source = inspect.getsource(load_direct_tfv_runtime_checkpoint)
    assert 'map_location="cpu"' in source
    assert "mmap=True" in source
    assert 'runtime_payload.pop("model_state_dict", None)' in source
    assert 'runtime_payload["runtime_model_state_dict_retained"] = False' in source
    assert "del state_dict, payload" in source
