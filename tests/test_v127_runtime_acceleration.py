from __future__ import annotations

import json
from pathlib import Path

from rtc.step2_eval_v127_fast import V127_FAST_EVAL_CONTRACT, _is_cuda_oom
from rtc.step2_gradient_v127_fast import V127_GRADIENT_FAST_CONTRACT


def test_cuda_oom_detection_is_narrow() -> None:
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not _is_cuda_oom(RuntimeError("ordinary shape mismatch"))


def test_acceleration_contracts_are_explicit() -> None:
    assert "FAST_FUSED_RANKING_HORIZON" in V127_FAST_EVAL_CONTRACT
    assert "CENTER_GRADIENT_REUSE" in V127_GRADIENT_FAST_CONTRACT


def test_canonical_execution_routes_to_fast_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs" / "v127_control_execution.json").read_text(encoding="utf-8"))
    assert payload["step2"]["fused_ranking_horizon_audit"] == "scripts/audit_step2_v127_fast.py"
    assert payload["step2"]["D2_gradient_audit"] == "scripts/audit_step2_v127_d2_gradients_fast.py"
    assert payload["step2"]["D5_gradient_finetune"] == "scripts/run_step2_v127_d5_gradient_fast.py"
    assert payload["runtime_acceleration"]["scientific_samples_removed"] is False
    assert payload["runtime_acceleration"]["existing_base_checkpoint_requires_retraining"] is False
