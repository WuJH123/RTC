from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_v122_wrapper_rejects_gradient_search_candidate_policy(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"v120_contract": "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1"}),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle.pt"
    import torch

    torch.save(
        {
            "candidate_policy": {"continuous_gradient_search": True},
            "value_gate": {"passed": True},
        },
        bundle,
    )
    import sys
    from scripts import run_policy_v122

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_policy_v122.py", "--config", str(config), "--step2", str(bundle)],
    )
    with pytest.raises(ValueError, match="continuous gradient search disabled"):
        run_policy_v122.main()
