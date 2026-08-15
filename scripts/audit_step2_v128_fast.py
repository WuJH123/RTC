"""Fused ranking + H30-H360 hydraulic audit for a strict V128 checkpoint."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

from rtc.checkpoint_v128 import V128_CHECKPOINT_CONTRACT, load_step2_v128
from rtc.step2_differentiable_v128 import V128_STEP2_CONTRACT

V128_FAST_RANKING_CONTRACT = "PROJECT7_V128_CONTROL_ORIENTED_RANKING_AUDIT_V1_FAST_FUSED"
V128_FAST_HORIZON_CONTRACT = "PROJECT7_V128_HYDRAULIC_ROLLOUT_HORIZON_AUDIT_V1_FAST_FUSED"
V128_FAST_TELEMETRY_CONTRACT = "PROJECT7_V128_FAST_FUSED_DEVELOPMENT_EVIDENCE_V1"


def _load_v127_audit() -> ModuleType:
    path = Path(__file__).with_name("audit_step2_v127_fast.py")
    spec = importlib.util.spec_from_file_location("_rtc_v127_fast_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical V127 fast audit: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arg_path(name: str) -> Path:
    try:
        index = sys.argv.index(name)
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"V128 fast audit requires {name}") from exc


def _rewrite(path: Path, *, contract: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract"] = contract
    payload["step2_contract"] = V128_STEP2_CONTRACT
    payload["checkpoint_contract"] = V128_CHECKPOINT_CONTRACT
    boundary = payload.setdefault("boundary", {})
    boundary["v128_checkpoint_loader_required"] = True
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    runner = _load_v127_audit()
    runner.load_step2_v127 = load_step2_v128
    ranking = _arg_path("--ranking-out")
    horizon = _arg_path("--horizon-out")
    telemetry = _arg_path("--telemetry-out")
    runner.main()
    _rewrite(ranking, contract=V128_FAST_RANKING_CONTRACT)
    _rewrite(horizon, contract=V128_FAST_HORIZON_CONTRACT)
    _rewrite(telemetry, contract=V128_FAST_TELEMETRY_CONTRACT)


if __name__ == "__main__":
    main()
