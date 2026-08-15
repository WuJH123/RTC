"""Fast group-local D2 TFV-gradient audit for a strict V128 checkpoint."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

from rtc.checkpoint_v128 import V128_CHECKPOINT_CONTRACT, load_step2_v128
from rtc.step2_differentiable_v128 import V128_STEP2_CONTRACT

V128_D2_FAST_GRADIENT_AUDIT_CONTRACT = (
    "PROJECT7_V128_INTERNAL_HOLDOUT_D2_GROUP_LOCAL_CAUSAL_GRADIENT_AUDIT_V1_FULL_GRADIENT_REUSE"
)


def _load_v127_audit() -> ModuleType:
    path = Path(__file__).with_name("audit_step2_v127_d2_gradients_fast.py")
    spec = importlib.util.spec_from_file_location("_rtc_v127_d2_fast_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical V127 D2 fast audit: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _out_dir() -> Path:
    try:
        index = sys.argv.index("--out-dir")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("V128 D2 gradient audit requires --out-dir") from exc


def main() -> None:
    runner = _load_v127_audit()
    runner.load_step2_v127 = load_step2_v128
    out = _out_dir()
    runner.main()
    path = out / "D2_INTERNAL_HOLDOUT_GRADIENT_METRICS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract"] = V128_D2_FAST_GRADIENT_AUDIT_CONTRACT
    payload["step2_contract"] = V128_STEP2_CONTRACT
    payload["checkpoint_contract"] = V128_CHECKPOINT_CONTRACT
    payload["predicted_gradient_semantics"] = (
        "smooth V128 typed-actuator optimization TFV full dJ/dU projected along exact D2 action direction"
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
