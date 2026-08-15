"""Fine-tune V128 on frozen D5-FIT directions and audit untouched D5-AUDIT."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

from rtc.checkpoint_v128 import (
    V128_CHECKPOINT_CONTRACT,
    load_step2_v128,
    save_step2_v128,
)
from rtc.step2_differentiable_v128 import V128_STEP2_CONTRACT

V128_D5_FAST_RUN_CONTRACT = (
    "PROJECT7_V128_D5_GRADIENT_FINETUNE_FAST_V1_TYPED_ACTUATOR_16GB_8GB"
)
V128_D5_REPORT_FILENAME = "STEP2_V128_D5_GRADIENT_REPORT.json"
V128_D5_CHECKPOINT_FILENAME = "step2_v128_d5_gradient.pt"


def _load_v127_runner() -> ModuleType:
    path = Path(__file__).with_name("run_step2_v127_d5_gradient_fast.py")
    spec = importlib.util.spec_from_file_location("_rtc_v127_d5_fast", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical V127 D5 fast runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _out_dir() -> Path:
    try:
        index = sys.argv.index("--out-dir")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("V128 D5 runner requires --out-dir") from exc


def main() -> None:
    runner = _load_v127_runner()
    out = _out_dir()
    runner.load_step2_v127 = load_step2_v128

    def save_v128_redirect(path, **kwargs):
        del path
        return save_step2_v128(out / V128_D5_CHECKPOINT_FILENAME, **kwargs)

    runner.save_step2_v127 = save_v128_redirect
    runner.V127_D5_FAST_RUN_CONTRACT = V128_D5_FAST_RUN_CONTRACT
    runner.main()

    old = out / "STEP2_V127_D5_GRADIENT_REPORT.json"
    new = out / V128_D5_REPORT_FILENAME
    if not old.is_file():
        raise RuntimeError("V128 D5 wrapper did not receive the canonical intermediate report")
    if new.exists():
        raise RuntimeError(f"refusing to overwrite existing V128 D5 report: {new}")
    payload = json.loads(old.read_text(encoding="utf-8"))
    payload["contract"] = V128_D5_FAST_RUN_CONTRACT
    payload["step2_contract"] = V128_STEP2_CONTRACT
    payload["checkpoint_contract"] = V128_CHECKPOINT_CONTRACT
    payload["base_checkpoint_contract_required"] = V128_CHECKPOINT_CONTRACT
    boundary = payload.setdefault("boundary", {})
    boundary["v128_typed_actuator_model"] = True
    old.unlink()
    new.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not (out / V128_D5_CHECKPOINT_FILENAME).is_file():
        raise RuntimeError("V128 D5 strict checkpoint was not created")


if __name__ == "__main__":
    main()
