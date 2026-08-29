"""Audit residual weights on train/deployment-parity and mismatch feature groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtc.direct_tfv_v28_residual_value import V28_RESIDUAL_FEATURE_NAMES
from rtc.direct_tfv_v28r1_supported_residual import (
    V28R1_SUPPORTED_FEATURE_NAMES,
    V28R1_ZERO_WEIGHT_FEATURE_NAMES,
)


CONTRACT = "PROJECT7_STEP3_V28R1_RESIDUAL_FEATURE_PARITY_AUDIT_V1"


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    names = tuple(str(item) for item in value.get("feature_names", ()))
    if names != V28_RESIDUAL_FEATURE_NAMES:
        raise ValueError(f"residual feature contract drifted: {path}")
    weights = np.asarray(value.get("weight"), dtype=np.float64).reshape(-1)
    if weights.shape != (len(names),) or not np.isfinite(weights).all():
        raise ValueError(f"invalid residual weights: {path}")
    scales = np.asarray(value.get("feature_scale"), dtype=np.float64).reshape(-1)
    if scales.shape != weights.shape or not np.isfinite(scales).all():
        raise ValueError(f"invalid residual feature scales: {path}")
    return {"payload": value, "names": names, "weights": weights, "scales": scales}


def _report(path: Path) -> dict[str, Any]:
    loaded = _checkpoint(path)
    names = loaded["names"]
    weights = loaded["weights"]
    mismatch = {
        name: float(weights[names.index(name)]) for name in V28R1_ZERO_WEIGHT_FEATURE_NAMES
    }
    supported = {
        name: float(weights[names.index(name)]) for name in V28R1_SUPPORTED_FEATURE_NAMES
    }
    mismatch_values = np.asarray(list(mismatch.values()), dtype=np.float64)
    return {
        "path": str(path.resolve()),
        "contract": loaded["payload"].get("contract"),
        "ridge": float(loaded["payload"].get("ridge", 0.0)),
        "mismatch_feature_weights": mismatch,
        "supported_feature_weights": supported,
        "mismatch_nonzero_count": int(np.sum(np.abs(mismatch_values) > 1.0e-12)),
        "mismatch_weight_l1": float(np.sum(np.abs(mismatch_values))),
        "mismatch_weight_max_abs": float(np.max(np.abs(mismatch_values), initial=0.0)),
        "mismatch_weights_exact_zero": bool(np.all(mismatch_values == 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v28-checkpoint", required=True)
    parser.add_argument("--v28r1-checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "contract": CONTRACT,
        "v28": _report(Path(args.v28_checkpoint)),
        "v28r1": None,
        "development_only": True,
    }
    if args.v28r1_checkpoint:
        result["v28r1"] = _report(Path(args.v28r1_checkpoint))
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
