"""Build the frozen V123 input normalization from causal TrainFit forecasts only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.step2_causal_rainfall_v123 import (
    derive_causal_input_normalization_v123,
    load_causal_forecast_store_v123,
)
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    deterministic_rainfall_split_v60,
)


def _digest_names(names: list[str]) -> str:
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_normalization(
    *, cache_manifest: str, causal_store: str, out: str, holdout_fraction: float = 0.20
) -> dict[str, object]:
    if abs(float(holdout_fraction) - 0.20) > 1e-12:
        raise ValueError("V123 normalization is frozen at holdout_fraction=0.20")
    cache = V60TrainCache(cache_manifest)
    store = load_causal_forecast_store_v123(causal_store)
    names = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=names, holdout_fraction=holdout_fraction
    )
    if set(fit) & set(holdout):
        raise ValueError("V123 normalization split overlap")
    normalization = derive_causal_input_normalization_v123(cache, store, fit)
    payload: dict[str, object] = {
        "contract": "PROJECT7_V123_CAUSAL_INPUT_NORMALIZATION_V1",
        "cache_manifest": str(Path(cache_manifest).resolve()),
        "causal_store": str(Path(causal_store).resolve()),
        "causal_store_sha256": hashlib.sha256(Path(causal_store).read_bytes()).hexdigest(),
        "source_tree_sha256": rtc_implementation_contract_sha256(),
        "holdout_fraction": float(holdout_fraction),
        "fit_group_count": len(fit),
        "holdout_group_count": len(holdout),
        "fit_group_sha256": _digest_names(fit),
        "holdout_group_sha256": _digest_names(holdout),
        "fit_events": sorted({cache.entry(name).event_id for name in fit}),
        "holdout_events": sorted({cache.entry(name).event_id for name in holdout}),
        "rainfall_source": {
            "kind": "V123_CAUSAL_FORECAST_STORE",
            "contract": store.forecast_contract,
            "future_realized_rainfall_used": False,
            "history_steps": 13,
            "model_step_seconds": 300,
            "horizon_steps": 72,
            "decay_per_step": 0.92,
            "scenario_multiplier": 1.0,
        },
        "state_mean": normalization.state_mean.tolist(),
        "state_std": normalization.state_std.tolist(),
        "flow_mean": normalization.flow_mean.tolist(),
        "flow_std": normalization.flow_std.tolist(),
        "rainfall_mean": normalization.rainfall_mean.tolist(),
        "rainfall_std": normalization.rainfall_std.tolist(),
        "boundary": {
            "new_swmm": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    path = Path(out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_normalization(**vars(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
