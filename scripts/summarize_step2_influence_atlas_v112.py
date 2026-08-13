"""Export human-readable top actuator->node support from a V11.2 atlas NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.step2_influence_support_v112 import STATE_EFFECT_NAMES_V112
from rtc.step2_v110_contract import HydraulicHorizonV110
from run_step2_v110 import _load_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--majority-probability",
        type=float,
        default=0.50,
        help="Reporting threshold only; never a training/reachability gate.",
    )
    args = ap.parse_args()
    if args.top_k <= 0 or not 0.0 < args.majority_probability < 1.0:
        raise ValueError("invalid V112 reporting options")

    z = np.load(args.atlas, allow_pickle=False)
    graph = _load_graph(Path(args.graph))
    minutes = np.asarray(HydraulicHorizonV110().response_minutes(), dtype=np.float64)
    rows: list[dict[str, object]] = []

    for phase in ("overall", "low", "mid", "high"):
        key = f"{phase}__state_support_probability"
        exposure_key = f"{phase}__exposure"
        if key not in z or exposure_key not in z:
            raise KeyError(f"V112 atlas missing {key}/{exposure_key}")
        support = np.asarray(z[key], dtype=np.float64)  # [A,T,N,5]
        exposure = np.asarray(z[exposure_key], dtype=np.int64)
        if support.shape != (len(graph.actuator_ids), len(minutes), len(graph.node_ids), 5):
            raise ValueError(f"V112 atlas shape mismatch for {phase}: {support.shape}")
        for a, actuator_id in enumerate(graph.actuator_ids):
            if exposure[a] <= 0 or not np.isfinite(support[a]).all():
                continue
            for c, channel in enumerate(STATE_EFFECT_NAMES_V112):
                node_peak = support[a, :, :, c].max(axis=0)
                order = np.argsort(-node_peak, kind="mergesort")[: args.top_k]
                for rank, node in enumerate(order, 1):
                    series = support[a, :, node, c]
                    peak_t = int(np.argmax(series))
                    majority = np.flatnonzero(series >= args.majority_probability)
                    rows.append({
                        "hydraulic_phase": phase,
                        "source_actuator_index": a,
                        "source_actuator_id": str(actuator_id),
                        "probe_exposure": int(exposure[a]),
                        "channel": channel,
                        "rank": rank,
                        "node_index": int(node),
                        "node_id": str(graph.node_ids[node]),
                        "max_support_probability": float(series[peak_t]),
                        "mean_support_probability": float(series.mean()),
                        "peak_support_min": float(minutes[peak_t]),
                        "first_majority_support_min": float(minutes[majority[0]]) if majority.size else None,
                        "reporting_threshold_only": float(args.majority_probability),
                    })

    frame = pd.DataFrame.from_records(rows)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(
        f"[V112_ATLAS_SUMMARY] rows={len(frame)} actuators={frame.source_actuator_id.nunique()} "
        f"nodes={frame.node_id.nunique()} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
