"""Build label-independent D3-HOLD joint-sequence support for Direct-TFV Step3 V6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_sequence_support import derive_direct_tfv_sequence_support
from rtc.production_cli import _load_graph
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


CURRENT_SEQUENCE_SUPPORT_BUILD_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_SEQUENCE_SUPPORT_BUILD_V1"
EXPECTED_D3_FIT_GROUPS = 112


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    graph = _load_graph(args.graph)
    _, _, checkpoint = load_direct_tfv_runtime_checkpoint(
        args.checkpoint,
        graph=graph,
        device=torch.device("cpu"),
    )
    base = V60TrainCache(args.cache_manifest)
    fit, _ = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d3 = sorted(name for name in fit if name.startswith("D3::"))
    expected = int(checkpoint["action_support"].get("source_groups", {}).get("D3", -1))
    if len(fit_d3) != EXPECTED_D3_FIT_GROUPS or expected != len(fit_d3):
        raise ValueError(
            "D3 TrainFit group count differs from the Step2 checkpoint lineage: "
            f"split={len(fit_d3)} checkpoint={expected}"
        )
    payload = derive_direct_tfv_sequence_support(
        base,
        fit_d3,
        actuator_ids=graph.actuator_ids,
        control_block_steps=2,
        free_control_blocks=12,
    )
    payload["build_contract"] = CURRENT_SEQUENCE_SUPPORT_BUILD_CONTRACT
    payload["lineage"] = {
        "step2_checkpoint_sha256": _sha(args.checkpoint),
        "graph_sha256": _sha(args.graph),
        "cache_manifest_sha256": _sha(args.cache_manifest),
        "d3_fit_group_count": len(fit_d3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
