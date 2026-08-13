import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.run_step2_v100_nonlocal_d2 import _checkpoint_payload_v100


def test_v100_checkpoint_payload_is_development_only_and_cpu_serializable():
    model = torch.nn.Linear(2, 1)
    payload = _checkpoint_payload_v100(
        model,
        lineage={"graph_sha256": "g", "cache_manifest_sha256": "c"},
        schedule={"seed": 42, "epochs": 4},
    )
    assert payload["production_compatible"] is False
    assert payload["contract"] == "PROJECT7_STEP2_V100_NONLOCAL_HYDRAULIC_OPERATOR_V1"
    assert all(not tensor.is_cuda for tensor in payload["state_dict"].values())
    assert payload["lineage"]["graph_sha256"] == "g"
