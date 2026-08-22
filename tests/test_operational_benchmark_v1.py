from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

from rtc.direct_tfv_operational_v21_runtime import load_v21_calibrator
from rtc.direct_tfv_policy_return_selected_boundary_v21 import DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT, SelectedBoundaryPartsV21
from rtc.operational_benchmark_v1 import EVENT_COUNT, OPERATIONAL_COMPARATORS


def _script(name: str):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module(name)


def test_operational_benchmark_is_five_events_and_four_competitive_baselines() -> None:
    assert EVENT_COUNT == 5
    assert OPERATIONAL_COMPARATORS == ("no_control", "internal_rtc", "auto_rbc", "efd")


def test_forcing_only_selector_covers_existing_rp_duration_range(tmp_path: Path) -> None:
    module = _script("freeze_project7_operational_benchmark5_current")
    names = [
        "PRR_RP002_D030_chicago.inp",
        "PRR_RP005_D060_chicago.inp",
        "PRR_RP010_D120_huff.inp",
        "PRR_RP025_D180_huff.inp",
        "PRR_RP050_D240_chicago.inp",
        "PRR_RP100_D360_huff.inp",
        "PRR_RP200_D480_chicago.inp",
    ]
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text("[TITLE]\n", encoding="utf-8")
        paths.append(path)
    selected = module._auto_select(paths)
    assert len(selected) == 5
    assert len(set(selected)) == 5
    rp = [module._descriptor(path)[0] for path in selected]
    assert min(rp) <= 5
    assert max(rp) >= 100


def test_operational_loader_may_read_failed_v21_only_as_development_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v21.pt"
    torch.save(
        {
            "contract": DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT,
            "development_only": True,
            "feature_scale": torch.ones(3),
            "svd_components": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "boundary_weight": torch.tensor([1.0, -1.0]),
            "magnitude_weight": torch.tensor([1.0, 1.0]),
            "target_scale_m3": 1000.0,
            "train_oof_boundary_supported": False,
        },
        checkpoint,
    )
    model, payload = load_v21_calibrator(checkpoint, device=torch.device("cpu"))
    assert payload["train_oof_boundary_supported"] is False
    output = model.predict(SelectedBoundaryPartsV21(feature=torch.zeros(3)))
    assert float(output.hold_score) == 0.0
    assert float(output.advantage_m3) == 0.0
    assert bool(output.execute) is False
