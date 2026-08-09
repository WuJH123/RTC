from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rtc.fresh_workspace import (
    initialize_fresh_workspace,
    load_fresh_workspace,
    require_path_inside_workspace,
    validate_fresh_run_index,
)
from rtc.splits import assign_rainfall_group_splits


def _events(path: Path, inp: Path) -> Path:
    frame = pd.DataFrame(
        {
            "event_id": [f"e{i:03d}" for i in range(160)],
            "rainfall_group": [f"g{i:03d}" for i in range(160)],
            "inp_path": [str(inp)] * 160,
        }
    )
    assign_rainfall_group_splits(frame, seed=42).to_csv(path, index=False)
    return path


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "model.inp"
    inp.write_text("[OPTIONS]\nFLOW_UNITS CMS\n", encoding="utf-8")
    priority = tmp_path / "priority.txt"
    priority.write_text("N1\n", encoding="utf-8")
    events = _events(tmp_path / "events.csv", inp)
    root = tmp_path / "fresh"
    initialize_fresh_workspace(
        root=root,
        frozen_inp=inp,
        priority_nodes=priority,
        event_registry=events,
    )
    return root, root / "FRESH_WORKSPACE_MANIFEST.json"


def test_fresh_workspace_starts_empty_and_copies_event_registry(tmp_path: Path) -> None:
    root, manifest = _workspace(tmp_path)
    loaded = load_fresh_workspace(manifest)
    assert Path(loaded["canonical_event_registry"]).is_file()
    assert Path(loaded["canonical_event_registry"]).parent.parent == root.resolve()
    assert loaded["rainfall_design"]["rainfall_groups"] == 160
    assert loaded["contract"] == "RTC_FRESH_WORKSPACE_V1_NO_HISTORICAL_OUTPUT_REUSE"


def test_nonempty_output_root_is_rejected(tmp_path: Path) -> None:
    inp = tmp_path / "model.inp"
    inp.write_text("x", encoding="utf-8")
    priority = tmp_path / "priority.txt"
    priority.write_text("N1\n", encoding="utf-8")
    events = _events(tmp_path / "events.csv", inp)
    root = tmp_path / "old_outputs"
    root.mkdir()
    (root / "historical.pt").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="must start empty"):
        initialize_fresh_workspace(
            root=root,
            frozen_inp=inp,
            priority_nodes=priority,
            event_registry=events,
        )


def test_workspace_path_guard_rejects_historical_output(tmp_path: Path) -> None:
    root = tmp_path / "fresh"
    root.mkdir()
    inside = root / "models" / "step1.pt"
    inside.parent.mkdir()
    inside.write_text("new", encoding="utf-8")
    require_path_inside_workspace(inside, root)
    historical = tmp_path / "Project6" / "step1.pt"
    historical.parent.mkdir()
    historical.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="outside fresh workspace"):
        require_path_inside_workspace(historical, root)


def test_training_index_must_reference_only_fresh_branch_metadata(tmp_path: Path) -> None:
    root, workspace_manifest = _workspace(tmp_path)
    branch = root / "d0" / "e1.json"
    branch.parent.mkdir(parents=True)
    branch.write_text("{}", encoding="utf-8")
    index = root / "step1" / "run_index.csv"
    index.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "e1",
                "scientific_split": "development",
                "development_fold": "train",
                "metadata_path": str(branch),
            }
        ]
    ).to_csv(index, index=False)
    evidence = validate_fresh_run_index(
        run_index_path=index,
        workspace_manifest_path=workspace_manifest,
    )
    assert evidence["all_metadata_inside_workspace"] is True

    old_branch = tmp_path / "Project6" / "old.json"
    old_branch.parent.mkdir()
    old_branch.write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "event_id": "old",
                "scientific_split": "development",
                "development_fold": "train",
                "metadata_path": str(old_branch),
            }
        ]
    ).to_csv(index, index=False)
    with pytest.raises(ValueError, match="outside fresh workspace"):
        validate_fresh_run_index(
            run_index_path=index,
            workspace_manifest_path=workspace_manifest,
        )
