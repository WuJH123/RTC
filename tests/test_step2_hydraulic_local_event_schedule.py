"""Correctness tests for the fair V9-event-schedule local D2 control."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_step2_hydraulic_local_event_schedule.py"
    spec = importlib.util.spec_from_file_location("step2_hydraulic_local_event_schedule", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_v9_event_schedule_has_56_optimizer_steps() -> None:
    module = _module()
    schedule = module.canonical_event_schedule(
        fit_d2_groups=112,
        fit_events=14,
        epochs=4,
    )
    assert schedule == {
        "fit_d2_groups": 112,
        "fit_events": 14,
        "groups_per_event": 8,
        "epochs": 4,
        "optimizer_updates_per_epoch": 14,
        "optimizer_updates_total": 56,
    }


def test_canonical_v9_event_schedule_rejects_noncanonical_exposure() -> None:
    module = _module()
    with pytest.raises(ValueError, match="112 D2 groups"):
        module.canonical_event_schedule(fit_d2_groups=111, fit_events=14, epochs=4)
    with pytest.raises(ValueError, match="14 TrainFit events"):
        module.canonical_event_schedule(fit_d2_groups=112, fit_events=13, epochs=4)
    with pytest.raises(ValueError, match="four epochs"):
        module.canonical_event_schedule(fit_d2_groups=112, fit_events=14, epochs=3)


def test_event_group_loss_is_mean_of_group_losses_not_row_weighted() -> None:
    module = _module()
    # The first group has one row with zero loss; the second has three rows with
    # smooth-L1 loss.  With beta=0.5, the nonzero rows each have loss 0.75.
    # A row-weighted reduction is 0.5625, while the V9 event contract is an
    # equal mean of the two checkpoint-group losses, 0.375.
    prediction = torch.zeros((4, 1), dtype=torch.float32)
    target = torch.as_tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)
    group_rows = [torch.as_tensor([0]), torch.as_tensor([1, 2, 3])]
    loss = module.event_group_mean_smooth_l1(prediction, target, group_rows)
    assert torch.isclose(loss, torch.as_tensor(0.375), atol=1e-7)


def test_event_group_rows_preserve_each_checkpoint_group() -> None:
    module = _module()
    dataset = {
        "event": np.asarray(["e1", "e1", "e1", "e2"], dtype=object),
        "group": np.asarray(["g1", "g1", "g2", "g3"], dtype=object),
    }
    grouped = module.event_group_rows(dataset)
    assert list(grouped) == ["e1", "e2"]
    assert [rows.tolist() for rows in grouped["e1"].values()] == [[0, 1], [2]]
    assert [rows.tolist() for rows in grouped["e2"].values()] == [[3]]


def test_actual_local_training_executes_one_equal_group_update_per_event() -> None:
    """The control must perform 14—not row-minibatch—updates per frozen epoch."""
    module = _module()
    groups = np.asarray([f"g{index:03d}" for index in range(112)], dtype=object)
    dataset = {
        "features": np.stack(
            [np.asarray([float(index), float(index % 8), 1.0], dtype=np.float32) for index in range(112)]
        ),
        "targets": np.ones((112, 7), dtype=np.float64),
        "event": np.asarray([f"e{index // 8:02d}" for index in range(112)], dtype=object),
        "group": groups,
    }
    schedule = module.canonical_event_schedule(fit_d2_groups=112, fit_events=14, epochs=4)
    predictions, metadata = module.fit_event_scheduled_local_mlp(
        dataset,
        (dataset["features"],),
        seed=42,
        epochs=4,
        device="cpu",
        expected_schedule=schedule,
    )
    assert predictions[0].shape == (112, 7)
    assert metadata["schedule"]["optimizer_updates_total"] == 56
    assert [row["event_updates"] for row in metadata["training_history"]] == [14, 14, 14, 14]
    assert [row["groups_per_event"] for row in metadata["training_history"]] == [8, 8, 8, 8]
