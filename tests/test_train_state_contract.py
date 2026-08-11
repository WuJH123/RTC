from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rtc.train_state import (
    _cpu_cuda_rng_states,
    restore_training_state,
    save_training_state,
    training_contract_sha,
)


def test_cuda_rng_states_are_normalized_to_cpu_byte_tensors() -> None:
    converted = _cpu_cuda_rng_states([torch.tensor([1, 257], dtype=torch.int64)])
    assert converted is not None
    assert len(converted) == 1
    assert converted[0].device.type == "cpu"
    assert converted[0].dtype == torch.uint8
    assert converted[0].tolist() == [1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA-mapped resume")
def test_cuda_mapped_training_state_restores_rng_state(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = torch.nn.Linear(2, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    contract, code_sha = training_contract_sha(
        "cuda-unit",
        {"data_sha256": "c" * 64, "epochs": 10, "learning_rate": 1e-3},
    )
    path = tmp_path / "cuda-state.pt"
    save_training_state(
        path,
        contract_sha256=contract,
        rtc_source_tree_sha256=code_sha,
        completed_epochs=2,
        model=model,
        optimizer=optimizer,
        scaler=None,
    )

    restored = torch.nn.Linear(2, 1).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    epoch, _ = restore_training_state(
        path,
        expected_contract_sha256=contract,
        expected_code_sha256=code_sha,
        model=restored,
        optimizer=restored_optimizer,
        scaler=None,
        map_location=device,
    )
    assert epoch == 2


def test_training_state_resumes_only_exact_same_experiment(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    contract, code_sha = training_contract_sha(
        "unit",
        {"data_sha256": "a" * 64, "epochs": 10, "learning_rate": 1e-3},
    )
    path = tmp_path / "state.pt"
    save_training_state(
        path,
        contract_sha256=contract,
        rtc_source_tree_sha256=code_sha,
        completed_epochs=3,
        model=model,
        optimizer=optimizer,
        scaler=None,
        extra_state={"history": [1.0, 0.8, 0.6]},
    )

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    epoch, extra = restore_training_state(
        path,
        expected_contract_sha256=contract,
        expected_code_sha256=code_sha,
        model=restored,
        optimizer=restored_optimizer,
        scaler=None,
        map_location="cpu",
    )
    assert epoch == 3
    assert extra["history"] == [1.0, 0.8, 0.6]

    wrong_contract, _ = training_contract_sha(
        "unit",
        {"data_sha256": "b" * 64, "epochs": 10, "learning_rate": 1e-3},
    )
    with pytest.raises(ValueError, match="different data/configuration"):
        restore_training_state(
            path,
            expected_contract_sha256=wrong_contract,
            expected_code_sha256=code_sha,
            model=restored,
            optimizer=restored_optimizer,
            scaler=None,
            map_location="cpu",
        )
