"""Development-only trainer for the Project7 V7 historical joint-action interaction arm.

V7 keeps the V6 preservation lesson: MAIN trains the complete additive single-facility backbone, then
JOINT/CONTROL may update only the new historical interaction head.  No split, truth, loss, seed,
epoch count, learning rate, target scale, or causal input contract is changed relative to V5/V6.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .step2_tfv_value_historical_v7 import HistoricalInteractionTFVValueModelV7
from .step2_tfv_value_training import _epoch_group_order, _graph_tensors
from .step2_tfv_value_training_v4 import DirectTFVTrainingDesignV4, _group_loss_v4
from .step2_train_response_v60 import InputNormalizationV60


HISTORICAL_INTERACTION_TRAINING_CONTRACT = (
    "PROJECT7_DIRECT_TFV_HISTORICAL_ACTION_IDENTITY_INTERACTION_TRAINING_V7"
)
HISTORICAL_INTERACTION_UPDATE_POLICY = "MAIN_ALL_THEN_HISTORICAL_INTERACTION_ONLY"


def _set_trainable_historical_v7(
    model: HistoricalInteractionTFVValueModelV7,
    *,
    stage: str,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "main":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    if stage in {"joint", "control"}:
        for parameter in model.historical_interaction_head.parameters():
            parameter.requires_grad_(True)
        return
    raise ValueError(f"unknown historical V7 Direct-TFV stage: {stage}")


def _train_stage_historical_v7(
    model: HistoricalInteractionTFVValueModelV7,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    stage: str,
    mode: str,
    epochs: int,
    learning_rate: float,
    design: DirectTFVTrainingDesignV4,
    control_decision_loss: bool,
) -> list[dict[str, float | int | str]]:
    _set_trainable_historical_v7(model, stage=stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"historical V7 stage {stage!r} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(design.weight_decay),
    )
    static = _graph_tensors(graph, device)
    history: list[dict[str, float | int | str]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        totals: dict[str, list[float]] = {
            "loss": [],
            "regression": [],
            "ranking": [],
            "control_sign": [],
            "interaction_l1": [],
            "joint_density_count": [],
        }
        groups_used = branches_used = 0
        for source, name in _epoch_group_order(source_groups, epoch=epoch, seed=design.seed):
            batch = source_caches[source].batch(name, normalization, device)
            result = _group_loss_v4(
                model,
                batch,
                mode=mode,
                graph_tensors=static,
                design=design,
                control_decision_loss=control_decision_loss,
            )
            if result is None:
                continue
            loss, metrics = result
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(design.grad_clip))
            optimizer.step()
            groups_used += 1
            branches_used += int(metrics["branches"])
            for key in totals:
                totals[key].append(float(metrics[key]))
        if not totals["loss"]:
            raise RuntimeError(f"historical V7 stage {stage!r} found no usable branches")
        history.append(
            {
                "contract": HISTORICAL_INTERACTION_TRAINING_CONTRACT,
                "update_policy": HISTORICAL_INTERACTION_UPDATE_POLICY,
                "stage": stage,
                "epoch": epoch,
                "groups": int(groups_used),
                "branches": int(branches_used),
                **{key: float(np.mean(values)) for key, values in totals.items()},
            }
        )
    return history


def train_direct_tfv_value_model_historical_v7(
    model: HistoricalInteractionTFVValueModelV7,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: DirectTFVTrainingDesignV4 = DirectTFVTrainingDesignV4(),
) -> dict[str, list[dict[str, float | int | str]]]:
    """Run the single allowed V7 existing-data follow-up after V6 fails the offline gate."""
    if not isinstance(model, HistoricalInteractionTFVValueModelV7):
        raise TypeError("historical V7 trainer requires HistoricalInteractionTFVValueModelV7")
    design.validate()
    main = _train_stage_historical_v7(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="main",
        mode="single",
        epochs=design.main_epochs,
        learning_rate=design.learning_rate,
        design=design,
        control_decision_loss=False,
    )
    joint = _train_stage_historical_v7(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        stage="joint",
        mode="joint",
        epochs=design.interaction_epochs,
        learning_rate=design.interaction_learning_rate,
        design=design,
        control_decision_loss=False,
    )
    if "D3" not in source_groups or not source_groups["D3"]:
        raise RuntimeError("historical V7 CONTROL requires D3 HOLD-reference TrainFit groups")
    control = _train_stage_historical_v7(
        model,
        source_caches=source_caches,
        source_groups={"D3": source_groups["D3"]},
        normalization=normalization,
        graph=graph,
        device=device,
        stage="control",
        mode="all",
        epochs=design.control_epochs,
        learning_rate=design.control_learning_rate,
        design=design,
        control_decision_loss=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {"main": main, "joint": joint, "control": control}


__all__ = [
    "HISTORICAL_INTERACTION_TRAINING_CONTRACT",
    "HISTORICAL_INTERACTION_UPDATE_POLICY",
    "_set_trainable_historical_v7",
    "train_direct_tfv_value_model_historical_v7",
]
