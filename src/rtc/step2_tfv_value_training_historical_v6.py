"""Development-only Direct-TFV retraining inspired by the strongest historical Step2 backbone.

This module intentionally does NOT replace the frozen V23/V5 publication policy.  It provides one
falsifiable experimental arm that tests a specific lesson from the V4.1/V4.2 development history:
a strong single-facility (D2-like) response backbone should not be overwritten while learning dense
joint actions.  The current V5 trainer updates the shared facility representation in JOINT and
CONTROL; this arm freezes the complete main-value backbone after MAIN and updates only the
interaction head thereafter.

The scientific target, data split, exact SWMM delta-TFV labels, causal state/rainfall inputs, 109
actuator contract and loss definitions are unchanged.  If this arm does not improve internal
holdout D3/decision metrics without regressing D2, it must be rejected rather than promoted.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .step2_tfv_value import DirectFacilityTFVValueModel
from .step2_tfv_value_training import _epoch_group_order, _graph_tensors
from .step2_tfv_value_training_v4 import (
    DirectTFVTrainingDesignV4,
    _group_loss_v4,
)
from .step2_train_response_v60 import InputNormalizationV60


HISTORICAL_RETRAIN_CONTRACT = "PROJECT7_DIRECT_TFV_HISTORICAL_PRESERVE_MAIN_RETRAIN_V6"
HISTORICAL_UPDATE_POLICY = "MAIN_ALL_THEN_INTERACTION_ONLY"


def _set_trainable_historical(model: DirectFacilityTFVValueModel, *, stage: str) -> None:
    """Freeze the main-value pathway after MAIN; only joint interaction may adapt later."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "main":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    if stage in {"joint", "control"}:
        for parameter in model.interaction_head.parameters():
            parameter.requires_grad_(True)
        return
    raise ValueError(f"unknown historical Direct-TFV stage: {stage}")


def _train_stage_historical(
    model: DirectFacilityTFVValueModel,
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
    _set_trainable_historical(model, stage=stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"historical Direct-TFV stage {stage!r} has no trainable parameters")
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
            raise RuntimeError(f"historical Direct-TFV stage {stage!r} found no usable branches")
        history.append(
            {
                "contract": HISTORICAL_RETRAIN_CONTRACT,
                "update_policy": HISTORICAL_UPDATE_POLICY,
                "stage": stage,
                "epoch": epoch,
                "groups": int(groups_used),
                "branches": int(branches_used),
                **{key: float(np.mean(values)) for key, values in totals.items()},
            }
        )
    return history


def train_direct_tfv_value_model_historical_v6(
    model: DirectFacilityTFVValueModel,
    *,
    source_caches: Mapping[str, Any],
    source_groups: Mapping[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: DirectTFVTrainingDesignV4 = DirectTFVTrainingDesignV4(),
) -> dict[str, list[dict[str, float | int | str]]]:
    """Run the preservation-aware arm on the same existing TrainFit data as V5."""
    design.validate()
    main = _train_stage_historical(
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
    joint = _train_stage_historical(
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
        raise RuntimeError("historical Direct-TFV CONTROL stage requires D3 HOLD-reference TrainFit groups")
    control = _train_stage_historical(
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
    "HISTORICAL_RETRAIN_CONTRACT",
    "HISTORICAL_UPDATE_POLICY",
    "_set_trainable_historical",
    "train_direct_tfv_value_model_historical_v6",
]
