"""Development-only NG1 Direct-TFV value model.

NG1-R3 keeps the frozen D2 facility main-value path and replaces both R1's
uniform all-pair pooling and R2's hard changed-changed mask with an
action-conditioned adaptive-connectivity interaction.  Pairs with at least one
changed actuator remain eligible so that changed-to-hydraulic-context
propagation is retained, while learned attention suppresses irrelevant pairs.
The D2 structural identity remains exact: a single changed facility receives no
interaction residual.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn

from .step2_tfv_value import DirectFacilityTFVValueModel, DirectTFVValueDesign


NG1_CONTRACT = "PROJECT7_STEP2_NG1_D2_PRESERVED_ADAPTIVE_CONNECTIVITY_V3"
NG1_GRAPH_CONTRACT = "PROJECT7_STEP2_NG1_COMPLETE_109_ACTUATOR_PAIR_GRAPH_V2_NORMALIZED_PHYSICS"


def _shortest_paths(edge_index: np.ndarray, node_count: int) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError("graph edge_index must be a rank-2 array")
    if edges.shape[0] == 2:
        pairs = edges.T
    elif edges.shape[1] == 2:
        pairs = edges
    else:
        raise ValueError("graph edge_index must have one dimension of length 2")
    for left, right in pairs.tolist():
        if 0 <= int(left) < node_count and 0 <= int(right) < node_count:
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
    result = np.full((node_count, node_count), np.inf, dtype=np.float64)
    for source in range(node_count):
        result[source, source] = 0.0
        queue: deque[int] = deque([source])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if not np.isfinite(result[source, neighbour]):
                    result[source, neighbour] = result[source, current] + 1.0
                    queue.append(neighbour)
    return result


def build_control_interaction_graph(graph: Any) -> dict[str, Any]:
    """Build a complete deterministic, label-independent actuator-pair graph."""
    actuator_ids = tuple(str(value) for value in graph.actuator_ids)
    if len(actuator_ids) != 109 or len(set(actuator_ids)) != 109:
        raise ValueError("NG1 requires 109 unique actuator IDs")
    upstream = np.asarray(graph.actuator_upstream, dtype=np.int64).reshape(-1)
    downstream = np.asarray(graph.actuator_downstream, dtype=np.int64).reshape(-1)
    physics = np.asarray(graph.actuator_physics, dtype=np.float64)
    if upstream.shape != (109,) or downstream.shape != (109,):
        raise ValueError("NG1 actuator endpoint arrays must have length 109")
    if physics.ndim != 2 or physics.shape[0] != 109 or not np.isfinite(physics).all():
        raise ValueError("NG1 actuator physics must be finite [109,feature]")
    node_count = len(tuple(getattr(graph, "node_ids", ())))
    if node_count <= 0:
        node_count = int(max(upstream.max(initial=0), downstream.max(initial=0)) + 1)
    distances = _shortest_paths(np.asarray(graph.edge_index), node_count)
    max_distance = max(float(node_count), 1.0)
    physics_scale = np.maximum(physics.std(axis=0, ddof=0), 1.0e-6)
    feature_names = (
        "same_upstream",
        "same_downstream",
        "downstream_to_other_upstream",
        "other_downstream_to_upstream",
        "endpoint_shortest_path_normalized",
        *(f"physics_abs_difference_standardized_{index}" for index in range(physics.shape[1])),
    )
    indices: list[tuple[int, int]] = []
    features: list[list[float]] = []
    for left in range(109):
        for right in range(left + 1, 109):
            endpoint_distance = min(
                distances[upstream[left], upstream[right]],
                distances[upstream[left], downstream[right]],
                distances[downstream[left], upstream[right]],
                distances[downstream[left], downstream[right]],
            )
            if not np.isfinite(endpoint_distance):
                endpoint_distance = max_distance + 1.0
            relation = [
                float(upstream[left] == upstream[right]),
                float(downstream[left] == downstream[right]),
                float(downstream[left] == upstream[right]),
                float(downstream[right] == upstream[left]),
                float(endpoint_distance / max_distance),
            ]
            relation.extend(
                (np.abs(physics[left] - physics[right]) / physics_scale)
                .astype(np.float64)
                .tolist()
            )
            indices.append((left, right))
            features.append(relation)
    pair_indices = np.asarray(indices, dtype=np.int64)
    pair_features = np.asarray(features, dtype=np.float32)
    if pair_indices.shape != (109 * 108 // 2, 2):
        raise RuntimeError("NG1 complete pair graph does not contain 5886 pairs")
    if pair_features.shape != (pair_indices.shape[0], len(feature_names)):
        raise RuntimeError("NG1 pair relation feature shape is inconsistent")
    payload = {
        "contract": NG1_GRAPH_CONTRACT,
        "actuator_ids": list(actuator_ids),
        "actuator_upstream": upstream.tolist(),
        "actuator_downstream": downstream.tolist(),
        "feature_names": list(feature_names),
        "pair_indices": pair_indices.tolist(),
        "pair_features": pair_features.astype(np.float64).tolist(),
        "label_independent": True,
        "physics_features_standardized": True,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        **payload,
        "pair_indices": pair_indices,
        "pair_features": pair_features,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def d2_magnitude_strata(values_m3: np.ndarray | list[float]) -> dict[str, Any]:
    values = np.asarray(values_m3, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("D2 magnitude strata require finite non-empty values")
    absolute = np.abs(values)
    q33, q67 = (
        float(np.quantile(absolute, q)) for q in (1.0 / 3.0, 2.0 / 3.0)
    )
    masks = {
        "small": absolute < q33,
        "medium": (absolute >= q33) & (absolute < q67),
        "large": absolute >= q67,
    }
    membership = sum(mask.astype(np.int64) for mask in masks.values())
    if not np.all(membership == 1):
        raise RuntimeError("D2 magnitude strata are not mutually exclusive and complete")
    return {
        "contract": "PROJECT7_STEP2_NG1_D2_MAGNITUDE_STRATA_Q33_Q67_V1",
        "q33_m3": q33,
        "q67_m3": q67,
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "masks": masks,
        "count": int(values.size),
    }


def d2_magnitude_weights(
    values_m3: np.ndarray | list[float], strata: dict[str, Any]
) -> np.ndarray:
    values = np.asarray(values_m3, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("D2 magnitude weights require non-empty values")
    q33, q67 = float(strata["q33_m3"]), float(strata["q67_m3"])
    masks = {
        "small": np.abs(values) < q33,
        "medium": (np.abs(values) >= q33) & (np.abs(values) < q67),
        "large": np.abs(values) >= q67,
    }
    weights = np.zeros(values.shape[0], dtype=np.float64)
    global_counts = strata.get("counts", {})
    for name, mask in masks.items():
        mask_array = np.asarray(mask, dtype=bool)
        count = int(global_counts.get(name, int(mask_array.sum())))
        if count:
            weights[mask_array] = 1.0 / float(count)
    if not np.isfinite(weights).all() or not np.any(weights > 0):
        raise ValueError("D2 magnitude weights are invalid")
    weights *= float(weights.size) / float(weights.sum())
    return weights.astype(np.float32)


class NG1ProcessAwareDirectTFVValueModel(DirectFacilityTFVValueModel):
    """Frozen D2 main path plus adaptive process-aware D3 connectivity."""

    contract = NG1_CONTRACT

    def __init__(
        self,
        *,
        state_dim: int,
        rainfall_dim: int,
        actuator_physics_dim: int,
        target_scale_m3: float,
        interaction_graph: dict[str, Any],
        design: DirectTFVValueDesign = DirectTFVValueDesign(),
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            rainfall_dim=rainfall_dim,
            actuator_physics_dim=actuator_physics_dim,
            target_scale_m3=target_scale_m3,
            design=design,
        )
        pair_indices = np.asarray(interaction_graph["pair_indices"], dtype=np.int64)
        pair_features = np.asarray(interaction_graph["pair_features"], dtype=np.float32)
        if pair_indices.shape != (109 * 108 // 2, 2):
            raise ValueError("NG1 model requires the complete 5886-pair graph")
        self.register_buffer(
            "pair_indices", torch.as_tensor(pair_indices, dtype=torch.long)
        )
        self.register_buffer(
            "pair_relation_features",
            torch.as_tensor(pair_features, dtype=torch.float32),
        )
        self.interaction_graph_sha256 = str(interaction_graph["sha256"])

        h = int(design.hidden_dim)
        common_dim = (
            2 * self.state_dim
            + self.actuator_physics_dim
            + design.actuator_embedding_dim
            + 1
            + 2 * h
        )
        joint_input = h + common_dim + design.action_blocks + 2 * h
        relation_dim = int(pair_features.shape[1])
        self.joint_context_encoder = nn.Sequential(
            nn.Linear(joint_input, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.pair_interaction_head = nn.Sequential(
            nn.Linear(2 * h + relation_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.pair_attention_head = nn.Sequential(
            nn.Linear(h + 4, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )
        self.interaction_value_head = nn.Sequential(
            nn.Linear(h + 2 * h + 2, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )

    def interaction_parameter_names(self) -> tuple[str, ...]:
        prefixes = (
            "joint_context_encoder.",
            "pair_interaction_head.",
            "pair_attention_head.",
            "interaction_value_head.",
        )
        return tuple(
            name for name, _ in self.named_parameters() if name.startswith(prefixes)
        )

    def main_parameter_names(self) -> tuple[str, ...]:
        interaction = set(self.interaction_parameter_names())
        return tuple(
            name for name, _ in self.named_parameters() if name not in interaction
        )

    def _joint_latent(
        self,
        common: torch.Tensor,
        latent: torch.Tensor,
        settings: torch.Tensor,
        state_context: torch.Tensor,
        rainfall_context: torch.Tensor,
    ) -> torch.Tensor:
        blocks = self._control_blocks(settings).transpose(1, 2)
        context = torch.cat(
            (
                latent,
                common,
                blocks,
                state_context[:, None].expand(
                    -1, self.design.actuator_count, -1
                ),
                rainfall_context[:, None].expand(
                    -1, self.design.actuator_count, -1
                ),
            ),
            dim=-1,
        )
        return self.joint_context_encoder(context)

    def _pair_activity_mask(self, changed: torch.Tensor) -> torch.Tensor:
        left_index = self.pair_indices[:, 0]
        right_index = self.pair_indices[:, 1]
        return changed.index_select(1, left_index) | changed.index_select(1, right_index)

    def _adaptive_pair_value(
        self,
        joint_latent: torch.Tensor,
        *,
        changed: torch.Tensor,
        action_magnitude: torch.Tensor,
        state_context: torch.Tensor,
        rainfall_context: torch.Tensor,
    ) -> torch.Tensor:
        left_index = self.pair_indices[:, 0]
        right_index = self.pair_indices[:, 1]
        left = joint_latent.index_select(1, left_index)
        right = joint_latent.index_select(1, right_index)
        relation = self.pair_relation_features[None].expand(
            joint_latent.shape[0], -1, -1
        )
        pair_hidden = self.pair_interaction_head(
            torch.cat((left, right, relation), dim=-1)
        )

        left_changed = changed.index_select(1, left_index)
        right_changed = changed.index_select(1, right_index)
        active = left_changed | right_changed
        active_float = active.to(dtype=pair_hidden.dtype)
        left_magnitude = action_magnitude.index_select(1, left_index)
        right_magnitude = action_magnitude.index_select(1, right_index)
        attention_features = torch.cat(
            (
                pair_hidden,
                left_magnitude[..., None],
                right_magnitude[..., None],
                left_changed.to(dtype=pair_hidden.dtype)[..., None],
                right_changed.to(dtype=pair_hidden.dtype)[..., None],
            ),
            dim=-1,
        )
        logits = self.pair_attention_head(attention_features).squeeze(-1)
        masked_logits = logits.masked_fill(~active, -1.0e9)
        weights = torch.softmax(masked_logits, dim=1) * active_float
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.sum(weights[..., None] * pair_hidden, dim=1)

        active_count = active_float.sum(dim=1)
        changed_float = changed.to(dtype=pair_hidden.dtype)
        changed_count = changed_float.sum(dim=1)
        density = torch.log1p(active_count) / float(np.log1p(109 * 108 // 2))
        strength = (action_magnitude * changed_float).sum(dim=1) / changed_count.clamp_min(1.0)
        value = self.interaction_value_head(
            torch.cat(
                (
                    pooled,
                    state_context,
                    rainfall_context,
                    density[:, None],
                    strength[:, None],
                ),
                dim=-1,
            )
        ).squeeze(-1)
        return value * (active_count > 0).to(dtype=value.dtype)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        actuator_upstream: torch.Tensor,
        actuator_downstream: torch.Tensor,
        actuator_physics: torch.Tensor,
    ):
        self._validate_inputs(
            current_state=current_state,
            rainfall=rainfall,
            reference_settings=reference_settings,
            candidate_settings=candidate_settings,
            previous_actuator_flow=previous_actuator_flow,
            actuator_upstream=actuator_upstream,
            actuator_downstream=actuator_downstream,
            actuator_physics=actuator_physics,
        )
        global_state = torch.cat(
            (current_state.mean(dim=1), current_state.amax(dim=1)), dim=-1
        )
        state_context = self.global_state_encoder(global_state)
        rainfall_context = self.rainfall_encoder(self._rainfall_summary(rainfall))
        common = self._facility_context(
            current_state=current_state,
            previous_actuator_flow=previous_actuator_flow,
            actuator_upstream=actuator_upstream,
            actuator_downstream=actuator_downstream,
            actuator_physics=actuator_physics,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        reference_latent = self._sequence_latent(common, reference_settings)
        candidate_latent = self._sequence_latent(common, candidate_settings)
        reference_main = self.facility_head(reference_latent).squeeze(-1)
        candidate_main = self.facility_head(candidate_latent).squeeze(-1)
        facility_effect = (
            candidate_main - reference_main
        ) * self.target_scale_m3

        changed = torch.any(
            torch.abs(candidate_settings - reference_settings) > 1.0e-7, dim=1
        )
        action_magnitude = torch.mean(
            torch.abs(candidate_settings - reference_settings), dim=1
        )
        reference_joint = self._joint_latent(
            common,
            reference_latent,
            reference_settings,
            state_context,
            rainfall_context,
        )
        candidate_joint = self._joint_latent(
            common,
            candidate_latent,
            candidate_settings,
            state_context,
            rainfall_context,
        )
        reference_value = self._adaptive_pair_value(
            reference_joint,
            changed=changed,
            action_magnitude=action_magnitude,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        candidate_value = self._adaptive_pair_value(
            candidate_joint,
            changed=changed,
            action_magnitude=action_magnitude,
            state_context=state_context,
            rainfall_context=rainfall_context,
        )
        changed_count = changed.sum(dim=-1)
        pair_gate = (changed_count > 1).to(dtype=facility_effect.dtype)
        interaction = (
            candidate_value - reference_value
        ) * self.target_scale_m3 * pair_gate

        blocks_ref = self._control_blocks(reference_settings)
        blocks_cand = self._control_blocks(candidate_settings)
        activity = torch.mean(
            torch.abs(blocks_cand - blocks_ref).transpose(1, 2), dim=-1
        )
        total = facility_effect.sum(dim=-1) + interaction
        from .step2_tfv_value import DirectTFVValueOutput

        return DirectTFVValueOutput(
            total_delta_tfv_m3=total,
            facility_main_effect_m3=facility_effect,
            interaction_residual_m3=interaction,
            action_activity=activity,
        )


__all__ = [
    "NG1_CONTRACT",
    "NG1_GRAPH_CONTRACT",
    "NG1ProcessAwareDirectTFVValueModel",
    "build_control_interaction_graph",
    "d2_magnitude_strata",
    "d2_magnitude_weights",
]
