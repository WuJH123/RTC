"""V4.4.1 conduit-only edge residual with corrected physical feature units."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from .step2_control_response_v44 import DifferentiableCounterfactualResponseModelV44
from .step2_edge_physics_v441 import (
    CONDUIT_FEATURE_NAMES_V441,
    PhysicalDirectedEdgeLineageV441,
    normalize_conduit_static_features_v441,
)


class DifferentiableCounterfactualResponseModelV441(
    DifferentiableCounterfactualResponseModelV44
):
    """V4.4 baseline plus a zero-start, conduit-only hydraulic residual.

    The inherited V4.4 edge message MLP remains the only new capacity.  This
    class changes only its static contract and dynamic head/gradient inputs:
    static features are shape-aware conduit features and dynamic features are
    dimensionless transforms using frozen Train-only scales.
    """

    def __init__(self, *args, edge_feature_dim: int = len(CONDUIT_FEATURE_NAMES_V441), **kwargs) -> None:
        super().__init__(*args, edge_feature_dim=edge_feature_dim, **kwargs)
        if int(edge_feature_dim) != len(CONDUIT_FEATURE_NAMES_V441):
            raise ValueError("V4.4.1 requires the fixed conduit feature contract")
        self.register_buffer("edge_hydraulic_head_scale_v441", torch.tensor(1.0), persistent=False)
        self.register_buffer("edge_hydraulic_gradient_scale_v441", torch.tensor(1.0), persistent=False)
        self.edge_hydraulic_contract_v441 = "EDGE_HYDRAULIC_PROPAGATION_V441_CONDUIT_ONLY"
        self.edge_hydraulic_debug_capture_v441 = False
        self._last_edge_node_residual_v441: torch.Tensor | None = None
        self.edge_hydraulic_link_types_v441: tuple[str, ...] = ()
        self.edge_hydraulic_edge_feature_names_v441: tuple[str, ...] = tuple(CONDUIT_FEATURE_NAMES_V441)

    def configure_edge_hydraulic_v441(
        self,
        lineage: PhysicalDirectedEdgeLineageV441,
        normalized_static_features: np.ndarray | torch.Tensor,
        *,
        head_scale_train: float,
        gradient_scale_train: float,
    ) -> None:
        edge_index = np.asarray(lineage.edge_index, dtype=np.int64)
        static = np.asarray(normalized_static_features, dtype=np.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("lineage.edge_index must be [2,E]")
        if static.shape != (edge_index.shape[1], len(CONDUIT_FEATURE_NAMES_V441)):
            raise ValueError("normalized V4.4.1 static features have the wrong shape")
        if len(lineage.edge_to_link_id) != edge_index.shape[1]:
            raise ValueError("every directed conduit edge needs one physical-link id")
        if tuple(set(lineage.edge_to_link_type)) != ("conduit",):
            raise ValueError("V4.4.1 propagation accepts conduits only")
        if not np.isfinite(static).all() or not np.isfinite(lineage.edge_static_features).all():
            raise ValueError("V4.4.1 edge features must be finite")
        lengths = np.asarray(lineage.edge_lengths_m, dtype=np.float32)
        if lengths.shape != (edge_index.shape[1],) or not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("V4.4.1 conduit edge lengths must be positive finite")
        if not np.isfinite(float(head_scale_train)) or float(head_scale_train) <= 0.0:
            raise ValueError("head_scale_train must be positive finite")
        if not np.isfinite(float(gradient_scale_train)) or float(gradient_scale_train) <= 0.0:
            raise ValueError("gradient_scale_train must be positive finite")
        if np.any(edge_index < 0):
            raise ValueError("edge indices must be non-negative")
        nodes = int(edge_index.max()) + 1 if edge_index.size else 0
        degree = np.bincount(edge_index[1], minlength=nodes).astype(np.float32)
        degree = np.maximum(degree, 1.0)
        # These are the same non-persistent buffers used by the V4.4 forward
        # path, but length is taken from the lineage rather than log_length.
        self.edge_hydraulic_edge_index_v44 = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_hydraulic_static_features_v44 = torch.as_tensor(static, dtype=torch.float32)
        self.edge_hydraulic_length_m_v44 = torch.as_tensor(lengths, dtype=torch.float32)
        self.edge_hydraulic_orientation_v44 = torch.as_tensor(
            np.asarray(lineage.orientation_signs, dtype=np.float32), dtype=torch.float32
        )
        self.edge_hydraulic_destination_degree_v44 = torch.as_tensor(degree, dtype=torch.float32)
        self.edge_hydraulic_link_ids_v44 = tuple(lineage.edge_to_link_id)
        self.edge_hydraulic_link_types_v44 = tuple(lineage.edge_to_link_type)
        self.edge_hydraulic_link_types_v441 = tuple(lineage.edge_to_link_type)
        self.edge_hydraulic_edge_feature_names_v44 = tuple(lineage.edge_static_feature_names)
        self.edge_hydraulic_edge_feature_names_v441 = tuple(lineage.edge_static_feature_names)
        self.edge_hydraulic_head_scale_v441 = torch.tensor(float(head_scale_train), dtype=torch.float32)
        self.edge_hydraulic_gradient_scale_v441 = torch.tensor(float(gradient_scale_train), dtype=torch.float32)

    def _edge_hydraulic_node_residual(
        self,
        *,
        base_node_latent: torch.Tensor,
        reference,
        interaction_time_gate: torch.Tensor,
    ) -> torch.Tensor:
        if base_node_latent.dim() != 5:
            raise ValueError("base_node_latent must be [B,C,H,N,R]")
        batch, candidates, horizon, nodes, rank = base_node_latent.shape
        if rank != self.effect_rank:
            raise ValueError("node latent rank does not match model effect_rank")
        if not self._edge_hydraulic_ready(nodes):
            raise RuntimeError("V4.4.1 edge residual requires a complete conduit lineage")
        device = base_node_latent.device
        dtype = base_node_latent.dtype
        edge_index = self.edge_hydraulic_edge_index_v44.to(device=device)
        static = self.edge_hydraulic_static_features_v44.to(device=device, dtype=dtype)
        lengths = self.edge_hydraulic_length_m_v44.to(device=device, dtype=dtype)
        orientation = self.edge_hydraulic_orientation_v44.to(device=device, dtype=dtype)
        degree = self.edge_hydraulic_destination_degree_v44.to(device=device, dtype=dtype)
        if degree.numel() < nodes:
            degree = torch.cat((degree, degree.new_ones(nodes - degree.numel())))
        src, dst = edge_index[0].long(), edge_index[1].long()
        edge_count = int(src.numel())
        reference_context = reference.node_context.to(device=device, dtype=dtype)
        context_rank = self.edge_hydraulic_context_encoder(reference_context)
        reference_states = reference.reference_states_physical.to(device=device, dtype=dtype)
        heads = reference_states[..., 1]
        output = base_node_latent.new_zeros(batch, candidates, horizon, nodes, rank)
        flat_count = batch * candidates * horizon
        flat_nodes = base_node_latent.reshape(flat_count, nodes, rank)
        flat_context = context_rank[:, None].expand(batch, candidates, horizon, nodes, rank).reshape(flat_count, nodes, rank)
        flat_heads = heads[:, None].expand(batch, candidates, horizon, nodes).reshape(flat_count, nodes)
        flat_output = output.reshape(flat_count, nodes, rank)
        head_scale = self.edge_hydraulic_head_scale_v441.to(device=device, dtype=dtype).clamp_min(torch.finfo(dtype).eps)
        gradient_scale = self.edge_hydraulic_gradient_scale_v441.to(device=device, dtype=dtype).clamp_min(torch.finfo(dtype).eps)
        for start in range(0, edge_count, self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, edge_count)
            chunk_src, chunk_dst = src[start:stop], dst[start:stop]
            src_latent, dst_latent = flat_nodes[:, chunk_src, :], flat_nodes[:, chunk_dst, :]
            src_context, dst_context = flat_context[:, chunk_src, :], flat_context[:, chunk_dst, :]
            delta_head = flat_heads[:, chunk_src] - flat_heads[:, chunk_dst]
            raw_gradient = delta_head / lengths[start:stop].clamp_min(torch.finfo(dtype).eps)[None, :]
            delta_head_norm = delta_head / head_scale
            gradient_norm = torch.sign(raw_gradient) * torch.log1p(torch.abs(raw_gradient) / gradient_scale)
            edge_static = static[start:stop][None].expand(flat_count, -1, -1)
            edge_orientation = orientation[start:stop][None].expand(flat_count, -1)
            message_input = torch.cat(
                (
                    src_latent,
                    dst_latent,
                    src_context,
                    dst_context,
                    edge_static,
                    delta_head_norm.unsqueeze(-1),
                    gradient_norm.unsqueeze(-1),
                    edge_orientation.unsqueeze(-1),
                ),
                dim=-1,
            )
            message = self.edge_hydraulic_message_encoder(message_input)
            message = message.reshape(batch, candidates, horizon, stop - start, rank)
            message = message * interaction_time_gate.detach().to(dtype).unsqueeze(-1).unsqueeze(-1)
            flat_output.index_add_(1, chunk_dst, message.reshape(flat_count, stop - start, rank))
        result = (flat_output / degree.reshape(1, nodes, 1)).reshape(batch, candidates, horizon, nodes, rank)
        if bool(getattr(self, "edge_hydraulic_debug_capture_v441", False)):
            self._last_edge_node_residual_v441 = result.detach().cpu()
        return result

    edge_hydraulic_node_residual_v441 = _edge_hydraulic_node_residual


def edge_hydraulic_parameter_names_v441(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _ in model.named_parameters() if name.startswith("edge_hydraulic_"))


def set_trainable_edge_hydraulic_v441(model: nn.Module, *, enabled: bool = True) -> tuple[str, ...]:
    names = set(edge_hydraulic_parameter_names_v441(model))
    if enabled and not names:
        raise RuntimeError("V4.4.1 edge parameter set is empty")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(enabled and name in names))
        if not (enabled and name in names):
            parameter.grad = None
    return tuple(sorted(names if enabled else ()))


__all__ = [
    "DifferentiableCounterfactualResponseModelV441",
    "edge_hydraulic_parameter_names_v441",
    "set_trainable_edge_hydraulic_v441",
]
