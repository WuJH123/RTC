"""V4.4 edge-hydraulic-conditioned interaction residual.

The V4.3.3 response operator is kept as the immutable baseline.  This module
adds one low-rank residual message path over the physical directed-link
lineage produced by :mod:`rtc.step2_edge_hydraulic_v44`.  The residual is
zero-initialized and is gated by the existing D3 multi-actuator causal gate,
so loading this model with an old V4.3.3 checkpoint is numerically equivalent
to the old model and D2 predictions cannot change during an edge-only update.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from .step2_control_response_v433 import (
    DifferentiableCounterfactualResponseModelV433,
)
from .step2_control_response_v41 import ReferenceEncodingV41, PreparedStaticV41, _mlp
from .step2_edge_hydraulic_v44 import (
    PhysicalDirectedEdgeLineageV44,
    STATIC_FEATURE_NAMES_V44,
)


def _zero_last_linear(module: nn.Module) -> None:
    """Zero the final linear projection of an MLP without changing capacity."""

    linear = next(
        (child for child in reversed(tuple(module.modules())) if isinstance(child, nn.Linear)),
        None,
    )
    if linear is None:
        raise RuntimeError("expected an MLP with a final Linear projection")
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


class DifferentiableCounterfactualResponseModelV44(
    DifferentiableCounterfactualResponseModelV433
):
    """V4.3.3 plus a zero-start residual over physical hydraulic edges.

    The old unweighted topology path remains untouched.  ``edge_hydraulic``
    parameters are the only new trainable parameters.  Static edge features
    and lineage are registered as non-persistent buffers so a development
    runner must explicitly configure them and no large graph artifact is
    silently serialized into a checkpoint.
    """

    def __init__(
        self,
        *args,
        edge_feature_dim: int = len(STATIC_FEATURE_NAMES_V44),
        edge_chunk_size: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(edge_feature_dim) <= 0:
            raise ValueError("edge_feature_dim must be positive")
        if int(edge_chunk_size) <= 0:
            raise ValueError("edge_chunk_size must be positive")
        self.edge_feature_dim = int(edge_feature_dim)
        self.edge_chunk_size = int(edge_chunk_size)

        # The message input is two low-rank node states, two low-rank local
        # hydraulic contexts, static edge attributes, and three directed
        # dynamic/orientation features (delta-head, gradient, orientation).
        message_input_dim = 4 * self.effect_rank + self.edge_feature_dim + 3
        self.edge_hydraulic_context_encoder = nn.Linear(
            self.hidden_dim, self.effect_rank
        )
        self.edge_hydraulic_message_encoder = _mlp(
            message_input_dim, self.hidden_dim, self.effect_rank
        )
        # Exact old-checkpoint equivalence: edge messages and their downstream
        # state/hidden projections are zero at initialization.  Existing V4.3
        # heads are used with zero-centering, so their biases cannot leak into
        # the residual when the edge node latent is zero.
        _zero_last_linear(self.edge_hydraulic_message_encoder)

        self.edge_hydraulic_residual_enabled = True
        self.edge_hydraulic_residual_active = True
        self.register_buffer(
            "edge_hydraulic_edge_index_v44",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "edge_hydraulic_static_features_v44",
            torch.empty((0, self.edge_feature_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "edge_hydraulic_length_m_v44",
            torch.empty((0,), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "edge_hydraulic_orientation_v44",
            torch.empty((0,), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "edge_hydraulic_destination_degree_v44",
            torch.empty((0,), dtype=torch.float32),
            persistent=False,
        )

    def configure_edge_hydraulic_v44(
        self,
        lineage: PhysicalDirectedEdgeLineageV44,
        normalized_static_features: np.ndarray | torch.Tensor,
    ) -> None:
        """Attach a validated physical directed-edge contract.

        This method is deliberately explicit and development-only.  It does
        not infer a physical link from the legacy ``PreparedStaticV41`` edge
        index, because that index may collapse parallel SWMM links.
        """

        edge_index = np.asarray(lineage.edge_index, dtype=np.int64)
        static = np.asarray(normalized_static_features, dtype=np.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("lineage.edge_index must be [2,E]")
        if static.ndim != 2 or static.shape != (edge_index.shape[1], self.edge_feature_dim):
            raise ValueError(
                "normalized edge static features must be [E, edge_feature_dim]"
            )
        if len(lineage.edge_to_link_id) != edge_index.shape[1]:
            raise ValueError("every directed edge must have one physical-link id")
        if len(lineage.orientation_signs) != edge_index.shape[1]:
            raise ValueError("every directed edge must have one orientation sign")
        if not np.isfinite(static).all():
            raise ValueError("normalized edge static features must be finite")
        if not np.isfinite(lineage.edge_static_features).all():
            raise ValueError("raw edge static features must be finite")
        raw_length = np.asarray(lineage.edge_static_features, dtype=np.float32)[:, 0]
        if raw_length.shape != (edge_index.shape[1],) or not np.isfinite(raw_length).all():
            raise ValueError("every directed edge must have a finite physical length")
        if np.any(edge_index < 0):
            raise ValueError("edge indices must be non-negative")
        nodes = int(edge_index.max()) + 1 if edge_index.size else 0
        destination_degree = np.bincount(
            edge_index[1], minlength=nodes
        ).astype(np.float32)
        destination_degree = np.maximum(destination_degree, 1.0)
        self.edge_hydraulic_edge_index_v44 = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_hydraulic_static_features_v44 = torch.as_tensor(static, dtype=torch.float32)
        self.edge_hydraulic_length_m_v44 = torch.as_tensor(
            np.maximum(raw_length, 1.0e-6), dtype=torch.float32
        )
        self.edge_hydraulic_orientation_v44 = torch.as_tensor(
            np.asarray(lineage.orientation_signs, dtype=np.float32), dtype=torch.float32
        )
        self.edge_hydraulic_destination_degree_v44 = torch.as_tensor(
            destination_degree, dtype=torch.float32
        )
        self.edge_hydraulic_link_ids_v44 = tuple(lineage.edge_to_link_id)
        self.edge_hydraulic_link_types_v44 = tuple(lineage.edge_to_link_type)
        self.edge_hydraulic_edge_feature_names_v44 = tuple(
            lineage.edge_static_feature_names
        )

    def _edge_hydraulic_ready(self, node_count: int) -> bool:
        edge_index = self.edge_hydraulic_edge_index_v44
        return bool(
            edge_index.numel()
            and edge_index.shape[0] == 2
            and self.edge_hydraulic_static_features_v44.shape[0] == edge_index.shape[1]
            and self.edge_hydraulic_destination_degree_v44.numel() <= node_count
            and int(edge_index.max()) < node_count
        )

    def _edge_hydraulic_node_residual(
        self,
        *,
        base_node_latent: torch.Tensor,
        reference: ReferenceEncodingV41,
        interaction_time_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute [B,C,H,N,R] physical-edge residual messages.

        Reference head/depth values are generated by the model's causal
        reference encoder.  No candidate/future SWMM truth or link flow enters
        this path.  Messages are accumulated at destination nodes using the
        physical directed-edge degree; parallel physical links remain separate
        rows in the index and therefore are not silently collapsed.
        """

        if base_node_latent.dim() != 5:
            raise ValueError("base_node_latent must be [B,C,H,N,R]")
        batch, candidates, horizon, nodes, rank = base_node_latent.shape
        if rank != self.effect_rank:
            raise ValueError("node latent rank does not match model effect_rank")
        if not self._edge_hydraulic_ready(nodes):
            raise RuntimeError(
                "V4.4 edge residual requires configure_edge_hydraulic_v44 with a complete lineage"
            )
        device = base_node_latent.device
        dtype = base_node_latent.dtype
        edge_index = self.edge_hydraulic_edge_index_v44.to(device=device)
        static = self.edge_hydraulic_static_features_v44.to(device=device, dtype=dtype)
        length = self.edge_hydraulic_length_m_v44.to(device=device, dtype=dtype)
        orientation = self.edge_hydraulic_orientation_v44.to(device=device, dtype=dtype)
        degree = self.edge_hydraulic_destination_degree_v44.to(device=device, dtype=dtype)
        if degree.numel() < nodes:
            degree = torch.cat((degree, degree.new_ones(nodes - degree.numel())))
        src, dst = edge_index[0].long(), edge_index[1].long()
        edge_count = int(src.numel())

        # Context is the causal reference trajectory.  Channel 1 is the
        # physical head in ReferenceEncodingV41 (head = invert + depth).
        reference_context = reference.node_context.to(device=device, dtype=dtype)
        context_rank = self.edge_hydraulic_context_encoder(reference_context)
        reference_states = reference.reference_states_physical.to(device=device, dtype=dtype)
        heads = reference_states[..., 1]
        output = base_node_latent.new_zeros(batch, candidates, horizon, nodes, rank)
        flat_count = batch * candidates * horizon
        flat_nodes = base_node_latent.reshape(flat_count, nodes, rank)
        # Candidate dimension shares the reference context.  Expand only the
        # flattened view and process edges in chunks to avoid [B,C,H,E,*]
        # allocations for the 2,552-edge contract.
        flat_context = context_rank[:, None].expand(
            batch, candidates, horizon, nodes, rank
        ).reshape(flat_count, nodes, rank)
        flat_heads = heads[:, None].expand(batch, candidates, horizon, nodes).reshape(
            flat_count, nodes
        )
        # ``interaction_time_gate`` has shape [B,C,H], so the reshape above is
        # the causal per-time gate used after the edge aggregation.
        flat_output = output.reshape(flat_count, nodes, rank)
        for start in range(0, edge_count, self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, edge_count)
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            src_latent = flat_nodes[:, chunk_src, :]
            dst_latent = flat_nodes[:, chunk_dst, :]
            src_context = flat_context[:, chunk_src, :]
            dst_context = flat_context[:, chunk_dst, :]
            delta_head = flat_heads[:, chunk_src] - flat_heads[:, chunk_dst]
            hydraulic_gradient = delta_head / length[start:stop].clamp_min(1.0e-6)[None, :]
            edge_static = static[start:stop][None].expand(flat_count, -1, -1)
            edge_orientation = orientation[start:stop][None].expand(flat_count, -1)
            message_input = torch.cat(
                (
                    src_latent,
                    dst_latent,
                    src_context,
                    dst_context,
                    edge_static,
                    delta_head.unsqueeze(-1),
                    hydraulic_gradient.unsqueeze(-1),
                    edge_orientation.unsqueeze(-1),
                ),
                dim=-1,
            )
            message = self.edge_hydraulic_message_encoder(message_input)
            # Every message is causal at the same forecast time.  No future
            # action can enter an earlier response through this spatial path.
            # ``flat_gate`` is expanded per edge below after recovering time.
            message = message.reshape(batch, candidates, horizon, stop - start, rank)
            message = message * interaction_time_gate.detach().to(dtype).unsqueeze(-1).unsqueeze(-1)
            message = message.reshape(flat_count, stop - start, rank)
            flat_output.index_add_(1, chunk_dst, message)
        flat_output = flat_output / degree.reshape(1, nodes, 1)
        return flat_output.reshape(batch, candidates, horizon, nodes, rank)

    # Alias used by diagnostics without exposing a separate learned path.
    edge_hydraulic_node_residual_v44 = _edge_hydraulic_node_residual

    def _topology_interaction(self, *, return_node_latent: bool = False, **kwargs):
        base_output = super()._topology_interaction(
            **kwargs, return_node_latent=True
        )
        base_state, base_hidden, base_node_latent = base_output
        if base_node_latent is None:
            raise RuntimeError("V4.4 requires the V4.3 node latent")
        if not bool(getattr(self, "edge_hydraulic_residual_enabled", True)) or not bool(
            getattr(self, "edge_hydraulic_residual_active", True)
        ):
            if return_node_latent:
                return base_state, base_hidden, base_node_latent
            return base_state, base_hidden
        edge_node_residual = self._edge_hydraulic_node_residual(
            base_node_latent=base_node_latent,
            reference=kwargs["reference"],
            interaction_time_gate=kwargs["interaction_time_gate"],
        )
        # Center the existing V4.3 heads at zero input.  This makes their
        # learned biases harmless and preserves exact baseline equivalence.
        zero_latent = torch.zeros_like(edge_node_residual)
        edge_state = self.topology_state_head(edge_node_residual) - self.topology_state_head(
            zero_latent
        )
        edge_hidden = self.topology_hidden_head(edge_node_residual.mean(dim=3)) - self.topology_hidden_head(
            zero_latent.mean(dim=3)
        )
        combined_state = base_state + edge_state if base_state is not None else edge_state
        combined_hidden = base_hidden + edge_hidden if base_hidden is not None else edge_hidden
        combined_node_latent = base_node_latent + edge_node_residual
        if return_node_latent:
            return combined_state, combined_hidden, combined_node_latent
        return combined_state, combined_hidden


def edge_hydraulic_parameter_names_v44(model: nn.Module) -> tuple[str, ...]:
    """Return only learned V4.4 edge-residual parameters."""

    return tuple(
        name
        for name, _ in model.named_parameters()
        if name.startswith("edge_hydraulic_")
    )


def set_trainable_edge_hydraulic_v44(
    model: nn.Module,
    *,
    enabled: bool = True,
) -> tuple[str, ...]:
    """Freeze the complete V4.3.3 backbone and optionally train edge params."""

    names = set(edge_hydraulic_parameter_names_v44(model))
    if enabled and not names:
        raise RuntimeError("V4.4 edge parameter set is empty")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(enabled and name in names))
        if not (enabled and name in names):
            parameter.grad = None
    return tuple(sorted(names if enabled else ()))


__all__ = [
    "DifferentiableCounterfactualResponseModelV44",
    "edge_hydraulic_parameter_names_v44",
    "set_trainable_edge_hydraulic_v44",
]
