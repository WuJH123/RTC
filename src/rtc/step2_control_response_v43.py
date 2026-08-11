"""Isolated V4.3 state/topology-conditioned D3 interaction prototype.

V4.3 reuses the tested V4.2.1 reference and additive single-actuator paths.  It
adds only a low-rank, same-time graph propagation path inside the D3 interaction
residual.  Reference and D2 parameters have explicit phase ownership so a D3
update cannot change an already-fitted D2 prediction.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import nn

from .step2_control_response_v41 import (
    CounterfactualGroupResponseV41,
    DifferentiableCounterfactualResponseModelV41,
    PreparedStaticV41,
    ReferenceEncodingV41,
    _directed_scatter,
    _mlp,
)


class DifferentiableCounterfactualResponseModelV43(
    DifferentiableCounterfactualResponseModelV41
):
    """V4.2.1 response model with a small state-conditioned graph residual."""

    def __init__(self, *args, topology_blocks: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if topology_blocks < 1:
            raise ValueError("topology_blocks must be positive")
        self.topology_blocks = int(topology_blocks)
        rank = self.effect_rank
        hidden = self.hidden_dim
        # Local actuator state + the V4.2.1 action token.  The output is a
        # low-rank endpoint seed, never a full [node, hidden] tensor.
        self.topology_seed_encoder = _mlp(4 * hidden + 1, hidden, rank)
        self.topology_context_encoder = nn.Linear(2 * hidden, rank)
        self.topology_message_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * rank, rank),
                nn.SiLU(),
                nn.Linear(rank, rank),
            )
            for _ in range(self.topology_blocks)
        )
        self.topology_state_head = nn.Linear(rank, self.state_dim)
        self.topology_hidden_head = nn.Linear(rank, hidden)
        self._topology_graph_enabled = True
        self._topology_local_state_enabled = True

    def set_topology_ablation(
        self, *, graph_enabled: bool = True, local_state_enabled: bool = True
    ) -> None:
        """Toggle read-only ablations without changing learned parameters."""

        self._topology_graph_enabled = bool(graph_enabled)
        self._topology_local_state_enabled = bool(local_state_enabled)

    def _topology_interaction(
        self,
        *,
        hidden_delta: torch.Tensor,
        delta_u: torch.Tensor,
        interaction_hidden: torch.Tensor,
        reference: ReferenceEncodingV41,
        prepared: PreparedStaticV41,
        interaction_time_gate: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        del interaction_hidden
        batch, candidates, horizon, actuators, _ = hidden_delta.shape
        nodes = reference.node_context.shape[2]
        dtype = hidden_delta.dtype
        device = hidden_delta.device
        if not self._topology_graph_enabled:
            zero_state = hidden_delta.new_zeros(batch, candidates, horizon, nodes, self.state_dim)
            zero_hidden = hidden_delta.new_zeros(batch, candidates, horizon, self.hidden_dim)
            return zero_state, zero_hidden

        upstream = reference.actuator_upstream.to(device=device)
        downstream = reference.actuator_downstream.to(device=device)
        upstream_context = reference.node_context[:, :, upstream, :]
        downstream_context = reference.node_context[:, :, downstream, :]
        upstream_context = upstream_context[:, None].expand(batch, candidates, -1, -1, -1)
        downstream_context = downstream_context[:, None].expand(batch, candidates, -1, -1, -1)
        actuator_static = reference.actuator_static.to(device=device, dtype=dtype)
        actuator_static = actuator_static[None, None, None].expand(
            batch, candidates, horizon, actuators, -1
        )
        seed_input = torch.cat(
            (
                hidden_delta,
                upstream_context,
                downstream_context,
                actuator_static,
                delta_u.unsqueeze(-1),
            ),
            dim=-1,
        )
        seed = self.topology_seed_encoder(seed_input)
        # The detached mask is a structural action gate.  It makes both zero
        # and numerically exact no-op actions produce an exact zero seed.
        action_mask = delta_u.detach().ne(0.0).to(dtype).unsqueeze(-1)
        seed = seed * action_mask
        outgoing, incoming = _directed_scatter(seed, upstream, downstream, nodes)
        node_latent = outgoing + incoming

        local_state = reference.node_context
        if not self._topology_local_state_enabled:
            local_state = reference.global_context.unsqueeze(2).expand(-1, -1, nodes, -1)
        static_context = reference.node_static_context.to(device=device, dtype=dtype)
        context = torch.cat(
            (
                local_state[:, None].expand(batch, candidates, horizon, nodes, -1),
                static_context[None, None, None].expand(batch, candidates, horizon, nodes, -1),
            ),
            dim=-1,
        )
        context_rank = torch.tanh(self.topology_context_encoder(context))

        edge_index = prepared.edge_index.to(device=device)
        if edge_index.numel() == 0:
            raise ValueError("V4.3 topology interaction requires at least one edge")
        src, dst = edge_index[0].long(), edge_index[1].long()
        degree = torch.bincount(dst, minlength=nodes).to(dtype=dtype).clamp_min(1.0)
        flat_count = batch * candidates * horizon
        for block in self.topology_message_blocks:
            flat = node_latent.reshape(flat_count, nodes, self.effect_rank)
            messages = flat[:, src, :]
            aggregate = flat.new_zeros(flat_count, nodes, self.effect_rank)
            aggregate.index_add_(1, dst, messages)
            aggregate = aggregate / degree.reshape(1, nodes, 1)
            combined = torch.cat(
                (
                    node_latent,
                    aggregate.reshape(batch, candidates, horizon, nodes, self.effect_rank),
                    context_rank,
                ),
                dim=-1,
            )
            node_latent = node_latent + block(combined)
            node_latent = torch.tanh(node_latent)

        time_gate = interaction_time_gate.detach().to(dtype).unsqueeze(-1).unsqueeze(-1)
        node_latent = node_latent * time_gate
        topology_state = self.topology_state_head(node_latent)
        topology_hidden = self.topology_hidden_head(node_latent.mean(dim=3))
        topology_state = topology_state * time_gate
        topology_hidden = topology_hidden * time_gate.squeeze(-1)
        return topology_state, topology_hidden


def _names_with_prefixes(
    model: nn.Module, prefixes: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        name for name, _ in model.named_parameters() if name.startswith(prefixes)
    )


def reference_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Parameters fitted in Phase R and frozen for D2/D3."""

    return _names_with_prefixes(
        model,
        (
            "node_static_encoder.",
            "actuator_static_encoder.",
            "actuator_identity.",
            "reference_",
        ),
    )


def single_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Parameters fitted in Phase D2 after the reference phase."""

    return _names_with_prefixes(
        model,
        ("temporal_identity.", "single_", "direct_single_tfv_head."),
    )


def interaction_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Parameters fitted in Phase D3; all reference/single parameters are frozen."""

    return _names_with_prefixes(
        model,
        ("interaction_", "direct_interaction_tfv_head.", "topology_"),
    )


def phase_parameter_names(model: nn.Module, phase: str) -> tuple[str, ...]:
    phase_key = phase.strip().lower()
    if phase_key in {"reference", "r"}:
        return reference_parameter_names(model)
    if phase_key in {"d2", "single"}:
        return single_parameter_names(model)
    if phase_key in {"d3", "interaction"}:
        return interaction_parameter_names(model)
    raise ValueError("phase must be reference, d2, or d3")


def set_trainable_phase(model: nn.Module, phase: str) -> tuple[str, ...]:
    """Make the phase ownership explicit and fail closed on unclassified params."""

    groups = {
        "reference": set(reference_parameter_names(model)),
        "d2": set(single_parameter_names(model)),
        "d3": set(interaction_parameter_names(model)),
    }
    phase_key = phase.strip().lower()
    if phase_key in {"r", "reference"}:
        selected = groups["reference"]
    elif phase_key in {"single", "d2"}:
        selected = groups["d2"]
    elif phase_key in {"interaction", "d3"}:
        selected = groups["d3"]
    else:
        raise ValueError("phase must be reference, d2, or d3")
    all_names = {name for name, _ in model.named_parameters()}
    classified = set().union(*groups.values())
    if classified != all_names:
        missing = sorted(all_names - classified)
        overlap = sorted(
            (groups["reference"] & groups["d2"])
            | (groups["reference"] & groups["d3"])
            | (groups["d2"] & groups["d3"])
        )
        raise RuntimeError(f"phase parameter partition is incomplete/overlapping: {missing=} {overlap=}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
        if name not in selected:
            parameter.grad = None
    return tuple(sorted(selected))


def parameter_sha256(model: nn.Module, names: Iterable[str]) -> str:
    selected = set(names)
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name in selected:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def topology_graph_contract(prepared: PreparedStaticV41) -> dict[str, int | bool | str]:
    """Describe the supplied graph without reconstructing or guessing edges."""

    edge = prepared.edge_index.detach().cpu().long()
    pairs = [tuple(row) for row in edge.t().tolist()]
    pair_set = set(pairs)
    reverse_count = sum((dst, src) in pair_set for src, dst in pairs)
    node_count = int(prepared.static_node_features.shape[0])
    degree = torch.bincount(edge.reshape(-1), minlength=node_count)
    return {
        "node_count": node_count,
        "edge_count": int(edge.shape[1]),
        "directed_contract": "bidirectional" if reverse_count == len(pairs) else "directed_or_mixed",
        "reverse_edge_count": int(reverse_count),
        "self_loop_count": int(sum(src == dst for src, dst in pairs)),
        "duplicate_edge_count": int(len(pairs) - len(pair_set)),
        "isolated_node_count": int(degree.eq(0).sum()),
        "edge_index_used_by_forward": True,
    }


__all__ = [
    "DifferentiableCounterfactualResponseModelV43",
    "interaction_parameter_names",
    "parameter_sha256",
    "phase_parameter_names",
    "reference_parameter_names",
    "set_trainable_phase",
    "single_parameter_names",
    "topology_graph_contract",
]
