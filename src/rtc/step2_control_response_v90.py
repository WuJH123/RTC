"""Project7 Step2 V9 signed hydraulic-effect decision surrogate.

V9 keeps the successful V7 Value model untouched and keeps V7 Hydraulic frozen as
reference.  The primary counterfactual outputs are signed, unclipped deltas.  Absolute
candidate trajectories are projected separately and are diagnostic only.

The same model class also implements the pre-registered D2 state-sufficiency ladder:
A = V8 boundary context only; B = A + frozen V7 predicted reference trajectory;
C = A + authoritative true reference trajectory (oracle diagnostic only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .step2_control_response_v70 import HydraulicResponseSurrogateV70
from .step2_control_response_v80 import (
    CausalPrefixActionProjectorV80,
    DirectHydraulicEffectSurrogateV80,
    PreparedStaticV80,
    _mlp,
    _scatter_actuators_to_nodes,
    prepare_static_v80,
)
from .step2_physical_edge_v90 import (
    ConduitPhysicalEdgeAssetsV90,
    causal_reference_dynamic_edge_features_v90,
)
from .step2_v90_contract import (
    DirectHydraulicEffectLossContractV90,
    LEVEL_A,
    LEVEL_B,
    LEVEL_C,
    validate_conditioning_level_v90,
)


@dataclass(frozen=True)
class HydraulicEffectOutputV90:
    horizon_indices: torch.Tensor
    reference_states_physical: torch.Tensor
    raw_delta_states_physical: torch.Tensor
    candidate_states_projected_physical: torch.Tensor
    reference_flows_physical: torch.Tensor
    raw_delta_flows_physical: torch.Tensor
    candidate_flows_projected_physical: torch.Tensor
    reference_flood_onset_logits: torch.Tensor
    candidate_flood_onset_logits: torch.Tensor
    joint_context_before_scatter: torch.Tensor

    # Compatibility aliases deliberately point to the signed primary effects, never to
    # projected-candidate-minus-reference.  New V9 code should use the explicit names.
    @property
    def delta_states_physical(self) -> torch.Tensor:
        return self.raw_delta_states_physical

    @property
    def delta_flows_physical(self) -> torch.Tensor:
        return self.raw_delta_flows_physical

    @property
    def candidate_states_physical(self) -> torch.Tensor:
        return self.candidate_states_projected_physical

    @property
    def candidate_flows_physical(self) -> torch.Tensor:
        return self.candidate_flows_projected_physical


def project_candidate_states_v90(
    reference_states_physical: torch.Tensor,
    raw_delta_states_physical: torch.Tensor,
    *,
    invert_elevation_m: torch.Tensor,
) -> torch.Tensor:
    """Project absolute state only; never recompute or mutate the signed raw delta."""
    if reference_states_physical.shape != raw_delta_states_physical.shape:
        raise ValueError("V9 reference/raw state shapes differ")
    if reference_states_physical.shape[-1] != 6:
        raise ValueError("V9 hydraulic state contract requires six channels")
    depth = (reference_states_physical[..., 0] + raw_delta_states_physical[..., 0]).clamp_min(0.0)
    invert = invert_elevation_m.to(depth).reshape(
        *((1,) * (depth.ndim - 1)), -1
    )
    return torch.stack(
        (
            depth,
            depth + invert,
            (reference_states_physical[..., 2] + raw_delta_states_physical[..., 2]).clamp_min(0.0),
            (reference_states_physical[..., 3] + raw_delta_states_physical[..., 3]).clamp_min(0.0),
            (reference_states_physical[..., 4] + raw_delta_states_physical[..., 4]).clamp_min(0.0),
            (reference_states_physical[..., 5] + raw_delta_states_physical[..., 5]).clamp_min(0.0),
        ),
        dim=-1,
    )


def project_candidate_flows_v90(
    reference_flows_physical: torch.Tensor,
    raw_delta_flows_physical: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct signed absolute managed flow without a global clamp.

    The frozen actuator-flow audit contains negative orifice and weir flows.
    Physical projection is therefore only a reconstruction of the authoritative
    signed quantity; it must not impose a false non-negative prior.
    """
    if reference_flows_physical.shape != raw_delta_flows_physical.shape:
        raise ValueError("V9 reference/raw flow shapes differ")
    return reference_flows_physical + raw_delta_flows_physical


def _signed_log1p(values: torch.Tensor) -> torch.Tensor:
    """Causal deterministic scaling for physical reference-trajectory conditioning."""
    return torch.sign(values) * torch.log1p(values.abs())


class DirectHydraulicEffectSurrogateV90(DirectHydraulicEffectSurrogateV80):
    """V8 direct effect with signed outputs and optional reference-trajectory context."""

    # This is a fixed corrective diffusion schedule, not a graph-depth sweep.
    # The original four-hop path made every more-remote node structurally zero
    # for a single changed actuator despite the H360 target containing material
    # remote response.  Existing zero-preserving blocks are deliberately reused
    # cyclically: reach expands to eight hops without adding a second parameter
    # stack or changing hidden width.
    _MULTISCALE_HOPS = (1, 2, 4, 8)
    _ACTUATOR_IDENTITY_DIM = 16

    def __init__(
        self,
        *,
        reference_model: HydraulicResponseSurrogateV70,
        temporal_basis: np.ndarray,
        control_block_steps: int,
        state_delta_scale: Sequence[float] | torch.Tensor,
        flow_delta_scale: Sequence[float] | torch.Tensor,
        physics_dim: int,
        node_static_dim: int,
        actuator_count: int,
        conditioning_level: str = LEVEL_B,
        contract: DirectHydraulicEffectLossContractV90 = DirectHydraulicEffectLossContractV90(),
    ) -> None:
        contract.validate()
        super().__init__(
            reference_model=reference_model,
            temporal_basis=temporal_basis,
            control_block_steps=control_block_steps,
            state_delta_scale=state_delta_scale,
            flow_delta_scale=flow_delta_scale,
            physics_dim=physics_dim,
            node_static_dim=node_static_dim,
            actuator_count=actuator_count,
            contract=contract,
        )
        self.conditioning_level = validate_conditioning_level_v90(conditioning_level)
        # Same capacity for A/B/C.  A receives zeros in these 13 slots, B receives
        # predicted reference endpoint states/flow, C receives oracle counterparts.
        trajectory_features = 2 * 6 + 1
        self.actuator_identity_embedding = nn.Embedding(
            self.actuator_count, self._ACTUATOR_IDENTITY_DIM
        )
        old_input_dim = int(self.actuator_effect_encoder[0].in_features)
        self.actuator_effect_encoder = _mlp(
            old_input_dim + trajectory_features + self._ACTUATOR_IDENTITY_DIM,
            self.hidden_dim,
            self.hidden_dim,
        )
        # Gates only mix signed, zero-preserving diffusion states.  A non-zero
        # gate bias cannot create an action effect when every diffusion state is
        # zero, so the candidate==reference structural invariant is retained.
        self.multiscale_gate = nn.Linear(
            len(self._MULTISCALE_HOPS) * self.hidden_dim,
            len(self._MULTISCALE_HOPS),
        )

    def _multiscale_diffuse_v90(
        self,
        seed: torch.Tensor,
        prepared: PreparedStaticV80,
        *,
        reference_states_physical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """Fuse fixed 1/2/4/8-hop signed diffusion fields.

        The four inherited graph blocks are reused in a deterministic cycle to
        prevent the former hard four-hop cutoff.  All operations remain
        zero-preserving because every source field is zero under zero action.
        """
        if not self.graph_blocks:
            raise RuntimeError("V9 multiscale diffusion requires inherited graph blocks")
        value = seed
        fields: list[torch.Tensor] = []
        wanted = set(self._MULTISCALE_HOPS)
        for hop in range(1, max(self._MULTISCALE_HOPS) + 1):
            block = self.graph_blocks[(hop - 1) % len(self.graph_blocks)]
            value = block(value, prepared.edge_index, prepared.node_degree)
            if hop in wanted:
                fields.append(value)
        if len(fields) != len(self._MULTISCALE_HOPS):
            raise RuntimeError("V9 multiscale diffusion did not retain every fixed hop")
        joined = torch.cat(fields, dim=-1)
        weights = torch.softmax(self.multiscale_gate(joined), dim=-1)
        fused = sum(
            field * weights[..., index, None]
            for index, field in enumerate(fields)
        )
        return fused, weights, self._MULTISCALE_HOPS

    def _trajectory_condition(
        self,
        *,
        base_output,
        indices: torch.Tensor,
        batch: int,
        retained_count: int,
        prepared: PreparedStaticV80,
        oracle_reference_states_physical: torch.Tensor | None,
        oracle_reference_flows_physical: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conditioning_level == LEVEL_A:
            states = base_output.reference_states_physical[:, 0].new_zeros(
                batch, retained_count, base_output.reference_states_physical.shape[-2], 6
            )
            flows = base_output.reference_flows_physical[:, 0].new_zeros(
                batch, retained_count, self.actuator_count
            )
        elif self.conditioning_level == LEVEL_B:
            states = base_output.reference_states_physical[:, 0]
            flows = base_output.reference_flows_physical[:, 0]
        elif self.conditioning_level == LEVEL_C:
            if oracle_reference_states_physical is None or oracle_reference_flows_physical is None:
                raise ValueError("V9 Level C requires explicit authoritative reference trajectory")
            states = oracle_reference_states_physical
            flows = oracle_reference_flows_physical
            if states.ndim != 4 or flows.ndim != 3:
                raise ValueError("V9 oracle reference must be [B,H,N,6] and [B,H,A]")
            if states.shape[1] != retained_count:
                states = states.index_select(1, indices)
            if flows.shape[1] != retained_count:
                flows = flows.index_select(1, indices)
            if states.shape[:2] != (batch, retained_count) or flows.shape[:2] != (batch, retained_count):
                raise ValueError("V9 oracle reference horizon/batch mismatch")
        else:  # validated in __init__, defensive only
            raise RuntimeError("unreachable V9 conditioning level")

        up_state = states[:, :, prepared.base.actuator_upstream]
        down_state = states[:, :, prepared.base.actuator_downstream]
        # Managed reference flow is already actuator-aligned.
        return _signed_log1p(up_state), _signed_log1p(down_state), _signed_log1p(flows)[..., None]

    def forward(
        self,
        initial_state: torch.Tensor,
        rainfall: torch.Tensor,
        reference_settings: torch.Tensor,
        candidate_settings: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        prepared: PreparedStaticV80,
        *,
        oracle_reference_states_physical: torch.Tensor | None = None,
        oracle_reference_flows_physical: torch.Tensor | None = None,
    ) -> HydraulicEffectOutputV90:
        if candidate_settings.ndim != 4:
            raise ValueError("V9 candidate settings must be [B,C,H,A]")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if actuators != self.actuator_count:
            raise ValueError("V9 actuator count mismatch")
        if previous_actuator_flow.shape != (batch, actuators):
            raise ValueError("V9 previous actuator flow must be [B,A]")
        if reference_settings.shape != (batch, horizon, actuators):
            raise ValueError("V9 reference setting shape mismatch")
        if self.conditioning_level != LEVEL_C and (
            oracle_reference_states_physical is not None or oracle_reference_flows_physical is not None
        ):
            raise ValueError("oracle reference trajectory is forbidden outside diagnostic Level C")

        base_output, node_context = self._reference_and_context(
            initial_state, rainfall, reference_settings, candidates, prepared
        )
        indices = base_output.horizon_indices
        retained_context = node_context.index_select(1, indices)
        retained_count, node_count = retained_context.shape[1], retained_context.shape[2]

        reference_expanded = reference_settings[:, None].expand_as(candidate_settings)
        action_delta = candidate_settings - reference_expanded
        prefix_delta = self.prefix(action_delta, indices)
        current_delta = action_delta.index_select(2, indices)[..., None]

        up = retained_context[:, :, prepared.base.actuator_upstream]
        down = retained_context[:, :, prepared.base.actuator_downstream]
        physics = prepared.base.actuator_physics[None, None].expand(batch, retained_count, -1, -1)
        previous_flow = previous_actuator_flow[:, None, :, None].expand(batch, retained_count, -1, -1)
        reference_current = reference_settings.index_select(1, indices)[..., None]
        actuator_ids = torch.arange(self.actuator_count, device=initial_state.device)
        identity = self.actuator_identity_embedding(actuator_ids)[None, None].expand(
            batch, retained_count, -1, -1
        )
        time = self.time_embedding(torch.arange(retained_count, device=initial_state.device))[
            None, :, None
        ].expand(batch, -1, actuators, -1)
        ref_up, ref_down, ref_flow = self._trajectory_condition(
            base_output=base_output,
            indices=indices,
            batch=batch,
            retained_count=retained_count,
            prepared=prepared,
            oracle_reference_states_physical=oracle_reference_states_physical,
            oracle_reference_flows_physical=oracle_reference_flows_physical,
        )
        base = torch.cat(
            (
                up,
                down,
                physics,
                previous_flow,
                reference_current,
                time,
                ref_up,
                ref_down,
                ref_flow,
                identity,
            ),
            dim=-1,
        )
        base = base[:, None].expand(batch, candidates, -1, -1, -1)
        effect_features = torch.cat((prefix_delta, current_delta), dim=-1)
        zeros = torch.zeros_like(effect_features)
        token = self.actuator_effect_encoder(torch.cat((base, effect_features), dim=-1))
        token = token - self.actuator_effect_encoder(torch.cat((base, zeros), dim=-1))

        node_effect = _scatter_actuators_to_nodes(
            token,
            prepared.base.actuator_upstream,
            prepared.base.actuator_downstream,
            node_count,
        )
        node_effect, _multiscale_weights, _ = self._multiscale_diffuse_v90(
            node_effect,
            prepared,
            reference_states_physical=base_output.reference_states_physical[:, 0],
        )

        static = prepared.base.node_static[None, None].expand(batch, retained_count, -1, -1)
        gate = self.context_gate(torch.cat((retained_context, static), dim=-1))
        node_effect = node_effect * gate[:, None]

        raw_state = self.node_delta_head(node_effect)
        scale = self.state_delta_scale.to(raw_state)
        delta_depth = raw_state[..., 0] * scale[0]
        raw_delta_states = torch.stack(
            (
                delta_depth,
                delta_depth,
                raw_state[..., 1] * scale[2],
                raw_state[..., 2] * scale[3],
                raw_state[..., 3] * scale[4],
                raw_state[..., 4] * scale[5],
            ),
            dim=-1,
        )

        reference_states = base_output.reference_states_physical.expand(
            batch, candidates, retained_count, node_count, 6
        )
        projected_states = project_candidate_states_v90(
            reference_states,
            raw_delta_states,
            invert_elevation_m=prepared.base.invert_elevation_m,
        )

        endpoint_effect = 0.5 * (
            node_effect[..., prepared.base.actuator_upstream, :]
            + node_effect[..., prepared.base.actuator_downstream, :]
        )
        flow_hidden = token + self.flow_node_projection(endpoint_effect)
        raw_delta_flows = self.flow_delta_head(flow_hidden).squeeze(-1)
        raw_delta_flows = raw_delta_flows * self.flow_delta_scale.to(raw_delta_flows).reshape(
            1, 1, 1, -1
        )
        reference_flows = base_output.reference_flows_physical.expand(
            batch, candidates, retained_count, actuators
        )
        projected_flows = project_candidate_flows_v90(reference_flows, raw_delta_flows)

        temperature = torch.exp(self.onset_log_temperature).clamp(0.25, 4.0)
        reference_logits = (
            base_output.reference_flood_onset_logits.expand(
                batch, candidates, retained_count, node_count
            )
            / temperature
            + self.onset_bias
        )
        candidate_logits = reference_logits + self.onset_delta_head(node_effect).squeeze(-1)

        same_action = torch.all(candidate_settings == reference_expanded, dim=(2, 3))
        state_mask = same_action[..., None, None, None]
        flow_mask = same_action[..., None, None]
        logit_mask = same_action[..., None, None]
        raw_delta_states = torch.where(state_mask, torch.zeros_like(raw_delta_states), raw_delta_states)
        raw_delta_flows = torch.where(flow_mask, torch.zeros_like(raw_delta_flows), raw_delta_flows)
        projected_states = torch.where(state_mask, reference_states, projected_states)
        projected_flows = torch.where(flow_mask, reference_flows, projected_flows)
        candidate_logits = torch.where(logit_mask, reference_logits, candidate_logits)
        token = torch.where(same_action[..., None, None, None], torch.zeros_like(token), token)

        return HydraulicEffectOutputV90(
            horizon_indices=indices,
            reference_states_physical=reference_states,
            raw_delta_states_physical=raw_delta_states,
            candidate_states_projected_physical=projected_states,
            reference_flows_physical=reference_flows,
            raw_delta_flows_physical=raw_delta_flows,
            candidate_flows_projected_physical=projected_flows,
            reference_flood_onset_logits=reference_logits,
            candidate_flood_onset_logits=candidate_logits,
            joint_context_before_scatter=token,
        )


class PhysicalConduitHydraulicEffectSurrogateV90(DirectHydraulicEffectSurrogateV90):
    """V9 signed effect branch with a frozen conduit physical multigraph.

    This is deliberately a representation control, not a revival of the V4
    additive residual.  It keeps the V9 direct signed output equation and
    endpoint action seed, but replaces the deduplicated, unweighted legacy
    adjacency in the effect diffusion with the conduit-only directed physical
    multigraph.  Physical parallel conduits remain separate ``index_add`` rows.

    Pumps, orifices and weirs are intentionally excluded by
    :class:`ConduitPhysicalEdgeAssetsV90`: their control effect already enters
    through the actuator endpoint seed and including them again would double
    count the action pathway.
    """

    def __init__(
        self,
        *args,
        physical_edge_assets: ConduitPhysicalEdgeAssetsV90,
        head_scale_m: float,
        depth_scale_m: float,
        gradient_scale: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.conditioning_level != LEVEL_B:
            raise ValueError(
                "V9 physical conduit control is predicted-reference-only and requires LEVEL_B"
            )
        assets = physical_edge_assets
        if assets.node_count <= 0 or assets.directed_edge_count <= 0:
            raise ValueError("V9 physical conduit propagation requires nonempty edge assets")
        if assets.regulator_propagation_edge_count != 0:
            raise ValueError("V9 physical conduit propagation must exclude regulators")
        if any(link_type != "conduit" for link_type in assets.edge_to_link_type):
            raise ValueError("V9 physical edge assets admitted a non-conduit propagation row")
        if assets.static_features_normalized.shape != (
            assets.directed_edge_count,
            33,
        ):
            raise ValueError("V9 physical edge assets require [E,33] static conduit features")
        if not torch.isfinite(assets.static_features_normalized).all():
            raise ValueError("V9 physical static edge features must be finite")
        if not torch.isfinite(assets.edge_length_m).all() or torch.any(assets.edge_length_m <= 0):
            raise ValueError("V9 physical conduit lengths must be positive finite")
        for name, value in (
            ("head_scale_m", head_scale_m),
            ("depth_scale_m", depth_scale_m),
            ("gradient_scale", gradient_scale),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive finite")

        self.physical_edge_assets = assets
        self.physical_head_scale_m = float(head_scale_m)
        self.physical_depth_scale_m = float(depth_scale_m)
        self.physical_gradient_scale = float(gradient_scale)
        # No bias may leak an effect from an all-zero action seed.  Static and
        # dynamic hydraulic attributes only gate/source-transform an existing
        # signed action field; they cannot broadcast a global offset.
        self.physical_edge_condition = nn.Sequential(
            nn.Linear(33 + 7, self.hidden_dim, bias=False),
            nn.Sigmoid(),
        )
        self.physical_edge_source = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.physical_edge_destination = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.physical_edge_output = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def _physical_edge_tensors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        node_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assets = self.physical_edge_assets
        if assets.node_count != node_count:
            raise ValueError(
                "physical conduit asset node count differs from the prepared V9 graph"
            )
        edge_index = assets.edge_index.to(device=device)
        static = assets.static_features_normalized.to(device=device, dtype=dtype)
        degree = assets.in_degree.to(device=device, dtype=dtype).clamp_min(1.0)
        if edge_index.ndim != 2 or edge_index.shape != (2, static.shape[0]):
            raise RuntimeError("invalid V9 physical directed edge contract")
        if int(edge_index.max()) >= node_count or int(edge_index.min()) < 0:
            raise RuntimeError("V9 physical directed edge indices are outside graph nodes")
        return edge_index, static, degree

    def _multiscale_diffuse_v90(
        self,
        seed: torch.Tensor,
        prepared: PreparedStaticV80,
        *,
        reference_states_physical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """Causal directed conduit message diffusion at fixed 1/2/4/8 hops."""
        return self._physical_multiscale_v90(
            seed,
            prepared,
            reference_states_physical=reference_states_physical,
            activation_checkpointing=True,
        )

    def _physical_message_update_v90(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        """One zero-preserving directed physical multiedge message update."""
        _, _, _, nodes, _ = value.shape
        source_field = value[..., source, :]
        destination_field = value[..., destination, :]
        message = self.physical_edge_source(source_field) + self.physical_edge_destination(
            destination_field
        )
        message = torch.tanh(message) * condition[:, None]
        message = self.physical_edge_output(message)
        accumulated = torch.zeros_like(value)
        accumulated.index_add_(-2, destination, message)
        update = accumulated / degree.reshape(1, 1, 1, nodes, 1)
        return value + 0.5 * update

    def _physical_multiscale_v90(
        self,
        seed: torch.Tensor,
        prepared: PreparedStaticV80,
        *,
        reference_states_physical: torch.Tensor | None,
        activation_checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """Physical diffusion with optional activation recomputation.

        Recomputing the E-by-H message tensors in backward preserves the exact
        full 24-candidate loss while avoiding a second experiment-only loss
        reduction.  It is a numerically equivalent execution guard, not an
        architectural or optimizer change.
        """
        if reference_states_physical is None:
            raise ValueError("physical conduit diffusion requires frozen predicted reference states")
        if seed.ndim != 5:
            raise ValueError("V9 physical diffusion seed must be [B,C,T,N,H]")
        batch, candidates, retained, nodes, hidden = seed.shape
        if hidden != self.hidden_dim:
            raise ValueError("V9 physical diffusion hidden width mismatch")
        if reference_states_physical.shape != (batch, retained, nodes, 6):
            raise ValueError("V9 physical diffusion reference trajectory shape mismatch")
        edge_index, static, degree = self._physical_edge_tensors(
            device=seed.device, dtype=seed.dtype, node_count=nodes
        )
        source, destination = edge_index[0], edge_index[1]
        dynamic = causal_reference_dynamic_edge_features_v90(
            self.physical_edge_assets,
            reference_head_m=reference_states_physical[..., 1],
            reference_depth_m=reference_states_physical[..., 0],
            head_scale_m=self.physical_head_scale_m,
            depth_scale_m=self.physical_depth_scale_m,
            gradient_scale=self.physical_gradient_scale,
        ).to(dtype=seed.dtype)
        # [B,T,E,H]: all dynamic quantities come from the frozen causal
        # reference trajectory, never candidate or future SWMM truth.
        edge_context = torch.cat(
            (
                static[None, None].expand(batch, retained, -1, -1),
                dynamic,
            ),
            dim=-1,
        )
        condition = self.physical_edge_condition(edge_context)
        value = seed
        fields: list[torch.Tensor] = []
        wanted = set(self._MULTISCALE_HOPS)
        for hop in range(1, max(self._MULTISCALE_HOPS) + 1):
            if activation_checkpointing and torch.is_grad_enabled():
                value = activation_checkpoint(
                    self._physical_message_update_v90,
                    value,
                    condition,
                    source,
                    destination,
                    degree,
                    use_reentrant=False,
                )
            else:
                value = self._physical_message_update_v90(
                    value, condition, source, destination, degree
                )
            if hop in wanted:
                fields.append(value)
        if len(fields) != len(self._MULTISCALE_HOPS):
            raise RuntimeError("V9 physical diffusion did not retain all fixed scales")
        joined = torch.cat(fields, dim=-1)
        weights = torch.softmax(self.multiscale_gate(joined), dim=-1)
        fused = sum(
            field * weights[..., index, None]
            for index, field in enumerate(fields)
        )
        return fused, weights, self._MULTISCALE_HOPS


__all__ = [
    "CausalPrefixActionProjectorV80",
    "DirectHydraulicEffectSurrogateV90",
    "PhysicalConduitHydraulicEffectSurrogateV90",
    "HydraulicEffectOutputV90",
    "PreparedStaticV80",
    "prepare_static_v80",
    "project_candidate_flows_v90",
    "project_candidate_states_v90",
]
