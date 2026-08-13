"""Project7 Step2 V10 regulator-aware nonlocal hydraulic-effect operator.

The V9 forensic campaign showed that endpoint-local D2 effects are learnable,
while finite-hop graph diffusion cannot place most H360 effect mass inside its
receptive field. V10 therefore removes the structural H-hop cutoff.

The model remains strictly counterfactual and causal: V7 Hydraulic is frozen;
candidate-reference action differences are the only effect source; pair
geometry is compiled only from the frozen INP/topology; no future SWMM truth or
online all-link flow is accepted; candidate==reference is exactly zero.

This is not ``sum(D2 effects) + residual``. Actuator source tokens are mixed in
latent operator space and passed through a zero-centred nonlinear node decoder,
so multi-actuator interactions are represented jointly after aggregation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .step2_control_response_v80 import PreparedStaticV80, _scatter_actuators_to_nodes
from .step2_control_response_v90 import (
    DirectHydraulicEffectSurrogateV90,
    HydraulicEffectOutputV90,
    _signed_log1p,
    project_candidate_flows_v90,
    project_candidate_states_v90,
)
from .step2_edge_physics_v441 import parse_frozen_inp_physical_links_v441
from .step2_v90_contract import LEVEL_B
from .step2_v100_contract import (
    NonlocalHydraulicEffectLossContractV100,
    V100_INFLUENCE_ASSET_CONTRACT,
)


PAIR_FEATURE_NAMES_V100 = (
    "is_upstream_endpoint",
    "is_downstream_endpoint",
    "undirected_up_proximity",
    "undirected_down_proximity",
    "forward_up_proximity",
    "forward_down_proximity",
    "reverse_up_proximity",
    "reverse_down_proximity",
    "forward_up_reachable",
    "forward_down_reachable",
    "reverse_up_reachable",
    "reverse_down_reachable",
    "same_hydraulic_component",
    "directional_balance_up",
    "directional_balance_down",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bfs(adjacency: Sequence[Sequence[int]], start: int) -> np.ndarray:
    distance = np.full(len(adjacency), -1, dtype=np.int32)
    distance[int(start)] = 0
    queue: deque[int] = deque([int(start)])
    while queue:
        node = queue.popleft()
        next_distance = int(distance[node]) + 1
        for nxt in adjacency[node]:
            if distance[nxt] < 0:
                distance[nxt] = next_distance
                queue.append(int(nxt))
    return distance


def _proximity(distance: np.ndarray) -> np.ndarray:
    values = np.zeros(distance.shape, dtype=np.float32)
    reachable = distance >= 0
    values[reachable] = 1.0 / (1.0 + distance[reachable].astype(np.float32))
    return values


@dataclass(frozen=True)
class ActuatorNodeInfluenceAssetsV100:
    contract: str
    inp_path: str
    inp_sha256: str
    node_count: int
    actuator_count: int
    physical_link_count: int
    conduit_count: int
    regulator_count: int
    outlet_count: int
    pair_feature_names: tuple[str, ...]
    pair_features: torch.Tensor
    same_component_mask: torch.Tensor
    actuator_ids: tuple[str, ...]
    uses_future_truth: bool = False
    uses_online_link_flow: bool = False

    @property
    def pair_count(self) -> int:
        return int(self.actuator_count * self.node_count)

    @property
    def reachable_pair_fraction(self) -> float:
        return float(self.same_component_mask.float().mean().item())


def build_actuator_node_influence_assets_v100(
    *,
    inp_path: str | Path,
    expected_inp_sha256: str,
    node_ids: Sequence[str],
    actuator_ids: Sequence[str],
    actuator_upstream: Sequence[int] | np.ndarray,
    actuator_downstream: Sequence[int] | np.ndarray,
) -> ActuatorNodeInfluenceAssetsV100:
    """Compile regulator-aware all-range actuator/node geometry from frozen INP.

    Unlike the V9 conduit message control, regulators remain in connectivity and
    direction descriptors. They are not extra action sources; action enters only
    once through the actuator token in the V10 model.
    """
    path = Path(inp_path).expanduser().resolve(strict=True)
    expected_sha = str(expected_inp_sha256).strip().lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("expected INP SHA256 must be a 64-character hexadecimal digest")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"frozen INP SHA256 mismatch: expected {expected_sha}, observed {actual_sha}")

    nodes = tuple(str(value) for value in node_ids)
    actuators = tuple(str(value) for value in actuator_ids)
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("V10 node IDs must be nonempty and unique")
    if not actuators or len(set(actuators)) != len(actuators):
        raise ValueError("V10 actuator IDs must be nonempty and unique")
    up = np.asarray(actuator_upstream, dtype=np.int64).reshape(-1)
    down = np.asarray(actuator_downstream, dtype=np.int64).reshape(-1)
    if up.shape != (len(actuators),) or down.shape != (len(actuators),):
        raise ValueError("V10 actuator endpoint arrays do not match actuator IDs")
    if np.any(up < 0) or np.any(up >= len(nodes)) or np.any(down < 0) or np.any(down >= len(nodes)):
        raise ValueError("V10 actuator endpoints are outside node range")

    node_index = {node_id: index for index, node_id in enumerate(nodes)}
    links = parse_frozen_inp_physical_links_v441(path, nodes)
    by_id = {link.link_id: link for link in links}
    missing = [actuator_id for actuator_id in actuators if actuator_id not in by_id]
    if missing:
        raise ValueError(f"V10 actuator IDs missing from frozen INP: {missing[:10]}")
    regulator_types = {"pump", "orifice", "weir"}
    for index, actuator_id in enumerate(actuators):
        link = by_id[actuator_id]
        if link.link_type not in regulator_types:
            raise ValueError(f"V10 actuator {actuator_id} is not a regulator link")
        if node_index[link.from_node] != int(up[index]) or node_index[link.to_node] != int(down[index]):
            raise ValueError(f"V10 actuator endpoint lineage mismatch for {actuator_id}")

    forward: list[list[int]] = [[] for _ in nodes]
    reverse: list[list[int]] = [[] for _ in nodes]
    undirected: list[list[int]] = [[] for _ in nodes]
    for link in links:
        src = node_index[link.from_node]
        dst = node_index[link.to_node]
        forward[src].append(dst)
        reverse[dst].append(src)
        undirected[src].append(dst)
        undirected[dst].append(src)

    features = np.zeros((len(actuators), len(nodes), len(PAIR_FEATURE_NAMES_V100)), dtype=np.float32)
    same_component = np.zeros((len(actuators), len(nodes)), dtype=bool)
    for actuator_index, (u, d) in enumerate(zip(up.tolist(), down.tolist())):
        und_u = _bfs(undirected, u)
        und_d = _bfs(undirected, d)
        fwd_u = _bfs(forward, u)
        fwd_d = _bfs(forward, d)
        rev_u = _bfs(reverse, u)
        rev_d = _bfs(reverse, d)
        p_und_u, p_und_d = _proximity(und_u), _proximity(und_d)
        p_fwd_u, p_fwd_d = _proximity(fwd_u), _proximity(fwd_d)
        p_rev_u, p_rev_d = _proximity(rev_u), _proximity(rev_d)
        features[actuator_index, u, 0] = 1.0
        features[actuator_index, d, 1] = 1.0
        features[actuator_index, :, 2] = p_und_u
        features[actuator_index, :, 3] = p_und_d
        features[actuator_index, :, 4] = p_fwd_u
        features[actuator_index, :, 5] = p_fwd_d
        features[actuator_index, :, 6] = p_rev_u
        features[actuator_index, :, 7] = p_rev_d
        features[actuator_index, :, 8] = (fwd_u >= 0).astype(np.float32)
        features[actuator_index, :, 9] = (fwd_d >= 0).astype(np.float32)
        features[actuator_index, :, 10] = (rev_u >= 0).astype(np.float32)
        features[actuator_index, :, 11] = (rev_d >= 0).astype(np.float32)
        connected = np.logical_or(und_u >= 0, und_d >= 0)
        same_component[actuator_index] = connected
        features[actuator_index, :, 12] = connected.astype(np.float32)
        features[actuator_index, :, 13] = p_fwd_u - p_rev_u
        features[actuator_index, :, 14] = p_fwd_d - p_rev_d

    if not np.isfinite(features).all():
        raise RuntimeError("V10 pair geometry contains non-finite values")
    if not bool(np.all(same_component[np.arange(len(actuators)), up])):
        raise RuntimeError("V10 actuator upstream endpoint lost its own component")
    if not bool(np.all(same_component[np.arange(len(actuators)), down])):
        raise RuntimeError("V10 actuator downstream endpoint lost its own component")

    type_counts = {
        kind: sum(link.link_type == kind for link in links)
        for kind in ("conduit", "pump", "orifice", "weir", "outlet")
    }
    return ActuatorNodeInfluenceAssetsV100(
        contract=V100_INFLUENCE_ASSET_CONTRACT,
        inp_path=str(path),
        inp_sha256=actual_sha,
        node_count=len(nodes),
        actuator_count=len(actuators),
        physical_link_count=len(links),
        conduit_count=int(type_counts["conduit"]),
        regulator_count=int(type_counts["pump"] + type_counts["orifice"] + type_counts["weir"]),
        outlet_count=int(type_counts["outlet"]),
        pair_feature_names=PAIR_FEATURE_NAMES_V100,
        pair_features=torch.from_numpy(features),
        same_component_mask=torch.from_numpy(same_component),
        actuator_ids=actuators,
    )


class RegulatorAwareNonlocalOperatorV100(nn.Module):
    """All-range actuator-to-node operator with a proven local shortcut."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        node_query_dim: int,
        actuator_count: int,
        node_count: int,
        assets: ActuatorNodeInfluenceAssetsV100,
        contract: NonlocalHydraulicEffectLossContractV100,
    ) -> None:
        super().__init__()
        contract.validate()
        if assets.actuator_count != actuator_count or assets.node_count != node_count:
            raise ValueError("V10 influence assets do not match model dimensions")
        if assets.uses_future_truth or assets.uses_online_link_flow:
            raise ValueError("V10 influence assets violate causal/static contract")
        self.hidden_dim = int(hidden_dim)
        self.rank = int(contract.operator_rank)
        self.actuator_count = int(actuator_count)
        self.node_count = int(node_count)
        self.register_buffer("pair_features", assets.pair_features.float())
        self.register_buffer("same_component_mask", assets.same_component_mask.bool())
        self.node_identity = nn.Embedding(node_count, contract.node_identity_dim)
        self.pair_actuator_identity = nn.Embedding(actuator_count, 8)
        pair_dim = len(assets.pair_feature_names) + contract.node_identity_dim + 8
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, contract.pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(contract.pair_hidden_dim, self.rank, bias=False),
        )
        self.source_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.node_query_gate = nn.Sequential(nn.Linear(node_query_dim, self.rank), nn.Sigmoid())
        self.rank_to_hidden = nn.Linear(self.rank, hidden_dim, bias=False)
        self.local_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim + node_query_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_weight = float(contract.local_shortcut_weight)
        self.nonlocal_weight = float(contract.nonlocal_weight)

    def _pair_kernel(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pair = self.pair_features.to(device=device, dtype=dtype)
        node_ids = torch.arange(self.node_count, device=device)
        actuator_ids = torch.arange(self.actuator_count, device=device)
        node_identity = self.node_identity(node_ids)[None].expand(self.actuator_count, -1, -1)
        actuator_identity = self.pair_actuator_identity(actuator_ids)[:, None].expand(-1, self.node_count, -1)
        kernel = torch.tanh(
            self.pair_encoder(torch.cat((pair, node_identity, actuator_identity), dim=-1))
        )
        mask = self.same_component_mask.to(device=device)[..., None]
        return torch.where(mask, kernel, torch.zeros_like(kernel))

    def forward(
        self,
        source_token: torch.Tensor,
        node_query: torch.Tensor,
        local_seed: torch.Tensor,
    ) -> torch.Tensor:
        if source_token.ndim != 5:
            raise ValueError("V10 source token must be [B,C,T,A,H]")
        if node_query.ndim != 4:
            raise ValueError("V10 node query must be [B,T,N,Q]")
        if local_seed.ndim != 5:
            raise ValueError("V10 local seed must be [B,C,T,N,H]")
        batch, candidates, retained, actuators, hidden = source_token.shape
        if actuators != self.actuator_count or hidden != self.hidden_dim:
            raise ValueError("V10 source token dimension mismatch")
        if node_query.shape[:3] != (batch, retained, self.node_count):
            raise ValueError("V10 node query dimension mismatch")
        if local_seed.shape != (batch, candidates, retained, self.node_count, hidden):
            raise ValueError("V10 local seed dimension mismatch")

        source_code = self.source_projection(source_token)
        pair_kernel = self._pair_kernel(source_code.dtype, source_code.device)
        global_rank = torch.einsum("bctar,anr->bctnr", source_code, pair_kernel)
        query_gate = self.node_query_gate(node_query)[:, None]
        global_hidden = self.rank_to_hidden(global_rank * query_gate)
        fused = self.nonlocal_weight * global_hidden + self.local_weight * self.local_projection(local_seed)

        query = node_query[:, None].expand(batch, candidates, retained, self.node_count, -1)
        zero = torch.zeros_like(fused)
        effect = self.interaction(torch.cat((fused, query), dim=-1))
        baseline = self.interaction(torch.cat((zero, query), dim=-1))
        return effect - baseline


class DirectHydraulicEffectSurrogateV100(DirectHydraulicEffectSurrogateV90):
    """V10 direct signed effect model with a nonlocal hydraulic operator."""

    def __init__(
        self,
        *args,
        influence_assets: ActuatorNodeInfluenceAssetsV100,
        contract: NonlocalHydraulicEffectLossContractV100 = NonlocalHydraulicEffectLossContractV100(),
        **kwargs,
    ) -> None:
        contract.validate()
        kwargs["conditioning_level"] = LEVEL_B
        kwargs["contract"] = contract
        super().__init__(*args, **kwargs)
        self.influence_assets = influence_assets
        node_static_dim = int(self.context_gate[0].in_features - self.reference_model.hidden_dim)
        node_query_dim = self.reference_model.hidden_dim + node_static_dim + 6
        self.nonlocal_operator = RegulatorAwareNonlocalOperatorV100(
            hidden_dim=self.hidden_dim,
            node_query_dim=node_query_dim,
            actuator_count=self.actuator_count,
            node_count=influence_assets.node_count,
            assets=influence_assets,
            contract=contract,
        )

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
        if oracle_reference_states_physical is not None or oracle_reference_flows_physical is not None:
            raise ValueError("V10 production-eligible operator forbids oracle reference trajectories")
        if candidate_settings.ndim != 4:
            raise ValueError("V10 candidate settings must be [B,C,H,A]")
        batch, candidates, horizon, actuators = candidate_settings.shape
        if actuators != self.actuator_count:
            raise ValueError("V10 actuator count mismatch")
        if previous_actuator_flow.shape != (batch, actuators):
            raise ValueError("V10 previous actuator flow must be [B,A]")
        if reference_settings.shape != (batch, horizon, actuators):
            raise ValueError("V10 reference setting shape mismatch")
        if prepared.base.node_static.shape[0] != self.influence_assets.node_count:
            raise ValueError("V10 influence assets and prepared graph node count differ")

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
            oracle_reference_states_physical=None,
            oracle_reference_flows_physical=None,
        )
        base = torch.cat(
            (up, down, physics, previous_flow, reference_current, time, ref_up, ref_down, ref_flow, identity),
            dim=-1,
        )
        base = base[:, None].expand(batch, candidates, -1, -1, -1)
        effect_features = torch.cat((prefix_delta, current_delta), dim=-1)
        zeros = torch.zeros_like(effect_features)
        token = self.actuator_effect_encoder(torch.cat((base, effect_features), dim=-1))
        token = token - self.actuator_effect_encoder(torch.cat((base, zeros), dim=-1))

        local_seed = _scatter_actuators_to_nodes(
            token,
            prepared.base.actuator_upstream,
            prepared.base.actuator_downstream,
            node_count,
        )
        static = prepared.base.node_static[None, None].expand(batch, retained_count, -1, -1)
        reference_states_base = base_output.reference_states_physical[:, 0]
        node_query = torch.cat(
            (retained_context, static, _signed_log1p(reference_states_base)), dim=-1
        )
        node_effect = self.nonlocal_operator(token, node_query, local_seed)

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


__all__ = [
    "ActuatorNodeInfluenceAssetsV100",
    "DirectHydraulicEffectSurrogateV100",
    "PAIR_FEATURE_NAMES_V100",
    "RegulatorAwareNonlocalOperatorV100",
    "build_actuator_node_influence_assets_v100",
]
