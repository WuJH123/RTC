"""Project7 Step2 V10 frozen development contract.

V10 is a representation replacement, not another graph-depth variant. It keeps
the successful V7 Value model and V9 signed-effect semantics, while replacing
endpoint-seeded finite-hop diffusion with a regulator-aware nonlocal
actuator-to-node hydraulic influence operator.
"""
from __future__ import annotations

from dataclasses import dataclass

from .step2_v90_contract import DirectHydraulicEffectLossContractV90

V100_CONTRACT = "PROJECT7_STEP2_V100_NONLOCAL_HYDRAULIC_OPERATOR_V1"
V100_INFLUENCE_ASSET_CONTRACT = "PROJECT7_STEP2_V100_ACTUATOR_NODE_INFLUENCE_ASSETS_V1"


@dataclass(frozen=True)
class NonlocalHydraulicEffectLossContractV100(DirectHydraulicEffectLossContractV90):
    """Fixed V10 mechanism-test capacity; no architecture or epoch sweep."""

    operator_rank: int = 24
    node_identity_dim: int = 16
    pair_hidden_dim: int = 48
    candidate_chunk_size: int = 4
    local_shortcut_weight: float = 1.0
    nonlocal_weight: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.operator_rank != 24:
            raise ValueError("V10 operator rank is frozen at 24")
        if self.node_identity_dim != 16 or self.pair_hidden_dim != 48:
            raise ValueError("V10 node/pair dimensions are frozen at 16/48")
        if self.candidate_chunk_size != 4:
            raise ValueError("V10 candidate execution chunk is frozen at 4")
        if self.local_shortcut_weight != 1.0 or self.nonlocal_weight != 1.0:
            raise ValueError("V10 local/nonlocal fusion weights are frozen at 1/1")


__all__ = [
    "NonlocalHydraulicEffectLossContractV100",
    "V100_CONTRACT",
    "V100_INFLUENCE_ASSET_CONTRACT",
]
