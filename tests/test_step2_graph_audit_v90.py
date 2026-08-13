from __future__ import annotations

import numpy as np

from rtc.step2_graph_audit_v90 import (
    endpoint_distances_v90,
    receptive_field_mass_v90,
    undirected_adjacency_v90,
)


def test_receptive_field_census_reports_authoritative_mass_outside_current_radius():
    # 0 -- 1 -- 2 -- 3; actuator 0 seeds node 0.
    adjacency = undirected_adjacency_v90(
        np.asarray([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        node_count=4,
    )
    distances = endpoint_distances_v90(adjacency, np.asarray([0]), np.asarray([0]))
    states = np.zeros((1, 1, 4, 6), dtype=np.float64)
    states[0, 0, 0, 0] = 1.0
    states[0, 0, 3, 0] = 3.0
    flows = np.asarray([[[1.0]]], dtype=np.float64)
    report = receptive_field_mass_v90(
        distances_by_changed_actuator=distances,
        changed_actuators=np.asarray([0]),
        delta_states=states,
        delta_flows=flows,
        actuator_upstream=np.asarray([0]),
        actuator_downstream=np.asarray([0]),
        hops=(1, 3),
    )
    assert report["h1"]["delta_depth_m_inside_mass_fraction"] == 0.25
    assert report["h1"]["delta_depth_m_outside_mass_fraction"] == 0.75
    assert report["h3"]["delta_depth_m_inside_mass_fraction"] == 1.0
    assert report["h3"]["delta_managed_flow_m3s_inside_mass_fraction"] == 1.0
