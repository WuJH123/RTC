from __future__ import annotations

import numpy as np


NODE_CONTEXT_FEATURE_NAMES = (
    "rainfall_mmhr",
    "outgoing_actuator_setting_mean",
    "incoming_actuator_setting_mean",
    "outgoing_actuator_flow_m3s",
    "incoming_actuator_flow_m3s",
)


def build_node_context(
    *,
    rainfall_mmhr: np.ndarray,
    actuator_setting: np.ndarray,
    actuator_flow_m3s: np.ndarray,
    actuator_upstream: np.ndarray,
    actuator_downstream: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Project actuator readback history to compact node-local causal features.

    Inputs may have leading time dimensions; the last dimensions are node/actuator. Flow is
    kept signed so reversed hydraulic response is not destroyed.
    """

    rain = np.asarray(rainfall_mmhr, dtype=np.float32)
    setting = np.asarray(actuator_setting, dtype=np.float32)
    flow = np.asarray(actuator_flow_m3s, dtype=np.float32)
    up = np.asarray(actuator_upstream, dtype=np.int64).reshape(-1)
    down = np.asarray(actuator_downstream, dtype=np.int64).reshape(-1)
    if setting.shape != flow.shape or setting.shape[-1] != up.size or up.size != down.size:
        raise ValueError("actuator setting/flow/topology dimensions do not align")
    if rain.shape[-1] == 1:
        rain = rain[..., 0]
    if rain.shape[-1] != node_count:
        raise ValueError("rainfall must have one value per node")
    leading = setting.shape[:-1]
    if rain.shape[:-1] != leading:
        raise ValueError("rainfall and actuator histories must share leading dimensions")

    flat_n = int(np.prod(leading)) if leading else 1
    s = setting.reshape(flat_n, -1)
    q = flow.reshape(flat_n, -1)
    r = rain.reshape(flat_n, node_count)
    output = np.zeros((flat_n, node_count, len(NODE_CONTEXT_FEATURE_NAMES)), dtype=np.float32)
    output[..., 0] = r
    out_count = np.zeros(node_count, dtype=np.float32)
    in_count = np.zeros(node_count, dtype=np.float32)
    np.add.at(out_count, up, 1.0)
    np.add.at(in_count, down, 1.0)
    for i in range(flat_n):
        out_setting = np.zeros(node_count, dtype=np.float32)
        in_setting = np.zeros(node_count, dtype=np.float32)
        out_flow = np.zeros(node_count, dtype=np.float32)
        in_flow = np.zeros(node_count, dtype=np.float32)
        np.add.at(out_setting, up, s[i])
        np.add.at(in_setting, down, s[i])
        np.add.at(out_flow, up, q[i])
        np.add.at(in_flow, down, q[i])
        output[i, :, 1] = out_setting / np.maximum(out_count, 1.0)
        output[i, :, 2] = in_setting / np.maximum(in_count, 1.0)
        output[i, :, 3] = out_flow
        output[i, :, 4] = in_flow
    return output.reshape(*leading, node_count, len(NODE_CONTEXT_FEATURE_NAMES))
