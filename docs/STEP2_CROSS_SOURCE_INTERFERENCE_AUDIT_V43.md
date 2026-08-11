# STEP2 CROSS-SOURCE INTERFERENCE AUDIT V4.3

```json
{
  "contract": "STEP2_CROSS_SOURCE_PARAMETER_INTERFERENCE_AUDIT_V43",
  "parent_load": {
    "missing": [],
    "unexpected": [],
    "contract": "STEP2_RESPONSE_CALIBRATION_V41_TRAIN_ONLY_DIAGNOSTIC"
  },
  "d2_group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
  "d3_group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
  "old_d3_allowed_parameter_count": 40,
  "old_d3_changed_reference_parameters": true,
  "d2_prediction_changed_after_old_d3": true,
  "prediction_shift": {
    "delta_states_max_abs": 1.39208984375,
    "direct_tfv_max_abs": 117.5810546875
  },
  "parameter_changes": {
    "reference_encoder": {
      "changed": true,
      "parameter_count": 6
    },
    "reference_state_head": {
      "changed": true,
      "parameter_count": 2
    },
    "reference_flow_encoder": {
      "changed": true,
      "parameter_count": 6
    },
    "reference_flow_head": {
      "changed": true,
      "parameter_count": 2
    },
    "node_static_encoder": {
      "changed": false,
      "parameter_count": 6
    },
    "actuator_static_encoder": {
      "changed": false,
      "parameter_count": 6
    },
    "actuator_identity": {
      "changed": false,
      "parameter_count": 1
    },
    "temporal_identity": {
      "changed": false,
      "parameter_count": 1
    },
    "single_effect_encoder": {
      "changed": false,
      "parameter_count": 6
    },
    "single_flow_head": {
      "changed": false,
      "parameter_count": 2
    },
    "single_state_head": {
      "changed": false,
      "parameter_count": 2
    },
    "single_network_coefficient_head": {
      "changed": false,
      "parameter_count": 2
    },
    "single_node_basis_head": {
      "changed": false,
      "parameter_count": 2
    },
    "direct_single_tfv_head": {
      "changed": false,
      "parameter_count": 6
    },
    "interaction_encoder": {
      "changed": true,
      "parameter_count": 6
    },
    "interaction_state_head": {
      "changed": true,
      "parameter_count": 2
    },
    "interaction_flow_head": {
      "changed": true,
      "parameter_count": 2
    },
    "direct_interaction_tfv_head": {
      "changed": true,
      "parameter_count": 6
    }
  },
  "root_cause": "old D3 partition allowed reference/static parameters that D2 forward consumes",
  "new_reference_parameter_sha256": "ac4706705e03811c21f24a78487c48445c96e4ed694ea61bfefa53a5f8c71dcd",
  "new_d2_single_parameter_sha256": "336d7d8b828085649cb2e501fe3ec9ef47b155e2dd2d284b22556da456527c9c",
  "new_interaction_parameter_sha256": "4a9e90966662a6e2708ba91d79b4e569b968529bbb29bb046fdeea45349b2c09",
  "new_reference_freeze_pass": true,
  "new_d2_freeze_during_d3_pass": true
}
```
