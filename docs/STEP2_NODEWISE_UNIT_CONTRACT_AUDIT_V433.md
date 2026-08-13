# STEP2 NODEWISE UNIT CONTRACT AUDIT V4.3.3

```json
{
  "contract": "STEP2_NODEWISE_UNIT_CONTRACT_AUDIT_V433",
  "old_nodewise_bug": {
    "nodewise_head_output_unit": "dimensionless unconstrained scalar",
    "nodewise_head_zero_latent_unique": [
      0.023615917190909386
    ],
    "nodewise_head_zero_latent_abs_sum": 1584.722900390625,
    "trapezoid_expected_input_unit": "m3/s",
    "trapezoid_old_raw_integrated_unit": "latent_seconds (not m3)",
    "trapezoid_old_raw_integrated_value": 472115.4375,
    "d3_tfv_scale_unit": "m3 scale quantity",
    "d3_tfv_scale_value": 117846.578125,
    "d3_state_flood_scale_unit": "m3/s scale quantity",
    "d3_state_flood_scale_value": 0.5808455944061279,
    "unit_contract_bug": true,
    "bias_accumulation": true,
    "old_nodewise_replaced_global_head": true,
    "replacement_path": "nodewise_tfv_enabled=True selects nodewise head instead of direct_interaction_tfv_head"
  },
  "corrected_contract": {
    "physical_nodewise_rate_unit": "m3/s",
    "per_node_integrated_volume_unit": "m3",
    "integrates_time_only": true,
    "post_integral_d3_tfv_scale_multiplication": false,
    "zero_centered_residual": true,
    "residual_is_additive_to_old_global_head": true,
    "initial_direct_prediction_max_difference_m3": 0.0,
    "baseline_preserved_at_initialization": true,
    "timestep_invariance": true
  },
  "best_d2_backbone_sha256": "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe"
}
```
