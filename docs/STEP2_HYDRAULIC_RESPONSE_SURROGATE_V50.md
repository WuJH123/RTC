# STEP2_HYDRAULIC_RESPONSE_SURROGATE_V50

Train-only bounded V5.0 artifact. SWMM, Validation and Final were not accessed.

```json
{
  "contract": "STEP2_V50_TRAJECTORY_LOSS_CONTRACT",
  "independent_parameters": true,
  "value_gradient_received": false,
  "fit_group_count": 6,
  "fit_group_names": [
    "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
    "D2::T20_D120_chicago::T20_D120_chicago::T20_D120_chicago:t9000",
    "D2::T5_D60_chicago::T5_D60_chicago::T5_D60_chicago:t9000",
    "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
    "D3::T20_D120_chicago::T20_D120_chicago::T20_D120_chicago:t9000",
    "D3::T5_D60_chicago::T5_D60_chicago::T5_D60_chicago:t9000"
  ],
  "evaluation_group_count": 6,
  "evaluation_group_names": [
    "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t9600",
    "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4800",
    "D2::T10_D180_chicago::T10_D180_chicago::T10_D180_chicago:t13200",
    "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t9600",
    "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4800",
    "D3::T10_D180_chicago::T10_D180_chicago::T10_D180_chicago:t13200"
  ],
  "training": {
    "contract": "STEP2_V50_TRAJECTORY_LOSS_CONTRACT",
    "value_gradient_received": false,
    "epochs": 2,
    "groups": 6,
    "history": [
      {
        "epoch": 1,
        "loss": 0.28646407773097354,
        "groups": 6
      },
      {
        "epoch": 2,
        "loss": 0.28646407276391983,
        "groups": 6
      }
    ]
  },
  "metrics": {
    "groups": 6,
    "depth_rmse": 0.2699100077152252,
    "flooding_rate_rmse": 0.5803422331809998,
    "storage_rmse": 365.68402099609375,
    "managed_flow_rmse": 4.663137435913086
  },
  "physical_outputs": [
    "delta_depth",
    "delta_flooding_rate",
    "delta_storage_state",
    "delta_managed_actuator_flow"
  ],
  "consistency_with_direct_value": "diagnostic_only"
}
```
