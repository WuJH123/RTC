# STEP2_V50_FINAL_DECISION

Train-only bounded V5.0 artifact. SWMM, Validation and Final were not accessed.

```json
{
  "contract": "PROJECT7_STEP2_V50_CANDIDATE_MANIFOLD_DIRECT_RESPONSE",
  "boundary": {
    "swmm_launched": false,
    "d3_regenerated": false,
    "validation_accessed": false,
    "final_accessed": false,
    "formal_run": false,
    "full_smoke_run": false
  },
  "git_parent": "2698194cfba02eea45545cee72feb042c49e8f0d",
  "identifiability": "MARGINAL",
  "d3_v2_generated": false,
  "d3_v2_reason": "MARGINAL identifiability: no D3_V2 authorization under deterministic gate; existing Train-only cache retained",
  "data": {
    "d3_groups": 144,
    "d3_candidates": 1152,
    "d2_groups": 144,
    "d2_candidates": 3456,
    "cache_manifest_sha256": "7c69211823f5419a7fabcae03c68f6578b364d1f948f0dbfd76534c0cc48f20d",
    "micro_cohort_sha256": "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3"
  },
  "historical_baselines": {
    "V4.3.3_D3": {
      "rank": 0.365079,
      "pairwise": 0.636905,
      "top1": "1/6",
      "mean_regret_m3": 88596.7083,
      "max_regret_m3": 228632.25,
      "large_ratio": 0.71737
    },
    "V4.4.1_D3": {
      "rank": 0.349206,
      "pairwise": 0.636905,
      "top1": "1/6",
      "mean_regret_m3": 87229.4583,
      "max_regret_m3": 228632.25,
      "large_ratio": 0.730205
    }
  },
  "model": {
    "contract": "PROJECT7_STEP2_V50_CANDIDATE_MANIFOLD_DIRECT_RESPONSE",
    "value_loss_contract": "STEP2_V50_VALUE_LOSS_CONTRACT",
    "trajectory_loss_contract": "STEP2_V50_TRAJECTORY_LOSS_CONTRACT",
    "reference_frozen": true,
    "candidate_reference_shared_operator": true,
    "state_conditioned_before_set_interaction": true,
    "separate_value_and_hydraulic_output_heads": true,
    "trajectory_gradient_into_value": false,
    "causal_temporal_operator": "left-padded causal Conv1d over 36 control blocks, expanded to H72",
    "actuator_count": 109,
    "mini_pack_names": [
      "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
      "D2::T20_D120_chicago::T20_D120_chicago::T20_D120_chicago:t9000",
      "D2::T5_D60_chicago::T5_D60_chicago::T5_D60_chicago:t9000",
      "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
      "D3::T20_D120_chicago::T20_D120_chicago::T20_D120_chicago:t9000",
      "D3::T5_D60_chicago::T5_D60_chicago::T5_D60_chicago:t9000"
    ],
    "mini_train_names": [
      "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
      "D2::T20_D120_chicago::T20_D120_chicago::T20_D120_chicago:t9000",
      "D2::T5_D60_chicago::T5_D60_chicago::T5_D60_chicago:t9000",
      "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
      "D3::T20_D120_chicago::T20_D120_chicago::T20_D120_chicago:t9000",
      "D3::T5_D60_chicago::T5_D60_chicago::T5_D60_chicago:t9000"
    ],
    "fit_group_count": 240,
    "holdout_group_count": 48,
    "mini_gate": {
      "exact_zero": true,
      "future_causality": true,
      "finite_gradients": false,
      "nonzero_gradients": true,
      "d2_rank": 0.28753579096756854,
      "d3_rank": 0.28571428571428575,
      "d3_pairwise": 0.619047619047619
    },
    "mini_metrics": {
      "D2": {
        "groups": 3,
        "rank": 0.28753579096756854,
        "pairwise": 0.5937684185046107,
        "sign": 0.5603864734299516,
        "top1": 1,
        "mean_regret_m3": 853.9375,
        "max_regret_m3": 1381.3125,
        "spread_ratio": 3.7012692564953944e-07,
        "tfv_mae_m3": 5813.009129842122
      },
      "D3": {
        "groups": 3,
        "rank": 0.28571428571428575,
        "pairwise": 0.619047619047619,
        "sign": 1.0,
        "top1": 1,
        "mean_regret_m3": 32798.020833333336,
        "max_regret_m3": 84962.0,
        "spread_ratio": 1.000286024045167e-05,
        "tfv_mae_m3": 66659.91276041667
      }
    },
    "fit_metrics": {
      "D2": {
        "groups": 120,
        "rank": 0.2133778571732396,
        "pairwise": 0.3752599482610845,
        "sign": 0.4240228370663154,
        "top1": 11,
        "mean_regret_m3": 11756.799479166666,
        "max_regret_m3": 47191.65625,
        "spread_ratio": 1.3644460619694935e-08,
        "tfv_mae_m3": 6236.361423577202
      },
      "D3": {
        "groups": 120,
        "rank": 0.10040157919675659,
        "pairwise": 0.5211309523809524,
        "sign": 0.8541666666666666,
        "top1": 17,
        "mean_regret_m3": 81639.49505208334,
        "max_regret_m3": 306374.0,
        "spread_ratio": 4.146092391089107e-07,
        "tfv_mae_m3": 93786.34294331868
      }
    },
    "holdout_metrics": {
      "D2": {
        "groups": 24,
        "rank": 0.2653490669840107,
        "pairwise": 0.39597250568320125,
        "sign": 0.38309727711901626,
        "top1": 3,
        "mean_regret_m3": 11368.463541666666,
        "max_regret_m3": 37200.25,
        "spread_ratio": 1.2981534289278519e-08,
        "tfv_mae_m3": 7692.213639153375
      },
      "D3": {
        "groups": 24,
        "rank": 0.2531382836808311,
        "pairwise": 0.5863095238095238,
        "sign": 0.8177083333333334,
        "top1": 5,
        "mean_regret_m3": 85884.84375,
        "max_regret_m3": 262837.0,
        "spread_ratio": 3.3558017415511e-07,
        "tfv_mae_m3": 96493.73678588867
      }
    },
    "holdout_magnitude": {
      "small": {
        "count": 64,
        "q33": 50246.5,
        "q67": 112447.5,
        "response_ratio": 1.2881885810264539e-06,
        "mae_m3": 26805.561630249023,
        "bias_m3": 18213.124221801758,
        "rank": 0.30638548567386237,
        "pairwise": 0.6351851851851852,
        "sign": 0.71875
      },
      "medium": {
        "count": 64,
        "q33": 50246.5,
        "q67": 112447.5,
        "response_ratio": 4.893932164802211e-07,
        "mae_m3": 84214.42967224121,
        "bias_m3": 77165.56407165527,
        "rank": 0.13363384751883034,
        "pairwise": 0.5574074074074074,
        "sign": 0.953125
      },
      "large": {
        "count": 64,
        "q33": 50246.5,
        "q67": 112447.5,
        "response_ratio": 2.385505072800649e-07,
        "mae_m3": 178461.21905517578,
        "bias_m3": 65685.55920410156,
        "rank": -0.16984126984126982,
        "pairwise": 0.4148148148148148,
        "sign": 0.78125
      }
    },
    "action_gradients": {
      "1": {
        "finite_fraction": 0.9999999403953552,
        "nonzero_fraction": 0.1249999925494194,
        "median_norm": 0.015389891341328621,
        "max_norm": 0.015389891341328621
      },
      "5": {
        "finite_fraction": 0.9999999403953552,
        "nonzero_fraction": 0.1249999925494194,
        "median_norm": 0.015392336994409561,
        "max_norm": 0.015392336994409561
      },
      "10": {
        "finite_fraction": 0.9999999403953552,
        "nonzero_fraction": 0.1249999925494194,
        "median_norm": 0.015395923517644405,
        "max_norm": 0.015395923517644405
      },
      "20": {
        "finite_fraction": 0.9999999403953552,
        "nonzero_fraction": 0.1249999925494194,
        "median_norm": 0.015402606688439846,
        "max_norm": 0.015402606688439846
      },
      "109": {
        "finite_fraction": 0.9999999403953552,
        "nonzero_fraction": 0.1249999925494194,
        "median_norm": 0.015426230616867542,
        "max_norm": 0.015426230616867542
      }
    },
    "exact_zero_causality": {
      "exact_zero": true,
      "zero_delta_max": 0.0,
      "future_causality": true,
      "state_conditioned_before_scatter": true,
      "nonnegative_candidate_depth": true,
      "nonnegative_candidate_flooding": true,
      "head_depth_consistency": true
    },
    "value_parameter_sha_before_hydraulic": "6c9bbbca6b2576a7d01a151cd66a2a29892592bda8a39e5342eec576e95cd850",
    "value_parameter_sha_after_hydraulic": "6c9bbbca6b2576a7d01a151cd66a2a29892592bda8a39e5342eec576e95cd850",
    "value_parameter_invariant": true,
    "parent": {
      "checkpoint": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_calibration_v42\\04_12_group_micro\\v42_12_group_micro.pt",
      "sha256": "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe",
      "loaded": 83,
      "missing": [
        "d2_state_scale",
        "d3_state_scale",
        "d2_flow_scale",
        "d3_flow_scale",
        "d2_tfv_scale",
        "d3_tfv_scale",
        "temporal_identity.weight",
        "action_set_encoder.token_encoder.0.weight",
        "action_set_encoder.token_encoder.0.bias",
        "action_set_encoder.token_encoder.2.weight",
        "action_set_encoder.token_encoder.2.bias",
        "action_set_encoder.token_encoder.4.weight",
        "action_set_encoder.token_encoder.4.bias",
        "action_set_encoder.set_attention.in_proj_weight",
        "action_set_encoder.set_attention.in_proj_bias",
        "action_set_encoder.set_attention.out_proj.weight",
        "action_set_encoder.set_attention.out_proj.bias",
        "action_set_encoder.joint_projection.0.weight",
        "action_set_encoder.joint_projection.0.bias",
        "action_set_encoder.joint_projection.2.weight",
        "action_set_encoder.joint_projection.2.bias",
        "action_set_encoder.joint_projection.4.weight",
        "action_set_encoder.joint_projection.4.bias",
        "action_set_encoder.temporal_conv.weight",
        "action_set_encoder.temporal_conv.bias",
        "control_value.rate_head.0.weight",
        "control_value.rate_head.0.bias",
        "control_value.rate_head.2.weight",
        "control_value.rate_head.2.bias",
        "control_value.rate_head.4.weight",
        "control_value.rate_head.4.bias",
        "hydraulic_response.node_head.0.weight",
        "hydraulic_response.node_head.0.bias",
        "hydraulic_response.node_head.2.weight",
        "hydraulic_response.node_head.2.bias",
        "hydraulic_response.node_head.4.weight",
        "hydraulic_response.node_head.4.bias",
        "hydraulic_response.flow_head.0.weight",
        "hydraulic_response.flow_head.0.bias",
        "hydraulic_response.flow_head.2.weight",
        "hydraulic_response.flow_head.2.bias",
        "hydraulic_response.flow_head.4.weight",
        "hydraulic_response.flow_head.4.bias"
      ],
      "unexpected": []
    },
    "fit_parent": {
      "checkpoint": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_calibration_v42\\04_12_group_micro\\v42_12_group_micro.pt",
      "sha256": "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe",
      "loaded": 83,
      "missing": [
        "d2_state_scale",
        "d3_state_scale",
        "d2_flow_scale",
        "d3_flow_scale",
        "d2_tfv_scale",
        "d3_tfv_scale",
        "temporal_identity.weight",
        "action_set_encoder.token_encoder.0.weight",
        "action_set_encoder.token_encoder.0.bias",
        "action_set_encoder.token_encoder.2.weight",
        "action_set_encoder.token_encoder.2.bias",
        "action_set_encoder.token_encoder.4.weight",
        "action_set_encoder.token_encoder.4.bias",
        "action_set_encoder.set_attention.in_proj_weight",
        "action_set_encoder.set_attention.in_proj_bias",
        "action_set_encoder.set_attention.out_proj.weight",
        "action_set_encoder.set_attention.out_proj.bias",
        "action_set_encoder.joint_projection.0.weight",
        "action_set_encoder.joint_projection.0.bias",
        "action_set_encoder.joint_projection.2.weight",
        "action_set_encoder.joint_projection.2.bias",
        "action_set_encoder.joint_projection.4.weight",
        "action_set_encoder.joint_projection.4.bias",
        "action_set_encoder.temporal_conv.weight",
        "action_set_encoder.temporal_conv.bias",
        "control_value.rate_head.0.weight",
        "control_value.rate_head.0.bias",
        "control_value.rate_head.2.weight",
        "control_value.rate_head.2.bias",
        "control_value.rate_head.4.weight",
        "control_value.rate_head.4.bias",
        "hydraulic_response.node_head.0.weight",
        "hydraulic_response.node_head.0.bias",
        "hydraulic_response.node_head.2.weight",
        "hydraulic_response.node_head.2.bias",
        "hydraulic_response.node_head.4.weight",
        "hydraulic_response.node_head.4.bias",
        "hydraulic_response.flow_head.0.weight",
        "hydraulic_response.flow_head.0.bias",
        "hydraulic_response.flow_head.2.weight",
        "hydraulic_response.flow_head.2.bias",
        "hydraulic_response.flow_head.4.weight",
        "hydraulic_response.flow_head.4.bias"
      ],
      "unexpected": []
    },
    "wall_seconds": 633.5666668000049
  },
  "hydraulic": {
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
  },
  "train_internal_holdout": {
    "contract": "STEP2_TRAIN_INTERNAL_HOLDOUT_V50_TRAIN_ONLY",
    "split": "deterministic SHA256 complete-group split on group identity; no row split",
    "fit_group_count": 240,
    "holdout_group_count": 48,
    "fit_metrics": {
      "D2": {
        "groups": 120,
        "rank": 0.2133778571732396,
        "pairwise": 0.3752599482610845,
        "sign": 0.4240228370663154,
        "top1": 11,
        "mean_regret_m3": 11756.799479166666,
        "max_regret_m3": 47191.65625,
        "spread_ratio": 1.3644460619694935e-08,
        "tfv_mae_m3": 6236.361423577202
      },
      "D3": {
        "groups": 120,
        "rank": 0.10040157919675659,
        "pairwise": 0.5211309523809524,
        "sign": 0.8541666666666666,
        "top1": 17,
        "mean_regret_m3": 81639.49505208334,
        "max_regret_m3": 306374.0,
        "spread_ratio": 4.146092391089107e-07,
        "tfv_mae_m3": 93786.34294331868
      }
    },
    "holdout_metrics": {
      "D2": {
        "groups": 24,
        "rank": 0.2653490669840107,
        "pairwise": 0.39597250568320125,
        "sign": 0.38309727711901626,
        "top1": 3,
        "mean_regret_m3": 11368.463541666666,
        "max_regret_m3": 37200.25,
        "spread_ratio": 1.2981534289278519e-08,
        "tfv_mae_m3": 7692.213639153375
      },
      "D3": {
        "groups": 24,
        "rank": 0.2531382836808311,
        "pairwise": 0.5863095238095238,
        "sign": 0.8177083333333334,
        "top1": 5,
        "mean_regret_m3": 85884.84375,
        "max_regret_m3": 262837.0,
        "spread_ratio": 3.3558017415511e-07,
        "tfv_mae_m3": 96493.73678588867
      }
    },
    "holdout_d3_magnitude": {
      "small": {
        "count": 64,
        "q33": 50246.5,
        "q67": 112447.5,
        "response_ratio": 1.2881885810264539e-06,
        "mae_m3": 26805.561630249023,
        "bias_m3": 18213.124221801758,
        "rank": 0.30638548567386237,
        "pairwise": 0.6351851851851852,
        "sign": 0.71875
      },
      "medium": {
        "count": 64,
        "q33": 50246.5,
        "q67": 112447.5,
        "response_ratio": 4.893932164802211e-07,
        "mae_m3": 84214.42967224121,
        "bias_m3": 77165.56407165527,
        "rank": 0.13363384751883034,
        "pairwise": 0.5574074074074074,
        "sign": 0.953125
      },
      "large": {
        "count": 64,
        "q33": 50246.5,
        "q67": 112447.5,
        "response_ratio": 2.385505072800649e-07,
        "mae_m3": 178461.21905517578,
        "bias_m3": 65685.55920410156,
        "rank": -0.16984126984126982,
        "pairwise": 0.4148148148148148,
        "sign": 0.78125
      }
    },
    "hydraulic_metrics": {
      "groups": 6,
      "depth_rmse": 0.2699100077152252,
      "flooding_rate_rmse": 0.5803422331809998,
      "storage_rmse": 365.68402099609375,
      "managed_flow_rmse": 4.663137435913086
    },
    "validation_accessed": false,
    "final_accessed": false
  },
  "primary_historical_blocker": "DATA_AND_REPRESENTATION: D3 is marginally identified and existing actions are dense/full-horizon; architecture is tested but not yet promoted",
  "v50_mechanism": "INSUFFICIENT",
  "architecture_lock_candidate": false,
  "ready_for_full_train18_smoke": false,
  "ready_for_production_wiring": false,
  "ready_for_formal": false,
  "next_action": "REDUCE_CONTROL_DIMENSION",
  "wall_seconds": 633.5992212999845
}
```
