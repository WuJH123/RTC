# PROJECT7 STEP2 NODEWISE TFV CORRECTNESS V4.3.3

```json
{
  "contract": "PROJECT7_STEP2_NODEWISE_TFV_CORRECTNESS_V433",
  "git_parent": "ef732e8d64e73fb89e17439f8e6d483e6d2d84dc",
  "branch": "agent/step2-nodewise-tfv-correctness-v433",
  "boundary": {
    "swmm_launched": false,
    "d2_regenerated": false,
    "d3_regenerated": false,
    "validation_outcomes_accessed": false,
    "final_accessed": false,
    "formal_run": false,
    "full_train_smoke_run": false,
    "closed_loop_run": false,
    "policy_lock_run": false,
    "precision": "fp32"
  },
  "cohort": {
    "tiny_groups": [
      "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
      "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300"
    ],
    "micro_groups_sha256": "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3",
    "micro_groups_sha_matches_prior": true
  },
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
  "best_d2_backbone": {
    "checkpoint": {
      "stage_result": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_calibration_v42\\04_12_group_micro\\stage_result.json",
      "checkpoint": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_calibration_v42\\04_12_group_micro\\v42_12_group_micro.pt",
      "sha256": "b2d5a0c006e21afd5e178118409d41472a9f16f5ed266cd0e5d7af0995e156fe",
      "best_epoch": 22,
      "selection_policy": "d3_magnitude",
      "group_metric_count": 12
    },
    "load": {
      "missing": [],
      "unexpected": [],
      "contract": "STEP2_RESPONSE_CALIBRATION_V41_TRAIN_ONLY_DIAGNOSTIC"
    },
    "metrics": {
      "groups": 6,
      "spread_ratio": 1.2574826200307097,
      "rank": 0.7065813992894324,
      "pairwise": 0.7879621707419217,
      "sign": 0.7965250329380765,
      "top1": 3,
      "mean_regret_m3": 260.0833333333333,
      "max_regret_m3": 1560.5
    },
    "expected": {
      "rank": 0.706581,
      "pairwise": 0.787962,
      "sign": 0.796525,
      "top1": 3,
      "max_regret_m3": 1560.5
    },
    "reproduced": true
  },
  "variants": {
    "A_old_global_residual_off": {
      "name": "A_old_global_residual_off",
      "residual_enabled": false,
      "d2_before_d3": {
        "groups": 6,
        "spread_ratio": 1.2574826200307097,
        "rank": 0.7065813992894324,
        "pairwise": 0.7879621707419217,
        "sign": 0.7965250329380765,
        "top1": 3,
        "mean_regret_m3": 260.0833333333333,
        "max_regret_m3": 1560.5
      },
      "initial_tiny": {
        "groups": 1,
        "spread_ratio": 1.9233550668712005,
        "rank": 0.4285714285714286,
        "pairwise": 0.6428571428571429,
        "sign": 1.0,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      },
      "initial_tiny_contributions": [
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -50199.0,
          "predicted_additive_single_delta_tfv_m3": -260313.5625,
          "predicted_old_global_interaction_delta_tfv_m3": 113823.109375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 113823.109375,
          "predicted_final_delta_tfv_m3": -146490.453125,
          "predicted_trajectory_delta_tfv_m3": -207978.8125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -82962.5,
          "predicted_additive_single_delta_tfv_m3": -140406.578125,
          "predicted_old_global_interaction_delta_tfv_m3": -4923.25732421875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -4923.25732421875,
          "predicted_final_delta_tfv_m3": -145329.828125,
          "predicted_trajectory_delta_tfv_m3": -143232.375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -35282.0,
          "predicted_additive_single_delta_tfv_m3": -377056.65625,
          "predicted_old_global_interaction_delta_tfv_m3": 203286.03125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 203286.03125,
          "predicted_final_delta_tfv_m3": -173770.625,
          "predicted_trajectory_delta_tfv_m3": -224305.890625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -39841.5,
          "predicted_additive_single_delta_tfv_m3": -157511.09375,
          "predicted_old_global_interaction_delta_tfv_m3": 48107.88671875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 48107.88671875,
          "predicted_final_delta_tfv_m3": -109403.203125,
          "predicted_trajectory_delta_tfv_m3": -125471.015625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -120244.0,
          "predicted_additive_single_delta_tfv_m3": -131144.171875,
          "predicted_old_global_interaction_delta_tfv_m3": -102930.625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -102930.625,
          "predicted_final_delta_tfv_m3": -234074.796875,
          "predicted_trajectory_delta_tfv_m3": -179969.71875
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -2384.5,
          "predicted_additive_single_delta_tfv_m3": -12975.697265625,
          "predicted_old_global_interaction_delta_tfv_m3": 5586.56689453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 5586.56689453125,
          "predicted_final_delta_tfv_m3": -7389.13037109375,
          "predicted_trajectory_delta_tfv_m3": -81123.203125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -111839.5,
          "predicted_additive_single_delta_tfv_m3": -17884.82421875,
          "predicted_old_global_interaction_delta_tfv_m3": -81507.0546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -81507.0546875,
          "predicted_final_delta_tfv_m3": -99391.875,
          "predicted_trajectory_delta_tfv_m3": -112905.3515625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -26624.5,
          "predicted_additive_single_delta_tfv_m3": -82073.421875,
          "predicted_old_global_interaction_delta_tfv_m3": -27948.427734375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -27948.427734375,
          "predicted_final_delta_tfv_m3": -110021.8515625,
          "predicted_trajectory_delta_tfv_m3": -100326.828125
        }
      ],
      "reference_sha_before_d3": "72f5bc81f4a08d5ffb59113aaefb71abfe7515ac33bc1860b94c3a081ded7798",
      "single_sha_before_d3": "ad7fbb9f67b0fc86cb9d3359ae9d54d36b250aafa7e7c9a4bed7d1e61eba9e30",
      "tiny": {
        "groups": 1,
        "spread_ratio": 1.9233550668712005,
        "rank": 0.4285714285714286,
        "pairwise": 0.6428571428571429,
        "sign": 1.0,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      },
      "d3": {
        "groups": 6,
        "spread_ratio": 0.6827296822968577,
        "rank": 0.2658730158730159,
        "pairwise": 0.6130952380952381,
        "sign": 0.7291666666666666,
        "top1": 2,
        "mean_regret_m3": 83857.04166666667,
        "max_regret_m3": 228632.25
      },
      "d3_magnitude_strata": {
        "small": {
          "count": 15,
          "mae_m3": 66076.93330078125,
          "bias_m3": -56335.02783203125,
          "response_ratio": 3.108472519575162,
          "rank": 0.625,
          "pairwise": 0.7583333333333333,
          "sign": 0.7333333333333333
        },
        "medium": {
          "count": 12,
          "mae_m3": 57059.34716796875,
          "bias_m3": 16131.223470052084,
          "response_ratio": 0.7194012603248028,
          "rank": -0.13333333333333333,
          "pairwise": 0.4444444444444445,
          "sign": 0.75
        },
        "large": {
          "count": 21,
          "mae_m3": 168980.68696521578,
          "bias_m3": -32880.26299758184,
          "response_ratio": 0.3460953878928711,
          "rank": 0.5399999999999999,
          "pairwise": 0.7333333333333333,
          "sign": 0.7142857142857143
        }
      },
      "interaction_cancellation": {
        "small": {
          "required_interaction": {
            "count": 15,
            "mean_signed_m3": 101517.88541666667,
            "mean_abs_m3": 119528.17421875
          },
          "predicted_old_interaction": {
            "count": 15,
            "mean_signed_m3": 45182.857482910156,
            "mean_abs_m3": 65771.32982177734
          },
          "predicted_local_residual": {
            "count": 15,
            "mean_signed_m3": 0.0,
            "mean_abs_m3": 0.0
          },
          "final_interaction": {
            "count": 15,
            "mean_signed_m3": 45182.857482910156,
            "mean_abs_m3": 65771.32982177734
          }
        },
        "medium": {
          "required_interaction": {
            "count": 12,
            "mean_signed_m3": 6895.784993489583,
            "mean_abs_m3": 86204.02750651042
          },
          "predicted_old_interaction": {
            "count": 12,
            "mean_signed_m3": 23027.007446289062,
            "mean_abs_m3": 49175.2607828776
          },
          "predicted_local_residual": {
            "count": 12,
            "mean_signed_m3": 0.0,
            "mean_abs_m3": 0.0
          },
          "final_interaction": {
            "count": 12,
            "mean_signed_m3": 23027.007446289062,
            "mean_abs_m3": 49175.2607828776
          }
        },
        "large": {
          "required_interaction": {
            "count": 21,
            "mean_signed_m3": 82629.55013020833,
            "mean_abs_m3": 180006.65745907737
          },
          "predicted_old_interaction": {
            "count": 21,
            "mean_signed_m3": 49749.2873113723,
            "mean_abs_m3": 80814.7334144229
          },
          "predicted_local_residual": {
            "count": 21,
            "mean_signed_m3": 0.0,
            "mean_abs_m3": 0.0
          },
          "final_interaction": {
            "count": 21,
            "mean_signed_m3": 49749.2873113723,
            "mean_abs_m3": 80814.7334144229
          }
        }
      },
      "candidate_decomposition": [
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -50199.0,
          "predicted_additive_single_delta_tfv_m3": -260313.5625,
          "predicted_old_global_interaction_delta_tfv_m3": 113823.109375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 113823.109375,
          "predicted_final_delta_tfv_m3": -146490.453125,
          "predicted_trajectory_delta_tfv_m3": -207978.8125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -82962.5,
          "predicted_additive_single_delta_tfv_m3": -140406.578125,
          "predicted_old_global_interaction_delta_tfv_m3": -4923.25732421875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -4923.25732421875,
          "predicted_final_delta_tfv_m3": -145329.828125,
          "predicted_trajectory_delta_tfv_m3": -143232.375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -35282.0,
          "predicted_additive_single_delta_tfv_m3": -377056.65625,
          "predicted_old_global_interaction_delta_tfv_m3": 203286.03125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 203286.03125,
          "predicted_final_delta_tfv_m3": -173770.625,
          "predicted_trajectory_delta_tfv_m3": -224305.875
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -39841.5,
          "predicted_additive_single_delta_tfv_m3": -157511.09375,
          "predicted_old_global_interaction_delta_tfv_m3": 48107.88671875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 48107.88671875,
          "predicted_final_delta_tfv_m3": -109403.203125,
          "predicted_trajectory_delta_tfv_m3": -125471.015625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -120244.0,
          "predicted_additive_single_delta_tfv_m3": -131144.171875,
          "predicted_old_global_interaction_delta_tfv_m3": -102930.625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -102930.625,
          "predicted_final_delta_tfv_m3": -234074.796875,
          "predicted_trajectory_delta_tfv_m3": -179969.71875
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -2384.5,
          "predicted_additive_single_delta_tfv_m3": -12975.697265625,
          "predicted_old_global_interaction_delta_tfv_m3": 5586.56689453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 5586.56689453125,
          "predicted_final_delta_tfv_m3": -7389.13037109375,
          "predicted_trajectory_delta_tfv_m3": -81123.203125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -111839.5,
          "predicted_additive_single_delta_tfv_m3": -17884.82421875,
          "predicted_old_global_interaction_delta_tfv_m3": -81507.0546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -81507.0546875,
          "predicted_final_delta_tfv_m3": -99391.875,
          "predicted_trajectory_delta_tfv_m3": -112905.3515625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -26624.5,
          "predicted_additive_single_delta_tfv_m3": -82073.421875,
          "predicted_old_global_interaction_delta_tfv_m3": -27948.419921875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -27948.419921875,
          "predicted_final_delta_tfv_m3": -110021.84375,
          "predicted_trajectory_delta_tfv_m3": -100326.828125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -131664.5,
          "predicted_additive_single_delta_tfv_m3": -186125.765625,
          "predicted_old_global_interaction_delta_tfv_m3": 150740.59375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 150740.59375,
          "predicted_final_delta_tfv_m3": -35385.171875,
          "predicted_trajectory_delta_tfv_m3": -116864.984375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": 935.5,
          "predicted_additive_single_delta_tfv_m3": -123041.0859375,
          "predicted_old_global_interaction_delta_tfv_m3": 98347.0625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 98347.0625,
          "predicted_final_delta_tfv_m3": -24694.0234375,
          "predicted_trajectory_delta_tfv_m3": -130411.703125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": 841121.0,
          "predicted_additive_single_delta_tfv_m3": 13074.66015625,
          "predicted_old_global_interaction_delta_tfv_m3": -17316.109375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -17316.109375,
          "predicted_final_delta_tfv_m3": -4241.44921875,
          "predicted_trajectory_delta_tfv_m3": -44466.6015625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": 165412.0,
          "predicted_additive_single_delta_tfv_m3": -116529.15625,
          "predicted_old_global_interaction_delta_tfv_m3": 92769.828125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 92769.828125,
          "predicted_final_delta_tfv_m3": -23759.328125,
          "predicted_trajectory_delta_tfv_m3": -104090.7109375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -31895.0,
          "predicted_additive_single_delta_tfv_m3": 29416.943359375,
          "predicted_old_global_interaction_delta_tfv_m3": -83510.9609375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -83510.9609375,
          "predicted_final_delta_tfv_m3": -54094.015625,
          "predicted_trajectory_delta_tfv_m3": -55669.25
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -146864.5,
          "predicted_additive_single_delta_tfv_m3": -166032.234375,
          "predicted_old_global_interaction_delta_tfv_m3": 126114.109375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 126114.109375,
          "predicted_final_delta_tfv_m3": -39918.125,
          "predicted_trajectory_delta_tfv_m3": -96400.5
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -115287.0,
          "predicted_additive_single_delta_tfv_m3": 46841.2890625,
          "predicted_old_global_interaction_delta_tfv_m3": -5976.09375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -5976.09375,
          "predicted_final_delta_tfv_m3": 40865.1953125,
          "predicted_trajectory_delta_tfv_m3": -58270.47265625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -22121.5,
          "predicted_additive_single_delta_tfv_m3": -167372.96875,
          "predicted_old_global_interaction_delta_tfv_m3": 63444.5859375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 63444.5859375,
          "predicted_final_delta_tfv_m3": -103928.3828125,
          "predicted_trajectory_delta_tfv_m3": -95528.484375
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -180903.5,
          "predicted_additive_single_delta_tfv_m3": -44583.421875,
          "predicted_old_global_interaction_delta_tfv_m3": -30120.15625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -30120.15625,
          "predicted_final_delta_tfv_m3": -74703.578125,
          "predicted_trajectory_delta_tfv_m3": -118173.015625
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -227755.0,
          "predicted_additive_single_delta_tfv_m3": -308489.5,
          "predicted_old_global_interaction_delta_tfv_m3": 193256.03125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 193256.03125,
          "predicted_final_delta_tfv_m3": -115233.46875,
          "predicted_trajectory_delta_tfv_m3": -258100.234375
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -72143.0,
          "predicted_additive_single_delta_tfv_m3": -122682.25,
          "predicted_old_global_interaction_delta_tfv_m3": -4956.5166015625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -4956.5166015625,
          "predicted_final_delta_tfv_m3": -127638.765625,
          "predicted_trajectory_delta_tfv_m3": -197362.78125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -164889.0,
          "predicted_additive_single_delta_tfv_m3": -281344.1875,
          "predicted_old_global_interaction_delta_tfv_m3": 71075.5703125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 71075.5703125,
          "predicted_final_delta_tfv_m3": -210268.625,
          "predicted_trajectory_delta_tfv_m3": -280494.28125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -11749.0,
          "predicted_additive_single_delta_tfv_m3": -73070.8984375,
          "predicted_old_global_interaction_delta_tfv_m3": -42253.23046875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -42253.23046875,
          "predicted_final_delta_tfv_m3": -115324.125,
          "predicted_trajectory_delta_tfv_m3": -155175.40625
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -89015.0,
          "predicted_additive_single_delta_tfv_m3": -97193.875,
          "predicted_old_global_interaction_delta_tfv_m3": -12628.4921875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -12628.4921875,
          "predicted_final_delta_tfv_m3": -109822.3671875,
          "predicted_trajectory_delta_tfv_m3": -197152.5
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -229381.0,
          "predicted_additive_single_delta_tfv_m3": -164077.453125,
          "predicted_old_global_interaction_delta_tfv_m3": 74137.34375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 74137.34375,
          "predicted_final_delta_tfv_m3": -89940.109375,
          "predicted_trajectory_delta_tfv_m3": -179691.90625
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -153885.5,
          "predicted_additive_single_delta_tfv_m3": 10980.8037109375,
          "predicted_old_global_interaction_delta_tfv_m3": -83906.3125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -83906.3125,
          "predicted_final_delta_tfv_m3": -72925.5078125,
          "predicted_trajectory_delta_tfv_m3": -82460.921875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": 176845.0,
          "predicted_additive_single_delta_tfv_m3": -141245.90625,
          "predicted_old_global_interaction_delta_tfv_m3": 113305.4453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 113305.4453125,
          "predicted_final_delta_tfv_m3": -27940.4609375,
          "predicted_trajectory_delta_tfv_m3": -93687.4296875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -54091.5,
          "predicted_additive_single_delta_tfv_m3": -170833.390625,
          "predicted_old_global_interaction_delta_tfv_m3": 69691.3046875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 69691.3046875,
          "predicted_final_delta_tfv_m3": -101142.0859375,
          "predicted_trajectory_delta_tfv_m3": -120052.6484375
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -61415.0,
          "predicted_additive_single_delta_tfv_m3": -145095.609375,
          "predicted_old_global_interaction_delta_tfv_m3": 38562.35546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 38562.35546875,
          "predicted_final_delta_tfv_m3": -106533.25,
          "predicted_trajectory_delta_tfv_m3": -119353.21875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": 371178.0,
          "predicted_additive_single_delta_tfv_m3": -59865.13671875,
          "predicted_old_global_interaction_delta_tfv_m3": 30048.166015625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 30048.166015625,
          "predicted_final_delta_tfv_m3": -29816.970703125,
          "predicted_trajectory_delta_tfv_m3": -61848.0546875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": 225425.0,
          "predicted_additive_single_delta_tfv_m3": -227655.109375,
          "predicted_old_global_interaction_delta_tfv_m3": 133064.765625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 133064.765625,
          "predicted_final_delta_tfv_m3": -94590.34375,
          "predicted_trajectory_delta_tfv_m3": -145595.53125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": 1672.0,
          "predicted_additive_single_delta_tfv_m3": -68950.46875,
          "predicted_old_global_interaction_delta_tfv_m3": 26484.134765625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 26484.134765625,
          "predicted_final_delta_tfv_m3": -42466.3359375,
          "predicted_trajectory_delta_tfv_m3": -78418.3125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -114162.5,
          "predicted_additive_single_delta_tfv_m3": -42069.38671875,
          "predicted_old_global_interaction_delta_tfv_m3": 7299.4189453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 7299.4189453125,
          "predicted_final_delta_tfv_m3": -34769.96875,
          "predicted_trajectory_delta_tfv_m3": -82054.0078125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -146690.0,
          "predicted_additive_single_delta_tfv_m3": -70600.7109375,
          "predicted_old_global_interaction_delta_tfv_m3": 458.0623474121094,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 458.0623474121094,
          "predicted_final_delta_tfv_m3": -70142.6484375,
          "predicted_trajectory_delta_tfv_m3": -95296.375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -133446.25,
          "predicted_additive_single_delta_tfv_m3": -166211.75,
          "predicted_old_global_interaction_delta_tfv_m3": 118572.0625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 118572.0625,
          "predicted_final_delta_tfv_m3": -47639.6875,
          "predicted_trajectory_delta_tfv_m3": -104522.640625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -67508.25,
          "predicted_additive_single_delta_tfv_m3": 25293.10546875,
          "predicted_old_global_interaction_delta_tfv_m3": -20826.958984375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -20826.958984375,
          "predicted_final_delta_tfv_m3": 4466.146484375,
          "predicted_trajectory_delta_tfv_m3": -58353.51171875
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -79917.25,
          "predicted_additive_single_delta_tfv_m3": -25492.005859375,
          "predicted_old_global_interaction_delta_tfv_m3": 20434.875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 20434.875,
          "predicted_final_delta_tfv_m3": -5057.130859375,
          "predicted_trajectory_delta_tfv_m3": -54766.359375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -40815.25,
          "predicted_additive_single_delta_tfv_m3": 32949.97265625,
          "predicted_old_global_interaction_delta_tfv_m3": -700.9312133789062,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -700.9312133789062,
          "predicted_final_delta_tfv_m3": 32249.041015625,
          "predicted_trajectory_delta_tfv_m3": -49943.18359375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -125242.75,
          "predicted_additive_single_delta_tfv_m3": -256236.921875,
          "predicted_old_global_interaction_delta_tfv_m3": 193286.9375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 193286.9375,
          "predicted_final_delta_tfv_m3": -62949.984375,
          "predicted_trajectory_delta_tfv_m3": -160778.390625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -146357.25,
          "predicted_additive_single_delta_tfv_m3": -131232.890625,
          "predicted_old_global_interaction_delta_tfv_m3": 66793.8828125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 66793.8828125,
          "predicted_final_delta_tfv_m3": -64439.0078125,
          "predicted_trajectory_delta_tfv_m3": -120812.5703125
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -66564.75,
          "predicted_additive_single_delta_tfv_m3": -35931.96484375,
          "predicted_old_global_interaction_delta_tfv_m3": 57343.95703125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 57343.95703125,
          "predicted_final_delta_tfv_m3": 21411.9921875,
          "predicted_trajectory_delta_tfv_m3": -68185.9375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -62088.5,
          "predicted_additive_single_delta_tfv_m3": 11264.556640625,
          "predicted_old_global_interaction_delta_tfv_m3": -16605.314453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -16605.314453125,
          "predicted_final_delta_tfv_m3": -5340.7578125,
          "predicted_trajectory_delta_tfv_m3": -36614.31640625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -6005.0,
          "predicted_additive_single_delta_tfv_m3": -124322.2265625,
          "predicted_old_global_interaction_delta_tfv_m3": 33959.60546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 33959.60546875,
          "predicted_final_delta_tfv_m3": -90362.625,
          "predicted_trajectory_delta_tfv_m3": -81563.7109375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -63585.75,
          "predicted_additive_single_delta_tfv_m3": 30372.310546875,
          "predicted_old_global_interaction_delta_tfv_m3": -20392.16796875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -20392.16796875,
          "predicted_final_delta_tfv_m3": 9980.142578125,
          "predicted_trajectory_delta_tfv_m3": -27877.5859375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -85455.0,
          "predicted_additive_single_delta_tfv_m3": -235868.96875,
          "predicted_old_global_interaction_delta_tfv_m3": 170310.359375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 170310.359375,
          "predicted_final_delta_tfv_m3": -65558.609375,
          "predicted_trajectory_delta_tfv_m3": -140524.0625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -18681.75,
          "predicted_additive_single_delta_tfv_m3": -135146.53125,
          "predicted_old_global_interaction_delta_tfv_m3": 82017.1640625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 82017.1640625,
          "predicted_final_delta_tfv_m3": -53129.3671875,
          "predicted_trajectory_delta_tfv_m3": -88759.2578125
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -72231.5,
          "predicted_additive_single_delta_tfv_m3": 58447.453125,
          "predicted_old_global_interaction_delta_tfv_m3": -76556.8125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -76556.8125,
          "predicted_final_delta_tfv_m3": -18109.359375,
          "predicted_trajectory_delta_tfv_m3": -48872.60546875
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": 71736.75,
          "predicted_additive_single_delta_tfv_m3": -136605.34375,
          "predicted_old_global_interaction_delta_tfv_m3": 146562.0625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 146562.0625,
          "predicted_final_delta_tfv_m3": 9956.71875,
          "predicted_trajectory_delta_tfv_m3": -98226.59375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": 44440.5,
          "predicted_additive_single_delta_tfv_m3": -125109.6953125,
          "predicted_old_global_interaction_delta_tfv_m3": 87408.953125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 87408.953125,
          "predicted_final_delta_tfv_m3": -37700.7421875,
          "predicted_trajectory_delta_tfv_m3": -74878.84375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -234637.25,
          "predicted_additive_single_delta_tfv_m3": 1942.7216796875,
          "predicted_old_global_interaction_delta_tfv_m3": -4430.83251953125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -4430.83251953125,
          "predicted_final_delta_tfv_m3": -2488.11083984375,
          "predicted_trajectory_delta_tfv_m3": -19143.79296875
        }
      ],
      "micro": {
        "trained": false
      },
      "d2_preserved": true
    },
    "B_corrected_nodewise_residual": {
      "name": "B_corrected_nodewise_residual",
      "residual_enabled": true,
      "d2_before_d3": {
        "groups": 6,
        "spread_ratio": 1.2574826200307097,
        "rank": 0.7065813992894324,
        "pairwise": 0.7879621707419217,
        "sign": 0.7965250329380765,
        "top1": 3,
        "mean_regret_m3": 260.0833333333333,
        "max_regret_m3": 1560.5
      },
      "initial_tiny": {
        "groups": 1,
        "spread_ratio": 1.9233550668712005,
        "rank": 0.4285714285714286,
        "pairwise": 0.6428571428571429,
        "sign": 1.0,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      },
      "initial_tiny_contributions": [
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -50199.0,
          "predicted_additive_single_delta_tfv_m3": -260313.5625,
          "predicted_old_global_interaction_delta_tfv_m3": 113823.109375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 113823.109375,
          "predicted_final_delta_tfv_m3": -146490.453125,
          "predicted_trajectory_delta_tfv_m3": -207978.8125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -82962.5,
          "predicted_additive_single_delta_tfv_m3": -140406.578125,
          "predicted_old_global_interaction_delta_tfv_m3": -4923.25732421875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -4923.25732421875,
          "predicted_final_delta_tfv_m3": -145329.828125,
          "predicted_trajectory_delta_tfv_m3": -143232.375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -35282.0,
          "predicted_additive_single_delta_tfv_m3": -377056.65625,
          "predicted_old_global_interaction_delta_tfv_m3": 203286.03125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 203286.03125,
          "predicted_final_delta_tfv_m3": -173770.625,
          "predicted_trajectory_delta_tfv_m3": -224305.890625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -39841.5,
          "predicted_additive_single_delta_tfv_m3": -157511.09375,
          "predicted_old_global_interaction_delta_tfv_m3": 48107.88671875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 48107.88671875,
          "predicted_final_delta_tfv_m3": -109403.203125,
          "predicted_trajectory_delta_tfv_m3": -125471.015625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -120244.0,
          "predicted_additive_single_delta_tfv_m3": -131144.171875,
          "predicted_old_global_interaction_delta_tfv_m3": -102930.625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -102930.625,
          "predicted_final_delta_tfv_m3": -234074.796875,
          "predicted_trajectory_delta_tfv_m3": -179969.71875
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -2384.5,
          "predicted_additive_single_delta_tfv_m3": -12975.697265625,
          "predicted_old_global_interaction_delta_tfv_m3": 5586.56689453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": 5586.56689453125,
          "predicted_final_delta_tfv_m3": -7389.13037109375,
          "predicted_trajectory_delta_tfv_m3": -81123.203125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -111839.5,
          "predicted_additive_single_delta_tfv_m3": -17884.82421875,
          "predicted_old_global_interaction_delta_tfv_m3": -81507.0546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -81507.0546875,
          "predicted_final_delta_tfv_m3": -99391.875,
          "predicted_trajectory_delta_tfv_m3": -112905.3515625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -26624.5,
          "predicted_additive_single_delta_tfv_m3": -82073.421875,
          "predicted_old_global_interaction_delta_tfv_m3": -27948.427734375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 0.0,
          "predicted_interaction_delta_tfv_m3": -27948.427734375,
          "predicted_final_delta_tfv_m3": -110021.8515625,
          "predicted_trajectory_delta_tfv_m3": -100326.828125
        }
      ],
      "reference_sha_before_d3": "72f5bc81f4a08d5ffb59113aaefb71abfe7515ac33bc1860b94c3a081ded7798",
      "single_sha_before_d3": "ad7fbb9f67b0fc86cb9d3359ae9d54d36b250aafa7e7c9a4bed7d1e61eba9e30",
      "initial_prediction_max_difference_vs_baseline": null,
      "initial_residual_gradient": {
        "objective": -128233.96875,
        "parameter_names": [
          "nodewise_residual_correction.0.weight",
          "nodewise_residual_correction.0.bias",
          "nodewise_residual_correction.2.weight",
          "nodewise_residual_correction.2.bias",
          "nodewise_residual_correction.4.weight",
          "nodewise_residual_correction.4.bias"
        ],
        "gradient_finite": true,
        "gradient_l2": 85878.625,
        "gradient_nonzero": true
      },
      "tiny": {
        "metrics": {
          "groups": 1,
          "spread_ratio": 1.9142236177530236,
          "rank": 0.4523809523809524,
          "pairwise": 0.6785714285714286,
          "sign": 1.0,
          "top1": 1,
          "mean_regret_m3": 0.0,
          "max_regret_m3": 0.0
        },
        "history": [
          {
            "epoch": 1,
            "loss": 4.140989780426025,
            "gradient_norm": 0.3154429495334625,
            "d3_rank": 0.4285714285714286,
            "d3_pairwise": 0.6428571428571429,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 2,
            "loss": 4.13913106918335,
            "gradient_norm": 0.29272639751434326,
            "d3_rank": 0.4285714285714286,
            "d3_pairwise": 0.6428571428571429,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 3,
            "loss": 4.1368608474731445,
            "gradient_norm": 0.2750684320926666,
            "d3_rank": 0.4285714285714286,
            "d3_pairwise": 0.6428571428571429,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 4,
            "loss": 4.134077072143555,
            "gradient_norm": 0.25965723395347595,
            "d3_rank": 0.4285714285714286,
            "d3_pairwise": 0.6428571428571429,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 5,
            "loss": 4.130676746368408,
            "gradient_norm": 0.24856089055538177,
            "d3_rank": 0.4285714285714286,
            "d3_pairwise": 0.6428571428571429,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 6,
            "loss": 4.126490592956543,
            "gradient_norm": 0.24567753076553345,
            "d3_rank": 0.4285714285714286,
            "d3_pairwise": 0.6428571428571429,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 7,
            "loss": 4.1213154792785645,
            "gradient_norm": 0.24841012060642242,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 8,
            "loss": 4.11492395401001,
            "gradient_norm": 0.25778910517692566,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 9,
            "loss": 4.107091903686523,
            "gradient_norm": 0.27409350872039795,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 10,
            "loss": 4.097610950469971,
            "gradient_norm": 0.29670852422714233,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 11,
            "loss": 4.086303234100342,
            "gradient_norm": 0.32407015562057495,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 12,
            "loss": 4.073046684265137,
            "gradient_norm": 0.3536618947982788,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 13,
            "loss": 4.057816982269287,
            "gradient_norm": 0.3820628225803375,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 14,
            "loss": 4.040750980377197,
            "gradient_norm": 0.4049759805202484,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          },
          {
            "epoch": 15,
            "loss": 4.0222249031066895,
            "gradient_norm": 0.41745927929878235,
            "d3_rank": 0.4523809523809524,
            "d3_pairwise": 0.6785714285714286,
            "d3_sign": 1.0,
            "d3_top1": 1,
            "d3_max_regret_m3": 0.0
          }
        ],
        "profile": {
          "forward_seconds": 1.2335544000379741,
          "backward_seconds": 0.3274626001366414,
          "optimizer_seconds": 0.13550919998669997,
          "wall_time_seconds": 2.758054300036747
        },
        "contributions": [
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 0,
            "true_delta_tfv_m3": -50199.0,
            "predicted_additive_single_delta_tfv_m3": -260313.5625,
            "predicted_old_global_interaction_delta_tfv_m3": 113823.109375,
            "predicted_nodewise_local_residual_delta_tfv_m3": 8223.6650390625,
            "predicted_interaction_delta_tfv_m3": 122046.7734375,
            "predicted_final_delta_tfv_m3": -138266.78125,
            "predicted_trajectory_delta_tfv_m3": -207978.8125
          },
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 1,
            "true_delta_tfv_m3": -82962.5,
            "predicted_additive_single_delta_tfv_m3": -140406.578125,
            "predicted_old_global_interaction_delta_tfv_m3": -4923.2568359375,
            "predicted_nodewise_local_residual_delta_tfv_m3": -7862.5263671875,
            "predicted_interaction_delta_tfv_m3": -12785.783203125,
            "predicted_final_delta_tfv_m3": -153192.359375,
            "predicted_trajectory_delta_tfv_m3": -143232.375
          },
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 2,
            "true_delta_tfv_m3": -35282.0,
            "predicted_additive_single_delta_tfv_m3": -377056.65625,
            "predicted_old_global_interaction_delta_tfv_m3": 203286.03125,
            "predicted_nodewise_local_residual_delta_tfv_m3": 11663.9814453125,
            "predicted_interaction_delta_tfv_m3": 214950.015625,
            "predicted_final_delta_tfv_m3": -162106.640625,
            "predicted_trajectory_delta_tfv_m3": -224305.890625
          },
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 3,
            "true_delta_tfv_m3": -39841.5,
            "predicted_additive_single_delta_tfv_m3": -157511.09375,
            "predicted_old_global_interaction_delta_tfv_m3": 48107.88671875,
            "predicted_nodewise_local_residual_delta_tfv_m3": -1442.8724365234375,
            "predicted_interaction_delta_tfv_m3": 46665.015625,
            "predicted_final_delta_tfv_m3": -110846.078125,
            "predicted_trajectory_delta_tfv_m3": -125471.015625
          },
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 4,
            "true_delta_tfv_m3": -120244.0,
            "predicted_additive_single_delta_tfv_m3": -131144.171875,
            "predicted_old_global_interaction_delta_tfv_m3": -102930.625,
            "predicted_nodewise_local_residual_delta_tfv_m3": -2927.6513671875,
            "predicted_interaction_delta_tfv_m3": -105858.2734375,
            "predicted_final_delta_tfv_m3": -237002.4375,
            "predicted_trajectory_delta_tfv_m3": -179969.71875
          },
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 5,
            "true_delta_tfv_m3": -2384.5,
            "predicted_additive_single_delta_tfv_m3": -12975.697265625,
            "predicted_old_global_interaction_delta_tfv_m3": 5586.56689453125,
            "predicted_nodewise_local_residual_delta_tfv_m3": -4003.883056640625,
            "predicted_interaction_delta_tfv_m3": 1582.683837890625,
            "predicted_final_delta_tfv_m3": -11393.013671875,
            "predicted_trajectory_delta_tfv_m3": -81123.203125
          },
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 6,
            "true_delta_tfv_m3": -111839.5,
            "predicted_additive_single_delta_tfv_m3": -17884.82421875,
            "predicted_old_global_interaction_delta_tfv_m3": -81507.0546875,
            "predicted_nodewise_local_residual_delta_tfv_m3": -10098.1943359375,
            "predicted_interaction_delta_tfv_m3": -91605.25,
            "predicted_final_delta_tfv_m3": -109490.078125,
            "predicted_trajectory_delta_tfv_m3": -112905.3515625
          },
          {
            "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
            "source_kind": "D3",
            "candidate_index": 7,
            "true_delta_tfv_m3": -26624.5,
            "predicted_additive_single_delta_tfv_m3": -82073.421875,
            "predicted_old_global_interaction_delta_tfv_m3": -27948.419921875,
            "predicted_nodewise_local_residual_delta_tfv_m3": -9492.216796875,
            "predicted_interaction_delta_tfv_m3": -37440.63671875,
            "predicted_final_delta_tfv_m3": -119514.0625,
            "predicted_trajectory_delta_tfv_m3": -100326.828125
          }
        ]
      },
      "tiny_pass": true,
      "d3": {
        "groups": 6,
        "spread_ratio": 0.7173724756594394,
        "rank": 0.36507936507936517,
        "pairwise": 0.6369047619047619,
        "sign": 0.7916666666666666,
        "top1": 1,
        "mean_regret_m3": 88596.70833333333,
        "max_regret_m3": 228632.25
      },
      "d3_magnitude_strata": {
        "small": {
          "count": 15,
          "mae_m3": 78527.79410807292,
          "bias_m3": -72024.15738932292,
          "response_ratio": 3.5913903364187787,
          "rank": 0.6749999999999999,
          "pairwise": 0.7833333333333333,
          "sign": 0.7333333333333333
        },
        "medium": {
          "count": 12,
          "mae_m3": 48327.344889322914,
          "bias_m3": -35.63818359375,
          "response_ratio": 0.8364486173939751,
          "rank": 0.0666666666666667,
          "pairwise": 0.5,
          "sign": 0.9166666666666666
        },
        "large": {
          "count": 21,
          "mae_m3": 160848.67764136905,
          "bias_m3": -52039.85546875,
          "response_ratio": 0.42130151415546263,
          "rank": 0.25999999999999995,
          "pairwise": 0.6,
          "sign": 0.7619047619047619
        }
      },
      "interaction_cancellation": {
        "small": {
          "required_interaction": {
            "count": 15,
            "mean_signed_m3": 101517.88541666667,
            "mean_abs_m3": 119528.17421875
          },
          "predicted_old_interaction": {
            "count": 15,
            "mean_signed_m3": 45182.85745442708,
            "mean_abs_m3": 65771.33089192708
          },
          "predicted_local_residual": {
            "count": 15,
            "mean_signed_m3": -15689.130338541667,
            "mean_abs_m3": 16343.222916666668
          },
          "final_interaction": {
            "count": 15,
            "mean_signed_m3": 29493.726725260418,
            "mean_abs_m3": 54454.70224609375
          }
        },
        "medium": {
          "required_interaction": {
            "count": 12,
            "mean_signed_m3": 6895.784993489583,
            "mean_abs_m3": 86204.02750651042
          },
          "predicted_old_interaction": {
            "count": 12,
            "mean_signed_m3": 23027.007446289062,
            "mean_abs_m3": 49175.2607828776
          },
          "predicted_local_residual": {
            "count": 12,
            "mean_signed_m3": -16166.861602783203,
            "mean_abs_m3": 17447.29350789388
          },
          "final_interaction": {
            "count": 12,
            "mean_signed_m3": 6860.1462809244795,
            "mean_abs_m3": 45926.94063313802
          }
        },
        "large": {
          "required_interaction": {
            "count": 21,
            "mean_signed_m3": 82629.55013020833,
            "mean_abs_m3": 180006.65745907737
          },
          "predicted_old_interaction": {
            "count": 21,
            "mean_signed_m3": 49749.287202380954,
            "mean_abs_m3": 80814.73344494047
          },
          "predicted_local_residual": {
            "count": 21,
            "mean_signed_m3": -19159.59156436012,
            "mean_abs_m3": 20443.117234002977
          },
          "final_interaction": {
            "count": 21,
            "mean_signed_m3": 30589.69503348214,
            "mean_abs_m3": 75300.19745163691
          }
        }
      },
      "candidate_decomposition": [
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -50199.0,
          "predicted_additive_single_delta_tfv_m3": -260313.5625,
          "predicted_old_global_interaction_delta_tfv_m3": 113823.109375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -6664.05419921875,
          "predicted_interaction_delta_tfv_m3": 107159.0546875,
          "predicted_final_delta_tfv_m3": -153154.5,
          "predicted_trajectory_delta_tfv_m3": -207978.8125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -82962.5,
          "predicted_additive_single_delta_tfv_m3": -140406.578125,
          "predicted_old_global_interaction_delta_tfv_m3": -4923.25732421875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 983.2523193359375,
          "predicted_interaction_delta_tfv_m3": -3940.0048828125,
          "predicted_final_delta_tfv_m3": -144346.578125,
          "predicted_trajectory_delta_tfv_m3": -143232.375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -35282.0,
          "predicted_additive_single_delta_tfv_m3": -377056.65625,
          "predicted_old_global_interaction_delta_tfv_m3": 203286.03125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -10795.1689453125,
          "predicted_interaction_delta_tfv_m3": 192490.859375,
          "predicted_final_delta_tfv_m3": -184565.796875,
          "predicted_trajectory_delta_tfv_m3": -224305.890625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -39841.5,
          "predicted_additive_single_delta_tfv_m3": -157511.09375,
          "predicted_old_global_interaction_delta_tfv_m3": 48107.88671875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -3249.49169921875,
          "predicted_interaction_delta_tfv_m3": 44858.39453125,
          "predicted_final_delta_tfv_m3": -112652.703125,
          "predicted_trajectory_delta_tfv_m3": -125471.015625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -120244.0,
          "predicted_additive_single_delta_tfv_m3": -131144.171875,
          "predicted_old_global_interaction_delta_tfv_m3": -102930.625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -6614.41162109375,
          "predicted_interaction_delta_tfv_m3": -109545.0390625,
          "predicted_final_delta_tfv_m3": -240689.21875,
          "predicted_trajectory_delta_tfv_m3": -179969.71875
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -2384.5,
          "predicted_additive_single_delta_tfv_m3": -12975.697265625,
          "predicted_old_global_interaction_delta_tfv_m3": 5586.56689453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 123.880859375,
          "predicted_interaction_delta_tfv_m3": 5710.44775390625,
          "predicted_final_delta_tfv_m3": -7265.24951171875,
          "predicted_trajectory_delta_tfv_m3": -81123.203125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -111839.5,
          "predicted_additive_single_delta_tfv_m3": -17884.82421875,
          "predicted_old_global_interaction_delta_tfv_m3": -81507.0546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 1214.008544921875,
          "predicted_interaction_delta_tfv_m3": -80293.046875,
          "predicted_final_delta_tfv_m3": -98177.875,
          "predicted_trajectory_delta_tfv_m3": -112905.3515625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -26624.5,
          "predicted_additive_single_delta_tfv_m3": -82073.421875,
          "predicted_old_global_interaction_delta_tfv_m3": -27948.427734375,
          "predicted_nodewise_local_residual_delta_tfv_m3": 1130.605712890625,
          "predicted_interaction_delta_tfv_m3": -26817.822265625,
          "predicted_final_delta_tfv_m3": -108891.2421875,
          "predicted_trajectory_delta_tfv_m3": -100326.828125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -131664.5,
          "predicted_additive_single_delta_tfv_m3": -186125.765625,
          "predicted_old_global_interaction_delta_tfv_m3": 150740.59375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -32096.4296875,
          "predicted_interaction_delta_tfv_m3": 118644.1640625,
          "predicted_final_delta_tfv_m3": -67481.6015625,
          "predicted_trajectory_delta_tfv_m3": -116864.984375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": 935.5,
          "predicted_additive_single_delta_tfv_m3": -123041.0859375,
          "predicted_old_global_interaction_delta_tfv_m3": 98347.0625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -25380.302734375,
          "predicted_interaction_delta_tfv_m3": 72966.7578125,
          "predicted_final_delta_tfv_m3": -50074.328125,
          "predicted_trajectory_delta_tfv_m3": -130411.703125
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": 841121.0,
          "predicted_additive_single_delta_tfv_m3": 13074.66015625,
          "predicted_old_global_interaction_delta_tfv_m3": -17316.111328125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -18323.912109375,
          "predicted_interaction_delta_tfv_m3": -35640.0234375,
          "predicted_final_delta_tfv_m3": -22565.36328125,
          "predicted_trajectory_delta_tfv_m3": -44466.6015625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": 165412.0,
          "predicted_additive_single_delta_tfv_m3": -116529.15625,
          "predicted_old_global_interaction_delta_tfv_m3": 92769.828125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -21472.57421875,
          "predicted_interaction_delta_tfv_m3": 71297.25,
          "predicted_final_delta_tfv_m3": -45231.90625,
          "predicted_trajectory_delta_tfv_m3": -104090.7109375
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -31895.0,
          "predicted_additive_single_delta_tfv_m3": 29416.943359375,
          "predicted_old_global_interaction_delta_tfv_m3": -83510.9609375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -13288.5634765625,
          "predicted_interaction_delta_tfv_m3": -96799.5234375,
          "predicted_final_delta_tfv_m3": -67382.578125,
          "predicted_trajectory_delta_tfv_m3": -55669.25
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -146864.5,
          "predicted_additive_single_delta_tfv_m3": -166032.234375,
          "predicted_old_global_interaction_delta_tfv_m3": 126114.109375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -22013.765625,
          "predicted_interaction_delta_tfv_m3": 104100.34375,
          "predicted_final_delta_tfv_m3": -61931.890625,
          "predicted_trajectory_delta_tfv_m3": -96400.5
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -115287.0,
          "predicted_additive_single_delta_tfv_m3": 46841.2890625,
          "predicted_old_global_interaction_delta_tfv_m3": -5976.09375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -69816.6484375,
          "predicted_interaction_delta_tfv_m3": -75792.7421875,
          "predicted_final_delta_tfv_m3": -28951.453125,
          "predicted_trajectory_delta_tfv_m3": -58270.47265625
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -22121.5,
          "predicted_additive_single_delta_tfv_m3": -167372.96875,
          "predicted_old_global_interaction_delta_tfv_m3": 63444.5859375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -22512.87890625,
          "predicted_interaction_delta_tfv_m3": 40931.70703125,
          "predicted_final_delta_tfv_m3": -126441.265625,
          "predicted_trajectory_delta_tfv_m3": -95528.484375
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -180903.5,
          "predicted_additive_single_delta_tfv_m3": -44583.421875,
          "predicted_old_global_interaction_delta_tfv_m3": -30120.15625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 2187.595947265625,
          "predicted_interaction_delta_tfv_m3": -27932.560546875,
          "predicted_final_delta_tfv_m3": -72515.984375,
          "predicted_trajectory_delta_tfv_m3": -118173.015625
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -227755.0,
          "predicted_additive_single_delta_tfv_m3": -308489.5,
          "predicted_old_global_interaction_delta_tfv_m3": 193256.03125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -9844.23828125,
          "predicted_interaction_delta_tfv_m3": 183411.796875,
          "predicted_final_delta_tfv_m3": -125077.703125,
          "predicted_trajectory_delta_tfv_m3": -258100.234375
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -72143.0,
          "predicted_additive_single_delta_tfv_m3": -122682.25,
          "predicted_old_global_interaction_delta_tfv_m3": -4956.5166015625,
          "predicted_nodewise_local_residual_delta_tfv_m3": 3490.72998046875,
          "predicted_interaction_delta_tfv_m3": -1465.78662109375,
          "predicted_final_delta_tfv_m3": -124148.0390625,
          "predicted_trajectory_delta_tfv_m3": -197362.78125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -164889.0,
          "predicted_additive_single_delta_tfv_m3": -281344.1875,
          "predicted_old_global_interaction_delta_tfv_m3": 71075.5703125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -5509.5361328125,
          "predicted_interaction_delta_tfv_m3": 65566.03125,
          "predicted_final_delta_tfv_m3": -215778.15625,
          "predicted_trajectory_delta_tfv_m3": -280494.28125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -11749.0,
          "predicted_additive_single_delta_tfv_m3": -73070.8984375,
          "predicted_old_global_interaction_delta_tfv_m3": -42253.23046875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 3651.207763671875,
          "predicted_interaction_delta_tfv_m3": -38602.0234375,
          "predicted_final_delta_tfv_m3": -111672.921875,
          "predicted_trajectory_delta_tfv_m3": -155175.40625
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -89015.0,
          "predicted_additive_single_delta_tfv_m3": -97193.875,
          "predicted_old_global_interaction_delta_tfv_m3": -12628.4921875,
          "predicted_nodewise_local_residual_delta_tfv_m3": 3208.609130859375,
          "predicted_interaction_delta_tfv_m3": -9419.8828125,
          "predicted_final_delta_tfv_m3": -106613.7578125,
          "predicted_trajectory_delta_tfv_m3": -197152.5
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -229381.0,
          "predicted_additive_single_delta_tfv_m3": -164077.453125,
          "predicted_old_global_interaction_delta_tfv_m3": 74137.34375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -5547.54931640625,
          "predicted_interaction_delta_tfv_m3": 68589.796875,
          "predicted_final_delta_tfv_m3": -95487.65625,
          "predicted_trajectory_delta_tfv_m3": -179691.90625
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -153885.5,
          "predicted_additive_single_delta_tfv_m3": 10980.8037109375,
          "predicted_old_global_interaction_delta_tfv_m3": -83906.3125,
          "predicted_nodewise_local_residual_delta_tfv_m3": 10075.4150390625,
          "predicted_interaction_delta_tfv_m3": -73830.8984375,
          "predicted_final_delta_tfv_m3": -62850.09375,
          "predicted_trajectory_delta_tfv_m3": -82460.921875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": 176845.0,
          "predicted_additive_single_delta_tfv_m3": -141245.90625,
          "predicted_old_global_interaction_delta_tfv_m3": 113305.4453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -21654.015625,
          "predicted_interaction_delta_tfv_m3": 91651.4296875,
          "predicted_final_delta_tfv_m3": -49594.4765625,
          "predicted_trajectory_delta_tfv_m3": -93687.4296875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -54091.5,
          "predicted_additive_single_delta_tfv_m3": -170833.390625,
          "predicted_old_global_interaction_delta_tfv_m3": 69691.3046875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -38214.046875,
          "predicted_interaction_delta_tfv_m3": 31477.2578125,
          "predicted_final_delta_tfv_m3": -139356.125,
          "predicted_trajectory_delta_tfv_m3": -120052.6484375
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -61415.0,
          "predicted_additive_single_delta_tfv_m3": -145095.609375,
          "predicted_old_global_interaction_delta_tfv_m3": 38562.35546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -20404.53125,
          "predicted_interaction_delta_tfv_m3": 18157.82421875,
          "predicted_final_delta_tfv_m3": -126937.78125,
          "predicted_trajectory_delta_tfv_m3": -119353.21875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": 371178.0,
          "predicted_additive_single_delta_tfv_m3": -59865.13671875,
          "predicted_old_global_interaction_delta_tfv_m3": 30048.166015625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -20128.537109375,
          "predicted_interaction_delta_tfv_m3": 9919.62890625,
          "predicted_final_delta_tfv_m3": -49945.5078125,
          "predicted_trajectory_delta_tfv_m3": -61848.0546875
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": 225425.0,
          "predicted_additive_single_delta_tfv_m3": -227655.109375,
          "predicted_old_global_interaction_delta_tfv_m3": 133064.765625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -22086.625,
          "predicted_interaction_delta_tfv_m3": 110978.140625,
          "predicted_final_delta_tfv_m3": -116676.96875,
          "predicted_trajectory_delta_tfv_m3": -145595.53125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": 1672.0,
          "predicted_additive_single_delta_tfv_m3": -68950.46875,
          "predicted_old_global_interaction_delta_tfv_m3": 26484.134765625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -17938.46484375,
          "predicted_interaction_delta_tfv_m3": 8545.669921875,
          "predicted_final_delta_tfv_m3": -60404.796875,
          "predicted_trajectory_delta_tfv_m3": -78418.3125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -114162.5,
          "predicted_additive_single_delta_tfv_m3": -42069.38671875,
          "predicted_old_global_interaction_delta_tfv_m3": 7299.41796875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -36780.12890625,
          "predicted_interaction_delta_tfv_m3": -29480.7109375,
          "predicted_final_delta_tfv_m3": -71550.09375,
          "predicted_trajectory_delta_tfv_m3": -82054.0078125
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -146690.0,
          "predicted_additive_single_delta_tfv_m3": -70600.7109375,
          "predicted_old_global_interaction_delta_tfv_m3": 458.0625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -16954.203125,
          "predicted_interaction_delta_tfv_m3": -16496.140625,
          "predicted_final_delta_tfv_m3": -87096.8515625,
          "predicted_trajectory_delta_tfv_m3": -95296.375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -133446.25,
          "predicted_additive_single_delta_tfv_m3": -166211.75,
          "predicted_old_global_interaction_delta_tfv_m3": 118572.0625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -43537.5390625,
          "predicted_interaction_delta_tfv_m3": 75034.5234375,
          "predicted_final_delta_tfv_m3": -91177.2265625,
          "predicted_trajectory_delta_tfv_m3": -104522.640625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -67508.25,
          "predicted_additive_single_delta_tfv_m3": 25293.10546875,
          "predicted_old_global_interaction_delta_tfv_m3": -20826.958984375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -14442.623046875,
          "predicted_interaction_delta_tfv_m3": -35269.58203125,
          "predicted_final_delta_tfv_m3": -9976.4765625,
          "predicted_trajectory_delta_tfv_m3": -58353.5078125
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -79917.25,
          "predicted_additive_single_delta_tfv_m3": -25492.005859375,
          "predicted_old_global_interaction_delta_tfv_m3": 20434.875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -19394.63671875,
          "predicted_interaction_delta_tfv_m3": 1040.23828125,
          "predicted_final_delta_tfv_m3": -24451.767578125,
          "predicted_trajectory_delta_tfv_m3": -54766.359375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -40815.25,
          "predicted_additive_single_delta_tfv_m3": 32949.97265625,
          "predicted_old_global_interaction_delta_tfv_m3": -700.931640625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -24287.015625,
          "predicted_interaction_delta_tfv_m3": -24987.947265625,
          "predicted_final_delta_tfv_m3": 7962.025390625,
          "predicted_trajectory_delta_tfv_m3": -49943.18359375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -125242.75,
          "predicted_additive_single_delta_tfv_m3": -256236.921875,
          "predicted_old_global_interaction_delta_tfv_m3": 193286.9375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -33675.2109375,
          "predicted_interaction_delta_tfv_m3": 159611.71875,
          "predicted_final_delta_tfv_m3": -96625.203125,
          "predicted_trajectory_delta_tfv_m3": -160778.390625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": -146357.25,
          "predicted_additive_single_delta_tfv_m3": -131232.890625,
          "predicted_old_global_interaction_delta_tfv_m3": 66793.8828125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -13754.8359375,
          "predicted_interaction_delta_tfv_m3": 53039.046875,
          "predicted_final_delta_tfv_m3": -78193.84375,
          "predicted_trajectory_delta_tfv_m3": -120812.5703125
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": -66564.75,
          "predicted_additive_single_delta_tfv_m3": -35931.96484375,
          "predicted_old_global_interaction_delta_tfv_m3": 57343.95703125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -24908.318359375,
          "predicted_interaction_delta_tfv_m3": 32435.638671875,
          "predicted_final_delta_tfv_m3": -3496.326171875,
          "predicted_trajectory_delta_tfv_m3": -68185.9375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -62088.5,
          "predicted_additive_single_delta_tfv_m3": 11264.556640625,
          "predicted_old_global_interaction_delta_tfv_m3": -16605.314453125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -10330.6640625,
          "predicted_interaction_delta_tfv_m3": -26935.978515625,
          "predicted_final_delta_tfv_m3": -15671.421875,
          "predicted_trajectory_delta_tfv_m3": -36614.31640625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 0,
          "true_delta_tfv_m3": -6005.0,
          "predicted_additive_single_delta_tfv_m3": -124322.2265625,
          "predicted_old_global_interaction_delta_tfv_m3": 33959.60546875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -21553.404296875,
          "predicted_interaction_delta_tfv_m3": 12406.201171875,
          "predicted_final_delta_tfv_m3": -111916.0234375,
          "predicted_trajectory_delta_tfv_m3": -81563.7109375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 1,
          "true_delta_tfv_m3": -63585.75,
          "predicted_additive_single_delta_tfv_m3": 30372.310546875,
          "predicted_old_global_interaction_delta_tfv_m3": -20392.16796875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -14176.7548828125,
          "predicted_interaction_delta_tfv_m3": -34568.921875,
          "predicted_final_delta_tfv_m3": -4196.611328125,
          "predicted_trajectory_delta_tfv_m3": -27877.5859375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 2,
          "true_delta_tfv_m3": -85455.0,
          "predicted_additive_single_delta_tfv_m3": -235868.96875,
          "predicted_old_global_interaction_delta_tfv_m3": 170310.359375,
          "predicted_nodewise_local_residual_delta_tfv_m3": -24478.86328125,
          "predicted_interaction_delta_tfv_m3": 145831.5,
          "predicted_final_delta_tfv_m3": -90037.46875,
          "predicted_trajectory_delta_tfv_m3": -140524.0625
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 3,
          "true_delta_tfv_m3": -18681.75,
          "predicted_additive_single_delta_tfv_m3": -135146.53125,
          "predicted_old_global_interaction_delta_tfv_m3": 82017.171875,
          "predicted_nodewise_local_residual_delta_tfv_m3": -33698.33203125,
          "predicted_interaction_delta_tfv_m3": 48318.83984375,
          "predicted_final_delta_tfv_m3": -86827.6875,
          "predicted_trajectory_delta_tfv_m3": -88759.2578125
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 4,
          "true_delta_tfv_m3": -72231.5,
          "predicted_additive_single_delta_tfv_m3": 58447.453125,
          "predicted_old_global_interaction_delta_tfv_m3": -76556.8125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -46243.796875,
          "predicted_interaction_delta_tfv_m3": -122800.609375,
          "predicted_final_delta_tfv_m3": -64353.15625,
          "predicted_trajectory_delta_tfv_m3": -48872.60546875
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 5,
          "true_delta_tfv_m3": 71736.75,
          "predicted_additive_single_delta_tfv_m3": -136605.34375,
          "predicted_old_global_interaction_delta_tfv_m3": 146562.0625,
          "predicted_nodewise_local_residual_delta_tfv_m3": -27304.7421875,
          "predicted_interaction_delta_tfv_m3": 119257.3203125,
          "predicted_final_delta_tfv_m3": -17348.0234375,
          "predicted_trajectory_delta_tfv_m3": -98226.59375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 6,
          "true_delta_tfv_m3": 44440.5,
          "predicted_additive_single_delta_tfv_m3": -125109.6953125,
          "predicted_old_global_interaction_delta_tfv_m3": 87408.953125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -22660.92578125,
          "predicted_interaction_delta_tfv_m3": 64748.02734375,
          "predicted_final_delta_tfv_m3": -60361.66796875,
          "predicted_trajectory_delta_tfv_m3": -74878.84375
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "source_kind": "D3",
          "candidate_index": 7,
          "true_delta_tfv_m3": -234637.25,
          "predicted_additive_single_delta_tfv_m3": 1942.7216796875,
          "predicted_old_global_interaction_delta_tfv_m3": -4430.83203125,
          "predicted_nodewise_local_residual_delta_tfv_m3": -16018.28125,
          "predicted_interaction_delta_tfv_m3": -20449.11328125,
          "predicted_final_delta_tfv_m3": -18506.390625,
          "predicted_trajectory_delta_tfv_m3": -19143.79296875
        }
      ],
      "micro": {
        "trained": true,
        "history": [
          {
            "epoch": 1,
            "loss": 5.5685990651448565,
            "gradient_norm": 0.32813734871645767,
            "d3_rank": 0.2817460317460318,
            "d3_pairwise": 0.6190476190476191,
            "d3_sign": 0.7291666666666666,
            "d3_top1": 2,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 2,
            "loss": 5.562181830406189,
            "gradient_norm": 0.3474699438860019,
            "d3_rank": 0.2936507936507937,
            "d3_pairwise": 0.625,
            "d3_sign": 0.7291666666666666,
            "d3_top1": 2,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 3,
            "loss": 5.551694830258687,
            "gradient_norm": 0.40777446205417317,
            "d3_rank": 0.28571428571428575,
            "d3_pairwise": 0.6190476190476191,
            "d3_sign": 0.75,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 4,
            "loss": 5.545103152592977,
            "gradient_norm": 0.4885798891385396,
            "d3_rank": 0.27777777777777785,
            "d3_pairwise": 0.6130952380952381,
            "d3_sign": 0.75,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 5,
            "loss": 5.52263085047404,
            "gradient_norm": 0.6306846588850021,
            "d3_rank": 0.29761904761904767,
            "d3_pairwise": 0.613095238095238,
            "d3_sign": 0.7708333333333334,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 6,
            "loss": 5.509798566500346,
            "gradient_norm": 0.6080164661010107,
            "d3_rank": 0.29761904761904767,
            "d3_pairwise": 0.613095238095238,
            "d3_sign": 0.7708333333333334,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 7,
            "loss": 5.511858264605205,
            "gradient_norm": 0.5733539660771688,
            "d3_rank": 0.36111111111111116,
            "d3_pairwise": 0.6309523809523809,
            "d3_sign": 0.7916666666666666,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 8,
            "loss": 5.502387960751851,
            "gradient_norm": 0.5305483539899191,
            "d3_rank": 0.3373015873015874,
            "d3_pairwise": 0.625,
            "d3_sign": 0.7916666666666666,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 9,
            "loss": 5.493231892585754,
            "gradient_norm": 0.5084889630476633,
            "d3_rank": 0.3373015873015874,
            "d3_pairwise": 0.625,
            "d3_sign": 0.7916666666666666,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 10,
            "loss": 5.492872953414917,
            "gradient_norm": 0.4885678291320801,
            "d3_rank": 0.3373015873015874,
            "d3_pairwise": 0.625,
            "d3_sign": 0.7916666666666666,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 11,
            "loss": 5.48512065410614,
            "gradient_norm": 0.44083184003829956,
            "d3_rank": 0.3373015873015874,
            "d3_pairwise": 0.625,
            "d3_sign": 0.7916666666666666,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          },
          {
            "epoch": 12,
            "loss": 5.482326825459798,
            "gradient_norm": 0.41167787959178287,
            "d3_rank": 0.36507936507936517,
            "d3_pairwise": 0.6369047619047619,
            "d3_sign": 0.7916666666666666,
            "d3_top1": 1,
            "d3_max_regret_m3": 228632.25
          }
        ],
        "profile": {
          "forward_seconds": 5.2640086996834725,
          "backward_seconds": 0.9550629000295885,
          "optimizer_seconds": 0.06610140018165112,
          "wall_time_seconds": 10.864205199992284
        }
      },
      "d2_after_d3": {
        "groups": 6,
        "spread_ratio": 1.2574826200307097,
        "rank": 0.7065813992894324,
        "pairwise": 0.7879621707419217,
        "sign": 0.7965250329380765,
        "top1": 3,
        "mean_regret_m3": 260.0833333333333,
        "max_regret_m3": 1560.5
      },
      "d2_prediction_invariance": {
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300": {
          "delta_states": 0.0,
          "delta_flows": 0.0,
          "direct_tfv": 0.0,
          "trajectory_tfv": 0.0
        },
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000": {
          "delta_states": 0.0,
          "delta_flows": 0.0,
          "direct_tfv": 0.0,
          "trajectory_tfv": 0.0
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600": {
          "delta_states": 0.0,
          "delta_flows": 0.0,
          "direct_tfv": 0.0,
          "trajectory_tfv": 0.0
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700": {
          "delta_states": 0.0,
          "delta_flows": 0.0,
          "direct_tfv": 0.0,
          "trajectory_tfv": 0.0
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500": {
          "delta_states": 0.0,
          "delta_flows": 0.0,
          "direct_tfv": 0.0,
          "trajectory_tfv": 0.0
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200": {
          "delta_states": 0.0,
          "delta_flows": 0.0,
          "direct_tfv": 0.0,
          "trajectory_tfv": 0.0
        },
        "prediction_invariant": true
      },
      "reference_sha_after_d3": "72f5bc81f4a08d5ffb59113aaefb71abfe7515ac33bc1860b94c3a081ded7798",
      "single_sha_after_d3": "ad7fbb9f67b0fc86cb9d3359ae9d54d36b250aafa7e7c9a4bed7d1e61eba9e30",
      "d2_preserved": true
    }
  },
  "mechanism_conclusion": {
    "old_nodewise_failure_caused_by_implementation": true,
    "nodewise_local_residual": "NEUTRAL",
    "large_effect_compression": "UNCHANGED",
    "remaining_primary_blocker": "state/topology-conditioned interaction calibration"
  },
  "verdict": "AMBER",
  "ready_for_full_smoke": false,
  "ready_for_formal": false,
  "ready_to_replace_active_step2": false,
  "need_new_swmm": false
}
```
