# PROJECT7 STEP2 TRAINING SEMANTICS V4.3.1

```json
{
  "contract": "PROJECT7_STEP2_TRAINING_SEMANTICS_V431",
  "git_parent": "d84a2a45279529fc2ba482d2c7e26acf58d3564b",
  "branch": "agent/step2-training-semantics-v431",
  "draft_pr_base": "agent/step2-topology-interaction-v43",
  "boundary": {
    "swmm_launched": false,
    "d2_regenerated": false,
    "d3_regenerated": false,
    "validation_outcomes_accessed": false,
    "final_accessed": false,
    "formal_run": false,
    "full_train_smoke_run": false
  },
  "initialization": {
    "immutable_parent": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_balance_v421\\03_tiny_combined\\v42_tiny_combined.pt",
    "tiny_parent": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_balance_v421\\03_tiny_combined\\v42_tiny_combined.pt",
    "micro_parent": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_balance_v421\\03_tiny_combined\\v42_tiny_combined.pt",
    "same_parent": true,
    "micro_loaded_tiny_checkpoint": false,
    "tiny_groups_sha256": "afdf78f48fa12ccb7ded10bba067836c9f00d448a25baedf973f74830e8c06a0",
    "micro_groups_sha256": "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3",
    "micro_groups_sha_matches_prior": true
  },
  "phase_loss": {
    "reference_only": true,
    "d2_only": true,
    "d3_interaction_only": true,
    "reference_counterfactual_gradient": "ZERO_BY_CONSTRUCTION",
    "d2_reference_interaction_gradient": "ZERO_BY_PHASE_FREEZE",
    "loss_gradient_audit": {
      "parameter_count": 48,
      "parameter_names": [
        "interaction_encoder.0.weight",
        "interaction_encoder.0.bias",
        "interaction_encoder.2.weight",
        "interaction_encoder.2.bias",
        "interaction_encoder.4.weight",
        "interaction_encoder.4.bias",
        "interaction_magnitude_encoder.0.weight",
        "interaction_magnitude_encoder.0.bias",
        "interaction_magnitude_encoder.2.weight",
        "interaction_magnitude_encoder.2.bias",
        "interaction_magnitude_encoder.4.weight",
        "interaction_magnitude_encoder.4.bias",
        "interaction_magnitude_residual.weight",
        "interaction_magnitude_residual.bias",
        "interaction_flow_head.weight",
        "interaction_flow_head.bias",
        "interaction_state_head.weight",
        "interaction_state_head.bias",
        "direct_interaction_tfv_head.0.weight",
        "direct_interaction_tfv_head.0.bias",
        "direct_interaction_tfv_head.2.weight",
        "direct_interaction_tfv_head.2.bias",
        "direct_interaction_tfv_head.4.weight",
        "direct_interaction_tfv_head.4.bias",
        "topology_seed_encoder.0.weight",
        "topology_seed_encoder.0.bias",
        "topology_seed_encoder.2.weight",
        "topology_seed_encoder.2.bias",
        "topology_seed_encoder.4.weight",
        "topology_seed_encoder.4.bias",
        "topology_context_encoder.weight",
        "topology_context_encoder.bias",
        "topology_message_blocks.0.0.weight",
        "topology_message_blocks.0.0.bias",
        "topology_message_blocks.0.2.weight",
        "topology_message_blocks.0.2.bias",
        "topology_message_blocks.1.0.weight",
        "topology_message_blocks.1.0.bias",
        "topology_message_blocks.1.2.weight",
        "topology_message_blocks.1.2.bias",
        "topology_message_blocks.2.0.weight",
        "topology_message_blocks.2.0.bias",
        "topology_message_blocks.2.2.weight",
        "topology_message_blocks.2.2.bias",
        "topology_state_head.weight",
        "topology_state_head.bias",
        "topology_hidden_head.weight",
        "topology_hidden_head.bias"
      ],
      "components": {
        "delta_state": {
          "gradient_l2": 0.008638544008135796,
          "gradient_linf": 0.001656392472796142,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.3909195065498352,
          "cosine_vs_ranking": -0.15055087208747864
        },
        "delta_flow": {
          "gradient_l2": 0.061511214822530746,
          "gradient_linf": 0.025740070268511772,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.004033168312162161,
          "cosine_vs_ranking": 0.010804433375597
        },
        "direct_TFV": {
          "gradient_l2": 0.33975744247436523,
          "gradient_linf": 0.03791401535272598,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.9999999403953552,
          "cosine_vs_ranking": 0.567444920539856
        },
        "centered_TFV": {
          "gradient_l2": 0.018973642960190773,
          "gradient_linf": 0.002521540503948927,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.2976275086402893,
          "cosine_vs_ranking": -0.6290403008460999
        },
        "trajectory_TFV": {
          "gradient_l2": 0.32004281878471375,
          "gradient_linf": 0.15483133494853973,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.49102523922920227,
          "cosine_vs_ranking": -0.1739840805530548
        },
        "ranking": {
          "gradient_l2": 0.09320250153541565,
          "gradient_linf": 0.011619525030255318,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.567444920539856,
          "cosine_vs_ranking": 1.0
        },
        "magnitude_calibration": {
          "gradient_l2": 0.04831024259328842,
          "gradient_linf": 0.00508603360503912,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.47946885228157043,
          "cosine_vs_ranking": 0.253356397151947
        },
        "interaction_energy": {
          "gradient_l2": 1.8042967319488525,
          "gradient_linf": 0.8911746740341187,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.007696580607444048,
          "cosine_vs_ranking": -0.009841544553637505
        }
      },
      "interaction_energy_weight": 0.01
    }
  },
  "state_diagnostic": {
    "current_state_source": "initial_state",
    "future_state_used_as_current": false,
    "rows": [
      {
        "source": "initial_state",
        "mean_depth": 2.4495548032899856,
        "high_depth_fraction": 0.8347639484978541,
        "flooding_active_fraction": 0.23927038626609443,
        "storage_utilization": 0.31775527952224025,
        "storage_source": "initial_state.node_volume/storage_capacity_m3",
        "mean_total_inflow": 7.014311629323599,
        "mean_total_outflow": 5.390270024620527,
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300"
      },
      {
        "source": "initial_state",
        "mean_depth": 0.14505399730022753,
        "high_depth_fraction": 0.01072961373390558,
        "flooding_active_fraction": 0.0,
        "storage_utilization": 0.0011705770114795564,
        "storage_source": "initial_state.node_volume/storage_capacity_m3",
        "mean_total_inflow": 0.0787492480969211,
        "mean_total_outflow": 0.06655981057984398,
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000"
      },
      {
        "source": "initial_state",
        "mean_depth": 2.9507243999102797,
        "high_depth_fraction": 0.8894849785407726,
        "flooding_active_fraction": 0.2317596566523605,
        "storage_utilization": 0.5348556767394269,
        "storage_source": "initial_state.node_volume/storage_capacity_m3",
        "mean_total_inflow": 7.614793686090273,
        "mean_total_outflow": 6.3484379805977795,
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600"
      },
      {
        "source": "initial_state",
        "mean_depth": 0.17292077147844703,
        "high_depth_fraction": 0.02145922746781116,
        "flooding_active_fraction": 0.0,
        "storage_utilization": 0.0016094653253429033,
        "storage_source": "initial_state.node_volume/storage_capacity_m3",
        "mean_total_inflow": 0.10245106474628247,
        "mean_total_outflow": 0.09102458301034619,
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700"
      },
      {
        "source": "initial_state",
        "mean_depth": 0.12007573420002947,
        "high_depth_fraction": 0.009656652360515022,
        "flooding_active_fraction": 0.0,
        "storage_utilization": 0.0009264860947356599,
        "storage_source": "initial_state.node_volume/storage_capacity_m3",
        "mean_total_inflow": 0.06225454612058258,
        "mean_total_outflow": 0.05006511703501156,
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500"
      },
      {
        "source": "initial_state",
        "mean_depth": 0.15980730879220717,
        "high_depth_fraction": 0.015021459227467811,
        "flooding_active_fraction": 0.0,
        "storage_utilization": 0.0013703983543033322,
        "storage_source": "initial_state.node_volume/storage_capacity_m3",
        "mean_total_inflow": 0.0911492757502719,
        "mean_total_outflow": 0.07895984462827524,
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200"
      }
    ]
  },
  "phased_no_topology_control": {
    "stage": "micro_control",
    "architecture": "V4.2.1 no-topology control",
    "selected_group_count": 12,
    "reference_group_count_raw": 12,
    "reference_group_count_unique": 6,
    "reference_deduplication_ratio": 0.5,
    "d2_metrics_before_d3": {
      "D2": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "D3": {
        "groups": 0
      }
    },
    "final_metrics": {
      "D2": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "D3": {
        "groups": 6,
        "spread_ratio": 0.4869722646121409,
        "rank": 0.2619047619047619,
        "pairwise": 0.5833333333333334,
        "sign": 0.8125,
        "top1": 1,
        "mean_regret_m3": 87313.54166666667,
        "max_regret_m3": 323535.0
      }
    },
    "d3_magnitude_strata": {
      "small": {
        "count": 15,
        "mae_m3": 39415.338541666664,
        "bias_m3": -38437.903125,
        "response_ratio": 2.247542064922422,
        "rank": -0.12500000000000003,
        "pairwise": 0.48333333333333334,
        "sign": 0.8
      },
      "medium": {
        "count": 12,
        "mae_m3": 43747.5244140625,
        "bias_m3": -8932.5107421875,
        "response_ratio": 0.9585154281071878,
        "rank": -0.06666666666666661,
        "pairwise": 0.38888888888888884,
        "sign": 0.9166666666666666
      },
      "large": {
        "count": 21,
        "mae_m3": 165187.69800967263,
        "bias_m3": -46279.633463541664,
        "response_ratio": 0.3929275704736995,
        "rank": 0.21999999999999992,
        "pairwise": 0.6266666666666667,
        "sign": 0.7619047619047619
      }
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
    "reference_sha_unchanged": true,
    "single_sha_unchanged": true,
    "interaction_sha_changed": true,
    "reference_forwards_per_group": 1,
    "profile_seconds": {
      "reference": {
        "forward_seconds": 0.6327459999592975,
        "backward_seconds": 0.09633849997771904,
        "optimizer_seconds": 0.01731679996009916,
        "wall_time_seconds": 1.37142420001328
      },
      "d2": {
        "forward_seconds": 1.6769332999247126,
        "backward_seconds": 4.122106399794575,
        "optimizer_seconds": 0.03805120021570474,
        "wall_time_seconds": 7.47412109997822
      },
      "d3": {
        "forward_seconds": 4.34499290009262,
        "backward_seconds": 2.0699566998519003,
        "optimizer_seconds": 0.09178520017303526,
        "wall_time_seconds": 9.98654480004916
      }
    }
  },
  "phased_topology": {
    "stage": "micro_topology",
    "architecture": "V4.3 topology",
    "selected_group_count": 12,
    "reference_group_count_raw": 12,
    "reference_group_count_unique": 6,
    "reference_deduplication_ratio": 0.5,
    "d2_metrics_before_d3": {
      "D2": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "D3": {
        "groups": 0
      }
    },
    "final_metrics": {
      "D2": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "D3": {
        "groups": 6,
        "spread_ratio": 0.4557475169865084,
        "rank": 0.2142857142857143,
        "pairwise": 0.5714285714285714,
        "sign": 0.8125,
        "top1": 1,
        "mean_regret_m3": 87313.54166666667,
        "max_regret_m3": 323535.0
      }
    },
    "d3_magnitude_strata": {
      "small": {
        "count": 15,
        "mae_m3": 53237.219010416666,
        "bias_m3": -53237.219010416666,
        "response_ratio": 2.821546820800748,
        "rank": -0.20000000000000004,
        "pairwise": 0.43333333333333335,
        "sign": 0.8
      },
      "medium": {
        "count": 12,
        "mae_m3": 41150.601888020836,
        "bias_m3": -23075.0986328125,
        "response_ratio": 1.1525544668447243,
        "rank": -0.06666666666666661,
        "pairwise": 0.38888888888888884,
        "sign": 0.9166666666666666
      },
      "large": {
        "count": 21,
        "mae_m3": 160265.37574404763,
        "bias_m3": -58627.900297619046,
        "response_ratio": 0.45375318229919775,
        "rank": 0.23999999999999994,
        "pairwise": 0.6466666666666667,
        "sign": 0.7619047619047619
      }
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
    "reference_sha_unchanged": true,
    "single_sha_unchanged": true,
    "interaction_sha_changed": true,
    "reference_forwards_per_group": 1,
    "profile_seconds": {
      "reference": {
        "forward_seconds": 1.700271499925293,
        "backward_seconds": 0.12057860009372234,
        "optimizer_seconds": 0.021816899941768497,
        "wall_time_seconds": 3.390756299952045
      },
      "d2": {
        "forward_seconds": 3.7435373999760486,
        "backward_seconds": 5.403682599833701,
        "optimizer_seconds": 0.03920589998597279,
        "wall_time_seconds": 12.84122139995452
      },
      "d3": {
        "forward_seconds": 5.078576699597761,
        "backward_seconds": 4.31729760003509,
        "optimizer_seconds": 0.1367735001258552,
        "wall_time_seconds": 14.468608499970287
      }
    }
  },
  "tiny_control": {
    "stage": "tiny_control",
    "architecture": "V4.2.1 no-topology control",
    "selected_group_count": 2,
    "reference_group_count_raw": 2,
    "reference_group_count_unique": 1,
    "reference_deduplication_ratio": 0.5,
    "d2_metrics_before_d3": {
      "D2": {
        "groups": 1,
        "spread_ratio": 1.5116044175078953,
        "rank": 0.990646284477494,
        "pairwise": 0.978021978021978,
        "sign": 0.9130434782608695,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      },
      "D3": {
        "groups": 0
      }
    },
    "final_metrics": {
      "D2": {
        "groups": 1,
        "spread_ratio": 1.5116044175078953,
        "rank": 0.990646284477494,
        "pairwise": 0.978021978021978,
        "sign": 0.9130434782608695,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      },
      "D3": {
        "groups": 1,
        "spread_ratio": 1.5372751825266524,
        "rank": 0.8571428571428572,
        "pairwise": 0.8571428571428571,
        "sign": 0.75,
        "top1": 0,
        "mean_regret_m3": 8404.5,
        "max_regret_m3": 8404.5
      }
    },
    "d3_magnitude_strata": {
      "small": {
        "count": 5,
        "mae_m3": 24029.17578125,
        "bias_m3": 22339.39453125,
        "response_ratio": 0.8903849779128046,
        "rank": 0.49999999999999994,
        "pairwise": 0.7,
        "sign": 0.6
      },
      "medium": {
        "count": 1,
        "mae_m3": 3156.34375,
        "bias_m3": -3156.34375,
        "response_ratio": 1.0380454271508213,
        "rank": NaN,
        "pairwise": NaN,
        "sign": 1.0
      },
      "large": {
        "count": 2,
        "mae_m3": 22179.765625,
        "bias_m3": -22179.765625,
        "response_ratio": 1.1911360835647515,
        "rank": -0.9999999999999999,
        "pairwise": 0.0,
        "sign": 1.0
      }
    },
    "d2_prediction_invariance": {
      "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300": {
        "delta_states": 0.0,
        "delta_flows": 0.0,
        "direct_tfv": 0.0,
        "trajectory_tfv": 0.0
      },
      "prediction_invariant": true
    },
    "reference_sha_unchanged": true,
    "single_sha_unchanged": true,
    "interaction_sha_changed": true,
    "reference_forwards_per_group": 1,
    "profile_seconds": {
      "reference": {
        "forward_seconds": 0.8038267000229098,
        "backward_seconds": 0.28489609999815,
        "optimizer_seconds": 0.07037330005550757,
        "wall_time_seconds": 1.3541146999923512
      },
      "d2": {
        "forward_seconds": 0.5307689000037499,
        "backward_seconds": 1.1194692999706604,
        "optimizer_seconds": 0.010843800089787692,
        "wall_time_seconds": 2.093308299954515
      },
      "d3": {
        "forward_seconds": 1.1372641999623738,
        "backward_seconds": 0.445346399967093,
        "optimizer_seconds": 0.019792099890764803,
        "wall_time_seconds": 2.417926400026772
      }
    }
  },
  "tiny_topology": {
    "stage": "tiny_topology",
    "architecture": "V4.3 topology",
    "selected_group_count": 2,
    "reference_group_count_raw": 2,
    "reference_group_count_unique": 1,
    "reference_deduplication_ratio": 0.5,
    "d2_metrics_before_d3": {
      "D2": {
        "groups": 1,
        "spread_ratio": 1.5116044175078953,
        "rank": 0.990646284477494,
        "pairwise": 0.978021978021978,
        "sign": 0.9130434782608695,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      },
      "D3": {
        "groups": 0
      }
    },
    "final_metrics": {
      "D2": {
        "groups": 1,
        "spread_ratio": 1.5116044175078953,
        "rank": 0.990646284477494,
        "pairwise": 0.978021978021978,
        "sign": 0.9130434782608695,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      },
      "D3": {
        "groups": 1,
        "spread_ratio": 1.5060507028495793,
        "rank": 0.880952380952381,
        "pairwise": 0.8928571428571429,
        "sign": 0.875,
        "top1": 1,
        "mean_regret_m3": 0.0,
        "max_regret_m3": 0.0
      }
    },
    "d3_magnitude_strata": {
      "small": {
        "count": 5,
        "mae_m3": 15784.264453125,
        "bias_m3": 2791.328515625,
        "response_ratio": 1.1881991722161387,
        "rank": 0.49999999999999994,
        "pairwise": 0.7,
        "sign": 0.8
      },
      "medium": {
        "count": 1,
        "mae_m3": 17511.890625,
        "bias_m3": -17511.890625,
        "response_ratio": 1.2110820024107278,
        "rank": NaN,
        "pairwise": NaN,
        "sign": 1.0
      },
      "large": {
        "count": 2,
        "mae_m3": 38484.9375,
        "bias_m3": -38484.9375,
        "response_ratio": 1.3316473381347662,
        "rank": 0.9999999999999999,
        "pairwise": 1.0,
        "sign": 1.0
      }
    },
    "d2_prediction_invariance": {
      "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300": {
        "delta_states": 0.0,
        "delta_flows": 0.0,
        "direct_tfv": 0.0,
        "trajectory_tfv": 0.0
      },
      "prediction_invariant": true
    },
    "reference_sha_unchanged": true,
    "single_sha_unchanged": true,
    "interaction_sha_changed": true,
    "reference_forwards_per_group": 1,
    "profile_seconds": {
      "reference": {
        "forward_seconds": 0.4777431999682449,
        "backward_seconds": 0.024556999967899173,
        "optimizer_seconds": 0.009269799978937954,
        "wall_time_seconds": 0.9114902000292204
      },
      "d2": {
        "forward_seconds": 1.0562578999670222,
        "backward_seconds": 1.4395654000109062,
        "optimizer_seconds": 0.012545100005809218,
        "wall_time_seconds": 3.542833899962716
      },
      "d3": {
        "forward_seconds": 0.9903337999712676,
        "backward_seconds": 0.6742547000176273,
        "optimizer_seconds": 0.030959299998357892,
        "wall_time_seconds": 2.635132599971257
      }
    }
  },
  "topology": {
    "graph_contract": {
      "node_count": 932,
      "edge_count": 2420,
      "directed_contract": "bidirectional",
      "reverse_edge_count": 2420,
      "self_loop_count": 0,
      "duplicate_edge_count": 0,
      "isolated_node_count": 0,
      "edge_index_used_by_forward": true
    },
    "old_v4_2_1_topology_active": false,
    "old_edge_ablation_max_abs_m3": 0.0,
    "topology_ablation": {
      "normal": {
        "groups": 6,
        "spread_ratio": 0.4557475225103878,
        "rank": 0.2142857142857143,
        "pairwise": 0.5714285714285714,
        "sign": 0.8125,
        "top1": 1,
        "mean_regret_m3": 87313.54166666667,
        "max_regret_m3": 323535.0
      },
      "graph_disabled": {
        "groups": 6,
        "spread_ratio": 0.4741594793693153,
        "rank": 0.253968253968254,
        "pairwise": 0.5833333333333333,
        "sign": 0.8125,
        "top1": 1,
        "mean_regret_m3": 87313.54166666667,
        "max_regret_m3": 323535.0
      },
      "prediction_change_max_abs_m3": 19023.8984375,
      "delta_rank": -0.03968253968253971,
      "delta_pairwise": -0.011904761904761862,
      "delta_mean_regret_m3": 0.0
    },
    "state_ablation": {
      "A_endpoint_on_message_on": {
        "metrics": {
          "groups": 6,
          "spread_ratio": 0.4557475225103878,
          "rank": 0.2142857142857143,
          "pairwise": 0.5714285714285714,
          "sign": 0.8125,
          "top1": 1,
          "mean_regret_m3": 87313.54166666667,
          "max_regret_m3": 323535.0
        },
        "prediction_max_change_vs_A_m3": 0.0
      },
      "B_endpoint_off_message_on": {
        "metrics": {
          "groups": 6,
          "spread_ratio": 0.45574703039019865,
          "rank": 0.2142857142857143,
          "pairwise": 0.5714285714285714,
          "sign": 0.8125,
          "top1": 1,
          "mean_regret_m3": 87313.54166666667,
          "max_regret_m3": 323535.0
        },
        "prediction_max_change_vs_A_m3": 22.75
      },
      "C_endpoint_on_message_off": {
        "metrics": {
          "groups": 6,
          "spread_ratio": 0.45575029317612564,
          "rank": 0.2142857142857143,
          "pairwise": 0.5714285714285714,
          "sign": 0.8125,
          "top1": 1,
          "mean_regret_m3": 87313.54166666667,
          "max_regret_m3": 323535.0
        },
        "prediction_max_change_vs_A_m3": 4.75
      },
      "D_endpoint_off_message_off": {
        "metrics": {
          "groups": 6,
          "spread_ratio": 0.45574983786051376,
          "rank": 0.2142857142857143,
          "pairwise": 0.5714285714285714,
          "sign": 0.8125,
          "top1": 1,
          "mean_regret_m3": 87313.54166666667,
          "max_regret_m3": 323535.0
        },
        "prediction_max_change_vs_A_m3": 21.0859375
      }
    },
    "pooling_audit": {
      "pooling": "node_mean_in_direct_tfv_head",
      "groups": [
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
          "mean_abs_node_response": 0.0031429086811840534,
          "p90_abs_node_response": 0.002395408693701029,
          "max_abs_node_response": 0.5138530731201172,
          "max_over_mean": 163.49602398454962
        },
        {
          "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
          "mean_abs_node_response": 0.00267362454906106,
          "p90_abs_node_response": 0.003002087352797389,
          "max_abs_node_response": 0.3854901194572449,
          "max_over_mean": 144.1825927251542
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
          "mean_abs_node_response": 0.005402887240052223,
          "p90_abs_node_response": 0.005935938563197851,
          "max_abs_node_response": 0.5618209838867188,
          "max_over_mean": 103.98532468378671
        },
        {
          "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
          "mean_abs_node_response": 0.0025198350194841623,
          "p90_abs_node_response": 0.002550458302721381,
          "max_abs_node_response": 0.24995601177215576,
          "max_over_mean": 99.19538772952068
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
          "mean_abs_node_response": 0.0025338137056678534,
          "p90_abs_node_response": 0.003965472802519798,
          "max_abs_node_response": 0.31447574496269226,
          "max_over_mean": 124.11162835659376
        },
        {
          "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
          "mean_abs_node_response": 0.0021554380655288696,
          "p90_abs_node_response": 0.0007024897495284677,
          "max_abs_node_response": 0.3796466290950775,
          "max_over_mean": 176.13432515952408
        }
      ],
      "large_effect_high_node_concentration": true
    }
  },
  "physical_causal": {
    "zero_action_exact_zero": true,
    "single_action_interaction_exact_zero": true,
    "multi_action_interaction_nonzero": true,
    "future_action_cannot_affect_past": true,
    "gradient_finite": true,
    "gradient_nonzero": true,
    "nonnegative_candidate_flooding": true,
    "head_depth_consistency": true,
    "horizon_steps": 72
  },
  "edge_hydraulic_audit": {
    "edge_features_available": false,
    "edge_features_aligned_to_edge_index": false,
    "candidate_graph_attributes": {},
    "node_conduit_aggregate_features": [
      "conduit_in_count",
      "conduit_out_count",
      "conduit_in_length_sum_m",
      "conduit_out_length_sum_m",
      "conduit_in_roughness_mean",
      "conduit_out_roughness_mean",
      "conduit_in_geom1_mean_m",
      "conduit_out_geom1_mean_m"
    ],
    "status": "EDGE_HYDRAULIC_FEATURES_NOT_IN_CURRENT_GRAPH_CONTRACT"
  },
  "compare": {
    "OLD_V4.2": {
      "D2": {
        "groups": 6,
        "spread_ratio": 1.2574826200307097,
        "rank": 0.7065813992894324,
        "pairwise": 0.7879621707419217,
        "sign": 0.7965250329380765,
        "top1": 3,
        "mean_regret_m3": 260.0833333333333,
        "max_regret_m3": 1560.5
      },
      "D3": {
        "groups": 6,
        "spread_ratio": 0.6836385492746212,
        "rank": 0.30952380952380953,
        "pairwise": 0.630952380952381,
        "sign": 0.75,
        "top1": 2,
        "mean_regret_m3": 83857.04166666667,
        "max_regret_m3": 228632.25
      }
    },
    "OLD_V4.2.1_CORRECTED": {
      "D2": {
        "groups": 6,
        "spread_ratio": 1.4487514468200609,
        "rank": 0.6131634488178036,
        "pairwise": 0.734971835959953,
        "sign": 0.6949385155906896,
        "top1": 2,
        "mean_regret_m3": 2629.5,
        "max_regret_m3": 14638.5
      },
      "D3": {
        "groups": 6,
        "spread_ratio": 0.8293908028953982,
        "rank": 0.376984126984127,
        "pairwise": 0.6428571428571429,
        "sign": 0.8125,
        "top1": 0,
        "mean_regret_m3": 80021.04166666667,
        "max_regret_m3": 162405.75,
        "magnitude_strata": {
          "small": {
            "count": 15,
            "mae_m3": 69223.19456380208,
            "bias_m3": -57544.053938802084,
            "response_ratio": 3.1600904212912115,
            "rank": 0.5249999999999999,
            "pairwise": 0.7583333333333333,
            "sign": 0.7333333333333333
          },
          "medium": {
            "count": 12,
            "mae_m3": 49901.962565104164,
            "bias_m3": -12196.9423828125,
            "response_ratio": 1.0453039038880456,
            "rank": -0.3999999999999999,
            "pairwise": 0.27777777777777773,
            "sign": 0.8333333333333334
          },
          "large": {
            "count": 21,
            "mae_m3": 151719.24504743304,
            "bias_m3": -58788.735746837796,
            "response_ratio": 0.4624876683752864,
            "rank": 0.36,
            "pairwise": 0.6666666666666667,
            "sign": 0.8571428571428571
          }
        }
      }
    },
    "OLD_V4.3": {
      "D2": {
        "groups": 6,
        "spread_ratio": 0.9212074639467834,
        "rank": 0.4046581751199058,
        "pairwise": 0.6579463166809805,
        "sign": 0.6305445761967502,
        "top1": 1,
        "mean_regret_m3": 6646.0,
        "max_regret_m3": 20020.5
      },
      "D3": {
        "groups": 6,
        "spread_ratio": 0.5045575758290036,
        "rank": 0.21031746031746035,
        "pairwise": 0.5892857142857143,
        "sign": 0.8125,
        "top1": 2,
        "mean_regret_m3": 50019.291666666664,
        "max_regret_m3": 162405.75,
        "magnitude_strata": {
          "small": {
            "count": 15,
            "mae_m3": 42590.807421875,
            "bias_m3": -41012.343359375,
            "response_ratio": 2.347394041168968,
            "rank": 0.475,
            "pairwise": 0.7083333333333333,
            "sign": 0.8
          },
          "medium": {
            "count": 12,
            "mae_m3": 39552.628580729164,
            "bias_m3": -2927.4482421875,
            "response_ratio": 0.8761248102039937,
            "rank": -0.3333333333333333,
            "pairwise": 0.3333333333333333,
            "sign": 0.9166666666666666
          },
          "large": {
            "count": 21,
            "mae_m3": 169859.34058779763,
            "bias_m3": -41729.51469494047,
            "response_ratio": 0.3705144041809961,
            "rank": 0.21999999999999997,
            "pairwise": 0.6066666666666667,
            "sign": 0.7619047619047619
          }
        }
      }
    },
    "V4.3.1_PHASED_NO_TOPOLOGY_CONTROL": {
      "D2": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "D3": {
        "groups": 6,
        "spread_ratio": 0.4869722646121409,
        "rank": 0.2619047619047619,
        "pairwise": 0.5833333333333334,
        "sign": 0.8125,
        "top1": 1,
        "mean_regret_m3": 87313.54166666667,
        "max_regret_m3": 323535.0,
        "magnitude_strata": {
          "small": {
            "count": 15,
            "mae_m3": 39415.338541666664,
            "bias_m3": -38437.903125,
            "response_ratio": 2.247542064922422,
            "rank": -0.12500000000000003,
            "pairwise": 0.48333333333333334,
            "sign": 0.8
          },
          "medium": {
            "count": 12,
            "mae_m3": 43747.5244140625,
            "bias_m3": -8932.5107421875,
            "response_ratio": 0.9585154281071878,
            "rank": -0.06666666666666661,
            "pairwise": 0.38888888888888884,
            "sign": 0.9166666666666666
          },
          "large": {
            "count": 21,
            "mae_m3": 165187.69800967263,
            "bias_m3": -46279.633463541664,
            "response_ratio": 0.3929275704736995,
            "rank": 0.21999999999999992,
            "pairwise": 0.6266666666666667,
            "sign": 0.7619047619047619
          }
        }
      }
    },
    "V4.3.1_PHASED_TOPOLOGY": {
      "D2": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "D3": {
        "groups": 6,
        "spread_ratio": 0.4557475169865084,
        "rank": 0.2142857142857143,
        "pairwise": 0.5714285714285714,
        "sign": 0.8125,
        "top1": 1,
        "mean_regret_m3": 87313.54166666667,
        "max_regret_m3": 323535.0,
        "magnitude_strata": {
          "small": {
            "count": 15,
            "mae_m3": 53237.219010416666,
            "bias_m3": -53237.219010416666,
            "response_ratio": 2.821546820800748,
            "rank": -0.20000000000000004,
            "pairwise": 0.43333333333333335,
            "sign": 0.8
          },
          "medium": {
            "count": 12,
            "mae_m3": 41150.601888020836,
            "bias_m3": -23075.0986328125,
            "response_ratio": 1.1525544668447243,
            "rank": -0.06666666666666661,
            "pairwise": 0.38888888888888884,
            "sign": 0.9166666666666666
          },
          "large": {
            "count": 21,
            "mae_m3": 160265.37574404763,
            "bias_m3": -58627.900297619046,
            "response_ratio": 0.45375318229919775,
            "rank": 0.23999999999999994,
            "pairwise": 0.6466666666666667,
            "sign": 0.7619047619047619
          }
        }
      }
    }
  },
  "topology_net_contribution": {
    "delta_rank": -0.047619047619047616,
    "delta_pairwise": -0.011904761904761973,
    "delta_mean_regret_m3": 0.0,
    "supported": false
  },
  "verdict": "RED",
  "ready_for_full_train_smoke": false,
  "ready_for_formal": false,
  "need_new_swmm": false,
  "next_bounded_action": "Keep production wiring frozen; if phased control recovers but topology is neutral/harmful, design V4.4 edge-hydraulic-conditioned interaction."
}
```
