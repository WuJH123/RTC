# PROJECT7 STEP2 STATE-TOPOLOGY INTERACTION V4.3

```json
{
  "contract": "PROJECT7_STEP2_STATE_TOPOLOGY_INTERACTION_V43",
  "git_parent": "84e31702c2c05ab79822611fd9aa4411c79f21dd",
  "branch": "agent/step2-topology-interaction-v43",
  "draft_pr_base": "agent/step2-control-response-v4",
  "boundary": {
    "scientific_split": [
      "development"
    ],
    "development_fold": [
      "train"
    ],
    "swmm_launched": false,
    "d2_regenerated": false,
    "d3_regenerated": false,
    "validation_outcomes_accessed": false,
    "final_accessed": false,
    "formal_step2_run": false,
    "closed_loop_run": false,
    "policy_lock_run": false,
    "full_train_smoke_run": false,
    "hyperparameter_grid_search": false
  },
  "cross_source_interference": {
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
  },
  "topology": {
    "contract": "STEP2_TOPOLOGY_INTERACTION_AUDIT_V43",
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
    "old_v4_2_1_edge_ablation_max_abs_m3": 0.0,
    "v4_3_topology_active": true,
    "topology_ablation": {
      "normal": {
        "groups": 6,
        "spread_ratio": 0.5045575758290036,
        "rank": 0.21031746031746035,
        "pairwise": 0.5892857142857143,
        "sign": 0.8125,
        "top1": 2,
        "mean_regret_m3": 50019.291666666664,
        "max_regret_m3": 162405.75
      },
      "ablated": {
        "groups": 6,
        "spread_ratio": 0.4946208051207716,
        "rank": 0.1785714285714286,
        "pairwise": 0.5714285714285715,
        "sign": 0.7291666666666666,
        "top1": 2,
        "mean_regret_m3": 50019.291666666664,
        "max_regret_m3": 162405.75
      },
      "delta_rank": 0.031746031746031744,
      "delta_pairwise": 0.017857142857142794,
      "delta_spread": 0.009936770708232012,
      "delta_mean_regret_m3": 0.0,
      "prediction_change_max_abs_m3": 42251.75
    },
    "state_context_ablation": {
      "normal": {
        "groups": 6,
        "spread_ratio": 0.5045575625451681,
        "rank": 0.21031746031746035,
        "pairwise": 0.5892857142857143,
        "sign": 0.8125,
        "top1": 2,
        "mean_regret_m3": 50019.291666666664,
        "max_regret_m3": 162405.75
      },
      "global_context_ablated": {
        "groups": 6,
        "spread_ratio": 0.5045577425646477,
        "rank": 0.21031746031746035,
        "pairwise": 0.5892857142857143,
        "sign": 0.8125,
        "top1": 2,
        "mean_regret_m3": 50019.291666666664,
        "max_regret_m3": 162405.75
      },
      "delta_rank": 0.0,
      "delta_pairwise": 0.0,
      "delta_mean_regret_m3": 0.0,
      "prediction_change_max_abs_m3": 2.03125
    },
    "topology_gradient": {
      "changed_actuator_gradient_nonzero_fraction": 1.0,
      "gradient_finite_fraction": 0.9999999403953552,
      "active_actuator_count": 109,
      "endpoint_node_count": 117,
      "response_by_topological_distance": {
        "0": {
          "node_count": 117,
          "mean_abs_interaction_response": 6.622970104217529,
          "max_abs_interaction_response": 96.34546661376953
        },
        "1": {
          "node_count": 128,
          "mean_abs_interaction_response": 2.5537095069885254,
          "max_abs_interaction_response": 8.370548248291016
        },
        "2": {
          "node_count": 89,
          "mean_abs_interaction_response": 2.3460140228271484,
          "max_abs_interaction_response": 10.856837272644043
        },
        "3": {
          "node_count": 93,
          "mean_abs_interaction_response": 2.23085618019104,
          "max_abs_interaction_response": 11.20804500579834
        }
      },
      "network_response_magnitude": 2.606915235519409,
      "all_nodes_nearly_identical": false
    },
    "local_hydraulic_state_conditioning": true,
    "zero_single_multi_invariants": {
      "zero_action_exact_zero": true,
      "single_action_interaction_exact_zero": true,
      "multi_action_interaction_nonzero": true,
      "future_action_cannot_affect_past": true,
      "gradient_finite": true,
      "gradient_nonzero": true,
      "nonnegative_candidate_flooding": true,
      "head_depth_consistency": true,
      "horizon_steps": 72
    }
  },
  "V4_2": {
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
  "V4_2_1": {
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
  "V4_3": {
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
  "state_topology_strata": [
    {
      "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
      "initial_hydraulic_wetness": 0.5333892321051065,
      "initial_flooding_activity": 0.06948915116833572,
      "initial_storage_utilization": 0.4458199427261612,
      "multi_actuator_spatial_dispersion": 23.347520976353927,
      "active_actuator_count": 108,
      "actuator_topological_distance": 36.0
    },
    {
      "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
      "initial_hydraulic_wetness": 0.4806903422968885,
      "initial_flooding_activity": 0.07542024320457796,
      "initial_storage_utilization": 0.35783166554009943,
      "multi_actuator_spatial_dispersion": 23.58782788093133,
      "active_actuator_count": 109,
      "actuator_topological_distance": 36.0
    },
    {
      "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
      "initial_hydraulic_wetness": 0.5709393946153708,
      "initial_flooding_activity": 0.06478004291845493,
      "initial_storage_utilization": 0.5083223295131927,
      "multi_actuator_spatial_dispersion": 23.585812356979407,
      "active_actuator_count": 108,
      "actuator_topological_distance": 36.0
    },
    {
      "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
      "initial_hydraulic_wetness": 0.5358807680300088,
      "initial_flooding_activity": 0.09938304721030043,
      "initial_storage_utilization": 0.3805417989875513,
      "multi_actuator_spatial_dispersion": 23.58782788093133,
      "active_actuator_count": 109,
      "actuator_topological_distance": 36.0
    },
    {
      "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
      "initial_hydraulic_wetness": 0.4025198265499995,
      "initial_flooding_activity": 0.03920779685264664,
      "initial_storage_utilization": 0.2963837328194575,
      "multi_actuator_spatial_dispersion": 23.58782788093133,
      "active_actuator_count": 108,
      "actuator_topological_distance": 36.0
    },
    {
      "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
      "initial_hydraulic_wetness": 0.43340748046476707,
      "initial_flooding_activity": 0.039550548402479735,
      "initial_storage_utilization": 0.3311849845040567,
      "multi_actuator_spatial_dispersion": 23.58782788093133,
      "active_actuator_count": 109,
      "actuator_topological_distance": 36.0
    }
  ],
  "performance": {
    "tiny": {
      "data_load_seconds": 0.0,
      "forward_seconds": 0.42427369992947206,
      "backward_seconds": 0.6346564999548718,
      "optimizer_seconds": 0.008535899978596717,
      "wall_time_seconds": 1.7121062999940477
    },
    "micro": {
      "data_load_seconds": 0.0,
      "forward_seconds": 1.5363830999704078,
      "backward_seconds": 2.309150099987164,
      "optimizer_seconds": 0.01966529997298494,
      "wall_time_seconds": 6.3979691999848
    },
    "reference_forwards_per_group": 1,
    "gpu": {
      "available": true,
      "mean_percent": 0.0,
      "p90_percent": 0.0,
      "max_percent": 0.0,
      "peak_memory_mib": 6793.0
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
  "scientific_interpretation": {
    "d3_topology_path_structurally_active": true,
    "topology_ablation_changed_prediction": true,
    "d2_regressed_vs_v4_2_1": true,
    "d3_consistent_improvement_vs_v4_2_1": false,
    "large_effect_response_ratio": "WORSE_OR_UNCHANGED",
    "magnitude_only_hypothesis": "INSUFFICIENT",
    "topology_conditioned_interaction_next": "YES"
  },
  "verdict": "RED",
  "ready_for_full_train_smoke": false,
  "ready_for_formal": false,
  "need_new_swmm": false,
  "next_bounded_action": "Do not promote V4.3; external review should decide a state/topology interaction redesign before any production wiring or full Train-only smoke."
}
```
