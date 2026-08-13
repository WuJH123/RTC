# STEP2 EDGE HYDRAULIC INTERACTION AUDIT V4.4

```json
{
  "contract": "STEP2_EDGE_HYDRAULIC_INTERACTION_AUDIT_V44",
  "edge_lineage": {
    "physical_links": 1276,
    "conduits": 1167,
    "pumps": 57,
    "orifices": 42,
    "weirs": 10,
    "others": 0,
    "new_physical_directed_edges": 2552,
    "parallel_link_node_pairs": 45,
    "ambiguous_old_mappings": 90,
    "mapping_complete": true,
    "legacy_graph_mapping_one_to_one": false,
    "normalization_sha256": "75bc18a592347a3f3479b37a5e30f65bf75dbe37ced49ef682104b163836926c",
    "edge_features_finite": true,
    "dynamic_features": {
      "head_src": true,
      "head_dst": true,
      "delta_head": true,
      "hydraulic_gradient": true,
      "source": "causal model-predicted reference trajectory",
      "future_truth_used": false,
      "link_flow_used": false
    },
    "status": "PASS_PHYSICAL_LINEAGE_WITH_LEGACY_MULTI_EDGE_AMBIGUITY"
  },
  "initial_equivalence": {
    "tiny_max_direct_tfv_difference_m3": 0.0,
    "micro_max_direct_tfv_difference_m3": 0.0,
    "pass": true
  },
  "tiny": {
    "baseline": {
      "groups": 1,
      "spread_ratio": 1.9142236177530236,
      "rank": 0.4523809523809524,
      "pairwise": 0.6785714285714286,
      "sign": 1.0,
      "top1": 1,
      "mean_regret_m3": 0.0,
      "max_regret_m3": 0.0
    },
    "edge_hydraulic": {
      "groups": 1,
      "spread_ratio": 1.8001338458079323,
      "rank": 0.880952380952381,
      "pairwise": 0.8928571428571429,
      "sign": 1.0,
      "top1": 1,
      "mean_regret_m3": 0.0,
      "max_regret_m3": 0.0
    },
    "baseline_equivalence_m3": 0.0,
    "edge_gradient": {
      "objective": -130226.421875,
      "gradient_finite": true,
      "gradient_l2": 4585626624.0,
      "gradient_nonzero": true,
      "trainable_parameter_count": 8
    },
    "edge_train_profile": {
      "forward_seconds": 1.9398741000331938,
      "backward_seconds": 1.177998699946329,
      "optimizer_seconds": 0.015514099912252277,
      "wall_time_seconds": 6.23345739999786,
      "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
      "gpu_peak_memory_allocated_bytes": 3113371136,
      "gpu_peak_memory_reserved_bytes": 4076863488,
      "gpu_utilization_mean_percent": 44.8125,
      "gpu_utilization_p90_percent": 73.0,
      "gpu_utilization_max_percent": 74.0,
      "gpu_memory_used_mean_mib": 3982.0,
      "gpu_memory_used_p90_mib": 4048.0,
      "gpu_memory_used_max_mib": 4048.0,
      "gpu_telemetry_samples": 16
    },
    "passed": true
  },
  "micro": {
    "in_sample_mechanism_micro": true,
    "baseline": {
      "groups": 6,
      "spread_ratio": 0.7173724724770453,
      "rank": 0.36507936507936517,
      "pairwise": 0.6369047619047619,
      "sign": 0.7916666666666666,
      "top1": 1,
      "mean_regret_m3": 88596.70833333333,
      "max_regret_m3": 228632.25
    },
    "edge_hydraulic": {
      "groups": 6,
      "spread_ratio": 0.7302114905682976,
      "rank": 0.3452380952380953,
      "pairwise": 0.6309523809523809,
      "sign": 0.5416666666666666,
      "top1": 1,
      "mean_regret_m3": 88596.70833333333,
      "max_regret_m3": 228632.25
    },
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
    "magnitude_baseline": {
      "small": {
        "count": 15,
        "mae_m3": 78527.79541015625,
        "bias_m3": -72024.15817057292,
        "response_ratio": 3.59139038692125,
        "rank": 0.6749999999999999,
        "pairwise": 0.7833333333333333,
        "sign": 0.7333333333333333
      },
      "medium": {
        "count": 12,
        "mae_m3": 48327.345052083336,
        "bias_m3": -35.638020833333336,
        "response_ratio": 0.8364486151608708,
        "rank": 0.0666666666666667,
        "pairwise": 0.5,
        "sign": 0.9166666666666666
      },
      "large": {
        "count": 21,
        "mae_m3": 160848.67764136905,
        "bias_m3": -52039.85398065476,
        "response_ratio": 0.4213015068253405,
        "rank": 0.25999999999999995,
        "pairwise": 0.6,
        "sign": 0.7619047619047619
      }
    },
    "magnitude_edge_hydraulic": {
      "small": {
        "count": 15,
        "mae_m3": 51849.585286458336,
        "bias_m3": -31443.645963541665,
        "response_ratio": 2.483561319333025,
        "rank": 0.6749999999999999,
        "pairwise": 0.7833333333333333,
        "sign": 0.6
      },
      "medium": {
        "count": 12,
        "mae_m3": 70205.4931640625,
        "bias_m3": 50287.607747395836,
        "response_ratio": 0.5453748898096981,
        "rank": 0.0666666666666667,
        "pairwise": 0.5,
        "sign": 0.4166666666666667
      },
      "large": {
        "count": 21,
        "mae_m3": 195772.94456845237,
        "bias_m3": 5544.8871837797615,
        "response_ratio": 0.25071868869085895,
        "rank": 0.29999999999999993,
        "pairwise": 0.6199999999999999,
        "sign": 0.5714285714285714
      }
    },
    "decomposition": {
      "small": {
        "required_interaction": {
          "count": 15,
          "mean_signed_m3": 101517.88541666667,
          "mean_abs_m3": 119528.17421875
        },
        "predicted_final_interaction": {
          "count": 15,
          "mean_signed_m3": 70074.24049479167,
          "mean_abs_m3": 78871.708203125
        },
        "predicted_edge_hydraulic_residual": {
          "count": 15,
          "mean_signed_m3": 40580.513899739584,
          "mean_abs_m3": 40580.513899739584
        },
        "required_vs_predicted_sign_agreement": 0.9333333333333333
      },
      "medium": {
        "required_interaction": {
          "count": 12,
          "mean_signed_m3": 6895.784993489583,
          "mean_abs_m3": 86204.02750651042
        },
        "predicted_final_interaction": {
          "count": 12,
          "mean_signed_m3": 57183.39217122396,
          "mean_abs_m3": 73778.44897460938
        },
        "predicted_edge_hydraulic_residual": {
          "count": 12,
          "mean_signed_m3": 50323.24556477865,
          "mean_abs_m3": 50323.24556477865
        },
        "required_vs_predicted_sign_agreement": 0.75
      },
      "large": {
        "required_interaction": {
          "count": 21,
          "mean_signed_m3": 82629.55013020833,
          "mean_abs_m3": 180006.65745907737
        },
        "predicted_final_interaction": {
          "count": 21,
          "mean_signed_m3": 88174.43843005953,
          "mean_abs_m3": 105982.57421875
        },
        "predicted_edge_hydraulic_residual": {
          "count": 21,
          "mean_signed_m3": 57584.74274553572,
          "mean_abs_m3": 57584.74274553572
        },
        "required_vs_predicted_sign_agreement": 0.5714285714285714
      }
    },
    "edge_ablation": {
      "edge_on": {
        "groups": 6,
        "spread_ratio": 0.7302114894151198,
        "rank": 0.3452380952380953,
        "pairwise": 0.6309523809523809,
        "sign": 0.5416666666666666,
        "top1": 1,
        "mean_regret_m3": 88596.70833333333,
        "max_regret_m3": 228632.25
      },
      "edge_off": {
        "groups": 6,
        "spread_ratio": 0.71737247055847,
        "rank": 0.36507936507936517,
        "pairwise": 0.6369047619047619,
        "sign": 0.7916666666666666,
        "top1": 1,
        "mean_regret_m3": 88596.70833333333,
        "max_regret_m3": 228632.25
      },
      "delta_rank": -0.019841269841269882,
      "delta_pairwise": -0.005952380952380931,
      "delta_top1": 0,
      "delta_mean_regret_m3": 0.0,
      "delta_max_regret_m3": 0.0
    },
    "interaction_alignment": {
      "count": 48,
      "required_vs_final_sign_agreement": 0.7291666666666666,
      "required_vs_final_spearman": 0.5125922709509335,
      "required_vs_edge_sign_agreement": 0.6458333333333334,
      "required_vs_edge_spearman": 0.026378636561007383
    },
    "spatial_ablation": {
      "endpoint": {
        "node_count": 351,
        "mean_abs_state_delta": 289.99917615224155,
        "max_abs_state_delta": 6272.79541015625
      },
      "1-hop": {
        "node_count": 384,
        "mean_abs_state_delta": 15.903543325761953,
        "max_abs_state_delta": 22.509737014770508
      },
      "2-hop": {
        "node_count": 267,
        "mean_abs_state_delta": 16.064287452773655,
        "max_abs_state_delta": 22.572473526000977
      },
      "3-hop": {
        "node_count": 279,
        "mean_abs_state_delta": 16.84676720463674,
        "max_abs_state_delta": 22.9193115234375
      },
      ">3-hop": {
        "node_count": 1515,
        "mean_abs_state_delta": 19.027045110192628,
        "max_abs_state_delta": 23.02712059020996
      }
    },
    "edge_contribution": "HARMFUL",
    "candidate_edge_contributions": [
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 0,
        "true_delta_tfv_m3": -648.0,
        "predicted_additive_single_delta_tfv_m3": -2282.369384765625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2282.369384765625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2282.369384765625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 1,
        "true_delta_tfv_m3": 1177.5,
        "predicted_additive_single_delta_tfv_m3": 3062.654296875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3062.654296875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3062.654296875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 2,
        "true_delta_tfv_m3": 4714.5,
        "predicted_additive_single_delta_tfv_m3": 3530.378173828125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3530.378173828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3530.378173828125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 3,
        "true_delta_tfv_m3": -7343.5,
        "predicted_additive_single_delta_tfv_m3": -14120.26953125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -14120.26953125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -14120.26953125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 4,
        "true_delta_tfv_m3": 7381.5,
        "predicted_additive_single_delta_tfv_m3": 9282.642578125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 9282.642578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 9282.642578125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 5,
        "true_delta_tfv_m3": 5092.0,
        "predicted_additive_single_delta_tfv_m3": 8987.8291015625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 8987.8291015625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 8987.8291015625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 6,
        "true_delta_tfv_m3": 4572.5,
        "predicted_additive_single_delta_tfv_m3": 6062.91357421875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 6062.91357421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 6062.91357421875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 7,
        "true_delta_tfv_m3": -5084.0,
        "predicted_additive_single_delta_tfv_m3": -6759.40673828125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -6759.40673828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -6759.40673828125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 8,
        "true_delta_tfv_m3": 6436.5,
        "predicted_additive_single_delta_tfv_m3": 9623.900390625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 9623.900390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 9623.900390625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 9,
        "true_delta_tfv_m3": -2938.5,
        "predicted_additive_single_delta_tfv_m3": -8065.91748046875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -8065.91748046875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -8065.91748046875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 10,
        "true_delta_tfv_m3": -7343.5,
        "predicted_additive_single_delta_tfv_m3": -14755.9853515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -14755.9853515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -14755.9853515625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 11,
        "true_delta_tfv_m3": -1661.0,
        "predicted_additive_single_delta_tfv_m3": -3123.502197265625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -3123.502197265625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -3123.502197265625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 12,
        "true_delta_tfv_m3": -2938.5,
        "predicted_additive_single_delta_tfv_m3": -6942.61962890625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -6942.61962890625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -6942.61962890625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 13,
        "true_delta_tfv_m3": 4310.5,
        "predicted_additive_single_delta_tfv_m3": 2363.08056640625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 2363.08056640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 2363.08056640625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 14,
        "true_delta_tfv_m3": -1661.0,
        "predicted_additive_single_delta_tfv_m3": -5423.2138671875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5423.2138671875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5423.2138671875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 15,
        "true_delta_tfv_m3": 28.5,
        "predicted_additive_single_delta_tfv_m3": 343.33966064453125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 343.33966064453125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 343.33966064453125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 16,
        "true_delta_tfv_m3": 10388.5,
        "predicted_additive_single_delta_tfv_m3": 10231.64453125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 10231.64453125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 10231.64453125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 17,
        "true_delta_tfv_m3": 2041.5,
        "predicted_additive_single_delta_tfv_m3": 4498.79931640625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 4498.79931640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 4498.79931640625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 18,
        "true_delta_tfv_m3": 8423.5,
        "predicted_additive_single_delta_tfv_m3": 13500.23046875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 13500.23046875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 13500.23046875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 19,
        "true_delta_tfv_m3": 0.0,
        "predicted_additive_single_delta_tfv_m3": 1350.86572265625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1350.86572265625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1350.86572265625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 20,
        "true_delta_tfv_m3": -67.0,
        "predicted_additive_single_delta_tfv_m3": -1150.6099853515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1150.6099853515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1150.6099853515625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 21,
        "true_delta_tfv_m3": 293.5,
        "predicted_additive_single_delta_tfv_m3": 3168.494140625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3168.494140625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3168.494140625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 22,
        "true_delta_tfv_m3": 10022.0,
        "predicted_additive_single_delta_tfv_m3": 16032.9853515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 16032.9853515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 16032.9853515625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D2",
        "candidate_index": 23,
        "true_delta_tfv_m3": -1786.5,
        "predicted_additive_single_delta_tfv_m3": -5662.998046875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5662.998046875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5662.998046875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 0,
        "true_delta_tfv_m3": 6221.5,
        "predicted_additive_single_delta_tfv_m3": 2067.117431640625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 2067.117431640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 2067.117431640625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 1,
        "true_delta_tfv_m3": 2892.5,
        "predicted_additive_single_delta_tfv_m3": 3377.876220703125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3377.876220703125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3377.876220703125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 2,
        "true_delta_tfv_m3": -436.0,
        "predicted_additive_single_delta_tfv_m3": 5751.28955078125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 5751.28955078125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 5751.28955078125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 3,
        "true_delta_tfv_m3": 1160.0,
        "predicted_additive_single_delta_tfv_m3": -4657.59326171875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -4657.59326171875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -4657.59326171875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 4,
        "true_delta_tfv_m3": 4225.0,
        "predicted_additive_single_delta_tfv_m3": 6062.9091796875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 6062.9091796875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 6062.9091796875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 5,
        "true_delta_tfv_m3": 4880.5,
        "predicted_additive_single_delta_tfv_m3": 2686.97412109375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 2686.97412109375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 2686.97412109375
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 6,
        "true_delta_tfv_m3": 2009.0,
        "predicted_additive_single_delta_tfv_m3": 1307.4095458984375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1307.4095458984375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1307.4095458984375
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 7,
        "true_delta_tfv_m3": 6756.0,
        "predicted_additive_single_delta_tfv_m3": 9395.5986328125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 9395.5986328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 9395.5986328125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 8,
        "true_delta_tfv_m3": 0.0,
        "predicted_additive_single_delta_tfv_m3": 343.36981201171875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 343.36981201171875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 343.36981201171875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 9,
        "true_delta_tfv_m3": 1182.0,
        "predicted_additive_single_delta_tfv_m3": 354.9830017089844,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 354.9830017089844,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 354.9830017089844
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 10,
        "true_delta_tfv_m3": 4880.5,
        "predicted_additive_single_delta_tfv_m3": -137.1268768310547,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -137.1268768310547,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -137.1268768310547
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 11,
        "true_delta_tfv_m3": -7110.0,
        "predicted_additive_single_delta_tfv_m3": -6670.31884765625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -6670.31884765625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -6670.31884765625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 12,
        "true_delta_tfv_m3": 1160.0,
        "predicted_additive_single_delta_tfv_m3": -622.8677978515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -622.8677978515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -622.8677978515625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 13,
        "true_delta_tfv_m3": -9107.5,
        "predicted_additive_single_delta_tfv_m3": -7055.732421875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -7055.732421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -7055.732421875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 14,
        "true_delta_tfv_m3": 6756.0,
        "predicted_additive_single_delta_tfv_m3": 11940.4580078125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 11940.4580078125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 11940.4580078125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 15,
        "true_delta_tfv_m3": 4880.5,
        "predicted_additive_single_delta_tfv_m3": -1692.7857666015625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1692.7857666015625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1692.7857666015625
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 16,
        "true_delta_tfv_m3": 4880.5,
        "predicted_additive_single_delta_tfv_m3": -598.8057861328125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -598.8057861328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -598.8057861328125
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 17,
        "true_delta_tfv_m3": 1283.5,
        "predicted_additive_single_delta_tfv_m3": 5148.25537109375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 5148.25537109375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 5148.25537109375
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 18,
        "true_delta_tfv_m3": 5068.0,
        "predicted_additive_single_delta_tfv_m3": 667.5578002929688,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 667.5578002929688,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 667.5578002929688
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 19,
        "true_delta_tfv_m3": 1857.0,
        "predicted_additive_single_delta_tfv_m3": -1430.4993896484375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1430.4993896484375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1430.4993896484375
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 20,
        "true_delta_tfv_m3": 2082.0,
        "predicted_additive_single_delta_tfv_m3": 2025.01171875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 2025.01171875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 2025.01171875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 21,
        "true_delta_tfv_m3": 2374.0,
        "predicted_additive_single_delta_tfv_m3": -558.75146484375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -558.75146484375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -558.75146484375
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 22,
        "true_delta_tfv_m3": -15382.5,
        "predicted_additive_single_delta_tfv_m3": -24006.82421875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -24006.82421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -24006.82421875
      },
      {
        "group": "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D2",
        "candidate_index": 23,
        "true_delta_tfv_m3": 144.0,
        "predicted_additive_single_delta_tfv_m3": -1732.3265380859375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1732.3265380859375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1732.3265380859375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 0,
        "true_delta_tfv_m3": 0.0,
        "predicted_additive_single_delta_tfv_m3": -4335.732421875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -4335.732421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -4335.732421875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 1,
        "true_delta_tfv_m3": -2877.0,
        "predicted_additive_single_delta_tfv_m3": 4913.64208984375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 4913.64208984375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 4913.64208984375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 2,
        "true_delta_tfv_m3": 0.0,
        "predicted_additive_single_delta_tfv_m3": 3810.9482421875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3810.9482421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3810.9482421875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 3,
        "true_delta_tfv_m3": 1862.0,
        "predicted_additive_single_delta_tfv_m3": 3493.934814453125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3493.934814453125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3493.934814453125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 4,
        "true_delta_tfv_m3": -493.5,
        "predicted_additive_single_delta_tfv_m3": 8979.8525390625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 8979.8525390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 8979.8525390625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 5,
        "true_delta_tfv_m3": 2120.0,
        "predicted_additive_single_delta_tfv_m3": -2924.507568359375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2924.507568359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2924.507568359375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 6,
        "true_delta_tfv_m3": -10190.5,
        "predicted_additive_single_delta_tfv_m3": -27.606531143188477,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -27.606531143188477,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -27.606531143188477
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 7,
        "true_delta_tfv_m3": -3882.0,
        "predicted_additive_single_delta_tfv_m3": -4730.91943359375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -4730.91943359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -4730.91943359375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 8,
        "true_delta_tfv_m3": 17465.5,
        "predicted_additive_single_delta_tfv_m3": 11509.0556640625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 11509.0556640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 11509.0556640625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 9,
        "true_delta_tfv_m3": -9532.0,
        "predicted_additive_single_delta_tfv_m3": -8457.2451171875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -8457.2451171875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -8457.2451171875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 10,
        "true_delta_tfv_m3": -389.5,
        "predicted_additive_single_delta_tfv_m3": -6923.35693359375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -6923.35693359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -6923.35693359375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 11,
        "true_delta_tfv_m3": 8658.5,
        "predicted_additive_single_delta_tfv_m3": 9259.072265625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 9259.072265625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 9259.072265625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 12,
        "true_delta_tfv_m3": -5875.5,
        "predicted_additive_single_delta_tfv_m3": -13913.3701171875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -13913.3701171875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -13913.3701171875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 13,
        "true_delta_tfv_m3": -4483.0,
        "predicted_additive_single_delta_tfv_m3": -3579.94091796875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -3579.94091796875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -3579.94091796875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 14,
        "true_delta_tfv_m3": -4755.0,
        "predicted_additive_single_delta_tfv_m3": -5663.1904296875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5663.1904296875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5663.1904296875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 15,
        "true_delta_tfv_m3": -5875.5,
        "predicted_additive_single_delta_tfv_m3": -13821.41015625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -13821.41015625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -13821.41015625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 16,
        "true_delta_tfv_m3": -1470.0,
        "predicted_additive_single_delta_tfv_m3": 2932.889892578125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 2932.889892578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 2932.889892578125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 17,
        "true_delta_tfv_m3": -18953.5,
        "predicted_additive_single_delta_tfv_m3": -11832.712890625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -11832.712890625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -11832.712890625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 18,
        "true_delta_tfv_m3": -7653.0,
        "predicted_additive_single_delta_tfv_m3": -7936.79443359375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -7936.79443359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -7936.79443359375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 19,
        "true_delta_tfv_m3": -20514.0,
        "predicted_additive_single_delta_tfv_m3": -12116.69140625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -12116.69140625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -12116.69140625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 20,
        "true_delta_tfv_m3": 3172.0,
        "predicted_additive_single_delta_tfv_m3": 8447.931640625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 8447.931640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 8447.931640625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 21,
        "true_delta_tfv_m3": -18953.5,
        "predicted_additive_single_delta_tfv_m3": -13999.888671875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -13999.888671875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -13999.888671875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 22,
        "true_delta_tfv_m3": -493.5,
        "predicted_additive_single_delta_tfv_m3": -6158.61474609375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -6158.61474609375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -6158.61474609375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D2",
        "candidate_index": 23,
        "true_delta_tfv_m3": -8767.5,
        "predicted_additive_single_delta_tfv_m3": -5375.65380859375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5375.65380859375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5375.65380859375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 0,
        "true_delta_tfv_m3": -6133.0,
        "predicted_additive_single_delta_tfv_m3": 4663.24169921875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 4663.24169921875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 4663.24169921875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 1,
        "true_delta_tfv_m3": 0.0,
        "predicted_additive_single_delta_tfv_m3": -1298.5263671875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1298.5263671875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1298.5263671875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 2,
        "true_delta_tfv_m3": -2112.0,
        "predicted_additive_single_delta_tfv_m3": -4193.80322265625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -4193.80322265625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -4193.80322265625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 3,
        "true_delta_tfv_m3": -4984.0,
        "predicted_additive_single_delta_tfv_m3": -5835.03173828125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5835.03173828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5835.03173828125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 4,
        "true_delta_tfv_m3": 1485.0,
        "predicted_additive_single_delta_tfv_m3": -376.8538513183594,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -376.8538513183594,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -376.8538513183594
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 5,
        "true_delta_tfv_m3": -5203.5,
        "predicted_additive_single_delta_tfv_m3": -2149.719970703125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2149.719970703125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2149.719970703125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 6,
        "true_delta_tfv_m3": -15341.0,
        "predicted_additive_single_delta_tfv_m3": -2925.682861328125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2925.682861328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2925.682861328125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 7,
        "true_delta_tfv_m3": -5624.0,
        "predicted_additive_single_delta_tfv_m3": -5005.171875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5005.171875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5005.171875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 8,
        "true_delta_tfv_m3": -5534.0,
        "predicted_additive_single_delta_tfv_m3": -1373.4224853515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1373.4224853515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1373.4224853515625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 9,
        "true_delta_tfv_m3": -12508.5,
        "predicted_additive_single_delta_tfv_m3": -2899.952880859375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2899.952880859375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2899.952880859375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 10,
        "true_delta_tfv_m3": 5622.0,
        "predicted_additive_single_delta_tfv_m3": -1569.09716796875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1569.09716796875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1569.09716796875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 11,
        "true_delta_tfv_m3": 34277.0,
        "predicted_additive_single_delta_tfv_m3": -5363.11474609375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5363.11474609375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5363.11474609375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 12,
        "true_delta_tfv_m3": -16137.0,
        "predicted_additive_single_delta_tfv_m3": -8207.6943359375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -8207.6943359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -8207.6943359375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 13,
        "true_delta_tfv_m3": -18717.0,
        "predicted_additive_single_delta_tfv_m3": -25293.912109375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -25293.912109375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -25293.912109375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 14,
        "true_delta_tfv_m3": -16137.0,
        "predicted_additive_single_delta_tfv_m3": -8514.9892578125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -8514.9892578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -8514.9892578125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 15,
        "true_delta_tfv_m3": 7755.0,
        "predicted_additive_single_delta_tfv_m3": -9702.400390625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -9702.400390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -9702.400390625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 16,
        "true_delta_tfv_m3": 2501.5,
        "predicted_additive_single_delta_tfv_m3": 1283.0682373046875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1283.0682373046875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1283.0682373046875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 17,
        "true_delta_tfv_m3": -10193.0,
        "predicted_additive_single_delta_tfv_m3": -7484.16455078125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -7484.16455078125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -7484.16455078125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 18,
        "true_delta_tfv_m3": -11404.0,
        "predicted_additive_single_delta_tfv_m3": -3395.506591796875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -3395.506591796875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -3395.506591796875
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 19,
        "true_delta_tfv_m3": -9931.0,
        "predicted_additive_single_delta_tfv_m3": 1621.8017578125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1621.8017578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1621.8017578125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 20,
        "true_delta_tfv_m3": -8905.5,
        "predicted_additive_single_delta_tfv_m3": -7847.7333984375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -7847.7333984375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -7847.7333984375
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 21,
        "true_delta_tfv_m3": 0.0,
        "predicted_additive_single_delta_tfv_m3": 305.6461181640625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 305.6461181640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 305.6461181640625
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 22,
        "true_delta_tfv_m3": -932.0,
        "predicted_additive_single_delta_tfv_m3": -2788.26611328125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2788.26611328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2788.26611328125
      },
      {
        "group": "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D2",
        "candidate_index": 23,
        "true_delta_tfv_m3": -8027.5,
        "predicted_additive_single_delta_tfv_m3": -2210.233642578125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2210.233642578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2210.233642578125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 0,
        "true_delta_tfv_m3": -5183.75,
        "predicted_additive_single_delta_tfv_m3": -12545.8046875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -12545.8046875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -12545.8046875
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 1,
        "true_delta_tfv_m3": 3805.5,
        "predicted_additive_single_delta_tfv_m3": -5376.1396484375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -5376.1396484375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -5376.1396484375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 2,
        "true_delta_tfv_m3": 7226.25,
        "predicted_additive_single_delta_tfv_m3": 4799.1767578125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 4799.1767578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 4799.1767578125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 3,
        "true_delta_tfv_m3": 3838.75,
        "predicted_additive_single_delta_tfv_m3": 1907.82861328125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1907.82861328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1907.82861328125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 4,
        "true_delta_tfv_m3": 11720.5,
        "predicted_additive_single_delta_tfv_m3": 8893.27734375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 8893.27734375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 8893.27734375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 5,
        "true_delta_tfv_m3": -4045.25,
        "predicted_additive_single_delta_tfv_m3": -10071.171875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -10071.171875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -10071.171875
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 6,
        "true_delta_tfv_m3": 429.25,
        "predicted_additive_single_delta_tfv_m3": -1465.0540771484375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1465.0540771484375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1465.0540771484375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 7,
        "true_delta_tfv_m3": 5804.5,
        "predicted_additive_single_delta_tfv_m3": 2114.52978515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 2114.52978515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 2114.52978515625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 8,
        "true_delta_tfv_m3": 5636.25,
        "predicted_additive_single_delta_tfv_m3": -8315.2421875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -8315.2421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -8315.2421875
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 9,
        "true_delta_tfv_m3": 14742.25,
        "predicted_additive_single_delta_tfv_m3": 14797.376953125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 14797.376953125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 14797.376953125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 10,
        "true_delta_tfv_m3": 9725.25,
        "predicted_additive_single_delta_tfv_m3": 10806.5732421875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 10806.5732421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 10806.5732421875
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 11,
        "true_delta_tfv_m3": 5636.25,
        "predicted_additive_single_delta_tfv_m3": 137.0226593017578,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 137.0226593017578,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 137.0226593017578
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 12,
        "true_delta_tfv_m3": 7719.5,
        "predicted_additive_single_delta_tfv_m3": 1128.85693359375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1128.85693359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1128.85693359375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 13,
        "true_delta_tfv_m3": 10445.75,
        "predicted_additive_single_delta_tfv_m3": 13443.0234375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 13443.0234375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 13443.0234375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 14,
        "true_delta_tfv_m3": 5636.25,
        "predicted_additive_single_delta_tfv_m3": -6465.58056640625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -6465.58056640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -6465.58056640625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 15,
        "true_delta_tfv_m3": 801.0,
        "predicted_additive_single_delta_tfv_m3": -3005.0302734375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -3005.0302734375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -3005.0302734375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 16,
        "true_delta_tfv_m3": -3292.25,
        "predicted_additive_single_delta_tfv_m3": -8033.0625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -8033.0625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -8033.0625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 17,
        "true_delta_tfv_m3": 1756.75,
        "predicted_additive_single_delta_tfv_m3": -4123.001953125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -4123.001953125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -4123.001953125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 18,
        "true_delta_tfv_m3": 11057.5,
        "predicted_additive_single_delta_tfv_m3": 143.2379150390625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 143.2379150390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 143.2379150390625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 19,
        "true_delta_tfv_m3": 10023.5,
        "predicted_additive_single_delta_tfv_m3": 3240.1708984375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3240.1708984375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3240.1708984375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 20,
        "true_delta_tfv_m3": 2121.5,
        "predicted_additive_single_delta_tfv_m3": 7366.6884765625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 7366.6884765625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 7366.6884765625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 21,
        "true_delta_tfv_m3": -530.25,
        "predicted_additive_single_delta_tfv_m3": 1908.7562255859375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1908.7562255859375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1908.7562255859375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 22,
        "true_delta_tfv_m3": 9725.25,
        "predicted_additive_single_delta_tfv_m3": 13307.6220703125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 13307.6220703125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 13307.6220703125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D2",
        "candidate_index": 23,
        "true_delta_tfv_m3": 3805.5,
        "predicted_additive_single_delta_tfv_m3": 8138.74853515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 8138.74853515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 8138.74853515625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 0,
        "true_delta_tfv_m3": -3569.5,
        "predicted_additive_single_delta_tfv_m3": -1413.77099609375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1413.77099609375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1413.77099609375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 1,
        "true_delta_tfv_m3": -3569.5,
        "predicted_additive_single_delta_tfv_m3": -4698.08984375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -4698.08984375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -4698.08984375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 2,
        "true_delta_tfv_m3": 238.5,
        "predicted_additive_single_delta_tfv_m3": 1941.430419921875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1941.430419921875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1941.430419921875
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 3,
        "true_delta_tfv_m3": 8632.25,
        "predicted_additive_single_delta_tfv_m3": 11952.37890625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 11952.37890625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 11952.37890625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 4,
        "true_delta_tfv_m3": -3569.5,
        "predicted_additive_single_delta_tfv_m3": -10136.9462890625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -10136.9462890625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -10136.9462890625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 5,
        "true_delta_tfv_m3": -1988.0,
        "predicted_additive_single_delta_tfv_m3": -2258.857666015625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -2258.857666015625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -2258.857666015625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 6,
        "true_delta_tfv_m3": 1754.5,
        "predicted_additive_single_delta_tfv_m3": -1465.0841064453125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -1465.0841064453125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -1465.0841064453125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 7,
        "true_delta_tfv_m3": 2586.75,
        "predicted_additive_single_delta_tfv_m3": 355.04766845703125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 355.04766845703125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 355.04766845703125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 8,
        "true_delta_tfv_m3": -1988.0,
        "predicted_additive_single_delta_tfv_m3": -136.20249938964844,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -136.20249938964844,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -136.20249938964844
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 9,
        "true_delta_tfv_m3": 10674.0,
        "predicted_additive_single_delta_tfv_m3": 14848.4453125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 14848.4453125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 14848.4453125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 10,
        "true_delta_tfv_m3": 10944.0,
        "predicted_additive_single_delta_tfv_m3": 10799.6962890625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 10799.6962890625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 10799.6962890625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 11,
        "true_delta_tfv_m3": 4419.5,
        "predicted_additive_single_delta_tfv_m3": 141.68569946289062,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 141.68569946289062,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 141.68569946289062
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 12,
        "true_delta_tfv_m3": 313.0,
        "predicted_additive_single_delta_tfv_m3": -7059.240234375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -7059.240234375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -7059.240234375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 13,
        "true_delta_tfv_m3": 3997.5,
        "predicted_additive_single_delta_tfv_m3": 1148.9364013671875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1148.9364013671875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1148.9364013671875
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 14,
        "true_delta_tfv_m3": 1763.5,
        "predicted_additive_single_delta_tfv_m3": 1853.7103271484375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1853.7103271484375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1853.7103271484375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 15,
        "true_delta_tfv_m3": -1988.0,
        "predicted_additive_single_delta_tfv_m3": -602.0758666992188,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -602.0758666992188,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -602.0758666992188
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 16,
        "true_delta_tfv_m3": 2816.5,
        "predicted_additive_single_delta_tfv_m3": 1633.8343505859375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 1633.8343505859375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 1633.8343505859375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 17,
        "true_delta_tfv_m3": 2613.5,
        "predicted_additive_single_delta_tfv_m3": 2040.95263671875,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 2040.95263671875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 2040.95263671875
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 18,
        "true_delta_tfv_m3": 2413.0,
        "predicted_additive_single_delta_tfv_m3": 3286.65478515625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 3286.65478515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 3286.65478515625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 19,
        "true_delta_tfv_m3": 8681.5,
        "predicted_additive_single_delta_tfv_m3": 12806.3291015625,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 12806.3291015625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 12806.3291015625
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 20,
        "true_delta_tfv_m3": 12288.0,
        "predicted_additive_single_delta_tfv_m3": 7366.71484375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 7366.71484375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 7366.71484375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 21,
        "true_delta_tfv_m3": -2640.5,
        "predicted_additive_single_delta_tfv_m3": -4735.4658203125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": -4735.4658203125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": -4735.4658203125
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 22,
        "true_delta_tfv_m3": 10944.0,
        "predicted_additive_single_delta_tfv_m3": 13307.1787109375,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 13307.1787109375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 13307.1787109375
      },
      {
        "group": "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D2",
        "candidate_index": 23,
        "true_delta_tfv_m3": 8354.0,
        "predicted_additive_single_delta_tfv_m3": 8135.5751953125,
        "predicted_interaction_delta_tfv_m3": 0.0,
        "predicted_final_delta_tfv_m3": 8135.5751953125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 0.0,
        "predicted_edge_off_direct_delta_tfv_m3": 8135.5751953125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 0,
        "true_delta_tfv_m3": -50199.0,
        "predicted_additive_single_delta_tfv_m3": -260313.5625,
        "predicted_interaction_delta_tfv_m3": 140393.8125,
        "predicted_final_delta_tfv_m3": -119919.75,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 33234.7578125,
        "predicted_edge_off_direct_delta_tfv_m3": -153154.5
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 1,
        "true_delta_tfv_m3": -82962.5,
        "predicted_additive_single_delta_tfv_m3": -140406.578125,
        "predicted_interaction_delta_tfv_m3": 30519.498046875,
        "predicted_final_delta_tfv_m3": -109887.078125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 34459.50244140625,
        "predicted_edge_off_direct_delta_tfv_m3": -144346.578125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 2,
        "true_delta_tfv_m3": -35282.0,
        "predicted_additive_single_delta_tfv_m3": -377056.65625,
        "predicted_interaction_delta_tfv_m3": 229122.828125,
        "predicted_final_delta_tfv_m3": -147933.828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 36631.96875,
        "predicted_edge_off_direct_delta_tfv_m3": -184565.796875
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 3,
        "true_delta_tfv_m3": -39841.5,
        "predicted_additive_single_delta_tfv_m3": -157511.09375,
        "predicted_interaction_delta_tfv_m3": 81829.671875,
        "predicted_final_delta_tfv_m3": -75681.421875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 36971.27734375,
        "predicted_edge_off_direct_delta_tfv_m3": -112652.703125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 4,
        "true_delta_tfv_m3": -120244.0,
        "predicted_additive_single_delta_tfv_m3": -131144.171875,
        "predicted_interaction_delta_tfv_m3": -84124.1875,
        "predicted_final_delta_tfv_m3": -215268.359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 25420.84375,
        "predicted_edge_off_direct_delta_tfv_m3": -240689.203125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 5,
        "true_delta_tfv_m3": -2384.5,
        "predicted_additive_single_delta_tfv_m3": -12975.697265625,
        "predicted_interaction_delta_tfv_m3": 39881.90625,
        "predicted_final_delta_tfv_m3": 26906.208984375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 34171.45849609375,
        "predicted_edge_off_direct_delta_tfv_m3": -7265.24951171875
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 6,
        "true_delta_tfv_m3": -111839.5,
        "predicted_additive_single_delta_tfv_m3": -17884.82421875,
        "predicted_interaction_delta_tfv_m3": -51937.05859375,
        "predicted_final_delta_tfv_m3": -69821.8828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 28355.98828125,
        "predicted_edge_off_direct_delta_tfv_m3": -98177.875
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
        "source_kind": "D3",
        "candidate_index": 7,
        "true_delta_tfv_m3": -26624.5,
        "predicted_additive_single_delta_tfv_m3": -82073.421875,
        "predicted_interaction_delta_tfv_m3": 7873.7783203125,
        "predicted_final_delta_tfv_m3": -74199.640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 34691.5927734375,
        "predicted_edge_off_direct_delta_tfv_m3": -108891.234375
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 0,
        "true_delta_tfv_m3": -131664.5,
        "predicted_additive_single_delta_tfv_m3": -186125.765625,
        "predicted_interaction_delta_tfv_m3": 150279.375,
        "predicted_final_delta_tfv_m3": -35846.390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 31635.2109375,
        "predicted_edge_off_direct_delta_tfv_m3": -67481.6015625
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 1,
        "true_delta_tfv_m3": 935.5,
        "predicted_additive_single_delta_tfv_m3": -123041.0859375,
        "predicted_interaction_delta_tfv_m3": 106078.78125,
        "predicted_final_delta_tfv_m3": -16962.3046875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 33112.0234375,
        "predicted_edge_off_direct_delta_tfv_m3": -50074.328125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 2,
        "true_delta_tfv_m3": 841121.0,
        "predicted_additive_single_delta_tfv_m3": 13074.66015625,
        "predicted_interaction_delta_tfv_m3": -1329.3515625,
        "predicted_final_delta_tfv_m3": 11745.30859375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 34310.671875,
        "predicted_edge_off_direct_delta_tfv_m3": -22565.36328125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 3,
        "true_delta_tfv_m3": 165412.0,
        "predicted_additive_single_delta_tfv_m3": -116529.15625,
        "predicted_interaction_delta_tfv_m3": 104281.484375,
        "predicted_final_delta_tfv_m3": -12247.671875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 32984.234375,
        "predicted_edge_off_direct_delta_tfv_m3": -45231.90625
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 4,
        "true_delta_tfv_m3": -31895.0,
        "predicted_additive_single_delta_tfv_m3": 29416.943359375,
        "predicted_interaction_delta_tfv_m3": -65981.0078125,
        "predicted_final_delta_tfv_m3": -36564.0625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 30818.515625,
        "predicted_edge_off_direct_delta_tfv_m3": -67382.578125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 5,
        "true_delta_tfv_m3": -146864.5,
        "predicted_additive_single_delta_tfv_m3": -166032.234375,
        "predicted_interaction_delta_tfv_m3": 135868.328125,
        "predicted_final_delta_tfv_m3": -30163.90625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 31767.984375,
        "predicted_edge_off_direct_delta_tfv_m3": -61931.890625
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 6,
        "true_delta_tfv_m3": -115287.0,
        "predicted_additive_single_delta_tfv_m3": 46841.2890625,
        "predicted_interaction_delta_tfv_m3": -49594.828125,
        "predicted_final_delta_tfv_m3": -2753.5390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 26197.9140625,
        "predicted_edge_off_direct_delta_tfv_m3": -28951.453125
      },
      {
        "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
        "source_kind": "D3",
        "candidate_index": 7,
        "true_delta_tfv_m3": -22121.5,
        "predicted_additive_single_delta_tfv_m3": -167372.96875,
        "predicted_interaction_delta_tfv_m3": 75160.4296875,
        "predicted_final_delta_tfv_m3": -92212.5390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 34228.72265625,
        "predicted_edge_off_direct_delta_tfv_m3": -126441.265625
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 0,
        "true_delta_tfv_m3": -180903.5,
        "predicted_additive_single_delta_tfv_m3": -44583.421875,
        "predicted_interaction_delta_tfv_m3": 115138.3046875,
        "predicted_final_delta_tfv_m3": 70554.8828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 143070.865234375,
        "predicted_edge_off_direct_delta_tfv_m3": -72515.984375
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 1,
        "true_delta_tfv_m3": -227755.0,
        "predicted_additive_single_delta_tfv_m3": -308489.5,
        "predicted_interaction_delta_tfv_m3": 326816.3125,
        "predicted_final_delta_tfv_m3": 18326.8125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 143404.515625,
        "predicted_edge_off_direct_delta_tfv_m3": -125077.703125
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 2,
        "true_delta_tfv_m3": -72143.0,
        "predicted_additive_single_delta_tfv_m3": -122682.25,
        "predicted_interaction_delta_tfv_m3": 139378.796875,
        "predicted_final_delta_tfv_m3": 16696.546875,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 140844.5830078125,
        "predicted_edge_off_direct_delta_tfv_m3": -124148.0390625
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 3,
        "true_delta_tfv_m3": -164889.0,
        "predicted_additive_single_delta_tfv_m3": -281344.1875,
        "predicted_interaction_delta_tfv_m3": 205588.15625,
        "predicted_final_delta_tfv_m3": -75756.03125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 140022.125,
        "predicted_edge_off_direct_delta_tfv_m3": -215778.15625
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 4,
        "true_delta_tfv_m3": -11749.0,
        "predicted_additive_single_delta_tfv_m3": -73070.8984375,
        "predicted_interaction_delta_tfv_m3": 103333.53125,
        "predicted_final_delta_tfv_m3": 30262.6328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 141935.5546875,
        "predicted_edge_off_direct_delta_tfv_m3": -111672.921875
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 5,
        "true_delta_tfv_m3": -89015.0,
        "predicted_additive_single_delta_tfv_m3": -97193.875,
        "predicted_interaction_delta_tfv_m3": 131142.765625,
        "predicted_final_delta_tfv_m3": 33948.890625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 140562.6474609375,
        "predicted_edge_off_direct_delta_tfv_m3": -106613.7578125
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 6,
        "true_delta_tfv_m3": -229381.0,
        "predicted_additive_single_delta_tfv_m3": -164077.453125,
        "predicted_interaction_delta_tfv_m3": 213442.28125,
        "predicted_final_delta_tfv_m3": 49364.828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 144852.484375,
        "predicted_edge_off_direct_delta_tfv_m3": -95487.65625
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
        "source_kind": "D3",
        "candidate_index": 7,
        "true_delta_tfv_m3": -153885.5,
        "predicted_additive_single_delta_tfv_m3": 10980.8037109375,
        "predicted_interaction_delta_tfv_m3": 66170.453125,
        "predicted_final_delta_tfv_m3": 77151.2578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 140001.3515625,
        "predicted_edge_off_direct_delta_tfv_m3": -62850.09375
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 0,
        "true_delta_tfv_m3": 176845.0,
        "predicted_additive_single_delta_tfv_m3": -141245.90625,
        "predicted_interaction_delta_tfv_m3": 122957.96875,
        "predicted_final_delta_tfv_m3": -18287.9375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 31306.5390625,
        "predicted_edge_off_direct_delta_tfv_m3": -49594.4765625
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 1,
        "true_delta_tfv_m3": -54091.5,
        "predicted_additive_single_delta_tfv_m3": -170833.390625,
        "predicted_interaction_delta_tfv_m3": 61687.640625,
        "predicted_final_delta_tfv_m3": -109145.75,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 30210.390625,
        "predicted_edge_off_direct_delta_tfv_m3": -139356.140625
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 2,
        "true_delta_tfv_m3": -61415.0,
        "predicted_additive_single_delta_tfv_m3": -145095.609375,
        "predicted_interaction_delta_tfv_m3": 48736.67578125,
        "predicted_final_delta_tfv_m3": -96358.9375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 30578.8515625,
        "predicted_edge_off_direct_delta_tfv_m3": -126937.78125
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 3,
        "true_delta_tfv_m3": 371178.0,
        "predicted_additive_single_delta_tfv_m3": -59865.13671875,
        "predicted_interaction_delta_tfv_m3": 42445.390625,
        "predicted_final_delta_tfv_m3": -17419.74609375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 32525.759765625,
        "predicted_edge_off_direct_delta_tfv_m3": -49945.5078125
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 4,
        "true_delta_tfv_m3": 225425.0,
        "predicted_additive_single_delta_tfv_m3": -227655.109375,
        "predicted_interaction_delta_tfv_m3": 141475.90625,
        "predicted_final_delta_tfv_m3": -86179.203125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 30497.765625,
        "predicted_edge_off_direct_delta_tfv_m3": -116676.96875
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 5,
        "true_delta_tfv_m3": 1672.0,
        "predicted_additive_single_delta_tfv_m3": -68950.46875,
        "predicted_interaction_delta_tfv_m3": 40996.953125,
        "predicted_final_delta_tfv_m3": -27953.515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 32451.283203125,
        "predicted_edge_off_direct_delta_tfv_m3": -60404.796875
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 6,
        "true_delta_tfv_m3": -114162.5,
        "predicted_additive_single_delta_tfv_m3": -42069.38671875,
        "predicted_interaction_delta_tfv_m3": 1255.2900390625,
        "predicted_final_delta_tfv_m3": -40814.09765625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 30735.9970703125,
        "predicted_edge_off_direct_delta_tfv_m3": -71550.09375
      },
      {
        "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
        "source_kind": "D3",
        "candidate_index": 7,
        "true_delta_tfv_m3": -146690.0,
        "predicted_additive_single_delta_tfv_m3": -70600.7109375,
        "predicted_interaction_delta_tfv_m3": 17264.681640625,
        "predicted_final_delta_tfv_m3": -53336.03125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 33760.822265625,
        "predicted_edge_off_direct_delta_tfv_m3": -87096.8515625
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 0,
        "true_delta_tfv_m3": -133446.25,
        "predicted_additive_single_delta_tfv_m3": -166211.75,
        "predicted_interaction_delta_tfv_m3": 105029.390625,
        "predicted_final_delta_tfv_m3": -61182.359375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 29994.8671875,
        "predicted_edge_off_direct_delta_tfv_m3": -91177.2265625
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 1,
        "true_delta_tfv_m3": -67508.25,
        "predicted_additive_single_delta_tfv_m3": 25293.10546875,
        "predicted_interaction_delta_tfv_m3": -3096.5810546875,
        "predicted_final_delta_tfv_m3": 22196.5234375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 32173.0009765625,
        "predicted_edge_off_direct_delta_tfv_m3": -9976.4765625
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 2,
        "true_delta_tfv_m3": -79917.25,
        "predicted_additive_single_delta_tfv_m3": -25492.005859375,
        "predicted_interaction_delta_tfv_m3": 36119.7421875,
        "predicted_final_delta_tfv_m3": 10627.736328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 35079.50390625,
        "predicted_edge_off_direct_delta_tfv_m3": -24451.767578125
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 3,
        "true_delta_tfv_m3": -40815.25,
        "predicted_additive_single_delta_tfv_m3": 32949.97265625,
        "predicted_interaction_delta_tfv_m3": 7976.9814453125,
        "predicted_final_delta_tfv_m3": 40926.953125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 32964.9287109375,
        "predicted_edge_off_direct_delta_tfv_m3": 7962.025390625
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 4,
        "true_delta_tfv_m3": -125242.75,
        "predicted_additive_single_delta_tfv_m3": -256236.921875,
        "predicted_interaction_delta_tfv_m3": 193180.09375,
        "predicted_final_delta_tfv_m3": -63056.828125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 33568.375,
        "predicted_edge_off_direct_delta_tfv_m3": -96625.203125
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 5,
        "true_delta_tfv_m3": -146357.25,
        "predicted_additive_single_delta_tfv_m3": -131232.890625,
        "predicted_interaction_delta_tfv_m3": 85518.1796875,
        "predicted_final_delta_tfv_m3": -45714.7109375,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 32479.1328125,
        "predicted_edge_off_direct_delta_tfv_m3": -78193.84375
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 6,
        "true_delta_tfv_m3": -66564.75,
        "predicted_additive_single_delta_tfv_m3": -35931.96484375,
        "predicted_interaction_delta_tfv_m3": 65782.03125,
        "predicted_final_delta_tfv_m3": 29850.06640625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 33346.392578125,
        "predicted_edge_off_direct_delta_tfv_m3": -3496.326171875
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
        "source_kind": "D3",
        "candidate_index": 7,
        "true_delta_tfv_m3": -62088.5,
        "predicted_additive_single_delta_tfv_m3": 11264.556640625,
        "predicted_interaction_delta_tfv_m3": 7013.771484375,
        "predicted_final_delta_tfv_m3": 18278.328125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 33949.748046875,
        "predicted_edge_off_direct_delta_tfv_m3": -15671.419921875
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 0,
        "true_delta_tfv_m3": -6005.0,
        "predicted_additive_single_delta_tfv_m3": -124322.2265625,
        "predicted_interaction_delta_tfv_m3": 44108.71484375,
        "predicted_final_delta_tfv_m3": -80213.515625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 31702.515625,
        "predicted_edge_off_direct_delta_tfv_m3": -111916.03125
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 1,
        "true_delta_tfv_m3": -63585.75,
        "predicted_additive_single_delta_tfv_m3": 30372.310546875,
        "predicted_interaction_delta_tfv_m3": -1418.806640625,
        "predicted_final_delta_tfv_m3": 28953.50390625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 33150.115234375,
        "predicted_edge_off_direct_delta_tfv_m3": -4196.611328125
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 2,
        "true_delta_tfv_m3": -85455.0,
        "predicted_additive_single_delta_tfv_m3": -235868.96875,
        "predicted_interaction_delta_tfv_m3": 176374.46875,
        "predicted_final_delta_tfv_m3": -59494.5,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 30542.96875,
        "predicted_edge_off_direct_delta_tfv_m3": -90037.46875
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 3,
        "true_delta_tfv_m3": -18681.75,
        "predicted_additive_single_delta_tfv_m3": -135146.53125,
        "predicted_interaction_delta_tfv_m3": 79369.46875,
        "predicted_final_delta_tfv_m3": -55777.0625,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 31050.62890625,
        "predicted_edge_off_direct_delta_tfv_m3": -86827.6875
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 4,
        "true_delta_tfv_m3": -72231.5,
        "predicted_additive_single_delta_tfv_m3": 58447.453125,
        "predicted_interaction_delta_tfv_m3": -95054.953125,
        "predicted_final_delta_tfv_m3": -36607.5,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 27745.65625,
        "predicted_edge_off_direct_delta_tfv_m3": -64353.15625
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 5,
        "true_delta_tfv_m3": 71736.75,
        "predicted_additive_single_delta_tfv_m3": -136605.34375,
        "predicted_interaction_delta_tfv_m3": 150703.296875,
        "predicted_final_delta_tfv_m3": 14097.953125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 31445.9765625,
        "predicted_edge_off_direct_delta_tfv_m3": -17348.0234375
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 6,
        "true_delta_tfv_m3": 44440.5,
        "predicted_additive_single_delta_tfv_m3": -125109.6953125,
        "predicted_interaction_delta_tfv_m3": 99280.1171875,
        "predicted_final_delta_tfv_m3": -25829.578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 34532.08984375,
        "predicted_edge_off_direct_delta_tfv_m3": -60361.66796875
      },
      {
        "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
        "source_kind": "D3",
        "candidate_index": 7,
        "true_delta_tfv_m3": -234637.25,
        "predicted_additive_single_delta_tfv_m3": 1942.7216796875,
        "predicted_interaction_delta_tfv_m3": 11937.0361328125,
        "predicted_final_delta_tfv_m3": 13879.7578125,
        "predicted_edge_hydraulic_residual_delta_tfv_m3": 32386.1494140625,
        "predicted_edge_off_direct_delta_tfv_m3": -18506.390625
      }
    ],
    "edge_train_profile": {
      "forward_seconds": 9.297560199978761,
      "backward_seconds": 5.5257914998801425,
      "optimizer_seconds": 0.0951807001256384,
      "wall_time_seconds": 24.458466099982616,
      "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
      "gpu_peak_memory_allocated_bytes": 4125114880,
      "gpu_peak_memory_reserved_bytes": 6689914880,
      "gpu_utilization_mean_percent": 64.76923076923077,
      "gpu_utilization_p90_percent": 76.8,
      "gpu_utilization_max_percent": 77.0,
      "gpu_memory_used_mean_mib": 6540.0,
      "gpu_memory_used_p90_mib": 6540.0,
      "gpu_memory_used_max_mib": 6540.0,
      "gpu_telemetry_samples": 13
    },
    "d2_invariance": {
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
    }
  },
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
  }
}
```
