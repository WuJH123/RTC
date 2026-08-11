# STEP2 PHASE LOSS AUDIT V4.3.1

```json
{
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
  "initialization": {
    "immutable_parent": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_balance_v421\\03_tiny_combined\\v42_tiny_combined.pt",
    "tiny_parent": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_balance_v421\\03_tiny_combined\\v42_tiny_combined.pt",
    "micro_parent": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_balance_v421\\03_tiny_combined\\v42_tiny_combined.pt",
    "same_parent": true,
    "micro_loaded_tiny_checkpoint": false,
    "tiny_groups_sha256": "afdf78f48fa12ccb7ded10bba067836c9f00d448a25baedf973f74830e8c06a0",
    "micro_groups_sha256": "6d4e916f08c78e0dc62af4787ea042667551be484da2f16fc669eb1a710529e3",
    "micro_groups_sha_matches_prior": true
  }
}
```
