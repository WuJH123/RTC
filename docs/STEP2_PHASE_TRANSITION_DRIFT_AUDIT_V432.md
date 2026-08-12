# STEP2 PHASE TRANSITION DRIFT AUDIT V4.3.2

```json
{
  "contract": "STEP2_PHASE_TRANSITION_DRIFT_AUDIT_V432",
  "parent_checkpoint": "E:\\RTC_sewer\\Project7\\study_v069\\step2_d3_magnitude_balance_v421\\03_tiny_combined\\v42_tiny_combined.pt",
  "parent_load": {
    "missing": [
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
      "topology_hidden_head.bias",
      "topology_nodewise_tfv_head.0.weight",
      "topology_nodewise_tfv_head.0.bias",
      "topology_nodewise_tfv_head.2.weight",
      "topology_nodewise_tfv_head.2.bias",
      "topology_nodewise_tfv_head.4.weight",
      "topology_nodewise_tfv_head.4.bias"
    ],
    "unexpected": [],
    "contract": "STEP2_RESPONSE_CALIBRATION_V41_TRAIN_ONLY_DIAGNOSTIC"
  },
  "selected_d2_groups": [
    "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
    "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
    "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
    "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
    "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
    "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200"
  ],
  "prediction_digest_shift": {
    "A0_to_A1": {
      "changed_tensors": 24,
      "total_tensors": 24
    },
    "A1_to_A2": {
      "changed_tensors": 24,
      "total_tensors": 24
    },
    "A2_to_A3": {
      "changed_tensors": 0,
      "total_tensors": 24
    }
  },
  "reference_sha_a0": "641dc5966e3967a1ed877733b4a7854479a84e4b7f39033a57bbe8f77a70bfa4",
  "reference_sha_a3": "61477224bcd7d7d626a681335eb47c46f62fd2dd27dccd9e1f9cba9e62e56761",
  "single_sha_a0": "be682db22f889ca49f5ae02c28fe6654b04c63d90b1a99659e434da9dba9b82a",
  "single_sha_a1": "be682db22f889ca49f5ae02c28fe6654b04c63d90b1a99659e434da9dba9b82a",
  "reference_representation_drift_confirmed": true,
  "first_degradation_stage": "REFERENCE",
  "stages": {
    "A0": {
      "metrics": {
        "groups": 6,
        "spread_ratio": 0.9664526610537157,
        "rank": 0.22019725926128061,
        "pairwise": 0.6031154370909206,
        "sign": 0.5510540184453228,
        "top1": 2,
        "mean_regret_m3": 7613.0,
        "max_regret_m3": 20020.5
      },
      "prediction_spreads": {
        "delta_state": 0.12739873925844827,
        "delta_flow": 0.036908903217408806,
        "direct_tfv": 23204.876139322918,
        "trajectory_tfv": 14972.942220052084
      },
      "reference_parameter_sha256": "641dc5966e3967a1ed877733b4a7854479a84e4b7f39033a57bbe8f77a70bfa4",
      "single_parameter_sha256": "be682db22f889ca49f5ae02c28fe6654b04c63d90b1a99659e434da9dba9b82a",
      "prediction_digests": {
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300": {
          "delta_states": "1072964a413ffe5b8d5aae12bb387b51ea86e7ce3e24e0782f5f9441f31222ba",
          "delta_flows": "bfee4005fd69e8946df7201718ef2f2e0b093989f0c8cdc52aab0e97f567f966",
          "direct_tfv": "76519c9aa88ecfd2a4244e572225ecf7397bc96d45c06412f4533ef7cfc58073",
          "trajectory_tfv": "3fb7547b2527073684a4d04e2d0acf619dab33d512ebcb50c65eb7c790577b6c"
        },
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000": {
          "delta_states": "a59898ee8c06a6b6179e18492ba6941009fcfbb99b941ed811d4f1c4d1d3c582",
          "delta_flows": "439ed37f2ce46bba28bf253ea5d1c2e31e1d6308247e6662ed9c92a7735b35cd",
          "direct_tfv": "7bb6918ce1b7bd71a56e3abb993424a17dfa386b447a7c0bfcc7e53eab117375",
          "trajectory_tfv": "42ab2e7cad3303916c7789ab2818f2e35ec999df1c0ce758c95b2df22428ef77"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600": {
          "delta_states": "6cd2b4586845969c7556a57e63607a1956c33290214a463cec324ed87db8c4a9",
          "delta_flows": "ae8c4d869cd9f10a7656d02fd4b9c210e40784f8fe2f3ec08b9314fbf0b3c94e",
          "direct_tfv": "9a176c55582263514c3793e8efae826fc1a7d68e71547f84a1507b54aa335a13",
          "trajectory_tfv": "c79676bd5efedc4ec4d937fda1cd5246ffcea6280acaec0c5a84e1a0682b23b5"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700": {
          "delta_states": "373b0be4107b8e1a63b0b1ee272432f27dfb453f41337f411f26d1281adda128",
          "delta_flows": "4475f9db5f37d321a09080a8febf7ebd1d2bfb2bc8a5b498fe6961008676800b",
          "direct_tfv": "9e5ee19f2c6d19e53168fffd255e95b09adaa750eaf9bd79064f9c59148cce42",
          "trajectory_tfv": "63ac0d55b221b9c40a2a1a0ca8e20cf84aad60ff9500b5197fd3627a296313eb"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500": {
          "delta_states": "1c806d8944c8080988101b3b9469017220e3234db136a57dfc29ceeff8fec47e",
          "delta_flows": "71f7418ac70195599f91f15cfbb9e2c3b315d5591a0875ea559dc484dad62808",
          "direct_tfv": "b08db43c96b4051b11af781bd9f26a0ae57ea69964ab232b03b5fb0c1c43a0f3",
          "trajectory_tfv": "c92a306b35ee60b052db963aca1a2f31da67b63a72375d508cb10f803eb80107"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200": {
          "delta_states": "b9fb97df9450e412ed293ddb9a96400be8196952fbccb8139d3d932207f3f3db",
          "delta_flows": "aac37415e350ce5e8aaf2a3caf3f9889b3db82bee14248edd1c5a8a508baa49f",
          "direct_tfv": "7ffa21bc0223cd0dde901de204c3ae27f678b3de65ddb988d6492989361e47ad",
          "trajectory_tfv": "4b11d88ff04f2614c05c132562851832f7a6b9e4054e3e1efb786fdce8e300ff"
        }
      }
    },
    "A1": {
      "metrics": {
        "groups": 6,
        "spread_ratio": 1.2148219972599839,
        "rank": 0.170751019310895,
        "pairwise": 0.568042067603752,
        "sign": 0.5662329819938515,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "prediction_spreads": {
        "delta_state": 0.1544371172785759,
        "delta_flow": 0.07156994422742476,
        "direct_tfv": 28182.937174479168,
        "trajectory_tfv": 8960.510538736979
      },
      "reference_parameter_sha256": "61477224bcd7d7d626a681335eb47c46f62fd2dd27dccd9e1f9cba9e62e56761",
      "single_parameter_sha256": "be682db22f889ca49f5ae02c28fe6654b04c63d90b1a99659e434da9dba9b82a",
      "prediction_digests": {
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300": {
          "delta_states": "b6698577d9992db4bac5e324849adf9676471262b7c58d6fb38796d1b735712c",
          "delta_flows": "f1f42a08de1d57fb80826baadcb7a21c6a2f92af1f20b92d43702adbc9c8a983",
          "direct_tfv": "f1071a9895d732c02e768bfb24db4fd2459d486a495648b9ba952d5d6be6cd20",
          "trajectory_tfv": "1bbf0ac073d4560f64dfc338f35d6ad06a4e8fe79dc862aa7e71cf2ad1f4a894"
        },
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000": {
          "delta_states": "f9d3b5f52d88282938b8caced7156082b81a252e31d9c2e9d0f41c2acf0ad9c8",
          "delta_flows": "c2c44b33b766a322efda0c3f6f8f4fa23dd0d8b3d7475856c99c9df580731545",
          "direct_tfv": "0a06b424e5c7b584ad775eac2f23ea0f7efff914042e3c0faaaed9540849a9fc",
          "trajectory_tfv": "41b73c3ba99e0ae92e0ea4450a48d802a8bf7c26ebf29f80016c2c3db152373a"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600": {
          "delta_states": "040fcd5b4e36627bab83bb0fd7d6cc214c9ea01f5fd6babc91554661abea3d0a",
          "delta_flows": "626e0c25fbb37b8c9f8da567a3b20d0629301cc81251c1009facbb8c5328c483",
          "direct_tfv": "a12fc15b5d22f8172eb1804eeb7f64c2e5742fd4c8d3c06516b646400a6d6f39",
          "trajectory_tfv": "b61fc1462d674212008d8acf6d4e9caa9bc5f56fa068385b02e1120b14681326"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700": {
          "delta_states": "791a60ff48138b307dbb7a9503161daee822d75a9bb38d70461ae10f0cb2147f",
          "delta_flows": "20d67bca896d252f37f3923d640bcd0e11aff81f8b5af418e606fcd835750f39",
          "direct_tfv": "c2322917fc3163e88cc851e8f15ab27d4372e90ee8f78d7aaa9b79a7cf36c3b6",
          "trajectory_tfv": "b4d0647065732f6a1224cc3aa0b67bace3338d65676fb82981d8bfd5c12241ca"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500": {
          "delta_states": "80afbc434ab4868074de8adeca9fd6b9174313cace41755c09a63c956ce6c6f1",
          "delta_flows": "048dce6408d2e41dbc16f63e9708efd9d7360c43bc17c3cb391e7db5fc287d78",
          "direct_tfv": "95cf4b807ff743c6eeab4f9251da1f2d54fd54ee3a6463ca7b1d79ba2570c194",
          "trajectory_tfv": "a2420b5570774b97d4ef84842cc2fa0351af458c389ec4c8486dc9e8805d489e"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200": {
          "delta_states": "47d7c715a3f42e4525b9583382d697bb012ea8d307a262eaaa947916fc1b9955",
          "delta_flows": "8ecbdd4766467e1b642d26494f9a393db999306e4aa9ea948fb7b1a2c8d42513",
          "direct_tfv": "617c11b5b0e06977a6653ebd1340a3c806e64294d122cb3cd286d5f72331281a",
          "trajectory_tfv": "305919f09a14372fb94ff23f28a99f62d39114ee0aa779153c7f89e9247222d0"
        }
      }
    },
    "A2": {
      "metrics": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "prediction_spreads": {
        "delta_state": 0.03639760489265124,
        "delta_flow": 0.02080303561524488,
        "direct_tfv": 8760.681722005209,
        "trajectory_tfv": 7282.3996175130205
      },
      "reference_parameter_sha256": "61477224bcd7d7d626a681335eb47c46f62fd2dd27dccd9e1f9cba9e62e56761",
      "single_parameter_sha256": "f2765760337240a17236523cb0c7b49f76a06b53a2ce8ede3fe27f4d5ab69018",
      "prediction_digests": {
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300": {
          "delta_states": "9b6b962565a1af1154dc23104369bf18f3dd6a355afca1b4f9de2375f05dacd2",
          "delta_flows": "033d70ffe0af0b69b600b7d6005c895e3707e2bc87079b8aa50e2cc728e95751",
          "direct_tfv": "1fc23b8ec4273fe31783038258b9b4e7e2f0cf6293995dafcaa8f86dfaf2bda8",
          "trajectory_tfv": "aec4561492790d56e32bcc218996cbf031e086bb799febabf3de6923ce76fe55"
        },
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000": {
          "delta_states": "68c9ca4b46f12df76ab2a3cc8e69367223a2df686bd1c819ef313810e3e5978c",
          "delta_flows": "aea4af89cc687ccd490a15a528b43b6060282a9fbbc616e86378011b0d2837f1",
          "direct_tfv": "936339bc0d56195d970599740c43ab45ad8e30b50b81691cd5b1940dc360b7f7",
          "trajectory_tfv": "cd12813498746c2cbd23cbecd6ac59f3b175a9806236297284bb44d73e4f54af"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600": {
          "delta_states": "c3be966a155ecd67d640b05ba334c2d2483989b9030222b73b43d2145dde04f6",
          "delta_flows": "eca9130791e2f18d8aba3e1ef13d943bc65120f3b965c9b45cd8782a6504c25e",
          "direct_tfv": "47feafc74080237ae8360349d15a99d726431b692f71b287313445d2713c2e7f",
          "trajectory_tfv": "177aeb51544a2a46587df1d0e6f21067159533be9709938f5e16f7aedb758d04"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700": {
          "delta_states": "99a0e60898c85d033798bf396d50aa08010c4151c5e3b5754eeb18246c41d311",
          "delta_flows": "e91b2e2924cf938d9af9d5d031605664a2551bb44a6f3e708935b5f3d043bd4b",
          "direct_tfv": "eefe4c33f88c5d91310699b0baa46b34c3c6134eb870895b8b3f101ad309c212",
          "trajectory_tfv": "6b8f61a598c4279e098b22359ddec6f264cb760333bf393f9493792d062a60c6"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500": {
          "delta_states": "1f4e97304e79d2021e833461564f48aba4bca0707ec9ae1946216a9cae871716",
          "delta_flows": "4fbafe276ea908ea0312db16c798862199eb23eff050681cb82110d80857a685",
          "direct_tfv": "dc530c9edcb3f2a57330fc250de8661b6a3f43152475477ba94c813243d711ca",
          "trajectory_tfv": "b55e4762b6378435a8601a1a2b311de394b3d03ce7e354b4deacfae6072c30b2"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200": {
          "delta_states": "5b6cb8b7a0726cc077a91be005b29e4b9c4593dbd3a0d47002e365fd804ad5c4",
          "delta_flows": "3a190acb8d18ad2c2c23b341ca354d549ae367bb03bd3dbbd3b5483334b2c7b5",
          "direct_tfv": "7053d526b94eb061bbc1fe666afab178969c139be02665c97f813037b78815e2",
          "trajectory_tfv": "d89a7441868524a3bfc8e8bb770ffb31476705299acaab6c06d497b1cd52e445"
        }
      }
    },
    "A3": {
      "metrics": {
        "groups": 6,
        "spread_ratio": 0.37636896861570995,
        "rank": 0.2360042296942441,
        "pairwise": 0.5902356389251645,
        "sign": 0.581357048748353,
        "top1": 1,
        "mean_regret_m3": 4561.25,
        "max_regret_m3": 20020.5
      },
      "prediction_spreads": {
        "delta_state": 0.03639760489265124,
        "delta_flow": 0.02080303561524488,
        "direct_tfv": 8760.681722005209,
        "trajectory_tfv": 7282.3996175130205
      },
      "reference_parameter_sha256": "61477224bcd7d7d626a681335eb47c46f62fd2dd27dccd9e1f9cba9e62e56761",
      "single_parameter_sha256": "f2765760337240a17236523cb0c7b49f76a06b53a2ce8ede3fe27f4d5ab69018",
      "prediction_digests": {
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300": {
          "delta_states": "9b6b962565a1af1154dc23104369bf18f3dd6a355afca1b4f9de2375f05dacd2",
          "delta_flows": "033d70ffe0af0b69b600b7d6005c895e3707e2bc87079b8aa50e2cc728e95751",
          "direct_tfv": "1fc23b8ec4273fe31783038258b9b4e7e2f0cf6293995dafcaa8f86dfaf2bda8",
          "trajectory_tfv": "aec4561492790d56e32bcc218996cbf031e086bb799febabf3de6923ce76fe55"
        },
        "D2::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000": {
          "delta_states": "68c9ca4b46f12df76ab2a3cc8e69367223a2df686bd1c819ef313810e3e5978c",
          "delta_flows": "aea4af89cc687ccd490a15a528b43b6060282a9fbbc616e86378011b0d2837f1",
          "direct_tfv": "936339bc0d56195d970599740c43ab45ad8e30b50b81691cd5b1940dc360b7f7",
          "trajectory_tfv": "cd12813498746c2cbd23cbecd6ac59f3b175a9806236297284bb44d73e4f54af"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600": {
          "delta_states": "c3be966a155ecd67d640b05ba334c2d2483989b9030222b73b43d2145dde04f6",
          "delta_flows": "eca9130791e2f18d8aba3e1ef13d943bc65120f3b965c9b45cd8782a6504c25e",
          "direct_tfv": "47feafc74080237ae8360349d15a99d726431b692f71b287313445d2713c2e7f",
          "trajectory_tfv": "177aeb51544a2a46587df1d0e6f21067159533be9709938f5e16f7aedb758d04"
        },
        "D2::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700": {
          "delta_states": "99a0e60898c85d033798bf396d50aa08010c4151c5e3b5754eeb18246c41d311",
          "delta_flows": "e91b2e2924cf938d9af9d5d031605664a2551bb44a6f3e708935b5f3d043bd4b",
          "direct_tfv": "eefe4c33f88c5d91310699b0baa46b34c3c6134eb870895b8b3f101ad309c212",
          "trajectory_tfv": "6b8f61a598c4279e098b22359ddec6f264cb760333bf393f9493792d062a60c6"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500": {
          "delta_states": "1f4e97304e79d2021e833461564f48aba4bca0707ec9ae1946216a9cae871716",
          "delta_flows": "4fbafe276ea908ea0312db16c798862199eb23eff050681cb82110d80857a685",
          "direct_tfv": "dc530c9edcb3f2a57330fc250de8661b6a3f43152475477ba94c813243d711ca",
          "trajectory_tfv": "b55e4762b6378435a8601a1a2b311de394b3d03ce7e354b4deacfae6072c30b2"
        },
        "D2::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200": {
          "delta_states": "5b6cb8b7a0726cc077a91be005b29e4b9c4593dbd3a0d47002e365fd804ad5c4",
          "delta_flows": "3a190acb8d18ad2c2c23b341ca354d549ae367bb03bd3dbbd3b5483334b2c7b5",
          "direct_tfv": "7053d526b94eb061bbc1fe666afab178969c139be02665c97f813037b78815e2",
          "trajectory_tfv": "d89a7441868524a3bfc8e8bb770ffb31476705299acaab6c06d497b1cd52e445"
        }
      }
    }
  }
}
```
