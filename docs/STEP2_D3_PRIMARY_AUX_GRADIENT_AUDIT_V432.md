# STEP2 D3 PRIMARY AUX GRADIENT AUDIT V4.3.2

```json
{
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
    "topology_hidden_head.bias",
    "topology_nodewise_tfv_head.0.weight",
    "topology_nodewise_tfv_head.0.bias",
    "topology_nodewise_tfv_head.2.weight",
    "topology_nodewise_tfv_head.2.bias",
    "topology_nodewise_tfv_head.4.weight",
    "topology_nodewise_tfv_head.4.bias"
  ],
  "primary_components": [
    "direct_TFV",
    "ranking"
  ],
  "auxiliary_components": [
    "delta_state",
    "delta_flow",
    "centered_TFV",
    "trajectory_TFV",
    "magnitude_calibration",
    "consistency",
    "interaction_energy"
  ],
  "fixed_loss_weights": {
    "reference_state": 0.05,
    "reference_flow": 0.05,
    "delta_state": 1.0,
    "delta_flow": 1.0,
    "direct_tfv": 2.0,
    "centered_tfv": 5.0,
    "trajectory_tfv": 1.0,
    "consistency": 1.0,
    "ranking": 5.0,
    "interaction_energy": 0.01,
    "magnitude_calibration": 1.0
  },
  "summary": {
    "groups": 6,
    "components": {
      "delta_state": {
        "gradient_l2_mean": 0.014452192777146896,
        "gradient_linf_mean": 0.0048618760192766786,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": -0.1255329872171084,
        "cosine_vs_ranking_mean": -0.01983377756550908,
        "cosine_vs_primary_mean": -0.03373969718813896,
        "fraction_cosine_vs_primary_negative": 0.6666666666666666
      },
      "delta_flow": {
        "gradient_l2_mean": 0.11075826485951741,
        "gradient_linf_mean": 0.047824383713304996,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": 0.0002560133192067345,
        "cosine_vs_ranking_mean": -0.004192207483962799,
        "cosine_vs_primary_mean": -0.0035541512382527194,
        "fraction_cosine_vs_primary_negative": 0.6666666666666666
      },
      "direct_TFV": {
        "gradient_l2_mean": 0.6686545349657536,
        "gradient_linf_mean": 0.07504571570704381,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": 1.0000000298023224,
        "cosine_vs_ranking_mean": 0.0936035265525182,
        "cosine_vs_primary_mean": 0.7734513557516038,
        "fraction_cosine_vs_primary_negative": 0.0
      },
      "centered_TFV": {
        "gradient_l2_mean": 0.044842307145396866,
        "gradient_linf_mean": 0.005661199034269278,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": -0.025214578956365585,
        "cosine_vs_ranking_mean": 0.6863112337887287,
        "cosine_vs_primary_mean": 0.15374940012892088,
        "fraction_cosine_vs_primary_negative": 0.3333333333333333
      },
      "trajectory_TFV": {
        "gradient_l2_mean": 0.6628562659025192,
        "gradient_linf_mean": 0.35650088389714557,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": 0.03898464826246103,
        "cosine_vs_ranking_mean": -0.023429481623073418,
        "cosine_vs_primary_mean": 0.12919624925901493,
        "fraction_cosine_vs_primary_negative": 0.3333333333333333
      },
      "ranking": {
        "gradient_l2_mean": 0.1739366032804052,
        "gradient_linf_mean": 0.02029368991497904,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": 0.0936035265525182,
        "cosine_vs_ranking_mean": 0.9999999900658926,
        "cosine_vs_primary_mean": 0.45598142345746356,
        "fraction_cosine_vs_primary_negative": 0.3333333333333333
      },
      "magnitude_calibration": {
        "gradient_l2_mean": 0.12527946134408316,
        "gradient_linf_mean": 0.013268167696272334,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": 0.5948724548021952,
        "cosine_vs_ranking_mean": 0.02220040000975132,
        "cosine_vs_primary_mean": 0.3523353996376197,
        "fraction_cosine_vs_primary_negative": 0.16666666666666666
      },
      "consistency": {
        "gradient_l2_mean": 0.84542948504289,
        "gradient_linf_mean": 0.3393876999616623,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": 0.4269276646276315,
        "cosine_vs_ranking_mean": 0.0629394951586922,
        "cosine_vs_primary_mean": 0.3376465483258168,
        "fraction_cosine_vs_primary_negative": 0.16666666666666666
      },
      "interaction_energy": {
        "gradient_l2_mean": 1.8180519143740337,
        "gradient_linf_mean": 0.8911745945612589,
        "finite_fraction_min": 1.0,
        "cosine_vs_direct_TFV_mean": 0.0036795036867260933,
        "cosine_vs_ranking_mean": -0.002456031118830045,
        "cosine_vs_primary_mean": -0.0018279789946973324,
        "fraction_cosine_vs_primary_negative": 0.5
      }
    }
  },
  "groups": [
    {
      "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t12300",
      "primary_gradient_l2": 1.0299588441848755,
      "primary_gradient_linf": 0.08473272621631622,
      "components": {
        "delta_state": {
          "gradient_l2": 0.012900445610284805,
          "gradient_linf": 0.0025341645814478397,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.42112404108047485,
          "cosine_vs_ranking": 0.169836163520813,
          "cosine_vs_primary": 0.08527299016714096
        },
        "delta_flow": {
          "gradient_l2": 0.05737772211432457,
          "gradient_linf": 0.027204135432839394,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.010645706206560135,
          "cosine_vs_ranking": 0.0007518717902712524,
          "cosine_vs_primary": 0.002992394845932722
        },
        "direct_TFV": {
          "gradient_l2": 0.10764255374670029,
          "gradient_linf": 0.017700299620628357,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 1.0,
          "cosine_vs_ranking": -0.19895729422569275,
          "cosine_vs_primary": 0.006010756827890873
        },
        "centered_TFV": {
          "gradient_l2": 0.08019006997346878,
          "gradient_linf": 0.009382646530866623,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.43547090888023376,
          "cosine_vs_ranking": 0.09544675797224045,
          "cosine_vs_primary": 0.006368644535541534
        },
        "trajectory_TFV": {
          "gradient_l2": 0.2688194215297699,
          "gradient_linf": 0.1485508680343628,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.320615291595459,
          "cosine_vs_ranking": 0.17720916867256165,
          "cosine_vs_primary": 0.11380494385957718
        },
        "ranking": {
          "gradient_l2": 0.21019013226032257,
          "gradient_linf": 0.01918545924127102,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.19895729422569275,
          "cosine_vs_ranking": 1.0,
          "cosine_vs_primary": 0.9787945747375488
        },
        "magnitude_calibration": {
          "gradient_l2": 0.047286178916692734,
          "gradient_linf": 0.0072041708044707775,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.9937839508056641,
          "cosine_vs_ranking": -0.1529202163219452,
          "cosine_vs_primary": 0.05168683081865311
        },
        "consistency": {
          "gradient_l2": 0.37207692861557007,
          "gradient_linf": 0.2149902880191803,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.2291732132434845,
          "cosine_vs_ranking": -0.015569997951388359,
          "cosine_vs_primary": 0.03201514109969139
        },
        "interaction_energy": {
          "gradient_l2": 1.7771793603897095,
          "gradient_linf": 0.8819583654403687,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.01209968701004982,
          "cosine_vs_ranking": -0.016651397570967674,
          "cosine_vs_primary": -0.014461660757660866
        }
      },
      "auxiliary_conflicts_vs_primary": {
        "delta_state": false,
        "delta_flow": false,
        "centered_TFV": false,
        "trajectory_TFV": false,
        "magnitude_calibration": false,
        "consistency": false,
        "interaction_energy": true
      }
    },
    {
      "group": "D3::T100_D180_chicago::T100_D180_chicago::T100_D180_chicago:t6000",
      "primary_gradient_l2": 2.5982542037963867,
      "primary_gradient_linf": 0.2915884256362915,
      "components": {
        "delta_state": {
          "gradient_l2": 0.011511299759149551,
          "gradient_linf": 0.0028962416108697653,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.16494539380073547,
          "cosine_vs_ranking": 0.009163285605609417,
          "cosine_vs_primary": -0.16700252890586853
        },
        "delta_flow": {
          "gradient_l2": 0.09046275913715363,
          "gradient_linf": 0.03237241506576538,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.003125592367723584,
          "cosine_vs_ranking": 0.019910426810383797,
          "cosine_vs_primary": -0.002373178955167532
        },
        "direct_TFV": {
          "gradient_l2": 1.318222999572754,
          "gradient_linf": 0.14969423413276672,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.9999999403953552,
          "cosine_vs_ranking": -0.3836860656738281,
          "cosine_vs_primary": 0.9993141293525696
        },
        "centered_TFV": {
          "gradient_l2": 0.005493618547916412,
          "gradient_linf": 0.0005579549469985068,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.40037888288497925,
          "cosine_vs_ranking": 0.998291552066803,
          "cosine_vs_primary": -0.36623528599739075
        },
        "trajectory_TFV": {
          "gradient_l2": 0.6370114684104919,
          "gradient_linf": 0.373815655708313,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.16812710464000702,
          "cosine_vs_ranking": -0.061249762773513794,
          "cosine_vs_primary": 0.16814246773719788
        },
        "ranking": {
          "gradient_l2": 0.020836586132645607,
          "gradient_linf": 0.0021389031317085028,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.3836860656738281,
          "cosine_vs_ranking": 1.0,
          "cosine_vs_primary": -0.34922856092453003
        },
        "magnitude_calibration": {
          "gradient_l2": 0.15592466294765472,
          "gradient_linf": 0.013233358040452003,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.9252794981002808,
          "cosine_vs_ranking": 0.05082785710692406,
          "cosine_vs_primary": -0.9368420839309692
        },
        "consistency": {
          "gradient_l2": 0.4560125768184662,
          "gradient_linf": 0.17675486207008362,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.7398111820220947,
          "cosine_vs_ranking": -0.3179492652416229,
          "cosine_vs_primary": 0.7379367351531982
        },
        "interaction_energy": {
          "gradient_l2": 1.8853551149368286,
          "gradient_linf": 0.8997670412063599,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.006361405365169048,
          "cosine_vs_ranking": -0.017632290720939636,
          "cosine_vs_primary": 0.005747906863689423
        }
      },
      "auxiliary_conflicts_vs_primary": {
        "delta_state": true,
        "delta_flow": true,
        "centered_TFV": true,
        "trajectory_TFV": false,
        "magnitude_calibration": true,
        "consistency": false,
        "interaction_energy": false
      }
    },
    {
      "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t18600",
      "primary_gradient_l2": 1.6040654182434082,
      "primary_gradient_linf": 0.16530939936637878,
      "components": {
        "delta_state": {
          "gradient_l2": 0.02973135933279991,
          "gradient_linf": 0.011829615570604801,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.26545804738998413,
          "cosine_vs_ranking": -0.13477852940559387,
          "cosine_vs_primary": 0.24919112026691437
        },
        "delta_flow": {
          "gradient_l2": 0.1163090318441391,
          "gradient_linf": 0.057359613478183746,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.008225219324231148,
          "cosine_vs_ranking": -0.0038809170946478844,
          "cosine_vs_primary": 0.007850045338273048
        },
        "direct_TFV": {
          "gradient_l2": 0.9306378960609436,
          "gradient_linf": 0.10275433957576752,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 1.0,
          "cosine_vs_ranking": -0.5300542116165161,
          "cosine_vs_primary": 0.9289724826812744
        },
        "centered_TFV": {
          "gradient_l2": 0.028512312099337578,
          "gradient_linf": 0.002609432674944401,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.43370628356933594,
          "cosine_vs_ranking": 0.9837621450424194,
          "cosine_vs_primary": -0.07382382452487946
        },
        "trajectory_TFV": {
          "gradient_l2": 1.8005826473236084,
          "gradient_linf": 0.8873028755187988,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.5067101716995239,
          "cosine_vs_ranking": -0.295754611492157,
          "cosine_vs_primary": 0.45885932445526123
        },
        "ranking": {
          "gradient_l2": 0.14003969728946686,
          "gradient_linf": 0.014768464490771294,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.5300542116165161,
          "cosine_vs_ranking": 0.9999999403953552,
          "cosine_vs_primary": -0.17853295803070068
        },
        "magnitude_calibration": {
          "gradient_l2": 0.2519325017929077,
          "gradient_linf": 0.0279402956366539,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.9984813928604126,
          "cosine_vs_ranking": -0.5598835945129395,
          "cosine_vs_primary": 0.9141893982887268
        },
        "consistency": {
          "gradient_l2": 0.34556472301483154,
          "gradient_linf": 0.18653163313865662,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.04922232776880264,
          "cosine_vs_ranking": -0.07666938006877899,
          "cosine_vs_primary": -0.09058241546154022
        },
        "interaction_energy": {
          "gradient_l2": 1.7820676565170288,
          "gradient_linf": 0.8658273220062256,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.009318976663053036,
          "cosine_vs_ranking": 0.01250745914876461,
          "cosine_vs_primary": -0.005353573709726334
        }
      },
      "auxiliary_conflicts_vs_primary": {
        "delta_state": false,
        "delta_flow": false,
        "centered_TFV": true,
        "trajectory_TFV": false,
        "magnitude_calibration": false,
        "consistency": true,
        "interaction_energy": true
      }
    },
    {
      "group": "D3::T100_D360_chicago::T100_D360_chicago::T100_D360_chicago:t8700",
      "primary_gradient_l2": 2.3931398391723633,
      "primary_gradient_linf": 0.3041321933269501,
      "components": {
        "delta_state": {
          "gradient_l2": 0.008661558851599693,
          "gradient_linf": 0.0020097168162465096,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.2038383185863495,
          "cosine_vs_ranking": -0.010771814733743668,
          "cosine_vs_primary": -0.19704599678516388
        },
        "delta_flow": {
          "gradient_l2": 0.27537932991981506,
          "gradient_linf": 0.12197385728359222,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.0034009874798357487,
          "cosine_vs_ranking": -0.019461100921034813,
          "cosine_vs_primary": -0.007199565880000591
        },
        "direct_TFV": {
          "gradient_l2": 1.1438686847686768,
          "gradient_linf": 0.12063737213611603,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 1.0,
          "cosine_vs_ranking": 0.11597061157226562,
          "cosine_vs_primary": 0.9794851541519165
        },
        "centered_TFV": {
          "gradient_l2": 0.045562468469142914,
          "gradient_linf": 0.006094041746109724,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.10235073417425156,
          "cosine_vs_ranking": 0.9993551969528198,
          "cosine_vs_primary": 0.30059731006622314
        },
        "trajectory_TFV": {
          "gradient_l2": 0.4072067439556122,
          "gradient_linf": 0.23119303584098816,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.06678084284067154,
          "cosine_vs_ranking": 0.028323018923401833,
          "cosine_vs_primary": 0.0695858970284462
        },
        "ranking": {
          "gradient_l2": 0.09710656851530075,
          "gradient_linf": 0.013306694105267525,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.11597061157226562,
          "cosine_vs_ranking": 1.0,
          "cosine_vs_primary": 0.3137481212615967
        },
        "magnitude_calibration": {
          "gradient_l2": 0.1643553525209427,
          "gradient_linf": 0.015884405001997948,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.5187913775444031,
          "cosine_vs_ranking": -0.7818166613578796,
          "cosine_vs_primary": 0.337322860956192
        },
        "consistency": {
          "gradient_l2": 1.4032642841339111,
          "gradient_linf": 0.5423706769943237,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.7350784540176392,
          "cosine_vs_ranking": 0.26949620246887207,
          "cosine_vs_primary": 0.7573797106742859
        },
        "interaction_energy": {
          "gradient_l2": 1.8580669164657593,
          "gradient_linf": 0.9173097610473633,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.006862102076411247,
          "cosine_vs_ranking": 0.015855319797992706,
          "cosine_vs_primary": 0.009776683524250984
        }
      },
      "auxiliary_conflicts_vs_primary": {
        "delta_state": true,
        "delta_flow": true,
        "centered_TFV": false,
        "trajectory_TFV": false,
        "magnitude_calibration": false,
        "consistency": false,
        "interaction_energy": false
      }
    },
    {
      "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t4500",
      "primary_gradient_l2": 2.362914562225342,
      "primary_gradient_linf": 0.30220094323158264,
      "components": {
        "delta_state": {
          "gradient_l2": 0.015708616003394127,
          "gradient_linf": 0.007734597194939852,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.08639758825302124,
          "cosine_vs_ranking": -0.016600893810391426,
          "cosine_vs_primary": -0.03036649525165558
        },
        "delta_flow": {
          "gradient_l2": 0.07857123762369156,
          "gradient_linf": 0.03634015470743179,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.003248437773436308,
          "cosine_vs_ranking": -0.019833873957395554,
          "cosine_vs_primary": -0.017785528674721718
        },
        "direct_TFV": {
          "gradient_l2": 0.21856266260147095,
          "gradient_linf": 0.02308240719139576,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 1.0000001192092896,
          "cosine_vs_ranking": 0.6709437966346741,
          "cosine_vs_primary": 0.7663173675537109
        },
        "centered_TFV": {
          "gradient_l2": 0.0811886116862297,
          "gradient_linf": 0.010403036139905453,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.2880677282810211,
          "cosine_vs_ranking": 0.24632734060287476,
          "cosine_vs_primary": 0.26671528816223145
        },
        "trajectory_TFV": {
          "gradient_l2": 0.5304205417633057,
          "gradient_linf": 0.30825096368789673,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.13665685057640076,
          "cosine_vs_ranking": 0.018840234726667404,
          "cosine_vs_primary": -0.008957043290138245
        },
        "ranking": {
          "gradient_l2": 0.409458190202713,
          "gradient_linf": 0.051207225769758224,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.6709437966346741,
          "cosine_vs_ranking": 1.0,
          "cosine_vs_primary": 0.9905468225479126
        },
        "magnitude_calibration": {
          "gradient_l2": 0.07906543463468552,
          "gradient_linf": 0.008442110382020473,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.9954361915588379,
          "cosine_vs_ranking": 0.7294896841049194,
          "cosine_vs_primary": 0.8161987662315369
        },
        "consistency": {
          "gradient_l2": 1.0019601583480835,
          "gradient_linf": 0.42496198415756226,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.5575196146965027,
          "cosine_vs_ranking": 0.09440642595291138,
          "cosine_vs_primary": 0.18493403494358063
        },
        "interaction_energy": {
          "gradient_l2": 1.9312258958816528,
          "gradient_linf": 0.9266482591629028,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.00043785199522972107,
          "cosine_vs_ranking": -0.015377368777990341,
          "cosine_vs_primary": -0.013404354453086853
        }
      },
      "auxiliary_conflicts_vs_primary": {
        "delta_state": true,
        "delta_flow": true,
        "centered_TFV": false,
        "trajectory_TFV": true,
        "magnitude_calibration": false,
        "consistency": false,
        "interaction_energy": true
      }
    },
    {
      "group": "D3::T100_D60_chicago::T100_D60_chicago::T100_D60_chicago:t7200",
      "primary_gradient_l2": 1.376710295677185,
      "primary_gradient_linf": 0.17858824133872986,
      "components": {
        "delta_state": {
          "gradient_l2": 0.008199877105653286,
          "gradient_linf": 0.002166920341551304,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.14235062897205353,
          "cosine_vs_ranking": -0.13585087656974792,
          "cosine_vs_primary": -0.1424872726202011
        },
        "delta_flow": {
          "gradient_l2": 0.046449508517980576,
          "gradient_linf": 0.01169612631201744,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.007559827994555235,
          "cosine_vs_ranking": -0.002639651531353593,
          "cosine_vs_primary": -0.004809074103832245
        },
        "direct_TFV": {
          "gradient_l2": 0.29299241304397583,
          "gradient_linf": 0.036405641585588455,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 1.0000001192092896,
          "cosine_vs_ranking": 0.8874043226242065,
          "cosine_vs_primary": 0.9606082439422607
        },
        "centered_TFV": {
          "gradient_l2": 0.02810676209628582,
          "gradient_linf": 0.004920082166790962,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.7278501391410828,
          "cosine_vs_ranking": 0.7946844100952148,
          "cosine_vs_primary": 0.7888742685317993
        },
        "trajectory_TFV": {
          "gradient_l2": 0.33309677243232727,
          "gradient_linf": 0.18989190459251404,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": -0.050438087433576584,
          "cosine_vs_ranking": -0.00794493779540062,
          "cosine_vs_primary": -0.026258094236254692
        },
        "ranking": {
          "gradient_l2": 0.16598844528198242,
          "gradient_linf": 0.02115539275109768,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.8874043226242065,
          "cosine_vs_ranking": 1.0,
          "cosine_vs_primary": 0.9805605411529541
        },
        "magnitude_calibration": {
          "gradient_l2": 0.053112637251615524,
          "gradient_linf": 0.0069046663120388985,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.9880213141441345,
          "cosine_vs_ranking": 0.8475053310394287,
          "cosine_vs_primary": 0.9314566254615784
        },
        "consistency": {
          "gradient_l2": 1.493698239326477,
          "gradient_linf": 0.49071675539016724,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.3492058515548706,
          "cosine_vs_ranking": 0.42392298579216003,
          "cosine_vs_primary": 0.4041960835456848
        },
        "interaction_energy": {
          "gradient_l2": 1.6744165420532227,
          "gradient_linf": 0.8555368185043335,
          "finite_fraction": 1.0,
          "cosine_vs_direct_TFV": 0.006510656327009201,
          "cosine_vs_ranking": 0.006562091410160065,
          "cosine_vs_primary": 0.006727124564349651
        }
      },
      "auxiliary_conflicts_vs_primary": {
        "delta_state": true,
        "delta_flow": true,
        "centered_TFV": false,
        "trajectory_TFV": true,
        "magnitude_calibration": false,
        "consistency": false,
        "interaction_energy": false
      }
    }
  ]
}
```
