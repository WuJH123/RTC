# STEP2 EDGE FEATURE SEMANTICS AUDIT V4.4.1

```json
{
  "contract": "PROJECT7_STEP2_EDGE_PHYSICS_CORRECTNESS_V441",
  "boundary": {
    "swmm_launched": false,
    "d2_regenerated": false,
    "d3_regenerated": false,
    "validation_outcomes_accessed": false,
    "final_accessed": false,
    "formal_run": false,
    "full_train_smoke_run": false,
    "production_wiring_modified": false
  },
  "frozen_inp": {
    "path": "E:\\RTC_sewer\\Project7\\inputs\\network\\wuhan_method_testbed_v067.inp",
    "sha256": "75f04166429f87ae20327cc083d8e8d50a0ed27f5e0add87f77103dba54ec0ea",
    "expected_sha256": "75f04166429f87ae20327cc083d8e8d50a0ed27f5e0add87f77103dba54ec0ea"
  },
  "link_offsets": "ELEVATION",
  "physical_link_census": {
    "physical_links": 1276,
    "conduits": 1167,
    "pumps": 57,
    "orifices": 42,
    "weirs": 10,
    "outlets": 0,
    "single_link_node_pairs": 1165,
    "multi_link_node_pairs": 45,
    "maximum_links_per_pair": 6
  },
  "shape_census": {
    "CIRCULAR": 569,
    "RECT_CLOSED": 594,
    "RECT_OPEN": 3,
    "TRAPEZOIDAL": 1
  },
  "old_graph": {
    "nodes": 932,
    "directed_edges": 2420,
    "unique_undirected_pairs": 1210,
    "ambiguous_old_mappings": 90,
    "unmapped_old_edges": 0,
    "status": "EDGE_TO_LINK_MAPPING_NOT_ONE_TO_ONE"
  },
  "physical_edge_contract": {
    "physical_links": 1276,
    "conduits": 1167,
    "new_directed_conduit_edges": 2334,
    "expected_conduits": 1167,
    "expected_directed_conduit_edges": 2334,
    "parallel_conduit_node_pairs": 31,
    "parallel_conduits_retained": true,
    "regulators_excluded_from_propagation": true,
    "mapping_complete": true
  },
  "parser_semantics": {
    "orifice_fields": true,
    "weir_fields": true,
    "pump_fields": true,
    "shape_aware_geometry": true,
    "unsupported_conduit_shapes": [],
    "barrels_min": 1.0,
    "barrels_median": 1.0,
    "barrels_max": 2.0,
    "zero_barrels_count": 0,
    "barrels_defaults_used": 0,
    "node_invert_mapping_complete": true
  },
  "conduit_invert_slope": {
    "stats": {
      "count": 1167,
      "min": -0.09657127012651083,
      "p01": -0.017334000000000224,
      "median": 0.00024261219495471788,
      "p95": 0.023112745436197568,
      "p99": 0.12625204233209644,
      "max": 0.2435752356617204
    },
    "positive_fraction": 0.8320479862896315,
    "negative_fraction": 0.10282776349614396,
    "zero_fraction": 0.06512425021422451
  },
  "zero_length_gradient_audit": {
    "zero_length_physical_links": 109,
    "zero_length_by_type": {
      "conduit": 0,
      "pump": 57,
      "orifice": 42,
      "weir": 10,
      "outlet": 0
    },
    "regulator_gradient_epsilon_denominator_used": false,
    "gradient_by_link_type": {
      "conduit": {
        "delta_head_m": {
          "count": 14004,
          "min": -2.886350631713867,
          "p01": -1.6291351318359375,
          "median": 0.04700469970703125,
          "p95": 1.5135135650634766,
          "p99": 2.8984384536743164,
          "max": 10.225000381469727
        },
        "hydraulic_gradient_dimensionless": {
          "count": 14004,
          "min": -0.0881851608256246,
          "p01": -0.012385724945655192,
          "median": 0.00043176191480392306,
          "p95": 0.013958000421525054,
          "p99": 0.08537450593307316,
          "max": 0.3018326759338379
        },
        "hydraulic_gradient_normalized": {
          "count": 14004,
          "min": -1.2884791137922875,
          "p01": -0.31408274945509895,
          "median": 0.012781282591897739,
          "p95": 0.3477266665368952,
          "p99": 1.2651231141937465,
          "max": 2.301822984279607
        },
        "zero_length_links": 0,
        "gradient_undefined_for_zero_length": false
      },
      "pump": {
        "delta_head_m": {
          "count": 684,
          "min": -11.034999370574951,
          "p01": -11.034999370574951,
          "median": 1.052999496459961,
          "p95": 24.97357940673828,
          "p99": 25.055999755859375,
          "max": 25.055999755859375
        },
        "hydraulic_gradient_dimensionless": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "hydraulic_gradient_normalized": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "zero_length_links": 57,
        "gradient_undefined_for_zero_length": true
      },
      "orifice": {
        "delta_head_m": {
          "count": 504,
          "min": -4.802202224731445,
          "p01": -4.3811018180847165,
          "median": 0.02686023712158203,
          "p95": 4.660588359832763,
          "p99": 8.198618178367614,
          "max": 8.20000171661377
        },
        "hydraulic_gradient_dimensionless": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "hydraulic_gradient_normalized": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "zero_length_links": 42,
        "gradient_undefined_for_zero_length": true
      },
      "weir": {
        "delta_head_m": {
          "count": 120,
          "min": -0.8899259567260742,
          "p01": -0.8899045753479004,
          "median": 0.00019359588623046875,
          "p95": 0.8180925369262695,
          "p99": 1.9874832916259777,
          "max": 2.0788230895996094
        },
        "hydraulic_gradient_dimensionless": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "hydraulic_gradient_normalized": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "zero_length_links": 10,
        "gradient_undefined_for_zero_length": true
      },
      "outlet": {
        "delta_head_m": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "hydraulic_gradient_dimensionless": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "hydraulic_gradient_normalized": {
          "count": 0,
          "min": NaN,
          "p01": NaN,
          "median": NaN,
          "p95": NaN,
          "p99": NaN,
          "max": NaN
        },
        "zero_length_links": 0,
        "gradient_undefined_for_zero_length": false
      }
    },
    "zero_length_regulator_gradient_bug": false
  },
  "edge_feature_contract": {
    "name": "EDGE_FEATURE_CONTRACT_V441",
    "feature_names": [
      "log_length_m",
      "roughness_n",
      "invert_slope",
      "inlet_relative_offset_m",
      "outlet_relative_offset_m",
      "full_depth_m",
      "width_or_base_width_m",
      "left_side_slope",
      "right_side_slope",
      "power_exponent",
      "barrels",
      "entrance_loss",
      "exit_loss",
      "average_loss",
      "flap_gate",
      "valid_length",
      "valid_roughness",
      "valid_invert_slope",
      "valid_inlet_relative_offset",
      "valid_outlet_relative_offset",
      "valid_full_depth",
      "valid_width_or_base_width",
      "valid_left_side_slope",
      "valid_right_side_slope",
      "valid_power_exponent",
      "valid_barrels",
      "link_type_conduit",
      "shape_circular",
      "shape_rect_open",
      "shape_rect_closed",
      "shape_trapezoidal",
      "shape_power",
      "shape_other"
    ],
    "normalized_finite": true,
    "normalization_source": "analytic/frozen Train-only static graph statistics",
    "future_truth_used": false,
    "link_flow_used": false,
    "link_flow_status": "LINK_FLOW_NOT_AVAILABLE_ONLINE"
  },
  "dynamic_normalization": {
    "contract": "EDGE_DYNAMIC_NORMALIZATION_V441",
    "source": "full Train18 stamped normalization plus frozen conduit geometry median; no Validation/Final",
    "source_manifest_sha256": "7c69211823f5419a7fabcae03c68f6578b364d1f948f0dbfd76534c0cc48f20d",
    "head_scale_train_m": 5.145068645477295,
    "gradient_scale_train_dimensionless": 0.03356537590421303,
    "conduit_length_median": 153.285,
    "transform": "delta_head/head_scale; signed_log1p(abs(delta_head/length)/gradient_scale)",
    "finite": true,
    "sha256": "eb9817ba95e630568e882b364fdf05fc35261591f39706094edca2ad110a8718"
  },
  "status": "PASS",
  "key_passes": {
    "orifice_parser": true,
    "weir_parser": true,
    "pump_parser": true,
    "xsection_units": true,
    "barrels": true,
    "node_invert_mapping": true,
    "conduit_slope": true,
    "zero_length_regulator_gradient_removed": true,
    "conduit_directed_edge_mapping": true,
    "dynamic_normalization": true
  }
}
```
