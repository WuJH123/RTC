# STEP2 EDGE HYDRAULIC LINEAGE AUDIT V4.4

```json
{
  "contract": "STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44",
  "boundary": {
    "swmm_launched": false,
    "d2_regenerated": false,
    "d3_regenerated": false,
    "validation_outcomes_accessed": false,
    "final_accessed": false,
    "formal_run": false,
    "full_train_smoke_run": false
  },
  "frozen_inp": {
    "path": "E:\\RTC_sewer\\Project7\\inputs\\network\\wuhan_method_testbed_v067.inp",
    "sha256": "75f04166429f87ae20327cc083d8e8d50a0ed27f5e0add87f77103dba54ec0ea",
    "expected_sha256": "75f04166429f87ae20327cc083d8e8d50a0ed27f5e0add87f77103dba54ec0ea"
  },
  "graph": {
    "path": "E:\\RTC_sewer\\Project7\\study_v069\\formal_assets\\graph_schema.npz",
    "sha256": "ebeceebaf8ec941abc830e950e579869cb3d411b9d0169e2c210298355efa071",
    "nodes": 932,
    "edge_index_shape": [
      2,
      2420
    ]
  },
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
  "legacy_graph_audit": {
    "nodes": 932,
    "directed_edges": 2420,
    "unique_directed_pairs": 2420,
    "unique_undirected_node_pairs": 1210,
    "bidirectional_undirected_pairs": 1210,
    "self_loops": 0,
    "duplicate_directed_edges": 0,
    "isolated_nodes": 0,
    "isolated_node_ids": [],
    "ambiguous_old_mappings": 90,
    "ambiguous_old_mapping_examples": [
      {
        "src": "HS1301283",
        "dst": "BZ10",
        "link_ids": [
          "HS1301283.1",
          "MC_6"
        ]
      },
      {
        "src": "HS1312178",
        "dst": "HS1312197",
        "link_ids": [
          "HS1312178.2",
          "MC_230"
        ]
      },
      {
        "src": "HS1322626",
        "dst": "HS1322659",
        "link_ids": [
          "HS1322626.2",
          "MC_305"
        ]
      },
      {
        "src": "HS1305028",
        "dst": "HS1304367",
        "link_ids": [
          "MC_104",
          "MC_105"
        ]
      },
      {
        "src": "VP0271848",
        "dst": "ADD424",
        "link_ids": [
          "MC_2",
          "MC_3"
        ]
      },
      {
        "src": "HS1318896",
        "dst": "HS1324158",
        "link_ids": [
          "MC_286",
          "MC_287"
        ]
      },
      {
        "src": "HS1328602",
        "dst": "HS1328591",
        "link_ids": [
          "MC_371",
          "MC_373"
        ]
      },
      {
        "src": "HS2221051",
        "dst": "HS2529052",
        "link_ids": [
          "MC_450",
          "MC_452"
        ]
      },
      {
        "src": "MH0001216",
        "dst": "H0000",
        "link_ids": [
          "MC_49",
          "MH0001216.2"
        ]
      },
      {
        "src": "MH0202568",
        "dst": "MH0200624",
        "link_ids": [
          "MC_817",
          "MH0202568.1"
        ]
      }
    ],
    "unmapped_old_edges": 0,
    "unmapped_old_edge_examples": [],
    "edge_source_contract": "formal_assets/graph_schema.npz edge_index has node adjacency only; no physical link IDs",
    "edge_to_link_mapping_status": "EDGE_TO_LINK_MAPPING_NOT_ONE_TO_ONE"
  },
  "physical_directed_edge_contract": {
    "new_physical_directed_edges": 2552,
    "one_directed_forward_and_reverse_per_physical_link": true,
    "mapping_complete": true,
    "unmapped_physical_links": 0,
    "parallel_links_retained": true,
    "edge_to_link_ids_unique_count": 1276,
    "orientation_signs": [
      -1,
      1
    ]
  },
  "edge_feature_contract": {
    "static_feature_names": [
      "length_m",
      "roughness_n",
      "inlet_offset_m",
      "outlet_offset_m",
      "geom1_m",
      "geom2_m",
      "geom3_m",
      "geom4_m",
      "barrels",
      "entrance_loss",
      "exit_loss",
      "average_loss",
      "flap_gate",
      "link_type_conduit",
      "link_type_pump",
      "link_type_orifice",
      "link_type_weir",
      "link_type_outlet",
      "shape_circular",
      "shape_rect_open",
      "shape_rect_closed",
      "shape_elliptical",
      "shape_arch",
      "shape_trapezoidal",
      "shape_custom",
      "shape_irregular",
      "shape_other"
    ],
    "dynamic_feature_names": [
      "head_src",
      "head_dst",
      "delta_head",
      "hydraulic_gradient"
    ],
    "type_availability": {
      "conduit": {
        "count": 1167,
        "length_present": true,
        "roughness_present": true,
        "shape_identity_present": true
      },
      "pump": {
        "count": 57,
        "length_present": false,
        "roughness_present": false,
        "shape_identity_present": false
      },
      "orifice": {
        "count": 42,
        "length_present": false,
        "roughness_present": false,
        "shape_identity_present": false
      },
      "weir": {
        "count": 10,
        "length_present": false,
        "roughness_present": false,
        "shape_identity_present": false
      },
      "outlet": {
        "count": 0,
        "length_present": false,
        "roughness_present": false,
        "shape_identity_present": false
      }
    },
    "shape_identity_preserved": true,
    "raw_geometry_preserved_in_physical_link_records": true,
    "edge_features_finite": true
  },
  "edge_feature_normalization": {
    "contract": "EDGE_FEATURE_NORMALIZATION_V44",
    "method": "analytic_static_robust_graph_statistics_median_iqr_with_unit_floor",
    "feature_names": [
      "length_m",
      "roughness_n",
      "inlet_offset_m",
      "outlet_offset_m",
      "geom1_m",
      "geom2_m",
      "geom3_m",
      "geom4_m",
      "barrels",
      "entrance_loss",
      "exit_loss",
      "average_loss",
      "flap_gate",
      "link_type_conduit",
      "link_type_pump",
      "link_type_orifice",
      "link_type_weir",
      "link_type_outlet",
      "shape_circular",
      "shape_rect_open",
      "shape_rect_closed",
      "shape_elliptical",
      "shape_arch",
      "shape_trapezoidal",
      "shape_custom",
      "shape_irregular",
      "shape_other"
    ],
    "transform": [
      "log1p",
      "log1p",
      "identity",
      "identity",
      "log1p",
      "log1p",
      "log1p",
      "log1p",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity",
      "identity"
    ],
    "location": [
      4.82475471496582,
      0.013902905397117138,
      18.40169906616211,
      18.19499969482422,
      0.9162907600402832,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "scale": [
      3.0624752044677734,
      1.0,
      3.8690004348754883,
      4.035000324249268,
      1.0,
      1.3862943649291992,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "normalized_shape": [
      2552,
      27
    ],
    "finite": true,
    "sha256": "75bc18a592347a3f3479b37a5e30f65bf75dbe37ced49ef682104b163836926c"
  },
  "dynamic_contract": {
    "head_src": true,
    "head_dst": true,
    "delta_head": true,
    "hydraulic_gradient": true,
    "source": "causal current/reference model state only",
    "future_truth_used": false,
    "link_flow_used": false,
    "link_flow_online_availability": "LINK_FLOW_NOT_AVAILABLE_ONLINE"
  },
  "mapping_complete": true,
  "old_graph_mapping_one_to_one": false,
  "status": "PASS_PHYSICAL_LINEAGE_WITH_LEGACY_MULTI_EDGE_AMBIGUITY"
}
```
