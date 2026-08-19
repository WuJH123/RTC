# Project7 — sparse-sensing Practical real-time control for urban drainage

Project7 is an idealized EPA-SWMM methodology testbed for a large drainage network. It is not a
field-calibrated digital twin. **Authoritative hydraulic truth always comes from SWMM.**

The current research question is:

> Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable
> learning controller reduce system-wide total flooding volume (TFV), while avoiding material
> deterioration of flooding at frozen Priority8 nodes?

## Current paper logic

Every 10 minutes:

```text
causal sparse hydraulic/rainfall history
  -> Step1 full-network state reconstruction
  -> frozen base Step2 H10 action-effect probes over all 109 actuators
  -> <=3 support-bounded first-action candidates
  -> receding-policy-return critic + one-sided admission
  -> execute H10 candidate or HOLD
  -> authoritative SWMM target write/readback
  -> re-observe
```

Objective hierarchy:

- **online primary:** system-wide cumulative TFV only;
- **secondary authoritative safety:** Priority8 PFV non-inferiority versus No-control, currently
  `PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`;
- **report-only:** Global Peak.

No future realised rainfall/state/flooding, no online SWMM candidate search, no PFV/Peak penalty and
no baseline imitation are permitted online.

## Why the deployed controller is not the historical 1308-D L-BFGS-B MPC

Historical Development found two distinct limitations:

1. 12 x 109 continuous full-plan optimization could exploit surrogate extrema and strain the frozen
   training-support geometry;
2. locally correct H360 action values did not reliably translate into better whole-event TFV under
   repeated H10 replanning.

The Practical online policy therefore does **not** call the historical L-BFGS-B optimizer. V12 remains
only an offline frozen parent `pi0` for the first paired policy-return label round and as an ablation.

## Exact receding-policy first-action value

The learned deployed target is:

```text
A^pi(x_t,u_t)
 = J(candidate H10 -> frozen continuation pi)
 - J(HOLD H10 -> the same frozen continuation pi)
```

Candidate/HOLD SWMM branches use the same raw causal prefix and same frozen continuation. Training,
calibration and runtime all encode the candidate identically as:

```text
H10 candidate target -> H350 HOLD target
```

This prevents the open-loop H360 action representation from silently re-entering the closed-loop
critic.

## Practical candidate layer

Frozen base Step2 is reused as an action-effect representation and cheap directional probe model. At
one decision it batch-scores support-bounded positive/negative H10 probes for all 109 facilities,
combines individually predicted-beneficial directions within q95 changed-facility support, and forms
at most:

1. `STEP2_H10_PROBE_SCALE_1.00`;
2. `STEP2_H10_PROBE_SCALE_0.50`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`.

Candidates are contracted to q95 joint-sequence support using the actual H10 pulse geometry. The
policy-return critic ranks them; a candidate executes only when its calibrated one-sided upper bound
is negative, otherwise HOLD is latched.

## Step1 / Step2 / critic roles

- **Step1:** sparse sensing -> causal full-network state.
- **Base Step2 V5:** frozen pairwise TFV action-effect representation and H10 direction generator;
  it is not treated as the deployed closed-loop value oracle.
- **Policy-return critic:** initialized from base Step2 and fine-tuned by default only in
  facility/action/interaction layers on exact paired SWMM labels.

Checkpoint selection is control-first: false-beneficial selection, same-query ranking, selected
regret, then event-balanced MAE.

## Engineering and support boundaries

- 109 writable actuators screened every decision;
- 300-s observation/model context and 600-s supervisory updates;
- execute the first 10-min target only;
- physical setting min/max;
- target movement <=0.5 per 10-min update;
- supervisory anchor = previous commanded `target_setting`;
- q95 TrainFit changed-facility and joint-sequence support;
- HOLD = retain the previous supervisory target;
- unchanged facilities retain their previous target;
- score == execute, with no material post-score projection;
- target write/readback and causal-history readiness fail closed.

## Actuator semantics and baselines

Hydraulic release intent is mapped to actuator-specific SWMM SETTING semantics. Pump, orifice, weir
and outlet settings are not interpreted as one universal opening coordinate.

Authoritative comparison retains:

- No-control — primary reference;
- Internal RTC — operational comparator;
- Auto-RBC — operational type-aware rule comparator;
- EFD — operational storage equal-filling comparator;
- all-max-setting (`all_open`) — diagnostic extreme;
- all-min-setting (`all_closed`) — diagnostic extreme.

The Proposed policy is not required to beat every operational comparator on every event, and the
numerical setting extremes are not mandatory competitive wins.

## Efficient exact paired data generation

Rainfall groups are role-disjoint: policy-return train >=48, model-selection validation >=12,
conformal calibration >=24, plus separate new Development probes.

The maintained workflow avoids unnecessary recomputation:

```text
prefix-only causal context capture
  -> Practical candidate portfolio
  -> one shared HOLD authoritative branch
  -> N sequential candidate authoritative branches
  -> same-query paired labels
```

Thus a query with N candidates requires `1 + N` full authoritative branches rather than `2N`.

## Path-safe current execution

Historical V* directories remain evidence, not execution selectors. Local Codex first discovers the
intended existing artifacts from `study_v069`, then freezes their **absolute paths and SHA-256** in one
manifest:

```text
scripts/build_project7_practical_asset_manifest_current.py
```

Downstream current scripts must reuse that manifest. Silent fallback to another historical path is
forbidden.

Current Practical scripts include:

```text
scripts/capture_direct_tfv_policy_return_context_current.py
scripts/design_direct_tfv_policy_return_portfolio_current.py
scripts/run_direct_tfv_policy_return_query_current.py
scripts/compile_direct_tfv_policy_return_dataset_current.py
scripts/train_direct_tfv_policy_return_current.py
scripts/score_direct_tfv_policy_return_calibration_current.py
scripts/calibrate_direct_tfv_policy_return_portfolio_admission_current.py
scripts/run_policy_direct_tfv_policy_return_development.py
scripts/run_six_baselines_development_current.py
scripts/compare_direct_tfv_baselines_current.py
scripts/add_pfv_to_direct_tfv_comparison_current.py
```

See **`CODEX_START_HERE.md`** and **`PROJECT7_PRACTICAL_RTC_V14.md`** for the full scientific and local
execution contracts.

## Development boundary

`READY_FOR_POLICY_LOCK=false` by default. Validation, Final, Formal and Policy Lock remain inaccessible
until role-disjoint policy-return training/validation/calibration, independent Development probes,
PFV-safe authoritative comparison and any necessary policy-iteration round are complete.
