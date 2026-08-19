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
  -> frozen base Step2 over all 109 actuators
  -> <=4 support-bounded H10 first-action candidates
       [Step2 full / Step2 half / type-aware hydraulic / 109-D projected gradient]
  -> receding-policy-return critic + one-sided admission
  -> execute first H10 target or HOLD
  -> authoritative SWMM target write/readback
  -> re-observe and replan
```

Objective hierarchy:

- **online primary:** system-wide cumulative TFV only;
- **secondary authoritative safety:** Priority8 PFV non-inferiority versus No-control, currently
  `PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`;
- **report-only:** Global Peak.

No future realised rainfall/state/flooding, no online SWMM candidate search, no PFV/Peak penalty and
no baseline imitation are permitted online.

## Hybrid first-action search, not historical full-plan L-BFGS-B

Historical Development found two distinct failure modes. A 12 x 109 continuous full-plan optimizer
could exploit surrogate extrema/support edges, but an excessively conservative executable-prefix
admission could also starve useful actions. The current controller therefore keeps differentiable
search capacity without restoring the historical 1308-dimensional optimizer.

At each decision the candidate family is at most:

1. `STEP2_H10_PROBE_SCALE_1.00`;
2. `STEP2_H10_PROBE_SCALE_0.50`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`;
4. `SUPPORT_CONSTRAINED_GRADIENT_H10`.

The fourth candidate is **109-dimensional and H10-only**. Autograd through frozen base Step2 proposes
one first target; every trial is immediately projected to actuator bounds, <=0.5 target slew, the
frozen per-facility q95 first-move radius and the q95 changed-facility ceiling. The gradient score
cannot authorize execution. Every distinct candidate is subsequently contracted to q95 joint-
sequence support and ranked by the separately trained receding-policy-return critic.

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

This is the critic representation of one first action, **not** an instruction to hold the real system
for 350 minutes. The authoritative controller re-observes and replans after the executed H10 block.

## Current first policy iteration

Historical V12/L-BFGS-B is archival evidence/ablation only. The first current paired-label round uses
`PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V2`: the same four-family H10 proposal/support geometry
without a policy-return critic. Frozen base Step2 ranks pi0 proposals; exact paired SWMM then supplies
`Q^pi0/A^pi0` labels. After critic training and matched calibration, pi1 uses policy-return UCB ranking.
If pi1 materially changes the state/action distribution, a new role-disjoint `Q^pi1` round is required
before Policy Lock.

## Step1 / Step2 / critic roles

- **Step1:** sparse sensing -> causal full-network state.
- **Base Step2 V5:** frozen TFV action-effect representation, H10 directional probe model and
  differentiable first-action proposer; it is not the deployed closed-loop value oracle.
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

The Proposed policy is not required to beat every operational comparator on every event. Performance
is assessed event-balanced on untouched authoritative SWMM events, with Priority8 PFV safety reported
on the same events.

## Efficient exact paired data generation

Rainfall groups are role-disjoint: policy-return train >=48, model-selection validation >=12,
conformal calibration >=24, plus separate new Development probes. Matched calibration must include
the candidate families that can appear online, including the projected-gradient H10 family.

The maintained workflow avoids unnecessary recomputation:

```text
hybrid pi0 parent trajectory
  -> prefix-only causal context capture
  -> four-family Practical candidate portfolio
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

The current manifest contains graph, sensors, config, Step1, base Step2, sequence support and
Priority8 only. Historical V12 policy/first-move admissions are not current dependencies. Downstream
current scripts must reuse the manifest; silent fallback to another historical path is forbidden.

Current Practical scripts include:

```text
scripts/run_policy_direct_tfv_base_hybrid_parent_current.py
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

Near-all-HOLD behavior is treated as an admission/data failure mode, not a successful conservative
RTC. `READY_FOR_POLICY_LOCK=false` by default. Validation, Final, Formal and Policy Lock remain
inaccessible until role-disjoint policy-return training/validation/calibration, independent
Development probes, useful non-starved TFV control, Priority8-PFV-safe positive claims and any
necessary policy-iteration round are complete.
