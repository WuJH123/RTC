# Codex start here — Project7 v0.6.9 V125 canonical RTC contract

This file is the **single current execution entrypoint**. Do not infer the active
method from historical version numbers, old PRs, or old runners. Read this file and
`configs/step2_current_contract.json` before doing any Project7 work.

## 1. Frozen research question

Project7 is an idealized SWMM methodology testbed, not a field digital twin.
Authoritative truth is EPA SWMM.

Every 10 minutes the Proposed controller must:

1. use sparse causal observations and the frozen accepted Step1 to reconstruct the current hydraulic state;
2. derive a causal engineering Sparse-RBC anchor from that reconstructed state;
3. score engineering-feasible finite joint-action first moves with Step2 Value;
4. keep the Sparse-RBC anchor by default;
5. execute a learned override only when calibrated evidence supports lower TFV than the anchor;
6. write the first 10-minute target command to all 109 writable pump/orifice/weir links;
7. verify target-latch/readback semantics, then reconstruct and re-optimise at the next update.

Primary objective: whole-system cumulative sewer-node overflow volume (`TFV`) minimization.
Priority8 `PFV` is one-sided **soft deterioration protection**, not a hard constraint.
A PFV improvement is not allowed to buy a candidate whose predicted TFV is worse than
the engineering anchor. Global Peak remains report-only.

Do not describe TFV as 2-D street inundation volume.

## 2. Canonical code surface

Read in this order:

1. `configs/step2_current_contract.json`
2. `src/rtc/step2_current.py`
3. `src/rtc/step2_policy_v125.py`
4. `src/rtc/controller_v125.py`
5. `scripts/run_policy_v125.py`
6. `src/rtc/step2_d4_action_support_v125.py`
7. `scripts/plan_step2_v125_d4_action_support.py`
8. `src/rtc/step3_calibration_v125.py`
9. `scripts/calibrate_step3_v125_anchor_override.py`

New production-facing code must import the current method through `rtc.step2_current`.
V120/V121/V122/V123 runners and V8–V113 hydraulic-effect models are retained only for
reproduction/forensics unless this document explicitly references them.

## 3. Current evidence that is already frozen

Do not spend a new optimisation cycle re-solving these points.

- Step1 Sparse-RBC parity is strong enough for the current control path: actuator-adjacent endpoint depth NSE ~0.991 and action Spearman ~0.988 on the frozen TrainFit audit.
- Frozen causal V70 Holdout D3 Value: rank ~0.414, pairwise ~0.652, top1 ~0.563.
- V124 interaction-aware attention model did **not** improve rank (about 0.410), so do not start another architecture sweep.
- Step3 first-move coverage is no longer collapsed: ~75–83 unique executable first moves and 109/109 actuator coverage in the V124 audit.
- Development T5: Sparse-RBC anchor-only reduced TFV ~10.17%; learned-only ~7.88%; old hybrid ~8.88%; Auto-RBC reference ~22.02%.
- The old V123 hybrid admitted learned actions relative to PASSIVE, not relative to the Sparse-RBC anchor. This is superseded by V125.
- V125 D4 support-gap audit confirmed substantial train/deploy action mismatch around the causal Sparse-RBC anchor: 112-group median nearest-anchor normalized L1 ~0.667; selected-48 median ~0.708.

Continuous MPC remains blocked.

## 4. V125 Step2/Step3 definition

### 4.1 Value model

The first V125 retraining experiment is a **data-support ablation**, not an architecture
experiment. Keep the accepted V124/V70 architecture settings frozen for the first run:

- hidden dim = 96
- attention heads = 4 when using V124
- listwise weight = 0.30
- seed = 42
- same causal rainfall/normalisation/splits

Only the training action support may change by adding D4-FIT supervision.

### 4.2 Engineering anchor

The default command is the causal Sparse-RBC first move computed from Step1 reconstructed
state and frozen actuator topology/physics. Authoritative current SWMM node depth is never
allowed into Proposed runtime anchor construction.

### 4.3 Learned override

The old condition “candidate beats PASSIVE by the V123 false-benefit margin” is necessary
but no longer sufficient.

V125 additionally requires a separately calibrated anchor-relative condition:

`predicted_TFV(candidate) - predicted_TFV(anchor) + anchor_override_margin < 0`

and the combined TFV + one-sided PFV-soft objective must also improve over the anchor.

Therefore PFV cannot compensate for a TFV-worse candidate relative to the anchor.

### 4.4 Rolling execution

- SWMM/model record step: 300 s.
- Control update: 600 s.
- Execute first 600-s block only.
- Maximum target-setting change: 0.5 per update.
- Score only already executable sequences.
- No projection after scoring.
- Controller returns a target for every writable actuator each decision.
- Native SWMM RTC controls are disabled for Proposed.
- Target-latch write/readback is authoritative for command acceptance; realised current-setting lag is a physical tracking diagnostic, not a write-failure test.

## 5. D4 V2 data contract — must replace the old V125 plan before SWMM

The old plan at commit `ab7a3b1` proved the action-support gap but its continuation rule is
superseded. Re-run the planner from the current main/merged V125 code before generating
D4 truth.

D4 V2 requirements:

- Development TrainFit only; frozen 112/32 D2 split retained.
- Select at most 48 high-gap checkpoints by deterministic rainfall-balanced geometry.
- Freeze D4 `fit` / `audit` roles **before** any D4 outcome is generated.
- Split unit is rainfall group, never branch/candidate.
- With 14 selected rainfall groups and audit_fraction=0.25 the current deterministic contract should yield 10 fit / 4 audit groups.
- D4-AUDIT is never used for training, checkpoint selection, calibration, or hyperparameter tuning.
- Candidate families remain local/interpretable around the Sparse-RBC anchor.
- The candidate may differ from the anchor only in the executable first 600 s.
- After the first block, **all candidates at that checkpoint must share the exact same Sparse-RBC anchor continuation**.
- Save the complete action sequence and its SHA256 in the frozen plan.

This isolates the causal marginal value of the current 10-minute decision and prevents
future-tail credit from being attributed to the first move.

## 6. D4 truth / retraining sequence

After the V2 plan passes correctness audit, one bounded authoritative SWMM labelling round
is allowed for the exact frozen plan. Do not redesign candidates after seeing outcomes.

Then:

1. build a new D4 evidence/cache source with exact SWMM cumulative TFV and Priority8 PFV;
2. train V125 using D2/D3 + **D4-FIT only**;
3. evaluate generic InternalHoldout D3 as a non-regression diagnostic;
4. evaluate D4-AUDIT anchor-neighbourhood ranking, sign, regret and false-benefit errors;
5. derive the anchor-relative one-sided TFV margin from D4-FIT only;
6. use D4-AUDIT only once as an untouched development audit of that margin;
7. only then run the V125 T5 closed-loop comparison.

If D4 support does not improve anchor-neighbourhood identification, stop adding network
capacity. The next research hypothesis is anchor-relative Advantage Value
`TFV(candidate)-TFV(anchor)`, not a larger Transformer/GNN.

## 7. Required V125 development metrics

Always report at least:

- generic InternalHoldout D3 rank, pairwise, sign, top1, MAE and regret;
- D4-AUDIT candidate-vs-anchor rank/pairwise/sign;
- best-action regret around anchor;
- beneficial-override precision/recall;
- false-benefit override rate among admitted overrides;
- mean/worst true TFV advantage of admitted overrides;
- PFV change of admitted overrides;
- 10-min decision runtime and deadline failures;
- target write/readback failures;
- score==execute and continuity violations;
- anchor / learned-override / passive fractions;
- authoritative closed-loop TFV, Priority8 PFV and Global Peak.

Aggregation across scientific events is event/rainfall-group balanced.

## 8. Continuous MPC gate

Continuous L-BFGS-B / 109-actuator differentiable search is forbidden unless the frozen
project gate passes:

- TFV rank >= 0.70
- top1 >= 0.50
- TFV gradient sign >= 0.70
- gradient cosine >= 0.60

These are Project7 preregistered engineering/scientific thresholds, not universal
literature thresholds. Do not lower them to enable continuous MPC.

## 9. Development boundaries

Until V125 finite development evidence passes:

- no Validation outcome access;
- no Final outcome access;
- no Formal;
- no Policy Lock;
- no continuous MPC;
- no tuning on D4-AUDIT;
- no future realised rainfall online;
- no future SWMM hydraulic truth online;
- no return to V8–V113 hydraulic-effect architecture search;
- no seed/hidden/head/loss sweep in the first D4 data-support experiment.

## 10. Scientific claim boundary

The Proposed controller may claim only the best action selected within the generated
engineering-feasible finite candidates around the current state/anchor. It must not claim
global optimality in the full continuous 109-dimensional space.

Final performance claims must come from authoritative SWMM and untouched scientific
Validation/Final only after the development method and policy are frozen.
