# Codex start here — Project7 v0.6.9 V125 canonical RTC contract

This file is the **single current execution entrypoint**. Do not infer the active method from historical version numbers, old PRs, old runners, or old branch names. Read this file and `configs/step2_current_contract.json` before doing any Project7 work.

## 1. Frozen research question

Project7 is an idealized SWMM methodology testbed, not a field digital twin. Authoritative truth is EPA SWMM.

Every 10 minutes the Proposed controller must:

1. use sparse causal observations and the frozen accepted Step1 to reconstruct the current hydraulic state;
2. derive a causal engineering Sparse-RBC anchor from that reconstructed state;
3. use that anchor as the **online Value reference**;
4. score a small, engineering-feasible local first-move neighbourhood with direct candidate-minus-anchor TFV/PFV Value;
5. keep the Sparse-RBC anchor by default;
6. execute a learned override only when calibrated evidence supports lower TFV than the anchor and the one-sided PFV-soft objective also improves;
7. write the first 10-minute target command to all 109 writable pump/orifice/weir links;
8. verify target-latch/readback semantics, then reconstruct and re-optimise at the next update.

Primary objective: whole-system cumulative sewer-node overflow volume (`TFV`) minimization. Priority8 `PFV` is one-sided **soft deterioration protection**, not a hard constraint. PFV improvement is never allowed to buy a candidate whose predicted TFV is worse than the engineering anchor. Global Peak remains report-only.

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
8. `scripts/build_step2_v125_d4_execution_manifest.py`
9. `src/rtc/d3_runner_guard.py` through CLI `rtc-run-d3-batch`
10. `src/rtc/step2_d4_cache_v125.py`
11. `scripts/build_step2_v125_d4_cache.py`
12. `scripts/run_step2_v125_value.py`
13. `scripts/run_step2_v125_pfv.py`
14. `scripts/build_step2_v125_anchor_override_evidence.py`
15. `src/rtc/step3_calibration_v125.py`
16. `scripts/calibrate_step3_v125_anchor_override.py`

New production-facing code must import the current method through `rtc.step2_current`. V120/V121/V122/V123 runners and V8–V113 hydraulic-effect models are retained only for reproduction/forensics unless this document explicitly references them.

## 3. Current evidence already frozen

Do not spend another optimisation cycle re-solving these points.

- Step1 Sparse-RBC parity is strong enough for the current control path: actuator-adjacent endpoint depth NSE ~0.991 and action Spearman ~0.988 on the frozen TrainFit audit.
- Frozen causal V70 Holdout D3 Value: rank ~0.414, pairwise ~0.652, top1 ~0.563.
- V124 interaction-aware attention model did **not** improve old passive-relative rank (~0.410), so do not start another architecture sweep.
- The old generic Step3 candidate generator no longer has first-move collapse (~75–83 unique executable first moves, 109/109 actuators), but that broad support is intentionally **not** the V125 online search domain because it is outside the new D4 local supervision.
- Development T5: Sparse-RBC anchor-only reduced TFV ~10.17%; learned-only ~7.88%; old hybrid ~8.88%; Auto-RBC reference ~22.02%.
- The old V123 hybrid admitted learned actions relative to PASSIVE, not directly relative to Sparse-RBC. This is superseded by V125.
- V125 D4 support-gap audit confirmed substantial train/deploy action mismatch around the causal Sparse-RBC anchor: 112-group median nearest-anchor normalized L1 ~0.667; selected-48 median ~0.708.

Continuous MPC remains blocked.

## 4. V125 Step2/Step3 definition

### 4.1 Production Value target

The production decision quantity is now direct anchor-relative advantage:

`DeltaTFV_anchor = TFV(candidate) - TFV(Sparse-RBC anchor)`

`DeltaPFV_anchor = PFV(candidate) - PFV(Sparse-RBC anchor)`

The online model receives the Sparse-RBC sequence as `reference_settings`; therefore anchor==reference must produce exact-zero TFV/PFV by architecture. Do not compute two passive-relative predictions and subtract them at runtime.

Historical D2/D3 candidate-minus-reference groups remain auxiliary generalization supervision. D4-FIT provides the decision-relevant anchor-relative supervision.

### 4.2 First V125 retraining is a data-support ablation

Do not change architecture in the first D4 run:

- TFV V124 hidden dim = 96;
- attention heads = 4;
- listwise weight = 0.30;
- seed = 42;
- PFV V70 architecture unchanged;
- same causal rainfall semantics;
- base TrainFit normalization and target scales remain frozen.

Only D4-FIT action support may change.

### 4.3 Engineering anchor and local candidates

The default command is the causal Sparse-RBC first move computed from Step1 reconstructed state and frozen actuator topology/physics. Authoritative current SWMM node depth is never allowed into Proposed runtime anchor construction.

The V125 online search domain is deliberately the same small local family used by D4:

- HOLD-first-move / anchor continuation;
- anchor scale 0.50;
- anchor scale 0.75;
- exact anchor 1.00;
- +/-25% perturbation of up to three strongest active hydraulic/control groups.

After clipping/dedup this is typically 7–10 first moves per decision. This is not a claim of global optimum; it is a decision-support domain with observed SWMM supervision.

Every non-anchor candidate may differ only during the executable first 600 s. All later 5-min steps are the exact same Sparse-RBC anchor continuation. Runtime candidate generation must fail closed if engineering projection changes a generated candidate before scoring.

### 4.4 Learned override

A learned candidate is eligible only if both are true:

`predicted_DeltaTFV_anchor + calibrated_TFV_false_benefit_margin < 0`

and

`anchor-relative TFV + one-sided PFV-soft objective < 0`.

The PFV model-error margin is also recalibrated on D4-FIT after PFV retraining. PFV improvement cannot compensate for TFV failure.

### 4.5 Rolling execution

- SWMM/model record step: 300 s.
- Control update: 600 s.
- Execute first 600-s block only.
- Maximum target-setting change: 0.5 per update.
- Score only already executable sequences.
- No projection after scoring.
- Controller returns a target for every writable actuator each decision.
- Native SWMM RTC controls are disabled for Proposed.
- Target-latch write/readback is authoritative for command acceptance; realised current-setting lag is a physical tracking diagnostic, not a write-failure test.

## 5. D4 V2 data contract — old `ab7a3b1` plan is superseded

The old V125 plan proved the action-support gap but its continuation rule is superseded. Re-run the planner from current `main` before any D4 truth generation.

D4 V2 requirements:

- Development TrainFit only; frozen 112/32 D2 split retained.
- Select at most 48 high-gap checkpoints by deterministic rainfall-balanced geometry.
- Freeze D4 `fit` / `audit` roles **before** any D4 outcome is generated.
- Split unit is rainfall group, never branch/candidate.
- With the current 14 selected rainfall groups and `audit_fraction=0.25`, expect 10 fit / 4 audit groups.
- D4-AUDIT is never used for training, checkpoint selection, calibration, normalization, scale derivation, or hyperparameter tuning.
- Reference is the causal Sparse-RBC anchor.
- Candidate may differ from the anchor only in the executable first 600 s.
- After the first block, all candidates at that checkpoint share the exact same Sparse-RBC anchor continuation.
- Save complete H72 scoring sequence and SHA256.

This isolates the causal marginal value of the current 10-minute decision and prevents future-tail credit from being attributed to the first move.

## 6. Authoritative D4 execution contract

Do not hand-write another PySWMM runner. Use the existing guarded path:

`scripts/build_step2_v125_d4_execution_manifest.py` -> `rtc-run-d3-batch`.

The execution adapter converts H72 x 5-min scoring sequences to 36 x 10-min control blocks. Every pair of 5-min targets must be exactly equal; otherwise fail closed. The execution manifest must contain one exact `anchor_scale_1.00` reference per checkpoint, 109-actuator complete target dictionaries, common continuation, rate feasibility and frozen event/checkpoint lineage.

Before SWMM, always run `rtc-run-d3-batch ... --census-only`. The census must show zero endpoint-invalid requests. Only then may the exact same frozen manifest be executed without `--census-only`. Reuse existing simulation assets where identity matches; never redesign candidates after outcomes are observed.

## 7. D4 cache and first retraining

Build physically separate caches from the same authoritative D4 run summary:

- `scripts/build_step2_v125_d4_cache.py --split-role fit`
- `scripts/build_step2_v125_d4_cache.py --split-role audit`

D4 uses `source_kind=D4`; the explicit reference is Sparse-RBC, not historical `D3_HOLD_REFERENCE`. Do not weaken or reinterpret old V60/D3 guards.

Then run:

- TFV: `scripts/run_step2_v125_value.py`
- PFV: `scripts/run_step2_v125_pfv.py`

The TFV first experiment keeps V124 hidden=96, heads=4, listwise=0.30, seed=42 and base normalization/scale unchanged. D4-FIT is the only new action-support supervision. PFV similarly retains its base architecture and base TrainFit scale.

**D4-AUDIT is an action-support holdout around states/rainfall groups already present in base TrainFit data. It is not independent state/rainfall validation.** Generic base InternalHoldout D3 remains the independent rainfall/state generalization diagnostic.

## 8. Direct anchor-relative calibration

After accepted TFV/PFV D4 training:

1. run `scripts/build_step2_v125_anchor_override_evidence.py` to write direct candidate-minus-anchor TFV/PFV truth/prediction rows for physically separate D4 FIT/AUDIT caches;
2. run `scripts/calibrate_step3_v125_anchor_override.py`.

D4-FIT alone estimates:

- one-sided TFV false-benefit margin;
- one-sided PFV under-predicted-deterioration model-error margin.

D4-AUDIT is used only once to report TFV false-benefit rate, beneficial-override precision/recall and PFV false-soft-safe rate. No D4-AUDIT outcome may change architecture, loss, seed, candidate design, split, calibration quantile or margins.

## 9. V125 closed-loop T5

Only after D4 data, training and calibration are frozen may `scripts/run_policy_v125.py` be executed on the existing development/debug T5 event.

Report:

- authoritative SWMM whole-system TFV;
- Priority8 PFV;
- Global Peak (diagnostic only);
- anchor / learned-override / passive fractions;
- selected candidate families;
- beneficial/false-benefit override statistics;
- PFV false-soft-safe statistics;
- 10-min decision runtime mean/p95/max and missed deadlines;
- target-latch/write failures;
- continuity violations;
- score==execute violations.

The primary development question is whether V125 improves on Sparse-RBC anchor-only TFV (~10.17%), not merely whether it beats passive. Auto-RBC (~22.02% on the frozen development T5 evidence) is an external comparator, not a Proposed candidate ceiling.

## 10. Decision tree

- If D4-AUDIT local identification improves and authoritative V125 T5 beats anchor-only without unacceptable PFV deterioration, freeze finite V125 for the next scientific stage.
- If D4-AUDIT local identification is good but T5 does not beat anchor, stop changing Step2 architecture and diagnose Step3/runtime/objective/decision-deadline behaviour.
- If D4-AUDIT local identification remains poor, stop architecture sweeps and classify the remaining blocker before any new SWMM: state/history sufficiency, local data support density, rainfall forecast information, or objective noise. Any second D4/active-learning round must be bounded and preregistered before outcomes.

## 11. Continuous MPC gate

Continuous L-BFGS-B / 109-actuator differentiable search is forbidden unless the frozen Project7 gate passes:

- TFV rank >= 0.70
- top1 >= 0.50
- TFV gradient sign >= 0.70
- gradient cosine >= 0.60

These are Project7 preregistered engineering/scientific thresholds, not universal literature thresholds. Do not lower them to enable continuous MPC.

## 12. Development boundaries

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
- no seed/hidden/head/loss sweep in the first D4 data-support experiment;
- no Global Peak objective/constraint.

## 13. Scientific claim boundary

The Proposed controller may claim only the best action selected within the generated engineering-feasible local candidate set around the current Sparse-RBC anchor. It must not claim global optimality in the full continuous 109-dimensional space.

Final performance claims must come from authoritative SWMM and untouched scientific Validation/Final only after the development method and policy are frozen.
