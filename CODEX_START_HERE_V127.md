# Codex start here — Project7 V127 final correctness path

This is the single canonical local execution guide for Project7 V127 development. Read this file together with `configs/step2_current_contract.json`. Historical V120–V126 prompts/PRs are evidence and forensics only when they conflict with the current contract.

> **Current Step2 routing:** the only canonical base-training entrypoint is `scripts/run_step2_v127_control_streaming.py`. `scripts/run_step2_v127.py` is a preserved historical implementation and must not be used to produce a current V127/V128 Step2 checkpoint.

## Scientific target

Project7 is an idealized EPA-SWMM methodology testbed. Every 600 s:

1. frozen Step1 reconstructs current full-network hydraulic state from sparse causal observations;
2. V127 Step2 rolls an action-conditioned differentiable hydraulic world model for H360 using causal current state, causal rainfall forecast, current actuator flow, graph/physics and proposed future targets;
3. PyTorch autograd supplies action gradients;
4. L-BFGS-B optimizes the exact `12 x 109 = 1308` H120 target-fraction variables;
5. only the first 600-s target is written to all 109 actuators, then SWMM is observed and optimization repeats.

Whole-system cumulative TFV is primary. Priority8 PFV is a one-sided soft secondary objective. Global Peak is report-only. RBC is a warm start, safety fallback and comparator; never a Step2 Value reference or action-space ceiling.

Model-quality scores are scientific evidence, not arbitrary runtime switches. Hard runtime correctness means causal inputs, finite computation, valid model/input semantics, engineering target bounds/slew, exact target write/readback, score==execute and completion inside the 600-s decision period.

## 1. Git and code baseline

Use only the final merged `origin/main` SHA supplied by the supervising response. Do not run scientific training from PR branches or old local main.

```powershell
cd E:\RTC_sewer\Project7\repo
git status
git fetch origin --prune
# Preserve local work first if needed.
git switch main
git reset --hard origin/main
git rev-parse HEAD
git status --short
```

The printed SHA must exactly equal the final SHA supplied by ChatGPT and the worktree must be clean.

Then run:

```powershell
python -m pytest -q
python -m py_compile `
  src/rtc/step2_differentiable_v127.py `
  src/rtc/step2_state_store_v127.py `
  src/rtc/step2_train_v127_control.py `
  src/rtc/step2_train_v127_streaming.py `
  src/rtc/step2_gradient_v127.py `
  src/rtc/checkpoint_v127.py `
  src/rtc/step3_mpc_v127.py `
  src/rtc/controller_v127.py `
  src/rtc/runtime.py `
  src/rtc/runtime_controller_guard.py `
  src/rtc/rule_baselines.py `
  src/rtc/execution_audit_v127.py `
  scripts/build_step2_v127_causal_state_store.py `
  scripts/run_step2_v127_control_streaming.py `
  scripts/plan_d5_gradient_v127.py `
  scripts/build_d5_execution_manifest_v127.py `
  scripts/build_d5_gradient_labels_v127.py `
  scripts/run_step2_v127_d5_gradient.py `
  scripts/audit_step2_v127_d2_gradients.py `
  scripts/audit_step2_v127_ranking.py `
  scripts/build_v127_continuous_gate.py `
  scripts/run_policy_v127.py `
  scripts/run_seven_strategies_v127.py
git diff --check
```

Do not bypass a real code/test failure. A stale historical test expectation should be updated only if the current scientific contract intentionally supersedes it; never delete test coverage just to make CI green.

## 2. Existing authoritative data

Locate local assets automatically under `E:\RTC_sewer\Project7\study_v069` and prior development outputs; do not ask for each path separately if it can be discovered from manifests.

Expected census to verify from files, not assume:

- D2 source: 4800 branches / 192 groups;
- D2 train-eligible: 3600 / 144 groups;
- D2 development-validation: 1200 / 48 groups, never train during this development run;
- canonical D2 TrainFit/InternalHoldout: 112/32 groups;
- targeted D3: 3600 / 144 groups, TrainFit/InternalHoldout 112/32;
- legacy D3: non-canonical historical pool, never silently concatenate;
- D4: 390 branches, FIT 269 / 33 groups / 10 rainfall groups, AUDIT 121 / 15 groups / 4 rainfall groups.

V127 branch tensors and authoritative SWMM cumulative node-flood-volume labels are always ordered `reference first, then candidates`. Never revert to raw shard row order.

Raw file SHA values are provenance/reproducibility aids. Do not fail training merely because semantically identical files were reserialized or comments changed. Hard compatibility is based on the actual model/input semantics required by the current code. Exact final Step2 identity across ranking/D2/D5 evidence remains mandatory because those reports describe a particular trained model.

## 3. Build the new causal Step1-state store

Because the final state-store contract uses semantic Step1 parameter identity and ordered sensor identity, rebuild the store once even if an older V127 state store exists.

```powershell
python scripts/build_step2_v127_causal_state_store.py `
  --graph <FROZEN_GRAPH> `
  --step1 <FROZEN_STEP1> `
  --sensors <FROZEN_SENSORS> `
  --train-index <FROZEN_TRAIN_INDEX> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --out <V127_ROOT>\STEP2_V127_CAUSAL_STATE_STORE.npz `
  --device cuda
```

Require unique/complete event-checkpoint identities, 109 actuators, `development/train/no_control` causal histories, no future hydraulic truth as model input. Preserve raw SHA provenance but use semantic Step1 state-dict and ordered sensor identities for train/deploy compatibility.

Do not retrain Step1 unless its own acceptance evidence or a real incompatibility fails. Step1 is frozen for this V127 run.

## 4. Retrain the final V127 Step2 once

All checkpoints produced before `PROJECT7_V127_STEP2_CHECKPOINT_V4_SEMANTIC_COMPATIBILITY` are stale. Retrain from the corrected label/objective contract rather than bypassing the loader. On the 16-GB RAM / RTX-4060 development workstation, use the memory-safe full-coverage control curriculum below; do **not** use the historical `scripts/run_step2_v127.py` implementation.

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python scripts/run_step2_v127_control_streaming.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --out-dir <V127_ROOT>\step2_existing `
  --device cuda
```

Training roles:

- D2: single-actuator hydraulic sensitivity/Jacobian support;
- targeted D3: coordinated multi-actuator nonlinear response;
- D4-FIT: local physical-response support;
- Stage A: full-coverage teacher-forced hydraulic transition + managed actuator flow;
- Stage B0: H60/H120 autoregressive truncated-rollout curriculum to reduce compounding drift;
- Stage B: H360 control-oriented objective with full candidate coverage using CPU streaming/GPU microbatches.

Stage-B objective semantics are important:

- hard clamp-based TFV is the physical surrogate and learns absolute authoritative SWMM cumulative TFV magnitude;
- the differentiable Softplus node-volume/TFV proxy is trained on same-prefix counterfactual action differences and ordering, not absolute SWMM volume. Its common zero-rate Softplus offset therefore cancels rather than biasing absolute flood-volume training;
- retained hydraulic states constrain long-horizon rollout drift.

Do not perform an architecture/hidden-size sweep before this corrected data/objective run. Historical V124 already showed that more representational capacity alone did not solve action ranking. First isolate the corrected supervision and gradient evidence.

## 5. Plan D5 before running any new SWMM

D5 is a small high-information gradient experiment, not a random candidate bank and not a 1308-coordinate sweep.

```powershell
python scripts/plan_d5_gradient_v127.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --out-dir <V127_ROOT>\d5_plan `
  --max-checkpoints 24 `
  --directions-per-center 6
```

The defaults are a **maximum budget**, not a target count that must be filled:

- at most 24 outcome-blind TrainFit checkpoints;
- up to 3 centres/checkpoint: HOLD, Sparse-RBC warm start, local non-RBC exploration near the HOLD↔RBC operating corridor;
- 6 directions per retained centre;
- maximum 936 SWMM branches and maximum 432 gradient pairs;
- actual counts may be lower when centres/actions collapse or duplicate after the exact decoder. This is desirable; do not add random replacements merely to reach 936.

Checkpoint selection is rainfall-balanced and uses causal state/command descriptors plus farthest-point diversity to avoid redundant hydraulic states. It must not inspect D5 outcomes.

Direction families are:

1. first-move single actuator;
2. persistent single actuator;
3. temporal single actuator;
4. first-move multi-actuator;
5. single-block spatial interaction;
6. sparse spatiotemporal interaction with first-move content.

All directions live in the exact `[12,109]` fraction space used online. Plus/minus variables are never clipped; reduce epsilon or reject the direction if bounds or the sequential decoder destroy central symmetry. Centre executability is independent from central-pair feasibility.

Before SWMM, review the plan for information value. Do not impose arbitrary score thresholds. Stop only for a real degeneracy such as duplicate physical actions surviving deduplication, missing required direction families, zero/non-finite physical displacement, broken +/- symmetry, wrong H120/H360 structure, or clearly collapsed checkpoint diversity. Report actuator/block/coordinate coverage and first-move/physical displacement. D5 does not need to touch all 1308 coordinates: D2 already provides local single-actuator/boundary finite-difference information; D5 tests representative continuous interior/projected gradient directions relevant to MPC.

## 6. Build D5 execution manifest and census

```powershell
python scripts/build_d5_execution_manifest_v127.py `
  --plan <V127_ROOT>\d5_plan\STEP2_V127_D5_GRADIENT_PLAN.csv `
  --checkpoints <FROZEN_CHECKPOINT_METADATA.csv> `
  --graph <FROZEN_GRAPH> `
  --out <V127_ROOT>\D5_EXECUTION_MANIFEST.csv

rtc-run-d3-batch `
  --manifest <V127_ROOT>\D5_EXECUTION_MANIFEST.csv `
  --out-dir <V127_ROOT>\d5_swmm `
  --control-block-seconds 600 `
  --stride-seconds 300 `
  --workers 16 `
  --swmm-threads-per-process 1 `
  --asset-root <EXISTING_SIMULATION_ASSET_ROOT> `
  --census-out <V127_ROOT>\D5_CENSUS.json `
  --census-only
```

Reduce workers if RAM/IO cannot sustain 16; do not alter the scientific manifest to fix a resource problem. Require no missing prefix/INP/identity/endpoint failure. Verify H72x109 action arrays, paired 5-min frames within each 10-min command, target-command slew <=0.5, H120 free control, H120→H360 terminal hold, and fraction/decoded central symmetry.

Only after census succeeds, run the identical batch command without `--census-only`. Resume exact simulation identities only.

## 7. Build authoritative D5 gradient truth

```powershell
python scripts/build_d5_gradient_labels_v127.py `
  --execution-manifest <V127_ROOT>\D5_EXECUTION_MANIFEST.csv `
  --run-summary <D5_RUN_SUMMARY.csv> `
  --graph <FROZEN_GRAPH> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --out <V127_ROOT>\D5_DIRECTIONAL_GRADIENT_LABELS.csv
```

Truth is authoritative SWMM `(TFV_plus - TFV_minus)/(2*epsilon)` along a unit direction in the exact online fraction tensor. D5 central differences mainly validate interior continuous directions; one-sided/boundary sensitivity evidence from D2 should be reported separately rather than forcing invalid central probes at active bounds.

## 8. D5-FIT gradient fine-tuning and untouched D5-AUDIT

```powershell
python scripts/run_step2_v127_d5_gradient.py `
  --graph <FROZEN_GRAPH> `
  --base-cache-manifest <CANONICAL_V60_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V127_EXISTING_DATA_CHECKPOINT> `
  --d5-execution-manifest <V127_ROOT>\D5_EXECUTION_MANIFEST.csv `
  --d5-gradient-labels <V127_ROOT>\D5_DIRECTIONAL_GRADIENT_LABELS.csv `
  --out-dir <V127_ROOT>\step2_d5 `
  --device cuda
```

D5-FIT uses symmetric smooth-TFV differences and ordinary first-order parameter training. D5-AUDIT never trains and evaluates the quantity used online: autograd with respect to the centre `[12,109]` fraction tensor through the same decoder, dotted with the frozen direction.

## 9. Audit the same final Step2 model

```powershell
python scripts/audit_step2_v127_ranking.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V127_D5_CHECKPOINT> `
  --out <V127_ROOT>\V127_RANKING_AUDIT.json `
  --device cuda

python scripts/audit_step2_v127_d2_gradients.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V127_D5_CHECKPOINT> `
  --out-dir <V127_ROOT>\d2_gradient_audit `
  --device cuda
```

D2 finite differences must stay inside each exact InternalHoldout counterfactual group and use the actual D2 action-sequence direction. Report tie-aware rank, pairwise, top1, TFV MAE/regret and D2/D5 gradient sign/cosine/MAE. Never train on InternalHoldout/D4-AUDIT/D5-AUDIT.

Compile evidence:

```powershell
python scripts/build_v127_continuous_gate.py `
  --ranking-report <V127_ROOT>\V127_RANKING_AUDIT.json `
  --d2-gradient-report <V127_ROOT>\d2_gradient_audit\D2_INTERNAL_HOLDOUT_GRADIENT_METRICS.json `
  --d5-gradient-report <V127_ROOT>\step2_d5\STEP2_V127_D5_GRADIENT_REPORT.json `
  --out <V127_ROOT>\V127_CONTINUOUS_EVIDENCE.json
```

The reports must describe the same final Step2 checkpoint and finite causal evidence. Do not invent rank/cosine thresholds to silently replace the method by RBC. Weak evidence is a scientific limitation to report.

## 10. Runtime semantics

V127 target slew is defined between consecutive **supervisory target settings**. Physical `current_setting` may lag due to device dynamics and is an input/diagnostic, not a second slew anchor. The controller must not project a scored V127 action after optimization. Score==execute is mandatory.

Every authoritative run is post-audited by `audit_target_write_readback_v127`: the 109 requested settings in each decision log must equal compact SWMM `target_setting` at the same elapsed epoch. Current setting is not used as target-write acceptance.

Rainfall online uses causal persistence/decay only (`history_steps_for_level=1`, `decay_per_step=0.92`, scenario 1.0). The runtime does not need the training rainfall-store file; it must reproduce the forecast algorithm semantics recorded by Step2.

## 11. Proposed plus six fixed baselines

Run one fixed development comparison with:

```powershell
python scripts/run_seven_strategies_v127.py `
  --inp <FROZEN_DEVELOPMENT_INP> `
  --event-id <FROZEN_DEVELOPMENT_EVENT_ID> `
  --sensors <FROZEN_SENSORS> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --config <FROZEN_RUNTIME_CONFIG> `
  --native-controls-template <FROZEN_NATIVE_CONTROLS_TEMPLATE> `
  --graph <FROZEN_GRAPH> `
  --step1 <FROZEN_STEP1> `
  --step2 <V127_D5_CHECKPOINT> `
  --continuous-gate <V127_ROOT>\V127_CONTINUOUS_EVIDENCE.json `
  --out-dir <V127_ROOT>\seven_strategies `
  --device cuda `
  --lbfgsb-maxiter 30 `
  --optimizer-deadline-seconds 480 `
  --decision-runtime-budget-seconds 540 `
  --pfv-soft-margin-m3 100 `
  --pfv-penalty-weight 1
```

There is intentionally **no `--causal-store` runtime argument**.

Formal comparison set is exactly:

1. Proposed V127 continuous differentiable MPC;
2. No-control;
3. Internal RTC;
4. Auto-RBC;
5. storage-volume EFD;
6. All-open;
7. All-closed.

No-control disables supervisory RTC but preserves intrinsic SWMM facility physics. Internal RTC uses the frozen native controls. Auto-RBC/EFD generate the next command from the current supervisory target latch, not a lagged physical current setting. EFD filling degree is storage volume/capacity from FUNCTIONAL/TABULAR geometry. All Python comparator commands use the same target-command slew semantics as Proposed. Internal native rules remain an external comparator and may naturally have their own native rule evaluation timing; do not falsify their method identity to force a Python command cadence.

All seven final rows must share source-event identity, SWMM engine, common observation/control clock where applicable, pass same-epoch target-write/readback audit, and use authoritative cumulative SWMM node statistics for TFV/PFV. Final Global Peak is routing-step frozen-decision replay, report-only; the 300-s sampled peak is diagnostic.

## 12. Interpretation and stop rule

Do not retune on the fixed development comparison event.

- poor ranking/gradient evidence => surrogate/action-effect identification remains weak;
- good gradients but many RBC fallbacks => inspect optimizer convergence, hard-vs-smooth disagreement, forecast/horizon and runtime budget;
- continuous actions execute but authoritative SWMM TFV worsens => surrogate control generalization/objective bias;
- runtime approaches 600 s => method is not yet real-time even if hydraulic benefit is good.

Do not automatically retrain Step1, expand D5, enlarge the neural network, add Global Peak penalties or introduce new safety thresholds after seeing one event. Diagnose the failing layer first.

## 13. Development boundaries

Do not train on D2 development-validation, InternalHoldout, D4-AUDIT or D5-AUDIT. Do not use future realized rainfall or future SWMM hydraulic state online. Do not change D5 checkpoint/action selection after seeing D5 outcomes. Do not access Validation, Final, Formal or Policy Lock automatically.

## 14. Required final development report

Write `PROJECT7_V127_CONTINUOUS_MPC_DEVELOPMENT.json` and `.md` with:

- exact final Git SHA and clean worktree;
- discovered frozen assets and provenance plus semantic Step1/sensor/graph identities;
- D2/D3/D4 census;
- D5 actual (not assumed maximum) checkpoint/centre/pair/branch counts, deduplication, diversity, direction-family/actuator/block/coordinate coverage, epsilon and physical/first-move displacement diagnostics;
- Stage-A and Stage-B training history;
- final Step2 checkpoint identity;
- ranking, D2 gradient and D5-AUDIT gradient evidence;
- seven-strategy authoritative TFV, TFV reduction vs No-control, Priority8 PFV/change, exact Global Peak and routing error;
- continuous/RBC/deadline counts and optimizer runtime mean/p95/max;
- target-write/readback, target-command continuity, physical tracking-lag and score==execute evidence;
- a professor-level conclusion on Step1 adequacy, hydraulic rollout, TFV/action-effect prediction, gradient credibility, continuous-action use, real-time feasibility and whether the current development evidence is scientifically convincing for a Water Research methodology paper.

Stop after this development report. Do not enter Validation/Final/Formal/Policy Lock automatically.
