# Codex start here — Project7 V127 corrected continuous differentiable MPC

This is the **single canonical local execution entrypoint**. Read this file and `configs/step2_current_contract.json` before running anything. If V120–V126 prompts, old PRs, or `CODEX_START_HERE_V069.md` conflict, this file and the current contract win.

## 1. Frozen scientific target

Project7 is an idealized EPA-SWMM methodology testbed, not a field digital twin. Every 600 s:

1. frozen Step1 reconstructs the current full-network hydraulic state from sparse causal observations;
2. V127 Step2 predicts future hydraulic response under continuous actuator targets using only causal current state, causal rainfall forecast, current actuator flow, graph/physics and proposed future targets;
3. PyTorch autograd supplies derivatives of a differentiable cumulative flooding objective;
4. L-BFGS-B optimizes the exact `12 x 109 = 1308` target-fraction variables covering H120 while Step2 predicts H360;
5. only the first 600-s target is written to all 109 actuators in authoritative SWMM; then the system is observed and optimized again.

Whole-system cumulative TFV is primary. Frozen Priority8 PFV is a one-sided soft secondary objective. Global Peak is report-only and is finally measured by routing-step frozen-decision replay.

**RBC provides a warm start and safety fallback; differentiable MPC provides optimization.** RBC is not a Step2 reference, candidate ceiling or default Proposed policy.

Model rank/gradient scores are scientific evidence. They must be finite, causal and tied to the same final Step2 checkpoint, but they are not universal numerical switches that turn continuous MPC on/off. Hard fail-closed conditions are causal/input lineage, finite computation, engineering bounds/rate continuity, exact write/readback semantics, and completing within the 600-s decision period.

## 2. Git and code baseline

Use only the merged `origin/main` SHA given by the supervising response. Do not cherry-pick V125/V126 work.

```powershell
cd E:\RTC_sewer\Project7\repo
git status
git fetch origin --prune
# Preserve any local work first if needed, then:
git switch main
git reset --hard origin/main
git rev-parse HEAD
git status --short

python -m pytest -q
python -m py_compile `
  src/rtc/step2_differentiable_v127.py `
  src/rtc/step2_state_store_v127.py `
  src/rtc/step2_train_v127.py `
  src/rtc/step2_gradient_v127.py `
  src/rtc/checkpoint_v127.py `
  src/rtc/step3_mpc_v127.py `
  src/rtc/controller_v127.py `
  src/rtc/rule_baselines.py `
  scripts/build_step2_v127_causal_state_store.py `
  scripts/run_step2_v127.py `
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

Any failure is a correctness blocker. Fix the failing layer and add a regression test before continuing.

## 3. Existing authoritative data — use, do not regenerate blindly

Verify local assets by SHA/manifest, never by a guessed filename.

- D2 source: 4800 branches / 192 groups.
- D2 train-eligible: 3600 / 144 groups.
- D2 development-validation: 1200 / 48 groups; do **not** train on these in this development run.
- canonical D2 TrainFit/InternalHoldout: 112/32 groups.
- targeted D3: 3600 / 144 groups; TrainFit/InternalHoldout 112/32.
- legacy D3: historical non-canonical pool; do not silently concatenate.
- D4: 390 branches; FIT 269 / 33 groups / 10 rainfall groups; AUDIT 121 / 15 groups / 4 rainfall groups.

InternalHoldout, D4-AUDIT and D5-AUDIT are read-only scientific evidence. Validation/Final/Formal/Policy Lock remain untouched.

Important corrected contract: V127 model branches and authoritative cumulative node-flood-volume labels are always ordered **reference first, then candidates**. Never use raw shard row order for the objective labels.

## 4. Build online-equivalent causal Step1 state store

Historical shards contain SWMM checkpoint truth, but online Step2 does not. Build causal Step1 state for every required training/evidence checkpoint:

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

Require all identities unique and complete, 109 actuators, causal `development/train/no_control` history, no future hydraulic state.

## 5. Train corrected V127 Step2 from D2/D3/D4

```powershell
python scripts/run_step2_v127.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --out-dir <V127_ROOT>\step2_existing `
  --device cuda
```

Training semantics:

- D2: single-actuator hydraulic/Jacobian support;
- targeted D3: joint multi-actuator nonlinear response;
- D4-FIT: local physical-response support;
- Stage A: teacher-forced hydraulic transition and managed-flow learning;
- Stage B: H360 rollout. The **smooth cumulative node flooding/TFV proxy that online MPC differentiates is directly supervised by authoritative SWMM cumulative node flooding volume**. Hard clamp-based TFV remains the physical surrogate metric and continuous-vs-RBC predicted check.

The checkpoint must bind actual V127 Step2 source fingerprints and semantic graph topology/features. A checkpoint from the pre-correctness V127 contract is stale; retrain rather than bypass the loader.

## 6. Plan high-value D5 in the exact online MPC variable space

D5 is a gradient experiment, not a random candidate bank and not a legacy group-basis experiment.

```powershell
python scripts/plan_d5_gradient_v127.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --out-dir <V127_ROOT>\d5_plan `
  --max-checkpoints 48 `
  --directions-per-center 8
```

Default design:

- 48 outcome-blind TrainFit checkpoints;
- 3 centres/checkpoint: HOLD, Sparse-RBC warm start, broad continuous centre;
- 8 unit-L2 directions/centre in the exact `[12,109]` L-BFGS-B fraction tensor;
- direction families cover executed first move, persistent/late/temporal single-actuator effects, first-move multi-actuator, single-block spatial interaction, sparse and broader spatiotemporal interactions;
- one centre plus exact `+epsilon/-epsilon` for every direction;
- plus/minus variables are never clipped; epsilon is reduced or direction rejected if `[0,1]` bounds or the exact online sequential decoder destroy central symmetry;
- H120 free targets, terminal target held to H360;
- default 2448 SWMM branches and 1152 authoritative directional-gradient pairs;
- FIT/AUDIT split frozen by rainfall group before outcomes.

Review plan diagnostics, including actuator/block/coordinate coverage and pair symmetry, before SWMM. Do not redesign D5 after seeing TFV.

## 7. Build execution manifest and census

```powershell
python scripts/build_d5_execution_manifest_v127.py `
  --plan <D5_PLAN.csv> `
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

If local RAM/IO cannot safely sustain 16 workers, reduce workers only; do not alter the scientific manifest. Require zero missing prefix/INP/identity/endpoint failures. Verify H72x109 target arrays, exact paired 5-min frames in every 10-min block, <=0.5 target change, H120 terminal hold and exact fraction-space +/- symmetry.

## 8. Run authoritative D5 and compile SWMM gradients

Run the identical `rtc-run-d3-batch` command without `--census-only`. Resume only exact simulation identities; do not mutate the manifest. When all rows are terminal and valid:

```powershell
python scripts/build_d5_gradient_labels_v127.py `
  --execution-manifest <V127_ROOT>\D5_EXECUTION_MANIFEST.csv `
  --run-summary <D5_RUN_SUMMARY.csv> `
  --graph <FROZEN_GRAPH> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --out <V127_ROOT>\D5_DIRECTIONAL_GRADIENT_LABELS.csv
```

Truth is authoritative SWMM central difference `(TFV_plus - TFV_minus)/(2*epsilon)` with respect to a unit direction in the exact online 1308-variable fraction tensor.

## 9. D5-FIT gradient fine-tune; D5-AUDIT untouched

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

D5-FIT uses symmetric smooth-TFV finite differences so parameter training remains first-order and memory-stable. D5-AUDIT computes the true online quantity: autograd with respect to the centre `[12,109]` fraction tensor, through the same decoder as L-BFGS-B, dotted with the frozen direction. D5-AUDIT never trains.

The final D5 checkpoint/report must expose the exact final Step2 SHA and causal rainfall forecast contract.

## 10. Audit the same final checkpoint

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

D2 finite differences must stay strictly inside each InternalHoldout counterfactual group. Rank must be tie-aware. Record rank/pairwise/top1/TFV MAE/regret and D2/D5 gradient sign/cosine/MAE. Do not use these evidence sets for training.

## 11. Compile structurally valid continuous evidence

```powershell
python scripts/build_v127_continuous_gate.py `
  --ranking-report <V127_ROOT>\V127_RANKING_AUDIT.json `
  --d2-gradient-report <V127_ROOT>\d2_gradient_audit\D2_INTERNAL_HOLDOUT_GRADIENT_METRICS.json `
  --d5-gradient-report <V127_ROOT>\step2_d5\STEP2_V127_D5_GRADIENT_REPORT.json `
  --out <V127_ROOT>\V127_CONTINUOUS_EVIDENCE.json
```

Require all three reports to refer to the identical final Step2 SHA, all metrics finite and causal Step1/rainfall verified. Do **not** invent or relax a numerical threshold after seeing results. Weak rank/gradient evidence is a scientific limitation to report, not a hidden switch that changes the method into RBC.

## 12. Run Proposed plus six fixed baselines

For the fixed development comparison:

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
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --continuous-gate <V127_ROOT>\V127_CONTINUOUS_EVIDENCE.json `
  --out-dir <V127_ROOT>\seven_strategies `
  --device cuda `
  --lbfgsb-maxiter 30 `
  --optimizer-deadline-seconds 480 `
  --decision-runtime-budget-seconds 540 `
  --pfv-soft-margin-m3 100 `
  --pfv-penalty-weight 1
```

Formal strategy set is exactly:
1. Proposed V127 continuous differentiable MPC;
2. No-control;
3. Internal RTC;
4. Auto-RBC;
5. storage-volume EFD;
6. All-open;
7. All-closed.

All seven must share the same source-event identity, SWMM engine and observation/control clock. Final TFV/PFV come from authoritative cumulative SWMM node statistics. Report-only Global Peak comes from routing-step frozen-decision replay for every strategy, not the 300-s callback sample.

For Proposed record every decision's 109 targets, target-latch/readback diagnostics, source (`MPC_V127_CONTINUOUS` or `RBC_SAFETY_V127`), predicted hard/smooth TFV/PFV, gradient norm, L-BFGS-B iterations, optimizer elapsed time and deadline status. No post-score projection is allowed. The action scored for the first 600 s must be exactly the action written for that 600 s.

## 13. Interpretation rules

Continuous MPC remains the method even if development evidence is weak. Interpret results rather than changing the method after T5:

- If ranking/gradient evidence is poor, state that the surrogate is not yet scientifically trustworthy and diagnose the data/model layer using TrainFit-only or newly frozen outcome-blind development evidence.
- If gradients are good but continuous decisions mostly fall back to RBC, diagnose optimization convergence, surrogate hard-vs-smooth disagreement, forecast/horizon effects or runtime budget.
- If continuous actions execute but authoritative SWMM TFV is worse, diagnose surrogate objective bias/generalization. Do not tune on the evaluation event.
- Auto-RBC is a strong external comparator, never an action-space ceiling.

Do not claim success simply because code runs or because Proposed beats No-control on one event.

## 14. Forbidden during this development run

Do not:
- train on the 1200 D2 development-validation branches;
- train on InternalHoldout, D4-AUDIT or D5-AUDIT;
- use future realized rainfall or future SWMM hydraulic state online;
- use RBC as Step2 reference, Value target or action-space ceiling;
- change D5 plan/FIT-AUDIT roles after SWMM outcomes;
- alter model-quality criteria after seeing a control result to manufacture a claim;
- access Validation/Final/Formal/Policy Lock automatically.

## 15. Required final local report

Write `PROJECT7_V127_CONTINUOUS_MPC_DEVELOPMENT.json` and `.md` containing:
- exact Git SHA and clean-worktree status;
- frozen graph/Step1/sensor/cache/rainfall/state-store/checkpoint SHAs;
- actual D2/D3/D4/D5 branch/group/rainfall census;
- D5 plan/manifest/run/label SHAs and direct-variable coverage;
- Step2 hydraulic/objective training history;
- final ranking and D2/D5 gradient metrics;
- exact final Step2 SHA used by every evidence file;
- seven-strategy authoritative TFV/PFV/exact Global Peak/routing error table;
- continuous/RBC/deadline counts and optimizer runtime mean/p95/max;
- write/readback, continuity and score==execute violations;
- a professor-level conclusion on whether the current surrogate/gradients/control benefit are scientifically convincing.

Stop after development reporting. Do not automatically enter Validation/Final/Formal/Policy Lock.
