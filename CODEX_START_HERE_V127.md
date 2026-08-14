# Codex start here — Project7 V127 continuous differentiable MPC

This is the **single canonical local execution entrypoint** for the V127 restoration. If any older V120–V126 prompt or `CODEX_START_HERE_V069.md` conflicts with this file, this V127 file and `configs/step2_current_contract.json` are authoritative.

## 1. Scientific question

Project7 is an idealized EPA-SWMM methodology testbed. Every 10 min:

1. reconstruct the current hydraulic state from sparse causal observations with frozen Step1;
2. predict future hydraulic response under a continuous action sequence with the differentiable V127 Step2 world model;
3. optimize all 109 writable pump/orifice/weir targets continuously over H120 inside an H360 prediction horizon;
4. execute only the first 10-min target in authoritative SWMM;
5. observe again and repeat.

TFV is the primary objective. Priority8 PFV is one-sided soft secondary protection. Global Peak is report-only.

**RBC provides safety; differentiable MPC provides optimization.** Sparse-RBC is a warm start, fail-safe and comparator. It is not the Step2 reference and is not an action-space ceiling.

## 2. Git discipline

Run only from `main` at the exact merged V127 SHA supplied by the supervising ChatGPT response. Do not merge a local V125/V126 commit into V127. If local `main` diverges, create a backup ref, fetch `origin`, and reset/switch safely to the supplied `origin/main` SHA. Working tree must be clean before any scientific run.

Run first:

```powershell
python -m pytest -q
python -m py_compile `
  src/rtc/step2_differentiable_v127.py `
  src/rtc/step2_state_store_v127.py `
  src/rtc/step2_train_v127.py `
  src/rtc/step2_gradient_v127.py `
  src/rtc/step3_mpc_v127.py `
  src/rtc/controller_v127.py `
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

Any failure is a correctness blocker. Fix only the failing layer, add regression coverage, and stop downstream work until the full test/compile/diff baseline passes.

## 3. Frozen existing data

Use the local assets whose SHA256 matches the supervising prompt/history. Do not guess by filename.

- D2 authoritative source: 4800 branches / 192 groups.
- D2 frozen train-eligible: 3600 branches / 144 groups.
- D2 development-validation: 1200 branches / 48 groups; **never train on these during this development run**.
- targeted D3: 3600 branches / 144 groups; canonical TrainFit/InternalHoldout = 112/32 groups.
- legacy D3: approximately 1318 historical branches; non-canonical and unused unless a separate lineage audit proves compatibility.
- D4 V2 authoritative: 390 branches; FIT 269 branches / 33 groups / 10 rainfall groups; AUDIT 121 branches / 15 groups / 4 rainfall groups.

D4-AUDIT and InternalHoldout are read-only. Validation/Final/Formal/Policy Lock are forbidden during V127 development.

## 4. Stage A — build online-equivalent causal Step1 state store

Historical SWMM shards contain authoritative checkpoint state as `initial_state`. That is useful truth but is not available online. V127 Step2 must train on exactly the causal Step1 state it will receive online.

Locate and verify the frozen graph, Step1 checkpoint, sensors, train index, canonical D2/D3 cache, D4 FIT cache and D4 AUDIT cache. Then run:

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

Verify every required event/checkpoint is present, hashes are finite/unique, actuator count is 109, and only `development/train/no_control` causal histories were used. No new SWMM is run here.

## 5. Stage B — existing D2/D3/D4 differentiable surrogate training

Run:

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

The model inputs must be causal Step1 state + causal rainfall + current actuator flow + continuous future targets. Future SWMM hydraulics/flood volumes are labels only.

Training roles:
- D2: single-actuator hydraulic sensitivity / Jacobian support;
- targeted D3: coordinated multi-actuator nonlinear response;
- D4-FIT: local physical-response support;
- D4-AUDIT: read-only local audit.

Stage A of training is teacher-forced hydraulic transition/managed-flow learning. Stage B is H360 end-to-end flood-objective rollout training. Hard physical TFV is used for magnitude/reporting; smooth positive TFV is used for differentiable ordering/optimization to avoid clamp gradient dead zones.

Do not tune hidden size, seed, epochs or loss weights after seeing holdout results in the first V127 run.

## 6. Stage C — plan high-value D5 gradient data

D5 is not another random candidate bank. It directly measures control gradients.

```powershell
python scripts/plan_d5_gradient_v127.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --out-dir <V127_ROOT>\d5_plan `
  --max-checkpoints 48 `
  --directions-per-center 4
```

Expected scientific design:
- 48 outcome-blind TrainFit checkpoints;
- three centres/checkpoint: HOLD, Sparse-RBC warm start, broad continuous manifold;
- four normalized spatiotemporal directions/centre;
- one center + exact `+epsilon/-epsilon` per direction;
- H120 free control then terminal target held to H360;
- expected 1296 branches and 576 gradient directions;
- FIT/AUDIT rainfall groups frozen before SWMM outcomes;
- if engineering decoding breaks central symmetry, epsilon is reduced or the direction is rejected.

Review the plan before SWMM. Do not alter candidates after seeing outcomes.

## 7. Stage D — build D5 execution manifest and census

```powershell
python scripts/build_d5_execution_manifest_v127.py `
  --plan <D5_PLAN.csv> `
  --checkpoints <FROZEN_CHECKPOINT_METADATA.csv> `
  --graph <FROZEN_GRAPH> `
  --out <V127_ROOT>\D5_EXECUTION_MANIFEST.csv
```

Verify H72 x 109 actions, paired 5-min targets exactly equal inside every 10-min block, max target change <=0.5, exactly one center per center ID and one +/- pair per direction.

Then use the existing validated runner in census-only mode:

```powershell
rtc-run-d3-batch `
  --manifest <D5_EXECUTION_MANIFEST.csv> `
  --out-dir <V127_ROOT>\d5_swmm `
  --control-block-seconds 600 `
  --stride-seconds 300 `
  --workers 16 `
  --swmm-threads-per-process 1 `
  --asset-root <EXISTING_SIMULATION_ASSET_ROOT> `
  --census-out <V127_ROOT>\D5_CENSUS.json `
  --census-only
```

Require endpoint/identity/missing-prefix/missing-INP failures = 0. If census fails, stop.

## 8. Stage E — execute authoritative D5 SWMM

Only after census passes, rerun the same command without `--census-only`. Do not modify the manifest. Allow exact simulation-identity reuse/resume only. Verify all requested branches are terminal, same-prefix/asset lineage is valid, routing error acceptable, and no NaN/Inf exists.

Then build authoritative directional-gradient labels:

```powershell
python scripts/build_d5_gradient_labels_v127.py `
  --execution-manifest <D5_EXECUTION_MANIFEST.csv> `
  --run-summary <D5_RUN_SUMMARY.csv> `
  --graph <FROZEN_GRAPH> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --out <V127_ROOT>\D5_DIRECTIONAL_GRADIENT_LABELS.csv
```

Each gradient label must be `(TFV_plus - TFV_minus)/(2*epsilon)` from authoritative SWMM. D5-AUDIT remains untouched.

## 9. Stage F — D5 FIT gradient fine-tune and untouched D5-AUDIT

```powershell
python scripts/run_step2_v127_d5_gradient.py `
  --graph <FROZEN_GRAPH> `
  --base-cache-manifest <CANONICAL_V60_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V127_EXISTING_DATA_CHECKPOINT> `
  --d5-execution-manifest <D5_EXECUTION_MANIFEST.csv> `
  --d5-gradient-labels <D5_DIRECTIONAL_GRADIENT_LABELS.csv> `
  --out-dir <V127_ROOT>\step2_d5 `
  --device cuda
```

D5-FIT uses symmetric smooth-TFV finite differences to train the model with ordinary first-order parameter gradients. D5-AUDIT evaluates the actual autograd directional gradient used online. D5-AUDIT never trains.

## 10. Stage G — final untouched ranking and gradient audits

Ranking:

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
```

InternalHoldout D2 gradient audit:

```powershell
python scripts/audit_step2_v127_d2_gradients.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V127_D5_CHECKPOINT> `
  --out-dir <V127_ROOT>\d2_gradient_audit `
  --device cuda
```

## 11. Stage H — continuous-MPC gate

```powershell
python scripts/build_v127_continuous_gate.py `
  --ranking-report <V127_RANKING_AUDIT.json> `
  --d2-gradient-report <D2_INTERNAL_HOLDOUT_GRADIENT_METRICS.json> `
  --d5-gradient-report <STEP2_V127_D5_GRADIENT_REPORT.json> `
  --out <V127_ROOT>\V127_CONTINUOUS_GATE.json
```

Continuous MPC is authorized only if all pass:
- InternalHoldout D3 rank >= 0.70;
- InternalHoldout D3 top1 >= 0.50;
- InternalHoldout D2 gradient sign >= 0.70;
- InternalHoldout D2 gradient cosine >= 0.60;
- D5-AUDIT gradient sign >= 0.70;
- D5-AUDIT gradient cosine >= 0.60;
- causal Step1 state and causal rainfall verified.

Do not lower thresholds after seeing results. If gate fails, stop before T5 and classify the failed evidence layer.

## 12. Stage I — V127 continuous T5 and seven strategies

Only with a passed gate, run the exact frozen development event comparison:

```powershell
python scripts/run_seven_strategies_v127.py `
  --inp <FROZEN_T5_INP> `
  --event-id T5_D180_chicago `
  --sensors <FROZEN_SENSORS> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --config <FROZEN_RUNTIME_CONFIG> `
  --native-controls-template <FROZEN_NATIVE_CONTROLS_TEMPLATE> `
  --graph <FROZEN_GRAPH> `
  --step1 <FROZEN_STEP1> `
  --step2 <V127_D5_CHECKPOINT> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --continuous-gate <V127_CONTINUOUS_GATE.json> `
  --out-dir <V127_ROOT>\t5_seven_strategies `
  --device cuda `
  --lbfgsb-maxiter 30 `
  --optimizer-deadline-seconds 480 `
  --decision-runtime-budget-seconds 540 `
  --pfv-soft-margin-m3 100 `
  --pfv-penalty-weight 1
```

Seven strategies are exactly: Proposed V127, No-control, Internal RTC, Auto-RBC, EFD, All-open, All-closed.

Report authoritative SWMM TFV, TFV reduction vs No-control, frozen Priority8 PFV and change vs No-control, Global Peak, routing error, decision count, continuous-MPC count, RBC safety-fallback count, optimizer deadline fallbacks, mean/p95/max decision runtime, write/readback failures, continuity violations and score==execute violations.

## 13. Interpretation

The immediate Project7 success criterion is not “beat RBC by construction.” Proposed must optimize a continuous action space and demonstrate positive authoritative TFV benefit under the fixed development comparison without material Priority8 PFV deterioration. Auto-RBC is a strong external comparator; it is not a candidate ceiling or training target.

If the continuous gate passes but T5 underperforms, do not retune on T5. Diagnose surrogate objective bias, forecast uncertainty, horizon/terminal effects and optimization convergence with new outcome-blind development evidence.

## 14. Forbidden

During this development run do not:
- train on the 1200 D2 development-validation branches;
- train on InternalHoldout, D4-AUDIT or D5-AUDIT;
- access Validation/Final/Formal/Policy Lock;
- use future realized rainfall or future SWMM state online;
- use Sparse-RBC as Step2 reference or action-space ceiling;
- restore V125/V126 anchor-relative finite policy as current method;
- lower continuous gates after observing results;
- alter D5 FIT/AUDIT or action plan after SWMM outcomes.

## 15. Required final local report

Write `PROJECT7_V127_CONTINUOUS_MPC_DEVELOPMENT.json` and `.md` with exact Git SHA, all frozen asset SHA256s, causal-state store SHA, D5 plan/manifest/run/label SHAs, D2/D3/D4/D5 data counts, all Step2 ranking and gradient metrics, gate verdict, seven-strategy table, continuous/fallback/deadline counts, runtime/execution evidence and a single final development verdict. Do not enter Validation/Final/Formal/Policy Lock automatically.
