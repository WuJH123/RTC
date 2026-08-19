# Project7 Direct-TFV — authoritative main workflow

This file is the single startup guide for local Codex after the web/GitHub review cycle.

## 0. Collaboration contract

The working model is deliberate:

1. **Web GPT** reviews the scientific logic and edits/merges GitHub.
2. **Local Codex** synchronizes `main`, runs the exact commands, monitors Windows/SWMM/CUDA, and reports evidence/errors.
3. The user returns the run log to Web GPT.
4. Web GPT fixes GitHub; local Codex then fast-forwards `main` and resumes.

Do not create an independent local scientific fork. Do not choose a historical V8/V9/V10/V11/V12 branch by filename. `origin/main` is authoritative.

Before every run:

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin --prune
git switch main
git pull --ff-only
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

If local `main` is not a clean fast-forward of `origin/main`, stop before expensive work and report the exact diff/status. Never `git clean -fd`, `reset --hard`, or pop an old stash blindly.

## 1. Research question is frozen

Every 600 s:

`causal sparse observations -> Step1 state reconstruction -> Direct-TFV value/MPC -> 109-D supervisory target -> execute first 10 min -> SWMM -> observe again`.

Frozen physical/control contract:

- model/observation step: 300 s;
- control update: 600 s;
- prediction horizon: H360;
- free control horizon: H120 = 12 control blocks;
- execute only the first 10-min target;
- 109 writable actuators;
- target movement <= 0.5 per update plus physical min/max;
- supervisory slew anchor = previous `target_setting`, not physical `current_setting`;
- unchanged facilities copy the previous commanded target exactly;
- `HOLD` = latch previous supervisory target;
- primary objective = system-wide cumulative TFV only;
- Priority8 PFV and Global Peak = report only;
- authoritative truth = SWMM;
- no future realized rainfall/state/flooding/Internal trajectory online;
- no online SWMM candidate search;
- no Auto-RBC/EFD imitation or warm start.

## 2. What V8–V12 established

The historical versions are evidence, not alternative current entrypoints.

- V8/V6 q95 joint support made continuous optimization much more reliable but conservative.
- V9 policy-matched admission reduced over-conservatism but did not beat strong rule baselines.
- V10 calibrated the wrong counterfactual (`H10 new -> old target H350`) and collapsed to all-HOLD.
- V11 fixed the real supervisory target-latch semantics and restored actions.
- V12 added a causal rainfall scenario ensemble `(0.8, 1.0, 1.2)`, history=3, decay=0.92 and scenario-mean TFV scoring. Exact accepted H360 replay was 9/9 sign-correct with 0/9 false-beneficial, but complete closed-loop TFV beat No-control on only 1/3 Development probes and Auto-RBC on 0/3.

Therefore the current scientific bottleneck is **OPEN_LOOP_VALUE_VS_RECEDING_CONTROL_MISMATCH**, not another scalar-margin problem.

## 3. Stable base versus current Development experiment

The stable engineering base remains:

- frozen Step1;
- frozen Direct-TFV V5 base Step2;
- q95 support;
- all-109 screening;
- target-latch first-move refinement/pruning;
- V12 causal rainfall scenario mean;
- memory-safe runtime telemetry graph release.

The **current Development experiment** adds a receding-policy-return critic. It does not replace the stable base until evidence passes.

Policy-return estimand:

`candidate H10 -> frozen continuation policy`

minus

`HOLD H10 -> the same frozen continuation policy`.

The two SWMM branches must share the exact authoritative prefix and the exact same continuation policy after H10.

## 4. Cheap gates before any expensive run

```powershell
python -m pip install -e ".[dev,swmm]"
python -m compileall -q src scripts tests
python scripts/lint_current_surface.py
python -m pytest -q tests/test_direct_tfv_first_move.py tests/test_direct_tfv_first_move_cli.py tests/test_direct_tfv_robust_rainfall.py tests/test_direct_tfv_policy_return.py tests/test_current_step2_routing.py
python -m pytest -q
```

Then run `--help` for every script you are about to invoke. Never invent CLI arguments from memory.

## 5. Do not regenerate generic D3 for first-move calibration

The candidate-free path is mandatory:

```text
D0 no-control causal prefix
  -> build_direct_tfv_first_move_context_current.py
  -> FirstMoveCalibrationContextStore
  -> HOLD + exactly one refined candidate per rainfall group
  -> exact authoritative SWMM branches
  -> matched admission
```

The first-move context store must report:

- `candidate_rows_used=false`;
- `generic_d3_candidate_dependency=false`;
- `causal_future_rainfall_used=false`.

If any script asks for generic D3 candidate rows as a context prerequisite, stop: that is a regression.

## 6. V12 matched calibration/runtime regression path

V12 is diagnostic/stable-base evidence, not a new acceptance event after its results have been read.

Candidate-free V12 panel:

```powershell
python scripts/design_direct_tfv_robust_rainfall_first_move_calibration_current.py --help
python scripts/merge_direct_tfv_robust_rainfall_first_move_panel_shards.py --help
python scripts/calibrate_direct_tfv_robust_rainfall_first_move_admission_current.py --help
python scripts/run_policy_direct_tfv_robust_rainfall_development.py --help
```

V12 admission must be bound to:

- `PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V10_CAUSAL_RAINFALL_SCENARIO_MEAN`;
- causal rainfall scenario contract;
- V12 behavioral SHA;
- Step2 SHA;
- q95 sequence-support SHA.

A V11 single-scenario admission must fail closed in V12.

## 7. Policy-return data roles

Before reading any paired branch outcome, preregister rainfall groups into four disjoint sets:

1. `policy_return_train`: at least 48 independent rainfall groups;
2. `policy_return_validation`: at least 12 independent rainfall groups for model selection only;
3. `policy_return_calibration`: at least 24 independent rainfall groups for one-sided conformal calibration only;
4. new Development probes: separate from all three sets.

Do not use Validation/Final/Formal/Policy Lock data.

T5, T8, T10, T20, T30, T80, P15/P35/P75, previous V10/V11/V12 probes and any already-observed mechanism events may be used only if explicitly downgraded to Development training/diagnostic roles; once used, never call them independent acceptance evidence again.

## 8. Freeze the parent policy before generating policy-return labels

Iteration 0 uses parent `pi_0 = frozen V12`.

For every selected rainfall group:

1. run the frozen parent policy closed loop and save its decision JSONL;
2. select decision indices by a forcing/state/action rule fixed before reading paired truth;
3. for each selected index run:

```powershell
python scripts/run_direct_tfv_policy_return_pair_current.py --help
```

with:

```text
--continuation-kind v12
```

Each pair produces:

- exact CANDIDATE branch;
- exact HOLD branch;
- identical-prefix verification;
- the same frozen continuation policy after H10;
- causal Step1 state at the branch point;
- causal rainfall scenarios at the branch point;
- authoritative `candidate TFV - HOLD TFV` label.

Do not infer this label from the old open-loop H360 replay.

## 9. Compile and train the policy-return critic

Compile role-pure datasets:

```powershell
python scripts/compile_direct_tfv_policy_return_dataset_current.py --help
```

Then train:

```powershell
python scripts/train_direct_tfv_policy_return_current.py --help
```

The first policy-return experiment intentionally keeps the same `DirectFacilityTFVValueModel` architecture and initializes from frozen V5. The first scientific change is the target/estimand, not network size.

Primary validation metrics are event-balanced:

- policy-return MAE;
- sign accuracy;
- false-beneficial rate;
- false-reject rate;
- within-group ranking.

Do not select a checkpoint from calibration labels.

## 10. Freeze critic, score untouched calibration, then calibrate

```powershell
python scripts/score_direct_tfv_policy_return_calibration_current.py --help
python scripts/calibrate_direct_tfv_policy_return_admission_current.py --help
```

The critic is frozen before calibration scoring. Admission uses rainfall-group one-sided residuals normalized by `sqrt(actual changed-facility count)`.

The old V11/V12 first-move margin and generic-D3 floor must not secretly control policy-return execution.

## 11. Run policy-return closed loop

```powershell
python scripts/run_policy_direct_tfv_policy_return_development.py --help
```

Run only preregistered, previously unread Development probes.

For every event report:

- Proposed TFV;
- No-control;
- Internal RTC;
- Auto-RBC;
- storage-volume EFD;
- All-open;
- All-closed;
- PFV report only;
- Global Peak report only;
- ACTION/HOLD count;
- actual changed-K distribution;
- runtime p50/p95/max;
- target write/readback;
- score==execute;
- support/engineering violations;
- routing error;
- fallback/deadline count.

Do not weaken any baseline because Proposed loses.

## 12. Policy iteration, not one-shot overfitting

If `pi_1` materially differs from frozen V12, the labels are Q-like values under `pi_0`, not yet a fixed-point value for `pi_1`.

Generate a new role-disjoint Development round using:

```text
--continuation-kind policy-return
--policy-return-checkpoint <Q_pi0 checkpoint>
--policy-return-admission <pi1 admission>
```

This evaluates `pi_1`, learns `Q^{pi_1}`, and defines `pi_2`.

Repeat only while Development evidence shows a material policy shift. Stop when preregistered fixed-point audits show stable action agreement and policy-return residuals. Do not tune on the final new probes.

## 13. Resource rules

Local target: RTX 4060 Laptop 8 GB, RAM 16 GB.

- One GPU training process at a time.
- Policy-return paired replays contain neural continuation MPC; do not blindly launch 24 GPU copies.
- Benchmark 1 then 2 concurrent pair runners. Increase only with actual VRAM/RAM evidence.
- Pure independent SWMM branches without neural continuation may use higher process counts if their real `--help` supports it.
- One SWMM simulation per Python process.
- Set child BLAS/OpenMP thread counts to 1 during process-level parallelism.
- Event-matrix parallelism is throughput evidence only; serial controller latency is the real-time evidence.

## 14. Stop rules

Stop expensive downstream work on:

- code/CLI/test/lint failure;
- causal leakage;
- source/artifact lineage mismatch;
- candidate/HOLD prefix mismatch;
- continuation-policy mismatch;
- CUDA OOM or paging that invalidates the intended runtime;
- Step2 policy-return sign/ranking scientific gate failure;
- engineering/support/readback/score-execute failure;
- new closed-loop Proposed worse than No-control on any preregistered acceptance probe.

Do not respond to a failure by lowering a margin, raising q95 to q99, increasing K, adding PFV/Peak penalties, or weakening baselines without separate preregistered evidence.

## 15. Promotion boundary

`READY_FOR_POLICY_LOCK=false` by default.

Do not enter Validation, Final, Formal or Policy Lock until a later explicit Web-GPT-reviewed change updates the promotion contract after all Development gates pass.
