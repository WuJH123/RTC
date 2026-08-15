# Codex start here — Project7 V128 control-identifiability development

This file is the **single V128 development path**. Read it together with
`configs/project7_execution_registry.json` and `configs/v128_control_execution.json`.
V128 is a development candidate; V127 remains the production identity until the evidence
and authoritative SWMM comparison below pass. Do not jump to historical V120–V127 training
entrypoints when executing V128.

## Frozen scientific target

Every 600 s:

1. use only causal sparse observations to reconstruct the current full-network hydraulic state;
2. use the V128 typed/physics-aware differentiable hydraulic surrogate over H360;
3. optimize exactly 12 x 109 future 10-min target fractions over H120;
4. enforce the frozen engineering envelope inside the differentiable decoder before scoring;
5. execute only the first 600-s target in authoritative SWMM;
6. verify same-epoch target write/readback, re-observe and repeat.

Whole-system cumulative TFV is primary. Priority8 PFV is one-sided soft secondary. Global
Peak is report-only. Default rainfall forecast is causal deterministic persistence/decay;
do **not** call the default run robust/stochastic MPC. Authoritative truth is SWMM.

## 0. Never run the wrong contract

Current registry rules:

- V127 production base trainer: `scripts/run_step2_v127_control_streaming.py`;
- `scripts/run_step2_v127.py` is historical and non-canonical;
- V128 base trainer: `scripts/run_step2_v128_control_4060.py`;
- V128 checkpoint loader: `rtc.checkpoint_v128.load_step2_v128`;
- V127 and V128 checkpoints are intentionally incompatible;
- all V128 ranking/D2/D5 evidence must name the identical final Step2 SHA256;
- no Validation/Final/Formal/Policy Lock while V128 remains development-only.

Before any expensive run:

```powershell
cd E:\RTC_sewer\Project7\repo
git status
git fetch origin --prune
git rev-parse HEAD
python -m pytest -q
python -m py_compile `
  src/rtc/step2_differentiable_v128.py `
  src/rtc/step2_train_v128_hydraulic.py `
  src/rtc/step2_train_v128.py `
  src/rtc/checkpoint_v128.py `
  src/rtc/engineering_v128.py `
  src/rtc/step3_mpc_v128.py `
  src/rtc/step3_runtime_v128.py `
  src/rtc/controller_v128.py `
  src/rtc/runtime_evidence_v128.py `
  src/rtc/v128_preflight.py `
  scripts/run_step2_v128_control_4060.py `
  scripts/audit_step2_v128_fast.py `
  scripts/audit_step2_v128_d2_gradients_fast.py `
  scripts/run_step2_v128_d5_gradient_fast.py `
  scripts/build_v128_continuous_evidence.py `
  scripts/run_policy_v128.py `
  scripts/run_seven_strategies_v128.py
git diff --check
```

Do not continue after a real test/compile failure.

## 1. RTX 4060 / 16-GB workstation execution profile

Use the existing memory-safe CPU-group/GPU-microbatch path. Do not increase SWMM workers
above 16 on the stated 16-GB RAM workstation.

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:RTC_V128_MATMUL_PRECISION = "high"
```

V128 keeps AMP and activation checkpointing off by default. `high` is an execution choice,
not a scientific result; later repeat the frozen comparison with `highest`.

Preflight the frozen graph before training:

```powershell
rtc-v128-preflight `
  --graph <FROZEN_GRAPH> `
  --device cuda `
  --out <V128_ROOT>\V128_PREFLIGHT_BEFORE_TRAINING.json
```

Require exactly 109 ordered actuators and an approximately 8-GB CUDA GPU. If no explicit
engineering-envelope file is supplied, the code uses the historical **idealized** graph
bounds plus 0.5 target change per 10 min. That default is not a field-device claim.

## 2. Train the V128 base Step2 from existing D2/D3/D4 only

No new SWMM is required for this base experiment. Reuse the frozen V127 causal data assets
and split discipline; do not train on InternalHoldout, D4-AUDIT, D5-AUDIT or the D2
development-validation pool.

```powershell
python scripts/run_step2_v128_control_4060.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --out-dir <V128_ROOT>\step2_base_high `
  --device cuda
```

Expected artifacts:

- `step2_v128_control_base.pt`;
- `STEP2_V128_CONTROL_BASE_REPORT.json`.

The checkpoint is source-strict: model-behavior source changes invalidate loading; the
package-local training-recipe fingerprint is also recorded. Stage A must report the typed
action context, and the H360 objective must report complete within-group candidate-pair
partitioning independent of GPU microbatch size.

## 3. Read-only base evidence

First diagnose the base checkpoint. Do not use these base reports for final V128 evidence
after D5 fine-tuning; they are diagnostic only.

```powershell
python scripts/audit_step2_v128_fast.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V128_ROOT>\step2_base_high\step2_v128_control_base.pt `
  --ranking-out <V128_ROOT>\base_high\ranking.json `
  --horizon-out <V128_ROOT>\base_high\horizon.json `
  --telemetry-out <V128_ROOT>\base_high\telemetry.json `
  --device cuda

python scripts/audit_step2_v128_d2_gradients_fast.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V128_ROOT>\step2_base_high\step2_v128_control_base.pt `
  --out-dir <V128_ROOT>\base_high\d2 `
  --device cuda
```

Interpretation order: action rank/pairwise/top1/regret and D2 gradient sign/cosine first;
then H30–H360 hydraulic drift. A low absolute hydraulic error does not rescue incorrect
action ordering or gradients.

## 4. D5-FIT fine-tuning and untouched D5-AUDIT

Reuse the already frozen outcome-blind D5 manifest/labels only with the historical idealized
0.5-per-10min decoder. A custom per-actuator envelope changes fraction-to-target derivative
coordinates and therefore requires a newly frozen envelope-bound D5 experiment; current
runtime fails closed instead of reusing the old D5 evidence.

```powershell
python scripts/run_step2_v128_d5_gradient_fast.py `
  --graph <FROZEN_GRAPH> `
  --base-cache-manifest <CANONICAL_V60_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V128_ROOT>\step2_base_high\step2_v128_control_base.pt `
  --d5-execution-manifest <FROZEN_D5_EXECUTION_MANIFEST> `
  --d5-gradient-labels <FROZEN_D5_GRADIENT_LABELS> `
  --out-dir <V128_ROOT>\step2_final_high `
  --device cuda
```

Expected final checkpoint: `step2_v128_d5_gradient.pt`. D5-AUDIT is read-only before and
after training and never enters optimization.

## 5. Re-audit the exact final checkpoint

This step is mandatory. Never combine base-checkpoint ranking/D2 reports with a D5-final
checkpoint.

```powershell
python scripts/audit_step2_v128_fast.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V128_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --ranking-out <V128_ROOT>\final_high\ranking.json `
  --horizon-out <V128_ROOT>\final_high\horizon.json `
  --telemetry-out <V128_ROOT>\final_high\telemetry.json `
  --device cuda

python scripts/audit_step2_v128_d2_gradients_fast.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_V60_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <V127_CAUSAL_STATE_STORE> `
  --step2 <V128_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --out-dir <V128_ROOT>\final_high\d2 `
  --device cuda
```

Compile structural continuous-MPC evidence:

```powershell
python scripts/build_v128_continuous_evidence.py `
  --ranking-report <V128_ROOT>\final_high\ranking.json `
  --d2-gradient-report <V128_ROOT>\final_high\d2\D2_INTERNAL_HOLDOUT_GRADIENT_METRICS.json `
  --d5-gradient-report <V128_ROOT>\step2_final_high\STEP2_V128_D5_GRADIENT_REPORT.json `
  --out <V128_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json
```

The compiler fails if ranking/D2/D5 do not reference the same final Step2 SHA256. Metric
scores are scientific evidence, not arbitrary runtime switches; causal provenance, finite
values and same-checkpoint identity are hard requirements.

## 6. Final preflight before any authoritative control run

```powershell
rtc-v128-preflight `
  --graph <FROZEN_GRAPH> `
  --step2 <V128_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --continuous-evidence <V128_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json `
  --device cuda `
  --out <V128_ROOT>\final_high\V128_PREFLIGHT_BEFORE_RUNTIME.json
```

If using `--engineering-envelope <FILE>`, the preflight and runtime require evidence valid
for that exact decoder space. Current frozen D5 supports only the historical idealized
envelope.

## 7. One authoritative development closed loop

Use a fixed development event only. Do not tune parameters after seeing the result.

```powershell
python scripts/run_policy_v128.py `
  --inp <FROZEN_DEVELOPMENT_INP> `
  --out-dir <V128_ROOT>\development_policy `
  --run-id <EVENT_ID>__proposed_v128 `
  --sensors <FROZEN_SENSORS> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --config <FROZEN_RUNTIME_CONFIG> `
  --graph <FROZEN_GRAPH> `
  --step1 <FROZEN_STEP1> `
  --step2 <V128_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --continuous-evidence <V128_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json `
  --device cuda `
  --lbfgsb-maxiter 30 `
  --optimizer-deadline-seconds 480 `
  --decision-runtime-budget-seconds 540 `
  --pfv-soft-margin-m3 100 `
  --pfv-penalty-weight 1
```

The run fails unless:

- every 600-s command is complete and within the frozen envelope;
- no post-score projection occurs;
- same-epoch SWMM target write/readback passes;
- every complete continuity-guarded supervisory callback is <600 s;
- score==execute and continuity evidence are explicitly true.

Report runtime mean/p50/p95/max and optimizer fallback/deadline counts. This is measured
real-time feasibility on the tested workstation/event, not a universal industrial
worst-case guarantee.

## 8. Proposed plus six fixed baselines

```powershell
python scripts/run_seven_strategies_v128.py `
  --inp <FROZEN_DEVELOPMENT_INP> `
  --event-id <FROZEN_DEVELOPMENT_EVENT_ID> `
  --sensors <FROZEN_SENSORS> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --config <FROZEN_RUNTIME_CONFIG> `
  --native-controls-template <FROZEN_NATIVE_CONTROLS_TEMPLATE> `
  --graph <FROZEN_GRAPH> `
  --step1 <FROZEN_STEP1> `
  --step2 <V128_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --continuous-evidence <V128_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json `
  --out-dir <V128_ROOT>\seven_strategies_high `
  --device cuda `
  --lbfgsb-maxiter 30 `
  --optimizer-deadline-seconds 480 `
  --decision-runtime-budget-seconds 540 `
  --pfv-soft-margin-m3 100 `
  --pfv-penalty-weight 1
```

The fixed comparison remains: Proposed V128, No-control, Internal RTC, Auto-RBC,
storage-volume EFD, All-open, All-closed. Final TFV/PFV come from authoritative SWMM node
statistics; report-only Global Peak uses frozen-decision routing-step replay.

## 9. RTX 4060 numerical-performance sensitivity

After the `high` recipe is frozen, repeat the same V128 base→D5→final evidence path in a
separate output root with:

```powershell
$env:RTC_V128_MATMUL_PRECISION = "highest"
```

Do not mix reports/checkpoints between `high` and `highest`. Compare rank/top1/regret,
D2/D5 gradient sign/cosine, H30-H360 errors, authoritative TFV/PFV and runtime. Promote
`high` only if it does not materially change scientific conclusions while providing useful
runtime benefit.

## Stop rule

V128 is not promoted merely because code tests pass. Stop in development and diagnose the
failing layer if any of the following occurs:

- action ranking/gradient evidence remains weak;
- long-horizon hydraulic drift materially worsens;
- continuous MPC rarely beats its RBC safety fallback;
- authoritative SWMM TFV does not improve/preserve performance relative to the frozen V127
  development comparison and fixed baselines;
- any guarded decision reaches 600 s;
- write/readback, continuity or score==execute fails.

Do not access Validation, Final, Formal or Policy Lock until V128 is explicitly promoted by
a separate, evidence-backed decision.
