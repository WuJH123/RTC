# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. Versioned implementation files remain only where they are required for reproducibility or shared internal orchestration. Do not select a trainer, runtime, or seven-strategy script by version number.

## Frozen research target

Project7 is an **idealized SWMM methodology testbed, not a field digital twin**. The current method is:

```text
causal sparse sensing
  -> Step1 current full-network hydraulic reconstruction
  -> typed/physics-aware differentiable Step2 action-conditioned hydraulic surrogate
  -> continuous 109-actuator MPC
  -> H360 prediction with 12 x 109 free 10-min target fractions over H120
  -> engineering envelope inside the differentiable decoder
  -> execute only the first 10-min target
  -> authoritative SWMM target write/readback
  -> re-observe and solve again after 600 s
```

Frozen objective hierarchy:

- whole-system cumulative **TFV is primary**;
- Priority8 PFV is a **one-sided soft secondary** quantity;
- Global Peak is **report-only**;
- authoritative truth is **SWMM**;
- default online rainfall is causal deterministic persistence/decay and must **not** be described as robust/stochastic MPC.

Frozen clock:

- model/observation step: 300 s;
- supervisory decision period: 600 s;
- prediction: 72 model steps = 360 min;
- free control: 12 x 10-min blocks = 120 min = 24 model steps;
- execute: first 10-min target only.

## Current code surface

Read these machine-readable contracts first:

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
```

User/Codex entrypoints are intentionally unversioned:

```text
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Current internal Step2 implementation is V128 exact-pairwise:

```text
src/rtc/step2_differentiable_v128.py
src/rtc/step2_train_v128_hydraulic.py
src/rtc/step2_train_v128_exact.py
src/rtc/checkpoint_v128.py
```

Do not recreate or import the deleted `src/rtc/step2_train_v128.py` detached-memory objective. Do not run `scripts/run_step2_v127.py` as a current trainer. V127 streaming/seven-strategy files may still be called internally as audited shared orchestration; they are not user entrypoints.

## Hard scientific boundaries

During current development:

- no Validation, Final, Formal, or Policy-Lock outcomes;
- no training on InternalHoldout, D4-AUDIT, D5-AUDIT, or D2 development-validation branches;
- no future realised rainfall, future SWMM state/flooding, or future Internal trajectory online;
- do not use RBC as a Value reference or an action-space ceiling;
- do not project an MPC command after scoring;
- custom per-actuator engineering envelopes require matching decoder-space D5 evidence;
- ranking, D2 and D5 evidence must all reference the **identical final Step2 SHA256**;
- a checkpoint is stale if either its model-source or exact-training-source fingerprint differs from current code.

## Workstation profile

Target workstation:

```text
GPU: NVIDIA RTX 4060 8 GB
RAM: 16 GB
SWMM workers: <= 16
SWMM threads/process: 1
```

Before training:

```powershell
cd E:\RTC_sewer\Project7\repo
git status
git pull
python -m pytest -q

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:RTC_V128_MATMUL_PRECISION = "high"
```

Keep AMP and activation checkpointing disabled unless a separately frozen experiment explicitly changes that contract. Later repeat the selected checkpoint/run path with `RTC_V128_MATMUL_PRECISION=highest` as a numerical/runtime sensitivity check.

## 0. Preflight before expensive work

```powershell
rtc-v128-preflight `
  --graph <FROZEN_GRAPH> `
  --device cuda `
  --out <CURRENT_ROOT>\PREFLIGHT_BEFORE_TRAINING.json
```

Require 109 unique ordered actuators and CUDA availability. The default engineering envelope remains the historical idealized graph bounds + 0.5 target change per 10 min; it is not a field-actuator claim.

## 1. Train current Step2 from frozen existing D2/D3/D4 assets

```powershell
python scripts/run_step2_current.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE> `
  --out-dir <CURRENT_ROOT>\step2_base_high `
  --device cuda
```

Expected current base artifacts:

```text
step2_v128_control_base.pt
STEP2_V128_CONTROL_BASE_REPORT.json
```

The H360 objective must report the exact two-pass pairwise contract and full directed-gradient coverage. If the report instead indicates a detached-memory V2 objective, stop: the wrong code was executed.

## 2. Read-only base diagnostics

```powershell
python scripts/audit_step2_v128_fast.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE> `
  --step2 <CURRENT_ROOT>\step2_base_high\step2_v128_control_base.pt `
  --ranking-out <CURRENT_ROOT>\base_high\ranking.json `
  --horizon-out <CURRENT_ROOT>\base_high\horizon.json `
  --telemetry-out <CURRENT_ROOT>\base_high\telemetry.json `
  --device cuda

python scripts/audit_step2_v128_d2_gradients_fast.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE> `
  --step2 <CURRENT_ROOT>\step2_base_high\step2_v128_control_base.pt `
  --out-dir <CURRENT_ROOT>\base_high\d2 `
  --device cuda
```

Interpret action ranking/top1/regret and gradient sign/cosine before treating low hydraulic RMSE as sufficient for control.

## 3. Frozen D5-FIT and untouched D5-AUDIT

Use the already frozen outcome-blind D5 assets only with the decoder space they were created for. Current historical D5 evidence is for the idealized 0.5-per-10-min envelope.

```powershell
python scripts/run_step2_v128_d5_gradient_fast.py `
  --graph <FROZEN_GRAPH> `
  --base-cache-manifest <CANONICAL_D2_D3_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE> `
  --step2 <CURRENT_ROOT>\step2_base_high\step2_v128_control_base.pt `
  --d5-execution-manifest <FROZEN_D5_EXECUTION_MANIFEST> `
  --d5-gradient-labels <FROZEN_D5_GRADIENT_LABELS> `
  --out-dir <CURRENT_ROOT>\step2_final_high `
  --device cuda
```

Expected final checkpoint:

```text
<CURRENT_ROOT>\step2_final_high\step2_v128_d5_gradient.pt
```

## 4. Re-audit the exact final checkpoint

Never combine base-checkpoint reports with the D5-final checkpoint. Re-run both ranking/horizon and D2 gradient audits on `step2_v128_d5_gradient.pt`, then compile evidence:

```powershell
python scripts/build_v128_continuous_evidence.py `
  --ranking-report <CURRENT_ROOT>\final_high\ranking.json `
  --d2-gradient-report <CURRENT_ROOT>\final_high\d2\D2_INTERNAL_HOLDOUT_GRADIENT_METRICS.json `
  --d5-gradient-report <CURRENT_ROOT>\step2_final_high\STEP2_V128_D5_GRADIENT_REPORT.json `
  --out <CURRENT_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json
```

The compiler must fail if the three reports do not identify one identical final Step2 SHA256.

## 5. Runtime preflight

```powershell
rtc-v128-preflight `
  --graph <FROZEN_GRAPH> `
  --step2 <CURRENT_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --continuous-evidence <CURRENT_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json `
  --device cuda `
  --out <CURRENT_ROOT>\final_high\PREFLIGHT_BEFORE_RUNTIME.json
```

## 6. One authoritative development closed loop

Use a preselected development event. Do not tune after looking at the result.

```powershell
python scripts/run_policy_current.py `
  --inp <FROZEN_DEVELOPMENT_INP> `
  --out-dir <CURRENT_ROOT>\development_policy `
  --run-id <EVENT_ID>__proposed_current `
  --sensors <FROZEN_SENSORS> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --config <FROZEN_RUNTIME_CONFIG> `
  --graph <FROZEN_GRAPH> `
  --step1 <FROZEN_STEP1> `
  --step2 <CURRENT_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --continuous-evidence <CURRENT_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json `
  --device cuda `
  --lbfgsb-maxiter 30 `
  --optimizer-deadline-seconds 480 `
  --decision-runtime-budget-seconds 540
```

A real-time acceptance statement requires every complete continuity-guarded supervisory callback to finish in less than 600 s, plus explicit score==execute, continuity, and target-write/readback evidence.

## 7. Seven-strategy authoritative SWMM comparison

```powershell
python scripts/run_seven_strategies_current.py `
  --inp <FROZEN_DEVELOPMENT_INP> `
  --event-id <EVENT_ID> `
  --sensors <FROZEN_SENSORS> `
  --priority-nodes <FROZEN_PRIORITY8> `
  --config <FROZEN_RUNTIME_CONFIG> `
  --native-controls-template <FROZEN_NATIVE_CONTROLS_TEMPLATE> `
  --graph <FROZEN_GRAPH> `
  --step1 <FROZEN_STEP1> `
  --step2 <CURRENT_ROOT>\step2_final_high\step2_v128_d5_gradient.pt `
  --continuous-evidence <CURRENT_ROOT>\final_high\V128_CONTINUOUS_EVIDENCE.json `
  --out-dir <CURRENT_ROOT>\seven_strategy `
  --device cuda
```

Strategies are Proposed current, No-control, Internal RTC, Auto-RBC, storage-volume EFD, All-open and All-closed. TFV/PFV are recomputed from authoritative node statistics; Global Peak is routing-step frozen-decision replay and remains report-only.

## 8. Stop/promotion rule

Merging current code into `main` means only that the repository has one unambiguous implementation surface. It does **not** mean the method has passed Policy Lock or Final evaluation.

Do not proceed to Validation/Final/Formal/Policy Lock unless the exact final checkpoint has coherent same-checkpoint ranking/D2/D5 evidence, acceptable H30-H360 hydraulic behavior, authoritative development TFV/PFV that supports the method, and measured sub-600-s control decisions without execution violations.
