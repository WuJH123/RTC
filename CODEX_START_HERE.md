# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. The immediate development goal is to debug the Proposed method quickly, identify why TFV control is weak, and reject bad model ideas before paying for a multi-hour full run. Versioned implementation files are archival/shared internals unless explicitly named below.

## Frozen research target

Project7 remains an **idealized SWMM methodology testbed, not a field digital twin**:

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

Frozen objective: whole-system cumulative **TFV primary**; Priority8 PFV **one-sided soft secondary**; Global Peak **report-only**; SWMM is authoritative truth. Clock: 300-s observation/model step, 600-s control update, H360 prediction, H120 free control, first 10-min target executed only.

## Current contracts and entrypoints

Read first:

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
```

User/Codex current entrypoints:

```text
rtc-current-preflight
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Current Step2 **requires an explicit cost profile**:

```text
--profile smoke
--profile dev
--profile full
```

There is intentionally no default. This prevents a small debugging change from silently starting a multi-hour full training run.

## Scientific boundaries

During development:

- never access Validation, Final, Formal or Policy Lock;
- never train on InternalHoldout, D4-AUDIT, D5-AUDIT, or D2 development-validation branches;
- never use future realised rainfall, future SWMM state/flooding, or future Internal trajectory online;
- smoke/dev are **screening only** and can never create a strict final Step2 checkpoint;
- smoke/dev stage checkpoints can never enter D5, runtime or Policy Lock;
- no post-score action projection; score must equal execute;
- do not fabricate ordinary-conduit flow labels or enable an incomplete mass-balance proxy as a physics loss;
- do not promote global attention, edge-aware propagation or influence shortcuts to full until held-out Development evidence supports them.

## Workstation profile

```text
GPU: RTX 4060 8 GB
RAM: 16 GB
SWMM workers: <=16; one thread/process
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RTC_V128_MATMUL_PRECISION=high
AMP off for current Step2
activation checkpointing off
```

The previous full run showed host-memory paging with GPU only partly utilised. Profile before changing chunks/workers.

## 0. Unit/preflight gate

```powershell
cd E:\RTC_sewer\Project7\repo
python -m pytest -q

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:RTC_V128_MATMUL_PRECISION = "high"

rtc-current-preflight `
  --graph <FROZEN_GRAPH> `
  --device cuda `
  --out <DEBUG_ROOT>\PREFLIGHT.json
```

Require 109 ordered actuators, CUDA and graph PASS.

## 1. First run: one-group profiler, not full

This exercises the real 109-actuator/H360/exact-pairwise code path on one deterministic group per source:

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --profile-one-group `
  --torch-profiler `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir <DEBUG_ROOT>\profile_one_group `
  --device cuda
```

Inspect `TRAINING_TELEMETRY.jsonl` and `TORCH_PROFILER_TRACE.json` before tuning memory/chunks. PyTorch profiler changes no scientific objective; it is execution diagnostics only.

## 2. Smoke training

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir <DEBUG_ROOT>\smoke `
  --device cuda
```

Smoke preserves the current architecture, 109 actuators, H360 objective and exact two-pass pairwise code path, but uses a tiny deterministic Development subset and one training repetition. It is **not paper evidence**.

Current exact pair training must report the canonical float32 truth-delta contract. The pair census, reported pair loss and live gradient use the same precomputed float32 SWMM candidate delta; the historical float64/float32 `544/542` coverage mismatch must not reappear.

### Stage checkpoint / resume

Training writes NONFINAL checkpoints after Stage A, B0 and objective:

```text
stage_a.pt
stage_b0.pt
stage_objective.pt
```

Pause deliberately, for example:

```text
--stop-after-stage stage_a
--stop-after-stage stage_b0
--stop-after-stage objective
```

Resume with the same profile/data/design:

```text
--resume-from <RUN>\stage_b0.pt
```

Resume fails closed if profile, graph, data lineage or training design differs. Intermediate stage checkpoints are never accepted by runtime/final loaders.

## 3. P0 spatial action-effect audit

After smoke objective finishes:

```powershell
python scripts/audit_step2_spatial_current.py `
  --profile smoke `
  --stage-checkpoint <DEBUG_ROOT>\smoke\stage_objective.pt `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out <DEBUG_ROOT>\smoke\SPATIAL_ACTION_EFFECT.json `
  --device cuda
```

Read held-out D2 action-effect sign/magnitude by actuator-to-node graph distance: `1-3`, `4-6`, `7-12`, `13+` hops. If far-field action effects collapse while near-field is good, do not spend a full run before testing the spatial P1 variants.

## 4. Step1 global-attention ablation

The current frozen Step1 remains the baseline. A V122 sensor-to-all-node attention model must be trained separately on the same Development TrainFit data, then compared on the identical Development validation windows:

```powershell
python scripts/audit_step1_global_attention_current.py `
  --run-index <STEP1_RUN_INDEX> `
  --graph <FROZEN_GRAPH> `
  --sensors <FROZEN_SENSORS> `
  --legacy-model <FROZEN_STEP1> `
  --attention-model <V122_ATTENTION_STEP1> `
  --out <DEBUG_ROOT>\STEP1_DISTANCE_ABLATION.json `
  --device cuda
```

Compare depth error by nearest-sensor hops. Do **not** hot-swap Step1. If attention is promoted, rebuild the causal Step1 state store and retrain Step2 from the beginning.

## 5. Edge-physics ablation only if spatial evidence warrants it

Compile current graph-edge physics from the frozen INP:

```powershell
python scripts/build_edge_physics_current.py `
  --inp <FROZEN_INP> `
  --graph <FROZEN_GRAPH> `
  --out-npz <DEBUG_ROOT>\EDGE_PHYSICS.npz `
  --out-json <DEBUG_ROOT>\EDGE_PHYSICS.json
```

Then run the same smoke/dev curriculum with V128 typed actuator messages plus edge-aware ordinary-network propagation:

```powershell
python scripts/run_step2_edge_aware_dev.py `
  --edge-physics <DEBUG_ROOT>\EDGE_PHYSICS.npz `
  --profile smoke `
  <the same graph/cache/causal-store arguments> `
  --out-dir <DEBUG_ROOT>\edge_smoke `
  --device cuda
```

This experiment includes static edge physics plus current head-difference/length-normalized hydraulic-gradient messages. `--profile full` is deliberately forbidden until held-out ranking/spatial evidence supports promotion.

## 6. Optional Development hydraulic-influence graph

Only after the baseline spatial audit identifies meaningful remote effects:

```powershell
python scripts/build_hydraulic_influence_current.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --out-npz <DEBUG_ROOT>\HYDRAULIC_INFLUENCE.npz `
  --out-json <DEBUG_ROOT>\HYDRAULIC_INFLUENCE.json
```

It is built from Development TrainFit D2 only. Never use Validation/Final to build shortcuts. The artifact is diagnostic/experimental until separately promoted.

## 7. Development proxy

Only variants that pass smoke and improve the relevant diagnostics progress to:

```powershell
python scripts/run_step2_current.py `
  --profile dev `
  <same frozen data arguments> `
  --out-dir <DEBUG_ROOT>\dev `
  --device cuda
```

Use the deterministic dev proxy to compare, at minimum:

- ranking / pairwise / top1 / selected regret;
- D2 gradient sign/cosine;
- H30-H360 rollout drift;
- spatial action-effect sign/magnitude by graph distance;
- training wall time, RAM/swap and CUDA use.

Reject variants that do not beat the frozen current baseline on the intended failure mode.

## 8. Full Step2 only after development promotion

Do not run full merely because code compiles. When a Proposal variant is selected and frozen:

```powershell
python scripts/run_step2_current.py `
  --profile full `
  <same frozen data arguments> `
  --out-dir <FULL_ROOT>\step2_base `
  --device cuda
```

`full` preserves the canonical 112 D2 + 112 D3 + 33 D4-FIT census, Stage A 4 epochs, H60/H120 rollout curriculum and H360 exact objective 3 epochs. Only this explicit profile can create `step2_v128_control_base.pt`, under strict V6 source/profile fingerprinting.

## 9. Full-only downstream sequence

Only after the full base checkpoint passes ranking/horizon + D2 gradient + spatial gates:

```text
D5-FIT using frozen D5-FIT only
-> re-run ranking/D2 on the exact D5-final checkpoint
-> compile same-SHA continuous evidence
-> runtime preflight
-> one preselected authoritative Development closed loop
-> seven-strategy authoritative Development comparison
```

Use existing current scripts:

```text
scripts/audit_step2_v128_fast.py
scripts/audit_step2_v128_d2_gradients_fast.py
scripts/run_step2_v128_d5_gradient_fast.py
scripts/build_v128_continuous_evidence.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Every guarded supervisory callback must be <600 s for a real-time claim; target write/readback, continuity and score==execute are mandatory.

## Promotion rule

The desired development loop is now:

```text
idea
 -> unit/preflight
 -> one-group profiler
 -> smoke
 -> spatial/ranking/gradient diagnosis
 -> dev
 -> reject or promote
 -> full once
 -> authoritative SWMM downstream evidence
```

Merging code into `main`, passing smoke, or producing a dev checkpoint does **not** constitute Policy Lock. Do not proceed to Validation/Final/Formal/Policy Lock until the selected full model has coherent same-checkpoint ranking/D2/D5 evidence, acceptable spatial/H30-H360 behavior, authoritative Development TFV/PFV benefit and measured real-time execution.
