# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. The immediate goal is to debug the Proposed method quickly, identify why TFV control is weak, reject weak ideas with bounded Development evidence, and run the expensive full pipeline only after one variant is clearly worth promoting.

Versioned implementation files are archival/shared internals unless explicitly named below. Do not infer the current workflow from the highest version number in a filename.

## 1. Frozen research target

Project7 is an **idealized EPA-SWMM methodology testbed, not a field digital twin**.

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

Frozen scientific hierarchy:

- whole-system cumulative **TFV = primary objective**;
- frozen Priority8 PFV = **one-sided soft secondary deterioration protection**;
- Global Peak = **report-only**;
- SWMM = authoritative truth.

Frozen clock/action contract:

- model/observation step = 300 s;
- control update = 600 s;
- H360 prediction = 72 model steps;
- H120 free control = 12 x 10-min blocks = 24 model steps;
- all 109 writable actuators remain eligible;
- continuous decision dimension = 12 x 109 = 1308;
- execute only the first 10-min target, then re-observe SWMM.

Sparse-RBC is a warm start, safety fallback and engineering comparator. It is **not** the Step2 Value reference and **not** an action-space ceiling.

## 2. Current machine contracts and stable entrypoints

Read first:

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
```

Stable user/Codex entrypoints:

```text
rtc-current-preflight
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Current Step2 requires an explicit cost profile:

```text
--profile smoke
--profile dev
--profile full
```

There is intentionally no default. Ordinary debugging must never silently start a multi-hour full run.

## 3. Development boundaries

Until the selected Proposal passes Development evidence:

- never access Validation, Final, Formal or Policy Lock;
- never train on InternalHoldout, D4-AUDIT, D5-AUDIT or D2 development-validation outcomes;
- never use future realised rainfall, future SWMM state/flooding or future Internal trajectory online;
- smoke/dev are **screening only** and can never create a strict final Step2 checkpoint;
- smoke/dev stage checkpoints can never enter D5, runtime or Policy Lock;
- no post-score action projection; the engineering envelope must be inside the differentiable decoder and score must equal execute;
- do not fabricate ordinary-conduit flow labels;
- do not enable the incomplete continuity proxy as a physics loss;
- do not promote global attention, edge-aware propagation or hydraulic-influence shortcuts to full until held-out Development evidence supports them.

## 4. Workstation profile

Current target workstation:

```text
GPU: RTX 4060 8 GB
RAM: 16 GB
SWMM workers: <=16; one thread/process
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RTC_V128_MATMUL_PRECISION=high
AMP off for current Step2
activation checkpointing off
```

The previous full run showed host-memory paging while CUDA was not saturated. Current training therefore uses lazy mmap branch streaming. Profile before changing chunks/workers.

## 5. Recover a mixed local checkout safely

The GitHub `main` branch is the source of truth. Do **not** merge a mixed local working tree into current code just to preserve old experiments.

Preferred recovery sequence in PowerShell:

```powershell
cd E:\RTC_sewer\Project7\repo

git status --short --branch
git remote -v
```

If the working tree contains anything you may want later, preserve it without applying it to current code:

```powershell
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
git diff > "E:\RTC_sewer\Project7\local_pre_sync_$Stamp.patch"
git diff --cached > "E:\RTC_sewer\Project7\local_pre_sync_index_$Stamp.patch"
git stash push -u -m "pre-current-main-sync-$Stamp"
```

Then make local `main` exactly equal to GitHub `origin/main`:

```powershell
git fetch origin --prune
git switch main
git reset --hard origin/main
git clean -fd

git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

The two SHAs must match and `git status --short` must be empty.

If the local repository itself is structurally unreliable, prefer a clean clone rather than repairing years of branch residue:

```powershell
cd E:\RTC_sewer\Project7
Rename-Item repo ("repo_legacy_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
git clone https://github.com/WuJH123/RTC.git repo
cd repo
git switch main
git pull --ff-only
```

Do not delete the study root or authoritative SWMM/cache assets when replacing the code checkout.

Install the current package from the synchronized checkout:

```powershell
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -m ruff check .
```

Key CLI smoke checks must all exit 0:

```powershell
python scripts/run_step2_current.py --help
python scripts/audit_step2_spatial_current.py --help
python scripts/audit_step2_gradient_current_dev.py --help
python scripts/train_step1_global_attention_dev.py --help
python scripts/audit_step1_global_attention_current.py --help
python scripts/run_step2_edge_aware_dev.py --help
python scripts/audit_step2_edge_spatial_current.py --help
python scripts/run_seven_strategies_current.py --help
```

## 6. Preflight gate

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:RTC_V128_MATMUL_PRECISION = "high"

rtc-current-preflight `
  --graph <FROZEN_GRAPH> `
  --device cuda `
  --out <DEBUG_ROOT>\PREFLIGHT.json
```

Require the current 109-actuator graph/device contract to pass before training.

## 7. First execution: one-group profiler, not full

This exercises the real 109-actuator/H360/exact-pairwise path on one deterministic group per source:

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

Inspect:

```text
TRAINING_TELEMETRY.jsonl
TORCH_PROFILER_TRACE.json
```

Do not increase hidden dimension, branch chunk or worker count until the profile shows what is actually limiting throughput.

## 8. Baseline smoke training

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir <DEBUG_ROOT>\baseline_smoke `
  --device cuda
```

Smoke preserves the current architecture, 109 actuators, H360 objective and exact two-pass pairwise code path, but uses a small deterministic Development subset and reduced repetition. It is **not paper evidence**.

The exact objective must use one canonical float32 SWMM candidate TFV delta for pair census, reported pair loss and live gradients. The historical float64/float32 `544/542` pair-coverage mismatch must never reappear.

### Stage checkpoint / resume

The runner writes NONFINAL stage-boundary checkpoints:

```text
stage_a.pt
stage_b0.pt
stage_objective.pt
```

Pause deliberately with:

```text
--stop-after-stage stage_a
--stop-after-stage stage_b0
--stop-after-stage objective
```

Resume with the same profile/data/design:

```text
--resume-from <RUN>\stage_b0.pt
```

Resume fails closed if profile, graph, data lineage, training design, model source, training source or model-class source differs. This is stage-boundary resume, not a mid-epoch resume claim.

## 9. Mandatory smoke diagnostics before any architecture change

### 9.1 Step2 spatial action-effect audit

```powershell
python scripts/audit_step2_spatial_current.py `
  --profile smoke `
  --stage-checkpoint <DEBUG_ROOT>\baseline_smoke\stage_objective.pt `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out <DEBUG_ROOT>\baseline_smoke\SPATIAL_ACTION_EFFECT.json `
  --device cuda
```

Read held-out D2 action-effect sign/magnitude/error by actuator-to-node graph distance:

```text
1-3 hops
4-6 hops
7-12 hops
13+ hops
```

If far-field action effects collapse while near-field is good, do not spend a full run before testing spatial P1 variants.

### 9.2 Smoke/dev D2 gradient audit

For `stage_objective.pt`, use the Development-only auditor, **not** the strict full-checkpoint auditor:

```powershell
python scripts/audit_step2_gradient_current_dev.py `
  --profile smoke `
  --stage-checkpoint <DEBUG_ROOT>\baseline_smoke\stage_objective.pt `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir <DEBUG_ROOT>\baseline_smoke\gradient `
  --device cuda
```

Report at minimum:

- TFV gradient sign accuracy;
- gradient cosine similarity;
- gradient MAE.

The strict `scripts/audit_step2_v128_d2_gradients_fast.py` is full-checkpoint evidence and must not be used to pretend a smoke/dev stage artifact is final.

### 9.3 Ranking/action-identification readout

Read `STEP2_V128_CURRENT_REPORT.json` for held-out ranking, pairwise, top1, TFV-delta MAE and selected regret. A lower training loss alone is not a promotion criterion.

## 10. Step1 global-attention ablation only when Step1 distance is a plausible blocker

The frozen production Step1 remains the baseline. Train a separate V122 sensor-to-all-node attention checkpoint on the same Development TrainFit data:

```powershell
python scripts/train_step1_global_attention_dev.py `
  --run-index <STEP1_RUN_INDEX> `
  --graph <FROZEN_GRAPH> `
  --sensors <FROZEN_SENSORS> `
  --out <DEBUG_ROOT>\step1_v122_attention.pt `
  --device cuda `
  --no-amp
```

Use the same frozen training split/sensor layout; do not overwrite the frozen Step1 checkpoint.

Then compare legacy and V122 on identical Development validation windows:

```powershell
python scripts/audit_step1_global_attention_current.py `
  --run-index <STEP1_RUN_INDEX> `
  --graph <FROZEN_GRAPH> `
  --sensors <FROZEN_SENSORS> `
  --legacy-model <FROZEN_STEP1> `
  --attention-model <DEBUG_ROOT>\step1_v122_attention.pt `
  --out <DEBUG_ROOT>\STEP1_DISTANCE_ABLATION.json `
  --device cuda
```

Compare depth error by nearest-sensor hops. Do **not** hot-swap Step1. If V122 is promoted, rebuild the causal Step1 state store and retrain Step2 from the beginning.

## 11. Edge-physics ablation only if Step2 far-field propagation is implicated

Compile edge physics from the frozen INP:

```powershell
python scripts/build_edge_physics_current.py `
  --inp <FROZEN_INP> `
  --graph <FROZEN_GRAPH> `
  --out-npz <DEBUG_ROOT>\EDGE_PHYSICS.npz `
  --out-json <DEBUG_ROOT>\EDGE_PHYSICS.json
```

Run the same smoke/dev curriculum with V128 typed actuator messages plus edge-aware ordinary-network propagation:

```powershell
python scripts/run_step2_edge_aware_dev.py `
  --edge-physics <DEBUG_ROOT>\EDGE_PHYSICS.npz `
  --profile smoke `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir <DEBUG_ROOT>\edge_smoke `
  --device cuda
```

`--profile full` is deliberately forbidden for this experimental architecture until held-out ranking/spatial/gradient evidence supports promotion.

## 12. Hydraulic-influence graph is diagnostic/experimental, not a default model change

Only after Development D2 shows meaningful remote actuator-node effects that baseline/edge-aware propagation still fails to learn:

```powershell
python scripts/build_hydraulic_influence_current.py `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --out-npz <DEBUG_ROOT>\HYDRAULIC_INFLUENCE.npz `
  --out-json <DEBUG_ROOT>\HYDRAULIC_INFLUENCE.json
```

The artifact may use Development TrainFit D2 only. Never construct influence shortcuts from Validation/Final/Formal data. Building the artifact does not promote it into the full model.

## 13. Physics diagnostics

Current V128 actuator injection is explicitly conservative, but the learned ordinary-network transition is not a full Saint-Venant/continuity solver. Current node state also lacks all terms needed for an exact node mass-balance equation and does not carry authoritative ordinary-conduit dynamic flow labels.

Therefore:

- continuity is a diagnostic proxy only;
- `continuity_proxy_training_loss` remains false;
- ordinary-conduit flow supervision is gated on real authoritative labels;
- do not fabricate missing link-flow labels to obtain a "physics-informed" claim.

## 14. Development proxy

Only variants that pass smoke and improve the intended failure mode progress to:

```powershell
python scripts/run_step2_current.py `
  --profile dev `
  <same frozen data arguments> `
  --out-dir <DEBUG_ROOT>\<variant>_dev `
  --device cuda
```

Repeat the same ranking, spatial and Development D2 gradient audits with `--profile dev`.

Promotion should consider at minimum:

- held-out ranking/pairwise/top1;
- selected regret;
- D2 gradient sign/cosine/MAE;
- 1-3, 4-6, 7-12 and 13+ hop action-effect behavior;
- H30-H360 rollout drift;
- wall time, RAM/swap and CUDA usage.

Reject a variant that does not improve the failure mode it was introduced to solve.

## 15. Full Step2 only after one Development winner is frozen

Do not run full merely because code compiles.

```powershell
python scripts/run_step2_current.py `
  --profile full `
  <same frozen data arguments> `
  --out-dir <FULL_ROOT>\step2_base `
  --device cuda
```

`full` preserves the canonical 112 D2 + 112 D3 + 33 D4-FIT census, Stage A four epochs, H60/H120 rollout curriculum and H360 exact objective three epochs. Only explicit `--profile full` can create `step2_v128_control_base.pt` under the strict V6 source/profile contract.

## 16. Full-only downstream sequence

Only after the full base checkpoint passes ranking/horizon + strict D2 gradient + spatial gates:

```text
D5-FIT using frozen D5-FIT only
-> re-run ranking/D2 on the exact D5-final checkpoint
-> compile same-SHA continuous evidence
-> runtime preflight
-> one preselected authoritative Development closed loop
-> seven-strategy authoritative Development comparison
```

Full-only evidence scripts:

```text
scripts/audit_step2_v128_fast.py
scripts/audit_step2_v128_d2_gradients_fast.py
scripts/run_step2_v128_d5_gradient_fast.py
scripts/build_v128_continuous_evidence.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Every guarded supervisory callback must be <600 s for a real-time claim; target write/readback, continuity and score==execute are mandatory.

## 17. Promotion rule

The current development loop is:

```text
idea
 -> unit/preflight
 -> one-group profiler
 -> smoke
 -> ranking + spatial + gradient diagnosis
 -> dev
 -> reject or promote
 -> full once
 -> D5 / authoritative Development SWMM evidence
 -> Policy Lock only after all gates
```

Merging code into `main`, passing smoke, or producing a dev checkpoint does **not** constitute Policy Lock. Do not proceed to Validation/Final/Formal/Policy Lock until the selected full model has coherent same-checkpoint ranking/D2/D5 evidence, acceptable spatial/H30-H360 behavior, authoritative Development TFV/PFV benefit and measured real-time execution.
