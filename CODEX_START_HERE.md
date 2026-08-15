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
configs/project7_current_lint_surface.json
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

The previous full runs showed host-memory paging while CUDA was not always saturated. Current training therefore uses lazy mmap branch streaming. `psutil` is installed in the Development extra so stage telemetry records RSS/VMS/private-or-USS memory, available RAM, swap and CUDA peaks.

A raw Torch Chrome trace across the whole one-group Stage A/B0/objective sequence is **not supported on the current 16-GB workstation**. A real run completed all three training stages but `export_chrome_trace(...)` drove process private memory to tens of GB and exhausted available RAM. The stable current entrypoint therefore rejects raw `--torch-profiler` before expensive work. This is a diagnostic-infrastructure rule, not a scientific-model change.

## 5. Synchronize local code to GitHub main

GitHub `main` is the code source of truth. If the user explicitly says old local changes are disposable, do not recover, merge or cherry-pick them.

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin --prune
git switch main
git reset --hard origin/main
git clean -fd
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

The two SHAs must match and the working tree must be clean. Never delete the separate study root or authoritative SWMM/cache assets when resetting the code checkout.

Install the synchronized package:

```powershell
python -m pip install -e ".[dev,swmm]"
```

## 6. Cheap code gates before any training

Run the complete unit/regression suite:

```powershell
python -m pytest -q
```

Then run the **current Project7 lint gate**:

```powershell
python scripts/lint_current_surface.py
```

Do **not** use `python -m ruff check .` as a smoke/dev stop-gate. The repository intentionally retains historical/archival Vxxx source for provenance and shared orchestration, and that historical tree contains pre-existing Ruff debt. Current execution/development files are fail-closed through `configs/project7_current_lint_surface.json` and GitHub Actions.

Do not run `ruff --fix .` or mass-format archival files during model debugging. Historical style cleanup is a separate maintenance task and must not alter scientific provenance or consume the current debugging cycle.

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

If pytest, current lint, or a current CLI help fails, stop before expensive work and report the exact failure. Do not substitute full-repository lint debt for a current-code failure.

## 7. Asset admission and preflight

Reuse existing frozen Development assets. Do not regenerate D2/D3/D4 merely because code was synchronized. Prefer the semantic-complete causal state store V2 and verify graph/Step1/sensor/rainfall lineage before training.

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:RTC_V128_MATMUL_PRECISION = "high"

rtc-current-preflight `
  --graph <FROZEN_GRAPH> `
  --device cuda `
  --out <DEBUG_ROOT>\PREFLIGHT.json
```

Require the current 109-actuator graph/device/engineering contract to pass. Preflight now applies the same V128 matmul-policy function used by training; with the environment above, `hardware.float32_matmul_precision` must be `high`. A preflight report that still says `highest` while `RTC_V128_MATMUL_PRECISION=high` is invalid and must be investigated before training.

## 8. First expensive action: one-group low-overhead resource profile

Do **not** add `--torch-profiler` on the current 16-GB workstation. Run the real one-group smoke path with ordinary stage/resource telemetry:

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --profile-one-group `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir <DEBUG_ROOT>\profile_one_group `
  --device cuda
```

Inspect `TRAINING_TELEMETRY.jsonl`. Record Stage A/B0/objective/evaluation wall time, process RSS/VMS/private-or-USS memory, available RAM, swap and CUDA allocated/reserved/peak values. If useful, sample `nvidia-smi` externally at low frequency for GPU utilisation/VRAM; do not start a second training process.

This resource profile is **diagnostic, not a scientific stop-gate**. Stop downstream only for a real training/runtime failure such as OOM, non-finite loss/gradient, stage error, lineage/contract mismatch, broken checkpoint, or an unreleased process/resource leak. Failure of an optional profiler/export/sampler after successful stage checkpoints is not evidence that Step2 failed.

Do not increase hidden dimension, chunks or workers merely because host memory is tight. First distinguish model/training memory from diagnostic instrumentation overhead.

## 9. Baseline smoke training

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

Smoke keeps the real V128 typed architecture, 109 actuators, H360 objective and exact two-pass pairwise path. It reduces Development coverage/repetition only and is **not paper evidence**.

The exact objective must share one canonical float32 SWMM candidate TFV delta across pair census, reported pair loss and live gradients. The historical `544/542` float64/float32 pair-coverage mismatch must never reappear.

### Stage checkpoint/resume

The runner writes NONFINAL stage-boundary checkpoints:

```text
stage_a.pt
stage_b0.pt
stage_objective.pt
```

Use `--stop-after-stage` and `--resume-from` only when profile, graph, data lineage, training design and source fingerprints remain compatible. This is stage-boundary resume, not mid-epoch resume. Smoke/dev stage checkpoints must never enter D5/runtime/Policy Lock.

## 10. Mandatory smoke diagnosis before architecture changes

### 10.1 Ranking/action identification

Read `STEP2_V128_CURRENT_REPORT.json` and report held-out ranking, pairwise, top1, TFV-delta MAE and selected regret. Training loss alone is not a promotion criterion.

### 10.2 Spatial action-effect audit

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

Report truth/predicted action effect, MAE/relative MAE and sign accuracy for `1-3`, `4-6`, `7-12`, and `13+` actuator-to-node graph hops.

### 10.3 Development D2 gradient audit

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

Report TFV gradient sign accuracy, gradient cosine similarity and gradient MAE. The strict `scripts/audit_step2_v128_d2_gradients_fast.py` is full-checkpoint evidence and must not be used on smoke/dev stage artifacts.

## 11. Failure-mode routing

Use the evidence to decide what to test next:

- ranking + gradient + near/far action effects all poor -> inspect local action identifiability, objective supervision and training coverage first;
- near-field good but `7-12`/`13+` action effects poor -> test the edge-aware Development path;
- Step1 error rises strongly with distance to nearest sensor -> test V122 sensor-to-all-node global attention;
- ranking acceptable but gradient sign/cosine poor -> fix gradient/objective path before Step1 or graph expansion;
- ranking/spatial/gradient all coherent -> only then investigate H360 autoregressive drift, optimizer/runtime, richer rainfall forecasting or horizon sensitivity.

Do not create a new version number merely because one smoke metric is weak.

## 12. Step1 global-attention ablation

Only when Step1 distance is a plausible blocker, train a separate Development V122 attention model and compare it against the frozen legacy Step1 on identical Development validation windows:

```powershell
python scripts/train_step1_global_attention_dev.py `
  --run-index <STEP1_RUN_INDEX> `
  --graph <FROZEN_GRAPH> `
  --sensors <FROZEN_SENSORS> `
  --out <DEBUG_ROOT>\step1_v122_attention.pt `
  --device cuda `
  --no-amp

python scripts/audit_step1_global_attention_current.py `
  --run-index <STEP1_RUN_INDEX> `
  --graph <FROZEN_GRAPH> `
  --sensors <FROZEN_SENSORS> `
  --legacy-model <FROZEN_STEP1> `
  --attention-model <DEBUG_ROOT>\step1_v122_attention.pt `
  --out <DEBUG_ROOT>\STEP1_DISTANCE_ABLATION.json `
  --device cuda
```

Compare nearest-sensor hop bins. Do not hot-swap Step1. If V122 is promoted, rebuild the causal Step1 state store and retrain Step2 from the beginning.

## 13. Edge-physics ablation

Only if Step2 far-field propagation is implicated:

```powershell
python scripts/build_edge_physics_current.py `
  --inp <FROZEN_INP> `
  --graph <FROZEN_GRAPH> `
  --out-npz <DEBUG_ROOT>\EDGE_PHYSICS.npz `
  --out-json <DEBUG_ROOT>\EDGE_PHYSICS.json

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

## 14. Influence graph and physics diagnostics

Build a hydraulic-influence artifact only after authoritative Development D2 proves meaningful remote effects that baseline/edge-aware propagation still fails to learn. Use Development TrainFit D2 only; never Validation/Final/Formal data. Building an influence artifact does not promote it into the full model.

Current V128 actuator injection is conservative, but the learned ordinary-network transition is not a full Saint-Venant solver and current training data do not contain all terms required for exact node continuity or authoritative ordinary-conduit flow supervision. Therefore continuity remains a diagnostic proxy, `continuity_proxy_training_loss=false`, and missing conduit-flow labels must never be fabricated.

## 15. Dev promotion

Only smoke variants that improve the intended failure mode progress to explicit `--profile dev`. Repeat ranking, spatial and Development D2 gradient audits and compare H30-H360 drift plus runtime/RAM/swap/CUDA usage.

Promote using evidence, not training loss alone. At minimum compare ranking/pairwise/top1, selected regret, TFV-delta error, D2 gradient sign/cosine/MAE, `1-3`/`4-6`/`7-12`/`13+` hop behavior, H360 drift and resource cost. Reject variants that do not solve their intended failure mode.

## 16. Full only after one Development winner

Do not run `--profile full` merely because the code compiles. The full profile retains the canonical 112 D2 + 112 D3 + 33 D4-FIT census, Stage A four epochs, H60/H120 rollout curriculum and H360 exact objective three epochs. Only an explicit full profile can create the strict V6 base checkpoint.

Only after a full checkpoint passes coherent ranking/horizon, strict D2 gradient and spatial gates may the workflow continue:

```text
D5-FIT using frozen D5-FIT only
-> re-run ranking/D2 on the exact D5-final checkpoint
-> compile same-SHA continuous evidence
-> runtime preflight
-> one preselected authoritative Development closed loop
-> seven-strategy authoritative Development comparison
```

Every guarded supervisory callback must be <600 s for a real-time claim; target write/readback, continuity and score==execute are mandatory.

## 17. Promotion rule

The current development loop is:

```text
idea
 -> pytest + current-surface lint + CLI help
 -> preflight
 -> one-group low-overhead resource profile
 -> smoke
 -> ranking + spatial + gradient diagnosis
 -> dev
 -> reject or promote
 -> full once
 -> D5 / authoritative Development SWMM evidence
 -> Policy Lock only after all gates
```

Merging code into `main`, passing smoke, or producing a dev checkpoint does **not** constitute Policy Lock. Do not proceed to Validation/Final/Formal/Policy Lock until the selected full model has coherent same-checkpoint ranking/D2/D5 evidence, acceptable spatial/H30-H360 behavior, authoritative Development TFV/PFV benefit and measured real-time execution.
