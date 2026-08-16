# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. GitHub `main` is the code source of truth. The immediate task is not to run an expensive full pipeline: it is to determine, with bounded Development evidence, whether the diagnosed action-identifiability repair actually restores the control sensitivity needed by the Proposed TFV-first MPC.

Do not infer the current workflow from the highest V-number in a filename. Versioned files are shared/archival internals unless this document or the machine contracts explicitly name them.

## 1. Frozen research target — do not redesign it

Project7 is an **idealized EPA-SWMM methodology testbed, not a field digital twin**.

```text
causal sparse sensing
  -> Step1 reconstruct current full-network hydraulic state
  -> Step2 predict action-conditioned future hydraulics / TFV
  -> continuous 109-actuator receding-horizon MPC
  -> H360 prediction; 12 x 109 free 10-min target fractions over H120
  -> engineering envelope inside differentiable decoder
  -> execute only the first 10-min target
  -> authoritative SWMM write/readback
  -> re-observe after 600 s and optimize again
```

Frozen hierarchy:

- system-wide cumulative **TFV = primary objective**;
- Priority8 PFV = **one-sided soft secondary deterioration protection**;
- Global Peak = **report-only**;
- SWMM = authoritative truth.

Frozen clock/action contract:

- model/observation step = 300 s;
- control update = 600 s;
- prediction horizon = H360 = 72 model steps;
- free control horizon = H120 = 24 model steps = 12 x 10-min blocks;
- 109 writable continuous actuators;
- 1308 free MPC variables;
- **execute only the first 10-min target**, then re-observe SWMM.

Sparse-RBC is a warm start, engineering comparator and fallback. It is not the Step2 value reference and not an action-space ceiling.

## 2. Why the current Step2 changed

The last source-strict Development diagnostics localized a **mixed action-chain failure**:

```text
setting -> managed actuator flow -> node hydraulics/flooding -> cumulative TFV
```

The old Stage-A checkpoint showed:

- TFV-gradient L2 ratio ≈ 0.0112;
- gradient cosine ≈ 0.0062;
- held-out action->flow sign accuracy ≈ 0.874, but action->flow L2 magnitude ratio ≈ 0.00559;
- some TrainFit-supported actuators had temporal residual scales more than one order of magnitude below authoritative candidate-reference flow response;
- substituting authoritative SWMM actuator flows into the old node transition did not repair held-out hydraulic/flood-volume action effects.

Therefore the current smoke/dev candidate repairs **both** halves of the action chain. It does not change the TFV-first research question.

## 3. Current repaired Step2

Read these machine contracts first:

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
configs/project7_current_lint_surface.json
```

Current model/training surface:

```text
scripts/run_step2_current.py
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_action_identifiable_v128.py
src/rtc/step2_current_dev_context_v128.py
src/rtc/step2_differentiable_v128_edge.py
src/rtc/edge_physics_current_v128.py
src/rtc/step2_train_v128_exact.py
```

The repair does four things:

1. **FIT-only hybrid actuator residual scale**
   - lower bound = historical temporal |q(t)-q(t-1)| q99.5;
   - action scale = TrainFit candidate-reference managed-flow |dq| q99.5 for actuators whose settings actually change;
   - training scale = max(temporal, action-conditioned), actuator by actuator;
   - D2 holdout, D4-AUDIT, Validation, Final and Formal outcomes are forbidden from defining this scale;
   - this scale is a numerical surrogate scale, **not an engineering actuator rate/ramp constraint**.

2. **Explicit setting-conditioned actuator response**
   - previous flow remains a causal feature;
   - setting enters an explicit centered linear/quadratic basis;
   - a learned sigmoid `responsiveness` can no longer multiply the whole flow residual toward zero;
   - the actuator response is still bounded and differentiable.

3. **Counterfactual hydraulic-effect supervision**
   - Stage A retains absolute next-state/managed-flow supervision;
   - additionally supervises candidate-minus-reference state and managed-flow effects;
   - B0 autoregressive rollout retains the same effect supervision;
   - this prevents good one-step forecasting from hiding a useless action Jacobian.

4. **Edge-physics hydraulic propagation**
   - current smoke/dev uses the frozen graph-edge-aligned SWMM link artifact;
   - static link descriptors plus current head-difference/head-gradient enter graph messages;
   - no ordinary-conduit dynamic-flow labels are fabricated;
   - the incomplete mass-balance proxy remains diagnostic-only, not a training loss.

The H360 exact two-pass within-group pairwise TFV objective is retained. After it, one low-learning-rate FIT-only H360 action-effect rehearsal is applied so the TFV objective cannot silently erase the hydraulic action sensitivity learned earlier.

## 4. Development boundaries

Until this repaired candidate passes Development gates:

- never access Validation, Final, Formal or Policy Lock;
- never train on D4-AUDIT, D5-AUDIT, InternalHoldout or D2 held-out outcomes;
- never use future realised rainfall, future SWMM hydraulic/flooding truth or future Internal trajectories online;
- never generate new D2/D3/D4 SWMM data merely because source code changed;
- never fabricate missing ordinary-conduit flow labels;
- never enable the incomplete continuity proxy as a physics loss;
- do not add hydraulic-influence shortcuts yet: current evidence did not identify a monotonic far-field-only failure;
- smoke/dev stage checkpoints are NONFINAL and cannot enter D5/runtime/Policy Lock.

**`--profile full` is intentionally disabled for the repaired architecture.** If it fails with a Development-only/full-block message, that is expected behavior, not a bug. Full can be enabled only by a later explicit code promotion after held-out Development evidence is convincing.

## 5. Workstation contract

```text
GPU: RTX 4060 Laptop, ~8 GB VRAM
RAM: 16 GB
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RTC_V128_MATMUL_PRECISION=high
AMP: OFF
activation checkpointing: OFF
SWMM workers: <= 16, one thread/process
```

Do not use raw `--torch-profiler` Chrome-trace export. It previously completed training stages but exhausted host memory during trace export. Use `TRAINING_TELEMETRY.jsonl` plus low-frequency external `nvidia-smi` sampling.

## 6. Hard-sync local code from GitHub

If old local modifications are disposable, do not recover old patch/stash/repo_legacy content.

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin --prune
git switch main
git reset --hard origin/main
git clean -fd
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
python -m pip install -e ".[dev,swmm]"
python -c "import importlib.metadata as m; print(m.version('wuhan-rtc'))"
```

HEAD and origin/main must match; working tree must be clean. The repaired package version is `0.8.1.dev0`.

Never delete or reset the separate `study_v069` data root while cleaning the code checkout.

## 7. Cheap gates before any GPU work

```powershell
python -m pytest -q
python scripts/lint_current_surface.py
python scripts/run_step2_current.py --help
python scripts/build_edge_physics_current.py --help
python scripts/audit_step2_actuator_flow_effect_current.py --help
python scripts/audit_step2_gradient_stage_current_dev.py --help
python scripts/audit_step2_gradient_current_dev.py --help
python scripts/audit_step2_spatial_current.py --help
```

Every command must exit 0. Do not replace the maintained-surface lint with repository-wide `ruff check .`.

## 8. Reuse frozen assets

Prefer the already-admitted Project7 assets. Known current locations from the Development study are:

```text
<FROZEN_GRAPH>
E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz

<CANONICAL_D2_D3_CACHE>
E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json

<D4_FIT_CACHE>
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_fit\training_cache\CACHE_MANIFEST.json

<D4_AUDIT_CACHE>
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_audit\training_cache\CACHE_MANIFEST.json

<FROZEN_CAUSAL_RAINFALL_STORE>
E:\RTC_sewer\Project7\study_v069\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz

<FROZEN_CAUSAL_STATE_STORE_V2>
E:\RTC_sewer\Project7\study_v069\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz
```

Resolve `<FROZEN_INP>` from the frozen Project7 contract/study assets; its semantic identity is `wuhan_method_testbed_v067.inp`. Do not guess another INP and do not regenerate SWMM outcomes.

## 9. Create a fresh DebugRoot and preflight

```powershell
$env:PATH += ';C:\Users\12480\AppData\Roaming\Python\Python314\Scripts'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:RTC_V128_MATMUL_PRECISION = 'high'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$DebugRoot = "E:\RTC_sewer\Project7\study_v069\debug_action_identifiable_$Stamp"
New-Item -ItemType Directory -Path $DebugRoot -Force | Out-Null

rtc-current-preflight `
  --graph <FROZEN_GRAPH> `
  --device cuda `
  --out "$DebugRoot\PREFLIGHT.json"
```

Require CUDA, 109 actuators, correct graph lineage and `float32_matmul_precision=high`.

## 10. Build/reuse the required edge-physics artifact

The artifact is deterministic from frozen INP + graph and contains no outcome labels.

```powershell
python scripts/build_edge_physics_current.py `
  --inp <FROZEN_INP> `
  --graph <FROZEN_GRAPH> `
  --out-npz "$DebugRoot\EDGE_PHYSICS_CURRENT.npz" `
  --out-json "$DebugRoot\EDGE_PHYSICS_CURRENT.json"

$Edge = "$DebugRoot\EDGE_PHYSICS_CURRENT.npz"
```

If an existing artifact is reused, verify both its INP SHA and graph SHA instead of trusting its filename.

## 11. FIRST expensive action: Stage A only

Do **not** run B0 or H360 first. The purpose is to find out cheaply whether the repaired `setting -> flow -> one-step hydraulics` pathway has recovered.

```powershell
$Run = "$DebugRoot\smoke_repair"

python scripts/run_step2_current.py `
  --profile smoke `
  --edge-physics $Edge `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir $Run `
  --device cuda `
  --stop-after-stage stage_a
```

Expected output includes `stage_a.pt` plus telemetry. Because model/training source and edge-artifact lineage changed, an old baseline Stage-A checkpoint must fail if used as `--resume-from`; do not try to bypass that guard.

## 12. Immediately audit Stage A, then STOP

### 12.1 Managed-flow action effect

```powershell
python scripts/audit_step2_actuator_flow_effect_current.py `
  --profile smoke `
  --stage stage_a `
  --stage-checkpoint "$Run\stage_a.pt" `
  --graph <FROZEN_GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir "$Run\action_flow_stage_a" `
  --device cuda
```

Report at minimum:

- flow-effect sign accuracy;
- cosine;
- MAE;
- predicted/true L2 magnitude ratio;
- median/Q10/Q90 magnitude ratio;
- the checkpoint delta-flow scale for the changed actuator.

Frozen old baseline for comparison:

```text
sign accuracy ≈ 0.8740
cosine ≈ 0.7195
L2 magnitude ratio ≈ 0.00559
median magnitude ratio ≈ 0.01732
```

The goal of this repair is **not** to sacrifice sign accuracy to create arbitrary large gradients. Magnitude must recover materially while direction remains credible.

### 12.2 Stage-A TFV directional gradient

```powershell
python scripts/audit_step2_gradient_stage_current_dev.py `
  --profile smoke `
  --stage stage_a `
  --stage-checkpoint "$Run\stage_a.pt" `
  --graph <FROZEN_GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir "$Run\gradient_stage_a" `
  --device cuda
```

Frozen old Stage-A comparison:

```text
TFV gradient L2 ratio ≈ 0.01120
median magnitude ratio ≈ 0.00390
cosine ≈ 0.00615
sign accuracy ≈ 0.4255
```

After these two audits, **STOP and report the evidence**. Do not automatically continue merely because the commands exited 0. The web/GitHub supervision loop should decide whether this candidate deserves B0.

## 13. Only after explicit approval: resume to B0

Use the exact same graph/data/edge/profile/training-design lineage and the same output directory:

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --edge-physics $Edge `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir $Run `
  --device cuda `
  --resume-from "$Run\stage_a.pt" `
  --stop-after-stage stage_b0
```

Then rerun action-flow and stage-gradient audits with `--stage stage_b0` and `stage_b0.pt`. Stop again if sensitivity regresses.

## 14. Only after B0 approval: resume through H360 objective

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --edge-physics $Edge `
  --graph <FROZEN_GRAPH> `
  --cache-manifest <CANONICAL_D2_D3_CACHE> `
  --d4-fit-cache <D4_FIT_CACHE> `
  --d4-audit-cache <D4_AUDIT_CACHE> `
  --causal-store <FROZEN_CAUSAL_RAINFALL_STORE> `
  --causal-state-store <FROZEN_CAUSAL_STATE_STORE_V2> `
  --out-dir $Run `
  --device cuda `
  --resume-from "$Run\stage_b0.pt" `
  --stop-after-stage objective
```

The exact two-pass H360 pair census must still report complete directed gradient coverage and canonical float32 truth partitioning. The repaired runner then applies its small FIT-only H360 action-effect anchor before writing `stage_objective.pt`.

After objective:

1. rerun action-flow audit with `--stage objective`;
2. run `audit_step2_gradient_current_dev.py` with the same `--edge-physics`;
3. run `audit_step2_spatial_current.py` with the same `--edge-physics`;
4. read held-out ranking/pairwise/top1/selected-regret from `STEP2_V128_CURRENT_REPORT.json`.

Compare with the frozen old smoke baseline, not with training loss alone:

```text
D4 audit rank ≈ 0.05595
D4 audit pairwise ≈ 0.54167
D4 audit top1 = 0
D4 selected regret ≈ 1096 m3
held-out D2 rank ≈ -0.10698
old objective gradient sign ≈ 0.4043
old objective gradient cosine ≈ 0.06884
old objective gradient L2 ratio ≈ 0.03359
```

A low hydraulic RMSE is never sufficient evidence that Step2 is useful for MPC.

## 15. Dev promotion

Only after the repaired smoke shows coherent improvement in **action-flow magnitude + TFV gradient + ranking + spatial effects** may the same architecture run `--profile dev`.

Dev must use a fresh output directory, the same frozen inputs and a graph/INP-matched edge artifact. Repeat the same stage-wise stop/audit logic; do not jump directly to the end.

## 16. Full / D5 / runtime / seven strategies

Not authorized by the current repaired code.

Do not attempt to bypass the full-profile block. Do not use old strict V128 final checkpoints with the repaired model. Do not run D5, MPC runtime, seven strategies, Validation, Final, Formal or Policy Lock yet.

A later GitHub change must explicitly promote this architecture, update strict checkpoint fingerprints/loaders, and re-enable full only after Development evidence supports it.

## 17. What Codex must return after each stage

Always report:

1. exact Git SHA and package version;
2. working-tree cleanliness;
3. pytest/current-lint/help results;
4. asset hashes and edge-artifact hashes;
5. exact stage checkpoint SHA;
6. stage wall time, RSS/private/VMS/available RAM/swap/CUDA peaks;
7. action-flow metrics;
8. TFV-gradient sign/cosine/L2/median magnitude ratios;
9. if objective completed: rank/pairwise/top1/regret and spatial bins;
10. explicit access flags proving Validation/Final/Formal were not touched;
11. one failure localization;
12. STOP unless the user explicitly authorizes the next stage.

Code execution success is not scientific success. The current purpose is to restore a trustworthy action-conditioned surrogate cheaply enough that a later full SWMM closed-loop test is worth paying for.
