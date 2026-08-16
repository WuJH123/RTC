# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. GitHub `main` is the
code source of truth. Do not infer the active method from the highest V-number in a filename.

The current task is Development failure elimination, not Formal evidence. The surrogate must
first learn the causal counterfactual hydraulic response needed by continuous MPC; autograd TFV
sensitivity is then checked as a solver signal rather than trained as a SWMM-labelled target.

## 1. Frozen research problem

Project7 is an **idealized EPA-SWMM methodology testbed, not a field digital twin**.

```text
causal sparse sensing
  -> Step1 current full-network state reconstruction
  -> Step2 action-conditioned differentiable hydraulic trajectory
  -> continuous 109-actuator receding-horizon MPC
  -> H360 prediction / H120 free targets
  -> execute first 10 minutes only
  -> authoritative SWMM write/readback
  -> re-observe after 600 s and optimize again
```

Never change this hierarchy while debugging Step2:

- system-wide cumulative **TFV = primary**;
- Priority8 PFV = **one-sided soft secondary deterioration protection**;
- Global Peak = **report-only**;
- SWMM = authoritative truth.

Frozen clock/action contract: 300 s model step, 600 s control update, H360=72 model steps,
H120=24 model steps=12 ten-minute target blocks, 109 writable continuous actuators, 1308 free
MPC variables, execute first 10 minutes only.

RBC is a warm start, fallback and engineering comparator. It is not the Step2 value reference and
not an action-space ceiling.

## 2. Why the current Step2 changed

PR #77 repaired the earlier near-zero action sensitivity, but the next Stage-A smoke showed that
the historical all-H72 teacher-forced action-to-flow metric was not a valid local actuator gate:

```text
full-H72 flow sign                 ~0.8516
full-H72 flow cosine               ~-0.1016
full-H72 predicted/true L2 ratio   ~0.8655
Stage-A TFV-gradient sign          ~0.4681
Stage-A TFV-gradient cosine        ~0.1751
```

Detailed diagnosis showed that later candidate/reference flow differences include hydraulic
feedback because each branch is conditioned on its already-diverged authoritative previous state
and flow. A single high-energy actuator can dominate the micro/global cosine. Therefore later H72
flow difference is a **network-feedback trajectory quantity**, not a local same-state `dq/du`.

The current correction is **counterfactual trajectory first**:

1. direct actuator labels come only from the first setting-divergence transition while reference
   and candidate still share the same authoritative hydraulic prefix;
2. temporal flow variation and direct setting-response scales remain separate;
3. the hydraulic transition is pre-trained with authoritative managed-flow injection before joint
   predicted-flow training;
4. sparse direct node effects use response-weighted loss rather than being diluted over ~932 nodes;
5. B0 is where full autoregressive network feedback belongs;
6. the exact H360 TFV objective remains downstream;
7. SWMM action-gradient labels are **not** a primary training objective. Gradients are Development
   diagnostics and, after surrogate promotion, an efficient online optimization signal.

## 3. Current code surface

Read the machine contracts before running anything:

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
configs/project7_current_lint_surface.json
```

Active Step2 files:

```text
scripts/run_step2_current.py
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_counterfactual_first_v128.py
src/rtc/step2_counterfactual_training_v4.py
src/rtc/step2_action_identifiable_v128.py
src/rtc/step2_differentiable_v128_edge.py
src/rtc/edge_physics_current_v128.py
src/rtc/step2_train_v128_exact.py
```

Current Development diagnostics:

```text
scripts/audit_step2_actuator_flow_effect_current.py
scripts/audit_step2_direct_hydraulic_effect_current.py
scripts/audit_step2_spatial_current.py
scripts/audit_step2_gradient_stage_current_dev.py
scripts/audit_step2_gradient_current_dev.py
```

`--profile full` is intentionally blocked. Strict historical full checkpoint creation also rejects
the unpromoted Development subclass. A future explicit promotion must define a matching production
checkpoint/runtime loader; do not bypass this guard.

## 4. Stage-A semantics

### A0 — direct setting -> managed flow

Train the actuator submodel with absolute managed-flow fit plus direct same-prefix
candidate-reference response. Only the first action divergence is a local actuator label. Full
H72 feedback differences are forbidden from the direct scale/loss.

### A1 — managed flow -> node hydraulics

Freeze actuator parameters. Inject authoritative SWMM managed flow and train one-step hydraulic
state propagation plus response-weighted direct next-state effect. This isolates the downstream
hydraulic transition without fabricating ordinary-conduit flow truth.

### A2 — joint direct counterfactual teacher forcing

Unfreeze the full model. Retain absolute one-step state/flow fit and direct same-prefix flow/state
effect losses using predicted managed flow.

Large truth arrays remain mmap-backed. Direct-pair extraction materializes only reference/candidate
slices up to the first divergence step.

## 5. Development boundaries

Until explicit promotion:

- no Validation, Final, Formal or Policy Lock;
- no D4-AUDIT/D5-AUDIT/held-out outcomes in training or numerical scale construction;
- no future realised rainfall or future SWMM state online;
- no new D2/D3/D4 SWMM outcomes merely because code changed;
- no fabricated ordinary-conduit flow labels;
- incomplete continuity proxy remains diagnostic-only;
- no hydraulic-influence graph/deeper-GNN promotion unless evidence isolates a spatial bottleneck;
- smoke/dev stage checkpoints are NONFINAL and cannot enter D5/runtime/Policy Lock.

## 6. Workstation contract

```text
GPU: RTX 4060 Laptop ~8 GB
RAM: 16 GB
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RTC_V128_MATMUL_PRECISION=high
AMP=OFF
activation checkpointing=OFF
SWMM workers<=16; one thread/process
```

Do not use raw Torch Chrome trace. Use `TRAINING_TELEMETRY.jsonl` plus low-frequency external
`nvidia-smi` sampling.

## 7. Hard-sync local code

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin --prune
git switch main
git reset --hard origin/main
git clean -fd
git rev-parse HEAD
git rev-parse origin/main
git status --short --branch
python -m pip install -e ".[dev,swmm]"
python -c "import importlib.metadata as m; print(m.version('wuhan-rtc'))"
```

HEAD must equal origin/main and the working tree must be clean. Current package is `0.8.2.dev0`.
Never delete/reset the separate `study_v069` data root.

## 8. Cheap gates

```powershell
python -m pytest -q
python scripts/lint_current_surface.py
python scripts/run_step2_current.py --help
python scripts/build_edge_physics_current.py --help
python scripts/audit_step2_actuator_flow_effect_current.py --help
python scripts/audit_step2_direct_hydraulic_effect_current.py --help
python scripts/audit_step2_gradient_stage_current_dev.py --help
python scripts/audit_step2_spatial_current.py --help
```

All must exit 0. Do not use full-repository `ruff check .` as the smoke gate.

## 9. Reuse frozen assets

Known Development assets:

```text
<GRAPH>
E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz

<BASE_CACHE>
E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json

<D4_FIT>
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_fit\training_cache\CACHE_MANIFEST.json

<D4_AUDIT>
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_audit\training_cache\CACHE_MANIFEST.json

<RAIN>
E:\RTC_sewer\Project7\study_v069\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz

<STATE_V2>
E:\RTC_sewer\Project7\study_v069\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz
```

Resolve frozen `wuhan_method_testbed_v067.inp` from the study contract. Do not guess another INP.

## 10. Preflight and edge artifact

```powershell
$env:PATH += ';C:\Users\12480\AppData\Roaming\Python\Python314\Scripts'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:RTC_V128_MATMUL_PRECISION='high'
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$DebugRoot="E:\RTC_sewer\Project7\study_v069\debug_counterfactual_$Stamp"
New-Item -ItemType Directory -Path $DebugRoot -Force | Out-Null

rtc-current-preflight --graph <GRAPH> --device cuda --out "$DebugRoot\PREFLIGHT.json"

python scripts/build_edge_physics_current.py `
  --inp <FROZEN_INP> --graph <GRAPH> `
  --out-npz "$DebugRoot\EDGE_PHYSICS_CURRENT.npz" `
  --out-json "$DebugRoot\EDGE_PHYSICS_CURRENT.json"
$Edge="$DebugRoot\EDGE_PHYSICS_CURRENT.npz"
```

If reusing an edge artifact, verify INP SHA and graph SHA; filenames are not lineage evidence.

## 11. First expensive action: Stage A only

```powershell
$Run="$DebugRoot\smoke_stage_a"
python scripts/run_step2_current.py `
  --profile smoke --edge-physics $Edge `
  --graph <GRAPH> --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> --causal-state-store <STATE_V2> `
  --out-dir $Run --device cuda --stop-after-stage stage_a
```

Require logs for A0, A1 and A2 and a source-strict `stage_a.pt`. Any PR #77 checkpoint must fail
lineage reuse because model/training semantics changed.

## 12. Stage-A audits — then STOP

### 12.1 Direct vs feedback managed-flow effect

```powershell
python scripts/audit_step2_actuator_flow_effect_current.py `
  --profile smoke --stage stage_a --stage-checkpoint "$Run\stage_a.pt" `
  --graph <GRAPH> --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> --d4-fit-cache <D4_FIT> --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> --causal-state-store <STATE_V2> `
  --out-dir "$Run\direct_flow" --device cuda
```

**Primary:** `direct_same_prefix_metrics` plus `direct_actuator_balanced_metrics` and top truth-L2
contributors. `feedback_full_horizon_metrics` is secondary trajectory feedback and must never be
the sole promotion/rejection metric.

### 12.2 Normal-flow vs oracle-flow direct hydraulic effect

```powershell
python scripts/audit_step2_direct_hydraulic_effect_current.py `
  --profile smoke --stage stage_a --stage-checkpoint "$Run\stage_a.pt" `
  --graph <GRAPH> --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> --d4-fit-cache <D4_FIT> --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> --causal-state-store <STATE_V2> `
  --out "$Run\DIRECT_HYDRAULIC_EFFECT.json" --device cuda
```

If oracle managed flow materially improves node response while normal flow is poor, `setting->q`
remains the bottleneck. If oracle and normal are both poor, `q->hydraulics` remains primary.

### 12.3 Stage-A TFV gradient diagnostic

Run `scripts/audit_step2_gradient_stage_current_dev.py` on the same checkpoint/edge/data lineage.
Interpret it **after** direct trajectory evidence. A large or pretty gradient cannot promote a
surrogate whose direct/trajectory effects are wrong.

STOP here and report. Do not automatically resume B0.

## 13. Promotion funnel after web-GPT review

Only an explicitly approved Stage-A candidate may continue:

```text
Stage A pass
  -> resume B0 from stage_a.pt
  -> trajectory/ranking/spatial audits
  -> if useful, H360 exact objective
  -> ranking/top1/regret + gradient diagnostic on same checkpoint
  -> deterministic dev profile confirmation
  -> Development closed-loop SWMM comparison
  -> explicit future code promotion enabling full
  -> strict promoted checkpoint/D5/runtime/Policy Lock
```

Gradient is not a standalone promotion gate. Candidate ranking, trajectory/action-effect fidelity,
causality, runtime and authoritative SWMM closed-loop performance remain required.

## 14. Stage-A return format

Return only:

1. exact Git SHA/package version and cheap gates;
2. graph/INP/edge/data lineage;
3. A0/A1/A2 losses and pair counts;
4. temporal vs direct action-scale telemetry;
5. direct same-prefix flow micro + actuator-balanced macro + top contributors;
6. full-H72 feedback flow metrics separately;
7. normal-vs-oracle direct depth/flood-rate effects;
8. Stage-A TFV gradient diagnostic;
9. wall time/RAM/swap/VRAM and nonfinite/OOM status;
10. one classification: `DIRECT_FLOW_FAILURE`, `HYDRAULIC_TRANSITION_FAILURE`, `MIXED`, or
    `STAGE_A_WORTH_B0`;
11. STOP.
