# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. GitHub `main` is the
code source of truth. Do not infer the active method from the highest V-number in a filename.

The immediate task is Development failure elimination, not Formal evidence. The Step2 surrogate
must first learn the causal counterfactual hydraulic trajectory and candidate ordering needed by
continuous MPC. Autograd TFV sensitivity is assessed downstream as a solver signal; it is not a
SWMM-labelled primary training target.

## 1. Frozen research problem — never redesign while debugging

Project7 is an **idealized EPA-SWMM methodology testbed, not a field digital twin**.

```text
causal sparse sensing
  -> Step1 current full-network state reconstruction
  -> Step2 action-conditioned differentiable hydraulic trajectory
  -> continuous 109-actuator receding-horizon MPC
  -> H360 prediction / H120 free targets
  -> engineering envelope before scoring
  -> execute first 10 minutes only
  -> authoritative SWMM write/readback
  -> re-observe after 600 s and optimize again
```

Frozen hierarchy:

- system-wide cumulative **TFV = primary**;
- Priority8 PFV = **one-sided soft secondary deterioration protection**;
- Global Peak = **report-only**;
- SWMM = authoritative truth.

Frozen clock/action contract: 300 s model step, 600 s control update, H360=72 model steps,
H120=24 model steps=12 ten-minute target blocks, 109 writable continuous actuators, 1308 free
MPC variables, execute first 10 minutes only.

RBC is a warm start, fallback and engineering comparator. It is not the Step2 value reference and
not an action-space ceiling.

## 2. What the latest smoke actually proved

The PR #77 Stage-A smoke did **not** justify continuing B0. It showed:

```text
historical full-H72 flow sign                 ~0.8516
historical full-H72 flow cosine               ~-0.1016
historical full-H72 predicted/true L2 ratio   ~0.8655
historical full-H72 median magnitude ratio    ~0.9046
Stage-A TFV-gradient sign                      ~0.4681
Stage-A TFV-gradient cosine                    ~0.1751
Stage-A TFV-gradient L2 ratio                  ~1.5568
```

The important diagnosis is that the historical H72 flow comparison was not a clean local actuator
metric. After candidate/reference hydraulic states have diverged, the flow difference contains
network feedback. One high-energy actuator can dominate a global cosine. Therefore full-H72 flow
response is a **trajectory-feedback quantity**, not a local same-state `setting -> flow` label.

A second code audit found two hidden P0 issues:

1. the old oracle-flow transition still passed branch-varying candidate setting into the typed
   action message, so `setting -> message -> state` could bypass authoritative managed flow;
2. the older Stage-A/B0 modules imported `_cpu_group` / `_select_to_device` by value before lazy
   streaming was installed, so later monkey-patching did not guarantee mmap-backed truth loading.

Both are fixed in the current Development surface.

## 3. Current scientific correction — counterfactual trajectory first

### A0 — direct same-prefix setting -> managed flow

Use only the **first setting divergence** for a local actuator effect. Reference and candidate must
share one authoritative pre-action state and previous managed flow. Temporal `q(t)-q(t-1)` scale
and direct action-response scale are separate. Later feedback cannot inflate the local scale.

### A1 — strict managed-flow-only -> hydraulics

Freeze actuator parameters and inject authoritative SWMM managed flow. Absolute teacher-forced A1
may use each branch's causal setting, but the **direct reference/candidate isolation** shares the
reference setting inside the typed action context. Thus authoritative managed flow is the only
branch-varying control signal reaching the transition. If this q-only oracle cannot reproduce the
direct hydraulic effect, the problem is downstream of the actuator model.

### A2 — joint direct counterfactual teacher forcing

Unfreeze the full model. Train absolute next-state/managed-flow prediction plus response-weighted
same-prefix flow and state effects using predicted managed flow.

### B0 — autoregressive network feedback

Only after Stage A passes, learn multi-step candidate/reference network feedback. B0 flow-effect
normalization uses actuator flow standard deviation, not the local direct-action scale. Full-H72
feedback belongs here.

### H360 exact TFV objective

Only after B0 trajectory/ranking/spatial evidence is coherent, run the existing source-strict
H360 two-pass within-group pairwise TFV objective. Follow it with one low-LR trajectory rehearsal.
No SWMM action-gradient labels are trained.

## 4. Active code surface

Read these machine contracts first:

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
configs/project7_current_lint_surface.json
```

Active Step2 implementation:

```text
scripts/run_step2_current.py
scripts/run_step2_action_identifiable_current.py
src/rtc/step2_counterfactual_first_v128.py
src/rtc/step2_oracle_isolation_v128.py
src/rtc/step2_counterfactual_training_v5.py
src/rtc/step2_differentiable_v128_edge.py
src/rtc/step2_train_v128_exact.py
src/rtc/step2_current_dev_context_v128.py
src/rtc/step2_lazy_stream_v128.py
```

Current Development diagnostics:

```text
scripts/audit_step2_actuator_flow_effect_current.py
scripts/audit_step2_direct_hydraulic_effect_current.py
scripts/audit_step2_gradient_stage_current_dev.py
scripts/audit_step2_gradient_current_dev.py
scripts/audit_step2_spatial_current.py
```

The active Stage A/B0/post-objective trajectory code calls the V128 lazy mmap helpers explicitly.
Do not replace it with superseded V3/V4 training modules.

## 5. Development boundaries

Until explicit production promotion:

- no Validation, Final, Formal or Policy Lock;
- no D4-AUDIT, D5-AUDIT, InternalHoldout or D2 held-out outcomes in training/scale fitting;
- no future realised rainfall, future SWMM hydraulic truth or future Internal trajectory online;
- no new D2/D3/D4 SWMM outcomes merely because source code changed;
- no fabricated ordinary-conduit dynamic-flow labels;
- incomplete continuity proxy remains diagnostic-only;
- no influence graph/deeper-GNN promotion unless held-out spatial evidence isolates a distance
  bottleneck;
- smoke/dev checkpoints are NONFINAL;
- `--profile full`, D5, runtime and seven-strategy Proposed evaluation stay blocked until a future
  production checkpoint factory/loader/runtime is explicitly implemented.

Do not bypass a fail-closed error by routing an older base-V128 checkpoint as the current Proposal.

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

Do not use raw Torch Chrome-trace export. Use `TRAINING_TELEMETRY.jsonl` and low-frequency
`nvidia-smi` sampling.

## 7. Hard-sync local code from GitHub

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

Require HEAD == origin/main, clean working tree and package `0.8.2.dev0`. Never reset/delete the
separate `study_v069` data root.

## 8. Cheap gates — all must pass before GPU training

```powershell
python -m pytest -q
python scripts/lint_current_surface.py
python scripts/run_step2_current.py --help
python scripts/build_edge_physics_current.py --help
python scripts/audit_step2_actuator_flow_effect_current.py --help
python scripts/audit_step2_direct_hydraulic_effect_current.py --help
python scripts/audit_step2_gradient_stage_current_dev.py --help
python scripts/audit_step2_gradient_current_dev.py --help
python scripts/audit_step2_spatial_current.py --help
python scripts/run_policy_current.py --help
python scripts/run_seven_strategies_current.py --help
python scripts/run_policy_current.py --promotion-status
python scripts/run_seven_strategies_current.py --promotion-status
```

The last two must report `runtime_enabled=false` / `seven_strategy_enabled=false`. Do not use
repository-wide `ruff check .` as the Development smoke gate.

## 9. Reuse frozen Development assets

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

Resolve the frozen `wuhan_method_testbed_v067.inp` from the study contract. Do not guess another
INP. Reuse an existing edge artifact only after validating both frozen INP SHA and graph SHA.

## 10. Preflight and edge artifact

```powershell
$env:PATH += ';C:\Users\12480\AppData\Roaming\Python\Python314\Scripts'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:RTC_V128_MATMUL_PRECISION='high'
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$DebugRoot="E:\RTC_sewer\Project7\study_v069\debug_counterfactual_v5_$Stamp"
New-Item -ItemType Directory -Path $DebugRoot -Force | Out-Null

rtc-current-preflight `
  --graph <GRAPH> `
  --device cuda `
  --out "$DebugRoot\PREFLIGHT.json"

python scripts/build_edge_physics_current.py `
  --inp <FROZEN_INP> `
  --graph <GRAPH> `
  --out-npz "$DebugRoot\EDGE_PHYSICS_CURRENT.npz" `
  --out-json "$DebugRoot\EDGE_PHYSICS_CURRENT.json"

$Edge="$DebugRoot\EDGE_PHYSICS_CURRENT.npz"
```

Require frozen graph/INP lineage, 109 actuators, CUDA, matmul `high`, AMP false and no outcome
labels in the edge artifact.

## 11. FIRST expensive action — Stage A only, then STOP

```powershell
$Run="$DebugRoot\smoke_stage_a"
python scripts/run_step2_current.py `
  --profile smoke `
  --edge-physics $Edge `
  --graph <GRAPH> `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out-dir $Run `
  --device cuda `
  --stop-after-stage stage_a
```

Require a new `stage_a.pt` and Stage-A log rows with the V5 contracts:

```text
A0 PROJECT7_V128_DIRECT_SAME_PREFIX_FLOW_A0_V5_EXPLICIT_LAZY
A1 PROJECT7_V128_ORACLE_FLOW_ONLY_HYDRAULIC_A1_V5_SETTING_BYPASS_BLOCKED
A2 PROJECT7_V128_JOINT_DIRECT_HYDRAULIC_A2_V5_EXPLICIT_LAZY
```

Also require:

```text
explicit_lazy_mmap_helpers=true
A1 direct_pair_setting_bypass_blocked=true
A1 oracle_isolation_contract=PROJECT7_V128_ORACLE_MANAGED_FLOW_ONLY_HYDRAULIC_ISOLATION_V1
gradient_label_used=false
```

Any PR #77/V4 stage checkpoint must fail source/lineage resume. Do not weaken that guard.

## 12. Stage-A audit 1 — direct and feedback managed-flow effects

```powershell
python scripts/audit_step2_actuator_flow_effect_current.py `
  --profile smoke `
  --stage stage_a `
  --stage-checkpoint "$Run\stage_a.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out-dir "$Run\direct_flow" `
  --device cuda
```

Primary gate:

- same-prefix direct micro sign/cosine/MAE/L2 ratio;
- actuator-balanced macro sign/cosine/magnitude;
- informative pair and actuator coverage;
- top truth-L2 contributors.

Secondary diagnostic only:

- historical full-H72 feedback micro metrics.

Never reject/promote solely from full-H72 global cosine.

## 13. Stage-A audit 2 — strict q-only hydraulic isolation

```powershell
python scripts/audit_step2_direct_hydraulic_effect_current.py `
  --profile smoke `
  --stage stage_a `
  --stage-checkpoint "$Run\stage_a.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out "$Run\DIRECT_HYDRAULIC_EFFECT.json" `
  --device cuda
```

Require output fields:

```text
oracle_flow_isolation_contract
oracle_direct_pair_setting_bypass_blocked=true
oracle_branch_varying_control_signal=authoritative_managed_flow_only
normal_flow_effect
depth.normal_vs_swmm
depth.oracle_flow_only_vs_swmm
flood_rate.normal_vs_swmm
flood_rate.oracle_flow_only_vs_swmm
```

Interpretation:

- direct flow poor -> actuator submodel/scale failure;
- direct flow good, q-only oracle hydraulics poor -> hydraulic transition / missing hydraulic state
  failure; STOP before B0;
- q-only oracle good, normal hydraulics poor -> setting-to-flow submodel remains the bottleneck;
- both direct flow and q-only hydraulics credible -> Stage A is eligible for B0 review.

## 14. Stage-A audit 3 — TFV gradient is diagnostic

Run the same-checkpoint Development gradient audit:

```powershell
python scripts/audit_step2_gradient_stage_current_dev.py `
  --profile smoke `
  --stage stage_a `
  --stage-checkpoint "$Run\stage_a.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out-dir "$Run\gradient_stage_a" `
  --device cuda
```

Do not demand a large gradient before the autoregressive world model exists. Do reject nonfinite,
pathologically unstable or causally impossible sensitivity. A large gradient cannot rescue wrong
direct effects.

**STOP after the three Stage-A audits.** Return the evidence for review before B0.

## 15. Conditional B0 — only after Stage-A approval

Resume the exact V5 Stage-A checkpoint rather than retraining another model:

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --edge-physics $Edge `
  --graph <GRAPH> `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out-dir $Run `
  --device cuda `
  --resume-from "$Run\stage_a.pt" `
  --stop-after-stage stage_b0
```

Require `PROJECT7_V128_COUNTERFACTUAL_AUTOREGRESSIVE_B0_V5_EXPLICIT_LAZY_FEEDBACK` and
`explicit_lazy_mmap_helpers=true`.

Before H360 objective, run same-checkpoint:

- action-flow audit at `stage_b0`;
- direct q-only hydraulic audit at `stage_b0`;
- trajectory/ranking audit using the current ranking script;
- `scripts/audit_step2_spatial_current.py` by graph-distance bins;
- stage-B0 gradient diagnostic.

If near-field is good but distant action effects collapse monotonically, then and only then open a
separate Development architecture change for edge depth/influence graph. Do not blindly add many
message-passing layers: depth alone is not a guaranteed cure for long-range graph sensitivity.

## 16. Conditional H360 exact objective

Only a B0 candidate with useful trajectory/ranking/spatial evidence may continue:

```powershell
python scripts/run_step2_current.py `
  --profile smoke `
  --edge-physics $Edge `
  --graph <GRAPH> `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out-dir $Run `
  --device cuda `
  --resume-from "$Run\stage_b0.pt" `
  --stop-after-stage objective
```

After `stage_objective.pt`, require on the **same checkpoint**:

- TFV candidate sign/ranking quality;
- top-1 selection accuracy and regret;
- trajectory/action-effect fidelity not destroyed by objective training;
- spatial bins remain coherent;
- H360 TFV gradient sign/cosine/magnitude as a solver diagnostic;
- no nonfinite/OOM/source-lineage mismatch.

Gradient is not a standalone promotion gate.

## 17. Conditional deterministic `dev` confirmation

Only a smoke winner should be repeated under `--profile dev`. Repeat Stage A -> B0 -> objective
with the same frozen assets and contracts, preserving all stop gates. Do not touch Validation,
Final, Formal or Policy Lock.

The next scientific evidence after a dev winner is **Development authoritative closed-loop SWMM**,
but the current production wrappers remain fail-closed until an explicit production model loader
exists. Do not bypass that by calling archival V128 runtime scripts manually.

## 18. Future production-promotion PR — required before D5/runtime/seven strategies

After Development evidence passes, a separate source change must implement all of the following as
one atomic promotion:

1. freeze the exact counterfactual-first model/training/source contract;
2. implement a production checkpoint saver **and loader for the exact promoted model class**;
3. bind edge artifact, graph, state/rainfall stores, engineering decoder and source hashes;
4. enable full profile explicitly and create a strict promoted checkpoint;
5. run D5-FIT only first; keep D5-AUDIT untouched;
6. verify candidate ranking and continuous-control gradient on the exact final checkpoint;
7. enable `run_policy_current.py` for authoritative Development SWMM only after loader verification;
8. verify decision latency <600 s, score==execute, target write/readback and first-10-min execution;
9. enable seven-strategy Development comparison;
10. only after all pre-lock gates pass may Policy Lock / untouched Validation/Final be discussed.

Until that PR exists, the following normal executions must fail closed:

```powershell
python scripts/run_policy_current.py
python scripts/run_seven_strategies_current.py
```

That failure is correct behavior, not a missing feature.

## 19. Stage-A return format

After the next local run, return only:

1. exact Git SHA/package version and cheap gates;
2. graph/INP/edge/data lineage;
3. A0/A1/A2 contracts, losses, pair counts and explicit-lazy flags;
4. temporal vs direct action-scale telemetry;
5. direct same-prefix flow micro + actuator-balanced macro + top truth-L2 contributors;
6. full-H72 feedback flow metrics separately;
7. strict q-only oracle depth/flood-rate metrics and normal-vs-oracle comparison;
8. Stage-A TFV gradient diagnostic;
9. wall time/RAM/swap/VRAM and nonfinite/OOM status;
10. exactly one classification:
    - `DIRECT_FLOW_FAILURE`
    - `Q_TO_HYDRAULICS_FAILURE`
    - `SETTING_TO_FLOW_REMAINS_BOTTLENECK`
    - `MIXED`
    - `STAGE_A_WORTH_B0`
11. STOP.
