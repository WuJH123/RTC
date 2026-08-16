# Project7 current execution guide

This is the **only user/Codex start document** for the current repository. GitHub `main` is the
code source of truth. Do not infer the active method from the highest V-number in a filename.

The immediate task is Development failure elimination, not Formal evidence. Step2 must first learn
the causal counterfactual hydraulic trajectory and candidate ordering required by continuous MPC.
Autograd TFV sensitivity is evaluated downstream as a solver signal; it is not a SWMM-labelled
primary training target.

## 1. Frozen research problem

Project7 is an **idealized EPA-SWMM methodology testbed, not a field digital twin**.

```text
causal sparse sensing
 -> Step1 current full-network state reconstruction
 -> Step2 differentiable action-conditioned hydraulic trajectory
 -> 109-actuator continuous receding-horizon MPC
 -> H360 prediction / H120 free targets
 -> engineering envelope before scoring
 -> execute first 10 minutes only
 -> authoritative SWMM write/readback
 -> re-observe after 600 s and optimize again
```

Never change this hierarchy while debugging:

- system-wide cumulative **TFV = primary**;
- Priority8 PFV = **one-sided soft secondary deterioration protection**;
- Global Peak = **report-only**;
- SWMM = authoritative truth.

Frozen clock/action contract: 300 s model step, 600 s control update, H360=72 model steps,
H120=24 model steps=12 ten-minute target blocks, 109 writable continuous actuators, 1308 free
MPC variables, execute first 10 minutes only. RBC is warm start/fallback/comparator, not a value
reference or action-space ceiling.

## 2. What the last smoke actually showed

```text
historical full-H72 flow sign                 ~0.8516
historical full-H72 flow cosine               ~-0.1016
historical full-H72 predicted/true L2 ratio   ~0.8655
historical full-H72 median magnitude ratio    ~0.9046
Stage-A TFV-gradient sign                      ~0.4681
Stage-A TFV-gradient cosine                    ~0.1751
Stage-A TFV-gradient L2 ratio                  ~1.5568
```

The historical H72 flow metric mixed direct actuator response with later hydraulic feedback, and a
single high-energy actuator could dominate the global cosine. Therefore later H72 flow difference
is a **network-feedback trajectory quantity**, not local same-state `dq/du`.

The source audit then found two additional P0s:

1. oracle-flow hydraulics still admitted a branch-varying `setting -> typed message -> state` side
   channel;
2. older Stage-A/B0 code imported streaming functions by value before lazy mmap hooks were
   installed, so host-memory laziness was not guaranteed.

Both are corrected in the current surface.

## 3. Current correction — counterfactual trajectory first

### A0 — direct same-prefix setting -> managed flow

Only the first setting divergence is a local actuator label. Reference/candidate must share one
authoritative pre-action state and previous managed flow. Temporal flow scale and direct setting
response scale are separate.

### A1 — strict managed-flow-only -> hydraulics

Freeze actuator parameters and inject authoritative SWMM managed flow. In the direct pair, both
branches share the reference setting inside the typed action context, so authoritative managed
flow is the only branch-varying control signal reaching the hydraulic transition.

### A2 — joint direct counterfactual teacher forcing

Unfreeze the model and train absolute state/flow plus response-weighted same-prefix flow/state
effects using predicted managed flow.

### B0 — autoregressive network feedback

Only after Stage A passes, train multi-step candidate/reference network feedback. Full feedback
flow effects are normalized by actuator flow standard deviation, not the direct-action scale.

### H360 exact TFV objective

Only after B0 ranking/horizon/spatial evidence is coherent, run the source-strict H360 two-pass
within-group pairwise TFV objective and one low-LR trajectory rehearsal. SWMM action gradients are
never training labels.

## 4. Active code surface

Machine contracts:

```text
configs/step2_current_contract.json
configs/project7_execution_registry.json
configs/v128_control_execution.json
configs/project7_current_lint_surface.json
```

Active implementation/diagnostics:

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
scripts/audit_step2_actuator_flow_effect_current.py
scripts/audit_step2_direct_hydraulic_effect_current.py
scripts/audit_step2_stage_ranking_current.py
scripts/audit_step2_spatial_current.py
scripts/audit_step2_gradient_stage_current_dev.py
scripts/audit_step2_gradient_current_dev.py
```

Superseded V3/V4 counterfactual Stage-A files are not current execution surfaces. The V4 file
created during this PR was deleted before merge.

## 5. Development boundaries

Until explicit production promotion:

- no Validation, Final, Formal or Policy Lock;
- no D4-AUDIT/D5-AUDIT/InternalHoldout/D2-holdout outcomes in training or scale fitting;
- no future realised rainfall or future SWMM hydraulic truth online;
- no new D2/D3/D4 SWMM outcomes merely because code changed;
- no fabricated ordinary-conduit dynamic-flow labels;
- incomplete continuity proxy remains diagnostic-only;
- no deeper-GNN/influence-graph promotion unless held-out spatial evidence isolates a distance
  bottleneck;
- smoke/dev stage checkpoints are NONFINAL;
- `--profile full`, D5, runtime and seven-strategy Proposed evaluation stay blocked until a future
  production checkpoint factory/loader/runtime is explicitly implemented.

Do not route an older base-V128 checkpoint as the current Proposal to bypass a fail-closed error.

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

Require HEAD == origin/main, clean tree and package `0.8.2.dev0`. Never reset/delete the separate
`study_v069` data root.

## 8. Cheap gates — all must pass before GPU training

```powershell
python -m pytest -q
python scripts/lint_current_surface.py
python scripts/run_step2_current.py --help
python scripts/build_edge_physics_current.py --help
python scripts/audit_step2_actuator_flow_effect_current.py --help
python scripts/audit_step2_direct_hydraulic_effect_current.py --help
python scripts/audit_step2_stage_ranking_current.py --help
python scripts/audit_step2_spatial_current.py --help
python scripts/audit_step2_gradient_stage_current_dev.py --help
python scripts/audit_step2_gradient_current_dev.py --help
python scripts/run_policy_current.py --help
python scripts/run_seven_strategies_current.py --help
python scripts/run_policy_current.py --promotion-status
python scripts/run_seven_strategies_current.py --promotion-status
```

The last two must report `runtime_enabled=false` and `seven_strategy_enabled=false`.

## 9. Frozen Development assets

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

## 10. Preflight and deterministic edge artifact

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

## 11. FIRST expensive action — Stage A only

```powershell
$Run="$DebugRoot\smoke"
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

Require a new `stage_a.pt` and:

```text
A0 PROJECT7_V128_DIRECT_SAME_PREFIX_FLOW_A0_V5_EXPLICIT_LAZY
A1 PROJECT7_V128_ORACLE_FLOW_ONLY_HYDRAULIC_A1_V5_SETTING_BYPASS_BLOCKED
A2 PROJECT7_V128_JOINT_DIRECT_HYDRAULIC_A2_V5_EXPLICIT_LAZY
explicit_lazy_mmap_helpers=true
A1 direct_pair_setting_bypass_blocked=true
A1 oracle_isolation_contract=PROJECT7_V128_ORACLE_MANAGED_FLOW_ONLY_HYDRAULIC_ISOLATION_V1
gradient_label_used=false
```

Any older PR #77/V4 stage checkpoint must fail source/lineage resume.

## 12. Stage-A direct-flow audit

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
  --out-dir "$Run\direct_flow_stage_a" `
  --device cuda
```

Primary gate: same-prefix direct micro metrics, actuator-balanced macro metrics, informative actuator
coverage and top truth-L2 contributors. Full-H72 feedback micro metrics are secondary only.

## 13. Stage-A strict q-only hydraulic audit

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
  --out "$Run\DIRECT_HYDRAULIC_STAGE_A.json" `
  --device cuda
```

Require:

```text
oracle_direct_pair_setting_bypass_blocked=true
oracle_branch_varying_control_signal=authoritative_managed_flow_only
depth.normal_vs_swmm
depth.oracle_flow_only_vs_swmm
flood_rate.normal_vs_swmm
flood_rate.oracle_flow_only_vs_swmm
```

Interpretation:

- direct flow poor -> actuator submodel/scale failure;
- direct flow good but q-only oracle hydraulics poor -> hydraulic transition/missing-state failure;
- q-only oracle good but normal hydraulics poor -> setting-to-flow remains bottleneck;
- both credible -> eligible for B0 review.

## 14. Stage-A TFV gradient diagnostic

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

A large gradient cannot rescue wrong direct effects. Weak Stage-A gradient alone does not reject a
candidate whose direct chain is credible. **STOP here and review evidence before B0.**

## 15. Conditional B0 — only after Stage-A approval

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

### 15.1 B0 source-strict ranking + H30-H360 trajectory audit

```powershell
python scripts/audit_step2_stage_ranking_current.py `
  --profile smoke `
  --stage stage_b0 `
  --stage-checkpoint "$Run\stage_b0.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --ranking-out "$Run\B0_RANKING.json" `
  --horizon-out "$Run\B0_HORIZON.json" `
  --telemetry-out "$Run\B0_RANKING_TELEMETRY.json" `
  --device cuda
```

This audit accepts NONFINAL stage checkpoints; do **not** call production-only
`audit_step2_v128_fast.py` as a workaround.

### 15.2 B0 spatial action-effect audit

```powershell
python scripts/audit_step2_spatial_current.py `
  --profile smoke `
  --stage stage_b0 `
  --stage-checkpoint "$Run\stage_b0.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out "$Run\B0_SPATIAL.json" `
  --device cuda
```

### 15.3 Repeat direct flow / q-only hydraulics / gradient on `stage_b0`

Use the Stage-A commands above with:

```text
--stage stage_b0
--stage-checkpoint "$Run\stage_b0.pt"
```

Only proceed if B0 improves useful ranking/trajectory without destroying direct-flow/q-only
hydraulic behavior. If near-field is good but distant effects collapse monotonically, then open a
separate Development edge-depth/influence-graph ablation. Do not blindly stack GNN layers.

## 16. Conditional exact H360 TFV objective

Only a B0 winner may continue:

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

Then repeat:

```powershell
python scripts/audit_step2_stage_ranking_current.py `
  --profile smoke `
  --stage objective `
  --stage-checkpoint "$Run\stage_objective.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --ranking-out "$Run\OBJECTIVE_RANKING.json" `
  --horizon-out "$Run\OBJECTIVE_HORIZON.json" `
  --telemetry-out "$Run\OBJECTIVE_RANKING_TELEMETRY.json" `
  --device cuda

python scripts/audit_step2_spatial_current.py `
  --profile smoke `
  --stage objective `
  --stage-checkpoint "$Run\stage_objective.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out "$Run\OBJECTIVE_SPATIAL.json" `
  --device cuda

python scripts/audit_step2_gradient_current_dev.py `
  --profile smoke `
  --stage-checkpoint "$Run\stage_objective.pt" `
  --graph <GRAPH> `
  --edge-physics $Edge `
  --cache-manifest <BASE_CACHE> `
  --d4-fit-cache <D4_FIT> `
  --d4-audit-cache <D4_AUDIT> `
  --causal-store <RAIN> `
  --causal-state-store <STATE_V2> `
  --out-dir "$Run\OBJECTIVE_GRADIENT" `
  --device cuda
```

Also repeat direct-flow and q-only hydraulic audits with `--stage objective`. Require useful TFV
candidate ranking/top-1/regret, acceptable horizon error growth, preserved action effects and a
nonpathological solver gradient. Gradient is not a standalone promotion gate.

## 17. Conditional deterministic `dev` confirmation

Only a smoke winner should be repeated under `--profile dev`, preserving Stage A -> STOP -> B0 ->
STOP -> objective stop gates and the same frozen assets. Do not access Validation/Final/Formal or
Policy Lock.

## 18. Future production-promotion PR — mandatory before D5/runtime/seven strategies

After Development evidence passes, a separate atomic source change must:

1. freeze the exact counterfactual-first model/training/source contract;
2. implement a production checkpoint saver **and loader for the exact promoted model class**;
3. bind graph/edge/state/rainfall/engineering/source hashes;
4. explicitly enable `--profile full` and create a strict promoted checkpoint;
5. run D5-FIT only first; keep D5-AUDIT untouched;
6. verify ranking and continuous-control sensitivity on the exact final checkpoint;
7. enable `run_policy_current.py` only after loader verification;
8. verify latency <600 s, score==execute, target write/readback and first-10-min execution;
9. enable seven-strategy Development comparison;
10. only after all pre-lock gates pass discuss Policy Lock / untouched Validation/Final.

Until then these normal invocations must fail:

```powershell
python scripts/run_policy_current.py
python scripts/run_seven_strategies_current.py
```

That is correct fail-closed behavior.

## 19. Return format after the next local run

After Stage A, return only:

1. exact Git SHA/package version and cheap gates;
2. graph/INP/edge/data lineage;
3. A0/A1/A2 contracts, losses, pair counts and explicit-lazy flags;
4. temporal vs direct action-scale telemetry;
5. direct same-prefix flow micro + actuator-balanced macro + top truth-L2 contributors;
6. full-H72 feedback flow metrics separately;
7. strict q-only oracle depth/flood metrics and normal-vs-oracle comparison;
8. Stage-A TFV gradient diagnostic;
9. wall time/RAM/swap/VRAM and nonfinite/OOM status;
10. exactly one classification:
    - `DIRECT_FLOW_FAILURE`
    - `Q_TO_HYDRAULICS_FAILURE`
    - `SETTING_TO_FLOW_REMAINS_BOTTLENECK`
    - `MIXED`
    - `STAGE_A_WORTH_B0`
11. STOP.
