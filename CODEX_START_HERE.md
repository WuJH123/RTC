# Project7 Practical RTC — authoritative local execution guide

This is the single startup guide for local Codex. Historical V* filenames are evidence/ablation, not
execution selectors. Web GPT owns scientific/code changes; local Codex discovers existing files,
executes the frozen commands, monitors resources and returns evidence/errors.

## 0. Git and collaboration contract

Never destroy the user's local documentation edits. Do not use `reset --hard`, `clean`, `restore .`,
`checkout .` or blind stash-pop.

For the current refactor, synchronize the exact PR branch/head supplied by Web GPT. After a later
scientific merge, switch to `main` only when explicitly instructed.

## 1. Frozen paper question

Every 600 s:

`causal sparse history -> Step1 reconstruction -> frozen Step2 H10 proposals -> <=4 supported first-action candidates -> receding-policy-return critic/admission -> execute H10 or HOLD -> authoritative SWMM write/readback -> observe again`.

Frozen contract:

- model/observation step = 300 s;
- supervisory decision = 600 s;
- value/forecast context = H360;
- only first H10 target executes before replanning;
- 109 writable actuators screened;
- actuator bounds and target slew <=0.5 per update;
- anchor = previous supervisory `target_setting`;
- unchanged facilities retain their previous target;
- HOLD = latch previous supervisory target;
- q95 TrainFit first-move density and joint-sequence support;
- authoritative truth = SWMM.

Objective hierarchy:

- online objective: system-wide cumulative TFV only;
- authoritative spatial safety for positive claims: `Priority8 PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`;
- Global Peak: report only.

Forbidden online: future realised rainfall/state/flooding/Internal trajectory, online SWMM candidate
search, PFV/Peak/action penalty, Auto-RBC/EFD warm start or baseline imitation.

## 2. Hybrid H10 proposal layer

The historical 12 x 109 = 1308-dimensional L-BFGS-B full-plan optimizer is not current. At each
causal state the current portfolio may contain at most four distinct candidates:

1. `STEP2_H10_PROBE_SCALE_1.00`;
2. `STEP2_H10_PROBE_SCALE_0.50`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`;
4. `SUPPORT_CONSTRAINED_GRADIENT_H10`.

The fourth candidate is a **109-D H10-only projected-gradient proposer** through frozen base Step2.
Every gradient trial is projected to bounds, 0.5 slew, per-facility q95 first-move radius and q95
changed-facility ceiling. It never optimizes a 12-block plan. The base-Step2 gradient score cannot
authorize execution; all candidates must still be ranked by the separately trained receding-policy-
return critic and admitted by its matched one-sided UCB.

The critic action token is exactly:

`H10 candidate target -> H350 HOLD target` versus `HOLD H360`.

This token is a first-action counterfactual representation, not an instruction to HOLD the real
network for 350 min. The actual controller re-observes and replans every 10 min.

## 3. First policy iteration

Historical V12/L-BFGS-B is archival only. Current first-round parent is:

`PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V2`.

It uses the same four-family proposal/support geometry but ranks proposals only with frozen base Step2
before a policy-return critic exists. Exact paired SWMM creates Q^pi0/A^pi0 labels. After critic
training/calibration, pi1 uses the critic. If pi1 materially changes the state/action distribution, a
new role-disjoint Q^pi1 round is required before Policy Lock.

## 4. Cheap gates

Always run before expensive work:

```powershell
python -m pip install -e ".[dev,swmm]"
python -m compileall -q src scripts tests
python scripts/lint_current_surface.py
python -m pytest -q tests/test_direct_tfv_policy_return_portfolio.py tests/test_direct_tfv_hybrid_gradient_portfolio.py tests/test_practical_rtc_v14_contract.py tests/test_direct_tfv_runtime_adapter.py
python -m pytest -q
```

Then run `--help` for every current script used. Never invent flags from memory.

## 5. Discover existing assets once and freeze paths

Search existing evidence before asking the user for paths:

```text
E:\RTC_sewer\Project7\study_v069
E:\RTC_sewer\Project7\repo
existing JSON/report/manifest files
```

Resolve files by scientific contract/SHA/semantic lineage, not by the newest-looking V* filename.
Required current assets are only:

- graph;
- sensors;
- runtime config;
- Step1;
- base Step2 V5;
- q95 sequence support;
- Priority8 list.

No historical V12 policy-admission or first-move-admission file belongs in the current manifest.
After discovery run:

```powershell
python scripts/build_project7_practical_asset_manifest_current.py --help
```

Freeze one `PRACTICAL_RTC_ASSETS.json` with absolute paths + SHA-256. Reuse it for all downstream
commands. If a recorded file disappears or changes SHA, report the exact asset and stop; never search
for a replacement mid-run.

## 6. One-query scientific mechanism smoke before bulk labels

Use a previously seen T30 case only as Development debugging, not independent evidence. Prefer the
existing prepared INP when its SHA/clock remain valid. First generate the **current hybrid pi0** parent
trajectory:

```powershell
python scripts/run_policy_direct_tfv_base_hybrid_parent_current.py --help
```

Choose the query point deterministically before reading candidate outcomes. Prefer the first
post-readiness non-HOLD parent decision; if the historical decision index 3 / elapsed 5400 s is used,
record that it is a seen mechanism query only.

Capture the causal pre-action context:

```powershell
python scripts/capture_direct_tfv_policy_return_context_current.py --help
```

Design the hybrid portfolio:

```powershell
python scripts/design_direct_tfv_policy_return_portfolio_current.py --help
```

Require:

- 2-4 distinct supported candidates when physically available;
- Step2 scale family present;
- type-aware hydraulic candidate present when non-HOLD after support;
- `SUPPORT_CONSTRAINED_GRADIENT_H10` present when its projected target remains distinct after dedup;
- gradient dimension = 109 and horizon = H10 only;
- q95 first-move/joint support;
- `lbfgsb_used=false`;
- no future rainfall and no SWMM inside candidate design.

Run exact truth efficiently:

```powershell
python scripts/run_direct_tfv_policy_return_query_current.py --help
```

Use `--continuation-kind base-probe` for first-round pi0. One query uses one shared HOLD plus N
sequential candidate branches. Every row must share one `query_set_id`, raw authoritative prefix and
continuation-policy SHA. Report source, base-Step2 H10 score if available, true `candidate TFV - HOLD
TFV`, changed K, support ratios and true rank.

If raw prefix, continuation, causality, engineering/readback/support checks fail, stop. If every
candidate is harmful, diagnose representation/proposal coverage before bulk generation. Do not tune
conformal margins from this seen smoke.

## 7. Freeze role-disjoint data roles

Before reading new paired outcomes, freeze rainfall-group identities for:

- policy-return train: >=48 independent groups;
- policy-return validation/model selection: >=12 groups;
- conformal calibration: >=24 groups;
- separate new Development closed-loop probes.

Do not count old T5/T10/T20/T8/T30/T80 or other previously inspected Development outcomes as new
independent evidence. Validation/Final/Formal/Policy Lock data remain inaccessible.

Initially choose one deterministic causal query point per rainfall group to control SWMM cost. Query
selection must not depend on candidate outcome.

For each group:

`hybrid pi0 parent trajectory -> causal context -> hybrid portfolio -> one shared HOLD + all distinct candidates`.

## 8. Resource rules for exact paired data

Target workstation: RTX 4060 Laptop 8 GB, 16 GB RAM.

- start with one heavy GPU/SWMM process;
- candidate/HOLD continuation controllers are sequential and released between branches;
- use the shared-HOLD query runner, not N independent candidate/HOLD pair processes;
- do not run multiple PySWMM simulations in one Python process;
- only consider a second process after real RAM/VRAM preflight proves stable;
- BLAS/OpenMP threads = 1 under process parallelism.

## 9. Compile and train policy-return critic

```powershell
python scripts/compile_direct_tfv_policy_return_dataset_current.py --help
python scripts/train_direct_tfv_policy_return_current.py --help
```

The dataset and checkpoint must report the hybrid V4 candidate contract. Default trainable scope is
`control-heads`: facility/action/interaction layers only. Base Step2 is not retrained.

Model-selection priority:

1. selected-action false-beneficial fraction;
2. same-query pairwise rank accuracy;
3. selected-action regret;
4. event-balanced MAE.

Also report sign accuracy, false-reject rate and same-query top1. Do not promote a critic from MAE
alone.

## 10. Freeze critic and calibrate matched admission

```powershell
python scripts/score_direct_tfv_policy_return_calibration_current.py --help
python scripts/calibrate_direct_tfv_policy_return_portfolio_admission_current.py --help
```

Calibration rainfall groups are untouched by critic fitting/model selection. The admission set must
contain the actual online families: Step2 H10 probe, type-aware hydraulic pressure and
`SUPPORT_CONSTRAINED_GRADIENT_H10`. Old three-family calibration cannot authorize hybrid execution.

Do not reduce coverage/margin simply to increase ACTION count.

## 11. Authoritative pi1 Development closed loop

```powershell
python scripts/run_policy_direct_tfv_policy_return_development.py --help
```

Require:

- `portfolio_mode=true`;
- `projected_gradient_h10_enabled=true`;
- `online_lbfgsb_used=false`;
- `legacy_v12_admission_required_online=false`;
- 109 facilities screened;
- ACTION/HOLD and candidate-source counts;
- changed-K and q95 support distribution;
- runtime p50/p95/max with every decision well below 600 s;
- score==execute;
- zero target write/readback, engineering and support violations;
- zero future-rainfall/online-SWMM leakage;
- routing error acceptable under the frozen SWMM contract.

Near-all-HOLD behavior is not considered successful RTC. If admission rejects nearly every action,
inspect role-pure residuals, false rejection and candidate truth before changing any threshold.

## 12. Baselines and Priority8 PFV

Run unchanged on the exact same event/INP/clock/SWMM engine:

- No-control — primary reference;
- Internal RTC;
- Auto-RBC;
- EFD;
- all-max-setting / all-min-setting as diagnostics only.

Do not weaken Auto-RBC/EFD or use them as Proposed warm starts. The Proposed method is not required to
beat Auto-RBC on every rainfall event.

Compute authoritative comparison and then:

```powershell
python scripts/add_pfv_to_direct_tfv_comparison_current.py --help
```

TFV remains the only online objective. A positive performance claim for an event requires Priority8
PFV non-inferiority. Global Peak is report-only.

## 13. Promotion logic

Development can support the paper direction when technical gates are clean and the controller shows
useful event-balanced TFV behavior versus No-control without action starvation, while Internal RTC,
Auto-RBC and EFD are reported honestly. Universal comparator superiority is not required.

If pi1 materially differs from pi0, run a new role-disjoint Q^pi1 iteration before Policy Lock.

`READY_FOR_POLICY_LOCK=false` until role-disjoint train/validation/calibration, independent new
Development probes, Priority8 PFV-safe positive claims and any necessary policy iteration are complete.
Do not enter Validation, Final, Formal or Policy Lock automatically.
