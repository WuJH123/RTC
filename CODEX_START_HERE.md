# Project7 Practical RTC — authoritative local execution guide

This is the single startup guide for local Codex after Web GPT merges the Practical refactor to
`main`. Do not select a scientific path from historical V* filenames.

## 0. Collaboration and git contract

1. **Web GPT** owns scientific/code changes and merges GitHub.
2. **Local Codex** fast-forwards `main`, discovers existing local assets, runs/monitors the frozen
   commands and returns evidence/errors.
3. Never create an independent local scientific fork or redesign the objective from run outcomes.

Before every run:

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin --prune
git switch main
git pull --ff-only
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

Preserve the user's existing documentation edits. Never use `reset --hard`, `clean`, `restore .`,
`checkout .` or blind stash-pop to make the tree look clean.

## 1. Frozen paper question

Every 600 s:

`causal sparse history -> Step1 full-state reconstruction -> frozen Step2 H10 probes -> <=3 supported
first-action candidates -> policy-return critic/admission -> execute H10 or HOLD -> authoritative SWMM
write/readback -> observe again`.

Frozen control contract:

- model/observation step = 300 s;
- supervisory update = 600 s;
- value context horizon = H360;
- only the first H10 command is executed;
- 109 writable actuators are screened;
- actuator min/max and target movement <= 0.5 per update;
- supervisory slew anchor = previous `target_setting`, not lagged physical `current_setting`;
- unchanged facilities retain their previous commanded target;
- HOLD = latch the previous supervisory target;
- q95 TrainFit changed-facility and joint-sequence support remain canonical;
- authoritative truth = SWMM.

Objective hierarchy:

- **online primary**: system-wide cumulative TFV only;
- **authoritative secondary safety**: `Priority8 PFV_proposed <= 100 m3 + 1.05 * PFV_no_control` under
  the current project-specific non-inferiority contract;
- **report only**: Global Peak.

No online PFV surrogate, PFV/Peak penalty, future realised rainfall/state/flooding/Internal trajectory,
online SWMM candidate search, Auto-RBC/EFD warm start or baseline imitation.

## 2. Why the old continuous optimizer is not current

Historical Development established two separate problems:

1. the 12 x 109 L-BFGS-B full-plan optimizer could bind q95 density/support, exploit surrogate extrema
   and generate plans whose exact SWMM action effect was less reliable than their predicted value;
2. even when accepted target-latched H360 samples were locally correct, repeated H10 replanning did not
   reliably improve whole-event TFV.

Therefore the current Practical online policy **does not call L-BFGS-B**. Historical V12 remains only
an offline frozen parent `pi0` for the first paired policy-return label round and an ablation.

## 3. Exact receding-policy estimand and tensor encoding

Authoritative label:

`A^pi(x_t,u_t) = J(candidate H10 -> frozen pi) - J(HOLD H10 -> same frozen pi)`.

Candidate/HOLD branches must share the same raw causal SWMM prefix and the same frozen continuation.
The critic input uses the same intervention:

`H10 candidate target -> H350 HOLD target`.

The reference is HOLD H360. Persistent-H360 candidate encoding is forbidden for policy-return
training, calibration and runtime.

## 4. Practical online proposal layer

At one causal state:

1. batch-score +/- first-move probes for all 109 facilities with frozen base Step2;
2. each probe is H10 candidate -> H350 HOLD and remains inside first-move support;
3. combine only individually predicted-beneficial directions, capped by q95 changed-facility support;
4. form at most three candidates after support/dedup:
   - `STEP2_H10_PROBE_SCALE_1.00`;
   - `STEP2_H10_PROBE_SCALE_0.50`;
   - `TYPE_AWARE_HYDRAULIC_PRESSURE`;
5. apply q95 joint-sequence support to the actual H10-pulse geometry;
6. policy-return critic ranks candidates; execute the minimum upper-bound candidate only if its
   one-sided upper bound is negative, otherwise HOLD.

Auto-RBC, EFD, all-max-setting and all-min-setting are never candidate sources.

## 5. Cheap gates

Always run before expensive work:

```powershell
python -m pip install -e ".[dev,swmm]"
python -m compileall -q src scripts tests
python scripts/lint_current_surface.py
python -m pytest -q
```

Then run `--help` for every script used in the current stage. Never invent CLI flags from memory.

## 6. Auto-discover existing assets once, then freeze paths

Do **not** ask the user for paths before searching existing local evidence. Search:

```text
E:\RTC_sewer\Project7\study_v069
E:\RTC_sewer\Project7\repo
existing JSON/report/manifest files
```

Resolve by content contract/SHA/semantic lineage, not by newest-looking V* filename. Prefer already
frozen compatible assets and do not regenerate expensive SWMM/Step1/Step2 data merely because an
older filename is present.

Required frozen assets:

- graph;
- sensors;
- runtime config;
- Step1;
- base Step2 V5;
- historical policy admission and V12 first-move admission **only for offline pi0 label generation**;
- q95 sequence support;
- Priority8 list.

After discovery write exactly one path manifest:

```powershell
python scripts/build_project7_practical_asset_manifest_current.py --help
```

The resulting `PRACTICAL_RTC_ASSETS.json` contains absolute paths + SHA-256 and forbids silent path
fallback. Reuse it for all subsequent commands. If a recorded file disappears or its SHA changes,
stop and report the exact asset; never search for a replacement mid-run.

## 7. One-query mechanism smoke before bulk labels

Use the already-seen T30 decision-3 mechanism case first. It is engineering/scientific debugging, not
independent acceptance evidence.

### 7.1 Capture causal context without full paired truth

```powershell
python scripts/capture_direct_tfv_policy_return_context_current.py --help
```

This replays only until the selected branch point, captures the pre-action Step1/rain/target context
and intentionally stops. It produces no TFV truth.

### 7.2 Design the Practical portfolio

```powershell
python scripts/design_direct_tfv_policy_return_portfolio_current.py --help
```

Require:

- 2-3 distinct supported candidates when physically available;
- `lbfgsb_used=false`;
- no future rainfall;
- no online SWMM;
- H10 action semantics;
- q95 density/joint support.

### 7.3 Run exact same-query truth efficiently

```powershell
python scripts/run_direct_tfv_policy_return_query_current.py --help
```

This runner executes one shared HOLD branch and each candidate sequentially. All candidate records
must have the same `query_set_id`, raw causal prefix and continuation-policy SHA. It should use
`1 + N_candidates` full authoritative branches, not `2 * N_candidates`.

Report each candidate's true `candidate TFV - HOLD TFV`, changed-K and true rank. If raw prefix,
continuation, causality, engineering or support checks fail, stop. If all three candidate families are
consistently harmful in the mechanism query, stop before bulk generation and report a proposal/
representation bottleneck instead of tuning admission margins.

## 8. Role-disjoint policy-return data

Before reading new paired outcomes, freeze rainfall-group roles:

1. `policy_return_train`: >=48 independent groups;
2. `policy_return_validation`: >=12 model-selection groups;
3. `policy_return_calibration`: >=24 conformal-calibration groups;
4. separate new Development probes.

Use existing untouched Development rainfall groups when sufficient. New synthetic forcing is needed
only if the existing role-pure pool is insufficient. Validation/Final/Formal/Policy Lock data remain
inaccessible.

For each selected prefix:

`prefix-only context capture -> Practical candidate design -> shared-HOLD multi-candidate SWMM query`.

Do not generate generic D3 candidates for this workflow.

## 9. Compile and train the policy-return critic

```powershell
python scripts/compile_direct_tfv_policy_return_dataset_current.py --help
python scripts/train_direct_tfv_policy_return_current.py --help
```

The base Step2 representation is reused. Default fine-tuning adapts only facility/action/interaction
layers, not the full expensive representation.

Model selection priority is:

1. selected-action false-beneficial fraction;
2. same-query pairwise ranking;
3. selected-action regret;
4. event-balanced MAE.

Also report sign accuracy, false reject and within-query top1. Do not promote a critic merely because
scalar MAE improved.

## 10. Freeze critic, score calibration, calibrate one-sided admission

```powershell
python scripts/score_direct_tfv_policy_return_calibration_current.py --help
python scripts/calibrate_direct_tfv_policy_return_portfolio_admission_current.py --help
```

The critic is frozen before calibration. Admission uses rainfall-group residuals normalized by
`sqrt(actual first-move changed K)` and must be calibrated on the same Practical multi-candidate query
family/action encoding used online.

## 11. Authoritative Practical Development closed loop

Run only:

```powershell
python scripts/run_policy_direct_tfv_policy_return_development.py --help
```

This current runtime requires `PRACTICAL_RTC_ASSETS.json`, the H10 policy-return checkpoint and matched
portfolio admission. It must report:

- `portfolio_mode=true`;
- `online_lbfgsb_used=false`;
- `legacy_v12_admission_required_online=false`;
- 109 facilities screened;
- ACTION/HOLD counts and changed-K distribution;
- runtime p50/p95/max and every guarded callback <600 s;
- score==execute, target write/readback, support and engineering checks;
- routing error and fallback/deadline counts.

## 12. Baselines and PFV safety

For the exact same event/INP/clock/SWMM engine, report:

- No-control — primary reference;
- Internal RTC — operational comparator;
- Auto-RBC — type-aware operational comparator;
- EFD — type-aware storage equal-filling comparator;
- all-max-setting (`all_open`) — diagnostic extreme;
- all-min-setting (`all_closed`) — diagnostic extreme.

Do not weaken baselines because Proposed loses. All-max/min are not universal physical max/min release
policies and are not mandatory wins.

Compute TFV and Priority8 PFV from authoritative node statistics. Apply:

```powershell
python scripts/add_pfv_to_direct_tfv_comparison_current.py --help
```

A positive method-performance claim on an event requires the frozen PFV non-inferiority envelope.
Global Peak remains report-only.

## 13. Policy iteration

First label round uses frozen historical V12 as `pi0`. After training/calibration, define Practical
`pi1`. If pi1 materially differs from pi0, generate a **new role-disjoint** round using pi1 as the
shared continuation and learn Q^pi1. Continue only while action/return behavior materially changes.
Do not claim a fixed deployed value from Q^pi0 after the policy has materially changed.

## 14. Resource rules

Target workstation: RTX 4060 Laptop 8 GB, 16 GB RAM.

- one neural GPU controller/trainer per process;
- candidate/HOLD continuation branches run sequentially on the GPU;
- branch controllers must be released between branches;
- shared-HOLD query runner is preferred to repeated pair runs;
- no full base Step2 retrain unless later evidence identifies representation error;
- pure independent CPU/SWMM work may use process parallelism only after real RAM/I/O preflight;
- one PySWMM Simulation per Python process; BLAS/OpenMP threads = 1 under process parallelism.

## 15. Stop and promotion rules

Stop expensive downstream work for code/test/CLI failures, future-information leakage, raw-prefix or
continuation mismatch, irrecoverable asset SHA drift, CUDA OOM/paging that invalidates intended
execution, severe policy-return false-beneficial/ranking failure, or engineering/support/readback/
score-execute violations.

A single storm losing to an operational comparator is **not** by itself a code failure and must not
trigger baseline weakening or objective redesign. Judge TFV benefit event-balanced against No-control,
report Internal/Auto-RBC/EFD honestly, and enforce PFV non-inferiority for positive method claims.

`READY_FOR_POLICY_LOCK=false` until role-disjoint training/validation/calibration, independent new
Development probes, PFV-safe authoritative comparison and any necessary policy-iteration round are
complete. Do not enter Validation, Final, Formal or Policy Lock automatically.
