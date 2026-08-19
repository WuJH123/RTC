# Project7 Practical RTC — authoritative local execution guide

Web GPT owns the scientific/code contract. Local Codex discovers existing files, executes the frozen
current branch, monitors resources and returns evidence/errors. Historical V* filenames are evidence
or ablation only, not automatic execution selectors.

## 0. Git and local-edit safety

Never destroy the user's local documentation edits. Do not use `reset --hard`, `clean`, `restore .`,
`checkout .`, or blind stash-pop. Synchronize the exact PR branch/head supplied by Web GPT. Do not
switch to `main` unless explicitly instructed after a later scientific merge.

## 1. Frozen paper question

Every 600 s:

`causal sparse history -> Step1 full-state reconstruction -> masked H10 proposals -> <=4 candidates -> receding-policy-return critic/admission -> execute H10 or HOLD -> SWMM write/readback -> observe again`.

The sole online objective is system-wide cumulative TFV. Priority8 PFV is an authoritative secondary
non-inferiority condition for positive event claims:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

Global Peak is report-only. Future realised rainfall/state/flooding/Internal trajectory, online SWMM
candidate search, PFV/Peak/action penalties, Auto-RBC/EFD warm starts and baseline imitation are
forbidden online.

## 2. New frozen action-space contract: 82 control freedoms inside 109 model channels

Do not convert Step1 or base Step2 into an 82-channel neural model.

- frozen Step2 representation: **109 hydraulic action channels**;
- native supervisory-control facilities: **82**;
- passive setting channels: **27**;
- on every passive channel: `candidate == HOLD/reference`.

The 82-facility mask is derived deterministically from explicit action clauses in the source INP
`[CONTROLS]`; the current Wuhan testbed must resolve to 82 or fail closed. The current expected type
counts are 57 pumps, 16 orifices and 9 weirs.

The five added `RTC_ST_*` Storage nodes remain part of the frozen hydraulic network/state and are not
control dimensions. **Do not retrain Step1 or base Step2 because of this control-mask change.**

## 3. Cheap label-independent migration before any new SWMM truth

First build the native control mask from the exact canonical source INP used by the graph:

```powershell
python scripts/build_native_supervisory_control_current.py --help
```

Then rebuild q95 sequence support **from the existing D3 TrainFit cache only**:

```powershell
python scripts/build_direct_tfv_sequence_support_current.py --help
```

This support rebuild must report `new_swmm_simulation_required=false` and freeze masked q95 changed-K,
first-block L1, H120 L1 and H120 total variation. Do not regenerate D3 SWMM data.

Only then build the new asset manifest:

```powershell
python scripts/build_project7_practical_asset_manifest_current.py --help
```

The manifest must contain exactly graph, sensors, config, Step1, base Step2, `supervisory_control`,
matching masked `sequence_support`, and Priority8. It must freeze 109 model channels, 82 control
freedoms, Step1-retrain=false and base-Step2-retrain=false.

## 4. Current Hybrid H10 portfolio

The historical 12 x 109 L-BFGS-B full-plan optimizer is not current. At most four distinct H10
candidates are offered:

1. `STEP2_H10_PROBE_SCALE_1.00`;
2. `STEP2_H10_PROBE_SCALE_0.50`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`;
4. `SUPPORT_CONSTRAINED_GRADIENT_H10`.

The projected-gradient tensor remains 109 channels but has only **82 free dimensions**. Its gradient is
zeroed on passive channels, and every trial is projected to the native mask, physical bounds, <=0.5
slew, per-facility q95 first-move radius and masked q95 changed-K. Final targets are also contracted to
masked q95 joint-sequence support.

The final post-mask/post-support base-Step2 H10 score must be stored in each candidate artifact. Base
Step2 score never authorizes pi1 execution; the trained policy-return critic + matched one-sided UCB
decides ACTION vs HOLD.

## 5. Exact receding-policy estimand

Authoritative label:

`A^pi(x_t,u_t) = J(candidate H10 -> frozen pi) - J(HOLD H10 -> same frozen pi)`.

Negative is beneficial. Training/calibration/runtime encode exactly:

`H10 candidate target -> H350 HOLD target` versus `HOLD H360`.

Candidate/HOLD branches must share the same raw causal prefix and frozen continuation. The query
identity also includes the supervisory-mask SHA. Derived Step1 floating-point differences are
only diagnostics and do not redefine physical prefix identity.

## 6. First current parent

Historical V12/L-BFGS-B is archival. The current first-round parent is:

`PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V3_82CONTROL_109REP`.

Pi0 uses the same native mask, masked support and four-family geometry as the eventual policy, but
ranks candidates with frozen base Step2 before a policy-return critic exists.

## 7. Cheap gates

Before expensive work run:

```powershell
python -m pip install -e ".[dev,swmm]"
python -m compileall -q src scripts tests
python scripts/lint_current_surface.py
python -m pytest -q tests/test_native_supervisory_control.py tests/test_direct_tfv_hybrid_gradient_portfolio.py tests/test_direct_tfv_policy_return_hold_aware_metrics.py tests/test_practical_rtc_assets.py tests/test_practical_rtc_v14_contract.py tests/test_direct_tfv_runtime_adapter.py tests/test_current_step2_routing.py
python -m pytest -q
```

Run `--help` for every script actually used. Never invent CLI flags. `capture_direct_tfv_policy_return_context_current.py --out` is an NPZ path; its companion JSON is written automatically.

## 8. Path discovery rules

Search existing evidence first under:

```text
E:\RTC_sewer\Project7\study_v069
E:\RTC_sewer\Project7\repo
```

Resolve graph, sensors, config, Step1, Step2, D3 cache manifest and Priority8 by scientific
contract/SHA/semantic lineage, not newest-looking filenames. The source INP used to derive the 82 mask
must be the exact canonical testbed lineage for the frozen graph; record its SHA. If any frozen asset
or mask/support SHA drifts, stop. Never silently replace it with another historical artifact.

## 9. Old T30 truth is historical after the control-mask change

The previous T30 elapsed-3600 query was produced under 109 free online channels. It taught us that
base-Step2/gradient value geometry can be wrong even when gradient descent numerically succeeds, and
that one all-harmful state should be treated as a valid HOLD example rather than universal model
failure.

However, because the deployed policy is now 82-control/109-representation, **do not reuse that old T30
candidate truth as current policy-return training/calibration/mechanism evidence**. It may be reported
only as historical Development diagnostics.

## 10. Re-run a small seen-event mechanism panel before bulk learning

Do not jump to 48/12/24. Under the new mask, run current hybrid-pi0 trajectories for already-inspected
T8, T30 and T80 Development events. Freeze two deterministic causal-ready query points per event
before reading new candidate truth:

1. first causal-ready non-HOLD decision;
2. later distinct non-HOLD decision with the most-negative recorded selected/base-Step2 score among
   remaining decisions.

If a selected point yields <2 distinct post-support candidates, move deterministically to the next
ranked point without reading SWMM candidate outcomes.

Write a `SEEN_MECHANISM_QUERY_PLAN_82CONTROL.json` containing event/INP/parent SHAs, query indices,
selection reasons, asset-manifest SHA and `selection_uses_candidate_truth=false`. Freeze its SHA before
running new paired truth.

For each query:

`masked hybrid pi0 -> causal context -> masked hybrid portfolio -> 1 shared HOLD + N sequential candidates`.

Use `run_direct_tfv_policy_return_query_current.py`, not the old pair runner. Require same raw prefix,
same continuation, mask SHA identity, passive-channel unchanged, zero future-rain leakage and zero
engineering/readback/support failures.

Aggregate with:

```powershell
python scripts/audit_direct_tfv_policy_return_mechanism_panel_current.py --help
```

Interpretation:

- fewer than six valid query sets: insufficient mechanism evidence;
- >=6 and at least one query has `true A^pi < 0`: useful generated-action coverage exists; fresh
  role-disjoint learning is justified;
- >=6 and no query has any beneficial candidate: stop before bulk and return a representation/proposal
  audit. Do **not** increase K/q95/gradient steps or add a fifth candidate family.

All-harmful means HOLD is oracle within the generated portfolio at that exact state, not that all
engineering-feasible actions are harmful.

## 11. What to report from the six-query panel

For every candidate report:

- event/query/elapsed;
- source;
- changed K;
- post-mask/post-support base-Step2 H10 score;
- shared-HOLD TFV and candidate TFV;
- exact `candidate - HOLD` policy return;
- first-move and joint-support ratios/contraction;
- gradient attempted/accepted steps when relevant;
- whether passive setting channels remained identical to HOLD.

Aggregate base-Step2 sign accuracy, false-beneficial, false-reject, within-query rank/top1; also report
per-family beneficial fraction and gradient beneficial/oracle-best fraction.

## 12. HOLD-aware critic evaluation

Runtime action set is `{HOLD=0} + generated candidates`.

- predicted ACTION only when minimum predicted return <0;
- oracle ACTION only when minimum true return <0;
- false-beneficial = predicted ACTION whose selected true return >=0;
- false-reject = predicted HOLD while a true beneficial candidate exists;
- regret is measured against `min(0,best true candidate)`.

Report hold-aware decision accuracy, predicted/oracle HOLD fractions, false-beneficial, false-reject,
same-query rank, candidate top1, selected regret and event-balanced MAE/sign metrics.

## 13. Role-disjoint learning only after the mechanism gate

Only if beneficial generated actions exist under the new 82-control policy, freeze fresh disjoint
rainfall groups for at least:

- train: 48;
- model-selection validation: 12;
- conformal calibration: 24;
- separate new Development closed-loop probes.

Previously inspected T5/T8/T10/T20/T30/T80 and all Validation/Final/Formal/Policy-Lock data are excluded
from new independent roles. Selection must be deterministic, forcing-only and label-blind. Initially
use one deterministic causal query per group to minimize SWMM cost.

Use the shared-HOLD query runner. All-harmful query sets stay in their assigned role because they are
necessary HOLD negatives.

## 14. Critic / calibration lineage

Current dataset, checkpoint, calibration records, admission artifact and runtime must all share:

- candidate portfolio contract;
- continuation-policy lineage;
- `supervisory_control_dimension=82`;
- `model_action_channel_count=109`;
- identical `supervisory_mask_sha256`;
- passive-setting unchanged evidence.

Default critic training scope remains `control-heads`; do not unfreeze the full representation first.
Matched calibration must contain Step2-probe, type-aware hydraulic and gradient family examples. Do not
reduce conformal coverage/margin to force more ACTION.

## 15. New Development pi1

Run only after critic + matched calibration are frozen. Require:

- 82 supervisory facilities screened, 109 model channels retained;
- projected-gradient free dimension 82, tensor channels 109, H10-only;
- no online L-BFGS-B;
- no historical V12 admission;
- reasonable ACTION/HOLD distribution;
- passive-setting changes = 0;
- q95 support/readback/engineering/score-execute violations = 0;
- every guarded callback <<600 s.

Compare unchanged No-control, Internal RTC, Auto-RBC and EFD on identical SWMM events. Positive event
claims additionally require Priority8 PFV non-inferiority; Global Peak remains report-only.

## 16. Resource rules

Workstation target: RTX 4060 Laptop 8 GB / 16 GB RAM. Start with one heavy process. Candidate/HOLD
branches are sequential and controller objects are released between branches. Use one shared HOLD per
query. Do not run multiple PySWMM Simulations in one Python process. Only consider a second external
worker after measured RAM/VRAM preflight proves safe. Set BLAS/OpenMP threads to 1 under process
parallelism.

## 17. Do not broaden the research question

Do not add pump-energy objectives, PID setpoints, new high/low water-level penalties, an online PFV
surrogate, a fifth heuristic candidate, or a new Step1/Step2 retraining campaign unless later evidence
specifically identifies such a need. The current correction is only the engineering definition of
online controllability.

## 18. Promotion boundary

Do not enter untouched Validation, Final, Formal or Policy Lock during this Development work. Do not
merge PR #106 based on CI or seen-event mechanism diagnostics alone.

`READY_FOR_POLICY_LOCK=false`.
