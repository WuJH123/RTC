# Project7 Practical RTC — authoritative local execution guide

Web GPT owns the scientific/code contract. Local Codex discovers existing files, executes the frozen
current branch, monitors resources and returns evidence/errors. Historical V* artifacts are evidence
or ablation only, not automatic execution selectors.

## 0. Git and local-edit safety

Never destroy the user's local documentation edits. Do not use `reset --hard`, `clean`, `restore .`,
`checkout .`, or blind stash-pop. Synchronize the exact PR branch/head supplied by Web GPT. Do not
switch to `main` unless explicitly instructed after a later scientific merge.

## 1. Frozen paper question

Every 600 s:

`causal sparse history -> frozen Step1 -> 82-control/109-representation -> <=3 H10 candidates -> receding-policy-return critic/admission -> execute H10 or HOLD -> SWMM readback -> observe again`.

Sole online objective: **system-wide cumulative TFV**.

Secondary authoritative Priority8 PFV safety for positive event claims:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

Global Peak is report-only. Future realised rainfall/state/flooding/Internal trajectory, online SWMM
candidate search, PFV/Peak/action penalties, Auto-RBC/EFD warm starts and baseline imitation are
forbidden online.

## 2. Frozen action-space and assets

Do not convert Step1 or base Step2 into an 82-channel neural model.

- frozen Step2 representation: 109 hydraulic/action channels;
- native supervisory controls: 82 facilities = 57 pumps + 16 orifices + 9 weirs;
- passive setting channels: 27;
- passive rule: `candidate == HOLD/reference`;
- five `RTC_ST_*` Storage nodes remain frozen hydraulic state/capacity features.

The existing native mask and masked q95 support were built label-independently from the canonical INP
and existing D3 TrainFit actions. **Reuse their SHAs; do not rebuild or rerun D3 SWMM unless lineage
actually drifts.** Step1/base Step2 are not retrained for this mask.

## 3. Completed mechanism evidence and what changed

The completed four-family 82-control seen panel used six label-blind queries across T8/T30/T80 and
passed same-prefix/continuation/causality/engineering gates. Every query contained a beneficial
candidate, so candidate coverage is not dead.

Observed base Step2 diagnostics were weak as a deployed return oracle: sign 0.45, false-beneficial
0.30, false-reject 0.25, pairwise rank 0.4167, top1 0.50. Type-aware hydraulic pressure was beneficial
5/6; projected gradient was beneficial 3/5 but oracle-best 0/5.

Therefore current online portfolio is reduced to exactly these possible families:

1. `STEP2_H10_PROBE_SCALE_0.50`;
2. `STEP2_H10_PROBE_SCALE_1.00`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`.

Projected gradient is now **Development ablation only**. Current pi0/pi1, dataset family matching,
calibration and online execution must not include `SUPPORT_CONSTRAINED_GRADIENT_H10`.

Historical gradient code remains for reproducibility; current CLI gradient knobs are compatibility-only
and must not alter the current policy.

## 4. Exact estimand and lineage consequence

Authoritative label:

`A^pi(x_t,u_t) = J(candidate H10 -> frozen pi) - J(HOLD H10 -> same frozen pi)`.

Negative is beneficial. The critic token is exactly `H10 candidate -> H350 HOLD` versus `HOLD H360`.

Removing gradient changes the pi0 continuation policy. Consequently, the completed four-family seen
exact returns are mechanism evidence only and may **not** be reused as current three-family
train/validation/calibration truth.

## 5. Cheap gates

Before any SWMM work:

```powershell
python -m pip install -e ".[dev,swmm]"
python -m compileall -q src scripts tests
python scripts/lint_current_surface.py
python -m pytest -q tests/test_native_supervisory_control.py tests/test_direct_tfv_hybrid_gradient_portfolio.py tests/test_direct_tfv_policy_return_portfolio.py tests/test_direct_tfv_policy_return_hold_aware_metrics.py tests/test_practical_rtc_assets.py tests/test_practical_rtc_v14_contract.py tests/test_direct_tfv_runtime_adapter.py tests/test_current_step2_routing.py
python -m pytest -q
```

Run `--help` for every script actually used. Never invent flags.

## 6. First next action: zero-SWMM forcing inventory

Before generating any new forcing or policy-return truth, inventory existing event/forcing metadata in
`study_v069` and current input manifests. This step must not run SWMM and must not inspect outcome
labels.

Exclude:

- all previously inspected Development families (including T5/T8/T10/T20/T30/T80, P15/P35/P75,
  V10 and V12 seen probes);
- existing Step2/D3 TrainFit rainfall groups from independent policy-return roles;
- Validation, Final, Formal and PolicyLock resources.

Return a deterministic inventory of eligible untouched forcing groups with forcing metadata only.
Do not guess that the pool is zero just because the previous role plan had not confirmed it.

Target eventual role counts remain 48 train / 12 model-selection validation / 24 calibration / 3 new
Development probes. Create only the **deficit** after inventory, using forcing-only deterministic
selection. Do not start bulk truth yet.

## 7. Minimal three-family continuation-specific mechanism recheck

Because pi0 changed, rerun three-family parent trajectories on already-seen T8/T30/T80. Before reading
new candidate truth, freeze **two** deterministic causal-ready query points per event:

- first query = first causal-ready non-HOLD decision;
- reserve query = later distinct non-HOLD decision with the most-negative recorded base-Step2 selected
  score among remaining decisions.

Write and hash the whole six-point plan before any new truth.

Then execute only the **first-wave three queries** initially: one per event. For each query use current
three-family context/portfolio and `1 shared HOLD + N sequential candidate branches`.

If all three first-wave queries contain at least one exact beneficial candidate and all technical gates
pass, STOP the mechanism recheck; do not spend SWMM on reserve queries. If any first-wave event has no
beneficial current candidate or the evidence is technically ambiguous, execute the already-frozen
reserve queries before considering model changes.

These seen queries are Development diagnostics only, never independent learning evidence.

## 8. Current pi0 and portfolio checks

Current parent contract:

`PROJECT7_PRACTICAL_BASE_H10_THREE_FAMILY_PARENT_PI0_V4_82CONTROL_109REP`.

Current portfolio contract contains `V6_H10_THREE_FAMILY_82CONTROL_109REP`.

For every parent/query require:

- supervisory controls = 82, model channels = 109;
- candidate count <=3;
- sources only Step2 0.50, Step2 1.00, hydraulic pressure;
- projected-gradient online = false;
- no L-BFGS-B;
- passive changes = 0;
- q95 support, engineering, routing and readback violations = 0;
- future realised rainfall online = false;
- same raw prefix and same frozen continuation for paired branches.

Base-Step2 score is diagnostic only. Do **not** drop the hydraulic candidate because Step2 predicts a
positive value; the completed mechanism panel specifically showed this false-reject mode.

## 9. Role-disjoint learning after both gates

Only after (a) forcing inventory/role identities are frozen and (b) the three-family mechanism recheck
passes, generate exact policy-return labels.

Initially use one deterministic causal query per rainfall group. One shared HOLD plus all distinct
three-family candidates minimizes authoritative branch count. All-harmful query sets remain valid HOLD
negative examples.

Dataset/checkpoint/calibration/runtime must share the current portfolio contract, continuation-policy
SHA, 82/109 control contract and supervisory-mask SHA.

Default critic training scope remains control/action/interaction heads. Do not unfreeze the whole
representation first. Model selection uses HOLD-aware false-beneficial, false-reject, same-query rank,
selected regret and then event-balanced MAE/sign.

Matched calibration must contain the **actual three-family** distribution and must reject projected-
gradient rows. Do not reduce conformal coverage or margins to force ACTION.

## 10. New Development pi1 and comparators

After critic + matched calibration are frozen, run new Development closed loops. Compare unchanged
No-control, Internal RTC, Auto-RBC and EFD on identical SWMM forcing. All-max/min are diagnostics only.
Positive claims additionally require the frozen Priority8 PFV envelope. Global Peak is report-only.

Near-all-HOLD behavior remains a failure mode to diagnose, not a successful conservative policy.

## 11. Resource rules

Target machine: RTX 4060 Laptop 8 GB / 16 GB RAM. Use one heavy process first. Candidate/HOLD branches
are sequential. One Python process must not run multiple PySWMM simulations concurrently. Set
OMP/MKL/OpenBLAS/NUMEXPR threads to 1. Only consider a second external worker after measured resource
preflight proves safe.

## 12. Do not broaden the question

Do not add pump-energy objectives, PID setpoints, new level penalties, online PFV surrogate, additional
online candidate families, wider q99/K support, more gradient steps, L-BFGS-B, or Step1/base-Step2
retraining without later independent evidence that specifically requires it.

Do not enter Validation, Final, Formal or Policy Lock. Do not merge PR #106 based on CI or seen
Development diagnostics alone.

`READY_FOR_POLICY_LOCK=false`.
