# Project7 Practical RTC — authoritative local execution guide

Web GPT owns the scientific/code contract. Local Codex discovers already-existing files, executes the frozen current branch, monitors resources, and returns evidence/errors. Historical V* filenames are evidence/ablation only, not automatic execution selectors.

## 0. Git and local-edit safety

Never destroy the user's local documentation edits. Do not use `reset --hard`, `clean`, `restore .`, `checkout .`, or blind stash-pop.

For the current experiment, synchronize the exact PR branch/head supplied by Web GPT. Do not switch to `main` unless explicitly instructed after a later scientific merge.

## 1. Frozen paper question

Every 600 s:

`causal sparse history -> Step1 full-state reconstruction -> frozen Step2 H10 proposals -> <=4 supported first-action candidates -> receding-policy-return critic/admission -> execute H10 or HOLD -> authoritative SWMM write/readback -> observe again`.

Frozen contract:

- observation/model step = 300 s;
- supervisory decision = 600 s;
- prediction/value context = H360;
- only first H10 target executes before replanning;
- 109 writable actuators are screened;
- physical bounds and target slew <=0.5 per update;
- HOLD = latch the previous supervisory target;
- unchanged facilities retain their previous target;
- frozen q95 TrainFit first-move and joint-sequence support;
- authoritative truth = SWMM.

Objective hierarchy:

- sole online objective: system-wide cumulative TFV;
- authoritative secondary spatial safety for positive event claims: `Priority8 PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`;
- Global Peak: report only.

Forbidden online: future realised rainfall/state/flooding/Internal trajectory, online SWMM candidate search, PFV/Peak/action penalties, Auto-RBC/EFD warm starts, or baseline imitation.

## 2. Current Hybrid H10 portfolio

The historical 12 x 109 L-BFGS-B plan optimizer is not current. At each state the portfolio may contain at most four distinct supported H10 candidates:

1. `STEP2_H10_PROBE_SCALE_1.00`;
2. `STEP2_H10_PROBE_SCALE_0.50`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`;
4. `SUPPORT_CONSTRAINED_GRADIENT_H10`.

The fourth is a 109-D H10-only projected-gradient proposer through frozen base Step2. Every trial is projected to bounds, <=0.5 slew, per-facility q95 first-move radius, and q95 changed-facility ceiling. Final targets are also contracted to q95 joint-sequence support. Base-Step2 score never authorizes deployed execution; after policy-return training, the critic + matched one-sided UCB decides ACTION vs HOLD.

The exact policy-return action token is:

`H10 candidate target -> H350 HOLD target` versus `HOLD H360`.

This is a first-action counterfactual representation. The real controller re-observes and replans every 10 min.

## 3. First policy iteration

Historical V12/L-BFGS-B is archival only. Current first-round parent is:

`PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V2`.

Pi0 uses the same proposal/support geometry but ranks candidates with frozen base Step2 before a policy-return critic exists. Exact-prefix paired SWMM creates `Q^pi0/A^pi0` labels. Pi1 uses the trained/calibrated critic. If pi1 materially changes the state/action distribution, a new role-disjoint `Q^pi1` round is required before Policy Lock.

## 4. Cheap gates

Always run before expensive work:

```powershell
python -m pip install -e ".[dev,swmm]"
python -m compileall -q src scripts tests
python scripts/lint_current_surface.py
python -m ruff check scripts/audit_direct_tfv_policy_return_mechanism_panel_current.py tests/test_direct_tfv_policy_return_hold_aware_metrics.py --select E4,E7,E9,F
python -m pytest -q tests/test_direct_tfv_policy_return_portfolio.py tests/test_direct_tfv_hybrid_gradient_portfolio.py tests/test_direct_tfv_policy_return_hold_aware_metrics.py tests/test_practical_rtc_v14_contract.py tests/test_direct_tfv_runtime_adapter.py tests/test_current_step2_routing.py
python -m pytest -q
```

Run `--help` for every script actually used. Never invent flags from memory. `capture_direct_tfv_policy_return_context_current.py --out` is an NPZ path; its companion JSON is written automatically.

## 5. Discover and freeze assets once

Search existing evidence first under:

```text
E:\RTC_sewer\Project7\study_v069
E:\RTC_sewer\Project7\repo
existing JSON/report/manifest files
```

Resolve by scientific contract/SHA/semantic lineage, not by the newest-looking filename. Current assets are only graph, sensors, runtime config, Step1, base Step2 V5, q95 sequence support, and Priority8 list. Historical V12 policy/first-move admissions do not belong in the current manifest.

Use `scripts/build_project7_practical_asset_manifest_current.py --help` and freeze one `PRACTICAL_RTC_ASSETS.json` with absolute paths and SHA-256. Reuse it downstream. If a recorded file disappears or changes SHA, stop; never silently substitute another V* artifact.

## 6. Revised mechanism gate: do not infer the whole method from one query

The first T30 query at elapsed 3600 s is already seen Development diagnostics. Its three distinct generated candidates were all harmful under exact policy-return truth. That result is important but has a narrow interpretation:

- HOLD was oracle-optimal **within the generated portfolio at that exact state**;
- the base Step2 local value/gradient surface was misaligned with exact receding-policy return at that state;
- it does **not** prove that every other state lacks a beneficial generated candidate;
- it does **not** prove that no beneficial engineering-feasible 109-D action exists outside the generated portfolio.

Do not start the 48/12/24 bulk yet, but also do not redesign Step2 from this single query.

Instead run a small, preregistered, label-blind **seen-event mechanism panel** using already-inspected Development events only. Preferred first panel:

- T8: two query points;
- T30: two query points (reuse the existing elapsed-3600 result as one of them);
- T80: two query points.

Freeze all six query points before reading any additional candidate SWMM truth. For each event choose deterministic causal-ready points from the current hybrid-pi0 trajectory; recommended rule is:

1. first causal-ready non-HOLD decision;
2. one later distinct non-HOLD decision with the most-negative recorded base-Step2 selected score among the remaining decisions.

If a selected point duplicates the first after support/dedup or lacks >=2 distinct candidates, move to the next deterministic ranked point without inspecting SWMM candidate outcomes.

These T8/T30/T80 results are mechanism debugging only. They must never count as independent train/validation/calibration/final evidence.

For each query:

`hybrid pi0 parent -> causal context -> hybrid portfolio -> 1 shared HOLD + N sequential candidate branches`.

Require same raw prefix, same continuation-policy SHA, zero future-rainfall leakage, zero engineering/readback/support failures.

Then aggregate the seen panel with:

```powershell
python scripts/audit_direct_tfv_policy_return_mechanism_panel_current.py --help
```

Pass every query records JSONL with repeated `--records-jsonl` arguments and write one audit JSON.

Interpretation:

- fewer than 6 valid query sets: insufficient mechanism evidence; do not change Step2 yet;
- >=6 queries and at least one query has a truly beneficial generated candidate: proposal coverage is not universally dead; proceed to role-disjoint policy-return data generation/training;
- >=6 queries and zero queries contain any beneficial generated candidate: stop before bulk and inspect proposal coverage plus Step2 state/value representation. Do not respond by increasing K, q95, gradient steps, or adding a fifth heuristic candidate.

An all-harmful query is still scientifically useful: it is a negative example teaching the critic that HOLD should win.

## 7. HOLD-aware critic evaluation

Runtime action set is `{HOLD with value 0} + generated candidates`. Training/model-selection metrics must respect that action set.

For each same-prefix query:

- predicted ACTION only when the minimum predicted candidate return is <0; otherwise predicted HOLD;
- oracle ACTION only when the minimum true candidate return is <0; otherwise oracle HOLD;
- false-beneficial = predicted ACTION but selected true return >=0;
- false-reject = predicted HOLD while at least one true candidate return <0;
- regret compares the realized selected action/HOLD against `min(0, best true candidate)`.

Report `predicted_hold_fraction`, `oracle_hold_optimal_fraction`, `hold_aware_decision_accuracy`, false-beneficial, false-reject, same-query rank accuracy, candidate top1, selected-action regret, and event-balanced MAE/sign metrics.

Do not interpret an all-harmful query that is correctly predicted as HOLD as a false-beneficial selection.

## 8. Role-disjoint policy-return data, only after the mechanism gate

Before reading new paired outcomes, freeze rainfall-group identities for at least:

- train: 48 independent groups;
- model-selection validation: 12 groups;
- conformal calibration: 24 groups;
- separate new Development closed-loop probes.

Old T5/T10/T20/T8/T30/T80 and other previously inspected Development outcomes cannot be counted as new independent evidence. Validation/Final/Formal/Policy Lock remain inaccessible.

Initially use one deterministic causal query point per group to control SWMM cost. Query selection must not depend on candidate truth.

For each group:

`hybrid pi0 parent -> causal context -> hybrid portfolio -> one shared HOLD + all distinct candidates`.

All-harmful query sets remain in their proper role dataset; they are necessary negative/HOLD examples.

## 9. Resource rules

Target workstation is RTX 4060 Laptop 8 GB / 16 GB RAM. Start with one heavy GPU/SWMM process. Candidate/HOLD branches are sequential and controllers are released between branches. Use shared-HOLD query generation. Do not run multiple PySWMM simulations inside one Python process. Only consider a second external worker after measured RAM/VRAM preflight proves safe. Set BLAS/OpenMP threads to 1 under process parallelism.

## 10. Train, calibrate, and Development-test pi1

Use current `--help` for:

```text
scripts/compile_direct_tfv_policy_return_dataset_current.py
scripts/train_direct_tfv_policy_return_current.py
scripts/score_direct_tfv_policy_return_calibration_current.py
scripts/calibrate_direct_tfv_policy_return_portfolio_admission_current.py
scripts/run_policy_direct_tfv_policy_return_development.py
```

Dataset ranking unit = same authoritative-prefix query set. Scientific split unit = rainfall group. Default training scope remains `control-heads`; do not unfreeze the full network initially.

Matched calibration must contain the actual online families, including Step2 probe, type-aware hydraulic pressure, and projected-gradient examples. Do not reduce coverage/margin to increase ACTION count.

In new Development pi1 closed loops require: portfolio mode, projected gradient enabled, no online L-BFGS-B, 109 screened, reasonable nonzero ACTION frequency, score==execute, q95 support, zero future-information/online-SWMM leakage, zero write/readback/engineering violations, and runtime far below 600 s.

Operational comparators remain No-control, Internal RTC, Auto-RBC, and EFD on the same INP/forcing/SWMM. All-max/all-min are diagnostic only. Positive event claims additionally require the frozen Priority8 PFV envelope. Global Peak is report-only.

## 11. If the six-query panel genuinely fails

Only if the preregistered seen mechanism panel has zero beneficial generated actions across all valid queries should Codex stop and return an upstream diagnosis. Do not edit the model itself. Return evidence needed for Web GPT to inspect:

- state/rain/action-token tensors and their normalization ranges;
- per-query base-Step2 prediction vs exact policy-return sign/rank;
- candidate K/support/contraction;
- candidate-source distributions;
- upstream/downstream depth, flooding, storage/headroom-related available state features around changed actuators;
- interaction-main decomposition if already exposed by the frozen model;
- whether beneficial exact historical first actions exist nearby but current portfolio fails to recover them.

Web GPT will decide whether the next Development ablation is representation improvement (e.g. explicit storage/headroom/downstream-capacity context or broader critic adaptation) rather than blind candidate expansion.

## 12. Promotion boundary

Do not enter untouched Validation, Final, Formal, or Policy Lock during this work. Do not merge PR #106 based on CI or seen Development diagnostics alone.

`READY_FOR_POLICY_LOCK=false` until the Development scientific gates are met.
