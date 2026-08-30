# Project7 policy-consistency and matched-baseline remediation R1

## Status

Development-only. This branch starts from V28R1 (`83ed02fb...`) because V28R1's supported-manifold
residual selection can fall back exactly to Q27. V29 remains a preserved negative result and is not
rescued or overwritten here.

No historical V26/V27/V28/V29 evidence is modified. No training or SWMM is performed by the
policy-consistency view builder or parity auditor.

## Zero-SWMM audit findings that motivate this branch

The completed read-only audit found:

- no conflict among *explicit* continuation hashes inside one causal context;
- roughly 55% of historical exact-return rows lack `continuation_policy_sha256`;
- V27/V28/V29 runtime decision logs do not carry a continuation/context/prefix hash sufficient for
  row-level runtime-to-training identity proof;
- native Auto-RBC observes 117 rule nodes versus 89 Proposed sensor nodes;
- native Auto-RBC changes about 87.7 actuators per transition on average, while Proposed is capped at
  the q95 changed-facility ceiling of 20;
- native Auto-RBC changes at least one of Proposed's passive 27 channels on every audited transition;
- both native Auto-RBC and Proposed share the 0.5 target-slew ceiling, so the important remaining
  mismatch is information basis and control authority, not target slew;
- V29's regime correction did not transfer closed-loop and must not motivate another regime-residual
  iteration.

These findings mean that `Proposed vs native Auto-RBC` mixes algorithm quality with information and
action-authority differences. It remains useful as an external operational reference, but not as the
sole fair learned-vs-rule superiority comparison.

## R1 scientific correction

### 1. Policy-consistent supervision view

`src/rtc/project7_policy_consistency.py` and
`scripts/build_project7_policy_consistent_exact_return_view.py` create a derived view without editing
source evidence.

A missing continuation hash is recovered only if the exact same
`causal_context_fingerprint_sha256` has exactly one explicit continuation hash. Cross-context
imputation is forbidden. Ambiguous/unresolved rows are never eligible for the policy-consistent
training view.

The pairwise decision identity is now defined conceptually as:

`(leakage_group, causal_context, resolved_continuation_policy_sha256)`.

This prevents a future pairwise learner from silently treating `Q^pi_A` and `Q^pi_B` as the same
estimand.

### 2. Information/action-authority matched active baselines

`src/rtc/project7_matched_baselines.py` reuses the Proposed outer runtime and replaces only the
decision rule. Matched Auto-RBC and matched EFD therefore share:

- the exact Proposed sparse sensor file;
- the exact frozen Step1 reconstruction;
- the same causal rainfall forecast;
- the 109-channel action representation;
- the native 82-channel writable supervisory mask;
- the same passive 27 channels;
- q95 changed-facility ceiling;
- q95 joint-sequence support;
- 0.5 target slew;
- target-latch semantics;
- authoritative SWMM execution and target-write/readback audit.

The rules see reconstructed hydraulic state only. They do not receive extra node depths from SWMM.

`src/rtc/project7_matched_internal.py` provides the narrow matched Internal RTC interpreter. It
parses the frozen source `[CONTROLS]` grammar and accepts only `NODE ... HEAD` conditions plus
catalogued `PUMP`, `ORIFICE`, `WEIR`, or `OUTLET` actions. Conditions are evaluated against the
Step1 reconstructed `head_m` channel; unsupported conditions fail closed. Parsed rule targets are
then sent through the same 82/27 mask, first-radius, q95 K ceiling, 0.5 slew, q95 sequence support,
and target-latch path. The source INP is a rule-definition artifact, not a source of online SWMM
state truth.

### 3. Comparator roles

- `No-control`: passive primary reference; no information/action authority is needed.
- `matched_auto_rbc`: fair active rule comparator.
- `matched_efd`: fair active EFD comparator.
- `matched_internal_rtc`: fair active comparator only when the reconstructed-state parser and static
  contract audit pass.
- native `auto_rbc` / native `efd`: external operational references, retained for context.
- native `internal_rtc`: external operational reference; its matched counterpart is a separate
  reconstructed-state implementation and must not be conflated with native execution.

## Decision rule for the next policy change

Do **not** create another residual/regime model before the matched benchmark is available.

1. If V28R1/Q27 is competitive with or better than both matched active baselines, the previous
   Auto-RBC gap was materially caused by comparator information/action authority. Keep the Q27-style
   decision-aware path and proceed to fresh validation/lock planning rather than increasing model
   complexity.
2. If matched Auto-RBC/EFD still outperform Proposed, build the policy-consistent target-continuation
   view and quantify usable Train/Validation/Test support. The next candidate policy should be a
   **same-context pairwise advantage learner on q95-supported actions**, using only one resolved
   continuation policy. Do not fit another cross-state regime residual.
3. If the matched rule's projected action is frequently geometrically distant from the current
   candidate portfolio and existing exact truth shows the projected rule action is beneficial, the
   bottleneck is candidate coverage. Add that projected rule structure as a candidate proposal but
   give it no execution privilege; it must be ranked under the same policy-consistent value model.
4. Step1/Step2 retraining, q95 removal, return-period/event-ID features, benchmark-specific thresholds,
   and Policy Lock/Final remain forbidden in this branch.

## Required gates before any matched SWMM run

- compileall;
- current-surface lint;
- focused tests for policy consistency and matched baselines;
- full pytest;
- CLI help for the new scripts;
- build the zero-SWMM policy-consistent view and review resolved/unresolved continuation counts;
- no new model training.

Only after those gates pass may the frozen Benchmark5 events run matched baselines. Existing matched
Internal results are reusable evidence; they are not regenerated by the policy-consistent refit. Any
new matched-baseline run must use a fresh output directory and must never overwrite the native
baseline cache.
