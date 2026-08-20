# Project7 — sparse-sensing Practical real-time control for urban drainage

Project7 is an idealized EPA-SWMM methodology testbed for a large drainage network. It is not a
field-calibrated digital twin. **Authoritative hydraulic truth always comes from SWMM.**

The current research question is:

> Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable
> learning controller reduce system-wide total flooding volume (TFV), while avoiding material
> deterioration of flooding at frozen Priority8 nodes?

## Current method in one line

```text
causal sparse history
 -> frozen Step1 full-state reconstruction
 -> frozen 109-channel Step2 representation
 -> 82-control native supervisory mask
 -> <=3 supported H10 candidates [Step2 0.5, Step2 1.0, type-aware hydraulic]
 -> exact-receding-policy-return critic + one-sided admission
 -> execute H10 or HOLD
 -> SWMM readback -> re-observe every 10 min
```

The **sole online objective is system-wide cumulative TFV**. Priority8 PFV is an authoritative
secondary non-inferiority safety check for positive event claims:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

Global Peak is report-only. Future realised rainfall/state/flooding, online SWMM candidate search,
PFV/Peak/action penalties and baseline imitation are forbidden online.

## 82 online controls inside a frozen 109-channel representation

The pretrained Step2 retains all 109 hydraulic setting/status channels. A deterministic artifact built
from explicit source-INP `[CONTROLS]` action clauses identifies 82 facilities that may change online:
57 pumps, 16 orifices and 9 weirs. The remaining 27 channels remain hydraulically represented but must
equal HOLD/reference in every candidate.

This does **not** trigger Step1 or base-Step2 retraining. The five added `RTC_ST_*` Storage nodes remain
unchanged hydraulic state/capacity information and are not supervisory action dimensions.

Masked q95 changed-facility/L1/total-variation support is recomputed label-independently from existing
D3 TrainFit actions, with no new SWMM simulation.

## What the completed 82-control mechanism panel established

Six query points were frozen label-blind before truth: two each from already-inspected T8, T30 and T80.
All six exact same-prefix queries contained at least one truly beneficial candidate. Therefore current
candidate coverage is **not** the principal failure mode.

The frozen base Step2 directional score, however, is not a reliable deployed return oracle on this
panel: sign accuracy 0.45, false-beneficial fraction 0.30, false-reject fraction 0.25, within-query
pairwise rank accuracy 0.4167 and top-1 accuracy 0.50.

Candidate-family mechanism evidence was especially informative:

- Step2 scale 0.50: beneficial 3/6;
- Step2 scale 1.00: beneficial 2/3 when distinct;
- type-aware hydraulic pressure: beneficial 5/6 and repeatedly oracle-best;
- projected gradient: beneficial 3/5 but oracle-best 0/5.

This supports a simpler conclusion: keep base Step2 as a pretrained representation/direction generator,
but learn the **actual receding first-action value/ranking** from exact SWMM pairs.

## Current three-family portfolio

The paper-facing online portfolio is now deliberately restricted to at most three distinct H10
candidates:

1. `STEP2_H10_PROBE_SCALE_0.50`;
2. `STEP2_H10_PROBE_SCALE_1.00`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`.

Projected gradient remains implemented only for an explicit Development ablation. It is not part of
current pi0, pi1, training/calibration family matching or online execution. Historical 12 x 109
L-BFGS-B remains archival.

Every candidate is bounded by the native 82-control mask, physical bounds, <=0.5 target slew,
per-facility q95 first-move radius and masked q95 joint-sequence support. Actuator release intent uses
actuator-type-specific SWMM SETTING semantics.

## Exact receding-policy target

The deployed learning target is:

```text
A^pi(x_t,u_t)
 = J(candidate H10 -> frozen continuation pi)
 - J(HOLD H10 -> the same frozen continuation pi)
```

Negative is beneficial. Training/calibration/runtime encode exactly:

`H10 candidate target -> H350 HOLD target` versus `HOLD H360`.

Candidate/HOLD branches share the same raw causal prefix, forcing, supervisory-mask SHA and frozen
continuation. The real controller re-observes and replans after each executed H10.

Because removing projected gradient changes the parent continuation policy, the completed four-family
seen panel remains **mechanism evidence only**. It is not silently relabelled as current three-family
training/calibration truth.

## Next Development gate: minimum confirmation, then forcing inventory

Before bulk labels, rerun the three-family parent on seen T8/T30/T80 and freeze two query points per
event before reading truth. Evaluate only the first query from each event initially. If all three have
a beneficial current candidate under the new continuation, the mechanism recheck is sufficient; the
already-frozen reserve queries remain unused. If the first wave is ambiguous, evaluate the reserves
before changing architecture.

Then perform a **zero-SWMM inventory** of existing forcing/event metadata against frozen exclusions.
Only create the missing forcing definitions needed to reach role-disjoint targets:

- policy-return train: 48 rainfall groups;
- model-selection validation: 12;
- conformal calibration: 24;
- new Development closed-loop probes: 3.

The current uploaded role audit confirmed no role-pure untouched pool yet; bulk generation remains
closed until forcing identities are frozen. Do not use existing Step2 TrainFit groups as independent
policy-return evidence.

## Critic and promotion logic

The runtime decision set is `{HOLD=0 + generated candidates}`. Critic selection prioritizes HOLD-aware
false-beneficial and false-reject behavior, same-query ranking and selected regret before scalar MAE.
Default fine-tuning remains control/action heads; do not unfreeze the full pretrained representation
unless later independent evidence requires it.

Operational comparison remains No-control, Internal RTC, Auto-RBC and EFD. All-max/min SETTING are
diagnostic extremes. Positive event claims additionally require Priority8 PFV safety.

See `CODEX_START_HERE.md`, `PROJECT7_PRACTICAL_RTC_V14.md` and `PROJECT7_HANDOFF_CURRENT.md`.

`READY_FOR_POLICY_LOCK=false`. Validation, Final, Formal and Policy Lock remain inaccessible during
current policy development.
