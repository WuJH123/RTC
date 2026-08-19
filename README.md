# Project7 — sparse-sensing Practical real-time control for urban drainage

Project7 is an idealized EPA-SWMM methodology testbed for a large drainage network. It is not a
field-calibrated digital twin. **Authoritative hydraulic truth always comes from SWMM.**

The current research question is:

> Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable
> learning controller reduce system-wide total flooding volume (TFV), while avoiding material
> deterioration of flooding at frozen Priority8 nodes?

## Current paper logic

Every 10 minutes:

```text
causal sparse hydraulic/rainfall history
  -> frozen Step1 full-network state reconstruction
  -> frozen 109-channel base Step2 representation
  -> native 82-facility supervisory mask
  -> <=4 masked/support-bounded H10 candidates
  -> receding-policy-return critic + one-sided admission
  -> execute first H10 target or HOLD
  -> authoritative SWMM target write/readback
  -> re-observe and replan
```

Objective hierarchy:

- **online primary:** system-wide cumulative TFV only;
- **secondary authoritative safety:** Priority8 PFV non-inferiority versus No-control,
  `PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`;
- **report-only:** Global Peak.

No future realised rainfall/state/flooding, no online SWMM candidate search, no PFV/Peak penalty and
no baseline imitation are permitted online.

## 82 online control freedoms, 109 hydraulic model channels

Project7 now separates **hydraulic representation** from **supervisory controllability**.

The pretrained Step2 retains all 109 hydraulic setting/status channels. A deterministic artifact built
from explicit action clauses in the source INP `[CONTROLS]` identifies the current 82 facilities that
may actually change online. The other 27 channels remain part of the hydraulic/model representation,
but every candidate must keep them identical to HOLD/reference.

The current expected controlled set is 57 pumps, 16 orifices and 9 weirs. A different count fails
closed rather than silently changing the research action space.

This change **does not trigger Step1 or base-Step2 retraining**. The five added `RTC_ST_*` Storage nodes
remain unchanged as hydraulic state/capacity information; they are not supervisory action dimensions.

## Cheap support migration instead of new SWMM data

Because the online control subspace changed, q95 changed-facility/L1/total-variation support is
recomputed from the existing D3 TrainFit actions after masking passive channels. This is
label-independent and requires **no new SWMM simulation**.

Build:

```text
scripts/build_native_supervisory_control_current.py
scripts/build_direct_tfv_sequence_support_current.py
scripts/build_project7_practical_asset_manifest_current.py
```

The resulting asset manifest freezes graph, sensors, config, Step1, 109-channel base Step2, native
supervisory mask, matching masked q95 support and Priority8 with absolute paths and SHA-256.

## Hybrid first-action search

Historical 12 x 109 L-BFGS-B full-plan optimization is archival. At most four current H10 candidates
are generated:

1. `STEP2_H10_PROBE_SCALE_1.00`;
2. `STEP2_H10_PROBE_SCALE_0.50`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`;
4. `SUPPORT_CONSTRAINED_GRADIENT_H10`.

The gradient tensor remains 109 channels but only 82 supervisory dimensions are free. Gradient values
on passive channels are zeroed, and every trial is projected to the native mask, physical bounds,
<=0.5 target slew, first-move q95 radius and masked q95 changed-K. Final candidates are contracted to
masked q95 H10 sequence support.

The gradient is only a proposer. Base-Step2 score never authorizes pi1 execution.

## Exact receding-policy first-action value

The deployed learning target is:

```text
A^pi(x_t,u_t)
 = J(candidate H10 -> frozen continuation pi)
 - J(HOLD H10 -> the same frozen continuation pi)
```

Candidate/HOLD branches share the same raw causal prefix, forcing and frozen continuation. Training,
calibration and runtime all encode:

```text
H10 candidate target -> H350 HOLD target
```

The query identity includes the supervisory-mask SHA. The real controller re-observes and replans after
every executed H10.

## First policy iteration and current mechanism gate

Historical V12/L-BFGS-B is archival. First-round labels use the masked hybrid parent:

`PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V3_82CONTROL_109REP`.

The earlier T30 exact query under 109 free channels remains historical Development diagnostics: it
showed that a numerically successful gradient can optimize a locally misaligned base-Step2 value
surface, and that all-harmful candidates at one state make HOLD the local portfolio oracle rather than
proving universal failure.

Because the online policy changed, that old exact truth is **not current 82-control training or
calibration evidence**. Before expensive 48/12/24 labels, rerun a small label-blind seen mechanism
panel under the new mask: two frozen causal query points each from already-inspected T8, T30 and T80.

If at least one of six valid query sets contains an exact beneficial generated action, proceed to fresh
role-disjoint policy-return learning. If none do, stop before bulk and inspect proposal/Step2
representation instead of widening K/q95/gradient steps or adding more heuristic candidates.

## HOLD-aware critic

The true action set is `{HOLD=0 + generated candidates}`. Critic model selection therefore reports
HOLD-aware false-beneficial, false-reject, selected regret and decision accuracy in addition to
same-query rank/top1 and event-balanced error/sign metrics. Correctly HOLDing an all-harmful query is
not an error.

## Baselines and safety

Operational comparison remains No-control, Internal RTC, Auto-RBC and EFD. All-max-setting and
all-min-setting are diagnostic extremes. Hydraulic release intent retains actuator-type-specific SWMM
SETTING semantics.

The Proposed policy is not required to beat every comparator on every event. Event-balanced TFV and
Priority8 PFV safety are assessed on untouched authoritative SWMM events after Development is frozen.

## Current execution path

```text
build native control mask
  -> rebuild masked q95 support from existing actions
  -> freeze asset manifest
  -> masked hybrid pi0 trajectory
  -> exact causal context
  -> masked four-family portfolio
  -> 1 shared HOLD + N sequential candidate SWMM branches
  -> seen mechanism audit
  -> only if justified: fresh role-disjoint dataset / critic / calibration
  -> new Development pi1 + unchanged baselines + TFV/PFV reporting
```

The old pair runner and historical V* admissions are not the current bulk label path. See
`CODEX_START_HERE.md` and `PROJECT7_PRACTICAL_RTC_V14.md` for the full frozen contracts.

`READY_FOR_POLICY_LOCK=false` until Development scientific gates are met. Validation, Final, Formal
and Policy Lock remain inaccessible during policy development.
