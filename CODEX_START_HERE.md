# Project7 current execution guide

GitHub `main` is the code source of truth. Project7 is an **idealized EPA-SWMM methodology testbed**,
not a field digital twin.

## Frozen research problem

```text
sparse causal sensors
 -> Step1 reconstruct CURRENT full-network hydraulic state
 -> Step2 learn 109-facility action effects on future system-wide cumulative TFV
 -> Step3 screen all 109 facilities relative to HOLD
 -> optimise a q95-supported H120 joint action sequence for H360 TFV
 -> admit the optimizer plan only when an optimizer-aware one-sided TFV residual upper bound < 0
 -> execute first 10-minute target only
 -> re-observe and repeat
```

Primary objective: **system-wide cumulative TFV minimization**. SWMM is authoritative offline truth.
No future realised rainfall, future SWMM state or future flooding truth is available online.

Frozen clock/action contract:
- state/model step = 300 s;
- control update = 600 s;
- H360 prediction = 72 five-minute steps;
- H120 free control = 12 ten-minute blocks;
- writable facilities = 109;
- all 109 are screened every decision;
- setting movement <= 0.5 per 10-minute update;
- execute first 10-minute target only, then re-observe.

## Current evidence and scientific bottleneck

Step2 V5 has 109/109 exact single-facility coverage and useful D3 HOLD-reference cached-candidate
metrics. Step3 V4 also executes correctly: zero support/engineering/readback violations and runtime
well below 600 s. Those are no longer the primary bottleneck.

Authoritative V5/V4 Development results exposed a more specific failure:

```text
T5_D120   TFV reduction vs No-control  +11.547%
T10_D180                              +0.615%
T20_D300                              -1.396%
```

Event-balanced mean reduction is only about 3.59%, median about 0.61%; Proposed loses to Auto-RBC on
all three events. More importantly, exact same-prefix T5 H360 replay has zero prefix/routing mismatch
but only 3/6 correct signs. Three optimizer-selected negative predictions are false-beneficial, and a
fourth predicted -124,660 m3 is essentially zero in SWMM (-96 m3).

Current classification:

```text
DEVELOPMENT_NO_CONTROL_BENEFIT_INCONSISTENT
```

The mechanism under test is **continuous-optimizer selection-induced optimism (optimizer's curse)**:
the cached D3 candidate set can look safe while L-BFGS-B selects much more optimistic extrema from a
larger continuous action space. Simply increasing active K from q90 to q95 did not solve the problem;
q95 was 32/32 ceiling-binding and T10/T20 still regressed. q99 remains diagnostic only.

## Step2 — keep V5 frozen for this diagnosis

Current training contract:

```text
PROJECT7_DIRECT_TFV_CORE_TRAINING_V5
```

Model remains shared pairwise `V(candidate)-V(reference)` with 109 facility contributions plus joint
interaction. MAIN is facility-balanced; JOINT is changed-facility-density balanced; CONTROL uses D3
HOLD-reference TrainFit groups. Do **not** retrain Step2 merely to respond to the current false-benefit
finding until the calibrated Step3 test is completed.

## Step3 V5 — optimizer-aware one-sided admission

Canonical contracts:

```text
PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V5
PROJECT7_DIRECT_TFV_OPTIMIZER_AWARE_ONE_SIDED_ADMISSION_V1
```

The optimizer is unchanged: all-109 screening, D3-HOLD q95 active-density support, bounded H120
L-BFGS-B, TFV-only objective. The new admission layer changes only whether the optimized plan is
executed:

```text
UCB_like = predicted_delta_TFV + one_sided_residual_margin
if UCB_like < 0 and first 10-minute block changes:
    execute first block
else:
    HOLD
```

The residual margin conservatively combines:
1. rainfall-disjoint D3 HOLD cached residuals (base value-model error); and
2. exact same-prefix H360 SWMM residuals from **optimizer-selected Development plans** (optimizer
   selection shift).

This is not a hand-tuned minimum-improvement threshold and not a new objective. It is a target-error
calibration derived from authoritative Development evidence. The event(s) used to create optimizer
replay residuals are calibration evidence and cannot later be claimed as independent post-calibration
validation.

A plan whose first executed block is unchanged is always HOLD, even if later hypothetical blocks have
predicted benefit, because those later blocks will be re-optimised after re-observation.

## Counterfactual audit V2

The runtime stores the raw optimized plan even when admission rejects it. The counterfactual selector
therefore samples both accepted and rejected raw optimizer plans. This allows the post-calibration
H360 audit to test the intended mechanism directly:

- admitted raw plans should remain truly beneficial more often;
- rejected raw plans should contain weak/false-beneficial extrema;
- all candidate/HOLD branches must have identical causal prefix, target latch, current setting and
  cumulative statistics;
- any replay mismatch remains `COUNTERFACTUAL_REPLAY_P0`.

## Baselines

Frozen Development panel remains:

```text
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

Do not weaken or retune any baseline. Current EFD is strict storage-volume EFD, not the older
Project6 depth-zone EFD-like heuristic. Auto-RBC remains the same causal actuator-adjacent rule
comparator.

## Frozen Development assets

```text
GRAPH
E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz

BASE_CACHE
E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json

RAIN
E:\RTC_sewer\Project7\study_v069\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz

STATE
E:\RTC_sewer\Project7\study_v069\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz
```

Reuse the accepted Step2 V5 checkpoint. Do not rerun full Step2 unless new evidence later identifies a
Step2 model defect.

## Canonical Development order

1. protect local uncommitted replay work before syncing GitHub;
2. hard-sync `main`, install, pytest, current lint, CUDA check;
3. reuse the accepted Step2 V5 checkpoint;
4. create admission calibration with `scripts/calibrate_direct_tfv_admission_current.py`, including the
   exact T5 optimizer H360 replay report;
5. require replay prefix mismatch = 0 and record optimizer replay calibration event IDs;
6. run `scripts/run_step3_direct_tfv_solver_calibrated_current.py` on the rainfall-disjoint D3 audit
   half;
7. T5 may be rerun only as an engineering/calibration-consistency smoke because its optimizer replay
   was used for calibration;
8. run T10/T20 as post-calibration Development event audits;
9. generate exact H360 replay on T10/T20, including both accepted and rejected raw optimizer plans;
10. require event-level No-control consistency and reduced false-beneficial rate before returning to
    Auto-RBC competitiveness;
11. stop before Validation/Final/Formal/Policy Lock.

## Scientific boundaries

- system-wide TFV remains the only optimization objective;
- no PFV or Global Peak gate/objective;
- no online SWMM candidate evaluation;
- no future realised rainfall online;
- no q99 canonical promotion from surrogate predictions alone;
- no event-wise Auto-RBC/EFD tuning;
- no use of Validation/Final/Formal/Policy Lock for admission calibration;
- no production promotion while Development benefit remains inconsistent.

## What decides the next branch

If post-calibration T10/T20 regain consistent TFV benefit vs No-control and exact H360 false-beneficial
rate falls materially, classify the current bottleneck as **optimizer-selection calibration limited**
and then reassess the remaining gap to Internal/Auto-RBC.

If event-level benefit remains inconsistent even after weak optimizer extrema are rejected, the next
scientific question is no longer admission. Then inspect whether Step2 lacks optimizer-distribution
training support or whether the H360 value estimand itself is misaligned with closed-loop 10-minute
receding control. Do not solve that by adding more objectives or weakening baselines.
