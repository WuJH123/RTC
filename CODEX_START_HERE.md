# Project7 current execution guide

GitHub `main` is the code source of truth. Do not infer the active method from the highest V-number
in a filename: several filenames are intentionally retained for compatibility while their embedded
scientific contract has advanced.

## 1. Frozen research problem

Project7 is an **idealized EPA-SWMM methodology testbed**, not a field digital twin.

```text
sparse causal sensors
 -> Step1 reconstruct CURRENT full-network hydraulic state
 -> Step2 learn 109-facility ACTION -> future delta TFV
 -> Step3 evaluate all 109 facilities relative to HOLD
 -> optimise the predicted-beneficial multi-facility subset inside TrainFit support
 -> execute first 10-minute target only
 -> observe again and repeat
```

Primary objective: **system-wide cumulative TFV minimization**. SWMM is authoritative truth for
offline labels and authoritative control evaluation; SWMM is not the online candidate evaluator.

Frozen clock/action contract:

- model/state step = 300 s;
- control update = 600 s;
- H360 prediction = 72 model steps;
- H120 free control = 12 ten-minute blocks;
- writable facilities = 109;
- every facility is eligible and screened every decision;
- setting movement <= 0.5 per 10-minute supervisory update;
- execute only the first 10-minute target, then re-observe.

## 2. Why the current V5/V4 update exists

Three Development events established two facts simultaneously:

1. Direct-TFV reduced TFV versus No-control on 3/3 events and same-prefix H360 SWMM replay supported
   the local control direction on 4/5 selected decisions; there was no evidence of catastrophic
   continuous-optimizer sign exploitation.
2. Direct-TFV remained weaker than Auto-RBC on 3/3 events. The strongest H360-beneficial decisions
   bound the old q90 active-set ceiling, while the testbed itself is release-dominated (All-open is a
   strong diagnostic extreme).

The response is **not** to weaken Auto-RBC or introduce a hand-tuned improvement threshold. The
current path strengthens learned joint-action support and then asks SWMM whether broader coordinated
control actually improves closed-loop TFV.

## 3. Current Step2 — Direct-TFV V5 training contract

Model architecture remains unchanged:

```text
causal Step1 current state
+ causal rainfall forecast
+ previous managed flow
+ complete H360 reference sequence
+ complete H360 candidate sequence
 -> delta TFV = V(candidate) - V(reference)
```

with 109 facility value differences plus a multi-facility interaction value difference.

Current training contract: `PROJECT7_DIRECT_TFV_CORE_TRAINING_V5`.

1. **MAIN** — exact single-facility branches learn 109 facility effects with facility-balanced
   regression.
2. **JOINT** — multi-facility branches are balanced by changed-facility count and update the shared
   facility encoder/head plus interaction head. This prevents numerous sparse joint branches from
   drowning out rarer dense coordinated branches.
3. **CONTROL** — only D3 HOLD-reference TrainFit groups fine-tune the late value representation with
   symmetric HOLD-relative sign loss.

D4 FIT/AUDIT remain reference-shift stress diagnostics, not Step3 gates. There is no extra harmful-
action bias, top-1 policy loss, uncertainty gate or calibrated minimum-improvement threshold.

### Action support

Per-facility q95 action magnitudes use all admitted TrainFit sources for identifiability/coverage.
The **joint changed-facility q50/q75/q90/q95/q99/max used by online Step3 are derived from D3
HOLD-reference branches only**. D2/D4 reference-shift geometry cannot inflate the online active set.

Legacy V4 checkpoints remain runtime-readable, but because they lack the additive V2 joint-density
support payload, Step3 V4 automatically fails back to q90.

## 4. Current Step3 — support-aware all-109 MPC V4

Canonical module:

```text
src/rtc/step3_tfv_value_mpc_v4.py
```

At every 10-minute decision:

1. build exact HOLD H360 reference;
2. screen all 109 facilities with up/down, half/full-radius, pulse/persistent probes;
3. rank predicted-beneficial facilities (`best predicted delta TFV < 0`);
4. choose dynamic active set;
5. default active-density support is D3-HOLD TrainFit **q95** for a current V5 checkpoint;
6. never exceed the maximum D3-HOLD joint changed-facility count observed in TrainFit;
7. run bounded L-BFGS-B over 12 free 10-minute blocks;
8. retain physical min/max, <=0.5 update and per-facility TrainFit magnitude geometry;
9. execute if optimised predicted delta TFV < 0, otherwise HOLD;
10. execute first 10-minute target only, then re-observe.

`q90` is the conservative ablation. `q99` is diagnostic only unless later authoritative SWMM evidence
supports promotion. Do not jump directly to an unrestricted 109-facility simultaneous move.

## 5. Counterfactual integrity

Same-prefix H360 replay must reproduce the authoritative closed-loop target-latch timing. Use:

```text
src/rtc/direct_tfv_replay_guard.py
PROJECT7_DIRECT_TFV_REPLAY_USES_AUTHORITATIVE_TARGET_LATCH_V1
```

Before branching, candidate/HOLD must match on current state, target latch, physical current settings
and cumulative statistics, and the branch target must match the logged HOLD reference. Any mismatch
is `COUNTERFACTUAL_REPLAY_P0` and blocks scientific interpretation.

## 6. Baseline semantics

The fixed Development panel remains:

```text
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

Scientific comparators are Internal RTC, Auto-RBC and EFD. All-open/all-closed are diagnostic
extremes.

Current `efd` is a **strict storage-volume Equal Filling Degree** comparator controlling writable
storage outflows only. It must not be described as the older Project6 depth-zone EFD-like heuristic.
Current Proposed uses no RBC warm start and no RBC safety fallback.

Do not weaken or retune a baseline in response to Proposed performance. Baseline information basis
(sensor counts and rule inputs) must be reported transparently.

## 7. Frozen Development assets

```text
GRAPH
E:\RTC_sewer\Project7\study_v069\formal_assets\graph_schema.npz

BASE_CACHE
E:\RTC_sewer\Project7\study_v069\step2_v60_control_latent_rebuild\training_cache_v60\CACHE_MANIFEST.json

D4_FIT
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_fit\training_cache\CACHE_MANIFEST.json

D4_AUDIT
E:\RTC_sewer\Project7\study_v069\step2_v125_d4_v2_bc04bc7\cache_audit\training_cache\CACHE_MANIFEST.json

RAIN
E:\RTC_sewer\Project7\study_v069\step2_v123_tfv_pfv_knowledge_guided_mpc\addbbd3\STEP2_V123_CAUSAL_FORECAST_STORE.npz

STATE
E:\RTC_sewer\Project7\study_v069\step2_v127_corrected_base_7634cd9\STEP2_V127_CAUSAL_STATE_STORE_V2.npz
```

## 8. Canonical Development order

1. protect any local uncommitted same-prefix replay work before syncing GitHub;
2. hard-sync `main`, install, full pytest, current lint, CUDA health;
3. Direct-TFV V5 smoke, then full V5 DEV if smoke is structurally sound;
4. require 109/109 exact single-facility coverage and report D3-HOLD q90/q95/q99/max joint support;
5. judge Step2 primarily from D3 HOLD-reference harmful fraction, selected true delta TFV, regret,
   pairwise and sign; D4 remains diagnostic;
6. run Step3 V4 q95 solver audit, with q90 as the pre-registered conservative ablation;
7. run one clean authoritative T5 Development SWMM closed loop;
8. run exact same-prefix H360 SWMM counterfactuals using the shared target-latch guard;
9. compare against the frozen provenance-verified six-baseline panel;
10. only if T5 execution/science passes, repeat on T10/T20 and aggregate event-balanced evidence;
11. stop before Validation/Final/Formal/Policy Lock.

## 9. Scientific boundaries

- no future realised rainfall online;
- no future SWMM state/flood truth online;
- no D4 AUDIT fitting;
- no return to V128 as the canonical target;
- no arbitrary minimum-improvement threshold;
- no uncertainty gate added to improve baseline ranking;
- no weakening or event-wise tuning of Auto-RBC/EFD;
- no unrestricted 109-way joint action without TrainFit/SWMM support;
- production/seven-strategy promotion remains fail-closed;
- Validation/Final/Formal/Policy Lock remain untouched during Development.

## 10. What decides success

A training metric improvement alone is not success. The decisive chain is:

```text
109/109 facility coverage
+ D3 HOLD-reference joint-density support
+ Step3 q95 support/engineering violations = 0
+ target write/readback = PASS
+ same-prefix H360 direction remains SWMM-supported
+ authoritative event TFV improves versus No-control
+ gap to competent Internal/Auto-RBC comparators narrows without weakening them
```

If q95 remains binding and authoritative same-prefix SWMM shows denser coordinated actions are truly
beneficial, the next step is a **small targeted dense-joint SWMM augmentation** around the relevant
changed-facility counts. Do not replace that evidence with a hand-tuned threshold or an unrestricted
109-dimensional optimizer.
