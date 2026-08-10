# Formal Pipeline — Wuhan RTC v0.6.6

This is the current fail-closed scientific evidence contract for fresh Wuhan RTC runs. It defines scientific correctness; sample-size targets remain recommendations rather than software startup gates.

## A. Objective and truth

Primary objective: minimize authoritative-SWMM system-wide cumulative **TFV**.

- PFV at the frozen eight priority nodes: soft secondary/diagnostic only.
- Priority depth: diagnostic only.
- Global Peak flooding rate: report only.
- Final TFV/PFV truth: cumulative SWMM node flooding-volume statistics.
- Final Global Peak: routing-step replay of the frozen executed decision schedule.

## B. Formal strategies

Exactly:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

Competitive baselines are No-control, Internal RTC, Auto-RBC and EFD. All-open/all-closed are diagnostic extremes. Hold is debug-only.

### Event-paired Internal RTC

The **prepared event INP** is authoritative for rainfall, DWF, initial conditions, hydraulic geometry and event clock. The **frozen network INP** is authoritative for the native `[CONTROLS]` payload.

Internal RTC runtime therefore equals:

```text
prepared event forcing + DWF + initial state + geometry
+
frozen network [CONTROLS] payload
```

No other event section may be taken from the frozen network. Formal evidence hashes the canonical native-controls payload and Final verifies it against the Policy-Locked frozen INP.

No-control and every Python strategy run on the same prepared event with native supervisory controls disabled.

### Information-budget disclosure

Internal RTC is intentionally a strong native engineering comparator: it may access true SWMM rule variables on the native `RULE_STEP`, including from simulation start. Auto-RBC uses current actuator-adjacent true SWMM depths. EFD uses current controlled-storage true SWMM depths. Proposed directly observes only the frozen sparse sensor layout, causal rainfall and actuator readbacks; its 932-node state is reconstructed.

The project makes **no equal-information/equal-frequency baseline claim**. Stronger comparator information budgets are disclosed rather than artificially weakened.

## C. Causal boundary

Allowed online at decision time `t`:

```text
sparse depth/head <= t
realised rainfall <= t
actuator target/current/flow readback <= t
static graph/device features
causal rainfall forecast/scenarios derived without future realised truth
```

Forbidden:

```text
event ID as policy feature
future realised rainfall/runoff
future SWMM state/flooding
future Internal trajectory
Final/locked hydraulic truth
offline future labels presented online
```

## D. Prepared event contract

Before a new v0.6.6 study, use `rtc-prepare-event-suite`.

The preparation contract:

1. canonicalizes rainfall time-series rows to explicit absolute timestamps;
2. preserves the original storm absolute clock and therefore the DWF phase at storm onset;
3. moves simulation/report start earlier to provide a dry/DWF causal history;
4. extends simulation end to an explicit post-rain recovery tail;
5. does not change rainfall intensity values, DWF values, hydraulic geometry or device definitions.

The default first attempt is 60 min pre-rain warm-up and 360 min post-rain tail. These are not immutable scientific constants: if Phase-0 recovery remains right-censored, extend the tail before production training and create a new fresh study root.

## E. Pretraining readiness

`rtc-check-study-readiness` must pass before production training. It binds:

- prepared event registry and hashes;
- sufficient pre-rain history for the locked Step1 history span;
- minimum post-rain tail;
- frozen native-controls template;
- sensor-layout provenance and byte hash;
- rainfall study/provenance scope;
- actuator claim scope.

Policy Lock must include the resulting `WUHAN_RTC_PRETRAINING_READINESS_V1` artifact.

## F. Actuation claim

All discovered SWMM-writable settings remain eligible online. The default research contract is:

```text
SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY
```

A fractional SWMM setting is not evidence of field VFD/continuous positioning. No artificial binary mask is introduced without engineering metadata. A field-deployment claim requires a hashed engineering capability map.

## G. Rainfall scope

The currently audited main Project7 library contains 180 independent design events:

```text
10,20,30,50,75,100 year labels
75,105,150,210,240,300 min durations
block/chicago_center/chicago_early/chicago_late/double_peak patterns
uniform single-gage spatial forcing
```

These actual bytes are valid controlled experimental inputs. The project does not claim independent regeneration from an official Wuhan IDF standard until an authoritative formula/generator provenance is supplied. Missing frequent/short/long/spatially heterogeneous conditions are robustness-scope limitations rather than permission to invent new forcing.

## H. Data roles

- D0: controls-disabled t=0-inclusive reference trajectories, including the dry/DWF prefix.
- D1: development/train-only controlled exploration for Step1 state coverage.
- D2: exact same-prefix local single-actuator perturbations; sampling budget does not reduce the 109-setting online action space.
- D3: joint multi-actuator, multi-control-block action sequences.
- Step1: sparse causal history -> current full six-channel state estimate.
- Step2: current state + future settings/rainfall -> future hydraulic state, actuator flow and flooding consequence.

## I. Exact counterfactual prefix

D2/D3 must replay the controls-disabled No-control source to the same checkpoint and match:

```text
elapsed time
complete node ordering
complete actuator ordering
all six hydraulic state channels
all actuator current settings/readback
SWMM engine version
```

Contract: `EXACT_NO_CONTROL_PREFIX_REPLAY_V1`. Any mismatch aborts before action.

## J. Timing

Keep separate:

```text
SWMM routing step
SWMM native rule step
model/observation step
Python supervisory update
prediction horizon
whole-event control duration
```

Production timing freezes only after non-censored development-only high-frequency evidence.

Hard relationships:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first Proposed control follows a complete causal history
first control aligns with model/control grids
D3 horizon contains complete supervisory control blocks
runtime budget < supervisory control interval
```

The previous h180 pilot was censored and is stale under v0.6.6. The new fresh pilot should test an evidence-motivated longer horizon (initially h210 is reasonable) and freeze only when `horizon_censored=false`.

## K. Step1 contract

Step1 maps sparse causal depth/head history + masks + causal rainfall/actuator context + graph/static features to the current full six-channel hydraulic state. Development train/validation rainfall groups must be disjoint. D1 is train-only.

Acceptance emphasizes unobserved-node state/depth, wet/high-depth subsets, priority diagnostics and event-balanced held-out performance.

## L. Step2 contract

For each model interval:

```text
initial_state = x_t
settings[k], rainfall[k] govern t_k -> t_(k+1)
target_states[k] = x_(k+1)
target_actuator_flows[k] = q_(k+1)
```

Training supervises future hydraulic trajectory, actuator-flow trajectory and exact cumulative SWMM node flooding volume. Action ranking, delta-TFV sign, regret and gradient direction are required acceptance evidence; state RMSE alone is insufficient.

## M. SWMM engine lineage

Step1 data/models, Step2 D2/D3/data/models, Proposed runtime and Final comparison must use compatible engine lineage. Do not upgrade PySWMM/SWMM during one locked experiment.

## N. Scientific event identity

Scientific event identity binds the event INP except policy-only `[CONTROLS]` and execution-only `THREADS`, plus referenced external forcing bytes. Thus all seven strategies on one event must share the same prepared forcing/DWF/clock identity even though Internal receives the frozen native rule payload.

## O. Ten-step acceptance flow

```text
Step 0  prepared inputs / readiness / split / graph / physical preflight
Step 1  Phase-0 high-frequency No-control D0 + checkpoints + small exact-prefix D2
Step 2  response timing + exact SWMM control leverage + non-censored timing freeze
Step 3  production D0/D1 + Step1 train/held-out acceptance
Step 4  production D2/D3 + Step2 train/held-out acceptance
Step 5  local gradient + joint ranking/regret acceptance
Step 6  development closed-loop Proposed vs No-control/Internal/Auto-RBC/EFD
Step 7  runtime/readback/deadline acceptance
Step 8  Policy Lock
Step 9  untouched seven-strategy authoritative-SWMM Final
```

Do not advance from a failed stage by weakening a guard.

## P. Policy Lock

Outer contract remains:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

The v0.6.6 implementation fingerprint additionally binds event preparation/readiness, event-paired Internal semantics, native-control payload identity, baseline information-budget disclosure and simulation-only actuation claims.

The baseline plan must be:

```text
FORMAL_BASELINE_PLAN_V6_EVENT_PAIRED_INFORMATION_DISCLOSED
```

If actuation remains simulation-only, Policy Lock must state `field_deployment_claim=false`.

## Q. Final

Every and only locked Final event must run exactly seven strategies. Each Formal run must match:

- locked prepared scientific event identity;
- locked physical network;
- locked SWMM engine;
- locked model/control cadence;
- Proposed model/config hashes;
- Internal native controls payload hash;
- current implementation identity.

Aggregate within independent rainfall group first, then give rainfall groups equal weight. Primary pairwise interpretation is Proposed vs No-control/Internal/Auto-RBC/EFD; diagnostic extremes remain separate.

## R. Recovery and surface-drainage interpretation

A configured 3 h/6 h/12 h simulation tail is not itself proof of hydraulic recovery. Recovery must be diagnosed relative to flooding cessation and an appropriate dry-weather/reference condition; DWF means total system flow need not approach zero. If the development pilot reaches the simulation endpoint while still flooding/recovering, mark it right-censored and extend the source event tail before production training.

## S. Safe reuse

Reuse requires compatible scientific implementation plus exact input/config/reference/action lineage and generated artifact hashes. File existence alone is never sufficient. Because v0.6.6 changes event preparation and Internal baseline semantics, old v0.6.5 D0/D1/D2/D3/models/acceptance/development/Policy Lock/Final artifacts are not Formal evidence for a v0.6.6 study.

Use `docs/LEAN_FRESH_RUN_V066.md`.
