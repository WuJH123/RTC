# Formal Pipeline — Wuhan RTC v0.6.7

This is the current fail-closed scientific evidence contract for fresh Wuhan RTC methodology-testbed runs. `docs/METHOD_TESTBED_V067.md` defines the physical/input assumptions.

## A. Objective and truth

Primary objective: minimize authoritative-SWMM system-wide cumulative **TFV**.

- PFV at the frozen eight priority nodes: soft secondary/diagnostic only.
- Priority depth: diagnostic only.
- Global Peak flooding rate: report only.
- Final TFV/PFV truth: cumulative SWMM node flooding-volume statistics.
- Final Global Peak: routing-step replay of the frozen executed decision schedule.

Scientific claim scope is deliberately narrow: demonstrate control of sewer-node overflow in an idealized simplified SWMM testbed. No field digital-twin or field-deployment claim is permitted.

## B. Active source/input contract

A fresh study must be reconstructed from the current source-only network INP, the frozen 89-node sensor layout and the frozen eight priority nodes using `rtc-build-method-testbed-v067`.

The generated network must preserve:

```text
932 hydraulic nodes
109 writable actuators
supplied idealized DWF
41 FREE outfalls
source SUBAREAS/infiltration
five retrofit storage curves
```

It additionally enforces:

```text
57 pump curves: PUMP2 -> PUMP4, endpoints unchanged
42 orifices: 10-min full travel time
RTC_IN_*/RTC_OUT_*: direction-specific flap gates
known copied OFF-rule defects repaired
simulation-only continuous settings in [0,1]
max supervisory setting delta = 0.5 per candidate 10-min update
```

No historical D0/D1/D2/D3/model/shard/acceptance/Policy-Lock/Final artifact is valid input to a fresh v0.6.7 study.

## C. Rainfall contract

Exactly 30 design events:

```text
return periods = 5,10,20,50,100 years
durations      = 60,120,180,240,300,360 min
pattern        = Chicago
r              = 0.39
rain step      = 5 min
spatial mode   = one uniform design gage
```

Wuhan DB4201/T 641-2020 formula:

```text
i = 9.686 * (1 + 0.887 * log10(P)) / (t + 11.23)^0.658   [mm/min]
```

Every five-minute Chicago block is generated analytically and checked against the formula-derived total depth. The rainfall library is a controlled design experiment, not observed historical rainfall.

## D. Event time contract

Default fresh events use:

```text
pre-rain causal prefix = 60 min
post-rain evaluation tail = 360 min
```

The 60-min prefix provides the complete default 13-frame causal history on a 5-min model grid. It is not proof of full dry-weather convergence. Before production scaling, use a small development-only sensitivity comparing storm-onset hydraulics under 60 and 120 min; extend to 180 min only if the onset state remains materially warm-up dependent.

The 360-min post-rain tail is a **fixed evaluation endpoint**, not a recovery-to-zero criterion. A run may be hydrologically right-censored at that endpoint while still remaining valid for the pre-registered fixed-window TFV comparison, provided every strategy uses the identical event clock.

## E. Formal strategies

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

Competitive baselines are No-control, Internal RTC, Auto-RBC and EFD. All-open/all-closed are diagnostic extremes.

Internal RTC receives the prepared event forcing/DWF/initial conditions plus only the frozen network `[CONTROLS]` payload. No-control and Python policies use the same event with native supervisory controls disabled.

Information budgets remain disclosed rather than artificially equalized: Internal may use true native rule variables on the SWMM rule clock; Auto-RBC uses current actuator-adjacent true depths; EFD uses current controlled-storage true depths; Proposed directly observes only the frozen sparse sensor layout, realised rainfall and actuator readback.

## F. Causal boundary

Allowed online at decision time `t`:

```text
sparse depth/head <= t
realised rainfall <= t
actuator target/current/flow readback <= t
static graph/device/hydrologic features
causal rainfall forecasts derived without future realised truth
```

Forbidden:

```text
event ID as a policy feature
future realised rainfall/runoff
future SWMM state/flooding
future Internal trajectory
Final/locked hydraulic truth
offline future labels presented online
```

## G. Static graph contract

The graph must preserve node/actuator ordering and expose the v0.6.7 physical feature vector, including:

```text
invert/max/initial/surcharge depth
node type and ponded area
storage capacity/full-depth area
incident conduit count/length/roughness/primary section scale
contributing subcatchment count/area/impervious area
area-weighted subcatchment width/slope
area-weighted Horton max/min infiltration rates
actuator capacity/geometry/coefficient/flap-gate features
```

These features are static INP properties and do not violate causality.

## H. Data roles

- D0: controls-disabled reference trajectories including the pre-rain prefix.
- D1: development/train-only controlled exploration for Step1 state coverage.
- D2: exact same-prefix local single-actuator perturbations; each branch still contains the complete 109-setting vector.
- D3: joint multi-actuator, multi-control-block sequences.
- Step1: sparse causal history -> current full hydraulic state.
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

Any mismatch aborts before candidate action.

## J. Timing

Keep separate:

```text
pre-rain initialization/history duration
SWMM routing step
SWMM native rule step
model/observation step
Python supervisory update
prediction horizon
fixed post-rain evaluation tail
whole-event control duration
```

Candidate model/control cadence remains 300 s / 600 s with 13 history frames, but prediction horizon must be frozen only from fresh v0.6.7 Phase-0 evidence. Do not infer the MPC horizon from the 360-min evaluation tail.

## K. Step1 acceptance

Training uses development/train only. Held-out development/validation rainfall groups must be disjoint. Acceptance emphasizes unobserved-node state/depth, wet/high-depth subsets, priority diagnostics and event-balanced performance.

## L. Step2 acceptance

Training supervises future hydraulic trajectory, actuator-flow trajectory and exact cumulative SWMM flooding-volume truth. Require action-effect evidence in addition to state error:

```text
D2 local finite-difference direction agreement
delta-TFV sign accuracy
candidate rank correlation
joint D2/D3 best-action regret
event-balanced held-out results
```

Do not enter MPC if Step2 cannot rank actions reliably.

## M. SWMM truth and units

Full-event TFV/PFV come from cumulative SWMM node flooding-volume statistics, not coarse sampled-rate integration. Branch outcomes use the cumulative-volume difference after the verified identical prefix. SI controller units are depth/head in m, flow in m3/s, rainfall in mm/h and flooding volume in m3.

Step1 data/models, Step2 data/models, Proposed runtime and Final must use compatible SWMM engine lineage.

## N. Fresh acceptance flow

```text
Step 0  source-only v0.6.7 input build / readiness / graph / physical audit
Step 1  small high-frequency No-control D0 + warm-up sensitivity + exact-prefix D2
Step 2  response timing + exact SWMM control leverage + timing freeze
Step 3  production D0/D1 + Step1 train/held-out acceptance
Step 4  production D2/D3 + Step2 train/held-out acceptance
Step 5  local gradient + joint ranking/regret acceptance
Step 6  development closed-loop Proposed vs No-control/Internal/Auto-RBC/EFD
Step 7  runtime/readback/deadline acceptance
Step 8  Policy Lock
Step 9  untouched seven-strategy authoritative-SWMM Final
```

Do not advance from a failed stage by weakening a guard or by reusing stale evidence.

## O. Policy Lock / Final

Policy Lock must bind the exact v0.6.7 generated network, 30-event registry/splits, rainfall provenance, sensor provenance, actuator scope, graph feature schema, timing contract, models, controller config, SWMM engine and source-tree identity.

Every locked Final event must run exactly the seven Formal strategies on the identical event forcing/DWF/clock/network. Aggregate first within rainfall group and then weight rainfall groups equally. All-open/all-closed remain diagnostic extremes.

## P. Reuse rule

Old v0.6.6 and earlier hydraulic trajectories/models are provenance only. v0.6.7 changes the network actuator semantics, graph static-feature schema and rainfall/event contract, so **all learned models and scientific evidence must be regenerated**.

Use `docs/METHOD_TESTBED_V067.md` and `scripts/bootstrap_project7_v067.ps1` for the active fresh workflow.
