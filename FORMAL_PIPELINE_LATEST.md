# Formal Pipeline — Wuhan RTC v0.6.8

This is the current fail-closed scientific evidence contract for the Project7 Wuhan methodology testbed. v0.6.8 does **not** change the TFV-first research question or the v0.6.7 physical testbed; it hardens time semantics, simulation identity, asset reuse and Phase-0 recovery diagnosis before production scaling.

## A. Objective and truth

Primary objective: minimize authoritative-SWMM system-wide cumulative **TFV**.

- PFV at the frozen eight priority nodes: soft secondary/diagnostic only.
- Priority depth: diagnostic only.
- Global Peak flooding rate: report only.
- Final TFV/PFV truth: cumulative SWMM node flooding-volume statistics.
- Final Global Peak: routing-step replay of the frozen executed decision schedule.

Scientific claim scope remains narrow: demonstrate control of sewer-node overflow in an idealized simplified SWMM testbed. No field digital-twin or field-deployment claim is permitted.

## B. Active source/input contract

The v0.6.7 method-testbed inputs remain the physical/input source contract:

```text
932 hydraulic nodes
109 writable actuators
supplied idealized DWF
41 FREE outfalls
source SUBAREAS/infiltration
five retrofit storage curves
57 pump curves migrated PUMP2 -> PUMP4, endpoints unchanged
42 orifices with 10-min full travel time
RTC_IN_*/RTC_OUT_* direction-specific flap gates
simulation-only continuous settings in [0,1]
max supervisory setting delta = 0.5 per candidate 10-min update
```

Exactly 30 Chicago design events remain frozen:

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

## C. Event time contract — no ambiguous `warmup_minutes`

Keep separate:

```text
source_pre_rain_prefix_minutes
additional_warmup_minutes
effective_warmup_minutes
post-rain evaluation tail
Phase-0 diagnostic extension tail
```

The source v0.6.7 event bundle already carries a 60-min pre-rain causal prefix. Therefore a request for an **effective 120-min initialization** adds only 60 additional minutes. New Formal preparation must use:

```text
--target-effective-warmup-minutes <target total prefix>
```

and must record all three warm-up fields above. The legacy `--warmup-minutes` option means *additional* prefix only and must not be reported as total effective warm-up.

The production/default whole-event evaluation tail remains 360 min. A longer 480/600-min tail may be used solely to make a Phase-0 checkpoint+horizon endpoint executable. Such a diagnostic extension must not silently redefine the Final evaluation clock.

## D. Formal strategies

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

No-control: native supervisory `[CONTROLS]` disabled and no Python actuator writes, while retaining physical device behavior and intrinsic pump logic.

Internal RTC: prepared event forcing/DWF/initial conditions plus only the frozen network `[CONTROLS]` payload; no Python override. If the native-controls template contains no executable payload or is topologically incompatible, fail closed.

## E. Causal boundary

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

## F. Static graph and data roles

The graph preserves v0.6.7 node/actuator ordering and 26-D physical/hydrologic features.

Data roles:

- D0: controls-disabled reference trajectories including pre-rain initialization.
- D1: development/train-only controlled exploration for Step1 coverage.
- D2: exact same-prefix local single-actuator sustained step perturbations; complete 109-setting vectors.
- Phase-0 pulse: development-only action-release diagnostic; one candidate control block then base action recovery.
- D3: joint multi-actuator, multi-control-block sequences.
- Step1: sparse causal history -> current full hydraulic state.
- Step2: current state + future settings/rainfall -> future hydraulic state, actuator flow and flooding consequence.

## G. Exact counterfactual prefix

D2/D3/Phase-0 pulse must replay controls-disabled No-control to the same checkpoint and match before any candidate write:

```text
elapsed time
complete node ordering
complete actuator ordering
all six hydraulic state channels
all actuator current settings/readback
SWMM engine version
```

Any mismatch aborts.

## H. Simulation identity and local data assets

Large data stays outside Git. Use the local simulation asset contract defined in `docs/SIMULATION_ASSET_MANAGEMENT_V068.md`.

D2 family identity binds:

```text
physical network
event forcing + START/effective warm-up (but not tail-only END extension)
exact checkpoint hydraulic/readback state
complete candidate action
native-controls-disabled execution semantics
SWMM engine
record stride
```

Simulation identity adds the requested horizon.

Changing network/forcing/warm-up/checkpoint state/action/engine invalidates reuse automatically. Merely extending a recovery tail does not split an otherwise identical family.

`VALID_REUSABLE` means a computation/artifact can satisfy an identity-matching request; it does not mean a scientific acceptance gate passed.

Before any large D2/D3 run, write a pre-run census. At minimum report requested rows, projected/deduplicated unique actions, endpoint-invalid requests, exact cache hits, covering long-trajectory hits, new executions and expected output location.

## I. Endpoint executability — before SWMM

For every D2 request:

```text
required_end = checkpoint_elapsed + horizon
```

For every D3/pulse request:

```text
required_end = checkpoint_elapsed + len(sequence) * control_block
```

Require:

```text
required_end <= event END - event START
```

before launching any SWMM worker. Never discover endpoint insufficiency after hundreds of branches have run.

Checkpoint design should still reserve an appropriate minimum tail, but the execution runner is the final fail-closed authority.

## J. Phase-0 timing — longest run first, no duplicate shorter simulations

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

Phase-0 high-frequency sampling remains <=60 s. PySWMM callback stride is an observation/intervention cadence, not the SWMM routing-step definition.

For a fixed event/checkpoint/action family, if the event clock safely supports the largest candidate Phase-0 horizon, prefer **one longest D2 simulation** and derive shorter timing views from its compact trajectory:

```text
h210 view ⊂ h240 view ⊂ h300 view ⊂ h360 trajectory
```

Use `rtc-phase0-timescale --analysis-horizon-minutes ...` to slice the same long trajectory. Do not rerun SWMM solely to observe a shorter prefix.

This prefix reuse is valid for trajectory/timing metrics. Shorter-horizon authoritative cumulative TFV requires an exact SWMM cumulative-statistics endpoint snapshot. New long D2 generation should therefore retain explicitly requested intermediate horizon snapshots; old trajectories without snapshots cannot be retroactively promoted by coarse flooding-rate integration.

Do not weaken the 5% sustained-step peak-near-horizon censor guard merely to pass timing.

## K. Sustained-step censoring vs post-release recovery

A sustained D2 step answers: “how does the system respond while this action remains applied?” It cannot determine how quickly the system decays after the action is removed.

If long sustained D2 shows late depth response after flow/flooding have largely converged, run the development-only Phase-0 pulse/release diagnostic before mechanically extending to ever-longer horizons:

```text
candidate action for one 10-min control block
then restore complete base action
observe flow/flood/depth recovery
```

Commands:

```text
rtc-design-phase0-pulses
rtc-run-d3-batch
rtc-analyse-phase0-pulses
```

Pulse/recovery evidence complements, but does not erase, the sustained-step censor report. Timing freeze must explain both sustained response and post-release recovery.

## L. Control leverage

`rtc-control-leverage-audit` remains a diagnostic rather than a hard acceptance gate. It uses exact SWMM cumulative flooding-volume statistics to determine whether sampled actuator changes measurably alter future TFV.

Do not solve weak leverage by generating an order of magnitude more neural-network data. First diagnose actuator semantics, controllable facilities, hydraulic connectivity, checkpoints and horizon.

## M. Step1 acceptance

Training uses development/train only. Held-out development/validation rainfall groups must remain disjoint. Acceptance emphasizes unobserved-node state/depth, wet/high-depth subsets, priority diagnostics and event-balanced performance.

## N. Step2 acceptance

Training supervises future hydraulic trajectory, actuator-flow trajectory and exact cumulative SWMM flooding-volume truth. Require:

```text
D2 local finite-difference direction agreement
delta-TFV sign accuracy
candidate rank correlation
joint D2/D3 best-action regret
event-balanced held-out results
```

Do not enter MPC if Step2 cannot rank actions reliably.

## O. SWMM truth and units

Full-event TFV/PFV use cumulative SWMM node flooding-volume statistics, not coarse sampled-rate integration. Branch outcomes use cumulative-volume differences after the verified identical prefix. SI controller units remain depth/head in m, flow in m3/s, rainfall in mm/h and flooding volume in m3.

Step1 data/models, Step2 data/models, Proposed runtime and Final must use compatible SWMM engine lineage.

## P. Fresh acceptance flow

```text
Step 0  source-only v0.6.7 physical input adoption / readiness / graph / audit
Step 1  small high-frequency D0 + explicit warm-up sensitivity + exact-prefix longest-horizon D2
Step 2  sustained-step timing + pulse/recovery timing + exact control leverage + timing freeze
Step 3  production D0/D1 + Step1 train/held-out acceptance
Step 4  production D2/D3 + Step2 train/held-out acceptance
Step 5  local gradient + joint ranking/regret acceptance
Step 6  development closed-loop Proposed vs No-control/Internal/Auto-RBC/EFD
Step 7  runtime/readback/deadline acceptance
Step 8  Policy Lock
Step 9  untouched seven-strategy authoritative-SWMM Final
```

Do not advance from a failed stage by weakening a guard, using Final truth, or treating a reusable asset as if it had passed an unrelated scientific gate.

## Q. Policy Lock / Final

Policy Lock binds the exact generated network, 30-event registry/splits, rainfall/sensor/priority provenance, actuator scope, graph schema, **effective warm-up contract**, timing contract, simulation-asset identity schema, models, controller config, SWMM engine and implementation contract.

Every locked Final event runs exactly seven Formal strategies on the identical forcing/DWF/clock/network. Aggregate first within rainfall group and then weight rainfall groups equally.

## R. Reuse rule

v0.6.6 and earlier learned models/evidence remain provenance only because v0.6.7 changed physical actuator semantics, graph features and rainfall/event contract.

Within the same v0.6.7 physical testbed, v0.6.8 may reuse an existing authoritative hydraulic simulation **only** when the new identity importer verifies the physical-state/action/engine family, exact prefix evidence and referenced artifact hashes. Orchestration/file-layout changes alone must not force an expensive identical SWMM rerun.

The first endpoint-failed h240 partial lineage is invalid and must not be imported as reusable evidence. Successful h210/h240-tail480/h300/h360 branches may be indexed as reusable computations, while their original censor findings remain unchanged.

## S. Acceleration boundary

Hot-start and runoff-interface acceleration are not automatically part of Formal v0.6.8. They require a dedicated equivalence audit against exact-prefix replay before adoption. Exact-prefix correctness has priority over speed.
