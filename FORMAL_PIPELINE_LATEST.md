# Formal Pipeline — Wuhan RTC v0.6.9 execution freeze

This is the active fail-closed scientific evidence contract for Project7. It keeps the v0.6.7 physical/rainfall methodology testbed and the v0.6.8 simulation-asset identity system, while freezing the v0.6.9 Train/Validation/Final split, controller/runtime settings, compute-sparing Phase-0 design and dimensionless acceptance gates.

## A. Scientific objective and claim

Primary objective: minimize authoritative-SWMM system-wide cumulative **TFV**.

- PFV at the frozen eight priority nodes: soft secondary/diagnostic only.
- Priority depth: diagnostic only.
- Global Peak flooding rate: report only.
- Final TFV/PFV truth: cumulative SWMM node flooding-volume statistics.
- Final Global Peak: routing-step replay of the frozen executed decision schedule.

Claim scope is exactly:

```text
IDEALIZED_METHODOLOGY_TESTBED_NOT_FIELD_DIGITAL_TWIN
```

No field-calibrated Wuhan digital-twin, field actuator-capability or deployment claim is permitted.

## B. Frozen physical/rainfall testbed and 30-event split

The v0.6.7 source bundle remains authoritative for physical/rainfall assumptions:

```text
932 hydraulic nodes
109 writable actuators
supplied idealized DWF background loading
41 FREE outfalls
source SUBAREAS/infiltration
five retrofit storage curves
57 pump curves PUMP2 -> PUMP4, endpoints unchanged
42 orifices with idealized 10-min full travel
RTC_IN_*/RTC_OUT_* directional flap gates
continuous simulation settings in [0,1]
max supervisory setting delta = 0.5 per 10-min update
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

The active split is frozen before fresh hydraulic generation:

```text
development/train       = 18 events
development/validation  = 6 events
Final                   = 6 events
calibration              = 0
safety_audit             = 0
```

Use only:

```text
configs/project7_v069_split_contract.json
configs/project7_v069_events_with_splits.csv
```

The split was constructed only from return period and duration; **no SWMM hydraulic outcome was used**. Train contains exactly three events at each duration. Validation contains exactly one event at each duration. Final contains exactly one event at each duration. Validation and Final each span all five return periods. Never reshuffle this split after observing hydraulic outcomes.

## C. Frozen runtime clock — keep every time concept separate

The active Project7 runtime contract is:

```text
SWMM simulation START                    = t=0
model/observation/record step            = 300 s
causal history                           = 13 frames = 0,5,...,60 min
first Proposed supervisory decision      = elapsed 60 min
D1 first exploration action              = elapsed 60 min
supervisory update                       = 600 s
prediction horizon                       = 360 min
prediction horizon in model steps        = 72
prediction horizon in control blocks     = 36
effective pre-rain warm-up               = 120 min
Formal post-rain evaluation tail         = 360 min
```

The source v0.6.7 event bundle already contains a 60-min pre-rain prefix. Formal production preparation must use:

```text
--target-effective-warmup-minutes 120
```

so only the missing additional 60 min is added. The registry must keep:

```text
source_pre_rain_prefix_minutes
additional_warmup_minutes
effective_warmup_minutes
```

For the production-prepared event, first Proposed control at elapsed 60 min occurs **60 min before positive design rainfall begins at elapsed 120 min**. This is a preregistered methodology-testbed timing choice.

The production runner must independently inspect the INP clock and fail if actual effective warm-up is not 120 min.

The 360-min post-rain evaluation tail and the 360-min prediction horizon are distinct quantities even though they currently share the same duration.

## D. Temporal actuator continuity — hard execution contract

Opening a new MPC horizon must never reset an actuator to its original/default state.

For every Python-controlled strategy:

1. **Hold between decisions.** Once a setting is commanded, its target remains active until the next 10-min supervisory epoch.
2. **Sequential future path.** Within the complete 360-min MPC/D3 path, the first block is anchored to current readback and every later block is anchored to the preceding block.
3. **Per-update movement bound.** For all actuators, `|u_k - u_(k-1)| <= 0.5` on the 10-min supervisory grid.
4. **Cross-decision continuity.** At the next rolling decision, the executed first move must be feasible relative to both actual `current_setting` and the **previous issued supervisory target**.
5. **No hidden reset.** Initial/default INP settings are not reused as an action anchor after control has begun.

A direction reversal is allowed only progressively. Under the frozen 0.5-per-update assumption, `1.0 -> 0.5 -> 0.0` is feasible; `1.0 -> 0.0` in one 10-min update is not.

For Proposed, the controller itself must emit an already continuous first move; the outer runtime guard is validation-only and must fail on a violation rather than silently repair Proposed output. Auto-RBC, EFD and All-open/All-closed are passed through the same physical continuity envelope; the diagnostic extremes therefore ramp toward 1/0. Internal RTC retains its native SWMM `[CONTROLS]` semantics and receives no Python override.

`max_setting_delta_per_update=0.5` is a simulation-side engineering smoothing assumption, not a field-validated actuator standard.

## E. Formal strategies and information budgets

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

Auto-RBC and EFD use the fixed parameters in `configs/project7_v069_parameter_register.json`. Event-specific or outcome-specific tuning is forbidden.

Information-budget differences are explicitly accepted and disclosed:

- Proposed: frozen sparse sensor layout + realised rainfall + actuator target/current/flow readback only.
- Internal RTC: true native rule variables available to frozen SWMM `[CONTROLS]`.
- Auto-RBC: current true actuator-adjacent node depths.
- EFD: current true controlled-storage depths.

Forbidden for every strategy:

```text
event ID as policy feature
future realised rainfall/runoff
future SWMM state/flooding
future Internal trajectory
Final/locked hydraulic truth
offline future labels presented online
```

## F. Graph and data roles

The graph preserves v0.6.7 node/actuator ordering and 26-D static physical/hydrologic features.

```text
D0  controls-disabled reference trajectory
D1  development/train controlled exploration; first action elapsed 60 min
D2  same-prefix sustained local actuator counterfactual
Phase0 pulse  conditional diagnostic only: one control block then release to base action
D3  joint multi-actuator, multi-control-block continuous sequences
Step1  sparse causal history -> current full hydraulic state
Step2  current state + future settings/rainfall -> future state/flow/flooding
MPC  optimize 360 min, execute first 10-min move, observe/read back, re-solve
```

D2/D3/pulse exact-prefix replay must match elapsed time, complete node/actuator ordering, six hydraulic state channels, current setting/readback and SWMM engine before candidate action.

## G. Simulation identity and local large-data assets

Large data remains outside Git. Use the local asset registry in `docs/SIMULATION_ASSET_MANAGEMENT_V068.md`.

D2/D3 identity binds physical network, event forcing and START/effective warm-up, exact checkpoint state/readback, complete action/sequence, execution semantics, SWMM engine and record stride. Horizon is an explicit identity dimension.

Changing network, rainfall, warm-up, checkpoint state, action/sequence, engine, split registry or time semantics invalidates downstream reuse automatically. A tail-only END extension does not split an otherwise identical physical family.

`VALID_REUSABLE` means the SWMM computation and referenced artifacts match identity/hash contracts. It never means a timing/model/policy gate passed.

Before any large D2/D3 run, write a pre-run census with requested rows, deduplicated unique actions/sequences, endpoint-invalid requests, exact cache hits, covering trajectory hits and required new executions.

## H. Endpoint executability

Before launching SWMM:

```text
D2 required_end = checkpoint_elapsed + horizon
D3 required_end = checkpoint_elapsed + len(sequence) * control_block
```

Require `required_end <= event END - event START`. Never discover endpoint insufficiency after a large batch has already executed.

## I. Phase-0 diagnostics — fixed compute-sparing design

Phase-0 is preregistered as:

```text
6 development/train events selected using forcing descriptors only
4 checkpoints per event
12 actuators per checkpoint
epsilon = 0.15
include center/reference action = yes
high-frequency record stride = 60 s
one authoritative horizon = h360 only
```

Use the rotating all-actuator probe design so the pilot covers the 109-actuator catalog efficiently without shrinking the production action space.

Do **not** launch separate h210/h240/h300 SWMM branches merely to inspect timing. For one identity-equivalent event/checkpoint/action family, use the h360 trajectory and derive shorter timing views in memory. Shorter-window exact TFV still requires an exact cumulative endpoint statistics snapshot and cannot be invented from the h360 final statistics.

Sustained-step censoring remains a diagnostic and does not select the production horizon. Production `horizon=360 min` is user-frozen.

Phase-0 pulse/release recovery is **conditional diagnostic only**. Do not run it automatically. Trigger it only if sustained h360 timing evidence remains materially ambiguous after the D2 timing/leverage review.

## J. Step1/Step2 training and preregistered acceptance

Step1/Step2 first-run architecture and training parameters are frozen at the current code defaults recorded in `configs/project7_v069_parameter_register.json`. Broad architecture/training sweeps are forbidden by default.

Step1 training uses development/train only, with rainfall-group-disjoint development/validation. Report event-balanced unobserved-node state/depth, wet/high-depth and priority diagnostics.

Step2 trains on future hydraulic trajectories, actuator flows and exact cumulative SWMM flooding-volume truth. Require held-out D2/D3 evidence before closed-loop MPC.

Hard dimensionless gates are preregistered in `configs/model_acceptance_contract_v4.json`:

```text
Step1 unobserved depth NSE                  >= 0.70
Step2 exact-TFV rank correlation            >= 0.70
TFV gradient sign accuracy                  >= 0.70
TFV gradient cosine similarity              >= 0.60
D2 TFV rank correlation                     >= 0.70
D2 top-1 hit rate                           >= 0.50
D3 TFV rank correlation                     >= 0.70
D3 top-1 hit rate                           >= 0.50
```

Absolute RMSE/MAE/regret and physical-plausibility metrics remain mandatory diagnostics but are not assigned post-hoc hard thresholds after seeing validation results. Never lower a hard gate because a model fails.

## K. Frozen controller/runtime acceptance

Use only the resolved controller config:

```text
configs/formal_controller_v5.json
```

Controller/forecast parameters are frozen at the current production-code defaults and must not be swept.

Runtime/readback values are frozen as:

```text
decision runtime budget       = 300 s
readback target tolerance     = 1e-6
readback current tolerance    = 0.05
```

Exactly one small development benchmark may verify these fixed values on intended hardware. If the benchmark fails, stop and report the blocker; do not auto-retune the values.

`rtc-accept-runtime` must use contract:

```text
DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V2_TEMPORAL_CONTINUITY
```

For Proposed development runs it must verify:

```text
first decision exactly elapsed 60 min
all decisions on 10-min grid
actual event effective warm-up = 120 min
prediction horizon = 360 min
complete Step1/Step2 engine/source lineage
continuity guard evidence on every decision
max command delta from current readback <= 0.5
max command delta from previous issued target <= 0.5
planned 360-min sequence max step delta <= 0.5
successful target/current readback
no runtime/readback/deadline fatal fallback
decision runtime <= 300 s
```

Any nonzero continuity violation or fatal runtime fallback fails runtime acceptance.

## L. Policy Lock and Final

Policy Lock requires:

```text
PRODUCTION_CONTROLLER_CONFIG_V5_TFV_FIRST_TEMPORAL_CONTINUITY
DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V2_TEMPORAL_CONTINUITY
PROJECT7_V069_30_EVENT_SPLIT_18TRAIN_6VALIDATION_6FINAL_V1
```

Policy Lock binds the exact network, active 30-event split registry, split contract, rainfall/sensor/priority provenance, effective-120 event clock, graph, simulation-asset identity schema, Step1/Step2 models, 60-min first-control epoch, 360-min horizon, temporal-continuity settings, runtime evidence, SWMM engine and implementation hash.

Every one of the **6 Final events** runs all seven strategies on identical event forcing/DWF/clock/network: 42 authoritative Final policy runs before routing-step peak replay/formalization overhead. Final tuning is forbidden.

## M. Fresh acceptance flow

```text
Step 0  adopt/build v0.6.7 physical inputs; overwrite local active split from the frozen v0.6.9 registry; readiness / graph / audit
Step 1  prepare effective-120 events; Phase-0 6×4×12 exact-prefix h360 D2 with 60-s sampling
Step 2  sustained timing + control leverage; conditional pulse only if ambiguity remains; bind already-fixed 360-min timing
Step 3  production D0/D1 on 18 Train + Step1 train; Step1 held-out acceptance on 6 Validation
Step 4  temporally feasible production D2/D3; Step2 train on Train and held-out acceptance on Validation
Step 5  local gradient + joint D2/D3 ranking acceptance using preregistered dimensionless gates
Step 6  development closed-loop Proposed vs No-control/Internal/Auto-RBC/EFD; no baseline tuning
Step 7  one runtime/readback benchmark + full runtime/deadline/temporal-continuity acceptance with 300-s budget
Step 8  Policy Lock
Step 9  untouched 6-event × 7-strategy authoritative-SWMM Final, then formalization/Global-Peak replay
```

Do not weaken guards, tune on Final, reset actions between horizons, rerun identity-equivalent branches, or treat reusable hydraulic assets as passed scientific evidence.

## N. Reuse and acceleration boundary

v0.6.6 and earlier learned evidence remains provenance only. Successful v0.6.7/v0.6.8 authoritative hydraulic branches may be reused only under verified simulation identity and artifact hashes **and only if the active split/time/action contract admits them**. Any data compiled under a different split, time or action contract must be rebuilt or explicitly shown equivalent before use.

Hot-start/runoff-interface acceleration remains equivalence-audit-gated. Exact-prefix correctness, active split lineage and action-time continuity have priority over speed.
