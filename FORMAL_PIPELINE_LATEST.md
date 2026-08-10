# Formal Pipeline — Wuhan RTC v0.6.9

This is the active fail-closed scientific evidence contract for Project7. v0.6.9 keeps the v0.6.7 physical/rainfall methodology testbed and the v0.6.8 simulation-asset identity system, while freezing realistic rolling-control time semantics.

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

## B. Frozen physical/rainfall testbed

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

## C. Frozen runtime clock — keep every time concept separate

The active Project7 runtime contract is:

```text
SWMM simulation START                    = t=0
model/observation/record step            = 300 s
causal history                           = 13 frames = 0,5,...,60 min
first Proposed supervisory decision      = elapsed 60 min
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

For the production-prepared event, first Proposed control at elapsed 60 min occurs **60 min before positive design rainfall begins at elapsed 120 min**. This is a pre-registered methodology-testbed timing choice.

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
D1  development/train controlled exploration
D2  same-prefix sustained local actuator counterfactual
Phase0 pulse  one control block then release to base action
D3  joint multi-actuator, multi-control-block continuous sequences
Step1  sparse causal history -> current full hydraulic state
Step2  current state + future settings/rainfall -> future state/flow/flooding
MPC  optimize 360 min, execute first 10-min move, observe/read back, re-solve
```

D2/D3/pulse exact-prefix replay must match elapsed time, complete node/actuator ordering, six hydraulic state channels, current setting/readback and SWMM engine before candidate action.

## G. Simulation identity and local large-data assets

Large data remains outside Git. Use the local asset registry in `docs/SIMULATION_ASSET_MANAGEMENT_V068.md`.

D2/D3 identity binds physical network, event forcing and START/effective warm-up, exact checkpoint state/readback, complete action/sequence, execution semantics, SWMM engine and record stride. Horizon is an explicit identity dimension.

Changing network, rainfall, warm-up, checkpoint state, action/sequence, engine or time semantics invalidates reuse automatically. A tail-only END extension does not split an otherwise identical physical family.

`VALID_REUSABLE` means the SWMM computation and referenced artifacts match identity/hash contracts. It never means a timing/model/policy gate passed.

Before any large D2/D3 run, write a pre-run census with requested rows, deduplicated unique actions/sequences, endpoint-invalid requests, exact cache hits, covering trajectory hits and required new executions.

## H. Endpoint executability

Before launching SWMM:

```text
D2 required_end = checkpoint_elapsed + horizon
D3 required_end = checkpoint_elapsed + len(sequence) * control_block
```

Require `required_end <= event END - event START`. Never discover endpoint insufficiency after a large batch has already executed.

## I. Phase-0 diagnostics and the fixed 360-min horizon

Phase-0 high-frequency D2 may use <=60 s Python sampling. This is a callback/observation cadence and does not redefine SWMM's routing step.

For one physical event/checkpoint/action family, prefer one longest executable trajectory and derive shorter timing views rather than rerunning SWMM. Exact shorter-window TFV requires an exact cumulative SWMM endpoint snapshot.

Sustained-step censoring and pulse/release recovery remain diagnostics. Preserve the original h360 censor result honestly.

**v0.6.9 does not use the censor flag as an automatic production-horizon selector.** Production `horizon=360 min` is a user-frozen methodology-testbed design choice. `rtc-freeze-phase0-timing` records the censor result but binds the fixed 300/600/60/360 grid.

## J. Step1/Step2 acceptance

Step1 training uses development/train only, with rainfall-group-disjoint development/validation. Report event-balanced unobserved-node state/depth, wet/high-depth and priority diagnostics.

Step2 trains on future hydraulic trajectories, actuator flows and exact cumulative SWMM flooding-volume truth. Require held-out evidence for:

```text
D2 finite-difference direction agreement
delta-TFV sign accuracy
candidate rank correlation
joint D2/D3 best-action regret
event-balanced performance
```

Do not enter closed-loop MPC if Step2 cannot rank actions reliably.

## K. Runtime acceptance

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
compute time within the frozen <600 s budget
```

Any nonzero continuity violation fails runtime acceptance.

## L. Policy Lock and Final

Policy Lock requires:

```text
PRODUCTION_CONTROLLER_CONFIG_V5_TFV_FIRST_TEMPORAL_CONTINUITY
DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V2_TEMPORAL_CONTINUITY
```

Legacy controller V4 and runtime-acceptance V1 cannot be Policy-Locked under v0.6.9.

Policy Lock binds the exact network, 30-event registry/splits, rainfall/sensor/priority provenance, effective-120 event clock, graph, simulation-asset identity schema, Step1/Step2 models, 60-min first-control epoch, 360-min horizon, temporal-continuity settings, runtime evidence, SWMM engine and implementation hash.

Every Final event runs all seven strategies on identical event forcing/DWF/clock/network. Final tuning is forbidden.

## M. Fresh acceptance flow

```text
Step 0  adopt/build v0.6.7 physical inputs / readiness / graph / audit
Step 1  prepare effective-120 events + high-frequency D0 + exact-prefix D2
Step 2  sustained timing + pulse/recovery + control leverage + bind fixed 360-min timing
Step 3  production D0/D1 + Step1 train/held-out acceptance
Step 4  temporally feasible production D2/D3 + Step2 train/held-out acceptance
Step 5  local gradient + joint ranking/regret acceptance
Step 6  development closed-loop Proposed vs No-control/Internal/Auto-RBC/EFD
Step 7  runtime/readback/deadline + temporal-continuity acceptance
Step 8  Policy Lock
Step 9  untouched seven-strategy authoritative-SWMM Final
```

Do not weaken guards, tune on Final, reset actions between horizons, or treat reusable hydraulic assets as passed scientific evidence.

## N. Reuse and acceleration boundary

v0.6.6 and earlier learned evidence remains provenance only. Successful v0.6.7/v0.6.8 authoritative hydraulic branches may be reused only under verified simulation identity and artifact hashes. Any data compiled under a different time/action contract must be rebuilt or explicitly shown equivalent before use.

Hot-start/runoff-interface acceleration remains equivalence-audit-gated. Exact-prefix correctness and action-time continuity have priority over speed.
