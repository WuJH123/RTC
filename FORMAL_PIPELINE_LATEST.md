# Formal Pipeline — Wuhan TFV-first causal fresh-data contract

This is the only admissible workflow for new Formal evidence.

## A. Frozen scientific question

Can a sparse-sensing, differentiable-surrogate, continuous receding-horizon controller reduce system-wide cumulative flooding volume in the Wuhan large sewer network, while using only causal online information and keeping observed priority-site deterioration as a soft secondary consideration?

Primary objective: **TFV minimum**.

Priority PFV/depth: **soft/diagnostic**, not a hard gate.

Global Peak: **report only**, obtained from routing-step replay.

All Final claims: **authoritative SWMM truth**.

## B. Frozen priority nodes

Exactly eight:

```text
MSLBZW001
HS1316314
YS2530050
HS2529198
MH0200773
HS1330349
HS2529139
HS2529052
```

Preflight and Policy Lock both require all eight IDs to exist in the frozen graph.

## C. No-control contract

Contract ID: `NO_SUPERVISORY_RTC_V2`.

No-control:

- removes executable user-defined `[CONTROLS]`;
- makes no Python setting writes;
- preserves the physical network and forcing;
- preserves pump curves and initial status;
- preserves intrinsic `[PUMPS]` Startup/Shutoff depth logic;
- preserves regulator physical/default behaviour.

It is neither All-open nor All-closed.

`rtc-inp-audit-v2` must record the remaining intrinsic local pump controls before any data generation.

## D. Formal strategy matrix

Exactly:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

`hold` is debug-only and is forbidden in Formal strategy matrices because it can collapse to No-control.

`internal_rtc` retains native `[CONTROLS]` and receives no Python actuator writes.

All other Formal strategies run on controls-disabled runtime INPs. All-open/All-closed share the same pre-control history as Proposed and start their commands at the frozen first-control epoch.

## E. Fresh-data rule

Historical RTC-derived data/model/evidence is not admissible.

Admissible existing inputs are limited to physical/observation/forcing definitions, for example:

- frozen physical INP;
- verified priority/sensor metadata;
- rainfall forcing definitions used to create the new event registry.

Create a new >=160-group event registry, validate it, then initialize an empty workspace:

```powershell
rtc-init-fresh-workspace `
  --root E:\RTC_sewer\RTC_fresh_v05 `
  --inp <FROZEN_INP> `
  --priority data/priority_nodes.txt `
  --events <NEW_EVENT_REGISTRY_WITH_SPLITS.csv>
```

The canonical copied registry is:

```text
<FRESH_ROOT>/contracts/event_registry_with_splits.csv
```

Policy Lock requires `FRESH_WORKSPACE_MANIFEST.json` and key RTC-derived artifacts inside this root.

## F. Rainfall design

Minimum independent rainfall groups: **160**.

Minimum role counts:

```text
development  >= 96
calibration  >= 24
safety_audit >= 16
final        >= 24
```

Development validation must contain >=19 independent rainfall groups at the minimum design.

No rainfall group may cross roles or development train/validation folds.

```powershell
rtc-validate-rainfall-design `
  --events <NEW_EVENT_REGISTRY_WITH_SPLITS.csv> `
  --out <FRESH_ROOT>/contracts/rainfall_design_evidence.json
```

The Final set remains untouched until Policy Lock.

## G. Time-scale discovery before production data

Do **not** freeze 5 min / 10 min / 120 min a priori.

Phase-0 D2 must sample <=60 s so sub-5-min responses are observable. SWMM internal Dynamic-Wave routing remains the INP routing step.

Phase-0 must report:

- actuator target/current-setting response;
- actuator-flow onset/peak/mass90;
- network flooding-rate onset/peak/mass90;
- network maximum-depth response.

```powershell
rtc-phase0-timescale `
  --manifest <FRESH_ROOT>/d2_phase0/probe_manifest.csv `
  --run-summary <FRESH_ROOT>/d2_phase0/runs/RUN_SUMMARY.csv `
  --detail-out <FRESH_ROOT>/phase0/timescale_detail.csv `
  --summary-out <FRESH_ROOT>/phase0/timescale_summary.json `
  --max-sample-seconds 60
```

If a selected diagnostic horizon censors response-mass estimates near its endpoint, lengthen the Phase-0 horizon before freezing the production horizon.

## H. Candidate 5-min/10-min causal implementation

If Phase-0 accepts:

```text
model/observation = 300 s
control update    = 600 s
history_steps     = 13
first control     = 60 min
```

then t=0 is explicitly included, giving exactly:

```text
0,5,10,...,55,60 min = 13 causal frames
```

First MPC = 60 min, then 70, 80, ... min.

One 10-min control block spans two 5-min model intervals.

`CausalTimingContract` fails closed if:

- control update is not divisible by model step;
- first control is off either time grid;
- first control precedes a complete causal history;
- horizon is shorter than one full control interval.

## I. Online information boundary

At decision time t, policy inputs may contain only:

- sparse depth/head observations <=t;
- realised rainfall <=t;
- actuator target/current-setting/flow readback <=t;
- static graph and actuator physics;
- causal forecast generated from observed rainfall history.

Forbidden:

- event ID as control signal;
- future realised rainfall/runoff;
- future SWMM state/flooding;
- future native/Internal trajectory;
- Final truth;
- offline labels masquerading as online state.

## J. Baseline cache / canonical D0

Generate fixed references once and reuse them:

```powershell
rtc-build-baseline-cache `
  --events <FRESH_ROOT>/contracts/event_registry_with_splits.csv `
  --config <FRESH_ROOT>/contracts/controller_resolved.json `
  --out-dir <FRESH_ROOT>/baseline_cache `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

Formal fixed strategies:

```text
no_control
internal_rtc
all_open
all_closed
```

The cache must validate actual `[CONTROLS]` state, Python-controller presence, decision logs, commanded extreme settings and physical-network hash.

Use:

- `STEP1_BASELINE_INDEX.csv` for fresh Step1 baselines;
- `NO_CONTROL_D0_INDEX.csv` for replayable D2/D3 checkpoints.

## K. D1 / Step1

D1 is new development/train controlled-state exploration for Step1 only.

D1 is forbidden as a D2/D3 checkpoint source.

Step1 must be trained only on fresh workspace data and validated on rainfall-group-disjoint development validation groups.

Step1 acceptance includes the preregistered reconstruction metric family and model hash.

## L. D2 same-prefix truth

Checkpoints come only from fresh controls-disabled, no-Python-write No-control trajectories.

D2 branches must preserve exact prefix lineage and save:

- current SI state;
- causal rainfall;
- requested/current actuator settings;
- actuator flow;
- H-step hydraulic target states;
- H-step actuator flows;
- exact cumulative node flooding-volume delta from SWMM statistics;
- checkpoint/action/event/rainfall hashes.

At setting bounds, gradient truth uses valid one-sided finite differences.

## M. D3 interaction truth

D3 uses the same replayable prefix contract and generates fresh multi-actuator, multi-step interaction sequences.

No fixed active facility subset is allowed in the scientific contract.

## N. Step2 index / training

Build one deduplicated D2+D3 index:

```powershell
rtc-build-step2-index `
  --d2-manifest <FRESH_ROOT>/d2/probe_manifest.csv `
  --d2-run-summary <FRESH_ROOT>/d2/runs/RUN_SUMMARY.csv `
  --d3-run-summary <FRESH_ROOT>/d3/RUN_SUMMARY.csv `
  --out <FRESH_ROOT>/step2/step2_run_index.csv
```

The builder must collapse repeated D2 center/base provenance and reject Final data.

Step2 training supervises hydraulic trajectory, actuator flow and exact cumulative SWMM node flooding volume.

## O. TFV/PFV truth

For node i over interval [t0,t1]:

```text
DeltaV_i = SWMM cumulative flooding volume_i(t1)
         - SWMM cumulative flooding volume_i(t0)
```

Then:

```text
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight priority nodes
```

`Node.flooding` is an instantaneous rate and cannot itself be reported as volume.

The MPC surrogate integrates predicted current+future flooding rates using the shared trapezoidal volume operator; that prediction is not substituted for authoritative SWMM truth.

## P. Global Peak truth

Formal Global Peak is:

```text
max_t sum_i max(flooding_rate_i(t), 0)
```

evaluated synchronously at routing time during frozen-decision replay.

It is not the sum of each node's individual historical peak and is not an MPC hard constraint.

## Q. Gradient/ranking acceptance

Before Proposed development closed-loop can be Formal-eligible:

- Step1 acceptance passes;
- Step2 acceptance passes;
- held-out exact-SWMM TFV gradient acceptance passes;
- held-out exact-SWMM candidate-ranking acceptance passes.

Predicted volume semantics must be identical between Step2, gradient evaluation, ranking evaluation and MPC.

## R. Real-time implementation acceptance

The controller must freeze `decision_runtime_budget_seconds` after workstation benchmarking. The budget must be strictly smaller than the control interval.

Every Proposed development decision records wall-clock Step1+forecast+MPC+projection time.

If a decision exceeds the budget, the stale candidate is rejected and `FALLBACK_COMPUTE_DEADLINE` is executed.

Run:

```powershell
rtc-accept-runtime `
  --run-index <FRESH_ROOT>/development/RUN_INDEX.csv `
  --config <FRESH_ROOT>/contracts/controller_resolved.json `
  --out <FRESH_ROOT>/acceptance/runtime_acceptance.json
```

A Formal-pass runtime gate requires:

```text
control-grid violations          = 0
first-decision violations        = 0
missing runtime diagnostics      = 0
FALLBACK_HISTORY_WARMUP          = 0
FALLBACK_READBACK                = 0
FALLBACK_RUNTIME_ERROR           = 0
FALLBACK_COMPUTE_DEADLINE        = 0
max decision runtime <= budget
```

## S. Pipeline ledger

The causal fresh-data ledger order is:

```text
inp_preflight
rainfall_split
phase0_timescale
d0_d1_coverage
d2_d3_generation
step1_acceptance
step2_acceptance
gradient_acceptance
candidate_ranking_acceptance
closed_loop_development
runtime_timing_acceptance
policy_lock
final_closed_loop_swmm
```

No later stage can pass before all previous required stages pass with hashed evidence.

## T. Policy Lock

Current contract:

`WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V3_CAUSAL_FRESH_DATA`

The artefact map must include at least:

```text
fresh_workspace_manifest
inp_preflight
frozen_inp
priority_nodes
sensor_layout
time_scale_config
step1_model
step2_model
graph_schema
state_schema
actuator_catalog
split_registry
model_acceptance_contract
step1_acceptance
step2_acceptance
gradient_acceptance
candidate_ranking_acceptance
controller_config
rainfall_forecast_config
fallback_policy
baseline_plan
runtime_acceptance
```

The lock verifies fresh-workspace provenance, hashes, rainfall cohort design, causal timing, real-time execution and the exact strategy matrix.

## U. Untouched Final

Only after Policy Lock:

1. generate/resume fixed Final baselines once;
2. run locked Proposed once per Final event;
3. formalize each run;
4. replay exact routing-step Global Peak;
5. compile with `rtc-compile-final-v4`.

Final compiler requires the complete event x strategy matrix:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

No Final result may alter models, thresholds, rainfall forecast, time scales, runtime budget, sensor layout or priority-node definitions.

## V. Field-deployment boundary

SWMM fractional settings provide a valid numerical control experiment. For pumps other than Type5, SWMM applies the setting as a multiplier to the pump-curve flow; regulators have their own fractional setting semantics.

This is not evidence that every physical Wuhan facility has VFD/continuous remote actuation.

A physical deployment claim additionally requires verified per-facility SCADA/operability metadata: discrete/continuous mode, rate/ramp, dwell, interlocks, communications latency, watchdog/fail-safe and manual override. Unknown hardware metadata must not be invented.
