# Formal Pipeline — Wuhan RTC v0.6.1

This is the fail-closed scientific contract for new Formal evidence.

## A. Scientific objective

Can a sparse-sensing, differentiable-surrogate, continuous receding-horizon controller reduce **system-wide cumulative Total Flood Volume (TFV)** in the Wuhan large sewer network while using only causal online information?

- TFV: primary objective.
- **PFV = Priority Flood Volume** at the frozen eight priority nodes: soft secondary/diagnostic only.
- Priority depth: soft/diagnostic only.
- Global Peak flooding rate: report only.
- Final truth: authoritative SWMM only.

PFV is not peak flooding flow in this project.

## B. Frozen priority nodes

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

All eight must exist in the actual frozen INP/graph.

## C. Online causality

Allowed at decision time `t`:

```text
sparse depth/head observations <=t
realised rainfall <=t
actuator target/current-setting/flow readback <=t
static graph/device features
rainfall forecast generated only from causal history
```

Forbidden:

```text
event ID as a policy signal
future realised rainfall/runoff
future SWMM state/flooding
future Internal-RTC trajectory
offline future labels masquerading as online state
Final truth
```

Runtime order is observe -> reconstruct current state -> causal forecast -> MPC -> project/write -> hold -> readback -> next decision.

## D. No-control and Formal strategies

No-control contract: `NO_SUPERVISORY_RTC_V2`.

No-control removes executable user `[CONTROLS]` and makes no Python actuator writes while preserving the physical network, forcing, pump curves/status, intrinsic `[PUMPS]` Startup/Shutoff logic and regulator physics.

Internal-RTC retains native `[CONTROLS]` and receives no Python writes.

Formal matrix is exactly:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

`hold` is debug-only.

## E. Rainfall-group contract

`rainfall_group` is the leakage/statistical unit.

Hard correctness requirements:

- unique event IDs;
- no rainfall group crosses scientific splits;
- development train/validation are rainfall-group-disjoint;
- at least one untouched Final group;
- Final absent from pre-lock training/tuning;
- referenced INPs exist.

About 160 independent rainfall groups remains a recommended paper-strength target rather than an execution gate.

Formal metrics first aggregate within rainfall group and then give independent groups equal weight.

## F. Phase-0 before production timing

Do not assume 5/10/120 min from historical work.

Use development-only controls-disabled D0/D2 data sampled at `<=60 s` while SWMM retains its internal routing step. Evaluate readback lag, actuator-flow `t10/t50/t90`, network response/peak timing and horizon censoring.

A Phase-0 branch whose time grid differs from production cannot enter production Step2 shards.

## G. Frozen timing contract

Required:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control aligns with model/control grids
full t=0-inclusive causal history exists before first MPC
prediction horizon >= one control interval
Formal D3 horizon contains an integer number of control blocks
decision_runtime_budget_seconds < control_update_seconds
```

Example accepted 5/10/120-min configuration:

```text
model step       = 300 s
control update   = 600 s
history          = 13 frames at 0,5,...,60 min
Step2 horizon    = 24 model steps
control block    = 2 model steps
D3 sequence      = 12 supervisory control blocks
```

v0.6.1 explicitly separates `model_horizon_steps` from `control_blocks`.

## H. Static Formal assets

Compile with:

```text
rtc-compile-formal-assets
```

The frozen graph/state/actuator catalog must retain the full discovered writable actuator set and valid priority/sensor mappings. No Engineering36/fixed active subset is created.

## I. Fixed baseline cache

After timing is frozen, generate fixed references once per event/timing/strategy contract and reuse only when generation/config/artifact hashes verify.

Use:

```text
STEP1_BASELINE_INDEX.csv -> Step1 baseline trajectories
NO_CONTROL_D0_INDEX.csv  -> replayable D2/D3 checkpoint source
```

## J. D1 / Step1

D1 is development/train controlled-state exploration for Step1 coverage only. It cannot be a D2/D3 prefix.

Step1 maps causal sparse history/context to the **current** full hydraulic state. Its trajectory spacing must equal the frozen model step. Validation is rainfall-group-disjoint and group-balanced.

## K. D2 same-prefix local truth

At checkpoint `t`:

```text
record pre-action x_t and cumulative SWMM statistics
write candidate u_t
run SWMM forward
record x_(t+1)...x_(t+H) and actuator flows
subtract exact cumulative node flooding-volume statistics
```

D2 supplies local/boundary action-effect and finite-difference gradient truth. At exact 0/1 setting bounds, feasible one-sided differences are used.

## L. D3 joint truth — model steps are not control blocks

D3 uses the same replayable No-control prefix and generates joint multi-actuator/multi-step setting sequences. Every discovered actuator is eligible.

The final `rtc-design-d3` reads the frozen controller config and stamps:

```text
model_horizon_steps
model_step_seconds
control_update_seconds
control_block_steps
control_blocks
D3_MODEL_STEP_CONTROL_BLOCK_ALIGNMENT_V1
```

The public `rtc-run-d3-batch` refuses runtime model/control clocks that differ from the design manifest.

This prevents a 24-step, 5-min Step2 horizon from being misinterpreted as 24 ten-minute D3 action blocks.

## M. Step2 model-based transition supervision

This is model-based MPC, not reinforcement learning; no reward column is required.

Step2 learns:

```text
current state
+ continuous actuator setting sequence
+ exogenous rainfall sequence
+ previous actuator flow/device physics
-> future hydraulic trajectory
+ future actuator flows
+ exact cumulative SWMM node flood-volume target
```

Each shard set has one immutable:

```text
model_step_seconds
horizon_steps
```

Mixed time grids and D2/D3 horizon mismatch fail closed.

## N. Step2 physical interpretation

Step2 is **physics-informed**, not a mathematically exact mass-conserving hydraulic solver.

The architecture predicts setting-dependent actuator flow and injects it with opposite upstream/downstream signs before a learned graph hydraulic transition. Formal training supervises state, managed flow and exact SWMM flood volume.

Formal Step2 acceptance additionally reports group-balanced:

```text
negative_depth_fraction
negative_flooding_rate_fraction
negative_node_volume_fraction
nonfinite_state_fraction
nonfinite_actuator_flow_fraction
```

These are physical-plausibility diagnostics. Authoritative hydraulics remain SWMM.

## O. One learned flood-volume operator

Step2 training, validation, MPC, gradient validation and ranking use the same predicted-volume operator:

```text
trapezoidal integration of checkpoint/current flooding rate + future predicted rates
```

The authoritative label remains SWMM cumulative node-statistics volume.

## P. TFV / PFV / Global Peak truth

For node `i` over `[t0,t1]`:

```text
DeltaV_i = SWMM cumulative flooding_volume_i(t1)
         - SWMM cumulative flooding_volume_i(t0)
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the eight priority nodes
```

`Node.flooding` is instantaneous rate only.

Global Peak:

```text
max_t sum_i max(flooding_rate_i(t), 0)
```

is obtained from routing-step observation of a frozen-decision replay that preserves the original Python target-write cadence.

## Q. All-actuator continuous MPC

MPC optimises direct continuous settings for every writable actuator and projects all future control blocks to the numerical/rate contract. No fixed active subset, runtime Top-K or artificial binary pump conversion is permitted.

Priority deterioration enters only as a secondary preference inside a TFV-near-optimal envelope.

## R. Required model/action-effect acceptance

Before Policy Lock:

1. Step1 held-out group-balanced reconstruction passes;
2. Step2 held-out trajectory/exact-TFV acceptance passes;
3. D2 local/boundary exact-SWMM TFV-gradient acceptance passes;
4. D2 local + D3 joint-sequence ranking/regret acceptance passes;
5. Proposed development closed-loop SWMM runs complete;
6. real-time history/readback/deadline acceptance passes.

Single-actuator D2 ranking alone is not sufficient evidence for the joint action space searched by MPC.

## S. Safe resume / storage

D0/D1/D2/D3 and fixed baselines resume only when compatible scientific/numerical generation keys and required artifact hashes verify. File existence alone is never sufficient.

Step1/Step2 training uses atomic epoch state with model, optimiser, scaler and RNG state.

Raw row-wise CSV and SWMM `.rpt/.out` are debug-only and normally deleted after successful compact evidence is produced.

## T. Ordered evidence ledger and Policy Lock

Use:

```text
rtc-record-pipeline-stage
rtc-build-policy-artifacts
rtc-policy-lock-v5
```

Required pre-lock stage order:

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
```

Policy Lock contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

It binds the scientific implementation contract plus exact experiment/model/config/evidence hashes.

## U. Untouched Final

Only after Policy Lock:

1. generate/resume the four fixed Final references once;
2. run locked Proposed once per untouched Final event;
3. formalize every run;
4. replay routing-step Global Peak;
5. compile the complete five-strategy matrix.

Final results may not alter models, timing, thresholds, forecast, runtime budget, sensors, priority nodes or rainfall splits.

## V. Evidence boundary

Passing GitHub CI proves implementation/unit contract integrity, not Wuhan scientific superiority. A claim such as “Proposed significantly reduces TFV” is admissible only after the actual frozen Wuhan untouched Final SWMM evaluation demonstrates it.

Field-deployment claims additionally require real facility SCADA/VFD/discrete-continuous capability, ramp/dwell/interlock, communication/readback latency and fail-safe metadata.
