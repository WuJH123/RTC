# Formal Pipeline — Wuhan RTC v0.6

This file defines the fail-closed scientific evidence sequence for the final TFV-first study. It deliberately distinguishes **scientific correctness gates** from **paper-strength sample-size recommendations**.

## A. Scientific question

Can a sparse-sensing, differentiable-surrogate, continuous receding-horizon controller reduce **system-wide cumulative TFV** in the Wuhan large sewer network while using only causal online information?

- TFV: primary objective.
- Eight-site PFV/depth: soft secondary/diagnostic.
- Global Peak: report only.
- Final truth: authoritative SWMM only.

## B. Frozen priority set

Exactly:

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

All eight must exist in the frozen graph/INP.

## C. No-control

Contract: `NO_SUPERVISORY_RTC_V2`.

No-control removes executable user `[CONTROLS]` and makes no Python writes, while preserving rainfall/runoff forcing, physical network, pump curves, initial status, intrinsic `[PUMPS]` Startup/Shutoff depth logic and regulator physics.

## D. Formal strategy matrix

Exactly:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

`hold` is debug-only. `internal_rtc` retains native `[CONTROLS]`; the other controlled strategies use the controls-disabled physical base.

## E. Rainfall/split contract

Hard correctness requirements:

- unique `event_id` rows;
- no `rainfall_group` crosses scientific splits;
- development contains group-disjoint train and validation folds;
- at least one untouched Final group;
- Final is absent from pre-lock training/tuning;
- referenced INPs exist.

Recommended paper-strength design: about **160 independent rainfall groups**, e.g. 96 development / 24 calibration / 16 safety-audit / 24 Final, with about 19 independent development-validation groups. These counts are **recommendations, not execution gates**.

## F. Study workspace and reuse

Initialize the study with `rtc-init-fresh-workspace`. The workspace binds canonical physical/input/split identities.

Large generated data may live on any local volume. Validity comes from:

```text
scientific/data contract
+ stable RTC implementation fingerprint
+ exact numerical input/config/timing/action lineage
+ generated-artifact hashes
```

The implementation fingerprint represents frozen scientific semantics; it is not a byte hash of every source file. File existence or directory location alone never authorizes reuse.

## G. Phase-0 time-scale discovery

Do not freeze 5/10/120 min a priori.

Use development-only controls-disabled D0/D2 data sampled at `<=60 s` while SWMM keeps its internal routing step. Evaluate:

- readback lag;
- actuator-flow `t10/t50/t90` and peak time;
- network flooding-rate response;
- network maximum-depth response;
- peak-near-horizon censoring.

Do not use sustained-step response-area `mass90` as a hydraulic time constant. Use D3/pulse-release sequences to assess recovery after control release.

Phase-0 branches whose time step differs from the frozen production model step cannot enter production Step2 shards.

## H. Causal production timing

If Phase-0 accepts 300 s observation/model + 600 s control with 13 history frames and first control at 60 min, the causal history is:

```text
0,5,10,...,55,60 min
```

First MPC = 60 min, then 70,80,... min.

Formal timing requires:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control aligns to model/control grids
first control occurs after complete causal history
horizon >= one full control interval
```

The SWMM Dynamic-Wave solver continues at its internal routing step between Python callbacks.

## I. Online information boundary

Allowed at decision time `t`:

- sparse depth/head observations `<=t`;
- realised rainfall `<=t`;
- actuator target/current-setting/flow readback `<=t`;
- static graph/device information;
- a forecast derived only from causal rainfall history.

Forbidden:

- event ID as policy signal;
- future realised rainfall/runoff;
- future SWMM state/flooding;
- future Internal trajectory;
- Final truth;
- offline future labels presented online.

## J. Baseline cache

After production timing is frozen, generate fixed references exactly once per event:

```text
no_control
internal_rtc
all_open
all_closed
```

Cache reuse requires matching event/physical/timing/strategy/implementation lineage and verified artifact hashes.

Use:

```text
STEP1_BASELINE_INDEX.csv  → Step1 baseline trajectories
NO_CONTROL_D0_INDEX.csv   → replayable D2/D3 checkpoint source
```

## K. D1 / Step1

D1 is development/train controlled-state exploration for Step1 coverage only. It is forbidden as a D2/D3 checkpoint source.

Step1 uses causal sparse history and reconstructs the **current** full state. Every trajectory must have the frozen model-step grid. Formal validation is rainfall-group balanced.

Step1 training state may resume only for the same run-index, graph, sensors, timing, architecture/training configuration and compatible implementation contract.

## L. D2 same-prefix truth

D2 checkpoints come only from controls-disabled No-control prefixes.

At checkpoint `t`:

1. record pre-action state/statistics;
2. write candidate action `u_t`;
3. run SWMM forward;
4. record future post-action state/flow;
5. compute exact cumulative SWMM node flooding-volume change.

Therefore the learned alignment is:

```text
x_t + u_t + exogenous rainfall trajectory → x_(t+1),...,x_(t+H)
```

Interior finite differences are central; exact 0/1 bounds use feasible one-sided differences.

## M. D3 interaction truth

D3 generates multi-actuator, multi-step sequences from the same replayable prefix contract. It supplies interaction training data and joint-action validation. Every discovered actuator is eligible; `change_probability` controls how often each actuator is perturbed and `perturbation_std` controls perturbation magnitude. These are data-coverage parameters only and do not create a runtime Top-K or fixed active subset.

## N. Step2 fixed-time contract

Build a deduplicated D2+D3 index, then train/validation shards. Every shard set must have one immutable:

```text
model_step_seconds
horizon_steps
```

Mixed time grids are rejected.

Step2 supervises hydraulic trajectory, actuator flow and exact cumulative SWMM node flooding volume.

## O. One predicted-volume operator

Step2 training, Step2 validation, MPC, gradient validation and ranking must all use:

```text
trapezoidal integration of checkpoint/current flooding rate + future predicted flooding rates
```

A future-only right-endpoint rectangle is forbidden because it would train and optimize different TFV objectives.

## P. Authoritative TFV/PFV

For node `i` over `[t0,t1]`:

```text
DeltaV_i = SWMM cumulative flooding volume_i(t1)
         - SWMM cumulative flooding volume_i(t0)
```

Then:

```text
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight priority nodes
```

`Node.flooding` is an instantaneous rate, not a volume.

## Q. Global Peak

```text
Global Peak = max_t sum_i max(flooding_rate_i(t), 0)
```

Formal peak comes from routing-step observation of a frozen-decision replay. The replay preserves the main run's Python target-write cadence so the reporting calculation cannot change the executed policy.

## R. Step1/Step2/action-effect gates

Before development closed-loop can become Formal-eligible:

1. rainfall-group-balanced Step1 held-out reconstruction passes;
2. rainfall-group-balanced Step2 held-out trajectory/exact-TFV acceptance passes;
3. D2 exact-SWMM local/boundary TFV-gradient acceptance passes;
4. D2 local + D3 joint-sequence ranking/regret acceptance passes.

Single-actuator ranking alone is insufficient because online MPC optimizes a joint 109-dimensional action space.

## S. All-actuator continuous MPC

MPC optimizes direct continuous settings for every writable actuator and projects every future control block to the feasible numerical/rate contract. No fixed active subset, Engineering36, runtime Top-K mask or artificial binary-pump conversion is permitted in the scientific controller.

Priority-site deterioration is minimized only as a secondary preference within a TFV-near-optimal solution envelope.

## T. Real-time execution gate

Freeze `decision_runtime_budget_seconds < control_update_seconds` after workstation benchmarking.

Every Proposed development decision records Step1 + forecast + MPC + projection wall-clock time. A stale candidate becomes `FALLBACK_COMPUTE_DEADLINE`.

Formal runtime acceptance requires:

```text
control-grid violations       = 0
first-decision violations     = 0
missing runtime diagnostics   = 0
FALLBACK_HISTORY_WARMUP       = 0
FALLBACK_READBACK             = 0
FALLBACK_RUNTIME_ERROR        = 0
FALLBACK_COMPUTE_DEADLINE     = 0
max decision runtime <= budget
```

## U. Pipeline ledger

Required evidence order:

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

## V. Policy Lock

Current contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

It binds the scientific implementation fingerprint, physical/input/split identities, priority/sensors, controller/time contract, graph, Step1/Step2 checkpoint/training lineage, acceptance evidence, runtime evidence and baseline plan.

Only direct scientific experiment/evidence artifacts are mandatory; unrelated files are not Policy-Lock gates.

## W. Untouched Final

Only after Policy Lock:

1. generate/resume the four fixed Final references once;
2. run locked Proposed once per untouched Final event;
3. formalize each run;
4. replay routing-step Global Peak;
5. compile the complete five-strategy paired matrix.

Final aggregation first collapses variants inside each rainfall group, then gives each independent rainfall group equal weight.

Final outcomes may not alter models, thresholds, forecast, timing, runtime budget, sensors, priority nodes or split assignments.

## X. Field-deployment boundary

SWMM fractional settings are valid numerical pump/orifice/weir controls. They do not prove that every physical Wuhan facility has continuous remote actuation. A field-deployment claim additionally requires verified SCADA/VFD/discrete-continuous mode, ramp/rate, dwell, interlock, communication/readback latency, watchdog/fail-safe and manual-override metadata.
