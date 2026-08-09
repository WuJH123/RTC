# Wuhan RTC — causal fresh-data TFV-first framework

This repository implements a **sparse-sensing → current full-state reconstruction → differentiable hydraulic world model → continuous receding-horizon MPC** workflow for the Wuhan large SWMM drainage model.

The current study is independent of historical RTC outputs. Historical discussions may be used to recover verified physical/observation definitions, but **all RTC-derived hydraulic trajectories, D1/D2/D3 branches, model checkpoints, development runs, acceptance evidence, Policy Lock and Final results must be regenerated with the current code inside one new Fresh Workspace**.

## 1. Frozen scientific objective

At decision time `t`, the policy may use only information available at or before `t`:

- sparse depth/head observations;
- realised rainfall;
- actuator target/current-setting readback and actuator flow;
- frozen graph/device features;
- a forecast produced only from causal rainfall history.

Forbidden online inputs include event ID, future realised rainfall/runoff, future SWMM states/flooding, future Internal-RTC trajectory, offline future labels and Final truth.

The control chain is:

```text
causal observations
      ↓
Step1 current-state reconstruction
      ↓
current full-network hydraulic state
      + causal rainfall scenarios
      ↓
Step2 differentiable hydraulic world model
      ↓
TFV-first continuous MPC
      ↓
engineering projection → target write → hold → readback → repeat
```

Primary objective: **minimise cumulative system-wide TFV**.

Priority-site PFV/depth: **soft secondary/diagnostic**, never a hard admission gate.

Global Peak: **report only**, never an MPC constraint.

Final conclusions: **authoritative SWMM only**.

## 2. Frozen Wuhan V8 lineage and priority nodes

For the audited Wuhan V8 lineage:

- `FLOW_UNITS = CMS`;
- `FLOW_ROUTING = DYNWAVE`;
- 932 hydraulic nodes;
- 1,167 conduits;
- 3,731 subcatchments;
- 57 pumps + 42 orifices + 10 weirs = 109 writable SWMM actuator links;
- source `ROUTING_STEP = 15 s`;
- source `RULE_STEP = 10 s`.

`data/priority_nodes.txt` is frozen exactly as:

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

These are the recovered waterlogging-matched PFV_CORE8 for the exact 932-node/109-actuator physical lineage. Current preflight remains fail-closed: all eight must still exist in the actual frozen INP/graph used for the new study.

## 3. No-control is No-supervisory-RTC

Formal `no_control` uses contract `NO_SUPERVISORY_RTC_V2`:

- remove executable user-defined `[CONTROLS]`;
- make no Python setting writes;
- preserve physical network and rainfall/runoff forcing;
- preserve pump curves and initial pump status;
- preserve intrinsic `[PUMPS]` Startup/Shutoff depth logic;
- preserve regulator physical/default behaviour.

Therefore No-control is **not** All-open or All-closed. Deleting `[CONTROLS]` removes the supervisory RTC layer; it does not erase local equipment behaviour encoded directly in pump properties.

After the Fresh Workspace exists, run:

```powershell
rtc-inp-audit-v2 `
  --inp <FROZEN_INP> `
  --priority data/priority_nodes.txt `
  --out <FRESH_ROOT>/preflight/inp_audit.json
```

The audit records remaining intrinsic pump Startup/Shutoff logic explicitly.

## 4. Formal strategy matrix

Exactly five strategies are admitted:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

- `proposed`: TFV-first sparse-state + differentiable MPC on the controls-disabled base.
- `no_control`: No-supervisory-RTC; no Python writes.
- `internal_rtc`: original native `[CONTROLS]`; no Python writes.
- `all_open`: controls-disabled common prefix, then command all eligible SWMM settings to `1.0` from the common first-control epoch.
- `all_closed`: same prefix, then command settings to `0.0`.

`hold` is debug-only and is excluded from the public Formal strategy CLI because, on a controls-disabled base, freezing the first readback can collapse to an effective No-control duplicate.

The frozen plan is `configs/formal_baseline_plan.v3.json`, contract `FORMAL_BASELINE_PLAN_V4_NO_DUPLICATE_HOLD`.

## 5. Rainfall/event design

For this 109-actuator study, the Formal minimum is **160 independent rainfall groups**. This is a conservative project design, not a universal hydrological constant.

At exactly 160 groups:

| Role | Independent groups |
|---|---:|
| Development | 96 |
| Calibration | 24 |
| Safety audit | 16 |
| Untouched Final | 24 |

Development is group-disjoint again into approximately 77 train and 19 validation groups at the minimum design.

`rainfall_group` is the leakage unit: variants sharing one forcing group may not cross scientific roles or development folds.

Prepare the new event registry **outside** the future output root. Recommended columns are:

```text
event_id
rainfall_group
inp_path
scientific_split
development_fold
total_depth_mm                 # if available
duration_minutes               # if available
peak_intensity_mmhr            # if available
antecedent_rainfall_mm         # if available
```

## 6. Fresh Workspace — no historical RTC output reuse

Initialize a new empty root:

```powershell
rtc-init-fresh-workspace `
  --root E:\RTC_sewer\RTC_fresh_v05 `
  --inp <FROZEN_PHYSICAL_INP> `
  --priority data/priority_nodes.txt `
  --events <NEW_EVENT_REGISTRY_WITH_SPLITS.csv>
```

The initializer validates the >=160-group design **before creating the root**, then copies the registry to:

```text
<FRESH_ROOT>/contracts/event_registry_with_splits.csv
```

and creates:

```text
<FRESH_ROOT>/FRESH_WORKSPACE_MANIFEST.json
```

For a standalone rainfall-design evidence file, run afterward:

```powershell
rtc-validate-rainfall-design `
  --events <FRESH_ROOT>/contracts/event_registry_with_splits.csv `
  --out <FRESH_ROOT>/contracts/rainfall_design_evidence.json
```

Formal Step1/Step2 public CLIs now require `--workspace-manifest`. They validate that the run index and every referenced branch/shard belong to the new root. A newly written model cannot therefore be trained silently from an old Project6 branch.

## 7. Three different clocks: 15 s, candidate 5 min, candidate 10 min

Do not confuse the clocks.

### Hydraulic clock

SWMM Dynamic-Wave routing remains at the INP routing scale, here 15 s.

`Simulation.step_advance(300)` only controls how often Python regains control; it does not turn the hydraulic routing step into 5 min.

### Candidate observation/model clock

`300 s = 5 min` is the current **candidate** Step1/Step2 sampling interval.

### Candidate supervisory control clock

`600 s = 10 min` is the current **candidate** MPC update interval.

If accepted, one control block spans two 5-min model intervals.

Neither 5 min nor 10 min is frozen merely because `REPORT_STEP/WET_STEP` happens to be 5 min.

## 8. Phase-0 must resolve the 5/10-min question before Formal data generation

A pilot sampled every 5 min cannot reveal whether an actuator/network response happened at 1–4 min. Therefore the Phase-0 D2 diagnostic grid must be **<=60 s** while SWMM continues its 15 s internal routing.

Phase-0 uses step-response metrics that are valid for a sustained setting change:

- readback separation lag;
- 10% response time `t10`;
- 50% response time `t50`;
- 90% response time `t90`;
- peak-effect time;
- whether the peak lies in the final 10% of the pilot horizon.

Do **not** use “90% of response area” as a response time under a sustained step input: the area necessarily grows with the chosen horizon and confounds system dynamics with experiment length.

### Phase-0 is a lower-level pilot, not the Formal baseline cache

The baseline cache requires resolved production timing, so it cannot logically be used to determine that timing. Use only development pilot groups:

```powershell
rtc-run-d0-batch `
  --events <DEVELOPMENT_PILOT_EVENTS.csv> `
  --strategy no_control `
  --out-dir <FRESH_ROOT>/phase0/d0 `
  --record-stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1

rtc-design-checkpoints `
  --run-index <FRESH_ROOT>/phase0/d0/D0_no_control_RUN_INDEX.csv `
  --out <FRESH_ROOT>/phase0/checkpoints.csv `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <PILOT_HISTORY_MINUTES>

rtc-design-probes `
  --inp <FROZEN_INP> `
  --checkpoints <FRESH_ROOT>/phase0/checkpoints.csv `
  --out <FRESH_ROOT>/phase0/probe_manifest.csv

rtc-run-probes `
  --manifest <FRESH_ROOT>/phase0/probe_manifest.csv `
  --out-dir <FRESH_ROOT>/phase0/d2 `
  --horizon-minutes <PILOT_HORIZON_MINUTES> `
  --stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1

rtc-phase0-timescale `
  --manifest <FRESH_ROOT>/phase0/probe_manifest.csv `
  --run-summary <FRESH_ROOT>/phase0/d2/RUN_SUMMARY.csv `
  --detail-out <FRESH_ROOT>/phase0/timescale_detail.csv `
  --summary-out <FRESH_ROOT>/phase0/timescale_summary.json `
  --max-sample-seconds 60
```

If >5% of active responses peak in the last 10% of the pilot horizon, the Phase-0 command fails and the horizon must be lengthened.

A sustained D2 step cannot identify recovery after releasing an action. Use D3/pulse-style sequences to study recovery/decay before freezing a long prediction horizon.

Only after Phase-0 and wall-clock benchmarking should model step, control update, history length, horizon and compute budget be frozen.

## 9. Correct t=0-inclusive causal timeline

The runtime records and observes the initial state at `t=0` before any supervisory write.

If Phase-0 accepts:

```text
model/observation step = 5 min
history_steps = 13
control update = 10 min
first control = 60 min
```

then the history is exactly:

```text
0, 5, 10, ..., 55, 60 min = 13 frames
```

so the first real MPC can occur at exactly 60 min rather than falling back for one extra control cycle.

At `t=60`:

1. read sparse hydraulics, realised rainfall, actuator target/current setting and flow;
2. append the current observation to causal history;
3. reconstruct current full state with Step1;
4. create causal rainfall scenarios;
5. optimise the Step2/MPC action sequence;
6. project the first move to the executable numerical/engineering contract;
7. write all actuator targets;
8. hold that 10-min control block while 5-min observations continue.

At `t=70`:

1. read current state again;
2. verify previous target/current readback;
3. reconstruct/reforecast/reoptimise;
4. write the next first move.

Then repeat at 80, 90, ... min.

`CausalTimingContract` fails closed if the control interval is not an integer multiple of the model step, the first control is off-grid, full history is unavailable, or the horizon covers less than one full control interval.

## 10. Real-time means wall-clock feasible

A simulation can wait for Python; a field controller cannot.

The resolved controller must freeze:

```text
decision_runtime_budget_seconds
```

The measured wall-clock interval includes Step1 reconstruction, rainfall forecast, Step2/MPC optimisation and first-move projection.

If a decision exceeds the budget, the stale action is rejected and the causal fallback is executed as:

```text
FALLBACK_COMPUTE_DEADLINE
```

After Proposed development runs:

```powershell
rtc-accept-runtime `
  --run-index <FRESH_ROOT>/development/RUN_INDEX.csv `
  --config <FRESH_ROOT>/contracts/controller_resolved.json `
  --out <FRESH_ROOT>/acceptance/runtime_acceptance.json
```

Policy Lock requires zero control-grid/first-decision violations, zero history/readback/runtime/deadline failures, complete runtime diagnostics and maximum measured decision runtime within the frozen budget. The budget itself must be shorter than the control interval.

## 11. Generate fixed baselines once after timing is frozen

After Phase-0 freezes timing, generate each fixed reference once per rainfall event:

```powershell
rtc-build-baseline-cache `
  --events <FRESH_ROOT>/contracts/event_registry_with_splits.csv `
  --config <FRESH_ROOT>/contracts/controller_resolved.json `
  --out-dir <FRESH_ROOT>/baseline_cache `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

Default fixed references:

```text
no_control
internal_rtc
all_open
all_closed
```

The cache is content-hash keyed. Unchanged evidence resumes; changing event INP, physical network, strategy or frozen timing intentionally invalidates the cache.

Key views:

- `BASELINE_CACHE_INDEX.csv`
- `NO_CONTROL_D0_INDEX.csv`
- `STEP1_BASELINE_INDEX.csv`
- `FINAL_BASELINE_RUN_INDEX.csv`

Later Steps read these files; they do not rerun the same baseline.

## 12. Step1 — current-state reconstruction

Use fresh development No-control/Internal trajectories and optional new development/train D1 exploration.

D1 is **Step1 coverage only**. It is not a D2/D3 checkpoint source because its current state contains prior exploration-action history.

Formal Step1 CLI example:

```powershell
rtc-train-step1-large `
  --workspace-manifest <FRESH_ROOT>/FRESH_WORKSPACE_MANIFEST.json `
  --run-index <FRESH_ROOT>/step1/train_run_index.csv `
  --graph <FRESH_ROOT>/formal_assets/graph_schema.npz `
  --sensors <SENSOR_FILE> `
  --history-steps <FROZEN_HISTORY_STEPS> `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --batch-size 4 --grad-accum 4 `
  --out <FRESH_ROOT>/models/step1.pt
```

Acceptance uses a rainfall-group-disjoint development validation index and the same `--workspace-manifest` guard.

## 13. D2 and D3 fresh action-effect data

Production D2/D3 checkpoints come only from fresh controls-disabled No-control prefixes:

```powershell
rtc-design-checkpoints `
  --run-index <FRESH_ROOT>/baseline_cache/NO_CONTROL_D0_INDEX.csv `
  --out <FRESH_ROOT>/checkpoints/checkpoint_settings.csv `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <FROZEN_HISTORY_READINESS>
```

D2 stores the checkpoint pre-action state/statistics, applies the candidate target setting, then stores future post-action states. The compiled supervision is therefore correctly aligned as:

```text
state_t + action_t + causal/exogenous rain_t  → state_(t+1)
```

D3 adds multi-actuator/multi-step interaction sequences.

Lower-level D0/D2/D3 pre-lock generators now fail if `scientific_split == final`.

Build one deduplicated Step2 index:

```powershell
rtc-build-step2-index `
  --d2-manifest <FRESH_ROOT>/d2/probe_manifest.csv `
  --d2-run-summary <FRESH_ROOT>/d2/RUN_SUMMARY.csv `
  --d3-run-summary <FRESH_ROOT>/d3/D3_RUN_SUMMARY.csv `
  --out <FRESH_ROOT>/step2/step2_run_index.csv
```

Repeated D2 center/base provenance is collapsed to one physically executed branch.

## 14. Step2 — one consistent physical TFV definition everywhere

Compile shards only from a fresh-workspace run index:

```powershell
rtc-compile-step2-shards `
  --workspace-manifest <FRESH_ROOT>/FRESH_WORKSPACE_MANIFEST.json `
  --run-index <FRESH_ROOT>/step2/step2_run_index.csv `
  --out-dir <FRESH_ROOT>/step2/train_shards `
  --development-fold train `
  --shard-size 128

rtc-train-step2-large `
  --workspace-manifest <FRESH_ROOT>/FRESH_WORKSPACE_MANIFEST.json `
  --manifest <FRESH_ROOT>/step2/train_shards/manifest.json `
  --graph <FRESH_ROOT>/formal_assets/graph_schema.npz `
  --batch-size 2 --grad-accum 4 `
  --out <FRESH_ROOT>/models/step2.pt
```

Step2 supervises hydraulic trajectory, actuator flow and exact cumulative node flooding volume.

Predicted cumulative flooding volume now uses the **same** operator in Step2 training, Step2 acceptance, MPC, gradient truth and ranking:

```text
trapezoidal integration of current flooding rate + future predicted flooding rates
```

A future-only right-endpoint rectangle is forbidden because it would train a different TFV objective from the one later optimised online.

## 15. Authoritative TFV and PFV

PySWMM `Node.flooding` is an instantaneous flooding **rate**. It is never TFV/PFV by itself.

For node `i` over exact interval `[t0,t1]`:

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(t1)
         - cumulative_SWMM_flooding_volume_i(t0)
```

Then:

```text
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight priority nodes
```

For a full event starting at the simulation origin, the start cumulative statistic is zero.

This is the authoritative engineering truth for training labels, held-out validation and Final reporting.

The surrogate is not allowed to query future cumulative SWMM truth online; it predicts future flooding rates and integrates them only for MPC prediction.

## 16. Global Peak

Formal Global Peak is:

```text
max_t [ sum_i max(flooding_rate_i(t), 0) ]
```

at synchronous routing time during frozen-decision replay.

It is not the sum of each node's individual historical peak and is not a control constraint.

## 17. Continuous SWMM settings versus field hardware

SWMM supports fractional settings for pump/orifice/weir control. For non-Type5 pumps, setting scales the flow obtained from the pump curve, so continuous-setting optimisation is a legitimate SWMM experiment.

This does **not** prove that all 109 physical Wuhan facilities support continuous remote actuation.

Physical deployment additionally requires verified per-facility SCADA/operability metadata:

- remote command availability;
- discrete versus continuous mode;
- VFD availability where continuous pump modulation is claimed;
- ramp/rate and minimum dwell;
- interlocks;
- command/readback latency;
- communications/watchdog behaviour;
- local fail-safe and manual override.

Unknown field constraints are not invented from the INP.

## 18. Formal acceptance and Policy Lock

Required order:

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

Current Policy Lock:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V3_CAUSAL_FRESH_DATA
```

It binds fresh-workspace provenance, physical network, priority/sensor mappings, >=160-group rainfall design, causal timing, models, acceptance evidence, runtime budget, No-control semantics and the exact non-duplicate strategy matrix.

## 19. Untouched Final

Only after Policy Lock:

1. generate/resume the four fixed Final references once;
2. run locked Proposed once per untouched Final event;
3. formalize each run;
4. replay routing-step Global Peak;
5. compile the complete five-strategy paired matrix.

```powershell
rtc-build-baseline-cache `
  --events <FRESH_ROOT>/contracts/event_registry_with_splits.csv `
  --out-dir <FRESH_ROOT>/final_baseline_cache `
  --stage final `
  --policy-lock <FRESH_ROOT>/policy_lock/policy_lock.json `
  --workers 16 `
  --swmm-threads-per-process 1
```

Final truth may not change models, thresholds, rainfall forecast, time scales, runtime budget, sensors or priority nodes.

## 20. Recommended clean execution order

```text
1  create a NEW >=160-group event registry outside the future root
2  initialize Fresh Workspace (rainfall design validated here)
3  run standalone rainfall-design evidence + INP preflight
4  create a development-only <=60 s D0/D2 Phase-0 pilot
5  audit readback/t10/t50/t90/peak and horizon censoring
6  use D3/pulse response where recovery after action release matters
7  benchmark real MPC wall-clock latency
8  freeze model step / control update / history / horizon / runtime budget
9  generate fixed pre-lock baselines once
10 generate fresh D1; train/accept Step1
11 design fresh No-control checkpoints
12 generate fresh production D2 + D3
13 build fresh Step2 index/shards; train/accept Step2
14 run exact-SWMM gradient + ranking gates
15 run Proposed development closed loop
16 pass real-time execution acceptance
17 Policy Lock
18 generate fixed Final baselines once
19 run untouched Proposed Final
20 routing-step Global Peak replay + Final compiler
```

Do not launch full 16-process production data generation before Phase-0 freezes the scientific time scales.

## 21. Compute/storage

For a workstation with 16 CPU workers and RTX 4060 8GB:

- SWMM generation: normally `--workers 16 --swmm-threads-per-process 1`;
- GPU training: separate phase, AMP + small micro-batches + gradient accumulation;
- successful `.rpt/.out` and raw row-wise CSV remain debug-only;
- Step1 windows are lazy;
- Step2 is sharded;
- compact SI arrays + exact node statistics + hashed manifests are canonical evidence.

Resuming is permitted only inside the bound Fresh Workspace.

## 22. Claim boundary

Passing all software, model, runtime and Final gates establishes a causal, reproducible RTC result **inside the frozen SWMM model and its simulated actuator-setting contract**.

A claim of direct Wuhan field deployment additionally requires verified telemetry reliability, physical actuator operability and SCADA safety/interlock metadata. Those properties are not inferred from the INP.

See `FORMAL_PIPELINE_LATEST.md` for the fail-closed evidence contract.
