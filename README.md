# Wuhan RTC — causal fresh-data TFV-first framework

This repository implements a **sparse-sensing → full-state reconstruction → differentiable hydraulic world model → continuous receding-horizon MPC** workflow for the Wuhan large SWMM drainage model.

The current scientific contract is intentionally independent of the historical Project6 pipeline. Historical discussions were used only to recover verified physical/observation definitions. **All RTC-derived hydraulic trajectories, counterfactual branches, model checkpoints, development runs, acceptance evidence and Final results must be regenerated under the current code and written into a new fresh workspace.**

---

## 1. Frozen research objective

At each online decision time, the controller may use only:

- sparse hydraulic observations available up to the current time;
- realised rainfall available up to the current time;
- actuator target/current-setting readback and actuator flow available up to the current time;
- static frozen network/actuator information;
- a rainfall forecast derived only from causal observations.

It may **not** use event ID as a policy input, future realised rainfall/runoff, future SWMM states/flooding, Final truth or rainfall-specific precomputed control schedules.

The online chain is:

```text
causal observations
      |
      v
Step1 sparse-state reconstruction
      |
      v
current full-network state
      |
      + causal rainfall scenarios
      v
Step2 differentiable hydraulic world model
      |
      v
continuous TFV-first MPC over all writable SWMM actuators
      |
      v
engineering projection -> target write -> hold -> readback -> repeat
```

### Objective

1. **Primary:** minimise predicted cumulative system-wide Total Flood Volume (TFV).
2. **Secondary:** inside the TFV-near-optimal solution set, prefer smaller positive deterioration at the eight observed priority ponding locations.
3. PFV and priority depth are **not hard MPC gates**.
4. Global Peak is reported but is not an MPC constraint.
5. Final scientific conclusions come from authoritative SWMM, never from surrogate predictions alone.

---

## 2. Frozen Wuhan V8 physical contract

For the audited Wuhan V8 lineage:

- `FLOW_UNITS = CMS`;
- `FLOW_ROUTING = DYNWAVE`;
- 932 hydraulic nodes;
- 1,167 conduits;
- 3,731 subcatchments;
- 57 pumps + 42 orifices + 10 weirs = **109 writable SWMM actuator links**;
- SWMM routing step = **15 s** in the source model;
- source native control-rule step = 10 s.

The eight verified waterlogging-matched priority nodes are now frozen in `data/priority_nodes.txt`:

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

Formal preflight fails if any of these eight nodes is absent from the frozen graph/INP.

---

## 3. No-control means **No-supervisory-RTC**, not All-open or All-closed

`no_control` has the explicit contract `NO_SUPERVISORY_RTC_V2`:

- remove all executable user-defined `[CONTROLS]` rules;
- make **no Python actuator writes**;
- preserve the same network geometry, storage, rainfall/runoff forcing and hydraulic parameters;
- preserve pump curves and initial pump status;
- preserve intrinsic `[PUMPS]` Startup/Shutoff water-depth logic;
- preserve regulator physical/default behaviour.

This distinction is important. SWMM pump Startup/Shutoff depth logic is a pump property, not a `[CONTROLS]` rule. Removing supervisory control rules must not silently mutate local equipment/protection behaviour.

Run the preflight audit after the fresh workspace exists:

```powershell
rtc-inp-audit-v2 `
  --inp <FROZEN_EVENT_OR_PHYSICAL_INP> `
  --priority data/priority_nodes.txt `
  --out <FRESH_ROOT>/preflight/inp_audit.json
```

The audit reports both the removed supervisory-rule layer and any intrinsic pump startup/shutoff controls that remain.

---

## 4. Formal comparison strategies — no duplicate Hold baseline

The Formal matrix is exactly:

1. `proposed`
2. `no_control`
3. `internal_rtc`
4. `all_open`
5. `all_closed`

Definitions:

- **Proposed:** TFV-first sparse-state + differentiable continuous MPC on the controls-disabled base.
- **No-control:** No-supervisory-RTC contract above; no Python writes.
- **Internal-RTC:** original frozen event INP with native `[CONTROLS]`; no Python writes.
- **All-open:** controls-disabled common prefix, then command all eligible link settings to `1.0` from the common first-control epoch.
- **All-closed:** same prefix, then command settings to `0.0`.

`hold` remains available only as an internal/debug helper. It is excluded from the public production strategy CLI and from Formal comparison because freezing the first controls-disabled readback can collapse to an effective No-control duplicate.

All-open/All-closed are **extreme diagnostic comparators**, not claims that real hardware can physically force every device to those states against local protection. The runtime records requested, target and current settings separately.

The plan is frozen in `configs/formal_baseline_plan.v3.json` under contract `FORMAL_BASELINE_PLAN_V4_NO_DUPLICATE_HOLD`.

---

## 5. Generate every fixed baseline once, then reuse it

A fixed baseline must never be rerun merely because another Step needs the result.

```powershell
rtc-build-baseline-cache `
  --events <FRESH_ROOT>/contracts/event_registry_with_splits.csv `
  --config <FRESH_ROOT>/contracts/controller_resolved.json `
  --out-dir <FRESH_ROOT>/baseline_cache `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

The default fixed set is:

```text
no_control
internal_rtc
all_open
all_closed
```

The content-hash cache binds event-INP identity, physical-network hash, strategy, observation/model cadence, control cadence, record cadence, first-control epoch and SWMM thread contract. Unchanged runs resume instead of rerunning SWMM; changed scientific contracts invalidate the cache intentionally.

Important views:

```text
BASELINE_CACHE_INDEX.csv
NO_CONTROL_D0_INDEX.csv
STEP1_BASELINE_INDEX.csv
FINAL_BASELINE_RUN_INDEX.csv
```

- `NO_CONTROL_D0_INDEX.csv` is the canonical replayable source for D2/D3 checkpoint design.
- `STEP1_BASELINE_INDEX.csv` supplies fresh development No-control/Internal trajectories to Step1.
- `FINAL_BASELINE_RUN_INDEX.csv` is generated only after Policy Lock.

Final baseline generation is locked to the frozen Final groups and Policy Lock:

```powershell
rtc-build-baseline-cache `
  --events <FRESH_ROOT>/contracts/event_registry_with_splits.csv `
  --out-dir <FRESH_ROOT>/final_baseline_cache `
  --stage final `
  --policy-lock <FRESH_ROOT>/policy_lock/policy_lock.json `
  --workers 16 `
  --swmm-threads-per-process 1
```

---

## 6. Rainfall/event design and fresh-data-only workspace

Do not point the new workflow at historical Project6 output/model folders.

First prepare a **new** event registry outside the future output root. It must already contain whole-rainfall-group assignments to `development / calibration / safety_audit / final` and development `train / validation` folds.

For this large 109-actuator study, the current Formal design requires at least **160 independent rainfall groups**. This is a project-level conservative evidence design, not a universal hydrological constant.

At the minimum 160-group design:

| Role | Groups |
|---|---:|
| Development | 96 |
| Calibration | 24 |
| Safety audit | 16 |
| Untouched Final | 24 |

Development contains approximately 77 train and 19 validation groups at the minimum design. A rainfall group is the leakage unit; variants sharing the same rainfall forcing must stay in the same group.

Then initialize an empty output root:

```powershell
rtc-init-fresh-workspace `
  --root E:\RTC_sewer\RTC_fresh_v05 `
  --inp <FROZEN_PHYSICAL_INP> `
  --priority data/priority_nodes.txt `
  --events <NEW_EVENT_REGISTRY_WITH_SPLITS.csv>
```

`rtc-init-fresh-workspace` validates the >=160-group design **before creating the root**. If the registry is invalid, no half-initialized output directory is left behind. If valid, it:

- refuses a pre-existing non-empty output root;
- copies the registry into `<FRESH_ROOT>/contracts/event_registry_with_splits.csv`;
- binds hashes for the frozen INP, priority file and event registry;
- records the validated rainfall-design summary;
- creates `FRESH_WORKSPACE_MANIFEST.json`.

For a standalone citable rainfall-design evidence file, run **after initialization** on the canonical copied registry:

```powershell
rtc-validate-rainfall-design `
  --events <FRESH_ROOT>/contracts/event_registry_with_splits.csv `
  --out <FRESH_ROOT>/contracts/rainfall_design_evidence.json
```

If available, include total depth, duration, peak intensity and antecedent-rainfall descriptors. The validator records their distribution but does not invent unsupported IDF/climatology limits.

Policy Lock rejects key RTC-derived artifacts located outside the fresh root. Historical RTC hydraulic trajectories, baseline outcomes, D1/D2/D3 branches, Step1/Step2 checkpoints and old Formal evidence are not admissible.

The physical INP, verified observation metadata and rainfall forcing definitions are scientific **inputs**, not reusable RTC-derived outputs.

---

## 7. The 15 s / 5 min / 10 min time hierarchy

These are three different clocks and must not be confused.

### Hydraulic clock

The source SWMM Dynamic-Wave routing step remains **15 s**. Python callback/model/control cadences do not replace the SWMM internal routing step.

### Candidate production observation/model clock

`300 s = 5 min` is the current **candidate** Step1/Step2 observation/model interval.

### Candidate supervisory control clock

`600 s = 10 min` is the current **candidate** MPC update interval. If 5/10 min is accepted, one optimized control block spans two 5-min model intervals.

### Why 5/10 min is not frozen a priori

A Phase-0 experiment sampled only every 5 min cannot reveal whether a meaningful hydraulic response occurred at 1, 2, 3 or 4 min. Therefore Phase-0 D2 must use a higher-frequency diagnostic grid.

New Formal Phase-0 requirement:

```text
Python D2 diagnostic sampling <= 60 s
SWMM internal routing remains 15 s
```

The Phase-0 report evaluates:

- target/current-setting separation/readback lag;
- actuator-flow response onset, peak and 90% response-mass time;
- network total-flooding-rate response;
- network maximum-depth response.

Run:

```powershell
rtc-phase0-timescale `
  --manifest <FRESH_ROOT>/d2_phase0/probe_manifest.csv `
  --run-summary <FRESH_ROOT>/d2_phase0/runs/RUN_SUMMARY.csv `
  --detail-out <FRESH_ROOT>/phase0/timescale_detail.csv `
  --summary-out <FRESH_ROOT>/phase0/timescale_summary.json `
  --max-sample-seconds 60
```

Only after reviewing high-frequency hydraulic response, readback and real computation latency should the production model step, control update, history length and prediction horizon be frozen.

---

## 8. Exact causal timeline if Phase-0 accepts 5 min observation + 10 min control

The current controller explicitly includes the initial causal observation at `t=0` before any supervisory write.

With:

```text
model/observation step = 5 min
history_steps = 13
control update = 10 min
first control = 60 min
```

the history is exactly:

```text
t = 0, 5, 10, ..., 55, 60 min   -> 13 frames
```

The first real MPC decision can therefore occur at exactly **t=60 min**.

The subsequent chain is:

```text
t=60:
  observe current sparse hydraulics + realised rain + actuator readback/flow
  reconstruct current full state
  build causal rainfall forecast
  optimise future action sequence
  project first action to executable setting bounds/rate contract
  write target settings

60–70 min:
  SWMM continues its 15 s Dynamic-Wave routing
  the 10-min control block is held while 5-min observations continue

t=70:
  observe again
  verify previous target/current readback
  reconstruct updated state
  reforecast
  reoptimise
  execute next first move

then 80, 90, ... min
```

The fail-closed timing validator enforces:

- control update is an integer multiple of model step;
- first control lies on both model and control grids;
- full causal history exists before the first decision;
- horizon covers at least one full control interval.

---

## 9. Real-time means wall-clock feasible, not just simulation-time feasible

A SWMM simulation can wait indefinitely while Python optimises. A field controller cannot.

`ControllerConfig` therefore supports:

```text
decision_runtime_budget_seconds
```

The measured wall-clock interval includes Step1 reconstruction, rainfall forecast, Step2/MPC optimisation and first-move projection. If it exceeds the frozen budget, the stale MPC result is rejected and the causal fallback is executed as `FALLBACK_COMPUTE_DEADLINE`.

Before Policy Lock run:

```powershell
rtc-accept-runtime `
  --run-index <FRESH_ROOT>/development/RUN_INDEX.csv `
  --config <FRESH_ROOT>/contracts/controller_resolved.json `
  --out <FRESH_ROOT>/acceptance/runtime_acceptance.json
```

Policy Lock requires:

- first Proposed decision exactly at the frozen first-control epoch;
- every decision on the frozen control grid;
- t=0 initial observation present;
- zero history-warmup fallback;
- zero readback fallback;
- zero runtime-error fallback;
- zero compute-deadline fallback;
- maximum measured decision runtime <= frozen runtime budget;
- runtime budget strictly smaller than the control interval.

---

## 10. Step1 — sparse-sensing current-state reconstruction

Generate all Step1 data fresh.

Recommended sources:

1. fresh development No-control/Internal compact trajectories from `STEP1_BASELINE_INDEX.csv`;
2. optional **new** D1 controlled exploration trajectories from development/train groups only.

D1 is Step1 state-coverage data only. It is **not** a D2/D3 checkpoint source because its current state already contains previous exploration-action history.

Step1 receives only causal sparse depth/head history plus node-local realised rainfall and actuator readback/flow context. It reconstructs the **current** full-network state; it is not given future hydraulic labels online.

Train and validate on rainfall-group-disjoint development folds.

---

## 11. D2 — same-prefix actuator-response truth

D2 checkpoints are designed only from fresh No-control prefixes:

```powershell
rtc-design-checkpoints `
  --run-index <FRESH_ROOT>/baseline_cache/NO_CONTROL_D0_INDEX.csv `
  --out <FRESH_ROOT>/checkpoints/checkpoint_settings.csv `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <FROZEN_HISTORY_READINESS>
```

Every D2 candidate starts from the same replayable controls-disabled No-control prefix and changes only the designed current action. This is the authoritative basis for learning action effects and validating TFV gradients.

For Phase-0, use a diagnostic stride <=60 s. After time scales are frozen, generate production Step2 branches on the frozen model grid.

Each branch stores compact SI arrays and exact cumulative SWMM node-flooding-volume truth.

---

## 12. D3 — multi-actuator interaction sequences

D3 uses the same fresh replayable No-control checkpoint contract and provides multi-actuator/multi-step interaction data. No Engineering36 or fixed runtime Top-K subset is introduced: every writable SWMM actuator remains part of the frozen actuator schema.

After D2/D3 generation:

```powershell
rtc-build-step2-index `
  --d2-manifest <FRESH_ROOT>/d2/probe_manifest.csv `
  --d2-run-summary <FRESH_ROOT>/d2/runs/RUN_SUMMARY.csv `
  --d3-run-summary <FRESH_ROOT>/d3/RUN_SUMMARY.csv `
  --out <FRESH_ROOT>/step2/step2_run_index.csv
```

The builder collapses repeated D2 center/base provenance to one physically executed branch and rejects Final leakage.

---

## 13. TFV and PFV — authoritative engineering definitions

`Node.flooding` is an instantaneous flooding **rate**. It is never TFV or PFV by itself.

For node `i` over an exact interval `[t0,t1]`, authoritative flooding volume is obtained from SWMM cumulative node statistics:

```text
DeltaV_i = cumulative_flooding_volume_i(t1)
         - cumulative_flooding_volume_i(t0)
```

Then:

```text
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the verified eight priority nodes
```

For a full event beginning at the simulation origin, the start cumulative value is zero and the event-end cumulative statistic is used directly.

This is physically meaningful because it measures the integrated volume of water lost to surface flooding/ponding rather than a single sampled flow rate.

The surrogate/MPC cannot query future SWMM cumulative truth online. It therefore predicts future flooding-rate states and numerically integrates them with the shared current+future trapezoidal-volume operator. Exact SWMM cumulative volume remains the training/validation/Final truth.

Global Peak is a separate metric:

```text
max over routing time of the simultaneous network sum of positive node flooding rates
```

It is obtained from frozen-decision routing-step replay for Formal reporting.

---

## 14. Continuous SWMM settings versus field hardware

SWMM supports fractional settings for pump/orifice/weir control. For non-Type5 pumps a pump setting scales the flow obtained from the pump curve; orifice/weir settings likewise have defined fractional meanings in SWMM.

Therefore continuous optimisation is a legitimate **SWMM control experiment**.

However, field deployment requires a separate site-specific actuator-operability/SCADA contract. Before physical implementation, verify for every controlled facility:

- remote command availability;
- actual discrete/continuous operating mode;
- VFD availability for pumps where continuous speed/flow modulation is claimed;
- command/readback latency;
- minimum on/off time, ramp/rate limits and interlocks;
- fail-safe local control and manual override;
- communications/watchdog behaviour.

The code must not claim that all 109 physical Wuhan facilities are already proven continuously controllable merely because SWMM accepts fractional settings. Unknown field metadata is not invented.

---

## 15. Step2 and acceptance gates

Compile bounded-memory shards and train Step2 only from newly generated D2/D3 branches.

Formal acceptance must pass, in order:

1. Step1 held-out reconstruction acceptance;
2. Step2 held-out trajectory/action-effect acceptance;
3. authoritative D2 SWMM TFV finite-difference gradient acceptance;
4. authoritative candidate-ranking acceptance;
5. Proposed development closed-loop SWMM;
6. real-time execution acceptance.

Only then can Policy Lock be created.

---

## 16. Policy Lock and Final

The Policy Lock contract is:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V3_CAUSAL_FRESH_DATA
```

It binds:

- fresh workspace identity;
- physical-network hash;
- verified eight priority nodes;
- sensor layout;
- >=160-group rainfall design;
- causal timing;
- Step1/Step2 models and acceptance evidence;
- TFV gradient/ranking evidence;
- real-time runtime acceptance;
- No-supervisory-RTC semantics;
- exact non-duplicate strategy matrix;
- resolved controller/forecast/fallback configuration.

The pipeline ledger requires `runtime_timing_acceptance` before `policy_lock`.

After lock:

1. generate/resume each fixed Final baseline exactly once;
2. run Proposed exactly once per untouched Final event;
3. formalize each frozen run and replay routing-step Global Peak;
4. compile the complete paired strategy matrix with `rtc-compile-final-v4`.

Final truth must never feed back into model training, time-scale selection, threshold selection, rainfall forecast tuning or controller hyperparameters.

---

## 17. Recommended execution order from a clean machine

```text
0. install current repo and SWMM extra
1. freeze physical INP; verify the eight priority nodes and sensor metadata
2. create NEW >=160-group event registry with group-disjoint roles OUTSIDE the future output root
3. initialize EMPTY fresh workspace (this validates the rainfall design and copies the registry)
4. write standalone rainfall-design evidence from the canonical copied registry
5. run INP preflight into the fresh root
6. generate small high-frequency No-control/Internal + D2 Phase-0 pilot
7. estimate readback/flow/network hydraulic time scales
8. benchmark online controller computation
9. freeze model step / control update / history / horizon / runtime budget
10. generate all fixed pre-lock baselines once into baseline cache
11. generate fresh D1 and train/accept Step1
12. design fresh No-control checkpoints
13. generate fresh D2 + D3
14. build Step2 index/shards; train/accept Step2
15. run TFV gradient + ranking gates
16. run Proposed development closed loop
17. run real-time execution acceptance
18. Policy Lock
19. generate fixed Final baselines once
20. run untouched Proposed Final
21. routing-step Global Peak replay + Final compiler
```

Do not start full 16-process generation before Phase-0 has frozen the scientific time scales.

---

## 18. Compute/storage guidance

For the stated workstation target:

- SWMM generation: normally `--workers 16 --swmm-threads-per-process 1`;
- do not multiply 16 processes by the source INP's internal `THREADS=2` unless explicitly benchmarked;
- Step1: lazy trajectory windows, small GPU batch, AMP, gradient accumulation;
- Step2: bounded-size shards, AMP, small micro-batch;
- `.rpt/.out` and raw row-wise CSV remain debug-only by default;
- compact SI arrays + exact node statistics + hashed manifests are the canonical saved evidence.

The workflow is resumable by hashed contracts, but resumability is allowed only **inside the new fresh workspace**; it is not permission to import historical Project6 results.

---

## 19. Current boundary of the claim

The repository is designed to support a scientifically causal and SWMM-authoritative RTC study. Passing all software/model/Final gates establishes correctness **within the frozen SWMM model and the specified simulated actuator-setting contract**.

A claim of direct field deployment additionally requires Wuhan-specific telemetry reliability, actuator-operability and SCADA safety/interlock metadata. Those data are not inferred from the INP and must be verified separately before physical implementation.

For the detailed fail-closed evidence sequence, read `FORMAL_PIPELINE_LATEST.md`.
