# Wuhan RTC v0.6.2 — causal sparse sensing, differentiable hydraulics and TFV-first MPC

This repository implements the Formal Wuhan large-network urban-drainage RTC workflow:

```text
causal sparse hydraulic observations + realised rainfall <= t
+ actuator target/current-setting/flow readback <= t
                         ↓
              Step1 current-state reconstruction
                         ↓
                current full-network state
                         ↓
          Step2 differentiable hydraulic world model
                         ↓
         all-actuator continuous rolling MPC
                         ↓
     target write → hold → readback → next decision
```

The primary objective is **minimum cumulative system-wide TFV**. The eight verified priority sites are a **soft secondary/diagnostic** consideration only. Global Peak is report-only. Final performance claims come from authoritative SWMM.

## 1. Metric terminology

In this project:

- **TFV = Total Flood Volume**: cumulative SWMM flooding-volume change summed over every hydraulic node for the defined horizon/event.
- **PFV = Priority Flood Volume**: the same cumulative flooding-volume change summed only over the frozen eight priority nodes.
- **Global Peak**: maximum over routing time of the simultaneous network sum of positive instantaneous flooding rates.
- `Node.flooding` is an instantaneous flooding rate and is never itself TFV or PFV.
- node `peak_flooding_rate` is distinct from PFV.

For node `i` over `[t0,t1]`:

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(t1)
         - cumulative_SWMM_flooding_volume_i(t0)
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the eight priority nodes
```

PFV/depth may deteriorate if necessary to obtain a better TFV solution; they are not hard MPC admission constraints.

## 2. Frozen causal information boundary

At decision time `t`, online control may use only information available at or before `t`:

- sparse depth/head observations;
- realised rainfall history;
- actuator target/current-setting/flow readback;
- frozen graph/device features;
- rainfall scenarios forecast only from causal rainfall history.

Forbidden online inputs include event ID as a control signal, future realised rainfall/runoff, future SWMM state/flooding, future Internal-RTC trajectory, offline future labels and Final truth.

Every Formal Step1 trajectory is now explicitly **t=0 inclusive**. If model step is 5 min and history length is 13, the first complete causal history is exactly `0,5,...,60 min`.

## 3. Wuhan V8 physical/reporting contract

For the audited Wuhan V8 lineage:

- `FLOW_UNITS = CMS`, `FLOW_ROUTING = DYNWAVE`;
- 932 hydraulic nodes, 1,167 conduits, 3,731 subcatchments;
- 57 pumps + 42 orifices + 10 weirs = **109 writable SWMM actuator links**;
- source routing step = 15 s; native rule step = 10 s.

`data/priority_nodes.txt` is frozen as:

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

Preflight and Policy Lock fail if these eight IDs are absent from the actual frozen INP/graph.

## 4. No-control and comparator semantics

`no_control` is `NO_SUPERVISORY_RTC_V2`:

- remove executable user `[CONTROLS]`;
- make no Python actuator writes;
- preserve rainfall/runoff forcing and physical hydraulics;
- preserve pump curves/initial status;
- preserve intrinsic `[PUMPS]` Startup/Shutoff depth logic;
- preserve regulator physical/default behaviour.

`internal_rtc` keeps original native `[CONTROLS]` and receives no Python writes.

Formal comparison is exactly:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

`hold` is debug-only.

## 5. All-actuator continuous control

Production MPC does **not** use Engineering36, a fixed active subset, runtime Top-K masking or an artificial binary-pump assumption.

Every discovered writable actuator remains in the action schema. MPC optimises direct continuous settings and projects every future control block to `[0,1]` and, when engineering evidence defines it, to sequential setting-rate limits. Which facilities move is therefore an outcome of hydraulic optimisation.

D3 training/validation sequences now use the **same frozen `max_setting_delta_per_update` contract** as production MPC when that value is configured. Sampling sparsity (`change_probability`) remains data coverage only; it is not runtime facility selection.

## 6. Three clocks and D3 horizon

Never confuse:

```text
SWMM hydraulic routing step
Step1/Step2 model-observation step
supervisory MPC control-update step
```

Production time scales are frozen only after high-frequency Phase-0 response/readback analysis.

Example if Phase-0 accepts 5/10/120 min:

```text
model_step_seconds       = 300
control_update_seconds   = 600
history_steps            = 13
first control            = 60 min
Step2 horizon            = 24 model steps = 120 min
D3 horizon               = 12 supervisory control blocks
```

`rtc-design-d3` reads the frozen controller config and derives control-block count automatically. The public D3 runner rejects time-cadence or rate-feasibility drift.

## 7. Data roles and saved trajectory contract

### D0 / fixed baselines

D0 and fixed baseline trajectories store compact SI evidence including:

```text
elapsed_seconds
node_ids
state_si[..., 6]
  depth_m
  head_m
  flooding_m3s
  volume_m3
  total_inflow_m3s
  total_outflow_m3s
rainfall_mmhr
actuator_ids
target_setting
current_setting
actuator_flow_m3s
```

Formal D0 starts at `elapsed_seconds=0` and then follows the frozen constant stride.

### D1 — Step1 state-space coverage

D1 is development/train-only controlled exploration for Step1 coverage. It is never a D2/D3 counterfactual prefix.

### D2 — local same-prefix action-effect truth

At checkpoint `t`, D2:

```text
replay controls-disabled No-control prefix
→ compare complete 6-channel state + all actuator readbacks
  against the saved No-control checkpoint
→ only if exact-prefix verification passes, snapshot cumulative statistics
→ write candidate action u_t
→ run SWMM forward
→ record x_(t+1)...x_(t+H) and actuator flows
→ subtract exact cumulative SWMM node flooding statistics
```

Reference trajectory metadata/compact hashes and SWMM engine version are part of D2 generation lineage.

### D3 — joint multi-actuator/multi-step truth

D3 applies the same exact-prefix verification before sequence execution. It supplies interaction data and joint action-ranking evidence. Every discovered actuator remains eligible.

## 8. Step1 and Step2 training datasets

This is model-based MPC, not reinforcement learning; an RL `reward` column is not required.

Step1 learns:

```text
causal sparse depth/head history
+ observation mask
+ causal rainfall/actuator local context history
+ graph/static features
→ current full 6-channel hydraulic state
```

Step2 learns:

```text
x_t
+ continuous action sequence
+ causal/exogenous rainfall sequence
+ previous actuator flow + device physics
→ future full hydraulic trajectory
+ future actuator-flow trajectory
+ exact cumulative SWMM node flood-volume truth
```

The Step2 compiler preserves interval alignment:

```text
initial_state = x_t
settings[k], rainfall[k] govern interval t_k → t_(k+1)
target_states[k] = x_(k+1)
target_actuator_flows[k] = q_(k+1)
```

Mixed model steps, horizons, node/actuator ordering or SWMM engine versions fail closed.

## 9. SWMM engine lineage

v0.6.2 treats the SWMM engine as part of the learned hydraulic operator:

- one SWMM engine per Step1 training/validation set;
- one engine per Step2 D2/D3 shard set;
- Step1 and Step2 locked models must have the same engine lineage;
- Proposed closed-loop execution must use that same engine;
- Final main run and Global Peak replay must use the same engine;
- all paired Final strategies are compiled under one engine identity.

Do not upgrade PySWMM/SWMM halfway through one Formal experiment.

## 10. Flood-volume truth and predicted operator

Authoritative D2/D3/Final volume truth is SWMM cumulative node statistics.

Step2 training, validation, MPC, gradient validation and ranking use one consistent predicted-volume operator:

```text
trapezoidal integration of checkpoint/current flooding rate
+ future predicted flooding rates
```

The numerical integral is a model objective/supervision bridge; Final truth remains SWMM statistics.

## 11. Step2 physical interpretation

Step2 is **physics-informed**, not a mathematically exact replacement hydraulic solver. It predicts setting-dependent actuator flow from local states/device physics, injects managed flow with opposite signs at upstream/downstream nodes, then advances a graph hydraulic residual transition.

Formal validation reports trajectory/flow/TFV performance plus negative-depth, negative-flood-rate, negative-node-volume and non-finite prediction fractions.

## 12. Rainfall-group and event-forcing lineage

`rainfall_group` is the statistical independence/leakage unit. Hard rules:

- unique `event_id` rows;
- no rainfall group crosses scientific splits;
- development train/validation are rainfall-group-disjoint;
- Final remains untouched before Policy Lock;
- referenced event INPs exist.

About 160 independent rainfall groups is a paper-strength recommendation, not an execution gate.

v0.6.2 additionally binds each Formal event to a **scientific event hash** containing all INP scientific/event content except policy-only `[CONTROLS]` and runtime-only `THREADS`. Referenced external `FILE` bytes are also hashed, so changing an external rainfall file in place invalidates the event identity.

## 13. Complete untouched Final

The Final compiler now requires:

1. the run index contains **every and only** event marked `final` in the locked split registry;
2. every Final event has exactly the five frozen strategies;
3. each run's `rainfall_group` matches the locked registry;
4. each strategy uses the registry-matched scientific event forcing hash;
5. all comparison runs use one SWMM engine;
6. statistics are paired by independent rainfall group and groups are equally weighted.

This prevents accidental event substitution and selective omission of an unfavourable Final event.

## 14. Safe resume and v0.6.2 invalidation rule

File existence alone never authorises reuse. Generated branches use semantic scientific implementation identity + exact numerical input/config/action/reference hashes + generated-artifact hashes. Step1/Step2 save atomic epoch resume state including optimizer/scaler/RNG.

**Do not reuse v0.6.1-derived RTC trajectories, D2/D3 branches, Step1/Step2 models, acceptance evidence or Policy Lock for v0.6.2.** The v0.6.2 semantic implementation fingerprint intentionally invalidates them because t=0, exact-prefix, engine and event-forcing semantics changed.

Reusable inputs are the frozen physical/event INPs, verified priority/sensor definitions and rainfall registry when their own scientific identities remain valid.

## 15. Compute/storage contract

Recommended workstation execution:

```text
SWMM generation: up to 16 independent Python processes × normally 1 SWMM thread/process
GPU training: RTX 4060 8 GB, AMP, small micro-batch + gradient accumulation
```

Successful runs default to compact NPZ + statistics + metadata/decision evidence. Raw row-wise CSV and `.rpt/.out` are debug-only by default.

## 16. Policy Lock and execution document

Policy Lock outer contract remains:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

The semantic implementation fingerprint inside the lock is v0.6.2 and binds the strengthened data lineage.

Use only:

- `docs/LOCAL_RUNBOOK_V062.md` — canonical Codex/local full workflow;
- `docs/FINAL_DATA_INVENTORY_V062.md` — authoritative v0.6.2 data inventory;
- `FORMAL_PIPELINE_LATEST.md` — scientific evidence sequence.

Install/test the merged release with:

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

GitHub CI verifies code/contracts and unit-level pipeline interfaces. It does **not** establish that Proposed significantly reduces Wuhan TFV; that claim is admissible only after the actual frozen Wuhan model completes the untouched Final five-strategy authoritative SWMM evaluation.
