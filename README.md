# Wuhan RTC v0.6.1 — causal sparse sensing, differentiable hydraulics and TFV-first MPC

This repository implements a large-network urban-drainage real-time-control research framework:

```text
causal sparse hydraulic observations
+ realised rainfall history
+ actuator target/current-setting/flow readback
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

## 1. Metric terminology — do not mix volume and peak rate

In this project:

- **TFV = Total Flood Volume**: cumulative SWMM flooding-volume change summed over every hydraulic node for the defined horizon/event.
- **PFV = Priority Flood Volume**: the same cumulative flooding-volume change, but summed only over the frozen eight priority nodes.
- **Global Peak**: maximum over routing time of the simultaneous network sum of positive instantaneous flooding rates.
- `Node.flooding` is an instantaneous rate and is never itself TFV or PFV.
- SWMM `peak_flooding_rate` is a node-level peak-rate statistic and is distinct from PFV.

PFV is **not** “peak flood flow” in this repository and is not a hard MPC admission constraint.

For node `i` over `[t0,t1]`:

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(t1)
         - cumulative_SWMM_flooding_volume_i(t0)
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the eight priority nodes
```

## 2. Frozen causal information boundary

At decision time `t`, online control may use only information available at or before `t`:

- sparse depth/head observations;
- realised rainfall history;
- actuator target/current-setting/flow readback;
- frozen graph/device features;
- rainfall scenarios forecast only from causal rainfall history.

Forbidden online inputs include event ID as a control signal, future realised rainfall/runoff, future SWMM state/flooding, future Internal-RTC trajectory, offline future labels and Final truth.

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

Therefore No-control is neither All-open nor All-closed. `internal_rtc` keeps the original native `[CONTROLS]` and receives no Python setting writes.

Formal comparison is exactly:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

`hold` is debug-only because on a controls-disabled base it can collapse to a No-control-like policy.

## 5. All-actuator continuous control

Production MPC does **not** use Engineering36, a fixed facility subset, runtime Top-K masking or an artificial binary-pump assumption.

Every discovered writable actuator remains in the action schema. MPC optimises direct continuous settings and projects every future control block to `[0,1]` and, when frozen from engineering evidence, to sequential setting-rate limits. Which facilities move is therefore an **outcome of hydraulic optimisation**, not a preselected list.

SWMM fractional settings are a valid numerical experiment. Immediate field-deployment claims require separate facility-specific SCADA/VFD/discrete-continuous mode, ramp, dwell, interlock, communication/readback and fail-safe metadata.

## 6. Three clocks and the D3 horizon rule

Do not confuse:

```text
SWMM hydraulic routing step
Step1/Step2 model-observation step
supervisory MPC control-update step
```

Production time scales are frozen only after `<=60 s` Phase-0 response/readback analysis.

If Phase-0 accepts:

```text
model_step_seconds       = 300
control_update_seconds   = 600
history_steps            = 13
first control            = 60 min
model horizon            = 120 min
```

then:

```text
0,5,10,...,60 min = 13 causal history frames
Step2 horizon      = 24 model steps
one control block = 2 model steps
D3 horizon         = 12 supervisory control blocks
```

v0.6.1 fixes an earlier ambiguous D3 interface: `rtc-design-d3` now reads the frozen controller config and derives the control-block count automatically. A D3 manifest cannot be executed with a different model/control cadence through the public runner.

## 7. Data roles and causal alignment

### Fixed baseline cache

Generate fixed baselines once per event/timing/strategy contract with `rtc-build-baseline-cache`; later stages reuse them only when lineage/config/artifact hashes still verify.

### D1 — Step1 state-space coverage

D1 is development/train-only controlled exploration for Step1. It is never a D2/D3 replay prefix.

### D2 — local same-prefix action-effect truth

At checkpoint `t`, D2:

```text
record x_t and cumulative statistics
→ write one candidate u_t
→ run SWMM forward
→ record x_(t+1)...x_(t+H), actuator flows
→ subtract exact cumulative SWMM node flooding statistics
```

### D3 — joint multi-actuator/multi-step truth

D3 uses the same replayable No-control prefix and generates joint continuous setting sequences. Every actuator is eligible in each sequence. `change_probability` and `perturbation_std` control data coverage only.

The D3 design manifest explicitly stores both:

```text
model_horizon_steps
control_blocks
control_block_steps
model_step_seconds
control_update_seconds
```

so model steps cannot be confused with control blocks.

## 8. Step1 and Step2 are model-based, not reinforcement learning

This project does **not** require an RL-style `(state, action, next_state, reward)` table.

Step1 training uses causal sparse-history → current full-state pairs.

Step2 training uses the equivalent model-based transition supervision:

```text
x_t
+ action sequence u_t...
+ exogenous rainfall sequence
+ previous actuator flow/device physics
→ future hydraulic state trajectory
+ future actuator-flow trajectory
+ exact cumulative SWMM node flood-volume label
```

There is no learned “reward” column. MPC differentiates through Step2 at runtime and constructs the TFV-first objective directly from predicted hydraulic/flooding trajectories.

## 9. Step2 physical interpretation

Step2 is **physics-informed**, not a replacement hydraulic solver with mathematically exact mass conservation.

Its architecture first predicts setting-dependent actuator flow from upstream/downstream states and device physics, injects that managed flow with opposite signs at the upstream/downstream nodes, and then advances the graph hydraulic state with a learned residual transition.

Formal training supervises:

1. future hydraulic-state trajectory;
2. managed actuator-flow trajectory;
3. exact cumulative SWMM node flooding volume.

Formal validation additionally reports negative-depth, negative-flood-rate, negative-node-volume and non-finite prediction fractions. Final hydraulic and flooding truth remains SWMM.

## 10. One flood-volume operator across the learned pipeline

Step2 training, Step2 validation, MPC, gradient validation and action ranking all use:

```text
trapezoidal integration of checkpoint/current flooding rate
+ future predicted flooding rates
```

The authoritative label is still the exact cumulative SWMM node-statistics delta, not the numerical integral.

## 11. Rainfall groups

`rainfall_group` is the leakage/statistical independence unit.

Hard correctness requirements are:

- unique `event_id`;
- no rainfall group crosses scientific splits;
- development train/validation are rainfall-group-disjoint;
- Final remains untouched before Policy Lock;
- referenced event INPs exist.

About **160 independent rainfall groups** remains the recommended paper-strength target (e.g. 96 development / 24 calibration / 16 safety-audit / 24 Final), not a software start-up gate.

Formal metrics first aggregate within rainfall group and then give independent rainfall groups equal weight.

## 12. Safe resume and storage control

File existence alone is never enough for reuse. D0/D1/D2/D3 and fixed baselines use deterministic scientific/numerical generation keys plus artifact hashes. Step1/Step2 training saves atomic epoch state including model, optimiser, scaler and RNG state.

Successful runs default to compact NPZ + statistics + metadata/decision evidence. Raw row-wise CSV and SWMM `.rpt/.out` files are debug-only and are not retained by default, preventing avoidable disk growth.

The semantic implementation fingerprint is deliberately stable across unrelated documentation/reporting edits; numerical input/config/action/model/artifact hashes still bind the actual computation.

## 13. Compute contract

For the target workstation:

```text
SWMM generation: up to 16 independent Python processes × normally 1 SWMM thread/process
GPU training: RTX 4060 8 GB, AMP, small micro-batch + gradient accumulation
```

Do not interpret “16 CPU workers” as one SWMM simulation using 16 internal threads; independent-event/branch multiprocessing is the intended parallelism.

## 14. Required pre-Policy-Lock evidence

Before Policy Lock:

1. physical/priority INP preflight;
2. high-frequency Phase-0 timing evidence;
3. Step1 held-out rainfall-group-balanced reconstruction gate;
4. Step2 held-out trajectory/exact-TFV gate plus physical plausibility diagnostics;
5. D2 exact-SWMM local/boundary TFV-gradient gate;
6. D2 local + D3 joint-sequence ranking/regret gate;
7. Proposed development closed-loop SWMM;
8. real-time runtime/readback/deadline acceptance.

Current lock contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

Only after lock may untouched Final events be evaluated.

## 15. Supported execution documents

- `docs/LOCAL_RUNBOOK_V06.md` — exact local command order;
- `docs/FINAL_DATA_INVENTORY_V06.md` — generated/reusable data inventory;
- `docs/CHECKLIST_FINAL_AUDIT_V061.md` — checklist-by-checklist audit status;
- `FORMAL_PIPELINE_LATEST.md` — scientific evidence contract.

Install/test the merged release with:

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

The GitHub CI can verify code/contracts and unit-level pipeline interfaces. It cannot establish that Proposed significantly reduces Wuhan TFV: that scientific claim is admissible only after the actual frozen Wuhan model has completed the untouched Final five-strategy SWMM evaluation.
