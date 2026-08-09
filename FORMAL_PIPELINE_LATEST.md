# Formal Pipeline — Wuhan RTC v0.6.5

This is the current fail-closed scientific evidence contract for fresh Wuhan RTC runs. It defines scientific correctness; paper-strength sample-size targets remain recommendations rather than software start-up gates.

## A. Scientific objective

Can a causal sparse-sensing, differentiable hydraulic world model and all-actuator continuous receding-horizon controller reduce **system-wide cumulative Total Flood Volume (TFV)** in the Wuhan large sewer network?

- TFV: primary objective.
- PFV = **Priority Flood Volume** at the frozen eight priority nodes: soft secondary/diagnostic only.
- Priority depth: soft/diagnostic only.
- Global Peak flooding rate: report only.
- Final truth: authoritative SWMM only.

PFV is not peak flooding flow.

## B. Frozen priority set

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

All eight must exist in the frozen INP/graph.

## C. No-control and Formal strategy matrix

`NO_SUPERVISORY_RTC_V2` removes executable user `[CONTROLS]` and makes no Python actuator writes while preserving forcing, hydraulic network, pump curves/status, intrinsic pump Startup/Shutoff and regulator physics.

`internal_rtc` retains native `[CONTROLS]` and receives no Python writes.

Formal matrix exactly:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

`auto_rrc` / `Auto-RRC` canonicalize to `auto_rbc`. `hold` is debug-only.

Auto-RBC is causal automatically parameterized local rule control from current normalized actuator-adjacent depths. EFD is causal storage-aware Equal Filling Degree from current normalized storage levels. Neither may use future realised rainfall/state or event-specific outcome tuning.

## D. Causal information boundary

Allowed at decision time `t`:

```text
sparse depth/head observations <= t
realised rainfall <= t
actuator target/current-setting/flow readback <= t
static graph/device features
rainfall scenarios inferred only from causal rainfall history
```

Forbidden online:

```text
event ID as policy feature
future realised rainfall/runoff
future SWMM state/flooding
future Internal trajectory
Final/locked hydraulic truth
offline future labels presented online
```

## E. t=0-inclusive Step1 timing

Every trajectory used by Step1 begins at `elapsed_seconds=0` and follows the frozen model step exactly. For a 300 s model step and 13 history frames, the first complete causal history is `0,5,10,...,60 min`.

## F. Three clocks and Phase-0 timing freeze

Keep distinct:

```text
SWMM routing step
model/observation step
supervisory control-update step
```

Production time scales are frozen only after development-only high-frequency Phase-0 response/readback evidence.

Hard relationships:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control aligns to model/control grids
first control follows a complete causal history
horizon >= one complete control interval
D3 horizon contains whole supervisory control blocks
decision runtime budget < control_update_seconds
```

v0.6.5 makes the previously manual Phase-0 links executable:

```text
rtc-design-phase0-events
```

selects a small development-only rainfall cohort using forcing descriptors only when available; hydraulic outcomes are never used for cohort selection.

```text
rtc-freeze-phase0-timing
```

refuses a horizon-censored timing report and writes a SHA-bound timing-only resolved contract after explicit timing values pass `CausalTimingContract` validation.

## G. Data roles

### D0

Controls-disabled/reference full-event trajectories. Formal D0 is t=0 inclusive.

### D1

Development/train-only controlled state-space exploration for Step1. D1 is never a D2/D3 counterfactual prefix.

### D2

Independent same-checkpoint single-actuator/local perturbations used for action-effect learning and finite-difference gradient evidence. The efficient rotating probe budget changes sampling cost only; it does not reduce the online action space.

### D3

Joint multi-actuator, multi-control-block sequences used for interaction learning and joint action ranking/regret. Every discovered writable actuator remains eligible.

## H. Exact No-control prefix contract

D2/D3 checkpoint selection records the source No-control trajectory metadata path. Before any candidate write, replay must match the saved No-control checkpoint in:

```text
exact checkpoint elapsed time
complete node ordering
complete actuator ordering
all six hydraulic state channels at every node
all actuator current settings/readback
SWMM engine version
```

Contract:

```text
EXACT_NO_CONTROL_PREFIX_REPLAY_V1
```

A mismatch aborts the branch before action.

## I. D3 engineering feasibility

D3 uses the same frozen controller timing contract and, when configured, the same sequential `max_setting_delta_per_update` as production MPC. Sampling sparsity is data coverage only, not runtime Top-K.

## J. Step1 data/model contract

Step1 maps:

```text
causal sparse depth/head history
+ masks
+ causal node-local rainfall/actuator context history
+ graph/static features
-> current full six-channel hydraulic state
```

Step1 requires t=0, one model step, locked node/actuator schema, rainfall-group-disjoint development train/validation and one SWMM-engine lineage.

## K. Step2 interval alignment

For each D2/D3 branch:

```text
initial_state = x_t
settings[k] and rainfall[k] govern interval t_k -> t_(k+1)
target_states[k] = x_(k+1)
target_actuator_flows[k] = q_(k+1)
```

The dataset also contains exact cumulative SWMM node flooding-volume truth for the full branch horizon. A Step2 shard set has one model step, horizon, SWMM engine, node ordering and actuator ordering.

## L. Step2 model and flood-volume operator

Step2 is physics-informed rather than a mathematically exact replacement hydraulic solver. It learns setting-dependent actuator flows and graph hydraulic evolution.

Training supervises future hydraulic trajectory, actuator-flow trajectory and exact cumulative SWMM node flooding volume. Step2 training, validation, MPC, gradient and ranking use the same predicted-volume operator: trapezoidal integration of current plus future predicted flooding rate. Authoritative volume truth remains SWMM cumulative statistics.

## M. SWMM engine lineage

The SWMM engine is part of the learned hydraulic operator. Step1 train/validation, Step2 D2/D3/train/validation, Proposed runtime and paired Final evaluation must preserve compatible engine lineage. Do not upgrade PySWMM/SWMM during one Formal experiment.

## N. Rainfall-group split

Hard rules:

```text
unique event_id
no rainfall_group crosses scientific split
development has group-disjoint train and validation
Final absent from pre-lock tuning/training
referenced event INPs exist
```

About 160 independent rainfall groups is a recommended paper-strength design, not an execution gate. Formal metrics give each independent rainfall group equal weight.

Phase-0 event selection is development-only and forcing-only. By default use development/train so held-out development validation remains available for model acceptance.

## O. Scientific event forcing identity

Physical-network identity alone is not sufficient for Final pairing. Scientific event identity binds event INP content except policy-only `[CONTROLS]` and execution-only `THREADS`, plus referenced external forcing-file bytes.

## P. TFV/PFV truth

For node `i` over `[t0,t1]`:

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(t1)
         - cumulative_SWMM_flooding_volume_i(t0)
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight priority nodes
```

Instantaneous `Node.flooding` is a flow rate and is not TFV/PFV.

## Q. Global Peak

```text
Global Peak = max_t sum_i max(flooding_rate_i(t), 0)
```

Formal Global Peak comes from routing-step replay of the frozen executed decision schedule while preserving target-write cadence and SWMM-engine lineage.

## R. Acceptance sequence

Before Policy Lock:

1. INP/priority preflight;
2. leakage-safe rainfall split;
3. forcing-only Phase-0 cohort selection;
4. high-frequency No-control D0 + exact-prefix D2 timing/control-leverage evidence;
5. non-censored production timing freeze;
6. production D0/D1 coverage;
7. Step1 held-out group-balanced acceptance;
8. production D2/D3 generation;
9. Step2 held-out trajectory/exact-TFV acceptance;
10. D2 local/boundary gradient acceptance;
11. D2 + D3 joint ranking/regret acceptance;
12. Proposed development closed-loop SWMM against No-control/Internal/Auto-RBC/EFD;
13. runtime/readback/deadline acceptance.

The development objective is to obtain real time-varying facility control that is not systematically worse than meaningful rule/reference baselines before spending the untouched Final budget.

## S. Policy Lock

Outer contract remains:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

The v0.6.5 semantic implementation fingerprint binds current causal timing, exact-prefix, SWMM-engine, event-forcing, seven-strategy Final, Phase-0 cohort-selection and timing-freeze semantics plus exact numerical artifact hashes.

The production controller config remains:

```text
PRODUCTION_CONTROLLER_CONFIG_V4_TFV_FIRST
```

The timing-only Phase-0 output is not itself a production policy config; forecast, optimizer, objective-near-optimality and runtime/readback fields must be resolved before Policy Lock.

## T. Complete untouched Final

The Final run index must contain every and only event marked `final` in the Policy-Locked split registry. For each event it must contain exactly:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

Each run must match locked event/rainfall-group identity, forcing hash, physical network, controller cadence, implementation identity and SWMM engine lineage. Final aggregation first pairs/collapses within independent rainfall group and then gives each group equal weight.

## U. Safe resume and invalidation

Reuse is authorized by:

```text
compatible semantic scientific implementation
+ exact numerical input/config/reference/action lineage
+ generated artifact hashes
```

File existence alone is never enough. For the current fresh-study workflow use `docs/LEAN_FRESH_RUN_V065.md`.
