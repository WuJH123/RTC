# Formal Pipeline — Wuhan RTC v0.6.2

This is the fail-closed scientific evidence contract for new Formal RTC runs. It defines scientific correctness; paper-strength sample-size targets remain recommendations rather than software start-up gates.

## A. Scientific objective

Can a causal sparse-sensing, differentiable hydraulic world-model and all-actuator continuous receding-horizon controller reduce **system-wide cumulative Total Flood Volume (TFV)** in the Wuhan large sewer network?

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
all_open
all_closed
```

`hold` is debug-only.

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

Every Formal trajectory used by Step1 begins at `elapsed_seconds=0` and then follows the frozen model step exactly.

For a 300 s model step and 13 history frames:

```text
0,5,10,...,60 min
```

is the first complete history. A D0 trajectory beginning at 5 min is not v0.6.2 Formal evidence.

## F. Three clocks

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

## G. Data roles

### D0

Controls-disabled/reference full-event trajectories. Formal v0.6.2 D0 explicitly stores t=0.

### D1

Development/train-only controlled state-space exploration for Step1. D1 is never a D2/D3 counterfactual prefix.

### D2

Independent same-checkpoint single-actuator/local perturbations used for action-effect learning and finite-difference gradient evidence.

### D3

Joint multi-actuator, multi-control-block sequences used for interaction learning and joint action ranking/regret.

All 109 discovered writable actuators remain eligible; D3 sampling sparsity is data coverage only.

## H. Exact No-control prefix contract

D2/D3 checkpoint selection records the source No-control trajectory metadata path.

Before any candidate D2/D3 write, the replayed branch must match the saved No-control checkpoint in:

```text
exact checkpoint elapsed time
complete node ordering
complete actuator ordering
all 6 hydraulic state channels at every node
all actuator current settings/readback
SWMM engine version
```

under the frozen numerical tolerances.

Contract:

```text
EXACT_NO_CONTROL_PREFIX_REPLAY_V1
```

The reference metadata SHA-256, compact trajectory SHA-256 and engine version are part of generation lineage. A prefix mismatch aborts the branch before action.

## I. D3 engineering feasibility

D3 uses the same controller time contract and, when configured, the same sequential `max_setting_delta_per_update` as production MPC.

Every actuator remains eligible. D3 does not create a fixed active subset or runtime Top-K.

## J. Step1 data/model contract

Step1 maps:

```text
causal sparse depth/head history
+ masks
+ causal node-local rainfall/actuator context history
+ graph/static features
→ current full 6-channel hydraulic state
```

Formal Step1 requires:

```text
t=0 first frame
one frozen model step
locked node/actuator schema
rainfall-group-disjoint development train/validation
one SWMM engine lineage
```

Training resume additionally binds run index, graph, sensors, timing, architecture/training configuration and semantic implementation identity.

## K. Step2 interval alignment

For each D2/D3 branch:

```text
initial_state = x_t
settings[k] and rainfall[k] govern interval t_k → t_(k+1)
target_states[k] = x_(k+1)
target_actuator_flows[k] = q_(k+1)
```

The dataset also contains exact cumulative SWMM node flooding-volume truth for the full branch horizon.

A Step2 shard set has exactly one:

```text
model_step_seconds
horizon_steps
SWMM engine version
node ordering
actuator ordering
```

Mixed Phase-0/production grids, mixed horizons or mixed engines are forbidden.

## L. Step2 model and flood-volume operator

Step2 is physics-informed rather than a mathematically exact replacement hydraulic solver. It learns setting-dependent actuator flows and graph hydraulic evolution.

Formal training supervises:

```text
future hydraulic trajectory
future actuator-flow trajectory
exact cumulative SWMM node flooding volume
```

Step2 training, validation, MPC, gradient and ranking all use the same predicted-volume operator:

```text
trapezoidal integration of checkpoint/current flooding rate
+ future predicted flooding rates
```

Authoritative flood-volume truth remains SWMM cumulative statistics.

## M. SWMM engine lineage

The SWMM engine is part of the learned hydraulic operator.

Formal v0.6.2 requires:

```text
one engine within Step1 data
Step1 train/validation same engine
one engine within Step2 D2/D3 shards
Step2 train/validation same engine
Step1 and Step2 locked models same engine
Proposed runtime same engine as trained models
main Final run and Global Peak replay same engine
paired Final comparison uses one engine lineage
```

Do not upgrade PySWMM/SWMM during one Formal experiment.

## N. Rainfall-group split

Hard rules:

```text
unique event_id
no rainfall_group crosses scientific split
development has group-disjoint train and validation
Final absent from pre-lock tuning/training
referenced event INPs exist
```

About 160 independent rainfall groups is a recommended paper-strength design, not an execution gate.

Formal metrics give each independent rainfall group equal weight.

## O. Scientific event forcing identity

Physical-network identity alone is not sufficient for Final pairing.

v0.6.2 defines a `scientific_event_sha256` from all scientific/event INP content except:

```text
[CONTROLS]  — policy-only difference
THREADS     — execution-only difference
```

External `FILE` rainfall/time-series inputs are identified by their file bytes, not by path spelling. Relocated runtime INPs therefore remain the same scientific event while an in-place forcing-file edit changes event identity.

Runtime INP construction rewrites relative external `FILE` references to absolute source paths so relocating an INP cannot silently lose its forcing.

## P. TFV/PFV truth

For node `i` over `[t0,t1]`:

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(t1)
         - cumulative_SWMM_flooding_volume_i(t0)
```

Then:

```text
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight priority nodes
```

Instantaneous `Node.flooding` is a flow rate and is not TFV/PFV.

## Q. Global Peak

```text
Global Peak = max_t sum_i max(flooding_rate_i(t), 0)
```

Formal Global Peak comes from routing-step replay of the frozen executed decision schedule. The replay preserves original Python target-write cadence and must use the same SWMM engine as the main run.

## R. Acceptance sequence

Before Policy Lock:

1. INP/priority preflight;
2. leakage-safe rainfall split;
3. Phase-0 time-scale evidence;
4. D0/D1 coverage;
5. exact-prefix D2/D3 generation;
6. Step1 held-out group-balanced acceptance;
7. Step2 held-out trajectory/exact-TFV acceptance;
8. D2 local/boundary gradient acceptance;
9. D2 + D3 joint ranking/regret acceptance;
10. Proposed development closed-loop SWMM;
11. runtime/readback/deadline acceptance.

## S. Policy Lock

Outer contract remains:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

The internal semantic implementation fingerprint is v0.6.2 and binds t=0, exact-prefix, SWMM-engine, event-forcing and complete-Final semantics plus exact numerical artifact hashes.

## T. Complete untouched Final

The Final run index must contain **every and only** event marked `final` in the Policy-Locked split registry.

For each event it must contain exactly:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

Each run must match:

```text
locked event_id
locked rainfall_group
locked scientific event forcing SHA-256
locked physical network
locked model/control cadence
current v0.6.2 implementation identity
one SWMM engine lineage
```

This prevents selective omission of an unfavourable Final event and prevents a different rainfall forcing from being mislabeled as the same event.

Final aggregation first pairs/collapses within independent rainfall group and then gives each group equal weight.

## U. Safe resume and invalidation

Reuse is authorized by:

```text
compatible semantic scientific implementation
+ exact numerical input/config/reference/action lineage
+ generated artifact hashes
```

File existence alone is never enough.

v0.6.1-derived RTC trajectories, D2/D3 branches, Step1/Step2 models, acceptance evidence and Policy Locks are **not** Formal-compatible with v0.6.2 because scientific data semantics changed.

Use `docs/LOCAL_RUNBOOK_V062.md` for the only canonical full execution order.
