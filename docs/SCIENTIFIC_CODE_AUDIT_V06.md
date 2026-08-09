# Independent scientific/code audit — RTC v0.6

## Audit question

Does the repository actually implement a large-network, causal, sparse-sensing, differentiable-world-model, continuous rolling-MPC framework that can determine online which facilities are hydraulically useful, how much to operate them and when to change them, with cumulative system TFV as the primary objective?

## Overall conclusion after v0.6 corrections

The architecture is scientifically coherent **provided that the new v0.6 fail-closed contracts are used from a new Fresh Workspace and all model/data/acceptance gates pass**. The code no longer relies on Engineering36, a fixed controlled subset, a runtime Top-K mask, event ID, future realised SWMM truth, or artificial binary pump actions.

The pre-v0.6 implementation, however, still contained several logic/reuse problems serious enough to invalidate a Formal run if left uncorrected. They are documented below because they explain why old RTC-derived outputs are intentionally not reusable.

## P0-1 — unsafe cache/resume semantics

### Problem

File existence or a partial scientific key was insufficient to prove that a cached D0/D2/D3/baseline branch came from the current implementation. A source-code change could therefore leave apparently reusable hydraulic evidence that was generated under different semantics.

### Correction

v0.6 introduces source-tree-bound generation keys. Baseline sidecars and D0/D1/D2/D3 metadata bind code, event/runtime INP, action/sequence, timing and generated-artifact hashes. Resume occurs only when all declared evidence verifies.

### Scientific effect

A code change can no longer silently mix old and new hydraulic labels inside one trained model.

## P0-2 — Step2 temporal-discretisation mismatch

### Problem

The world model does not receive `dt` as an online input, so a 60 s transition and a 300 s transition are different learned operators. Earlier data plumbing could compile branches without a single frozen model-step contract, risking Phase-0 and production data being mixed.

### Correction

Every production Step2 branch/shard/model now carries one `model_step_seconds` and one `horizon_steps`. Mixed temporal grids fail during dataset compilation/sharding, model acceptance and production loading.

### Scientific effect

Step2 is now a well-defined discrete-time hydraulic transition model instead of an ambiguous mixture of different transition horizons.

## P0-3 — boundary actuator optimisation

### Problem

Using `sigmoid(logit(current_setting))` as the MPC parameterisation makes the derivative effectively vanish near exact 0 or 1. The Wuhan source model contains many pumps initially at OFF/0, so this can make an actuator that should be opened practically unoptimisable.

### Correction

v0.6 optimises direct continuous setting variables and projects them after every optimiser step to `[0,1]`. If a rate limit is frozen, every future control block is projected sequentially relative to the preceding block/current readback.

### Scientific effect

The MPC can move inward from both exact bounds and all future planned settings remain numerically executable. Online facility selection therefore truly remains available over the full actuator catalogue.

## P0-4 — future MPC feasibility

### Problem

A rate limit applied only to the first executed move leaves the surrogate objective free to evaluate physically impossible future setting jumps, which can bias the first move itself.

### Correction

The setting-rate constraint is now part of projected optimisation for every future control block, not merely post-processing of the first move.

## P0-5 — local D2 validation did not prove joint MPC ranking

### Problem

Single-actuator D2 perturbations are ideal for local action-effect/gradient truth, but online MPC searches a joint multi-actuator, multi-step space. Good D2 ranking alone cannot prove that Step2 ranks joint action sequences correctly.

### Correction

Formal candidate-ranking acceptance now requires both:

- D2 local/single-actuator ordering;
- D3 joint multi-actuator, multi-step sequence ordering/top-1/regret.

## P0-6 — training interruption could waste GPU work

### Problem

Only the final model checkpoint was canonical, so interruption could require training again from epoch 0.

### Correction

Step1 and Step2 now write atomic per-epoch train states containing model, optimiser, scaler/RNG state and completed epoch. Resume is accepted only for the exact same data/config/code experiment.

## P0-7 — rainfall-group statistical weighting

### Problem

Window/branch/event-row metrics can overweight a rainfall group merely because more windows, checkpoints, candidates or event variants were generated from it. This conflicts with `rainfall_group` being the independent evidence unit.

### Correction

Formal Step1, Step2, gradient/ranking and Final summaries now first aggregate within rainfall group and then give independent rainfall groups equal weight.

## P0-8 — checkpoint time truncation

### Problem

The branch replay API addresses checkpoints in integer minutes. A high-frequency Phase-0 state at 90 s must not silently become a 1 min replay prefix.

### Correction

Checkpoint design admits only exact minute-aligned states until the replay API itself is upgraded to second-level addressing.

## P0-9 — Global Peak replay isolation

### Problem

Routing-step replay without unique report/output files can cause collision or large residual SWMM engine files during parallel Final evaluation.

### Correction

Formal Global Peak replay uses run-specific temporary `.rpt/.out` files and removes them in `finally`.

## P0-10 — evidence provenance did not fully bind the implementation

### Problem

A model SHA or `passed=true` is insufficient if the raw metrics/model were generated under older code or a different discrete-time contract.

### Correction

Policy Lock V4 is code/time/data-bound. It verifies current source-tree hashes, model checkpoint contracts, Step1 history/model step, Step2 model step/horizon, raw metric contracts/aggregation, preregistered thresholds, runtime evidence and artifact hashes.

## Causal data flow after correction

### Online information at decision time t

Allowed:

```text
sparse depth/head history <= t
realised rainfall history <= t
actuator target/current readback <= t
actuator flow readback <= t
frozen graph/device features
causal rainfall forecast constructed from observed history
```

Forbidden:

```text
rainfall event ID as a policy feature
future realised rainfall/runoff
future SWMM hydraulic state/flooding
future Internal-RTC trajectory
offline future labels
Final truth
```

### Step1

Maps the causal sparse history to the **current** full hydraulic state. Its target contains six physical hydraulic state channels; validation focuses on held-out unobserved-node reconstruction and is rainfall-group balanced.

### Step2

Learns the frozen discrete-time transition:

```text
current full state
+ current/future actuator setting sequence
+ causal/exogenous rainfall sequence
+ actuator physics/readback flow
→ future hydraulic trajectory + actuator flow
```

Offline training uses realised future rainfall because it is an exogenous ground-truth driver needed to identify the physical transition. Online inference replaces it with causal forecast scenarios; future realised rainfall is never available to the controller.

### MPC

Optimises all actuator settings continuously. There is no fixed facility set. Facilities whose optimal setting remains equal/near the current setting are naturally inactive; hydraulically useful facilities receive a non-zero setting change. `active_actuator_count_1e4` is reporting only, not a Top-K gate.

Primary objective is risk-adjusted cumulative TFV. Priority-site deterioration is a soft secondary preference within the TFV-near-optimal region.

## TFV/PFV truth

Instantaneous `Node.flooding` is a flow rate, not flooding volume.

Authoritative branch/event truth is:

```text
DeltaV_i = SWMM cumulative flooding_volume_i(end)
         - SWMM cumulative flooding_volume_i(start)

TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight PFV_CORE8 nodes
```

The surrogate cannot query future SWMM cumulative statistics online. Its predicted volume is the shared trapezoidal integration of current + future predicted flooding rates.

## Baseline semantics

Formal matrix:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

No-control means no supervisory RTC: `[CONTROLS]` removed, no Python writes, intrinsic pump Startup/Shutoff/local physical behaviour preserved. It is neither All-open nor All-closed.

Fixed baselines are generated once and reused through a code-bound cache rather than recomputed by later stages.

## Remaining boundary between simulation research and field deployment

The repository now provides a coherent **SWMM-authoritative real-time-control experiment**. Direct field-deployment claims require additional Wuhan SCADA/equipment metadata that cannot be inferred safely from the INP:

- whether each facility is remotely writable;
- discrete versus continuous hardware mode;
- VFD availability for pumps;
- true facility-specific ramp/dwell/interlock limits;
- communication/readback latency and watchdog behaviour;
- local fail-safe/manual override rules;
- telemetry availability/reliability for actuator flow/readback channels.

Until those data are verified, the correct claim is simulation-based RTC feasibility under the frozen SWMM actuator-setting contract, not immediate physical deployment of all 109 facilities.

## Final admissibility rule

No Formal/Final claim should be made until the new v0.6 data are generated from scratch under the final merged source tree and all Step1, Step2, D2 gradient, D3 joint-ranking, runtime and Policy Lock gates pass. Old Project6/earlier-RTC outputs are intentionally invalid for the new evidence chain.
