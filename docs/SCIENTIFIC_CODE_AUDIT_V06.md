# Independent scientific/code audit — RTC v0.6

## Audit question

Does the repository implement a large-network, causal, sparse-sensing, differentiable-world-model, continuous rolling-MPC framework that can decide online which facilities are hydraulically useful, how much to operate them and when to change them, with cumulative system TFV as the primary objective?

## Final conclusion after v0.6 corrections

The architecture is scientifically coherent for a **SWMM-authoritative RTC experiment** provided that the v0.6 data/model/acceptance sequence is used and all Formal gates pass.

The final controller no longer relies on Engineering36, a fixed controlled subset, runtime Top-K masking, event ID, future realised SWMM truth or artificial binary pump actions.

The principal pre-v0.6 problems and their final corrections are recorded below.

## P0-1 — unsafe cache/resume semantics

**Problem:** file existence or a partial key could be mistaken for valid reusable hydraulic evidence.

**Correction:** D0/D1/D2/D3 and baseline reuse now require a compatible scientific implementation fingerprint, exact event/runtime/timing/action/config lineage and verified generated-artifact hashes. The implementation fingerprint represents stable scientific semantics rather than byte-hashing every Python file, so unrelated documentation/reporting edits do not invalidate expensive computation.

## P0-2 — Step2 temporal-discretisation ambiguity

**Problem:** the world model does not receive `dt` as an online input, so mixing 60 s and 300 s transitions creates an ill-defined learned operator.

**Correction:** every production Step2 branch/shard/model carries one `model_step_seconds` and one `horizon_steps`; mixed time grids are rejected during sharding, validation and production loading.

## P0-3 — boundary actuator optimisation

**Problem:** `sigmoid(logit(current_setting))` makes gradients nearly vanish near exact 0/1, potentially preventing an OFF pump from being activated by MPC.

**Correction:** v0.6 optimizes direct continuous settings and projects them after every optimizer step to `[0,1]`. Optional setting-rate constraints are projected sequentially across all future control blocks.

## P0-4 — future MPC feasibility

**Problem:** constraining only the first executed move allows the surrogate objective to exploit infeasible future setting jumps.

**Correction:** setting-rate feasibility is enforced throughout the optimized future sequence, not only during first-move post-processing.

## P0-5 — single-actuator validation did not prove joint MPC behavior

**Problem:** D2 local perturbations alone cannot demonstrate that Step2 ranks joint multi-actuator/multi-step sequences correctly.

**Correction:** Formal action-effect acceptance includes both D2 local/boundary gradient/ranking evidence and D3 joint-sequence ordering/top-1/regret evidence.

## P0-6 — interrupted training wasted GPU work

**Problem:** only final model checkpoints were canonical.

**Correction:** Step1 and Step2 save atomic per-epoch states containing model, optimizer, scaler/RNG state and completed epoch. Resume is accepted only for the same compatible data/time/model/training experiment.

## P0-7 — rainfall-group statistical overweighting

**Problem:** long events or groups with more windows/checkpoints/candidates could dominate metrics merely because more rows were generated.

**Correction:** Formal Step1, Step2, gradient/ranking and Final statistics aggregate within rainfall group first and then give independent rainfall groups equal weight.

## P0-8 — checkpoint-time truncation

**Problem:** the current replay interface addresses checkpoints in integer minutes, so a 90 s high-frequency state must not silently become a 1 min prefix.

**Correction:** checkpoint design admits only exact minute-aligned states until replay addressing is upgraded to seconds.

## P0-9 — Global Peak replay semantics

**Problem:** routing-step reporting could accidentally change the target-write cadence and therefore change the trajectory it was meant to measure.

**Correction:** Global Peak replay observes flooding every SWMM routing callback but reasserts actuator targets only on the original main Python callback grid. Run-specific temporary `.rpt/.out` files are removed in `finally`.

## P0-10 — evidence-schema/lineage gaps

**Problem:** `passed=true` or a model SHA alone is not enough evidence if raw metrics/models use incompatible scientific semantics or time contracts.

**Correction:** Policy Lock V4 validates stable implementation fingerprints, model checkpoint/time contracts, source metric contracts/aggregation, preregistered thresholds, runtime evidence and direct artifact hashes. The mandatory artifact set is limited to objects that define or prove the scientific experiment; unrelated repository files are not gates.

## Causal data flow

### Online information allowed at time t

```text
sparse depth/head history <= t
realised rainfall history <= t
actuator target/current readback <= t
actuator flow readback <= t
frozen graph/device features
causal rainfall forecast derived from observed history
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

Maps causal sparse history to the **current** full hydraulic state. Its trajectory grid must equal the frozen model step; held-out validation is rainfall-group balanced.

### Step2

Learns the fixed discrete-time transition:

```text
current full state
+ actuator setting sequence
+ exogenous rainfall sequence
+ actuator physics/readback flow
→ future hydraulic trajectory + actuator flow
```

Offline training may use realised future rainfall as an exogenous physical driver. Online inference replaces it with causal forecast scenarios; future realised rainfall is never available to the controller.

### MPC

Optimizes every writable actuator continuously. Facilities whose optimum stays near the current setting are naturally inactive; useful facilities receive a setting change. There is no fixed facility set. Diagnostic active-actuator counts do not create a Top-K gate.

Primary objective is risk-adjusted cumulative TFV. Priority-site deterioration is a soft secondary preference inside the TFV-near-optimal region.

## TFV/PFV truth

Instantaneous `Node.flooding` is a rate, not a volume.

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(end)
         - cumulative_SWMM_flooding_volume_i(start)
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight PFV_CORE8 nodes
```

The surrogate predicts volume using the shared current+future trapezoidal flooding-rate operator; it never replaces authoritative SWMM cumulative truth in Final reporting.

## Baseline semantics

Formal matrix:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

No-control removes supervisory `[CONTROLS]` and makes no Python writes while preserving intrinsic pump Startup/Shutoff/local physical behavior. Fixed baselines are generated once and safely resumed by scientific/numerical lineage rather than recomputed by every downstream stage.

## Rainfall study size

Leakage-free rainfall-group splitting is mandatory. About 160 independent rainfall groups remains the recommended large-study target, not a hard software gate; pilot and development studies may run with smaller valid cohorts.

## Simulation-to-field boundary

The repository supports a coherent SWMM-authoritative RTC experiment. Direct physical deployment still requires verified Wuhan facility/SCADA metadata including remotely writable status, discrete/continuous mode, VFD availability, true ramp/dwell/interlock limits, communication/readback latency, watchdog/fail-safe behavior and telemetry reliability.

Until those data exist, the defensible claim is simulation-based RTC feasibility under the frozen SWMM actuator-setting contract, not immediate field deployment of all 109 facilities.

## Final admissibility

No Formal/Final performance claim should be made until v0.6 RTC-derived evidence has passed physical/priority preflight, Phase-0 timing, Step1, Step2, D2 gradient, D2+D3 joint ranking, development real-time acceptance, Policy Lock V4 and untouched five-strategy Final SWMM evaluation.
