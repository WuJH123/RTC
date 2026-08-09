# Final data inventory — RTC v0.6

This inventory defines the authoritative inputs, generated data and evidence for the Wuhan sparse-sensing → differentiable-world-model → TFV-first MPC study.

## 1. Reusable scientific inputs

These are scientific inputs rather than historical RTC results:

1. frozen physical/event SWMM INPs;
2. `data/priority_nodes.txt` with the eight verified PFV_CORE8 nodes;
3. frozen sparse-sensor node list;
4. rainfall/event registry with `event_id`, `rainfall_group`, `inp_path`, `scientific_split`, `development_fold`;
5. static network/actuator information encoded by the frozen INP.

They are reusable while their exact scientific identities remain unchanged.

## 2. Study workspace

Required study-level artifacts:

```text
FRESH_WORKSPACE_MANIFEST.json
contracts/event_registry_with_splits.csv
contracts/rainfall_design_evidence.json
preflight/inp_audit.json
```

The workspace binds the canonical inputs/splits. Large generated trajectories/shards may live on another local disk. Scientific validity is determined by lineage, the stable implementation-contract fingerprint, numerical input/config keys and generated-artifact hashes — not by physical directory location.

The recommended paper-strength rainfall target is about 160 independent groups, but software correctness requires only leakage-free development train/validation plus untouched Final groups.

## 3. Phase-0 timing evidence

Development-only high-frequency evidence:

```text
phase0/d0/...                         controls-disabled No-control trajectories, <=60 s sampling
phase0/checkpoints.csv
phase0/probe_manifest.csv
phase0/d2/*.json/*.compact.npz
phase0/d2/RUN_SUMMARY.csv
phase0/timescale_detail.csv
phase0/timescale_summary.json
phase0/d3/...                         optional pulse/release recovery evidence
```

Phase-0 is used to freeze model step, control interval, history, horizon and runtime budget. If its time step differs from production, it is forbidden from production Step2 shards.

## 4. Fixed pre-lock baseline cache

After timing is frozen:

```text
baseline_cache/BASELINE_CACHE_INDEX.csv
baseline_cache/NO_CONTROL_D0_INDEX.csv
baseline_cache/STEP1_BASELINE_INDEX.csv
baseline_cache/<event>/<strategy>/*.json
baseline_cache/<event>/<strategy>/*.compact.npz
baseline_cache/<event>/<strategy>/*.node_statistics.csv.gz
baseline_cache/<event>/<strategy>/*.decisions.jsonl
baseline_cache/<event>/<strategy>/*.baseline_cache.json
```

Formal fixed strategies:

```text
no_control
internal_rtc
all_open
all_closed
```

Resume requires a matching strategy/event/physical/timing/config generation key plus verification of the cached artifact hashes. File existence alone is not sufficient.

## 5. Step1 data/model evidence

Sources:

```text
baseline_cache/STEP1_BASELINE_INDEX.csv
d1/D1_RUN_INDEX.csv                    optional development/train coverage
step1/train_run_index.csv
step1/validation_run_index.csv
```

D1 is Step1 coverage only and cannot be a D2/D3 prefix.

Model/evidence:

```text
models/step1.pt
models/step1.pt.json
models/step1.pt.trainstate.pt
acceptance/step1_metrics.json
acceptance/step1_gate.json
```

The train-state resumes only for the same run index, graph, sensors, history/model step, architecture/training configuration and compatible implementation contract.

Formal Step1 validation gives equal weight to independent rainfall groups.

## 6. Production D2 data

Inputs:

```text
baseline_cache/NO_CONTROL_D0_INDEX.csv
checkpoints/checkpoint_settings.csv
```

Outputs:

```text
d2/probe_manifest.csv
d2/RUN_SUMMARY.csv
d2/<branch>.json
d2/<branch>.compact.npz
d2/<branch>.node_statistics.csv.gz
```

Each branch binds event/rainfall/split, checkpoint, candidate action SHA, source/runtime INP hashes, horizon, sample step, implementation contract and generated-artifact hashes.

D2 supplies same-prefix local/boundary action-effect truth.

## 7. Production D3 data

```text
d3/sequence_manifest.csv
d3/D3_RUN_SUMMARY.csv
d3/<branch>.json
d3/<branch>.compact.npz
d3/<branch>.node_statistics.csv.gz
```

D3 supplies multi-actuator/multi-step interaction evidence used both for Step2 training and joint-action ranking validation.

`max-active` is a D3 exploration parameter only; it does not impose a runtime MPC Top-K.

## 8. Step2 index and time-locked shards

```text
step2/step2_run_index.csv
step2/train_shards/manifest.json
step2/train_shards/step2_*.npz
step2/validation_shards/manifest.json
step2/validation_shards/step2_*.npz
```

Every production shard set has one immutable:

```text
model_step_seconds
horizon_steps
```

Mixed time grids are rejected.

Model/evidence:

```text
models/step2.pt
models/step2.pt.json
models/step2.pt.trainstate.pt
acceptance/step2_metrics.json
acceptance/step2_gate.json
```

Step2 supervises future hydraulic trajectory, actuator flow and exact cumulative SWMM node flooding volume. Predicted volume uses the same current+future trapezoidal operator later used by MPC/validation/gradient/ranking.

## 9. Action-effect acceptance

```text
acceptance/gradient_detail.csv
acceptance/gradient_metrics.json
acceptance/gradient_gate.json
acceptance/ranking.json
acceptance/ranking.detail.csv
```

D2 validates local/boundary finite-difference TFV gradients. D2 + D3 ranking validates local and joint multi-actuator/multi-step action ordering and regret. Formal metrics are rainfall-group balanced.

## 10. Proposed development closed-loop evidence

```text
development/<event>/*.json
development/<event>/*.compact.npz
development/<event>/*.node_statistics.csv.gz
development/<event>/*.decisions.jsonl
development/RUN_INDEX.csv
acceptance/runtime_acceptance.json
```

Each Proposed run is bound to the resolved controller, graph and Step1/Step2 model identities. Runtime acceptance verifies the event clock, t=0 history, readback, fatal fallbacks and wall-clock decision budget.

## 11. Policy Lock

```text
policy_lock/policy_lock.json
```

Contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

It binds only artifacts that define the scientific experiment or prove acceptance: physical/input identities, canonical split, priority/sensors, controller/time contract, graph, model checkpoints/training lineage, acceptance evidence, runtime evidence and baseline plan.

The implementation identifier is a stable scientific-semantics fingerprint. Unrelated source/documentation changes are not intended to invalidate evidence; a scientific semantic change requires the implementation contract to be bumped.

## 12. Untouched Final evidence

Fixed references:

```text
final_baseline_cache/FINAL_BASELINE_RUN_INDEX.csv
```

Every Final Proposed/reference run receives:

```text
<run>.formal_manifest.json
<run>.formal_manifest.peak_replay.json
```

Final compiler outputs:

```text
final/formal_final_detail.csv
final/formal_final_group_detail.csv
final/formal_final_summary.csv
final/proposed_vs_no_control.csv
final/proposed_vs_internal_rtc.csv
final/proposed_vs_all_open.csv
final/proposed_vs_all_closed.csv
```

Final aggregation first collapses variants inside each rainfall group and then gives every independent rainfall group equal weight.

## 13. What must not be reused

Do not treat the following as valid v0.6 evidence:

- Project6 trajectories/models that do not satisfy current contracts;
- old baseline results without a current compatible cache sidecar;
- D1/D2/D3 branches without valid generation keys/artifact hashes;
- Step1/Step2 checkpoints without the current checkpoint/time/training lineage;
- Phase-0 data mixed into production when its time grid differs;
- pre-lock training/acceptance containing Final groups;
- old Policy Locks/Final evidence from incompatible scientific semantics.

## 14. Safe resume rule

A computation is reusable when its compatible scientific contract, exact numerical inputs/configuration and required artifact hashes verify. Rerunning an interrupted command should therefore resume matching work automatically; manually copying old files to force a skip is not valid reuse.
