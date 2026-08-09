# Final data inventory — Wuhan RTC v0.6.2

This inventory defines the authoritative data/evidence chain for the v0.6.2 sparse-sensing → differentiable-world-model → TFV-first MPC study.

## 1. Reusable scientific inputs

Reusable only while their own identities remain unchanged:

```text
frozen physical/event SWMM INPs
external rainfall/time-series FILE inputs referenced by those INPs
data/priority_nodes.txt
frozen sparse-sensor list
rainfall/event registry with scientific splits
graph/device information encoded by the frozen INP
```

The scientific event identity includes external `FILE` content hashes. Policy-only `[CONTROLS]` edits and execution-only `THREADS` changes do not define a different rainfall event.

## 2. v0.6.1 derived evidence is not reusable

Do not import the following into a v0.6.2 Formal workspace:

```text
v0.6.1 D0/D1 trajectories
v0.6.1 D2/D3 branches
v0.6.1 Step1/Step2 models or train states
v0.6.1 acceptance/gradient/ranking evidence
v0.6.1 development runtime acceptance
v0.6.1 Policy Lock
v0.6.1 Final evidence
```

The v0.6.2 semantic implementation fingerprint intentionally invalidates them.

## 3. Workspace-level evidence

```text
FRESH_WORKSPACE_MANIFEST.json
contracts/event_registry_with_splits.csv
contracts/rainfall_design_evidence.json
preflight/inp_audit.json
formal_assets/graph_schema.npz
formal_assets/state_schema.json
formal_assets/actuator_catalog.json
formal_assets/physical_contract.json
formal_assets/formal_asset_audit.json
```

Large generated data may live on another local volume; validity is lineage/hash based, not path based.

## 4. Phase-0

```text
phase0/d0/D0_no_control_RUN_INDEX.csv
phase0/d0/<event>/*.compact.npz
phase0/d0/<event>/*.node_statistics.csv.gz
phase0/d0/<event>/*.json
phase0/checkpoints.csv
phase0/probe_manifest.csv
phase0/d2/RUN_SUMMARY.csv
phase0/d2/*.compact.npz
phase0/d2/*.node_statistics.csv.gz
phase0/d2/*.json
phase0/timescale_detail.csv
phase0/timescale_summary.json
```

Formal D0 compact trajectories begin at `elapsed_seconds=0` and use one fixed sample stride.

## 5. Fixed pre-lock baseline cache

```text
baseline_cache/BASELINE_CACHE_INDEX.csv
baseline_cache/NO_CONTROL_D0_INDEX.csv
baseline_cache/STEP1_BASELINE_INDEX.csv
baseline_cache/<event>/<strategy>/*.compact.npz
baseline_cache/<event>/<strategy>/*.node_statistics.csv.gz
baseline_cache/<event>/<strategy>/*.decisions.jsonl
baseline_cache/<event>/<strategy>/*.json
baseline_cache/<event>/<strategy>/*.baseline_cache.json
```

Fixed strategies:

```text
no_control
internal_rtc
all_open
all_closed
```

`NO_CONTROL_D0_INDEX.csv` is the replay-reference source for production D2/D3.

## 6. Step1 data/model

Sources:

```text
baseline_cache/STEP1_BASELINE_INDEX.csv
d1/D1_RUN_INDEX.csv
step1/train_run_index.csv
step1/validation_run_index.csv
```

Every Formal Step1 source must:

- start at t=0;
- follow the frozen model step;
- have the graph node/actuator ordering;
- carry SWMM engine version;
- use one engine per training/validation contract.

Model/evidence:

```text
models/step1.pt
models/step1.pt.json
models/step1.pt.trainstate.pt
acceptance/step1_metrics.json
acceptance/step1_gate.json
```

## 7. Production D2

```text
checkpoints/checkpoint_settings.csv
d2/probe_manifest.csv
d2/RUN_SUMMARY.csv
d2/<branch>.compact.npz
d2/<branch>.node_statistics.csv.gz
d2/<branch>.json
```

Each Formal D2 branch binds:

```text
event/rainfall/split/checkpoint
candidate action SHA
source/runtime INP hashes
saved No-control reference metadata hash
saved No-control reference compact hash
SWMM engine version
model step + horizon
exact-prefix verification evidence
generated-artifact hashes
```

Before candidate write, the complete six-channel state and all actuator current settings must match the saved reference checkpoint within the frozen numerical tolerances.

## 8. Production D3

```text
d3/sequence_manifest.csv
d3/D3_RUN_SUMMARY.csv
d3/<branch>.compact.npz
d3/<branch>.node_statistics.csv.gz
d3/<branch>.json
```

D3 binds the same exact No-control prefix and engine lineage as D2. Its manifest also stores:

```text
model_horizon_steps
model_step_seconds
control_update_seconds
control_block_steps
control_blocks
max_setting_delta_per_update
sequence_rate_feasible
```

Every actuator is eligible. Sampling sparsity is not runtime Top-K.

## 9. Step2 index/shards/model

```text
step2/step2_run_index.csv
step2/train_shards/manifest.json
step2/train_shards/step2_*.npz
step2/validation_shards/manifest.json
step2/validation_shards/step2_*.npz
models/step2.pt
models/step2.pt.json
models/step2.pt.trainstate.pt
acceptance/step2_metrics.json
acceptance/step2_gate.json
```

Each shard set has exactly one:

```text
model_step_seconds
horizon_steps
SWMM engine version
node ordering
actuator ordering
```

The tensors are interval-aligned as `x_t + u_t + rainfall_t -> x_(t+1)` and include exact cumulative SWMM node flood-volume truth.

## 10. Action-effect acceptance

```text
acceptance/gradient_detail.csv
acceptance/gradient_metrics.json
acceptance/gradient_gate.json
acceptance/ranking.json
acceptance/ranking.detail.csv
```

D2 supplies local/boundary finite-difference truth. D3 supplies joint multi-actuator/multi-step ordering/regret evidence.

## 11. Proposed development runtime evidence

```text
development/<event>/*.json
development/<event>/*.compact.npz
development/<event>/*.node_statistics.csv.gz
development/<event>/*.decisions.jsonl
development/RUN_INDEX.csv
acceptance/runtime_acceptance.json
```

Proposed runtime must use the same SWMM engine lineage as both trained models.

## 12. Policy Lock

```text
policy_lock/pipeline_ledger.json
policy_lock/artifacts.json
policy_lock/policy_lock.json
```

Outer contract remains:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

The lock contains the v0.6.2 semantic implementation fingerprint plus model/engine/timing/data/acceptance identities.

## 13. Untouched Final

Fixed baseline cache:

```text
final_baseline_cache/FINAL_BASELINE_RUN_INDEX.csv
```

Every Final Formal run contains:

```text
main metadata + hash
node statistics + hash
decision log + hash
peak replay + hash
physical network hash
scientific event forcing hash
SWMM engine version
```

The scientific event hash binds event INP scientific contents and external forcing bytes while ignoring `[CONTROLS]` and `THREADS` differences.

Final index:

```text
final/FINAL_RUN_INDEX.csv
```

must contain every and only locked Final event and all five strategies per event.

Compiler outputs:

```text
final/formal_final_detail.csv
final/formal_final_group_detail.csv
final/formal_final_summary.csv
final/proposed_vs_no_control.csv
final/proposed_vs_internal_rtc.csv
final/proposed_vs_all_open.csv
final/proposed_vs_all_closed.csv
```

Final aggregation gives each independent rainfall group equal weight.

## 14. Metric truth

```text
TFV = sum exact cumulative SWMM flooding-volume delta over all nodes
PFV = sum exact cumulative flooding-volume delta over the eight frozen priority nodes
Global Peak = routing-step synchronous max of network positive flooding-rate sum
```

PFV/depth and Global Peak are reported but are not hard MPC gates.

## 15. Resume rule

Rerun the same command after interruption. Never copy/rename old outputs merely to force a skip.

Reuse requires the current v0.6.2 scientific implementation identity, exact numerical/config/reference/action lineage and verified generated-artifact hashes.
