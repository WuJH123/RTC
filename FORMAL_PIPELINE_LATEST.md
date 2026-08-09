# Formal Pipeline — Wuhan large-system TFV-first contract (latest)

This document supersedes every earlier V1/V2/V3 workflow for **new Formal evidence**.
The target is a large Dynamic-Wave SWMM network controlled online without rainfall-event
lookup, a fixed active actuator subset, or binary-pump assumptions.

## 1. Frozen scientific objective

1. **Primary objective:** minimise cumulative system-wide TFV over the causal prediction horizon, optionally risk-adjusted across causal rainfall scenarios.
2. **Eight observed ponding sites:** retain site-wise cumulative PFV and maximum depth as important diagnostics and a **soft secondary preference** among TFV-near-optimal controls. PFV is not a hard admission/veto condition.
3. **Hard runtime conditions only:** finite numerics, continuous setting bounds, engineering projection, target write/readback, frozen schema/lineage and strict causality.
4. **No-control and Internal-RTC are different policies:**
   - `no_control`: same physical network/forcing, native `[CONTROLS]` disabled, no Python writes;
   - `internal_rtc`: event INP with native `[CONTROLS]` enabled, no Python writes.
5. Proposed, Hold, D1, D2, D3, All-open and All-closed use a controls-disabled runtime. Hold/All-open/All-closed share the same controls-disabled pre-control prefix as Proposed until the frozen first-control epoch.

## 2. Mandatory preflight

```powershell
rtc-inp-audit-v2 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --out outputs/preflight/inp_audit.json
```

A supplied priority ID absent from the INP is a hard blocker. Never substitute a similar string. If the eight observed sites are available as coordinates:

```powershell
rtc-resolve-priority `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --points data/priority_points.csv `
  --out-csv outputs/preflight/priority_mapping.csv `
  --out-nodes data/priority_nodes.txt
```

Review mapping distances and duplicate mappings before Formal use.

## 3. Rainfall-group split

Assign whole rainfall groups to mutually exclusive:

`development / calibration / safety_audit / final`

and split development again by rainfall group into `train / validation`. Final remains untouched until Policy Lock.

Historical rainfall time series may be reused if forcing lineage is valid. Historical hydraulic outcomes are not automatically admitted.

## 4. Canonical fixed-baseline cache

Fixed baselines must be generated once per event/runtime contract and then reused by Step1, checkpoint design, development comparisons and Final.

For pre-lock development/calibration/safety events:

```powershell
rtc-build-baseline-cache `
  --events outputs/contracts/event_registry_with_splits.csv `
  --config <resolved_controller_runtime_config.json> `
  --out-dir outputs/baseline_cache `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

For an early Phase-0 pilot, `--strategies no_control,internal_rtc` is sufficient. After time scales/control timing are frozen, rerun with the complete fixed baseline set; cache keys decide whether an existing result is still valid.

The cache key binds the complete event-INP SHA, forcing-independent physical-network SHA, strategy, observation/model step, control update, record stride, first-control epoch, SWMM thread contract and cache version.

The runner validates executed evidence, not just requested strategy names:

- No-control: no executable native controls, no controller, empty decision log;
- Internal-RTC: native controls present, no Python controller, empty decision log;
- Hold: one frozen readback vector for all decisions;
- All-open: every eligible actuator exactly `1.0` after the common first-control epoch;
- All-closed: every eligible actuator exactly `0.0` after the common first-control epoch;
- all runtimes: physical-network hash must equal the source event.

Primary reusable outputs:

- `BASELINE_CACHE_INDEX.csv`;
- `NO_CONTROL_D0_INDEX.csv` — canonical D2/D3 checkpoint source;
- `STEP1_BASELINE_INDEX.csv` — development No-control/Internal trajectories;
- per-event compact SI trajectory, cumulative node statistics, decision log and cache sidecar.

`rtc-run-d0-batch` remains a compatibility/lower-level command; new Formal work should prefer the baseline cache.

## 5. Optional D1 controlled-state coverage for Step1 only

```powershell
rtc-run-d1-exploration ...
```

D1 is development/train only and exposes Step1 to controlled states. Every writable actuator remains eligible.

**D1 is not a D2/D3 checkpoint source.** Its state contains the effects of its preceding exploration-action history. A fresh No-control prefix at the same clock time is not the same hydraulic state.

## 6. Step1 — sparse-state reconstruction

Recommended current inputs:

- `outputs/baseline_cache/STEP1_BASELINE_INDEX.csv`;
- optionally newly generated development/train D1 compact trajectories;
- frozen graph and sensor layout.

```powershell
rtc-train-step1-large `
  --run-index outputs/step1/step1_run_index.csv `
  --graph outputs/formal_assets/graph_schema.npz `
  --sensors data/sensor_nodes.txt `
  --history-steps <FROZEN_HISTORY_STEPS> `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --batch-size 4 --grad-accum 4 `
  --out outputs/models/step1.pt
```

Step1 reads sparse depth/head history plus topology-local realised rainfall and actuator readback/flow context. Windows are sliced lazily from compact trajectories and are not duplicated on disk.

Historical Train1600/Project6 trajectories are optional auxiliary material only after explicit physical-hash/units/causality/schema admission; they are not required for Formal Step1.

## 7. Replayable D2/D3 checkpoint design

Only a controls-disabled, no-Python-write No-control prefix is replayable by the current fresh-branch design.

```powershell
rtc-design-checkpoints `
  --run-index outputs/baseline_cache/NO_CONTROL_D0_INDEX.csv `
  --out outputs/checkpoints/checkpoint_settings.csv `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <HISTORY_READINESS_MINUTES>
```

When the source is the baseline cache, checkpoint design reuses cached hydraulics but restores the **original event INP path** in the checkpoint manifest. D2/D3 therefore build one new controls-disabled runtime from the original source rather than copying a cached runtime INP again.

Final groups, D1 controlled trajectories and Internal-RTC trajectories are rejected.

## 8. D2 — same-checkpoint single-actuator truth

```powershell
rtc-design-probes ...
rtc-run-probes `
  --manifest outputs/d2/probe_manifest.csv `
  --out-dir outputs/d2/runs `
  --horizon-minutes <DEVELOPMENT_HORIZON> `
  --stride-seconds <FROZEN_MODEL_STEP> `
  --workers 16 `
  --swmm-threads-per-process 1
```

Every D2 branch saves:

- exact current SI state;
- causal node rainfall;
- continuous action setting;
- previous actuator flow;
- full H-step SI target trajectory;
- H-step actuator-flow target;
- exact cumulative node flooding volume from SWMM node statistics;
- action/checkpoint/event/rainfall lineage.

At actuator bounds, Formal TFV gradient truth uses valid one-sided finite differences instead of discarding OFF/fully-open facilities.

## 9. Empirical hydraulic time scale

```powershell
python -m rtc.phase0_timescale ...
```

Review response onset, peak-effect time and 90%-response-mass distributions. Freeze model step, control update, history and horizon only after development evidence. Do not automatically inherit Project6 5/10/120 minutes.

## 10. Frozen graph and schemas

Only after the eight observed sites are correctly mapped:

```powershell
python -m rtc.formal_assets_v3 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --sensors data/sensor_nodes.txt `
  --out-dir outputs/formal_assets
```

The graph stores SI node geometry, pump-curve/orifice/weir features and stable actuator order. Step2 also learns a locked actuator identity embedding. No Engineering36/fixed active subset is created.

## 11. D3 — multi-actuator interaction trajectories

```powershell
rtc-design-d3 ...
rtc-run-d3-batch ... --workers 16 --swmm-threads-per-process 1
```

D3 uses the same replayable controls-disabled checkpoint contract as D2 and stores compact settings/state/flow/rainfall trajectories plus exact cumulative node-flooding truth.

## 12. Build one lineage-safe Step2 run index

Do not manually merge D2 manifest rows and executed branches. The D2 center/base action may be repeated once per actuator probe while physically executed only once.

```powershell
rtc-build-step2-index `
  --d2-manifest outputs/d2/probe_manifest.csv `
  --d2-run-summary outputs/d2/runs/RUN_SUMMARY.csv `
  --d3-run-summary outputs/d3/RUN_SUMMARY.csv `
  --out outputs/step2/step2_run_index.csv
```

This compiler collapses repeated D2 base-action provenance, verifies rainfall/split/checkpoint invariants, rejects duplicated executed metadata and rejects Final branches from Step2 train/validation.

## 13. Step2 — differentiable hydraulic world model

```powershell
rtc-compile-step2-shards `
  --run-index outputs/step2/step2_run_index.csv `
  --out-dir outputs/step2/train_shards `
  --development-fold train `
  --shard-size 128

rtc-train-step2-large `
  --manifest outputs/step2/train_shards/manifest.json `
  --graph outputs/formal_assets/graph_schema.npz `
  --batch-size 2 --grad-accum 4 `
  --out outputs/models/step2.pt
```

Formal Step2 training supervises full state trajectories, actuator flow and exact cumulative SWMM node flooding volume. Validation uses independent development/validation rainfall groups.

Old Project6 action-effect datasets are not automatically admitted to current Step2 because their No-control/action subset/objective/data contracts differ and prior action-effect truth was not sufficient to authorize gradient MPC.

## 14. Gradient/ranking acceptance

Before Proposed enters Formal-eligible closed loop:

- verify held-out SWMM TFV finite-difference gradient sign/direction/magnitude;
- verify held-out candidate TFV ranking/top-1/regret;
- use exact SWMM cumulative volume as truth;
- use the same trapezoidal predicted-volume definition as Step2/MPC.

Priority-PFV gradient may be reported but is not a hard TFV-policy veto.

## 15. Development closed loop

Run Proposed only after Step1, Step2, gradient and ranking gates pass:

```powershell
rtc-run-policy --strategy proposed ...
```

Compare it against the **already cached** fixed baselines for the same event. Do not rerun No-control/Internal/Hold/All-open/All-closed inside each development analysis script.

## 16. TFV-first Policy Lock

Freeze physical assets, split registry, Step1/Step2 models, acceptance evidence, controller config, rainfall forecast, fallback policy and baseline plan:

```powershell
rtc-policy-lock-v5 `
  --ledger outputs/evidence/tfv_pipeline_ledger.json `
  --artifacts outputs/contracts/formal_policy_artifacts.json `
  --out outputs/policy_lock/policy_lock.json
```

PFV remains a soft/diagnostic priority metric, not a mandatory safety-pass gate.

## 17. Untouched Final

Only after Policy Lock generate fixed Final baselines:

```powershell
rtc-build-baseline-cache `
  --events outputs/contracts/event_registry_with_splits.csv `
  --out-dir outputs/final_baseline_cache `
  --stage final `
  --policy-lock outputs/policy_lock/policy_lock.json `
  --workers 16 `
  --swmm-threads-per-process 1
```

Final mode obtains the locked controller timing and baseline plan from Policy Lock, runs/resumes each fixed baseline once, and creates its Formal manifest plus exact routing-step Global Peak replay.

Then:

1. run Proposed once per untouched Final event using the locked policy;
2. formalize each Proposed run with `rtc-formalize-run`;
3. combine Proposed Formal rows with `outputs/final_baseline_cache/FINAL_BASELINE_RUN_INDEX.csv`;
4. compile with `rtc-compile-final-v4`.

Formal conclusions use:

- TFV: cumulative SWMM flooding volume over all nodes;
- PFV: cumulative SWMM flooding volume at the eight verified observed-site nodes;
- Global Peak: synchronous network flooding-rate peak from frozen-decision routing replay;
- priority maximum depth: site-wise diagnostic;
- paired, event-balanced comparisons against No-control and Internal-RTC separately.

Final truth may not be used for fitting, threshold selection, priority mapping, sensor selection, time-scale selection, uncertainty calibration or hyperparameter tuning.

## 18. Hardware/storage contract

For a 16-worker CPU + RTX 4060 8GB workstation:

- SWMM: independent processes, normally `--workers 16 --swmm-threads-per-process 1`;
- do not retain source-INP `THREADS=2` inside 16 concurrent processes;
- GPU training: separate phase, AMP + small micro-batches + gradient accumulation;
- successful `.out/.rpt` files are deleted by default;
- raw row-wise node/actuator CSV is debug-only;
- Step1 windows are lazy;
- Step2 is sharded;
- baseline/runtime/branch reuse is content-hash driven, never filename driven.

Compact hydraulic state remains:

`depth_m, head_m, flooding_m3s, volume_m3, total_inflow_m3s, total_outflow_m3s`.

`Node.flooding` is an instantaneous rate. Authoritative PFV/TFV always use cumulative SWMM node flooding volume over the exact event/horizon.
