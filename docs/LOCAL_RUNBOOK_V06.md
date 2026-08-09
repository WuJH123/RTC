# Local runbook — RTC v0.6 final code-bound workflow

Use this sequence after the final PR is merged to `main`. Do not start large data generation from an unmerged development branch.

## 0. Freeze the source tree first

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

Once the first RTC-derived data is generated, do **not** edit `src/rtc` during the study. v0.6 intentionally binds data/model resume keys to the entire RTC Python source tree. A source change invalidates stale generated evidence by design.

Recommended local environment:

```text
SWMM: up to 16 independent Python processes, normally 1 SWMM thread/process
GPU: RTX 4060 8 GB, AMP on, small micro-batches + gradient accumulation
```

## 1. Prepare a new event registry

Build a new CSV outside the future Fresh Workspace containing at least 160 independent rainfall groups and these columns:

```text
event_id
rainfall_group
inp_path
scientific_split
development_fold
```

Recommended descriptors when available:

```text
total_depth_mm
duration_minutes
peak_intensity_mmhr
antecedent_rainfall_mm
```

Use whole rainfall groups for splitting. Final groups remain untouched until Policy Lock.

## 2. Initialize the Fresh Workspace

Example:

```powershell
$Root     = "E:\RTC_sewer\RTC_fresh_v06"
$Inp      = "E:\RTC_sewer\data\wuhan_v8_storage_retrofit.inp"
$Events   = "E:\RTC_sewer\contracts\events_v06.csv"
$Priority = "E:\RTC_sewer\RTC\data\priority_nodes.txt"
$Sensors  = "E:\RTC_sewer\contracts\sensor_nodes.txt"

rtc-init-fresh-workspace `
  --root $Root `
  --inp $Inp `
  --priority $Priority `
  --events $Events

rtc-validate-rainfall-design `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out "$Root\contracts\rainfall_design_evidence.json"
```

The Fresh Workspace must start empty.

## 3. Preflight the physical model and priority mapping

```powershell
rtc-inp-audit-v2 `
  --inp $Inp `
  --priority $Priority `
  --out "$Root\preflight\inp_audit.json"
```

Do not continue unless:

- all eight priority nodes are present;
- physical census matches the intended Wuhan V8 lineage;
- No-control contract is `NO_SUPERVISORY_RTC_V2`;
- native supervisory controls are identified;
- intrinsic pump Startup/Shutoff logic is reported rather than erased.

## 4. Phase-0 — determine production time scales before large generation

Create a **development-only** pilot event CSV from the canonical registry. Do not include Final rows.

Generate high-frequency controls-disabled No-control trajectories:

```powershell
rtc-run-d0-batch `
  --events "$Root\contracts\phase0_development_events.csv" `
  --strategy no_control `
  --out-dir "$Root\phase0\d0" `
  --record-stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1
```

Design exact minute-aligned checkpoints:

```powershell
rtc-design-checkpoints `
  --run-index "$Root\phase0\d0\D0_no_control_RUN_INDEX.csv" `
  --out "$Root\phase0\checkpoints.csv" `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes 60
```

Design D2 setting probes:

```powershell
rtc-design-probes `
  --inp $Inp `
  --checkpoints "$Root\phase0\checkpoints.csv" `
  --out "$Root\phase0\probe_manifest.csv"
```

Run high-frequency D2:

```powershell
rtc-run-probes `
  --manifest "$Root\phase0\probe_manifest.csv" `
  --out-dir "$Root\phase0\d2" `
  --horizon-minutes 120 `
  --stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1
```

Analyse readback/t10/t50/t90/peak timing:

```powershell
rtc-phase0-timescale `
  --manifest "$Root\phase0\probe_manifest.csv" `
  --run-summary "$Root\phase0\d2\RUN_SUMMARY.csv" `
  --detail-out "$Root\phase0\timescale_detail.csv" `
  --summary-out "$Root\phase0\timescale_summary.json" `
  --max-sample-seconds 60
```

If the horizon is censored, extend it. If recovery after releasing a control action matters, generate D3 pulse/release sequences before freezing the production horizon.

Do not put 60 s Phase-0 branches into production Step2 shards.

## 5. Freeze the production timing/controller skeleton

After Phase-0, create resolved files inside the Fresh Workspace:

```text
contracts/time_scale_config.json
contracts/controller_resolved.json
contracts/rainfall_forecast_config.json
contracts/fallback_policy.json
```

The controller must satisfy:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control lies on model and control grids
full causal history exists before first MPC
horizon >= one complete control interval
decision_runtime_budget_seconds < control_update_seconds
```

If 5/10 min is accepted, a representative contract is:

```text
model_step_seconds       = 300
control_update_seconds   = 600
history_steps            = 13
control_start_minutes    = 60
record_stride_seconds    = 300
```

`horizon_steps` must come from Phase-0/validation rather than being assumed.

## 6. Build the frozen graph/formal static assets

Build the graph from the frozen physical INP and write graph/state/actuator schemas inside the Fresh Workspace using the repository's formal-asset utilities. The resulting canonical artifacts must include at least:

```text
formal_assets/graph_schema.npz
formal_assets/state_schema.json
formal_assets/actuator_catalog.json
```

All 109 writable SWMM links remain in the actuator schema. No Engineering36/Top-K subset is introduced.

## 7. Generate fixed pre-lock baselines exactly once

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --config "$Root\contracts\controller_resolved.json" `
  --out-dir "$Root\baseline_cache" `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

The safe-resume cache is code/input/time/hash bound. Subsequent Steps read the cache indexes instead of rerunning the same baseline.

## 8. Generate D1 development/train exploration for Step1

Create a CSV containing only development/train events, then:

```powershell
rtc-run-d1-batch `
  --events "$Root\contracts\development_train_events.csv" `
  --sensors $Sensors `
  --out-dir "$Root\d1" `
  --seed 42 `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --control-update-seconds <FROZEN_CONTROL_UPDATE> `
  --control-start-minutes <FROZEN_CONTROL_START> `
  --workers 16 `
  --swmm-threads-per-process 1
```

D1 resumes by event and cannot accept validation/calibration/safety-audit/Final rows.

Build Step1 train/validation run indexes from fresh baseline-cache trajectories plus D1 train trajectories as appropriate. Do not use D1 as a D2/D3 replay prefix.

## 9. Train Step1 with epoch resume

```powershell
rtc-train-step1-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step1\train_run_index.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --sensors $Sensors `
  --history-steps <FROZEN_HISTORY_STEPS> `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --epochs 30 `
  --batch-size 4 `
  --grad-accum 4 `
  --out "$Root\models\step1.pt"
```

If interrupted, rerun the exact same command. The default `step1.pt.trainstate.pt` resumes the next epoch only when data/config/code contracts still match.

Validate:

```powershell
rtc-accept-step1-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step1\validation_run_index.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --sensors $Sensors `
  --model "$Root\models\step1.pt" `
  --history-steps <FROZEN_HISTORY_STEPS> `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --priority $Priority `
  --out "$Root\acceptance\step1_metrics.json"

rtc-accept-gate `
  --metrics "$Root\acceptance\step1_metrics.json" `
  --contract "$Root\contracts\model_acceptance_contract.json" `
  --section step1 `
  --out "$Root\acceptance\step1_gate.json"
```

Step1 metrics are equal-weight per independent rainfall group.

## 10. Generate production D2/D3

Use the code-bound No-control baseline index:

```powershell
rtc-design-checkpoints `
  --run-index "$Root\baseline_cache\NO_CONTROL_D0_INDEX.csv" `
  --out "$Root\checkpoints\checkpoint_settings.csv" `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <HISTORY_READY_MINUTES>
```

Design and run production D2 on the frozen model step/horizon. Example for a 300 s model step and 120 min horizon:

```powershell
rtc-design-probes `
  --inp $Inp `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --out "$Root\d2\probe_manifest.csv"

rtc-run-probes `
  --manifest "$Root\d2\probe_manifest.csv" `
  --out-dir "$Root\d2" `
  --horizon-minutes 120 `
  --stride-seconds 300 `
  --workers 16 `
  --swmm-threads-per-process 1
```

Design D3 sequences from the same replayable checkpoints:

```powershell
rtc-design-d3 `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --out "$Root\d3\sequence_manifest.csv" `
  --horizon-steps <FROZEN_HORIZON_STEPS> `
  --sequences-per-checkpoint 12 `
  --max-active 6 `
  --max-delta 0.20 `
  --seed 42

rtc-run-d3-batch `
  --manifest "$Root\d3\sequence_manifest.csv" `
  --out-dir "$Root\d3" `
  --control-block-seconds <FROZEN_CONTROL_UPDATE> `
  --stride-seconds <FROZEN_MODEL_STEP> `
  --workers 16 `
  --swmm-threads-per-process 1
```

`max-active` here is a **data-exploration sequence-design parameter**, not a runtime MPC Top-K/fixed controlled set. Production MPC still optimizes all actuators.

## 11. Build the Step2 run index and time-locked shards

```powershell
rtc-build-step2-index `
  --d2-manifest "$Root\d2\probe_manifest.csv" `
  --d2-run-summary "$Root\d2\RUN_SUMMARY.csv" `
  --d3-run-summary "$Root\d3\D3_RUN_SUMMARY.csv" `
  --out "$Root\step2\step2_run_index.csv"
```

Train shards:

```powershell
rtc-compile-step2-shards `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step2\step2_run_index.csv" `
  --out-dir "$Root\step2\train_shards" `
  --development-fold train `
  --shard-size 128 `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --horizon-steps <FROZEN_HORIZON_STEPS>
```

Validation shards use the same command with `--development-fold validation` and a separate output directory.

## 12. Train/accept Step2 with epoch resume

```powershell
rtc-train-step2-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --manifest "$Root\step2\train_shards\manifest.json" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --epochs 30 `
  --batch-size 2 `
  --grad-accum 4 `
  --out "$Root\models\step2.pt"
```

Rerun the same command after interruption to resume from the code/data-bound train-state.

```powershell
rtc-accept-step2-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --manifest "$Root\step2\validation_shards\manifest.json" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --model "$Root\models\step2.pt" `
  --priority $Priority `
  --out "$Root\acceptance\step2_metrics.json"

rtc-accept-gate `
  --metrics "$Root\acceptance\step2_metrics.json" `
  --contract "$Root\contracts\model_acceptance_contract.json" `
  --section step2 `
  --out "$Root\acceptance\step2_gate.json"
```

## 13. Validate local gradient and joint action ordering

```powershell
rtc-formal-gradient-v2 `
  --manifest "$Root\d2\probe_manifest.csv" `
  --run-summary "$Root\d2\RUN_SUMMARY.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --step2 "$Root\models\step2.pt" `
  --detail-out "$Root\acceptance\gradient_detail.csv" `
  --metrics-out "$Root\acceptance\gradient_metrics.json"

rtc-accept-gate `
  --metrics "$Root\acceptance\gradient_metrics.json" `
  --contract "$Root\contracts\model_acceptance_contract.json" `
  --section gradient `
  --out "$Root\acceptance\gradient_gate.json"
```

Then validate the action space actually used by MPC:

```powershell
rtc-formal-ranking `
  --manifest "$Root\d2\probe_manifest.csv" `
  --run-summary "$Root\d2\RUN_SUMMARY.csv" `
  --d3-run-summary "$Root\d3\D3_RUN_SUMMARY.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --step2 "$Root\models\step2.pt" `
  --priority $Priority `
  --thresholds "$Root\contracts\model_acceptance_contract.json" `
  --out "$Root\acceptance\ranking.json"
```

Formal ranking must pass both D2 local and D3 joint-sequence thresholds.

## 14. Run Proposed development closed-loop and real-time gate

For each selected development event, call the public `rtc-run-policy --strategy proposed` with the frozen graph/Step1/Step2/controller/sensors/priority inputs and write outputs under `$Root\development`.

Create `development/RUN_INDEX.csv`, then:

```powershell
rtc-accept-runtime `
  --run-index "$Root\development\RUN_INDEX.csv" `
  --config "$Root\contracts\controller_resolved.json" `
  --out "$Root\acceptance\runtime_acceptance.json"
```

Do not Policy Lock if there is any history/readback/runtime/deadline fatal fallback or if maximum decision wall-clock time exceeds the frozen budget.

## 15. Create Policy Lock V4

Complete the TFV pipeline ledger and the artifact-map JSON, then:

```powershell
rtc-policy-lock-v5 `
  --ledger "$Root\policy_lock\pipeline_ledger.json" `
  --artifacts "$Root\policy_lock\artifacts.json" `
  --out "$Root\policy_lock\policy_lock.json"
```

Expected contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

After this point, do not change code, models, timing, thresholds, forecast, sensors, priority nodes or splits.

## 16. Generate fixed Final references once

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out-dir "$Root\final_baseline_cache" `
  --stage final `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --workers 16 `
  --swmm-threads-per-process 1
```

The current source tree must exactly equal the source tree recorded in Policy Lock.

## 17. Run Proposed Final and formalize every run

Run locked Proposed once for every untouched Final event using `rtc-run-policy` and no tuning.

For every Proposed/fixed reference run:

```powershell
rtc-formalize-run `
  --main-metadata <RUN_METADATA.json> `
  --strategy <STRATEGY> `
  --event-id <EVENT_ID> `
  --rainfall-group <RAINFALL_GROUP> `
  --out <FORMAL_MANIFEST.json>
```

The command replays routing-step Global Peak using isolated temporary SWMM engine files and deletes those files afterward.

## 18. Compile the paired Final result

Create a complete run index containing exactly one Formal manifest per event × strategy, then:

```powershell
rtc-compile-final-v4 `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --run-index "$Root\final\FINAL_RUN_INDEX.csv" `
  --out-dir "$Root\final"
```

The Final summary is equal-weight per independent rainfall group, not per raw event row.

## 19. Safe restart rule

After a crash, rerun the same command. Safe resume is automatic for code-bound data caches and epoch train-states. Do not manually copy/rename old outputs to satisfy a missing path.

If the source tree changes, expect previous RTC-derived artifacts to fail their code hash. That is deliberate: regenerate under the new final code rather than forcing reuse.

## 20. Success condition

Only claim the final framework is complete when all of the following are simultaneously true:

- priority/physical preflight passes;
- Phase-0 timing is not censored;
- Step1 group-balanced gate passes;
- Step2 group-balanced gate passes;
- D2 gradient gate passes;
- D3 joint-action ranking gate passes;
- Proposed development runtime gate passes;
- Policy Lock V4 is created;
- untouched Final contains the complete five-strategy matrix;
- all Final metrics are authoritative SWMM truth;
- no Final information was used for tuning.
