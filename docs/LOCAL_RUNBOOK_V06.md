# Local runbook — RTC v0.6 final workflow

Use this sequence only after the final PR has been merged to `main`. The workflow is designed to be scientifically fail-closed where correctness requires it, while allowing safe resume of expensive SWMM generation and model training.

## 0. Install the frozen release

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

v0.6 uses a **stable scientific implementation fingerprint**, not a byte-for-byte hash of every source file. Expensive results are reused when the implementation semantics plus their exact numerical inputs/configuration and generated-artifact hashes still match.

Recommended workstation execution:

```text
SWMM generation: up to 16 independent Python processes, normally 1 SWMM thread/process
GPU training: RTX 4060 8 GB, AMP, small micro-batch + gradient accumulation
```

## 1. Prepare the event registry

Create a new CSV with:

```text
event_id
rainfall_group
inp_path
scientific_split
development_fold
```

Optional but recommended descriptors:

```text
total_depth_mm
duration_minutes
peak_intensity_mmhr
antecedent_rainfall_mm
```

Correctness requires whole rainfall groups to remain inside one scientific split and one development fold. Final groups remain untouched until Policy Lock.

For publication-strength evidence, target roughly 160 independent rainfall groups (for example 96 development / 24 calibration / 16 safety-audit / 24 Final). This is a **recommended study size, not a software execution gate**.

## 2. Initialize the study workspace

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

The root must be empty on first initialization. Large generated data may be placed on another local volume; scientific reuse is validated by lineage/config/artifact hashes, not by directory location.

## 3. Physical/priority preflight

```powershell
rtc-inp-audit-v2 `
  --inp $Inp `
  --priority $Priority `
  --out "$Root\preflight\inp_audit.json"
```

Continue only when the intended Wuhan physical census is confirmed, all eight priority nodes are present, and No-control is reported as `NO_SUPERVISORY_RTC_V2` with local pump Startup/Shutoff behavior preserved.

## 4. Phase-0: determine the time scales first

Create a **development-only** pilot CSV; never include Final rows.

```powershell
rtc-run-d0-batch `
  --events "$Root\contracts\phase0_development_events.csv" `
  --strategy no_control `
  --out-dir "$Root\phase0\d0" `
  --record-stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1

rtc-design-checkpoints `
  --run-index "$Root\phase0\d0\D0_no_control_RUN_INDEX.csv" `
  --out "$Root\phase0\checkpoints.csv" `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes 60

rtc-design-probes `
  --inp $Inp `
  --checkpoints "$Root\phase0\checkpoints.csv" `
  --out "$Root\phase0\probe_manifest.csv"

rtc-run-probes `
  --manifest "$Root\phase0\probe_manifest.csv" `
  --out-dir "$Root\phase0\d2" `
  --horizon-minutes 120 `
  --stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1

rtc-phase0-timescale `
  --manifest "$Root\phase0\probe_manifest.csv" `
  --run-summary "$Root\phase0\d2\RUN_SUMMARY.csv" `
  --detail-out "$Root\phase0\timescale_detail.csv" `
  --summary-out "$Root\phase0\timescale_summary.json" `
  --max-sample-seconds 60
```

Use readback lag, `t10/t50/t90`, peak timing and horizon censoring. If the response remains censored, lengthen the pilot horizon. Use D3/pulse-release sequences when recovery after releasing an action must be identified.

**Do not put 60 s Phase-0 branches into production Step2 shards.**

## 5. Freeze the production timing/controller contract

Write resolved files such as:

```text
$Root\contracts\time_scale_config.json
$Root\contracts\controller_resolved.json
$Root\contracts\model_acceptance_contract.json
```

Required timing relationships:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control lies on model and control grids
first control occurs only after a complete causal history
horizon >= one complete control interval
decision_runtime_budget_seconds < control_update_seconds
```

If Phase-0 supports 5/10 min, a representative contract is:

```text
model_step_seconds      = 300
control_update_seconds  = 600
history_steps           = 13
control_start_minutes   = 60
record_stride_seconds   = 300
```

`horizon_steps` must be selected from Phase-0/validation rather than assumed.

## 6. Build graph/static assets

Build the graph and static schemas from the frozen INP using the repository utilities. Canonical artifacts should include:

```text
$Root\formal_assets\graph_schema.npz
$Root\formal_assets\state_schema.json
$Root\formal_assets\actuator_catalog.json
```

The actuator schema must retain all 109 writable SWMM links. Do not introduce Engineering36 or a fixed runtime Top-K subset.

## 7. Generate fixed pre-lock references once

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --config "$Root\contracts\controller_resolved.json" `
  --out-dir "$Root\baseline_cache" `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

Formal fixed references are:

```text
no_control
internal_rtc
all_open
all_closed
```

The cache safely resumes matching event/strategy evidence. Later stages reuse `NO_CONTROL_D0_INDEX.csv` and `STEP1_BASELINE_INDEX.csv` instead of rerunning the same SWMM baselines.

## 8. Generate D1 for Step1 coverage

Create a CSV containing **development/train only**, then:

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

D1 resumes event-by-event. It is Step1 state-space coverage only and must not be used as a D2/D3 replay prefix.

Build Step1 indexes:

```powershell
rtc-build-step1-index `
  --baseline-index "$Root\baseline_cache\STEP1_BASELINE_INDEX.csv" `
  --d1-index "$Root\d1\D1_RUN_INDEX.csv" `
  --out-dir "$Root\step1"
```

## 9. Train and accept Step1

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

If interrupted, rerun the exact command. The default `step1.pt.trainstate.pt` resumes only for the same data/timing/model/training contract.

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

## 10. Generate production D2 and D3

Use the cached No-control prefix:

```powershell
rtc-design-checkpoints `
  --run-index "$Root\baseline_cache\NO_CONTROL_D0_INDEX.csv" `
  --out "$Root\checkpoints\checkpoint_settings.csv" `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <HISTORY_READY_MINUTES>

rtc-design-probes `
  --inp $Inp `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --out "$Root\d2\probe_manifest.csv"

rtc-run-probes `
  --manifest "$Root\d2\probe_manifest.csv" `
  --out-dir "$Root\d2" `
  --horizon-minutes <FROZEN_HORIZON_MINUTES> `
  --stride-seconds <FROZEN_MODEL_STEP> `
  --workers 16 `
  --swmm-threads-per-process 1
```

D3:

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

`--max-active` controls exploration sparsity in D3 data generation only. Online MPC still optimizes all actuators.

## 11. Build the time-locked Step2 data

```powershell
rtc-build-step2-index `
  --d2-manifest "$Root\d2\probe_manifest.csv" `
  --d2-run-summary "$Root\d2\RUN_SUMMARY.csv" `
  --d3-run-summary "$Root\d3\D3_RUN_SUMMARY.csv" `
  --out "$Root\step2\step2_run_index.csv"
```

Compile train shards:

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

Repeat with `--development-fold validation` and `--out-dir "$Root\step2\validation_shards"`.

## 12. Train and accept Step2

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

Rerun the same command to resume an interrupted epoch sequence.

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

## 13. Validate the action-effect model

Local/boundary TFV gradient:

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

Joint action ranking:

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

Formal ranking must pass both D2 local and D3 multi-actuator/multi-step evidence.

## 14. Proposed development closed loop and real-time gate

Run each selected development event with:

```powershell
rtc-run-policy `
  --strategy proposed `
  --inp <EVENT_INP> `
  --out-dir <EVENT_OUTPUT_DIR> `
  --run-id <RUN_ID> `
  --sensors $Sensors `
  --priority $Priority `
  --config "$Root\contracts\controller_resolved.json" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --step1 "$Root\models\step1.pt" `
  --step2 "$Root\models\step2.pt"
```

Create `development/RUN_INDEX.csv`, then:

```powershell
rtc-accept-runtime `
  --run-index "$Root\development\RUN_INDEX.csv" `
  --config "$Root\contracts\controller_resolved.json" `
  --out "$Root\acceptance\runtime_acceptance.json"
```

Do not Policy Lock if any fatal history/readback/runtime/deadline fallback occurs or the frozen compute budget is exceeded.

## 15. Policy Lock V4

Complete the TFV pipeline ledger and artifact-map JSON, then:

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

After lock, do not change the scientific implementation contract, models, timing, thresholds, forecast, sensors, priority nodes or split registry used by Final.

## 16. Final fixed references and Proposed

Generate/resume fixed Final references exactly once:

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out-dir "$Root\final_baseline_cache" `
  --stage final `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --workers 16 `
  --swmm-threads-per-process 1
```

Run the locked Proposed controller once per untouched Final event. Do not tune anything from Final outcomes.

Formalize every Proposed/fixed run:

```powershell
rtc-formalize-run `
  --main-metadata <RUN_METADATA.json> `
  --strategy <proposed|no_control|internal_rtc|all_open|all_closed> `
  --event-id <EVENT_ID> `
  --rainfall-group <RAINFALL_GROUP> `
  --out <RUN.formal_manifest.json>
```

Build one complete Final run index with:

```text
event_id,rainfall_group,strategy,formal_manifest_path
```

and compile:

```powershell
rtc-compile-final-v4 `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --run-index "$Root\final\FINAL_RUN_INDEX.csv" `
  --out-dir "$Root\final"
```

The compiler first collapses variants inside each rainfall group, then gives every independent rainfall group equal weight.

## 17. Safe resume rule

Rerun an interrupted command normally. Do **not** manually copy old outputs into a new run to make a command skip work.

A computation is safely reusable only when its declared contract/key matches and required artifact hashes verify. File existence alone is never sufficient.

If a scientific implementation contract, numerical input, timing, action/sequence manifest, graph/model or training manifest changes, the affected result must be regenerated/retrained. Unrelated documentation/reporting edits should not invalidate expensive evidence.

## 18. Completion checklist

The final framework is complete only when all of these are true:

- physical/priority preflight passes;
- Phase-0 timing is not censored;
- production timing is frozen;
- Step1 group-balanced gate passes;
- Step2 group-balanced gate passes;
- D2 gradient gate passes;
- D2 + D3 joint-action ranking gate passes;
- Proposed development real-time gate passes;
- Policy Lock V4 is created;
- untouched Final contains the complete five-strategy matrix;
- all Final TFV/PFV/Global Peak metrics are authoritative SWMM truth;
- no Final information was used for training or tuning.
