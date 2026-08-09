# Local runbook — Wuhan RTC v0.6.1

This is the canonical executable workflow for the causal TFV-first study. Use the final merged `main` branch only.

## 0. Install and test

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

Recommended hardware contract:

```text
SWMM generation: up to 16 independent Python processes, normally 1 SWMM thread/process
GPU: RTX 4060 8 GB, AMP, small micro-batches + gradient accumulation
```

## 1. Define inputs

```powershell
$Root     = "E:\RTC_sewer\RTC_fresh_v061"
$Inp      = "E:\RTC_sewer\data\wuhan_v8_storage_retrofit.inp"
$Events   = "E:\RTC_sewer\contracts\events_v061.csv"
$Priority = "E:\RTC_sewer\RTC\data\priority_nodes.txt"
$Sensors  = "E:\RTC_sewer\contracts\sensor_nodes.txt"
$Plan     = "E:\RTC_sewer\RTC\configs\formal_baseline_plan.v3.json"
```

The event registry requires:

```text
event_id,rainfall_group,inp_path,scientific_split,development_fold
```

Rainfall groups must not cross scientific roles or development train/validation. Final remains untouched until Policy Lock.

## 2. Initialize workspace and preflight

```powershell
rtc-init-fresh-workspace `
  --root $Root `
  --inp $Inp `
  --priority $Priority `
  --events $Events

rtc-validate-rainfall-design `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out "$Root\contracts\rainfall_design_evidence.json"

New-Item -ItemType Directory -Force "$Root\preflight" | Out-Null

rtc-inp-audit-v2 `
  --inp $Inp `
  --priority $Priority `
  --out "$Root\preflight\inp_audit.json"
```

Do not continue if the intended physical census/units or eight-node priority mapping fail. PFV in this project means **Priority Flood Volume**, not peak flooding rate.

## 3. Phase-0 high-frequency timing audit

Create a representative development-only pilot registry. A simple starting scaffold is:

```powershell
$Registry = Import-Csv "$Root\contracts\event_registry_with_splits.csv"
$Registry |
  Where-Object { $_.scientific_split -eq "development" } |
  Export-Csv "$Root\contracts\phase0_development_events.csv" -NoTypeInformation
```

You may replace this with a smaller rainfall-diverse development subset after inspecting descriptors.

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

Freeze production model step, control interval, history and horizon only after reviewing readback lag, `t10/t50/t90`, peak timing and censoring. Phase-0 60 s branches cannot enter a production 300 s Step2 dataset.

## 4. Resolve production configuration

Create:

```text
$Root\contracts\time_scale_config.json
$Root\contracts\controller_resolved.json
$Root\contracts\model_acceptance_contract.json
```

Use development-only evidence and repository templates. Formal timing requires:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control aligns with model/control grids
full t=0-inclusive causal history exists before first MPC
prediction horizon >= one complete control interval
Formal D3 horizon contains a whole number of control blocks
decision_runtime_budget_seconds < control_update_seconds
```

Read all later values from the frozen controller JSON:

```powershell
$Controller     = "$Root\contracts\controller_resolved.json"
$Cfg            = Get-Content $Controller -Raw | ConvertFrom-Json
$ModelStep      = [int]$Cfg.model_step_seconds
$ControlUpdate  = [int]$Cfg.control_update_seconds
$ControlStart   = [int]$Cfg.control_start_minutes
$HistorySteps   = [int]$Cfg.controller.history_steps
$HorizonSteps   = [int]$Cfg.controller.horizon_steps
$HorizonMinutes = [int](($HorizonSteps * $ModelStep) / 60)

if (($ControlUpdate % $ModelStep) -ne 0) { throw "control/model step mismatch" }
if ((($HorizonSteps * $ModelStep) % $ControlUpdate) -ne 0) {
  throw "Formal D3 requires whole control blocks"
}
```

Example: 300 s model step + 600 s control + 120 min horizon = 24 Step2 model steps = 12 D3 control blocks.

## 5. Compile frozen static assets

```powershell
rtc-compile-formal-assets `
  --inp $Inp `
  --priority $Priority `
  --sensors $Sensors `
  --out-dir "$Root\formal_assets"
```

Expected:

```text
formal_assets/graph_schema.npz
formal_assets/state_schema.json
formal_assets/actuator_catalog.json
formal_assets/physical_contract.json
formal_assets/formal_asset_audit.json
```

For the audited Wuhan V8 lineage the action schema should contain all 109 writable links and no fixed active subset.

## 6. Generate fixed baselines once

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --config $Controller `
  --out-dir "$Root\baseline_cache" `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

Reuse `STEP1_BASELINE_INDEX.csv` for Step1 and `NO_CONTROL_D0_INDEX.csv` for D2/D3 prefixes. Formal fixed references are `no_control`, `internal_rtc`, `all_open`, `all_closed`.

## 7. Generate D1 development/train coverage

```powershell
$Registry = Import-Csv "$Root\contracts\event_registry_with_splits.csv"
$Registry |
  Where-Object {
    $_.scientific_split -eq "development" -and $_.development_fold -eq "train"
  } |
  Export-Csv "$Root\contracts\development_train_events.csv" -NoTypeInformation

rtc-run-d1-batch `
  --events "$Root\contracts\development_train_events.csv" `
  --sensors $Sensors `
  --out-dir "$Root\d1" `
  --seed 42 `
  --model-step-seconds $ModelStep `
  --control-update-seconds $ControlUpdate `
  --control-start-minutes $ControlStart `
  --workers 16 `
  --swmm-threads-per-process 1

rtc-build-step1-index `
  --baseline-index "$Root\baseline_cache\STEP1_BASELINE_INDEX.csv" `
  --d1-index "$Root\d1\D1_RUN_INDEX.csv" `
  --out-dir "$Root\step1"
```

D1 is Step1 state-space coverage only; never use it as a D2/D3 replay prefix.

## 8. Train and accept Step1

```powershell
rtc-train-step1-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step1\train_run_index.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --sensors $Sensors `
  --history-steps $HistorySteps `
  --model-step-seconds $ModelStep `
  --epochs 30 `
  --batch-size 4 `
  --grad-accum 4 `
  --out "$Root\models\step1.pt"

rtc-accept-step1-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step1\validation_run_index.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --sensors $Sensors `
  --model "$Root\models\step1.pt" `
  --history-steps $HistorySteps `
  --model-step-seconds $ModelStep `
  --priority $Priority `
  --out "$Root\acceptance\step1_metrics.json"

rtc-accept-gate `
  --metrics "$Root\acceptance\step1_metrics.json" `
  --contract "$Root\contracts\model_acceptance_contract.json" `
  --section step1 `
  --out "$Root\acceptance\step1_gate.json"
```

Rerun the identical training command after interruption to resume compatible epoch state.

## 9. Production D2 and D3

```powershell
rtc-design-checkpoints `
  --run-index "$Root\baseline_cache\NO_CONTROL_D0_INDEX.csv" `
  --out "$Root\checkpoints\checkpoint_settings.csv" `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes $ControlStart

rtc-design-probes `
  --inp $Inp `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --out "$Root\d2\probe_manifest.csv"

rtc-run-probes `
  --manifest "$Root\d2\probe_manifest.csv" `
  --out-dir "$Root\d2" `
  --horizon-minutes $HorizonMinutes `
  --stride-seconds $ModelStep `
  --workers 16 `
  --swmm-threads-per-process 1
```

D3 v0.6.1 derives action-block count from the same frozen controller config:

```powershell
rtc-design-d3 `
  --inp $Inp `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --controller-config $Controller `
  --out "$Root\d3\sequence_manifest.csv" `
  --sequences-per-checkpoint 12 `
  --perturbation-std 0.20 `
  --change-probability 0.25 `
  --seed 42

rtc-run-d3-batch `
  --manifest "$Root\d3\sequence_manifest.csv" `
  --out-dir "$Root\d3" `
  --control-block-seconds $ControlUpdate `
  --stride-seconds $ModelStep `
  --workers 16 `
  --swmm-threads-per-process 1
```

The public D3 runner refuses a runtime model/control clock that differs from the design manifest.

## 10. Compile fixed-time Step2 shards

```powershell
rtc-build-step2-index `
  --d2-manifest "$Root\d2\probe_manifest.csv" `
  --d2-run-summary "$Root\d2\RUN_SUMMARY.csv" `
  --d3-run-summary "$Root\d3\D3_RUN_SUMMARY.csv" `
  --out "$Root\step2\step2_run_index.csv"

rtc-compile-step2-shards `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step2\step2_run_index.csv" `
  --out-dir "$Root\step2\train_shards" `
  --development-fold train `
  --shard-size 128 `
  --model-step-seconds $ModelStep `
  --horizon-steps $HorizonSteps

rtc-compile-step2-shards `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step2\step2_run_index.csv" `
  --out-dir "$Root\step2\validation_shards" `
  --development-fold validation `
  --shard-size 128 `
  --model-step-seconds $ModelStep `
  --horizon-steps $HorizonSteps
```

Mixed Phase-0/production steps or D2/D3 horizon mismatch fail closed.

## 11. Train and accept Step2

```powershell
rtc-train-step2-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --manifest "$Root\step2\train_shards\manifest.json" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --epochs 30 `
  --batch-size 2 `
  --grad-accum 4 `
  --out "$Root\models\step2.pt"

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

Review the diagnostic physical-plausibility fractions in `step2_metrics.json`: negative depth, flooding rate, node volume, and non-finite state/managed-flow values. The world model is physics-informed but is not claimed to be an exact mass-conserving solver.

## 12. Validate gradient and joint action ranking

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

## 13. Proposed development closed loop

For every selected development event:

```powershell
rtc-run-policy `
  --strategy proposed `
  --inp <EVENT_INP> `
  --out-dir <EVENT_OUTPUT_DIR> `
  --run-id <RUN_ID> `
  --sensors $Sensors `
  --priority $Priority `
  --config $Controller `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --step1 "$Root\models\step1.pt" `
  --step2 "$Root\models\step2.pt"
```

Build `$Root\development\RUN_INDEX.csv`, then:

```powershell
rtc-accept-runtime `
  --run-index "$Root\development\RUN_INDEX.csv" `
  --config $Controller `
  --out "$Root\acceptance\runtime_acceptance.json"
```

Do not lock if history/readback/runtime/deadline failures occur or the frozen runtime budget is exceeded.

## 14. Record the ordered pre-lock evidence ledger

```powershell
$Ledger = "$Root\policy_lock\pipeline_ledger.json"

rtc-record-pipeline-stage --ledger $Ledger --stage inp_preflight --passed `
  --evidence "$Root\preflight\inp_audit.json"
rtc-record-pipeline-stage --ledger $Ledger --stage rainfall_split --passed `
  --evidence "$Root\contracts\rainfall_design_evidence.json"
rtc-record-pipeline-stage --ledger $Ledger --stage phase0_timescale --passed `
  --evidence "$Root\phase0\timescale_summary.json"
rtc-record-pipeline-stage --ledger $Ledger --stage d0_d1_coverage --passed `
  --evidence "$Root\baseline_cache\STEP1_BASELINE_INDEX.csv" `
  --evidence "$Root\d1\D1_RUN_INDEX.csv"
rtc-record-pipeline-stage --ledger $Ledger --stage d2_d3_generation --passed `
  --evidence "$Root\d2\RUN_SUMMARY.csv" `
  --evidence "$Root\d3\D3_RUN_SUMMARY.csv" `
  --evidence "$Root\step2\step2_run_index.csv"
rtc-record-pipeline-stage --ledger $Ledger --stage step1_acceptance --passed `
  --evidence "$Root\acceptance\step1_gate.json"
rtc-record-pipeline-stage --ledger $Ledger --stage step2_acceptance --passed `
  --evidence "$Root\acceptance\step2_gate.json"
rtc-record-pipeline-stage --ledger $Ledger --stage gradient_acceptance --passed `
  --evidence "$Root\acceptance\gradient_gate.json"
rtc-record-pipeline-stage --ledger $Ledger --stage candidate_ranking_acceptance --passed `
  --evidence "$Root\acceptance\ranking.json"
rtc-record-pipeline-stage --ledger $Ledger --stage closed_loop_development --passed `
  --evidence "$Root\development\RUN_INDEX.csv"
rtc-record-pipeline-stage --ledger $Ledger --stage runtime_timing_acceptance --passed `
  --evidence "$Root\acceptance\runtime_acceptance.json"
```

The ledger hashes evidence and refuses to pass later stages when prerequisites are missing.

## 15. Build Policy-Lock artifacts and create the lock

No hand-written JSON is required:

```powershell
rtc-build-policy-artifacts `
  --root $Root `
  --frozen-inp $Inp `
  --priority $Priority `
  --sensors $Sensors `
  --baseline-plan $Plan `
  --out "$Root\policy_lock\artifacts.json"

rtc-policy-lock-v5 `
  --ledger $Ledger `
  --artifacts "$Root\policy_lock\artifacts.json" `
  --out "$Root\policy_lock\policy_lock.json"
```

Expected contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

After lock, do not tune scientific code semantics, models, timing, forecast, thresholds, sensor/priority layout or rainfall splits from Final outcomes.

## 16. Untouched Final

Generate/resume fixed references:

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out-dir "$Root\final_baseline_cache" `
  --stage final `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --workers 16 `
  --swmm-threads-per-process 1
```

Run the **locked** Proposed controller once for each untouched Final event with exactly the locked controller/graph/models/sensors/priority files. Do not retrain or retune from Final results.

## 17. Formalize and compile Final

For every Proposed and fixed run:

```powershell
rtc-formalize-run `
  --main-metadata <RUN_METADATA.json> `
  --strategy <proposed|no_control|internal_rtc|all_open|all_closed> `
  --event-id <EVENT_ID> `
  --rainfall-group <RAINFALL_GROUP> `
  --out <RUN.formal_manifest.json>
```

Create `$Root\final\FINAL_RUN_INDEX.csv`:

```text
event_id,rainfall_group,strategy,formal_manifest_path
```

Then:

```powershell
rtc-compile-final-v4 `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --run-index "$Root\final\FINAL_RUN_INDEX.csv" `
  --out-dir "$Root\final"
```

Final compiler requires the complete five-strategy matrix and gives every independent rainfall group equal statistical weight.

## 18. Completion rule

The codebase is implementation-complete only after the exact audit-PR head passes CI and is merged. The research is scientifically complete only after the real Wuhan local run passes preflight, Phase-0, Step1, Step2, D2 gradient, D2+D3 joint ranking, development runtime, Policy Lock and untouched Final SWMM.

Do not claim a significant TFV improvement until Final SWMM proves it. PFV/depth and Global Peak must be reported honestly but are not hard MPC gates.

## 19. Resume rule

After interruption, rerun the same command. Never copy/rename old outputs merely to force a skip. Reuse requires compatible scientific semantics, exact numerical input/config/action lineage and verified generated-artifact hashes.
