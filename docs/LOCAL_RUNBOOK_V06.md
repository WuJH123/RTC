# Local runbook — Wuhan RTC v0.6.1

This is the supported execution order for the final causal TFV-first study. Run expensive generation only from the final merged `main` branch.

## 0. Install and verify the release

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

Recommended workstation contract:

```text
SWMM generation: up to 16 independent Python processes, normally 1 SWMM thread/process
GPU training: RTX 4060 8 GB, AMP, small micro-batches + gradient accumulation
```

Do not interpret 16 workers as one SWMM simulation using 16 engine threads.

## 1. Prepare the event registry and sensor list

Create an event registry with at least:

```text
event_id,rainfall_group,inp_path,scientific_split,development_fold
```

Hard requirements are rainfall-group-disjoint development train/validation, no cross-split rainfall-group leakage and untouched Final groups. About 160 independent rainfall groups is the recommended paper-strength target, not a startup gate.

Define paths:

```powershell
$Root     = "E:\RTC_sewer\RTC_fresh_v061"
$Inp      = "E:\RTC_sewer\data\wuhan_v8_storage_retrofit.inp"
$Events   = "E:\RTC_sewer\contracts\events_v061.csv"
$Priority = "E:\RTC_sewer\RTC\data\priority_nodes.txt"
$Sensors  = "E:\RTC_sewer\contracts\sensor_nodes.txt"
```

Initialize the clean study root and validate split design:

```powershell
rtc-init-fresh-workspace `
  --root $Root `
  --inp $Inp `
  --priority $Priority `
  --events $Events

rtc-validate-rainfall-design `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out "$Root\contracts\rainfall_design_evidence.json"
```

Large generated data may live on another disk; scientific reuse is lineage/config/artifact-hash based, not path based.

## 2. Physical model and priority preflight

```powershell
New-Item -ItemType Directory -Force "$Root\preflight" | Out-Null

rtc-inp-audit-v2 `
  --inp $Inp `
  --priority $Priority `
  --out "$Root\preflight\inp_audit.json"
```

Do not continue if:

- the intended Wuhan physical census/units differ;
- any of the eight priority nodes is absent;
- No-control is not `NO_SUPERVISORY_RTC_V2`;
- unexpected native-control or pump-local-control semantics are found.

Remember: in this repository PFV means **Priority Flood Volume**, not peak flood flow.

## 3. Phase-0 — determine model/control time scales before production data

Create a representative **development-only** pilot CSV from the canonical registry. Never include Final rows.

Example filtering scaffold:

```powershell
$Registry = Import-Csv "$Root\contracts\event_registry_with_splits.csv"
$Registry |
  Where-Object { $_.scientific_split -eq "development" } |
  Export-Csv "$Root\contracts\phase0_development_events.csv" -NoTypeInformation
```

For compute efficiency you may replace that file with a smaller rainfall-diverse development subset after reviewing rainfall descriptors.

Generate high-frequency No-control data:

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

Review readback lag, actuator-flow `t10/t50/t90`, network response/peak timing and horizon censoring. Extend the pilot horizon if responses remain censored. Use pulse/release experiments if recovery after control release is important.

Do **not** put 60 s Phase-0 branches into a 300 s production Step2 dataset.

## 4. Freeze controller/time/acceptance contracts

Using Phase-0 plus development-only sensitivity/runtime benchmarking, create resolved:

```text
$Root\contracts\time_scale_config.json
$Root\contracts\controller_resolved.json
$Root\contracts\model_acceptance_contract.json
```

Start from repository templates where applicable. Required relationships include:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control aligns with model and control grids
first control follows a complete t=0-inclusive causal history
horizon >= one complete control interval
for Formal D3: horizon_steps is an integer number of control blocks
decision_runtime_budget_seconds < control_update_seconds
```

After writing `controller_resolved.json`, read all later timing values directly from it:

```powershell
$Controller    = "$Root\contracts\controller_resolved.json"
$Cfg           = Get-Content $Controller -Raw | ConvertFrom-Json
$ModelStep     = [int]$Cfg.model_step_seconds
$ControlUpdate = [int]$Cfg.control_update_seconds
$ControlStart  = [int]$Cfg.control_start_minutes
$HistorySteps  = [int]$Cfg.controller.history_steps
$HorizonSteps  = [int]$Cfg.controller.horizon_steps
$HorizonMinutes = [int](($HorizonSteps * $ModelStep) / 60)

if (($ControlUpdate % $ModelStep) -ne 0) { throw "control/model step mismatch" }
if ((($HorizonSteps * $ModelStep) % $ControlUpdate) -ne 0) {
  throw "Formal D3 requires a horizon containing whole control blocks"
}
```

For example, 300 s model step + 600 s control + 120 min horizon gives 24 model steps but only 12 D3 control blocks. v0.6.1 derives this automatically.

## 5. Compile frozen Formal static assets

v0.6.1 exposes the complete asset compiler as a public CLI:

```powershell
rtc-compile-formal-assets `
  --inp $Inp `
  --priority $Priority `
  --sensors $Sensors `
  --out-dir "$Root\formal_assets"
```

Expected outputs include:

```text
formal_assets/graph_schema.npz
formal_assets/state_schema.json
formal_assets/actuator_catalog.json
formal_assets/physical_contract.json
formal_assets/formal_asset_audit.json
```

Confirm `actuator_count = 109`, no fixed active subset and all priority/sensor IDs valid.

## 6. Generate fixed pre-lock baselines exactly once

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --config $Controller `
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

Later stages reuse:

```text
baseline_cache/STEP1_BASELINE_INDEX.csv
baseline_cache/NO_CONTROL_D0_INDEX.csv
```

The cache validates actual strategy semantics and artifact hashes before resuming.

## 7. Generate D1 development/train coverage for Step1

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

D1 is Step1 coverage only and must never become a D2/D3 replay prefix.

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
```

Rerun the identical command after interruption to resume the compatible epoch state.

```powershell
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

Do not continue to Policy Lock unless the preregistered gate passes.

## 9. Generate production D2 and D3

Design replayable checkpoints from cached No-control:

```powershell
rtc-design-checkpoints `
  --run-index "$Root\baseline_cache\NO_CONTROL_D0_INDEX.csv" `
  --out "$Root\checkpoints\checkpoint_settings.csv" `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes $ControlStart
```

D2:

```powershell
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

D3 v0.6.1 — **do not manually pass a model-step horizon as an action-block count**:

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

The public D3 runner fails before SWMM if runtime model/control clocks differ from the design manifest.

## 10. Build fixed-time Step2 data

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
  --model-step-seconds $ModelStep `
  --horizon-steps $HorizonSteps
```

Validation shards:

```powershell
rtc-compile-step2-shards `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step2\step2_run_index.csv" `
  --out-dir "$Root\step2\validation_shards" `
  --development-fold validation `
  --shard-size 128 `
  --model-step-seconds $ModelStep `
  --horizon-steps $HorizonSteps
```

If D2/D3 durations do not match exactly, shard compilation fails rather than silently padding/truncating.

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

Review the additional Step2 physical-plausibility diagnostics even if they are not hard preregistered thresholds:

```text
negative_depth_fraction
negative_flooding_rate_fraction
negative_node_volume_fraction
nonfinite_state_fraction
nonfinite_actuator_flow_fraction
```

The surrogate is physics-informed, not claimed to be a strict mass-conserving hydraulic solver.

## 12. Validate local gradients and joint action ordering

D2 finite-difference gradient:

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

D2 + D3 ranking:

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

D2 alone is not sufficient evidence for the joint action space searched by MPC.

## 13. Proposed development closed-loop and real-time acceptance

For each selected development event:

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

Build `$Root\development\RUN_INDEX.csv` from the completed Proposed metadata, then:

```powershell
rtc-accept-runtime `
  --run-index "$Root\development\RUN_INDEX.csv" `
  --config $Controller `
  --out "$Root\acceptance\runtime_acceptance.json"
```

Do not Policy Lock if there is any fatal history/readback/runtime/deadline fallback or if the measured decision time exceeds the frozen compute budget.

## 14. Build the ordered evidence ledger

v0.6.1 provides a public evidence-ledger command. The following calls must be made in order after the corresponding stage has genuinely passed:

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

The command hashes each evidence file and refuses to pass a later stage if an earlier prerequisite has not passed.

## 15. Build the Policy-Lock artifact map and lock

Create the exact artifact map with PowerShell:

```powershell
$Artifacts = [ordered]@{
  fresh_workspace_manifest      = "$Root\FRESH_WORKSPACE_MANIFEST.json"
  inp_preflight                 = "$Root\preflight\inp_audit.json"
  frozen_inp                    = $Inp
  priority_nodes                = $Priority
  sensor_layout                 = $Sensors
  time_scale_config             = "$Root\contracts\time_scale_config.json"
  step1_model                   = "$Root\models\step1.pt"
  step2_model                   = "$Root\models\step2.pt"
  graph_schema                  = "$Root\formal_assets\graph_schema.npz"
  split_registry                = "$Root\contracts\event_registry_with_splits.csv"
  model_acceptance_contract     = "$Root\contracts\model_acceptance_contract.json"
  step1_acceptance              = "$Root\acceptance\step1_gate.json"
  step2_acceptance              = "$Root\acceptance\step2_gate.json"
  gradient_acceptance           = "$Root\acceptance\gradient_gate.json"
  candidate_ranking_acceptance  = "$Root\acceptance\ranking.json"
  controller_config             = $Controller
  baseline_plan                 = "E:\RTC_sewer\RTC\configs\formal_baseline_plan.v3.json"
  runtime_acceptance            = "$Root\acceptance\runtime_acceptance.json"
}

New-Item -ItemType Directory -Force "$Root\policy_lock" | Out-Null
$Artifacts | ConvertTo-Json -Depth 4 | Set-Content "$Root\policy_lock\artifacts.json" -Encoding UTF8

rtc-policy-lock-v5 `
  --ledger $Ledger `
  --artifacts "$Root\policy_lock\artifacts.json" `
  --out "$Root\policy_lock\policy_lock.json"
```

Expected lock contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

After this point do not tune code semantics, models, timing, forecast, thresholds, sensors, priority nodes or rainfall splits from Final results.

## 16. Generate untouched Final fixed references

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out-dir "$Root\final_baseline_cache" `
  --stage final `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --workers 16 `
  --swmm-threads-per-process 1
```

Run locked Proposed once for each untouched Final event using the same `rtc-run-policy --strategy proposed` command and exactly the locked graph/models/controller/sensors/priority files.

No Final result may trigger retraining or parameter changes.

## 17. Formalize every Final run and compile the five-strategy matrix

For each Proposed/fixed run:

```powershell
rtc-formalize-run `
  --main-metadata <RUN_METADATA.json> `
  --strategy <proposed|no_control|internal_rtc|all_open|all_closed> `
  --event-id <EVENT_ID> `
  --rainfall-group <RAINFALL_GROUP> `
  --out <RUN.formal_manifest.json>
```

Create:

```text
$Root\final\FINAL_RUN_INDEX.csv
```

with columns:

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

Final compiler requires one complete five-strategy matrix per event and gives each independent rainfall group equal statistical weight.

## 18. What constitutes success

### Implementation-complete

The repository code/tests/CLI contracts are complete when the exact audit-PR head passes GitHub Actions and is merged.

### Scientifically complete

The Wuhan study is complete only when the local frozen model has passed:

```text
preflight
Phase-0
Step1 acceptance
Step2 acceptance + physical diagnostics
D2 gradient
D2+D3 joint ranking
Proposed development runtime
Policy Lock
untouched five-strategy Final SWMM
```

Do not claim that Proposed significantly reduces TFV until the untouched Final SWMM files demonstrate it. PFV/depth and Global Peak must be reported honestly, but PFV is a soft secondary/diagnostic quantity rather than a hard decision gate.

## 19. Safe resume rule

After interruption, rerun the same command. Do not manually copy/rename historical outputs to force a skip.

Reuse is allowed only when the relevant scientific contract, numerical inputs/configuration/action sequence and generated-artifact hashes verify. Unrelated documentation edits should not invalidate expensive hydraulic evidence.
