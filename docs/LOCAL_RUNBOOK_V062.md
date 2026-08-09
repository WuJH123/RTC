# Canonical Codex/local runbook — Wuhan RTC v0.6.2

This is the only recommended full execution sequence for the v0.6.2 scientific implementation.

**Start from a new workspace. Do not point these commands at a v0.6.1 RTC-derived workspace.** v0.6.2 changed the Formal time origin, counterfactual-prefix verification, SWMM-engine lineage and Final event-forcing identity; old D0/D1/D2/D3 branches, Step1/Step2 models, acceptance evidence and Policy Locks are intentionally incompatible.

## 0. Pull the merged release and test

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

Use one Python environment and one PySWMM/SWMM engine version for all Formal data generation, training validation, Proposed runs and Final evaluation. Do not upgrade it in the middle of the experiment.

Recommended workstation execution:

```text
SWMM: up to 16 independent Python processes × 1 SWMM thread/process
RTX 4060 8 GB: AMP + small micro-batches + gradient accumulation
```

## 1. Define paths

Adjust only paths that genuinely differ on the local machine:

```powershell
$Repo     = "E:\RTC_sewer\RTC"
$Root     = "E:\RTC_sewer\RTC_fresh_v062"
$Inp      = "E:\RTC_sewer\data\wuhan_v8_storage_retrofit.inp"
$Events   = "E:\RTC_sewer\contracts\events_v062.csv"
$Priority = "$Repo\data\priority_nodes.txt"
$Sensors  = "E:\RTC_sewer\contracts\sensor_nodes.txt"
$Plan     = "$Repo\configs\formal_baseline_plan.v3.json"
```

The event registry requires at least:

```text
event_id,rainfall_group,inp_path,scientific_split,development_fold
```

Required split semantics:

- each `event_id` is unique;
- each `rainfall_group` belongs to one scientific split only;
- development groups are separated into `train` and `validation`;
- non-development rows have empty `development_fold`;
- Final stays untouched before Policy Lock.

About 160 independent rainfall groups is a paper-strength recommendation, not a software start-up gate.

## 2. Initialize a genuinely fresh workspace and preflight the INP

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

For the audited Wuhan V8 lineage verify the expected census/units and all eight priority nodes. `no_control` must be `NO_SUPERVISORY_RTC_V2`.

If event INPs use external `FILE` rainfall/time-series inputs, v0.6.2 binds their content hashes and preserves their paths when creating relocated runtime INPs.

## 3. Phase-0 high-frequency timing audit

Create a rainfall-diverse development-only pilot registry. Starting with all development events is valid; Codex may reduce it only using forcing descriptors, never hydraulic outcome labels from Final.

```powershell
$Registry = Import-Csv "$Root\contracts\event_registry_with_splits.csv"
$Registry |
  Where-Object { $_.scientific_split -eq "development" } |
  Export-Csv "$Root\contracts\phase0_development_events.csv" -NoTypeInformation
```

Run t=0-inclusive No-control trajectories at <=60 s sample spacing:

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

v0.6.2 D2 will verify that every replayed checkpoint reaches the saved No-control six-channel state and all actuator readbacks before applying the candidate.

Review readback lag, actuator-flow `t10/t50/t90`, peak timing, flooding/depth response and horizon censoring. Extend Phase-0 horizon if response remains censored.

**Stop here before production generation until the model/control/history/horizon clocks are scientifically frozen.**

## 4. Resolve the production timing/controller/acceptance files

Create resolved files under the fresh workspace using repository templates plus development-only Phase-0/model sensitivity evidence:

```text
$Root\contracts\time_scale_config.json
$Root\contracts\controller_resolved.json
$Root\contracts\model_acceptance_contract.json
```

Required timing relationships:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control aligns to model and control grids
full t=0-inclusive Step1 history exists before first MPC
horizon >= one complete control interval
Formal D3 horizon = whole number of control blocks
decision_runtime_budget_seconds < control_update_seconds
```

Load all later timing values from the one frozen controller file:

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
  throw "D3 horizon must contain whole control blocks"
}
```

Example only if Phase-0 supports it: 300 s model step + 600 s control update + 120 min horizon = 24 Step2 model steps = 12 D3 control blocks.

## 5. Compile frozen graph/state/actuator assets

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

The action schema should contain all 109 writable Wuhan links and no fixed active subset/binary mask.

## 6. Generate fixed pre-lock baselines once

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --config $Controller `
  --out-dir "$Root\baseline_cache" `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

Formal fixed references are `no_control`, `internal_rtc`, `all_open`, `all_closed`.

Reuse views:

```text
STEP1_BASELINE_INDEX.csv -> Step1 baseline state coverage
NO_CONTROL_D0_INDEX.csv  -> production D2/D3 replay prefixes
```

Do not regenerate a fixed baseline merely because a later stage needs it. Resume is valid only when lineage/config/artifact hashes verify.

## 7. Generate D1 development/train state-space coverage

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

D1 is Step1 coverage only. Never use D1 as a D2/D3 replay prefix.

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

Formal Step1 now requires every trajectory to start at t=0, use the frozen model step and share one SWMM engine. Training and validation engine lineage must match.

Rerunning an identical interrupted training command resumes compatible epoch state.

## 9. Generate production D2 and D3

Design production checkpoints from the cached controls-disabled No-control trajectory:

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

D2 must report successful `EXACT_NO_CONTROL_PREFIX_REPLAY_V1` before any action is executed.

D3 derives the control-block count and setting-rate feasibility from the frozen controller:

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

The D3 runner rejects time/rate-contract drift and verifies the exact No-control prefix before the first sequence command.

## 10. Build the Step2 index and fixed-time/engine shards

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

One shard set must have exactly one:

```text
model_step_seconds
horizon_steps
SWMM engine version
node ordering
actuator ordering
```

Mixed Phase-0/production grids or mixed engines fail closed.

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

Inspect trajectory/managed-flow/exact-TFV metrics and physical-plausibility diagnostics. Final hydraulic truth remains SWMM.

## 12. Validate local gradients and joint action ordering

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

D2 validates local/boundary action effects; D3 validates joint multi-actuator/multi-step ordering. D2 alone is not evidence for the joint MPC search space.

## 13. Run Proposed on development events and accept real-time execution

For every selected development event, run the Proposed controller with the same frozen graph/models/config/sensor/priority files:

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

v0.6.2 refuses to validate a Proposed run if its actual SWMM engine differs from the engine bound into Step1/Step2.

Build `$Root\development\RUN_INDEX.csv`, then:

```powershell
rtc-accept-runtime `
  --run-index "$Root\development\RUN_INDEX.csv" `
  --config $Controller `
  --out "$Root\acceptance\runtime_acceptance.json"
```

Do not lock if history/readback/runtime/deadline failures occur or runtime exceeds the frozen decision budget.

## 14. Record the ordered evidence ledger

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

## 15. Build Policy-Lock artifacts and lock the experiment

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

Expected outer lock contract:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

The lock contains the v0.6.2 semantic implementation fingerprint and the single SWMM engine lineage for Step1/Step2.

After lock, do not tune code semantics, models, timing, forecasts, thresholds, sensors, priority nodes or rainfall splits from Final outcomes.

## 16. Generate untouched Final fixed baselines

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out-dir "$Root\final_baseline_cache" `
  --stage final `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --workers 16 `
  --swmm-threads-per-process 1
```

This uses the Policy-Locked controller timing/baseline/split/physical contract.

## 17. Run locked Proposed on every Final event

Run **every and only** event whose locked `scientific_split` is `final` exactly once with `--strategy proposed` and the locked models/config/assets. Do not inspect a Final result and then retrain or alter parameters.

For each completed Proposed main run, formalize it:

```powershell
rtc-formalize-run `
  --main-metadata <RUN_METADATA.json> `
  --strategy proposed `
  --event-id <EVENT_ID> `
  --rainfall-group <RAINFALL_GROUP> `
  --out <RUN.formal_manifest.json>
```

The Formal manifest binds:

- scientific event forcing hash;
- external `FILE` forcing bytes where present;
- physical network hash;
- SWMM engine version;
- main-run evidence;
- frozen-decision routing-step Global Peak replay.

Final fixed baseline cache already formalizes its fixed strategy runs.

## 18. Build the complete Final run index and compile

Create:

```text
$Root\final\FINAL_RUN_INDEX.csv
```

with:

```text
event_id,rainfall_group,strategy,formal_manifest_path
```

It must contain exactly five rows per locked Final event:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

Then:

```powershell
rtc-compile-final-v4 `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --run-index "$Root\final\FINAL_RUN_INDEX.csv" `
  --out-dir "$Root\final"
```

v0.6.2 refuses to compile if:

- any locked Final event is omitted;
- an extra/non-Final event is inserted;
- any event lacks one of the five strategies;
- `rainfall_group` differs from the locked registry;
- a run uses a different scientific event forcing;
- main/peak replay or paired strategies use inconsistent SWMM engine lineage.

Formal outputs include system TFV, eight-site PFV, Global Peak and paired Proposed-vs-baseline reductions. Independent rainfall groups receive equal statistical weight.

## 19. Scientific completion rule

Code/CI completion does **not** mean the Proposed controller is scientifically superior.

The Wuhan result can be interpreted only after the actual local run has completed:

```text
preflight
→ Phase-0 timing
→ fixed baselines
→ Step1 train/accept
→ exact-prefix D2/D3
→ Step2 train/accept
→ D2 gradient
→ D2+D3 ranking
→ Proposed development runtime
→ Policy Lock
→ complete untouched five-strategy Final SWMM
```

Only the authoritative Final SWMM evidence may support a claim that Proposed significantly reduces TFV. PFV/depth and Global Peak must be reported even when they worsen.

## 20. Resume rule

After interruption, rerun the same command. Do not rename/copy old outputs merely to trigger a skip.

Safe reuse requires compatible v0.6.2 scientific semantics plus matching numerical inputs/configuration/reference trajectories/actions and verified artifact hashes. A v0.6.1-derived RTC model or trajectory is not Formal-compatible with v0.6.2.
