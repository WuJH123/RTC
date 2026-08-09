# Local runbook — RTC v0.6 final workflow

Run this sequence only from the final merged `main` branch. Correctness gates protect causal/data/model semantics; recommended cohort sizes are not software blockers.

## 0. Install and test

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

Recommended workstation execution: up to 16 independent SWMM Python processes with one SWMM thread/process; GPU training uses AMP and small micro-batches with gradient accumulation.

v0.6 resume keys use a stable **scientific implementation fingerprint** plus exact numerical inputs/configuration and generated-artifact hashes. They do not hash every Python file byte-for-byte.

## 1. Event registry and workspace

Create a new registry outside the future workspace with:

```text
event_id,rainfall_group,inp_path,scientific_split,development_fold
```

Optional descriptors include total depth, duration, peak intensity and antecedent rainfall.

Hard rules: rainfall groups do not cross scientific splits; development train/validation are group-disjoint; Final remains untouched. About 160 independent rainfall groups is the recommended paper-strength target, not an execution gate.

```powershell
$Root     = "E:\RTC_sewer\RTC_fresh_v06"
$Inp      = "E:\RTC_sewer\data\wuhan_v8_storage_retrofit.inp"
$Events   = "E:\RTC_sewer\contracts\events_v06.csv"
$Priority = "E:\RTC_sewer\RTC\data\priority_nodes.txt"
$Sensors  = "E:\RTC_sewer\contracts\sensor_nodes.txt"

rtc-init-fresh-workspace `
  --root $Root --inp $Inp --priority $Priority --events $Events

rtc-validate-rainfall-design `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out "$Root\contracts\rainfall_design_evidence.json"

rtc-inp-audit-v2 `
  --inp $Inp --priority $Priority `
  --out "$Root\preflight\inp_audit.json"
```

Do not continue unless the intended Wuhan physical census and all eight priority nodes are present and No-control is `NO_SUPERVISORY_RTC_V2`.

Large generated data may live on another local volume; validity depends on lineage/config/artifact hashes, not directory location.

## 2. Phase-0 — freeze the time scales before production generation

Create `$Root\contracts\phase0_development_events.csv` containing development-only pilot events.

```powershell
rtc-run-d0-batch `
  --events "$Root\contracts\phase0_development_events.csv" `
  --strategy no_control `
  --out-dir "$Root\phase0\d0" `
  --record-stride-seconds 60 `
  --workers 16 --swmm-threads-per-process 1

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
  --workers 16 --swmm-threads-per-process 1

rtc-phase0-timescale `
  --manifest "$Root\phase0\probe_manifest.csv" `
  --run-summary "$Root\phase0\d2\RUN_SUMMARY.csv" `
  --detail-out "$Root\phase0\timescale_detail.csv" `
  --summary-out "$Root\phase0\timescale_summary.json" `
  --max-sample-seconds 60
```

Use readback lag, `t10/t50/t90`, peak timing and horizon censoring. Extend the pilot horizon when response peaks remain censored. Use pulse/release D3 experiments if recovery dynamics must be identified.

Do not put Phase-0 branches into production Step2 when their time grid differs.

## 3. Freeze controller/time contracts

Create resolved:

```text
contracts/time_scale_config.json
contracts/controller_resolved.json
contracts/model_acceptance_contract.json
```

Required relationships:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control aligns with model/control grids
first control follows a complete causal history
horizon >= one complete control interval
decision_runtime_budget_seconds < control_update_seconds
```

If Phase-0 supports 5/10 min, a representative timing is 300 s model step, 600 s control update, 13 history frames and first MPC at 60 min. Do not assume the horizon; freeze it from Phase-0/validation.

## 4. Build graph/static assets

Use `rtc-build-graph` and the formal-asset utilities to create at least:

```text
formal_assets/graph_schema.npz
formal_assets/state_schema.json
formal_assets/actuator_catalog.json
```

The graph/action schema must retain all 109 writable actuators.

## 5. Generate fixed baselines once

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --config "$Root\contracts\controller_resolved.json" `
  --out-dir "$Root\baseline_cache" `
  --stage prelock `
  --workers 16 --swmm-threads-per-process 1
```

Reuse `STEP1_BASELINE_INDEX.csv` for Step1 and `NO_CONTROL_D0_INDEX.csv` for D2/D3 checkpoints. Fixed Formal references are no_control, internal_rtc, all_open and all_closed.

## 6. D1 and Step1

Create `$Root\contracts\development_train_events.csv` containing development/train only.

```powershell
rtc-run-d1-batch `
  --events "$Root\contracts\development_train_events.csv" `
  --sensors $Sensors `
  --out-dir "$Root\d1" `
  --seed 42 `
  --model-step-seconds <MODEL_STEP> `
  --control-update-seconds <CONTROL_UPDATE> `
  --control-start-minutes <CONTROL_START> `
  --workers 16 --swmm-threads-per-process 1

rtc-build-step1-index `
  --baseline-index "$Root\baseline_cache\STEP1_BASELINE_INDEX.csv" `
  --d1-index "$Root\d1\D1_RUN_INDEX.csv" `
  --out-dir "$Root\step1"
```

D1 is Step1 coverage only; never use it as a D2/D3 replay prefix.

```powershell
rtc-train-step1-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step1\train_run_index.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --sensors $Sensors `
  --history-steps <HISTORY_STEPS> `
  --model-step-seconds <MODEL_STEP> `
  --epochs 30 --batch-size 4 --grad-accum 4 `
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
  --history-steps <HISTORY_STEPS> `
  --model-step-seconds <MODEL_STEP> `
  --priority $Priority `
  --out "$Root\acceptance\step1_metrics.json"

rtc-accept-gate `
  --metrics "$Root\acceptance\step1_metrics.json" `
  --contract "$Root\contracts\model_acceptance_contract.json" `
  --section step1 `
  --out "$Root\acceptance\step1_gate.json"
```

## 7. Production D2/D3

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
  --horizon-minutes <HORIZON_MINUTES> `
  --stride-seconds <MODEL_STEP> `
  --workers 16 --swmm-threads-per-process 1
```

Design D3 using the **actual v0.6 CLI**:

```powershell
rtc-design-d3 `
  --inp $Inp `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --out "$Root\d3\sequence_manifest.csv" `
  --horizon-steps <HORIZON_STEPS> `
  --sequences-per-checkpoint 12 `
  --perturbation-std 0.20 `
  --change-probability 0.25 `
  --seed 42

rtc-run-d3-batch `
  --manifest "$Root\d3\sequence_manifest.csv" `
  --out-dir "$Root\d3" `
  --control-block-seconds <CONTROL_UPDATE> `
  --stride-seconds <MODEL_STEP> `
  --workers 16 --swmm-threads-per-process 1
```

Every actuator is eligible in every D3 sequence. `change-probability` only controls stochastic data coverage and is not an online Top-K/fixed subset.

## 8. Step2 index/shards/train/accept

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
  --model-step-seconds <MODEL_STEP> `
  --horizon-steps <HORIZON_STEPS>
```

Repeat shard compilation with `--development-fold validation` and a separate validation output directory.

```powershell
rtc-train-step2-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --manifest "$Root\step2\train_shards\manifest.json" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --epochs 30 --batch-size 2 --grad-accum 4 `
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

## 9. Gradient and joint-action ranking

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

D2 validates local/boundary gradients; D3 validates joint multi-actuator/multi-step sequence ordering.

## 10. Proposed development and real-time gate

For each development event:

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

Build `development/RUN_INDEX.csv`, then:

```powershell
rtc-accept-runtime `
  --run-index "$Root\development\RUN_INDEX.csv" `
  --config "$Root\contracts\controller_resolved.json" `
  --out "$Root\acceptance\runtime_acceptance.json"
```

Do not lock if there is a fatal history/readback/runtime/deadline fallback or compute time exceeds the frozen budget.

## 11. Policy Lock and Final

After all ledger stages pass:

```powershell
rtc-policy-lock-v5 `
  --ledger "$Root\policy_lock\pipeline_ledger.json" `
  --artifacts "$Root\policy_lock\artifacts.json" `
  --out "$Root\policy_lock\policy_lock.json"
```

Expected contract: `WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND`.

Then generate/resume fixed Final references:

```powershell
rtc-build-baseline-cache `
  --events "$Root\contracts\event_registry_with_splits.csv" `
  --out-dir "$Root\final_baseline_cache" `
  --stage final `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --workers 16 --swmm-threads-per-process 1
```

Run locked Proposed once per untouched Final event, with no tuning from Final outcomes.

Formalize every Final run:

```powershell
rtc-formalize-run `
  --main-metadata <RUN_METADATA.json> `
  --strategy <STRATEGY> `
  --event-id <EVENT_ID> `
  --rainfall-group <RAINFALL_GROUP> `
  --out <RUN.formal_manifest.json>
```

Create `final/FINAL_RUN_INDEX.csv` with:

```text
event_id,rainfall_group,strategy,formal_manifest_path
```

Compile:

```powershell
rtc-compile-final-v4 `
  --policy-lock "$Root\policy_lock\policy_lock.json" `
  --run-index "$Root\final\FINAL_RUN_INDEX.csv" `
  --out-dir "$Root\final"
```

Final summaries give every independent rainfall group equal weight.

## 12. Resume rule

Rerun an interrupted command normally. Do not manually copy old files to force a skip.

Safe reuse requires a compatible scientific implementation contract, matching numerical inputs/configuration and verified artifact hashes. File existence alone is never sufficient.

The project is ready for scientific interpretation only after preflight, Phase-0, Step1, Step2, D2 gradient, D2+D3 ranking, real-time development, Policy Lock and untouched five-strategy Final all pass.
