# Lean fresh-data execution plan — RTC v0.6.3

Use this plan when previous derived RTC data cannot be trusted. Start from source event INPs, event/rainfall metadata, sensor definitions and the current repository only. Generate all hydraulic trajectories, learned datasets, models and evaluation evidence again.

The goal is to discover failure cheaply before committing to a large SWMM data campaign.

## Hardware target

```text
System RAM: 16 GB
GPU: NVIDIA RTX 4060 8 GB
SWMM CPU parallelism: up to 16 independent simulations
```

Recommended execution:

```text
SWMM workers = 16
SWMM threads per process = 1
```

If observed RAM use repeatedly exceeds about 14 GB or Windows begins paging, reduce to 12 workers. Do not give each of 16 independent SWMM processes multiple internal threads.

Training defaults for the target GPU:

```text
Step1: batch-size 8, grad-accum 2, AMP on
Step2: batch-size 2, grad-accum 4, AMP on
```

If Step2 CUDA memory has comfortable headroom, test batch-size 4 / grad-accum 2. If it OOMs, use batch-size 1 / grad-accum 8. Do not reduce horizon merely to fit VRAM after the hydraulic horizon has been selected.

Keep Step2 shard size around 128 to limit system-RAM pressure.

---

## Stage 0 — fresh root and code self-test

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -c "import importlib.metadata; print(importlib.metadata.version('wuhan-rtc'))"
```

Expected release after this audit:

```text
0.6.3
```

Use a new workspace, for example:

```powershell
$Repo     = "E:\RTC_sewer\RTC"
$Root     = "E:\RTC_sewer\RTC_fresh_v063"
$Inp      = "E:\RTC_sewer\data\wuhan_v8_storage_retrofit.inp"
$Events   = "E:\RTC_sewer\contracts\events_v063.csv"
$Priority = "$Repo\data\priority_nodes.txt"
$Sensors  = "E:\RTC_sewer\contracts\sensor_nodes.txt"
```

Do not copy D0/D1/D2/D3, Step1, Step2 or acceptance/model artifacts from an older workspace.

Run workspace/preflight and compile the graph before expensive generation.

### Data created

```text
fresh workspace manifest
event/split registry
INP audit
graph/state/actuator schema
```

### Expected learning/result

No ML yet. Confirm what the code is actually controlling: node count, discovered actuator count/types, sensor mapping, units and writable setting schema.

---

## Stage 1 — small Phase-0 timing pilot

Do **not** start with all rainfall events.

Select about 6–8 development rainfall groups using forcing descriptors only, spanning short/long and lower/higher intensity patterns when those descriptors exist. Never select Final events using hydraulic outcomes.

Generate t=0-inclusive No-control trajectories at a high observation rate, normally 60 s:

```powershell
rtc-run-d0-batch `
  --events <PHASE0_DEVELOPMENT_EVENTS.csv> `
  --strategy no_control `
  --out-dir "$Root\phase0\d0" `
  --record-stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1

rtc-design-checkpoints `
  --run-index "$Root\phase0\d0\D0_no_control_RUN_INDEX.csv" `
  --out "$Root\phase0\checkpoints.csv" `
  --checkpoints-per-event 4 `
  --minimum-elapsed-minutes 30
```

Use efficient local probes rather than exhaustive all-actuator probes:

```powershell
rtc-design-probes-efficient `
  --inp $Inp `
  --checkpoints "$Root\phase0\checkpoints.csv" `
  --out "$Root\phase0\d2_manifest.csv" `
  --epsilon 0.15 `
  --actuators-per-checkpoint 12 `
  --seed 42

rtc-run-probes `
  --manifest "$Root\phase0\d2_manifest.csv" `
  --out-dir "$Root\phase0\d2" `
  --horizon-minutes 120 `
  --stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1
```

With 6 events × 4 checkpoints × 12 probed actuators, the unique D2 burden is roughly:

```text
24 checkpoints × (1 center + about 24 +/- branches)
≈ 600 independent SWMM branches
```

rather than probing every actuator at every checkpoint.

Run the existing timescale analysis and also the new leverage diagnosis:

```powershell
rtc-phase0-timescale `
  --manifest "$Root\phase0\d2_manifest.csv" `
  --run-summary "$Root\phase0\d2\RUN_SUMMARY.csv" `
  --detail-out "$Root\phase0\timescale_detail.csv" `
  --summary-out "$Root\phase0\timescale_summary.json" `
  --max-sample-seconds 60

rtc-control-leverage-audit `
  --d2-manifest "$Root\phase0\d2_manifest.csv" `
  --d2-run-summary "$Root\phase0\d2\RUN_SUMMARY.csv" `
  --out "$Root\phase0\control_leverage.json"
```

### Data created

```text
high-frequency No-control hydraulic trajectories
replayable hydraulic checkpoints
small exact-SWMM local action-effect branches
exact node flood-volume deltas
actuator-flow response trajectories
```

### Expected learning/result

Freeze a plausible model step, control update interval and prediction horizon from hydraulic response timing.

More importantly, decide whether exact SWMM shows a real TFV response to control. This is a diagnosis, not a hard software gate.

If almost no sampled checkpoint has a meaningful exact TFV improvement, stop large Step2 generation and inspect the hydraulic/action definition first.

---

## Stage 2 — freeze only the timing needed for production data

After Phase-0, create one `controller_resolved.json` with:

```text
model_step_seconds
control_update_seconds
control_start_minutes
history_steps
horizon_steps
optimizer_iterations
optimizer_learning_rate
optional max_setting_delta_per_update
```

Do not pre-register dozens of secondary thresholds at this stage. The minimum engineering relations are:

```text
control_update_seconds % model_step_seconds == 0
record stride == model step
first control occurs after a full causal history
prediction horizon covers at least one control interval
D3 horizon contains whole control blocks
```

The current online action set is non-empty because hold-current is always available.

---

## Stage 3 — production-grid D0 and D1 for Step1

Generate new model-step No-control trajectories for development train and validation groups. One SWMM run per event is cheap compared with D2/D3, so use broader rainfall coverage here.

A good first campaign is approximately:

```text
Step1 train: 40–60 independent rainfall groups
Step1 validation: 10–15 independent rainfall groups
```

Expand only if held-out reconstruction remains rainfall-pattern dependent.

Generate D1 only for development/train groups:

```powershell
rtc-run-d1-batch `
  --events <DEVELOPMENT_TRAIN_EVENTS.csv> `
  --sensors $Sensors `
  --out-dir "$Root\d1" `
  --seed 42 `
  --model-step-seconds $ModelStep `
  --control-update-seconds $ControlUpdate `
  --control-start-minutes $ControlStart `
  --perturbation-std 0.12 `
  --change-probability 0.35 `
  --max-delta 0.20 `
  --workers 16 `
  --swmm-threads-per-process 1
```

Compile Step1 index from fresh No-control baseline D0 plus D1 exploration. D1 is not needed in validation.

### Data created

Each trajectory contains a regular time series of:

```text
full six-channel SWMM node state
node rainfall forcing
actuator target setting
actuator current setting
actuator flow
```

Step1 converts these trajectories into causal history windows internally; there is no need to materialize a giant flat window CSV.

### Expected learning/result

Learn:

```text
sparse observed hydraulic history
+ causal rainfall/actuator context
+ graph/static features
-> current full-network hydraulic state
```

The main check is unobserved-node depth/state reconstruction on rainfall-group-disjoint validation events.

---

## Stage 4 — Step1 training

Recommended first configuration on RTX 4060 8 GB:

```powershell
rtc-train-step1-large `
  --workspace-manifest "$Root\FRESH_WORKSPACE_MANIFEST.json" `
  --run-index "$Root\step1\train_run_index.csv" `
  --graph "$Root\formal_assets\graph_schema.npz" `
  --sensors $Sensors `
  --history-steps $HistorySteps `
  --model-step-seconds $ModelStep `
  --epochs 30 `
  --batch-size 8 `
  --grad-accum 2 `
  --out "$Root\models\step1.pt"
```

If GPU memory is tight, use batch 4 / accumulation 4. AMP is enabled by default on CUDA.

### Expected result

Do not ask whether every hydraulic variable is perfect. Ask whether the reconstructed current state is accurate enough that changing the state estimator does not substantially change the action ranking produced by Step2/MPC later.

---

## Stage 5 — first production-grid action-effect campaign

This stage should remain **progressive**.

Start with roughly:

```text
24–32 development/train rainfall groups
8–12 development/validation rainfall groups
4 hydraulic checkpoints per event
16–24 locally probed actuators per checkpoint
4 random D3 sequences + 1 hold reference per checkpoint
```

For D2 use:

```powershell
rtc-design-probes-efficient `
  --inp $Inp `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --out "$Root\d2\probe_manifest.csv" `
  --epsilon 0.15 `
  --actuators-per-checkpoint 24 `
  --seed 42
```

The deterministic rotation makes every discovered actuator eligible for sampling across checkpoints. Do not interpret the per-checkpoint probe budget as a runtime active-facility subset.

Run D2 at the frozen production model step/horizon with 16 processes × 1 SWMM thread.

Design a small D3 set after the controller timing is frozen:

```powershell
rtc-design-d3 `
  --inp $Inp `
  --checkpoints "$Root\checkpoints\checkpoint_settings.csv" `
  --controller-config "$Root\contracts\controller_resolved.json" `
  --out "$Root\d3\sequence_manifest.csv" `
  --sequences-per-checkpoint 4 `
  --perturbation-std 0.20 `
  --change-probability 0.25 `
  --seed 42
```

`rtc-design-d3` also includes a hold reference.

Run D3 with 16 independent SWMM processes and one SWMM thread per process.

Immediately run the leverage audit again including D3:

```powershell
rtc-control-leverage-audit `
  --d2-manifest "$Root\d2\probe_manifest.csv" `
  --d2-run-summary "$Root\d2\RUN_SUMMARY.csv" `
  --d3-run-summary "$Root\d3\D3_RUN_SUMMARY.csv" `
  --out "$Root\diagnostics\control_leverage_production_grid.json"
```

### Data created

D2 learns local single-facility action effects; D3 adds coupled multi-facility/multi-step action effects.

Every branch stores:

```text
checkpoint full state
future rainfall sequence
full actuator setting sequence
previous actuator flow
future full hydraulic state trajectory
future actuator flow trajectory
exact SWMM cumulative node flood-volume delta
```

### Expected learning/result

Before training Step2, exact SWMM should show non-trivial TFV variation between actions at at least a meaningful subset of flooded checkpoints. If all candidate TFVs are practically identical, the correct response is not “generate ten times more data”.

---

## Stage 6 — Step2 sharding and training

Compile train and validation shards separately. Use shard size about 128 for 16 GB RAM.

Recommended first training run:

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

### Expected learning/result

Step2 must learn three things together:

```text
future hydraulic state trajectory
future actuator-flow trajectory
cumulative flood-volume consequence of the setting sequence
```

For realtime control, prioritize held-out evidence in this order:

1. D3 joint sequence TFV ranking / selected regret;
2. D2 local TFV gradient sign/ranking;
3. exact TFV error;
4. managed-flow error;
5. hydraulic-state error and physical-plausibility diagnostics.

Small state RMSE with wrong action ranking is not a useful control surrogate.

If Round-1 Step2 is weak, add **targeted** data instead of blindly doubling everything:

- rainfall/hydraulic strata with high validation error;
- actuators with uncertain or inconsistent action effect;
- checkpoints where D3 joint ranking is poor.

---

## Stage 7 — development closed-loop SWMM before any large Final

Run Proposed on several development validation events and compare first against:

```text
no_control
internal_rtc
```

All-open/all-closed are useful diagnostics but are not needed to decide whether the learned controller is worth continuing.

Inspect:

```text
actual authoritative SWMM TFV difference
fraction of decisions using MPC vs fallback
active_actuator_count_1e4
setting_change_l1 / setting_change_max
decision runtime
readback failures
predicted TFV vs actual realized direction
```

The v0.6.3 production MPC retains its best optimization iterate and refuses to execute a candidate that does not even beat hold/fallback in predicted TFV.

### Expected result

A useful development controller should satisfy all of the following qualitatively:

- not remain at hold for essentially every flooded decision;
- not move every actuator indiscriminately at every decision;
- complete each decision before the control interval expires;
- improve authoritative SWMM TFV on a meaningful fraction of development events;
- avoid systematic deterioration versus no-control.

If exact leverage was strong and Step2 ranking was good but closed-loop SWMM still fails, focus on forecast/horizon/optimizer behavior rather than generating more random training data.

---

## Stage 8 — only now expand baselines and Final evidence

Once development closed-loop value is demonstrated, generate the complete fixed comparison matrix and untouched Final data using the repository's Formal/Policy-Lock tools.

Do not spend the full five-strategy Final budget merely to discover that the controller never learned a useful action direction.

---

# Fresh-data inventory

When previous derived results are not trusted, regenerate all of the following:

| Stage | Fresh data/model | Purpose |
| --- | --- | --- |
| 0 | graph / actuator / sensor / event contracts | define current executable problem |
| 1 | high-frequency D0 | response timing and replay prefix |
| 1 | small D2 pilot | exact physical action leverage |
| 3 | production-grid D0 | uncontrolled hydraulic state coverage and replay prefixes |
| 3 | D1 exploration | controlled state-space coverage for Step1 |
| 4 | Step1 model | sparse history -> current full state |
| 5 | budgeted D2 | local setting/action effect |
| 5 | small D3 | joint/multi-step interaction effect |
| 6 | Step2 shards/model | differentiable hydraulic/action-effect world model |
| 7 | development closed-loop runs | verify realtime optimizer + surrogate + forecast together |
| 8 | complete baselines + untouched Final | final scientific comparison |

Do not reuse an old D0/D1/D2/D3 trajectory, Step1/Step2 checkpoint or Policy Lock merely because its filename looks compatible.

# The central decision rule

```text
First prove that SWMM itself contains useful control leverage.
Then prove the surrogate learns the action ranking.
Then prove MPC can exploit it in closed loop.
Only then scale the experiment.
```

This is both cheaper and scientifically stronger than generating a huge dataset first and diagnosing controllability after training.
