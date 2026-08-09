# Lean fresh-data execution plan — RTC v0.6.5

This is the active fresh-study workflow. Historical v0.6.2/v0.6.3 runbooks are not the current execution contract.

The central rule is:

```text
prove physical SWMM control leverage
-> prove the learned action model ranks actions
-> prove MPC exploits that ranking in closed loop
-> only then spend the untouched Final budget
```

## Directory contract for Project7

Keep everything belonging to the new study under one parent directory, but do not use the git checkout itself as the fresh study workspace because `rtc-init-fresh-workspace` requires an empty root.

```text
E:\RTC_sewer\Project7\repo
E:\RTC_sewer\Project7\inputs
E:\RTC_sewer\Project7\study
E:\RTC_sewer\Project7\logs
```

`repo` is the GitHub checkout. `inputs` contains copied authoritative source assets only. `study` contains newly generated RTC evidence. Do not copy historical D0/D1/D2/D3, Step1/Step2 checkpoints, development runs, Policy Locks or Final outputs into the new study.

## Hardware target

```text
RAM: 16 GB
GPU: NVIDIA RTX 4060 8 GB
SWMM workers: normally 16 independent processes
SWMM threads/process: 1
```

If Windows starts paging or RAM remains above about 14 GB, reduce SWMM workers to 12.

Training defaults later:

```text
Step1 batch 8 / grad accumulation 2 / AMP
Step2 batch 2 / grad accumulation 4 / AMP
Step2 shard size about 128
```

---

# Step 0 — sync code, localize source inputs, validate the executable problem

## 0.1 Sync current GitHub main

```powershell
$Base = "E:\RTC_sewer\Project7"
$Repo = "$Base\repo"
$Inputs = "$Base\inputs"
$Study = "$Base\study"
$Logs = "$Base\logs"

New-Item -ItemType Directory -Force $Base,$Inputs,$Logs | Out-Null

if (-not (Test-Path "$Repo\.git")) {
  git clone https://github.com/WuJH123/RTC.git $Repo
}
cd $Repo
git fetch origin
git checkout main
git pull --ff-only origin main
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -c "import importlib.metadata; print(importlib.metadata.version('wuhan-rtc'))"
```

Expected release:

```text
0.6.5
```

Do not run `git reset --hard`, delete unrelated directories or modify source assets outside `Project7`.

## 0.2 Copy authoritative source assets into Project7

Required source inputs are:

```text
frozen Wuhan network/event source INPs and any referenced external rainfall files
event/rainfall metadata registry or source table
sensor-node definition
priority-node definition
```

The repository already contains `data/priority_nodes.txt`; copy it into `Project7\inputs\contracts` so the new study is self-contained.

If authoritative event INPs or sensor definitions currently live elsewhere under `E:\RTC_sewer`, inspect them read-only and copy the intended source files into `Project7\inputs`. Do not reuse derived RTC outputs merely because their names look compatible.

The event registry used by the new study must point its `inp_path` column to the copied `Project7\inputs` event INPs. Relative/external rainfall files referenced by those INPs must also remain resolvable after copying.

Required registry columns after split assignment:

```text
event_id
rainfall_group
inp_path
scientific_split
development_fold
```

Recommended forcing-descriptor columns, when available:

```text
total_depth_mm
duration_minutes
peak_intensity_mmhr
antecedent_rainfall_mm
```

If `scientific_split` and `development_fold` do not yet exist, create them once using:

```powershell
rtc-split-groups `
  --input "$Inputs\contracts\events_source.csv" `
  --out "$Inputs\contracts\events_with_splits.csv" `
  --seed 42
```

Then validate:

```powershell
rtc-validate-rainfall-design `
  --events "$Inputs\contracts\events_with_splits.csv" `
  --out "$Logs\rainfall_design.json"
```

## 0.3 Initialize the fresh study workspace

`$Study` must not contain old outputs before initialization.

```powershell
rtc-init-fresh-workspace `
  --root $Study `
  --inp "$Inputs\network\wuhan_v8_storage_retrofit.inp" `
  --priority "$Inputs\contracts\priority_nodes.txt" `
  --events "$Inputs\contracts\events_with_splits.csv"
```

The command copies the split registry into `$Study\contracts\event_registry_with_splits.csv` and binds source hashes.

## 0.4 INP and graph/actuator/sensor audit

```powershell
rtc-inp-audit-v2 `
  --inp "$Inputs\network\wuhan_v8_storage_retrofit.inp" `
  --priority "$Inputs\contracts\priority_nodes.txt" `
  --out "$Study\audit\inp_preflight.json"

rtc-compile-formal-assets `
  --inp "$Inputs\network\wuhan_v8_storage_retrofit.inp" `
  --priority "$Inputs\contracts\priority_nodes.txt" `
  --sensors "$Inputs\contracts\sensor_nodes.txt" `
  --out-dir "$Study\formal_assets"
```

Step 0 is complete only when the audit confirms the actual node/actuator schema, units, native controls, No-control semantics, priority mapping and sensor IDs. No ML is trained here.

---

# Step 1 — high-frequency Phase-0 D0 + small exact-SWMM D2 pilot

## 1.1 Select 6–8 development/train rainfall groups without hydraulic-outcome selection

v0.6.5 provides an executable selector. Use 8 groups when at least 8 development/train groups are available; otherwise use 6 when at least 6 are available. If fewer than 6 exist, stop and report the limitation instead of silently using Final or validation groups.

```powershell
rtc-design-phase0-events `
  --events "$Study\contracts\event_registry_with_splits.csv" `
  --out "$Study\phase0\events.csv" `
  --groups 8 `
  --development-fold train `
  --seed 42
```

When forcing descriptors exist the selector uses deterministic farthest-point coverage in forcing space. If they do not exist, it records a seeded group-only fallback. Hydraulic outcomes are never used for Phase-0 cohort selection.

## 1.2 Generate high-frequency No-control D0

```powershell
rtc-run-d0-batch `
  --events "$Study\phase0\events.csv" `
  --strategy no_control `
  --out-dir "$Study\phase0\d0" `
  --record-stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1
```

No-control means native supervisory `[CONTROLS]` disabled, no Python setting writes, while intrinsic/local equipment physics remains.

## 1.3 Design four hydraulic checkpoints/event

```powershell
rtc-design-checkpoints `
  --run-index "$Study\phase0\d0\D0_no_control_RUN_INDEX.csv" `
  --out "$Study\phase0\checkpoints.csv" `
  --checkpoints-per-event 4 `
  --minimum-elapsed-minutes 30 `
  --seed 42
```

## 1.4 Design budgeted D2 probes

```powershell
rtc-design-probes-efficient `
  --inp "$Inputs\network\wuhan_v8_storage_retrofit.inp" `
  --checkpoints "$Study\phase0\checkpoints.csv" `
  --out "$Study\phase0\d2_manifest.csv" `
  --epsilon 0.15 `
  --actuators-per-checkpoint 12 `
  --seed 42
```

For 6 groups x 4 checkpoints x 12 locally probed actuators, the unique SWMM burden is about 600 branches after center-action deduplication. Every branch still contains the complete actuator-setting vector; this sampling budget is not runtime Top-K.

## 1.5 Run exact-prefix D2 branches

Start with a 120 min horizon:

```powershell
rtc-run-probes `
  --manifest "$Study\phase0\d2_manifest.csv" `
  --out-dir "$Study\phase0\d2_h120" `
  --horizon-minutes 120 `
  --stride-seconds 60 `
  --workers 16 `
  --swmm-threads-per-process 1
```

Every branch must replay and verify the exact saved No-control prefix before any candidate action write.

---

# Step 2 — timing/control-leverage diagnosis and evidence-bound timing freeze

## 2.1 Run Phase-0 step-response timing analysis

```powershell
rtc-phase0-timescale `
  --manifest "$Study\phase0\d2_manifest.csv" `
  --run-summary "$Study\phase0\d2_h120\RUN_SUMMARY.csv" `
  --detail-out "$Study\phase0\timescale_detail.csv" `
  --summary-out "$Study\phase0\timescale_summary.json" `
  --max-sample-seconds 60
```

If the command exits with code 2 because `horizon_censored=true`, that is a scientific signal rather than a code failure. Re-run the D2 pilot with a longer horizon (normally 180 min, then at most 240 min for this pilot) into a new output directory, then rerun the timing analysis against that summary. Do not freeze timing from a horizon-censored report.

## 2.2 Quantify exact SWMM control leverage

```powershell
rtc-control-leverage-audit `
  --d2-manifest "$Study\phase0\d2_manifest.csv" `
  --d2-run-summary "$Study\phase0\d2_h120\RUN_SUMMARY.csv" `
  --out "$Study\phase0\control_leverage.json"
```

If a longer D2 horizon was required, point this command at that latest run summary.

Interpretation:

```text
PROMISING_CONTROL_LEVERAGE
    proceed to production-grid Step1/Step2 data

WEAK_OR_STATE_DEPENDENT_CONTROL_LEVERAGE
    expand the pilot across more flooded/high-depth development states before large Step2 generation

LITTLE_MEASURABLE_CONTROL_LEVERAGE_IN_PILOT
    stop large ML generation and inspect actuator semantics/control space/hydraulic response first
```

This report is diagnostic, not a Formal hard gate.

## 2.3 Freeze timing

Use the measured report to choose the production cadence explicitly. The current candidate is:

```text
model step = 300 s
control update = 600 s
history = 13 frames -> 60 min causal span from t=0
prediction horizon = 120 min -> 24 model steps / 12 control blocks
first control = 60 min
```

Do not freeze these values blindly if the high-frequency response evidence contradicts them. The selected horizon must not be censored and should cover the relevant network response time. Round timing to valid model/control grids.

When the 300/600/13/120/60 design remains justified, write the timing-only resolved contract:

```powershell
rtc-freeze-phase0-timing `
  --phase0-summary "$Study\phase0\timescale_summary.json" `
  --out "$Study\contracts\timing_resolved.json" `
  --model-step-seconds 300 `
  --control-update-seconds 600 `
  --history-steps 13 `
  --horizon-minutes 120 `
  --control-start-minutes 60
```

Only add `--max-setting-delta-per-update` when supported by engineering metadata/runtime evidence. Otherwise leave it null at this timing stage.

`timing_resolved.json` is deliberately **not** a complete production policy config. It is the frozen data/controller clock contract for production D0/D1 and D3 design. Forecast, optimizer, near-optimal TFV secondary preferences and runtime/readback fields are resolved later before closed-loop development/Policy Lock.

Step 2 is complete only when:

```text
Phase-0 event selection provenance exists
high-frequency D0 exists
D2 exact-prefix branches exist
Phase-0 timing report is not horizon-censored
exact-SWMM control-leverage report exists
timing_resolved.json validates causal/model/control/D3 alignment
```

---

# Later stages — implementation status and intended order

The repository already contains executable code for the remaining stages:

```text
Step 3: production D0 + D1 -> Step1 index/train/held-out acceptance
Step 4: production budgeted D2 + small D3 -> Step2 shards/train/held-out acceptance
Step 5: D2 gradient + D2/D3 ranking/regret evidence
Step 6: causal closed-loop Proposed + No-control/Internal/Auto-RBC/EFD development comparison
Step 7: runtime/readback/deadline acceptance
Step 8: Policy Lock
Step 9: untouched seven-strategy authoritative SWMM Final + pairwise/event-balanced reporting
```

Do not launch Policy Lock or Final merely because the code path exists. First obtain a development controller that actually changes facility settings over time, completes decisions within the control interval and is not the worst TFV strategy among No-control, Internal RTC, Auto-RBC and EFD.
