# Lean Fresh Run v0.6.6 — Wuhan RTC

This is the active fresh-run contract after the Project7 pre-training audit. It supersedes the v0.6.5 runbook for new evidence.

## Scientific invariants

- Primary objective: minimize authoritative-SWMM system TFV.
- PFV/priority depth: soft/reporting only.
- Global Peak: report-only.
- Proposed: sparse causal observations -> Step1 -> differentiable Step2 -> continuous receding-horizon MPC.
- All 109 SWMM-writable actuators remain eligible in the simulation action space.
- Default actuation claim is `SWMM_MODEL_CONTINUOUS_SIMULATION_ONLY`; no field VFD/binary claim without engineering metadata.
- No Final hydraulic truth enters pre-lock training/tuning.
- Internal RTC is a strong native comparator, not an equal-information/equal-frequency comparator.

## Required input correction before Step 0

Use the event-specific INPs as the authoritative source for rainfall, DWF and initial conditions. Do not use the frozen network INP as an event forcing source.

Prepare a new v0.6.6 event suite. For this large Wuhan Dynamic-Wave model the first engineering attempt uses a **6 h dry/DWF hydraulic warm-up**, while Step1 still uses only the latest 60 min causal history:

```powershell
rtc-prepare-event-suite `
  --events E:\RTC_sewer\Project7\inputs\contracts\events_with_splits.csv `
  --out-dir E:\RTC_sewer\Project7\inputs\prepared_v066\events `
  --out-registry E:\RTC_sewer\Project7\inputs\prepared_v066\events_with_splits.csv `
  --warmup-minutes 360 `
  --post-rain-tail-minutes 360
```

The preparation contract keeps each storm at the same absolute clock and therefore preserves its DWF phase. It moves only the simulation/report start earlier, canonicalizes rain time-series rows to explicit absolute timestamps and extends the simulation end. It does not change rainfall intensities or DWF values.

`warmup_minutes` and `history_span_minutes` are different quantities. Six hours is a conservative first initialization attempt, **not proof of dry-weather convergence**. Before large D2/production data, compare the No-control hydraulic state at storm onset under 6 h versus a longer 12 h warm-up on a small forcing-only development/train pilot. If the storm-onset state is still materially dependent on warm-up length, test 24 h and adopt the shortest duration for which the hydraulic state is insensitive enough for the study. If the warm-up changes, re-prepare the full event registry and start a new fresh study root; do not mix evidence from different event clocks.

Create/use:

```text
configs/sensor_layout_provenance.project7.v1.json
configs/rainfall_provenance.project7.v1.json
configs/actuator_scope.v1.json
```

If the active Project7 sensor file is outside the repo, copy the repo provenance JSON into the study contract directory and verify its SHA still matches the sensor file.

Then run:

```powershell
rtc-check-study-readiness `
  --events E:\RTC_sewer\Project7\inputs\prepared_v066\events_with_splits.csv `
  --frozen-inp E:\RTC_sewer\Project7\inputs\network\wuhan_v8_storage_retrofit.inp `
  --sensors E:\RTC_sewer\Project7\inputs\contracts\sensor_nodes.txt `
  --sensor-provenance E:\RTC_sewer\Project7\repo\configs\sensor_layout_provenance.project7.v1.json `
  --rainfall-provenance E:\RTC_sewer\Project7\repo\configs\rainfall_provenance.project7.v1.json `
  --actuator-scope E:\RTC_sewer\Project7\repo\configs\actuator_scope.v1.json `
  --history-span-minutes 60 `
  --minimum-post-rain-tail-minutes 360 `
  --out E:\RTC_sewer\Project7\logs\study_readiness_v066.json
```

Do not enter production training if readiness fails.

## Internal RTC event pairing

Internal RTC must never use a controls-free event source directly. For any direct policy run:

```text
--inp <prepared event INP>
--native-controls-template <frozen network INP>
```

The runtime contains the prepared event rainfall/DWF/initial state plus the frozen `[CONTROLS]` payload. Formalization hashes that controls payload and Final verifies it against the Policy-Locked frozen INP.

For baseline caches use:

```text
rtc-build-baseline-cache ... --frozen-inp <frozen network INP>
```

## Step 0 — fresh v0.6.6 workspace and source audit

Create a new study root, for example:

```text
E:\RTC_sewer\Project7\study_v066
```

Do not delete the v0.6.5 study.

Run fresh workspace initialization, INP preflight, rainfall-group validation and formal asset compilation using the **prepared v0.6.6 event registry**. Record the passed readiness JSON as a study artifact.

## Step 1 — Phase-0 physical pilot

Select 8 development/train rainfall groups by forcing descriptors only.

Run high-frequency No-control D0 at 60 s. With the default 360 min warm-up, rainfall begins at elapsed 360 min. Hydraulic action-effect checkpoints must be selected after rainfall onset; the first useful flooded-state checkpoint should normally be no earlier than `rainfall_onset + 30 min` (390 min for the default preparation), not simply 30 min after simulation start. Reserve enough future tail for the candidate horizon.

Before expensive D2, use one or a few development/train D0 runs to compare storm-onset hydraulics under 360 min versus 720 min warm-up. This is an initialization sensitivity check, not an outcome-based rainfall selection. If the difference is still material, repeat with 1440 min and regenerate the prepared suite using the selected initialization duration.

Design rotating D2 local probes with 12 actuators/checkpoint, full 109-dimensional candidate vectors and exact No-control prefix replay.

## Step 2 — response timing and control leverage

The old h180 evidence is scientifically stale under the v0.6.6 implementation/event-preparation contract and must not be reused as Formal evidence.

Start the new pilot at 210 min because the earlier audit found p90 network-flood/depth peaks around 160–162.5 min and h180 was censored. If h210 is still censored, test the next grid-aligned horizon (normally 220 or 240 min) using a forcing-only development/train timing cohort with sufficient future tail. Never lower the censor threshold to force a pass.

Run exact-SWMM control leverage and freeze timing only when `horizon_censored=false`.

The likely starting clocks are:

```text
hydraulic warm-up = 360 min first attempt; evidence may extend it
model step = 300 s
control update = 600 s
Step1 history = 13 frames = 60 min
first Python supervisory control = 60 min after simulation start
rainfall onset = warm-up duration (360 min in the first prepared suite)
```

Starting the Python policies after the first complete 60 min history means they operate causally during most of the dry/DWF warm-up, as a continuously running RTC would. Internal RTC remains a deliberately strong native comparator and may evaluate its native rules from simulation start. The prediction horizon is evidence-dependent and is not the same as the whole-event control duration.

Also inspect recovery at the 6 h post-rain tail. If the development pilot remains right-censored, re-prepare the event suite with a 12 h tail **before production training**, create a new study root and repeat Step 0–2. Do not call a censored endpoint recovered.

## Step 3 — production D0/D1 and Step1

After a passed timing freeze:

- production No-control D0: development/train + development/validation;
- D1 controlled exploration: development/train only;
- Step1 train: train D0 + D1;
- Step1 validation: development/validation D0 only;
- no rainfall-group leakage.

Use event-balanced unobserved-node depth/state metrics and wet/high-depth subsets. Do not advance on a cosmetically low aggregate RMSE if held-out hydraulic reconstruction is poor.

## Step 4 — production D2/D3 and Step2

First campaign target:

```text
24–32 development/train rainfall groups
8–12 development/validation rainfall groups
4 checkpoints/event
16–24 rotating D2 actuators/checkpoint
4 random D3 joint sequences + 1 hold/checkpoint
```

Every D2/D3 action is still a full 109-setting vector. D3 horizon and control blocks must come from the frozen timing contract.

Train Step2 on future hydraulic trajectories, actuator flows and exact cumulative SWMM flooding-volume truth.

## Step 5 — gradient and ranking acceptance

Require action-effect evidence, not only state RMSE:

- local/boundary D2 finite-difference direction agreement;
- delta-TFV sign accuracy;
- candidate rank correlation;
- joint D2/D3 best-action regret;
- event-balanced held-out results.

Do not enter MPC if Step2 cannot rank actions reliably.

## Step 6 — development closed-loop comparison

Use development events only. Run:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
```

All-open/all-closed can be run as diagnostics but are not competitive success criteria.

For Internal RTC always pair the prepared event with the frozen native-controls template. Disclose that Internal uses native true-state rules at the native rule step, Auto-RBC uses actuator-adjacent true depths and EFD uses controlled-storage true depths. Do not claim equal information budgets.

Development success requires real time-varying Proposed actions, runtime within the supervisory interval, correct readback and authoritative-SWMM TFV that is not systematically worst among meaningful comparators.

## Step 7 — runtime/readback/deadline acceptance

Check:

- decision latency distribution and worst case < control interval;
- requested target -> SWMM target/current readback;
- no silent actuator ordering change;
- deterministic fallback;
- no NaN/invalid action;
- generated runtime/event/native-rule lineage hashes.

## Step 8 — Policy Lock

Policy Lock must bind the v0.6.6 study-readiness artifact in addition to the existing models/config/graph/split/timing/acceptance/runtime artefacts.

The baseline plan contract must be:

```text
FORMAL_BASELINE_PLAN_V6_EVENT_PAIRED_INFORMATION_DISCLOSED
```

If actuation scope remains simulation-only, Policy Lock must state `field_deployment_claim=false`.

## Step 9 — untouched Final

Run every locked Final event under exactly seven strategies:

```text
proposed
no_control
internal_rtc
auto_rbc
efd
all_open
all_closed
```

Final verifies:

- prepared scientific event identity;
- physical network identity;
- SWMM engine lineage;
- Proposed model/config hashes;
- Internal native controls payload hash against the frozen INP;
- model/control cadence;
- complete event × strategy matrix.

Aggregate first within rainfall group and then weight independent rainfall groups equally.

Primary comparisons are Proposed vs No-control/Internal/Auto-RBC/EFD. Report All-open/All-closed as diagnostic extremes.

## Stop conditions

Stop before the next expensive stage when any of the following is unresolved:

```text
prepared event/readiness contract fails
pre-rain hydraulic state remains materially warm-up-dependent
native Internal rule pairing fails
response horizon remains censored
recovery tail remains censored and has not been extended
Step1 held-out acceptance fails
Step2 action ranking/gradient acceptance fails
runtime/readback acceptance fails
Policy Lock hashes do not match
```

Never solve a physical/source-data problem by weakening a guard or by generating more neural-network data.
