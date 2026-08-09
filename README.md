# Wuhan RTC — large-system TFV-first framework

Research code for **sparse-sensing, differentiable-surrogate, state-adaptive real-time control of a large SWMM drainage network**.

The controller does **not** use rainfall-event IDs, rainfall-specific lookup schedules, a fixed active actuator subset, a pre-enumerated finite action library, or hard-binary pump assumptions. At each control update it reconstructs the current hydraulic state and decides online **which facilities are worth controlling, by how much, and when**.

---

## 1. Scientific objective

```text
Sparse hydraulic sensors + realised causal rainfall + actuator readback
                           |
                           v
                 Step 1: state reconstruction
                           |
                           v
                 full current hydraulic state
                           |
                 causal rainfall scenarios
                           |
                           v
       Step 2: differentiable hydraulic world model
       setting -> facility flow -> network trajectory
                           |
                           v
          Step 3: continuous receding-horizon MPC
          every writable actuator remains eligible
                           |
               PRIMARY: minimise cumulative TFV
                           |
          SECONDARY: reduce priority-site deterioration
            within a TFV-near-optimal solution set
                           |
                           v
              execute first block + readback
                           |
                           +---- observe and repeat
```

### TFV/PFV contract

- **TFV is the primary control objective.** It is cumulative flooding volume over the defined prediction horizon/event, summed over all nodes.
- **PFV is not a hard admission constraint.** The eight observed ponding locations remain site-wise diagnostics and a soft lexicographic secondary preference. Limited PFV deterioration may be accepted when necessary for a meaningful system-wide TFV reduction.
- `Node.flooding` is an instantaneous flow rate. It is never labelled PFV/TFV by itself.
- Formal PFV/TFV truth comes from cumulative SWMM node statistics over the exact event/horizon; sampled-rate integration is surrogate/diagnostic only.
- Global Peak is reported separately and is not a hard MPC condition.

---

## 2. Uploaded Wuhan V8 physical contract

The user-supplied `wuhan_v8_storage_retrofit(2).inp` was audited directly on 2026-08-09. See [`docs/WUHAN_V8_UPLOADED_INP_AUDIT_2026-08-09.md`](docs/WUHAN_V8_UPLOADED_INP_AUDIT_2026-08-09.md).

Key facts from that exact file:

- `FLOW_UNITS = CMS`, `FLOW_ROUTING = DYNWAVE`;
- 932 hydraulic nodes, 1,167 conduits, 3,731 subcatchments;
- 57 pumps + 42 orifices + 10 weirs = **109 eligible continuous actuators**;
- 82/109 actuators are affected by native `[CONTROLS]`, including all 57 pumps;
- source `ROUTING_STEP = 15 s`, `RULE_STEP = 10 s`, `REPORT/WET_STEP = 5 min`;
- the eight IDs currently stored in the historical `data/priority_nodes.txt` are **0/8 present in this uploaded INP**.

Therefore Formal priority/PFV evidence is blocked until the eight observed ponding sites are mapped to valid nodes in this exact model. The code fails fast instead of guessing IDs.

---

## 3. Historical local data: what should actually be reused?

Correctness has priority over historical-data reuse. The default policy is:

> **reuse forcing/physical assets when lineage is provable; regenerate authoritative hydraulics, action-effect labels and baseline evidence under the current contracts.**

The following assets were identified in the earlier Project6 workflow.

| Historical local asset | Current reuse decision | Allowed role now |
|---|---|---|
| `E:\RTC_sewer\Project6\data\wuhan_v8_storage_retrofit.inp` | **Reuse only after SHA / physical-network hash matches the current frozen INP.** | Physical source model. Do not rely on filename equality alone. |
| `E:\RTC_sewer\Project6\outputs\rainfall_library_v8_storage_variablepump\rainfall_event_table.csv` and `rainfall_event_table.formal_adapter.json` | **Reusable as rainfall/forcing inventory.** | Build the new event registry and rainfall-group split. Historical records identified a 36-event `formal_blind_v33` subset (`084–119`); keep any still-designated blind/final groups out of training. |
| `E:\RTC_sewer\project5_json\Train1600` (~1600 historical instances; 64 events; flow/depth/flood/settings/metadata) | **Do not directly admit to Formal.** | Optional auxiliary Step1 pretraining/smoke only after explicit physical hash, units, causal feature and sensor-schema audit. Regenerating compact D0/D1 is preferred. |
| `E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4` (historically ~3000 instances; later multiple 81/84-rainfall-group/candidate assets) | **Do not reuse for current Step2/MPC by default.** | Offline audit only. Old action-effect labels were produced under earlier No-control/control-subset/objective contracts, and later Project6 Step2 action-effect truth was not sufficient to authorize gradient MPC. |
| Historical V4 action-effect / Train1600 V3 / Pilot V3 / Fast Core (~4481 candidate rows) / Round3–5 / 7908-candidate assets | **Not Formal training data unless re-admitted by lineage.** | Diagnostics, sensitivity comparisons, or optional Step1-only auxiliary material after audit. Do not mix them into current D2/D3 Step2 shards automatically. |
| Old No-control trajectories/metrics | **Regenerate.** | Historical Project6 had a known No-control semantic failure; current No-control is controls-disabled + no Python writes. |
| Old Internal-RTC trajectories | Prefer **regeneration in the baseline cache**. | They can only be salvaged if original native `[CONTROLS]`, forcing, engine/runtime lineage and no-Python-write semantics are all proven. Regeneration is safer and cheap relative to D2/D3. |
| Old Step1/Step2 model checkpoints | **Do not use as Formal locked models.** | Architecture/state schema/actuator physics/context/objective have changed. They may be retained only as historical diagnostics. |

The repository does **not** assume that old Project6 folders are absent. They may remain on disk, but the current pipeline will not silently ingest them.

---

## 4. Fixed baseline policy semantics

These strategies must never be conflated:

- **Internal-RTC**: event INP with native `[CONTROLS]` retained; **no Python setting writes**.
- **No-control**: same physical network/forcing with native `[CONTROLS]` disabled; **no Python setting writes**. It is not all-open.
- **Hold**: controls-disabled common prefix; at the first common control decision freeze the observed actuator readback vector for the rest of the event.
- **All-open**: controls-disabled common prefix; from the first common control decision command every eligible setting to `1.0` and keep it there.
- **All-closed**: controls-disabled common prefix; from the first common control decision command every eligible setting to `0.0` and keep it there.
- **Proposed** is not a fixed baseline and is never placed in the fixed-baseline cache.

The frozen plan is [`configs/formal_baseline_plan.v3.json`](configs/formal_baseline_plan.v3.json).

### Why the common pre-control prefix matters

Proposed needs a causal history before its first decision. Hold/All-open/All-closed therefore use the same controls-disabled prefix until the frozen `control_start_minutes`, so differences after that epoch are attributable to the policy rather than to different initial histories. No-control remains uncontrolled for the entire event; Internal-RTC follows the original native rules for the entire event.

---

## 5. Generate each baseline once per rainfall event

Use the baseline cache instead of rerunning baselines inside Step1, development evaluation and Final.

```powershell
rtc-build-baseline-cache `
  --events outputs/contracts/event_registry_with_splits.csv `
  --config configs/formal_controller_v4.json `
  --out-dir outputs/baseline_cache `
  --stage prelock `
  --workers 16 `
  --swmm-threads-per-process 1
```

For an early Phase-0 pilot, it is acceptable to generate only the two physically important reference baselines:

```powershell
rtc-build-baseline-cache `
  --events outputs/contracts/event_registry_with_splits.csv `
  --config configs/development_controller_resolved.json `
  --out-dir outputs/baseline_cache `
  --stage prelock `
  --strategies no_control,internal_rtc `
  --workers 16 `
  --swmm-threads-per-process 1
```

The cache key includes:

- complete event-INP SHA-256;
- forcing/control-independent physical-network SHA-256;
- strategy ID;
- model/observation step;
- control update interval;
- record stride;
- common control-start epoch;
- SWMM threads/process;
- baseline-cache contract version.

If none of these change, a second command resumes the existing result instead of rerunning SWMM. If one changes, the cache is invalidated **on purpose** rather than reusing scientifically incompatible evidence.

For `--stage prelock`, rows assigned to `final` are not generated.

After Policy Lock, Final baselines are generated once with the locked controller timing and locked baseline plan:

```powershell
rtc-build-baseline-cache `
  --events outputs/contracts/event_registry_with_splits.csv `
  --out-dir outputs/final_baseline_cache `
  --stage final `
  --policy-lock outputs/policy_lock/policy_lock.json `
  --workers 16 `
  --swmm-threads-per-process 1
```

Final baseline cache generation also creates each baseline's Formal manifest and exact routing-step Global Peak replay. It does not rerun Proposed.

### Baseline cache outputs

```text
outputs/baseline_cache/
  BASELINE_CACHE_INDEX.csv
  NO_CONTROL_D0_INDEX.csv
  STEP1_BASELINE_INDEX.csv
  FINAL_BASELINE_RUN_INDEX.csv
  _runtime_inp/
  <event_id>/<strategy>/
      *.compact.npz
      *.node_statistics.csv.gz
      *.decisions.jsonl
      *.json
      *.baseline_cache.json
```

- `BASELINE_CACHE_INDEX.csv`: canonical inventory of every generated fixed baseline.
- `NO_CONTROL_D0_INDEX.csv`: direct input to checkpoint design for D2/D3.
- `STEP1_BASELINE_INDEX.csv`: development No-control + Internal-RTC compact trajectories suitable for Step1.
- `FINAL_BASELINE_RUN_INDEX.csv`: populated in Final mode with Formal baseline manifests.

Every completed baseline is re-validated from its actual runtime evidence:

- No-control must have no executable `[CONTROLS]`, no controller and an empty decision log;
- Internal-RTC must retain executable `[CONTROLS]`, have no Python controller and an empty decision log;
- Hold must log one unchanged actuator vector at every control decision;
- All-open must log `1.0` for every eligible actuator at every control decision;
- All-closed must log `0.0` for every eligible actuator at every control decision;
- the runtime physical-network hash must equal the source-event physical-network hash.

`rtc-run-d0-batch` remains available as a lower-level compatibility utility, but the baseline cache is the preferred path for new work.

---

## 6. Canonical end-to-end data flow

### Step 0 — physical preflight and site/sensor mapping

**Inputs**

- frozen Wuhan INP / event INP source;
- eight observed ponding-site coordinates or verified node mappings;
- sensor layout.

**Generate**

- INP audit;
- physical-network hash;
- verified priority-node map;
- graph/schema assets.

```powershell
rtc-inp-audit-v2 --inp <frozen_inp> --priority data/priority_nodes.txt --out outputs/preflight/inp_audit.json
rtc-resolve-priority --inp <frozen_inp> --points data/priority_points.csv --out-csv outputs/preflight/priority_mapping.csv --out-nodes data/priority_nodes.txt
python -m rtc.formal_assets_v3 --inp <frozen_inp> --priority data/priority_nodes.txt --sensors data/sensor_nodes.txt --out-dir outputs/formal_assets
```

Historical data reuse: the old INP is reusable only after hash validation; old priority IDs are not reusable for the uploaded V8 model until mapped correctly.

### Step A — event registry and rainfall-group split

**Inputs**

- reusable rainfall/forcing library;
- event-specific INP paths.

**Generate**

`event_registry_with_splits.csv`, with whole rainfall groups assigned to:

`development / calibration / safety_audit / final`

Development is split again by rainfall group into `train / validation`.

Historical rainfall time series may be reused; old hydraulic outcomes do not need to be reused.

### Step B — fixed baseline cache / canonical D0

Generate No-control, Internal-RTC and, after the control timing is frozen, Hold/All-open/All-closed once per event.

**Primary reusable outputs**

- full compact SI trajectory;
- actuator readback/flow history;
- causal node rainfall;
- cumulative node flooding statistics;
- fixed policy decision log;
- lineage sidecar.

This single cache supplies multiple later steps instead of rerunning the same baseline.

### Step C — optional D1 controlled-state coverage for Step1

D1 is **development/train only** and is used only to broaden Step1 hydraulic-state coverage.

```powershell
rtc-run-d1-exploration ...
```

All actuators remain eligible. D1 is not a Formal baseline and is not a D2/D3 checkpoint source.

Historical Train1600 may be considered only as auxiliary Step1 material after an explicit converter/lineage audit; the default current pipeline does not need it.

### Step 1 — sparse-sensing state reconstruction

**Recommended current inputs**

- `outputs/baseline_cache/STEP1_BASELINE_INDEX.csv`;
- optionally newly generated development/train D1 compact trajectories;
- frozen graph and sensor layout.

**Do not require** old Project6 action-effect data.

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

Step1 windows are sliced lazily from compact trajectories; the same frame is not written repeatedly to disk.

**Generate**

- `step1.pt`;
- held-out development/validation acceptance evidence.

### Step D — replayable checkpoint design

Only No-control prefixes are valid current D2/D3 checkpoint sources.

```powershell
rtc-design-checkpoints `
  --run-index outputs/baseline_cache/NO_CONTROL_D0_INDEX.csv `
  --out outputs/checkpoints/checkpoint_settings.csv `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <HISTORY_READINESS_MINUTES>
```

When the input comes from the baseline cache, checkpoint design uses the cached hydraulic state but restores the **original event INP path** for D2/D3 generation. This avoids nesting controls-disabled runtime copies and unnecessary ~25 MB-per-event duplicate INPs.

D1 controlled states are intentionally rejected because their prior action history is not reproduced by a fresh No-control branch.

### Step E — D2 same-checkpoint actuator truth

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

Each D2 branch saves enough information for Step2 without rerunning SWMM:

- exact current SI state;
- causal node rainfall;
- continuous action setting;
- previous actuator flow;
- full H-step SI hydraulic target trajectory;
- H-step actuator-flow target;
- exact cumulative node flooding volume from SWMM node statistics;
- action SHA and checkpoint/event/rainfall lineage.

### Step F — empirical time-scale selection

Use D2 responses to estimate onset / peak-effect / 90%-response-mass times.

```powershell
python -m rtc.phase0_timescale ...
```

Do not inherit Project6 5/10/120 minutes automatically. Freeze model step, control interval, history and horizon after this development evidence.

### Step G — D3 multi-actuator interactions

D3 uses fresh controls-disabled branches from the same replayable No-control checkpoint lineage.

```powershell
rtc-design-d3 ...
rtc-run-d3-batch ... --workers 16 --swmm-threads-per-process 1
```

**Generate** compact multi-step settings, state trajectories, actuator flows, rainfall and exact cumulative node-flooding truth.

### Step H — build the Step2 run index once

Do not manually concatenate D2 manifest rows with D2 run results. The center/base action is repeated in the probe manifest for multiple actuator probes even though it is physically executed only once.

Use:

```powershell
rtc-build-step2-index `
  --d2-manifest outputs/d2/probe_manifest.csv `
  --d2-run-summary outputs/d2/runs/RUN_SUMMARY.csv `
  --d3-run-summary outputs/d3/RUN_SUMMARY.csv `
  --out outputs/step2/step2_run_index.csv
```

The compiler:

- collapses repeated D2 base-action provenance to one executed branch;
- verifies event/rainfall/split/checkpoint invariants;
- preserves the list of probe actuators represented by the collapsed base action;
- rejects duplicate metadata branches;
- rejects Final branches from the Step2 train/validation index;
- combines D2 and D3 into one lineage-safe index.

### Step 2 — differentiable hydraulic world model

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

Step2 is trained on current-contract D2/D3 evidence, not old Project6 action-effect labels. It supervises:

- future hydraulic state;
- actuator flow;
- exact cumulative SWMM node flooding volume.

Then compile development/validation shards and run `rtc-accept-step2-large`, SWMM finite-difference gradient truth and candidate-ranking truth.

### Step 3 — development closed-loop MPC

Only after Step1/Step2/gradient/ranking acceptance may Proposed enter authoritative development closed-loop SWMM.

```powershell
rtc-run-policy --strategy proposed ...
```

Development comparisons should reference the **cached** No-control/Internal/Hold/All-open/All-closed outputs for the same event rather than rerunning them.

### Step I — Policy Lock

Freeze physical assets, split registry, models, acceptance evidence, controller config, rainfall forecast, baseline plan and runtime semantics.

```powershell
rtc-policy-lock-v5 ...
```

### Step J — untouched Final

Only after Policy Lock:

1. generate/resume fixed Final baselines with `rtc-build-baseline-cache --stage final --policy-lock ...`;
2. run Proposed once for each untouched Final event;
3. formalize Proposed with `rtc-formalize-run`;
4. combine Proposed Formal rows with `FINAL_BASELINE_RUN_INDEX.csv`;
5. compile Final using `rtc-compile-final-v4`.

Final conclusions use authoritative SWMM cumulative TFV/PFV and exact routing-step Global Peak replay. Final truth must not change training, time scales, thresholds, priority mapping, sensor layout or hyperparameters.

---

## 7. Compact storage contract

Formal new data is stored as compressed SI arrays plus exact node statistics and JSON lineage. Raw node/actuator CSV and SWMM `.out/.rpt` are debug-only and disabled by default. Per-subcatchment realised runoff is not persisted as a Formal model input.

Compact state channels are:

`depth_m, head_m, flooding_m3s, volume_m3, total_inflow_m3s, total_outflow_m3s`.

Step1 windows are lazy slices of compact trajectories. Step2 branches are compiled into bounded-size shards. Changing a model batch size does not require rerunning SWMM.

---

## 8. Hardware path: 16 CPU workers + RTX 4060 8GB

For SWMM generation use independent processes, normally:

```text
--workers 16 --swmm-threads-per-process 1
```

The source INP has `THREADS=2`; using 16 SWMM processes without overriding that would create avoidable CPU oversubscription. The runtime builders keep one SWMM engine thread/process and cache runtime INPs by content hash.

For GPU training use the `*-large` commands. Defaults use AMP, small micro-batches and gradient accumulation. Step1 uses trajectory-local lazy batches; Step2 streams shards so an 8GB GPU does not need the complete Wuhan dataset in memory.

Do not intentionally run the full 16-process SWMM pool and GPU training at the same time unless a measured local resource audit shows a benefit.

---

## 9. What must be regenerated versus what should be cached

### Regenerate under the current contract

- No-control authoritative trajectories;
- Internal-RTC authoritative trajectories unless fully re-admitted by lineage;
- D1 development exploration if used;
- D2 same-state action-effect branches;
- D3 interaction branches;
- Step1 and Step2 Formal model checkpoints;
- development/Final Proposed closed-loop runs.

### Generate once and cache

- fixed No-control/Internal/Hold/All-open/All-closed result for each event and frozen runtime contract;
- controls-disabled runtime INP per unique event source SHA + SWMM thread contract;
- graph/schema/physical-network manifests;
- D2/D3 compact branches;
- Step1/Step2 acceptance evidence;
- Formal exact-peak replay after the corresponding decision schedule is frozen.

### Reuse when lineage is valid

- rainfall time series / event forcing library;
- physical INP if exact SHA/physical-network hash is verified;
- observed-site coordinates;
- sensor metadata if it still maps to the frozen graph;
- historical hydraulic trajectories only as explicitly audited auxiliary material, never by directory name alone.

---

## 10. Evidence boundary

Only authoritative SWMM runs can support final PFV/TFV/Global-Peak claims. Surrogate outputs are used for state reconstruction, prediction, gradient search and online decision-making; they do not replace Final SWMM truth.

Read [`FORMAL_PIPELINE_LATEST.md`](FORMAL_PIPELINE_LATEST.md) for the stricter Formal evidence/locking sequence. The README is the operational data-flow entry point; the Formal document defines the scientific fail-closed gates.
