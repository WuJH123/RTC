# Formal Pipeline — Wuhan large-system TFV-first contract (latest)

This document supersedes every earlier V1/V2/V3 pipeline note for **new Formal evidence**.
The target system is a large Dynamic-Wave SWMM network. The scientific objective is:

> reconstruct the current network state from sparse causal observations, learn the
> continuous hydraulic consequences of every writable actuator, and at each control update
> decide online which facilities to move, by how much, and when — without rainfall-ID
> lookup, a fixed active subset, or binary-pump assumptions.

## Frozen scientific objective

1. **Primary optimization objective:** minimise cumulative system-wide TFV over the causal
   prediction horizon (risk-adjusted across causal rainfall scenarios when used).
2. **Eight observed ponding locations:** retained as site-wise PFV/max-depth diagnostics and
   a **soft secondary preference** among TFV-near-optimal controls. PFV is **not** a hard
   admission/veto condition. A priority site may worsen if this is necessary for a
   meaningful system-wide TFV improvement.
3. **Hard runtime conditions:** finite numerics, continuous setting bounds, engineering
   projection, write/readback and strict causality. Failure here falls back; PFV alone does
   not.
4. **No-control and Internal-RTC are different policies:**
   - `no_control`: same physical network/forcing, native `[CONTROLS]` disabled, no Python
     writes;
   - `internal_rtc`: original event INP with native `[CONTROLS]` enabled, no Python writes.
5. Proposed, Hold, D1, D2, D3, all-open and all-closed run on a **controls-disabled** copy
   from simulation start. This prevents SWMM native rules from overwriting Python actions
   between control callbacks.

## Mandatory preflight before expensive SWMM

```powershell
rtc-inp-audit-v2 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --out outputs/preflight/inp_audit.json
```

The command fails if a supplied priority node does not exist. Never substitute a visually
similar node ID. If the eight observed sites are available as coordinates, map them with:

```powershell
rtc-resolve-priority `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --points data/priority_points.csv `
  --out-csv outputs/preflight/priority_mapping.csv `
  --out-nodes data/priority_nodes.txt
```

`priority_points.csv` columns are `priority_id,x,y`. Review mapping distances before Formal
use.

## Hardware contract for a 16-CPU / RTX 4060 8GB workstation

- SWMM data generation: independent **processes**, normally `--workers 16` with
  `--swmm-threads-per-process 1`. Do not run 16 processes × `THREADS=2` and oversubscribe
  the CPU.
- GPU training: run after SWMM generation. Step1 default micro-batch 4 / gradient
  accumulation 4; Step2 default micro-batch 2 / accumulation 4; AMP enabled on CUDA.
- Do not run the CPU-heavy 16-process SWMM pool concurrently with GPU training unless a
  measured resource audit proves it is beneficial.

## Storage contract

New Formal data uses compact compressed arrays, not repeated row-wise hydraulic CSVs:

- D0/D1: one `*.compact.npz` trajectory + exact cumulative node statistics + JSON lineage;
- D2/D3: one compact branch + exact cumulative node statistics + JSON lineage;
- raw node/actuator CSV and SWMM `.out/.rpt` are debug-only and off by default;
- Step1 windows are sliced lazily from trajectories and are **not materialized**;
- Step2 is compiled to bounded-size shards, not one monolithic dataset;
- successful SWMM engine `.out/.rpt` files are deleted by default;
- data runners resume from content-complete hashed branch evidence.

The compact state schema is SI:

`depth_m, head_m, flooding_m3s, volume_m3, total_inflow_m3s, total_outflow_m3s`.

`Node.flooding` is an instantaneous rate. It is never called PFV/TFV by itself.
Authoritative PFV/TFV use SWMM cumulative `Node.statistics.flooding_volume` over the exact
specified horizon/event. Sampled rate integration is surrogate/diagnostic only.

## Stage A — rainfall-group split

Assign whole rainfall groups to mutually exclusive:

`development / calibration / safety_audit / final`

and split development again by rainfall group into `train / validation`. Final is untouched
until after Policy Lock.

## Stage B — D0 state coverage

For No-control D0, each event is automatically converted to a controls-disabled runtime INP:

```powershell
rtc-run-d0-batch `
  --events outputs/contracts/event_registry_with_splits.csv `
  --strategy no_control `
  --out-dir outputs/d0_no_control `
  --record-stride-seconds <FROZEN_MODEL_STEP> `
  --workers 16 `
  --swmm-threads-per-process 1
```

Internal-RTC is generated separately with:

```powershell
rtc-run-d0-batch ... --strategy internal_rtc
```

Do not merge their policy semantics.

## Stage C — optional development-only D1 controlled coverage

D1 gives Step1 controlled-state coverage without defining a runtime policy lookup:

```powershell
rtc-run-d1-exploration ...
```

All writable facilities remain eligible; the random mask is a data-sampling device only.
D1 is development/train only.

## Stage D — checkpoint design

Never hand-pick states after seeing outcomes. Build stratified, history-ready checkpoints
from compact D0/D1 trajectories:

```powershell
rtc-design-checkpoints `
  --run-index outputs/d0_d1_run_index.csv `
  --out outputs/checkpoints/checkpoint_settings.csv `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <HISTORY_READINESS_MINUTES>
```

Final groups are automatically excluded.

## Stage E — D2 same-checkpoint actuator truth

Design `u-epsilon / u / u+epsilon` branches, then run compact D2:

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

Every branch is run on the same-prefix **controls-disabled** base. Native Internal-RTC rules
must never contaminate action-effect labels.

## Stage F — Phase-0 hydraulic time scale

Run `python -m rtc.phase0_timescale` on D2 compact branches. Inspect response onset, peak and
90%-mass distributions. Freeze model step, control update and horizon only after this
review plus development sensitivity. Do not inherit Project6 5/10/120 minutes by default.

## Stage G — frozen graph/assets

Only after the eight observed sites are correctly mapped:

```powershell
python -m rtc.formal_assets_v2 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --sensors data/sensor_nodes.txt `
  --out-dir outputs/formal_assets
```

The graph stores SI node geometry plus actuator type, pump-curve capacity/range,
orifice/weir geometry and a stable actuator ordering. Step2 also learns a locked
actuator-identity embedding. No Engineering36/fixed subset is created.

## Stage H — Step1 training/acceptance (RTX 4060 path)

Do **not** use the legacy materialized-window pipeline for this model.

```powershell
rtc-train-step1-large `
  --run-index outputs/d0_d1_run_index.csv `
  --graph outputs/formal_assets/graph_schema.npz `
  --sensors data/sensor_nodes.txt `
  --history-steps <FROZEN_HISTORY_STEPS> `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --batch-size 4 --grad-accum 4 `
  --out outputs/models/step1.pt

rtc-accept-step1-large ... --out outputs/evidence/step1_metrics.json
rtc-accept-gate `
  --metrics outputs/evidence/step1_metrics.json `
  --contract configs/model_acceptance_contract_v3.json `
  --section step1 `
  --out outputs/evidence/step1_acceptance.json
```

Step1 context is topology-local: realised node rainfall + incoming/outgoing actuator
setting/flow summaries. The full 109-actuator vector is not broadcast to every node.

## Stage I — D3 interactions and Step2 shards

Generate D3 on controls-disabled bases with the parallel runner:

```powershell
rtc-run-d3-batch ... --workers 16 --swmm-threads-per-process 1
```

Merge D2/D3 run metadata into one lineage-preserving run index and compile shards:

```powershell
rtc-compile-step2-shards `
  --run-index outputs/step2/run_index.csv `
  --out-dir outputs/step2/train_shards `
  --development-fold train `
  --shard-size 128
```

## Stage J — Step2 training/acceptance (RTX 4060 path)

```powershell
rtc-train-step2-large `
  --manifest outputs/step2/train_shards/manifest.json `
  --graph outputs/formal_assets/graph_schema.npz `
  --batch-size 2 --grad-accum 4 `
  --out outputs/models/step2.pt
```

Compile a separate development/validation shard manifest and run
`rtc-accept-step2-large`. TFV/PFV KPI acceptance uses exact cumulative SWMM node-statistic
volume, not sampled-rate labels. Then apply the frozen `step2` threshold section with
`rtc-accept-gate`.

## Stage K — gradient and candidate-ranking truth

Before gradient MPC is Formal-eligible, held-out D2 must verify SWMM finite-difference TFV
gradient direction/magnitude and candidate TFV ranking/regret. Priority-PFV gradient may be
reported but cannot veto a TFV-valid policy.

## Stage L — development closed loop

Run:

```powershell
rtc-run-policy `
  --strategy proposed `
  --inp <event_inp> `
  --config configs/formal_controller_v4.json `
  --graph outputs/formal_assets/graph_schema.npz `
  --step1 outputs/models/step1.pt `
  --step2 outputs/models/step2.pt `
  --priority data/priority_nodes.txt `
  --sensors data/sensor_nodes.txt `
  ...
```

The runner creates a controls-disabled runtime copy. Internal-RTC is run separately with
`--strategy internal_rtc`; No-control separately with `--strategy no_control`.

## Stage M — TFV-first Policy Lock

Start from:

- `configs/formal_controller_v4.template.json`
- `configs/model_acceptance_contract_v3.template.json`
- `configs/formal_baseline_plan.v3.json`

Freeze numerical development choices, then use:

```powershell
rtc-policy-lock-v5 `
  --ledger outputs/evidence/tfv_pipeline_ledger.json `
  --artifacts outputs/contracts/formal_policy_artifacts.json `
  --out outputs/policy_lock/policy_lock.json
```

PFV calibration may be locked to support uncertainty-aware **soft** priority ranking and
reporting, but it is not a mandatory safety-pass gate.

## Stage N — untouched Final

For each locked Final event, run the complete strategy matrix. Formal main runs set
`exact_global_peak=false`; afterward bind a routing-step replay with `rtc-formalize-run`.
Compile only with:

```powershell
rtc-compile-final-v4 `
  --policy-lock outputs/policy_lock/policy_lock.json `
  --run-index outputs/final/run_index.csv `
  --out-dir outputs/final/evidence
```

Formal conclusions use:

- TFV: cumulative SWMM flooding volume over all nodes;
- PFV: cumulative SWMM flooding volume at the eight verified observed-site nodes;
- Global Peak: synchronous network flooding-rate peak from frozen-decision routing replay;
- priority max depth: site-wise diagnostic;
- event-balanced paired comparisons against No-control and Internal-RTC separately.

No Final truth may be used for fitting, threshold selection, priority mapping, sensor
selection, time-scale selection or hyperparameter tuning.
