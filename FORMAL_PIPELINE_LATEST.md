# Formal Pipeline — Wuhan large-system TFV-first contract (latest)

This document supersedes every earlier V1/V2/V3 workflow for **new Formal evidence**.
The target is a large Dynamic-Wave SWMM network controlled online without rainfall-event
lookup, a fixed active actuator subset, or binary-pump assumptions.

## 1. Frozen scientific objective

1. **Primary objective:** minimise cumulative system-wide TFV over the causal prediction
   horizon, optionally risk-adjusted across causal rainfall scenarios.
2. **Eight observed ponding sites:** retain site-wise cumulative PFV and maximum depth as
   important diagnostics and a **soft secondary preference** among TFV-near-optimal controls.
   PFV is not a hard admission/veto condition; limited PFV deterioration may be accepted
   when necessary for a meaningful system-wide TFV improvement.
3. **Hard runtime conditions only:** finite numerics, continuous setting bounds, engineering
   projection, target write/readback, frozen schema/lineage and strict causality.
4. **No-control and Internal-RTC are different policies:**
   - `no_control`: same physical network/forcing, native `[CONTROLS]` disabled, no Python writes;
   - `internal_rtc`: original event INP, native `[CONTROLS]` enabled, no Python writes.
5. Proposed, Hold, D1, D2, D3, all-open and all-closed execute on a controls-disabled copy
   from simulation start. This prevents fast native rules from changing Python actions
   between RTC callbacks.

## 2. Mandatory preflight

```powershell
rtc-inp-audit-v2 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --out outputs/preflight/inp_audit.json
```

A supplied priority ID absent from the INP is a hard blocker. Never substitute a similar
string. If the eight observed sites are available as coordinates, create an auditable map:

```powershell
rtc-resolve-priority `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --points data/priority_points.csv `
  --out-csv outputs/preflight/priority_mapping.csv `
  --out-nodes data/priority_nodes.txt
```

Review nearest-node distances and duplicate mappings before Formal use.

## 3. Hardware and storage contract

For a workstation with 16 CPU workers and an RTX 4060 8GB:

- SWMM data generation: independent processes, normally
  `--workers 16 --swmm-threads-per-process 1`;
- do not run 16 processes while retaining source-INP `THREADS=2`, which would oversubscribe
  the CPU;
- run GPU training as a separate phase; default large-model commands use AMP, small
  micro-batches and gradient accumulation;
- successful SWMM `.out/.rpt` files are deleted by default;
- raw node/actuator CSV is debug-only;
- D0/D1 and D2/D3 save compact compressed SI arrays plus exact node statistics and JSON
  lineage;
- Step1 windows are sliced lazily and never duplicated on disk;
- Step2 is compiled into bounded-size shards rather than one monolithic dataset;
- runners resume only branches whose metadata, compact file and exact-statistics evidence
  are complete.

Compact hydraulic state:

`depth_m, head_m, flooding_m3s, volume_m3, total_inflow_m3s, total_outflow_m3s`.

`Node.flooding` is an instantaneous rate, not PFV/TFV. Authoritative PFV/TFV use cumulative
SWMM node flooding volume over the exact event/horizon. The surrogate converts predicted
rates to volume with trapezoidal integration using the **current** rate plus all future
predicted rates; Step2 training/acceptance additionally uses exact cumulative SWMM volume.

## 4. Stage A — rainfall-group split

Assign entire rainfall groups to mutually exclusive:

`development / calibration / safety_audit / final`

and split development again by rainfall group into `train / validation`. Final remains
untouched until Policy Lock.

## 5. Stage B — replayable D0 state coverage

Generate No-control D0 with native controls disabled from time zero:

```powershell
rtc-run-d0-batch `
  --events outputs/contracts/event_registry_with_splits.csv `
  --strategy no_control `
  --out-dir outputs/d0_no_control `
  --record-stride-seconds <FROZEN_MODEL_STEP> `
  --workers 16 `
  --swmm-threads-per-process 1
```

Generate Internal-RTC separately:

```powershell
rtc-run-d0-batch ... --strategy internal_rtc
```

Never merge their policy semantics.

## 6. Stage C — optional D1 controlled-state coverage for Step1 only

```powershell
rtc-run-d1-exploration ...
```

D1 is development/train only and exists to expose Step1 to hydraulically reachable
controlled states. Every writable actuator remains eligible.

**D1 trajectories are NOT D2/D3 checkpoint sources in the current pipeline.** A D1 state
contains the effects of its entire preceding exploration-action history. A fresh D2 branch
that simply replays a No-control prefix to the same clock time is therefore not the same
hydraulic state. D1 may become a D2 source only after an explicit prefix-action replay or
verified hot-start lineage is implemented and validated.

## 7. Stage D — D2/D3 checkpoint design from No-control D0 only

Use only controls-disabled, no-Python-write D0 trajectories, because their prefix can be
reproduced exactly by every fresh D2/D3 branch:

```powershell
rtc-design-checkpoints `
  --run-index outputs/d0_no_control/D0_no_control_RUN_INDEX.csv `
  --out outputs/checkpoints/checkpoint_settings.csv `
  --checkpoints-per-event 8 `
  --minimum-elapsed-minutes <HISTORY_READINESS_MINUTES>
```

The command rejects D1 controlled trajectories, Internal-RTC trajectories and Final groups.
It records `prefix_contract=CONTROLS_DISABLED_NO_CONTROL_FROM_T0`.

## 8. Stage E — D2 same-checkpoint actuator truth

Design `u-epsilon / u / u+epsilon`, then run compact D2:

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

Every branch starts from the same replayable controls-disabled prefix. Exact horizon TFV/PFV
labels come from the difference in cumulative SWMM node statistics between checkpoint and
aligned horizon endpoint.

## 9. Stage F — empirical hydraulic time scale

Use compact D2 response evidence:

```powershell
python -m rtc.phase0_timescale ...
```

Review response onset, peak-effect time and 90%-response-mass distributions. Freeze model
step, control update and horizon only after this development evidence and sensitivity
analysis. Do not automatically inherit Project6 5/10/120 minutes.

## 10. Stage G — frozen graph and schemas

Only after the eight observed sites are correctly mapped:

```powershell
python -m rtc.formal_assets_v2 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --sensors data/sensor_nodes.txt `
  --out-dir outputs/formal_assets
```

The graph stores SI node geometry and device-specific pump-curve/orifice/weir features.
Step2 also learns a stable actuator-identity embedding. No Engineering36 or fixed active
subset is created.

## 11. Stage H — Step1 large-model path

Do not materialise repeated 13-frame windows for this network.

```powershell
rtc-train-step1-large `
  --run-index outputs/d0_d1_step1_run_index.csv `
  --graph outputs/formal_assets/graph_schema.npz `
  --sensors data/sensor_nodes.txt `
  --history-steps <FROZEN_HISTORY_STEPS> `
  --model-step-seconds <FROZEN_MODEL_STEP> `
  --batch-size 4 --grad-accum 4 `
  --out outputs/models/step1.pt
```

Step1 reads sparse depth/head history plus topology-local causal context: realised node
rainfall and incoming/outgoing actuator setting/flow summaries. The full 109-actuator
vector is not broadcast to every node. Use development/validation for acceptance and apply
the preregistered `step1` threshold section with `rtc-accept-gate`.

## 12. Stage I — D3 interactions and Step2 shards

Generate D3 on controls-disabled replayable prefixes:

```powershell
rtc-run-d3-batch ... --workers 16 --swmm-threads-per-process 1
```

Merge D2/D3 metadata into a lineage-preserving run index and compile bounded shards:

```powershell
rtc-compile-step2-shards `
  --run-index outputs/step2/run_index.csv `
  --out-dir outputs/step2/train_shards `
  --development-fold train `
  --shard-size 128
```

## 13. Stage J — Step2 large-model path

```powershell
rtc-train-step2-large `
  --manifest outputs/step2/train_shards/manifest.json `
  --graph outputs/formal_assets/graph_schema.npz `
  --batch-size 2 --grad-accum 4 `
  --out outputs/models/step2.pt
```

Formal Step2 training supervises full state trajectories, actuator flow and exact cumulative
SWMM node flooding volume. Build a separate development/validation shard manifest and run
`rtc-accept-step2-large`; then apply the frozen `step2` thresholds using `rtc-accept-gate`.

## 14. Stage K — gradient and candidate-ranking truth

Before gradient MPC is Formal-eligible, held-out D2 must verify TFV gradient
sign/direction/magnitude against authoritative SWMM finite differences and verify candidate
TFV ranking/top-1 regret. Predicted cumulative volume uses the same trapezoid contract as
MPC/training. Priority-PFV gradient may be reported but does not veto TFV-valid control.

## 15. Stage L — development closed loop

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

The runner uses a cached controls-disabled runtime INP for Proposed/No-control/Hold and
other Python policies. Internal-RTC always uses the original event INP.

## 16. Stage M — TFV-first Policy Lock

Start from:

- `configs/formal_controller_v4.template.json`
- `configs/model_acceptance_contract_v3.template.json`
- `configs/formal_baseline_plan.v3.json`

Then freeze with:

```powershell
rtc-policy-lock-v5 `
  --ledger outputs/evidence/tfv_pipeline_ledger.json `
  --artifacts outputs/contracts/formal_policy_artifacts.json `
  --out outputs/policy_lock/policy_lock.json
```

Priority calibration may be locked for uncertainty-aware soft ranking/reporting, but PFV is
not a mandatory safety-pass gate.

## 17. Stage N — untouched Final

Run the complete locked strategy matrix on untouched Final groups. Main closed-loop runs
keep `exact_global_peak=false`; afterward use frozen-decision routing-step replay through
`rtc-formalize-run`. Compile only with:

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
- paired, event-balanced comparisons against No-control and Internal-RTC separately.

Final truth may not be used for fitting, threshold selection, priority mapping, sensor
selection, time-scale selection, uncertainty calibration or hyperparameter tuning.
