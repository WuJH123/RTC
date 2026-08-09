# Wuhan RTC v0.6 — sparse sensing, differentiable hydraulics and TFV-first MPC

This repository implements the final Wuhan large-network research workflow:

```text
causal sparse observations + realised rainfall + actuator readback
                         ↓
               Step1 current-state reconstruction
                         ↓
                 full current hydraulic state
                         ↓
       Step2 differentiable hydraulic world model
                         ↓
       all-actuator continuous receding-horizon MPC
                         ↓
        executable setting write → hold → readback
```

The scientific objective is **system-wide cumulative TFV minimisation**. The eight observed priority sites are a **soft secondary/diagnostic** objective only. Global Peak is a reporting metric only. All final performance claims come from authoritative SWMM.

## 1. Online causality contract

At decision time `t`, the controller may use only information available at or before `t`:

- sparse depth/head observations;
- realised rainfall history;
- actuator target/current-setting readback and actuator flow;
- frozen graph/device information;
- rainfall scenarios generated from causal rainfall history.

Forbidden online information includes event ID as a control signal, future realised rainfall/runoff, future SWMM state/flooding, future Internal-RTC trajectory, offline future labels and Final truth.

## 2. Frozen Wuhan physical/reporting contract

For the audited Wuhan V8 lineage:

- `FLOW_UNITS = CMS`;
- `FLOW_ROUTING = DYNWAVE`;
- 932 hydraulic nodes;
- 1,167 conduits;
- 3,731 subcatchments;
- 57 pumps + 42 orifices + 10 weirs = **109 writable SWMM actuator links**;
- source `ROUTING_STEP = 15 s`;
- source `RULE_STEP = 10 s`.

The verified PFV_CORE8 file is `data/priority_nodes.txt` and must contain exactly:

```text
MSLBZW001
HS1316314
YS2530050
HS2529198
MH0200773
HS1330349
HS2529139
HS2529052
```

Preflight and Policy Lock both fail if these nodes are absent from the frozen graph/INP.

## 3. No-control semantics

`no_control` means **No-supervisory-RTC**, contract `NO_SUPERVISORY_RTC_V2`:

- executable user-defined `[CONTROLS]` are removed;
- Python makes no actuator writes;
- physical network and rainfall/runoff forcing are preserved;
- pump curves and initial status are preserved;
- intrinsic `[PUMPS]` Startup/Shutoff depth logic is preserved;
- regulator physical/default behaviour is preserved.

Therefore No-control is neither All-open nor All-closed. `internal_rtc` retains the original native `[CONTROLS]` and receives no Python actuator writes.

Formal comparison is exactly:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

`hold` remains debug-only because it can collapse to a No-control-like policy on a controls-disabled base.

## 4. Continuous all-actuator MPC

Production MPC does **not** use Engineering36, a fixed controlled subset, runtime Top-K masking or artificial binary-pump conversion.

All writable actuators remain in the frozen action schema. The optimizer uses direct continuous settings and projects every future control block to `[0,1]` and, when configured, to the sequential rate limit. This preserves usable inward gradients even when the current setting is exactly 0 or 1.

`active_actuator_count_*` in runtime diagnostics is reporting only; it does not select a fixed set.

SWMM supports fractional pump/orifice/weir settings as a numerical control experiment. A claim of physical field deployment additionally requires Wuhan-specific SCADA/VFD/interlock/dwell/ramp/readback metadata; those properties are not inferred from the INP.

## 5. Rainfall groups: correctness requirements versus paper-strength target

`rainfall_group` is the independent leakage/statistical unit.

**Hard correctness requirements** are only:

- unique `event_id` rows;
- no rainfall group crosses `scientific_split`;
- development contains rainfall-group-disjoint train and validation folds;
- Final contains at least one untouched rainfall group;
- Final is never used for training/tuning;
- referenced event INPs exist.

For a publication-strength large-network experiment, the current **recommended target** is about **160 independent rainfall groups**, for example:

```text
development   96
calibration   24
safety_audit  16
final         24
```

This is a study-design recommendation, **not a software execution gate**. Pilot runs and smaller development studies may proceed with fewer groups if the required leakage invariants hold.

## 6. Fresh study workspace and safe reuse

Start the final study with a new empty workspace:

```powershell
rtc-init-fresh-workspace `
  --root E:\RTC_sewer\RTC_fresh_v06 `
  --inp <FROZEN_INP> `
  --priority data/priority_nodes.txt `
  --events <NEW_EVENT_REGISTRY_WITH_SPLITS.csv>
```

The workspace binds the canonical physical/input/split identities. Large RTC-derived data may live on another disk/volume.

Reuse is **not** decided by directory location or file existence. A generated result is reusable only when its:

1. scientific/data contract is compatible;
2. stable RTC implementation-contract fingerprint matches;
3. numerical inputs/config/timing/action/sequence lineage matches;
4. required generated-artifact hashes still verify.

The implementation fingerprint is intentionally a stable scientific-semantics contract, **not a byte-for-byte hash of every Python file**. Unrelated documentation/reporting/error-message edits therefore do not invalidate expensive SWMM evidence. When scientific semantics change, the corresponding implementation contract ID must be bumped.

Historical Project6 RTC trajectories/models/evidence are not admissible in the v0.6 study unless they explicitly satisfy the current contracts; the intended final workflow regenerates RTC-derived evidence with v0.6.

## 7. Phase-0 before production timing is frozen

Do not assume 5 min observation, 10 min control or a 120 min horizon just because they were used historically.

Phase-0 uses **development-only** controls-disabled D0/D2 evidence with `<=60 s` Python sampling while SWMM retains its internal routing step. Analyse:

- setting/readback lag;
- actuator-flow `t10/t50/t90` and peak time;
- network flooding-rate response;
- network maximum-depth response;
- whether the response peak is censored near the pilot horizon.

Do not use response-area `mass90` for a sustained step input; that quantity is confounded by the selected experiment horizon. If recovery after releasing an action matters, use D3/pulse-release sequences.

Phase-0 data whose step differs from the final production step is timing evidence only and is rejected from production Step2 shards.

## 8. Candidate 5/10-min causal timeline

If Phase-0 accepts:

```text
model/observation step = 300 s
control update         = 600 s
history_steps          = 13
first control          = 60 min
record_stride          = 300 s
```

then the t=0-inclusive history is exactly:

```text
0, 5, 10, ..., 55, 60 min = 13 causal frames
```

The first real MPC can occur at `t=60 min`, then `70,80,... min`. One 10-min control block spans two 5-min model intervals while the SWMM Dynamic-Wave solver continues at its internal routing step.

Formal timing requires:

```text
record_stride_seconds == model_step_seconds
control_update_seconds % model_step_seconds == 0
first control lies on both model and control grids
full causal history exists before first MPC
prediction horizon >= one complete control interval
```

## 9. Data-generation roles

### Fixed baseline cache

After production timing is frozen, generate each fixed reference once per event with `rtc-build-baseline-cache`. The cache validates strategy semantics, physical-network identity, timing/config lineage and generated artifact hashes before reuse.

Canonical views include:

```text
BASELINE_CACHE_INDEX.csv
NO_CONTROL_D0_INDEX.csv
STEP1_BASELINE_INDEX.csv
FINAL_BASELINE_RUN_INDEX.csv
```

### D1

`rtc-run-d1-batch` accepts **development/train only**. It provides controlled-state coverage for Step1. D1 is never a D2/D3 checkpoint source.

### D2

D2 starts from replayable controls-disabled No-control prefixes. At checkpoint `t`, it records the pre-action state/statistics, writes one candidate action, then records the future trajectory and exact cumulative SWMM node flooding-volume change. It provides local/boundary action-effect truth and finite-difference gradients.

### D3

D3 starts from the same replayable prefix contract and provides multi-actuator, multi-step interaction sequences. Its `max-active` generation parameter is a data-coverage choice only; production MPC still optimizes all actuators.

All D0/D1/D2/D3 generators are resumable by deterministic generation keys plus artifact hashes and reject Final rows before Policy Lock.

## 10. Step1 data and training

Step1 reconstructs the **current** full hydraulic state from causal history. Its compact trajectories must have a time grid exactly equal to the frozen `model_step_seconds`.

Use:

- development baseline trajectories from `STEP1_BASELINE_INDEX.csv`;
- optional development/train D1 trajectories.

Train/validation must remain rainfall-group-disjoint. Formal validation metrics are averaged with equal weight per independent rainfall group.

Training saves an atomic epoch state containing model, optimizer, scaler and RNG state. Rerunning the identical command resumes only when the run-index, graph, sensors, timing, architecture/hyperparameters and implementation contract still match.

## 11. Step2 data and training

`rtc-build-step2-index` combines deduplicated D2 and D3 branches. `rtc-compile-step2-shards` then enforces one immutable discrete-time contract across every shard:

```text
model_step_seconds
horizon_steps
```

This prevents Phase-0 60 s branches from being mixed with a production 300 s world model.

Step2 is supervised on:

1. future hydraulic state trajectory;
2. actuator-flow trajectory;
3. exact cumulative SWMM node flooding volume.

The predicted cumulative-volume operator is identical in Step2 training, Step2 validation, MPC, gradient validation and action ranking:

```text
trapezoidal integration of current flooding rate + future predicted flooding rates
```

The authoritative label remains SWMM cumulative node flooding volume.

Step2 training also has atomic epoch resume bound to shard manifest, graph, fixed time contract and training configuration.

## 12. Authoritative TFV/PFV/Global Peak

`Node.flooding` is an instantaneous rate, not a flooding volume.

For node `i` over `[t0,t1]`:

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(t1)
         - cumulative_SWMM_flooding_volume_i(t0)
```

Then:

```text
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the frozen eight priority nodes
```

Global Peak is:

```text
max_t sum_i max(flooding_rate_i(t), 0)
```

Formal Global Peak is obtained by routing-step observation of a frozen-decision replay. The replay preserves the original Python actuator-write cadence so the reporting calculation does not alter the executed control trajectory.

## 13. Model/action-effect acceptance

Before Policy Lock, v0.6 requires:

1. rainfall-group-balanced held-out Step1 reconstruction acceptance;
2. rainfall-group-balanced held-out Step2 trajectory/exact-TFV acceptance;
3. held-out D2 exact-SWMM local TFV-gradient acceptance, with one-sided differences at 0/1 bounds;
4. D2 local + D3 joint-sequence ranking/regret acceptance;
5. Proposed development closed-loop SWMM;
6. real-time execution acceptance.

D3 ranking is required because online MPC optimizes the joint multi-actuator action space; single-actuator D2 ranking alone is insufficient evidence.

## 14. Real-time execution contract

The resolved controller freezes `decision_runtime_budget_seconds`, which must be smaller than the control interval.

Each Proposed decision records wall-clock Step1 + rainfall forecast + MPC + projection time. A candidate that exceeds the deadline is discarded and the causal fallback is executed as `FALLBACK_COMPUTE_DEADLINE`.

Runtime acceptance requires correct event-clock decision timing, t=0 history, successful readback and zero fatal history/readback/runtime/deadline fallbacks.

## 15. Policy Lock and Final

Current Policy Lock:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

It binds only artifacts that define the scientific experiment or demonstrate its acceptance: physical model, priority/sensors, canonical split, controller/time contract, graph, Step1/Step2 checkpoints, acceptance evidence, runtime evidence and baseline plan.

After Policy Lock:

1. generate/resume the four fixed Final references once;
2. run locked Proposed once per untouched Final event;
3. formalize every run and replay routing-step Global Peak;
4. compile the complete five-strategy matrix with `rtc-compile-final-v4`.

Final statistics first collapse variants inside each rainfall group and then give every independent rainfall group equal weight.

## 16. Public command sequence

The complete supported local workflow is documented in:

- `docs/LOCAL_RUNBOOK_V06.md` — exact execution order and commands;
- `docs/FINAL_DATA_INVENTORY_V06.md` — required/reusable data artifacts;
- `FORMAL_PIPELINE_LATEST.md` — fail-closed scientific evidence contract.

Install and smoke-test the final merged main branch with:

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

Do not begin expensive production SWMM generation until Phase-0 has frozen the time contract and the final v0.6 source has been merged.
