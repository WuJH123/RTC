# Wuhan RTC v0.6 — sparse sensing, differentiable hydraulics and TFV-first MPC

This repository implements the final large-network workflow:

```text
causal sparse hydraulics + realised rainfall + actuator readback
                            ↓
                 Step1 current-state reconstruction
                            ↓
                    full current hydraulic state
                            ↓
             Step2 differentiable world model
                            ↓
        all-actuator continuous receding-horizon MPC
                            ↓
          executable setting write → hold → readback
```

The primary scientific objective is **minimum cumulative system-wide TFV**. The eight observed priority sites are a **soft secondary/diagnostic** objective only. Global Peak is report-only. Final performance claims come from authoritative SWMM.

## 1. Causal online contract

At decision time `t`, the policy may use only information available at or before `t`: sparse depth/head observations, realised rainfall history, actuator target/current-setting/flow readback, frozen graph/device information and rainfall scenarios generated causally from observed rainfall history.

Forbidden online information includes event ID as a control signal, future realised rainfall/runoff, future SWMM state/flooding, future Internal-RTC trajectory, offline future labels and Final truth.

## 2. Wuhan physical/reporting contract

Audited Wuhan V8 lineage:

- `FLOW_UNITS = CMS`, `FLOW_ROUTING = DYNWAVE`;
- 932 hydraulic nodes, 1,167 conduits, 3,731 subcatchments;
- 57 pumps + 42 orifices + 10 weirs = **109 writable SWMM actuator links**;
- source routing step = 15 s; native rule step = 10 s.

`data/priority_nodes.txt` is frozen exactly as:

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

Preflight/Policy Lock fail if these eight nodes are absent from the frozen graph/INP.

## 3. No-control and Formal baselines

`no_control` is `NO_SUPERVISORY_RTC_V2`: remove executable user `[CONTROLS]`, make no Python actuator writes, preserve forcing/network physics, pump curves/initial status, intrinsic `[PUMPS]` Startup/Shutoff logic and regulator physical/default behavior.

It is neither All-open nor All-closed. `internal_rtc` retains the original native `[CONTROLS]` and receives no Python writes.

Formal matrix:

```text
proposed
no_control
internal_rtc
all_open
all_closed
```

`hold` is debug-only because it can collapse to No-control on the controls-disabled base.

## 4. All-actuator continuous MPC

Production MPC does **not** use Engineering36, a fixed controlled subset, runtime Top-K or artificial binary-pump conversion.

All writable links remain in the action schema. MPC optimizes direct continuous settings and projects every future control block to `[0,1]` and, when configured, to sequential rate limits. This retains usable inward gradients for actuators currently at exact 0 or 1.

SWMM fractional pump/orifice/weir settings are valid numerical controls. Physical Wuhan deployment still requires verified SCADA/VFD/discrete-continuous mode, ramp/rate, dwell, interlock, communication/readback and fail-safe metadata.

## 5. Rainfall-group design without over-gating

`rainfall_group` is the leakage/statistical unit.

Hard correctness requirements:

- unique `event_id`;
- no rainfall group crosses scientific splits;
- development train/validation are rainfall-group-disjoint;
- Final contains untouched groups and is absent from pre-lock training/tuning;
- referenced event INPs exist.

For publication-strength evidence, target about **160 independent rainfall groups** (for example 96 development / 24 calibration / 16 safety-audit / 24 Final). This is a **recommended study size, not a software execution gate**.

## 6. Fresh study and safe resume

Initialize the final study with `rtc-init-fresh-workspace`. The workspace binds canonical physical/input/split identities; large generated data may live on another local volume.

Reuse requires:

```text
compatible scientific/data contract
+ stable RTC implementation fingerprint
+ matching numerical input/config/timing/action lineage
+ verified generated-artifact hashes
```

The implementation fingerprint represents stable scientific semantics, **not a byte hash of every Python file**. File existence or directory location alone never authorizes reuse.

D0/D1/D2/D3 generation and Step1/Step2 training are resumable. When a scientific/numerical contract changes, affected work is regenerated; unrelated documentation/reporting changes should not invalidate expensive computation.

## 7. Time hierarchy and Phase-0

The SWMM Dynamic-Wave solver keeps its internal routing step. Python observation/model and supervisory-control clocks are separate.

Do not freeze 5 min observation, 10 min control or a 120 min horizon a priori. Phase-0 uses development-only controls-disabled D0/D2 data sampled at `<=60 s` and evaluates:

- readback lag;
- actuator-flow `t10/t50/t90` and peak time;
- network flooding-rate response;
- network maximum-depth response;
- peak-near-horizon censoring.

Do not use sustained-step response-area `mass90` as a hydraulic time constant. Use D3/pulse-release experiments for recovery after control release.

If Phase-0 accepts 300 s observation/model + 600 s control, 13 history frames and first control at 60 min, the t=0-inclusive history is exactly `0,5,...,60 min`. Formal production timing also requires `record_stride_seconds == model_step_seconds`.

## 8. Data roles

### Fixed baseline cache

After production timing is frozen, `rtc-build-baseline-cache` generates each fixed reference once and later stages reuse verified indexes rather than rerunning SWMM.

### D1

`rtc-run-d1-batch` accepts development/train only. D1 provides controlled-state coverage for Step1 and is never a D2/D3 replay prefix.

### D2

D2 starts from replayable controls-disabled No-control prefixes. It records the checkpoint pre-action state/statistics, applies one candidate action, runs SWMM forward and stores future trajectory plus exact cumulative SWMM node flooding-volume change. D2 provides local/boundary action-effect and finite-difference gradient truth.

### D3

D3 starts from the same replayable prefix contract. In each generated sequence **every discovered actuator is eligible**; `change_probability` and `perturbation_std` only control stochastic data coverage. There is no fixed active subset. D3 provides multi-actuator/multi-step interaction data and joint-action ranking evidence.

## 9. Step1/Step2 contracts

Step1 reconstructs the **current** full state from causal history. Compact trajectory spacing must equal the frozen model step. Train/validation are rainfall-group-disjoint and Formal metrics give equal weight to independent rainfall groups.

Step2 shards enforce one immutable:

```text
model_step_seconds
horizon_steps
```

so Phase-0 and production time grids cannot be mixed.

Step2 supervises future hydraulic state, actuator flow and exact cumulative SWMM node flood volume. The predicted-volume operator is identical in Step2 training, validation, MPC, gradient and ranking:

```text
trapezoidal integration of checkpoint/current flooding rate + future predicted flooding rates
```

Both model trainers save atomic epoch resume states with model/optimizer/scaler/RNG lineage.

## 10. Authoritative metrics and acceptance

`Node.flooding` is an instantaneous rate, not a volume.

For node `i` over `[t0,t1]`:

```text
DeltaV_i = cumulative_SWMM_flooding_volume_i(t1)
         - cumulative_SWMM_flooding_volume_i(t0)
TFV = sum DeltaV_i over all hydraulic nodes
PFV = sum DeltaV_i over the eight priority nodes
```

Global Peak is `max_t sum_i max(flooding_rate_i(t),0)` and is obtained by routing-step observation of a frozen-decision replay that preserves the original Python target-write cadence.

Before Policy Lock, the workflow requires:

1. Step1 held-out rainfall-group-balanced reconstruction acceptance;
2. Step2 held-out rainfall-group-balanced trajectory/exact-TFV acceptance;
3. D2 exact-SWMM local/boundary TFV-gradient acceptance;
4. D2 local + D3 joint-sequence ranking/regret acceptance;
5. Proposed development closed-loop SWMM;
6. real-time execution acceptance.

The resolved controller freezes a decision wall-clock budget smaller than the control interval. A stale optimization result is rejected as `FALLBACK_COMPUTE_DEADLINE`.

## 11. Policy Lock, Final and supported local workflow

Current Policy Lock:

```text
WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND
```

After lock, generate/resume four fixed Final references, run locked Proposed once per untouched Final event, formalize each run, replay routing-step Global Peak and compile the complete five-strategy matrix.

Final statistics first collapse variants inside each rainfall group, then give every independent rainfall group equal weight.

Exact local commands and data assets are documented in:

- `docs/LOCAL_RUNBOOK_V06.md` — supported execution order and CLI;
- `docs/FINAL_DATA_INVENTORY_V06.md` — data/evidence inventory and reuse rules;
- `FORMAL_PIPELINE_LATEST.md` — scientific evidence contract.

Install/test the final merged release with:

```powershell
cd E:\RTC_sewer\RTC
git checkout main
git pull
python -m pip install -e ".[dev,swmm]"
python -m pytest -q
```

Do not begin expensive production SWMM generation until Phase-0 has frozen the production time contract.
