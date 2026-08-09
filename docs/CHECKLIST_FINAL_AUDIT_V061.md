# Checklist final audit — Wuhan RTC v0.6.1

This document maps the final engineering/scientific checklist to the implemented repository contract. `PASS` means the code path and fail-closed contract are present and reviewable. `FIXED v0.6.1` identifies defects found by this checklist audit. `LOCAL SWMM REQUIRED` means no repository/CI review can honestly substitute for execution of the frozen Wuhan model.

## 0. One terminology correction before applying the checklist

The supplied checklist used PFV to mean a rolling peak flooding flow. That is **not the frozen project definition**.

For this repository:

```text
TFV = Total Flood Volume
    = sum of cumulative SWMM node flooding-volume change over all hydraulic nodes

PFV = Priority Flood Volume
    = same cumulative-volume change restricted to the eight verified priority nodes

Global Peak
    = max over routing time of the simultaneous network sum of positive flooding rates
```

`Node.flooding` is an instantaneous rate. SWMM node `peak_flooding_rate` is also a rate statistic. Neither is PFV in this study.

PFV/depth are soft/diagnostic only; they are not hard MPC safety gates.

## I. Project objective and architecture

| Checklist item | Status | Audit conclusion |
|---|---|---|
| Sparse sensing -> full-state reconstruction -> differentiable hydraulics -> rolling MPC | PASS | Public workflow is Step1 -> Step2 -> continuous TFV-first MPC. |
| Only causal observations/rain/readback online | PASS | Event ID, future realised rainfall/runoff/state/flooding, future Internal trajectory and Final truth are forbidden by the runtime/scientific contract. |
| No predefined rainfall-specific policy | PASS | Online forecast is generated from causal rainfall history; event ID is not a policy feature. |
| No fixed facility subset / no Engineering36 / no runtime Top-K | PASS | Full discovered actuator catalogue is passed to MPC; active actuator count is diagnostic only. |
| Do not artificially binarise pumps | PASS | SWMM setting space remains continuous `[0,1]`; field-deployment capability is kept as a separate engineering question. |
| Historical Project6 failure modes isolated | PASS | v0.6+ uses new causal/data/model contracts; historical RTC results without compatible lineage are not Formal evidence. |

Important Project6 failure modes explicitly addressed in v0.6+ include stale/incomplete outputs, No-control/Internal confusion, future-information leakage, inconsistent TFV/PFV semantics, mixed Step2 time grids, duplicated D2 centre branches, weak cache resume, boundary MPC gradients and row-count-biased statistics.

## II. Frozen INP and units

| Checklist item | Status | Audit conclusion |
|---|---|---|
| INP node/link/subcatchment/actuator census | PASS + LOCAL PREFLIGHT | The audited Wuhan V8 lineage is 932 hydraulic nodes, 1,167 conduits, 3,731 subcatchments and 109 writable pump/orifice/weir links. Actual local file is re-audited before generation. |
| FLOW_UNITS/system-unit conversion | PASS | Runtime tensors convert flow to m3/s, length to m, volume to m3 and rainfall to mm/h. CMS input is identity-converted. |
| SWMM routing vs Python callback clocks | PASS | Dynamic-Wave routing remains the INP routing step; Python model/control callbacks are higher-level clocks. |
| Priority-node membership in actual INP | LOCAL SWMM/INP REQUIRED | `rtc-inp-audit-v2` fails if any frozen priority ID is absent. Historical lineage is not allowed to bypass current membership checking. |

## III. Data generation and reuse

### Required data roles

```text
rainfall/event registry
Phase-0 high-frequency No-control + D2 response data
fixed baseline cache
D1 development/train Step1 coverage
D2 local same-prefix action-effect branches
D3 joint multi-actuator/multi-step branches
Step1 train/validation indexes
Step2 D2+D3 run index and fixed-time shards
model checkpoints/train states
acceptance evidence
Proposed development closed-loop evidence
Policy Lock
untouched Final five-strategy SWMM evidence
```

| Checklist item | Status | Audit conclusion |
|---|---|---|
| Incremental/resumable SWMM generation | PASS | D0/D1/D2/D3 and fixed baselines use deterministic generation/cache keys plus generated-artifact SHA verification. File existence alone is insufficient. |
| Interrupted GPU training resumes | PASS | Step1/Step2 atomic epoch state stores model, optimiser, scaler and RNG state and validates the training contract before resume. |
| Baselines generated once per event | PASS | Baseline cache writes reusable indexes and verifies executed strategy semantics. |
| Compact storage | PASS | Successful runs default to compressed tensors + statistics + metadata/decision evidence; raw row CSV and `.rpt/.out` are debug-only. |
| Downstream-required fields present | PASS | Compact branches include state, rainfall, setting/readback, actuator flow, node/actuator IDs and exact SWMM node-flood-volume label; provenance indexes carry event/rainfall/split/checkpoint/action hashes. |

This project is model-based MPC rather than reinforcement learning; therefore an RL `reward` column is not required. Step2 directly learns `x_t + action + rainfall -> future states/flows`, and MPC constructs the TFV objective from predicted trajectories.

## IV. Time causality and control chain

Correct runtime order is:

```text
observe state/rain/readback at t
-> append causal history
-> Step1 reconstruct x_t
-> forecast rainfall from history <=t
-> optimise Step2/MPC
-> project first move
-> write SWMM target
-> hold until next control epoch
-> read back target/current setting and flow
-> next decision
```

| Checklist item | Status | Audit conclusion |
|---|---|---|
| t=0 causal frame retained | PASS | Corrected history contains the true initial observation before supervisory writes. |
| 1/5/10-min clock distinction | PASS | Phase-0 <=60 s is separate from production model and control clocks; production time is frozen only after response analysis. |
| Step2 transition alignment | PASS | D2/D3 record pre-action `x_t`; interval action governs the subsequent transition to future states. |
| Mixed Phase-0/production Step2 grids rejected | PASS | Shard compilation and model loading require one frozen `model_step_seconds` and `horizon_steps`. |
| D3 model-step vs control-block horizon | **FIXED v0.6.1** | Previous CLI could interpret a 24-step 5-min model horizon as 24 ten-minute control blocks. New `rtc-design-d3` reads the controller config and derives control blocks automatically; public execution rechecks the design timing. |
| Full causal history before first MPC | PASS | `CausalTimingContract` validates t=0-inclusive history span and first-control grid. |

For a representative accepted 5/10/120-min design:

```text
model step       = 300 s
control update   = 600 s
model horizon    = 24 steps
control block    = 2 model steps
D3 sequence      = 12 control blocks
```

## V. No-control versus Internal-RTC

| Checklist item | Status | Audit conclusion |
|---|---|---|
| No-control independent from Internal | PASS | No-control removes executable `[CONTROLS]` and makes no Python writes. Internal retains native `[CONTROLS]` and receives no Python writes. |
| Preserve intrinsic equipment behaviour | PASS | Pump curve, initial state and `[PUMPS]` Startup/Shutoff logic remain. Removing `[CONTROLS]` does not mean deleting physical/local pump properties. |
| Baseline execution validated after run | PASS | Cache validation checks actual `[CONTROLS]` payload, controller presence, decision logs, commanded all-open/all-closed values and physical-network identity. |

## VI. Priority nodes and online facility selection

Frozen priority IDs:

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

They were historically reconciled to the same audited Wuhan physical lineage. Formal use remains conditional on current-INP membership preflight.

Online facility selection is implicit continuous optimisation over all discovered writable actuators: there is no preselected active list. A facility is operationally “selected” when its optimised first-move setting differs from the current readback.

## VII. Step1 / Step2 training and physical logic

| Checklist item | Status | Audit conclusion |
|---|---|---|
| Step1 input/output time alignment | PASS | Causal sparse history and context end at the target current state; validation is group-disjoint. |
| Step2 `state/action/next_state` equivalent samples | PASS | D2/D3 compile into initial state, rainfall sequence, setting sequence, previous flow, future states/flows and exact flood-volume truth. |
| Exact flood-volume training target | PASS | Formal Step2 includes exact cumulative SWMM node-volume supervision in addition to state/flow losses. |
| Same predicted TFV operator in train/validation/MPC/ranking | PASS | Current + future flooding rates use one trapezoidal operator everywhere. |
| Strict mass conservation | NOT CLAIMED | Architecture is physics-informed, not a mathematically exact continuity solver. Managed actuator flow is injected with upstream/downstream signs, but the learned graph residual is not an exact SWMM solver. |
| Basic physical plausibility diagnostics | **ADDED v0.6.1** | Step2 acceptance reports negative depth/flood-rate/node-volume fractions and non-finite state/flow fractions, rainfall-group balanced. |
| Gradient at 0/1 actuator bounds | PASS | Direct projected setting optimisation preserves inward gradients; D2 finite differences use feasible one-sided truth at bounds. |
| Joint action validation | PASS | D2 local evidence is supplemented by D3 joint multi-actuator/multi-step ranking and regret. |

## VIII. Rainfall-event count

Correctness is governed by group leakage, not an arbitrary hard row count. The software requires development train/validation separation and untouched Final groups.

For paper-strength evidence, about 160 independent rainfall groups remains the current recommended design, not an execution blocker. The actual number should be justified from event diversity, rainfall descriptors, computational budget and convergence/sensitivity evidence.

## IX. Code quality, I/O and compute

| Checklist item | Status | Audit conclusion |
|---|---|---|
| Clear package/core layout | PASS | Core logic is under `src/rtc`; public executable workflow is exposed through installed CLI entry points. A separate `scripts/` folder is not required for correctness. |
| Avoid disk inflation | PASS | Compact files are default; engine output/raw CSV are opt-in/debug and normally deleted. |
| Hash checks not over-gating | PASS | Scientific implementation fingerprint is semantic/stable; numerical inputs/config/actions/artifacts are separately hashed. |
| CPU parallelism | PASS | Data generation uses independent process workers; recommended target is 16 processes x 1 SWMM thread/process to avoid oversubscription. |
| RTX 4060 8 GB | PASS | Formal large trainers support AMP, small micro-batches and gradient accumulation. |
| Errors fail visibly | PASS | Missing fields, time mismatches, leakage, invalid settings, stale artifacts and incompatible model/runtime contracts raise explicit errors or controlled fallbacks. |

## X. E2E and Policy Lock completeness

| Checklist item | Status | Audit conclusion |
|---|---|---|
| Formal static assets executable from public CLI | **FIXED v0.6.1** | Added `rtc-compile-formal-assets`; no manual Python import is required. |
| D3 design executable without manual horizon conversion | **FIXED v0.6.1** | Public design reads the frozen controller config. |
| Pipeline ledger executable from public CLI | **FIXED v0.6.1** | Added `rtc-record-pipeline-stage`, which hashes evidence and enforces stage order. |
| Code/unit CI | MUST PASS BEFORE MERGE | This audit branch is not mergeable as final until its exact head passes GitHub Actions. |
| Full Wuhan SWMM end-to-end scientific run | **LOCAL SWMM REQUIRED** | The large frozen INP/data are not stored in GitHub CI. The local runbook must be executed on the real model. |
| Proposed significantly reduces Final TFV | **NOT YET A VALID CLAIM** | Must be demonstrated on untouched Final SWMM results after Policy Lock. No code review can manufacture this result. |
| PFV “acceptable” | REPORT, NOT HARD GATE | Report priority flood-volume/depth deterioration honestly. TFV is primary; PFV may worsen if the TFV-first near-optimal/soft-priority policy chooses that trade-off. |

## XI. Final admissibility rule

The codebase may be called **implementation-complete** after the v0.6.1 audit PR passes CI and is merged. The research may be called **scientifically complete** only after the actual local Wuhan workflow passes, in order:

```text
INP/priority preflight
rainfall split validation
Phase-0 timing
fixed baseline cache + D1 coverage
Step1 acceptance
D2+D3 generation
Step2 acceptance + physical diagnostics
D2 gradient acceptance
D2+D3 joint ranking acceptance
Proposed development closed-loop
runtime/readback/deadline acceptance
Policy Lock
untouched five-strategy Final SWMM
```

Only the final step can support a performance statement such as “Proposed reduces cumulative TFV relative to No-control/Internal-RTC.”
