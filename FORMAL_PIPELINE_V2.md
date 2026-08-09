# Wuhan RTC — Formal Scientific Pipeline V2

This document is the executable research contract for the actuator-agnostic continuous RTC repository. It supersedes any earlier fixed Engineering36, binary-pump, fixed Top-K, or fixed 5/10/120-minute assumptions.

## Frozen scientific question

At decision time `t`, use only causal sparse hydraulic observations, actuator readback and observed rainfall history to reconstruct the current network state, forecast causal rainfall scenarios, predict the hydraulic effect of continuous settings for every eligible SWMM pump/orifice/weir/outlet, and solve a receding-horizon MPC. A candidate is admissible only when all eight observed ponding sites satisfy their own calibrated upper safety bound; one site cannot compensate for deterioration at another. Within the admitted set, minimize network total flooding volume (or its configured forecast-scenario CVaR). Execute only the first control block and re-observe.

Formal conclusions come only from authoritative SWMM runs. The differentiable model is an online search/screening model, not the source of final PFV/TFV/Global-Peak claims.

## Non-negotiable invariants

- No hard-coded binary actuator mask.
- No fixed Engineering36 or fixed active Top-K subset.
- Every actuator discovered from `[PUMPS]`, `[ORIFICES]`, `[WEIRS]`, `[OUTLETS]` remains eligible with continuous setting in `[0,1]` unless separately frozen engineering metadata proves a narrower range.
- Exactly eight real observed ponding nodes are protected site-by-site.
- Step1 estimates the **current** full state; it never predicts a future state from future SWMM truth.
- Step2 learns `setting -> actuator flow -> hydraulic trajectory`; it does not learn a control policy label.
- Future realized SWMM runoff/state/flooding and Final/Locked truth are forbidden online features.
- Rainfall forecast and model-error uncertainty are different objects. Forecast scenarios describe forcing uncertainty; a separate calibration split estimates the one-sided surrogate safety error.
- Calibration rainfall groups, independent safety-audit groups and untouched Final groups must be disjoint.
- The model observation step and the control update step are separate. Step1 history is updated every model step; control can be held over multiple model steps.
- Online fallback is causal. The default is `CAUSAL_HOLD_CURRENT_V1`; authoritative future SWMM/native-rule trajectories may never be queried online to construct fallback.
- Formal PFV/TFV use SWMM cumulative node flooding statistics. Formal Global Peak is the routing-step synchronized network flooding-rate peak, not the sum of per-node individual peaks.

---

# Stage 0 — Install and compile frozen assets

```powershell
pip install -e ".[dev,swmm]"

python -m rtc.formal_assets `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --priority data/priority_nodes.txt `
  --sensors data/sensor_nodes.txt `
  --out-dir outputs/formal_assets
```

Outputs include:

- `graph_schema.npz`
- `state_schema.json`
- `actuator_catalog.json`
- `physical_contract.json`
- `formal_asset_audit.json`

The physical-network fingerprint intentionally excludes event forcing and `[CONTROLS]`, so event-specific rainfall INPs and passive-No-RTC variants can still prove that they use the same physical drainage system.

Do not proceed if the eight priority nodes or sensors are absent from the INP.

# Stage 1 — Rainfall-group scientific split

Create an event registry containing at minimum `event_id`, `rainfall_group` and the event INP/forcing lineage, then:

```powershell
rtc-split-groups --input events.csv --out outputs/split_registry.csv --seed 42
```

The split is by whole rainfall group:

- `development`
  - `train`
  - held-out `validation`
- `calibration`
- `safety_audit`
- `final`

No rainfall group may occur in more than one role. `final` remains untouched until after Policy Lock.

# Stage 2 — Phase-0 hydraulic time-scale pilot

Do **not** assume a 5-minute model step, 10-minute update or 120-minute horizon from an older project. Use representative dry/wet/rising/high-water states and D2 response probes to estimate response and settling times. Freeze the resulting values in a copy of:

- `configs/formal_time_scale.template.json`
- `configs/formal_controller.template.json`

Replace every placeholder before Formal execution. The strict Policy Lock rejects inconsistent controller/time-scale files.

# Stage 3 — D0/D1 full-event hydraulic trajectories

Generate authoritative full-event trajectories with **no Python actuator writes**. Use the original INP for native-rule trajectories and a separately hashed passive-No-RTC INP when needed.

```powershell
rtc-run-trajectory `
  --inp <event.inp> `
  --out-dir outputs/d0_d1/<event> `
  --run-id <event_id> `
  --stride-seconds <MODEL_STEP_SECONDS>
```

Build a run index with `metadata_path,event_id,rainfall_group,scientific_split,development_fold`.

D0/D1 are used for Step1 state reconstruction and base hydraulic dynamics. They are not Final evidence unless rerun under the locked Final protocol.

# Stage 4 — Step1 causal sparse-state reconstruction

```powershell
rtc-compile-step1 `
  --run-index outputs/d0_d1/run_index.csv `
  --sensors data/sensor_nodes.txt `
  --history-steps <FROZEN_HISTORY_STEPS> `
  --out outputs/step1/windows.npz

rtc-train-step1 `
  --dataset outputs/step1/windows.npz `
  --graph outputs/formal_assets/graph_schema.npz `
  --out outputs/models/step1.pt
```

Training consumes only `development/train` rainfall groups.

Evaluate only on `development/validation` groups using a numeric threshold JSON copied from the `step1` section of the preregistered `MODEL_ACCEPTANCE_CONTRACT_V2`:

```powershell
rtc-accept-step1 `
  --dataset outputs/step1/windows.npz `
  --graph outputs/formal_assets/graph_schema.npz `
  --model outputs/models/step1.pt `
  --priority data/priority_nodes.txt `
  --thresholds outputs/contracts/step1_thresholds.json `
  --out outputs/acceptance/step1.json
```

Training loss is never model-acceptance evidence.

# Stage 5 — D2 independent same-checkpoint actuator probes

Each branch starts from the **same fresh SWMM prefix**. Never execute pulse A, then pulse B on the modified state.

Prepare checkpoints containing the complete baseline setting vector for every discovered actuator, plus event/split lineage.

```powershell
rtc-design-probes `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --checkpoints outputs/checkpoints.csv `
  --out outputs/d2/probe_manifest.csv `
  --epsilon <DEVELOPMENT_VALUE>

rtc-run-probes `
  --manifest outputs/d2/probe_manifest.csv `
  --out-dir outputs/d2/runs `
  --horizon-minutes <FROZEN_HORIZON_MINUTES> `
  --stride-seconds <MODEL_STEP_SECONDS>
```

Every D2 branch records:

- exact pre-action state;
- requested/target/current setting;
- actuator flow;
- node depth/head/flooding/volume;
- subcatchment rainfall/runoff for audit;
- cumulative SWMM node flooding statistics at checkpoint and branch end.

The exact post-action flooding-volume truth is the SWMM cumulative-statistics difference, not a coarse sampled-rate integral.

# Stage 6 — D3 multi-actuator interaction/free-rollout data

D3 is training coverage for interaction and long-rollout dynamics. Random sparse perturbation is allowed for efficient **data design**, but it never becomes an online fixed active subset.

```powershell
rtc-design-d3 `
  --inp data/wuhan_v8_storage_retrofit.inp `
  --checkpoints outputs/checkpoints.csv `
  --out outputs/d3/manifest.csv `
  --horizon-steps <FROZEN_HORIZON_STEPS>

rtc-run-d3 `
  --manifest outputs/d3/manifest.csv `
  --out-dir outputs/d3/runs `
  --control-block-seconds <CONTROL_UPDATE_SECONDS> `
  --stride-seconds <MODEL_STEP_SECONDS>
```

D4 active-learning selection may add states with high error/uncertainty/gradient disagreement. D4 may only draw from the development/calibration/audit role appropriate to the intended use; it must never draw Final rainfall truth before Policy Lock.

# Stage 7 — Step2 trajectory world model

Compile development/train D2+D3 branches into full-trajectory tensors. Future SWMM runoff is deliberately excluded from formal online inputs; only causal rainfall is an exogenous input.

```powershell
rtc-compile-step2 `
  --run-index outputs/step2/train_run_index.csv `
  --development-fold train `
  --out outputs/step2/train.npz

rtc-train-step2 `
  --dataset outputs/step2/train.npz `
  --graph outputs/formal_assets/graph_schema.npz `
  --out outputs/models/step2.pt
```

Compile a separate `development/validation` tensor set and run:

```powershell
rtc-accept-step2 `
  --dataset outputs/step2/validation.npz `
  --graph outputs/formal_assets/graph_schema.npz `
  --model outputs/models/step2.pt `
  --priority data/priority_nodes.txt `
  --thresholds outputs/contracts/step2_thresholds.json `
  --out outputs/acceptance/step2.json
```

Step2 acceptance must cover hydraulic trajectory errors, managed actuator flow, horizon-end behavior, PFV/TFV/Global-Peak-derived metrics, event-balanced metrics and candidate/action-effect skill defined by the frozen acceptance contract.

# Stage 8 — Gradient truth and candidate-ranking truth

Gradient MPC is not authorized merely because PyTorch can back-propagate.

Use held-out development/validation D2 lower/center/upper triplets:

```powershell
python -m rtc.formal_gradient `
  --manifest outputs/d2/probe_manifest.csv `
  --run-summary outputs/d2/runs/RUN_SUMMARY.csv `
  --graph outputs/formal_assets/graph_schema.npz `
  --step2 outputs/models/step2.pt `
  --priority data/priority_nodes.txt `
  --thresholds outputs/contracts/gradient_thresholds.json `
  --out outputs/acceptance/gradient.json
```

SWMM finite differences use exact cumulative flooding-volume truth. Model gradients use autograd with the same action tied over the branch horizon.

Candidate ranking is a separate gate so repeated D2 center actions cannot bias the statistic:

```powershell
python -m rtc.formal_ranking `
  --manifest outputs/d2/probe_manifest.csv `
  --run-summary outputs/d2/runs/RUN_SUMMARY.csv `
  --graph outputs/formal_assets/graph_schema.npz `
  --step2 outputs/models/step2.pt `
  --priority data/priority_nodes.txt `
  --thresholds outputs/contracts/ranking_thresholds.json `
  --out outputs/acceptance/ranking.json
```

Ranking deduplicates by event/checkpoint/action SHA.

# Stage 9 — Independent safety calibration

Build calibration residual cases automatically from D2 branches assigned to the `calibration` split:

```powershell
python -m rtc.calibration_cases `
  --manifest outputs/d2/probe_manifest.csv `
  --run-summary outputs/d2/runs/RUN_SUMMARY.csv `
  --graph outputs/formal_assets/graph_schema.npz `
  --step2 outputs/models/step2.pt `
  --priority data/priority_nodes.txt `
  --out outputs/calibration/cases.csv

rtc-calibrate-safety `
  --input outputs/calibration/cases.csv `
  --priority data/priority_nodes.txt `
  --coverage <PREREGISTERED_COVERAGE> `
  --out outputs/calibration/sitewise_ucb.json
```

The calibration residual is `truth - prediction` at each protected site. Do not estimate this UCB on safety-audit or Final data.

# Stage 10 — Independent pre-lock safety audit

Use only rainfall groups assigned to `safety_audit`:

```powershell
python -m rtc.safety_audit_cases `
  --manifest outputs/d2/probe_manifest.csv `
  --run-summary outputs/d2/runs/RUN_SUMMARY.csv `
  --graph outputs/formal_assets/graph_schema.npz `
  --step2 outputs/models/step2.pt `
  --priority data/priority_nodes.txt `
  --calibration outputs/calibration/sitewise_ucb.json `
  --budget-config outputs/contracts/controller.json `
  --out outputs/safety_audit/selected_cases.csv

python -m rtc.formal_safety_audit `
  --cases outputs/safety_audit/selected_cases.csv `
  --priority data/priority_nodes.txt `
  --calibration outputs/calibration/sitewise_ucb.json `
  --budget-config outputs/contracts/controller.json `
  --out outputs/safety_audit/evidence.json
```

The formal audit re-checks that calibration and audit rainfall groups do not overlap, then reports selected-action false-safe rate, event-balanced false-safe rate, site-wise empirical coverage and fallback frequency.

This D2 audit validates the calibrated site-wise admission mechanism on independent states. It does **not** replace the subsequent full closed-loop development evaluation.

# Stage 11 — Full causal closed-loop development run

Run the real proposed chain on development-only events:

```powershell
rtc-run-policy `
  --strategy proposed `
  --inp <event.inp> `
  --out-dir outputs/dev_closed_loop/<event> `
  --run-id <event_id>__proposed `
  --sensors data/sensor_nodes.txt `
  --priority data/priority_nodes.txt `
  --config outputs/contracts/controller.json `
  --graph outputs/formal_assets/graph_schema.npz `
  --step1 outputs/models/step1.pt `
  --step2 outputs/models/step2.pt `
  --calibration outputs/calibration/sitewise_ucb.json
```

Also run the frozen baselines listed in `configs/formal_baseline_plan.v2.json` using the same event forcing and physical network.

The production runner separates observation and control cadences. For example, if a pilot eventually freezes a 5-minute model step and 10-minute control update, Step1 still receives 5-minute history while each MPC control block is held for two model steps. Do not infer these example values as the Wuhan contract.

A decision fails closed to the causal fallback when history is not ready, readback fails, the optimizer fails, the result is non-finite, or the calibrated site-wise safety gate rejects it.

# Stage 12 — Record the fail-closed stage ledger

After each stage passes, record its hashed evidence with `rtc-record-stage`. Stages cannot be passed out of order. The ledger re-hashes earlier evidence before Policy Lock so a modified report cannot silently remain “passed”.

# Stage 13 — Strict Policy Lock

First freeze numeric copies of the templates:

- controller configuration;
- rainfall forecast configuration;
- time-scale contract;
- model acceptance contract;
- baseline plan;
- fallback policy;
- split registry.

Prepare `formal_policy_artifacts.json` from `configs/formal_policy_artifacts.template.json`, then:

```powershell
python -m rtc.formal_lock_v3 `
  --ledger outputs/evidence/pipeline_ledger.json `
  --artifacts outputs/contracts/formal_policy_artifacts.json `
  --out outputs/policy_lock/policy_lock.json
```

Lock revision 3 verifies:

- all preceding scientific stages passed;
- all prior evidence hashes are unchanged;
- exactly eight priority nodes;
- sensor nodes are present in the locked graph;
- rainfall groups do not cross scientific roles;
- `development/train` and `development/validation` are group-disjoint;
- Step1/Step2/gradient/ranking actually used the preregistered thresholds;
- acceptance model SHA equals the model being locked;
- gradient/ranking evidence comes from the locked Step2 model;
- independent safety audit passed and reports zero calibration/audit group overlap;
- controller time scales equal the frozen time-scale contract;
- runtime rainfall-forecast settings equal the separately frozen forecast file;
- baseline plan and fallback are frozen;
- frozen INP, graph, state schema, actuator catalogue and physical-network fingerprint are hashed.

No hyperparameter, model, threshold, split, fallback, forecast, baseline or controller setting may change after this point without invalidating the lock and rerunning downstream evidence.

# Stage 14 — Untouched Final

Only now run every event whose rainfall group is locked as `final`, for every strategy in the locked baseline plan. Build a Final run index containing:

`event_id,rainfall_group,strategy,metadata_path`

Every formal run must use `exact_global_peak=true`.

Compile Final evidence with the strict V2 compiler:

```powershell
python -m rtc.formal_final_v2 `
  --policy-lock outputs/policy_lock/policy_lock.json `
  --run-index outputs/final/run_index.csv `
  --out-dir outputs/final/evidence
```

The compiler re-hashes every locked artifact, checks every Final rainfall group, checks the complete event × strategy matrix, verifies the forcing-independent physical-network hash for every run, and refuses sampled PFV/TFV or non-exact Global Peak.

Formal KPI truth is:

- `TFV`: sum of SWMM cumulative node flooding volumes;
- `PFV`: sum of cumulative flooding volumes at the eight observed priority nodes, with all eight node-specific values also retained;
- `Global Peak`: maximum over routing steps of the synchronized sum of node flooding rates;
- priority maximum depth: reported per protected site;
- flow-routing error and all evidence SHA values: retained for audit.

## Final evidence boundary

Passing unit tests or completing this software pipeline does **not** prove that the proposed controller improves Wuhan flooding. That claim requires the frozen Wuhan INP, event forcing, trained accepted models, calibration, independent audit, Policy Lock and untouched authoritative Final SWMM results. The repository is designed to fail closed until those real evidence files exist.
