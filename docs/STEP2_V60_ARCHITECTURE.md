# Project7 Step2 V6.0 — Control-Latent Dual-Surrogate Rebuild

**Status:** bounded development only. **Do not merge into production Step2 yet.**

V6 is a clean scientific fork from the V5 prototype. It does **not** inherit the V4/V5 response model or legacy multitask loss. Existing D2 remains useful supervision; legacy dense-random D3 is diagnostic-only and is explicitly refused by the V6 training cache.

## Why V6 exists

The V1–V5 history narrowed the Step2 failure to three coupled structural problems:

1. **Action identifiability:** raw control has `109 actuators × 36 blocks = 3924` dimensions, while legacy D3 exposes only a handful of alternatives at each hydraulic state and usually perturbs almost all actuators at once.
2. **Nonlinear multi-actuator response:** the historical `sum(single effects) + interaction residual` decomposition is not a reliable ordering baseline for D3.
3. **Objective interference:** full hydraulic trajectory fidelity and MPC candidate ordering are different learning tasks; one legacy multitask objective caused negative transfer and response collapse.

V6 addresses these problems at the data, representation, objective and optimization levels together.

---

## P0 — frozen redesign

### 1. Low-dimensional control manifold

`step2_control_basis_v60.py` defines the future MPC search coordinates **before** choosing the learning algorithm:

- deterministic topology-aware actuator zones derived from network connectivity and actuator endpoints;
- zones are separated by actuator type, preserving pump/orifice/weir identity;
- six smooth temporal basis functions cover the 36 ten-minute control blocks;
- coefficients are decoded differentiably to the full 109-actuator H72 setting sequence;
- bounds and the frozen maximum setting change per update are applied during decode.

The effective optimization dimension is `control_groups × 6`, rather than 3924 independent settings.

The low-dimensional coefficients are the intended V6 MPC decision variables. Production wiring is intentionally not changed in this branch.

### 2. Targeted D3-v2 instead of legacy dense-random D3

`step2_d3_design_v60.py` creates one exact HOLD plus structured candidates at each Train-only checkpoint:

- single-control-group anchors;
- same-zone coordinated actions;
- cross-zone coordinated actions;
- sparse low-discrepancy combinations;
- multiple action magnitudes;
- all 109 actuators remain globally eligible, but each candidate lies on the V6 control manifold.

The default is 24 candidates/checkpoint. This can be changed only as a frozen data-design contract, not tuned against Validation/Final outcomes.

Every design is bound to:

- sequence SHA;
- V6 control-basis SHA;
- V6 D3 design-contract SHA;
- exact checkpoint/rainfall/event provenance.

### 3. Legacy D3 cannot leak into V6 training

The V6 lineage is deliberately separate:

1. `build_step2_v60_run_index.py` accepts existing D2 + targeted D3-v2 only;
2. every D3 same-prefix group must contain exactly one `D3_HOLD_REFERENCE`;
3. legacy `D3_MULTI_ACTUATOR_ROLLOUT` / `D3_MULTI_ACTUATOR_SEQUENCE` roles are rejected;
4. `compile_step2_shards_v60.py` writes V6 basis/design hashes into the shard manifest;
5. `build_step2_training_cache_v60.py` and `run_step2_v60_guarded.py` verify cache → shard → V6 design lineage before training.

Do not bypass these V6 scripts with the legacy Step2 index/shard path.

---

## P0 — state-conditioned nonlinear control response

### 4. State and control policy are joint boundary conditions

The causal boundary encoder consumes:

- current six-channel hydraulic state;
- causal rainfall horizon;
- reference control policy;
- static network/node attributes.

For actuator `a` at time `t`, interaction is formed only after local state conditioning:

`h[a,t] = phi(physics[a], x_up[a,t], x_down[a,t], u_ref[a,t], u_cand[a,t], delta_u[a,t], time[t])`

The actuator set interaction therefore depends on the hydraulic operating state **before** different actuators interact.

### 5. Direct joint-action response

V6 does not use `sum(D2) + D3 residual` as the multi-actuator forward equation. Candidate and reference actions pass through the same state-conditioned operator, and the response is obtained by structural subtraction. Therefore candidate == reference gives exact zero without a penalty term.

### 6. Stable Koopman/control-latent dynamics

`StableKoopmanControlOperatorV60` evolves a bounded latent control state using:

- a stable diagonal linear Koopman-like core;
- causal control/boundary forcing;
- a bounded nonlinear residual.

This is deliberately not an unrestricted 72-step hydraulic-state autoregressive rollout. It captures delayed control influence in latent space while avoiding the recurrent instability observed in V1/V2.

---

## P0 — two independent Step2 surrogates

### 7. ControlValueSurrogateV60

Purpose: future MPC candidate scoring/optimization.

Primary supervision only:

- authoritative exact cumulative `delta TFV`;
- within-group centred value;
- listwise candidate ranking;
- differentiable regret surrogate.

No hydraulic trajectory loss is allowed to backpropagate into this model.

### 8. HydraulicResponseSurrogateV60

Purpose: hydraulic response fidelity, mechanism interpretation and physical diagnostics.

Outputs:

- node depth/head;
- node flooding rate;
- node volume/storage response;
- managed actuator flow;
- flooding-onset logits.

Physical output structure keeps:

- nonnegative depth;
- nonnegative flooding rate;
- nonnegative volume/inflow/outflow;
- exact `head = invert elevation + depth`.

`DualStep2SurrogateV60.assert_disjoint_parameters()` fails if the Value and Hydraulic models share trainable parameters.

---

## P1 — hydraulic fidelity and long-horizon stability

### 9. Multi-resolution H360

Hydraulic supervision/decoding keeps high resolution where RTC decisions matter most and coarser resolution later:

- approximately 0–30 min: 5-min points;
- 30–120 min: 10-min points;
- 120–360 min: 30-min points;
- final H360 endpoint is always retained.

The Value surrogate still integrates the complete causal H72 latent value-rate sequence using physical elapsed seconds.

### 10. Critical-hydraulic loss

Hydraulic loss is node-balanced and increases supervision for:

- wet nodes using depth/max-depth ratio;
- near-surcharge states using the explicit graph `max_depth + surcharge_depth` contract;
- storage nodes close to capacity using volume/storage-capacity ratio;
- flooding onset via binary classification;
- flooding-rate regression;
- managed-flow response.

No surcharge threshold is guessed from outcome labels.

### 11. Explicit event balancing

`step2_optimization_v60.py` uses one optimizer step per event. Multiple checkpoint groups belonging to the same event contribute `1/groups_in_event` gradient weight. D2 and targeted D3 are source-balanced, and D3 joint training replays event-balanced D2 anchors to preserve the high-SNR single-actuator sensitivity learned from D2.

D2 and D3 keep separate Train-only TFV scales so large D3 responses cannot numerically suppress D2 sensitivity.

---

## P2 — targeted active learning

After the first targeted-D3 training round, a second bounded SWMM batch may be selected with `select_step2_d3_active_learning_v60.py`.

The selector can use only model/manifold quantities:

- ensemble uncertainty;
- predicted candidate-rank margin;
- Value gradient norm;
- control-manifold novelty;
- rainfall-group balancing.

It explicitly rejects score tables containing authoritative future TFV outcomes. Validation and Final are never used for active-learning selection.

---

# Local Codex execution order

Use this exact logical order after syncing the branch. Resolve actual study paths from the existing Project7 readiness/workspace manifests; do not invent alternate INP, graph, checkpoint or D2 assets.

## A. Sync and verify source

```powershell
cd E:\RTC_sewer\Project7\repo
git fetch origin
git switch agent/step2-v60-control-latent-rebuild
git pull --ff-only
python -m pytest -q tests/test_step2_v60.py
python -m pytest -q
```

Do not run legacy `rtc-train-step2-large` for V6.

## B. Audit the old D3 distribution against the new control manifold

```powershell
python scripts/audit_step2_control_manifold_v60.py `
  --graph <FROZEN_GRAPH_SCHEMA_NPZ> `
  --cache-manifest <LEGACY_TRAIN_ONLY_STEP2_CACHE_MANIFEST> `
  --out <V60_WORKDIR>\STEP2_LEGACY_D3_MANIFOLD_AUDIT_V60.json
```

This is diagnostic-only. Legacy D3 must not enter V6 training.

## C. Build the targeted D3-v2 design

Use a forcing/checkpoint-only Train checkpoint table that contains the exact current 109 settings plus existing D3 execution metadata (`inp_path`, `trajectory_metadata_path`, checkpoint/event/rainfall lineage).

```powershell
python scripts/design_step2_d3_v2_v60.py `
  --inp <FROZEN_METHOD_TESTBED_INP> `
  --graph <FROZEN_GRAPH_SCHEMA_NPZ> `
  --checkpoints <TRAIN_ONLY_CHECKPOINT_TABLE> `
  --out <V60_WORKDIR>\D3_V60_MANIFEST.csv `
  --summary-out <V60_WORKDIR>\D3_V60_DESIGN_SUMMARY.json `
  --basis-out <V60_WORKDIR>\CONTROL_BASIS_V60.json
```

Before SWMM, inspect candidate-family counts, active control groups, active actuators, sequence uniqueness, basis SHA and design SHA.

## D. D3 execution preflight — no SWMM first

```powershell
rtc-run-d3-batch `
  --manifest <V60_WORKDIR>\D3_V60_MANIFEST.csv `
  --out-dir <V60_WORKDIR>\d3_v60_runs `
  --control-block-seconds 600 `
  --stride-seconds 300 `
  --census-only
```

Fail closed on endpoint, same-prefix, identity or asset-lineage errors.

## E. Run only targeted Train D3-v2

After the census is clean:

```powershell
rtc-run-d3-batch `
  --manifest <V60_WORKDIR>\D3_V60_MANIFEST.csv `
  --out-dir <V60_WORKDIR>\d3_v60_runs `
  --control-block-seconds 600 `
  --stride-seconds 300 `
  --workers <SAFE_LOCAL_WORKER_COUNT> `
  --asset-root <EXISTING_SIMULATION_ASSET_REGISTRY_IF_USED>
```

This is the only new SWMM generation authorised by the V6 design. It is Train-only. Do not run Validation or Final.

## F. Build the dedicated V6 run index

```powershell
python scripts/build_step2_v60_run_index.py `
  --d2-manifest <EXISTING_D2_MANIFEST> `
  --d2-run-summary <EXISTING_D2_RUN_SUMMARY> `
  --d3-v60-manifest <V60_WORKDIR>\D3_V60_MANIFEST.csv `
  --d3-v60-run-summary <V60_WORKDIR>\d3_v60_runs\D3_RUN_SUMMARY.csv `
  --out <V60_WORKDIR>\STEP2_V60_RUN_INDEX.csv
```

The command must report `legacy_dense_d3_present = false` and non-empty matching basis/design hashes.

## G. Validate and compile V6 shards

First use the existing fresh-workspace validator if the current workspace contract requires it. Then use the V6 compiler, not the legacy CLI:

```powershell
python scripts/compile_step2_shards_v60.py `
  --run-index <V60_WORKDIR>\STEP2_V60_RUN_INDEX.csv `
  --out-dir <V60_WORKDIR>\shards `
  --model-step-seconds 300 `
  --horizon-steps 72
```

## H. Build the lineage-checked mmap cache

```powershell
python scripts/build_step2_training_cache_v60.py `
  --manifest <V60_WORKDIR>\shards\manifest.json `
  --out-dir <V60_WORKDIR>\training_cache
```

## I. Train/evaluate the dual surrogate

Use the guarded entrypoint only:

```powershell
python scripts/run_step2_v60_guarded.py `
  --graph <FROZEN_GRAPH_SCHEMA_NPZ> `
  --cache-manifest <V60_WORKDIR>\training_cache\CACHE_MANIFEST.json `
  --out-dir <V60_WORKDIR>\development `
  --device cuda `
  --seed 42
```

The runner creates a deterministic rainfall-group TrainFit/InternalHoldout split. It does not access Validation6 or Final.

---

# Development decision

Do **not** accept V6 because loss decreased. Review separately:

### Value surrogate

- D2 and D3 candidate rank;
- pairwise accuracy;
- top-1 / regret;
- TFV MAE;
- control-coefficient gradient finite/non-zero;
- no candidate-spread collapse.

### Hydraulic surrogate

- depth RMSE/NSE when available;
- flooding-rate error;
- storage-volume error;
- managed-flow error;
- flooding-onset balanced accuracy;
- critical wet/surcharge/storage strata.

### Structural gates

- candidate == reference exact zero;
- future action cannot affect past output;
- Value/Hydraulic parameter sets are disjoint;
- no legacy dense D3 in cache;
- V6 basis/design hashes survive design → run index → shard → cache;
- Validation/Final untouched;
- production wiring unchanged.

Only if the TrainInternalHoldout value metrics improve materially over the V4.3.3/V4.4.1/V5 evidence without D2 sensitivity collapse should V6 be considered an architecture-lock candidate. Formal thresholds are not changed in this branch.

If targeted D3-v2 is still insufficient after one bounded active-learning round, stop enlarging neural architecture. Re-audit whether the V6 coefficient dimension must be reduced further (fewer hydraulic zones / temporal bases) rather than returning to unrestricted 3924-D actions.
