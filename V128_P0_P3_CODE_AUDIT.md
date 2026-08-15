# Project7 current P0-P3 code audit and remediation ledger

Status: **current development implementation, not Policy Locked**. Current priority is rapid Proposal diagnosis and model selection; expensive full/downstream evidence comes only after smoke/dev promotion.

Frozen target remains:

```text
causal sparse sensing -> Step1 current full-network state
-> typed differentiable hydraulic/action Step2
-> 12 x 109 H120 continuous MPC inside H360
-> execute first 600 s only -> authoritative SWMM re-observation
```

TFV primary; Priority8 PFV one-sided soft secondary; Global Peak report-only.

## P0 — correctness, routing and diagnostic gates

### P0.1 Exact pair coverage failure 544/542 — FIXED

The first full V128 attempt failed inside the exact H360 objective, not from CUDA OOM. Pair census reduced float32 SWMM node-volume labels in NumPy float64 while live CUDA truth/loss was float32. Around the frozen 1 m3 informative threshold the same candidate pair could therefore be classified differently.

`src/rtc/step2_train_v128_exact.py` now precomputes one canonical float32 candidate TFV/delta tensor. Census, reported pair loss, candidate delta losses and live directed pair gradients all consume that same tensor. The objective contract is V4 canonical-float32 truth, and a deterministic regression test reproduces the precision-drift class.

### P0.2 Accidental full runs — FIXED

`scripts/run_step2_current.py` now requires explicit `--profile smoke|dev|full`; no default exists. The obsolete full-only `scripts/run_step2_v128_control_4060.py` was deleted.

- smoke: tiny deterministic Development subset, nonfinal;
- dev: larger deterministic Development proxy, nonfinal;
- full: canonical 112 D2 + 112 D3 + 33 D4-FIT and complete curriculum.

Smoke/dev keep 109 actuators, H360 and the exact pairwise code path. They reduce statistical coverage/repetitions only and cannot enter D5/runtime/Policy Lock.

### P0.3 Stage checkpoint / resume — ADDED

`src/rtc/stage_checkpoint_v128.py` writes NONFINAL stage-boundary checkpoints after Stage A, B0 and objective. It stores graph/data/design identity plus Python/NumPy/Torch RNG and fails closed on mismatches. `--stop-after-stage` and `--resume-from` prevent repeated Stage-A/B0 work after compatible interruptions. Stage checkpoints are deliberately incompatible with final/runtime loaders.

### P0.4 Resource/profiler evidence — ADDED

`TRAINING_TELEMETRY.jsonl` records stage wall time, RSS, available RAM/swap and CUDA memory. `--profile-one-group --torch-profiler` exercises the real H360/exact code path on bounded data and exports a Chrome trace. The previous full run showed severe host paging while GPU utilisation remained partial; performance work must therefore start from profiling rather than blind chunk growth.

### P0.5 Spatial-distance evidence — ADDED

`src/rtc/spatial_diagnostics_v128.py` and `scripts/audit_step2_spatial_current.py` quantify held-out D2 action-effect sign/magnitude at 1-3, 4-6, 7-12 and 13+ actuator-to-node graph hops. This is the required gate before attributing poor TFV control to long-range graph propagation.

`scripts/audit_step1_global_attention_current.py` compares frozen legacy Step1 with separately trained V122 sensor-to-all-node attention on identical Development validation windows and reports depth error by distance to nearest sensor. Sample indices are carried explicitly through the trajectory-group sampler so rainfall-group evidence cannot be misattributed.

## P1 — representation experiments, not automatic promotion

### P1.1 Typed actuator representation — RETAINED

Current V128 direction-aware actuator messages retain endpoint state, setting, previous/predicted managed flow, responsiveness, type/physics and actuator identity. Managed-flow injection remains a separate locally conservative pathway.

### P1.2 Edge physics — DEVELOPMENT PATH ADDED

The current graph topology is correct, but generic node message passing does not distinguish physical conduit properties per edge. New development components:

```text
src/rtc/edge_physics_current_v128.py
scripts/build_edge_physics_current.py
src/rtc/step2_differentiable_v128_edge.py
scripts/run_step2_edge_aware_dev.py
scripts/audit_step2_edge_spatial_current.py
```

They reuse audited SWMM link parsing semantics to align conduit length/roughness/slope/geometry/loss/type/orientation descriptors with current graph edges, and add current head difference plus length-normalized head-gradient messages. The V128 typed actuator pathway is retained. The edge-aware runner forbids `--profile full`; promotion requires held-out spatial/ranking improvement.

### P1.3 Ordinary conduit flow supervision — GATED, NOT FABRICATED

The current training cache exposes node states and managed actuator flows, but no authoritative ordinary-conduit dynamic-flow label field was found. `src/rtc/physics_diagnostics_v128.py` reports this readiness explicitly. No synthetic conduit-flow target is introduced. Extend the SWMM data contract only if edge-aware smoke/dev evidence demonstrates enough value to justify new data production.

### P1.4 Step1 global attention — ABLATION, NOT HOT SWAP

V122 global sensor-to-all-node attention already exists. It is now evaluated as an explicit distance-stratified ablation. If promoted, causal Step1 state stores must be rebuilt and Step2 retrained from the beginning; current frozen state stores cannot be reused under a changed Step1 model.

## P2 — long-range structure and physics diagnostics

### P2.1 Do not solve long range by depth alone

The development plan may compare modest graph depths, but no 10/20-layer default is introduced. Long-range performance is first measured. Edge-aware messages and optional sparse influence shortcuts are preferred experiments when far-field evidence fails.

### P2.2 Development-only hydraulic influence graph — ADDED

`src/rtc/hydraulic_influence_v128.py` and `scripts/build_hydraulic_influence_current.py` derive sparse remote actuator-to-node shortcut candidates from Development TrainFit D2 same-prefix SWMM effects only, including graph hops, effect magnitude/sign and first detectable flooding-rate effect time. The artifact is not enabled in full and can never be built from Validation/Final/Formal outcomes.

### P2.3 Continuity/physics loss — DIAGNOSTIC FIRST

EPA SWMM couples conduit St-Venant routing with volume conservation at nodes. Current Step2 state omits some node loss terms and ordinary conduit dynamic flow, so directly enforcing a supposedly exact node mass-balance loss would be scientifically unsafe. `src/rtc/physics_diagnostics_v128.py` therefore implements a clearly labeled continuity proxy for gross-diagnostic use and keeps `training_loss_enabled=false` until the state/data contract contains the required authoritative terms.

### P2.4 H360/rainfall — UNCHANGED UNTIL SPATIAL/ACTION IDENTIFICATION IS DIAGNOSED

Default rainfall remains causal persistence/decay. H360/H120/first-10-min receding control is retained while spatial/ranking/gradient failure modes are resolved. Horizon sensitivity comes later so multiple scientific dimensions are not changed simultaneously.

## P3 — strict artifacts and current surface

Current strict checkpoint contract:

`PROJECT7_V128_STEP2_CHECKPOINT_V6_CURRENT_PROFILE_TRAINING_SOURCE_STRICT`

It requires explicit full-profile evidence, current execution-profile contract, exact model/training-source fingerprints, graph/time/schema identity and development lineage. Smoke/dev artifacts are rejected by runtime. Historical D5 remains valid only for its frozen idealized 0.5-per-10-min decoder space.

Current user surface:

```text
CODEX_START_HERE.md
rtc-current-preflight
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

Historical/shared V127 files remain only when still required for frozen lineage, reproducibility or a shared audited implementation. Dead current-facing files are deleted rather than left as competing entrypoints.

## Promotion evidence

Before full: smoke/dev must diagnose the intended failure mode with ranking/gradient/spatial evidence and execution telemetry. Before Policy Lock: the selected explicit full checkpoint must then provide same-checkpoint ranking/D2/D5 evidence, H30-H360 behavior, authoritative Development TFV/PFV, score==execute/continuity/readback and every guarded control callback <600 s. Code merge, smoke PASS or dev PASS is never Policy Lock.
