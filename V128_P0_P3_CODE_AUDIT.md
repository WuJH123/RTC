# Project7 current P0-P3 code audit and remediation ledger

Status: **current development implementation, not Policy Locked**. Merging to `main` removes repository routing ambiguity; it does not constitute scientific promotion or Final evidence.

Frozen target remains:

```text
causal sparse sensing -> Step1 current full-network state
-> typed differentiable hydraulic/action Step2
-> 12 x 109 H120 continuous MPC inside H360
-> execute first 600 s only -> authoritative SWMM re-observation
```

TFV is primary; Priority8 PFV is one-sided soft secondary; Global Peak is report-only.

## P0 — execution and artifact identity

### P0.1 Conflicting current guides

`CODEX_START_HERE_V069.md`, `CODEX_START_HERE_V127.md` and `CODEX_START_HERE_V128.md` each claimed current/single status. They were removed. `CODEX_START_HERE.md` is the only root execution guide.

Stable user/Codex entrypoints are:

```text
rtc-current-preflight
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

### P0.2 Current contract still selected historical V127 user routes

`configs/step2_current_contract.json` and `configs/project7_execution_registry.json` now select one unversioned current surface. Remaining V127-named files are explicitly archival or audited shared orchestration, not user entrypoints.

### P0.3 Exact pairwise implementation existed but canonical training did not call it

`src/rtc/step2_train_v128_exact.py` already implemented the two-pass exact first-order pairwise algorithm, while the canonical runner still imported the older detached-memory `src/rtc/step2_train_v128.py`.

Current training now imports only `step2_train_v128_exact`. The obsolete detached-memory module was deleted. Regression tests inspect the real runner source and prohibit the old module from reappearing.

### P0.4 Typed model versus inherited Stage-A teacher forcing

Current `src/rtc/step2_train_v128_hydraulic.py` owns Stage A and uses the same typed endpoint-state/setting/flow/physics/identity action pathway as online rollout.

### P0.5 Checkpoint training-source strictness was incomplete

The previous loader compared model-source SHA but did not compare training-source SHA with current code. Current contract is:

`PROJECT7_V128_STEP2_CHECKPOINT_V5_MODEL_BASE_D5_TRAINING_SOURCE_STRICT`

It fails closed on either model-source or training-source mismatch. The training fingerprint includes base Stage A/B0/B semantics **and** D5-FIT gradient training plus the frozen D5 fraction decoder, because D5 changes the final runtime weights. V127/older-V128 checkpoints are rejected by contract.

### P0.6 Historical D5 decoder space versus current engineering decoder

D5 gradient assets use the historical 0.5-per-10-min fraction decoder. A regression test now proves that this decoder is numerically identical to the V128 decoder **only when** the current engineering envelope is the idealized graph-bounds + exact 0.5-per-10-min envelope. A custom per-actuator envelope therefore requires a new matching D5 experiment; old D5 evidence cannot be relabeled.

## P1 — control-identification and engineering semantics

### P1.1 Large-event action-order deadband

Current V128 uses a fixed 1 m3 SWMM action-effect floor rather than enlarging the training deadband with event TFV.

### P1.2 Actuator aliasing

`src/rtc/step2_differentiable_v128.py` uses typed, direction-aware actuator-to-node messages containing upstream/downstream hydraulic state, target setting, previous/predicted managed flow, responsiveness, actuator physics/type features and actuator identity. Managed-flow injection remains a separate physical pathway.

### P1.3 Engineering envelope and score==execute

`src/rtc/engineering_v128.py` defines ordered/hashable per-actuator min/max/max-delta semantics. `src/rtc/step3_mpc_v128.py` applies them inside the differentiable decoder before scoring. Post-score projection is forbidden. The historical 0.5 default is explicitly idealized, not a field-device claim.

### P1.4 Tracking lag versus command slew

The hard supervisory rate anchor is the previous issued `target_setting`; realised `current_setting` is hydraulic state/tracking diagnostic. RBC warm-start/fallback and current MPC use the same command semantics.

## P2 — exact ranking gradients and workstation efficiency

### P2.1 Full within-state first-order pairwise gradient under 8-GB VRAM

`src/rtc/step2_train_v128_exact.py` uses two passes at one parameter snapshot:

1. no-grad H360 pass caches every candidate smooth-TFV delta;
2. gradient H360 pass recomputes candidates in small GPU microbatches;
3. every live candidate is compared with all cached candidates;
4. every informative unordered pair is visited from both endpoints;
5. directed terms are normalized by the original unordered-pair denominator.

A pure autograd regression test compares this constructed gradient elementwise with the complete unordered pair-loss gradient. Only one H360 autograd microbatch is resident at a time.

### P2.2 Repeated immutable tensors inside L-BFGS-B

`src/rtc/step3_runtime_v128.py` caches topology, actuator endpoint/physics and static-node tensors by device/dtype without changing scoring equations.

### P2.3 RTX 4060 / 16-GB execution profile

Current execution keeps CPU-group/GPU-microbatch training, AMP off and activation checkpointing off. FP32 matmul `high` is the main workstation profile and `highest` is the frozen numerical/runtime sensitivity comparison. SWMM generation should remain at <=16 workers and one SWMM thread/process.

### P2.4 Real-time acceptance and finite diagnostics

`src/rtc/runtime_controller_guard.py` measures the complete wrapped supervisory callback. `src/rtc/runtime_evidence_v128.py` requires exact 600-s decision spacing, every guarded callback below 600 s, explicit score==execute and continuity evidence. Same-epoch SWMM target write/readback remains a separate authoritative execution audit.

If no valid gradient was evaluated before a deadline/fallback, current V128 records `gradient_evaluated=false` rather than serializing NaN as if it were scientific evidence.

## P3 — current-surface and evidence hygiene

- `CODEX_START_HERE.md` is the only current root guide.
- `configs/step2_current_contract.json` defines the research/method surface.
- `configs/project7_execution_registry.json` defines one current routing surface.
- `configs/v128_control_execution.json` binds the selected exact implementation and workstation contract.
- `rtc-current-preflight` is the stable user preflight alias.
- ranking, D2 and D5 evidence must reference one identical final Step2 SHA256.
- obsolete root Formal guides were removed; Formal implementation/configs remain only as historical/frozen provenance and are not current routes.

## Deliberate non-changes

1. Default rainfall remains one causal persistence/decay scenario; the default method is not robust/stochastic MPC.
2. Existing D5 evidence applies only to its frozen idealized decoder/envelope space.
3. Code correctness/CI cannot prove hydraulic control benefit.
4. Validation, Final, Formal and Policy Lock remain untouched during current development.
5. Historical configs/internal modules are not mass-deleted when they still carry frozen data lineage, reproducibility or shared audited orchestration.

## Required empirical promotion evidence

Before any Policy-Lock/Final claim, the exact final checkpoint must provide:

- InternalHoldout D2/D3 rank, pairwise, top1 and selected regret;
- D2 and untouched D5-AUDIT TFV gradient sign/cosine/MAE;
- H30-H360 hydraulic/managed-flow and TFV-delta error growth;
- one preselected authoritative development closed loop with every guarded decision <600 s;
- target write/readback PASS, continuity PASS and score==execute PASS;
- authoritative seven-strategy TFV/Priority8 PFV/report-only Global Peak comparison;
- RTX 4060 `high` versus `highest` numerical/runtime sensitivity.
