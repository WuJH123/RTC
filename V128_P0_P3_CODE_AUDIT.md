# Project7 current P0-P3 code audit and remediation ledger

Status: **current development implementation, not Policy Locked**. Merging the code to `main`
removes repository routing ambiguity; it does not constitute scientific promotion or Final evidence.

Frozen scientific target remains:

```text
causal sparse sensing -> Step1 current full-network state
-> typed differentiable hydraulic/action Step2
-> 12 x 109 H120 continuous MPC inside H360
-> execute first 600 s only -> authoritative SWMM re-observation
```

TFV is primary; Priority8 PFV is one-sided soft secondary; Global Peak is report-only.

## P0 — execution and artifact identity

### P0.1 Multiple files claimed to be the single current start guide

**Problem.** `CODEX_START_HERE_V069.md`, `CODEX_START_HERE_V127.md` and
`CODEX_START_HERE_V128.md` each described themselves as current/single entrypoints.

**Fix.** They were removed. `CODEX_START_HERE.md` is now the only root start guide. User/Codex
training, runtime and seven-strategy entrypoints are stable and unversioned:

```text
scripts/run_step2_current.py
scripts/run_policy_current.py
scripts/run_seven_strategies_current.py
```

### P0.2 Current contract still routed users to V127

**Problem.** `configs/step2_current_contract.json` and the execution registry still selected V127
paths even after V128 became the active development implementation.

**Fix.** Current routing now selects the unversioned surface. V127 files that remain are explicitly
classified as archival or internal shared orchestration, never user entrypoints.

### P0.3 Exact V128 pairwise implementation existed but canonical training still imported the old objective

**Problem.** `src/rtc/step2_train_v128_exact.py` implemented the two-pass exact first-order pairwise
algorithm, while `scripts/run_step2_v128_control_4060.py` still imported the older detached-memory
`src/rtc/step2_train_v128.py`. PR/docs could therefore claim exact gradients while a real local run
executed the older objective.

**Fix.** The canonical V128 implementation now imports only `step2_train_v128_exact`. The old
`step2_train_v128.py` module was deleted. Regression tests verify the current runner source and the
absence of the obsolete module.

### P0.4 Typed V128 model versus inherited Stage-A teacher forcing

**Problem.** Replacing only the V127 model builder would leave teacher forcing on the legacy action
context.

**Fix.** `src/rtc/step2_train_v128_hydraulic.py` owns current Stage A and uses the same typed
endpoint-state/setting/flow/physics/identity action path used by online rollout.

### P0.5 Checkpoint contract claimed training-source strictness without enforcing it

**Problem.** the previous loader compared model-source SHA but only checked that training-source SHA
looked syntactically valid. A checkpoint trained under the obsolete detached-memory objective could
therefore load under current code.

**Fix.** `PROJECT7_V128_STEP2_CHECKPOINT_V4_MODEL_AND_EXACT_TRAINING_SOURCE_STRICT` compares both
current model-source and exact-training-source fingerprints. Either mismatch fails closed and requires
retraining/re-audit. V127/older-V128 checkpoints are also rejected by contract.

## P1 — control-identification and engineering semantics

### P1.1 Large-event action-order deadband

Current V128 uses a fixed 1 m3 SWMM action-effect floor for explicit ranking/sign supervision rather
than scaling the training deadband with event TFV.

### P1.2 Physically different actuator combinations aliased by setting sums

`src/rtc/step2_differentiable_v128.py` uses typed, direction-aware actuator-to-node messages containing
upstream/downstream hydraulic state, target setting, previous/predicted managed flow, responsiveness,
actuator physics/type features and actuator identity. Managed-flow injection remains a separate physical
pathway.

### P1.3 Engineering envelope and score==execute

`src/rtc/engineering_v128.py` defines ordered/hashable per-actuator min/max/max-delta semantics.
`src/rtc/step3_mpc_v128.py` applies that envelope inside the differentiable decoder before scoring.
Post-score projection is forbidden. The historical 0.5-per-10-min default is explicitly idealized,
not a field-device claim, and a custom envelope requires matching decoder-space D5 evidence.

### P1.4 Tracking lag versus supervisory command slew

The hard command-rate anchor is the previous issued `target_setting`; realised `current_setting` is
hydraulic state/tracking diagnostic. RBC warm-start/fallback and current MPC follow the same command
semantics.

## P2 — exact ranking gradients and workstation efficiency

### P2.1 Full within-state pairwise first-order gradient under 8-GB VRAM

`src/rtc/step2_train_v128_exact.py` uses two passes at one parameter snapshot:

1. no-grad H360 pass caches every candidate smooth-TFV delta;
2. gradient H360 pass recomputes candidates in small GPU microbatches;
3. every live candidate is compared with all cached candidates;
4. every informative unordered candidate pair is visited from both endpoints;
5. directed terms are divided by the original unordered-pair denominator.

A pure autograd regression test compares this constructed gradient elementwise with the complete
unordered pair-loss gradient. Only one H360 autograd microbatch is resident at a time.

### P2.2 Repeated immutable graph tensors inside L-BFGS-B

`src/rtc/step3_runtime_v128.py` caches topology, actuator endpoint/physics and static-node tensors by
device/dtype. This is an execution optimization; scoring semantics are unchanged.

### P2.3 RTX 4060 / 16-GB execution profile

Current execution keeps CPU-group/GPU-microbatch training, AMP off and activation checkpointing off.
FP32 matmul `high` is the default workstation profile and `highest` is the frozen numerical/runtime
sensitivity comparison. SWMM generation should remain at <=16 workers and one SWMM thread/process on
the stated 16-GB workstation.

### P2.4 Real-time acceptance

`src/rtc/runtime_controller_guard.py` measures the wrapped supervisory callback and
`src/rtc/runtime_evidence_v128.py` requires exact 600-s decision spacing, every guarded callback below
600 s, explicit score==execute and continuity evidence. Same-epoch SWMM target write/readback remains a
separate authoritative execution audit.

## P3 — version and evidence hygiene

- `CODEX_START_HERE.md` is the only current human execution guide.
- `configs/step2_current_contract.json` defines the research/method surface.
- `configs/project7_execution_registry.json` defines one current routing surface.
- `configs/v128_control_execution.json` binds the selected exact V128 implementation and workstation.
- `rtc-v128-preflight` fails closed on wrong graph/actuator/CUDA/checkpoint/evidence/envelope identity.
- ranking, D2 and D5 evidence must reference one identical final Step2 SHA256.
- root/versioned historical guides were removed; Git history preserves their provenance.

## Deliberate non-changes

1. Default rainfall is one causal persistence/decay scenario; the default method is not robust/stochastic MPC.
2. Existing D5 evidence applies only to its frozen decoder/envelope space.
3. Code correctness/CI cannot prove hydraulic control benefit.
4. Validation, Final, Formal and Policy Lock remain untouched during current development.

## Required empirical promotion evidence

Before any Policy-Lock/Final claim, the exact final checkpoint must provide:

- InternalHoldout D2/D3 rank, pairwise, top1 and selected regret;
- D2 and untouched D5-AUDIT TFV gradient sign/cosine/MAE;
- H30-H360 hydraulic/managed-flow and TFV-delta error growth;
- one preselected authoritative development closed loop with every guarded decision <600 s;
- target write/readback PASS, continuity PASS and score==execute PASS;
- authoritative seven-strategy TFV/Priority8 PFV/report-only Global Peak comparison;
- RTX 4060 `high` versus `highest` numerical/runtime sensitivity.
