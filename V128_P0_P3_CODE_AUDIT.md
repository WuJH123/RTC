# Project7 V128 P0-P3 code audit and remediation ledger

Status: **development candidate only**. V127 remains the production identity until V128
passes the same-checkpoint evidence and authoritative SWMM development comparison.

Scientific target is unchanged: sparse causal sensing -> Step1 current full-network state ->
action-conditioned differentiable hydraulic surrogate -> 12x109 H120 continuous MPC inside
H360 prediction -> execute first 600 s only -> authoritative SWMM re-observation. TFV is
primary; Priority8 PFV is one-sided soft secondary; Global Peak is report-only.

## P0 — wrong execution/training contracts

### P0.1 Stale Step2 trainer could be called canonical

**Problem.** `scripts/run_step2_v127.py` and the memory-safe/current control-oriented
`run_step2_v127_control_streaming.py` had different training semantics while current docs
and configs could point to the old entrypoint.

**Fix.** Current V127 registry/guide now routes base training to
`run_step2_v127_control_streaming.py`; `run_step2_v127.py` is explicitly historical.
Regression coverage: `tests/test_current_step2_routing.py`.

### P0.2 V128 typed model was initially incompatible with inherited Stage-A teacher forcing

**Problem.** V127 Stage A directly called the legacy two-channel `_setting_context()` rather
than `model.rollout()`. Replacing only the model builder would therefore bypass the typed
V128 action path during teacher forcing.

**Fix.** `src/rtc/step2_train_v128_hydraulic.py` versions Stage A and uses the same typed
endpoint-state/setting/flow/physics/identity message as online rollout. The V128 runner
explicitly replaces both Stage A and H360 objective training.

### P0.3 V127/V128 checkpoint and evidence identities could otherwise be mixed

**Fix.** `src/rtc/checkpoint_v128.py` provides a distinct source-strict V128 checkpoint.
V127 and V128 loaders reject each other's checkpoints. Model-behavior source SHA is checked
at load time; package-local training-source SHA is recorded. Ranking, D2 and D5 evidence
must describe the identical final Step2 file SHA before V128 continuous evidence can be
compiled.

### P0.4 V128 runtime called a nonexistent `ControllerConfig.validate()`

**Problem.** `ControllerConfig` is a dataclass and has no `validate()` method. This would
have failed immediately before authoritative runtime.

**Fix.** `scripts/run_policy_v128.py` now performs explicit H72/2x300-s/legacy-slew-off and
runtime-budget checks rather than invoking a nonexistent API.

## P1 — control-identification and engineering semantics

### P1.1 Large events suppressed useful action-order labels

**Problem.** V127 used `max(1 m3, 0.001 * reference TFV)` for explicit sign/ranking terms.
Large floods could discard operationally relevant candidate differences.

**Fix.** V128 uses a frozen 1 m3 absolute action-effect floor; no event-size proportional
ranking deadband.

### P1.2 Direct hydraulic action context aliased physically different devices

**Problem.** V127 direct context reduced all incident actuator targets to outgoing/incoming
raw setting sums. Different pump/orifice/weir combinations could therefore have identical
direct context even when their physical effects differ.

**Fix.** `src/rtc/step2_differentiable_v128.py` uses typed/physics-aware actuator-to-node
messages containing endpoint hydraulic state, setting, previous and predicted managed flow,
responsiveness, actuator physics/type features and actuator identity. Physical flow injection
remains a separate conservation-oriented pathway.

### P1.3 Uniform 0.5 slew was implicit for heterogeneous devices

**Fix.** `src/rtc/engineering_v128.py` defines an ordered, hashable per-actuator envelope:
min setting, max setting and max target change per 10 min. The historical 0.5 envelope is
still available, but explicitly labeled idealized and cannot be presented as field-device
truth. A file cannot spoof the idealized source label while changing bounds/rates.

### P1.4 Engineering projection had to preserve score==execute

**Fix.** `src/rtc/step3_mpc_v128.py` applies bounds/rates in the differentiable fraction-to-
physical-target decoder before surrogate scoring. Post-score projection is forbidden.
`src/rtc/controller_v128.py` revalidates every returned command, including fallback actions.

### P1.5 Sparse-RBC warm start/fallback used the wrong rate anchor under tracking lag

**Fix.** V128 may use physical current setting as hydraulic feedback, but its supervisory
target is projected against the active target latch plus/minus the same per-actuator envelope
before scoring. Runtime continuity also uses the previous supervisory target as the hard
command-rate anchor.

## P2 — training coverage and workstation efficiency

### P2.1 Candidate-candidate ranking was limited by GPU microbatch partition

**Problem.** With 24 candidates and a two-candidate H360 microbatch, most within-state
candidate pairs did not directly enter the pairwise objective.

**Fix.** `src/rtc/step2_train_v128.py` uses detached same-parameter-snapshot cross-microbatch
prediction memory. Every informative candidate pair is covered without retaining prior H360
autograd graphs. Pair loss is accumulated as a sum and divided by the exact group-level
informative-pair count, so the ranking objective is invariant to microbatch partition.

### P2.2 Repeated immutable CPU->GPU graph tensor construction inside L-BFGS-B

**Fix.** `src/rtc/step3_runtime_v128.py` caches topology, endpoint indices, actuator physics
and static-node tensors per device/dtype. The scoring equations are unchanged.

### P2.3 RTX 4060 / 16-GB workstation execution

V128 reuses CPU-group/GPU-microbatch training. Default scientific execution keeps AMP and
activation checkpointing off. `torch.set_float32_matmul_precision("high")` is available as
an auditable performance profile; `highest` must be run as a frozen numerical-sensitivity
comparison. Existing SWMM generation remains at no more than 16 workers, one SWMM thread
per process, subject to RAM/IO telemetry.

### P2.4 Real-time status previously relied on configured deadlines rather than measured run evidence

**Fix.** `src/rtc/runtime_controller_guard.py` measures the complete wrapped supervisory
callback (inner decision plus continuity checks). `src/rtc/runtime_evidence_v128.py` requires
all guarded decision runtimes, exact 600-s decision spacing, explicit score==execute and
continuity evidence. Any guarded callback >=600 s fails measured real-time acceptance.
Same-epoch target write/readback remains a separate authoritative execution audit.

## P3 — version drift and evidence hygiene

- `configs/project7_execution_registry.json` separates current V127 production from V128
  development candidate.
- `configs/v128_control_execution.json` freezes V128 time, architecture, checkpoint,
  engineering, hardware, evidence and runtime contracts.
- `src/rtc/v128_preflight.py` / `rtc-v128-preflight` fail closed before expensive runtime on
  wrong actuator count, unavailable CUDA, wrong checkpoint/evidence SHA or unsupported
  engineering-envelope evidence.
- `CODEX_START_HERE_V128.md` is the single V128 execution order.
- V128 ranking/horizon, D2, D5, continuous-evidence, runtime and seven-strategy scripts use
  V128-specific checkpoint/contract identities rather than silently relabeling V127 assets.

## Deliberate non-changes / claims that remain forbidden

1. Default online rainfall remains one causal persistence/decay scenario. CVaR machinery is
   present, but the default V128 run is **not** claimed robust/stochastic MPC.
2. A custom per-actuator engineering envelope changes decoder-space gradients. Existing D5
   supports only the historical idealized 0.5 envelope; custom envelopes require newly frozen
   matching D5 evidence.
3. Code correctness and CI do not prove hydraulic/control benefit. V128 must still be trained
   on the user's frozen local assets and evaluated on the exact final checkpoint.
4. No Validation, Final, Formal or Policy Lock access is allowed while V128 is development-only.

## Required empirical promotion evidence

Promotion requires the exact final V128 checkpoint to provide, without model selection on
holdouts:

- InternalHoldout D2/D3 rank, pairwise, top1 and selected regret;
- D2 and untouched D5-AUDIT TFV gradient sign/cosine/MAE;
- H30-H360 hydraulic/managed-flow and TFV-delta error growth;
- one fixed authoritative development closed loop with every guarded decision <600 s,
  target write/readback PASS, continuity PASS and score==execute PASS;
- authoritative seven-strategy TFV/Priority8 PFV/report-only Global Peak comparison;
- RTX 4060 `high` versus `highest` numerical/runtime sensitivity.

Only after those results are reviewed should V128 replace the V127 production identity.
