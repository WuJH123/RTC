# Codex start here — Project7 v0.6.9 current execution contract

This file is the single active entrypoint.  Do **not** infer current Step2
semantics from the highest historical version number in the repository.

## 1. Study identity

- Project: Project7 urban-drainage real-time control methodology testbed.
- Frozen INP: `wuhan_method_testbed_v067.inp`.
- Authoritative truth: EPA SWMM.
- Study positioning: idealized SWMM methodology testbed, not a field digital twin.
- Final control objective: system-wide cumulative sewer-node overflow volume
  (`TFV`) minimization.
- PFV at 8 priority nodes and Global Peak are diagnostics, not current hard
  objectives.
- No explicit surface ponding/2-D routing is represented; TFV must be described
  as **sewer-node overflow volume**, not street inundation volume.

## 2. Frozen physical/time contract

- 109 controlled links = 57 pumps + 42 orifices + 10 weirs.
- ~932 nodes, 1167 conduits, 10 storages, 89 sensors.
- SWMM/model sample step = 300 s.
- MPC update = 600 s.
- Long value horizon = 360 min / 72 model steps / 36 control blocks.
- Maximum setting change = 0.5 per 10-min update.
- Effective warm-up = 120 min; tail = 360 min.
- Chicago events = 5 return periods × 6 durations = 30.
- Frozen forcing split = 18 Train / 6 Validation / 6 Final.
- Validation and Final are not tuning data.

## 3. Current canonical Step2 contract

Read:

1. `configs/step2_current_contract.json`
2. `docs/STEP2_V110_ARCHITECTURE.md`
3. `src/rtc/step2_current.py`

New development/production-facing code must import Step2 through
`rtc.step2_current`.

### 3.1 Long-horizon Control Value — V7, supported and frozen

`rtc.step2_control_response_v70.ControlValueSurrogateV70`

Input:
- current causal state,
- causal rainfall boundary,
- reference action,
- candidate action,
- previous controlled-link flow,
- actuator physics/identity.

Output:
- direct signed authoritative `Delta-TFV` in m3.

Meaning:
`Delta-TFV = TFV_candidate - TFV_reference`.

V7 does not route the objective through a tiny flooding-rate head and H72
integration.  It is the current long-horizon 0–360 min MPC value model.

Do not redesign or retrain V7 merely because the Hydraulic Effect model is
under development.

### 3.2 Short/medium-horizon Hydraulic Effect — V11, development

`rtc.step2_control_response_v110.ActuatorSetHydraulicResponseV110`

Hydraulic horizon is deliberately **0–120 min**, not six-hour centimetre-level
nodewise prediction.  The 360-min anti-myopia objective remains V7 Delta-TFV.

Authoritative target at response time tau remains the same-prefix
counterfactual:

`Delta x(t+tau) = x_candidate(t+tau) - x_reference(t+tau)`.

This target is correct for delayed effects.  Lag is handled by ensuring that
the prediction at tau can only see candidate/reference actions that have
already occurred by tau.

V11 predicts signed changes in:
- node depth/head,
- node flooding rate,
- node volume/storage,
- node total inflow/outflow,
- 109 managed-link flows.

It does not require flooding to occur before a hydraulic effect exists.

## 4. V11 architecture requirements

The design handles four coupled RTC properties.

### Lag
For each retained response time, use only the causal action prefix.  A setting
change scheduled after that response time must have exactly zero influence on
that earlier output.

### Nonlocality
A changed pump/orifice/weir may affect remote upstream/downstream nodes.
Hydraulic influence is not restricted to a fixed 1/2/4/8-hop receptive field.
Every node/time query may attend to the changed-actuator set with static
all-range graph-relation bias.

### Multi-actuator combination
The changed facilities form a variable-size set.  Use actuator self-attention
before node/time cross-attention.  D3 is a joint nonlinear response problem.

Forbidden:
`SUM(predicted D2 effects) + interaction residual`.

D2 is mechanism supervision/anchor, never a formula for D3 truth.

### Rolling MPC
Every 10 min the real system is observed/reconstructed and the MPC problem is
solved again.  V11 therefore focuses detailed hydraulics on 0–120 min while V7
keeps the 0–360 min value objective.  Only the first 10-min control block is
executed before re-estimation/re-optimization.

## 5. Hydraulic Effect learning target

Each node/variable/time response is decomposed into:

1. `active`: is the candidate-reference effect locally meaningful?
2. `sign`: if active, is it positive or negative?
3. `magnitude`: if active, how large is it?

The raw signed counterfactual delta is primary.  Physical projection of an
absolute candidate trajectory must never clip or rewrite the raw signed delta.

### Active-effect definition

Do not use one global channel median/scale.

V11 freezes TrainFit-D2-only local thresholds using:
`max(0.25 * P90(abs(Delta)), physical floor)`.

Physical floors:
- depth: >= 0.01 m and >= 1% of local maximum node depth,
- flooding: >= 1e-5 m3/s,
- volume: >= 1e-3 m3 and, for storages, >= 0.5% capacity,
- node inflow/outflow: >= 1e-4 m3/s,
- managed flow: >= 1e-4 m3/s.

Holdout, Validation and Final never select these thresholds.

## 6. Time-domain supervision

Retained V11 responses:
5, 10, 15, 20, 25, 30 min, then every 10 min through 120 min.

The model is direct response, not recurrent free-run.  In addition to
active/sign/magnitude supervision, compare authoritative finite differences
between adjacent retained response times.  This teaches response rise/decay and
lag without recursively feeding predicted state into the next state.

## 7. Existing Step2 data

Use the existing lineage-valid V6 counterfactual-group-preserving cache:
- canonical D2 groups,
- targeted D3-v2 groups,
- same-prefix reference/candidate branches,
- no legacy dense D3 training.

Do not regenerate SWMM because the surrogate architecture changed.

D2 has already shown learnable single-actuator control signal.  The historic
finite-hop full-network Hydraulic failures do not imply that D2 is unlearnable.

## 8. V11 development order

Canonical runner:

`scripts/run_step2_v110.py`

### Stage D2
- TrainFit D2 only.
- 4 epochs.
- seed 42.
- FP32 AdamW.
- no sweep.
- evaluate independent TrainInternalHoldout D2.

D3 remains blocked unless holdout skill-vs-zero is > 0 for:
- depth,
- flooding,
- volume,
- managed flow.

### Stage D3
Only after an accepted V11 D2 report/checkpoint:
- targeted TrainFit D3,
- 10 epochs,
- D3/D2 authoritative supervision = 0.75/0.25,
- same model and seed contract,
- no hyperparameter sweep.

D2 outputs are never summed to synthesize D3 labels.

## 9. Development boundaries

Until V11 D2 and D3 development gates pass:

- no Validation tuning,
- no Final access,
- no Formal,
- no Policy Lock,
- no production wiring,
- no new SWMM,
- no new all-link-flow data,
- no active learning,
- no V7 Value redesign.

Future SWMM truth is training/evaluation label only and is forbidden online.

## 10. Historical Step2 modules

V4–V10 Hydraulic branches/files are forensic provenance, not active
implementations.  They document failed or superseded hypotheses:
- additive D2 superposition,
- local finite-hop propagation,
- rate-integration value collapse,
- post-projection signed-effect corruption,
- state-sufficiency and history diagnostics,
- V10 nonlocal prototype before the final dual-timescale/set formulation.

Do not import those modules into new canonical code.

The only active import surface is:
`rtc.step2_current`.

## 11. Final MPC scientific claim

The controller may claim only:

> the best control found within the generated engineering-feasible candidates /
> frozen control manifold.

Do not claim a global optimum over the full continuous 109-dimensional action
space.

All final control-effect claims remain authoritative SWMM results.
