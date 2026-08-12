# Project7 Step2 V7.0 — direct control-objective surrogate

V7 is an isolated development branch built from the frozen V6 data/cache lineage after the V6 response-collapse forensics. It does not modify production Step2 and does not require new SWMM data.

## Why V7 exists

The frozen targeted D3-v2 data contain a learnable control signal: a simple action-only ridge diagnostic retained substantial candidate ordering and response spread. In contrast, V6 preserved non-zero joint/action and latent representations but collapsed at its final reference/candidate flooding-rate head and TFV integration path. V7 therefore removes the rate-head integration path from the MPC-facing value model.

## MPC-facing value contract

The primary value target is the authoritative signed counterfactual volume

`Delta TFV = TFV(candidate) - TFV(reference)`

in cubic metres. The model predicts this quantity directly.

Inputs are strictly causal and already present in the frozen V6 cache:

- current six-channel node hydraulic state;
- causal H72 rainfall boundary;
- H72 reference and candidate actuator settings;
- current/previous actuator flow;
- actuator upstream/downstream node identity;
- actuator physics and actuator identity.

The 72-step setting sequence is compressed with the same six frozen temporal control basis functions used by the V6 MPC manifold. Each actuator is encoded only after conditioning on its local upstream/downstream state, current actuator flow, physics, reference setting profile, future candidate-reference action profile and global rainfall/state context. Multi-actuator interaction is then formed by permutation-compatible sum/mean/max and second-order pooled features.

The reference identity is structural: `candidate == reference -> Delta TFV == 0` exactly. Physical output uses a TrainFit-only robust volume scale and a signed asinh/sinh transform. There is no flooding-rate softplus subtraction and no H72 rate integration in the primary value path.

## Value objective

Magnitude calibration is primary:

1. signed transformed magnitude loss;
2. physical standardized Delta-TFV loss;
3. pairwise Delta-TFV difference calibration;
4. pairwise sign as a small auxiliary term.

Listwise/ranking/regret losses are deliberately removed from the canonical V7 training objective. Candidate ordering is still evaluated, but it cannot be obtained by sacrificing physical response scale.

D2 is a sensitivity warm start. During joint training targeted D3-v2 is primary (0.75 event weight) and D2 is an anchor (0.25). Events, not branch rows, are the optimization unit.

## Hydraulic-response contract

The hydraulic model remains parameter-disjoint from the value model. V7 retains the already-tested V6 multi-resolution hydraulic decoder for efficiency, but changes supervision:

- absolute reference/candidate state fidelity;
- explicit candidate-minus-reference hydraulic state fidelity;
- absolute actuator-flow fidelity;
- explicit candidate-minus-reference actuator-flow fidelity;
- true dry-to-flood onset transition, rather than flood occurrence.

Onset uses the current physical flooding state at the checkpoint, TrainFit-only class prevalence, capped positive weighting and focal BCE. Event-balanced counterfactual evaluation reports Delta-depth, Delta-flooding, Delta-storage and Delta-managed-flow errors/response ratios separately from absolute trajectory metrics.

## Efficiency contract

V7 deliberately makes the MPC-facing path cheaper than V6:

- no H72 x 109 multi-head attention in Value;
- no recurrent Koopman rollout in Value;
- no reference/candidate rate-head subtraction;
- no rate-to-volume integration in Value;
- one direct differentiable score per candidate.

The existing 102-dimensional control-basis decode remains differentiable, so MPC gradients are taken directly through `Delta TFV(alpha)`.

## Canonical execution

Always run `scripts/run_step2_v70_guarded.py`, never the implementation runner directly. It reuses the already validated V6 cache lineage (existing D2 + targeted D3-v2 only) and the identical deterministic rainfall/event split.

The runner trains Value first. If TrainFit D3 still exhibits catastrophic response collapse (`spread_ratio < 1e-3`), it fails closed and skips Hydraulic training to avoid wasting compute. If Value clears that failure mode, Hydraulic training proceeds and both absolute and counterfactual-effect metrics are reported.

## Frozen boundaries

V7 development must not:

- run SWMM;
- regenerate D2 or D3;
- access Validation or Final;
- run Formal/closed-loop evaluation;
- use legacy dense D3;
- initialize from V4/V5/V6 Step2 checkpoints;
- change production Step2 wiring;
- merge before external scientific review.
