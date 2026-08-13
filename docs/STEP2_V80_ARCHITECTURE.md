# Project7 Step2 V8.0 — Direct Hydraulic-Effect Surrogate

## Decision carried from V7

V7 solved the primary Value-model failure: direct signed authoritative Delta-TFV removed the historical near-zero physical response collapse. The V7 Value checkpoint is therefore preserved unchanged in V8.

The remaining V7 failure is isolated to Hydraulic candidate-minus-reference response. During 10 hydraulic epochs, absolute-state/flow losses fell substantially while the Delta-state and Delta-flow losses were almost flat. The V7 Hydraulic class also remained an empty subclass of the V6 absolute trajectory architecture: candidate effects were obtained by subtracting two independently decoded large absolute trajectories. This is not retained as the V8 effect path.

## V8 contract

V8 changes only Hydraulic counterfactual learning.

### Frozen reference path

The trained V7 Hydraulic checkpoint is loaded as an immutable reference trajectory model. It runs once per group with the reference action sequence. Its parameters never receive V8 gradients.

### Direct candidate-effect path

For every candidate and retained hydraulic time:

1. construct the causal candidate-reference control prefix;
2. condition each actuator token on the frozen reference hydraulic context at its upstream/downstream nodes, actuator physics, previous actuator flow, reference setting and retained time;
3. construct the action token as `encoder(base + action_delta) - encoder(base + zero)`, giving structural zero action effect;
4. scatter actuator effects to physical endpoints;
5. propagate the effect over the frozen sewer graph with four zero-preserving message-passing blocks;
6. decode signed node Delta-depth, Delta-flooding, Delta-storage, Delta-inflow and Delta-outflow directly;
7. decode signed managed-actuator Delta-flow directly;
8. form candidate physical states as `frozen_reference + direct_effect`, with physical non-negativity and exact head-depth consistency.

The primary Delta prediction is therefore **not** obtained by subtracting candidate and reference absolute neural predictions.

## Causality

The effect at retained time `j` can consume only candidate actions at or before that time. Future action blocks are explicitly masked by `CausalPrefixActionProjectorV80`. No future SWMM target state, target flow, exact flooding volume or Delta-TFV enters the forward inputs.

Available causal inputs are reused from the frozen cache: current six-channel node state, causal rainfall boundary, reference/candidate settings, previous actuator flow, static topology, actuator physics and actuator endpoints. Recent hydraulic history and all-link current flow are not invented because they are absent from the frozen V6/V7 cache contract.

## Sparse-effect objective

A dense all-node mean is retained only as a regularizer. The primary effect objective combines:

- active-effect Smooth-L1 on cells whose authoritative effect exceeds 0.25 of the TrainFit-only channel RMS scale;
- group response-magnitude calibration;
- active-effect sign consistency;
- a smaller dense regularizer;
- corresponding direct managed-flow terms.

Flooding and storage receive higher channel importance than inflow/outflow. The loss does not include absolute reference-state error because the reference model is frozen and V8 must not spend effect capacity correcting common reference bias.

## Staged training

- Stage A: 3 epochs on D2 single-actuator branches to learn local spatial response and action-to-hydraulic sensitivity.
- Stage B: 8 epochs on targeted D3 multi-actuator groups with source balance D3 0.75 / D2 anchor 0.25.

No V7 Value retraining, new SWMM, active learning or hyperparameter search is part of V8 development.

## Onset

Dry-to-flood transition remains the target. V7 used a capped inverse-prevalence positive weight of 50 and evaluated at logit 0; that yielded high balanced accuracy but very low precision. V8 instead:

- derives a milder square-root imbalance weight from TrainFit only, capped at 20;
- calibrates the operating logit threshold on TrainFit D3 only by F1;
- freezes that threshold before InternalHoldout evaluation;
- reports F1, prevalence and precision lift in addition to recall/specificity/balanced accuracy.

Onset remains diagnostic and does not override the primary direct hydraulic-effect acceptance.

## Evaluation

For Delta-depth, Delta-flood, Delta-storage and Delta-managed-flow V8 reports:

- RMSE;
- response ratio;
- skill versus the zero-effect predictor;
- active-effect skill versus zero;
- active-effect sign accuracy;
- active-effect fraction.

Candidate absolute trajectory RMSE is retained as a secondary diagnostic.

A development architecture-lock candidate requires the successful V7 Value mechanism plus positive Holdout-D3 skill versus zero for all primary hydraulic-effect families and no catastrophic (<0.10) Delta-depth/flood/storage response ratio. This is not Formal or production approval.

## Boundaries

V8 is Train-only development. It must not run SWMM, regenerate D2/D3, access Validation or Final, run Formal, alter production Step2 wiring, or merge the draft PR before external scientific review.
