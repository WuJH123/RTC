# Project7 V127 restoration audit

## Frozen scientific question

Project7 is an idealized SWMM methodology testbed. Every 10 min, sparse causal sensing is reconstructed into a control-sufficient current hydraulic state. A differentiable action-conditioned hydraulic surrogate predicts future system response under continuous pump/orifice/weir target sequences. A bounded continuous receding-horizon optimizer minimizes system-wide cumulative TFV over H360 with H120 free control, executes only the first 10-min target, and repeats. SWMM is authoritative truth.

**RBC provides safety; differentiable MPC provides optimization.** Sparse-RBC is a warm start, safety fallback and engineering comparator. It is not the Step2 target/reference and is not an action-space ceiling.

Priority8 PFV is one-sided soft secondary protection. Global Peak is report-only.

## Why V126 is superseded scientifically

V125/V126 solved several correctness and diagnosis problems but converted the main control problem into local policy improvement around Sparse-RBC. The completed V125 evidence showed InternalHoldout D3 TFV rank about 0.413, D4-AUDIT rank about 0.479, a 6018.6 m3 false-benefit margin, and zero learned overrides in 60 T5 decisions. Those results demonstrate that the scalar anchor-relative Value was not sufficiently identifiable for the intended controller; they do not demonstrate that continuous model-based MPC is scientifically invalid.

The earlier V122 objective already stated the correct end goal: all fixed baselines are external comparators and must never define Proposed's online action ceiling. V127 restores that objective while retaining the later correctness lessons.

## V126/V125 items retained

- authoritative SWMM lineage, branch identity, same-prefix and endpoint preflight;
- D2 source/train/development-validation separation;
- targeted-D3 lineage and rainfall-group holdout;
- D4 FIT/AUDIT physical separation and outcome-blind split;
- branch/group/unique-state census;
- causal rainfall forecast; no future realized rainfall online;
- V125 evidence-builder reference-index correctness repair;
- exact target identity protection against float32 projection noise;
- target-latch write/readback as command authority;
- current-setting tracking diagnostics;
- 300 s observation/model stride and 600 s control decisions;
- all 109 writable actuators eligible;
- continuity <=0.5 target change per decision;
- score==execute and no post-score engineering projection;
- independent/OOF calibration principle;
- No-control, Internal RTC, Auto-RBC, EFD, All-open and All-closed remain external authoritative baselines;
- Validation/Final/Formal/Policy Lock remain untouched during development.

## Anchor-centric items revoked from the current method

- Sparse-RBC as Step2 Value reference;
- candidate-minus-anchor scalar Value as final Step2 scientific target;
- Sparse-RBC as the default Proposed action when the learned model is healthy;
- local anchor-neighbourhood finite candidate generation as the final optimization space;
- D4 anchor-advantage data as the primary training distribution;
- anchor-policy-continuation D5 as the preferred new-data definition;
- continuous-gradient search being permanently disabled by method definition.

Historical code remains for reproducibility; `rtc.step2_current` and the current contract must not route new runs through it.

## V127 data roles

- **D2**: single-actuator hydraulic sensitivity / local Jacobian support.
- **targeted D3**: coordinated multi-actuator nonlinear response and ranking/generalization.
- **D4 FIT**: local physical-response support. Its old anchor label is not the V127 objective.
- **D4 AUDIT**: untouched local response/ranking diagnostic.
- **D5 FIT**: symmetric central-difference directional-gradient supervision.
- **D5 AUDIT**: untouched continuous-gradient audit.

## V127 D5 value

D5 is designed directly for differentiable MPC rather than branch-count expansion. Forty-eight outcome-blind TrainFit checkpoints are balanced across rainfall groups and hydraulic severity. Each checkpoint has three action centres (HOLD, Sparse-RBC warm start, broad continuous manifold), four normalized spatiotemporal directions per centre, and an exact +/- epsilon SWMM pair. The design is 1296 branches and 576 central finite-difference directions. If the engineering decoder destroys symmetry, epsilon is reduced or the direction is rejected before SWMM.

## Continuous optimizer

The online free vector is 12 x 109 = 1308 continuous target fractions. A differentiable sequential transform enforces each physical actuator bound and <=0.5 change from the active target every 10 min. The last free H120 target is held through H360. PyTorch autograd supplies derivatives of the smooth flood-volume proxy; SciPy L-BFGS-B performs the bounded optimization. Hard physical predicted TFV remains available for reporting/checking. RBC is a warm start and fallback only.

## Fail-closed evidence

Continuous MPC may execute only after the frozen development gate passes:

- InternalHoldout D3 rank >= 0.70;
- InternalHoldout D3 top1 >= 0.50;
- InternalHoldout D2 TFV gradient sign accuracy >= 0.70;
- InternalHoldout D2 TFV gradient cosine >= 0.60;
- D5-AUDIT TFV gradient sign accuracy >= 0.70;
- D5-AUDIT TFV gradient cosine >= 0.60;
- causal Step1 state and causal rainfall verified.

Failure means the continuous controller remains blocked; thresholds are not lowered to obtain a positive T5 result.
