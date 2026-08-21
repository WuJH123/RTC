# Project7 Practical RTC — paper-aligned scientific contract

## Research question

Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable learning
controller reduce **system-wide total flooding volume (TFV)** in a large SWMM drainage network while
avoiding material deterioration of flooding at the frozen Priority8 nodes?

No-control is the primary reference. Internal SWMM rules, Auto-RBC and EFD are operational comparators.
All-maximum and all-minimum SETTING policies are diagnostic extremes only. Universal superiority on
every event is not required.

## Frozen hydraulic representation versus supervisory freedom

Project7 separates the learned hydraulic representation from the online control degrees of freedom:

- frozen base Step2 action representation: **109 channels**;
- native supervisory-control facilities: **82**;
- passive setting channels: **27**;
- every passive channel satisfies `candidate == HOLD/reference`.

The 82-facility mask is derived deterministically from explicit source-INP `[CONTROLS]` actions and is
frozen by source-INP SHA, graph actuator order and mask SHA. The current testbed census is 57 pumps,
16 orifices and 9 weirs.

The five RTC Storage additions are retained as hydraulic network/state features for storage capacity
and headroom. They are not supervisory action dimensions. This control-space restriction **does not
retrain Step1 or base Step2**; it removes online action freedoms without changing the frozen network or
109-channel representation.

## Online information, timing and objective

Every 600 s the controller may use only causal information available up to the current decision:
13-frame sparse hydraulic history, target/current-setting and managed-flow readback, observed rainfall
history with frozen persistence/decay scenarios, network/static physics, the native control mask and
training-support artifacts.

The **sole online optimization objective is system-wide cumulative TFV**. Future realised rainfall,
future SWMM state/flooding, future Internal-rule trajectory, online SWMM candidate search, PFV/Peak
action penalties and baseline imitation are forbidden.

The authoritative first-action estimand is:

`A^pi(x_t,u_t) = J(candidate H10 -> frozen continuation pi) - J(HOLD H10 -> same frozen continuation pi)`.

Negative is beneficial. The critic action token is exactly:

`H10 candidate target -> H350 current HOLD target`

versus `HOLD H360`. The real controller observes on the 300-s grid and replans after every 600-s H10
command.

## Priority8 PFV and Global Peak

Priority8 PFV is a secondary **authoritative SWMM safety condition**, not an online objective or
surrogate. Unless later preregistered otherwise, a positive event claim requires:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

This is a project-specific engineering non-inferiority tolerance, not a universal regulatory limit.
Global Peak is report-only.

## Cheap masked q95 support correction

The previous q95 action geometry was measured in a 109-free Development distribution. Current q95
support is recomputed offline from existing D3 TrainFit action tensors after zeroing differences on the
27 passive channels. No new SWMM outcome simulation is required.

The current 82-control artifact freezes q95 changed-K, first-block L1, H120 L1 and H120 total variation,
plus mask/Step2/cache lineage. q95 remains canonical and may not be loosened merely to increase ACTION
frequency.

## What the completed 82-control seen mechanism panel established

Two query points each from already-inspected T8, T30 and T80 were frozen label-blind before new exact
SWMM truth. All six query sets passed same raw prefix, same continuation, causal rainfall, passive-
channel, engineering, readback and support gates.

Observed mechanism summary:

- six query sets / three events / 20 candidate rows;
- every query had at least one true beneficial candidate;
- base Step2 sign accuracy = 0.45;
- base Step2 false-beneficial fraction = 0.30;
- base Step2 false-reject fraction = 0.25;
- within-query pairwise rank accuracy = 0.4167;
- candidate top-1 accuracy = 0.50.

Candidate-family beneficial fractions were Step2 0.50 = 3/6, Step2 1.00 = 2/3 when distinct,
type-aware hydraulic pressure = 5/6, and projected gradient = 3/5. Projected gradient was oracle-best
in 0/5 queries where it existed. Type-aware hydraulic pressure repeatedly recovered large beneficial
actions that base Step2 scored as non-beneficial.

Therefore the current bottleneck is **receding-policy-return sign/rank generalization**, not absence of
useful executable candidates. Base Step2 remains valuable as a pretrained action-effect representation
and Step2-direction generator, but it is not a deployed closed-loop return oracle.

## Current three-family Practical portfolio

To reduce compute and method complexity without sacrificing the useful mechanisms observed in the
seen panel, the paper-facing online portfolio contains at most three distinct H10 candidates:

1. `STEP2_H10_PROBE_SCALE_0.50`;
2. `STEP2_H10_PROBE_SCALE_1.00`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`.

Every candidate is projected/contracted to the 82-control mask, physical bounds, <=0.5 target slew,
per-facility q95 first-move radius and masked q95 joint-sequence support. The final post-support base-
Step2 H10 score is retained only for diagnostics.

The historical `12 x 109 = 1308-dimensional L-BFGS-B` optimizer remains archival. **Projected gradient
is now Development ablation only**: its implementation is retained for reproducibility, but it cannot
enter current pi0, pi1, policy-return training, matched calibration or online execution.

This reduction is evidence-driven: in the completed seen panel a non-gradient beneficial candidate
remained in all six queries, while gradient was never oracle-best. It removes online autograd and one
candidate family rather than expanding the research problem.

## Current first policy iteration

The current parent is:

`PROJECT7_PRACTICAL_BASE_H10_THREE_FAMILY_PARENT_PI0_V4_82CONTROL_109REP`.

Pi0 uses the same three-family proposal/support geometry as the future pi1, but ranks candidates with
frozen base Step2 before a return critic exists. Exact same-prefix SWMM pairs then teach the policy-
return critic the actual first-action value under this continuation. Pi1 uses the trained critic plus
matched one-sided admission.

Removing gradient changes pi0 continuation. Therefore the completed four-family seen exact returns
remain **Development mechanism evidence only**; they are not relabelled as current three-family
training/calibration data.

## Minimum continuation-specific recheck before bulk

Because the parent continuation changed, run a small new three-family mechanism confirmation before
bulk labels. Freeze two deterministic causal-ready queries per already-seen T8/T30/T80 event before
reading any new truth:

- first wave: first causal-ready non-HOLD query from each event (3 queries total);
- reserve wave: one later deterministic non-HOLD query per event, frozen in advance.

Evaluate only the three first-wave queries initially. If all three contain at least one beneficial
current candidate and all technical gates pass, no reserve truth is needed. If the first wave is
ambiguous, evaluate the already-frozen reserve queries before considering any architecture change.

This staged gate is Development-only and cannot become train/validation/calibration evidence.

## Role-disjoint data: inventory before generation

After the three-family mechanism recheck passes, first perform a **zero-SWMM forcing inventory** over
existing event/forcing metadata. Exclude all previously inspected Development families, existing Step2
TrainFit role groups and untouched Validation/Final/Formal/PolicyLock resources.

Only then create the missing forcing definitions needed to freeze at least:

- policy-return train: 48 rainfall groups;
- model-selection validation: 12;
- conformal calibration: 24;
- separate new Development probes: 3.

Selection is forcing-only, deterministic and label-blind. The current role audit has not confirmed an
eligible untouched pool, so bulk policy-return simulation remains closed until this inventory and role
assignment are frozen.

Initially use one deterministic causal query per assigned group to minimize authoritative SWMM cost.
One shared HOLD is reused for all distinct current candidates in that query.

## Critic training and matched admission

The policy-return critic is initialized from base Step2. Default training adapts only control/action
and interaction heads; global representation layers remain frozen initially. Full-model retraining is
not justified by the current mechanism evidence.

Model selection uses the actual decision set `{HOLD=0 + generated candidates}` and prioritizes:

- selected-action false-beneficial fraction;
- selected-action false-reject fraction;
- same-query pairwise rank / candidate top-1;
- selected-action regret;
- event-balanced sign and MAE.

Matched calibration contains only candidate families that can appear online: Step2 probes and type-
aware hydraulic pressure. Projected-gradient rows are rejected by the current admission contract.
Rainfall group is the split-conformal independent unit; coverage is not manually weakened to force
more ACTION.

## Engineering and causality boundaries

- 109 hydraulic/model channels remain represented;
- exactly 82 native supervisory facilities may change online;
- passive channels equal HOLD/reference;
- physical bounds and <=0.5 target slew hold;
- masked q95 changed-K and H10 sequence support hold;
- HOLD latches the previous supervisory target;
- score equals execute with no material post-score projection;
- target write/readback and causal-history readiness fail closed;
- no future realised rainfall or online SWMM search.

Hydraulic release intent uses actuator-type-specific SWMM SETTING semantics. The mask answers which
facilities the paper controller may change; it does not remove passive links from hydraulic simulation.

## Baselines and promotion

Operational comparators remain No-control, Internal RTC, Auto-RBC and EFD. All-max/min SETTING are
diagnostic extremes. No comparator is weakened or used as a warm start.

Development succeeds only after the three-family continuation-specific mechanism gate, role-disjoint
critic training/validation/calibration, independent new Development closed loops with useful TFV
behavior, Priority8-PFV-safe positive claims, zero engineering/causal violations and adequate runtime.
If pi1 materially changes the policy distribution, collect a role-disjoint Q^pi1 round before Policy
Lock.

Do not add pump-energy objectives, PID setpoints, new level penalties, an online PFV surrogate, a fourth
or fifth online candidate family, or a new Step1/base-Step2 retraining campaign without later evidence
that specifically requires it.

Validation, Final, Formal and Policy Lock remain closed during this work.

`READY_FOR_POLICY_LOCK=false`.
