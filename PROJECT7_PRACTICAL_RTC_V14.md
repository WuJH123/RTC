# Project7 Practical RTC — paper-aligned scientific contract

## Research question

Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable learning
controller reduce **system-wide total flooding volume (TFV)** in a large SWMM drainage network while
avoiding material deterioration of flooding at the frozen Priority8 nodes?

The paper method is not required to beat every comparator on every storm. No-control is the primary
reference. Internal SWMM rules, Auto-RBC and EFD are operational comparators. Numerical all-maximum
and all-minimum SETTING policies are diagnostic extremes and are reported for interpretation only.

## Online information and objective

Every 10 minutes the controller may use only:

- the causal 13-frame sparse hydraulic history available up to the current decision;
- actuator target/current-setting and managed-flow readback available up to that time;
- causal rainfall history and frozen persistence/decay rainfall scenarios;
- frozen network topology, static node/actuator physics and training-support artifacts.

It must not use future realised rainfall, future SWMM state/flooding, future Internal-rule trajectory,
untouched evaluation truth, or an online SWMM candidate search.

The **sole online optimization objective is system-wide cumulative TFV** under the receding-policy
first-action estimand:

`A^pi(x_t,u_t) = J(candidate H10 -> frozen continuation pi) - J(HOLD H10 -> same frozen continuation pi)`.

Negative return is beneficial.

## Exact action encoding

The paired authoritative label changes the supervisory command for H10 only before handing both
branches to the same continuation policy. The learned policy-return critic must therefore see the
same intervention. Its candidate sequence is encoded as:

`H10 candidate target -> H350 current HOLD target`.

The reference is HOLD through H360. Repeating the candidate target through H360 is forbidden for the
policy-return critic because it changes the estimand back into an open-loop/target-latched value.
Training, calibration and online scoring share one implementation of this action-token encoding.

## Secondary Priority8 PFV safety

PFV means **Priority8 Flooding Volume**, derived from authoritative SWMM node flooding volumes. It is
not a second online objective and no online PFV surrogate is required. Unless a later preregistered
contract changes the tolerance, an event is eligible for a performance claim only when:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

The 100 m3 / 5% envelope is a project-specific engineering non-inferiority tolerance rather than a
universal regulatory threshold. Global Peak remains report-only.

## Step1, base Step2 and policy-return roles

- **Step1** reconstructs the causal full-network hydraulic state from sparse observations.
- **Base Step2 V5** is a frozen pretrained TFV action-effect representation. It is reused for cheap
  batched first-action directional probes rather than retrained or trusted as a deployed closed-loop
  value oracle.
- **Policy-return critic** is initialized from the base Step2 weights and fine-tuned on paired exact-
  prefix SWMM labels for the actual receding continuation. With limited labels, global state/rainfall
  encoders stay frozen by default and only facility/action/interaction layers are adapted.

Acceptance emphasizes control utility: false-beneficial selections, within-same-prefix candidate
ranking, selected-action regret, sign accuracy and event-balanced metrics. Scalar MAE alone cannot
promote a critic.

## Practical online candidate portfolio

The deployed Practical controller does **not** solve the historical 12 x 109 = 1308-dimensional
L-BFGS-B problem online. That optimizer produced useful historical diagnostics, but Development replay
showed surrogate-extremum exploitation and support/extrapolation sensitivity, while local H360 value
correctness still failed to guarantee event-level receding benefit.

At each decision the current controller instead:

1. batch-scores positive/negative, support-bounded **H10 -> H350 HOLD** probes for all 109 facilities
   with frozen base Step2 under causal rainfall scenarios;
2. keeps the best direction for each facility and combines only individually predicted-beneficial
   directions, capped by the TrainFit q95 changed-facility support;
3. forms at most three engineering-supported first targets after projection/deduplication:
   - Step2 H10-probe direction at 100% magnitude;
   - Step2 H10-probe direction at 50% magnitude;
   - one type-aware hydraulic-pressure direction using storage/depth headroom, downstream congestion
     and causal rainfall;
4. contracts each candidate to q95 joint-sequence support using the **actual H10 pulse** geometry;
5. ranks the candidates with the calibrated policy-return critic and executes the candidate with the
   smallest one-sided upper bound only when that bound is negative; otherwise it executes HOLD;
6. writes only the first H10 target to authoritative SWMM and re-observes after 10 minutes.

This is a finite, support-aware policy-improvement search, not a claim of continuous 109-dimensional
global optimality.

## Role of historical V12 / L-BFGS-B

The frozen V12 target-latch controller remains useful only as an **offline parent continuation pi0**
for the first policy-return data round and for historical ablation. It may still contain the old
L-BFGS-B path because the counterfactual estimand requires the same frozen continuation in candidate
and HOLD branches. It is not the final Practical online action generator.

If the learned pi1 materially differs from pi0, the next data round must use frozen pi1 as the shared
continuation, yielding Q^pi1 rather than pretending Q^pi0 remains the deployed value function.

## Training-support and engineering boundaries

- 109 writable facilities remain screened; no hand-picked small control subset is introduced.
- physical setting bounds are respected;
- supervisory target slew is at most 0.5 per 10-minute update;
- first-move changed-facility density stays inside frozen TrainFit q95 support;
- the actual H10 counterfactual sequence stays inside frozen q95 joint-sequence support;
- HOLD means latch the previous supervisory target;
- unchanged facilities preserve their last commanded targets;
- score equals execute: no material post-score projection is allowed;
- target write/readback and causal-history readiness remain fail-closed.

## Actuator semantics

A normalized hydraulic **release intent** is converted into the correct SWMM SETTING coordinate for
each actuator type. Pump, orifice, weir and outlet SETTING values are not assumed to share a universal
"larger setting = more release" meaning. The same type-aware conversion is used by the Proposed
hydraulic candidate and rule-based comparators.

## Baseline roles

The six-strategy evidence panel remains for transparent reporting:

- No-control — primary reference: no supervisory RTC, native `[CONTROLS]` disabled, intrinsic/local
  device physics preserved;
- Internal RTC — operational comparator using frozen native SWMM rules;
- Auto-RBC — causal automatically parameterized rule-based comparator;
- EFD — causal storage equal-filling comparator;
- All-max-setting — diagnostic numerical extreme, legacy ID `all_open`;
- All-min-setting — diagnostic numerical extreme, legacy ID `all_closed`.

The Proposed controller is not required to beat diagnostic extremes or every operational comparator
on every event. Any superiority claim must come from untouched authoritative evaluation.

## Data roles and policy iteration

Rainfall groups, not individual decision rows, are the independent split unit. Before bulk label
generation, role-disjoint groups are frozen for at least:

- policy-return train: 48 groups;
- model-selection validation: 12 groups;
- conformal calibration: 24 groups;
- separate new Development probes.

Candidate ranking is learned only among actions sharing one exact authoritative query prefix. Within
a query set the HOLD branch is scientifically identical for every candidate and may be computed once
and reused when prefix/forcing/continuation identity is proven, reducing authoritative SWMM cost.

## Development success before Policy Lock

The next Development stage succeeds only when:

1. zero future-information, engineering, readback, support and score/execute violations occur;
2. paired labels verify the same raw causal authoritative prefix and same frozen continuation policy;
3. the critic has acceptable false-beneficial, same-query ranking and selected-regret behavior on
   role-disjoint validation;
4. independent Development closed loops show useful event-balanced TFV behavior relative to
   No-control, with Internal RTC, Auto-RBC and EFD reported fairly;
5. every event used for a positive method-performance claim satisfies the frozen Priority8 PFV
   envelope;
6. the 10-minute supervisory runtime budget is satisfied with substantial margin;
7. if pi1 materially differs from pi0, a new role-disjoint Q^pi1 round is completed before Policy Lock.

Universal superiority is not a promotion requirement. Validation, Final, Formal and Policy Lock
remain inaccessible during policy development.

`READY_FOR_POLICY_LOCK=false` until these Development conditions are met.
