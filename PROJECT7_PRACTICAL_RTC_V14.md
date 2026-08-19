# Project7 Practical RTC — paper-aligned scientific contract

## Research question

Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable learning
controller reduce **system-wide total flooding volume (TFV)** in a large SWMM drainage network while
avoiding material deterioration of flooding at the frozen Priority8 nodes?

The paper method is not required to beat every comparator on every storm. No-control is the primary
reference. Internal SWMM rules, Auto-RBC and EFD are operational comparators. Numerical all-maximum
and all-minimum SETTING policies are diagnostic extremes only.

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
branches to the same continuation policy. The learned critic sees the identical intervention:

`H10 candidate target -> H350 current HOLD target`.

The reference is HOLD through H360. Repeating the candidate target through H360 is forbidden for the
policy-return critic because it changes the estimand back into an open-loop/target-latched value.
Training, calibration and online scoring share this H10 action-token implementation.

## Secondary Priority8 PFV safety

PFV means **Priority8 Flooding Volume**, derived from authoritative SWMM node flooding volumes. It is
not a second online objective and no online PFV surrogate is introduced. Unless a later preregistered
contract changes the tolerance, a positive method-performance claim for an event requires:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

The 100 m3 / 5% envelope is a project-specific engineering non-inferiority tolerance rather than a
universal regulatory threshold. Global Peak remains report-only.

This placement is deliberate: TFV remains the only online optimization target, while Priority8 PFV is
an authoritative spatial safety condition checked on the same untouched SWMM event. Adding PFV to the
online objective, using future Priority8 truth, or calling SWMM online to enforce it would change the
research question.

## Step1, base Step2 and policy-return roles

- **Step1** reconstructs the causal full-network hydraulic state from sparse observations.
- **Base Step2 V5** is a frozen pretrained TFV action-effect representation. It is reused for cheap
  batched first-action directional probes and for one differentiable first-action proposal. It is not
  trusted as the deployed closed-loop value oracle.
- **Policy-return critic** is initialized from base Step2 and fine-tuned on paired exact-prefix SWMM
  labels for the actual receding continuation. With limited labels, global state/rainfall encoders stay
  frozen by default and facility/action/interaction layers are adapted.

Acceptance emphasizes control utility: false-beneficial selections, within-same-prefix ranking,
selected-action regret, sign accuracy and event-balanced metrics. Scalar MAE alone cannot promote a
critic.

## Hybrid Practical online candidate portfolio

The deployed Practical controller does **not** solve the historical 12 x 109 = 1308-dimensional
L-BFGS-B full-plan problem online. Historical Development showed that high-dimensional surrogate
optimization could bind support ceilings and exploit model extrema, while locally correct H360 values
did not guarantee better repeated H10 control.

Completely removing differentiability is also unnecessary. At every decision all 109 facilities are
still screened, and the controller forms at most **four** distinct, support-constrained first-action
candidates:

1. `STEP2_H10_PROBE_SCALE_1.00` — full supported learned H10 direction;
2. `STEP2_H10_PROBE_SCALE_0.50` — half-magnitude learned direction;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE` — causal hydraulic pressure/headroom direction with actuator-type
   SWMM SETTING semantics;
4. `SUPPORT_CONSTRAINED_GRADIENT_H10` — one **109-dimensional H10-only projected-gradient candidate**.

The gradient proposal is not a return oracle and is not an optimizer that directly controls SWMM. It
uses autograd through frozen base Step2 only to propose one first target. Each gradient trial is
immediately projected to:

- physical actuator bounds;
- at most 0.5 supervisory target change per 10-minute update;
- frozen per-facility q95 first-move radius;
- frozen q95 changed-facility ceiling.

After all candidates are formed, each is contracted again to frozen q95 joint-sequence support using
the **actual H10 pulse** geometry. The policy-return critic scores all distinct candidates, adds the
matched one-sided calibration margin, and executes only the candidate with the smallest upper bound
when that upper bound is negative. Otherwise the controller HOLDs the previous supervisory target.

Thus the continuous gradient is only a **candidate proposer**. Final online choice remains:

`candidate family -> policy-return critic -> one-sided UCB admission -> H10 action or HOLD`.

No PFV/Peak penalty, action penalty, Auto-RBC/EFD warm start, future realised rainfall or online SWMM
candidate search is used.

## Rolling / receding control remains continuous in time

Finite candidate count does not make the controller static. Authoritative SWMM is observed on the
300-s model/observation grid and a new supervisory decision is made every 600 s. Only the current H10
first target is written. After that 10-minute interval the controller re-observes, updates causal
history, reconstructs state, regenerates the four-family portfolio, rescoring with the policy-return
critic and selects a new first action.

The closed-loop sequence is therefore:

`observe -> reconstruct -> forecast causally -> propose/rank -> execute H10 -> re-observe -> repeat`.

H10 candidate -> H350 HOLD is the **critic action encoding for a first-action counterfactual**, not an
instruction to hold the real drainage network unchanged for the next 350 minutes.

## First policy iteration: current hybrid pi0, not historical V12

Historical V12/L-BFGS-B is retained only as archival evidence/ablation. It is not the current online
controller and is not required to generate current policy-return data.

The first paired-label round uses frozen:

`PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V2`.

This pi0 uses the same four-family H10 candidate/support geometry as the final method but, before a
policy-return critic exists, ranks the candidates deterministically with frozen base Step2 H10 score
and executes only a predicted-beneficial supported first target. Exact paired SWMM then supplies
`Q^pi0` / `A^pi0` labels.

After critic training and calibration, the Practical policy becomes pi1. If pi1 materially changes the
state/action distribution relative to pi0, a new role-disjoint `Q^pi1` data round is required before
Policy Lock.

## Training-support and engineering boundaries

- 109 writable facilities remain screened; no hand-picked small actuator subset is introduced.
- physical setting bounds are respected;
- supervisory target slew is at most 0.5 per 10-minute update;
- first-move changed-facility density stays inside frozen TrainFit q95 support;
- the actual H10 counterfactual sequence stays inside frozen q95 joint-sequence support;
- HOLD means latch the previous supervisory target;
- unchanged facilities preserve their last commanded targets;
- score equals execute: no material post-score projection is allowed;
- target write/readback and causal-history readiness remain fail-closed.

## Exact-prefix authoritative policy-return data

For one query set, candidate and HOLD branches must share:

- the identical recorded supervisory action prefix;
- identical raw causal sensor/rainfall information up to the branch time;
- the same physical INP/forcing;
- the same frozen continuation policy after the H10 intervention.

Derived Step1 floating-point reconstruction differences are diagnostic and do not redefine physical
prefix identity. The bulk query runner computes one shared HOLD branch and then N sequential candidate
branches, so N candidates require `1 + N` authoritative branches rather than `2N`.

All candidate rows from one query share the same `query_set_id`; candidate ranking loss is computed
only inside these same-prefix query sets.

## Data roles

Rainfall group, not individual decision row, is the independent split unit. Before bulk label
generation, freeze role-disjoint groups for at least:

- policy-return train: 48 groups;
- model-selection validation: 12 groups;
- conformal calibration: 24 groups;
- separate new Development probes.

The matched calibration set must contain the actual online candidate family, including Step2-probe,
type-aware hydraulic and `SUPPORT_CONSTRAINED_GRADIENT_H10` examples. A calibration produced for the
old three-family portfolio cannot control four-family execution.

## Actuator semantics and baselines

A normalized hydraulic release intent is converted into the correct SWMM SETTING coordinate for each
actuator type. Pump, orifice, weir and outlet SETTING values are not assumed to share one universal
"larger setting = more release" meaning. The same type-aware semantics are used by the Proposed
hydraulic candidate and rule comparators.

Operational comparison remains:

- No-control — primary reference;
- Internal RTC — frozen native SWMM rules;
- Auto-RBC — causal rule-based comparator;
- EFD — causal storage equal-filling comparator.

All-max-setting (`all_open`) and all-min-setting (`all_closed`) remain diagnostic extremes. The
Proposed method is not required to beat every operational comparator on every event; superiority must
be assessed event-balanced on untouched authoritative SWMM events without weakening a comparator.

## Path-safe current execution

Historical V* directories are evidence, not automatic execution selectors. Current scripts consume a
single SHA-verified absolute asset manifest containing graph, sensors, config, Step1, base Step2,
sequence support and Priority8 nodes. Silent fallback to historical V12 policy admissions or guessed
paths is forbidden.

The current first-round sequence is:

1. `scripts/run_policy_direct_tfv_base_hybrid_parent_current.py`
2. `scripts/capture_direct_tfv_policy_return_context_current.py`
3. `scripts/design_direct_tfv_policy_return_portfolio_current.py`
4. `scripts/run_direct_tfv_policy_return_query_current.py`
5. `scripts/compile_direct_tfv_policy_return_dataset_current.py`
6. `scripts/train_direct_tfv_policy_return_current.py`
7. `scripts/score_direct_tfv_policy_return_calibration_current.py`
8. `scripts/calibrate_direct_tfv_policy_return_portfolio_admission_current.py`
9. `scripts/run_policy_direct_tfv_policy_return_development.py`
10. unchanged baseline panel + authoritative TFV/Priority8 PFV reporting.

## Development success before Policy Lock

Development succeeds only when:

1. future-information, engineering, readback, support and score/execute violations are all zero;
2. paired labels verify the same raw causal prefix and same frozen continuation;
3. projected-gradient candidates are finite, support-bounded, H10-only and add useful candidate
   coverage without dominating execution through the base Step2 score;
4. the critic has acceptable false-beneficial, same-query ranking and selected-regret behavior on
   role-disjoint validation;
5. independent Development closed loops show useful event-balanced TFV behavior relative to
   No-control, with Internal RTC, Auto-RBC and EFD reported fairly;
6. action frequency is scientifically reasonable — a return to near-all-HOLD behavior is treated as
   an admission/data problem, not as successful RTC;
7. every event used for a positive method-performance claim satisfies the frozen Priority8 PFV
   envelope;
8. the 10-minute supervisory runtime budget is satisfied with substantial margin;
9. if pi1 materially differs from pi0, a new role-disjoint Q^pi1 round is completed before Policy
   Lock.

Universal superiority is not a promotion requirement. Validation, Final, Formal and Policy Lock
remain inaccessible during policy development.

`READY_FOR_POLICY_LOCK=false` until these Development conditions are met.
