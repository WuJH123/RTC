# Project7 Practical RTC — paper-aligned scientific contract

## Research question

Can a sparse-sensor, strictly causal, training-support-constrained and engineering-executable learning
controller reduce **system-wide total flooding volume (TFV)** in a large SWMM drainage network while
avoiding material deterioration of flooding at the frozen Priority8 nodes?

The paper method is not required to beat every comparator on every storm. No-control is the primary
reference. Internal SWMM rules, Auto-RBC and EFD are operational comparators. Numerical all-maximum
and all-minimum SETTING policies are diagnostic extremes only.

## Keep hydraulic representation separate from supervisory-control freedom

The source testbed contains 109 hydraulic links represented by the pretrained Direct-TFV Step2 action
tensor. Project7 now distinguishes those **model action channels** from facilities with explicit
supervisory-control evidence in the source INP ``[CONTROLS]`` section.

- frozen Step2 hydraulic action representation: **109 channels**;
- current native supervisory-control subspace: **82 facilities**;
- remaining passive setting channels: **27**;
- every candidate must satisfy `candidate[i] == HOLD/reference[i]` on those 27 passive channels.

The 82-facility set is not a hand-picked performance subset. It is generated deterministically from
explicit action clauses in the source SWMM ``[CONTROLS]`` and is frozen by actuator order, mask SHA
and source-INP SHA. The current Wuhan testbed is expected to resolve to 57 pumps, 16 orifices and 9
weirs. A different count fails closed instead of silently changing the scientific action space.

This contract deliberately **does not retrain Step1 or base Step2**. Restricting candidate differences
to a subspace of the action representation removes online freedoms; it does not change the hydraulic
network represented by Step1/Step2.

## The five added Storage nodes remain hydraulic state, not control dimensions

The five RTC Storage additions are retained exactly as part of the frozen SWMM network. Their role is
to expose/quantify storage capacity and headroom in the hydraulic state representation. They are not
removed, redefined or counted as supervisory-control facilities when the online action freedom is
reduced from 109 to 82.

Accordingly:

- network topology and storage-capacity features remain unchanged;
- frozen Step1 remains unchanged;
- frozen 109-channel base Step2 remains unchanged;
- only the online candidate/reference **difference** is masked to the 82-facility supervisory space.

## Online information and objective

Every 10 minutes the controller may use only:

- the causal 13-frame sparse hydraulic history available up to the current decision;
- actuator target/current-setting and managed-flow readback available up to that time;
- causal rainfall history and frozen persistence/decay rainfall scenarios;
- frozen network topology, static node/actuator physics, native supervisory mask and training-support
  artifacts.

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

## Step1, base Step2 and policy-return roles

- **Step1** reconstructs the causal full-network hydraulic state from sparse observations. It is not
  retrained merely because the online supervisory mask changed.
- **Base Step2 V5** retains its frozen 109-channel TFV action-effect representation. It is reused for
  cheap first-action directional probes and one differentiable first-action proposal. It is not the
  deployed closed-loop value oracle and is not retrained merely to remove 27 online freedoms.
- **Policy-return critic** is initialized from base Step2 and fine-tuned on paired exact-prefix SWMM
  labels collected under the current 82-control mask and the actual receding continuation.

Acceptance emphasizes control utility: HOLD-aware false-beneficial/false-reject behavior, within-query
ranking, selected-action regret, sign accuracy and event-balanced metrics. Scalar MAE alone cannot
promote a critic.

## Cheap masked q95 support update instead of expensive retraining

The old q95 action geometry was measured in the 109-free-action Development distribution. Current
support is recomputed **offline** from the already-existing D3 TrainFit action tensors after masking
the 27 passive channels to zero action difference. No new SWMM simulation or new outcome label is
needed for this step.

The resulting artifact freezes, on the 82-control subspace:

- q95 changed-facility count;
- q95 first-block L1;
- q95 H120 L1;
- q95 H120 total variation;
- native supervisory-mask SHA and Step2/cache lineage.

This is the least-compute scientifically consistent response to changing the online control freedom:
reuse the learned hydraulic representation, but update the support envelope of the policy actually
being deployed.

## Hybrid Practical candidate portfolio

The deployed controller does **not** solve the historical 12 x 109 = 1308-dimensional L-BFGS-B
full-plan problem online. It forms at most four distinct H10 candidates on the 82-facility control
subspace while retaining 109-channel tensors:

1. `STEP2_H10_PROBE_SCALE_1.00`;
2. `STEP2_H10_PROBE_SCALE_0.50`;
3. `TYPE_AWARE_HYDRAULIC_PRESSURE`;
4. `SUPPORT_CONSTRAINED_GRADIENT_H10`.

The gradient proposal now has **82 free supervisory dimensions embedded in a 109-channel tensor**. It
is H10-only. Every gradient trial is projected to:

- the native supervisory mask;
- physical actuator bounds;
- at most 0.5 supervisory target change per 10-minute update;
- frozen per-facility q95 first-move radius from base Step2 training;
- the newly recomputed masked q95 changed-facility ceiling.

After candidate formation, each candidate is contracted to the newly recomputed masked q95
joint-sequence support using the actual H10 pulse geometry. The final post-mask/post-support
base-Step2 H10 score is frozen in the candidate artifact for later sign/rank diagnostics.

The gradient remains only a **candidate proposer**. Final online choice is:

`masked candidate family -> policy-return critic -> one-sided UCB admission -> H10 action or HOLD`.

No PFV/Peak penalty, action penalty, Auto-RBC/EFD warm start, future realised rainfall or online SWMM
candidate search is used.

## What the completed T30 109-free diagnostic taught us

The earlier seen T30 query, run before the native supervisory mask was introduced, produced an all-
harmful candidate set even though base Step2 predicted negative return for two candidates. The
projected-gradient candidate completed six accepted surrogate-improvement steps yet had the worst
exact receding-policy return. This established two useful lessons:

1. the gradient implementation was numerically functioning; increasing gradient steps or restoring
   high-dimensional L-BFGS-B is not the justified response to a value-landscape mismatch;
2. one all-harmful state means HOLD is oracle within that generated portfolio at that state, not that
   Step2 or all feasible actions are universally useless.

Because the supervisory policy has now changed from 109 free channels to 82, that old T30 exact truth
is retained only as **historical Development mechanism evidence**. It cannot be reused as current
82-control policy-return training/calibration truth.

## Rolling / receding control remains continuous in time

Finite candidate count and the 82-control mask do not make the controller static. Authoritative SWMM
is observed on the 300-s grid and a new supervisory decision is made every 600 s. Only the current
H10 target is written. The controller then re-observes, reconstructs state, regenerates the masked
portfolio and makes another decision.

`observe -> reconstruct -> forecast causally -> masked propose/rank -> execute H10 -> re-observe -> repeat`.

H10 candidate -> H350 HOLD is the critic encoding for a first-action counterfactual, not a command to
leave the real network unchanged for 350 minutes.

## First policy iteration: masked hybrid pi0

Historical V12/L-BFGS-B is archival only. The first current paired-label round uses:

`PROJECT7_PRACTICAL_BASE_H10_HYBRID_PARENT_PI0_V3_82CONTROL_109REP`.

Pi0 uses the same native mask, masked q95 support and four-family H10 proposal geometry as the eventual
policy, but ranks candidates with frozen base Step2 before a policy-return critic exists. Exact paired
SWMM supplies `Q^pi0` / `A^pi0` labels. Pi1 then uses the trained and matched-calibrated critic. If pi1
materially changes the state/action distribution, collect a role-disjoint `Q^pi1` round before Policy
Lock.

## Training-support and engineering boundaries

- 109 hydraulic/model action channels remain represented;
- exactly 82 current supervisory facilities may change online;
- the remaining 27 setting channels must equal HOLD/reference in every candidate;
- physical setting bounds are respected;
- supervisory target slew is at most 0.5 per 10-minute update;
- changed-facility density and H10 joint-sequence geometry remain inside masked TrainFit q95 support;
- HOLD means latch the previous supervisory target;
- score equals execute: no material post-score projection is allowed;
- target write/readback and causal-history readiness remain fail-closed.

The current scientific contract intentionally does **not** add pump-energy objectives, PID targets,
new high/low-level penalty terms, a pump-specific surrogate, or an online PFV surrogate. Such additions
would broaden the research question without evidence that they are needed for the present bottleneck.

## Exact-prefix authoritative policy-return data

For one query set, candidate and HOLD branches must share the identical recorded supervisory prefix,
raw causal sensor/rainfall information, physical INP/forcing and frozen continuation policy after H10.
The query identity now also includes the supervisory-mask SHA.

The bulk query runner computes one shared HOLD branch and N sequential candidate branches, requiring
`1 + N` authoritative branches instead of `2N`. Each authoritative record freezes:

- exact SWMM policy return;
- post-support base-Step2 H10 score;
- 82-control/109-channel contract;
- supervisory-mask SHA;
- proof that passive setting channels were unchanged;
- same-prefix and same-continuation audits.

## Seen-event mechanism gate before expensive bulk labels

Do not jump directly into 48/12/24 after changing the policy. First run a small label-blind seen-event
mechanism panel under the **new 82-control mask**: two deterministic causal-ready queries each from the
already-inspected T8, T30 and T80 Development events. Freeze all six query points before reading new
candidate SWMM truth.

These six queries are diagnostics only. They do not count as independent training, validation,
calibration or final evidence.

- If at least one valid query contains a truly beneficial generated candidate, useful candidate
  coverage exists and fresh role-disjoint exact policy-return learning is justified.
- If all six contain no beneficial generated candidate, stop before bulk and inspect proposal coverage
  plus Step2 state/value representation. Do not increase K/q95/gradient steps or add another heuristic
  family merely to force actions.

All-harmful query classification means HOLD is oracle **within the generated portfolio at that exact
state**; it is not a claim that no beneficial engineering-feasible action exists anywhere in the
109-channel hydraulic model.

## Data roles after the mechanism gate

Only after the new masked mechanism gate supports continued learning, freeze role-disjoint rainfall
groups for at least:

- policy-return train: 48 groups;
- model-selection validation: 12 groups;
- conformal calibration: 24 groups;
- separate new Development probes.

Train, validation and calibration datasets must share the same supervisory-mask SHA and the current
masked hybrid candidate contract. The critic checkpoint and conformal admission artifact carry that
same mask SHA, and runtime rejects any lineage mismatch.

## HOLD-aware critic selection

The actual decision set is `{HOLD=0 + generated candidates}`. Therefore model selection reports at
least:

- selected-action false-beneficial fraction;
- selected-action false-reject fraction;
- HOLD-aware selected regret;
- HOLD-aware decision accuracy;
- predicted-HOLD and oracle-HOLD fractions;
- within-query pairwise rank and candidate top-1;
- event-balanced MAE/sign accuracy.

A query where all true candidate returns are positive and the critic predicts all candidates positive
is a correct HOLD decision, not a false-beneficial failure.

## Actuator semantics and baselines

Hydraulic release intent retains actuator-specific SWMM SETTING semantics. The native supervisory mask
only answers **which facilities can be changed by the paper RTC**; it does not remove passive links
from hydraulic simulation or change comparator definitions.

Operational comparison remains No-control, Internal RTC, Auto-RBC and EFD. All-max-setting and
all-min-setting remain diagnostic extremes. No comparator is weakened or used as a warm start.

## Path-safe current execution

The single SHA-verified asset manifest now contains:

- graph;
- sensors;
- runtime config;
- frozen Step1;
- frozen 109-channel base Step2;
- native supervisory-control mask;
- matching masked q95 sequence support;
- Priority8 nodes.

Build the two cheap control artifacts before the manifest:

1. `scripts/build_native_supervisory_control_current.py` — source INP -> 82-of-109 mask;
2. `scripts/build_direct_tfv_sequence_support_current.py` — existing D3 TrainFit actions -> masked q95
   support, **no new SWMM**.

Then current execution is:

3. `scripts/build_project7_practical_asset_manifest_current.py`;
4. `scripts/run_policy_direct_tfv_base_hybrid_parent_current.py`;
5. `scripts/capture_direct_tfv_policy_return_context_current.py`;
6. `scripts/design_direct_tfv_policy_return_portfolio_current.py`;
7. `scripts/run_direct_tfv_policy_return_query_current.py`;
8. `scripts/audit_direct_tfv_policy_return_mechanism_panel_current.py`;
9. only if justified: compile/train/score/calibrate policy return;
10. new Development closed loop + unchanged baselines + authoritative TFV/Priority8 PFV reporting.

The old pair runner and historical V* admissions are archival. Do not use them as the current bulk
label path.

## Development success before Policy Lock

Development succeeds only when:

1. future-information, engineering, readback, support, passive-channel and score/execute violations are
   all zero;
2. paired labels verify the same raw causal prefix and same frozen continuation;
3. the current mechanism panel demonstrates that useful generated first actions exist before expensive
   bulk learning proceeds;
4. the critic has acceptable HOLD-aware false-beneficial/false-reject, same-query ranking and selected
   regret on role-disjoint validation;
5. independent Development closed loops show useful event-balanced TFV behavior relative to
   No-control, with Internal RTC, Auto-RBC and EFD reported fairly;
6. action frequency is scientifically reasonable — near-all-HOLD is diagnosed, not celebrated;
7. positive event claims satisfy the frozen Priority8 PFV envelope;
8. the 600-s supervisory runtime budget is met with substantial margin;
9. if pi1 materially differs from pi0, a role-disjoint Q^pi1 round is completed before Policy Lock.

Universal superiority is not a promotion requirement. Validation, Final, Formal and Policy Lock
remain inaccessible during policy development.

`READY_FOR_POLICY_LOCK=false`.
