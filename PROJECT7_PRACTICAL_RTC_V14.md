# Project7 Practical RTC V14 — simplified scientific contract

## Research question

Can sparse-sensor, causal, support-aware receding control reduce **system-wide total flooding volume (TFV)** relative to operating the same drainage network without supervisory RTC, while avoiding material deterioration of flooding at the frozen Priority8 nodes and satisfying executable actuator constraints?

The method is **not required to beat every comparator on every storm**. Internal SWMM rules, Auto-RBC and EFD are operational comparators. Numerical all-maximum/all-minimum SETTING policies are diagnostic extremes and are reported for interpretation only.

## Online information and objective

At each 10-minute control decision the controller may use only:

- the causal 13-frame sparse hydraulic history available up to the decision time;
- actuator target/current-setting and flow readback available up to the decision time;
- causal rainfall history and the frozen persistence/decay rainfall scenarios;
- frozen network topology, static node/actuator physics and support artifacts.

It must not use future realised rainfall, future SWMM state, future flooding labels or online SWMM candidate evaluation.

The **sole online optimization objective is system-wide cumulative TFV** under the receding-policy first-action value:

`A^pi(x_t, u_t) = J(candidate H10 -> frozen continuation pi) - J(HOLD H10 -> same frozen continuation pi)`.

Negative values are beneficial.

## Secondary PFV safety

PFV means **Priority8 Flooding Volume**, not peak flow. It is derived from authoritative SWMM flooding volume at the eight frozen priority nodes.

PFV is a secondary safety requirement, not a second online optimization objective. Unless a later preregistered study contract changes the tolerance, a Proposed event is PFV-safe when:

`PFV_proposed <= 100 m3 + 1.05 * PFV_no_control`.

This 100 m3 / 5% envelope is a Project-specific engineering non-inferiority tolerance, not a universal regulation. Global Peak remains report-only.

## Step1 / Step2 / policy-return roles

- **Step1** reconstructs causal full-network hydraulic state from sparse observations.
- **Base Step2** remains a lightweight pretrained TFV action-effect representation and **direction generator**. It is not assumed to be a sufficiently accurate deployed closed-loop value critic.
- **Policy-return critic** learns the first-action return under the actual frozen receding continuation. With limited authoritative labels, default fine-tuning adapts control/action heads first rather than retraining the complete representation.

Step2 acceptance for RTC must emphasize action utility: sign accuracy, false-beneficial rate, same-state candidate ranking, selected-action regret and event-balanced performance. Scalar MAE alone is insufficient.

## Practical candidate portfolio

To keep the control and evidence path tractable, each state uses at most three distinct engineering-supported candidate families after projection/deduplication:

1. learned V12 direction at 100% magnitude;
2. learned V12 direction at 50% magnitude;
3. type-aware hydraulic-pressure direction using storage/depth headroom, downstream congestion and causal rainfall.

Auto-RBC, EFD and numerical all-max/min policies are never Proposed warm starts or candidate actions.

All candidates remain inside existing actuator bounds, 0.5 target slew, q95 active-density support and q95 joint sequence support.

## Actuator semantics

A normalized hydraulic **release intent** must be converted to the correct SWMM SETTING coordinate for each actuator type. Pump, orifice, weir and outlet SETTING values are not assumed to share one physical meaning. The same type-aware conversion is applied to Proposed hydraulic candidates and to rule-based comparators.

## Baseline roles

The six-strategy evidence panel is retained for transparency and backwards-compatible reporting:

- No-control — primary reference: no supervisory RTC, native `[CONTROLS]` disabled, intrinsic device/local physics preserved;
- Internal RTC — operational comparator;
- Auto-RBC — operational causal rule-based comparator;
- EFD — operational storage equal-filling comparator;
- All-max-setting — diagnostic numerical extreme (legacy ID `all_open`);
- All-min-setting — diagnostic numerical extreme (legacy ID `all_closed`).

The Proposed policy is **not required to beat the diagnostic extremes**. Hold remains debug-only.

## Development success before Policy Lock

The next Development stage is successful only if all of the following are true:

1. zero engineering, readback, support and future-information violations;
2. exact paired labels verify the same raw causal authoritative prefix and same frozen continuation policy;
3. the policy-return critic improves or at least preserves strong-storm false-beneficial/sign/ranking behaviour relative to the base Step2 direction;
4. independent Development probes show useful TFV control relative to No-control on an event-balanced basis;
5. every claimed PFV-safe Proposed event satisfies the frozen PFV envelope;
6. Internal RTC, Auto-RBC and EFD are reported fairly with the same SWMM forcing/model and corrected actuator semantics.

Universal superiority over Internal RTC, Auto-RBC, EFD or diagnostic setting extremes is **not** a promotion requirement. Any superiority claim must be supported by untouched evaluation evidence.

`READY_FOR_POLICY_LOCK=false` until role-disjoint training/validation/calibration and new Development probes are completed.
