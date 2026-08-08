# Wuhan RTC Scientific Contract V1

## 1. Research question

Can a large urban drainage network be controlled online from sparse observations by reconstructing its hydraulic state, predicting the hydraulic consequences of continuous actuator settings with a differentiable world model, and repeatedly solving a safety-constrained MPC problem without rainfall-event-specific schedules or a preselected actuator subset?

## 2. Authoritative system

- Physical truth: EPA SWMM using the frozen `wuhan_v8_storage_retrofit.inp` supplied locally.
- Surrogate models are never final evidence.
- Final claims must be derived from closed-loop authoritative SWMM simulations on rainfall groups not used for model fitting, calibration, architecture selection, actuator screening, or policy tuning.

## 3. Actuator contract

All physically writable SWMM control links discovered from `[PUMPS]`, `[ORIFICES]`, `[WEIRS]`, and `[OUTLETS]` are eligible controls.

- No fixed Engineering36 or other manually selected active subset is part of the scientific contract.
- No pump is hard-coded as binary.
- The default normalized target setting is continuous in `[0, 1]` for every discovered actuator. Device-specific physical bounds may narrow this interval only when demonstrated by the frozen model or engineering metadata.
- The optimizer may learn that an actuator is hydraulically inactive at a given state; inactivity is state-dependent and must not be confused with permanent exclusion.

## 4. Priority-site safety

The eight priority nodes in `data/priority_nodes.txt` originate from observed ponding locations and remain first-class safety sites.

For each decision time, compare a proposed control trajectory with an operational fallback trajectory starting from the same reconstructed state and using the same causal rainfall ensemble.

Safety is **site-wise**: an improvement at one observed ponding location must not compensate for a large deterioration at another. For every priority node `j`, the controller must satisfy independently calibrated one-sided uncertainty limits for:

- flooding-volume deterioration `V_flood,cand,j - V_flood,fallback,j`; and
- maximum-depth deterioration over the prediction horizon.

An optional aggregate priority-flood-volume limit may be added, but it never replaces the per-site limits. An optional risk-transfer constraint may also prevent material new flooding outside the eight priority sites.

The numerical tolerances are calibration/engineering outputs, not arbitrary constants embedded in model code.

## 5. Optimisation objective

The optimisation is lexicographic:

1. satisfy actuator bounds and runtime executability;
2. satisfy all eight site-wise safety constraints;
3. satisfy the optional new-flood risk-transfer constraint;
4. minimise system-wide TFV over the prediction horizon; with rainfall ensembles, minimise a frozen risk statistic such as CVaR of TFV;
5. use control movement / energy only as a tie-breaker when flood performance is practically equivalent.

Global peak and other hydraulic statistics may be reported but are not automatically hard constraints.

## 6. Step 1: sparse hydraulic state estimation

Input at decision time `t`:

- a causal history of sparse sensor observations;
- observed rainfall up to `t`;
- actuator setting/readback history;
- frozen network topology and static attributes.

Output is the reconstructed network-wide hydraulic state required by Step 2, including node depth/head and, where supervised, edge/actuator flow, storage state and flooding state. Step 1 does not learn the control policy.

## 7. Step 2: differentiable hydraulic world model

Step 2 predicts physical consequences, not optimal actions.

Its intended causal factorisation is:

`state_k + setting_k + device physics -> actuator flow_k`

followed by

`state_k + actuator-flow injections + rainfall/runoff_k + graph -> state_(k+1)`.

The transition is rolled forward to the prediction horizon. Priority flooding and TFV are derived from predicted hydraulic trajectories; a direct KPI head may be auxiliary but never the sole primary world model. Training must include same-state counterfactual supervision so that control effects are not drowned by between-event severity.

## 8. Step 3: online MPC

At every update:

1. Step 1 reconstructs the current state.
2. Obtain a causal rainfall forecast or forecast ensemble.
3. Produce the fallback trajectory.
4. Step 2 evaluates differentiable control trajectories for all eligible actuators.
5. Optimise continuous settings subject to safety.
6. Perform hard post-optimisation safety admission.
7. Execute only the first control move.
8. Verify requested setting versus actual readback.
9. Observe the new state and repeat.

Historical experience may provide warm starts but must never restrict the optimiser to previously seen rainfall events or pre-enumerated actions.

## 9. Time-scale contract

Sampling interval, setting duration, control horizon and prediction horizon are configuration parameters. Before formal training, use hydraulic-response experiments to identify suitable values from the Wuhan network. Initial engineering values may be used for pilot runs but are not immutable scientific constants.

## 10. Data programme

### D0/D1 — baseline hydraulic coverage

Generate diverse rainfall and initial-condition trajectories that cover meaningful hydraulic regimes: filling, surcharge, flooding onset, high-flow and recession. Use these for Step 1 and base hydraulic dynamics.

### D2 — independent actuator probes

Every probe must start from the same authoritative checkpoint as its controls. Perturb exactly one actuator while holding all other actuator settings fixed. Never use sequential pulse experiments where one actuator inherits the state altered by a preceding pulse.

D2 learns state-dependent actionability, setting-to-flow response, local finite-difference truth, and hydraulic response delays/settling times.

### D3 — multi-actuator rollouts

After D2 is validated, generate multi-actuator trajectories to teach interaction and network propagation over the full prediction horizon.

### D4 — active learning

Add SWMM simulations only where the current model is uncertain, near hydraulic thresholds, where free rollouts drift, or where model gradients disagree with authoritative finite differences.

## 11. Required acceptance sequence

Do not skip layers:

1. INP / topology / actuator identity contract.
2. Step 1 reconstruction acceptance.
3. actuator setting-to-flow acceptance.
4. teacher-forced one-step hydraulic transition acceptance.
5. coupled one-step acceptance with predicted actuator flow.
6. short free-rollout acceptance.
7. full-horizon free-rollout acceptance.
8. same-state candidate ranking / control-effect acceptance.
9. finite-difference gradient-direction acceptance.
10. uncertainty and safety calibration on an independent set.
11. closed-loop development evaluation.
12. Policy Lock.
13. untouched final / blind closed-loop SWMM evaluation.

A failure at one layer blocks downstream claims.

## 12. Leakage prohibitions

Online Step 2 / Step 3 must not use future authoritative SWMM states, future realised rainfall unavailable at decision time, future outcomes of candidate actions, event IDs as policy features, or final/blind data for model fitting, actuator screening, calibration or hyperparameter selection.
