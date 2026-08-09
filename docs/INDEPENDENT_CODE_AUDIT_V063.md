# Independent code audit — RTC v0.6.3

This audit starts from the executable code and deliberately does not infer the intended study from earlier project descriptions.

## 1. What the code actually implements

The executable Proposed path is a model-based receding-horizon controller:

```text
causal sparse hydraulic observations + realised rainfall + actuator readback
                              |
                              v
                 Step1 sparse-state estimator
                              |
                        current full state
                              |
                              v
              Step2 differentiable world model
 current state + future setting sequence + causal rainfall scenarios
                              |
             future hydraulic/flooding trajectories
                              |
                              v
            continuous projected-gradient TFV MPC
                              |
                    first setting move only
                              |
                              v
                    authoritative SWMM
```

The code therefore attempts to solve three different learning/control problems, not one:

1. **State observability** — infer the current full hydraulic state from sparse measurements.
2. **Action-effect learning** — learn how actuator settings alter future hydraulic states, managed flows and flood volume.
3. **Online optimization** — exploit the learned action effect fast enough to select a useful first move before the next control deadline.

The action catalog is discovered directly from SWMM `PUMPS`, `ORIFICES`, `WEIRS` and `OUTLETS`. The actual count for a new study must be taken from the current INP audit rather than assumed from an old facility list.

## 2. What objective the current controller can pursue

The production MPC minimizes predicted cumulative system-wide flooding volume over its horizon. It uses continuous direct settings projected to the executable `[0,1]` interval and optionally to a sequential per-update setting-change limit.

Priority-site flooding/depth is a soft secondary preference. It is not what makes the mathematical action set feasible or infeasible.

This distinction matters: with the current code, **holding the current settings is always an available executable sequence**. Therefore the main early scientific risk is not an empty feasible set. The real risks are:

- actuator changes have little physical influence on future TFV at relevant states;
- Step2 does not learn the sign/ranking of those action effects;
- gradient optimization fails to exploit an otherwise learnable effect;
- rainfall forecasting is too weak for useful horizon decisions;
- the optimized decision takes longer than the supervisory update interval.

## 3. P0 defect found by this audit

The v0.6.2 Step2 trainer writes the model time contract:

```text
STEP2_FIXED_DISCRETE_TIME_ENGINE_V2
```

and stores `swmm_engine_version` in `model_config`.

The v0.6.2 production loader still required the old `STEP2_FIXED_DISCRETE_TIME_V1` and did not remove `swmm_engine_version` before constructing `DifferentiableHydraulicWorldModel`.

Consequences of a fresh run were severe:

```text
fresh Step2 training -> checkpoint successfully written
                     -> Step2 load/acceptance/Proposed runtime rejected or TypeError
```

v0.6.3 aligns the train/load/runtime contract and preserves the engine identity as runtime metadata.

## 4. MPC robustness defect found by this audit

The previous optimizer treated any finite final Adam iterate as a valid candidate. It did not:

- preserve the lowest-TFV iterate already found;
- require the selected candidate to beat the explicit hold/fallback sequence.

A late optimization step could therefore be worse than an earlier one and still be executed.

v0.6.3 production MPC now:

1. retains the best primary TFV iterate seen during optimization;
2. retains the best priority-secondary iterate inside the TFV near-optimal set;
3. computes the explicit hold/fallback TFV under the same rainfall scenarios;
4. admits the candidate only when predicted TFV is lower than fallback by the configured minimum improvement.

The default minimum improvement is zero plus a numerical margin, so this is a lightweight dominance check rather than a new scientific safety gate.

## 5. Why the old full D2 design is unnecessarily expensive

The exhaustive local design probes every actuator at every checkpoint with center and +/- perturbations. If the INP contains `A` controllable links and there are `C` checkpoints, the raw design is approximately:

```text
C * (1 + 2A) unique SWMM branches
```

For a large network this becomes expensive before we know whether many of those actuator/state combinations have any useful hydraulic leverage.

v0.6.3 adds a **budgeted rotating D2 design**. At each checkpoint only `B` actuators receive local perturbations, while every candidate still contains a complete setting vector for every actuator. Across successive checkpoints the sampled actuator IDs rotate deterministically.

Hence:

```text
training SWMM budget is reduced
online MPC action dimension is NOT reduced
no fixed Top-K subset is introduced
```

For `B=24`, a typical checkpoint needs roughly 49 unique branches instead of roughly `1+2A`.

## 6. The most important pre-training experiment

Before generating a large Step2 dataset, run a small **exact-SWMM control-leverage pilot**.

For each representative replayable checkpoint, compare exact SWMM future TFV for:

- the center/hold action;
- local D2 perturbations;
- optionally a few joint D3 sequences.

Report:

```text
reference TFV
best exact TFV
best exact TFV improvement
best reduction percentage
action-to-action TFV spread
fraction of checkpoints with meaningful improvement
number of sampled actuators with measurable TFV effect
```

This answers a more fundamental question than surrogate accuracy:

> Is there a real hydraulic signal for the learner and optimizer to exploit?

The v0.6.3 command is:

```text
rtc-control-leverage-audit
```

It is intentionally **diagnostic only** and does not create another Policy-Lock prerequisite.

Interpretation guidance:

- `PROMISING_CONTROL_LEVERAGE`: enough sampled states show useful exact-SWMM improvement to justify expanded Step2 generation.
- `WEAK_OR_STATE_DEPENDENT_CONTROL_LEVERAGE`: expand the pilot over more high-depth/high-flood states before large training.
- `LITTLE_MEASURABLE_CONTROL_LEVERAGE_IN_PILOT`: do not scale Step2 yet; first inspect actuator semantics, controllable assets and checkpoint placement.

## 7. Four practical stop/go decisions — not a forest of gates

For engineering development, use only four major decisions:

### A. Structural sanity

Can SWMM run the fresh event files, are units/topology/sensors valid, and do discovered settings/readbacks behave as expected?

### B. Physical control leverage

Do exact SWMM perturbations measurably alter TFV at representative flooded states?

### C. Learnability

Does Step1 reconstruct unobserved state sufficiently for control, and does Step2 rank/action-effect predict held-out SWMM candidates sufficiently well?

The most important Step2 metrics for control are action ranking/regret and gradient sign, not merely small state RMSE.

### D. Closed-loop value

On development events, does Proposed authoritative SWMM actually reduce TFV versus hold/no-control while meeting the computation deadline and producing non-degenerate actuator movement?

Only after D is positive is a large untouched Final evaluation worth the cost.

Existing detailed diagnostics can remain available for publication-quality evidence. They should not all be treated as independent reasons to stop early development unless they expose a genuine numerical/physical failure.

## 8. Failure diagnosis map

```text
Exact SWMM leverage weak
    -> physical/action-space/checkpoint problem; more ML data is unlikely to solve it

Exact leverage strong, Step2 ranking/gradient weak
    -> data coverage/model-learning problem

Step2 action effect good, MPC predicts no better-than-hold action
    -> optimizer/forecast/horizon problem

MPC predicts improvement, authoritative SWMM development does not
    -> surrogate extrapolation or rainfall forecast error

Development SWMM improves but runtime exceeds update interval
    -> computation/optimizer iteration problem
```

This decomposition prevents spending days generating data without knowing which layer is limiting the controller.
