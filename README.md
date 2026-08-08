# Wuhan RTC

Clean-slate research code for **state-adaptive real-time control (RTC) of a large urban drainage system** using `wuhan_v8_storage_retrofit.inp`.

## Scientific goal

The controller must not depend on rainfall-event IDs, rainfall-specific schedules, a fixed actuator subset, or a pre-enumerated action library. At every update it should:

1. reconstruct the current network-wide hydraulic state from sparse observations;
2. predict future hydraulics under causal rainfall information and proposed continuous actuator settings;
3. protect each of the eight observed real-world priority ponding locations from material deterioration relative to the fallback operation;
4. minimise system-wide total flood volume (TFV) among safe controls;
5. execute only the first move, verify readback, then re-observe and re-optimise.

```text
Sparse sensors + realised rainfall + actuator readback
                    |
                    v
          Step 1: state estimator
                    |
                    v
       reconstructed hydraulic state
                    |
       causal rainfall forecast ensemble
                    |
                    v
 Step 2: differentiable hydraulic world model
      setting -> facility flow -> network state
                    |
                    v
 Step 3: continuous receding-horizon MPC
       all writable actuators eligible
                    |
          site-wise priority safety
                    |
         minimise risk-adjusted TFV
                    |
                    v
        execute first move + readback
                    |
                    +---- repeat
```

## Deliberate departures from the previous Project6 design

- **No Engineering36 / fixed active actuator list.** Actuators are discovered from the INP (`PUMPS`, `ORIFICES`, `WEIRS`, `OUTLETS`).
- **No binary pump assumption.** SWMM settings are represented continuously in `[0, 1]`; no code path hard-binarises pumps.
- **No rainfall-ID policy.** Event names are metadata only, never policy features.
- **No direct `state + action -> PFV/TFV` primary surrogate.** Flood objectives are derived from predicted hydraulic trajectories.
- **No fixed K/Top-K control constraint.** Runtime actionability emerges from hydraulic state and the differentiable model.
- **The eight priority locations remain first-class safety sites** because they come from observed ponding information.
- **Priority safety is site-wise.** Improvement at one priority location cannot compensate large deterioration at another.
- **Time horizons are experiment-derived configuration**, not immutable scientific constants.

## Safety / optimisation contract

The policy is lexicographic:

1. actuator settings must be physically executable;
2. independently calibrated one-sided bounds on flood-volume and depth deterioration must pass at **each** observed priority site relative to the fallback trajectory;
3. optional risk-transfer protection may prevent material new flooding elsewhere;
4. among safe controls, minimise rainfall-ensemble risk-adjusted TFV;
5. use control movement/energy only as a tie-breaker when flood performance is practically equivalent.

Safety tolerances are frozen from independent calibration and/or justified engineering tolerances; they are not hard-coded scientific constants.

## Data strategy

- **D0/D1 hydraulic-state trajectories:** diverse rainfall/hydraulic regimes for Step 1 and base dynamics.
- **D2 independent actuator probes:** each actuator is perturbed from the *same checkpoint* while all others remain fixed. This prevents the sequential-pulse contamination found in the previous project.
- **D3 multi-actuator rollouts:** full-horizon trajectories for interactions and network propagation.
- **D4 active learning:** add SWMM simulations only where the current model is uncertain, near thresholds, or has poor rollout/gradient fidelity.

## Quick start

The large Wuhan INP is intentionally not duplicated here. Reference the frozen file locally, for example:

```text
data/wuhan_v8_storage_retrofit.inp
```

Install the core package:

```bash
python -m pip install -e .[dev]
```

For authoritative local SWMM data generation also install:

```bash
python -m pip install -e .[swmm]
```

Audit the INP and automatically discover all writable actuators:

```bash
rtc-audit-inp \
  --inp data/wuhan_v8_storage_retrofit.inp \
  --priority data/priority_nodes.txt
```

Prepare a checkpoint CSV with `checkpoint_id`, `checkpoint_minutes`, optional event/rainfall metadata, and one column `setting:<actuator_id>` for every discovered actuator. Then create independent single-actuator probe branches:

```bash
rtc-design-probes \
  --inp data/wuhan_v8_storage_retrofit.inp \
  --checkpoints checkpoint_settings.csv \
  --out outputs/probe_manifest.csv
```

Run a small authoritative SWMM probe batch locally:

```bash
rtc-run-probes \
  --manifest outputs/probe_manifest.csv \
  --inp data/wuhan_v8_storage_retrofit.inp \
  --out-dir outputs/d2_probe_runs \
  --limit 10
```

Each SWMM branch starts a fresh simulation, replays the same native prefix without Python overrides, records the pre-action checkpoint, applies the complete continuous action setting, and records node depth/head/flooding/volume plus actuator target/current setting and flow.

Run unit tests:

```bash
pytest -q
```

## Status

V1 intentionally does not import old Project6 training artefacts as authority. Old assets may only be reused after explicit physical-lineage, compatibility and leakage audits.
