# Project7 Step2 NG1: D2-preserving process-aware interaction

NG1 is an isolated Development-only candidate. It does not replace the
publication V23 Step2, does not modify V15/V21, and does not change production
routing.

## Scientific contract

`PROJECT7_STEP2_NG1_D2_PRESERVED_PROCESS_AWARE_INTERACTION_V1` keeps the V5
facility main-value path for exact single-facility D2 branches and adds a
separate interaction value for multi-facility D3 branches:

`delta_TFV_hat = M_phi(candidate, reference) + I_psi(candidate, reference)`.

The interaction path uses the complete 109-actuator pair graph (5,886 pairs),
current causal state and rainfall context, actuator endpoint/physics/identity,
previous managed flow, and complete H360 action sequences. The graph is made
only from frozen graph assets; it contains no truth labels.

The following are structural invariants, not soft penalties:

- candidate equal to reference gives exact zero;
- swapping candidate and reference gives exact sign reversal;
- one changed facility gives an exact zero interaction residual;
- after D2 MAIN, all main parameters are frozen bitwise during JOINT/CONTROL.

D2 magnitude strata use q33/q67 computed from D2 TrainFit labels only, with
mutually exclusive and exhaustive small/medium/large bins. D3 density strata
use only TrainFit action metadata. The primary target remains authoritative
exact SWMM delta TFV; D4 is diagnostic-only and no hydraulic auxiliary is used
in the first NG1 run.

## Data and firewall

NG1 reuses the canonical existing Development cache and deterministic split:

`fit_d2=112`, `fit_d3=112`, `hold_d2=32`, `hold_d3=32`, with 14 fit rainfall
groups and 4 holdout rainfall groups and zero overlap. It creates no SWMM
runs, rainfall, policy-return truth, Validation access, Final access, Formal
tuning, or Policy Lock artifact.

The first run is fixed at seed 42 and the V5 development epochs and learning
rates. A failing historical-frontier comparison stops development; it does
not authorize V8, Step3, or a production promotion.
