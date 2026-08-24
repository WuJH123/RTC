# Project7 Step2 NG1: D2-preserving adaptive connectivity

NG1 is an isolated Development-only candidate. It does not replace the
publication V23 Step2, does not modify V15/V21, and does not change production
routing.

## Historical progression

R1 separated the D2 main-effect path from D3 interaction learning. On the
frozen seed-42 Development panel it improved D2 and improved D3 global
rank/pairwise/sign, but D3 Top1 and selected regret were worse than V5.

R2 tested a hard changed-changed actuator-pair interaction together with a more
decision-focused training variant. Under the same seed/data, D3 rank, pairwise,
Top1 and selected regret all deteriorated relative to R1. R2 is therefore not a
candidate for multi-seed rescue. Its main mechanistic implication is that
changed-to-unchanged hydraulic context appears necessary for joint-action
response representation.

## R3 scientific contract

`PROJECT7_STEP2_NG1_D2_PRESERVED_ADAPTIVE_CONNECTIVITY_V3` keeps the R1 D2
facility main-value path and restores the R1 loss/weighting semantics. The only
new scientific mechanism is the D3 interaction representation:

`delta_TFV_hat = M_phi(candidate, reference) + I_psi(candidate, reference)`.

The interaction path uses the complete 109-actuator relation graph as a
label-independent pair universe. A pair is eligible when at least one actuator
is changed. This retains changed-changed and changed-to-context propagation,
while learned masked attention uses pair latent state, action magnitude,
changed indicators, topology and standardized actuator physics to suppress
irrelevant relations.

The adaptive interaction value also receives eligible-pair density and mean
changed-actuator magnitude as context. It does not use outcome labels to build
connectivity.

The following remain exact structural invariants:

- candidate equal to reference gives exact zero;
- swapping candidate and reference gives exact sign reversal;
- one changed facility gives an exact zero interaction residual;
- after D2 MAIN, all main parameters are frozen bitwise during JOINT/CONTROL.

R3 intentionally restores the R1 training semantics:

- D2 corrected q33/q67 magnitude balancing and facility-balanced MAIN;
- D3 q50/q75/q90 TrainFit action-density balancing as in R1;
- selection temperature from the TrainFit q75 absolute delta-TFV scale;
- regression + pairwise rank + sign + soft selection regret + interaction L1;
- no R2 oracle-best margin loss and no R2 changed-changed hard mask.

This design isolates the adaptive-connectivity hypothesis from training-loss
changes.

## Data and firewall

NG1 reuses the canonical existing Development cache and deterministic split:

`fit_d2=112`, `fit_d3=112`, `hold_d2=32`, `hold_d3=32`, with 14 fit rainfall
groups and 4 Development model-selection rainfall groups and zero overlap.
It creates no SWMM runs, rainfall, policy-return truth, Validation access,
Final access, Formal tuning, or Policy Lock artifact.

The 32/32 panel has already been inspected during R1/R2 development and is
therefore a Development model-selection panel, not a future untouched
confirmatory holdout.

## Replication policy

One seed is sufficient to falsify an architectural change under a paired,
same-data comparison when it is directionally worse than its predecessor.
Accordingly, R2 is not rerun across seeds.

R3 first runs exactly once at seed 42. Only if it satisfies the frozen Pareto
Development gate is robustness replication authorized. The replication seeds
are predeclared as `42`, `2026`, and `3407`; all are reported and best-seed
selection is forbidden.
