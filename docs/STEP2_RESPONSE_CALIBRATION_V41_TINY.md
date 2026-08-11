# Step2 response calibration V4.1 — Train-only tiny checkpoint

Status: **TINY D2 PASS; TINY D3 PASS; TWO-GROUP COMBINED PASS**.

This is development/train mechanism evidence only. It is not Formal Step2 evidence and
does not authorize a full Train-only smoke, validation access, Final access, or closed-loop
MPC. No SWMM process was launched and neither D2 nor D3 was regenerated.

## Corrected root cause

The V4 action pathway existed, but the training target and response calibration were
misaligned. The old D2 `true spread = 132,976 m3` was a sampled flooding-rate rectangle
surrogate, not authoritative cumulative SWMM TFV. Against the authoritative D2 spread of
`17,732 m3`, the old V4 predicted spread of `3,929 m3` had ratio `0.222`, not `0.0295`.

V4.1 uses source-separated physical counterfactual scales derived once from the full
development/train manifest (`144 D2 + 144 D3 groups`). It trains direct DeltaTFV against
authoritative cumulative SWMM volume, uses all meaningful within-group candidate pairs,
and keeps physical flooding non-negative through a latent positive mapping.

The D3 debugging gate found and corrected two additional mechanism defects:

1. the additive single-actuator branch was incorrectly multiplied by the D3 scale inside
   D3; it now retains its D2 physical calibration in both sources;
2. the interaction summary diluted 107–109 active actuators through a global mean and a
   `/108` gate; it now uses a binary `>=2 active actuators` gate plus identity-weighted and
   pairwise moments. One active actuator still gives an exact-zero interaction residual.

## Tiny results

| stage | source | spread ratio | Spearman | pairwise | sign | top1 | regret m3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| single group | D2 | 1.194 | 0.987 | 0.967 | 0.957 | PASS | 0 |
| single group | D3 | 1.609 | 0.976 | 0.964 | 0.875 | PASS | 0 |
| combined | D2 | 1.396 | 0.982 | 0.963 | 0.913 | PASS | 0 |
| combined | D3 | 2.172 | 0.857 | 0.893 | 0.875 | PASS | 0 |

D3 single-effect spread was `93.6k m3`, interaction-residual spread was `177.2k m3`,
and final spread was `189.6k m3`; the interaction branch did not collapse the total
response to zero. All tiny candidate-setting gradients were finite and all changed-action
gradients were non-zero. The physical flooding negative fraction was exactly zero.

## Performance

- D2 tiny: 80 epochs, 25.17 s wall time, 2.43 GB peak allocated GPU memory.
- D3 tiny: best epoch 54, early stop at epoch 79, 15.60 s wall time, 0.93 GB peak.
- two-group combined: best epoch 7, early stop at epoch 32, 15.52 s wall time, 2.46 GB peak.
- one group is one GPU batch; the reference is encoded once and all candidates run in
  parallel. Static graph and actuator encodings are computed once per group forward.

Next authorized bounded action: evaluate/train the unchanged frozen 12-group
development/train micro cohort. Do not run full smoke or Formal Step2.
