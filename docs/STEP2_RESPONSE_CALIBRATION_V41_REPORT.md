# Project7 Step2 response calibration V4.1

Verdict: **AMBER**.

This is development/train evidence only. No SWMM run, D2/D3 regeneration, Validation
outcome access, Final access, Formal Step2, full Train-only smoke, closed-loop MPC, or
formal-threshold change occurred.

## Result

The V4.1 mechanism is now capable of fitting the frozen complete D2 and D3 tiny groups.
The action pathway, single/interaction decomposition, exact-zero action response,
group-aware ranking, reference deduplication, direct authoritative DeltaTFV head,
non-negative physical flooding, and 1/5/10/20-actuator gradients all pass.

| cohort | source | spread ratio | rank | pairwise | sign | top1 | mean regret m3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | D2 | 1.194 | 0.987 | 0.967 | 0.957 | 1/1 | 0 |
| tiny | D3 | 1.609 | 0.976 | 0.964 | 0.875 | 1/1 | 0 |
| 12-group micro | D2 | 1.311 | 0.663 | 0.762 | 0.754 | 3/6 | 2,063 |
| 12-group micro | D3 | 0.911 | 0.278 | 0.607 | 0.708 | 1/6 | 121,941 |

The full-Train source-specific authoritative DeltaTFV RMS scales are `8,885.9 m3`
for D2 and `117,846.6 m3` for D3. The old V4 D2 ratio `0.0295` compared its prediction
against a sampled-rate surrogate; aligned to authoritative exact TFV it was `0.2216`.

V4.1 improves cross-group ranking, especially for D2, but D3 remains insufficient. In
the Train-only magnitude audit, large D3 effects have a mean-absolute response ratio of
only `0.411` and rank `0.331`. The maximum D3 best-action regret is `372,115 m3`.
Consequently, V4.1 is not ready to replace the active Step2 trainer and does not authorize
a full smoke or Formal Step2.

## Performance

- Old V4 tiny + micro: approximately `239.9 s`.
- Successful V4.1 tiny/combined + micro: `137.0 s`; micro alone `80.7 s`.
- Reference rows per group: D2 `48 -> 1`; D3 `16 -> 1`.
- Micro GPU utilization: mean `57.7%`, p90 `88%`, max `96%`.
- Micro GPU memory: Torch peak `3.01 GB`; `nvidia-smi` max `3,371 MiB`.
- Physical flooding negative fraction: exactly `0`.

Next bounded action: retain the same Train-only 12-group cohort and repair D3
magnitude-conditioned interaction calibration, especially large-effect under-response.
Do not run full smoke, Formal, Validation, Final, or new SWMM before that mechanism gate.

The complete machine-readable report is produced outside the repository under
`study_v069/step2_response_calibration_v41/STEP2_RESPONSE_CALIBRATION_V41_REPORT.json`.
