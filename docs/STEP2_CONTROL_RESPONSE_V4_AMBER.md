# Step2 control-response V4 AMBER checkpoint

This document freezes the isolated V4 mechanism state before V4.1 response-calibration work. It is development/train evidence only and is not wired into the active Step2 trainer.

## Boundary

- Git baseline: `49115c49c0ef895243abf2b071c30d2b1feba7e6`
- Branch: `agent/step2-control-response-v4`
- SWMM launched: **NO**
- D2/D3 regenerated: **NO**
- Validation outcomes: **NOT ACCESSED**
- Final: **NOT ACCESSED**
- Formal Step2 / closed-loop: **NOT RUN**

## Mechanism result

- V3 later-segment branch-specific teacher forcing leakage: **300/300** later checks failed same-prefix.
- V4 reference/action-effect separation: **PASS**.
- Structural zero-action exact-zero: **PASS**.
- Future-action causality: **PASS**.
- Actuator identity and directed scatter: **PASS**.
- Five- and ten-actuator simultaneous action differentiation: **PASS**.
- Physical flooding-rate output: **FAIL/AMBER**; maximum observed negative fraction was about `0.536`.

## Train-only evidence

| Cohort | Source | Predicted spread (m3) | True spread (m3) | Spread ratio | Rank | Sign | Top-1 |
|---|---|---:|---:|---:|---:|---:|---|
| tiny | D2 | 3,929 | 132,976 | 0.0295 | -0.0165 | 0.708 | FAIL |
| tiny | D3 | 53,524 | 111,030 | 0.482 | 0.571 | 1.000 | FAIL |
| 12-group mean | D2 | 3,773 | 27,771 | 0.163 | 0.045 | 0.424 | 0/6 |
| 12-group mean | D3 | 35,690 | 375,544 | 0.145 | -0.016 | 0.646 | 1/6 |

The pathway is nonzero, but response magnitude, candidate ranking and flooding physics are not accepted. The V4 verdict is **AMBER**. It is not ready to replace the active trainer, run a full Train-only smoke, or run Formal Step2.

Next bounded action: audit Train-only target scales and per-loss gradients, then develop V4.1 without accessing Validation or Final and without generating new SWMM data.
