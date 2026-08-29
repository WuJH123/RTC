# Project7 V28: q95-matched residual value RTC

V28 is a Development-only extension of the V27 q95-supported controller.  It preserves the
V27 Q27 checkpoint and the q95 joint-sequence support layer, then applies a deterministic ridge
residual correction to the supported candidate score:

```text
Q28(x, u_q95) = Q27(x, u_q95) + r28(z)
```

The residual features describe only the current causal prefix, q95 contraction geometry,
candidate family, and current candidate score.  Event identifiers and future realized states
are excluded.  The raw V23 proposals are diagnostic only; after physical projection and q95
projection, float32-equivalent supported targets are deduplicated before scoring.  The selected
supported candidate is compared with `HOLD = 0`, and only the q95-supported sequence can be
executed.

The exact policy-return estimand is unchanged:

```text
true_policy_return_delta_tfv_m3
  = J(candidate H10 + frozen causal continuation)
  - J(HOLD H10 + identical continuation)
```

H120 is diagnostic only.  PFV remains the secondary safety check and Global Peak remains
report-only.  The V28 lane does not modify or promote V23, V27, V27R1, V15, V21, Policy Lock,
or Final artifacts.

## Development data and lineage

The residual fit reuses the 478-row V27 exact-return dataset and adds only the 32 missing
q95-supported exact-return records selected by the causal targeted-truth planner.  The augmented
dataset has 510 unique rows, 216 causal contexts, and 93 leakage groups.  Its rainfall/context
split is train 374 rows / 65 groups, validation 69 / 14, and test 67 / 14.  The leakage audit
passed, and test rows are not used for fitting or ridge selection.

The executed residual checkpoint is bound to the frozen V27 value checkpoint and the augmented
dataset manifest.  The selected ridge is 0.0 and the optional pairwise weight is 0.0; these are
reported Development selections, not runtime gates.

## Reproducible result artifacts

The result audit is generated only from existing metadata, structured decision JSONL, and the
Benchmark5 comparison; it does not invoke SWMM:

- `scripts/audit_project7_v28_q95_results.py`
- `V28_Q95_MATCHED_RESIDUAL_BENCHMARK5_AUDIT.json`
- `V28_Q95_MATCHED_RESIDUAL_BENCHMARK5_AUDIT.csv`
- `V28_Q95_MATCHED_RESIDUAL_BENCHMARK5_AUDIT.md`

The authoritative Development Benchmark5 run executed Proposed V28 for five events and reused
the immutable No-control, Internal RTC, Auto-RBC, and EFD baseline cache.  Policy Lock and Final
were not accessed.
