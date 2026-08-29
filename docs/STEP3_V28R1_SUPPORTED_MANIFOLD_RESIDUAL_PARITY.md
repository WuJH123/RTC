# Project7 Step3 V28R1 — supported-manifold residual parity

## Status

Development-only. This lane is stacked on V28 and does not alter Step1, Step2, Q27, q95 support,
the candidate portfolio, PFV semantics, or the five-event Development benchmark.

## Why V28R1 exists

V28 produced a small closed-loop improvement over V27, but the residual model did not generalize:
Validation/Test ranking was weak and Test Spearman was negative. Code review found a concrete
train/deployment feature-parity problem.

The V28 residual feature vector contains raw-to-q95 contraction geometry. At deployment the runtime
starts from a true raw V23 proposal and computes a q95-supported target, so these fields can be
non-zero. In the residual trainer, however, the learning rows are already q95-supported action
identities; the stored supported target was passed through the projector again and used as both the
raw and supported target. Therefore raw-to-supported geometry was neutral during fitting but could
be material at deployment.

This is a distribution mismatch, not evidence that q95 should be removed. q95 remains mandatory.

## V28R1 hypothesis

A residual correction should only use features whose semantics are identical in the learning rows
and in the online q95-supported candidate manifold.

V28R1 therefore gives statistical weight only to:

- frozen Q27 supported-candidate score;
- supported first-move magnitude;
- changed-facility count;
- current causal network stress, rainfall level and strong-storm blend;
- candidate-family indicators.

The V28 raw/q95 contraction fields remain in the checkpoint/runtime feature contract for backward
compatibility, but receive exact zero residual-model weight:

- q95 contraction scale;
- q95 max support ratio;
- q95 binding flag;
- raw first-move L1;
- raw-to-supported first-move L1;
- raw-to-supported H120 L1;
- raw-to-supported total-variation L1.

This prevents deployment-only raw geometry from changing the V28R1 residual prediction.

## Model selection

V28R1 selects both ridge regularization and residual shrinkage using Train leakage-group CV only.
Validation and Test are report-only and cannot select the correction.

Residual shrinkage candidates are:

`0.00, 0.25, 0.50, 0.75, 1.00`.

`alpha = 0` exactly recovers frozen Q27. This is not a safety gate; it is a statistical null model in
the same model-selection problem. If the residual lacks reproducible Train-group decision value,
V28R1 can prefer no correction instead of forcing a noisy residual into the runtime.

The selection priority is:

1. mean decision regret on Train group-CV;
2. pairwise rank accuracy;
3. Q28 RMSE;
4. smaller residual shrinkage;
5. smaller ridge only as the final deterministic tie-break.

## Data and truth

No new rainfall or SWMM truth is required for V28R1. Reuse the V28 augmented dataset and the 32
already-computed targeted q95-supported exact-return records.

The V28 targeted-truth records contain `raw_candidate_target`, but the V28 augmented dataset did not
preserve that field. The raw target is useful provenance and should be retained in future dataset
materializations, but it is deliberately not used by the V28R1 residual until equivalent raw-action
lineage exists across the historical q95-exact learning population.

## Runtime

V28R1 reuses the existing V28 runtime and checkpoint contract. The fitted model remains a 17-wide
`V28ResidualValueModel`; deployment-mismatched columns simply have zero weights. Therefore the
runtime continues to:

1. generate the frozen V23/V27 candidate portfolio;
2. apply mandatory q95 support;
3. deduplicate equivalent supported targets;
4. score frozen Q27 plus the V28R1 residual;
5. compare candidates with HOLD = 0;
6. execute only the supported H10 action;
7. re-observe after 600 s.

No UCB, ACTION quota, stress escape, source priority, future rainfall or online SWMM candidate search
is introduced.

## Interpretation

V28R1 is supported if it improves cross-context ranking and/or closed-loop TFV without sacrificing
PFV/engineering execution. If Train CV selects `alpha = 0`, the scientific conclusion is that the
current residual supervision is not reproducibly better than frozen Q27; that is a valid result and
should not be overridden by Validation or Benchmark-specific tuning.
