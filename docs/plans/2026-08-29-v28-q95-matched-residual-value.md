# Project7 V28 q95-matched residual value RTC

## Objective

Build a Development-only V28 controller from the V27 reference that keeps the
authoritative q95 joint-sequence support and applies a frozen-Q27, lightweight
residual value correction. Raw portfolio proposals remain diagnostic only;
only post-q95 supported actions may be scored and executed.

## Boundaries

- Base the implementation on V27 reference commit `17cb028657ef0b8054b4c97481a2dc975b064d9c`.
- Keep Step1, Step2, V15, V21, V27, V23, and V24 artifacts immutable.
- Reuse existing exact-return evidence first; generate no new SWMM truth unless
  a deterministic planner proves a q95-supported action is absent.
- Do not access or tune Validation, Policy Lock, or Final artifacts.
- Use no event identifier, future rainfall, future state, or benchmark outcome
  as a model feature or selection signal.

## Implementation steps

1. Audit the V27 runtime, dataset, model checkpoint, baseline lineage, and
   archived local patch without modifying the original dirty worktree.
2. Add unit tests for q95-only execution, exact post-q95 deduplication,
   frozen-Q27 plus residual scoring, structured telemetry, and Development
   firewall behavior; run them red before implementation.
3. Add the V28 residual model, runtime factory, policy runner, Benchmark5
   runner, targeted truth planner, and end-to-end Development wrapper.
4. Reuse and audit existing q95-compatible exact-return records. Produce a
   deterministic targeted truth plan and run only missing Development
   counterfactuals if exact action identity is absent.
5. Fit the residual ridge only on the resulting group-isolated Development
   data, select ridge on Validation, and report Test once without using it for
   selection.
6. Run compile, lint, focused tests, full tests, and all CLI help checks.
7. Run one V28 Development Benchmark5 with immutable baseline cache, audit
   TFV/PFV/engineering/telemetry outputs, and do not tune or repeat the five
   events.
8. Commit only intended source, test, documentation, and small evidence files;
   push the branch and create a Draft PR for external review.

## Verification checklist

- q95 is mandatory and raw targets are never executable.
- post-q95 float32 target bytes are deduplicated and contributor sources are
  retained.
- HOLD has exact score zero; Q28 equals Q27 plus residual.
- zero residual reproduces V27 supported selection.
- passive channels, actuator bounds, first-radius, changed-facility ceiling,
  0.5 slew, target latch, and readback remain enforced.
- residual training uses the exact policy-return target and group-isolated
  Train/Validation/Test roles.
- no new training truth, rainfall, baseline SWMM, Validation, Policy Lock, or
  Final access occurs.
