# Codex handoff — Step2 historical-retrain V6

Continue only on branch `agent/step2-historical-retrain-v6`.

The web-side patch already added a preservation-aware trainer, a dedicated runner, a same-lineage
baseline/candidate comparator, pre-registered gates and tests. The next job is to validate the patch
locally, run the existing-data A/B experiment using exactly the frozen V5 Development inputs, and stop
at the offline comparator unless every hard gate passes.

Do not modify V23 lineage, Validation, Final, Policy Lock or existing Formal evidence during this
stage. Do not generate new SWMM truth until the existing-data comparator passes.
