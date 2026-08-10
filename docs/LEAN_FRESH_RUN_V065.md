# Superseded — Lean fresh-data execution plan v0.6.5

This file is retained as a historical version marker only. **Do not use v0.6.5 for a new Project7 run.**

The Project7 pre-training audit subsequently found scientific P0 issues that v0.6.5 did not close: event INPs lacked the native rules needed for a valid Internal-RTC comparator, rainfall started before a complete causal history, the 180 min post-rain tail was right-censored, h180 response timing remained censored, and field actuator semantics were not proven.

The active contract is now:

```text
docs/LEAN_FRESH_RUN_V066.md
FORMAL_PIPELINE_LATEST.md
```

v0.6.6 adds event-forcing/native-rule pairing, dry/DWF hydraulic initialization, explicit recovery-tail preparation, fail-closed pretraining readiness, native-rule payload lineage, baseline information-budget disclosure and simulation-only actuator claims.

Historical v0.6.5 details remain available in Git history and must not be mixed with new v0.6.6 Formal evidence.
