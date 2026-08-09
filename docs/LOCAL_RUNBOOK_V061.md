# Local runbook — Wuhan RTC v0.6.1 (deprecated)

Do **not** use this workflow for a new Formal run.

The v0.6.2 audit changed scientific data semantics that invalidate v0.6.1-derived RTC trajectories/models/evidence:

- D0 is explicitly t=0 inclusive;
- D2/D3 verify the complete saved No-control hydraulic/readback checkpoint before action;
- Step1/Step2/Proposed/Final bind one SWMM engine lineage;
- D3 respects the frozen sequential setting-rate contract;
- Final is bound to the locked event forcing (including external `FILE` bytes) and must contain every locked Final event.

Use the canonical workflow instead:

```text
docs/LOCAL_RUNBOOK_V062.md
```

A v0.6.1 workspace may be retained for audit/history, but its derived RTC data, trained models, acceptance evidence and Policy Lock are not Formal-compatible with v0.6.2.
