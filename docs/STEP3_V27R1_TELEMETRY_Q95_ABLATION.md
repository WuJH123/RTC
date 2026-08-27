# Project7 Step3 V27R1 — telemetry correction and q95 ablation

V27 is retained as the numerical reference policy. The completed five-event Development Benchmark5
showed three findings that motivate V27R1 without redesigning Step3:

1. V27 improved TFV versus V26 and V23 on all five benchmark events and closed the aggregate gap to
   Auto-RBC to about 0.42%.
2. The runtime telemetry adapter incorrectly labelled every portfolio decision HOLD because the
   low-level `policy_return_admission_passed` field was not present after wrapping, even though the
   equivalent `admission_passed`/`candidate_valid` decision survived and 144 numerical commands
   changed.
3. q95 joint-sequence contraction changed the preferred candidate on 43.65% of decisions and changed
   the predicted ACTION/HOLD sign on 16.61%. This proves material decision impact, but it does not by
   itself prove q95 is harmful.

## Telemetry fix

`controller_direct_tfv_portfolio.py` now uses the runtime wrapper's preserved
`admission_passed`/`candidate_valid` decision when the historical
`policy_return_admission_passed` field is absent. The fix affects reporting only. V26/V27 direct-value
policies are also explicitly marked as not using the historical calibrated one-sided admission layer.

## q95 ablation

The primary V27 policy is unchanged. V27R1 adds a separate Development-only ablation that removes only
the learned q95 joint-sequence L1/TV contraction. It preserves:

- the 82/109 supervisory mask;
- actuator bounds and type semantics;
- first-move radius;
- changed-facility ceiling;
- the 0.5 maximum setting update;
- causal state/rainfall inputs;
- target write/readback verification;
- the frozen V27 value checkpoint and candidate generator;
- Auto-RBC as a candidate only.

The ablation therefore asks a single question: after an engineering-feasible first move has already
been constructed, does additionally shrinking the full H10 sequence toward HOLD according to the
historical q95 L1/TV support improve authoritative SWMM TFV?

No offline AUC, pairwise accuracy, precision or harmful-action statistic blocks this experiment. The
ablation reruns only Proposed on the same five-event benchmark and reuses the immutable baselines.
PFV and engineering execution remain reported from SWMM. The result is Development evidence only and
cannot by itself justify Policy Lock or Final.
