# Project7 Step3 V27 — decision-aware exact-return control with Auto-RBC shadow proposals

V27 is a Development-only successor to V26. It preserves the frozen physical RTC contract and the
historical exact-return dataset strategy, but addresses seven scientific/code bottlenecks that became
visible after V26 expanded from 47 to hundreds of unique historical state-action observations.

## 1. Pointwise value regression did not match the deployed ranking problem

V26 fitted each candidate's exact TFV return independently, then deployed the fitted values through an
`argmin` over candidates. V27 trains the same exact-return quantity but augments pointwise equations
with same-state pairwise return-difference equations. Pairwise terms are generated only among actions
belonging to the same canonical causal decision unit.

The model family remains compact and auditable. A leakage-group CV inside Train evaluates a fixed
ridge x pairwise-weight grid. Only a small Train-CV shortlist is refit on all Train and compared on
Validation. Test is read once after model selection and never changes the chosen configuration.

## 2. Reporting-value clipping no longer controls ranking

V26 inverted the asinh target transform with `sinh(clip(z,-8,8))` and used the resulting m3 value for
candidate ranking. Several out-of-support candidates could therefore collapse to the same saturated
reported value.

V27 uses the unclipped latent coordinate `z` for candidate ordering and ACTION/HOLD. Because sinh is
strictly monotone and physical zero return maps to latent zero, this preserves the intended ordering
without requiring unbounded m3 reporting. The clipped inverse transform remains diagnostic only.
Every split report records latent range and reporting clip-hit fraction; closed-loop decisions also
record clip-hit counts.

## 3. Auto-RBC is now an engineering proposal, not just a source of aggregate features

Auto-RBC remains a comparator and does not receive ACTION privilege. V27 reproduces its causal
upstream-fill/downstream-congestion/type-aware release logic to generate one `AUTO_RBC_SHADOW_TOPK`
proposal. Before value scoring, this proposal is projected through the same 82/109 supervisory mask,
first-move support, actuator bounds, 0.5 slew and changed-facility ceiling used by the learned
portfolio. The same V27 value model then decides whether the shadow candidate, another candidate, or
HOLD is preferable.

This is intentionally a projected shadow of the comparator, not a claim that the Proposed runtime is
identical to the baseline Auto-RBC controller.

## 4. Historical missing-context recovery uses composite lineage

V26 could stop on an ambiguous `query_set_id` even when a more specific prefix or event-decision
identity separated the historical states. V27 indexes all available causal contexts by stronger
identities first: context-file SHA, prefix + continuation, prefix, event/decision/elapsed and
query+action. Available strong identities are intersected before weaker query/group identities are
considered. A peer's candidate action is never borrowed; only the causal context can be recovered.

Rows that still cannot be uniquely resolved remain excluded individually. They do not abort the rest
of the historical dataset and no nearest-neighbour guessing is used.

## 5. Hydraulic representation preserves more local spatial structure

The V26 aggregate hydraulic features remain. V27 additionally appends top-k local values for upstream
pressure, downstream congestion, pressure gradient, downstream headroom, action magnitude and signed
release-gradient alignment. It also adds action-conditioned summaries for pump, orifice, weir and
outlet groups. No stress/filling threshold directly authorizes ACTION.

## 6. Model selection is group-aware and Test stays untouched

V27 performs deterministic leakage-group cross-validation entirely inside Train. CV is used to rank
regularization/pairwise-weight configurations by decision regret, realized selected exact return,
pairwise ranking accuracy and candidate RMSE. Validation chooses among the Train-CV shortlist. Test is
reported only after the checkpoint is fixed.

Offline AUC, sign accuracy, precision or harmful-action counts remain scientific diagnostics. They do
not block the Development Benchmark5 run.

## 7. q95 sequence support is kept for execution but audited as a causal intervention

V26 observed frequent q95 binding. V27 does not silently remove the support contract. For every online
candidate it scores both the first-move-supported raw target and the q95-contracted target with the
same value model. Actual execution still uses q95. The decision log records whether q95 changes the
preferred source/value, how many candidates bind, and the raw/supported source transition.

`audit_project7_v27_decision_diagnostics.py` aggregates these diagnostics after Benchmark5. If q95
rarely changes ranking, it is unlikely to explain the remaining Auto-RBC gap. If it frequently changes
the preferred action, a separate Development ablation can be designed without retrospectively
changing this V27 policy.

## End-to-end Development path

`run_project7_v27_end_to_end_development.py` performs:

1. historical exact-return inventory;
2. V27 composite-lineage dataset rebuild and leakage-safe Train/Validation/Test split;
3. decision-aware value training with Train group-CV and Validation selection;
4. five-event Proposed Benchmark5 with immutable baseline reuse.

No new training-truth SWMM is generated. No offline model-quality threshold is allowed to short-circuit
Benchmark5. Physical/engineering execution and data-integrity failures remain fail-closed.

After Benchmark5, run the V27 decision diagnostic audit and compare TFV/PFV/engineering results with
V26, V23 and Auto-RBC. V27 remains Development-only until independent evidence justifies a later
Policy Lock; this branch must not relabel Development results as Formal/Final evidence.
