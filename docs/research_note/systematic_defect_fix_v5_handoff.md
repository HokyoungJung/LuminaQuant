# Systematic Defect Fix v5 — Data-PC Handoff (2026-07-10)

8-class systematic defect hunt over the research/evaluation stack (workflow journal
preserved in-session; synthesis in `.omc/research/systematic-defect-hunt-v5-20260710.md`).
Everything below is on branch `fix/silent-proxy-substitution-v5` (stacked on main
after PR #46).

## CRITICAL — prior walkforward rejects are INVALID for research_only lanes

**Finding #1 (silent proxy substitution).** The research signal dispatcher fell back
to ONE shared generic 64-bar momentum z proxy for every `strategy_class` without a
bespoke handler — silently, with no marker in the output rows. 84/111 candidate
classes (2833/3674 candidates), including **all 33 new research_only sleeves from
waves v2/v2b/v2c**, were never executed by the engine; their recorded metrics
measured the shared proxy, not the lane. Two different lanes produced byte-identical
return streams.

Consequences for previously reported results:
- The `full_universe_walkforward` rejects for research_only sleeve lanes measured
  the proxy. They say nothing about the lanes. (Handler-mapped classes — the
  incumbent families — were evaluated correctly and their numbers stand.)
- The 2026-07-09 postmortem of the "6 measured rows" examined lane code that never
  ran in the engine.

**Fix.** `strategy_signal_dispatch.dispatch` now:
- routes unmapped-but-registered classes through the REAL strategy class via the
  registry simulator (window-capable sleeves get MARKET_WINDOW feeds), gated by
  `research.route_unmapped_registered_strategies` (**ON in
  `configs/profiles/research.yaml`**, OFF-default byte-identical elsewhere);
- labels every row's evaluation path in meta: `evaluation_mode` ∈
  {`handler`, `registry_simulator`, `generic_fallback_proxy`} — **audit this column
  in the next run; research_only lanes must NOT be `generic_fallback_proxy`**;
- fixes an `np.roll` wrap-around that leaked end-of-sample exposure into bar 0.

## Also fixed in v5 (changes measurement behavior)

1. **LeadLagSpillover handler look-ahead**: full-sample sigma normalisation →
   per-bar expanding sigma (32-bar warmup) + zero-fill lag shift (no head wrap).
   LeadLag handler-path numbers from prior runs were optimistic; re-measure.
2. **Cost-stress double-charge**: `_stress_metrics` in the efficiency-repair
   script charged ROUND-TRIP bps per ONE-WAY trade event (base sim uses `/2`).
   The 15/20bps stress gates were 2x too harsh; some previously stress-rejected
   candidates may now pass.
3. **Zero-alloc → engine-default resize (all lane templates)**: entries with
   computed `alloc <= 0` omitted `target_allocation` from metadata and the engine
   sized them at the config default (~10% of equity, unsized/un-vol-gated). All
   `_emit_targets`/`_emit_weighted` template copies + slow_leadlag + the crash-gate
   rotation now skip (or flatten) zero-weight targets instead of emitting.
4. **3 remaining inert vol-target throttles** (missed by v4): flow_share_rotation,
   diversified_multifactor_ensemble, cross_sectional_funding_momentum_carry now
   annualize per-bar portfolio vol from observed bar spacing before the
   Moreira-Muir clamp (same pattern as the 14 lanes fixed in PR #46).
5. **information_discreteness 1d/fip_4wk_p33 cell was structurally dead**
   (closes deque 43 < warmup gate 70 → zero signals forever). Deque now sized to
   the min-history floor; the 1d cell will produce its first real measurements.
6. **offsession_tugofwar EST pre-open contamination**: ambiguous-hour set
   {13,20} → {13,14,20} (NYSE open at :30 straddles hour 13 in EDT but 14 in EST).
   Known remaining (documented, unfixed): US holidays (~10 weekdays/yr) still
   count as cash days.
7. **seasonal_xs_persistence gap pollution**: a multi-week data gap booked its
   whole return as one "weekly" observation into a single bucket (probe: 5-week
   gap, 2x move → +0.107 score inflation persisting ~6 quarters). Now gap-guarded
   via grid-adjacency (`prev_week_key`), state-serialized.

## Verified CORRECT (no action)

- Engines do NOT flatten between emissions (HOLD semantics confirmed on all three
  consumption paths) — weekly-cadence lanes do not churn.
- No same-bar-fill look-ahead in core paths (`signal[i] → next_return[i]`).
- VolManagedRiskOverlay: not defective.

## What to run on the data-PC

1. Re-run the full-universe walkforward with the updated tree
   (`configs/profiles/research.yaml` already routes unmapped classes):
   `.venv/bin/python scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py ...`
   (same invocation as the v3 handoff `alpha_pool_evaluability_v3_handoff.md`).
2. In the output rows, group by `evaluation_mode`:
   - `registry_simulator` rows are the FIRST-EVER real measurements of the 33
     research_only sleeves — apply the standard two-tier gate to them.
   - any research_only lane still showing `generic_fallback_proxy` is a wiring bug;
     report it back instead of gating it.
3. Efficiency-repair reruns will show slightly laxer 15/20bps stress gates (fix #2);
   re-select with the same `robust_score_params`.
4. Watch lanes fixed in #4/#5 for changed turnover/exposure profiles (throttles now
   engage in vol storms; the dead 1d ID cell now trades).
