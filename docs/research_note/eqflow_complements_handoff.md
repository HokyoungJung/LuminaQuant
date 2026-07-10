# Equity-Flow Complements (eq-flow v5 follow-up) — Data-PC Handoff (2026-07-10)

Five ADDITIVE walk-forward variants / ranking treatments designed from the
fold-by-fold equity-flow diagnosis of the promising candidates
(`.omc/research/equity-flow-analysis-20260710.md`; run right after the v5
systematic-defect fixes in `systematic_defect_fix_v5_handoff.md`).

Everything is config-gated OFF by default (byte-identical legacy output; the
72-row dynamic-conviction / 5-row router count pins hold) and turned ON in
`configs/profiles/research.yaml` (`research.walkforward_*` flags), or per-flag
via `--eqflow-*` CLI switches on
`scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`.
Variant parameters are PRE-REGISTERED module constants — reused verbatim from
the registered overlay schemas where applicable — not tunable grids.

## The flow diagnosis these attack (one line each)

- The lead-lag router champion (+46–63% comp OOS, the only distributed alpha
  on the board) has ONE common tail: the 2026-07 whipsaw fold, a
  high-cross-correlation event (-8.6%). Its idiosyncratic wins are
  low-correlation.
- dynamic_conviction_switch ("clean #1", +7.99%) is a 1-fold artifact with a
  pro-cyclical cliff: it levers MOST (val +180.5%) exactly into its worst OOS
  fold (-16.71%), a levered entry right after the June run-up.
- The terminal 2026-07 fold is a 4-DAY PARTIAL fold that nevertheless sets
  headline MDD / latest-OOS in the rankings.
- strict_efficiency:static_guarded is an always-on shallow bleeder (two wins
  eaten by eight small losses); the conviction family already demonstrates
  that gating it to up-vol validation regimes removes the bleed.
- Momentum families are UN-diversified from each other (same 2025-11 win, same
  2026-07 loss); the only real fold-complementarity is momentum <-> low-beta
  carry (tradfi_intraday, dense_pairs state_vwap).

## What ships

| # | Flag (`research.*`) | New output | Mechanism |
|---|---|---|---|
| C1 | `walkforward_corr_guard_router_variant` | `codex_lagged_leaf_router_grid:..._fallback_mdd20_cap2_eqcorr_guard` | Engle-Kelly avg-pairwise-correlation of the symbol panel on the PRE-OOS window; z >= 1.5 AND rho >= 0.35 → fold deployment x0.35 (registered overlay band, verbatim) |
| B2 | `walkforward_crash_scaled_conviction_variant` | `dynamic_conviction_switch:..._dm_crash_scaled` (mdd20/mdd15 calmar80 cells) | Daniel-Moskowitz bear x rebound state on the selected stream's PRE-OOS window → OOS x {1.0, 0.5, 0.0} |
| A3a | `walkforward_val_saturation_conviction_variant` | `dynamic_conviction_switch:..._val_sat80` (same cells) | validation return reflected past a 0.80/window (~40%/mo) ceiling before conviction/scale scoring — extreme validation now REDUCES deploy scale |
| A3b | `walkforward_partial_fold_bar_count_weighting` | `bar_weighted_*` aggregate fields + bar-weighted sort | fold weight = min(1, bar_count / (0.75 x max)); a 4-day partial fold cannot set headline compounded/MDD ranks |
| D4 | `walkforward_regime_gated_strict_efficiency_variant` | `strict_efficiency:static_guarded_regime_gated` | deploy the static blend only when its own validation window shows return > 0 AND mdd <= 0.12; else cash for the fold |
| F1 | (no flag — committed data) | `configs/research/allocation_cells/eqflow_momentum_lowbeta_composite_cell.json` | pre-registered 3-sleeve M2 cell: LL mdd20_cap2 (momentum) + tradfi_intraday open_impulse (tradfi_carry) + state_vwap pair (pair_carry), ERC, min_families=3 |

## What to run on the data-PC

1. Re-run the monthly-refit walkforward with the research profile (flags already
   ON there) — same invocation as `systematic_defect_fix_v5_handoff.md`, which
   MUST be absorbed first (the v5 proxy fix invalidates all prior research_only
   rejects).
2. Read the new labels' rows next to their bases:
   - C1 vs `..._fallback_mdd20_cap2`: the claim is 2026-07-shaped folds trimmed
     (guard engaged) with Jan/Feb/Jun-shaped folds untouched (guard neutral).
     Audit `corr_guard_engaged` / `corr_guard_rho` / `corr_guard_z` per fold: if
     the guard engages in the idiosyncratic win folds, the band is wrong —
     report back, do not tune in place.
   - B2/A3a vs their base cells: expect the terminal-fold loss cut (mu < 1 /
     saturated scale) at the cost of some upside; judge on return/MaxMDD and
     fold-count breadth, not compounded alone.
   - D4 vs `static_guarded`: expect the Sep/Dec/Mar/Apr bleed folds gated to
     cash; the two win folds (Nov/Feb) must survive the gate.
3. Rankings: compare `bar_weighted_compounded_oos_return` ordering vs raw — any
   candidate whose headline rank depended on the 4-day partial fold will move.
4. F1: materialize the three member streams (walkforward artifact for the two
   labels; dense-pairs artifact for state_vwap), fill `returns`/`turnover` in a
   COPY of the cell JSON, run `scripts/research/build_quality_gated_allocation.py`
   on it, and evaluate the composite on the standard two-tier gate. Membership
   is pre-registered; only the ERC weights are measured.

## Guardrails / honesty notes

- All variants are de-risk-only or selection-layer: none can lever a base
  candidate beyond its own intent, so none can manufacture a Sharpe lift — the
  thesis is drawdown/Calmar variance reduction and ranking de-noising.
- OFF-path is byte-identical by construction and pinned by tests
  (`tests/test_eqflow_*.py` + the existing count pins).
- These are research_only measurements; nothing here implies promotion. The
  D5 universe-survivorship re-run (85 vs 110 symbol June divergence) remains
  the top open data-PC experiment before trusting the lead-lag headline.
