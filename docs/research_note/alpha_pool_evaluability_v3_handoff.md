# Alpha Pool Evaluability v3 — Data-PC Handoff (2026-07-10)

Response to the 2026-07-09 cost-grid rounds where **169/175 candidate rows closed as
`insufficient_data`** (local parquet coverage is 1h/4h; every 1d-only row was never
measured) and only 6 rows (N4@4h, off-session@1h) were actually evaluated. This wave
makes the whole pool measurable on the coverage that exists, adds a coverage
dry-run tool, and locks the postmortem of the two measured failures.

## 1. What changed

### 4h/1h slice cells on every lane (27 modules)
Every previously-1d-only lane now carries `"4h"` and `"1h"` slice cells mirroring its
1d variants: bar-denominated windows scale x6/x24 (wall-clock preserved),
decision-denominated counters and timestamp clocks (ISO-week/quarter, per-UTC-day)
unchanged, episodic lanes carry stricter sub-daily entry thresholds so episode
frequency does not scale with bar count. Candidate builders were widened to
`_present("1h", "4h", "1d")`.

**Verification on a 7-symbol 1h/4h-ONLY universe (your coverage reality): all 29
lane families materialize — 791 candidate rows, zero inert.** Per-lane judgment
calls (price-delay lag order, ID sign-census granularity, vol-shock z bump at 1h)
are documented in the module docstrings/slice comments.

### Coverage dry-run tool — RUN THIS FIRST
```
.venv/bin/python scripts/research/report_data_coverage.py --timeframes 1h 4h 1d --json coverage.json
.venv/bin/python scripts/research/report_data_coverage.py --check-manifest <your_manifest>.json
```
`--check-manifest` mirrors research_runner's cache-miss semantics and prints, per
candidate row, evaluable/missing_symbols BEFORE the grid — no more blind
insufficient_data rounds. Include `coverage.json` and the check-manifest summary
in the result report.

### Postmortem of the two measured failures (regression-locked)
Verdict for BOTH `stationarity_gated_residual_reversion_4h_strict_adf` (Sharpe
−18.41) and `offsession_tugofwar_1h_tow_42d_fade` (Sharpe −33.95): **GENUINE NULL
— no sign bug, no churn, no leverage inflation.** Sign paths are theory-faithful
and now regression-locked (a silent sign flip fails
`test_pm_sign_is_reversion_buy_the_loser_sell_the_winner` /
`test_pm_fade_sign_shorts_high_tow_longs_low_tow`); decision clocks proven weekly;
turnover cost ~0.1–0.3%/wk explains <10% of the loss.

**do_not_rerun_as_is** for these two exact configurations. FORBIDDEN: "fixing" by
flipping the reversion/fade sign (curve-fitting one OOS cell; blocked by the
regression tests). Revival preconditions (documented, not applied): off-session —
fix the inert vol-target horizon mismatch (per-1h-bar vol vs annualized 0.20) +
add a benchmark hedge + widen the universe; N4 — its cadence clock is bar-count
keyed (rebalance_bars assumes fixed bar size), pin the feed cadence. Neither
rescues the sign.

## 2. Re-run instructions (same contract as v2/v2b/v2c)
1. `report_data_coverage.py` first; attach output.
2. **UNIVERSE MANDATE — use the FULL research universe, not a hand-picked subset:**
   build the manifest with `BINANCE_EXTENDED_RESEARCH_SYMBOLS` (110 crypto
   symbols; add the TradFi perp set for the off-session lane) INTERSECTED with
   actual coverage from step 1, and REPORT the resulting universe size alongside
   the results. Cross-sectional lanes take the whole basket per row — their
   statistical power scales with universe breadth, and thin books were flagged as
   loss amplifiers in the postmortem. Two former in-lane caps are widened this
   wave (long-run `max_universe` 12→128; slow-leadlag book 3/3→10/10); no other
   lane caps its cross-section by fixed count (quantile-based books scale
   automatically).
3. Build the manifest at the timeframes your coverage supports (~791 rows on
   1h/4h with 7 symbols; substantially more with the full universe). Dry-run
   with `--check-manifest`; report the evaluable count BEFORE running.
4. Evaluate on `backtest_cost_realistic.yaml`, cost grid 10/15/20/30bps,
   strict gate (DSR 0.90 / SPA 0.05 / PBO 0.50), `emit_candidate_overfit_stats=ON`,
   admit+merge through `selection.apply_selection_reject_and_dedup`.
5. Two-tier gate unchanged (pool admission relaxed / promotion (a)-(f) strict).
6. NO-SURVIVORSHIP: report ALL rows, including insufficient_data rows WITH their
   `metadata.missing_symbols` so the next iteration can target coverage gaps.
7. Allocator 3-arm (base M2 / MR1 tilt / family tilt) re-run once ≥2 sleeve
   families produce non-empty streams — last round had only 2 streams, which is
   below the min_families=3 floor.
8. 0% real allocation; research_only; no execution paths.

Everything is byte-identical when the new cells are not requested: the 1d
candidate set is unchanged, defaults untouched, and the manifest snapshot was
re-pinned once (165→209 in the 2-symbol 1h/4h fixture) to admit the new cells.

## 3. Sizing-discipline addendum (v4, same branch cycle)

The postmortem's inert-vol-target finding generalized: a repo-wide audit found
the annualized-scale `target_vol` (~0.20) compared against PER-BAR vol
estimates across most sleeves — throttles pinned at their clamps (including a
2x-ceiling pin in price_volume) at every timeframe. **Fixed in 14 throttle
sites across the pool** (the offsession flagship included; its postmortem
diagnostic now asserts the FIXED behavior): per-bar vol is annualized via
`sqrt(bars_per_year)` inferred from observed bar spacing (canonical helpers in
`indicators/annualization.py`, 365.25-day year), unknown cadence falls through
to a unity scalar (never inflates). Normalized inverse-vol WEIGHTS were
verified horizon-free (annualization cancels) and left untouched, as were the
two per-bar-consistent overlays and `VolManagedRiskOverlay` (explicit per-bar
target — not defective).

**Impact on your grid: MDD/Calmar/vol of every cell now reflect an ACTIVE
de-risk throttle** (at 1d too — correct Moreira-Muir behavior). Also emitted
when `emit_candidate_overfit_stats=ON`: per-candidate `rpt_bps` (activates
automatically once a turnover series/scalar is threaded onto the candidate
row — see the runner's `_candidate_rpt_bps`), automating the RPT>=10bps gate.
After the grid, run `scripts/research/analyze_candidate_grid.py <report-dir>`
for the two-tier classification, cross-cost stability flags and the
second-generation shortlist.
