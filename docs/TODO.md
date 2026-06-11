# Tracked follow-ups (post-overhaul)

Items deliberately deferred from the 2026-06 overhaul + hardening pass. Each entry
is specified so a future agent can implement it without re-deriving context.

---

## 1. Warmup context end-to-end in the event-driven engine (priority: HIGH before live walk-forward param selection)

**Status:** DONE (2026-06-11) — implemented per the plan below; see the
resolution section in `docs/divergences/walk_forward_warmup_engine_gap.md`.
Implementation summary:

- `Backtest(warmup_bars=N)`: data handler loads from
  `start_date - warmup_bars x strategy_timeframe`; `start_date` stays the
  trading boundary. Raises `InsufficientWarmupError` when the loaded data
  provides fewer than `warmup_bars` strategy-timeframe buckets of context.
  Skip-ahead is disabled inside the warmup region (it would drop warmup bars).
- `TradingEngine(live_start_ms=...)`: warmup events still reach
  `strategy.calculate_signals` (and the timeframe aggregator), but
  `update_timeindex` / `check_open_orders` / signal→order→fill handling are
  suppressed — per-bar inside `handle_market_window_event` for straddling
  windows. Warmup boundary uses naive-UTC bar-time semantics (matches
  `HistoricCSVDataHandler._bar_time_ms`, NOT local-tz `datetime.timestamp()`).
- `run_backtest_chunked(warmup_bars=N)`: warmup attaches to the first
  non-empty chunk only; the runner extends that chunk's `data_loader` call
  backwards by the warmup span (`cli/optimize.py`'s `_chunk_loader` therefore
  passes `warmup_bars=0` to the repo — single point of extension).
- `cli/optimize.py`: `BT_CHUNK_WARMUP_BARS` honoured in BOTH fold-eval paths
  (chunked parquet + in-memory slice); fold windows are filtered against
  `data_start + warmup_span` so every fold has full context;
  `InsufficientWarmupError` propagates; `-999.0` sentinel kept ONLY for
  no-data windows and `ValueError` infeasible-param combos (`"infeasible":
  True` row) — all other exceptions now propagate.
- Acceptance tests: `tests/test_engine_warmup.py`,
  `tests/test_optimize_warmup_sentinels.py`.

Original analysis and plan kept below for context.

**Previous status:** documented gap — see `docs/divergences/walk_forward_warmup_engine_gap.md`.
The Variant B golden (`baseline/golden/walk_forward_results_warmup.json`) is a
**numpy reference oracle only**; the production optimizer evaluates every
fold/chunk with cold indicators.

**Why it matters (assessed 2026-06-11):**
- **Walk-forward parameter-selection bias:** each fold's first `indicator_lookback`
  bars produce no/partial signals, so long-lookback combos are systematically
  penalized vs short-lookback combos (e.g. MA(120) in a 90-bar val window emits
  ZERO signals). Since walk-forward output selects real-money parameters, the
  selection process itself is biased until this is fixed.
- **Sim-to-live mismatch at window starts:** live trading runs continuously
  (indicators always warm); fold backtests cold-start. The mismatch is
  concentrated in each window's first bars and grows as folds get shorter.
- **Regression blind spot:** Variant B golden does not constrain the engine, so
  early-window engine regressions are not caught by the rtol gate.

**Not affected:** live trading itself; long single backtests (first-chunk-only,
amortized); current goldens/gates (consistent with documented behavior).

**Interim mitigation:** keep eval windows ≥ 5–10× the max indicator lookback.

**Implementation plan (scouted against the code, 2026-06-11):**
1. `Backtest.__init__`: add `warmup_bars: int = 0`. Keep `start_date` as the
   trading boundary (`live_start`); compute `effective_load_start = start_date −
   warmup_bars × strategy_timeframe`. Default 0 ⇒ bit-identical current behavior.
2. `compute/ohlcv_loader.py` (`OHLCVFrameLoader`): filter at
   `effective_load_start`, not `start_date` (today the `datetime >= start_date`
   filter drops warmup rows before the strategy sees them).
3. Engine suppression (`core/engine.py`): during warmup
   (`event_time < live_start`) still call `strategy.calculate_signals(event)` so
   stateful incremental indicators advance and the data handler accumulates
   history, but suppress `portfolio.update_timeindex`, `check_open_orders`, and
   early-return in `handle_signal_event`/`handle_order_event`/`handle_fill_event`
   — no trades, no equity rows in the warmup region. **Crux:** the windowed
   optimizer path delivers `MARKET_WINDOW` events that carry many bars — the
   suppression must be applied PER-BAR inside `handle_market_window_event`
   (and `handle_market_batch_event`), not per-event.
4. `backtesting/chunked_runner.py`: thread `warmup_bars` into the FIRST chunk's
   `Backtest` only (later chunks stay warm via the existing carry-over); verify
   the chunk `data_loader` actually loads from `chunk_start − warmup`.
5. `cli/optimize.py`: honour `BT_CHUNK_WARMUP_BARS` end-to-end (today it is dead
   plumbing defaulting to 0); wire `check_warmup_sufficient` /
   `InsufficientWarmupError` into the fold evaluator; and SEPARATE the `-999.0`
   sentinel semantics: legitimate infeasible-combo/no-data pruning may keep the
   worst-score sentinel for optuna, but unexpected exceptions must propagate
   (today `except Exception` swallows real bugs as `-999.0`).
6. Equity-curve baseline: first live bar must mark from initial capital
   (positions are flat during warmup since orders are suppressed), so skipping
   `update_timeindex` during warmup must not corrupt the first live equity row.

**Acceptance criteria (falsifiable):**
- With `warmup_bars ≥ lookback`, the strategy's indicator state at the first
  in-window bar equals the state from a run fed the full prior history
  (assert via a stateful indicator probe, e.g. IncrementalRsi value).
- Zero orders/fills/equity rows timestamped before `start_date`.
- `warmup_bars=0` ⇒ byte-identical results to today (golden gate stays green).
- Note: the engine will NOT reproduce Variant B at rtol 1e-8 (the oracle is
  costless signal×returns; the engine applies ExecutionModel costs) — the test
  asserts warmup SEMANTICS (above), not oracle-value equality.
- `InsufficientWarmupError` raised (not -999) when requested warmup data is
  unavailable; unexpected exceptions in fold evaluation propagate.
- Full suite + ruff + golden rtol gate + CI green.

---

## 1a. Warmup operational hardening for gappy real data (priority: medium, before unattended live walk-forward)

Follow-ups from the item-1 review (mechanism is correct; these are blast-radius
controls for long unattended runs):

- `main()` shifts fold windows by *calendar* warmup span, but
  `Backtest._assert_warmup_available` counts *loaded* strategy-tf buckets — a
  data gap inside the warmup region of a calendar-valid fold raises
  `InsufficientWarmupError` and aborts the whole run. Decide per-fold policy:
  catch at the fold level, log loudly, and skip the fold (never -999).
- Non-`ValueError` exceptions now abort the entire study/grid (intended loud
  contract). If unattended runs need resilience, pass a curated `catch=` set to
  `study.optimize` rather than re-widening `_execute_backtest`.
- Optional strictness: exclude the partial bucket straddling `live_start` from
  the warmup-availability count when base data is finer than the strategy tf.

## 2. Vectorized bar engine (priority: low — current throughput ~6.4k bars/sec suffices)

See `docs/perf/phase4-results.md` ("future work"). Only worth it for very large
parameter grids; measure first.

## 3. God-module decomposition seams (priority: low)

See AGENTS.md "Monolithic-module disposition" — `data_sync.py`, `trader.py`,
`cutover_surfaces_service.py` decomposition seams are documented there; extract
only along those seams.
