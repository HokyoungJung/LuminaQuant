# Broader redesign + Ralph audit summary — 2026-04-12

## Team attempt
- `implement-the-approved-prd-tes` stalled without producing diffs or mailbox progress and was force-shut down after repeated unblock attempts.
- Work was completed directly in the leader session instead.

## Architect verification
- Final architect verdict: **APPROVE**.

## LuminaQuant implementation
- Added **opt-in scalar RLS adaptive hedge** to the pair strategy.
- Added sparse-fold diagnostics and penalties (`active_fold_ratio`, `inactive_fold_count`, `failed_fold_ratio`).
- Added **broader non-pair / non-threshold method**: `VolatilityRegimeResidualBasketReversionStrategy`.
- Added **portfolio-level novelty**: `build_sparse_fold_aware_ensemble` and builder script comparing incumbent-only vs new-method-only vs combined.
- `train total_return == 0 && train trade_count == 0` is now strongly demoted/rejected in both ranking and followup gates.

## Focused low-memory research runs
- All heavy financial runs remained leader-only and strictly sequential.
- No article batch `01~44` was rerun.

### Adaptive pair follow-up
- Path: `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_adaptive_rls_followup_current/research_run/candidate_research_latest.json`
- Envelope: `POLARS_MAX_THREADS=1`, `LQ_BACKTEST_LOW_MEMORY=1`, `LQ_AUTO_COLLECT_DB=0`
- Memory evidence: `max RSS = 1,137,160 KiB`, wall `21.03s`
- `pair_spread_1h_adaptive_rls_fast_bnbusdt_trxusdt_2.5_0.65`
  - train: return `-11.9632%`, sharpe `-3.5468`, trades `40`, pbo `0.000`
  - val: return `-5.1051%`, sharpe `-2.6379`, trades `28`, pbo `0.000`
  - oos: return `-3.4780%`, sharpe `-4.9902`, trades `12`, pbo `0.000`, active_fold_ratio `0.750`
  - hard reject: `{'oos_sharpe': -4.990214745560854, 'stress_x2_sharpe': -6.134097107464911, 'stress_x3_sharpe': -7.149390213587263}`
- `pair_spread_1h_adaptive_rls_stable_bnbusdt_trxusdt_2.6_0.70`
  - train: return `-8.5611%`, sharpe `-3.1080`, trades `32`, pbo `0.000`
  - val: return `-4.2326%`, sharpe `-2.6086`, trades `16`, pbo `0.000`
  - oos: return `-3.8980%`, sharpe `-5.4545`, trades `8`, pbo `0.000`, active_fold_ratio `0.500`
  - hard reject: `{'oos_sharpe': -5.454510850342113, 'stress_x2_sharpe': -6.191031961593753, 'stress_x3_sharpe': -6.847335149099589}`

### Broader volatility-regime residual follow-up
- Path: `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/volatility_regime_residual_followup_current/research_run/candidate_research_latest.json`
- Envelope: `POLARS_MAX_THREADS=1`, `LQ_BACKTEST_LOW_MEMORY=1`, `LQ_AUTO_COLLECT_DB=0`
- Memory evidence: `max RSS = 1,153,088 KiB`, wall `66.14s`
- `volatility_regime_residual_basket_reversion_15m_volcap_guarded_lo_64_2.00`
  - train: return `-0.1519%`, sharpe `-0.9705`, trades `139`, pbo `0.375`
  - val: return `-2.6143%`, sharpe `-3.6772`, trades `80`, pbo `0.000`
  - oos: return `-0.1237%`, sharpe `-1.4803`, trades `45`, pbo `0.250`, active_fold_ratio `1.000`
  - hard reject: `{'oos_sharpe': -1.4803144857882453, 'stress_x2_sharpe': -2.590576474089705, 'stress_x3_sharpe': -3.6759427976488475}`
- `volatility_regime_residual_basket_reversion_15m_volcap_ls_48_1.80`
  - train: return `-11.2954%`, sharpe `-3.0418`, trades `938`, pbo `0.000`
  - val: return `-1.6728%`, sharpe `-1.5158`, trades `311`, pbo `0.250`
  - oos: return `-2.3631%`, sharpe `-3.2509`, trades `192`, pbo `0.000`, active_fold_ratio `1.000`
  - hard reject: `{'oos_sharpe': -3.2509123282534067, 'stress_x2_sharpe': -5.916494210491676, 'stress_x3_sharpe': -8.502519609083727}`

### Sparse-fold-aware portfolio comparison
- Path: `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/sparse_fold_ensemble_followup_current/sparse_fold_ensemble_followup_latest.json`
- incumbent_only OOS: return `3.3809%`, sharpe `4.8678`
- new_methods_only OOS: return `-0.1248%`, sharpe `-0.2892`
- combined OOS: return `3.3808%`, sharpe `4.8678`
- combined weight `1.0000` -> `pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.5_0.65` (oos_pbo `0.625`, active_fold_ratio `0.000`)
- combined weight `0.0000` -> `pair_spread_1h_adaptive_rls_fast_bnbusdt_trxusdt_2.5_0.65` (oos_pbo `0.000`, active_fold_ratio `0.750`)
- combined weight `0.0000` -> `volatility_regime_residual_basket_reversion_15m_volcap_guarded_lo_64_2.00` (oos_pbo `0.250`, active_fold_ratio `1.000`)

## LuminaQuant conclusion
- The repo now has materially broader method coverage and a portfolio-level combiner under the hardened sparse-fold-aware ranking stack.
- However, both focused new-method runs were still negative, so the sparse-fold-aware ensemble correctly stayed concentrated in the incumbent rather than forcing weak new sleeves into production.

## oh-my-codex / Ralph audit
- Added owner/session provenance to mode-state writes.
- Added a scope-safe stale-state sweep helper.
- Legacy global stale states are auto-demoted, but older live session-scoped states are preserved.
- Session-scoped state queries no longer mutate unrelated session state.

## Verification performed
- LuminaQuant:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q ...` -> **32 passed**
  - `uv run ruff check ...` -> **pass**
  - `uv run python -m compileall ...` -> **pass**
- oh-my-codex:
  - `npm run build` -> **pass**
  - `node --test dist/cli/__tests__/session-scoped-runtime.test.js dist/mcp/__tests__/state-server.test.js dist/modes/__tests__/base-tmux-pane.test.js dist/state/__tests__/mode-state-context.test.js` -> **12 passed**
  - `lsp_diagnostics_directory` on repo -> **0 errors, 0 warnings**
