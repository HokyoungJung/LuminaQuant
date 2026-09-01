# Profit moonshot precious-metal pair aggressive alpha

Generated: `2026-05-05T13:22:24.517807+00:00`
Decision: **rejected / not deployment-ready**.

## What was built

- New `TimeframePairZScoreReversionStrategy`: completed-1h hedge-adjusted pair z-score reversion with explicit sizing metadata, max-order caps, stops/take-profit/max-hold, correlation/beta checks, and current-volume entry guards.
- New mode `profit_moonshot_precious_metal_pair_aggressive_mode` includes all requested metals: XAU/XAG primary sleeve and XPT/XPD secondary sleeve.
- Existing `PairTradingZScoreStrategy` now emits explicit `target_allocation` / `max_order_value` metadata so profit modes do not rely on hidden unbounded-child fallback sizing.

## External rationale

- CME documents gold/silver ratio and metals spread trading as established relative-value constructs, but also notes gold and silver have distinct safe-haven vs industrial drivers, so the relationship is dynamic rather than mechanically mean-reverting.
- CME also launched exchange-traded Gold/Silver Ratio, Gold/Platinum Spread, and Platinum/Palladium Spread contracts, supporting the alpha-family choice.
- Mean-reversion literature emphasizes transaction costs; this is why the engine result, not the vector screen, is the promotion gate.

## Evidence

### Research screen (not promotion evidence)
- XAU/XAG selected screen: train `+0.4386%`, val `+1.5459%`, OOS `+4.2731%`, OOS Sharpe `16.1551`, trades `10`.
- XPT/XPD selected screen: train `-4.1132%`, val `+3.1279%`, OOS `+3.0109%`, OOS Sharpe `7.8471`, trades `32`.

### Live-equivalent/raw-first gate
- status: `blocked_missing_raw_first_market_data`
- blockers: `XAU/USDT:train_raw_first_incomplete, XAU/USDT:val_raw_first_incomplete, XAG/USDT:train_raw_first_incomplete, XAG/USDT:val_raw_first_incomplete, XPT/USDT:train_raw_first_incomplete, XPT/USDT:val_raw_first_incomplete, XPD/USDT:train_raw_first_incomplete, XPD/USDT:val_raw_first_incomplete`

### Legacy-windowed engine split (single mode, all four metals)
- train: return `-0.0570%`, MDD `0.2624%`, Sharpe `-0.024200`, trades `70`, liquidations `0`.
- val: return `+0.1914%`, MDD `0.4715%`, Sharpe `0.037224`, trades `70`, liquidations `0`.
- oos: return `-0.0478%`, MDD `0.0528%`, Sharpe `-0.164066`, trades `13`, liquidations `0`.

## Tail refresh

- Tail refresh was attempted for XAU/XAG/XPT/XPD from 2026-03-28 through 2026-05-05 with one worker and <7GB soft RSS.
- Observed peak RSS: `1537.184 MiB`; XAU raw archive reached `date=2026-05-04`; the process was killed after ~28 minutes with no further progress, so XAG/XPT/XPD tails remain incomplete.

## Final decision

- Do **not** promote this mode. It is useful code + negative evidence, not a win.
- Current OOS-return best remains `profit_moonshot_hourly_shock_reversion_eth_12h_mode`; Sharpe/MDD shadow remains `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`; conservative legacy candidate remains `profit_moonshot_momentum_hybrid_safe_mode`.

## Verification

- Targeted pytest: `8 passed in 0.31s`.
- Ruff on changed strategy/live-selection/tests: `All checks passed`.
- `python -m compileall -q src tests`: passed.
- Live-equivalent standard gate executed and blocked on raw-first metals train/val coverage.
- Available-window split backtest peak RSS: `1019.396 MiB`; no duplicate backtest process left running.
