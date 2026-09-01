# Portfolio Superiority Overlay Decision

- verdict: `better_but_not_promotable_yet`
- current_default: `production_guarded_mode`

## Why
- Some requested overlays improve one axis (return or Sharpe or drawdown), but none beat their base on both OOS return and OOS Sharpe.
- aligned wave2 pair OOS: `+0.1181%` / Sharpe `0.2376` / MaxDD `1.4112%`

## Requested overlay scoreboard
- `production_guarded` standalone-bar rows=`15` challenger rows=`0`
- `hybrid_guarded` standalone-bar rows=`0` challenger rows=`0`
- `soft_three_way` standalone-bar rows=`0` challenger rows=`0`

## Requested 10%~30% overlay highlights
- `production_guarded` best requested Sharpe row = base `90.0%` / pair `10.0%` / cash `0.0%` -> OOS `+0.3424%` / Sharpe `1.6145` / MaxDD `0.3232%`
- `hybrid_guarded` best requested Sharpe row = base `90.0%` / pair `10.0%` / cash `0.0%` -> OOS `+0.1591%` / Sharpe `0.7977` / MaxDD `0.4774%`
- `soft_three_way` best requested Sharpe row = base `50.0%` / pair `15.0%` / cash `35.0%` -> OOS `+0.0651%` / Sharpe `0.4264` / MaxDD `0.3419%`

## Executed broad follow-up after overlay failure
- resliced wave2 survivor count on the current baseline contract: `1`
- `pair_spread_1h_exec_tightstop_tp_btcusdt_xauusdt_2.2_0.55` | val `-1.5507%` / `-1.3437` | oos `+0.5245%` / `0.8835`
- `pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.2_0.55` | val `+9.3886%` / `3.6691` | oos `+0.1181%` / `0.2376`

## Next broad wave
- lane: `metals_crypto_relative_value_dense_pairs`
- The aligned current-window overlay lane failed; the BNB/TRX pair no longer dominates once February-only gains are removed.
- The resliced broader wave2 leaderboard shows BTC/XAU pair relative value has the best short aligned OOS return, but its train/val are still negative.
- The next heavy sweep should therefore focus on denser metals/crypto pair variants and participation-aware donor filters, not on enlarging the current overlay cap.
