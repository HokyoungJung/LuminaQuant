# Crypto/FX Alpha Zoo real-data screen

- factor_count: `63`
- row_count: `58845`
- source_path: `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet`
- selection_policy: `train_validation_only_locked_oos_report_only`
- uses_locked_oos_for_selection: `False`
- calendar_primary: `False`
- strategy_validity_pass: `True`
- ledger_records: `45160`

## Source coverage
- `BNB/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `BTC/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `ETH/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `SOL/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `TRX/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`

## Selected factors
- `btc_residual_ret_48` score `0.055211920686826156`
- `ret_48` score `0.0392254965169943`
- `ret_12` score `0.03891193946966292`
- `ret_24` score `0.03467169999305067`
- `btc_residual_ret_24` score `0.03417818351609867`
- `ret_8` score `0.02906146109237773`
- `range_position_24` score `0.028413420806017473`
- `range_position_48` score `0.02766611615101623`
- `range_position_12` score `0.02289153851025238`
- `btc_residual_ret_12` score `0.021831903960780052`
- `crypto_leadership_rank_48` score `0.020808456928474193`
- `ret_2` score `0.016202386388177826`
- `ret_4` score `0.015392494868434305`
- `range_position_8` score `0.014096108266559992`
- `breakout_failure_48` score `0.013284208599777596`
- `kmid` score `0.013145391386184146`
- `btc_residual_z_48` score `0.011220448650335191`
- `mom_z_8` score `0.009116453111102178`
- `mom_z_24` score `0.008796967409898717`
- `range_position_4` score `0.008461324305350095`
