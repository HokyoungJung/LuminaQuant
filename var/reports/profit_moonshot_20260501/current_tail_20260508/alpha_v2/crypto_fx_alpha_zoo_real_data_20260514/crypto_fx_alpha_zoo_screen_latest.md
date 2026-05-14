# Crypto/FX Alpha Zoo real-data screen

- factor_count: `63`
- row_count: `58845`
- source_path: `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet`
- selection_policy: `train_validation_only_locked_oos_report_only`
- uses_locked_oos_for_selection: `False`
- calendar_primary: `False`
- strategy_validity_pass: `True`
- ledger_records: `67259`

## Source coverage
- `BNB/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `BTC/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `ETH/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `SOL/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`
- `TRX/USDT` rows `11769` observed `close,funding_rate,high,liquidation_long_notional,liquidation_short_notional,low,open,open_interest,taker_buy_base_volume,taker_buy_quote_volume,taker_sell_base_volume,taker_sell_quote_volume,volume` imputed `vwap` required_ohlcv_observed `True`

## Selected factors
- `ret_24` score `0.04863241095607805`
- `ret_48` score `0.031163198055628366`
- `btc_residual_ret_24` score `0.029240275519924183`
- `range_position_24` score `0.027646934133252112`
- `range_position_48` score `0.025435257820772616`
- `btc_residual_ret_48` score `0.023679904358334027`
- `kup` score `0.02351458599277505`
- `mom_z_24` score `0.02140614321331276`
- `volume_shock_z_24` score `0.01767730036033069`
- `volume_shock_z_48` score `0.0174968839572537`
- `btc_residual_z_48` score `0.014907642038503097`
- `crypto_leadership_rank_24` score `0.014827996055154518`
- `crypto_leadership_rank_48` score `0.012813233741609234`
- `breakout_failure_48` score `0.011287495908385703`
- `klen` score `0.009884554767328788`
- `range_position_2` score `0.008866806222611748`
- `ret_2` score `0.008623285347581555`
- `volume_vwap_pressure_24` score `0.008395636761265918`
- `btc_residual_z_24` score `0.007906960909058675`
- `mom_z_48` score `0.007886285662330204`
