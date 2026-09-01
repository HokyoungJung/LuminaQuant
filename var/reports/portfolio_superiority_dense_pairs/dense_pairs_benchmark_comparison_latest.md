# Dense Pairs Benchmark Comparison

- generated_at: `2026-04-22T12:25:46.405817+00:00`
- current_default: `production_guarded_mode`

## Standalone challengers
- new pair `pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.2_0.55` -> OOS `+1.2611%` / Sharpe `3.1892` / MaxDD `0.4038%` | train `-6.9727%` / `-0.7193`
- old pair `pair_spread_1h_robust_core_strict_bnbusdt_trxusdt_2.6_0.70` -> OOS `+0.2892%` / Sharpe `3.2765` / MaxDD `0.0000%`
- dense_pairs optimized portfolio -> OOS `+0.3252%` / Sharpe `0.8127` / MaxDD `0.9273%`

## Baselines
- `production_guarded` -> OOS `+0.3654%` / Sharpe `1.6080` / MaxDD `0.3184%`
- `hybrid_guarded` -> OOS `+0.1618%` / Sharpe `0.7763` / MaxDD `0.4897%`
- `soft_three_way` -> OOS `+0.0887%` / Sharpe `0.7423` / MaxDD `0.5734%`

## New pair overlay frontier best rows
- `production_guarded` + new pair best Sharpe row = base `25.0%` / pair `30.0%` / cash `45.0%` -> OOS `+0.4705%` / Sharpe `3.5958` / MaxDD `0.1897%` | beats_base return=`True` sharpe=`True` dd=`True`
- `hybrid_guarded` + new pair best Sharpe row = base `10.0%` / pair `30.0%` / cash `60.0%` -> OOS `+0.3948%` / Sharpe `3.2274` / MaxDD `0.1591%` | beats_base return=`True` sharpe=`True` dd=`True`
- `soft_three_way` + new pair best Sharpe row = base `5.0%` / pair `30.0%` / cash `65.0%` -> OOS `+0.3830%` / Sharpe `3.2018` / MaxDD `0.1388%` | beats_base return=`True` sharpe=`True` dd=`True`
