# Profit Moonshot Alpha Zoo Real-Data Strict Policy Result — 2026-05-14

## Decision

Restored the hard OOS return/MDD promotion gate for `CryptoFxAlphaZooStateStrategy`. The 2026-05-13 relaxed deployability conclusion is superseded. Current strict result is `deployable_success=false` because the train/validation-selected `alpha_zoo_conservative_exit` strict `6.0x` row has OOS return/MDD `3.007073`, below the invalid current-base reference `6.916878`.

## Evidence

- Real data: `var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet` with lagged FRED context.
- Screen: `58,845` rows, `63` factors, `20` selected factor cards.
- Ledger: `67,259` triple-barrier outcomes; train+validation `45,311`; locked-OOS `21,948`.
- Calibration: train/validation records only; locked-OOS calibration records `0`; calibrated edge keys `12`.
- Replay grid: `9` formulaic candidates; selected on train/validation only; locked-OOS hidden until candidate freeze.
- Strict lane: highest zero-liquidation integer `6.0x`; OOS +41.0967%, MDD 13.6667%, return/MDD 3.007073, Sharpe 2.143209, Sortino 2.841936, smart Sortino 2.500237, zero liquidations, min buffer 9049.125962.
- Hard failure: `oos_return_mdd_beats_current_base=false`.
- Diagnostic 5x/6x lane remains non-promotional with `promotion_allowed=false`.
- Peak RSS: `626.7266 MiB` < 8 GiB.

## Artifacts

`var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/`

## Source ledger note

No global research inventory/source-ledger regeneration was required because no new external source family or global source class was introduced.
