# Alpha Zoo asset-diverse strategy discovery

Generated: `2026-05-23T12:42:18.947171Z`

Research/paper-testnet only. Single-symbol state rules use cross-asset filters;
locked-OOS remains gate/report-only after train+validation ranking freeze.

- Candidates evaluated: `97560`
- Asset groups: `{'crypto_exchange_beta': 7020, 'crypto_high_beta_alt': 35100, 'crypto_major': 13320, 'crypto_payment_alt': 14040, 'precious_metal_proxy': 28080}`
- Paper candidate gate pass: `4`
- `ready_for_real=false`
- `real_money_execution=false`
- Runner peak RSS MiB: `1972.680`

## Top train+validation-ranked rows

| Rank | Symbol | Group | TF | Family | Train | Val | OOS | RPT train/val/OOS | Decision |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.4386% | 32.0985% | 0.0000% | 14.10/65.51/NA | no_promotion_shadow_or_reject |
| 2 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.4386% | 32.0985% | 0.0000% | 14.10/65.51/NA | no_promotion_shadow_or_reject |
| 3 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.4386% | 32.0985% | 0.0000% | 14.10/65.51/NA | no_promotion_shadow_or_reject |
| 4 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.4386% | 32.0985% | 0.0000% | 14.10/65.51/NA | no_promotion_shadow_or_reject |
| 5 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 41.1040% | 31.5558% | 0.0000% | 15.90/64.40/NA | no_promotion_shadow_or_reject |
| 6 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 41.1040% | 31.5558% | 0.0000% | 15.90/64.40/NA | no_promotion_shadow_or_reject |
| 7 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 41.1040% | 31.5558% | 0.0000% | 15.90/64.40/NA | no_promotion_shadow_or_reject |
| 8 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 41.1040% | 31.5558% | 0.0000% | 15.90/64.40/NA | no_promotion_shadow_or_reject |
| 9 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.8071% | 30.6467% | 0.0000% | 14.29/62.54/NA | no_promotion_shadow_or_reject |
| 10 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.8071% | 30.6467% | 0.0000% | 14.29/62.54/NA | no_promotion_shadow_or_reject |
| 11 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.8071% | 30.6467% | 0.0000% | 14.29/62.54/NA | no_promotion_shadow_or_reject |
| 12 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 36.8071% | 30.6467% | 0.0000% | 14.29/62.54/NA | no_promotion_shadow_or_reject |
| 13 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 43.2802% | 30.1099% | 0.0000% | 16.81/61.45/NA | no_promotion_shadow_or_reject |
| 14 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 43.2802% | 30.1099% | 0.0000% | 16.81/61.45/NA | no_promotion_shadow_or_reject |
| 15 | XRPUSDT | crypto_payment_alt | 1h | cross_asset_rank_chandelier_breakout | 43.2802% | 30.1099% | 0.0000% | 16.81/61.45/NA | no_promotion_shadow_or_reject |

## Paper/testnet-only candidates

| Rank | Model | Symbol | Group | TF | Family | Train | Val | OOS |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 4581 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5` | ETHUSDT | crypto_major | 2h | relative_residual_reclaim | 16.8301% | 4.7367% | 4.8120% |
| 4582 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_5p0x_0p1_cf067261` | ETHUSDT | crypto_major | 2h | relative_residual_reclaim | 16.8301% | 4.7367% | 4.8120% |
| 5579 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_2p0x_0p15_1a9aa250` | ETHUSDT | crypto_major | 2h | relative_residual_reclaim | 10.1047% | 2.8751% | 2.8746% |
| 5580 | `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_3p0x_0p1_06295c43` | ETHUSDT | crypto_major | 2h | relative_residual_reclaim | 10.1047% | 2.8751% | 2.8746% |
