# Alpha Zoo debounced efficiency repair discovery

Generated: `2026-05-23T02:52:54.063760Z`

Focused ETH/SOL debounced momentum hysteresis repair pass.
Locked-OOS is gate/report-only after train+validation candidate freeze.
Real-money remains blocked.

## Summary

- Candidates evaluated: `36000`
- Rows with train >= validation: `14465`
- Train-dominant sample gate pass: `274`
- Execution-efficiency proxy gate pass: `954`
- Full paper candidate gate pass: `82`
- Decision: `paper_testnet_candidate_after_fill_preflight`
- `ready_for_real=false`, `real_money_execution=false`

## Top train+validation-ranked rows

| Rank | Symbol | TF | Side | Rule | Train | Val | OOS | RPT train/val/OOS | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | SOLUSDT | 1h | long_short | lb12 e0.02 x0.005 hold48 cool0 none | 78.5282% | 33.6023% | -3.0944% | 58.56/108.22/-22.18 | no_promotion_shadow_or_reject |
| 2 | SOLUSDT | 1h | long_short | lb12 e0.025 x0.0 hold36 cool0 none | 48.2810% | 29.6453% | -8.1090% | 32.51/83.39/-48.70 | no_promotion_shadow_or_reject |
| 3 | SOLUSDT | 1h | long_short | lb6 e0.03 x-0.005 hold36 cool0 none | 30.9170% | 28.5599% | -1.5297% | 25.26/97.64/-17.89 | no_promotion_shadow_or_reject |
| 4 | SOLUSDT | 1h | long_short | lb12 e0.02 x0.005 hold48 cool0 none | 68.5414% | 29.5202% | -2.7344% | 57.50/106.96/-22.05 | no_promotion_shadow_or_reject |
| 5 | SOLUSDT | 1h | long_short | lb6 e0.02 x0.005 hold48 cool6 none | 57.3567% | 27.8825% | -7.9133% | 46.52/95.32/-56.73 | no_promotion_shadow_or_reject |
| 6 | SOLUSDT | 1h | long_short | lb6 e0.02 x0.005 hold48 cool6 adx20 | 60.6561% | 27.8100% | -7.9133% | 49.19/95.08/-56.73 | no_promotion_shadow_or_reject |
| 7 | ETHUSDT | 2h | long_short | lb6 e0.03 x-0.005 hold18 cool0 adx20 | 38.8250% | 27.0944% | -9.8883% | 35.95/98.70/-99.88 | no_promotion_shadow_or_reject |
| 8 | SOLUSDT | 2h | long_short | lb6 e0.02 x0.005 hold18 cool0 trend_strength2 | 49.6543% | 27.5194% | -4.4413% | 39.69/105.44/-39.48 | no_promotion_shadow_or_reject |
| 9 | SOLUSDT | 1h | long_short | lb12 e0.02 x0.005 hold36 cool6 none | 38.7977% | 27.6219% | -11.5057% | 25.51/75.78/-73.05 | no_promotion_shadow_or_reject |
| 10 | SOLUSDT | 1h | long_short | lb12 e0.025 x0.0 hold36 cool0 adx20 | 67.4877% | 26.8519% | -8.1090% | 45.72/77.49/-48.70 | no_promotion_shadow_or_reject |

## Paper/testnet-only candidates

These rows pass strict train-dominant, sample, locked-OOS report, liquidation/wipeout, and return-per-turnover proxy gates. They remain `ready_for_real=false` and require paper/testnet preflight plus monitoring before any forward observation.

| Rank | Symbol | TF | Side | Train | Val | OOS | Trades | RPT train/val/OOS |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 180 | SOLUSDT | 1h | short_only | 28.2198% | 16.9294% | 2.4704% | 254/63/26 | 24.69/59.72/21.11 |
| 283 | SOLUSDT | 1h | short_only | 25.3406% | 15.0157% | 2.2043% | 254/63/26 | 24.94/59.59/21.19 |
| 350 | SOLUSDT | 1h | long_short | 26.3570% | 15.1313% | 1.7643% | 250/61/22 | 23.43/55.12/17.82 |
| 395 | SOLUSDT | 1h | short_only | 24.5154% | 13.6239% | 2.4704% | 230/59/26 | 23.69/51.31/21.11 |
| 485 | SOLUSDT | 1h | short_only | 23.3233% | 13.8100% | 2.4704% | 246/63/26 | 21.07/48.71/21.11 |
| 497 | SOLUSDT | 1h | long_short | 23.9228% | 13.4658% | 1.5867% | 250/61/22 | 23.92/55.19/18.03 |
| 564 | SOLUSDT | 1h | short_only | 22.0625% | 12.1097% | 2.2043% | 230/59/26 | 23.98/51.31/21.19 |
| 677 | SOLUSDT | 1h | short_only | 21.0497% | 12.2825% | 2.2043% | 246/63/26 | 21.39/48.74/21.19 |
| 709 | SOLUSDT | 1h | short_only | 19.3343% | 11.2087% | 1.6656% | 254/63/26 | 25.37/59.31/21.35 |
| 710 | SOLUSDT | 1h | short_only | 19.3343% | 11.2087% | 1.6656% | 254/63/26 | 25.37/59.31/21.35 |

## Train-dominant sample-gate shadows

Rows here pass train/validation/OOS sample and risk gates but may still fail execution efficiency.

| Rank | Symbol | TF | Train | Val | OOS | Rejection focus |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 167 | SOLUSDT | 1h | 29.9143% | 16.4858% | 0.5808% | locked_oos_return_per_turnover_proxy_bps_4.964_not_above_10.000 |
| 181 | SOLUSDT | 1h | 31.3156% | 16.1679% | 0.5808% | locked_oos_return_per_turnover_proxy_bps_4.964_not_above_10.000 |
| 263 | SOLUSDT | 1h | 26.5494% | 14.5909% | 0.5220% | locked_oos_return_per_turnover_proxy_bps_5.019_not_above_10.000 |
| 264 | SOLUSDT | 1h | 25.2092% | 15.4514% | 0.2475% | locked_oos_return_per_turnover_proxy_bps_2.500_not_above_10.000 |
| 295 | SOLUSDT | 1h | 27.7640% | 14.3128% | 0.5220% | locked_oos_return_per_turnover_proxy_bps_5.019_not_above_10.000 |
| 394 | SOLUSDT | 1h | 22.7211% | 13.7270% | 0.2350% | locked_oos_return_per_turnover_proxy_bps_2.670_not_above_10.000 |
| 427 | SOLUSDT | 1h | 32.0799% | 13.1166% | 0.7561% | locked_oos_return_per_turnover_proxy_bps_5.601_not_above_10.000 |
| 463 | SOLUSDT | 1h | 34.8231% | 12.8080% | 0.7561% | locked_oos_return_per_turnover_proxy_bps_5.601_not_above_10.000 |
| 595 | SOLUSDT | 1h | 28.4874% | 11.6489% | 0.6783% | locked_oos_return_per_turnover_proxy_bps_5.653_not_above_10.000 |
| 650 | SOLUSDT | 1h | 30.8586% | 11.3780% | 0.6783% | locked_oos_return_per_turnover_proxy_bps_5.653_not_above_10.000 |
