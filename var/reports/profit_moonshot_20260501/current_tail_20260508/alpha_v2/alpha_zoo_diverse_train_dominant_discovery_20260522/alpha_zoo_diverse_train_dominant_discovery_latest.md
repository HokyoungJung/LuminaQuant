# Alpha Zoo diverse train-dominant discovery

Generated: `2026-05-22T13:56:07.687901Z`

This run enforces `train_return >= validation_return` for promotion trust and explores diverse strategy families.
Locked-OOS is gate/report-only after train+validation candidate freeze. Real-money remains blocked.

## Summary

- Candidates evaluated: `22800`
- Rows with train >= validation: `7968`
- Train-dominant sample gate pass: `50`
- Execution-efficiency proxy gate pass: `151`
- Full paper candidate gate pass: `0`
- Max validation return: `0.3458328953710983`
- `ready_for_real=false`, `real_money_execution=false`

## Top train+validation-ranked rows

| Rank | Family | Symbol | TF | Train | Val | OOS | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | debounced_momentum_hysteresis | DOGEUSDT | 1h | 99.3970% | 28.1765% | 0.0000% | no_promotion_shadow_or_reject |
| 2 | debounced_momentum_hysteresis | DOGEUSDT | 1h | 27.0530% | 28.9934% | 0.0000% | no_promotion_shadow_or_reject |
| 3 | stateful_momentum_hysteresis | DOGEUSDT | 2h | 52.4765% | 27.4475% | 0.0000% | no_promotion_shadow_or_reject |
| 4 | debounced_momentum_hysteresis | ETHUSDT | 1h | 25.0916% | 23.5473% | 1.3122% | train_dominant_shadow_until_execution_efficiency |
| 5 | debounced_momentum_hysteresis | DOGEUSDT | 1h | 85.8505% | 24.7857% | 0.0000% | no_promotion_shadow_or_reject |
| 6 | debounced_momentum_hysteresis | DOGEUSDT | 1h | 24.3803% | 25.4791% | 0.0000% | no_promotion_shadow_or_reject |
| 7 | debounced_momentum_hysteresis | XRPUSDT | 1h | 31.1357% | 24.0532% | 0.0000% | no_promotion_shadow_or_reject |
| 8 | debounced_momentum_hysteresis | DOGEUSDT | 2h | 50.7467% | 22.6686% | 0.0000% | no_promotion_shadow_or_reject |
| 9 | debounced_momentum_hysteresis | ETHUSDT | 1h | 22.4142% | 20.7652% | 1.1768% | train_dominant_shadow_until_execution_efficiency |
| 10 | debounced_momentum_hysteresis | SOLUSDT | 1h | 28.0417% | 21.5917% | -6.2136% | no_promotion_shadow_or_reject |

## Train-dominant sample-gate shadows

These rows satisfy train>=validation, split sample counts, positive locked-OOS, zero liquidation/account wipeout, and validation return/MDD gates; they are still shadow-only until execution efficiency passes.

| Rank | Family | Symbol | TF | Train | Val | OOS | Rejection focus |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 4 | debounced_momentum_hysteresis | ETHUSDT | 1h | 25.0916% | 23.5473% | 1.3122% | train_return_per_turnover_proxy_bps_9.356_not_above_10.000; locked_oos_return_per_turnover_proxy_bps_5.302_not_above_10.000 |
| 9 | debounced_momentum_hysteresis | ETHUSDT | 1h | 22.4142% | 20.7652% | 1.1768% | train_return_per_turnover_proxy_bps_9.402_not_above_10.000; locked_oos_return_per_turnover_proxy_bps_5.349_not_above_10.000 |
| 30 | debounced_momentum_hysteresis | ETHUSDT | 1h | 16.9461% | 15.3254% | 0.8981% | train_return_per_turnover_proxy_bps_9.478_not_above_10.000; locked_oos_return_per_turnover_proxy_bps_5.443_not_above_10.000 |
| 67 | debounced_momentum_hysteresis | ETHUSDT | 1h | 17.0655% | 11.5261% | 0.0530% | locked_oos_return_per_turnover_proxy_bps_0.421_not_above_10.000 |
| 68 | debounced_momentum_hysteresis | ETHUSDT | 1h | 20.9587% | 12.2708% | 0.1437% | train_return_per_turnover_proxy_bps_5.750_not_above_10.000; locked_oos_return_per_turnover_proxy_bps_0.523_not_above_10.000 |
| 77 | debounced_momentum_hysteresis | ETHUSDT | 1h | 19.6942% | 11.6835% | 0.8903% | locked_oos_return_per_turnover_proxy_bps_6.595_not_above_10.000 |
| 82 | debounced_momentum_hysteresis | ETHUSDT | 1h | 11.3611% | 10.0515% | 0.6090% | train_return_per_turnover_proxy_bps_9.531_not_above_10.000; locked_oos_return_per_turnover_proxy_bps_5.537_not_above_10.000 |
| 85 | debounced_momentum_hysteresis | ETHUSDT | 1h | 18.7415% | 10.8988% | 0.1349% | train_return_per_turnover_proxy_bps_5.784_not_above_10.000; locked_oos_return_per_turnover_proxy_bps_0.553_not_above_10.000 |
| 87 | debounced_momentum_hysteresis | ETHUSDT | 1h | 15.2256% | 10.2139% | 0.0545% | locked_oos_return_per_turnover_proxy_bps_0.487_not_above_10.000 |
| 101 | debounced_momentum_hysteresis | ETHUSDT | 1h | 17.5688% | 10.3716% | 0.7951% | locked_oos_return_per_turnover_proxy_bps_6.626_not_above_10.000 |
