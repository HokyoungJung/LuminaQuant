# Strict No-Leak Leaf Selector Report

- generated: `2026-06-06T14:24:50.381592+00:00`
- eligible symbols: `10` / `85`
- monitor-only symbols: `75`
- selection: train + previous 2M validation only; next month OOS report-only
- nested hybrids: forbidden; raw leaf rules only

## Policy comparison

| Cost | Policy | OOS comp | MDD | Sharpe | Sortino | PF | Pos months | Min month | Latest |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10bps` | `best_single` | `54.56%` | `30.63%` | `1.26` | `1.79` | `1.2096102974868224` | `6/10` | `-17.54%` | `24.47%` |
| `10bps` | `top3_equal` | `-20.64%` | `39.81%` | `-0.46` | `-0.51` | `0.9222291810816932` | `5/10` | `-25.01%` | `7.84%` |
| `10bps` | `top5_equal` | `-1.00%` | `20.49%` | `0.14` | `0.18` | `1.0252389190117328` | `4/10` | `-10.81%` | `13.07%` |
| `10bps` | `cash_gated_top3` | `-20.64%` | `39.81%` | `-0.46` | `-0.51` | `0.9222291810816932` | `5/10` | `-25.01%` | `7.84%` |
| `20bps` | `best_single` | `27.10%` | `43.63%` | `0.82` | `1.09` | `1.1376704168954466` | `6/10` | `-32.16%` | `24.47%` |
| `20bps` | `top3_equal` | `-8.46%` | `27.37%` | `-0.13` | `-0.18` | `0.9796524853340398` | `4/10` | `-10.14%` | `10.66%` |
| `20bps` | `top5_equal` | `-0.09%` | `23.68%` | `0.15` | `0.21` | `1.0235668110896226` | `5/10` | `-7.38%` | `5.03%` |
| `20bps` | `cash_gated_top3` | `-8.46%` | `27.37%` | `-0.13` | `-0.18` | `0.9796524853340398` | `4/10` | `-10.14%` | `10.66%` |

## Best 10bps policy by utility: `best_single`

| Fold | OOS | Chosen leaves |
| --- | ---: | --- |
| `2025-09` | `-0.74%` | ts_mom_vol:ETHUSDT:12h:lb48:th0 |
| `2025-10` | `5.51%` | ts_mom_vol:ETHUSDT:12h:lb3:th0 |
| `2025-11` | `16.68%` | ts_mom_vol:ADAUSDT:12h:lb3:th0 |
| `2025-12` | `23.90%` | ts_mom_vol:DOGEUSDT:12h:lb48:th0 |
| `2026-01` | `-17.54%` | basket_reversal:basket:12h:lb24:th1.5 |
| `2026-02` | `8.29%` | basket_reversal:basket:1d:lb3:th1 |
| `2026-03` | `-11.06%` | ts_mom_vol:SOLUSDT:4h:lb48:th0 |
| `2026-04` | `-1.52%` | basket_reversal:basket:1d:lb3:th1 |
| `2026-05` | `4.87%` | ts_mom_vol:TRXUSDT:12h:lb48:th0 |
| `2026-06` | `24.47%` | ts_mom_vol:ETHUSDT:1d:lb24:th0 |

## Interpretation

- Any claim based on the fixed leaf OOS screen is diagnostic only; it is not clean selection evidence.
- The rows above are the clean no-leak policy results: every fold chooses leaves from train/validation only, then reports locked OOS.
- If 20bps materially degrades a policy, it is not ready for real execution without fill telemetry.
