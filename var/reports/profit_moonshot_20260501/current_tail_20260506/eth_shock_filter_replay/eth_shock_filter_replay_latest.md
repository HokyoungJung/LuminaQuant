# ETH 12h shock-reversion stateful replay

Generated: `2026-05-06T10:14:43.247133Z`
OOS window ends: `2026-05-04`

## Gate policy

- Replay survivors are selected only relative to the replayed incumbent and funding-guard shapes; they are **not** final wins.
- OOS return must beat `+0.8284%`.
- OOS Sharpe/MDD must beat funding-guard shadow Sharpe `0.111225` and MDD `0.1778%`.
- Those absolute thresholds are final live-equivalent backtest gates; Sharpe > 1.0 is required for final success and sub-1 full-backtest survivors are shadow-only.
- Replay enforces one ETH position, 0.8% target allocation, $175 max order value, taker fee/slippage, 10% bar-volume fill cap, cooldown, 5% stop, 10% take-profit, and 72h max hold.

## Result

- Specs evaluated: `130`
- Replay-relative survivors for one-at-a-time full backtest slots: `8`
- Replay rows with absolute final-gate shape and Sharpe>1: `0`

| rank | replay spec | survivor | train ret | val ret | OOS ret | OOS Sharpe | OOS MDD | OOS trips | top OOS rejects |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `replay_base_taker_flow_1h_10pct` | `True` | +1.0395% | +0.1820% | +0.4329% | 1.022035 | +0.0879% | 19 | `{"taker_buy_exhaustion_missing": 40, "taker_flow_missing": 18, "taker_sell_exhaustion_missing": 36}` |
| 2 | `replay_base_sol_any_regime_350bp` | `True` | +0.1584% | +0.1933% | +0.3747% | 0.822076 | +0.1195% | 22 | `{"regime_counterguard": 69}` |
| 3 | `replay_funding_guard_taker_flow_1h_10pct` | `True` | +0.0627% | +0.2877% | +0.3408% | 0.808203 | +0.1087% | 20 | `{"funding_hour_excluded": 31, "taker_buy_exhaustion_missing": 41, "taker_sell_exhaustion_missing": 38}` |
| 4 | `replay_base_btc_sol_regime_150bp_rv48_q70` | `True` | +0.6916% | +0.1014% | +0.3097% | 0.830850 | +0.0676% | 17 | `{"realized_vol_too_high": 59, "regime_counterguard": 172}` |
| 5 | `replay_base_btc_sol_any_regime_350bp` | `True` | +0.3467% | +0.1933% | +0.2832% | 0.628635 | +0.1550% | 22 | `{"regime_counterguard": 84}` |
| 6 | `replay_base_btc_sol_regime_150bp_rv48_q55` | `True` | +0.8725% | +0.0493% | +0.2129% | 0.596526 | +0.0813% | 17 | `{"realized_vol_too_high": 83, "regime_counterguard": 211}` |
| 7 | `replay_base_sol_any_regime_150bp` | `True` | +0.6434% | +0.0123% | +0.2049% | 0.487939 | +0.1272% | 20 | `{"regime_counterguard": 119}` |
| 8 | `replay_funding_guard_sol_any_regime_350bp` | `True` | +0.2435% | +0.1471% | +0.1969% | 0.438235 | +0.1472% | 23 | `{"funding_hour_excluded": 24, "regime_counterguard": 54}` |
| 9 | `replay_base_taker_flow_3h_5pct` | `False` | -0.0404% | -0.3613% | +0.4270% | 0.964761 | +0.0796% | 21 | `{"taker_buy_exhaustion_missing": 22, "taker_sell_exhaustion_missing": 23}` |
| 10 | `replay_base_rv72_q55_cap` | `False` | -0.5087% | -0.2273% | +0.3380% | 0.816440 | +0.1032% | 19 | `{"realized_vol_too_high": 149}` |
| 11 | `replay_base_sol_any_regime_250bp` | `False` | +0.4925% | -0.0392% | +0.3313% | 0.769421 | +0.0791% | 20 | `{"regime_counterguard": 111}` |
| 12 | `replay_base_btc_sol_regime_150bp_flow3h` | `False` | +0.4278% | -0.1180% | +0.2918% | 0.802543 | +0.0668% | 14 | `{"regime_counterguard": 249, "taker_buy_exhaustion_missing": 26, "taker_sell_exhaustion_missing": 50}` |
| 13 | `replay_base_btc_sol_all_regime_150bp` | `False` | -0.1457% | +0.1051% | +0.2873% | 0.654529 | +0.1271% | 20 | `{"regime_counterguard": 95}` |
| 14 | `replay_funding_guard_rv48_q85_cap` | `False` | -0.1474% | +0.2051% | +0.2850% | 0.617953 | +0.1604% | 25 | `{"funding_hour_excluded": 11, "realized_vol_too_high": 19}` |
| 15 | `replay_base_btc_sol_regime_250bp_rv24_q70` | `False` | -0.0836% | -0.1265% | +0.2827% | 0.711806 | +0.0985% | 20 | `{"realized_vol_too_high": 50, "regime_counterguard": 141}` |
| 16 | `replay_funding_guard_taker_flow_3h_5pct` | `False` | +0.1221% | -0.2695% | +0.2778% | 0.619394 | +0.1353% | 23 | `{"funding_hour_excluded": 24, "taker_buy_exhaustion_missing": 13, "taker_sell_exhaustion_missing": 37}` |
| 17 | `replay_base_btc_sol_regime_250bp_rv48_q70` | `False` | +0.2294% | -0.0574% | +0.2569% | 0.673566 | +0.0784% | 17 | `{"realized_vol_too_high": 72, "regime_counterguard": 150}` |
| 18 | `replay_base_taker_flow_1h_5pct` | `False` | +0.3523% | -0.2123% | +0.2431% | 0.533231 | +0.1236% | 23 | `{"taker_buy_exhaustion_missing": 16, "taker_flow_missing": 18, "taker_sell_exhaustion_missing": 23}` |
| 19 | `replay_funding_guard_rv24_q85_cap` | `False` | -0.0775% | +0.0628% | +0.2418% | 0.521131 | +0.1536% | 25 | `{"funding_hour_excluded": 12, "realized_vol_too_high": 22}` |
| 20 | `replay_base_taker_flow_6h_5pct` | `False` | +0.1941% | -0.1255% | +0.2377% | 0.553521 | +0.0944% | 19 | `{"taker_buy_exhaustion_missing": 43, "taker_sell_exhaustion_missing": 60}` |
| 21 | `replay_base_btc_sol_regime_250bp_rv24_q85` | `False` | +0.1786% | -0.1781% | +0.2366% | 0.559044 | +0.0967% | 20 | `{"realized_vol_too_high": 12, "regime_counterguard": 107}` |
| 22 | `replay_funding_guard_rv72_q55_cap` | `False` | -0.2549% | -0.0769% | +0.2277% | 0.544478 | +0.1171% | 20 | `{"funding_hour_excluded": 46, "realized_vol_too_high": 108}` |
| 23 | `replay_base_btc_any_regime_250bp` | `False` | -0.6145% | +0.0250% | +0.1885% | 0.423100 | +0.1071% | 21 | `{"regime_counterguard": 89}` |
| 24 | `replay_base_btc_sol_regime_250bp_flow3h` | `False` | -0.0777% | -0.0828% | +0.1871% | 0.480900 | +0.0949% | 18 | `{"regime_counterguard": 164, "taker_buy_exhaustion_missing": 24, "taker_sell_exhaustion_missing": 56}` |
| 25 | `replay_funding_wide_threshold_80bp` | `False` | -0.1836% | -0.1271% | +0.1854% | 0.401529 | +0.1800% | 26 | `{"funding_hour_excluded": 12}` |
| 26 | `replay_funding_guard_taker_flow_1h_5pct` | `False` | -0.2335% | -0.0310% | +0.1819% | 0.396122 | +0.1348% | 23 | `{"funding_hour_excluded": 20, "taker_buy_exhaustion_missing": 17, "taker_sell_exhaustion_missing": 19}` |
| 27 | `replay_funding_guard_btc_sol_regime_150bp_flow1h` | `False` | +0.1037% | -0.4129% | +0.1752% | 0.437831 | +0.1097% | 18 | `{"funding_hour_excluded": 70, "regime_counterguard": 143, "taker_buy_exhaustion_missing": 9, "taker_sell_exhaustion_miss` |
| 28 | `replay_funding_guard_taker_flow_6h_5pct` | `False` | +0.0960% | +0.0464% | +0.1723% | 0.406816 | +0.1628% | 19 | `{"funding_hour_excluded": 48, "taker_buy_exhaustion_missing": 45, "taker_sell_exhaustion_missing": 71}` |
| 29 | `replay_base_btc_sol_all_regime_250bp` | `False` | -0.4214% | +0.1201% | +0.1661% | 0.371555 | +0.1554% | 22 | `{"regime_counterguard": 85}` |
| 30 | `replay_base_btc_sol_any_regime_250bp` | `False` | +0.4763% | -0.0483% | +0.1608% | 0.379043 | +0.1271% | 20 | `{"regime_counterguard": 123}` |

## Full-backtest slots

Run at most one live-equivalent raw-first mode at a time, starting from the first replay survivor above. A replay survivor is only a slot candidate; promotion still requires train/val/OOS engine evidence and the final gates.
