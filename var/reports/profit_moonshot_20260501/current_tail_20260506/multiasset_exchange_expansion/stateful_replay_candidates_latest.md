# Multiasset exchange stateful replay candidates

Generated: `2026-05-07T10:00:17.164056Z`
OOS window ends: `2026-05-04`

## Gate policy

- Hyperliquid and Tickmill are read-only feature/regime sources only.
- Replay enforces one ETH position, fee/spread/slippage/fill, cooldown, 5% stop, 10% take-profit, and 72h max hold.
- Full live-equivalent backtest slots require replay-relative survivor status; final success still requires OOS return > +0.8284%, MDD < 0.1778%, Sharpe > 1.0, liquidations 0, separated train/val/OOS raw-first evidence, and RSS < 8GB.

## Data metadata

- Binance/ETH feature rows: `2106573`
- Hyperliquid/ETH feature rows: `11736`
- Hyperliquid OI history usable: `False`
- Tickmill macro replay status: `blocked`

## Results

- Specs evaluated: `38`
- Replay survivors for one-at-a-time full backtest slots: `0`

| rank | replay spec | survivor | train ret | val ret | OOS ret | OOS Sharpe | OOS MDD | OOS trips | top OOS rejects |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `hl_funding_divergence_funding_guard_50ppm` | `False` | -0.0641% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | `{"funding_hour_excluded": 30, "hl_binance_funding_sign_divergence": 48}` |
| 2 | `hl_funding_divergence_funding_guard_75ppm` | `False` | -0.0367% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | `{"funding_hour_excluded": 30, "hl_binance_funding_sign_divergence": 48}` |
| 3 | `hl_funding_divergence_funding_guard_100ppm` | `False` | -0.0792% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | `{"funding_hour_excluded": 30, "hl_binance_funding_sign_divergence": 48}` |
| 4 | `hl_funding_divergence_funding_guard_150ppm` | `False` | -0.0792% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | `{"funding_hour_excluded": 30, "hl_binance_funding_sign_divergence": 48}` |
| 5 | `hl_funding_divergence_funding_guard_250ppm` | `False` | -0.0792% | -0.3573% | +0.2064% | 0.460760 | +0.1175% | 22 | `{"funding_hour_excluded": 30, "hl_binance_funding_sign_divergence": 48}` |
| 6 | `hl_funding_divergence_base_50ppm` | `False` | -0.0525% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | `{"hl_binance_funding_sign_divergence": 63}` |
| 7 | `hl_funding_divergence_base_75ppm` | `False` | -0.1794% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | `{"hl_binance_funding_sign_divergence": 63}` |
| 8 | `hl_funding_divergence_base_100ppm` | `False` | -0.2217% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | `{"hl_binance_funding_sign_divergence": 63}` |
| 9 | `hl_funding_divergence_base_150ppm` | `False` | -0.2217% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | `{"hl_binance_funding_sign_divergence": 63}` |
| 10 | `hl_funding_divergence_base_250ppm` | `False` | -0.2217% | -0.2336% | +0.1530% | 0.338964 | +0.1264% | 22 | `{"hl_binance_funding_sign_divergence": 63}` |
| 11 | `replay_base_12h_threshold_100bp` | `False` | -0.3704% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | `{}` |
| 12 | `hl_funding_abs_cap_base_50ppm` | `False` | -0.2962% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | `{}` |
| 13 | `hl_funding_abs_cap_base_75ppm` | `False` | -0.3289% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | `{}` |
| 14 | `hl_funding_abs_cap_base_100ppm` | `False` | -0.3704% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | `{}` |
| 15 | `hl_funding_abs_cap_base_150ppm` | `False` | -0.3704% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | `{}` |
| 16 | `hl_funding_abs_cap_base_250ppm` | `False` | -0.3704% | -0.0889% | +0.1196% | 0.251137 | +0.1607% | 26 | `{}` |
| 17 | `replay_funding_guard_current_threshold_80bp` | `False` | +0.2206% | -0.3307% | +0.0825% | 0.177490 | +0.1800% | 27 | `{"funding_hour_excluded": 9}` |
| 18 | `hl_funding_abs_cap_funding_guard_50ppm` | `False` | -0.0006% | -0.3307% | +0.0825% | 0.177490 | +0.1800% | 27 | `{"funding_hour_excluded": 9}` |
| 19 | `hl_funding_abs_cap_funding_guard_75ppm` | `False` | +0.2631% | -0.3307% | +0.0825% | 0.177490 | +0.1800% | 27 | `{"funding_hour_excluded": 9}` |
| 20 | `hl_funding_abs_cap_funding_guard_100ppm` | `False` | +0.2206% | -0.3307% | +0.0825% | 0.177490 | +0.1800% | 27 | `{"funding_hour_excluded": 9}` |
| 21 | `hl_funding_abs_cap_funding_guard_150ppm` | `False` | +0.2206% | -0.3307% | +0.0825% | 0.177490 | +0.1800% | 27 | `{"funding_hour_excluded": 9}` |
| 22 | `hl_funding_abs_cap_funding_guard_250ppm` | `False` | +0.2206% | -0.3307% | +0.0825% | 0.177490 | +0.1800% | 27 | `{"funding_hour_excluded": 9}` |
| 23 | `hl_oi_exhaustion_base_z05` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"hl_oi_history_missing": 813}` |
| 24 | `hl_oi_exhaustion_base_z10` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"hl_oi_history_missing": 813}` |
| 25 | `hl_mark_basis_stress_base_10bp` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"hl_mark_history_missing": 813}` |
| 26 | `hl_mark_basis_stress_base_25bp` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"hl_mark_history_missing": 813}` |
| 27 | `hl_mark_basis_stress_base_50bp` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"hl_mark_history_missing": 813}` |
| 28 | `tickmill_macro_usd_risk_base` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"tickmill_mt5_macro_data_unavailable": 813}` |
| 29 | `tickmill_macro_xau_xag_stress_base` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"tickmill_mt5_macro_data_unavailable": 813}` |
| 30 | `tickmill_macro_indices_risk_off_base` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"tickmill_mt5_macro_data_unavailable": 813}` |
| 31 | `hl_oi_exhaustion_funding_guard_z05` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "hl_oi_history_missing": 689}` |
| 32 | `hl_oi_exhaustion_funding_guard_z10` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "hl_oi_history_missing": 689}` |
| 33 | `hl_mark_basis_stress_funding_guard_10bp` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "hl_mark_history_missing": 689}` |
| 34 | `hl_mark_basis_stress_funding_guard_25bp` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "hl_mark_history_missing": 689}` |
| 35 | `hl_mark_basis_stress_funding_guard_50bp` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "hl_mark_history_missing": 689}` |
| 36 | `tickmill_macro_usd_risk_funding_guard` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "tickmill_mt5_macro_data_unavailable": 689}` |
| 37 | `tickmill_macro_xau_xag_stress_funding_guard` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "tickmill_mt5_macro_data_unavailable": 689}` |
| 38 | `tickmill_macro_indices_risk_off_funding_guard` | `False` | +0.0000% | +0.0000% | +0.0000% | 0.000000 | +0.0000% | 0 | `{"funding_hour_excluded": 234, "tickmill_mt5_macro_data_unavailable": 689}` |

## No full-backtest slot

No Hyperliquid/Tickmill read-only filter earned a replay survivor slot. Vector-only or context-only ideas are rejected before live-equivalent backtesting.
