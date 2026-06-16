# New Strategy Evaluation Report — 2026-06-16

## Scope

- Repository updated from `private/main` to `5e7715d6` before evaluation.
- Newly added strategy classes/candidates evaluated: 14 candidates across 6 strategy classes.
- Primary direct backtest universe: `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`.
- Backtest window: `2026-05-16T00:00:00Z -> 2026-06-13T09:25:00Z`.
- Market data root: `data/market_parquet`; no synthetic fills.

## Verification

- Targeted unit tests: `78 passed in 0.91s`.
- Test command:

```bash
uv run pytest -q tests/test_cross_sectional_anomaly_alpha_sleeves.py tests/test_selection_aware_alpha_sleeves.py tests/test_universe_selection.py tests/test_strategy_factory_library.py
```

- Full research runner was attempted on the 14-candidate manifest, but the 106-symbol/default resource loading path timed out after 900s.
- A restricted top-10 crypto research run also timed out after 1200s while loading resource bundles.
- Direct backtest path completed and produced the comparison below.

## New Strategy Results

| Strategy | Best candidate | Status | Return | Sharpe | MDD | Trades | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `TrendEfficiencyMomentumStrategy` | `trend_efficiency_momentum_1h_teff_lo_20_20` | pass | `+0.1495%` | `11.935` | `0.2551%` | `55` | Shadow/paper only |
| `DispersionConditionedReversionStrategy` | `dispersion_conditioned_reversion_1h_disp_lo_5_0.020` | pass | `-0.4775%` | `-76.417` | `0.4953%` | `7` | Reject / retune |
| `SelectionGatedMomentumStrategy` | `selection_gated_momentum_1h_screened_lo_48_12` | pass | `-0.5374%` | `-34.259` | `0.7346%` | `20` | Reject / retune |
| `SelectionGatedReversionStrategy` | `selection_gated_reversion_1h_screened_fade_lo_6_12` | pass | `-0.7551%` | `-35.525` | `1.2465%` | `77` | Reject / retune |
| `IdiosyncraticVolatilityStrategy` | `idiosyncratic_volatility_4h_ivol_lo_120_60` | pass | `0.0000%` | `-999.000` | `0.0000%` | `0` | Reject / no trades |
| `LotterySkewnessStrategy` | `lottery_skewness_4h_lottery_lo_60_20` | pass | `0.0000%` | `-999.000` | `0.0000%` | `0` | Reject / no trades |

Aggregate direct result: 14 candidates, 13 passed, 6 traded, 1 excluded, 0 failed.

## Incumbent Comparison

Best existing TopCap candidate on the same crypto5/window:

| Candidate | Return | Sharpe | MDD | Trades |
|---|---:|---:|---:|---:|
| `topcap_tsmom_1h_exec_tightstop_tp_16_4_0.015` | `+2.3985%` | `24.795` | `1.5959%` | `157` |

The incumbent TopCap family remains materially stronger for absolute return than all newly added strategies in this sample.

## Blend Probe

Compared `topcap_tsmom_1h_exec_tightstop_tp_16_4_0.015` against `trend_efficiency_momentum_1h_teff_lo_20_20`.

- Per-step return correlation: `0.2545`.
- TrendEfficiency standalone return: `+0.1487%`, 55 trades.
- TopCap standalone return: `+2.3985%`, 157 trades.

| TrendEfficiency weight | Blended return | Blended MDD |
|---:|---:|---:|
| 5% | `2.2865%` | `1.5177%` |
| 10% | `2.1745%` | `1.4395%` |
| 20% | `1.9502%` | `1.2829%` |
| 30% | `1.7257%` | `1.1260%` |
| 50% | `1.2761%` | `0.8118%` |

## Recommendation

- Do not promote any newly added strategy to live trading yet.
- Only `trend_efficiency_momentum_1h_teff_lo_20_20` is worth paper/shadow monitoring.
- If added to existing allocation, cap TrendEfficiency at a small 5–10% risk-sleeve weight; it may reduce drawdown but is not a return enhancer versus the current TopCap incumbent.
- Reject or retune SelectionGated momentum/reversion, DispersionConditioned reversion, IdiosyncraticVolatility, and LotterySkewness variants before further allocation work.

## Artifacts

- `direct_backtest_crypto5/new_strategy_direct_backtest_latest.md`
- `direct_backtest_crypto5/new_strategy_direct_backtest_latest.json`
- `direct_backtest_crypto5/blend_probe_latest.md`
- `baseline_compare_crypto5/baseline_compare_latest.json`
- `new_strategy_manifest.json`
