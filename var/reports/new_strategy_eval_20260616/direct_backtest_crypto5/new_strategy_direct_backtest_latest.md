# New strategy direct backtest report

- window: `2026-05-16T00:00:00Z -> 2026-06-13T09:25:00Z`
- symbols: `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`
- candidates: `14`; pass `13`; traded `6`; excluded `1`; fail `0`
- best: `trend_efficiency_momentum_1h_teff_lo_20_20` ret `0.1495%` sharpe `11.935` mdd `0.2551%` trades `55`

## Candidate rows

| Strategy | Candidate | TF | Status | Return | Sharpe | MDD | Trades | Fills | Note |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| DispersionConditionedReversionStrategy | `dispersion_conditioned_reversion_1h_disp_lo_5_0.020` | 1h | pass | -0.4775% | -76.417 | 0.4953% | 7 | 7 | negative |
| DispersionConditionedReversionStrategy | `dispersion_conditioned_reversion_4h_disp_ls_5_0.030` | 4h | pass | 0.0000% | -999.000 | 0.0000% | 0 | 0 | no trades in window |
| IdiosyncraticVolatilityStrategy | `idiosyncratic_volatility_1d_ivol_ls_120_60` | 1d | excluded | 0.0000% | 0.000 | 0.0000% | 0 | 0 | OHLCV audit failed for required warmup/common window |
| IdiosyncraticVolatilityStrategy | `idiosyncratic_volatility_4h_ivol_lo_120_60` | 4h | pass | 0.0000% | -999.000 | 0.0000% | 0 | 0 | no trades in window |
| LotterySkewnessStrategy | `lottery_skewness_1d_lottery_ls_60_20` | 1d | pass | 0.0000% | -999.000 | 0.0000% | 0 | 0 | no trades in window |
| LotterySkewnessStrategy | `lottery_skewness_4h_lottery_lo_60_20` | 4h | pass | 0.0000% | -999.000 | 0.0000% | 0 | 0 | no trades in window |
| SelectionGatedMomentumStrategy | `selection_gated_momentum_1h_screened_lo_48_12` | 1h | pass | -0.5374% | -34.259 | 0.7346% | 20 | 20 | negative |
| SelectionGatedMomentumStrategy | `selection_gated_momentum_1h_screened_ls_72_10` | 1h | pass | -0.5819% | -21.548 | 1.0070% | 33 | 33 | negative |
| SelectionGatedMomentumStrategy | `selection_gated_momentum_4h_screened_swing_30_10` | 4h | pass | 0.0000% | -999.000 | 0.0000% | 0 | 0 | no trades in window |
| SelectionGatedReversionStrategy | `selection_gated_reversion_1h_screened_fade_lo_6_12` | 1h | pass | -0.7551% | -35.525 | 1.2465% | 77 | 77 | negative |
| SelectionGatedReversionStrategy | `selection_gated_reversion_1h_screened_fade_ls_8_10` | 1h | pass | -0.9766% | -43.772 | 1.5560% | 157 | 157 | negative |
| SelectionGatedReversionStrategy | `selection_gated_reversion_4h_screened_fade_swing_4_10` | 4h | pass | 0.0000% | -999.000 | 0.0000% | 0 | 0 | no trades in window |
| TrendEfficiencyMomentumStrategy | `trend_efficiency_momentum_1h_teff_lo_20_20` | 1h | pass | 0.1495% | 11.935 | 0.2551% | 55 | 55 | positive |
| TrendEfficiencyMomentumStrategy | `trend_efficiency_momentum_4h_teff_ls_20_20` | 4h | pass | 0.0000% | -999.000 | 0.0000% | 0 | 0 | no trades in window |

## Strategy rollup

| Strategy | Candidates | Traded | Best return | Best Sharpe | Verdict |
|---|---:|---:|---:|---:|---|
| DispersionConditionedReversionStrategy | 2 | 1 | -0.4775% | -76.417 | reject/needs retune |
| IdiosyncraticVolatilityStrategy | 2 | 0 | 0.0000% | -999.000 | reject/needs retune |
| LotterySkewnessStrategy | 2 | 0 | 0.0000% | -999.000 | reject/needs retune |
| SelectionGatedMomentumStrategy | 3 | 2 | -0.5374% | -34.259 | reject/needs retune |
| SelectionGatedReversionStrategy | 3 | 2 | -0.7551% | -35.525 | reject/needs retune |
| TrendEfficiencyMomentumStrategy | 2 | 1 | 0.1495% | 11.935 | candidate |
