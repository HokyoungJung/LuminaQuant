# Strategy backtest scoreboard

- scope: `all`
- generated_at: `2026-09-03T13:58:02.566688Z`
- data_root: `/home/hoky/Quants-agent/LuminaQuant/data/market_parquet`
- exchange: `binance`
- timeframe: `1m`
- period `[start,end)`: `2026-05-01T00:00:00Z` → `2026-05-03T00:00:00Z`
- annual_periods: `525600`
- strategy_count: `162`
- pass_count: `63`
- excluded_count: `15`

## Data integrity summary

- exchange symbol audit: `skipped` (matched=n/a, unmatched=0)
- feature audit statuses: `{'not_required': 144, 'fail': 15, 'pass': 3}`
- OHLCV warning rows: `1061`
- OHLCV policy: exact requested [start,end) 1m grid, no gap fill, no interpolation, and no synthetic rows; any missing, duplicate, off-grid, or out-of-window bar fails before simulation.
- Required external features must resolve on every 1m bar in the same half-open window under the bounded 8h stale policy; sparse columns fail before simulation.
- Zero-volume bars are reported as warnings, not imputed.

## Top 12 traded performers by total return

- Pass-status strategies with zero completed trades are not treated as performance winners; they are listed in zero-trade diagnostics instead.
| Rank | Strategy | Tier | Symbols | Return | Sharpe | CAGR | MDD | Trades | Signals | Zero-trade reason |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `GoldSilverRatioTrendStrategy` | `live_default` | 2 | 0.00% | 0.119 | 0.02% | 0.01% | 8 | 8 | `` |
| 2 | `FalseBreakoutReversalStrategy` | `live_default` | 14 | -0.01% | -7.966 | -2.07% | 0.02% | 8 | 8 | `` |
| 3 | `RoundNumberBarrierStrategy` | `research_only` | 14 | -0.04% | -27.389 | -7.70% | 0.04% | 10 | 10 | `` |
| 4 | `ProfitMoonshotBreakoutStrategy` | `live_opt_in` | 14 | -0.07% | -23.432 | -11.72% | 0.07% | 4 | 4 | `` |
| 5 | `ProfitMoonshotTrendStrategy` | `live_opt_in` | 14 | -0.33% | -26.531 | -45.21% | 0.33% | 12 | 10 | `` |
| 6 | `BenchmarkLeadLagContinuationStrategy` | `live_default` | 14 | -0.47% | -38.686 | -57.82% | 0.47% | 79 | 64 | `` |
| 7 | `VolatilitySqueezeBreakoutStrategy` | `live_default` | 14 | -0.55% | -73.180 | -63.47% | 0.55% | 68 | 50 | `` |
| 8 | `LastDayLiquidityRegimeStrategy` | `live_opt_in` | 14 | -2.06% | -48.142 | -97.76% | 2.06% | 28 | 20 | `` |
| 9 | `DisagreementGatedEnsembleStrategy` | `research_only` | 14 | -5.10% | -99.867 | -99.99% | 5.10% | 371 | 2 | `` |
| 10 | `DonchianAtrTrendStrategy` | `live_default` | 14 | -7.37% | -252.415 | -100.00% | 7.37% | 996 | 507 | `` |
| 11 | `VarianceRatioTrendRiderStrategy` | `research_only` | 14 | -53.54% | -183.650 | -100.00% | 53.54% | 641 | 226 | `` |
| 12 | `OvernightSessionReturnRiderStrategy` | `live_default` | 14 | -56.08% | -124.175 | -100.00% | 56.08% | 462 | 644 | `` |

## Zero-trade diagnostics

| Strategy | Market events | Signals | Orders | Fills | Reason |
|---|---:|---:|---:|---:|---|
| `AlphaZooOptunaHybridLiveStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `BullBearRegimeRotationStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CalendarSeasonalityOverlayStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CompressionBreakoutContinuationStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossCryptoSlowDiffusionStrategy` | 8640 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossSectionalIntermediateEchoMomentumStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossSectionalOffSessionTugOfWarStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossSectionalPathConvexityStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossSectionalPriceDelayPremiumStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossSectionalRegressionTrendQualityStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossSectionalSeasonalPersistenceStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `CrossSectionalTimeUnderWaterStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `DispersionConditionedReversionStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `DownsideTailRiskPremiumStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `EquityMetalRiskRegimeRotationStrategy` | 34560 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `HourlyShockReversionStrategy` | 8640 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `IdiosyncraticSkewInnovationStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `IdiosyncraticVolatilityStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `InformationDiscretenessMomentumStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `IntermarketLeadLagContinuationStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `IntradayFlowPressureRiderStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `KalmanTrendRiderStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `LagConvergenceStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `LeadLagSpilloverStrategy` | 5760 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `LiquidityShockReversionStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `LotterySkewnessStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `LowTurnoverTrendPersistenceStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `MetalEquityDivergenceReversalStrategy` | 34560 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `MomentumCrashDynamicScalingOverlayStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `NoiseFilteredVolatilityBreakoutStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `PanicReboundMeanReversionStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `PrevDayBoxQuartileReversionStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `PriceVolumeCorrContinuationStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `ProspectTheoryValueStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `RareEventScoreStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `RegimeRouterConfirmedRotationStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `SalienceTheoryValueStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `SemisLeadLagRotationStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `SessionFilteredPairCarryStrategy` | 5760 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `SlowCrossSectionalLeadLagStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `SpectralCycleRiderStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `TimeframePairZScoreReversionStrategy` | 5760 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `TopCapTimeSeriesMomentumStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `TrendEfficiencyMomentumStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `TrendGatedResidualMomentumStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `VWAPCompressionReversionStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `VolOfVolRegimeTrendGateStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `VolumeClockMomentumRiderStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |
| `VwapReversionStrategy` | 40320 | 0 | 0 | 0 | `no_signal_generated_under_default_params_window` |

## All strategy results

| Strategy | Tier | Symbols | Return | Sharpe | CAGR | MDD | Trades | Signals | Feature audit | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `AbnormalReturnContinuationStrategy` | `live_opt_in` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `AccelerationRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `AdaptiveRegimeMomentumStrategy` | `live_opt_in` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `AdaptiveTrendRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `AdfGatedReversionRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `Alpha101FormulaStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `AlphaZooOptunaHybridLiveStrategy` | `live_opt_in` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `AmihudIlliquidityMomentumRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `AvgCorrelationCrashGuardOverlayStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `BasisFundingGapConvergenceStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `BenchmarkLeadLagContinuationStrategy` | `live_default` | 14 | -0.47% | -38.686 | -57.82% | 0.47% | 79 | 64 | `not_required` | `pass` |
| `BettingAgainstBetaStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `BitcoinBuyHoldStrategy` | `live_default` | 1 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `BreadthRegimeTrendTimerStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `BullBearRegimeRotationStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CalendarSeasonalityOverlayStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CarryTrendConfluenceRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `CompositeTrendStrategy` | `live_opt_in` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CompressionBreakoutContinuationStrategy` | `live_opt_in` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `ConfidenceGatedTrendStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossAssetDiversifiedTrendStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossCryptoSlowDiffusionStrategy` | `live_opt_in` | 3 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CrossSectionalCapitalGainsOverhangStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalCloseLocationAccumulationStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalDownsideBetaAsymmetryStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalEquityMomentumStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalFlowShareRotationStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalFundingMomentumCarryStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `CrossSectionalIntermediateEchoMomentumStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CrossSectionalNearHighAnchoringStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalNearLowRecoveryStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalOffSessionTugOfWarStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CrossSectionalPathConvexityStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CrossSectionalPriceDelayPremiumStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CrossSectionalRegressionTrendQualityStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CrossSectionalResidualTakerFlowStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `CrossSectionalSeasonalPersistenceStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CrossSectionalShortTermReversalStrategy` | `live_default` | 12 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `CrossSectionalTimeUnderWaterStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `CryptoFxAlphaZooStateStrategy` | `live_opt_in` | 3 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `CusumChangePointTrendRiderStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `DeepLearningForecastGateStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `DerivativesFlowSqueezeStrategy` | `live_opt_in` | 3 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `DisagreementGatedEnsembleStrategy` | `research_only` | 14 | -5.10% | -99.867 | -99.99% | 5.10% | 371 | 2 | `not_required` | `pass` |
| `DispersionConditionedReversionStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `DiversifiedMultiFactorEnsembleStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `DonchianAtrTrendStrategy` | `live_default` | 14 | -7.37% | -252.415 | -100.00% | 7.37% | 996 | 507 | `not_required` | `pass` |
| `DownsideTailRiskPremiumStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `DualMomentumDefensiveRotationStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `DualMomentumIndexRotationStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `EquityBenchmarkResidualReversalStrategy` | `live_default` | 12 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `EquityCurveKillSwitchOverlayStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `EquityMetalRiskRegimeRotationStrategy` | `live_default` | 12 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `FalseBreakoutReversalStrategy` | `live_default` | 14 | -0.01% | -7.966 | -2.07% | 0.02% | 8 | 8 | `not_required` | `pass` |
| `FundingDislocationTrendCarryStrategy` | `live_default` | 3 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `pass` | `fail` |
| `FundingHarvestCarryStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `GarchInnovationRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `GoldSilverRatioMeanReversionStrategy` | `live_default` | 2 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `GoldSilverRatioTrendStrategy` | `live_default` | 2 | 0.00% | 0.119 | 0.02% | 0.01% | 8 | 8 | `not_required` | `pass` |
| `HourlyShockReversionStrategy` | `live_opt_in` | 3 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `HurstRegimeGatedStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `IdiosyncraticSkewInnovationStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `IdiosyncraticVolatilityStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `InformationDiscretenessMomentumStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `IntermarketLeadLagContinuationStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `IntradayFlowPressureRiderStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `IntradaySeasonalMomentumRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `KalmanPairsStatArbStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `KalmanTrendRiderStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `LagConvergenceStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `LastDayLiquidityRegimeStrategy` | `live_opt_in` | 14 | -2.06% | -48.142 | -97.76% | 2.06% | 28 | 20 | `not_required` | `pass` |
| `LeadLagSpilloverStrategy` | `live_opt_in` | 2 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `LeveragedTrendTimingRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `LiquidationCascadeReversionStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `LiquidityShockReversionStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `LongRunOverreactionReversalStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `LotterySkewnessStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `LowTurnoverTrendPersistenceStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `LowVolatilityMomentumStrategy` | `live_default` | 12 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `MaScoreVolTargetRotationStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `MeanReversionStdStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `MetalEquityDivergenceReversalStrategy` | `live_default` | 12 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `MetalsRelativeValueBasketStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `MicroRangeExpansion1sStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `MomentumCrashDynamicScalingOverlayStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `MovingAverageCrossStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `MultiTimeframeTrendEnsembleStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `NearHighMomentumStrategy` | `live_default` | 12 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `NoiseFilteredVolatilityBreakoutStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `OffSessionBasisDislocationStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `OpenInterestGrowthPressureStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `OpenInterestTrendConfirmationRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `OpeningRangeBreakoutRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `OpeningRangeContinuationStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `OrderBookImbalanceReversionStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `OvernightSessionReturnRiderStrategy` | `live_default` | 14 | -56.08% | -124.175 | -100.00% | 56.08% | 462 | 644 | `not_required` | `pass` |
| `PairSpreadZScoreStrategy` | `live_opt_in` | 2 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `PairTradingZScoreStrategy` | `live_default` | 2 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `PairsSpreadMeanReversionStrategy` | `live_default` | 4 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `PanicReboundMeanReversionStrategy` | `live_opt_in` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `PcaResidualStatArbStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `PermutationEntropyTrendRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `PerpCrowdingCarryStrategy` | `live_opt_in` | 3 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `PrevDayBoxQuartileReversionStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `PriceVolumeCorrContinuationStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `ProfitMoonshotBreakoutStrategy` | `live_opt_in` | 14 | -0.07% | -23.432 | -11.72% | 0.07% | 4 | 4 | `not_required` | `pass` |
| `ProfitMoonshotReversionStrategy` | `live_opt_in` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `ProfitMoonshotTrendStrategy` | `live_opt_in` | 14 | -0.33% | -26.531 | -45.21% | 0.33% | 12 | 10 | `not_required` | `pass` |
| `ProspectTheoryValueStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `PullbackTrendContinuationStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RareEventScoreStrategy` | `live_opt_in` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `RealizedSemivarianceTrendRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RealizedVolTermStructureStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RebalancingPremiumHarvestStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RegimeAdaptiveDisagreementEnsembleStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RegimeBreakoutCandidateStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RegimeRouterConfirmedRotationStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `ResearchOnlyFourHourFundingHarvestCarryStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `fail` | `excluded` |
| `ResidualEquityMomentumStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `ResidualMomentumRotationStrategy` | `live_default` | 12 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RollingBreakoutStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RoundNumberBarrierStrategy` | `research_only` | 14 | -0.04% | -27.389 | -7.70% | 0.04% | 10 | 10 | `not_required` | `pass` |
| `RsiDivergenceScaleOutStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `RsiStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `SalienceTheoryValueStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `SeasonalMicroBreakoutRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `SelectionGatedMomentumStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `SelectionGatedReversionStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `SemisLeadLagRotationStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `SessionFilteredPairCarryStrategy` | `live_opt_in` | 2 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `SessionHighBreakoutScalpStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `SilentVolumeShockResolutionStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `SlowCrossSectionalLeadLagStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `SpectralCycleRiderStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `SpreadStressLiquidityReversionStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `StationarityGatedResidualReversionStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `SystematicCoskewnessPremiumStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `TailIndexRegimeRiderStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `TakerFlowExhaustionReversalStrategy` | `live_opt_in` | 3 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `pass` | `fail` |
| `TakerFlowImbalanceContinuationStrategy` | `live_default` | 3 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `pass` | `fail` |
| `TimeframePairZScoreReversionStrategy` | `live_opt_in` | 2 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `TopCapTimeSeriesMomentumStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `TrendEfficiencyMomentumStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `TrendGatedIbsReversionStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `TrendGatedResidualMomentumStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `TurtleUnitPyramidingStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VWAPCompressionReversionStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `VarianceRatioTrendRiderStrategy` | `research_only` | 14 | -53.54% | -183.650 | -100.00% | 53.54% | 641 | 226 | `not_required` | `pass` |
| `VolCompressionVWAPReversionStrategy` | `live_opt_in` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VolCompressionVwapReversionStrategy` | `live_opt_in` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VolManagedMomentumCrashGateStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VolManagedRiskOverlayStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VolOfVolRegimeTrendGateStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `VolatilityBreakoutRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VolatilityCompressionReversionStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VolatilitySqueezeBreakoutRiderStrategy` | `live_default` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VolatilitySqueezeBreakoutStrategy` | `live_default` | 14 | -0.55% | -73.180 | -63.47% | 0.55% | 68 | 50 | `not_required` | `pass` |
| `VolumeClockMomentumRiderStrategy` | `research_only` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |
| `VpinToxicityRiderStrategy` | `research_only` | 14 | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `fail` |
| `VwapReversionStrategy` | `live_default` | 14 | 0.00% | -999.000 | 0.00% | 0.00% | 0 | 0 | `not_required` | `pass` |

## Issues and audit notes

- `fail` `AbnormalReturnContinuationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XRP/USDT' at boundary 1777622400000")
- `fail` `AccelerationRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `AdaptiveRegimeMomentumStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAG/USDT' at boundary 1777651200000")
- `fail` `AdaptiveTrendRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `AdfGatedReversionRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XPD/USDT' at boundary 1777622400000")
- `fail` `Alpha101FormulaStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `AmihudIlliquidityMomentumRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAU/USDT' at boundary 1777622400000")
- `fail` `AvgCorrelationCrashGuardOverlayStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `warn` `BasisFundingGapConvergenceStrategy`: required feature audit failed for all symbols
- `fail` `BettingAgainstBetaStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BNB/USDT' at boundary 1777622400000")
- `fail` `BitcoinBuyHoldStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `BreadthRegimeTrendTimerStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `warn` `CarryTrendConfluenceRiderStrategy`: required feature audit failed for all symbols
- `fail` `CompositeTrendStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XPT/USDT' at boundary 1777708800000")
- `fail` `ConfidenceGatedTrendStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `CrossAssetDiversifiedTrendStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAG/USDT' at boundary 1777651200000")
- `fail` `CrossSectionalCapitalGainsOverhangStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'SOL/USDT' at boundary 1777622400000")
- `fail` `CrossSectionalCloseLocationAccumulationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'SOL/USDT' at boundary 1777622400000")
- `fail` `CrossSectionalDownsideBetaAsymmetryStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'SOL/USDT' at boundary 1777622400000")
- `fail` `CrossSectionalEquityMomentumStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `CrossSectionalFlowShareRotationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `warn` `CrossSectionalFundingMomentumCarryStrategy`: required feature audit failed for all symbols
- `fail` `CrossSectionalNearHighAnchoringStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777680000000")
- `fail` `CrossSectionalNearLowRecoveryStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `warn` `CrossSectionalResidualTakerFlowStrategy`: required feature audit failed for all symbols
- `fail` `CrossSectionalShortTermReversalStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `warn` `CryptoFxAlphaZooStateStrategy`: required feature audit failed for all symbols
- `fail` `CusumChangePointTrendRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777651200000")
- `warn` `DeepLearningForecastGateStrategy`: required feature audit failed for all symbols
- `warn` `DerivativesFlowSqueezeStrategy`: required feature audit failed for all symbols
- `warn` `DiversifiedMultiFactorEnsembleStrategy`: required feature audit failed for all symbols
- `fail` `DualMomentumDefensiveRotationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777651200000")
- `fail` `DualMomentumIndexRotationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777651200000")
- `fail` `EquityBenchmarkResidualReversalStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'QQQ/USDT' at boundary 1777622400000")
- `fail` `EquityCurveKillSwitchOverlayStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `FundingDislocationTrendCarryStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `warn` `FundingHarvestCarryStrategy`: required feature audit failed for all symbols
- `fail` `GarchInnovationRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAG/USDT' at boundary 1777622400000")
- `fail` `GoldSilverRatioMeanReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAU/USDT' at boundary 1777622400000")
- `fail` `HurstRegimeGatedStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `IntradaySeasonalMomentumRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `KalmanPairsStatArbStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `LeveragedTrendTimingRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TON/USDT' at boundary 1777622400000")
- `warn` `LiquidationCascadeReversionStrategy`: required feature audit failed for all symbols
- `fail` `LongRunOverreactionReversalStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TRX/USDT' at boundary 1777622400000")
- `fail` `LowVolatilityMomentumStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777651200000")
- `fail` `MaScoreVolTargetRotationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `MeanReversionStdStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TON/USDT' at boundary 1777622400000")
- `fail` `MetalsRelativeValueBasketStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAU/USDT' at boundary 1777622400000")
- `fail` `MicroRangeExpansion1sStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'SOL/USDT' at boundary 1777622400000")
- `fail` `MovingAverageCrossStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `MultiTimeframeTrendEnsembleStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `NearHighMomentumStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAG/USDT' at boundary 1777651200000")
- `warn` `OffSessionBasisDislocationStrategy`: required feature audit failed for all symbols
- `warn` `OpenInterestGrowthPressureStrategy`: required feature audit failed for all symbols
- `warn` `OpenInterestTrendConfirmationRiderStrategy`: required feature audit failed for all symbols
- `fail` `OpeningRangeBreakoutRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XPD/USDT' at boundary 1777622400000")
- `fail` `OpeningRangeContinuationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `warn` `OrderBookImbalanceReversionStrategy`: required feature audit failed for all symbols
- `fail` `PairSpreadZScoreStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `PairTradingZScoreStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `PairsSpreadMeanReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `PcaResidualStatArbStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `PermutationEntropyTrendRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ADA/USDT' at boundary 1777622400000")
- `fail` `PerpCrowdingCarryStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `ProfitMoonshotReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `PullbackTrendContinuationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TON/USDT' at boundary 1777622400000")
- `fail` `RealizedSemivarianceTrendRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `RealizedVolTermStructureStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'AVAX/USDT' at boundary 1777622400000")
- `fail` `RebalancingPremiumHarvestStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `RegimeAdaptiveDisagreementEnsembleStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XPT/USDT' at boundary 1777622400000")
- `fail` `RegimeBreakoutCandidateStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TRX/USDT' at boundary 1777622400000")
- `warn` `ResearchOnlyFourHourFundingHarvestCarryStrategy`: required feature audit failed for all symbols
- `fail` `ResidualEquityMomentumStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TRX/USDT' at boundary 1777622400000")
- `fail` `ResidualMomentumRotationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAU/USDT' at boundary 1777622400000")
- `fail` `RollingBreakoutStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `RsiDivergenceScaleOutStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `RsiStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `SeasonalMicroBreakoutRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `SelectionGatedMomentumStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `SelectionGatedReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TRX/USDT' at boundary 1777622400000")
- `fail` `SessionHighBreakoutScalpStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `SilentVolumeShockResolutionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XRP/USDT' at boundary 1777651200000")
- `fail` `SpreadStressLiquidityReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'SOL/USDT' at boundary 1777622400000")
- `fail` `StationarityGatedResidualReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'BTC/USDT' at boundary 1777622400000")
- `fail` `SystematicCoskewnessPremiumStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `TailIndexRegimeRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TON/USDT' at boundary 1777622400000")
- `fail` `TakerFlowExhaustionReversalStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777680000000")
- `fail` `TakerFlowImbalanceContinuationStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `TrendGatedIbsReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'SOL/USDT' at boundary 1777622400000")
- `fail` `TurtleUnitPyramidingStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'TRX/USDT' at boundary 1777622400000")
- `fail` `VolCompressionVWAPReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'DOGE/USDT' at boundary 1777622400000")
- `fail` `VolCompressionVwapReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'DOGE/USDT' at boundary 1777622400000")
- `fail` `VolManagedMomentumCrashGateStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAU/USDT' at boundary 1777622400000")
- `fail` `VolManagedRiskOverlayStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'ETH/USDT' at boundary 1777622400000")
- `fail` `VolatilityBreakoutRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'SOL/USDT' at boundary 1777622400000")
- `fail` `VolatilityCompressionReversionStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'DOGE/USDT' at boundary 1777622400000")
- `fail` `VolatilitySqueezeBreakoutRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XPT/USDT' at boundary 1777680000000")
- `fail` `VpinToxicityRiderStrategy`: ValueError("require_funding_coverage: missing exact funding settlement data for symbol 'XAU/USDT' at boundary 1777622400000")

## Output files

- latest_json: `/home/hoky/Quants-agent/LuminaQuant/var/backtests/rigorous-20260903/coarse_screen/strategy_screen_latest.json`
- latest_markdown: `/home/hoky/Quants-agent/LuminaQuant/var/backtests/rigorous-20260903/coarse_screen/strategy_screen_latest.md`
- timestamped_json: `/home/hoky/Quants-agent/LuminaQuant/var/backtests/rigorous-20260903/coarse_screen/strategy_screen_20260903T135802_566688Z.json`
- timestamped_markdown: `/home/hoky/Quants-agent/LuminaQuant/var/backtests/rigorous-20260903/coarse_screen/strategy_screen_20260903T135802_566688Z.md`
