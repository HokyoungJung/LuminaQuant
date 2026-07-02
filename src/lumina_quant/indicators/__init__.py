"""Reusable indicator primitives for strategy composition."""

from .accelerated import NUMBA_AVAILABLE as NUMBA_AVAILABLE
from .accelerated import POLARS_AVAILABLE as POLARS_AVAILABLE
from .accelerated import TALIB_AVAILABLE as TALIB_AVAILABLE
from .accelerated import close_to_close_volatility as close_to_close_volatility
from .accelerated import compute_fast_alpha_bundle as compute_fast_alpha_bundle
from .accelerated import garman_klass_volatility as garman_klass_volatility
from .accelerated import linear_decay_latest as linear_decay_latest
from .accelerated import rogers_satchell_volatility as rogers_satchell_volatility
from .accelerated import rolling_corr_latest_numpy as rolling_corr_latest_numpy
from .accelerated import rolling_feature_frame_polars as rolling_feature_frame_polars
from .accelerated import rolling_mean_latest_numpy as rolling_mean_latest_numpy
from .accelerated import rolling_std_latest_numpy as rolling_std_latest_numpy
from .accelerated import talib_feature_pack as talib_feature_pack
from .accelerated import yang_zhang_volatility as yang_zhang_volatility
from .advanced_alpha import cross_leadlag_spillover as cross_leadlag_spillover
from .alpha_features import AlphaFeaturePack as AlphaFeaturePack
from .alpha_features import amihud_illiquidity as amihud_illiquidity
from .alpha_features import basis_bps as basis_bps
from .alpha_features import clipped_tanh_score as clipped_tanh_score
from .alpha_features import drawdown_from_peak as drawdown_from_peak
from .alpha_features import finite_floats as finite_floats
from .alpha_features import log_return as log_return
from .alpha_features import order_flow_imbalance as order_flow_imbalance
from .alpha_features import range_zscore as range_zscore
from .alpha_features import realized_volatility as realized_volatility
from .alpha_features import simple_return as simple_return
from .alpha_features import trend_efficiency as trend_efficiency
from .alpha_features import volatility_ratio as volatility_ratio
from .alpha_features import volume_zscore as volume_zscore
from .atr import average_true_range as average_true_range
from .fast_ops import cross_sectional_rank as cross_sectional_rank
from .atr import true_range as true_range
from .bands import bollinger_bands as bollinger_bands
from .bands import donchian_channel as donchian_channel
from .bands import keltner_channel as keltner_channel
from .common import safe_float as safe_float
from .common import safe_int as safe_int
from .common import time_key as time_key
from .ensemble_weights import blend_weights as blend_weights
from .ensemble_weights import direction_hit_rate as direction_hit_rate
from .ensemble_weights import disagreement_coefficient as disagreement_coefficient
from .ensemble_weights import inverse_error_weights as inverse_error_weights
from .ensemble_weights import softmax_weights as softmax_weights
from .flow_share import cdf_extremeness as cdf_extremeness
from .flow_share import cross_sectional_share as cross_sectional_share
from .flow_share import dense_rank_desc as dense_rank_desc
from .flow_share import normal_cdf as normal_cdf
from .flow_share import share_weighted_return as share_weighted_return
from .formulaic_alpha import alpha_001 as alpha_001
from .formulaic_alpha import alpha_002 as alpha_002
from .formulaic_alpha import alpha_003 as alpha_003
from .formulaic_alpha import alpha_004 as alpha_004
from .formulaic_alpha import alpha_005 as alpha_005
from .formulaic_alpha import alpha_006 as alpha_006
from .formulaic_alpha import alpha_007 as alpha_007
from .formulaic_alpha import alpha_008 as alpha_008
from .formulaic_alpha import alpha_009 as alpha_009
from .formulaic_alpha import alpha_010 as alpha_010
from .formulaic_alpha import alpha_011 as alpha_011
from .formulaic_alpha import alpha_012 as alpha_012
from .formulaic_alpha import alpha_013 as alpha_013
from .formulaic_alpha import alpha_014 as alpha_014
from .formulaic_alpha import alpha_015 as alpha_015
from .formulaic_alpha import alpha_016 as alpha_016
from .formulaic_alpha import alpha_018 as alpha_018
from .formulaic_alpha import alpha_019 as alpha_019
from .formulaic_alpha import alpha_020 as alpha_020
from .formulaic_alpha import alpha_025 as alpha_025
from .formulaic_alpha import alpha_041 as alpha_041
from .formulaic_alpha import alpha_042 as alpha_042
from .formulaic_alpha import alpha_043 as alpha_043
from .formulaic_alpha import alpha_044 as alpha_044
from .formulaic_alpha import alpha_053 as alpha_053
from .formulaic_alpha import alpha_054 as alpha_054
from .formulaic_alpha import alpha_055 as alpha_055
from .formulaic_alpha import alpha_101 as alpha_101
from .formulaic_operators import decay_linear as decay_linear
from .formulaic_operators import delay as delay
from .formulaic_operators import delta as delta
from .formulaic_operators import rank_pct as rank_pct
from .formulaic_operators import returns_from_close as returns_from_close
from .formulaic_operators import signed_power as signed_power
from .formulaic_operators import ts_argmax as ts_argmax
from .formulaic_operators import ts_argmin as ts_argmin
from .formulaic_operators import ts_correlation as ts_correlation
from .formulaic_operators import ts_covariance as ts_covariance
from .formulaic_operators import ts_max as ts_max
from .formulaic_operators import ts_mean as ts_mean
from .formulaic_operators import ts_min as ts_min
from .formulaic_operators import ts_product as ts_product
from .formulaic_operators import ts_rank as ts_rank
from .formulaic_operators import ts_stddev as ts_stddev
from .formulaic_operators import ts_sum as ts_sum
from .futures_fast import NUMBA_AVAILABLE as FUTURES_NUMBA_AVAILABLE
from .futures_fast import normalized_true_range_latest as normalized_true_range_latest
from .futures_fast import (
    rolling_log_return_volatility_latest as rolling_log_return_volatility_latest,
)
from .futures_fast import trend_efficiency_latest as trend_efficiency_latest
from .futures_fast import volume_shock_zscore_latest as volume_shock_zscore_latest
from .garch import garch11_fit as garch11_fit
from .garch import garch11_next_variance as garch11_next_variance
from .momentum import chande_momentum_oscillator as chande_momentum_oscillator
from .momentum import cumulative_return as cumulative_return
from .momentum import detrended_price_oscillator as detrended_price_oscillator
from .momentum import fisher_transform as fisher_transform
from .momentum import kaufman_efficiency_ratio as kaufman_efficiency_ratio
from .momentum import momentum_return as momentum_return
from .momentum import momentum_spread as momentum_spread
from .moving_average import RollingMeanWindow as RollingMeanWindow
from .moving_average import double_exponential_moving_average as double_exponential_moving_average
from .moving_average import exponential_moving_average as exponential_moving_average
from .moving_average import exponential_moving_average_series as exponential_moving_average_series
from .moving_average import hull_moving_average as hull_moving_average
from .moving_average import simple_moving_average as simple_moving_average
from .moving_average import triple_exponential_moving_average as triple_exponential_moving_average
from .moving_average import weighted_moving_average as weighted_moving_average
from .oscillators import awesome_oscillator as awesome_oscillator
from .oscillators import commodity_channel_index as commodity_channel_index
from .oscillators import money_flow_index as money_flow_index
from .oscillators import percentile_rank as percentile_rank
from .oscillators import rate_of_change as rate_of_change
from .oscillators import relative_strength_index as relative_strength_index
from .oscillators import stochastic_oscillator as stochastic_oscillator
from .oscillators import stochastic_rsi as stochastic_rsi
from .oscillators import true_strength_index as true_strength_index
from .oscillators import ultimate_oscillator as ultimate_oscillator
from .oscillators import williams_r as williams_r
from .oscillators import zscore as zscore
from .rare_event import RareEventScore as RareEventScore
from .rare_event import load_close_tail_from_lazy as load_close_tail_from_lazy
from .rare_event import local_extremum_score_latest as local_extremum_score_latest
from .rare_event import rare_event_scores_from_frame as rare_event_scores_from_frame
from .rare_event import rare_event_scores_latest as rare_event_scores_latest
from .rare_event import rare_return_score_latest as rare_return_score_latest
from .rare_event import rare_streak_score_latest as rare_streak_score_latest
from .rare_event import trend_break_score_latest as trend_break_score_latest
from .research_factors import leadlag_spillover as leadlag_spillover
from .research_factors import perp_crowding_score as perp_crowding_score
from .research_factors import pv_trend_score as pv_trend_score
from .research_factors import volcomp_vwap_pressure as volcomp_vwap_pressure
from .rolling_stats import RollingZScoreWindow as RollingZScoreWindow
from .rolling_stats import ewma_volatility as ewma_volatility
from .rolling_stats import hurst_exponent as hurst_exponent
from .rolling_stats import rolling_beta as rolling_beta
from .rolling_stats import rolling_corr as rolling_corr
from .rolling_stats import rolling_quantile as rolling_quantile
from .rolling_stats import sample_std as sample_std
from .rolling_stats import ts_regression_intercept as ts_regression_intercept
from .rolling_stats import ts_regression_rsquared as ts_regression_rsquared
from .rolling_stats import ts_regression_slope as ts_regression_slope
from .rsi import IncrementalRsi as IncrementalRsi
from .spectral_cycle import cycle_phase_fraction as cycle_phase_fraction
from .spectral_cycle import dominant_cycle as dominant_cycle
from .stationarity import adf_critical_value as adf_critical_value
from .stationarity import adf_t_statistic as adf_t_statistic
from .stationarity import ar1_half_life as ar1_half_life
from .trend import aroon_indicator as aroon_indicator
from .trend import average_directional_index as average_directional_index
from .trend import ichimoku_cloud as ichimoku_cloud
from .trend import linear_regression_slope as linear_regression_slope
from .trend import moving_average_convergence_divergence as moving_average_convergence_divergence
from .trend import percentage_price_oscillator as percentage_price_oscillator
from .trend import supertrend as supertrend
from .trend import (
    triple_exponential_average_rate_of_change as triple_exponential_average_rate_of_change,
)
from .trend import vortex_indicator as vortex_indicator
from .volatility import atr_percent as atr_percent
from .volatility import bollinger_bandwidth as bollinger_bandwidth
from .volatility import choppiness_index as choppiness_index
from .volatility import conditional_value_at_risk as conditional_value_at_risk
from .volatility import downside_volatility as downside_volatility
from .volatility import historical_volatility as historical_volatility
from .volatility import log_returns as log_returns
from .volatility import max_drawdown as max_drawdown
from .volatility import parkinson_volatility as parkinson_volatility
from .volatility import rolling_sharpe_ratio as rolling_sharpe_ratio
from .volatility import rolling_sortino_ratio as rolling_sortino_ratio
from .volatility import ulcer_index as ulcer_index
from .volatility import value_at_risk as value_at_risk
from .volume import accumulation_distribution_line as accumulation_distribution_line
from .volume import chaikin_money_flow as chaikin_money_flow
from .volume import chaikin_oscillator as chaikin_oscillator
from .volume import ease_of_movement as ease_of_movement
from .volume import force_index as force_index
from .volume import negative_volume_index as negative_volume_index
from .volume import on_balance_volume as on_balance_volume
from .volume import positive_volume_index as positive_volume_index
from .volume import price_volume_correlation as price_volume_correlation
from .volume import volume_oscillator as volume_oscillator
from .volume import volume_price_trend as volume_price_trend
from .volume import volume_weighted_moving_average as volume_weighted_moving_average
from .vwap import rolling_vwap as rolling_vwap
from .vwap import vwap_deviation as vwap_deviation
from .vwap import vwap_from_sums as vwap_from_sums

__all__ = [
    "FUTURES_NUMBA_AVAILABLE",
    "NUMBA_AVAILABLE",
    "POLARS_AVAILABLE",
    "TALIB_AVAILABLE",
    "AlphaFeaturePack",
    "IncrementalRsi",
    "RareEventScore",
    "RollingMeanWindow",
    "RollingZScoreWindow",
    "accumulation_distribution_line",
    "adf_critical_value",
    "adf_t_statistic",
    "alpha_001",
    "alpha_002",
    "alpha_003",
    "alpha_004",
    "alpha_005",
    "alpha_006",
    "alpha_007",
    "alpha_008",
    "alpha_009",
    "alpha_010",
    "alpha_011",
    "alpha_012",
    "alpha_013",
    "alpha_014",
    "alpha_015",
    "alpha_016",
    "alpha_018",
    "alpha_019",
    "alpha_020",
    "alpha_025",
    "alpha_041",
    "alpha_042",
    "alpha_043",
    "alpha_044",
    "alpha_053",
    "alpha_054",
    "alpha_055",
    "alpha_101",
    "amihud_illiquidity",
    "ar1_half_life",
    "aroon_indicator",
    "atr_percent",
    "average_directional_index",
    "average_true_range",
    "awesome_oscillator",
    "basis_bps",
    "blend_weights",
    "bollinger_bands",
    "bollinger_bandwidth",
    "cdf_extremeness",
    "chaikin_money_flow",
    "chaikin_oscillator",
    "chande_momentum_oscillator",
    "choppiness_index",
    "clipped_tanh_score",
    "close_to_close_volatility",
    "commodity_channel_index",
    "compute_fast_alpha_bundle",
    "conditional_value_at_risk",
    "cross_leadlag_spillover",
    "cross_sectional_rank",
    "cross_sectional_share",
    "cumulative_return",
    "cycle_phase_fraction",
    "decay_linear",
    "delay",
    "delta",
    "dense_rank_desc",
    "detrended_price_oscillator",
    "direction_hit_rate",
    "disagreement_coefficient",
    "dominant_cycle",
    "donchian_channel",
    "double_exponential_moving_average",
    "downside_volatility",
    "drawdown_from_peak",
    "ease_of_movement",
    "ewma_volatility",
    "exponential_moving_average",
    "exponential_moving_average_series",
    "finite_floats",
    "fisher_transform",
    "force_index",
    "garch11_fit",
    "garch11_next_variance",
    "garman_klass_volatility",
    "historical_volatility",
    "hull_moving_average",
    "hurst_exponent",
    "ichimoku_cloud",
    "inverse_error_weights",
    "kaufman_efficiency_ratio",
    "keltner_channel",
    "leadlag_spillover",
    "linear_regression_slope",
    "load_close_tail_from_lazy",
    "local_extremum_score_latest",
    "log_return",
    "log_returns",
    "max_drawdown",
    "momentum_return",
    "momentum_spread",
    "money_flow_index",
    "moving_average_convergence_divergence",
    "negative_volume_index",
    "normal_cdf",
    "normalized_true_range_latest",
    "on_balance_volume",
    "order_flow_imbalance",
    "parkinson_volatility",
    "percentage_price_oscillator",
    "percentile_rank",
    "perp_crowding_score",
    "positive_volume_index",
    "price_volume_correlation",
    "pv_trend_score",
    "range_zscore",
    "rank_pct",
    "rare_event_scores_from_frame",
    "rare_event_scores_latest",
    "rare_return_score_latest",
    "rare_streak_score_latest",
    "rate_of_change",
    "realized_volatility",
    "relative_strength_index",
    "returns_from_close",
    "rogers_satchell_volatility",
    "rolling_beta",
    "rolling_corr",
    "rolling_corr_latest_numpy",
    "rolling_feature_frame_polars",
    "rolling_log_return_volatility_latest",
    "rolling_mean_latest_numpy",
    "rolling_quantile",
    "rolling_sharpe_ratio",
    "rolling_sortino_ratio",
    "rolling_std_latest_numpy",
    "rolling_vwap",
    "safe_float",
    "safe_int",
    "sample_std",
    "share_weighted_return",
    "signed_power",
    "simple_moving_average",
    "simple_return",
    "softmax_weights",
    "stochastic_oscillator",
    "stochastic_rsi",
    "supertrend",
    "talib_feature_pack",
    "time_key",
    "trend_break_score_latest",
    "trend_efficiency",
    "trend_efficiency_latest",
    "triple_exponential_average_rate_of_change",
    "triple_exponential_moving_average",
    "true_range",
    "true_strength_index",
    "ts_argmax",
    "ts_argmin",
    "ts_correlation",
    "ts_covariance",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_product",
    "ts_rank",
    "ts_regression_intercept",
    "ts_regression_rsquared",
    "ts_regression_slope",
    "ts_stddev",
    "ts_sum",
    "ulcer_index",
    "ultimate_oscillator",
    "value_at_risk",
    "volatility_ratio",
    "volcomp_vwap_pressure",
    "volume_oscillator",
    "volume_price_trend",
    "volume_shock_zscore_latest",
    "volume_weighted_moving_average",
    "volume_zscore",
    "vortex_indicator",
    "vwap_deviation",
    "vwap_from_sums",
    "weighted_moving_average",
    "williams_r",
    "yang_zhang_volatility",
    "zscore",
]
