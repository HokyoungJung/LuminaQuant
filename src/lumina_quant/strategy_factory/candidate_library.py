"""Candidate-library builder for advanced multi-sleeve quant research."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from lumina_quant.research_universe import (
    BINANCE_TRADFI_COMMODITY_SYMBOLS,
    BINANCE_TRADFI_EQUITY_SYMBOLS,
    BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS,
)
from lumina_quant.strategies.pair_spread_zscore import bounded_pair_retune_params
from lumina_quant.symbols import (
    CANONICAL_STRATEGY_TIMEFRAMES,
    canonicalize_symbol_list,
    normalize_strategy_timeframes,
)
from lumina_quant.strategy_factory.runtime_settings import (
    current_research_market_data_settings,
    default_research_symbol_universe,
)

_PAIR_ANCHORS: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("BTC/USDT", "TRX/USDT"),
    ("BNB/USDT", "TRX/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT", "XAG/USDT"),
    ("XPT/USDT", "XPD/USDT"),
    ("BTC/USDT", "XAU/USDT"),
    ("ETH/USDT", "XAU/USDT"),
    ("BNB/USDT", "XAU/USDT"),
    ("BTC/USDT", "XAG/USDT"),
)

_CRYPTO_LEADERS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"}
_METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}


DEFAULT_BINANCE_TOP10_PLUS_METALS: tuple[str, ...] = default_research_symbol_universe()

DEFAULT_TIMEFRAMES: tuple[str, ...] = CANONICAL_STRATEGY_TIMEFRAMES

_COMPOSITE_TREND_OOS_STABILITY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "stable_ls_core",
            "long_threshold": 0.60,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0038,
            "max_signal_strength": 1.35,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.0,
            "max_hold_bars": 960,
            "crowding_reduce_threshold": 0.50,
            "crowding_block_threshold": 0.78,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_highconv",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0036,
            "max_signal_strength": 1.20,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.1,
            "max_hold_bars": 960,
            "crowding_reduce_threshold": 0.50,
            "crowding_block_threshold": 0.78,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_tefilter",
            "long_threshold": 0.60,
            "short_threshold": 0.45,
            "te_min": 0.25,
            "vr_min": 0.80,
            "exit_score_cross": 0.04,
            "chop_max": 58.0,
            "vol_window": 160,
            "risk_target_vol": 0.0035,
            "max_signal_strength": 1.15,
            "atr_stop_mult": 2.1,
            "trail_atr_mult": 3.2,
            "max_hold_bars": 960,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.75,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_crashguard",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.82,
            "exit_score_cross": 0.03,
            "chop_max": 58.0,
            "vol_window": 144,
            "risk_target_vol": 0.0034,
            "max_signal_strength": 1.10,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.2,
            "max_hold_bars": 768,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.72,
            "benchmark_regime_ma": 96,
            "benchmark_symbol": "BTC/USDT",
            "allow_short": True,
        },
        {
            "variant": "stable_ls_exec_trail",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.02,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0035,
            "max_signal_strength": 1.15,
            "atr_stop_mult": 1.8,
            "trail_atr_mult": 2.4,
            "max_hold_bars": 768,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.76,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_exec_shorthold",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.04,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0035,
            "max_signal_strength": 1.10,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 2.8,
            "max_hold_bars": 640,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.76,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "stable_lo_core",
            "long_threshold": 0.60,
            "short_threshold": 0.75,
            "te_min": 0.25,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 58.0,
            "vol_window": 168,
            "risk_target_vol": 0.0030,
            "max_signal_strength": 1.10,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.0,
            "max_hold_bars": 720,
            "crowding_reduce_threshold": 0.45,
            "crowding_block_threshold": 0.70,
            "allow_short": False,
        },
        {
            "variant": "stable_lo_highconv",
            "long_threshold": 0.75,
            "short_threshold": 0.75,
            "te_min": 0.25,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 58.0,
            "vol_window": 168,
            "risk_target_vol": 0.0028,
            "max_signal_strength": 1.00,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.1,
            "max_hold_bars": 720,
            "crowding_reduce_threshold": 0.45,
            "crowding_block_threshold": 0.70,
            "allow_short": False,
        },
        {
            "variant": "stable_lo_guarded",
            "long_threshold": 0.75,
            "short_threshold": 0.60,
            "te_min": 0.25,
            "vr_min": 0.95,
            "exit_score_cross": 0.02,
            "chop_max": 56.0,
            "vol_window": 192,
            "risk_target_vol": 0.0028,
            "max_signal_strength": 0.95,
            "atr_stop_mult": 2.1,
            "trail_atr_mult": 3.2,
            "max_hold_bars": 720,
            "crowding_reduce_threshold": 0.40,
            "crowding_block_threshold": 0.65,
            "allow_short": False,
        },
    ),
}

_PAIR_RETUNE_FOCUS_PAIRS_15M: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "TRX/USDT"),
    ("BNB/USDT", "TRX/USDT"),
)

_PAIR_RETUNE_FOCUS_PAIRS_30M: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "BNB/USDT"),
    ("BTC/USDT", "TRX/USDT"),
    ("BNB/USDT", "TRX/USDT"),
    ("ETH/USDT", "SOL/USDT"),
)

_PAIR_RETUNE_FOCUS_PAIRS_4H: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT", "XAG/USDT"),
    ("XPT/USDT", "XPD/USDT"),
    ("BTC/USDT", "XAU/USDT"),
    ("ETH/USDT", "XAU/USDT"),
    ("BNB/USDT", "XAU/USDT"),
    ("BTC/USDT", "XAG/USDT"),
)

_PAIR_RETUNE_FOCUS_PAIRS_1D: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("BTC/USDT", "TRX/USDT"),
    ("XPT/USDT", "XPD/USDT"),
    ("BTC/USDT", "XAU/USDT"),
    ("ETH/USDT", "XAU/USDT"),
    ("BNB/USDT", "XAU/USDT"),
    ("BTC/USDT", "XAG/USDT"),
)

_PAIR_RETUNE_SPECS_BY_TIMEFRAME: dict[str, tuple[tuple[float, float, float], ...]] = {
    "30m": (
        (2.0, 0.50, 3.6),
        (2.4, 0.60, 4.0),
    ),
    "15m": (
        (2.6, 0.70, 4.2),
        (3.0, 0.85, 4.8),
    ),
    "1h": (
        (1.8, 0.45, 3.4),
        (2.2, 0.55, 3.9),
        (2.6, 0.70, 4.2),
    ),
    "4h": (
        (1.6, 0.35, 3.0),
        (1.8, 0.45, 3.4),
        (2.0, 0.50, 3.6),
        (2.2, 0.55, 3.9),
        (2.6, 0.70, 4.2),
    ),
    "1d": (
        (1.4, 0.30, 2.8),
        (1.5, 0.33, 2.9),
        (1.6, 0.35, 3.0),
        (1.8, 0.45, 3.4),
        (2.2, 0.55, 3.9),
    ),
}

_PAIR_RETUNE_PARAM_SETS_BY_TIMEFRAME: dict[str, tuple[dict[str, float | int | str], ...]] = {
    "30m": (
        {
            "variant": "sector",
            "lookback_window": 120,
            "hedge_window": 240,
            "min_correlation": 0.18,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 192,
            "stop_loss_pct": 0.025,
        },
    ),
    "1h": (
        {
            "variant": "core",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.20,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 240,
            "stop_loss_pct": 0.030,
        },
        {
            "variant": "state_vwap",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.25,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.030,
            "vwap_window": 72,
            "min_volume_window": 24,
            "min_volume_ratio": 0.20,
        },
        {
            "variant": "state_volconv",
            "lookback_window": 120,
            "hedge_window": 240,
            "min_correlation": 0.22,
            "cooldown_bars": 10,
            "reentry_z_buffer": 0.30,
            "max_hold_bars": 192,
            "stop_loss_pct": 0.025,
            "vol_lag_bars": 2,
            "min_vol_convergence": 0.60,
            "beta_stop_scale_min": 0.85,
            "beta_stop_scale_max": 2.0,
        },
        {
            "variant": "state_atr",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.25,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.025,
            "atr_window": 14,
            "atr_max_pct": 0.04,
        },
        {
            "variant": "exec_takeprofit",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.20,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.030,
            "take_profit_pct": 0.10,
        },
        {
            "variant": "exec_tightstop_tp",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.20,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.08,
        },
    ),
    "4h": (
        {
            "variant": "participation",
            "lookback_window": 72,
            "hedge_window": 144,
            "min_correlation": 0.05,
            "cooldown_bars": 4,
            "reentry_z_buffer": 0.15,
            "max_hold_bars": 96,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "balanced",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.08,
            "cooldown_bars": 5,
            "reentry_z_buffer": 0.18,
            "max_hold_bars": 120,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "stability",
            "lookback_window": 120,
            "hedge_window": 240,
            "min_correlation": 0.12,
            "cooldown_bars": 6,
            "reentry_z_buffer": 0.22,
            "max_hold_bars": 144,
            "stop_loss_pct": 0.020,
        },
        {
            "variant": "fast_cycle",
            "lookback_window": 84,
            "hedge_window": 168,
            "min_correlation": 0.03,
            "cooldown_bars": 3,
            "reentry_z_buffer": 0.12,
            "max_hold_bars": 72,
            "stop_loss_pct": 0.030,
        },
    ),
    "1d": (
        {
            "variant": "participation",
            "lookback_window": 48,
            "hedge_window": 96,
            "min_correlation": 0.00,
            "cooldown_bars": 1,
            "reentry_z_buffer": 0.10,
            "max_hold_bars": 28,
            "stop_loss_pct": 0.020,
        },
        {
            "variant": "balanced",
            "lookback_window": 64,
            "hedge_window": 128,
            "min_correlation": 0.04,
            "cooldown_bars": 2,
            "reentry_z_buffer": 0.12,
            "max_hold_bars": 36,
            "stop_loss_pct": 0.020,
        },
        {
            "variant": "short_window",
            "lookback_window": 40,
            "hedge_window": 80,
            "min_correlation": 0.00,
            "cooldown_bars": 1,
            "reentry_z_buffer": 0.08,
            "max_hold_bars": 24,
            "stop_loss_pct": 0.020,
        },
    ),
}

_PAIR_ADAPTIVE_RLS_1H_SPECS: tuple[dict[str, float | int | str], ...] = (
    {
        "variant": "adaptive_rls_fast",
        "lookback_window": 96,
        "hedge_window": 192,
        "entry_z": 2.5,
        "exit_z": 0.65,
        "stop_z": 4.1,
        "min_correlation": 0.18,
        "cooldown_bars": 6,
        "reentry_z_buffer": 0.20,
        "max_hold_bars": 168,
        "stop_loss_pct": 0.025,
        "hedge_mode": "rls",
        "hedge_forgetting_factor": 0.985,
        "hedge_covariance_init": 8.0,
        "take_profit_pct": 0.06,
    },
    {
        "variant": "adaptive_rls_stable",
        "lookback_window": 120,
        "hedge_window": 240,
        "entry_z": 2.6,
        "exit_z": 0.70,
        "stop_z": 4.2,
        "min_correlation": 0.20,
        "cooldown_bars": 8,
        "reentry_z_buffer": 0.22,
        "max_hold_bars": 168,
        "stop_loss_pct": 0.025,
        "hedge_mode": "rls",
        "hedge_forgetting_factor": 0.992,
        "hedge_covariance_init": 10.0,
        "atr_window": 14,
        "atr_max_pct": 0.04,
    },
)

_LAG_CONVERGENCE_FOCUS_PAIRS_BY_TIMEFRAME: dict[str, tuple[tuple[str, str], ...]] = {
    "4h": (
        ("XAU/USDT", "XAG/USDT"),
        ("XPT/USDT", "XPD/USDT"),
    ),
    "1d": (
        ("XAU/USDT", "XAG/USDT"),
        ("XPT/USDT", "XPD/USDT"),
    ),
}

_LAG_CONVERGENCE_SPECS_BY_TIMEFRAME: dict[str, tuple[dict[str, float | int | str], ...]] = {
    "4h": (
        {
            "variant": "metals_core",
            "lag_bars": 2,
            "entry_threshold": 0.018,
            "exit_threshold": 0.006,
            "stop_threshold": 0.060,
            "max_hold_bars": 36,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "metals_fast",
            "lag_bars": 1,
            "entry_threshold": 0.014,
            "exit_threshold": 0.004,
            "stop_threshold": 0.050,
            "max_hold_bars": 24,
            "stop_loss_pct": 0.030,
        },
    ),
    "1d": (
        {
            "variant": "metals_core",
            "lag_bars": 1,
            "entry_threshold": 0.012,
            "exit_threshold": 0.004,
            "stop_threshold": 0.040,
            "max_hold_bars": 14,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "metals_patience",
            "lag_bars": 2,
            "entry_threshold": 0.015,
            "exit_threshold": 0.005,
            "stop_threshold": 0.050,
            "max_hold_bars": 18,
            "stop_loss_pct": 0.030,
        },
    ),
}

_ROLLING_BREAKOUT_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "loose_lo",
            "lookback_bars": 48,
            "breakout_buffer": 0.001,
            "atr_window": 14,
            "atr_stop_multiplier": 2.2,
            "stop_loss_pct": 0.025,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "lookback_bars": 64,
            "breakout_buffer": 0.002,
            "atr_window": 21,
            "atr_stop_multiplier": 2.8,
            "stop_loss_pct": 0.030,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "loose_lo",
            "lookback_bars": 36,
            "breakout_buffer": 0.001,
            "atr_window": 14,
            "atr_stop_multiplier": 2.0,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "lookback_bars": 48,
            "breakout_buffer": 0.002,
            "atr_window": 18,
            "atr_stop_multiplier": 2.5,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
    ),
}

_REGIME_BREAKOUT_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "trend_guarded",
            "lookback_window": 48,
            "slope_window": 21,
            "volatility_fast_window": 12,
            "volatility_slow_window": 48,
            "range_entry_threshold": 0.68,
            "slope_entry_threshold": 0.001,
            "momentum_floor": 0.003,
            "max_volatility_ratio": 1.8,
            "stop_loss_pct": 0.025,
            "allow_short": False,
        },
        {
            "variant": "trend_ls",
            "lookback_window": 64,
            "slope_window": 24,
            "volatility_fast_window": 16,
            "volatility_slow_window": 64,
            "range_entry_threshold": 0.72,
            "slope_entry_threshold": 0.0015,
            "momentum_floor": 0.004,
            "max_volatility_ratio": 1.7,
            "stop_loss_pct": 0.030,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "trend_guarded",
            "lookback_window": 36,
            "slope_window": 18,
            "volatility_fast_window": 10,
            "volatility_slow_window": 40,
            "range_entry_threshold": 0.65,
            "slope_entry_threshold": 0.0008,
            "momentum_floor": 0.002,
            "max_volatility_ratio": 1.9,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
        {
            "variant": "trend_ls",
            "lookback_window": 48,
            "slope_window": 21,
            "volatility_fast_window": 12,
            "volatility_slow_window": 48,
            "range_entry_threshold": 0.70,
            "slope_entry_threshold": 0.001,
            "momentum_floor": 0.003,
            "max_volatility_ratio": 1.8,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
    ),
}

_MEAN_REVERSION_STD_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "15m": (
        {
            "variant": "balanced_ls",
            "window": 64,
            "entry_z": 2.0,
            "exit_z": 0.50,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 96,
            "entry_z": 2.4,
            "exit_z": 0.40,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
        {
            "variant": "resid_btc_ls",
            "window": 64,
            "entry_z": 2.0,
            "exit_z": 0.50,
            "stop_loss_pct": 0.025,
            "allow_short": True,
            "residualize_btc": True,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "resid_btc_guarded_lo",
            "window": 96,
            "entry_z": 2.4,
            "exit_z": 0.40,
            "stop_loss_pct": 0.020,
            "allow_short": False,
            "residualize_btc": True,
            "btc_symbol": "BTC/USDT",
        },
    ),
    "30m": (
        {
            "variant": "balanced_ls",
            "window": 48,
            "entry_z": 1.8,
            "exit_z": 0.45,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 72,
            "entry_z": 2.2,
            "exit_z": 0.35,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
    ),
}

_LIQUIDITY_SHOCK_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "thin_ls",
            "volume_window": 64,
            "range_window": 48,
            "volume_shock_z": 1.4,
            "range_shock_z": 1.0,
            "return_shock_pct": 0.008,
            "revert_fraction": 0.45,
            "max_hold_bars": 18,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "thin_lo",
            "volume_window": 72,
            "range_window": 64,
            "volume_shock_z": 1.8,
            "range_shock_z": 1.2,
            "return_shock_pct": 0.010,
            "revert_fraction": 0.40,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
    "15m": (
        {
            "variant": "thin_ls",
            "volume_window": 48,
            "range_window": 36,
            "volume_shock_z": 1.2,
            "range_shock_z": 0.9,
            "return_shock_pct": 0.012,
            "revert_fraction": 0.50,
            "max_hold_bars": 10,
            "stop_loss_pct": 0.022,
            "allow_short": True,
        },
        {
            "variant": "thin_lo",
            "volume_window": 64,
            "range_window": 48,
            "volume_shock_z": 1.5,
            "range_shock_z": 1.1,
            "return_shock_pct": 0.015,
            "revert_fraction": 0.45,
            "max_hold_bars": 8,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
    ),
}

_SESSION_LIQUIDITY_VACUUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "utc_ls",
            "volume_window": 48,
            "range_window": 36,
            "volume_shock_z": 1.0,
            "range_shock_z": 0.8,
            "return_shock_pct": 0.006,
            "revert_fraction": 0.40,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.018,
            "allow_short": True,
            "session_window_minutes": 30,
        },
        {
            "variant": "utc_guarded_lo",
            "volume_window": 64,
            "range_window": 48,
            "volume_shock_z": 1.3,
            "range_shock_z": 1.0,
            "return_shock_pct": 0.008,
            "revert_fraction": 0.35,
            "max_hold_bars": 10,
            "stop_loss_pct": 0.016,
            "allow_short": False,
            "session_window_minutes": 25,
        },
    ),
}

_FUNDING_LIQUIDATION_CROWDING_FADE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "balanced_ls",
            "window": 96,
            "crowding_entry": 0.85,
            "crowding_exit": 0.25,
            "liquidation_z_min": 1.0,
            "return_shock_pct": 0.010,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 128,
            "crowding_entry": 1.00,
            "crowding_exit": 0.30,
            "liquidation_z_min": 1.2,
            "return_shock_pct": 0.012,
            "max_hold_bars": 10,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_DEEP_RESEARCH_FUNDING_DISLOCATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "balanced_ls",
            "fast_lookback_bars": 24,
            "mid_lookback_bars": 72,
            "slow_lookback_bars": 168,
            "rebalance_bars": 4,
            "signal_threshold": 0.45,
            "max_longs": 3,
            "max_shorts": 2,
            "vol_window": 48,
            "crowding_window": 72,
            "trend_weight": 0.55,
            "carry_weight": 0.25,
            "basis_weight": 0.10,
            "crowding_penalty_weight": 0.10,
            "stop_loss_pct": 0.045,
            "max_abs_exposure": 1.0,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "fast_lookback_bars": 24,
            "mid_lookback_bars": 96,
            "slow_lookback_bars": 240,
            "rebalance_bars": 6,
            "signal_threshold": 0.55,
            "max_longs": 3,
            "max_shorts": 0,
            "vol_window": 72,
            "crowding_window": 96,
            "trend_weight": 0.60,
            "carry_weight": 0.25,
            "basis_weight": 0.10,
            "crowding_penalty_weight": 0.05,
            "stop_loss_pct": 0.040,
            "max_abs_exposure": 1.0,
            "allow_short": False,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "fast_lookback_bars": 12,
            "mid_lookback_bars": 36,
            "slow_lookback_bars": 90,
            "rebalance_bars": 3,
            "signal_threshold": 0.40,
            "max_longs": 3,
            "max_shorts": 2,
            "vol_window": 36,
            "crowding_window": 54,
            "trend_weight": 0.58,
            "carry_weight": 0.24,
            "basis_weight": 0.08,
            "crowding_penalty_weight": 0.10,
            "stop_loss_pct": 0.060,
            "max_abs_exposure": 1.0,
            "allow_short": True,
        },
    ),
}

_DEEP_RESEARCH_VOL_MANAGED_MOMENTUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "balanced_ls",
            "momentum_lookback_bars": 96,
            "rebalance_bars": 4,
            "vol_window": 48,
            "target_vol": 0.018,
            "max_leverage": 1.0,
            "signal_threshold": 0.35,
            "max_longs": 3,
            "max_shorts": 2,
            "crash_window_bars": 24,
            "crash_return_pct": 0.055,
            "vol_ratio_window": 192,
            "vol_ratio_max": 2.4,
            "stress_reduce": 0.25,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "momentum_lookback_bars": 120,
            "rebalance_bars": 6,
            "vol_window": 72,
            "target_vol": 0.016,
            "max_leverage": 0.8,
            "signal_threshold": 0.45,
            "max_longs": 3,
            "max_shorts": 0,
            "crash_window_bars": 36,
            "crash_return_pct": 0.070,
            "vol_ratio_window": 240,
            "vol_ratio_max": 2.2,
            "stress_reduce": 0.15,
            "allow_short": False,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "momentum_lookback_bars": 60,
            "rebalance_bars": 3,
            "vol_window": 36,
            "target_vol": 0.026,
            "max_leverage": 1.0,
            "signal_threshold": 0.35,
            "max_longs": 3,
            "max_shorts": 2,
            "crash_window_bars": 12,
            "crash_return_pct": 0.085,
            "vol_ratio_window": 120,
            "vol_ratio_max": 2.3,
            "stress_reduce": 0.30,
            "allow_short": True,
        },
    ),
}

_DEEP_RESEARCH_FLOW_IMBALANCE_LIQUIDATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "majors_sweep_ls",
            "window": 72,
            "entry_score": 0.35,
            "exit_score": 0.08,
            "liquidation_z_min": 1.2,
            "return_shock_pct": 0.006,
            "max_spread_bps": 12.0,
            "max_hold_bars": 18,
            "stop_loss_pct": 0.016,
            "allow_short": True,
        },
        {
            "variant": "majors_guarded_lo",
            "window": 96,
            "entry_score": 0.45,
            "exit_score": 0.10,
            "liquidation_z_min": 1.5,
            "return_shock_pct": 0.008,
            "max_spread_bps": 10.0,
            "max_hold_bars": 14,
            "stop_loss_pct": 0.014,
            "allow_short": False,
        },
    ),
    "15m": (
        {
            "variant": "majors_sweep_ls",
            "window": 64,
            "entry_score": 0.35,
            "exit_score": 0.08,
            "liquidation_z_min": 1.1,
            "return_shock_pct": 0.010,
            "max_spread_bps": 14.0,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
    ),
}

_DEEP_RESEARCH_FLOW_MAJOR_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "TRX/USDT",
)

_BASIS_SNAPBACK_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "balanced_ls",
            "window": 96,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 128,
            "entry_z": 2.2,
            "exit_z": 0.35,
            "max_hold_bars": 10,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_VOL_OF_VOL_EXHAUSTION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "15m": (
        {
            "variant": "balanced_ls",
            "vol_window": 24,
            "vol_z_window": 48,
            "return_z_window": 24,
            "vol_entry_z": 1.8,
            "return_entry_z": 1.2,
            "max_hold_bars": 8,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "vol_window": 32,
            "vol_z_window": 64,
            "return_z_window": 32,
            "vol_entry_z": 2.2,
            "return_entry_z": 1.5,
            "max_hold_bars": 6,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_BREADTH_THRUST_FAILURE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "balanced_ls",
            "momentum_lookback": 16,
            "breadth_entry": 0.80,
            "breadth_exit": 0.55,
            "basket_return_floor": 0.003,
            "max_hold_bars": 8,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "momentum_lookback": 24,
            "breadth_entry": 0.85,
            "breadth_exit": 0.60,
            "basket_return_floor": 0.004,
            "max_hold_bars": 6,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_RESIDUAL_BASKET_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "15m": (
        {
            "variant": "resid_btc_ls",
            "residual_window": 48,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.020,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "resid_btc_guarded_lo",
            "residual_window": 64,
            "entry_z": 2.2,
            "exit_z": 0.35,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 0,
            "stop_loss_pct": 0.018,
            "allow_short": False,
            "btc_symbol": "BTC/USDT",
        },
    ),
}

_SESSION_GATED_RESIDUAL_BASKET_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "resid_btc_ls",
            "residual_window": 64,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.020,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
            "session_window_minutes": 30,
        },
        {
            "variant": "resid_btc_guarded_lo",
            "residual_window": 80,
            "entry_z": 2.0,
            "exit_z": 0.35,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 0,
            "stop_loss_pct": 0.018,
            "allow_short": False,
            "btc_symbol": "BTC/USDT",
            "session_window_minutes": 25,
        },
    ),
}

_VOL_REGIME_RESIDUAL_BASKET_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "15m": (
        {
            "variant": "volcap_ls",
            "residual_window": 48,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.020,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
            "btc_vol_fast": 12,
            "btc_vol_slow": 60,
            "btc_vol_ratio_cap": 1.15,
            "dispersion_floor": 0.0020,
        },
        {
            "variant": "volcap_guarded_lo",
            "residual_window": 64,
            "entry_z": 2.0,
            "exit_z": 0.35,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 0,
            "stop_loss_pct": 0.018,
            "allow_short": False,
            "btc_symbol": "BTC/USDT",
            "btc_vol_fast": 16,
            "btc_vol_slow": 72,
            "btc_vol_ratio_cap": 1.05,
            "dispersion_floor": 0.0025,
        },
    ),
}

_LIQUIDATION_CONTAGION_FADE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "balanced_ls",
            "window": 64,
            "leader_liq_z_min": 1.2,
            "return_shock_pct": 0.006,
            "exit_z": 0.3,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 96,
            "leader_liq_z_min": 1.5,
            "return_shock_pct": 0.008,
            "exit_z": 0.25,
            "max_hold_bars": 10,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_MULTI_HORIZON_TREND_EXHAUSTION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "balanced_ls",
            "short_window": 16,
            "entry_z": 1.6,
            "exit_z": 0.3,
            "max_hold_bars": 10,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "short_window": 24,
            "entry_z": 2.0,
            "exit_z": 0.25,
            "max_hold_bars": 8,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_VWAP_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "balanced_ls",
            "window": 48,
            "entry_dev": 0.012,
            "exit_dev": 0.003,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 64,
            "entry_dev": 0.016,
            "exit_dev": 0.004,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
    "15m": (
        {
            "variant": "balanced_ls",
            "window": 36,
            "entry_dev": 0.010,
            "exit_dev": 0.002,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 48,
            "entry_dev": 0.014,
            "exit_dev": 0.003,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_TOPCAP_TSMOM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "balanced",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "resid_btc",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.010,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
            "residualize_btc": True,
            "residualize_mean": False,
        },
        {
            "variant": "resid_beta_neutral",
            "lookback_bars": 24,
            "rebalance_bars": 4,
            "signal_threshold": 0.008,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
            "residualize_btc": True,
            "residualize_mean": True,
        },
        {
            "variant": "defensive",
            "lookback_bars": 24,
            "rebalance_bars": 6,
            "signal_threshold": 0.020,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 1,
            "min_price": 0.10,
            "btc_regime_ma": 64,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "crashguard",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 48,
            "benchmark_drawdown_limit": 0.08,
        },
        {
            "variant": "exec_tightstop",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.05,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_fastrebalance",
            "lookback_bars": 16,
            "rebalance_bars": 2,
            "signal_threshold": 0.012,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_takeprofit",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.10,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_tightstop_tp",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_fastrebalance_tp",
            "lookback_bars": 16,
            "rebalance_bars": 2,
            "signal_threshold": 0.012,
            "stop_loss_pct": 0.07,
            "take_profit_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
    ),
    "4h": (
        {
            "variant": "balanced",
            "lookback_bars": 10,
            "rebalance_bars": 2,
            "signal_threshold": 0.020,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 18,
            "btc_symbol": "BTC/USDT",
        },
    ),
}

_ADAPTIVE_REGIME_MOMENTUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "profit_reboot_balanced",
            "lookback_bars": 360,
            "short_lookback_bars": 24,
            "regime_lookback_bars": 360,
            "volatility_lookback_bars": 60,
            "rebalance_bars": 72,
            "signal_threshold": 0.040,
            "broad_threshold": 0.0,
            "max_longs": 1,
            "max_shorts": 2,
            "gross_exposure": 0.005,
            "max_order_value": 200.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "trailing_exit_pct": 0.0,
            "max_hold_bars": 0,
        },
        {
            "variant": "profit_reboot_defensive",
            "lookback_bars": 168,
            "short_lookback_bars": 24,
            "regime_lookback_bars": 168,
            "volatility_lookback_bars": 60,
            "rebalance_bars": 360,
            "signal_threshold": 0.080,
            "broad_threshold": 0.0,
            "max_longs": 1,
            "max_shorts": 2,
            "gross_exposure": 0.005,
            "max_order_value": 200.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "trailing_exit_pct": 0.0,
            "max_hold_bars": 0,
        },
        {
            "variant": "profit_reboot_short_bias",
            "lookback_bars": 168,
            "short_lookback_bars": 72,
            "regime_lookback_bars": 168,
            "volatility_lookback_bars": 60,
            "rebalance_bars": 360,
            "signal_threshold": 0.080,
            "broad_threshold": 0.0,
            "max_longs": 0,
            "max_shorts": 2,
            "gross_exposure": 0.005,
            "max_order_value": 200.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "trailing_exit_pct": 0.0,
            "max_hold_bars": 0,
        },
    ),
    "4h": (
        {
            "variant": "profit_reboot_slow_defensive",
            "lookback_bars": 60,
            "short_lookback_bars": 6,
            "regime_lookback_bars": 60,
            "volatility_lookback_bars": 24,
            "rebalance_bars": 6,
            "signal_threshold": 0.020,
            "broad_threshold": 0.0,
            "max_longs": 1,
            "max_shorts": 1,
            "gross_exposure": 0.005,
            "max_order_value": 200.0,
            "stop_loss_pct": 0.040,
            "take_profit_pct": 0.090,
            "trailing_exit_pct": 0.050,
            "max_hold_bars": 180,
        },
    ),
}

_PANIC_REBOUND_MEAN_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "fast_confirm",
            "history_bars": 144,
            "return_window": 18,
            "volume_window": 18,
            "vwap_window": 12,
            "shock_return_z": 1.6,
            "shock_return_pct": 0.018,
            "volume_z": 0.8,
            "confirmation_bars": 2,
            "min_rebound_pct": 0.004,
            "vwap_recovery_pct": 0.0,
            "stop_loss_pct": 0.016,
            "take_profit_pct": 0.025,
            "trailing_exit_pct": 0.014,
            "max_hold_bars": 10,
            "target_allocation": 0.06,
            "max_order_value": 250.0,
        },
        {
            "variant": "volume_strict",
            "history_bars": 192,
            "return_window": 32,
            "volume_window": 32,
            "vwap_window": 24,
            "shock_return_z": 2.1,
            "shock_return_pct": 0.025,
            "volume_z": 1.2,
            "confirmation_bars": 3,
            "min_rebound_pct": 0.006,
            "vwap_recovery_pct": 0.001,
            "stop_loss_pct": 0.018,
            "take_profit_pct": 0.035,
            "trailing_exit_pct": 0.018,
            "max_hold_bars": 18,
            "target_allocation": 0.08,
            "max_order_value": 300.0,
        },
    ),
    "15m": (
        {
            "variant": "slow_confirm",
            "history_bars": 160,
            "return_window": 24,
            "volume_window": 24,
            "vwap_window": 16,
            "shock_return_z": 1.8,
            "shock_return_pct": 0.030,
            "volume_z": 1.0,
            "confirmation_bars": 2,
            "min_rebound_pct": 0.008,
            "vwap_recovery_pct": 0.0,
            "stop_loss_pct": 0.020,
            "take_profit_pct": 0.045,
            "trailing_exit_pct": 0.020,
            "max_hold_bars": 12,
            "target_allocation": 0.08,
            "max_order_value": 300.0,
        },
    ),
}

_SESSION_FILTERED_PAIR_CARRY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "bnbtrx_overlap",
            "lookback_window": 96,
            "hedge_window": 192,
            "entry_z": 2.2,
            "exit_z": 0.50,
            "stop_z": 3.8,
            "min_correlation": 0.18,
            "max_hold_bars": 72,
            "cooldown_bars": 6,
            "reentry_z_buffer": 0.25,
            "min_z_turn": 0.02,
            "stop_loss_pct": 0.020,
            "take_profit_pct": 0.045,
            "allowed_session_utc_hours": "0,1,8,9,13,14,15,20,21",
            "min_expected_move_pct": 0.0015,
        },
        {
            "variant": "bnbtrx_strict",
            "lookback_window": 120,
            "hedge_window": 240,
            "entry_z": 2.6,
            "exit_z": 0.65,
            "stop_z": 4.2,
            "min_correlation": 0.24,
            "max_hold_bars": 48,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.30,
            "min_z_turn": 0.03,
            "stop_loss_pct": 0.018,
            "take_profit_pct": 0.040,
            "allowed_session_utc_hours": "0,1,8,9,13,14,15,20,21",
            "min_expected_move_pct": 0.0025,
        },
    ),
    "4h": (
        {
            "variant": "bnbtrx_asia_us",
            "lookback_window": 72,
            "hedge_window": 144,
            "entry_z": 1.8,
            "exit_z": 0.45,
            "stop_z": 3.3,
            "min_correlation": 0.10,
            "max_hold_bars": 36,
            "cooldown_bars": 4,
            "reentry_z_buffer": 0.18,
            "min_z_turn": 0.01,
            "stop_loss_pct": 0.018,
            "take_profit_pct": 0.035,
            "allowed_session_utc_hours": "0,4,8,12,16,20",
            "min_expected_move_pct": 0.0010,
        },
    ),
}

_COMPRESSION_BREAKOUT_CONTINUATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast",
            "lookback_bars": 32,
            "compression_window": 16,
            "compression_history_bars": 96,
            "compression_percentile": 0.30,
            "breakout_buffer": 0.003,
            "broad_lookback_bars": 12,
            "broad_threshold": 0.0,
            "stop_loss_pct": 0.020,
            "take_profit_pct": 0.045,
            "trailing_exit_pct": 0.025,
            "max_hold_bars": 36,
            "target_allocation": 0.08,
            "max_order_value": 300.0,
        },
        {
            "variant": "balanced",
            "lookback_bars": 48,
            "compression_window": 24,
            "compression_history_bars": 160,
            "compression_percentile": 0.25,
            "breakout_buffer": 0.002,
            "broad_lookback_bars": 24,
            "broad_threshold": 0.0,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.060,
            "trailing_exit_pct": 0.030,
            "max_hold_bars": 72,
            "target_allocation": 0.10,
            "max_order_value": 350.0,
        },
    ),
    "1h": (
        {
            "variant": "slow",
            "lookback_bars": 36,
            "compression_window": 18,
            "compression_history_bars": 120,
            "compression_percentile": 0.25,
            "breakout_buffer": 0.002,
            "broad_lookback_bars": 18,
            "broad_threshold": 0.0,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.070,
            "trailing_exit_pct": 0.035,
            "max_hold_bars": 60,
            "target_allocation": 0.10,
            "max_order_value": 350.0,
        },
    ),
}

_PROFIT_MOONSHOT_TREND_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "balanced",
            "lookback_bars": 60,
            "fast_lookback_bars": 12,
            "slow_lookback_bars": 240,
            "rebalance_bars": 12,
            "entry_threshold": 0.012,
            "exit_threshold": 0.002,
            "max_longs": 2,
            "max_shorts": 2,
            "gross_exposure": 0.10,
            "max_order_value": 1_000.0,
            "stop_loss_pct": 0.045,
            "take_profit_pct": 0.120,
            "trailing_exit_pct": 0.055,
            "max_hold_bars": 240,
            "breadth_threshold": -0.002,
        },
        {
            "variant": "defensive",
            "lookback_bars": 72,
            "fast_lookback_bars": 18,
            "slow_lookback_bars": 288,
            "rebalance_bars": 24,
            "entry_threshold": 0.018,
            "exit_threshold": 0.004,
            "max_longs": 1,
            "max_shorts": 1,
            "gross_exposure": 0.05,
            "max_order_value": 500.0,
            "stop_loss_pct": 0.035,
            "take_profit_pct": 0.090,
            "trailing_exit_pct": 0.040,
            "max_hold_bars": 240,
            "breadth_threshold": 0.000,
        },
    ),
    "4h": (
        {
            "variant": "slow",
            "lookback_bars": 36,
            "fast_lookback_bars": 6,
            "slow_lookback_bars": 120,
            "rebalance_bars": 6,
            "entry_threshold": 0.014,
            "exit_threshold": 0.003,
            "max_longs": 2,
            "max_shorts": 1,
            "gross_exposure": 0.08,
            "max_order_value": 800.0,
            "stop_loss_pct": 0.055,
            "take_profit_pct": 0.140,
            "trailing_exit_pct": 0.060,
            "max_hold_bars": 180,
            "breadth_threshold": -0.004,
        },
    ),
}

_PROFIT_MOONSHOT_BREAKOUT_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "expansion",
            "lookback_bars": 48,
            "fast_lookback_bars": 8,
            "slow_lookback_bars": 168,
            "rebalance_bars": 3,
            "entry_threshold": 0.006,
            "exit_threshold": 0.002,
            "max_longs": 2,
            "max_shorts": 2,
            "gross_exposure": 0.10,
            "max_order_value": 1_000.0,
            "stop_loss_pct": 0.040,
            "take_profit_pct": 0.110,
            "trailing_exit_pct": 0.050,
            "max_hold_bars": 144,
            "breakout_buffer": 0.002,
            "squeeze_ratio_max": 1.60,
            "volume_z_min": -0.25,
        },
    ),
    "4h": (
        {
            "variant": "slow_expansion",
            "lookback_bars": 30,
            "fast_lookback_bars": 5,
            "slow_lookback_bars": 90,
            "rebalance_bars": 2,
            "entry_threshold": 0.008,
            "exit_threshold": 0.003,
            "max_longs": 2,
            "max_shorts": 1,
            "gross_exposure": 0.08,
            "max_order_value": 800.0,
            "stop_loss_pct": 0.050,
            "take_profit_pct": 0.140,
            "trailing_exit_pct": 0.060,
            "max_hold_bars": 96,
            "breakout_buffer": 0.003,
            "squeeze_ratio_max": 1.50,
            "volume_z_min": -0.10,
        },
    ),
}

_PROFIT_MOONSHOT_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "shock_fade",
            "lookback_bars": 36,
            "fast_lookback_bars": 4,
            "slow_lookback_bars": 96,
            "rebalance_bars": 2,
            "entry_threshold": 0.75,
            "exit_threshold": 0.25,
            "max_longs": 2,
            "max_shorts": 2,
            "gross_exposure": 0.07,
            "max_order_value": 700.0,
            "stop_loss_pct": 0.030,
            "take_profit_pct": 0.060,
            "trailing_exit_pct": 0.030,
            "max_hold_bars": 72,
            "return_z_min": 1.20,
            "volume_z_min": 0.25,
            "range_z_min": 0.25,
        },
    ),
    "4h": (
        {
            "variant": "slow_shock_fade",
            "lookback_bars": 24,
            "fast_lookback_bars": 3,
            "slow_lookback_bars": 60,
            "rebalance_bars": 1,
            "entry_threshold": 0.85,
            "exit_threshold": 0.30,
            "max_longs": 2,
            "max_shorts": 1,
            "gross_exposure": 0.06,
            "max_order_value": 600.0,
            "stop_loss_pct": 0.035,
            "take_profit_pct": 0.080,
            "trailing_exit_pct": 0.035,
            "max_hold_bars": 48,
            "return_z_min": 1.30,
            "volume_z_min": 0.20,
            "range_z_min": 0.20,
        },
    ),
}

_LAST_DAY_LIQUIDITY_REGIME_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "liquid_momo_ls",
            "momentum_lookback_bars": 24,
            "signal_skip_bars": 1,
            "liquidity_window": 24,
            "volatility_window": 24,
            "rebalance_bars": 6,
            "signal_threshold": 0.012,
            "liquidity_quantile": 0.60,
            "max_longs": 2,
            "max_shorts": 1,
            "min_price": 0.10,
            "max_realized_vol": 0.09,
            "stop_loss_pct": 0.05,
            "allow_short": True,
            "illiquid_reversal": True,
        },
        {
            "variant": "guarded_lo",
            "momentum_lookback_bars": 24,
            "signal_skip_bars": 1,
            "liquidity_window": 36,
            "volatility_window": 24,
            "rebalance_bars": 12,
            "signal_threshold": 0.015,
            "liquidity_quantile": 0.70,
            "max_longs": 2,
            "max_shorts": 0,
            "min_price": 0.10,
            "max_realized_vol": 0.07,
            "stop_loss_pct": 0.04,
            "allow_short": False,
            "illiquid_reversal": False,
        },
    ),
    "1d": (
        {
            "variant": "liquid_momo_ls",
            "momentum_lookback_bars": 1,
            "signal_skip_bars": 1,
            "liquidity_window": 20,
            "volatility_window": 20,
            "rebalance_bars": 1,
            "signal_threshold": 0.008,
            "liquidity_quantile": 0.60,
            "max_longs": 2,
            "max_shorts": 1,
            "min_price": 0.10,
            "max_realized_vol": 0.15,
            "stop_loss_pct": 0.08,
            "allow_short": True,
            "illiquid_reversal": True,
        },
        {
            "variant": "guarded_lo",
            "momentum_lookback_bars": 1,
            "signal_skip_bars": 1,
            "liquidity_window": 20,
            "volatility_window": 20,
            "rebalance_bars": 1,
            "signal_threshold": 0.006,
            "liquidity_quantile": 0.70,
            "max_longs": 2,
            "max_shorts": 0,
            "min_price": 0.10,
            "max_realized_vol": 0.12,
            "stop_loss_pct": 0.07,
            "allow_short": False,
            "illiquid_reversal": False,
        },
    ),
}

_ABNORMAL_RETURN_CONTINUATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "event_ls",
            "return_z_window": 20,
            "entry_z": 1.4,
            "exit_z": 0.25,
            "hold_bars": 2,
            "stop_loss_pct": 0.06,
            "allow_short": True,
        },
        {
            "variant": "event_lo",
            "return_z_window": 24,
            "entry_z": 1.8,
            "exit_z": 0.35,
            "hold_bars": 1,
            "stop_loss_pct": 0.05,
            "allow_short": False,
        },
    ),
}

_CARRY_TREND_FACTOR_ROTATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "balanced_lo",
            "lookback_bars": 24,
            "rebalance_bars": 8,
            "signal_threshold": 0.20,
            "stop_loss_pct": 0.06,
            "max_longs": 3,
            "max_shorts": 0,
            "min_price": 0.10,
            "btc_regime_ma": 72,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 48,
            "benchmark_drawdown_limit": 0.08,
            "vol_window": 48,
            "crowding_window": 72,
            "trend_weight": 0.55,
            "carry_weight": 0.20,
            "defensive_weight": 0.15,
            "crowding_weight": 0.10,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "lookback_bars": 32,
            "rebalance_bars": 8,
            "signal_threshold": 0.15,
            "stop_loss_pct": 0.05,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 96,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 72,
            "benchmark_drawdown_limit": 0.10,
            "vol_window": 72,
            "crowding_window": 96,
            "trend_weight": 0.45,
            "carry_weight": 0.20,
            "defensive_weight": 0.20,
            "crowding_weight": 0.15,
            "allow_short": True,
        },
        {
            "variant": "production_lo_guarded",
            "lookback_bars": 48,
            "rebalance_bars": 12,
            "signal_threshold": 0.25,
            "stop_loss_pct": 0.045,
            "max_longs": 2,
            "max_shorts": 0,
            "min_price": 0.10,
            "btc_regime_ma": 120,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 96,
            "benchmark_drawdown_limit": 0.06,
            "vol_window": 96,
            "crowding_window": 120,
            "trend_weight": 0.50,
            "carry_weight": 0.15,
            "defensive_weight": 0.25,
            "crowding_weight": 0.10,
            "allow_short": False,
            "production_ready": True,
        },
    ),
    "4h": (
        {
            "variant": "balanced_lo",
            "lookback_bars": 12,
            "rebalance_bars": 3,
            "signal_threshold": 0.20,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 0,
            "min_price": 0.10,
            "btc_regime_ma": 24,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 18,
            "benchmark_drawdown_limit": 0.10,
            "vol_window": 24,
            "crowding_window": 36,
            "trend_weight": 0.50,
            "carry_weight": 0.20,
            "defensive_weight": 0.20,
            "crowding_weight": 0.10,
            "allow_short": False,
        },
        {
            "variant": "carry_guarded_ls",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.15,
            "stop_loss_pct": 0.06,
            "max_longs": 2,
            "max_shorts": 1,
            "min_price": 0.10,
            "btc_regime_ma": 24,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 18,
            "benchmark_drawdown_limit": 0.10,
            "vol_window": 24,
            "crowding_window": 36,
            "trend_weight": 0.40,
            "carry_weight": 0.25,
            "defensive_weight": 0.20,
            "crowding_weight": 0.15,
            "allow_short": True,
        },
        {
            "variant": "production_lo_trendcarry",
            "lookback_bars": 24,
            "rebalance_bars": 6,
            "signal_threshold": 0.20,
            "stop_loss_pct": 0.05,
            "max_longs": 2,
            "max_shorts": 0,
            "min_price": 0.10,
            "btc_regime_ma": 36,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 24,
            "benchmark_drawdown_limit": 0.08,
            "vol_window": 36,
            "crowding_window": 48,
            "trend_weight": 0.45,
            "carry_weight": 0.20,
            "defensive_weight": 0.25,
            "crowding_weight": 0.10,
            "allow_short": False,
            "production_ready": True,
        },
    ),
}

_ALPHA101_SIGNAL_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "a005_vwap_tuned",
            "alpha_id": 5,
            "rank_window": 24,
            "history_window": 144,
            "score_window": 64,
            "entry_z": 1.15,
            "exit_z": 0.30,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": True,
            "alpha_param_overrides": {
                "alpha101.5.const.001": 8.0,
                "alpha101.5.const.002": 14.0,
            },
        },
        {
            "variant": "a011_flow_tuned",
            "alpha_id": 11,
            "rank_window": 24,
            "history_window": 144,
            "score_window": 64,
            "entry_z": 1.10,
            "exit_z": 0.30,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": True,
            "alpha_param_overrides": {
                "alpha101.11.const.001": 4.0,
                "alpha101.11.const.002": 4.0,
                "alpha101.11.const.003": 5.0,
            },
        },
        {
            "variant": "a017_turn_tuned",
            "alpha_id": 17,
            "rank_window": 20,
            "history_window": 160,
            "score_window": 64,
            "entry_z": 1.00,
            "exit_z": 0.25,
            "signal_sign": -1.0,
            "stop_loss_pct": 0.03,
            "allow_short": False,
            "alpha_param_overrides": {
                "alpha101.17.const.001": 12.0,
                "alpha101.17.const.002": 7.0,
            },
        },
        {
            "variant": "a101_bodyrange_tuned",
            "alpha_id": 101,
            "rank_window": 20,
            "history_window": 96,
            "score_window": 48,
            "entry_z": 1.00,
            "exit_z": 0.25,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": False,
            "alpha_param_overrides": {
                "alpha101.101.const.001": 0.01,
            },
        },
    ),
    "4h": (
        {
            "variant": "a011_flow_swing",
            "alpha_id": 11,
            "rank_window": 20,
            "history_window": 96,
            "score_window": 32,
            "entry_z": 1.05,
            "exit_z": 0.25,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.035,
            "allow_short": True,
            "alpha_param_overrides": {
                "alpha101.11.const.001": 5.0,
                "alpha101.11.const.002": 5.0,
                "alpha101.11.const.003": 4.0,
            },
        },
        {
            "variant": "a101_bodyrange_swing",
            "alpha_id": 101,
            "rank_window": 16,
            "history_window": 80,
            "score_window": 24,
            "entry_z": 0.90,
            "exit_z": 0.20,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.035,
            "allow_short": False,
            "alpha_param_overrides": {
                "alpha101.101.const.001": 0.02,
            },
        },
    ),
}

_VOLCOMP_RETUNE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "guarded_lo_core",
            "vwap_window": 96,
            "z_window": 192,
            "entry_z": 2.2,
            "exit_z": 0.18,
            "compression_percentile": 0.12,
            "compression_vol_ratio": 0.72,
            "atr_stop_pct": 0.012,
            "max_hold_bars": 24,
            "allow_short": False,
        },
        {
            "variant": "guarded_lo_strict",
            "vwap_window": 120,
            "z_window": 240,
            "entry_z": 2.6,
            "exit_z": 0.15,
            "compression_percentile": 0.10,
            "compression_vol_ratio": 0.68,
            "atr_stop_pct": 0.010,
            "max_hold_bars": 18,
            "allow_short": False,
        },
    ),
    "15m": (
        {
            "variant": "guarded_lo_core",
            "vwap_window": 72,
            "z_window": 168,
            "entry_z": 2.0,
            "exit_z": 0.22,
            "compression_percentile": 0.16,
            "compression_vol_ratio": 0.78,
            "atr_stop_pct": 0.016,
            "max_hold_bars": 36,
            "allow_short": False,
        },
        {
            "variant": "guarded_lo_strict",
            "vwap_window": 96,
            "z_window": 192,
            "entry_z": 2.4,
            "exit_z": 0.18,
            "compression_percentile": 0.12,
            "compression_vol_ratio": 0.72,
            "atr_stop_pct": 0.014,
            "max_hold_bars": 28,
            "allow_short": False,
        },
    ),
}


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """Serializable strategy-candidate definition."""

    candidate_id: str
    name: str
    family: str
    strategy_class: str
    timeframe: str
    symbols: tuple[str, ...]
    params: dict[str, Any]
    notes: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        timeframe = str(self.timeframe)
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "family": self.family,
            "strategy_class": self.strategy_class,
            "strategy": self.strategy_class,
            "strategy_timeframe": timeframe,
            # Legacy alias retained for compatibility.
            "timeframe": timeframe,
            "symbols": list(self.symbols),
            "params": dict(self.params),
            "notes": self.notes,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


def _normalize_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(canonicalize_symbol_list(values))


def _candidate_id(
    *, name: str, timeframe: str, params: dict[str, Any], symbols: tuple[str, ...]
) -> str:
    payload = {
        "name": name,
        "timeframe": str(timeframe),
        "params": params,
        "symbols": list(symbols),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _article_pipeline_family_ids(
    *,
    strategy_class: str,
    timeframe: str,
    symbols: Sequence[str],
) -> tuple[str, ...]:
    symbol_set = set(canonicalize_symbol_list(symbols))
    strategy_token = str(strategy_class or "").strip()
    timeframe_token = str(timeframe or "").strip()
    if strategy_token == "CompositeTrendStrategy":
        return ("regime-conditioned-composite-trend",)
    if strategy_token == "VolCompressionVWAPReversionStrategy":
        return ("vol-compression-break-reversion",)
    if strategy_token == "LeadLagSpilloverStrategy":
        return ("lead-lag-regime-spillover",)
    if strategy_token == "LagConvergenceStrategy":
        return ("metals-lag-convergence",) if symbol_set.intersection(_METALS) else ()
    if strategy_token == "TopCapTimeSeriesMomentumStrategy":
        return ("topcap-rotation-relative-momentum",)
    if strategy_token == "AdaptiveRegimeMomentumStrategy":
        return ("profit-reboot-adaptive-regime-momentum",)
    if strategy_token == "PanicReboundMeanReversionStrategy":
        return ("profit-reboot-panic-rebound-mean-reversion",)
    if strategy_token == "SessionFilteredPairCarryStrategy":
        return ("profit-reboot-session-filtered-pair-carry",)
    if strategy_token == "CompressionBreakoutContinuationStrategy":
        return ("profit-reboot-compression-breakout-continuation",)
    if strategy_token == "ProfitMoonshotTrendStrategy":
        return ("profit-moonshot-cross-sectional-trend",)
    if strategy_token == "ProfitMoonshotBreakoutStrategy":
        return ("profit-moonshot-range-expansion-breakout",)
    if strategy_token == "ProfitMoonshotReversionStrategy":
        return ("profit-moonshot-shock-reversion",)
    if strategy_token == "CarryTrendFactorRotationStrategy":
        return ("carry-trend-factor-rotation",)
    if strategy_token == "Alpha101FormulaStrategy":
        return ("formulaic-alpha101-research",)
    if strategy_token in {"RollingBreakoutStrategy", "RegimeBreakoutCandidateStrategy"}:
        return ("regime-breakout-thrust",)
    if strategy_token == "MeanReversionStdStrategy":
        return ("single-asset-zscore-reversion",)
    if strategy_token == "LiquidityShockReversionStrategy":
        return ("liquidity-shock-reversion",)
    if strategy_token == "SessionLiquidityVacuumFadeStrategy":
        return ("session-transition-liquidity-vacuum-fade",)
    if strategy_token == "FundingLiquidationCrowdingFadeStrategy":
        return ("funding-liquidation-crowding-fade",)
    if strategy_token == "FundingDislocationTrendCarryStrategy":
        return ("deep-research-funding-dislocation-trend-carry",)
    if strategy_token == "VolManagedMomentumCrashGateStrategy":
        return ("deep-research-vol-managed-momentum-crash-gate",)
    if strategy_token == "FlowImbalanceLiquidationSweepStrategy":
        return ("deep-research-flow-imbalance-liquidation-sweep",)
    if strategy_token == "BasisSnapbackReversionStrategy":
        return ("basis-snapback-reversion",)
    if strategy_token == "VolOfVolExhaustionFadeStrategy":
        return ("vol-of-vol-exhaustion-fade",)
    if strategy_token == "VwapReversionStrategy":
        return ("intraday-vwap-reversion",)
    if strategy_token == "BreadthThrustFailureReversalStrategy":
        return ("breadth-thrust-failure-reversal",)
    if strategy_token == "ResidualBasketReversionStrategy":
        return ("cross-sectional-residual-basket-reversion",)
    if strategy_token == "SessionGatedResidualBasketReversionStrategy":
        return ("session-gated-residual-basket-reversion",)
    if strategy_token == "CrossAssetLiquidationContagionFadeStrategy":
        return ("cross-asset-liquidation-contagion-fade",)
    if strategy_token == "MultiHorizonTrendExhaustionFadeStrategy":
        return ("multi-horizon-trend-exhaustion-fade",)
    if strategy_token == "PairSpreadZScoreStrategy":
        if symbol_set.intersection(_METALS):
            return ("crypto-metal-residual-pairs",)
        if (
            timeframe_token in {"15m", "30m", "1h"}
            and symbol_set
            and symbol_set.isdisjoint(_METALS)
        ):
            return ("sector-dispersion-reversion",)
    return ()


def _with_article_pipeline_provenance(
    *,
    strategy_class: str,
    timeframe: str,
    symbols: Sequence[str],
    tags: Sequence[str] | None,
    metadata: dict[str, Any] | None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    merged_metadata = dict(metadata or {})
    family_ids = list(
        dict.fromkeys(
            [
                str(item).strip()
                for item in list(merged_metadata.get("article_pipeline_family_ids") or [])
                if str(item).strip()
            ]
            + list(
                _article_pipeline_family_ids(
                    strategy_class=strategy_class,
                    timeframe=timeframe,
                    symbols=symbols,
                )
            )
        )
    )
    merged_tags = [str(tag).strip() for tag in list(tags or []) if str(tag).strip()]
    if family_ids:
        merged_tags.extend(["article_pipeline", *[f"article_family:{item}" for item in family_ids]])
        merged_metadata["article_pipeline_family_ids"] = list(family_ids)
        merged_metadata["hypothesis_origin"] = "article_research_pipeline"
    return tuple(dict.fromkeys(merged_tags)), merged_metadata


def _has_perp_support_data() -> bool:
    candidates: list[Path] = []
    market_data_root = current_research_market_data_settings().get(
        "parquet_root",
        "data/market_parquet",
    )

    for raw in (
        market_data_root,
        os.getenv("LQ__STORAGE__MARKET_DATA_PARQUET_PATH", ""),
        os.getenv("LQ_MARKET_PARQUET_PATH", ""),
        "data/market_parquet",
    ):
        token = str(raw or "").strip()
        if not token:
            continue
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        candidates.append(path / "feature_points")

    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        candidates.append(parent / "data" / "market_parquet" / "feature_points")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return True
    return False


def _add_candidate(
    out: list[StrategyCandidate],
    *,
    name: str,
    family: str,
    strategy_class: str,
    timeframe: str,
    symbols: Sequence[str],
    params: dict[str, Any],
    notes: str,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    symbol_tuple = tuple(canonicalize_symbol_list(symbols))
    if not symbol_tuple:
        return
    normalized_tags, enriched_metadata = _with_article_pipeline_provenance(
        strategy_class=strategy_class,
        timeframe=str(timeframe),
        symbols=symbol_tuple,
        tags=tags,
        metadata=metadata,
    )
    metadata_payload = {
        "timeframe": str(timeframe),
        "family": str(family),
        **enriched_metadata,
    }
    out.append(
        StrategyCandidate(
            candidate_id=_candidate_id(
                name=name,
                timeframe=timeframe,
                params=params,
                symbols=symbol_tuple,
            ),
            name=name,
            family=family,
            strategy_class=strategy_class,
            timeframe=str(timeframe),
            symbols=symbol_tuple,
            params=dict(params),
            notes=notes,
            tags=normalized_tags,
            metadata=metadata_payload,
        )
    )


def _pairs_in_universe(symbols: Sequence[str]) -> list[tuple[str, str]]:
    universe = set(symbols)
    out: list[tuple[str, str]] = []
    for left, right in _PAIR_ANCHORS:
        if left in universe and right in universe:
            out.append((left, right))
    return out


def _build_alpha101_formula_params(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "alpha_id": int(spec["alpha_id"]),
        "rank_window": int(spec["rank_window"]),
        "history_window": int(spec["history_window"]),
        "score_window": int(spec["score_window"]),
        "entry_z": float(spec["entry_z"]),
        "exit_z": float(spec["exit_z"]),
        "signal_sign": float(spec["signal_sign"]),
        "stop_loss_pct": float(spec["stop_loss_pct"]),
        "allow_short": bool(spec["allow_short"]),
        "alpha_param_overrides": dict(spec["alpha_param_overrides"]),
    }


def _add_alpha101_formula_candidates(
    out: list[StrategyCandidate],
    *,
    timeframes: Sequence[str],
    symbols: Sequence[str],
) -> None:
    for timeframe in timeframes:
        tf_tag = timeframe.replace("/", "-")
        for spec in _ALPHA101_SIGNAL_SLICE.get(timeframe, ()):
            signal_sign = float(spec["signal_sign"])
            alpha_param_overrides = dict(spec["alpha_param_overrides"])
            direction_tag = "inv" if signal_sign < 0.0 else "dir"
            _add_candidate(
                out,
                name=(
                    f"alpha101_formula_{tf_tag}_a{int(spec['alpha_id']):03d}_{spec['variant']}_{direction_tag}"
                ),
                family="formulaic_alpha",
                strategy_class="Alpha101FormulaStrategy",
                timeframe=timeframe,
                symbols=symbols,
                params=_build_alpha101_formula_params(spec),
                notes=(
                    "Single-asset Alpha101 factor sleeve with explicit constant overrides "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("alpha101", "formulaic", "single_asset", "factor"),
                metadata={
                    "timeframe": timeframe,
                    "alpha_id": int(spec["alpha_id"]),
                    "signal_sign": signal_sign,
                    "allow_short": bool(spec["allow_short"]),
                    "alpha_param_override_keys": sorted(alpha_param_overrides.keys()),
                    "retune_profile": str(spec["variant"]),
                },
            )


@dataclass
class _CandidateBuildContext:
    normalized_timeframes: tuple[str, ...]
    normalized_symbols: tuple[str, ...]
    candidates: list[StrategyCandidate] = field(default_factory=list)
    pairs: list[tuple[str, str]] = field(init=False)
    trend_tfs: list[str] = field(init=False)
    mean_rev_tfs: list[str] = field(init=False)
    std_mean_rev_tfs: list[str] = field(init=False)
    liquidity_tfs: list[str] = field(init=False)
    panic_rebound_tfs: list[str] = field(init=False)
    session_liquidity_tfs: list[str] = field(init=False)
    funding_crowding_tfs: list[str] = field(init=False)
    basis_snapback_tfs: list[str] = field(init=False)
    vol_of_vol_tfs: list[str] = field(init=False)
    session_residual_tfs: list[str] = field(init=False)
    contagion_tfs: list[str] = field(init=False)
    breakout_tfs: list[str] = field(init=False)
    compression_breakout_tfs: list[str] = field(init=False)
    breadth_tfs: list[str] = field(init=False)
    trend_exhaustion_tfs: list[str] = field(init=False)
    topcap_tfs: list[str] = field(init=False)
    liquidity_regime_tfs: list[str] = field(init=False)
    abnormal_return_tfs: list[str] = field(init=False)
    alpha101_tfs: list[str] = field(init=False)
    pair_tfs: list[str] = field(init=False)
    session_pair_carry_tfs: list[str] = field(init=False)
    residual_basket_tfs: list[str] = field(init=False)
    lag_convergence_tfs: list[str] = field(init=False)
    carry_tfs: list[str] = field(init=False)
    micro_tfs: list[str] = field(init=False)
    crypto_symbols: list[str] = field(init=False)
    crypto_only_symbols: list[str] = field(init=False)
    laggard_symbols: list[str] = field(init=False)
    perp_support_data_available: bool = field(init=False)

    def __post_init__(self) -> None:
        self.pairs = list(_pairs_in_universe(self.normalized_symbols))
        self.trend_tfs = self._present("30m", "1h")
        self.mean_rev_tfs = self._present("5m", "15m")
        self.std_mean_rev_tfs = self._present("15m", "30m")
        self.liquidity_tfs = self._present("5m", "15m")
        self.panic_rebound_tfs = self._present("5m", "15m")
        self.session_liquidity_tfs = self._present("5m")
        self.funding_crowding_tfs = self._present("30m")
        self.basis_snapback_tfs = self._present("30m")
        self.vol_of_vol_tfs = self._present("15m")
        self.session_residual_tfs = self._present("5m")
        self.contagion_tfs = self._present("5m")
        self.breakout_tfs = self._present("30m", "1h")
        self.compression_breakout_tfs = self._present("30m", "1h")
        self.breadth_tfs = self._present("30m")
        self.trend_exhaustion_tfs = self._present("30m")
        self.topcap_tfs = self._present("1h", "4h")
        self.liquidity_regime_tfs = self._present("1h", "1d")
        self.abnormal_return_tfs = self._present("1d")
        self.alpha101_tfs = self._present("1h", "4h")
        self.pair_tfs = self._present("15m", "30m", "1h", "4h", "1d")
        self.session_pair_carry_tfs = self._present("1h", "4h")
        self.residual_basket_tfs = self._present("15m")
        self.lag_convergence_tfs = self._present("4h", "1d")
        self.carry_tfs = self._present("30m", "1h", "4h")
        self.micro_tfs = self._present("1s")
        self.crypto_symbols = [
            symbol for symbol in self.normalized_symbols if symbol not in _METALS
        ]
        # Crypto-ONLY: drop every tradfi perp (equity/ETF/commodity/premarket) so
        # the per-symbol crypto riders never trade equity/ETF perps (which would
        # both leak tradfi into a crypto sleeve and double-trade the dedicated
        # equity single-name rider). Tradfi perps are routed through the explicit
        # equity/ETF builders that intersect named research-universe constants.
        _tradfi_perps = set(canonicalize_symbol_list(BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS))
        self.crypto_only_symbols = [
            symbol for symbol in self.crypto_symbols if symbol not in _tradfi_perps
        ]
        self.laggard_symbols = [
            symbol for symbol in self.crypto_symbols if symbol not in _CRYPTO_LEADERS
        ]
        self.perp_support_data_available = _has_perp_support_data()

    def _present(self, *choices: str) -> list[str]:
        return [tf for tf in choices if tf in self.normalized_timeframes]

    def build(self) -> list[StrategyCandidate]:
        _build_primary_trend_candidates(self)
        _build_core_mean_reversion_candidates(self)
        _build_intraday_alpha_candidates(self)
        _build_cross_sectional_rotation_candidates(self)
        _build_cross_asset_mean_reversion_candidates(self)
        _build_formula_and_breadth_candidates(self)
        _build_breakout_candidates(self)
        _build_pair_and_intermarket_candidates(self)
        _build_deep_research_report_candidates(self)
        _build_optional_carry_and_micro_candidates(self)
        # New decorrelated alpha sleeves (Pass-1 live universe + dormant tranche).
        _build_hurst_regime_gated_candidates(self)
        _build_confidence_gated_trend_candidates(self)
        # Per-symbol directional RETURN-RIDER sleeves (>=30m only; ride winners
        # with ATR trailing stops + pyramiding for high compound return).
        _build_adaptive_trend_rider_candidates(self)
        _build_volatility_breakout_rider_candidates(self)
        _build_acceleration_rider_candidates(self)
        # Micro-signal-informed sleeves: LOOK at intrabar 1s tape / perp features
        # but DECIDE at 30m (class-pinned decision_cadence_seconds=1800). Core
        # signals are computed from accumulated >=30m decision bars; the last
        # window's 1s tape is only a fresh micro confirm at the decision instant.
        _build_intraday_flow_pressure_rider_candidates(self)
        _build_vol_of_vol_regime_trend_gate_candidates(self)
        _build_vwap_compression_reversion_candidates(self)
        # Aggressive per-symbol directional RETURN-maximizing sleeves (>=30m only;
        # multi-horizon trend agreement, buy-the-dip continuation, funding-carry
        # harvest — ride winners with ATR trailing stops + pyramiding).
        _build_multi_timeframe_trend_ensemble_candidates(self)
        _build_pullback_trend_continuation_candidates(self)
        _build_funding_harvest_carry_candidates(self)
        # Carry-trend CONFLUENCE rider (per-symbol single-asset, crypto-perp): only
        # rides a trend when the funding/carry sign AGREES with the trend sign, and
        # the volatility-SQUEEZE breakout rider (per-symbol single-asset, OHLCV):
        # a low-vol contraction precondition gating a directional breakout ride.
        # Both >=30m only; single-asset so they bypass the multi-asset gate.
        _build_carry_trend_confluence_rider_candidates(self)
        _build_volatility_squeeze_breakout_rider_candidates(self)
        # Session opening-range-breakout rider (per-symbol single-asset, OHLCV):
        # a SESSION-anchored opening range (reset each UTC day) arms a one-shot
        # breakout; and the open-interest trend-confirmation rider (per-symbol
        # single-asset, crypto-perp): rides a trend ONLY when rising OI confirms
        # it (fresh money, not short-covering). Both >=30m only; single-asset so
        # they bypass the multi-asset gate. The OI sleeve is perp-gated.
        _build_opening_range_breakout_rider_candidates(self)
        _build_open_interest_trend_confirmation_rider_candidates(self)
        _build_intraday_seasonal_momentum_rider_candidates(self)
        _build_overnight_session_return_rider_candidates(self)
        _build_kalman_trend_rider_candidates(self)
        _build_realized_semivariance_trend_rider_candidates(self)
        _build_permutation_entropy_trend_rider_candidates(self)
        _build_amihud_illiquidity_momentum_rider_candidates(self)
        _build_cusum_change_point_trend_rider_candidates(self)
        _build_variance_ratio_trend_rider_candidates(self)
        _build_metals_relative_value_basket_candidates(self)
        _build_liquidation_cascade_reversion_candidates(self)
        _build_orderbook_imbalance_reversion_candidates(self)
        _build_selection_gated_momentum_candidates(self)
        _build_selection_gated_reversion_candidates(self)
        _build_cross_sectional_equity_momentum_candidates(self)
        _build_residual_equity_momentum_candidates(self)
        _build_betting_against_beta_candidates(self)
        _build_semis_leadlag_rotation_candidates(self)
        _build_dual_momentum_index_rotation_candidates(self)
        # Directional, long-biased EQUITY/ETF return sleeves (S-EQ1/2/3): ride
        # secular single-name winners, time leveraged ETFs above-trend with decay-
        # aware sizing, and dual-momentum-rotate with a long defensive leg. Each
        # targets the equity/ETF universe cleanly via _intersect_universe (never
        # ctx.crypto_symbols) so no crypto leaks in.
        _build_equity_single_name_trend_rider_candidates(self)
        # Commodity/macro MANAGED-FUTURES trend riders (S-CMDY1/2/3): REUSE the
        # three proven return-rider classes routed to the 8 commodity perps via
        # _intersect_universe(_COMMODITY_TREND_UNIVERSE) (never ctx.crypto_symbols).
        # Long AND short — commodities trend strongly both ways. 4h + 1d only.
        _build_commodity_adaptive_trend_rider_candidates(self)
        _build_commodity_breakout_rider_candidates(self)
        _build_commodity_acceleration_rider_candidates(self)
        # Equity 52-WEEK-HIGH breakout momentum rider (S-EQ-52WH): REUSE
        # VolatilityBreakoutRiderStrategy with a long-only ~252-bar new-high
        # Donchian lookback (George/Hwang 52-week-high momentum). 1d primary + 4h.
        _build_equity_new_high_breakout_rider_candidates(self)
        _build_leveraged_trend_timing_candidates(self)
        _build_dual_momentum_defensive_candidates(self)
        _build_calendar_seasonality_overlay_candidates(self)
        # Cross-sectional anomaly sleeves (decorrelated factor families).
        _build_idiosyncratic_volatility_candidates(self)
        _build_lottery_skewness_candidates(self)
        _build_trend_efficiency_momentum_candidates(self)
        _build_dispersion_conditioned_reversion_candidates(self)
        # Coordinated cross-asset managed-futures trend book (inverse-vol risk
        # parity + portfolio vol-targeting over the FULL universe) and the
        # commodity->equity intermarket lead-lag continuation sleeve (oil->energy
        # ETF). Both >=30m (4h + 1d) and admission-safe cross_sectional.
        _build_cross_asset_diversified_trend_candidates(self)
        _build_intermarket_leadlag_continuation_candidates(self)
        # Realized-vol-term-structure recovery rider (per-symbol single-asset) and
        # the breadth-gated total-exposure trend timer (cross_sectional basket).
        # Both >=30m only; the RV-term sleeve keys on the RV_s/RV_l ratio itself
        # and the breadth timer gates TOTAL exposure on cross-sectional breadth.
        _build_realized_vol_term_structure_candidates(self)
        _build_breadth_regime_trend_timer_candidates(self)
        return self.candidates


def _build_primary_trend_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    normalized_symbols = ctx.normalized_symbols
    trend_tfs = ctx.trend_tfs
    # Primary trend sleeve (RG_PVTM) with explicit 30m/1h OOS-stability retune only.
    for timeframe in trend_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _COMPOSITE_TREND_OOS_STABILITY_SLICE.get(timeframe, ()):
            params = {
                "long_threshold": float(spec["long_threshold"]),
                "short_threshold": float(spec["short_threshold"]),
                "exit_score_cross": float(spec["exit_score_cross"]),
                "te_min": float(spec["te_min"]),
                "vr_min": float(spec["vr_min"]),
                "chop_max": float(spec["chop_max"]),
                "vol_window": int(spec["vol_window"]),
                "risk_target_vol": float(spec["risk_target_vol"]),
                "max_signal_strength": float(spec["max_signal_strength"]),
                "atr_stop_mult": float(spec["atr_stop_mult"]),
                "trail_atr_mult": float(spec["trail_atr_mult"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "crowding_reduce_threshold": float(spec["crowding_reduce_threshold"]),
                "crowding_block_threshold": float(spec["crowding_block_threshold"]),
                "allow_short": bool(spec["allow_short"]),
            }
            if "benchmark_regime_ma" in spec:
                params["benchmark_regime_ma"] = int(spec["benchmark_regime_ma"])
            if "benchmark_symbol" in spec:
                params["benchmark_symbol"] = str(spec["benchmark_symbol"])
            regime_tag = "ls" if bool(spec["allow_short"]) else "lo"
            tags = ["trend", "trend-following", "momentum", "oos-stability"]
            note_suffix = ""
            if int(spec.get("benchmark_regime_ma", 0) or 0) > 0:
                tags.append("crash_aware")
                note_suffix = (
                    f" Crash-aware long gate uses {spec.get('benchmark_symbol', 'BTC/USDT')} "
                    f"vs {int(spec['benchmark_regime_ma'])}-bar MA."
                )
            if "exec_" in str(spec.get("variant") or ""):
                tags.append("execution_risk")
                note_suffix = (
                    f"{note_suffix} Execution-risk retune."
                    if note_suffix
                    else " Execution-risk retune."
                )
            _add_candidate(
                candidates,
                name=(
                    "composite_trend_stable_"
                    f"{tf_tag}_{spec['variant']}_{regime_tag}_"
                    f"{float(spec['long_threshold']):.2f}_{float(spec['short_threshold']):.2f}_"
                    f"{float(spec['te_min']):.2f}_{float(spec['vr_min']):.2f}"
                ),
                family="trend",
                strategy_class="CompositeTrendStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Primary RG_PVTM trend sleeve with bounded 30m/1h OOS-stability retune "
                    f"({spec['variant']}, {'long-only' if not bool(spec['allow_short']) else 'long/short'})."
                    f"{note_suffix}"
                ),
                tags=tuple(tags),
                metadata={
                    "timeframe": timeframe,
                    "regime": "ls" if bool(spec["allow_short"]) else "lo",
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "benchmark_regime_ma": int(spec.get("benchmark_regime_ma", 0) or 0),
                    "benchmark_symbol": str(spec.get("benchmark_symbol") or ""),
                },
            )


def _build_vwap_mean_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    normalized_symbols = ctx.normalized_symbols
    mean_rev_tfs = ctx.mean_rev_tfs

    # Vol-compression VWAP reversion sleeve.
    for timeframe in mean_rev_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _VOLCOMP_RETUNE_SLICE.get(timeframe, ()):
            params = {
                "vwap_window": int(spec["vwap_window"]),
                "z_window": int(spec["z_window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "compression_percentile": float(spec["compression_percentile"]),
                "compression_vol_ratio": float(spec["compression_vol_ratio"]),
                "atr_stop_pct": float(spec["atr_stop_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"volcomp_vwap_rev_guarded_{tf_tag}_{spec['variant']}_"
                    f"{float(spec['entry_z']):.2f}_{float(spec['compression_percentile']):.2f}"
                ),
                family="mean_reversion",
                strategy_class="VolCompressionVWAPReversionStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Compression-gated VWAP mean reversion with bounded low-turnover guardrails "
                    f"for {timeframe} follow-up ({spec['variant']})."
                ),
                tags=("mean_reversion", "vol_compression", "vwap", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "entry_guard": "zscore",
                    "allow_short": bool(spec["allow_short"]),
                },
            )

    # Classic VWAP deviation reversion sleeve.
    for timeframe in mean_rev_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _VWAP_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "window": int(spec["window"]),
                "entry_dev": float(spec["entry_dev"]),
                "exit_dev": float(spec["exit_dev"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"vwap_reversion_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['window'])}_{float(spec['entry_dev']):.3f}"
                ),
                family="mean_reversion",
                strategy_class="VwapReversionStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Rolling VWAP deviation mean reversion with bounded entry/exit bands "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "vwap", "single_asset", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                },
            )


def _build_zscore_mean_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    normalized_symbols = ctx.normalized_symbols
    std_mean_rev_tfs = ctx.std_mean_rev_tfs

    # Classic z-score mean reversion sleeve.
    for timeframe in std_mean_rev_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _MEAN_REVERSION_STD_SLICE.get(timeframe, ()):
            params = {
                "window": int(spec["window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            tags = ["mean_reversion", "zscore", "single_asset", "bounded"]
            note_suffix = ""
            if bool(spec.get("residualize_btc", False)):
                params["residualize_btc"] = True
                params["btc_symbol"] = str(spec.get("btc_symbol") or "BTC/USDT")
                tags.append("btc_beta_neutral")
                note_suffix = " BTC-beta-neutral residual signal."
            _add_candidate(
                candidates,
                name=(
                    f"mean_reversion_std_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['window'])}_{float(spec['entry_z']):.2f}"
                ),
                family="mean_reversion",
                strategy_class="MeanReversionStdStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Single-asset rolling z-score mean reversion with bounded stop rules "
                    f"for {timeframe} ({spec['variant']}).{note_suffix}"
                ),
                tags=tuple(tags),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "residualize_btc": bool(spec.get("residualize_btc", False)),
                    "btc_symbol": str(spec.get("btc_symbol") or ""),
                },
            )


def _build_liquidity_event_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    normalized_symbols = ctx.normalized_symbols
    liquidity_tfs = ctx.liquidity_tfs
    session_liquidity_tfs = ctx.session_liquidity_tfs
    crypto_symbols = tuple(symbol for symbol in normalized_symbols if symbol not in _METALS)

    # Liquidity-shock event reversion sleeve.
    for timeframe in liquidity_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _LIQUIDITY_SHOCK_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "volume_window": int(spec["volume_window"]),
                "range_window": int(spec["range_window"]),
                "volume_shock_z": float(spec["volume_shock_z"]),
                "range_shock_z": float(spec["range_shock_z"]),
                "return_shock_pct": float(spec["return_shock_pct"]),
                "revert_fraction": float(spec["revert_fraction"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"liquidity_shock_reversion_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['volume_window'])}_{float(spec['return_shock_pct']):.3f}"
                ),
                family="mean_reversion",
                strategy_class="LiquidityShockReversionStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Event-triggered liquidity-shock mean reversion that fades outsized intraday moves "
                    f"when range and volume dislocations spike on {timeframe} ({spec['variant']})."
                ),
                tags=(
                    "mean_reversion",
                    "liquidity_shock",
                    "event_driven",
                    "single_asset",
                    "bounded",
                ),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )

    for timeframe in session_liquidity_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _SESSION_LIQUIDITY_VACUUM_SLICE.get(timeframe, ()):
            params = {
                "volume_window": int(spec["volume_window"]),
                "range_window": int(spec["range_window"]),
                "volume_shock_z": float(spec["volume_shock_z"]),
                "range_shock_z": float(spec["range_shock_z"]),
                "return_shock_pct": float(spec["return_shock_pct"]),
                "revert_fraction": float(spec["revert_fraction"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
                "session_window_minutes": int(spec["session_window_minutes"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"session_liquidity_vacuum_fade_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['volume_window'])}_{float(spec['return_shock_pct']):.3f}"
                ),
                family="mean_reversion",
                strategy_class="SessionLiquidityVacuumFadeStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Session-transition liquidity vacuum fade that only reacts around repeated UTC handoff windows "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    "mean_reversion",
                    "session_transition",
                    "liquidity_shock",
                    "event_driven",
                    "bounded",
                ),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )


def _build_panic_rebound_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    panic_rebound_tfs = ctx.panic_rebound_tfs
    crypto_symbols = tuple(symbol for symbol in ctx.normalized_symbols if symbol not in _METALS)
    if not crypto_symbols:
        return

    for timeframe in panic_rebound_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _PANIC_REBOUND_MEAN_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "history_bars": int(spec["history_bars"]),
                "return_window": int(spec["return_window"]),
                "volume_window": int(spec["volume_window"]),
                "vwap_window": int(spec["vwap_window"]),
                "shock_return_z": float(spec["shock_return_z"]),
                "shock_return_pct": float(spec["shock_return_pct"]),
                "volume_z": float(spec["volume_z"]),
                "confirmation_bars": int(spec["confirmation_bars"]),
                "min_rebound_pct": float(spec["min_rebound_pct"]),
                "vwap_recovery_pct": float(spec["vwap_recovery_pct"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "take_profit_pct": float(spec["take_profit_pct"]),
                "trailing_exit_pct": float(spec["trailing_exit_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "target_allocation": float(spec["target_allocation"]),
                "max_order_value": float(spec["max_order_value"]),
                "min_price": 0.10,
            }
            _add_candidate(
                candidates,
                name=(
                    f"panic_rebound_mr_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['return_window'])}_{float(spec['shock_return_pct']):.3f}"
                ),
                family="profit_reboot_mean_reversion",
                strategy_class="PanicReboundMeanReversionStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Profit-reboot long-only panic rebound sleeve that waits for post-shock "
                    "VWAP/rebound confirmation before entering and uses fast stop/time exits "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    "profit_reboot_20260501",
                    "profit_moonshot_20260501",
                    "mean_reversion",
                    "panic_rebound",
                    "liquidation_rebound",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                    "confirmation_required": True,
                },
            )


def _build_derivatives_mean_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    normalized_symbols = ctx.normalized_symbols
    funding_crowding_tfs = ctx.funding_crowding_tfs
    basis_snapback_tfs = ctx.basis_snapback_tfs
    vol_of_vol_tfs = ctx.vol_of_vol_tfs
    crypto_symbols = tuple(symbol for symbol in normalized_symbols if symbol not in _METALS)

    for timeframe in funding_crowding_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _FUNDING_LIQUIDATION_CROWDING_FADE_SLICE.get(timeframe, ()):
            params = {
                "window": int(spec["window"]),
                "crowding_entry": float(spec["crowding_entry"]),
                "crowding_exit": float(spec["crowding_exit"]),
                "liquidation_z_min": float(spec["liquidation_z_min"]),
                "return_shock_pct": float(spec["return_shock_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"funding_liquidation_crowding_fade_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['window'])}_{float(spec['crowding_entry']):.2f}"
                ),
                family="mean_reversion",
                strategy_class="FundingLiquidationCrowdingFadeStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Fade derivative crowding/liquidation exhaustion after aligned funding, OI, and liquidation shocks "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "crowding", "liquidation", "derivatives", "event_driven"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )

    for timeframe in basis_snapback_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _BASIS_SNAPBACK_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "window": int(spec["window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"basis_snapback_reversion_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['window'])}_{float(spec['entry_z']):.1f}"
                ),
                family="mean_reversion",
                strategy_class="BasisSnapbackReversionStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Mean-revert derivatives basis dislocations when mark-vs-index spread becomes extreme "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "basis", "derivatives", "event_driven", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )

    for timeframe in vol_of_vol_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _VOL_OF_VOL_EXHAUSTION_SLICE.get(timeframe, ()):
            params = {
                "vol_window": int(spec["vol_window"]),
                "vol_z_window": int(spec["vol_z_window"]),
                "return_z_window": int(spec["return_z_window"]),
                "vol_entry_z": float(spec["vol_entry_z"]),
                "return_entry_z": float(spec["return_entry_z"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"vol_of_vol_exhaustion_fade_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['vol_window'])}_{float(spec['vol_entry_z']):.1f}"
                ),
                family="mean_reversion",
                strategy_class="VolOfVolExhaustionFadeStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Fade second-order volatility exhaustion after realized-vol spikes "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "vol_of_vol", "volatility_exhaustion", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )


def _build_core_mean_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    _build_vwap_mean_reversion_candidates(ctx)
    _build_zscore_mean_reversion_candidates(ctx)
    _build_liquidity_event_reversion_candidates(ctx)
    _build_panic_rebound_candidates(ctx)
    _build_derivatives_mean_reversion_candidates(ctx)


def _build_intraday_alpha_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    laggard_symbols = ctx.laggard_symbols
    mean_rev_tfs = ctx.mean_rev_tfs
    normalized_symbols = ctx.normalized_symbols
    # Lead/lag spillover sleeve (metals excluded).
    if laggard_symbols:
        for timeframe in mean_rev_tfs:
            tf_tag = timeframe.replace("/", "-")
            for entry_score, max_lag in product((0.25, 0.35, 0.50), (2, 3, 4)):
                params = {
                    "entry_score": float(entry_score),
                    "exit_score": 0.08,
                    "max_lag": int(max_lag),
                    "ridge_alpha": 1.0,
                    "max_hold_bars": 24,
                    "stop_loss_pct": 0.02,
                    "allow_short": True,
                }
                _add_candidate(
                    candidates,
                    name=f"leadlag_spillover_{tf_tag}_{entry_score:.2f}_lag{max_lag}",
                    family="intraday_alpha",
                    strategy_class="LeadLagSpilloverStrategy",
                    timeframe=timeframe,
                    symbols=tuple(
                        sorted(set(_CRYPTO_LEADERS).intersection(normalized_symbols))
                        + laggard_symbols
                    ),
                    params=params,
                    notes="Cross-asset lead-lag predictor (crypto only, metals excluded).",
                    tags=("leadlag", "cross-asset", "intraday", "alpha"),
                    metadata={
                        "timeframe": timeframe,
                        "symbol_scope": "crypto_excluding_metals",
                        "lag_bands": [2, 3, 4],
                    },
                )


def _build_cross_sectional_rotation_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    crypto_symbols = ctx.crypto_symbols
    topcap_tfs = ctx.topcap_tfs
    liquidity_regime_tfs = ctx.liquidity_regime_tfs
    abnormal_return_tfs = ctx.abnormal_return_tfs
    residual_basket_tfs = ctx.residual_basket_tfs
    session_residual_tfs = ctx.session_residual_tfs
    if len(crypto_symbols) >= 4:
        for timeframe in topcap_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _TOPCAP_TSMOM_SLICE.get(timeframe, ()):
                params = {
                    "lookback_bars": int(spec["lookback_bars"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "signal_threshold": float(spec["signal_threshold"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "min_price": float(spec["min_price"]),
                    "btc_regime_ma": int(spec["btc_regime_ma"]),
                    "btc_symbol": str(spec["btc_symbol"]),
                }
                if "take_profit_pct" in spec:
                    params["take_profit_pct"] = float(spec["take_profit_pct"])
                if "residualize_btc" in spec:
                    params["residualize_btc"] = bool(spec["residualize_btc"])
                if "residualize_mean" in spec:
                    params["residualize_mean"] = bool(spec["residualize_mean"])
                if "benchmark_drawdown_window" in spec:
                    params["benchmark_drawdown_window"] = int(spec["benchmark_drawdown_window"])
                if "benchmark_drawdown_limit" in spec:
                    params["benchmark_drawdown_limit"] = float(spec["benchmark_drawdown_limit"])
                tags = ["cross_sectional", "relative_momentum", "topcap", "crypto"]
                residual_notes = []
                if bool(spec.get("residualize_btc", False)):
                    tags.append("residual_momentum")
                    residual_notes.append("BTC-common-move residualization")
                if bool(spec.get("residualize_mean", False)):
                    tags.append("factor_neutral")
                    residual_notes.append("cross-sectional mean neutralization")
                if (
                    int(spec.get("benchmark_drawdown_window", 0) or 0) > 0
                    and float(spec.get("benchmark_drawdown_limit", 0.0) or 0.0) > 0.0
                ):
                    tags.append("crash_aware")
                    residual_notes.append(
                        f"benchmark drawdown gate {int(spec['benchmark_drawdown_window'])} bars/{float(spec['benchmark_drawdown_limit']):.1%}"
                    )
                if str(spec.get("variant") or "").startswith("exec_"):
                    tags.append("execution_risk")
                    residual_notes.append("execution-risk retune")
                if float(spec.get("take_profit_pct", 0.0) or 0.0) > 0.0:
                    tags.append("take_profit")
                    residual_notes.append(f"take profit {float(spec['take_profit_pct']):.1%}")
                note_suffix = " with " + " + ".join(residual_notes) + "." if residual_notes else "."
                _add_candidate(
                    candidates,
                    name=(
                        f"topcap_tsmom_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['lookback_bars'])}_{int(spec['rebalance_bars'])}_{float(spec['signal_threshold']):.3f}"
                    ),
                    family="cross_sectional",
                    strategy_class="TopCapTimeSeriesMomentumStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Top-cap long/short relative-momentum rotation with BTC regime gating "
                        f"for {timeframe} ({spec['variant']}){note_suffix}"
                    ),
                    tags=tuple(tags),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "residualize_btc": bool(spec.get("residualize_btc", False)),
                        "residualize_mean": bool(spec.get("residualize_mean", False)),
                        "benchmark_drawdown_window": int(
                            spec.get("benchmark_drawdown_window", 0) or 0
                        ),
                        "benchmark_drawdown_limit": float(
                            spec.get("benchmark_drawdown_limit", 0.0) or 0.0
                        ),
                    },
                )

        for timeframe in topcap_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _ADAPTIVE_REGIME_MOMENTUM_SLICE.get(timeframe, ()):
                params = {
                    "lookback_bars": int(spec["lookback_bars"]),
                    "short_lookback_bars": int(spec["short_lookback_bars"]),
                    "regime_lookback_bars": int(spec["regime_lookback_bars"]),
                    "volatility_lookback_bars": int(spec["volatility_lookback_bars"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "signal_threshold": float(spec["signal_threshold"]),
                    "broad_threshold": float(spec["broad_threshold"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "gross_exposure": float(spec["gross_exposure"]),
                    "max_order_value": float(spec["max_order_value"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "take_profit_pct": float(spec["take_profit_pct"]),
                    "trailing_exit_pct": float(spec["trailing_exit_pct"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "btc_symbol": "BTC/USDT",
                    "min_price": 0.10,
                    "risk_off_exit": True,
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"adaptive_regime_momentum_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['lookback_bars'])}_{int(spec['rebalance_bars'])}_{float(spec['signal_threshold']):.3f}"
                    ),
                    family="profit_reboot_cross_sectional",
                    strategy_class="AdaptiveRegimeMomentumStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Profit-reboot adaptive regime momentum sleeve that compresses each market-window "
                        f"decision into one bar and switches between long, short, and cash for {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "profit_reboot_20260501",
                        "profit_moonshot_20260501",
                        "cross_sectional",
                        "adaptive_regime",
                        "momentum",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "market_window_one_bar_per_decision": True,
                    },
                )

        moonshot_specs = (
            (
                "trend",
                "profit_moonshot_cross_sectional",
                "ProfitMoonshotTrendStrategy",
                _PROFIT_MOONSHOT_TREND_SLICE,
            ),
            (
                "breakout",
                "profit_moonshot_breakout",
                "ProfitMoonshotBreakoutStrategy",
                _PROFIT_MOONSHOT_BREAKOUT_SLICE,
            ),
            (
                "reversion",
                "profit_moonshot_reversion",
                "ProfitMoonshotReversionStrategy",
                _PROFIT_MOONSHOT_REVERSION_SLICE,
            ),
        )
        for sleeve, family, strategy_class, spec_by_timeframe in moonshot_specs:
            for timeframe in topcap_tfs:
                tf_tag = timeframe.replace("/", "-")
                for spec in spec_by_timeframe.get(timeframe, ()):
                    params = {
                        "lookback_bars": int(spec["lookback_bars"]),
                        "fast_lookback_bars": int(spec["fast_lookback_bars"]),
                        "slow_lookback_bars": int(spec["slow_lookback_bars"]),
                        "rebalance_bars": int(spec["rebalance_bars"]),
                        "entry_threshold": float(spec["entry_threshold"]),
                        "exit_threshold": float(spec["exit_threshold"]),
                        "max_longs": int(spec["max_longs"]),
                        "max_shorts": int(spec["max_shorts"]),
                        "gross_exposure": float(spec["gross_exposure"]),
                        "max_order_value": float(spec["max_order_value"]),
                        "stop_loss_pct": float(spec["stop_loss_pct"]),
                        "take_profit_pct": float(spec["take_profit_pct"]),
                        "trailing_exit_pct": float(spec["trailing_exit_pct"]),
                        "max_hold_bars": int(spec["max_hold_bars"]),
                        "min_price": 0.10,
                        "allow_shorts": True,
                    }
                    for optional_key in (
                        "breadth_threshold",
                        "breakout_buffer",
                        "squeeze_ratio_max",
                        "volume_z_min",
                        "return_z_min",
                        "range_z_min",
                    ):
                        if optional_key in spec:
                            params[optional_key] = spec[optional_key]
                    _add_candidate(
                        candidates,
                        name=(
                            f"profit_moonshot_{sleeve}_{tf_tag}_{spec['variant']}_"
                            f"{int(spec['lookback_bars'])}_{float(spec['entry_threshold']):.3f}"
                        ),
                        family=family,
                        strategy_class=strategy_class,
                        timeframe=timeframe,
                        symbols=crypto_symbols,
                        params=params,
                        notes=(
                            "Profit-moonshot MARKET_WINDOW sleeve that bypasses TimeframeAggregator "
                            f"and targets higher participation for {timeframe} ({spec['variant']}, {sleeve})."
                        ),
                        tags=(
                            "profit_moonshot_20260501",
                            "market_window",
                            "no_timeframe_aggregator",
                            sleeve,
                            "crypto",
                        ),
                        metadata={
                            "timeframe": timeframe,
                            "retune_profile": str(spec["variant"]),
                            "symbol_scope": "crypto",
                            "market_window_one_bar_per_decision": True,
                        },
                    )

        for timeframe in topcap_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _CARRY_TREND_FACTOR_ROTATION_SLICE.get(timeframe, ()):
                params = {
                    "lookback_bars": int(spec["lookback_bars"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "signal_threshold": float(spec["signal_threshold"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "min_price": float(spec["min_price"]),
                    "btc_regime_ma": int(spec["btc_regime_ma"]),
                    "btc_symbol": str(spec["btc_symbol"]),
                    "benchmark_drawdown_window": int(spec["benchmark_drawdown_window"]),
                    "benchmark_drawdown_limit": float(spec["benchmark_drawdown_limit"]),
                    "vol_window": int(spec["vol_window"]),
                    "crowding_window": int(spec["crowding_window"]),
                    "trend_weight": float(spec["trend_weight"]),
                    "carry_weight": float(spec["carry_weight"]),
                    "defensive_weight": float(spec["defensive_weight"]),
                    "crowding_weight": float(spec["crowding_weight"]),
                    "allow_short": bool(spec["allow_short"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"carry_trend_factor_rotation_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['lookback_bars'])}_{int(spec['rebalance_bars'])}_{float(spec['signal_threshold']):.3f}"
                    ),
                    family="cross_sectional",
                    strategy_class="CarryTrendFactorRotationStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Article-inspired factor rotation that combines trend persistence, carry/crowding pressure, "
                        f"and defensive volatility scaling for {timeframe} ({spec['variant']})."
                    ),
                    tags=("cross_sectional", "factor", "carry", "momentum", "defensive", "crypto"),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "allow_short": bool(spec["allow_short"]),
                        "production_ready": bool(spec.get("production_ready", False)),
                        "data_dependent": True,
                        "article_reference": "quant-company-profit-mechanisms",
                    },
                )

    if len(crypto_symbols) >= 2:
        for timeframe in liquidity_regime_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _LAST_DAY_LIQUIDITY_REGIME_SLICE.get(timeframe, ()):
                params = {
                    "momentum_lookback_bars": int(spec["momentum_lookback_bars"]),
                    "signal_skip_bars": int(spec["signal_skip_bars"]),
                    "liquidity_window": int(spec["liquidity_window"]),
                    "volatility_window": int(spec["volatility_window"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "signal_threshold": float(spec["signal_threshold"]),
                    "liquidity_quantile": float(spec["liquidity_quantile"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "min_price": float(spec["min_price"]),
                    "max_realized_vol": float(spec["max_realized_vol"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "allow_short": bool(spec["allow_short"]),
                    "illiquid_reversal": bool(spec["illiquid_reversal"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"last_day_liquidity_regime_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['momentum_lookback_bars'])}_{int(spec['rebalance_bars'])}_{float(spec['signal_threshold']):.3f}"
                    ),
                    family="cross_sectional",
                    strategy_class="LastDayLiquidityRegimeStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Liquidity-conditioned last-day-return continuation/reversal sleeve "
                        f"for {timeframe} ({spec['variant']}) based on liquid-momentum / illiquid-reversal evidence."
                    ),
                    tags=(
                        "cross_sectional",
                        "pure_momentum",
                        "liquidity_conditioned",
                        "last_day_return",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "allow_short": bool(spec["allow_short"]),
                        "illiquid_reversal": bool(spec["illiquid_reversal"]),
                    },
                )

    if len(crypto_symbols) >= 2:
        for timeframe in abnormal_return_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _ABNORMAL_RETURN_CONTINUATION_SLICE.get(timeframe, ()):
                for symbol in crypto_symbols:
                    params = {
                        "return_z_window": int(spec["return_z_window"]),
                        "entry_z": float(spec["entry_z"]),
                        "exit_z": float(spec["exit_z"]),
                        "hold_bars": int(spec["hold_bars"]),
                        "stop_loss_pct": float(spec["stop_loss_pct"]),
                        "allow_short": bool(spec["allow_short"]),
                    }
                    _add_candidate(
                        candidates,
                        name=(
                            f"abnormal_return_continuation_{tf_tag}_{spec['variant']}_"
                            f"{symbol.replace('/', '').lower()}_{float(spec['entry_z']):.1f}_{int(spec['hold_bars'])}"
                        ),
                        family="event_alpha",
                        strategy_class="AbnormalReturnContinuationStrategy",
                        timeframe=timeframe,
                        symbols=(symbol,),
                        params=params,
                        notes=(
                            "Abnormal one-day return continuation sleeve that follows large daily shocks "
                            f"for {symbol} on {timeframe} ({spec['variant']})."
                        ),
                        tags=(
                            "event_alpha",
                            "abnormal_return",
                            "continuation",
                            "single_asset",
                            "crypto",
                        ),
                        metadata={
                            "timeframe": timeframe,
                            "retune_profile": str(spec["variant"]),
                            "symbol_scope": symbol,
                            "allow_short": bool(spec["allow_short"]),
                        },
                    )

    if len(crypto_symbols) >= 4:
        for timeframe in residual_basket_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _RESIDUAL_BASKET_REVERSION_SLICE.get(timeframe, ()):
                params = {
                    "residual_window": int(spec["residual_window"]),
                    "entry_z": float(spec["entry_z"]),
                    "exit_z": float(spec["exit_z"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "allow_short": bool(spec["allow_short"]),
                    "btc_symbol": str(spec["btc_symbol"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"residual_basket_reversion_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['residual_window'])}_{float(spec['entry_z']):.2f}"
                    ),
                    family="cross_sectional",
                    strategy_class="ResidualBasketReversionStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Cross-sectional residual basket reversion using BTC-neutralized residual zscores "
                        f"for {timeframe} ({spec['variant']})."
                    ),
                    tags=("cross_sectional", "residual_reversion", "btc_beta_neutral", "crypto"),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "btc_symbol": str(spec["btc_symbol"]),
                    },
                )

    if len(crypto_symbols) >= 3:
        for timeframe in session_residual_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _SESSION_GATED_RESIDUAL_BASKET_REVERSION_SLICE.get(timeframe, ()):
                params = {
                    "residual_window": int(spec["residual_window"]),
                    "entry_z": float(spec["entry_z"]),
                    "exit_z": float(spec["exit_z"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "allow_short": bool(spec["allow_short"]),
                    "btc_symbol": str(spec["btc_symbol"]),
                    "session_window_minutes": int(spec["session_window_minutes"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"session_gated_residual_basket_reversion_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['residual_window'])}_{float(spec['entry_z']):.2f}"
                    ),
                    family="cross_sectional",
                    strategy_class="SessionGatedResidualBasketReversionStrategy",
                    timeframe=timeframe,
                    symbols=tuple(symbol for symbol in crypto_symbols[:3]),
                    params=params,
                    notes=(
                        "Session-gated residual basket reversion using BTC-neutral residual zscores "
                        f"for {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "cross_sectional",
                        "residual_reversion",
                        "session_transition",
                        "btc_beta_neutral",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "btc_symbol": str(spec["btc_symbol"]),
                    },
                )

    if len(crypto_symbols) >= 3:
        for timeframe in residual_basket_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _VOL_REGIME_RESIDUAL_BASKET_REVERSION_SLICE.get(timeframe, ()):
                params = {
                    "residual_window": int(spec["residual_window"]),
                    "entry_z": float(spec["entry_z"]),
                    "exit_z": float(spec["exit_z"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "allow_short": bool(spec["allow_short"]),
                    "btc_symbol": str(spec["btc_symbol"]),
                    "btc_vol_fast": int(spec["btc_vol_fast"]),
                    "btc_vol_slow": int(spec["btc_vol_slow"]),
                    "btc_vol_ratio_cap": float(spec["btc_vol_ratio_cap"]),
                    "dispersion_floor": float(spec["dispersion_floor"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"volatility_regime_residual_basket_reversion_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['residual_window'])}_{float(spec['entry_z']):.2f}"
                    ),
                    family="cross_sectional",
                    strategy_class="VolatilityRegimeResidualBasketReversionStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Volatility-regime-gated residual basket reversion using BTC-neutral residual zscores "
                        f"for {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "cross_sectional",
                        "residual_reversion",
                        "volatility_regime",
                        "btc_beta_neutral",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "btc_symbol": str(spec["btc_symbol"]),
                    },
                )


def _build_cross_asset_mean_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    crypto_symbols = ctx.crypto_symbols
    normalized_symbols = ctx.normalized_symbols
    contagion_tfs = ctx.contagion_tfs
    trend_exhaustion_tfs = ctx.trend_exhaustion_tfs
    for timeframe in contagion_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _LIQUIDATION_CONTAGION_FADE_SLICE.get(timeframe, ()):
            params = {
                "window": int(spec["window"]),
                "leader_liq_z_min": float(spec["leader_liq_z_min"]),
                "return_shock_pct": float(spec["return_shock_pct"]),
                "exit_z": float(spec["exit_z"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"liquidation_contagion_fade_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['window'])}_{float(spec['leader_liq_z_min']):.1f}"
                ),
                family="mean_reversion",
                strategy_class="CrossAssetLiquidationContagionFadeStrategy",
                timeframe=timeframe,
                symbols=tuple(symbol for symbol in crypto_symbols[:3]),
                params=params,
                notes=(
                    "Fade secondary-asset moves after extreme leader liquidation contagion "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "liquidation", "contagion", "cross_asset", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                },
            )

    for timeframe in trend_exhaustion_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _MULTI_HORIZON_TREND_EXHAUSTION_SLICE.get(timeframe, ()):
            params = {
                "short_window": int(spec["short_window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"multi_horizon_trend_exhaustion_fade_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['short_window'])}_{float(spec['entry_z']):.1f}"
                ),
                family="mean_reversion",
                strategy_class="MultiHorizonTrendExhaustionFadeStrategy",
                timeframe=timeframe,
                symbols=tuple(symbol for symbol in normalized_symbols if symbol not in _METALS),
                params=params,
                notes=(
                    "Fade short-horizon trend exhaustion when multi-horizon momentum disagrees "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "trend_exhaustion", "multi_horizon", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )


def _build_formula_and_breadth_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    alpha101_tfs = ctx.alpha101_tfs
    crypto_symbols = ctx.crypto_symbols
    breadth_tfs = ctx.breadth_tfs
    normalized_symbols = ctx.normalized_symbols
    if crypto_symbols:
        _add_alpha101_formula_candidates(
            candidates,
            timeframes=alpha101_tfs,
            symbols=crypto_symbols,
        )

    for timeframe in breadth_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _BREADTH_THRUST_FAILURE_SLICE.get(timeframe, ()):
            params = {
                "momentum_lookback": int(spec["momentum_lookback"]),
                "breadth_entry": float(spec["breadth_entry"]),
                "breadth_exit": float(spec["breadth_exit"]),
                "basket_return_floor": float(spec["basket_return_floor"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"breadth_thrust_failure_reversal_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['momentum_lookback'])}_{float(spec['breadth_entry']):.2f}"
                ),
                family="cross_sectional",
                strategy_class="BreadthThrustFailureReversalStrategy",
                timeframe=timeframe,
                symbols=tuple(symbol for symbol in normalized_symbols if symbol not in _METALS),
                params=params,
                notes=(
                    "Fade failed basket breadth thrusts after overly one-sided crypto participation "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("cross_sectional", "breadth", "mean_reversion", "basket"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )


def _build_breakout_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    breakout_tfs = ctx.breakout_tfs
    normalized_symbols = ctx.normalized_symbols
    # Single-asset breakout sleeves.
    for timeframe in breakout_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _ROLLING_BREAKOUT_SLICE.get(timeframe, ()):
            params = {
                "lookback_bars": int(spec["lookback_bars"]),
                "breakout_buffer": float(spec["breakout_buffer"]),
                "atr_window": int(spec["atr_window"]),
                "atr_stop_multiplier": float(spec["atr_stop_multiplier"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"rolling_breakout_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['lookback_bars'])}_{float(spec['breakout_buffer']):.3f}"
                ),
                family="trend",
                strategy_class="RollingBreakoutStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Single-asset channel breakout with ATR-aware protective stops "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("trend", "breakout", "single_asset", "atr"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                },
            )
        for spec in _REGIME_BREAKOUT_SLICE.get(timeframe, ()):
            params = {
                "lookback_window": int(spec["lookback_window"]),
                "slope_window": int(spec["slope_window"]),
                "volatility_fast_window": int(spec["volatility_fast_window"]),
                "volatility_slow_window": int(spec["volatility_slow_window"]),
                "range_entry_threshold": float(spec["range_entry_threshold"]),
                "slope_entry_threshold": float(spec["slope_entry_threshold"]),
                "momentum_floor": float(spec["momentum_floor"]),
                "max_volatility_ratio": float(spec["max_volatility_ratio"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"regime_breakout_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['lookback_window'])}_{float(spec['range_entry_threshold']):.2f}"
                ),
                family="trend",
                strategy_class="RegimeBreakoutCandidateStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Regime-gated breakout candidate with trend and volatility filters "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("trend", "breakout", "regime", "single_asset"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                },
            )


def _build_pair_and_intermarket_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    pair_tfs = ctx.pair_tfs
    session_pair_carry_tfs = ctx.session_pair_carry_tfs
    pairs = ctx.pairs
    lag_convergence_tfs = ctx.lag_convergence_tfs
    # Pair spread sleeve.
    for timeframe in pair_tfs:
        tf_tag = timeframe.replace("/", "-")
        tuned_params = bounded_pair_retune_params(timeframe)
        tuned_param_sets = tuple(
            dict(item)
            for item in _PAIR_RETUNE_PARAM_SETS_BY_TIMEFRAME.get(timeframe, (dict(tuned_params),))
        )
        pair_universe = list(pairs)
        if timeframe == "15m":
            pair_universe = [pair for pair in pair_universe if pair in _PAIR_RETUNE_FOCUS_PAIRS_15M]
        elif timeframe == "30m":
            pair_universe = [pair for pair in pair_universe if pair in _PAIR_RETUNE_FOCUS_PAIRS_30M]
        elif timeframe == "4h":
            pair_universe = [pair for pair in pair_universe if pair in _PAIR_RETUNE_FOCUS_PAIRS_4H]
        elif timeframe == "1d":
            pair_universe = [pair for pair in pair_universe if pair in _PAIR_RETUNE_FOCUS_PAIRS_1D]
        for symbol_x, symbol_y in pair_universe:
            pair_token = f"{symbol_x.replace('/', '').lower()}_{symbol_y.replace('/', '').lower()}"
            for tuned_spec in tuned_param_sets:
                variant = str(tuned_spec.get("variant") or "core")
                for entry_z, exit_z, stop_z in _PAIR_RETUNE_SPECS_BY_TIMEFRAME.get(timeframe, ()):
                    params = {
                        "lookback_window": int(tuned_spec["lookback_window"]),
                        "hedge_window": int(tuned_spec["hedge_window"]),
                        "entry_z": float(entry_z),
                        "exit_z": float(exit_z),
                        "stop_z": float(stop_z),
                        "max_hold_bars": int(tuned_spec["max_hold_bars"]),
                        "min_correlation": float(tuned_spec["min_correlation"]),
                        "cooldown_bars": int(tuned_spec["cooldown_bars"]),
                        "reentry_z_buffer": float(tuned_spec["reentry_z_buffer"]),
                        "stop_loss_pct": float(tuned_spec["stop_loss_pct"]),
                        "symbol_x": symbol_x,
                        "symbol_y": symbol_y,
                    }
                    for optional_key in (
                        "vwap_window",
                        "min_volume_window",
                        "min_volume_ratio",
                        "vol_lag_bars",
                        "min_vol_convergence",
                        "atr_window",
                        "atr_max_pct",
                        "beta_stop_scale_min",
                        "beta_stop_scale_max",
                        "take_profit_pct",
                    ):
                        if optional_key in tuned_spec:
                            params[optional_key] = tuned_spec[optional_key]
                    tags = ["market_neutral", "pair", "spread", "zscore"]
                    state_notes = []
                    if int(tuned_spec.get("vwap_window", 0) or 0) > 0:
                        tags.append("pair_state")
                        state_notes.append(f"VWAP normalization {int(tuned_spec['vwap_window'])}")
                    if float(tuned_spec.get("min_volume_ratio", 0.0) or 0.0) > 0.0:
                        tags.append("pair_state")
                        state_notes.append(
                            f"volume ratio >= {float(tuned_spec['min_volume_ratio']):.2f}"
                        )
                    if float(tuned_spec.get("min_vol_convergence", 0.0) or 0.0) > 0.0:
                        tags.append("pair_state")
                        state_notes.append(
                            f"vol convergence z >= {float(tuned_spec['min_vol_convergence']):.2f}"
                        )
                    if int(tuned_spec.get("atr_window", 0) or 0) > 0:
                        tags.append("pair_state")
                        state_notes.append(
                            f"ATR filter {int(tuned_spec['atr_window'])}/{float(tuned_spec.get('atr_max_pct', 1.0)):.2f}"
                        )
                    if float(tuned_spec.get("take_profit_pct", 0.0) or 0.0) > 0.0:
                        tags.append("execution_risk")
                        tags.append("take_profit")
                        state_notes.append(
                            f"take profit {float(tuned_spec['take_profit_pct']):.1%}"
                        )
                    note_suffix = " " + "; ".join(state_notes) + "." if state_notes else ""
                    _add_candidate(
                        candidates,
                        name=f"pair_spread_{tf_tag}_{variant}_{pair_token}_{entry_z:.1f}_{exit_z:.2f}",
                        family="market_neutral",
                        strategy_class="PairSpreadZScoreStrategy",
                        timeframe=timeframe,
                        symbols=(symbol_x, symbol_y),
                        params=params,
                        notes=(
                            "Rolling-beta spread z-score with bounded turnover/correlation guardrails"
                            + (
                                " and 15m evidence-focused pair pruning."
                                if timeframe == "15m"
                                else ""
                            )
                            + (
                                " and 30m sector-dispersion pair caps for the new-hypothesis refresh."
                                if timeframe == "30m"
                                else ""
                            )
                            + (
                                f" {timeframe} uses {variant} tuning to balance participation, stability, and PBO."
                                if timeframe in {"4h", "1d"}
                                else "."
                            )
                            + note_suffix
                        ),
                        tags=tuple(dict.fromkeys(tags)),
                        metadata={
                            "timeframe": timeframe,
                            "pair": f"{symbol_x}_{symbol_y}",
                            "pair_variant": variant,
                        },
                    )

        if timeframe == "1h" and ("BNB/USDT", "TRX/USDT") in pair_universe:
            pair_token = "bnbusdt_trxusdt"
            for adaptive_spec in _PAIR_ADAPTIVE_RLS_1H_SPECS:
                params = {
                    "lookback_window": int(adaptive_spec["lookback_window"]),
                    "hedge_window": int(adaptive_spec["hedge_window"]),
                    "entry_z": float(adaptive_spec["entry_z"]),
                    "exit_z": float(adaptive_spec["exit_z"]),
                    "stop_z": float(adaptive_spec["stop_z"]),
                    "max_hold_bars": int(adaptive_spec["max_hold_bars"]),
                    "min_correlation": float(adaptive_spec["min_correlation"]),
                    "cooldown_bars": int(adaptive_spec["cooldown_bars"]),
                    "reentry_z_buffer": float(adaptive_spec["reentry_z_buffer"]),
                    "stop_loss_pct": float(adaptive_spec["stop_loss_pct"]),
                    "symbol_x": "BNB/USDT",
                    "symbol_y": "TRX/USDT",
                    "hedge_mode": str(adaptive_spec["hedge_mode"]),
                    "hedge_forgetting_factor": float(adaptive_spec["hedge_forgetting_factor"]),
                    "hedge_covariance_init": float(adaptive_spec["hedge_covariance_init"]),
                }
                for optional_key in ("take_profit_pct", "atr_window", "atr_max_pct"):
                    if optional_key in adaptive_spec:
                        params[optional_key] = adaptive_spec[optional_key]
                _add_candidate(
                    candidates,
                    name=(
                        f"pair_spread_{tf_tag}_{adaptive_spec['variant']}_{pair_token}_"
                        f"{float(adaptive_spec['entry_z']):.1f}_{float(adaptive_spec['exit_z']):.2f}"
                    ),
                    family="market_neutral",
                    strategy_class="PairSpreadZScoreStrategy",
                    timeframe=timeframe,
                    symbols=("BNB/USDT", "TRX/USDT"),
                    params=params,
                    notes=(
                        "Adaptive scalar-RLS hedge update for BNB/TRX 1h pair trading. "
                        "Focused broader-redesign follow-up candidate with capped count and explicit sparse-fold validation."
                    ),
                    tags=(
                        "market_neutral",
                        "pair",
                        "spread",
                        "zscore",
                        "adaptive_hedge",
                        "focused_followup",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "pair": "BNB/USDT_TRX/USDT",
                        "pair_variant": str(adaptive_spec["variant"]),
                        "focused_followup": True,
                    },
                )

    for timeframe in session_pair_carry_tfs:
        if ("BNB/USDT", "TRX/USDT") not in pairs:
            continue
        tf_tag = timeframe.replace("/", "-")
        for spec in _SESSION_FILTERED_PAIR_CARRY_SLICE.get(timeframe, ()):
            params = {
                "symbol_x": "BNB/USDT",
                "symbol_y": "TRX/USDT",
                "lookback_window": int(spec["lookback_window"]),
                "hedge_window": int(spec["hedge_window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "stop_z": float(spec["stop_z"]),
                "min_correlation": float(spec["min_correlation"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "cooldown_bars": int(spec["cooldown_bars"]),
                "reentry_z_buffer": float(spec["reentry_z_buffer"]),
                "min_z_turn": float(spec["min_z_turn"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "take_profit_pct": float(spec["take_profit_pct"]),
                "allowed_session_utc_hours": str(spec["allowed_session_utc_hours"]),
                "min_expected_move_pct": float(spec["min_expected_move_pct"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"session_filtered_pair_carry_{tf_tag}_{spec['variant']}_"
                    f"{float(spec['entry_z']):.1f}_{float(spec['exit_z']):.2f}"
                ),
                family="profit_reboot_pair_carry",
                strategy_class="SessionFilteredPairCarryStrategy",
                timeframe=timeframe,
                symbols=("BNB/USDT", "TRX/USDT"),
                params=params,
                notes=(
                    "Profit-reboot BNB/TRX pair carry sleeve that only opens spread mean-reversion "
                    "positions during configured UTC sessions when expected move clears fee/slippage "
                    f"thresholds for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    "profit_reboot_20260501",
                    "profit_moonshot_20260501",
                    "market_neutral",
                    "pair",
                    "session_filter",
                    "expected_move_gate",
                ),
                metadata={
                    "timeframe": timeframe,
                    "pair": "BNB/USDT_TRX/USDT",
                    "pair_variant": str(spec["variant"]),
                    "session_filtered": True,
                },
            )

    for timeframe in lag_convergence_tfs:
        tf_tag = timeframe.replace("/", "-")
        pair_universe = [
            pair
            for pair in _LAG_CONVERGENCE_FOCUS_PAIRS_BY_TIMEFRAME.get(timeframe, ())
            if pair in pairs
        ]
        for symbol_x, symbol_y in pair_universe:
            pair_token = f"{symbol_x.replace('/', '').lower()}_{symbol_y.replace('/', '').lower()}"
            for spec in _LAG_CONVERGENCE_SPECS_BY_TIMEFRAME.get(timeframe, ()):
                params = {
                    "symbol_x": symbol_x,
                    "symbol_y": symbol_y,
                    "lag_bars": int(spec["lag_bars"]),
                    "entry_threshold": float(spec["entry_threshold"]),
                    "exit_threshold": float(spec["exit_threshold"]),
                    "stop_threshold": float(spec["stop_threshold"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"lag_convergence_{tf_tag}_{spec['variant']}_{pair_token}_"
                        f"{int(spec['lag_bars'])}_{float(spec['entry_threshold']):.3f}"
                    ),
                    family="intermarket",
                    strategy_class="LagConvergenceStrategy",
                    timeframe=timeframe,
                    symbols=(symbol_x, symbol_y),
                    params=params,
                    notes=(
                        "Lagged relative-momentum convergence for short-history metals pairs "
                        f"on {timeframe} ({spec['variant']})."
                    ),
                    tags=("lag_convergence", "metals", "pair", "relative_momentum"),
                    metadata={
                        "timeframe": timeframe,
                        "pair": f"{symbol_x}_{symbol_y}",
                        "pair_variant": str(spec["variant"]),
                    },
                )


def _build_deep_research_report_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    crypto_symbols = tuple(ctx.crypto_symbols)
    if len(crypto_symbols) < 3:
        return

    shared_metadata = {
        "source_report": "desktop-deep-research-report-20260608",
        "leaf_only": True,
        "research_only": True,
        "no_nested_oos_mining": True,
        "requires_fresh_forward_shadow": True,
        "deployment_gate": "blocked_until_fresh_forward_cost_telemetry_pbo_dsr_psr",
    }

    if ctx.perp_support_data_available:
        for timeframe in ctx._present("1h", "4h"):
            tf_tag = timeframe.replace("/", "-")
            for spec in _DEEP_RESEARCH_FUNDING_DISLOCATION_SLICE.get(timeframe, ()):
                params = {
                    "fast_lookback_bars": int(spec["fast_lookback_bars"]),
                    "mid_lookback_bars": int(spec["mid_lookback_bars"]),
                    "slow_lookback_bars": int(spec["slow_lookback_bars"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "signal_threshold": float(spec["signal_threshold"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "vol_window": int(spec["vol_window"]),
                    "crowding_window": int(spec["crowding_window"]),
                    "trend_weight": float(spec["trend_weight"]),
                    "carry_weight": float(spec["carry_weight"]),
                    "basis_weight": float(spec["basis_weight"]),
                    "crowding_penalty_weight": float(spec["crowding_penalty_weight"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "max_abs_exposure": float(spec["max_abs_exposure"]),
                    "allow_short": bool(spec["allow_short"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"deep_research_funding_dislocation_trend_carry_{tf_tag}_"
                        f"{spec['variant']}_{int(spec['mid_lookback_bars'])}_"
                        f"{float(spec['signal_threshold']):.2f}"
                    ),
                    family="deep_research_leaf",
                    strategy_class="FundingDislocationTrendCarryStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Desktop deep-research leaf: cross-sectional trend-carry score using "
                        "multi-horizon momentum, funding/basis dislocation, and OI/crowding "
                        f"penalty for {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "deep_research_report_20260608",
                        "leaf_alpha",
                        "trend",
                        "carry",
                        "funding",
                        "derivatives",
                    ),
                    metadata={
                        **shared_metadata,
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto_excluding_metals",
                        "requires_perp_feature_points": True,
                        "data_dependent": True,
                    },
                )

    for timeframe in ctx._present("1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _DEEP_RESEARCH_VOL_MANAGED_MOMENTUM_SLICE.get(timeframe, ()):
            params = {
                "momentum_lookback_bars": int(spec["momentum_lookback_bars"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "vol_window": int(spec["vol_window"]),
                "target_vol": float(spec["target_vol"]),
                "max_leverage": float(spec["max_leverage"]),
                "signal_threshold": float(spec["signal_threshold"]),
                "max_longs": int(spec["max_longs"]),
                "max_shorts": int(spec["max_shorts"]),
                "crash_window_bars": int(spec["crash_window_bars"]),
                "crash_return_pct": float(spec["crash_return_pct"]),
                "vol_ratio_window": int(spec["vol_ratio_window"]),
                "vol_ratio_max": float(spec["vol_ratio_max"]),
                "stress_reduce": float(spec["stress_reduce"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"deep_research_vol_managed_momentum_crash_gate_{tf_tag}_"
                    f"{spec['variant']}_{int(spec['momentum_lookback_bars'])}_"
                    f"{float(spec['target_vol']):.3f}"
                ),
                family="deep_research_leaf",
                strategy_class="VolManagedMomentumCrashGateStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Desktop deep-research leaf: volatility-managed cross-sectional momentum "
                    "with benchmark crash/volatility gate and bounded leverage."
                ),
                tags=(
                    "deep_research_report_20260608",
                    "leaf_alpha",
                    "momentum",
                    "volatility_targeting",
                    "crash_gate",
                ),
                metadata={
                    **shared_metadata,
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_excluding_metals",
                },
            )

    flow_symbols = tuple(
        symbol for symbol in _DEEP_RESEARCH_FLOW_MAJOR_SYMBOLS if symbol in ctx.normalized_symbols
    )
    if ctx.perp_support_data_available and flow_symbols:
        for timeframe in ctx._present("5m", "15m"):
            tf_tag = timeframe.replace("/", "-")
            for spec in _DEEP_RESEARCH_FLOW_IMBALANCE_LIQUIDATION_SLICE.get(timeframe, ()):
                params = {
                    "window": int(spec["window"]),
                    "entry_score": float(spec["entry_score"]),
                    "exit_score": float(spec["exit_score"]),
                    "liquidation_z_min": float(spec["liquidation_z_min"]),
                    "return_shock_pct": float(spec["return_shock_pct"]),
                    "max_spread_bps": float(spec["max_spread_bps"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "allow_short": bool(spec["allow_short"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"deep_research_flow_imbalance_liquidation_sweep_{tf_tag}_"
                        f"{spec['variant']}_{int(spec['window'])}_"
                        f"{float(spec['entry_score']):.2f}"
                    ),
                    family="deep_research_leaf",
                    strategy_class="FlowImbalanceLiquidationSweepStrategy",
                    timeframe=timeframe,
                    symbols=flow_symbols,
                    params=params,
                    notes=(
                        "Desktop deep-research leaf: major-asset order-flow/liquidation sweep "
                        "sleeve using taker imbalance, depth/BBO quality, and liquidation flush "
                        f"confirmation for {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "deep_research_report_20260608",
                        "leaf_alpha",
                        "order_flow",
                        "liquidation",
                        "microstructure",
                    ),
                    metadata={
                        **shared_metadata,
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "major_crypto_only",
                        "requires_perp_feature_points": True,
                        "initial_gross_cap_hint": 0.15,
                    },
                )


def _build_optional_carry_and_micro_candidates(ctx: _CandidateBuildContext) -> None:
    candidates = ctx.candidates
    carry_tfs = ctx.carry_tfs
    micro_tfs = ctx.micro_tfs
    crypto_symbols = ctx.crypto_symbols
    perp_support_data_available = ctx.perp_support_data_available
    # Optional carry/crowding sleeve.
    if perp_support_data_available and carry_tfs:
        for timeframe in carry_tfs:
            tf_tag = timeframe.replace("/", "-")
            for entry, exit_th in ((0.25, 0.08), (0.35, 0.10), (0.45, 0.15)):
                params = {
                    "entry_threshold": float(entry),
                    "exit_threshold": float(exit_th),
                    "mild_funding": 0.0002,
                    "extreme_funding": 0.0012,
                    "stop_loss_pct": 0.02,
                    "max_hold_bars": 72,
                    "allow_short": True,
                }
                _add_candidate(
                    candidates,
                    name=f"perp_crowding_carry_{tf_tag}_{entry:.2f}_{exit_th:.2f}",
                    family="carry",
                    strategy_class="PerpCrowdingCarryStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes="Funding/OI crowding-aware carry sleeve.",
                    tags=("carry", "perp", "funding", "crowding"),
                    metadata={
                        "timeframe": timeframe,
                        "data_dependent": perp_support_data_available,
                        "symbol_scope": "crypto",
                    },
                )

    # Research-only micro sleeve.
    for timeframe in micro_tfs:
        tf_tag = timeframe.replace("/", "-")
        for lookback, range_z, vol_z in ((20, 1.2, 0.8), (30, 1.5, 1.0), (45, 2.0, 1.2)):
            params = {
                "lookback": int(lookback),
                "range_z_threshold": float(range_z),
                "volume_z_threshold": float(vol_z),
                "max_hold_bars": 20,
                "allow_short": True,
            }
            _add_candidate(
                candidates,
                name=f"micro_range_expansion_{tf_tag}_{lookback}_{range_z:.1f}_{vol_z:.1f}",
                family="micro",
                strategy_class="MicroRangeExpansion1sStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes="Research-only micro breakout sleeve with strict turnover controls.",
                tags=("micro", "range", "breakout", "research"),
                metadata={
                    "timeframe": timeframe,
                    "research_only": True,
                },
            )


# ---------------------------------------------------------------------------
# New decorrelated alpha sleeves (Pass-1 live universe + dormant equity tranche)
# ---------------------------------------------------------------------------
#
# Selection-admission contract (selection.py:209,223-253): any candidate whose
# ``candidate_mix_type`` is ``"multi"`` (>=3 symbols) is dropped from the default
# shortlist UNLESS it is wired ``family="cross_sectional"`` AND its tags are a
# superset of {"cross_sectional", "carry", "momentum"}.  EVERY multi-symbol sleeve
# below therefore mirrors ``CarryTrendFactorRotationStrategy`` exactly.  The
# single-asset sleeves (S13 ConfidenceGatedTrend per-symbol, S11 calendar overlay
# per-index, S4 dual-momentum per-index) emit one symbol per candidate, are
# ``candidate_mix_type=="single"``, and intentionally use their natural family.

# Equity/ETF perp universe for the DORMANT tranche.  These are compact-form
# symbols; ``_add_candidate``/``canonicalize_symbol_list`` slash + canonicalize
# them and self-skip on empty intersections.
_EQUITY_FACTOR_UNIVERSE: tuple[str, ...] = tuple(dict.fromkeys(BINANCE_TRADFI_EQUITY_SYMBOLS))
_INDEX_ROTATION_UNIVERSE: tuple[str, ...] = tuple(dict.fromkeys(BINANCE_TRADFI_ETF_INDEX_SYMBOLS))
# Commodity/macro managed-futures trend universe: the 8 Binance TradFi commodity
# perps (precious metals XAU/XAG/XPT/XPD + energy/industrial COPPER/CL/BZ/NATGAS).
# Commodities trend strongly BOTH ways, so the managed-futures riders routed here
# are long AND short (allow_short=True).
_COMMODITY_TREND_UNIVERSE: tuple[str, ...] = tuple(dict.fromkeys(BINANCE_TRADFI_COMMODITY_SYMBOLS))
# Semis lead-lag follower basket (leader = SOXLUSDT lives in the ETF/index set).
_SEMIS_FOLLOWER_SYMBOLS: tuple[str, ...] = (
    "NVDAUSDT",
    "AMDUSDT",
    "AVGOUSDT",
    "MUUSDT",
    "TSMUSDT",
    "QCOMUSDT",
    "MRVLUSDT",
    "ARMUSDT",
)
_SEMIS_LEADLAG_UNIVERSE: tuple[str, ...] = ("SOXLUSDT", *_SEMIS_FOLLOWER_SYMBOLS)
_INDEX_PER_ASSET_PREFERENCES: tuple[str, ...] = (
    "SPYUSDT",
    "QQQUSDT",
)
# Leveraged / high-beta thematic ETFs that live in the ETF/index perp universe
# (verified members of BINANCE_TRADFI_ETF_INDEX_SYMBOLS): a long-only trend timer
# converts their volatility-decay buy-hold disaster into a high-CAGR vehicle.
_LEVERAGED_ETF_UNIVERSE: tuple[str, ...] = (
    "SOXLUSDT",
    "URNMUSDT",
)
# Defensive risk-off rotation leg (verified members of the tradfi perp universe):
# precious metal (XAU), energy (XLE), long-vol (UVXY). Rotated LONG into the best
# of these by its own momentum when the absolute-momentum gate is risk-off.
_DEFENSIVE_ROTATION_UNIVERSE: tuple[str, ...] = (
    "XAUUSDT",
    "XLEUSDT",
    "UVXYUSDT",
)


def _intersect_universe(
    universe: Sequence[str], normalized_symbols: Sequence[str]
) -> tuple[str, ...]:
    """Intersect a compact-form universe with the canonicalized live symbols."""
    available = set(canonicalize_symbol_list(normalized_symbols))
    out: list[str] = []
    for symbol in canonicalize_symbol_list(universe):
        if symbol in available and symbol not in out:
            out.append(symbol)
    return tuple(out)


# Cross_sectional/carry/momentum trio is MANDATORY for every multi-symbol sleeve.
_CROSS_SECTIONAL_ADMISSION_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "carry",
    "momentum",
)


_HURST_REGIME_GATED_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "balanced_lo",
            "hurst_window": 60,
            "trend_threshold": 0.55,
            "mr_threshold": 0.45,
            "hysteresis": 0.04,
            "mid_weight": 0.30,
            "ema_fast": 20,
            "ema_slow": 60,
            "zscore_window": 60,
            "entry_z": 1.5,
            "lookback_bars": 120,
            "rebalance_band": 0.25,
            "max_positions": 5,
            "stop_loss_pct": 0.060,
            "max_hold_bars": 240,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "hurst_window": 96,
            "trend_threshold": 0.58,
            "mr_threshold": 0.42,
            "hysteresis": 0.05,
            "mid_weight": 0.30,
            "ema_fast": 24,
            "ema_slow": 96,
            "zscore_window": 96,
            "entry_z": 1.8,
            "lookback_bars": 192,
            "rebalance_band": 0.30,
            "max_positions": 4,
            "stop_loss_pct": 0.050,
            "max_hold_bars": 360,
            "allow_short": True,
        },
    ),
    "4h": (
        {
            "variant": "balanced_lo",
            "hurst_window": 48,
            "trend_threshold": 0.55,
            "mr_threshold": 0.45,
            "hysteresis": 0.04,
            "mid_weight": 0.30,
            "ema_fast": 12,
            "ema_slow": 48,
            "zscore_window": 48,
            "entry_z": 1.5,
            "lookback_bars": 96,
            "rebalance_band": 0.25,
            "max_positions": 4,
            "stop_loss_pct": 0.070,
            "max_hold_bars": 120,
            "allow_short": False,
        },
    ),
}


def _build_hurst_regime_gated_candidates(ctx: _CandidateBuildContext) -> None:
    """S9 — Hurst-gated trend/mean-reversion basket overlay (live crypto)."""
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _HURST_REGIME_GATED_SLICE.get(timeframe, ()):
            params = {
                "hurst_window": int(spec["hurst_window"]),
                "trend_threshold": float(spec["trend_threshold"]),
                "mr_threshold": float(spec["mr_threshold"]),
                "hysteresis": float(spec["hysteresis"]),
                "mid_weight": float(spec["mid_weight"]),
                "ema_fast": int(spec["ema_fast"]),
                "ema_slow": int(spec["ema_slow"]),
                "zscore_window": int(spec["zscore_window"]),
                "entry_z": float(spec["entry_z"]),
                "lookback_bars": int(spec["lookback_bars"]),
                "rebalance_band": float(spec["rebalance_band"]),
                "max_positions": int(spec["max_positions"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"hurst_regime_gated_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['hurst_window'])}_{float(spec['trend_threshold']):.2f}"
                ),
                family="cross_sectional",
                strategy_class="HurstRegimeGatedStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Hurst-gated regime overlay that blends a trend child and a "
                    "mean-reversion child per symbol with hysteresis to avoid "
                    f"thrash near H~0.5 for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "regime",
                    "hurst",
                    "mean_reversion",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "allow_short": bool(spec["allow_short"]),
                    "decision_cadence_seconds": 3600,
                },
            )


_CONFIDENCE_GATED_TREND_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "calm_lo",
            "ema_fast": 20,
            "ema_slow": 60,
            "confidence_threshold": 0.55,
            "vol_ratio_short_window": 16,
            "vol_ratio_long_window": 96,
            "max_vol_ratio": 1.4,
            "funding_z_window": 96,
            "max_funding_z": 2.0,
            "win_rate_window": 24,
            "trend_lookback_bars": 24,
            "lookback_bars": 120,
            "rebalance_band": 0.15,
            "allow_short": False,
            "stop_loss_pct": 0.040,
            "take_profit_pct": 0.0,
            "max_hold_bars": 240,
        },
        {
            "variant": "conviction_ls",
            "ema_fast": 24,
            "ema_slow": 96,
            "confidence_threshold": 0.62,
            "vol_ratio_short_window": 24,
            "vol_ratio_long_window": 144,
            "max_vol_ratio": 1.3,
            "funding_z_window": 120,
            "max_funding_z": 1.8,
            "win_rate_window": 36,
            "trend_lookback_bars": 36,
            "lookback_bars": 192,
            "rebalance_band": 0.20,
            "allow_short": True,
            "stop_loss_pct": 0.035,
            "take_profit_pct": 0.0,
            "max_hold_bars": 360,
        },
    ),
}


def _build_confidence_gated_trend_candidates(ctx: _CandidateBuildContext) -> None:
    """S13 — per-symbol confidence-gated trend (single-asset, outside the gate)."""
    crypto_symbols = ctx.crypto_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("1h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _CONFIDENCE_GATED_TREND_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "ema_fast": int(spec["ema_fast"]),
                    "ema_slow": int(spec["ema_slow"]),
                    "confidence_threshold": float(spec["confidence_threshold"]),
                    "vol_ratio_short_window": int(spec["vol_ratio_short_window"]),
                    "vol_ratio_long_window": int(spec["vol_ratio_long_window"]),
                    "max_vol_ratio": float(spec["max_vol_ratio"]),
                    "funding_z_window": int(spec["funding_z_window"]),
                    "max_funding_z": float(spec["max_funding_z"]),
                    "win_rate_window": int(spec["win_rate_window"]),
                    "trend_lookback_bars": int(spec["trend_lookback_bars"]),
                    "lookback_bars": int(spec["lookback_bars"]),
                    "rebalance_band": float(spec["rebalance_band"]),
                    "allow_short": bool(spec["allow_short"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "take_profit_pct": float(spec["take_profit_pct"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"confidence_gated_trend_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['confidence_threshold']):.2f}"
                    ),
                    family="trend",
                    strategy_class="ConfidenceGatedTrendStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol rule-based confidence-gated trend sleeve that "
                        "only trades high-conviction setups (calm funding, low "
                        "vol ratio, positive recent win-rate) to lift deflated "
                        f"Sharpe and lower PBO for {symbol} on {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "confidence_gated",
                        "single_asset",
                        "meta_gate",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": 3600,
                    },
                )


# Per-symbol directional RETURN-RIDER sleeves: meaningful TopCap-scale exposure,
# ATR trailing stops that let winners run, and pyramiding into continuation for
# high compound return.  Wired ONLY at >=30m timeframes (never 1s/1m/5m/15m) with
# SHORT windows so they fire many trades on a ~1-month window.
# Per-timeframe decision cadence (seconds) for the >=30m return riders, so a 4h
# row decides every 4h and a 1d row every day (not every 30m).
_RIDER_TF_CADENCE_SECONDS: dict[str, int] = {
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_ADAPTIVE_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "kama_period": 10,
            "kama_fast": 2,
            "kama_slow": 24,
            "min_efficiency": 0.28,
            "slope_lookback": 2,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "kama_period": 12,
            "kama_fast": 2,
            "kama_slow": 30,
            "min_efficiency": 0.30,
            "slope_lookback": 3,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "kama_period": 10,
            "kama_fast": 2,
            "kama_slow": 24,
            "min_efficiency": 0.30,
            "slope_lookback": 2,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "kama_period": 8,
            "kama_fast": 2,
            "kama_slow": 20,
            "min_efficiency": 0.30,
            "slope_lookback": 1,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

_VOLATILITY_BREAKOUT_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "donchian_window": 20,
            "atr_expansion_mult": 1.05,
            "atr_baseline_window": 48,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "donchian_window": 24,
            "atr_expansion_mult": 1.10,
            "atr_baseline_window": 60,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "donchian_window": 18,
            "atr_expansion_mult": 1.10,
            "atr_baseline_window": 48,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "donchian_window": 14,
            "atr_expansion_mult": 1.10,
            "atr_baseline_window": 30,
            "trail_atr_mult": 4.5,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

_ACCELERATION_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "roc_period": 8,
            "min_roc": 0.0,
            "decel_tolerance": 0.0,
            "trail_atr_mult": 2.5,
            "atr_period": 14,
            "max_adds": 4,
            "add_step_atr": 0.75,
            "vol_window": 48,
            "target_vol": 0.025,
            "max_hold_bars": 160,
            "allow_short": True,
            "add_alloc_fraction": 0.6,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "roc_period": 10,
            "min_roc": 0.0,
            "decel_tolerance": 0.0,
            "trail_atr_mult": 2.8,
            "atr_period": 14,
            "max_adds": 4,
            "add_step_atr": 0.75,
            "vol_window": 48,
            "target_vol": 0.025,
            "max_hold_bars": 150,
            "allow_short": True,
            "add_alloc_fraction": 0.6,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "roc_period": 8,
            "min_roc": 0.0,
            "decel_tolerance": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 0.75,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 100,
            "allow_short": True,
            "add_alloc_fraction": 0.6,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "roc_period": 6,
            "min_roc": 0.0,
            "decel_tolerance": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 0.75,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.6,
        },
    ),
}


def _build_adaptive_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol KAMA/efficiency adaptive-trend rider (single-asset, return-max)."""
    # Crypto-only: tradfi equity/ETF perps are routed through the dedicated
    # equity single-name rider, so this crypto sleeve must not trade them.
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _ADAPTIVE_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "kama_period": int(spec["kama_period"]),
                    "kama_fast": int(spec["kama_fast"]),
                    "kama_slow": int(spec["kama_slow"]),
                    "min_efficiency": float(spec["min_efficiency"]),
                    "slope_lookback": int(spec["slope_lookback"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"adaptive_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['trail_atr_mult']):.1f}"
                    ),
                    family="trend",
                    strategy_class="AdaptiveTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol KAMA/efficiency-confirmed adaptive-trend rider "
                        "that rides winners with an ATR trailing stop and pyramids "
                        "into continuation for high compound return on "
                        f"{symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


def _build_volatility_breakout_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol Donchian/ATR-expansion breakout rider (single-asset, return-max)."""
    # Crypto-only: tradfi equity/ETF perps are routed through the dedicated
    # equity/leveraged-ETF trend builders, so this crypto sleeve excludes them.
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _VOLATILITY_BREAKOUT_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "donchian_window": int(spec["donchian_window"]),
                    "atr_expansion_mult": float(spec["atr_expansion_mult"]),
                    "atr_baseline_window": int(spec["atr_baseline_window"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"volatility_breakout_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['donchian_window'])}"
                    ),
                    family="breakout",
                    strategy_class="VolatilityBreakoutRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol Donchian breakout rider confirmed by ATR "
                        "expansion that rides the move with an ATR trailing stop "
                        "and pyramids on follow-through to capture explosive "
                        f"range expansions on {symbol} at {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "breakout",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


def _build_acceleration_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol accelerating-momentum rider (single-asset, return-max)."""
    # Crypto-only: tradfi equity/ETF perps are routed through the dedicated
    # equity trend/rotation builders, so this crypto sleeve excludes them.
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _ACCELERATION_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "roc_period": int(spec["roc_period"]),
                    "min_roc": float(spec["min_roc"]),
                    "decel_tolerance": float(spec["decel_tolerance"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"acceleration_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['roc_period'])}"
                    ),
                    family="momentum",
                    strategy_class="AccelerationRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol accelerating-momentum rider that enters when "
                        "rate-of-change is positive and rising (or negative and "
                        "falling), rides with an ATR trailing stop, and pyramids "
                        "while acceleration persists to capture parabolic moves on "
                        f"{symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "momentum",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# --------------------------------------------------------------------------- #
# Micro-signal-informed sleeves: LOOK intrabar, TRADE at >=30m.
#
# Every candidate below carries decision_cadence_seconds=1800 in metadata REGARDLESS
# of its base candidate timeframe -- the underlying classes hard-pin the 30m decision
# throttle as a class attribute, so the candidate timeframe is only the bar grain the
# sleeve accumulates as DECISION BARS. Candidate timeframes are kept >=30m
# (30m/1h/4h/1d) for consistency with the >=30m mandate.
# --------------------------------------------------------------------------- #
# Cadence stamped on every micro-signal candidate's metadata. Unlike the return
# riders (which scale cadence with the base TF), these sleeves decide at 30m for
# EVERY base timeframe because the class attribute pins it there.
_MICRO_SIGNAL_DECISION_CADENCE_SECONDS = 1800

# #1 IntradayFlowPressureRiderStrategy: per-symbol taker-flow CONTINUATION ride.
# Crypto-only (taker fields are crypto-perp) and gated on perp_support_data.
_INTRADAY_FLOW_PRESSURE_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "flow_z_window": 6,
            "entry_z": 1.4,
            "pyramid_z": 1.9,
            "tick_agree_frac": 0.55,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "flow_z_window": 8,
            "entry_z": 1.5,
            "pyramid_z": 2.0,
            "tick_agree_frac": 0.55,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "flow_z_window": 6,
            "entry_z": 1.6,
            "pyramid_z": 2.1,
            "tick_agree_frac": 0.55,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "flow_z_window": 5,
            "entry_z": 1.6,
            "pyramid_z": 2.1,
            "tick_agree_frac": 0.55,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

# #2 VolOfVolRegimeTrendGateStrategy: directional trend SIZED by a GK/rv vol-of-vol
# governor + KER cleanliness gate. OHLCV-only -> widest crypto universe.
_VOL_OF_VOL_REGIME_TREND_GATE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "trend_z_window": 48,
            "trend_score_min": 0.10,
            "gk_window": 20,
            "rv_window": 20,
            "ker_period": 10,
            "clean_threshold": 0.40,
            "choppy_threshold": 0.20,
            "gk_rv_history_window": 32,
            "gk_rv_veto_rel": 1.6,
            "gk_rv_clean_rel": 1.0,
            "downsize_floor": 0.40,
            "upsize_cap": 1.5,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "trend_z_window": 48,
            "trend_score_min": 0.12,
            "gk_window": 20,
            "rv_window": 20,
            "ker_period": 12,
            "clean_threshold": 0.42,
            "choppy_threshold": 0.22,
            "gk_rv_history_window": 32,
            "gk_rv_veto_rel": 1.6,
            "gk_rv_clean_rel": 1.0,
            "downsize_floor": 0.40,
            "upsize_cap": 1.5,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "trend_z_window": 36,
            "trend_score_min": 0.12,
            "gk_window": 16,
            "rv_window": 16,
            "ker_period": 10,
            "clean_threshold": 0.42,
            "choppy_threshold": 0.22,
            "gk_rv_history_window": 32,
            "gk_rv_veto_rel": 1.6,
            "gk_rv_clean_rel": 1.0,
            "downsize_floor": 0.40,
            "upsize_cap": 1.5,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "trend_z_window": 24,
            "trend_score_min": 0.12,
            "gk_window": 12,
            "rv_window": 12,
            "ker_period": 8,
            "clean_threshold": 0.42,
            "choppy_threshold": 0.22,
            "gk_rv_history_window": 32,
            "gk_rv_veto_rel": 1.6,
            "gk_rv_clean_rel": 1.0,
            "downsize_floor": 0.40,
            "upsize_cap": 1.5,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

# #5 VWAPCompressionReversionStrategy: vol-compression-gated VWAP-deviation mean
# reversion (wires volcomp_vwap_pressure). OHLCV-only.
_VWAP_COMPRESSION_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "vwap_window": 48,
            "z_window": 96,
            "bandwidth_window": 48,
            "percentile_window": 192,
            "compression_percentile": 0.30,
            "compression_vol_ratio": 0.85,
            "entry_z": 2.0,
            "exit_z": 0.0,
            "max_hold_bars": 48,
            "stop_loss_pct": 0.03,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "vwap_window": 48,
            "z_window": 96,
            "bandwidth_window": 48,
            "percentile_window": 192,
            "compression_percentile": 0.30,
            "compression_vol_ratio": 0.85,
            "entry_z": 2.0,
            "exit_z": 0.0,
            "max_hold_bars": 36,
            "stop_loss_pct": 0.03,
            "allow_short": True,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "vwap_window": 36,
            "z_window": 72,
            "bandwidth_window": 36,
            "percentile_window": 160,
            "compression_percentile": 0.30,
            "compression_vol_ratio": 0.85,
            "entry_z": 2.1,
            "exit_z": 0.0,
            "max_hold_bars": 24,
            "stop_loss_pct": 0.04,
            "allow_short": True,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "vwap_window": 24,
            "z_window": 60,
            "bandwidth_window": 24,
            "percentile_window": 120,
            "compression_percentile": 0.30,
            "compression_vol_ratio": 0.85,
            "entry_z": 2.1,
            "exit_z": 0.0,
            "max_hold_bars": 16,
            "stop_loss_pct": 0.05,
            "allow_short": True,
        },
    ),
}


def _build_intraday_flow_pressure_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol taker-flow CONTINUATION ride (#1, single-asset, crypto-perp).

    Gated on perp-support data because the core taker-flow read is a crypto-perp
    field; crypto-only symbols (tradfi perps routed elsewhere). Decides at 30m for
    EVERY base timeframe (the class hard-pins decision_cadence_seconds=1800).
    """
    if not ctx.perp_support_data_available:
        return
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _INTRADAY_FLOW_PRESSURE_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "flow_z_window": int(spec["flow_z_window"]),
                    "entry_z": float(spec["entry_z"]),
                    "pyramid_z": float(spec["pyramid_z"]),
                    "tick_agree_frac": float(spec["tick_agree_frac"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"intraday_flow_pressure_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['entry_z']):.1f}"
                    ),
                    family="flow",
                    strategy_class="IntradayFlowPressureRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol taker-flow PRESSURE continuation rider: a 30m "
                        "decision-bar flow-imbalance z-score gate confirmed by the "
                        "latest 1s tick-agreement, ridden with an ATR trailing stop "
                        "and pyramided on flow extension. Looks intrabar, decides at "
                        f"30m on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "flow",
                        "taker_flow",
                        "continuation",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                        "micro_signal",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "data_dependent": ctx.perp_support_data_available,
                        "decision_cadence_seconds": _MICRO_SIGNAL_DECISION_CADENCE_SECONDS,
                    },
                )


def _build_vol_of_vol_regime_trend_gate_candidates(ctx: _CandidateBuildContext) -> None:
    """Directional trend SIZED by a GK/rv vol-of-vol governor (#2, OHLCV-only).

    Crypto-only universe (tradfi perps routed through the equity builders).
    Decides at 30m for every base timeframe (class hard-pins the throttle).
    """
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _VOL_OF_VOL_REGIME_TREND_GATE_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "trend_z_window": int(spec["trend_z_window"]),
                    "trend_score_min": float(spec["trend_score_min"]),
                    "gk_window": int(spec["gk_window"]),
                    "rv_window": int(spec["rv_window"]),
                    "ker_period": int(spec["ker_period"]),
                    "clean_threshold": float(spec["clean_threshold"]),
                    "choppy_threshold": float(spec["choppy_threshold"]),
                    "gk_rv_history_window": int(spec["gk_rv_history_window"]),
                    "gk_rv_veto_rel": float(spec["gk_rv_veto_rel"]),
                    "gk_rv_clean_rel": float(spec["gk_rv_clean_rel"]),
                    "downsize_floor": float(spec["downsize_floor"]),
                    "upsize_cap": float(spec["upsize_cap"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"vol_of_vol_regime_trend_gate_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['gk_rv_veto_rel']):.1f}"
                    ),
                    family="trend",
                    strategy_class="VolOfVolRegimeTrendGateStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol directional trend (pv_trend_score) whose SIZE is "
                        "governed by a Garman-Klass-vs-realized vol-of-vol read plus "
                        "a Kaufman-efficiency cleanliness gate: hidden intrabar stress "
                        "down-sizes/vetoes, clean efficient trends up-size. Decides at "
                        f"30m on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "vol_of_vol",
                        "regime_gate",
                        "return_rider",
                        "trailing_stop",
                        "single_asset",
                        "crypto",
                        "micro_signal",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _MICRO_SIGNAL_DECISION_CADENCE_SECONDS,
                    },
                )


def _build_vwap_compression_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    """Vol-compression-gated VWAP-deviation mean reversion (#5, OHLCV-only).

    Crypto-only universe. Decides at 30m for every base timeframe (the class
    hard-pins decision_cadence_seconds=1800).
    """
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _VWAP_COMPRESSION_REVERSION_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "vwap_window": int(spec["vwap_window"]),
                    "z_window": int(spec["z_window"]),
                    "bandwidth_window": int(spec["bandwidth_window"]),
                    "percentile_window": int(spec["percentile_window"]),
                    "compression_percentile": float(spec["compression_percentile"]),
                    "compression_vol_ratio": float(spec["compression_vol_ratio"]),
                    "entry_z": float(spec["entry_z"]),
                    "exit_z": float(spec["exit_z"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "allow_short": bool(spec["allow_short"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"vwap_compression_reversion_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['entry_z']):.1f}"
                    ),
                    family="mean_reversion",
                    strategy_class="VWAPCompressionReversionStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol VWAP-deviation mean reversion gated by a coiled "
                        "vol-compression regime (volcomp_vwap_pressure): only acts when "
                        "compression is active, anchored to the boundary window's 1s "
                        "volume-weighted typical price for a precise deviation. Decides "
                        f"at 30m on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "mean_reversion",
                        "vwap",
                        "vol_compression",
                        "single_asset",
                        "crypto",
                        "micro_signal",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _MICRO_SIGNAL_DECISION_CADENCE_SECONDS,
                    },
                )


# S-EQ1 single-name EQUITY trend rider: REUSES AdaptiveTrendRiderStrategy with
# equity-tuned, LONG-ONLY params (equities drift up; shorting single names invites
# squeeze risk). One single-asset candidate per equity perp, 1d primary + 4h
# faster variant. Long, slower trend windows than the crypto rider profiles.
_EQUITY_SINGLE_NAME_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "swing_long",
            "kama_period": 14,
            "kama_fast": 2,
            "kama_slow": 30,
            "min_efficiency": 0.32,
            "slope_lookback": 2,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.020,
            "max_hold_bars": 360,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_long",
            "kama_period": 20,
            "kama_fast": 2,
            "kama_slow": 40,
            "min_efficiency": 0.35,
            "slope_lookback": 3,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 20,
            "target_vol": 0.025,
            "max_hold_bars": 252,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_equity_single_name_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """S-EQ1 — single-name EQUITY trend rider (reuse AdaptiveTrendRider, long-only)."""
    equity_symbols = _intersect_universe(_EQUITY_FACTOR_UNIVERSE, ctx.normalized_symbols)
    if not equity_symbols:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _EQUITY_SINGLE_NAME_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in equity_symbols:
                params = {
                    "kama_period": int(spec["kama_period"]),
                    "kama_fast": int(spec["kama_fast"]),
                    "kama_slow": int(spec["kama_slow"]),
                    "min_efficiency": float(spec["min_efficiency"]),
                    "slope_lookback": int(spec["slope_lookback"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"equity_single_name_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['trail_atr_mult']):.1f}"
                    ),
                    family="trend",
                    strategy_class="AdaptiveTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "DORMANT equity tranche: long-only single-name KAMA/"
                        "efficiency adaptive-trend rider that rides secular equity "
                        "winners with an ATR trailing stop and pyramids into "
                        f"continuation for high compound return on {symbol} at "
                        f"{timeframe} ({spec['variant']}); self-skips until equity "
                        "perps materialize."
                    ),
                    tags=(
                        "trend",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "equity",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "dormant_tranche": True,
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# S-CMDY1/2/3 commodity/macro MANAGED-FUTURES trend riders: REUSE the three proven
# return-rider classes (AdaptiveTrendRiderStrategy / VolatilityBreakoutRiderStrategy /
# AccelerationRiderStrategy) routed to the 8 commodity perps. Commodities trend
# strongly BOTH ways (oil/metals/gas have long structural bull AND bear legs), so
# every commodity rider is LONG AND SHORT (allow_short=True). One single-asset
# candidate per commodity symbol; commodity-tuned slice params at 4h + 1d
# (short-enough trend/breakout/ROC windows that the sleeves actually trade).
_COMMODITY_ADAPTIVE_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "macro_swing_ls",
            "kama_period": 12,
            "kama_fast": 2,
            "kama_slow": 28,
            "min_efficiency": 0.30,
            "slope_lookback": 2,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.020,
            "max_hold_bars": 240,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_trend_ls",
            "kama_period": 10,
            "kama_fast": 2,
            "kama_slow": 24,
            "min_efficiency": 0.30,
            "slope_lookback": 1,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

_COMMODITY_BREAKOUT_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "macro_swing_ls",
            "donchian_window": 20,
            "atr_expansion_mult": 1.10,
            "atr_baseline_window": 48,
            "trail_atr_mult": 3.8,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.020,
            "max_hold_bars": 240,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_trend_ls",
            "donchian_window": 20,
            "atr_expansion_mult": 1.10,
            "atr_baseline_window": 40,
            "trail_atr_mult": 4.2,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

_COMMODITY_ACCELERATION_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "macro_swing_ls",
            "roc_period": 10,
            "min_roc": 0.0,
            "decel_tolerance": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 0.75,
            "vol_window": 36,
            "target_vol": 0.025,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.6,
        },
    ),
    "1d": (
        {
            "variant": "macro_trend_ls",
            "roc_period": 8,
            "min_roc": 0.0,
            "decel_tolerance": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 0.75,
            "vol_window": 24,
            "target_vol": 0.030,
            "max_hold_bars": 100,
            "allow_short": True,
            "add_alloc_fraction": 0.6,
        },
    ),
}


def _build_commodity_adaptive_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """S-CMDY1 — commodity managed-futures KAMA trend rider (reuse AdaptiveTrendRider, long/short)."""
    commodity_symbols = _intersect_universe(_COMMODITY_TREND_UNIVERSE, ctx.normalized_symbols)
    if not commodity_symbols:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _COMMODITY_ADAPTIVE_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in commodity_symbols:
                params = {
                    "kama_period": int(spec["kama_period"]),
                    "kama_fast": int(spec["kama_fast"]),
                    "kama_slow": int(spec["kama_slow"]),
                    "min_efficiency": float(spec["min_efficiency"]),
                    "slope_lookback": int(spec["slope_lookback"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"commodity_adaptive_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['trail_atr_mult']):.1f}"
                    ),
                    family="trend",
                    strategy_class="AdaptiveTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Commodity/macro managed-futures KAMA/efficiency trend rider "
                        "that rides commodity trends BOTH ways (long and short) with "
                        "an ATR trailing stop and pyramids into continuation for high "
                        f"compound return on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "commodity",
                        "managed_futures",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


def _build_commodity_breakout_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """S-CMDY2 — commodity managed-futures Donchian breakout rider (reuse VolBreakoutRider, long/short)."""
    commodity_symbols = _intersect_universe(_COMMODITY_TREND_UNIVERSE, ctx.normalized_symbols)
    if not commodity_symbols:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _COMMODITY_BREAKOUT_RIDER_SLICE.get(timeframe, ()):
            for symbol in commodity_symbols:
                params = {
                    "donchian_window": int(spec["donchian_window"]),
                    "atr_expansion_mult": float(spec["atr_expansion_mult"]),
                    "atr_baseline_window": int(spec["atr_baseline_window"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"commodity_breakout_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['donchian_window'])}"
                    ),
                    family="breakout",
                    strategy_class="VolatilityBreakoutRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Commodity/macro managed-futures Donchian breakout rider "
                        "confirmed by ATR expansion that rides commodity range "
                        "expansions BOTH ways (long and short) with an ATR trailing "
                        "stop and pyramids on follow-through for high compound return "
                        f"on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "breakout",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "commodity",
                        "managed_futures",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


def _build_commodity_acceleration_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """S-CMDY3 — commodity managed-futures accelerating-momentum rider (reuse AccelerationRider, long/short)."""
    commodity_symbols = _intersect_universe(_COMMODITY_TREND_UNIVERSE, ctx.normalized_symbols)
    if not commodity_symbols:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _COMMODITY_ACCELERATION_RIDER_SLICE.get(timeframe, ()):
            for symbol in commodity_symbols:
                params = {
                    "roc_period": int(spec["roc_period"]),
                    "min_roc": float(spec["min_roc"]),
                    "decel_tolerance": float(spec["decel_tolerance"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"commodity_acceleration_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['roc_period'])}"
                    ),
                    family="momentum",
                    strategy_class="AccelerationRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Commodity/macro managed-futures accelerating-momentum rider "
                        "that enters on positive-and-rising (or negative-and-falling) "
                        "rate-of-change, rides commodity trends BOTH ways (long and "
                        "short) with an ATR trailing stop, and pyramids while "
                        "acceleration persists for high compound return on "
                        f"{symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "momentum",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "commodity",
                        "managed_futures",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# S-EQ-52WH equity 52-WEEK-HIGH breakout momentum rider: REUSES
# VolatilityBreakoutRiderStrategy with a LONG ~252-bar (52-week daily) Donchian
# new-high lookback. George/Hwang 52-week-high momentum — stocks trading near
# their 52-week high keep outperforming; ride the new high with the inherited ATR
# trailing stop. LONG-ONLY (allow_short=False — the 52w-high effect is a long
# anomaly). One single-asset candidate per equity perp, 1d primary (~252 = 52
# weeks) + 4h faster variant (~252 4h bars). donchian_window schema max is 4096,
# so 252 is well within range.
_EQUITY_NEW_HIGH_BREAKOUT_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "fast_52wh_long",
            "donchian_window": 252,
            "atr_expansion_mult": 1.05,
            "atr_baseline_window": 96,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.020,
            "max_hold_bars": 504,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_52wh_long",
            "donchian_window": 252,
            "atr_expansion_mult": 1.05,
            "atr_baseline_window": 60,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 20,
            "target_vol": 0.025,
            "max_hold_bars": 252,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_equity_new_high_breakout_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """S-EQ-52WH — equity 52-week-high breakout momentum rider (reuse VolBreakoutRider, long-only)."""
    equity_symbols = _intersect_universe(_EQUITY_FACTOR_UNIVERSE, ctx.normalized_symbols)
    if not equity_symbols:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _EQUITY_NEW_HIGH_BREAKOUT_RIDER_SLICE.get(timeframe, ()):
            for symbol in equity_symbols:
                params = {
                    "donchian_window": int(spec["donchian_window"]),
                    "atr_expansion_mult": float(spec["atr_expansion_mult"]),
                    "atr_baseline_window": int(spec["atr_baseline_window"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"equity_new_high_breakout_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['donchian_window'])}"
                    ),
                    family="breakout",
                    strategy_class="VolatilityBreakoutRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "George/Hwang 52-week-high momentum: long-only single-name "
                        "equity rider that buys a fresh ~252-bar (52-week) new high "
                        "and rides it with the inherited ATR trailing stop, pyramiding "
                        "on follow-through. Stocks near their 52-week high keep "
                        f"outperforming on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "breakout",
                        "momentum",
                        "52w_high",
                        "single_asset",
                        "equity",
                        "return_rider",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# S-EQ2 leveraged-ETF trend timing: NEW LeveragedTrendTimingRiderStrategy, LONG
# only above the long regime SMA + golden cross, with decay-aware sizing. One
# single-asset candidate per leveraged/high-beta ETF, 1d primary + 4h variant.
_LEVERAGED_TREND_TIMING_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "swing_timer",
            "regime_sma_bars": 200,
            "fast_sma_bars": 50,
            "slow_sma_bars": 200,
            "confirm_bars": 3,
            "trend_buffer": 0.0,
            "decay_vol_ref": 0.035,
            "decay_floor": 0.25,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 30,
            "target_vol": 0.020,
            "max_hold_bars": 360,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_timer",
            "regime_sma_bars": 200,
            "fast_sma_bars": 50,
            "slow_sma_bars": 200,
            "confirm_bars": 3,
            "trend_buffer": 0.0,
            "decay_vol_ref": 0.030,
            "decay_floor": 0.25,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 20,
            "target_vol": 0.025,
            "max_hold_bars": 252,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_leveraged_trend_timing_candidates(ctx: _CandidateBuildContext) -> None:
    """S-EQ2 — leveraged-ETF trend timer with decay-aware sizing (single-asset, long-only)."""
    letf_symbols = _intersect_universe(_LEVERAGED_ETF_UNIVERSE, ctx.normalized_symbols)
    if not letf_symbols:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _LEVERAGED_TREND_TIMING_SLICE.get(timeframe, ()):
            for symbol in letf_symbols:
                params = {
                    "regime_sma_bars": int(spec["regime_sma_bars"]),
                    "fast_sma_bars": int(spec["fast_sma_bars"]),
                    "slow_sma_bars": int(spec["slow_sma_bars"]),
                    "confirm_bars": int(spec["confirm_bars"]),
                    "trend_buffer": float(spec["trend_buffer"]),
                    "decay_vol_ref": float(spec["decay_vol_ref"]),
                    "decay_floor": float(spec["decay_floor"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": False,
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"leveraged_trend_timing_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['regime_sma_bars'])}"
                    ),
                    family="trend",
                    strategy_class="LeveragedTrendTimingRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "DORMANT index tranche: long-only leveraged-ETF trend timer "
                        "that holds only above the long regime SMA and a confirmed "
                        "golden cross, rides with an ATR trailing stop, and shrinks "
                        "size as realized volatility rises (decay penalty) on "
                        f"{symbol} at {timeframe} ({spec['variant']}); self-skips "
                        "until the leveraged ETF perp materializes."
                    ),
                    tags=(
                        "trend",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "index",
                        "leveraged_etf",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": False,
                        "dormant_tranche": True,
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# S-EQ3 dual-momentum index rotation WITH a defensive leg: NEW
# DualMomentumDefensiveRotationStrategy. Risk-on -> rank index/ETF universe and
# hold top-N above SMA; risk-off -> rotate 100% LONG into the best of the
# defensive universe (XAU/XLE/UVXY) by its own momentum. Basket/multi, 1d only.
_DUAL_MOMENTUM_DEFENSIVE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "absrel_defensive_top3",
            "absolute_lookback_bars": 12,
            "blend_lookbacks": "1,3,6,12",
            "sma_bars": 200,
            "rebalance_bars": 21,
            "max_holdings": 3,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 252,
        },
    ),
}


def _build_dual_momentum_defensive_candidates(ctx: _CandidateBuildContext) -> None:
    """S-EQ3 — dual-momentum index rotation with a long defensive leg (multi basket)."""
    filtered = _intersect_universe(_INDEX_ROTATION_UNIVERSE, ctx.normalized_symbols)
    defensive = _intersect_universe(_DEFENSIVE_ROTATION_UNIVERSE, ctx.normalized_symbols)
    # Need >=4 rotation ETFs AND at least one defensive instrument so the risk-off
    # leg has somewhere to rotate; otherwise self-skip (no flat-only duplication of
    # the plain dual-momentum rotation sleeve).
    if len(filtered) < 4 or not defensive:
        return
    union_symbols = tuple(dict.fromkeys((*filtered, *defensive)))
    for timeframe in ctx._present("1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _DUAL_MOMENTUM_DEFENSIVE_SLICE.get(timeframe, ()):
            params = {
                "defensive_symbols": ",".join(defensive),
                "absolute_lookback_bars": int(spec["absolute_lookback_bars"]),
                "blend_lookbacks": str(spec["blend_lookbacks"]),
                "sma_bars": int(spec["sma_bars"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "max_holdings": int(spec["max_holdings"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"dual_momentum_defensive_rotation_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['max_holdings'])}_{int(spec['sma_bars'])}"
                ),
                family="cross_sectional",
                strategy_class="DualMomentumDefensiveRotationStrategy",
                timeframe=timeframe,
                symbols=union_symbols,
                params=params,
                notes=(
                    "DORMANT index tranche: dual-momentum rotation that gates on an "
                    "absolute-return filter, rotates into the top blended-momentum "
                    "index perps above their SMA when risk-on, and rotates 100% LONG "
                    "into the best defensive instrument (metal/energy/long-vol) by "
                    f"its own momentum when risk-off for {timeframe} "
                    f"({spec['variant']}); self-skips until index perps materialize."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "index",
                    "rotation",
                    "defensive",
                    "dormant",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "index",
                    "dormant_tranche": True,
                    "decision_cadence_seconds": 86400,
                },
            )


# Per-symbol multi-horizon trend-ENSEMBLE rider (enter only when short/medium/long
# horizons agree; ride + pyramid). Short windows so it fires often on a ~1-month
# >=30m crypto5 window. Single-asset, family ``trend``.
_MULTI_TF_TREND_ENSEMBLE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "short_lookback": 6,
            "mid_lookback": 18,
            "long_lookback": 48,
            "align_threshold": 2,
            "min_horizon_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "short_lookback": 6,
            "mid_lookback": 18,
            "long_lookback": 48,
            "align_threshold": 2,
            "min_horizon_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "short_lookback": 4,
            "mid_lookback": 12,
            "long_lookback": 36,
            "align_threshold": 2,
            "min_horizon_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "short_lookback": 3,
            "mid_lookback": 8,
            "long_lookback": 21,
            "align_threshold": 2,
            "min_horizon_roc": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

# Per-symbol buy-the-dip / sell-the-rally trend-continuation rider. Short windows
# so it fires often. Single-asset, family ``trend``.
_PULLBACK_TREND_CONTINUATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "short_ma_window": 10,
            "pullback_roc_period": 3,
            "pullback_atr_mult": 0.5,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "short_ma_window": 10,
            "pullback_roc_period": 3,
            "pullback_atr_mult": 0.5,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "short_ma_window": 8,
            "pullback_roc_period": 2,
            "pullback_atr_mult": 0.5,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "trend_lookback": 21,
            "trend_ma_window": 21,
            "short_ma_window": 5,
            "pullback_roc_period": 2,
            "pullback_atr_mult": 0.5,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

# Per-symbol directional funding-carry HARVEST rider (collect persistent funding
# sign as directional income). Single-asset, family ``carry``. Funding feature
# required -> builder gates on ``ctx.perp_support_data_available``.
_FUNDING_HARVEST_CARRY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "funding_window": 8,
            "entry_funding": 0.00005,
            "exit_funding": 0.0,
            "funding_scale": 0.0003,
            "no_fight_roc_period": 6,
            "no_fight_roc": 0.05,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 480,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "funding_window": 8,
            "entry_funding": 0.00005,
            "exit_funding": 0.0,
            "funding_scale": 0.0003,
            "no_fight_roc_period": 6,
            "no_fight_roc": 0.05,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 360,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "funding_window": 6,
            "entry_funding": 0.00005,
            "exit_funding": 0.0,
            "funding_scale": 0.0003,
            "no_fight_roc_period": 4,
            "no_fight_roc": 0.06,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "funding_window": 4,
            "entry_funding": 0.00005,
            "exit_funding": 0.0,
            "funding_scale": 0.0003,
            "no_fight_roc_period": 3,
            "no_fight_roc": 0.08,
            "trail_atr_mult": 4.5,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 90,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_multi_timeframe_trend_ensemble_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol multi-horizon trend-ensemble rider (single-asset, return-max)."""
    crypto_symbols = ctx.crypto_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _MULTI_TF_TREND_ENSEMBLE_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "short_lookback": int(spec["short_lookback"]),
                    "mid_lookback": int(spec["mid_lookback"]),
                    "long_lookback": int(spec["long_lookback"]),
                    "align_threshold": int(spec["align_threshold"]),
                    "min_horizon_roc": float(spec["min_horizon_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"multi_timeframe_trend_ensemble_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['align_threshold'])}of3"
                    ),
                    family="trend",
                    strategy_class="MultiTimeframeTrendEnsembleStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol multi-horizon trend-ensemble rider that enters "
                        "only when the short/medium/long-horizon trends agree, sizes "
                        "by agreement conviction, and rides winners with an ATR "
                        "trailing stop + pyramiding for high compound return on "
                        f"{symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "return_rider",
                        "multi_horizon",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


def _build_pullback_trend_continuation_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol buy-the-dip / sell-the-rally continuation rider (single-asset)."""
    crypto_symbols = ctx.crypto_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _PULLBACK_TREND_CONTINUATION_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "short_ma_window": int(spec["short_ma_window"]),
                    "pullback_roc_period": int(spec["pullback_roc_period"]),
                    "pullback_atr_mult": float(spec["pullback_atr_mult"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"pullback_trend_continuation_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['short_ma_window'])}"
                    ),
                    family="trend",
                    strategy_class="PullbackTrendContinuationStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol trend-continuation rider that establishes the "
                        "long-horizon trend then enters on a pullback in the trend "
                        "direction (buy-the-dip / sell-the-rally) for a better entry "
                        "price, riding with an ATR trailing stop + pyramiding on "
                        f"{symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "return_rider",
                        "pullback",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


def _build_funding_harvest_carry_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol directional funding-carry harvest rider (single-asset, feature-gated)."""
    if not ctx.perp_support_data_available:
        return
    crypto_symbols = ctx.crypto_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _FUNDING_HARVEST_CARRY_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "funding_window": int(spec["funding_window"]),
                    "entry_funding": float(spec["entry_funding"]),
                    "exit_funding": float(spec["exit_funding"]),
                    "funding_scale": float(spec["funding_scale"]),
                    "no_fight_roc_period": int(spec["no_fight_roc_period"]),
                    "no_fight_roc": float(spec["no_fight_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"funding_harvest_carry_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['funding_window'])}"
                    ),
                    family="carry",
                    strategy_class="FundingHarvestCarryStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol directional funding-carry harvest that goes long "
                        "into persistently negative funding (longs paid) and short "
                        "into persistently positive funding (shorts paid), with a "
                        "trend-no-fight guard, riding the carry with an ATR trailing "
                        f"stop + pyramiding on {symbol} at {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "carry",
                        "funding",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "data_dependent": True,
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Carry-trend CONFLUENCE rider: ride a trend ONLY when the funding/carry sign
# agrees with the trend sign (long needs low/negative funding so the long perp
# receives carry; short needs high-positive funding). Per-symbol single-asset,
# crypto-perp only, >=30m. SHORT trend/funding windows so it fires on a ~1-month
# window. ``long_carry_funding`` / ``short_carry_funding`` are the confluence
# thresholds on the trailing-average funding rate.
_CARRY_TREND_CONFLUENCE_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "funding_window": 8,
            "long_carry_funding": 0.00005,
            "short_carry_funding": 0.00005,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 240,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "funding_window": 8,
            "long_carry_funding": 0.00005,
            "short_carry_funding": 0.00005,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "min_trend_roc": 0.0,
            "funding_window": 6,
            "long_carry_funding": 0.00005,
            "short_carry_funding": 0.00005,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "trend_lookback": 24,
            "trend_ma_window": 24,
            "min_trend_roc": 0.0,
            "funding_window": 4,
            "long_carry_funding": 0.00005,
            "short_carry_funding": 0.00005,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_carry_trend_confluence_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol carry-trend CONFLUENCE rider (single-asset, crypto-perp, feature-gated)."""
    if not ctx.perp_support_data_available:
        return
    # Crypto-only: funding is a crypto-perp field; tradfi perps are routed through
    # the dedicated equity/commodity builders, so this crypto sleeve excludes them.
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _CARRY_TREND_CONFLUENCE_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "funding_window": int(spec["funding_window"]),
                    "long_carry_funding": float(spec["long_carry_funding"]),
                    "short_carry_funding": float(spec["short_carry_funding"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"carry_trend_confluence_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['funding_window'])}"
                    ),
                    family="carry",
                    strategy_class="CarryTrendConfluenceRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol carry-trend CONFLUENCE rider: rides a trend ONLY "
                        "when the funding/carry sign AGREES with the trend (long into "
                        "an uptrend with low/negative funding so the long perp receives "
                        "carry; short into a downtrend with high-positive funding), "
                        "riding the winner with an ATR trailing stop + pyramiding on "
                        f"{symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "carry",
                        "funding",
                        "trend",
                        "confluence",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "data_dependent": True,
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Volatility-SQUEEZE breakout rider: a low-vol contraction regime (Bollinger
# Bandwidth at a multi-bar percentile low) is a required PRECONDITION; on the
# volatility expansion + prior-N-bar range break it enters in the breakout
# direction and rides. Per-symbol single-asset, OHLCV-only, >=30m. SHORT windows
# so it fires on a ~1-month window.
_VOLATILITY_SQUEEZE_BREAKOUT_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "bandwidth_window": 20,
            "bandwidth_num_std": 2.0,
            "squeeze_percentile_window": 48,
            "squeeze_percentile": 0.25,
            "expansion_mult": 1.4,
            "breakout_window": 20,
            "require_bb_in_kc": False,
            "keltner_window": 20,
            "keltner_atr_window": 10,
            "keltner_atr_mult": 1.5,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "bandwidth_window": 20,
            "bandwidth_num_std": 2.0,
            "squeeze_percentile_window": 60,
            "squeeze_percentile": 0.25,
            "expansion_mult": 1.5,
            "breakout_window": 24,
            "require_bb_in_kc": False,
            "keltner_window": 20,
            "keltner_atr_window": 10,
            "keltner_atr_mult": 1.5,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "bandwidth_window": 18,
            "bandwidth_num_std": 2.0,
            "squeeze_percentile_window": 48,
            "squeeze_percentile": 0.25,
            "expansion_mult": 1.5,
            "breakout_window": 18,
            "require_bb_in_kc": False,
            "keltner_window": 18,
            "keltner_atr_window": 10,
            "keltner_atr_mult": 1.5,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "bandwidth_window": 14,
            "bandwidth_num_std": 2.0,
            "squeeze_percentile_window": 30,
            "squeeze_percentile": 0.25,
            "expansion_mult": 1.5,
            "breakout_window": 14,
            "require_bb_in_kc": False,
            "keltner_window": 14,
            "keltner_atr_window": 10,
            "keltner_atr_mult": 1.5,
            "trail_atr_mult": 4.5,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_volatility_squeeze_breakout_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol volatility-SQUEEZE breakout rider (single-asset, OHLCV-only)."""
    # Crypto-only: tradfi equity/ETF perps are routed through the dedicated
    # equity/commodity breakout builders, so this crypto sleeve excludes them.
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _VOLATILITY_SQUEEZE_BREAKOUT_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "bandwidth_window": int(spec["bandwidth_window"]),
                    "bandwidth_num_std": float(spec["bandwidth_num_std"]),
                    "squeeze_percentile_window": int(spec["squeeze_percentile_window"]),
                    "squeeze_percentile": float(spec["squeeze_percentile"]),
                    "expansion_mult": float(spec["expansion_mult"]),
                    "breakout_window": int(spec["breakout_window"]),
                    "require_bb_in_kc": bool(spec["require_bb_in_kc"]),
                    "keltner_window": int(spec["keltner_window"]),
                    "keltner_atr_window": int(spec["keltner_atr_window"]),
                    "keltner_atr_mult": float(spec["keltner_atr_mult"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"volatility_squeeze_breakout_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['breakout_window'])}"
                    ),
                    family="breakout",
                    strategy_class="VolatilitySqueezeBreakoutRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol volatility-SQUEEZE breakout rider: a low-vol "
                        "Bollinger-Bandwidth contraction regime (percentile-low) is a "
                        "required PRECONDITION; on the volatility expansion + prior-bar "
                        "range break it enters in the breakout direction and rides with "
                        f"an ATR trailing stop + pyramiding on {symbol} at {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "breakout",
                        "volatility_squeeze",
                        "contraction_expansion",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Session opening-range-breakout rider: anchor a SESSION by the UTC calendar day,
# accumulate the opening-range high/low over the first ``opening_range_bars``
# decision bars (no trading), then arm a one-shot breakout (LONG above the range
# high + ATR buffer; SHORT below the range low) and ride. Per-symbol single-asset,
# OHLCV-only, >=30m. The opening range RESETS each UTC day.
_OPENING_RANGE_BREAKOUT_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "opening_range_bars": 4,
            "session_start_minute_utc": 0,
            "buffer_atr_mult": 0.10,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 48,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "opening_range_bars": 3,
            "session_start_minute_utc": 0,
            "buffer_atr_mult": 0.10,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 24,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "opening_range_bars": 1,
            "session_start_minute_utc": 0,
            "buffer_atr_mult": 0.15,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 18,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_opening_range_breakout_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol session opening-range-breakout rider (single-asset, OHLCV-only)."""
    # Crypto-only: tradfi equity/ETF perps are routed through the dedicated
    # equity/commodity breakout builders, so this crypto sleeve excludes them.
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    # 1d is excluded: a daily-bar session would be a single bar per UTC day, so an
    # intraday opening-range anchor is meaningless; ORB lives at 30m/1h/4h.
    for timeframe in ctx._present("30m", "1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _OPENING_RANGE_BREAKOUT_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "opening_range_bars": int(spec["opening_range_bars"]),
                    "session_start_minute_utc": int(spec["session_start_minute_utc"]),
                    "buffer_atr_mult": float(spec["buffer_atr_mult"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"opening_range_breakout_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['opening_range_bars'])}"
                    ),
                    family="breakout",
                    strategy_class="OpeningRangeBreakoutRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol SESSION opening-range-breakout rider: anchors a "
                        "session by the UTC calendar day, accumulates the opening-range "
                        "high/low over the first k decision bars (no trading), then arms "
                        "a one-shot breakout (LONG above the range high + ATR buffer; "
                        "SHORT below the range low) and rides with an ATR trailing stop + "
                        f"pyramiding on {symbol} at {timeframe} ({spec['variant']}). The "
                        "opening range resets each UTC day."
                    ),
                    tags=(
                        "breakout",
                        "opening_range",
                        "session_anchored",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Open-interest trend-confirmation rider: ride a trend ONLY when RISING open
# interest confirms it (fresh money committing -> healthy/persistent trend);
# suppress entries when OI is falling (unwind / short-covering -> low conviction).
# Per-symbol single-asset, crypto-perp only, >=30m. SHORT trend/OI windows so it
# fires on a ~1-month window. ``oi_rise_threshold`` is the min fractional OI rise
# over ``oi_lookback`` required to confirm.
_OPEN_INTEREST_TREND_CONFIRMATION_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "oi_lookback": 8,
            "oi_rise_threshold": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 240,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "oi_lookback": 8,
            "oi_rise_threshold": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "min_trend_roc": 0.0,
            "oi_lookback": 6,
            "oi_rise_threshold": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "trend_lookback": 24,
            "trend_ma_window": 24,
            "min_trend_roc": 0.0,
            "oi_lookback": 4,
            "oi_rise_threshold": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_open_interest_trend_confirmation_rider_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """Per-symbol open-interest trend-confirmation rider (single-asset, perp-gated)."""
    if not ctx.perp_support_data_available:
        return
    # Crypto-only: open-interest is a crypto-perp field; tradfi perps are routed
    # through the dedicated equity/commodity builders, so this sleeve excludes them.
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _OPEN_INTEREST_TREND_CONFIRMATION_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "oi_lookback": int(spec["oi_lookback"]),
                    "oi_rise_threshold": float(spec["oi_rise_threshold"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"open_interest_trend_confirmation_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{int(spec['oi_lookback'])}"
                    ),
                    family="trend",
                    strategy_class="OpenInterestTrendConfirmationRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol open-interest trend-confirmation rider: rides a "
                        "trend ONLY when RISING open interest confirms it (fresh money "
                        "committing -> healthy, persistent trend: LONG into an uptrend "
                        "with rising OI, SHORT into a downtrend with rising OI), and "
                        "SUPPRESSES entries when OI is falling (unwind / short-covering "
                        "-> low conviction), riding the winner with an ATR trailing stop "
                        f"+ pyramiding on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "open_interest",
                        "confirmation",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "data_dependent": True,
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Intraday time-of-day seasonal-momentum rider: ride a trend ONLY in a UTC
# time-of-day slot whose decay-weighted drift is statistically significant AND
# aligned with the trend. Per-symbol single-asset, crypto-only, >=30m. 1d is
# excluded (a daily bar has no sub-day slot structure).
_INTRADAY_SEASONAL_MOMENTUM_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "hourly_core_ls",
            "slot_minutes": 60,
            "seasonal_decay": 0.05,
            "min_slot_observations": 12,
            "slot_t_threshold": 1.5,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 48,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "hourly_swing_ls",
            "slot_minutes": 60,
            "seasonal_decay": 0.04,
            "min_slot_observations": 12,
            "slot_t_threshold": 1.5,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 24,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "quartile_swing_ls",
            "slot_minutes": 240,
            "seasonal_decay": 0.04,
            "min_slot_observations": 10,
            "slot_t_threshold": 1.5,
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 18,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_intraday_seasonal_momentum_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol intraday time-of-day seasonal-momentum rider (single-asset)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    # 1d excluded: a daily bar maps to a single intraday slot, so the time-of-day
    # statistic degenerates; the sleeve lives at 30m/1h/4h.
    for timeframe in ctx._present("30m", "1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _INTRADAY_SEASONAL_MOMENTUM_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "slot_minutes": int(spec["slot_minutes"]),
                    "seasonal_decay": float(spec["seasonal_decay"]),
                    "min_slot_observations": int(spec["min_slot_observations"]),
                    "slot_t_threshold": float(spec["slot_t_threshold"]),
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"intraday_seasonal_momentum_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="momentum",
                    strategy_class="IntradaySeasonalMomentumRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol intraday seasonal-momentum rider: rides the trend "
                        "ONLY in a UTC time-of-day slot whose decay-weighted drift is "
                        "statistically significant and aligned with the trend; an "
                        "unfavorable/insufficient-history slot suppresses entry. ATR "
                        f"trailing stop + pyramiding on {symbol} at {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "momentum",
                        "intraday_seasonality",
                        "time_of_day",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Overnight-session return-tilt rider: structurally tilt LONG during the UTC
# session whose decay-weighted mean return is significantly positive (SHORT when
# negative), suppress otherwise. NOT trend-conditioned. Per-symbol single-asset,
# crypto-only, >=30m. 1d excluded (a daily bar cannot resolve overnight vs active).
_OVERNIGHT_SESSION_RETURN_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "asia_window_ls",
            "overnight_start_hour_utc": 0,
            "overnight_end_hour_utc": 8,
            "session_decay": 0.02,
            "min_session_observations": 12,
            "session_t_threshold": 1.5,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 24,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "asia_window_swing_ls",
            "overnight_start_hour_utc": 0,
            "overnight_end_hour_utc": 8,
            "session_decay": 0.02,
            "min_session_observations": 12,
            "session_t_threshold": 1.5,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 12,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "asia_window_macro_ls",
            "overnight_start_hour_utc": 0,
            "overnight_end_hour_utc": 8,
            "session_decay": 0.02,
            "min_session_observations": 10,
            "session_t_threshold": 1.5,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 1,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 6,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_overnight_session_return_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol overnight-session return-tilt rider (single-asset, OHLCV-only)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    # 1d excluded: a daily bar cannot resolve the overnight vs active session.
    for timeframe in ctx._present("30m", "1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _OVERNIGHT_SESSION_RETURN_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "overnight_start_hour_utc": int(spec["overnight_start_hour_utc"]),
                    "overnight_end_hour_utc": int(spec["overnight_end_hour_utc"]),
                    "session_decay": float(spec["session_decay"]),
                    "min_session_observations": int(spec["min_session_observations"]),
                    "session_t_threshold": float(spec["session_t_threshold"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"overnight_session_return_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="seasonality",
                    strategy_class="OvernightSessionReturnRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol overnight-session return-tilt rider: structurally "
                        "tilts LONG during the UTC session whose decay-weighted mean "
                        "return is significantly positive (SHORT when negative), "
                        "suppresses otherwise. NOT trend-conditioned; ATR trailing stop "
                        f"manages exits on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "seasonality",
                        "overnight_session",
                        "time_of_day",
                        "return_rider",
                        "trailing_stop",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Kalman state-space trend rider: ride when a local-linear-trend Kalman SLOPE on
# the log close is significant. Per-symbol single-asset, crypto-only, >=30m.
_KALMAN_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "responsive_ls",
            "obs_noise": 0.0001,
            "level_process_noise": 2e-5,
            "slope_process_noise": 2e-7,
            "slope_t": 2.0,
            "min_slope_frac": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 96,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "obs_noise": 0.0001,
            "level_process_noise": 1e-5,
            "slope_process_noise": 1e-7,
            "slope_t": 2.0,
            "min_slope_frac": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 72,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "obs_noise": 0.0001,
            "level_process_noise": 5e-6,
            "slope_process_noise": 5e-8,
            "slope_t": 2.2,
            "min_slope_frac": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 36,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "obs_noise": 0.0001,
            "level_process_noise": 5e-6,
            "slope_process_noise": 5e-8,
            "slope_t": 2.2,
            "min_slope_frac": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 30,
            "target_vol": 0.030,
            "max_hold_bars": 30,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_kalman_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol Kalman state-space trend rider (single-asset, OHLCV-only)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _KALMAN_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "obs_noise": float(spec["obs_noise"]),
                    "level_process_noise": float(spec["level_process_noise"]),
                    "slope_process_noise": float(spec["slope_process_noise"]),
                    "slope_t": float(spec["slope_t"]),
                    "min_slope_frac": float(spec["min_slope_frac"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"kalman_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="trend",
                    strategy_class="KalmanTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol Kalman state-space trend rider: a local-linear-trend "
                        "Kalman filter on the log close yields a low-lag, uncertainty-aware "
                        "slope; a ride is armed when the filtered slope is statistically "
                        "significant (slope/sqrt(var) >= slope_t). ATR trailing stop + "
                        f"pyramiding on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "kalman",
                        "state_space",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Realized-semivariance trend rider: ride a trend confirmed by an upside/downside
# realized-semivariance asymmetry (Patton-Sheppard good/bad volatility). Per-symbol
# single-asset, crypto-only, >=30m.
_REALIZED_SEMIVARIANCE_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "core_ls",
            "semivar_window": 48,
            "semivar_threshold": 0.20,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 96,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "swing_ls",
            "semivar_window": 48,
            "semivar_threshold": 0.20,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 72,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "macro_ls",
            "semivar_window": 36,
            "semivar_threshold": 0.22,
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 36,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_realized_semivariance_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol realized-semivariance-asymmetry trend rider (single-asset)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _REALIZED_SEMIVARIANCE_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "semivar_window": int(spec["semivar_window"]),
                    "semivar_threshold": float(spec["semivar_threshold"]),
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"realized_semivariance_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="trend",
                    strategy_class="RealizedSemivarianceTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol realized-semivariance trend rider: rides the trend only "
                        "when the upside/downside realized-semivariance asymmetry "
                        "SJ=(RS+ - RS-)/(RS+ + RS-) clears a threshold AND agrees with the "
                        "trend (Patton-Sheppard good/bad volatility). ATR trailing stop + "
                        f"pyramiding on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "realized_semivariance",
                        "good_bad_volatility",
                        "signed_jump",
                        "return_rider",
                        "trailing_stop",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Permutation-entropy trend rider: ride only in a low-entropy (predictable)
# regime. Per-symbol single-asset, crypto-only, >=30m.
_PERMUTATION_ENTROPY_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "core_ls",
            "pe_dim": 3,
            "pe_window": 64,
            "pe_threshold": 0.85,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 96,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "swing_ls",
            "pe_dim": 3,
            "pe_window": 64,
            "pe_threshold": 0.85,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 72,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "macro_ls",
            "pe_dim": 3,
            "pe_window": 48,
            "pe_threshold": 0.85,
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 36,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "daily_ls",
            "pe_dim": 3,
            "pe_window": 48,
            "pe_threshold": 0.85,
            "trend_lookback": 30,
            "trend_ma_window": 30,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 30,
            "target_vol": 0.030,
            "max_hold_bars": 30,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_permutation_entropy_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol permutation-entropy-gated trend rider (single-asset, OHLCV-only)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _PERMUTATION_ENTROPY_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "pe_dim": int(spec["pe_dim"]),
                    "pe_window": int(spec["pe_window"]),
                    "pe_threshold": float(spec["pe_threshold"]),
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"permutation_entropy_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="trend",
                    strategy_class="PermutationEntropyTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol permutation-entropy-gated trend rider: rides the trend "
                        "only when the normalized Bandt-Pompe permutation entropy of recent "
                        "closes is below pe_threshold (a predictable/structured regime) AND "
                        f"the trend is confirmed. ATR trailing stop + pyramiding on {symbol} "
                        f"at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "permutation_entropy",
                        "predictability_regime",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Amihud illiquidity-premium momentum rider: ride a confirmed trend only when the
# Amihud illiquidity is elevated vs its rolling median. Per-symbol single-asset,
# crypto-only, >=30m.
_AMIHUD_ILLIQUIDITY_MOMENTUM_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "core_ls",
            "amihud_window": 48,
            "amihud_history_window": 64,
            "illiquidity_rel": 1.0,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 96,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "swing_ls",
            "amihud_window": 48,
            "amihud_history_window": 64,
            "illiquidity_rel": 1.0,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 72,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "macro_ls",
            "amihud_window": 36,
            "amihud_history_window": 48,
            "illiquidity_rel": 1.0,
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 36,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "daily_ls",
            "amihud_window": 30,
            "amihud_history_window": 48,
            "illiquidity_rel": 1.0,
            "trend_lookback": 30,
            "trend_ma_window": 30,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 30,
            "target_vol": 0.030,
            "max_hold_bars": 30,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_amihud_illiquidity_momentum_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol Amihud illiquidity-premium momentum rider (single-asset)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _AMIHUD_ILLIQUIDITY_MOMENTUM_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "amihud_window": int(spec["amihud_window"]),
                    "amihud_history_window": int(spec["amihud_history_window"]),
                    "illiquidity_rel": float(spec["illiquidity_rel"]),
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"amihud_illiquidity_momentum_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="momentum",
                    strategy_class="AmihudIlliquidityMomentumRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol Amihud illiquidity-premium momentum rider: rides a "
                        "confirmed trend only when the Amihud illiquidity (|return|/dollar "
                        "volume) is elevated vs its rolling median (an illiquidity-premium "
                        f"regime). ATR trailing stop + pyramiding on {symbol} at {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "momentum",
                        "amihud_illiquidity",
                        "illiquidity_premium",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# CUSUM change-point trend rider: ride the drift regime shift a two-sided CUSUM
# control chart declares. Per-symbol single-asset, crypto-only, >=30m.
_CUSUM_CHANGE_POINT_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "core_ls",
            "cusum_vol_window": 48,
            "cusum_k": 0.5,
            "cusum_h": 5.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 96,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "swing_ls",
            "cusum_vol_window": 48,
            "cusum_k": 0.5,
            "cusum_h": 5.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 72,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "macro_ls",
            "cusum_vol_window": 36,
            "cusum_k": 0.5,
            "cusum_h": 5.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 36,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "daily_ls",
            "cusum_vol_window": 30,
            "cusum_k": 0.5,
            "cusum_h": 4.5,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 30,
            "target_vol": 0.030,
            "max_hold_bars": 30,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_cusum_change_point_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol CUSUM change-point trend rider (single-asset, OHLCV-only)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _CUSUM_CHANGE_POINT_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "cusum_vol_window": int(spec["cusum_vol_window"]),
                    "cusum_k": float(spec["cusum_k"]),
                    "cusum_h": float(spec["cusum_h"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"cusum_change_point_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="trend",
                    strategy_class="CusumChangePointTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol CUSUM change-point trend rider: a two-sided CUSUM "
                        "control chart on vol-standardized returns declares an up/down drift "
                        "regime shift; the detected direction opens a ride with an ATR "
                        f"trailing stop + pyramiding on {symbol} at {timeframe} "
                        f"({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "cusum",
                        "change_point",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Variance-ratio trend rider: ride a confirmed trend only when the Lo-MacKinlay
# variance ratio signals persistence (VR >= 1 + threshold). Per-symbol single-asset,
# crypto-only, >=30m.
_VARIANCE_RATIO_TREND_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "core_ls",
            "vr_window": 96,
            "vr_k": 4,
            "vr_threshold": 0.10,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 96,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "swing_ls",
            "vr_window": 96,
            "vr_k": 4,
            "vr_threshold": 0.10,
            "trend_lookback": 48,
            "trend_ma_window": 48,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 72,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "macro_ls",
            "vr_window": 72,
            "vr_k": 4,
            "vr_threshold": 0.10,
            "trend_lookback": 36,
            "trend_ma_window": 36,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 36,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "daily_ls",
            "vr_window": 60,
            "vr_k": 4,
            "vr_threshold": 0.10,
            "trend_lookback": 30,
            "trend_ma_window": 30,
            "min_trend_roc": 0.0,
            "trail_atr_mult": 4.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 30,
            "target_vol": 0.030,
            "max_hold_bars": 30,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_variance_ratio_trend_rider_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol variance-ratio-gated trend rider (single-asset, OHLCV-only)."""
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _VARIANCE_RATIO_TREND_RIDER_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "vr_window": int(spec["vr_window"]),
                    "vr_k": int(spec["vr_k"]),
                    "vr_threshold": float(spec["vr_threshold"]),
                    "trend_lookback": int(spec["trend_lookback"]),
                    "trend_ma_window": int(spec["trend_ma_window"]),
                    "min_trend_roc": float(spec["min_trend_roc"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"variance_ratio_trend_rider_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="trend",
                    strategy_class="VarianceRatioTrendRiderStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol variance-ratio trend rider: rides a confirmed trend only "
                        "when the Lo-MacKinlay variance ratio VR(k) >= 1 + vr_threshold (the "
                        "random walk is rejected toward persistence). ATR trailing stop + "
                        f"pyramiding on {symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "trend",
                        "variance_ratio",
                        "random_walk_rejection",
                        "return_rider",
                        "trailing_stop",
                        "pyramiding",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


_METALS_RELATIVE_VALUE_BASKET_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "rv_core",
            "zscore_window": 90,
            "entry_z": 2.0,
            "exit_z": 0.40,
            "max_hold_bars": 120,
        },
        {
            "variant": "rv_patient",
            "zscore_window": 120,
            "entry_z": 2.3,
            "exit_z": 0.30,
            "max_hold_bars": 180,
        },
    ),
    "1d": (
        {
            "variant": "rv_swing",
            "zscore_window": 60,
            "entry_z": 1.8,
            "exit_z": 0.40,
            "max_hold_bars": 30,
        },
    ),
}


def _build_metals_relative_value_basket_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S7 — market-neutral precious-metal relative-value basket (live metals)."""
    metals = tuple(symbol for symbol in ctx.normalized_symbols if symbol in _METALS)
    if len(metals) < 3:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _METALS_RELATIVE_VALUE_BASKET_SLICE.get(timeframe, ()):
            params = {
                "zscore_window": int(spec["zscore_window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"metals_relative_value_basket_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['zscore_window'])}_{float(spec['entry_z']):.1f}"
                ),
                family="cross_sectional",
                strategy_class="MetalsRelativeValueBasketStrategy",
                timeframe=timeframe,
                symbols=metals,
                params=params,
                notes=(
                    "Market-neutral relative-value basket across the precious-metal "
                    "perps (XAU/XAG/XPT/XPD) that fades z-scored ratio dislocations "
                    f"on an event-banded cadence for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "metals",
                    "relative_value",
                    "market_neutral",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "metals",
                    "decision_cadence_seconds": 14400,
                },
            )


_LIQUIDATION_CASCADE_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "cascade_core",
            "btc_return_trigger": -0.03,
            "volume_mult": 2.0,
            "oi_drop_trigger": 0.05,
            "hold_bars": 12,
            "basket_size": 4,
            "beta_window": 72,
        },
        {
            "variant": "cascade_deep",
            "btc_return_trigger": -0.05,
            "volume_mult": 2.5,
            "oi_drop_trigger": 0.07,
            "hold_bars": 18,
            "basket_size": 4,
            "beta_window": 96,
        },
    ),
    "4h": (
        {
            "variant": "cascade_swing",
            "btc_return_trigger": -0.04,
            "volume_mult": 2.0,
            "oi_drop_trigger": 0.06,
            "hold_bars": 6,
            "basket_size": 4,
            "beta_window": 48,
        },
    ),
}


def _build_liquidation_cascade_reversion_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S10 — crisis-conditional liquidation-cascade reversion (live crypto)."""
    if not ctx.perp_support_data_available:
        return
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _LIQUIDATION_CASCADE_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "btc_return_trigger": float(spec["btc_return_trigger"]),
                "volume_mult": float(spec["volume_mult"]),
                "oi_drop_trigger": float(spec["oi_drop_trigger"]),
                "hold_bars": int(spec["hold_bars"]),
                "basket_size": int(spec["basket_size"]),
                "beta_window": int(spec["beta_window"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"liquidation_cascade_reversion_{tf_tag}_{spec['variant']}_"
                    f"{abs(float(spec['btc_return_trigger'])):.2f}_{int(spec['hold_bars'])}"
                ),
                family="cross_sectional",
                strategy_class="LiquidationCascadeReversionStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Crisis-conditional sleeve that buys beaten-down alts into a "
                    "BTC-led deleveraging cascade (sharp drop + volume spike + OI "
                    "contraction), BTC-beta-hedged, with a fixed hold and no-re-"
                    f"layer lock for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "liquidation",
                    "crisis",
                    "reversion",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "data_dependent": True,
                    "decision_cadence_seconds": 3600,
                },
            )


_ORDERBOOK_IMBALANCE_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "fade_core",
            "imbalance_z_window": 120,
            "entry_z": 2.0,
            "exit_z": 0.40,
            "spread_z_window": 120,
            "max_spread_z": 3.0,
            "spread_quantile_window": 240,
            "max_spread_quantile": 0.90,
            "hold_bars": 12,
            "stop_pct": 0.010,
            "cooldown_bars": 6,
            "allow_short": True,
        },
        {
            "variant": "fade_tight",
            "imbalance_z_window": 180,
            "entry_z": 2.4,
            "exit_z": 0.30,
            "spread_z_window": 180,
            "max_spread_z": 2.5,
            "spread_quantile_window": 360,
            "max_spread_quantile": 0.85,
            "hold_bars": 8,
            "stop_pct": 0.008,
            "cooldown_bars": 8,
            "allow_short": True,
        },
    ),
    "15m": (
        {
            "variant": "fade_swing",
            "imbalance_z_window": 96,
            "entry_z": 2.0,
            "exit_z": 0.40,
            "spread_z_window": 96,
            "max_spread_z": 3.0,
            "spread_quantile_window": 192,
            "max_spread_quantile": 0.90,
            "hold_bars": 6,
            "stop_pct": 0.012,
            "cooldown_bars": 4,
            "allow_short": True,
        },
    ),
}


def _build_orderbook_imbalance_reversion_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S12 — microstructure order-book imbalance reversion (live crypto)."""
    if not ctx.perp_support_data_available:
        return
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("5m", "15m"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _ORDERBOOK_IMBALANCE_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "imbalance_z_window": int(spec["imbalance_z_window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "spread_z_window": int(spec["spread_z_window"]),
                "max_spread_z": float(spec["max_spread_z"]),
                "spread_quantile_window": int(spec["spread_quantile_window"]),
                "max_spread_quantile": float(spec["max_spread_quantile"]),
                "hold_bars": int(spec["hold_bars"]),
                "stop_pct": float(spec["stop_pct"]),
                "cooldown_bars": int(spec["cooldown_bars"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"orderbook_imbalance_reversion_{tf_tag}_{spec['variant']}_"
                    f"{float(spec['entry_z']):.1f}_{int(spec['hold_bars'])}"
                ),
                family="cross_sectional",
                strategy_class="OrderBookImbalanceReversionStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Microstructure mean-reversion sleeve that fades extreme "
                    "order-book depth-imbalance/spread z-scores with a short hold, "
                    "tight stop, and per-symbol cooldown to bound turnover for "
                    f"{timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "microstructure",
                    "order_book",
                    "reversion",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "data_dependent": True,
                    "decision_cadence_seconds": 300,
                },
            )


# --- Selection-aware cross-sectional sleeves (live crypto) -----------------
# These compose the multi-factor target-pool selector with a directional signal:
# each rebalance screens the universe into a tradeable pool, then trades only
# inside it.  They are multi-symbol (>=3) so they MUST be wired
# family="cross_sectional" with the mandatory admission trio.

_SELECTION_GATED_MOMENTUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "screened_lo",
            "momentum_lookback_bars": 48,
            "vol_window": 24,
            "rebalance_bars": 24,
            "history_window": 120,
            "pool_size": 12,
            "top_fraction": 0.34,
            "selection_tilt": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.060,
            "max_hold_bars": 240,
        },
        {
            "variant": "screened_ls",
            "momentum_lookback_bars": 72,
            "vol_window": 36,
            "rebalance_bars": 24,
            "history_window": 168,
            "pool_size": 10,
            "top_fraction": 0.30,
            "selection_tilt": 0.30,
            "allow_short": True,
            "stop_loss_pct": 0.050,
            "max_hold_bars": 360,
        },
    ),
    "4h": (
        {
            "variant": "screened_swing",
            "momentum_lookback_bars": 30,
            "vol_window": 14,
            "rebalance_bars": 12,
            "history_window": 96,
            "pool_size": 10,
            "top_fraction": 0.34,
            "selection_tilt": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.070,
            "max_hold_bars": 120,
        },
    ),
}


def _build_selection_gated_momentum_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """Selection-gated cross-sectional momentum on a screened crypto pool."""
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        cadence = 3600 if timeframe == "1h" else 14400
        for spec in _SELECTION_GATED_MOMENTUM_SLICE.get(timeframe, ()):
            params = {
                "momentum_lookback_bars": int(spec["momentum_lookback_bars"]),
                "vol_window": int(spec["vol_window"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "history_window": int(spec["history_window"]),
                "pool_size": int(spec["pool_size"]),
                "top_fraction": float(spec["top_fraction"]),
                "selection_tilt": float(spec["selection_tilt"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"selection_gated_momentum_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['momentum_lookback_bars'])}_{int(spec['pool_size'])}"
                ),
                family="cross_sectional",
                strategy_class="SelectionGatedMomentumStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Selection-aware sleeve that screens the universe into the "
                    "multi-factor target pool (price-position/volume/volatility/"
                    "vwap) each rebalance, then runs vol-scaled cross-sectional "
                    f"momentum only inside that pool for {timeframe} "
                    f"({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "selection_aware",
                    "universe_selection",
                    "factor",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "allow_short": bool(spec["allow_short"]),
                    "decision_cadence_seconds": cadence,
                },
            )


_SELECTION_GATED_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "screened_fade_lo",
            "reversion_lookback_bars": 6,
            "vol_window": 24,
            "rebalance_bars": 12,
            "history_window": 120,
            "pool_size": 12,
            "top_fraction": 0.34,
            "selection_tilt": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.040,
            "max_hold_bars": 96,
        },
        {
            "variant": "screened_fade_ls",
            "reversion_lookback_bars": 8,
            "vol_window": 36,
            "rebalance_bars": 12,
            "history_window": 168,
            "pool_size": 10,
            "top_fraction": 0.30,
            "selection_tilt": 0.30,
            "allow_short": True,
            "stop_loss_pct": 0.035,
            "max_hold_bars": 144,
        },
    ),
    "4h": (
        {
            "variant": "screened_fade_swing",
            "reversion_lookback_bars": 4,
            "vol_window": 14,
            "rebalance_bars": 6,
            "history_window": 96,
            "pool_size": 10,
            "top_fraction": 0.34,
            "selection_tilt": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.050,
            "max_hold_bars": 48,
        },
    ),
}


def _build_selection_gated_reversion_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """Selection-gated short-horizon reversion on a screened crypto pool."""
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        cadence = 3600 if timeframe == "1h" else 14400
        for spec in _SELECTION_GATED_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "reversion_lookback_bars": int(spec["reversion_lookback_bars"]),
                "vol_window": int(spec["vol_window"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "history_window": int(spec["history_window"]),
                "pool_size": int(spec["pool_size"]),
                "top_fraction": float(spec["top_fraction"]),
                "selection_tilt": float(spec["selection_tilt"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"selection_gated_reversion_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['reversion_lookback_bars'])}_{int(spec['pool_size'])}"
                ),
                family="cross_sectional",
                strategy_class="SelectionGatedReversionStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Selection-aware sleeve that screens the universe into the "
                    "multi-factor target pool each rebalance, then fades the most "
                    "recent return only on screened mid-range liquid names for "
                    f"{timeframe} ({spec['variant']}); decorrelated from the "
                    "selection-gated momentum sleeve."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "selection_aware",
                    "universe_selection",
                    "reversion",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "allow_short": bool(spec["allow_short"]),
                    "decision_cadence_seconds": cadence,
                },
            )


# --- DORMANT equity / index / calendar tranche -----------------------------
# These self-skip until the equity/ETF perps materialize on the test PC.

_CROSS_SECTIONAL_EQUITY_MOMENTUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "xs_12_1_lo",
            "lookback_bars": 252,
            "skip_bars": 21,
            "vol_window": 63,
            "regime_sma_bars": 200,
            "rebalance_bars": 21,
            "quintile_pct": 0.20,
            "allow_short": False,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 252,
        },
        {
            "variant": "xs_12_1_ls",
            "lookback_bars": 252,
            "skip_bars": 21,
            "vol_window": 63,
            "regime_sma_bars": 200,
            "rebalance_bars": 21,
            "quintile_pct": 0.20,
            "allow_short": True,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 252,
        },
    ),
}


def _build_cross_sectional_equity_momentum_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S1 (DORMANT) — vol-scaled 12-1 cross-sectional equity momentum."""
    filtered = _intersect_universe(_EQUITY_FACTOR_UNIVERSE, ctx.normalized_symbols)
    if len(filtered) < 4:
        return
    for timeframe in ctx._present("1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _CROSS_SECTIONAL_EQUITY_MOMENTUM_SLICE.get(timeframe, ()):
            params = {
                "lookback_bars": int(spec["lookback_bars"]),
                "skip_bars": int(spec["skip_bars"]),
                "vol_window": int(spec["vol_window"]),
                "regime_sma_bars": int(spec["regime_sma_bars"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "quintile_pct": float(spec["quintile_pct"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"cross_sectional_equity_momentum_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['lookback_bars'])}_{int(spec['skip_bars'])}"
                ),
                family="cross_sectional",
                strategy_class="CrossSectionalEquityMomentumStrategy",
                timeframe=timeframe,
                symbols=filtered,
                params=params,
                notes=(
                    "DORMANT equity tranche: vol-scaled 12-1 cross-sectional "
                    "momentum on equity perps with a basket-SMA short gate for "
                    f"{timeframe} ({spec['variant']}); self-skips until equity "
                    "perps materialize."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "equity",
                    "factor",
                    "dormant",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "equity",
                    "dormant_tranche": True,
                    "decision_cadence_seconds": 86400,
                },
            )


_RESIDUAL_EQUITY_MOMENTUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "imom_ls",
            "lookback_bars": 252,
            "skip_bars": 21,
            "beta_window": 252,
            "rebalance_bars": 21,
            "quintile_pct": 0.20,
            "allow_short": True,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 252,
        },
    ),
}


def _build_residual_equity_momentum_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S2 (DORMANT) — residual (idiosyncratic) momentum vs benchmark."""
    filtered = _intersect_universe(_EQUITY_FACTOR_UNIVERSE, ctx.normalized_symbols)
    if len(filtered) < 4:
        return
    for timeframe in ctx._present("1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _RESIDUAL_EQUITY_MOMENTUM_SLICE.get(timeframe, ()):
            params = {
                "lookback_bars": int(spec["lookback_bars"]),
                "skip_bars": int(spec["skip_bars"]),
                "beta_window": int(spec["beta_window"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "quintile_pct": float(spec["quintile_pct"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"residual_equity_momentum_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['lookback_bars'])}_{int(spec['beta_window'])}"
                ),
                family="cross_sectional",
                strategy_class="ResidualEquityMomentumStrategy",
                timeframe=timeframe,
                symbols=filtered,
                params=params,
                notes=(
                    "DORMANT equity tranche: residual (idiosyncratic) momentum "
                    "that strips benchmark beta before ranking, inverse-residual-"
                    f"vol sized, for {timeframe} ({spec['variant']}); self-skips "
                    "until equity perps materialize."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "equity",
                    "residual",
                    "factor",
                    "dormant",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "equity",
                    "dormant_tranche": True,
                    "decision_cadence_seconds": 86400,
                },
            )


_BETTING_AGAINST_BETA_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "bab_ls",
            "beta_window": 252,
            "rebalance_bars": 21,
            "quintile_pct": 0.20,
            "beta_z_window": 63,
            "allow_short": True,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 252,
        },
    ),
}


def _build_betting_against_beta_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S3 (DORMANT) — betting-against-beta (long low-beta / short high-beta)."""
    filtered = _intersect_universe(_EQUITY_FACTOR_UNIVERSE, ctx.normalized_symbols)
    if len(filtered) < 4:
        return
    for timeframe in ctx._present("1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _BETTING_AGAINST_BETA_SLICE.get(timeframe, ()):
            params = {
                "beta_window": int(spec["beta_window"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "quintile_pct": float(spec["quintile_pct"]),
                "beta_z_window": int(spec["beta_z_window"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"betting_against_beta_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['beta_window'])}_{float(spec['quintile_pct']):.2f}"
                ),
                family="cross_sectional",
                strategy_class="BettingAgainstBetaStrategy",
                timeframe=timeframe,
                symbols=filtered,
                params=params,
                notes=(
                    "DORMANT equity tranche: betting-against-beta sleeve that goes "
                    "long the low-beta quintile and short the high-beta quintile, "
                    f"beta-neutralized, for {timeframe} ({spec['variant']}); self-"
                    "skips until equity perps materialize."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "equity",
                    "low_beta",
                    "factor",
                    "dormant",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "equity",
                    "dormant_tranche": True,
                    "decision_cadence_seconds": 86400,
                },
            )


_SEMIS_LEADLAG_ROTATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "soxl_lead",
            "leader_lookback_bars": 6,
            "leader_z_window": 96,
            "follower_lookback_bars": 6,
            "leader_sigma": 1.0,
            "spillover_window": 96,
            "rebalance_bars": 2,
            "max_longs": 3,
            "allow_short": True,
            "stop_loss_pct": 0.08,
            "max_hold_bars": 96,
        },
    ),
    "1d": (
        {
            "variant": "soxl_lead_swing",
            "leader_lookback_bars": 3,
            "leader_z_window": 60,
            "follower_lookback_bars": 3,
            "leader_sigma": 1.0,
            "spillover_window": 60,
            "rebalance_bars": 1,
            "max_longs": 3,
            "allow_short": False,
            "stop_loss_pct": 0.08,
            "max_hold_bars": 21,
        },
    ),
}


def _build_semis_leadlag_rotation_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S8 (DORMANT) — semis lead-lag rotation (SOXL leads the chip names)."""
    filtered = _intersect_universe(_SEMIS_LEADLAG_UNIVERSE, ctx.normalized_symbols)
    # Require the leader plus at least three followers.
    leader = canonicalize_symbol_list(("SOXLUSDT",))
    has_leader = bool(leader) and leader[0] in filtered
    if not has_leader or len(filtered) < 4:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _SEMIS_LEADLAG_ROTATION_SLICE.get(timeframe, ()):
            params = {
                "leader_lookback_bars": int(spec["leader_lookback_bars"]),
                "leader_z_window": int(spec["leader_z_window"]),
                "follower_lookback_bars": int(spec["follower_lookback_bars"]),
                "leader_sigma": float(spec["leader_sigma"]),
                "spillover_window": int(spec["spillover_window"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "max_longs": int(spec["max_longs"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"semis_leadlag_rotation_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['leader_lookback_bars'])}_{int(spec['rebalance_bars'])}"
                ),
                family="cross_sectional",
                strategy_class="SemisLeadLagRotationStrategy",
                timeframe=timeframe,
                symbols=filtered,
                params=params,
                notes=(
                    "DORMANT equity tranche: semis lead-lag rotation where a SOXL "
                    "move beyond one sigma tilts the book toward under-reacted "
                    f"laggard chip names for {timeframe} ({spec['variant']}); self-"
                    "skips until semis perps materialize."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "equity",
                    "lead_lag",
                    "semis",
                    "dormant",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "equity",
                    "dormant_tranche": True,
                    "decision_cadence_seconds": 14400,
                },
            )


# Commodity -> equity intermarket lead-lag continuation universe: the energy
# leaders (WTI ``CL``, Brent ``BZ``, industrial ``COPPER``) plus the energy ETF
# follower ``XLE``. Verified members of the TradFi commodity + ETF/index sets.
_INTERMARKET_LEADLAG_UNIVERSE: tuple[str, ...] = (
    "CLUSDT",
    "BZUSDT",
    "COPPERUSDT",
    "XLEUSDT",
)
# Default leader>follower continuation pairs (oil -> energy ETF). Copper is held
# in the universe for an optional industrial-proxy extension but is only added to
# the pair spec when a copper-led equity follower also materializes.
_INTERMARKET_LEADLAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("CL/USDT", "XLE/USDT"),
    ("BZ/USDT", "XLE/USDT"),
)


_CROSS_ASSET_DIVERSIFIED_TREND_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "diversified_ls_core",
            "trend_lookback": 90,
            "vol_window": 60,
            "min_trend": 0.02,
            "max_positions": 10,
            "target_vol": 0.020,
            "rebalance_band": 0.25,
            "confirm_sma_slope": True,
            "rebalance_bars": 5,
            "allow_short": True,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 252,
        },
        {
            "variant": "diversified_lo_slow",
            "trend_lookback": 120,
            "vol_window": 90,
            "min_trend": 0.03,
            "max_positions": 8,
            "target_vol": 0.018,
            "rebalance_band": 0.30,
            "confirm_sma_slope": True,
            "rebalance_bars": 7,
            "allow_short": False,
            "stop_loss_pct": 0.14,
            "max_hold_bars": 300,
        },
    ),
    "4h": (
        {
            "variant": "diversified_ls_swing",
            "trend_lookback": 60,
            "vol_window": 48,
            "min_trend": 0.025,
            "max_positions": 8,
            "target_vol": 0.030,
            "rebalance_band": 0.25,
            "confirm_sma_slope": True,
            "rebalance_bars": 6,
            "allow_short": True,
            "stop_loss_pct": 0.10,
            "max_hold_bars": 180,
        },
    ),
}


def _build_cross_asset_diversified_trend_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """Risk-budgeted cross-asset trend book over the FULL normalized universe.

    One coordinated managed-futures book: per-symbol TSMOM trend, inverse-vol
    risk-parity weights, portfolio vol-targeting, long up-trenders / short
    down-trenders. Targets ``ctx.normalized_symbols`` (crypto + equity +
    commodity + metals) with a >=``min_symbols`` guard so it only fires once a
    genuinely diversified book is available.
    """
    normalized_symbols = ctx.normalized_symbols
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _CROSS_ASSET_DIVERSIFIED_TREND_SLICE.get(timeframe, ()):
            min_symbols = 6
            if len(normalized_symbols) < min_symbols:
                continue
            params = {
                "trend_lookback": int(spec["trend_lookback"]),
                "vol_window": int(spec["vol_window"]),
                "min_trend": float(spec["min_trend"]),
                "max_positions": int(spec["max_positions"]),
                "target_vol": float(spec["target_vol"]),
                "rebalance_band": float(spec["rebalance_band"]),
                "confirm_sma_slope": bool(spec["confirm_sma_slope"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "min_symbols": int(min_symbols),
                "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"cross_asset_diversified_trend_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['trend_lookback'])}_{int(spec['vol_window'])}"
                ),
                family="cross_sectional",
                strategy_class="CrossAssetDiversifiedTrendStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Coordinated cross-asset managed-futures trend book: per-symbol "
                    "TSMOM trend with inverse-vol risk-parity weights and portfolio "
                    f"vol-targeting over the full universe for {timeframe} "
                    f"({spec['variant']}); diversification across weakly-correlated "
                    "asset classes raises risk-adjusted return."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "trend",
                    "managed_futures",
                    "risk_parity",
                    "cross_asset",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "cross_asset",
                    "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                },
            )


_INTERMARKET_LEADLAG_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "oil_energy_ls",
            "lead_lookback": 5,
            "follow_lookback": 3,
            "z_window": 60,
            "entry_z": 1.5,
            "min_leader_move": 0.01,
            "max_follower_move": 0.01,
            "catch_up_fraction": 0.70,
            "allow_short": True,
            "stop_loss_pct": 0.04,
            "max_hold_bars": 20,
        },
    ),
    "4h": (
        {
            "variant": "oil_energy_fast",
            "lead_lookback": 6,
            "follow_lookback": 4,
            "z_window": 96,
            "entry_z": 1.5,
            "min_leader_move": 0.012,
            "max_follower_move": 0.012,
            "catch_up_fraction": 0.70,
            "allow_short": True,
            "stop_loss_pct": 0.04,
            "max_hold_bars": 36,
        },
    ),
}


def _build_intermarket_leadlag_continuation_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """Commodity -> equity intermarket lead-lag continuation (oil -> energy ETF).

    Self-skips unless at least one configured leader>follower pair has BOTH legs
    live in the intersected commodity+ETF universe.
    """
    filtered = _intersect_universe(_INTERMARKET_LEADLAG_UNIVERSE, ctx.normalized_symbols)
    available = set(filtered)
    live_pairs = [
        (leader, follower)
        for leader, follower in _INTERMARKET_LEADLAG_PAIRS
        if leader in available and follower in available
    ]
    if not live_pairs:
        return
    pair_spec = ",".join(f"{leader}>{follower}" for leader, follower in live_pairs)
    leg_symbols = tuple(dict.fromkeys(symbol for pair in live_pairs for symbol in pair))
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _INTERMARKET_LEADLAG_SLICE.get(timeframe, ()):
            params = {
                "pair_spec": pair_spec,
                "lead_lookback": int(spec["lead_lookback"]),
                "follow_lookback": int(spec["follow_lookback"]),
                "z_window": int(spec["z_window"]),
                "entry_z": float(spec["entry_z"]),
                "min_leader_move": float(spec["min_leader_move"]),
                "max_follower_move": float(spec["max_follower_move"]),
                "catch_up_fraction": float(spec["catch_up_fraction"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"intermarket_lead_lag_continuation_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['lead_lookback'])}_{int(spec['follow_lookback'])}"
                ),
                family="cross_sectional",
                strategy_class="IntermarketLeadLagContinuationStrategy",
                timeframe=timeframe,
                symbols=leg_symbols,
                params=params,
                notes=(
                    "Commodity -> equity intermarket lead-lag continuation: an oil "
                    "(CL/BZ) leader move beyond an entry z-threshold enters the energy "
                    f"ETF follower (XLE) in the same direction for {timeframe} "
                    f"({spec['variant']}); exits on follower catch-up / stop / max-hold; "
                    "self-skips until both legs materialize."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "lead_lag",
                    "intermarket",
                    "commodity",
                    "equity",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "commodity_equity",
                    "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                },
            )


# Per-symbol realized-vol TERM-STRUCTURE recovery rider (single-asset, return-max).
# RV_s/RV_l spike (panic vol backwardation) -> LONG the V-recovery, ride with the
# inherited ATR trailing stop; exit on vol normalization. Wired >=30m only with
# SHORT windows so it fires on many spikes over a ~1-month window.
_RV_TERM_STRUCTURE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_recovery",
            "short_window": 6,
            "long_window": 48,
            "shock_ratio": 1.8,
            "exit_ratio": 1.1,
            "use_micro_short": True,
            "fade_upside_short": False,
            "upside_return": 0.04,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.025,
            "max_hold_bars": 160,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_recovery",
            "short_window": 6,
            "long_window": 48,
            "shock_ratio": 1.9,
            "exit_ratio": 1.1,
            "use_micro_short": True,
            "fade_upside_short": False,
            "upside_return": 0.05,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.025,
            "max_hold_bars": 120,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_recovery",
            "short_window": 5,
            "long_window": 36,
            "shock_ratio": 2.0,
            "exit_ratio": 1.15,
            "use_micro_short": True,
            "fade_upside_short": False,
            "upside_return": 0.06,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 90,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_recovery",
            "short_window": 4,
            "long_window": 24,
            "shock_ratio": 2.0,
            "exit_ratio": 1.15,
            "use_micro_short": False,
            "fade_upside_short": False,
            "upside_return": 0.08,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 1,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 45,
            "allow_short": False,
            "add_alloc_fraction": 0.5,
        },
    ),
}


def _build_realized_vol_term_structure_candidates(ctx: _CandidateBuildContext) -> None:
    """Per-symbol realized-vol-term-structure recovery rider (single-asset, return-max).

    Crypto-only: tradfi equity/ETF perps are routed through the dedicated equity
    builders, so this crypto sleeve excludes them (never ``ctx.crypto_symbols``).
    """
    crypto_symbols = ctx.crypto_only_symbols
    if not crypto_symbols:
        return
    for timeframe in ctx._present("30m", "1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _RV_TERM_STRUCTURE_SLICE.get(timeframe, ()):
            for symbol in crypto_symbols:
                params = {
                    "short_window": int(spec["short_window"]),
                    "long_window": int(spec["long_window"]),
                    "shock_ratio": float(spec["shock_ratio"]),
                    "exit_ratio": float(spec["exit_ratio"]),
                    "use_micro_short": bool(spec["use_micro_short"]),
                    "fade_upside_short": bool(spec["fade_upside_short"]),
                    "upside_return": float(spec["upside_return"]),
                    "trail_atr_mult": float(spec["trail_atr_mult"]),
                    "atr_period": int(spec["atr_period"]),
                    "max_adds": int(spec["max_adds"]),
                    "add_step_atr": float(spec["add_step_atr"]),
                    "vol_window": int(spec["vol_window"]),
                    "target_vol": float(spec["target_vol"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "allow_short": bool(spec["allow_short"]),
                    "add_alloc_fraction": float(spec["add_alloc_fraction"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"realized_vol_term_structure_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}_"
                        f"{float(spec['shock_ratio']):.1f}"
                    ),
                    family="mean_reversion",
                    strategy_class="RealizedVolTermStructureStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "Per-symbol realized-vol-term-structure recovery rider: a "
                        "short/long realized-vol ratio spike (panic vol "
                        "backwardation) enters LONG the V-recovery and rides it with "
                        "an ATR trailing stop, exiting on vol normalization, on "
                        f"{symbol} at {timeframe} ({spec['variant']})."
                    ),
                    tags=(
                        "mean_reversion",
                        "volatility",
                        "term_structure",
                        "return_rider",
                        "trailing_stop",
                        "single_asset",
                        "crypto",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "allow_short": bool(spec["allow_short"]),
                        "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                    },
                )


# Top-down BREADTH-gated total-exposure trend timer over the crypto basket. Each
# decision bar scales TOTAL net-long gross by cross-sectional breadth (fraction of
# the basket above its own trend); strong breadth -> long the up-trenders with a
# breadth-scaled gross, collapsing breadth -> flat (risk-off). Multi-symbol basket
# (admission-safe cross_sectional). >=30m only (1h/4h/1d).
_BREADTH_REGIME_TREND_TIMER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "breadth_timer_core",
            "trend_window": 50,
            "risk_on_breadth": 0.60,
            "risk_off_breadth": 0.40,
            "require_positive_return": True,
            "max_positions": 8,
            "min_trend": 0.0,
            "max_gross": 0.80,
            "rebalance_bars": 3,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 240,
        },
    ),
    "4h": (
        {
            "variant": "breadth_timer_swing",
            "trend_window": 40,
            "risk_on_breadth": 0.58,
            "risk_off_breadth": 0.38,
            "require_positive_return": True,
            "max_positions": 8,
            "min_trend": 0.0,
            "max_gross": 0.85,
            "rebalance_bars": 3,
            "stop_loss_pct": 0.12,
            "max_hold_bars": 180,
        },
    ),
    "1d": (
        {
            "variant": "breadth_timer_macro",
            "trend_window": 50,
            "risk_on_breadth": 0.55,
            "risk_off_breadth": 0.35,
            "require_positive_return": True,
            "max_positions": 6,
            "min_trend": 0.0,
            "max_gross": 0.90,
            "rebalance_bars": 2,
            "stop_loss_pct": 0.14,
            "max_hold_bars": 90,
        },
    ),
}


def _build_breadth_regime_trend_timer_candidates(ctx: _CandidateBuildContext) -> None:
    """Breadth-gated total-exposure crypto trend timer (cross_sectional basket).

    Crypto-only basket over ``ctx.crypto_only_symbols`` with a >=``min_symbols``
    guard so it only fires once a genuinely broad basket is available; never leaks
    ``ctx.crypto_symbols`` / raw symbols.
    """
    crypto_symbols = tuple(ctx.crypto_only_symbols)
    for timeframe in ctx._present("1h", "4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _BREADTH_REGIME_TREND_TIMER_SLICE.get(timeframe, ()):
            min_symbols = 5
            if len(crypto_symbols) < min_symbols:
                continue
            params = {
                "trend_window": int(spec["trend_window"]),
                "risk_on_breadth": float(spec["risk_on_breadth"]),
                "risk_off_breadth": float(spec["risk_off_breadth"]),
                "require_positive_return": bool(spec["require_positive_return"]),
                "max_positions": int(spec["max_positions"]),
                "min_trend": float(spec["min_trend"]),
                "max_gross": float(spec["max_gross"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "min_symbols": int(min_symbols),
                "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"breadth_regime_trend_timer_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['trend_window'])}_{int(spec['risk_on_breadth'] * 100)}"
                ),
                family="cross_sectional",
                strategy_class="BreadthRegimeTrendTimerStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Top-down breadth-gated trend timer: cross-sectional breadth "
                    "(fraction of the crypto basket above its own trend) scales TOTAL "
                    "net-long gross exposure; strong breadth holds a breadth-scaled "
                    "long basket of up-trenders, collapsing breadth goes flat "
                    f"(risk-off) for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "breadth",
                    "regime",
                    "market_timing",
                    "trend",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto_basket",
                    "decision_cadence_seconds": _RIDER_TF_CADENCE_SECONDS.get(timeframe, 1800),
                },
            )


_DUAL_MOMENTUM_INDEX_ROTATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "absrel_top3",
            "absolute_lookback_bars": 12,
            "blend_lookbacks": "1,3,6,12",
            "sma_bars": 200,
            "rebalance_bars": 21,
            "max_holdings": 3,
            "stop_loss_pct": 0.10,
            "max_hold_bars": 252,
        },
    ),
}


def _build_dual_momentum_index_rotation_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S4 (DORMANT) — dual-momentum index rotation (multi-symbol cross_sectional basket)."""
    filtered = _intersect_universe(_INDEX_ROTATION_UNIVERSE, ctx.normalized_symbols)
    if len(filtered) < 4:
        return
    for timeframe in ctx._present("1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _DUAL_MOMENTUM_INDEX_ROTATION_SLICE.get(timeframe, ()):
            params = {
                "absolute_lookback_bars": int(spec["absolute_lookback_bars"]),
                "blend_lookbacks": str(spec["blend_lookbacks"]),
                "sma_bars": int(spec["sma_bars"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "max_holdings": int(spec["max_holdings"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"dual_momentum_index_rotation_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['max_holdings'])}_{int(spec['sma_bars'])}"
                ),
                family="cross_sectional",
                strategy_class="DualMomentumIndexRotationStrategy",
                timeframe=timeframe,
                symbols=filtered,
                params=params,
                notes=(
                    "DORMANT index tranche: dual-momentum rotation that gates on an "
                    "absolute-return filter then rotates into the top blended-"
                    f"momentum index perps for {timeframe} ({spec['variant']}); "
                    "self-skips until index perps materialize."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "index",
                    "dual_momentum",
                    "rotation",
                    "dormant",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "index",
                    "dormant_tranche": True,
                    "decision_cadence_seconds": 86400,
                },
            )


_CALENDAR_SEASONALITY_OVERLAY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "tom_dow",
            "turn_of_month_pre_days": 3,
            "turn_of_month_post_days": 3,
            "enable_turn_of_month": True,
            "enable_day_of_week": True,
            "weekday_long_mask": "1,1,0,0,1,0,0",
            "hold_bars": 2,
            "stop_loss_pct": 0.05,
        },
    ),
}


def _build_calendar_seasonality_overlay_candidates(
    ctx: _CandidateBuildContext,
) -> None:
    """S11 (DORMANT) — per-index calendar seasonality tilt (single-asset)."""
    filtered = _intersect_universe(_INDEX_ROTATION_UNIVERSE, ctx.normalized_symbols)
    if not filtered:
        return
    # Wire one single-asset candidate per preferred index that is present.
    preferred = _intersect_universe(_INDEX_PER_ASSET_PREFERENCES, ctx.normalized_symbols)
    index_symbols = preferred or filtered
    for timeframe in ctx._present("1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _CALENDAR_SEASONALITY_OVERLAY_SLICE.get(timeframe, ()):
            for symbol in index_symbols:
                params = {
                    "index_symbol": symbol,
                    "turn_of_month_pre_days": int(spec["turn_of_month_pre_days"]),
                    "turn_of_month_post_days": int(spec["turn_of_month_post_days"]),
                    "enable_turn_of_month": bool(spec["enable_turn_of_month"]),
                    "enable_day_of_week": bool(spec["enable_day_of_week"]),
                    "weekday_long_mask": str(spec["weekday_long_mask"]),
                    "hold_bars": int(spec["hold_bars"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                }
                _add_candidate(
                    ctx.candidates,
                    name=(
                        f"calendar_seasonality_overlay_{tf_tag}_{spec['variant']}_"
                        f"{symbol.replace('/', '').lower()}"
                    ),
                    family="seasonality",
                    strategy_class="CalendarSeasonalityOverlayStrategy",
                    timeframe=timeframe,
                    symbols=(symbol,),
                    params=params,
                    notes=(
                        "DORMANT index tranche: per-index turn-of-month and day-of-"
                        "week seasonality tilt computed from the calendar only for "
                        f"{symbol} on {timeframe} ({spec['variant']}); no-ops until "
                        "the index perp materializes."
                    ),
                    tags=(
                        "seasonality",
                        "calendar",
                        "single_asset",
                        "index",
                        "dormant",
                    ),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": symbol,
                        "dormant_tranche": True,
                        "decision_cadence_seconds": 86400,
                    },
                )


_IDIOSYNCRATIC_VOLATILITY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "ivol_lo",
            "beta_window": 120,
            "vol_window": 60,
            "rebalance_bars": 6,
            "quantile_pct": 0.25,
            "rebalance_band": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.080,
            "max_hold_bars": 120,
        },
    ),
    "1d": (
        {
            "variant": "ivol_ls",
            "beta_window": 120,
            "vol_window": 60,
            "rebalance_bars": 5,
            "quantile_pct": 0.25,
            "rebalance_band": 0.30,
            "allow_short": True,
            "stop_loss_pct": 0.080,
            "max_hold_bars": 120,
        },
    ),
}


def _build_idiosyncratic_volatility_candidates(ctx: _CandidateBuildContext) -> None:
    """Idiosyncratic-volatility anomaly basket (long low / short high residual vol)."""
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _IDIOSYNCRATIC_VOLATILITY_SLICE.get(timeframe, ()):
            params = {
                "beta_window": int(spec["beta_window"]),
                "vol_window": int(spec["vol_window"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "quantile_pct": float(spec["quantile_pct"]),
                "rebalance_band": float(spec["rebalance_band"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"idiosyncratic_volatility_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['beta_window'])}_{int(spec['vol_window'])}"
                ),
                family="cross_sectional",
                strategy_class="IdiosyncraticVolatilityStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Idiosyncratic-volatility anomaly sleeve: ranks symbols by the "
                    "volatility of their benchmark-residual returns and goes long the "
                    "low-idio-vol quantile / short the high one for "
                    f"{timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "anomaly",
                    "low_vol",
                    "idiosyncratic",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "allow_short": bool(spec["allow_short"]),
                    "decision_cadence_seconds": 86400,
                },
            )


_LOTTERY_SKEWNESS_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "lottery_lo",
            "skew_window": 60,
            "max_window": 20,
            "max_weight": 0.50,
            "rebalance_bars": 6,
            "quantile_pct": 0.25,
            "rebalance_band": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.080,
            "max_hold_bars": 120,
        },
    ),
    "1d": (
        {
            "variant": "lottery_ls",
            "skew_window": 60,
            "max_window": 20,
            "max_weight": 0.50,
            "rebalance_bars": 5,
            "quantile_pct": 0.25,
            "rebalance_band": 0.30,
            "allow_short": True,
            "stop_loss_pct": 0.080,
            "max_hold_bars": 120,
        },
    ),
}


def _build_lottery_skewness_candidates(ctx: _CandidateBuildContext) -> None:
    """Lottery-preference (skewness / MAX) anomaly basket (short lottery names)."""
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("4h", "1d"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _LOTTERY_SKEWNESS_SLICE.get(timeframe, ()):
            params = {
                "skew_window": int(spec["skew_window"]),
                "max_window": int(spec["max_window"]),
                "max_weight": float(spec["max_weight"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "quantile_pct": float(spec["quantile_pct"]),
                "rebalance_band": float(spec["rebalance_band"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"lottery_skewness_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['skew_window'])}_{int(spec['max_window'])}"
                ),
                family="cross_sectional",
                strategy_class="LotterySkewnessStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Lottery-preference anomaly sleeve: blends trailing return "
                    "skewness with the Bali-Cakici-Whitelaw MAX single-bar return, "
                    "shorting high-lottery names and going long low-skew names for "
                    f"{timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "anomaly",
                    "skewness",
                    "lottery",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "allow_short": bool(spec["allow_short"]),
                    "decision_cadence_seconds": 86400,
                },
            )


_TREND_EFFICIENCY_MOMENTUM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "teff_lo",
            "efficiency_period": 20,
            "trend_lookback_bars": 20,
            "rebalance_bars": 6,
            "quantile_pct": 0.25,
            "signal_threshold": 0.10,
            "rebalance_band": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.080,
            "max_hold_bars": 120,
        },
    ),
    "4h": (
        {
            "variant": "teff_ls",
            "efficiency_period": 20,
            "trend_lookback_bars": 20,
            "rebalance_bars": 5,
            "quantile_pct": 0.25,
            "signal_threshold": 0.10,
            "rebalance_band": 0.30,
            "allow_short": True,
            "stop_loss_pct": 0.080,
            "max_hold_bars": 120,
        },
    ),
}


def _build_trend_efficiency_momentum_candidates(ctx: _CandidateBuildContext) -> None:
    """Trend-quality (Kaufman efficiency) momentum basket (long clean uptrends)."""
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _TREND_EFFICIENCY_MOMENTUM_SLICE.get(timeframe, ()):
            params = {
                "efficiency_period": int(spec["efficiency_period"]),
                "trend_lookback_bars": int(spec["trend_lookback_bars"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "quantile_pct": float(spec["quantile_pct"]),
                "signal_threshold": float(spec["signal_threshold"]),
                "rebalance_band": float(spec["rebalance_band"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"trend_efficiency_momentum_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['efficiency_period'])}_{int(spec['trend_lookback_bars'])}"
                ),
                family="cross_sectional",
                strategy_class="TrendEfficiencyMomentumStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Trend-quality momentum sleeve: scores each symbol by Kaufman "
                    "efficiency ratio times the sign of its trailing trend, going long "
                    "clean high-efficiency uptrends / short low-efficiency downtrends "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "anomaly",
                    "trend_quality",
                    "efficiency",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "allow_short": bool(spec["allow_short"]),
                    "decision_cadence_seconds": 86400,
                },
            )


_DISPERSION_CONDITIONED_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "disp_lo",
            "reversion_lookback_bars": 5,
            "dispersion_threshold": 0.020,
            "rebalance_bars": 3,
            "quantile_pct": 0.25,
            "rebalance_band": 0.25,
            "allow_short": False,
            "stop_loss_pct": 0.060,
            "max_hold_bars": 48,
        },
    ),
    "4h": (
        {
            "variant": "disp_ls",
            "reversion_lookback_bars": 5,
            "dispersion_threshold": 0.030,
            "rebalance_bars": 2,
            "quantile_pct": 0.25,
            "rebalance_band": 0.30,
            "allow_short": True,
            "stop_loss_pct": 0.060,
            "max_hold_bars": 48,
        },
    ),
}


def _build_dispersion_conditioned_reversion_candidates(ctx: _CandidateBuildContext) -> None:
    """Regime-gated cross-sectional reversion basket (fade extremes in high dispersion)."""
    crypto_symbols = ctx.crypto_symbols
    if len(crypto_symbols) < 4:
        return
    for timeframe in ctx._present("1h", "4h"):
        tf_tag = timeframe.replace("/", "-")
        for spec in _DISPERSION_CONDITIONED_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "reversion_lookback_bars": int(spec["reversion_lookback_bars"]),
                "dispersion_threshold": float(spec["dispersion_threshold"]),
                "rebalance_bars": int(spec["rebalance_bars"]),
                "quantile_pct": float(spec["quantile_pct"]),
                "rebalance_band": float(spec["rebalance_band"]),
                "allow_short": bool(spec["allow_short"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
            }
            _add_candidate(
                ctx.candidates,
                name=(
                    f"dispersion_conditioned_reversion_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['reversion_lookback_bars'])}_"
                    f"{float(spec['dispersion_threshold']):.3f}"
                ),
                family="cross_sectional",
                strategy_class="DispersionConditionedReversionStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes=(
                    "Regime-gated cross-sectional reversion sleeve: only trades when "
                    "cross-sectional return dispersion exceeds a threshold, then fades "
                    "extreme movers toward the basket mean for "
                    f"{timeframe} ({spec['variant']})."
                ),
                tags=(
                    *_CROSS_SECTIONAL_ADMISSION_TAGS,
                    "anomaly",
                    "mean_reversion",
                    "dispersion",
                    "regime",
                    "crypto",
                ),
                metadata={
                    "timeframe": timeframe,
                    "retune_profile": str(spec["variant"]),
                    "symbol_scope": "crypto",
                    "allow_short": bool(spec["allow_short"]),
                    "decision_cadence_seconds": 86400,
                },
            )


def build_binance_futures_candidates(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> list[StrategyCandidate]:
    """Build candidate universe for RG_PVTM and diversifier sleeves."""
    normalized_timeframes = tuple(
        normalize_strategy_timeframes(
            list(timeframes),
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    normalized_symbols = _normalize_unique(symbols)
    if not normalized_timeframes:
        raise ValueError("timeframes must not be empty")
    if len(normalized_symbols) < 2:
        raise ValueError("symbols must include at least two instruments")

    return _CandidateBuildContext(
        normalized_timeframes=normalized_timeframes,
        normalized_symbols=normalized_symbols,
    ).build()


def build_article_pipeline_candidates(
    *,
    timeframes: Sequence[str] = ("5m", "15m", "30m", "1h", "4h"),
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
    max_per_family: int = 0,
    max_total: int = 0,
) -> list[StrategyCandidate]:
    """Build only candidates tagged for the article-driven research pipeline.

    Defaults deliberately exclude the 1s micro sleeve and 1d long-horizon sweep so
    the resulting manifest remains lightweight for low-memory sequential research.
    """
    rows = build_binance_futures_candidates(timeframes=timeframes, symbols=symbols)
    article_rows = [row for row in rows if "article_pipeline" in row.tags]
    article_rows.sort(key=lambda row: (row.family, row.timeframe, row.strategy_class, row.name))

    if max_per_family > 0:
        family_counts: dict[str, int] = {}
        limited_rows: list[StrategyCandidate] = []
        for row in article_rows:
            count = family_counts.get(row.family, 0)
            if count >= max_per_family:
                continue
            family_counts[row.family] = count + 1
            limited_rows.append(row)
        article_rows = limited_rows

    if max_total > 0:
        article_rows = article_rows[: max(1, int(max_total))]
    return article_rows


def build_candidate_manifest(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> dict[str, Any]:
    """Build a JSON-ready manifest with aggregate metadata."""
    normalized_symbols = tuple(canonicalize_symbol_list(symbols))
    normalized_timeframes = tuple(
        normalize_strategy_timeframes(
            list(timeframes),
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    candidates = build_binance_futures_candidates(
        timeframes=normalized_timeframes,
        symbols=normalized_symbols,
    )

    family_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}

    for candidate in candidates:
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
        strategy_counts[candidate.strategy_class] = (
            strategy_counts.get(candidate.strategy_class, 0) + 1
        )
        timeframe_counts[candidate.timeframe] = timeframe_counts.get(candidate.timeframe, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol_universe": list(normalized_symbols),
        "timeframes": list(normalized_timeframes),
        "candidate_count": len(candidates),
        "family_counts": family_counts,
        "strategy_counts": strategy_counts,
        "timeframe_counts": timeframe_counts,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def build_article_pipeline_manifest(
    *,
    timeframes: Sequence[str] = ("5m", "15m", "30m", "1h", "4h"),
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
    max_per_family: int = 0,
    max_total: int = 0,
) -> dict[str, Any]:
    normalized_symbols = tuple(canonicalize_symbol_list(symbols))
    normalized_timeframes = tuple(
        normalize_strategy_timeframes(
            list(timeframes),
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    candidates = build_article_pipeline_candidates(
        timeframes=normalized_timeframes,
        symbols=normalized_symbols,
        max_per_family=max_per_family,
        max_total=max_total,
    )

    family_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    article_family_counts: dict[str, int] = {}
    for candidate in candidates:
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
        strategy_counts[candidate.strategy_class] = (
            strategy_counts.get(candidate.strategy_class, 0) + 1
        )
        timeframe_counts[candidate.timeframe] = timeframe_counts.get(candidate.timeframe, 0) + 1
        for family_id in list(candidate.metadata.get("article_pipeline_family_ids") or []):
            token = str(family_id)
            article_family_counts[token] = article_family_counts.get(token, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol_universe": list(normalized_symbols),
        "timeframes": list(normalized_timeframes),
        "candidate_count": len(candidates),
        "family_counts": family_counts,
        "strategy_counts": strategy_counts,
        "timeframe_counts": timeframe_counts,
        "article_family_counts": article_family_counts,
        "max_per_family": int(max_per_family),
        "max_total": int(max_total),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
