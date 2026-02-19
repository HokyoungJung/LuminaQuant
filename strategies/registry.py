"""Centralized strategy registry and default parameter bundles."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any, cast

from lumina_quant.strategy import Strategy

from .bitcoin_buy_hold import BitcoinBuyHoldStrategy
from .lag_convergence import LagConvergenceStrategy
from .mean_reversion_std import MeanReversionStdStrategy
from .pair_trading_zscore import PairTradingZScoreStrategy
from .topcap_tsmom import TopCapTimeSeriesMomentumStrategy
from .vwap_reversion import VwapReversionStrategy


def _optional_strategy_class(module_name: str, class_name: str):
    package = __package__ or "strategies"
    try:
        module = importlib.import_module(f"{package}.{module_name}")
    except Exception:
        return None
    return getattr(module, class_name, None)


MovingAverageCrossStrategy = _optional_strategy_class(
    "moving_average", "MovingAverageCrossStrategy"
)
RollingBreakoutStrategy = _optional_strategy_class("rolling_breakout", "RollingBreakoutStrategy")
RsiStrategy = _optional_strategy_class("rsi_strategy", "RsiStrategy")

StrategyClass = type[Strategy]

DEFAULT_STRATEGY_NAME = "RsiStrategy" if RsiStrategy is not None else "MeanReversionStdStrategy"

_RAW_STRATEGY_MAP: dict[str, StrategyClass | None] = {
    "BitcoinBuyHoldStrategy": BitcoinBuyHoldStrategy,
    "LagConvergenceStrategy": LagConvergenceStrategy,
    "MeanReversionStdStrategy": MeanReversionStdStrategy,
    "RsiStrategy": RsiStrategy,
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
    "PairTradingZScoreStrategy": PairTradingZScoreStrategy,
    "RollingBreakoutStrategy": RollingBreakoutStrategy,
    "TopCapTimeSeriesMomentumStrategy": TopCapTimeSeriesMomentumStrategy,
    "VwapReversionStrategy": VwapReversionStrategy,
}
_STRATEGY_MAP: dict[str, StrategyClass] = {
    name: cast(StrategyClass, cls) for name, cls in _RAW_STRATEGY_MAP.items() if cls is not None
}

_DEFAULT_STRATEGY_PARAMS: dict[str, dict[str, Any]] = {
    "BitcoinBuyHoldStrategy": {
        "symbol": "BTC/USDT",
        "strength": 1.0,
    },
    "LagConvergenceStrategy": {
        "symbol_x": "",
        "symbol_y": "",
        "lag_bars": 3,
        "entry_threshold": 0.015,
        "exit_threshold": 0.004,
        "stop_threshold": 0.05,
        "max_hold_bars": 96,
        "stop_loss_pct": 0.03,
    },
    "MeanReversionStdStrategy": {
        "window": 64,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_loss_pct": 0.03,
        "allow_short": True,
    },
    "RsiStrategy": {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "allow_short": True,
    },
    "MovingAverageCrossStrategy": {
        "short_window": 10,
        "long_window": 30,
        "allow_short": True,
    },
    "PairTradingZScoreStrategy": {
        "lookback_window": 96,
        "hedge_window": 192,
        "entry_z": 2.0,
        "exit_z": 0.35,
        "stop_z": 3.5,
        "min_correlation": 0.15,
        "max_hold_bars": 240,
        "cooldown_bars": 6,
        "reentry_z_buffer": 0.20,
        "min_z_turn": 0.05,
        "stop_loss_pct": 0.04,
        "min_abs_beta": 0.02,
        "max_abs_beta": 6.0,
        "min_volume_window": 24,
        "min_volume_ratio": 0.0,
        "symbol_x": "",
        "symbol_y": "",
        "use_log_price": True,
    },
    "RollingBreakoutStrategy": {
        "lookback_bars": 48,
        "breakout_buffer": 0.0,
        "atr_window": 14,
        "atr_stop_multiplier": 2.5,
        "stop_loss_pct": 0.03,
        "allow_short": False,
    },
    "TopCapTimeSeriesMomentumStrategy": {
        "lookback_bars": 16,
        "rebalance_bars": 16,
        "signal_threshold": 0.04,
        "stop_loss_pct": 0.08,
        "max_longs": 6,
        "max_shorts": 5,
        "min_price": 0.1,
        "btc_regime_ma": 48,
        "btc_symbol": "BTC/USDT",
    },
    "VwapReversionStrategy": {
        "window": 64,
        "entry_dev": 0.02,
        "exit_dev": 0.005,
        "stop_loss_pct": 0.03,
        "allow_short": True,
    },
}

_DEFAULT_OPTUNA_CONFIG: dict[str, dict[str, Any]] = {
    "LagConvergenceStrategy": {
        "n_trials": 24,
        "params": {
            "lag_bars": {"type": "int", "low": 1, "high": 12},
            "entry_threshold": {"type": "float", "low": 0.004, "high": 0.05, "step": 0.001},
            "exit_threshold": {"type": "float", "low": 0.001, "high": 0.02, "step": 0.001},
            "stop_threshold": {"type": "float", "low": 0.01, "high": 0.12, "step": 0.002},
            "max_hold_bars": {"type": "int", "low": 12, "high": 240},
            "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.08, "step": 0.005},
        },
    },
    "MeanReversionStdStrategy": {
        "n_trials": 24,
        "params": {
            "window": {"type": "int", "low": 16, "high": 256},
            "entry_z": {"type": "float", "low": 0.8, "high": 3.5, "step": 0.1},
            "exit_z": {"type": "float", "low": 0.1, "high": 1.5, "step": 0.05},
            "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.12, "step": 0.005},
            "allow_short": {"type": "categorical", "choices": [True, False]},
        },
    },
    "RsiStrategy": {
        "n_trials": 20,
        "params": {
            "rsi_period": {"type": "int", "low": 5, "high": 30},
            "oversold": {"type": "int", "low": 20, "high": 40},
            "overbought": {"type": "int", "low": 60, "high": 90},
            "allow_short": {"type": "categorical", "choices": [True, False]},
        },
    },
    "MovingAverageCrossStrategy": {
        "n_trials": 20,
        "params": {
            "short_window": {"type": "int", "low": 5, "high": 80},
            "long_window": {"type": "int", "low": 20, "high": 250},
            "allow_short": {"type": "categorical", "choices": [True, False]},
        },
    },
    "PairTradingZScoreStrategy": {
        "n_trials": 32,
        "params": {
            "lookback_window": {"type": "int", "low": 48, "high": 240},
            "hedge_window": {"type": "int", "low": 96, "high": 480},
            "entry_z": {"type": "float", "low": 1.2, "high": 3.0, "step": 0.1},
            "exit_z": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.05},
            "stop_z": {"type": "float", "low": 2.5, "high": 5.0, "step": 0.1},
            "min_correlation": {
                "type": "float",
                "low": -0.2,
                "high": 0.8,
                "step": 0.05,
            },
            "max_hold_bars": {"type": "int", "low": 24, "high": 480},
            "cooldown_bars": {"type": "int", "low": 0, "high": 48},
            "reentry_z_buffer": {
                "type": "float",
                "low": 0.0,
                "high": 0.8,
                "step": 0.05,
            },
            "min_z_turn": {"type": "float", "low": 0.0, "high": 0.8, "step": 0.05},
            "stop_loss_pct": {
                "type": "float",
                "low": 0.005,
                "high": 0.12,
                "step": 0.005,
            },
            "min_abs_beta": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.01},
            "max_abs_beta": {"type": "float", "low": 1.0, "high": 10.0, "step": 0.1},
            "min_volume_window": {"type": "int", "low": 4, "high": 128},
            "min_volume_ratio": {
                "type": "float",
                "low": 0.0,
                "high": 1.2,
                "step": 0.05,
            },
        },
    },
    "RollingBreakoutStrategy": {
        "n_trials": 24,
        "params": {
            "lookback_bars": {"type": "int", "low": 10, "high": 240},
            "breakout_buffer": {"type": "float", "low": 0.0, "high": 0.02, "step": 0.001},
            "atr_window": {"type": "int", "low": 5, "high": 48},
            "atr_stop_multiplier": {"type": "float", "low": 0.8, "high": 6.0, "step": 0.1},
            "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.12, "step": 0.005},
            "allow_short": {"type": "categorical", "choices": [True, False]},
        },
    },
    "VwapReversionStrategy": {
        "n_trials": 24,
        "params": {
            "window": {"type": "int", "low": 16, "high": 256},
            "entry_dev": {"type": "float", "low": 0.002, "high": 0.08, "step": 0.001},
            "exit_dev": {"type": "float", "low": 0.0, "high": 0.03, "step": 0.001},
            "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.12, "step": 0.005},
            "allow_short": {"type": "categorical", "choices": [True, False]},
        },
    },
}

_DEFAULT_GRID_CONFIG: dict[str, dict[str, Any]] = {
    "LagConvergenceStrategy": {
        "params": {
            "lag_bars": [1, 2, 3, 5, 8],
            "entry_threshold": [0.008, 0.012, 0.015, 0.02, 0.03],
            "exit_threshold": [0.002, 0.004, 0.006, 0.01],
            "stop_threshold": [0.03, 0.05, 0.08],
            "max_hold_bars": [24, 48, 96, 160],
            "stop_loss_pct": [0.01, 0.02, 0.03, 0.04],
        }
    },
    "MeanReversionStdStrategy": {
        "params": {
            "window": [24, 48, 64, 96, 128],
            "entry_z": [1.2, 1.6, 2.0, 2.4, 2.8],
            "exit_z": [0.2, 0.4, 0.6, 0.8],
            "stop_loss_pct": [0.01, 0.02, 0.03, 0.04],
            "allow_short": [True, False],
        }
    },
    "RsiStrategy": {
        "params": {
            "rsi_period": [10, 14, 20],
            "oversold": [20, 25, 30],
            "overbought": [70, 75, 80],
            "allow_short": [True, False],
        }
    },
    "MovingAverageCrossStrategy": {
        "params": {
            "short_window": [10, 20, 30],
            "long_window": [40, 80, 120],
            "allow_short": [True, False],
        }
    },
    "PairTradingZScoreStrategy": {
        "params": {
            "lookback_window": [72, 96, 144],
            "hedge_window": [144, 192, 288],
            "entry_z": [1.6, 2.0, 2.4],
            "exit_z": [0.25, 0.35, 0.5],
            "stop_z": [3.0, 3.5, 4.0],
            "min_correlation": [0.0, 0.15, 0.3],
            "max_hold_bars": [96, 240, 384],
            "cooldown_bars": [0, 6, 12],
            "reentry_z_buffer": [0.0, 0.2, 0.35],
            "min_z_turn": [0.0, 0.05, 0.15],
            "stop_loss_pct": [0.02, 0.04, 0.08],
            "min_abs_beta": [0.0, 0.02, 0.1],
            "max_abs_beta": [3.0, 6.0, 9.0],
            "min_volume_window": [12, 24, 48],
            "min_volume_ratio": [0.0, 0.2, 0.5],
        }
    },
    "RollingBreakoutStrategy": {
        "params": {
            "lookback_bars": [16, 32, 48, 64, 96],
            "breakout_buffer": [0.0, 0.001, 0.002, 0.005],
            "atr_window": [8, 14, 21, 34],
            "atr_stop_multiplier": [1.2, 1.8, 2.5, 3.5],
            "stop_loss_pct": [0.01, 0.02, 0.03, 0.05],
            "allow_short": [True, False],
        }
    },
    "VwapReversionStrategy": {
        "params": {
            "window": [24, 48, 64, 96, 128],
            "entry_dev": [0.006, 0.01, 0.015, 0.02, 0.03],
            "exit_dev": [0.0, 0.002, 0.004, 0.006],
            "stop_loss_pct": [0.01, 0.02, 0.03, 0.05],
            "allow_short": [True, False],
        }
    },
}


def get_strategy_map() -> dict[str, StrategyClass]:
    return dict(_STRATEGY_MAP)


def get_strategy_names() -> list[str]:
    return sorted(_STRATEGY_MAP.keys())


def resolve_strategy_class(
    name: str | None, default_name: str = DEFAULT_STRATEGY_NAME
) -> StrategyClass:
    requested = str(name or "").strip()
    if requested in _STRATEGY_MAP:
        return _STRATEGY_MAP[requested]

    fallback = str(default_name).strip()
    if fallback in _STRATEGY_MAP:
        return _STRATEGY_MAP[fallback]
    if DEFAULT_STRATEGY_NAME in _STRATEGY_MAP:
        return _STRATEGY_MAP[DEFAULT_STRATEGY_NAME]
    if _STRATEGY_MAP:
        return next(iter(_STRATEGY_MAP.values()))
    raise ValueError("No strategy classes are available in registry")


def get_default_strategy_params(strategy_name: str) -> dict[str, Any]:
    return deepcopy(_DEFAULT_STRATEGY_PARAMS.get(str(strategy_name), {}))


def get_default_optuna_config(strategy_name: str) -> dict[str, Any]:
    return deepcopy(_DEFAULT_OPTUNA_CONFIG.get(str(strategy_name), {}))


def get_default_grid_config(strategy_name: str) -> dict[str, Any]:
    return deepcopy(_DEFAULT_GRID_CONFIG.get(str(strategy_name), {}))
