"""Public sample configuration loading.

The public configuration schema is intentionally small and generic. It only
controls local sample-data replay parameters and does not contain credentials,
connectors, research metadata, or private strategy settings.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATA_PATH = "sample_data/sample_ohlcv.csv"
DEFAULT_FAST_WINDOW = 3
DEFAULT_SLOW_WINDOW = 8
DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_FEE_BPS = 1.0
DEFAULT_FAST_GRID = "2,3,4"
DEFAULT_SLOW_GRID = "6,8,10"
DEFAULT_OPTIMIZATION_METHOD = "grid"
DEFAULT_OPTUNA_TRIALS = 16
DEFAULT_SAMPLER_SEED = 7


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    data_path: str = DEFAULT_DATA_PATH
    fast_window: int = DEFAULT_FAST_WINDOW
    slow_window: int = DEFAULT_SLOW_WINDOW
    initial_cash: float = DEFAULT_INITIAL_CASH
    fee_bps: float = DEFAULT_FEE_BPS
    engine: str = "python"


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    data_path: str = DEFAULT_DATA_PATH
    method: str = DEFAULT_OPTIMIZATION_METHOD
    fast_grid: str = DEFAULT_FAST_GRID
    slow_grid: str = DEFAULT_SLOW_GRID
    initial_cash: float = DEFAULT_INITIAL_CASH
    fee_bps: float = DEFAULT_FEE_BPS
    n_trials: int = DEFAULT_OPTUNA_TRIALS
    sampler_seed: int = DEFAULT_SAMPLER_SEED


@dataclass(frozen=True, slots=True)
class PublicPipelineConfig:
    backtest: BacktestConfig
    paper_live: BacktestConfig
    optimization: OptimizationConfig


def default_public_pipeline_config() -> PublicPipelineConfig:
    backtest = BacktestConfig()
    return PublicPipelineConfig(
        backtest=backtest,
        paper_live=BacktestConfig(engine="python"),
        optimization=OptimizationConfig(),
    )


def load_public_pipeline_config(path: str | Path) -> PublicPipelineConfig:
    """Load the public sample TOML config."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    defaults = default_public_pipeline_config()
    common = _table(raw, "sample")
    return PublicPipelineConfig(
        backtest=_load_backtest_config(_table(raw, "backtest"), common, defaults.backtest),
        paper_live=_load_backtest_config(_table(raw, "paper_live"), common, defaults.paper_live),
        optimization=_load_optimization_config(
            _table(raw, "optimization"),
            common,
            defaults.optimization,
        ),
    )


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _load_backtest_config(
    section: dict[str, Any],
    common: dict[str, Any],
    defaults: BacktestConfig,
) -> BacktestConfig:
    engine = _string(section, "engine", defaults.engine)
    if engine not in {"python", "rust"}:
        raise ValueError("backtest engine must be 'python' or 'rust'")
    return BacktestConfig(
        data_path=_string(section, "data_path", _string(common, "data_path", defaults.data_path)),
        fast_window=_integer(section, "fast_window", defaults.fast_window),
        slow_window=_integer(section, "slow_window", defaults.slow_window),
        initial_cash=_number(
            section,
            "initial_cash",
            _number(common, "initial_cash", defaults.initial_cash),
        ),
        fee_bps=_number(section, "fee_bps", _number(common, "fee_bps", defaults.fee_bps)),
        engine=engine,
    )


def _load_optimization_config(
    section: dict[str, Any],
    common: dict[str, Any],
    defaults: OptimizationConfig,
) -> OptimizationConfig:
    method = _string(section, "method", defaults.method)
    if method not in {"grid", "optuna"}:
        raise ValueError("optimization method must be 'grid' or 'optuna'")
    n_trials = _integer(section, "n_trials", defaults.n_trials)
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return OptimizationConfig(
        data_path=_string(section, "data_path", _string(common, "data_path", defaults.data_path)),
        method=method,
        fast_grid=_grid(section, "fast_grid", defaults.fast_grid),
        slow_grid=_grid(section, "slow_grid", defaults.slow_grid),
        initial_cash=_number(
            section,
            "initial_cash",
            _number(common, "initial_cash", defaults.initial_cash),
        ),
        fee_bps=_number(section, "fee_bps", _number(common, "fee_bps", defaults.fee_bps)),
        n_trials=n_trials,
        sampler_seed=_integer(section, "sampler_seed", defaults.sampler_seed),
    )


def _string(section: dict[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    if not value:
        raise ValueError(f"{key} must not be empty")
    return value


def _integer(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(section: dict[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _grid(section: dict[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{key} must not be empty")
        return value
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{key} must contain at least one value")
        converted = []
        for item in value:
            if not isinstance(item, int):
                raise ValueError(f"{key} values must be integers")
            converted.append(str(item))
        return ",".join(converted)
    raise ValueError(f"{key} must be a comma string or integer list")
