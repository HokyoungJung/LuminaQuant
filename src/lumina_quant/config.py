"""Public sample configuration loading.

The public configuration schema is intentionally small and generic. It only
controls local sample-data replay parameters and does not contain credentials,
connectors, research metadata, or private strategy settings.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumina_quant.strategy_loader import DEFAULT_STRATEGY_REF, is_sample_strategy_ref

DEFAULT_DATA_PATH = "sample_data/sample_ohlcv.csv"
DEFAULT_FAST_WINDOW = 3
DEFAULT_SLOW_WINDOW = 8
DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_FEE_BPS = 1.0
DEFAULT_FAST_GRID = "2,3,4"
DEFAULT_SLOW_GRID = "6,8,10"
DEFAULT_OPTIMIZATION_METHOD = "optuna"
DEFAULT_OPTUNA_TRIALS = 16
DEFAULT_SAMPLER_SEED = 7


@dataclass(frozen=True, slots=True)
class SearchParam:
    kind: str
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    choices: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "choices": list(self.choices),
        }


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    data_path: str = DEFAULT_DATA_PATH
    strategy_ref: str = DEFAULT_STRATEGY_REF
    strategy_params: dict[str, Any] = field(default_factory=dict)
    fast_window: int = DEFAULT_FAST_WINDOW
    slow_window: int = DEFAULT_SLOW_WINDOW
    initial_cash: float = DEFAULT_INITIAL_CASH
    fee_bps: float = DEFAULT_FEE_BPS
    engine: str = "python"


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    data_path: str = DEFAULT_DATA_PATH
    strategy_ref: str = DEFAULT_STRATEGY_REF
    strategy_params: dict[str, Any] = field(default_factory=dict)
    method: str = DEFAULT_OPTIMIZATION_METHOD
    fast_grid: str = DEFAULT_FAST_GRID
    slow_grid: str = DEFAULT_SLOW_GRID
    search_space: dict[str, SearchParam] = field(default_factory=dict)
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
    sample_params = {"fast_window": DEFAULT_FAST_WINDOW, "slow_window": DEFAULT_SLOW_WINDOW}
    return PublicPipelineConfig(
        backtest=BacktestConfig(strategy_params=dict(sample_params)),
        paper_live=BacktestConfig(engine="python", strategy_params=dict(sample_params)),
        optimization=OptimizationConfig(search_space=_default_sample_search_space()),
    )


def load_public_pipeline_config(path: str | Path) -> PublicPipelineConfig:
    """Load the public sample TOML config."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    defaults = default_public_pipeline_config()
    common = _table(raw, "sample")
    strategy = _table(raw, "strategy")
    return PublicPipelineConfig(
        backtest=_load_backtest_config(
            _table(raw, "backtest"),
            common,
            strategy,
            defaults.backtest,
        ),
        paper_live=_load_backtest_config(
            _table(raw, "paper_live"),
            common,
            strategy,
            defaults.paper_live,
        ),
        optimization=_load_optimization_config(
            _table(raw, "optimization"),
            common,
            strategy,
            defaults.optimization,
        ),
    )


def _default_sample_search_space() -> dict[str, SearchParam]:
    return {
        "fast_window": SearchParam(kind="int", low=2, high=4, step=1),
        "slow_window": SearchParam(kind="int", low=6, high=10, step=1),
    }


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _load_backtest_config(
    section: dict[str, Any],
    common: dict[str, Any],
    strategy: dict[str, Any],
    defaults: BacktestConfig,
) -> BacktestConfig:
    engine = _string(section, "engine", defaults.engine)
    if engine not in {"python", "rust"}:
        raise ValueError("backtest engine must be 'python' or 'rust'")
    strategy_ref = _strategy_ref(section, strategy, defaults.strategy_ref)
    default_params = defaults.strategy_params if is_sample_strategy_ref(strategy_ref) else {}
    strategy_params = _merged_strategy_params(strategy, section, default_params)
    fast_window = _integer(section, "fast_window", defaults.fast_window)
    slow_window = _integer(section, "slow_window", defaults.slow_window)
    if "fast_window" in section:
        strategy_params["fast_window"] = fast_window
    if "slow_window" in section:
        strategy_params["slow_window"] = slow_window
    return BacktestConfig(
        data_path=_string(section, "data_path", _string(common, "data_path", defaults.data_path)),
        strategy_ref=strategy_ref,
        strategy_params=strategy_params,
        fast_window=fast_window,
        slow_window=slow_window,
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
    strategy: dict[str, Any],
    defaults: OptimizationConfig,
) -> OptimizationConfig:
    method = _string(section, "method", defaults.method)
    if method not in {"grid", "optuna"}:
        raise ValueError("optimization method must be 'grid' or 'optuna'")
    n_trials = _integer(section, "n_trials", defaults.n_trials)
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    search_space = _search_space(_table(section, "search_space"), defaults.search_space)
    return OptimizationConfig(
        data_path=_string(section, "data_path", _string(common, "data_path", defaults.data_path)),
        strategy_ref=_strategy_ref(section, strategy, defaults.strategy_ref),
        strategy_params=_merged_strategy_params(strategy, section, defaults.strategy_params),
        method=method,
        fast_grid=_grid(section, "fast_grid", defaults.fast_grid),
        slow_grid=_grid(section, "slow_grid", defaults.slow_grid),
        search_space=search_space,
        initial_cash=_number(
            section,
            "initial_cash",
            _number(common, "initial_cash", defaults.initial_cash),
        ),
        fee_bps=_number(section, "fee_bps", _number(common, "fee_bps", defaults.fee_bps)),
        n_trials=n_trials,
        sampler_seed=_integer(section, "sampler_seed", defaults.sampler_seed),
    )


def _strategy_ref(section: dict[str, Any], strategy: dict[str, Any], default: str) -> str:
    raw = (
        section.get("strategy")
        or section.get("strategy_ref")
        or strategy.get("class_path")
        or default
    )
    if not isinstance(raw, str) or not raw:
        raise ValueError("strategy reference must be a non-empty string")
    return raw


def _merged_strategy_params(
    strategy: dict[str, Any],
    section: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = dict(defaults)
    params.update(_scalar_table(_table(strategy, "params"), "strategy.params"))
    params.update(_scalar_table(_table(section, "strategy_params"), "strategy_params"))
    return params


def _search_space(raw: dict[str, Any], defaults: dict[str, SearchParam]) -> dict[str, SearchParam]:
    if not raw:
        return dict(defaults)
    return {name: _search_param(name, conf) for name, conf in raw.items()}


def _search_param(name: str, conf: Any) -> SearchParam:
    if not isinstance(conf, dict):
        raise ValueError(f"search_space.{name} must be a TOML table")
    kind = _string(conf, "type", "categorical")
    if kind == "int":
        low = _integer(conf, "low", 0)
        high = _integer(conf, "high", low)
        step = _integer(conf, "step", 1)
        if step < 1 or high < low:
            raise ValueError(f"invalid integer search space for {name}")
        return SearchParam(kind=kind, low=low, high=high, step=step)
    if kind == "float":
        low = _number(conf, "low", 0.0)
        high = _number(conf, "high", low)
        step = conf.get("step")
        if high < low:
            raise ValueError(f"invalid float search space for {name}")
        return SearchParam(
            kind=kind,
            low=low,
            high=high,
            step=float(step) if isinstance(step, int | float) else None,
        )
    if kind == "categorical":
        choices = conf.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"categorical search space for {name} needs choices")
        return SearchParam(kind=kind, choices=tuple(choices))
    raise ValueError(f"unsupported search parameter type for {name}: {kind}")


def _scalar_table(raw: dict[str, Any], label: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            raise ValueError(f"{label}.{key} must be a scalar value")
        out[key] = value
    return out


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
