"""Centralized strategy registry + parameter schema integration."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, cast

from lumina_quant.strategy import Strategy
from lumina_quant.tuning import ParamRegistry

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


def _has_perp_support_data() -> bool:
    explicit = str(os.getenv("LQ_PERP_SUPPORT_DATA_PATH", "")).strip()
    if explicit:
        return Path(explicit).exists()

    fallback = Path("data") / "market_parquet" / "feature_points"
    return fallback.exists()


MovingAverageCrossStrategy = _optional_strategy_class(
    "moving_average", "MovingAverageCrossStrategy"
)
RollingBreakoutStrategy = _optional_strategy_class("rolling_breakout", "RollingBreakoutStrategy")
RareEventScoreStrategy = _optional_strategy_class("rare_event_score", "RareEventScoreStrategy")
RsiStrategy = _optional_strategy_class("rsi_strategy", "RsiStrategy")
RegimeBreakoutCandidateStrategy = _optional_strategy_class(
    "candidate_regime_breakout", "RegimeBreakoutCandidateStrategy"
)
VolatilityCompressionReversionStrategy = _optional_strategy_class(
    "candidate_vol_compression_reversion", "VolatilityCompressionReversionStrategy"
)
CompositeTrendStrategy = _optional_strategy_class("composite_trend", "CompositeTrendStrategy")
VolCompressionVWAPReversionStrategy = _optional_strategy_class(
    "vol_compression_vwap_reversion", "VolCompressionVWAPReversionStrategy"
)
VolCompressionVwapReversionStrategy = _optional_strategy_class(
    "vol_compression_vwap_reversion", "VolCompressionVwapReversionStrategy"
)
LeadLagSpilloverStrategy = _optional_strategy_class(
    "leadlag_spillover", "LeadLagSpilloverStrategy"
)
PairSpreadZScoreStrategy = _optional_strategy_class("pair_spread_zscore", "PairSpreadZScoreStrategy")
MicroRangeExpansion1sStrategy = _optional_strategy_class(
    "micro_range_expansion_1s", "MicroRangeExpansion1sStrategy"
)
PerpCrowdingCarryStrategy = _optional_strategy_class(
    "perp_crowding_carry", "PerpCrowdingCarryStrategy"
)

StrategyClass = type[Strategy]

DEFAULT_STRATEGY_NAME = "RsiStrategy" if RsiStrategy is not None else "MeanReversionStdStrategy"

_RAW_STRATEGY_MAP: dict[str, StrategyClass | None] = {
    "BitcoinBuyHoldStrategy": BitcoinBuyHoldStrategy,
    "LagConvergenceStrategy": LagConvergenceStrategy,
    "MeanReversionStdStrategy": MeanReversionStdStrategy,
    "RsiStrategy": RsiStrategy,
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
    "PairTradingZScoreStrategy": PairTradingZScoreStrategy,
    "PairSpreadZScoreStrategy": PairSpreadZScoreStrategy,
    "RareEventScoreStrategy": RareEventScoreStrategy,
    "RegimeBreakoutCandidateStrategy": RegimeBreakoutCandidateStrategy,
    "RollingBreakoutStrategy": RollingBreakoutStrategy,
    "TopCapTimeSeriesMomentumStrategy": TopCapTimeSeriesMomentumStrategy,
    "VolatilityCompressionReversionStrategy": VolatilityCompressionReversionStrategy,
    "VwapReversionStrategy": VwapReversionStrategy,
    "CompositeTrendStrategy": CompositeTrendStrategy,
    "VolCompressionVWAPReversionStrategy": VolCompressionVWAPReversionStrategy,
    "VolCompressionVwapReversionStrategy": VolCompressionVwapReversionStrategy,
    "LeadLagSpilloverStrategy": LeadLagSpilloverStrategy,
    "MicroRangeExpansion1sStrategy": MicroRangeExpansion1sStrategy,
}
if _has_perp_support_data():
    _RAW_STRATEGY_MAP["PerpCrowdingCarryStrategy"] = PerpCrowdingCarryStrategy

_STRATEGY_MAP: dict[str, StrategyClass] = {
    name: cast(StrategyClass, cls) for name, cls in _RAW_STRATEGY_MAP.items() if cls is not None
}

_STRATEGY_TIER_HINTS: dict[str, str] = {
    "BitcoinBuyHoldStrategy": "live_default",
    "LagConvergenceStrategy": "live_default",
    "MeanReversionStdStrategy": "live_default",
    "RsiStrategy": "live_default",
    "MovingAverageCrossStrategy": "live_default",
    "PairTradingZScoreStrategy": "live_default",
    "RollingBreakoutStrategy": "live_default",
    "TopCapTimeSeriesMomentumStrategy": "live_default",
    "VwapReversionStrategy": "live_default",
    "RareEventScoreStrategy": "live_opt_in",
    "PairSpreadZScoreStrategy": "live_opt_in",
    "CompositeTrendStrategy": "live_opt_in",
    "VolCompressionVWAPReversionStrategy": "live_opt_in",
    "VolCompressionVwapReversionStrategy": "live_opt_in",
    "LeadLagSpilloverStrategy": "live_opt_in",
    "PerpCrowdingCarryStrategy": "live_opt_in",
    "RegimeBreakoutCandidateStrategy": "research_only",
    "VolatilityCompressionReversionStrategy": "research_only",
    "MicroRangeExpansion1sStrategy": "research_only",
}

_STRATEGY_METADATA: dict[str, dict[str, Any]] = {
    name: {
        "name": name,
        "tier": str(_STRATEGY_TIER_HINTS.get(name, "live_default")),
    }
    for name in _STRATEGY_MAP
}

_OPTUNA_TRIAL_OVERRIDES: dict[str, str] = {
    "LagConvergenceStrategy": "24",
    "MeanReversionStdStrategy": "24",
    "RsiStrategy": "20",
    "MovingAverageCrossStrategy": "20",
    "PairTradingZScoreStrategy": "32",
    "PairSpreadZScoreStrategy": "24",
    "RareEventScoreStrategy": "28",
    "RollingBreakoutStrategy": "24",
    "RegimeBreakoutCandidateStrategy": "24",
    "VwapReversionStrategy": "24",
    "VolatilityCompressionReversionStrategy": "24",
    "CompositeTrendStrategy": "24",
    "VolCompressionVWAPReversionStrategy": "24",
    "VolCompressionVwapReversionStrategy": "24",
    "LeadLagSpilloverStrategy": "24",
    "PerpCrowdingCarryStrategy": "16",
    "MicroRangeExpansion1sStrategy": "16",
}


def _resolve_optuna_trial_budget(raw_value: str | None) -> int:
    token = str(raw_value or os.getenv("LQ_OPTUNA_DEFAULT_TRIALS", "20")).strip()
    try:
        parsed = int(token)
    except Exception:
        return 16
    return max(1, parsed)


def _strategy_schema(strategy_cls: StrategyClass) -> dict[str, Any]:
    getter = getattr(strategy_cls, "get_param_schema", None)
    if not callable(getter):
        return {}
    try:
        schema = getter()
    except Exception:
        return {}
    return dict(schema) if isinstance(schema, dict) else {}


_PARAM_REGISTRY = ParamRegistry()
for _strategy_name, _strategy_cls in _STRATEGY_MAP.items():
    _schema = _strategy_schema(_strategy_cls)
    if not _schema:
        continue
    _PARAM_REGISTRY.register(
        _strategy_name,
        _schema,
        optuna_trials=_resolve_optuna_trial_budget(_OPTUNA_TRIAL_OVERRIDES.get(_strategy_name)),
    )


def get_strategy_metadata(strategy_name: str) -> dict[str, Any]:
    """Return strategy metadata (tier and name)."""

    token = str(strategy_name)
    return dict(_STRATEGY_METADATA.get(token, {"name": token, "tier": "live_default"}))


def get_strategy_tier(strategy_name: str) -> str:
    return str(get_strategy_metadata(strategy_name).get("tier", "live_default"))


def get_strategy_map() -> dict[str, StrategyClass]:
    return dict(_STRATEGY_MAP)


def get_live_strategy_map(*, include_opt_in: bool = True) -> dict[str, StrategyClass]:
    allowed = {"live_default", "live_opt_in"} if include_opt_in else {"live_default"}
    return {
        name: cls
        for name, cls in _STRATEGY_MAP.items()
        if get_strategy_tier(name) in allowed
    }


def get_strategy_names(*, include_research_only: bool = True) -> list[str]:
    if include_research_only:
        return sorted(_STRATEGY_MAP.keys())
    return sorted(
        name for name in _STRATEGY_MAP if get_strategy_tier(name) != "research_only"
    )


def get_live_strategy_names(*, include_opt_in: bool = True) -> list[str]:
    return sorted(get_live_strategy_map(include_opt_in=include_opt_in).keys())


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


def get_strategy_param_schema(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.get_schema(str(strategy_name))


def get_strategy_canonical_param_names(strategy_name: str) -> dict[str, str]:
    return _PARAM_REGISTRY.get_canonical_names(str(strategy_name))


def resolve_strategy_params(strategy_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return _PARAM_REGISTRY.resolve_params(str(strategy_name), overrides or {}, keep_unknown=True)


def get_default_strategy_params(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.default_params(str(strategy_name))


def resolve_optuna_config(strategy_name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    return _PARAM_REGISTRY.resolve_optuna_config(str(strategy_name), override or {})


def get_default_optuna_config(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.default_optuna_config(str(strategy_name))


def resolve_grid_config(strategy_name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    return _PARAM_REGISTRY.resolve_grid_config(str(strategy_name), override or {})


def get_default_grid_config(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.default_grid_config(str(strategy_name))
