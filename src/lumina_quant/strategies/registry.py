"""Centralized strategy registry + parameter schema integration."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, cast

from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import ParamRegistry

from .adaptive_regime_momentum import AdaptiveRegimeMomentumStrategy
from .alpha101_formula import Alpha101FormulaStrategy
from .alpha_zoo_optuna_hybrid_live import AlphaZooOptunaHybridLiveStrategy
from .bitcoin_buy_hold import BitcoinBuyHoldStrategy
from .compression_breakout_continuation import CompressionBreakoutContinuationStrategy
from .crypto_fx_alpha_zoo_state import CryptoFxAlphaZooStateStrategy
from .lag_convergence import LagConvergenceStrategy
from .mean_reversion_std import MeanReversionStdStrategy
from .pair_trading_zscore import PairTradingZScoreStrategy
from .panic_rebound_mean_reversion import PanicReboundMeanReversionStrategy
from .profit_moonshot import (
    ProfitMoonshotBreakoutStrategy,
    ProfitMoonshotReversionStrategy,
    ProfitMoonshotTrendStrategy,
)
from .session_filtered_pair_carry import SessionFilteredPairCarryStrategy
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
LeadLagSpilloverStrategy = _optional_strategy_class("leadlag_spillover", "LeadLagSpilloverStrategy")
AbnormalReturnContinuationStrategy = _optional_strategy_class(
    "abnormal_return_continuation", "AbnormalReturnContinuationStrategy"
)
LastDayLiquidityRegimeStrategy = _optional_strategy_class(
    "last_day_liquidity_regime", "LastDayLiquidityRegimeStrategy"
)
PairSpreadZScoreStrategy = _optional_strategy_class(
    "pair_spread_zscore", "PairSpreadZScoreStrategy"
)
MicroRangeExpansion1sStrategy = _optional_strategy_class(
    "micro_range_expansion_1s", "MicroRangeExpansion1sStrategy"
)
PerpCrowdingCarryStrategy = _optional_strategy_class(
    "perp_crowding_carry", "PerpCrowdingCarryStrategy"
)
DerivativesFlowSqueezeStrategy = _optional_strategy_class(
    "derivatives_flow_squeeze", "DerivativesFlowSqueezeStrategy"
)
CrossCryptoSlowDiffusionStrategy = _optional_strategy_class(
    "cross_crypto_slow_diffusion", "CrossCryptoSlowDiffusionStrategy"
)
HourlyShockReversionStrategy = _optional_strategy_class(
    "hourly_shock_reversion", "HourlyShockReversionStrategy"
)
TakerFlowExhaustionReversalStrategy = _optional_strategy_class(
    "taker_flow_exhaustion_reversal", "TakerFlowExhaustionReversalStrategy"
)
TimeframePairZScoreReversionStrategy = _optional_strategy_class(
    "timeframe_pair_zscore_reversion", "TimeframePairZScoreReversionStrategy"
)

StrategyClass = type[Strategy]

DEFAULT_STRATEGY_NAME = "RsiStrategy" if RsiStrategy is not None else "MeanReversionStdStrategy"

_RAW_STRATEGY_MAP: dict[str, StrategyClass | None] = {
    "AdaptiveRegimeMomentumStrategy": AdaptiveRegimeMomentumStrategy,
    "Alpha101FormulaStrategy": Alpha101FormulaStrategy,
    "AlphaZooOptunaHybridLiveStrategy": AlphaZooOptunaHybridLiveStrategy,
    "BitcoinBuyHoldStrategy": BitcoinBuyHoldStrategy,
    "CompressionBreakoutContinuationStrategy": CompressionBreakoutContinuationStrategy,
    "CryptoFxAlphaZooStateStrategy": CryptoFxAlphaZooStateStrategy,
    "LagConvergenceStrategy": LagConvergenceStrategy,
    "MeanReversionStdStrategy": MeanReversionStdStrategy,
    "PanicReboundMeanReversionStrategy": PanicReboundMeanReversionStrategy,
    "ProfitMoonshotBreakoutStrategy": ProfitMoonshotBreakoutStrategy,
    "ProfitMoonshotReversionStrategy": ProfitMoonshotReversionStrategy,
    "ProfitMoonshotTrendStrategy": ProfitMoonshotTrendStrategy,
    "RsiStrategy": RsiStrategy,
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
    "PairTradingZScoreStrategy": PairTradingZScoreStrategy,
    "PairSpreadZScoreStrategy": PairSpreadZScoreStrategy,
    "SessionFilteredPairCarryStrategy": SessionFilteredPairCarryStrategy,
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
    "AbnormalReturnContinuationStrategy": AbnormalReturnContinuationStrategy,
    "LastDayLiquidityRegimeStrategy": LastDayLiquidityRegimeStrategy,
    "MicroRangeExpansion1sStrategy": MicroRangeExpansion1sStrategy,
}
if PerpCrowdingCarryStrategy is not None:
    _RAW_STRATEGY_MAP["PerpCrowdingCarryStrategy"] = PerpCrowdingCarryStrategy
if DerivativesFlowSqueezeStrategy is not None:
    _RAW_STRATEGY_MAP["DerivativesFlowSqueezeStrategy"] = DerivativesFlowSqueezeStrategy
if CrossCryptoSlowDiffusionStrategy is not None:
    _RAW_STRATEGY_MAP["CrossCryptoSlowDiffusionStrategy"] = CrossCryptoSlowDiffusionStrategy
if HourlyShockReversionStrategy is not None:
    _RAW_STRATEGY_MAP["HourlyShockReversionStrategy"] = HourlyShockReversionStrategy
if TakerFlowExhaustionReversalStrategy is not None:
    _RAW_STRATEGY_MAP["TakerFlowExhaustionReversalStrategy"] = TakerFlowExhaustionReversalStrategy
if TimeframePairZScoreReversionStrategy is not None:
    _RAW_STRATEGY_MAP["TimeframePairZScoreReversionStrategy"] = TimeframePairZScoreReversionStrategy

_STRATEGY_MAP: dict[str, StrategyClass] = {
    name: cast(StrategyClass, cls) for name, cls in _RAW_STRATEGY_MAP.items() if cls is not None
}

_STRATEGY_TIER_HINTS: dict[str, str] = {
    "AdaptiveRegimeMomentumStrategy": "live_opt_in",
    "Alpha101FormulaStrategy": "research_only",
    "AlphaZooOptunaHybridLiveStrategy": "live_opt_in",
    "BitcoinBuyHoldStrategy": "live_default",
    "CompressionBreakoutContinuationStrategy": "live_opt_in",
    "CryptoFxAlphaZooStateStrategy": "live_opt_in",
    "LagConvergenceStrategy": "live_default",
    "MeanReversionStdStrategy": "live_default",
    "PanicReboundMeanReversionStrategy": "live_opt_in",
    "ProfitMoonshotBreakoutStrategy": "live_opt_in",
    "ProfitMoonshotReversionStrategy": "live_opt_in",
    "ProfitMoonshotTrendStrategy": "live_opt_in",
    "RsiStrategy": "live_default",
    "MovingAverageCrossStrategy": "live_default",
    "PairTradingZScoreStrategy": "live_default",
    "RollingBreakoutStrategy": "live_default",
    "SessionFilteredPairCarryStrategy": "live_opt_in",
    "TopCapTimeSeriesMomentumStrategy": "live_default",
    "VwapReversionStrategy": "live_default",
    "RareEventScoreStrategy": "live_opt_in",
    "PairSpreadZScoreStrategy": "live_opt_in",
    "CompositeTrendStrategy": "live_opt_in",
    "VolCompressionVWAPReversionStrategy": "live_opt_in",
    "VolCompressionVwapReversionStrategy": "live_opt_in",
    "LeadLagSpilloverStrategy": "live_opt_in",
    "AbnormalReturnContinuationStrategy": "live_opt_in",
    "LastDayLiquidityRegimeStrategy": "live_opt_in",
    "PerpCrowdingCarryStrategy": "live_opt_in",
    "DerivativesFlowSqueezeStrategy": "live_opt_in",
    "CrossCryptoSlowDiffusionStrategy": "live_opt_in",
    "HourlyShockReversionStrategy": "live_opt_in",
    "TakerFlowExhaustionReversalStrategy": "live_opt_in",
    "TimeframePairZScoreReversionStrategy": "live_opt_in",
    "RegimeBreakoutCandidateStrategy": "research_only",
    "VolatilityCompressionReversionStrategy": "research_only",
    "MicroRangeExpansion1sStrategy": "research_only",
    # New multi-factor book — research_only until backtest-validated on the
    # data-bearing machine (see docs/COST_REALISM_REMEASUREMENT.md). Promote to
    # live_opt_in only after the cost-realistic walk-forward passes the gates.
    "DiversifiedMultiFactorEnsembleStrategy": "research_only",
    # Alpha research batch (research_only until cost-realistic walk-forward passes
    # on the data PC — see docs/COST_REALISM_REMEASUREMENT.md):
    "VolManagedRiskOverlayStrategy": "research_only",
    "CrossSectionalFundingMomentumCarryStrategy": "research_only",
    # Directional bull/bear basket router: research_only until cost-realistic
    # walk-forward proves that the explicit short sleeve improves bear-regime
    # capture without increasing fold-level drawdown concentration.
    "BullBearRegimeRotationStrategy": "research_only",
    # Alpha-hunt meta-spine batch (2026-07-03 consensus plan): research_only until
    # the data-PC clean walk-forward passes the operationalized promotion rule in
    # .omc/plans/alpha-hunt-consensus-plan.md. Registration and these hints land
    # atomically (live-safety) and are enforced by tests/test_strategy_tier_guard.py.
    "DisagreementGatedEnsembleStrategy": "research_only",
    "CrossSectionalFlowShareRotationStrategy": "research_only",
    "RegimeRouterConfirmedRotationStrategy": "research_only",
}

_STRATEGY_METADATA: dict[str, dict[str, Any]] = {
    name: {
        "name": name,
        "tier": str(_STRATEGY_TIER_HINTS.get(name, "live_default")),
    }
    for name in _STRATEGY_MAP
}

_OPTUNA_TRIAL_OVERRIDES: dict[str, str] = {
    "AdaptiveRegimeMomentumStrategy": "24",
    "Alpha101FormulaStrategy": "24",
    "AlphaZooOptunaHybridLiveStrategy": "12",
    "LagConvergenceStrategy": "24",
    "CompressionBreakoutContinuationStrategy": "20",
    "CryptoFxAlphaZooStateStrategy": "16",
    "MeanReversionStdStrategy": "24",
    "PanicReboundMeanReversionStrategy": "20",
    "ProfitMoonshotBreakoutStrategy": "20",
    "ProfitMoonshotReversionStrategy": "20",
    "ProfitMoonshotTrendStrategy": "20",
    "RsiStrategy": "20",
    "MovingAverageCrossStrategy": "20",
    "PairTradingZScoreStrategy": "32",
    "SessionFilteredPairCarryStrategy": "20",
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
    "AbnormalReturnContinuationStrategy": "20",
    "LastDayLiquidityRegimeStrategy": "24",
    "PerpCrowdingCarryStrategy": "16",
    "DerivativesFlowSqueezeStrategy": "16",
    "CrossCryptoSlowDiffusionStrategy": "16",
    "HourlyShockReversionStrategy": "16",
    "TakerFlowExhaustionReversalStrategy": "16",
    "TimeframePairZScoreReversionStrategy": "16",
    "MicroRangeExpansion1sStrategy": "16",
    "BullBearRegimeRotationStrategy": "16",
    "CrossSectionalFlowShareRotationStrategy": "16",
    "RegimeRouterConfirmedRotationStrategy": "16",
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


def _register_strategy_schema(strategy_name: str, strategy_cls: StrategyClass) -> None:
    if _PARAM_REGISTRY.has_strategy(strategy_name):
        return
    schema = _strategy_schema(strategy_cls)
    if not schema:
        return
    _PARAM_REGISTRY.register(
        strategy_name,
        schema,
        optuna_trials=_resolve_optuna_trial_budget(_OPTUNA_TRIAL_OVERRIDES.get(strategy_name)),
    )


def _sync_param_registry(strategy_map: dict[str, StrategyClass]) -> None:
    for strategy_name, strategy_cls in strategy_map.items():
        _register_strategy_schema(strategy_name, strategy_cls)


_sync_param_registry(_STRATEGY_MAP)


def get_strategy_metadata(strategy_name: str) -> dict[str, Any]:
    """Return strategy metadata (tier and name)."""
    token = str(strategy_name)
    if token in _STRATEGY_METADATA:
        return dict(_STRATEGY_METADATA[token])
    # Glob-discovered strategies are not in _STRATEGY_METADATA (which is built over
    # the curated _STRATEGY_MAP), but an explicit _STRATEGY_TIER_HINTS entry must
    # still be honored so a new drop-in sleeve can pin itself to research_only /
    # live_opt_in without being edited into _STRATEGY_MAP. Fail safe to the hint.
    return {"name": token, "tier": str(_STRATEGY_TIER_HINTS.get(token, "live_default"))}


def get_strategy_tier(strategy_name: str) -> str:
    return str(get_strategy_metadata(strategy_name).get("tier", "live_default"))


# ---------------------------------------------------------------------------
# Plugin auto-discovery (AC8: add ONE template file + reference it by name)
# ---------------------------------------------------------------------------
# A user registers a new strategy by dropping ONE conforming module into this
# package decorated with ``@register("strategy", "ClassName", ...)`` from
# ``lumina_quant.core.plugin_registry`` — no edit to this file is required.
# Discovery imports any not-yet-imported sibling module so its ``@register``
# decorators execute and populate GLOBAL_REGISTRY; resolution then consults the
# registry alongside the curated ``_STRATEGY_MAP``.  Discovery is LAZY (runs on
# the first map/resolve call, never at module import) to avoid the
# strategies-package circular import while ``strategies/__init__`` is still
# initialising.
_PLUGIN_DISCOVERY_DONE: set[str] = set()
_PLUGIN_DISCOVERY_SKIP: frozenset[str] = frozenset({"registry", "plugin_interface"})


def _discover_plugin_strategies() -> None:
    """Import sibling strategy modules so their ``@register`` decorators fire.

    Idempotent and resilient: a malformed dropped module is skipped without
    breaking resolution of the others.  Newly-dropped files are picked up on the
    next call because their module name is not yet in the discovered set.

    The directory is scanned with ``glob`` (a fresh ``scandir`` every call) rather
    than ``pkgutil.iter_modules`` so a file dropped at runtime is seen even after
    the import system's ``FileFinder`` cached an earlier listing; when a genuinely
    new module appears we call ``invalidate_caches`` before importing it.
    """
    package = __package__ or "lumina_quant.strategies"
    pkg_dir = Path(__file__).resolve().parent
    try:
        candidates = {entry.stem for entry in pkg_dir.glob("*.py") if entry.name != "__init__.py"}
    except OSError:
        return
    new_names = candidates - _PLUGIN_DISCOVERY_DONE - _PLUGIN_DISCOVERY_SKIP
    if not new_names:
        return
    # A new module file appeared since the last scan — make the import system
    # aware of it before importing (FileFinder caches directory listings).
    importlib.invalidate_caches()
    for name in sorted(new_names):
        _PLUGIN_DISCOVERY_DONE.add(name)
        if name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{package}.{name}")
        except Exception:
            # A bad drop-in module must not break resolution of the others.
            continue


def _registry_strategy_classes() -> dict[str, StrategyClass]:
    """Operational ``@register``-discovered strategy classes from GLOBAL_REGISTRY.

    Private/test names (leading underscore) are excluded so transient test-only
    registrations never leak into the operational strategy map.
    """
    out: dict[str, StrategyClass] = {}
    for name, cls in GLOBAL_REGISTRY.get_all("strategy").items():
        token = str(name)
        if token.startswith("_"):
            continue
        if isinstance(cls, type) and issubclass(cls, Strategy):
            out[token] = cast(StrategyClass, cls)
    return out


def get_strategy_map() -> dict[str, StrategyClass]:
    _discover_plugin_strategies()
    combined: dict[str, StrategyClass] = dict(_STRATEGY_MAP)
    combined.update(_registry_strategy_classes())
    _sync_param_registry(combined)
    return combined


def get_live_strategy_map(*, include_opt_in: bool = True) -> dict[str, StrategyClass]:
    allowed = {"live_default", "live_opt_in"} if include_opt_in else {"live_default"}
    return {
        name: cls for name, cls in get_strategy_map().items() if get_strategy_tier(name) in allowed
    }


def get_strategy_names(*, include_research_only: bool = True) -> list[str]:
    names = get_strategy_map().keys()
    if include_research_only:
        return sorted(names)
    return sorted(name for name in names if get_strategy_tier(name) != "research_only")


def get_live_strategy_names(*, include_opt_in: bool = True) -> list[str]:
    return sorted(get_live_strategy_map(include_opt_in=include_opt_in).keys())


def resolve_strategy_class(
    name: str | None, default_name: str = DEFAULT_STRATEGY_NAME
) -> StrategyClass:
    strategy_map = get_strategy_map()
    requested = str(name or "").strip()
    if requested in strategy_map:
        return strategy_map[requested]

    fallback = str(default_name).strip()
    if fallback in strategy_map:
        return strategy_map[fallback]
    if DEFAULT_STRATEGY_NAME in strategy_map:
        return strategy_map[DEFAULT_STRATEGY_NAME]
    if strategy_map:
        return next(iter(strategy_map.values()))
    raise ValueError("No strategy classes are available in registry")


def get_strategy_param_schema(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.get_schema(str(strategy_name))


def get_strategy_canonical_param_names(strategy_name: str) -> dict[str, str]:
    return _PARAM_REGISTRY.get_canonical_names(str(strategy_name))


def resolve_strategy_params(
    strategy_name: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _PARAM_REGISTRY.resolve_params(str(strategy_name), overrides or {}, keep_unknown=True)


def get_default_strategy_params(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.default_params(str(strategy_name))


def resolve_optuna_config(
    strategy_name: str, override: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _PARAM_REGISTRY.resolve_optuna_config(str(strategy_name), override or {})


def get_default_optuna_config(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.default_optuna_config(str(strategy_name))


def resolve_grid_config(
    strategy_name: str, override: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _PARAM_REGISTRY.resolve_grid_config(str(strategy_name), override or {})


def get_default_grid_config(strategy_name: str) -> dict[str, Any]:
    return _PARAM_REGISTRY.default_grid_config(str(strategy_name))
