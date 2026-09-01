"""Helpers to load and apply live-selection artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_LIVE_DECISION_PATH = Path(
    "var/reports/exact_window_backtests/followup_status/portfolio_live_readiness_decision_latest.json"
)
SUPPORTED_LIVE_PORTFOLIO_MODES = frozenset(
    {
        "aggressive_realized_mode",
        "hybrid_guarded_mode",
        "legacy_no_highvol_hybrid_mode",
        "retuned_live_portfolio_hybrid_mode",
        "balanced_overlay_mode",
        "defensive_overlay_mode",
        "core_mode",
        "pair_tactical_mode",
        "production_guarded_state_vwap_pair_mode",
        "strict_autoresearch_practical_mode",
        "risk_off_mode",
        "incumbent",
        "incumbent_only",
        "autoresearch_55_45",
        "blend_85_15",
        "static_blend_76_24",
        "production_guarded_portfolio",
        "strict_autoresearch_1x",
        "soft_three_way_regime",
        "three_way_regime",
        "balanced_overlay_80_20",
        "pair_fast_exit",
        "state_vwap_pair",
        "wave2_pair",
        "profit_reboot_adaptive_momentum_mode",
        "profit_reboot_adaptive_momentum_defensive_mode",
        "profit_reboot_adaptive_momentum_short_bias_mode",
        "profit_reboot_panic_rebound_mode",
        "profit_reboot_session_pair_carry_mode",
        "profit_reboot_compression_breakout_mode",
        "profit_moonshot_adaptive_momentum_mode",
        "profit_moonshot_adaptive_momentum_120_mode",
        "profit_moonshot_adaptive_momentum_130_mode",
        "profit_moonshot_adaptive_momentum_140_mode",
        "profit_moonshot_adaptive_momentum_boost_mode",
        "profit_moonshot_adaptive_momentum_governed_mode",
        "profit_moonshot_adaptive_momentum_vol_target_mode",
        "profit_moonshot_adaptive_momentum_vol_target_132_mode",
        "profit_moonshot_adaptive_momentum_asym_dynamic_mode",
        "profit_moonshot_adaptive_momentum_volume_guard_mode",
        "profit_moonshot_momentum_hybrid_return_mode",
        "profit_moonshot_momentum_hybrid_safe_mode",
        "profit_moonshot_momentum_hybrid_core_mode",
        "profit_moonshot_panic_rebound_mode",
        "profit_moonshot_session_pair_carry_mode",
        "profit_moonshot_balanced_mode",
        "profit_moonshot_trend_mode",
        "profit_moonshot_breakout_mode",
        "profit_moonshot_reversion_mode",
        "profit_moonshot_ensemble_mode",
        "derivatives_flow_squeeze_mode",
        "profit_moonshot_derivatives_taker_flow_mode",
        "profit_moonshot_derivatives_taker_flow_sparse_mode",
        "profit_moonshot_leadlag_slow_diffusion_mode",
        "profit_moonshot_leadlag_slow_diffusion_sol_eth_mode",
        "profit_moonshot_leadlag_slow_diffusion_ensemble_mode",
        "profit_moonshot_hourly_shock_reversion_eth_mode",
        "profit_moonshot_hourly_shock_reversion_eth_12h_mode",
        "profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode",
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode",
        "profit_moonshot_filtered_shock_reversion_diversified_mode",
        "profit_moonshot_taker_flow_exhaustion_eth_mode",
        "profit_moonshot_taker_flow_exhaustion_eth_reactive_mode",
        "profit_moonshot_taker_flow_exhaustion_eth_hold_mode",
        "profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode",
        "profit_moonshot_precious_metal_pair_aggressive_mode",
        "artifact_manifest_mode",
    }
)


def normalize_portfolio_mode_reference(reference: str) -> str:
    """Normalize a portfolio-mode reference shared by live and backtest CLIs.

    Live decisions sometimes store the bare mode name while operator-facing
    commands often use ``ArtifactPortfolioModeStrategy[mode]``.  Treat both as
    the same target so backtests can exercise exactly the same portfolio-mode
    runtime that live trading will instantiate.
    """
    token = str(reference or "").strip()
    prefix = "ArtifactPortfolioModeStrategy["
    if token.startswith(prefix) and token.endswith("]"):
        token = token[len(prefix) : -1].strip()
    return token


def resolve_selection_file(selection_file: str = "") -> Path | None:
    token = str(selection_file or "").strip()
    if token:
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Live selection file not found: {path}")
        return path

    root = Path("best_optimized_parameters/live")
    if not root.exists():
        return None
    files = sorted(root.glob("live_selection_*.json"), key=lambda item: item.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def resolve_live_decision_file(decision_file: str = "") -> Path | None:
    token = str(decision_file or "").strip()
    if token:
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Live decision file not found: {path}")
        return path
    path = DEFAULT_LIVE_DECISION_PATH
    if path.exists():
        return path.resolve()
    return None


def supports_live_portfolio_mode(reference: str) -> bool:
    token = normalize_portfolio_mode_reference(reference)
    return token in SUPPORTED_LIVE_PORTFOLIO_MODES or token.startswith("manifest:")


def resolve_portfolio_mode_runtime_config(reference: str) -> dict[str, Any]:
    """Return the canonical runtime config for a supported live portfolio mode.

    This intentionally returns class-free data so both live and backtest entry
    points can wire their own runtime class while sharing the same symbol list
    and strategy params.
    """
    portfolio_mode = normalize_portfolio_mode_reference(reference)
    if not supports_live_portfolio_mode(portfolio_mode):
        raise ValueError(f"Unsupported live portfolio mode: {reference}")

    from lumina_quant.strategies.artifact_portfolio_mode import (
        resolve_portfolio_mode_definition,
    )

    definition = resolve_portfolio_mode_definition(portfolio_mode)
    return {
        "portfolio_mode": portfolio_mode,
        "strategy_name": f"ArtifactPortfolioModeStrategy[{portfolio_mode}]",
        "strategy_params": {"portfolio_mode": portfolio_mode},
        "symbols": list(definition.symbols),
        "cash_weight": float(definition.cash_weight),
    }


def load_selection_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selection payload shape: {path}")
    return payload


def load_live_decision_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid live decision payload shape: {path}")
    return payload


def _normalize_symbol_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _decision_strategy_params(payload: dict) -> dict[str, Any]:
    params = _as_mapping(payload.get("strategy_params"))
    if not params:
        params = _as_mapping(payload.get("params"))
    return params


def _decision_runtime_overrides(payload: dict) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    symbols = _normalize_symbol_list(
        payload.get("symbols") or payload.get("trading_symbols") or payload.get("selected_symbols")
    )
    if symbols:
        overrides["symbols"] = symbols

    params = _decision_strategy_params(payload)
    if params:
        overrides["strategy_params"] = params

    strategy_timeframe = str(
        payload.get("strategy_timeframe") or payload.get("timeframe") or ""
    ).strip()
    if strategy_timeframe:
        overrides["strategy_timeframe"] = strategy_timeframe

    exchange = _as_mapping(payload.get("exchange"))
    live_exchange = _as_mapping(payload.get("live_exchange"))
    exchange_override = {**exchange, **live_exchange}
    leverage = _optional_int(
        payload.get("leverage") or payload.get("live_leverage") or exchange_override.get("leverage")
    )
    if leverage is not None:
        exchange_override["leverage"] = int(leverage)
        overrides["leverage"] = int(leverage)
    if exchange_override:
        overrides["exchange"] = exchange_override

    target_allocation = _optional_float(payload.get("target_allocation"))
    if target_allocation is not None:
        overrides["target_allocation"] = float(target_allocation)

    sizing_mode = str(
        payload.get("target_allocation_mode")
        or payload.get("sizing_mode")
        or _as_mapping(payload.get("live_equivalent_contract")).get("target_allocation_mode")
        or _as_mapping(payload.get("live_equivalent_contract")).get("sizing_mode")
        or ""
    ).strip()
    if sizing_mode:
        overrides["target_allocation_mode"] = sizing_mode

    risk_caps = {
        **_as_mapping(payload.get("risk_caps")),
        **_as_mapping(payload.get("live_risk_caps")),
    }
    for key in (
        "max_order_value",
        "max_order_notional_pct",
        "max_symbol_exposure_pct",
        "max_total_margin_pct",
        "max_total_notional_pct",
    ):
        numeric = _optional_float(payload.get(key))
        if numeric is None:
            numeric = _optional_float(risk_caps.get(key))
        if numeric is not None:
            overrides[key] = float(numeric)

    for source_key, target_key in (
        ("window_seconds", "window_seconds"),
        ("ingest_window_seconds", "ingest_window_seconds"),
        ("decision_cadence_seconds", "decision_cadence_seconds"),
    ):
        numeric = _optional_int(payload.get(source_key))
        if numeric is not None:
            overrides[target_key] = int(numeric)
    return overrides


def infer_strategy_class_name(candidate_name: str) -> str | None:
    token = str(candidate_name or "").strip().lower()
    if not token:
        return None
    if (
        token.startswith("alpha_zoo_integer_leverage_optuna_hybrid")
        or token.startswith("alpha_zoo_optuna_hybrid")
        or token.startswith("hybrid_v3_5_optuna_three_profile_blend")
        or token.startswith("hybrid_v3_6_optuna_three_profile_blend")
    ):
        return "AlphaZooOptunaHybridLiveStrategy"
    if (
        token.startswith("crypto_fx_alpha_zoo")
        or token.startswith("alpha_zoo")
        or token.startswith("profit_moonshot_alpha_zoo")
    ):
        return "CryptoFxAlphaZooStateStrategy"
    if token.startswith("bitcoin_buy_hold"):
        return "BitcoinBuyHoldStrategy"
    if token.startswith("lag_convergence"):
        return "LagConvergenceStrategy"
    if token.startswith("mean_reversion_std"):
        return "MeanReversionStdStrategy"
    if token.startswith("rolling_breakout"):
        return "RollingBreakoutStrategy"
    if token.startswith("topcap_tsmom"):
        return "TopCapTimeSeriesMomentumStrategy"
    if (
        token.startswith("adaptive_regime_momentum")
        or token.startswith("profit_reboot_adaptive_momentum")
        or token.startswith("profit_moonshot_adaptive_momentum")
    ):
        return "AdaptiveRegimeMomentumStrategy"
    if (
        token.startswith("panic_rebound")
        or token.startswith("profit_reboot_panic_rebound")
        or token.startswith("profit_moonshot_panic_rebound")
    ):
        return "PanicReboundMeanReversionStrategy"
    if (
        token.startswith("session_filtered_pair_carry")
        or token.startswith("profit_reboot_session_pair_carry")
        or token.startswith("profit_moonshot_session_pair_carry")
    ):
        return "SessionFilteredPairCarryStrategy"
    if token.startswith("profit_moonshot_perp_crowding") or token.startswith("perp_crowding"):
        return "PerpCrowdingCarryStrategy"
    if (
        token.startswith("dfse_")
        or token.startswith("derivatives_flow_squeeze")
        or token.startswith("profit_moonshot_dfse")
        or token.startswith("profit_moonshot_derivatives")
    ):
        return "DerivativesFlowSqueezeStrategy"
    if token.startswith("profit_moonshot_leadlag_slow_diffusion") or token.startswith(
        "cross_crypto_slow_diffusion"
    ):
        return "CrossCryptoSlowDiffusionStrategy"
    if token.startswith("profit_moonshot_taker_flow_exhaustion") or token.startswith(
        "taker_flow_exhaustion"
    ):
        return "TakerFlowExhaustionReversalStrategy"
    if token.startswith("profit_moonshot_hourly_shock_reversion") or token.startswith(
        "hourly_shock_reversion"
    ):
        return "HourlyShockReversionStrategy"
    if token.startswith("profit_moonshot_precious_metal_pair") or token.startswith(
        "timeframe_pair_zscore_reversion"
    ):
        return "TimeframePairZScoreReversionStrategy"
    if token.startswith("profit_moonshot_trend"):
        return "ProfitMoonshotTrendStrategy"
    if token.startswith("profit_moonshot_breakout"):
        return "ProfitMoonshotBreakoutStrategy"
    if token.startswith("profit_moonshot_reversion"):
        return "ProfitMoonshotReversionStrategy"
    if token.startswith("compression_breakout") or token.startswith(
        "profit_reboot_compression_breakout"
    ):
        return "CompressionBreakoutContinuationStrategy"
    if token.startswith("pair_"):
        return "PairTradingZScoreStrategy"
    if token.startswith("vwap_reversion"):
        return "VwapReversionStrategy"
    if token.startswith("rsi"):
        return "RsiStrategy"
    if token.startswith("moving"):
        return "MovingAverageCrossStrategy"
    return None


def extract_live_decision_config(payload: dict) -> dict:
    decision = str(payload.get("decision") or "").strip().lower()
    reference = str(
        payload.get("selected_mode")
        or payload.get("candidate_mode")
        or payload.get("candidate_key")
        or ""
    ).strip()
    explicit_strategy_name = str(
        payload.get("strategy_name") or payload.get("strategy_class") or ""
    ).strip()
    strategy_name = explicit_strategy_name or infer_strategy_class_name(reference)
    overrides = _decision_runtime_overrides(payload)
    if decision == "keep_incumbent":
        return {
            "decision": decision,
            "reference": reference,
            "target_kind": "incumbent_fallback",
            "strategy_name": strategy_name,
            **overrides,
        }
    if strategy_name:
        return {
            "decision": decision,
            "reference": reference,
            "target_kind": "strategy_class",
            "strategy_name": strategy_name,
            **overrides,
        }
    if reference:
        return {
            "decision": decision,
            "reference": reference,
            "target_kind": "portfolio_mode",
            "strategy_name": None,
            **overrides,
        }
    return {
        "decision": decision,
        "reference": reference,
        "target_kind": "unknown",
        "strategy_name": None,
        **overrides,
    }


def extract_selection_config(payload: dict) -> dict:
    selected = payload.get("selected_candidate") or {}
    selected = selected if isinstance(selected, dict) else {}

    symbols = selected.get("symbols")
    if not isinstance(symbols, list):
        symbols = []
    symbols = [str(item).strip().upper() for item in symbols if str(item).strip()]

    params = selected.get("params")
    if not isinstance(params, dict):
        params = {}

    split = payload.get("split") or {}
    split = split if isinstance(split, dict) else {}
    strategy_timeframe = str(
        split.get("strategy_timeframe") or payload.get("best_timeframe") or ""
    ).strip()
    if not strategy_timeframe:
        strategy_timeframe = None

    return {
        "candidate_name": str(selected.get("name") or "").strip(),
        "symbols": symbols,
        "params": params,
        "strategy_timeframe": strategy_timeframe,
        "base_timeframe": str(payload.get("base_timeframe") or "").strip() or None,
        "mode": str(payload.get("mode") or "").strip() or None,
    }
