"""Configuration loader with env overrides and typed coercion."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from lumina_quant.configuration.schema import (
    BacktestExternalConfig,
    BacktestRuntimeConfig,
    DeepLearningRuntimeConfig,
    DataConfig,
    ExecutionConfig,
    LiveExchangeConfig,
    LiveExternalConfig,
    LivePolymarketConfig,
    LiveRuntimeConfig,
    MarketWindowConfig,
    MemoryConfig,
    OptimizationRuntimeConfig,
    PortfolioConfig,
    PromotionGateConfig,
    ResearchConfig,
    RiskConfig,
    RuntimeConfig,
    StorageConfig,
    SystemConfig,
    TradingConfig,
    StrategyQualityConfig,
    ValidationConfig,
    _VALID_DATA_KINDS,
)
from lumina_quant.core.order_policy import canonical_order_type, normalize_limit_price_mode

DEFAULT_MARKET_DATA_PARQUET_PATH = "data/market_parquet"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except TypeError, ValueError:
        return int(default)


def _normalize_timeframe_token(value: Any, default: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return str(default)
    return token


def _as_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_live_source(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"", "committed"}:
        return "committed"
    if token in {"binance_futures", "binance_futures_live", "binance_live", "binance", "live"}:
        return "binance_futures"
    if token in {"external", "custom"}:
        return "external"
    if token in {"polymarket_live", "polymarket"}:
        return "polymarket_live"
    return token


def _normalize_exchange_driver(driver: Any, *, exchange_name: Any = None) -> str:
    token = str(driver or "").strip().lower().replace("-", "_")
    name = str(exchange_name or "").strip().lower()
    if token in {"", "binance_futures", "binance_native", "binance"}:
        return "binance_futures" if name in {"", "binance"} else token
    return token


def _normalize_order_state_source(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"", "polling", "poll"}:
        return "polling"
    if token in {"user_stream", "userstream", "stream"}:
        return "user_stream"
    return token


def _normalize_timeframe_list(value: Any) -> list[str]:
    items: list[Any]
    if isinstance(value, str):
        token = value.strip()
        if token.startswith("[") and token.endswith("]"):
            try:
                parsed = json.loads(token)
            except TypeError, ValueError:
                parsed = [part for part in token.split(",")]
            items = list(parsed) if isinstance(parsed, list) else [token]
        else:
            items = [part for part in token.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []

    normalized: list[str] = []
    for item in items:
        tf = str(item or "").strip().lower()
        if tf:
            normalized.append(tf)
    if not normalized:
        return []

    deduped = list(dict.fromkeys(normalized))
    if "1s" in deduped:
        deduped = ["1s", *[token for token in deduped if token != "1s"]]
    return deduped


def _resolve_materializer_required_timeframes(raw: Any, defaults: list[str]) -> list[str]:
    resolved = _normalize_timeframe_list(raw)
    if not resolved:
        resolved = _normalize_timeframe_list(defaults)
    if not resolved:
        resolved = ["1s"]
    if "1s" not in resolved:
        resolved = ["1s", *resolved]
    elif resolved[0] != "1s":
        resolved = ["1s", *[token for token in resolved if token != "1s"]]
    return list(dict.fromkeys(resolved))


def load_yaml_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config from project root or an absolute path."""
    project_root = Path(__file__).resolve().parents[2]
    raw_path = Path(config_path)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                project_root / config_path,
                Path.cwd() / config_path,
                raw_path,
            ]
        )

    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        tried = ", ".join(str(candidate.absolute()) for candidate in candidates)
        raise FileNotFoundError(f"Configuration file not found. Tried: {tried}")
    with open(path, encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _parse_env_scalar(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if raw.strip().startswith(("[", "{")):
        try:
            return json.loads(raw)
        except TypeError, ValueError:
            return raw
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_nested_value(container: dict[str, Any], path_tokens: list[str], value: Any) -> None:
    cur: dict[str, Any] = container
    for token in path_tokens[:-1]:
        key = token.lower()
        node = cur.get(key)
        if not isinstance(node, dict):
            node = {}
            cur[key] = node
        cur = node
    cur[path_tokens[-1].lower()] = value


def apply_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Apply `LQ__` env overrides onto the config dictionary."""
    merged = dict(data)
    for key, raw_value in env.items():
        if not key.startswith("LQ__"):
            continue
        value = _parse_env_scalar(raw_value)
        tokens = [token for token in key[4:].split("__") if token]
        if not tokens:
            continue
        _set_nested_value(merged, tokens, value)
    return merged


def _coerce_dataclass_kwargs(raw: dict[str, Any], model_cls: type[Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(model_cls)}
    return {key: value for key, value in raw.items() if key in allowed}


def _resolve_storage_path(path_value: str) -> str:
    normalized = str(path_value or "").strip().replace("\\", "/")
    if not normalized:
        return DEFAULT_MARKET_DATA_PARQUET_PATH
    return normalized


def _raw_section(mapped: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapped.get(key, {})
    return value if isinstance(value, dict) else {}


def _extract_runtime_sections(mapped: dict[str, Any]) -> dict[str, dict[str, Any]]:
    live_raw = _raw_section(mapped, "live")
    backtest_raw = _raw_section(mapped, "backtest")
    return {
        "system": _raw_section(mapped, "system"),
        "trading": _raw_section(mapped, "trading"),
        "risk": _raw_section(mapped, "risk"),
        "execution": _raw_section(mapped, "execution"),
        "storage": _raw_section(mapped, "storage"),
        "data": _raw_section(mapped, "data"),
        "deep_learning": _raw_section(mapped, "deep_learning"),
        "strategy_quality": _raw_section(mapped, "strategy_quality"),
        "portfolio": _raw_section(mapped, "portfolio"),
        "backtest": backtest_raw,
        "backtest_external": _raw_section(backtest_raw, "external"),
        "live": live_raw,
        "live_exchange": _raw_section(live_raw, "exchange"),
        "live_external": _raw_section(live_raw, "external"),
        "live_polymarket": _raw_section(live_raw, "polymarket"),
        "optimization": _raw_section(mapped, "optimization"),
        "promotion_gate": _raw_section(mapped, "promotion_gate"),
        "market_window": _raw_section(mapped, "market_window"),
        "memory": _raw_section(mapped, "memory"),
        "validation": _raw_section(mapped, "validation"),
        "research": _raw_section(mapped, "research"),
    }


def _build_live_runtime_config(
    *,
    live_raw: dict[str, Any],
    exchange_raw: dict[str, Any],
    live_external_raw: dict[str, Any],
    live_polymarket_raw: dict[str, Any],
    env: Mapping[str, str],
) -> LiveRuntimeConfig:
    live_kwargs = _coerce_dataclass_kwargs(live_raw, LiveRuntimeConfig)
    for reserved_key in (
        "exchange",
        "external",
        "polymarket",
        "api_key",
        "secret_key",
        "telegram_bot_token",
        "telegram_chat_id",
    ):
        live_kwargs.pop(reserved_key, None)

    live = LiveRuntimeConfig(
        **live_kwargs,
        exchange=LiveExchangeConfig(**_coerce_dataclass_kwargs(exchange_raw, LiveExchangeConfig)),
        external=LiveExternalConfig(
            **_coerce_dataclass_kwargs(live_external_raw, LiveExternalConfig)
        ),
        polymarket=LivePolymarketConfig(
            **_coerce_dataclass_kwargs(live_polymarket_raw, LivePolymarketConfig)
        ),
        api_key=str(env.get("BINANCE_API_KEY") or env.get("EXCHANGE_API_KEY") or ""),
        secret_key=str(env.get("BINANCE_SECRET_KEY") or env.get("EXCHANGE_SECRET_KEY") or ""),
        telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN") or live_raw.get("telegram_bot_token"),
        telegram_chat_id=env.get("TELEGRAM_CHAT_ID") or live_raw.get("telegram_chat_id"),
    )
    if not live.mode:
        live.mode = "paper"
    return live


def _normalize_backend_token(value: Any, *, field_name: str) -> str:
    token = str(value or "").strip().lower().replace("_", "-")
    if token in {"", "auto"}:
        return "auto"
    if token in {"cpu", "gpu"}:
        return token
    if token in {"force-gpu", "forcegpu", "forcedgpu", "forced-gpu"}:
        return "forced-gpu"
    raise ValueError(
        f"{field_name} must be one of: auto, cpu, gpu, forced-gpu. Received: {value!r}"
    )


def _normalize_storage_runtime_section(runtime: RuntimeConfig, storage_raw: dict[str, Any]) -> None:
    runtime.storage.market_data_parquet_path = _resolve_storage_path(
        runtime.storage.market_data_parquet_path
    )
    runtime.storage.wal_max_bytes = max(0, _as_int(runtime.storage.wal_max_bytes, 268435456))
    runtime.storage.wal_compact_on_threshold = _as_bool(
        runtime.storage.wal_compact_on_threshold, True
    )
    runtime.storage.wal_compaction_interval_seconds = max(
        0,
        _as_int(runtime.storage.wal_compaction_interval_seconds, 3600),
    )
    runtime.storage.collector_periodic_enabled = _as_bool(
        getattr(runtime.storage, "collector_periodic_enabled", True),
        True,
    )
    runtime.storage.collector_poll_seconds = _as_int(
        getattr(runtime.storage, "collector_poll_seconds", 2),
        2,
    )
    runtime.storage.collector_bootstrap_lookback_hours = max(
        1,
        _as_int(getattr(runtime.storage, "collector_bootstrap_lookback_hours", 24), 24),
    )
    runtime.storage.materializer_periodic_enabled = _as_bool(
        getattr(runtime.storage, "materializer_periodic_enabled", True),
        True,
    )
    runtime.storage.materializer_poll_seconds = _as_int(
        getattr(runtime.storage, "materializer_poll_seconds", 5),
        5,
    )
    runtime.storage.materializer_base_timeframe = _normalize_timeframe_token(
        getattr(runtime.storage, "materializer_base_timeframe", "1s"),
        "1s",
    )
    explicit_required = (
        isinstance(storage_raw, dict) and "materializer_required_timeframes" in storage_raw
    )
    raw_required = (
        storage_raw.get("materializer_required_timeframes")
        if explicit_required
        else getattr(runtime.storage, "materializer_required_timeframes", [])
    )
    if explicit_required:
        runtime.storage.materializer_required_timeframes = _normalize_timeframe_list(raw_required)
    else:
        runtime.storage.materializer_required_timeframes = (
            _resolve_materializer_required_timeframes(
                raw_required,
                list(StorageConfig().materializer_required_timeframes),
            )
        )


def _normalize_trading_and_risk_runtime_section(runtime: RuntimeConfig) -> None:
    runtime.trading.timeframe = _normalize_timeframe_token(runtime.trading.timeframe, "1m")
    normalized_timeframes: list[str] = []
    raw_timeframes = getattr(runtime.trading, "timeframes", None)
    if isinstance(raw_timeframes, (list, tuple, set)):
        for item in raw_timeframes:
            token = _normalize_timeframe_token(item, "")
            if token:
                normalized_timeframes.append(token)
    if not normalized_timeframes:
        normalized_timeframes = [str(runtime.trading.timeframe)]
    if str(runtime.trading.timeframe) not in normalized_timeframes:
        normalized_timeframes.insert(0, str(runtime.trading.timeframe))
    runtime.trading.timeframes = list(dict.fromkeys(normalized_timeframes))

    runtime.trading.initial_capital = _as_float(runtime.trading.initial_capital, 10000.0)
    runtime.trading.target_allocation = _as_float(runtime.trading.target_allocation, 0.1)
    runtime.trading.target_allocation_mode = (
        str(
            getattr(runtime.trading, "target_allocation_mode", "legacy_notional_cap")
            or "legacy_notional_cap"
        )
        .strip()
        .lower()
    )
    runtime.trading.min_trade_qty = _as_float(runtime.trading.min_trade_qty, 0.001)
    runtime.risk.risk_per_trade = _as_float(runtime.risk.risk_per_trade, 0.005)
    runtime.risk.max_daily_loss_pct = _as_float(runtime.risk.max_daily_loss_pct, 0.03)
    runtime.risk.max_order_notional_pct = _as_float(
        getattr(runtime.risk, "max_order_notional_pct", 0.0),
        0.0,
    )
    runtime.risk.max_total_notional_pct = _as_float(
        getattr(runtime.risk, "max_total_notional_pct", 0.0),
        0.0,
    )


def _normalize_execution_runtime_section(runtime: RuntimeConfig, exec_raw: dict[str, Any]) -> None:
    runtime.execution.slippage_rate = _as_float(runtime.execution.slippage_rate, 0.0005)

    raw_gpu_mode = exec_raw.get("gpu_mode") if isinstance(exec_raw, dict) else None
    raw_compute_backend = exec_raw.get("compute_backend") if isinstance(exec_raw, dict) else None
    normalized_gpu_mode = _normalize_backend_token(
        raw_gpu_mode if str(raw_gpu_mode or "").strip() else runtime.execution.gpu_mode,
        field_name="execution.gpu_mode",
    )
    normalized_compute_backend = _normalize_backend_token(
        (
            raw_compute_backend
            if str(raw_compute_backend or "").strip()
            else runtime.execution.compute_backend
        ),
        field_name="execution.compute_backend",
    )
    requested_gpu_mode = (
        normalized_gpu_mode
        if str(raw_gpu_mode or "").strip()
        else (
            normalized_compute_backend
            if str(raw_compute_backend or "").strip()
            else normalized_gpu_mode
        )
    )
    runtime.execution.gpu_mode = requested_gpu_mode
    runtime.execution.compute_backend = requested_gpu_mode
    runtime.execution.gpu_vram_gb = max(0.0, _as_float(runtime.execution.gpu_vram_gb, 0.0))


def _normalize_live_exchange_runtime_section(runtime: RuntimeConfig) -> None:
    runtime.live.exchange.driver = _normalize_exchange_driver(
        runtime.live.exchange.driver,
        exchange_name=runtime.live.exchange.name,
    )
    runtime.live.exchange.leverage = _as_int(runtime.live.exchange.leverage, 3)


def _normalize_backtest_risk_free_runtime_section(runtime: RuntimeConfig) -> None:
    runtime.backtest.risk_free_mode = (
        str(
            getattr(runtime.backtest, "risk_free_mode", "us_treasury_constant")
            or "us_treasury_constant"
        )
        .strip()
        .lower()
    )
    runtime.backtest.risk_free_tenor = (
        str(getattr(runtime.backtest, "risk_free_tenor", "3m") or "3m").strip().lower()
    )
    runtime.backtest.risk_free_annual = _as_float(
        getattr(runtime.backtest, "risk_free_annual", runtime.backtest.risk_free_rate),
        runtime.backtest.risk_free_rate,
    )
    runtime.backtest.risk_free_series_path = str(
        getattr(runtime.backtest, "risk_free_series_path", "") or ""
    ).strip()
    runtime.backtest.sortino_target_mode = (
        str(getattr(runtime.backtest, "sortino_target_mode", "same_as_rf") or "same_as_rf")
        .strip()
        .lower()
    )
    runtime.backtest.sortino_target_annual = _as_float(
        getattr(runtime.backtest, "sortino_target_annual", runtime.backtest.risk_free_annual),
        runtime.backtest.risk_free_annual,
    )


def _normalize_live_runtime_section(runtime: RuntimeConfig, live_raw: dict[str, Any]) -> None:
    runtime.live.market_data_source = _normalize_live_source(
        getattr(runtime.live, "market_data_source", "committed")
    )
    runtime.live.order_state_source = _normalize_order_state_source(
        getattr(runtime.live, "order_state_source", "polling")
    )
    runtime.live.shadow_live_enabled = _as_bool(
        getattr(runtime.live, "shadow_live_enabled", False),
        False,
    )
    runtime.live.reconciliation_poll_fallback_enabled = _as_bool(
        getattr(runtime.live, "reconciliation_poll_fallback_enabled", True),
        True,
    )
    runtime.live.book_ticker_enabled = _as_bool(
        getattr(runtime.live, "book_ticker_enabled", False),
        False,
    )
    runtime.live.default_order_type = canonical_order_type(
        getattr(runtime.live, "default_order_type", "LMT"),
        default="LMT",
    )
    runtime.live.allow_market_orders = _as_bool(
        getattr(runtime.live, "allow_market_orders", False),
        False,
    )
    runtime.live.limit_price_mode = normalize_limit_price_mode(
        getattr(runtime.live, "limit_price_mode", "one_tick_worse"),
    )
    runtime.live.limit_price_offset_ticks = max(
        0,
        _as_int(getattr(runtime.live, "limit_price_offset_ticks", 1), 1),
    )
    runtime.live.limit_price_tick_fallback = max(
        0.0,
        _as_float(getattr(runtime.live, "limit_price_tick_fallback", 0.0), 0.0),
    )
    runtime.live.limit_time_in_force = (
        str(getattr(runtime.live, "limit_time_in_force", "GTC") or "GTC").strip().upper()
    )
    protective_style = (
        str(getattr(runtime.live, "protective_order_style", "limit") or "limit").strip().lower()
    )
    runtime.live.protective_order_style = (
        protective_style if protective_style in {"limit", "market"} else "limit"
    )
    runtime.live.startup_reconciliation_hard_fail = _as_bool(
        getattr(runtime.live, "startup_reconciliation_hard_fail", False),
        False,
    )
    runtime.live.main_loop_error_retry_limit = max(
        1,
        _as_int(getattr(runtime.live, "main_loop_error_retry_limit", 3), 3),
    )
    runtime.live.main_loop_error_window_seconds = max(
        1,
        _as_int(getattr(runtime.live, "main_loop_error_window_seconds", 60), 60),
    )
    runtime.live.poll_interval = max(1, _as_int(runtime.live.poll_interval, 20))
    live_poll_raw = (
        live_raw.get("live_poll_seconds")
        if isinstance(live_raw, dict) and "live_poll_seconds" in live_raw
        else (
            live_raw.get("poll_seconds")
            if isinstance(live_raw, dict) and "poll_seconds" in live_raw
            else (
                live_raw.get("poll_interval")
                if isinstance(live_raw, dict) and "poll_interval" in live_raw
                else runtime.live.poll_interval
            )
        )
    )
    runtime.live.poll_seconds = max(1, _as_int(live_poll_raw, runtime.live.poll_interval))
    runtime.live.live_poll_seconds = int(runtime.live.poll_seconds)
    runtime.live.poll_interval = int(runtime.live.poll_seconds)
    ingest_window_raw = (
        live_raw.get("ingest_window_seconds")
        if isinstance(live_raw, dict) and "ingest_window_seconds" in live_raw
        else (
            live_raw.get("window_seconds")
            if isinstance(live_raw, dict) and "window_seconds" in live_raw
            else runtime.live.poll_seconds
        )
    )
    runtime.live.window_seconds = max(1, _as_int(ingest_window_raw, runtime.live.poll_seconds))
    runtime.live.ingest_window_seconds = int(runtime.live.window_seconds)
    runtime.live.decision_cadence_seconds = max(
        1,
        _as_int(runtime.live.decision_cadence_seconds, runtime.live.poll_seconds),
    )
    runtime.live.materialized_staleness_threshold_seconds = max(
        1,
        _as_int(getattr(runtime.live, "materialized_staleness_threshold_seconds", 45), 45),
    )
    runtime.live.materialized_staleness_alert_cooldown_seconds = max(
        1,
        _as_int(getattr(runtime.live, "materialized_staleness_alert_cooldown_seconds", 60), 60),
    )
    runtime.live.reconciliation_interval_sec = _as_int(runtime.live.reconciliation_interval_sec, 30)
    runtime.live.external.source_kind = (
        str(getattr(runtime.live.external, "source_kind", "jsonl") or "jsonl").strip().lower()
    )
    runtime.live.external.path = str(getattr(runtime.live.external, "path", "") or "").strip()
    runtime.live.external.poll_seconds = max(
        1,
        _as_int(getattr(runtime.live.external, "poll_seconds", 2), 2),
    )
    runtime.live.external.allow_stale_seconds = max(
        1,
        _as_int(getattr(runtime.live.external, "allow_stale_seconds", 45), 45),
    )
    runtime.live.external.schema = (
        str(getattr(runtime.live.external, "schema", "market_window_v1") or "market_window_v1")
        .strip()
        .lower()
    )
    runtime.live.external.symbol_map = dict(getattr(runtime.live.external, "symbol_map", {}) or {})
    runtime.live.polymarket.host = str(getattr(runtime.live.polymarket, "host", "") or "").strip()
    runtime.live.polymarket.gamma_host = str(
        getattr(runtime.live.polymarket, "gamma_host", "") or ""
    ).strip()
    runtime.live.polymarket.data_host = str(
        getattr(runtime.live.polymarket, "data_host", "") or ""
    ).strip()
    runtime.live.polymarket.market_ws_url = str(
        getattr(runtime.live.polymarket, "market_ws_url", "") or ""
    ).strip()
    runtime.live.polymarket.user_ws_url = str(
        getattr(runtime.live.polymarket, "user_ws_url", "") or ""
    ).strip()
    runtime.live.polymarket.chain_id = max(
        1,
        _as_int(getattr(runtime.live.polymarket, "chain_id", 137), 137),
    )
    runtime.live.polymarket.asset_ids = [
        str(item).strip()
        for item in list(getattr(runtime.live.polymarket, "asset_ids", []) or [])
        if str(item).strip()
    ]
    runtime.live.polymarket.allow_real_execution = _as_bool(
        getattr(runtime.live.polymarket, "allow_real_execution", False),
        False,
    )


def _normalize_backtest_runtime_section(
    runtime: RuntimeConfig, backtest_raw: dict[str, Any]
) -> None:
    runtime.backtest.random_seed = _as_int(runtime.backtest.random_seed, 42)
    runtime.backtest.leverage = _as_int(runtime.backtest.leverage, 3)
    runtime.backtest.margin_mode = (
        str(getattr(runtime.backtest, "margin_mode", "isolated") or "isolated").strip().lower()
    )
    runtime.backtest.data_source = (
        str(getattr(runtime.backtest, "data_source", "auto") or "auto").strip().lower()
    )
    runtime.backtest.external.source_kind = (
        str(getattr(runtime.backtest.external, "source_kind", "csv") or "csv").strip().lower()
    )
    runtime.backtest.external.root_path = str(
        getattr(runtime.backtest.external, "root_path", "") or ""
    ).strip()
    runtime.backtest.external.symbol_map = dict(
        getattr(runtime.backtest.external, "symbol_map", {}) or {}
    )
    backtest_poll_raw = (
        backtest_raw.get("backtest_poll_seconds")
        if isinstance(backtest_raw, dict) and "backtest_poll_seconds" in backtest_raw
        else (
            backtest_raw.get("poll_seconds")
            if isinstance(backtest_raw, dict) and "poll_seconds" in backtest_raw
            else runtime.live.poll_seconds
        )
    )
    runtime.backtest.poll_seconds = max(1, _as_int(backtest_poll_raw, runtime.live.poll_seconds))
    runtime.backtest.backtest_poll_seconds = int(runtime.backtest.poll_seconds)
    backtest_window_raw = (
        backtest_raw.get("backtest_window_seconds")
        if isinstance(backtest_raw, dict) and "backtest_window_seconds" in backtest_raw
        else (
            backtest_raw.get("window_seconds")
            if isinstance(backtest_raw, dict) and "window_seconds" in backtest_raw
            else runtime.live.window_seconds
        )
    )
    runtime.backtest.window_seconds = max(
        1, _as_int(backtest_window_raw, runtime.live.window_seconds)
    )
    runtime.backtest.backtest_window_seconds = int(runtime.backtest.window_seconds)
    backtest_decision_raw = (
        backtest_raw.get("decision_cadence_seconds")
        if isinstance(backtest_raw, dict) and "decision_cadence_seconds" in backtest_raw
        else (
            backtest_raw.get("backtest_decision_seconds")
            if isinstance(backtest_raw, dict) and "backtest_decision_seconds" in backtest_raw
            else (
                backtest_raw.get("decision_seconds")
                if isinstance(backtest_raw, dict) and "decision_seconds" in backtest_raw
                else runtime.live.decision_cadence_seconds
            )
        )
    )
    runtime.backtest.decision_cadence_seconds = max(
        1,
        _as_int(backtest_decision_raw, runtime.live.decision_cadence_seconds),
    )
    backtest_mode = str(getattr(runtime.backtest, "mode", "windowed") or "windowed").strip().lower()
    if backtest_mode not in {"windowed", "legacy_batch", "legacy_1s"}:
        backtest_mode = "windowed"
    runtime.backtest.mode = backtest_mode
    runtime.backtest.backtest_decision_seconds = int(runtime.backtest.decision_cadence_seconds)
    runtime.backtest.chunk_days = max(1, _as_int(runtime.backtest.chunk_days, 2))
    runtime.backtest.chunk_warmup_bars = max(0, _as_int(runtime.backtest.chunk_warmup_bars, 0))
    runtime.backtest.skip_ahead_enabled = _as_bool(runtime.backtest.skip_ahead_enabled, True)


def _normalize_optimization_runtime_section(runtime: RuntimeConfig) -> None:
    runtime.optimization.walk_forward_folds = _as_int(runtime.optimization.walk_forward_folds, 3)
    runtime.optimization.overfit_penalty = _as_float(runtime.optimization.overfit_penalty, 0.5)
    runtime.optimization.max_workers = _as_int(runtime.optimization.max_workers, 1)
    runtime.optimization.persist_best_params = _as_bool(
        runtime.optimization.persist_best_params, False
    )
    runtime.optimization.validation_days = max(
        0,
        _as_int(getattr(runtime.optimization, "validation_days", 30), 30),
    )
    runtime.optimization.oos_days = max(
        1,
        _as_int(getattr(runtime.optimization, "oos_days", 30), 30),
    )


def _normalize_market_window_runtime_section(
    runtime: RuntimeConfig,
    live_raw: dict[str, Any],
    backtest_raw: dict[str, Any],
    market_window_raw: dict[str, Any],
) -> None:
    deprecated_live_parity = (
        live_raw.get("market_window_parity_v2_enabled")
        if isinstance(live_raw, dict) and "market_window_parity_v2_enabled" in live_raw
        else None
    )
    deprecated_backtest_parity = (
        backtest_raw.get("market_window_parity_v2_enabled")
        if isinstance(backtest_raw, dict) and "market_window_parity_v2_enabled" in backtest_raw
        else None
    )
    runtime.live.market_window_parity_v2_enabled = _as_bool_or_none(deprecated_live_parity)
    runtime.backtest.market_window_parity_v2_enabled = _as_bool_or_none(deprecated_backtest_parity)

    parity_flag_explicit = (
        isinstance(market_window_raw, dict) and "parity_v2_enabled" in market_window_raw
    )
    if parity_flag_explicit:
        runtime.market_window.parity_v2_enabled = _as_bool(
            runtime.market_window.parity_v2_enabled, False
        )
    else:
        for candidate in (
            runtime.live.market_window_parity_v2_enabled,
            runtime.backtest.market_window_parity_v2_enabled,
        ):
            if candidate is not None:
                runtime.market_window.parity_v2_enabled = bool(candidate)
                break
        else:
            runtime.market_window.parity_v2_enabled = _as_bool(
                runtime.market_window.parity_v2_enabled,
                False,
            )
    runtime.market_window.metrics_log_path = str(
        getattr(runtime.market_window, "metrics_log_path", "")
        or "logs/live/market_window_metrics.ndjson"
    )


def _normalize_data_runtime_section(runtime: RuntimeConfig) -> None:
    raw_kinds = getattr(runtime.data, "kinds", None)
    if isinstance(raw_kinds, (list, tuple)):
        normalized: list[str] = []
        for item in raw_kinds:
            token = str(item or "").strip().lower()
            if not token:
                continue  # skip blank entries
            if token not in _VALID_DATA_KINDS:
                raise ValueError(
                    f"Invalid data.kinds token: {token!r}. "
                    f"Valid values: {sorted(_VALID_DATA_KINDS)}"
                )
            normalized.append(token)
        runtime.data.kinds = list(dict.fromkeys(normalized)) or list(DataConfig().kinds)
    else:
        runtime.data.kinds = list(DataConfig().kinds)
    runtime.data.tick_path = str(getattr(runtime.data, "tick_path", "") or "").strip()


def _normalize_deep_learning_model_list(value: Any) -> list[str]:
    supported = {"fits": "FITS", "cyclenet": "CycleNet", "cmamba": "CMamba", "patchtst": "PatchTST"}
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(part).strip() for part in value]
    else:
        raw = []

    models: list[str] = []
    for item in raw:
        model = supported.get(item.lower())
        if model is not None and model not in models:
            models.append(model)
    return models or list(DeepLearningRuntimeConfig().models)


def _normalize_deep_learning_runtime_section(runtime: RuntimeConfig) -> None:
    cfg = runtime.deep_learning
    cfg.enabled = _as_bool(getattr(cfg, "enabled", False), False)
    cfg.repo_path = str(getattr(cfg, "repo_path", "") or "../../DeepLearning").strip()
    cfg.config_path = str(getattr(cfg, "config_path", "") or "configs/config.yaml").strip()
    cfg.test_config_path = str(
        getattr(cfg, "test_config_path", "") or "configs/config_test.yaml"
    ).strip()
    cfg.dataset_path = str(getattr(cfg, "dataset_path", "") or "").strip()
    cfg.feature_matrix_path = str(
        getattr(cfg, "feature_matrix_path", "") or "var/data/deeplearning/features"
    ).strip()
    cfg.prediction_path = str(
        getattr(cfg, "prediction_path", "") or "var/data/deeplearning/predictions"
    ).strip()
    cfg.manifest_path = str(
        getattr(cfg, "manifest_path", "") or "var/data/deeplearning/pipeline_manifest.json"
    ).strip()
    cfg.training_state_path = str(
        getattr(cfg, "training_state_path", "") or "var/data/deeplearning/training_state.json"
    ).strip()
    cfg.metrics_path = str(
        getattr(cfg, "metrics_path", "") or "var/data/deeplearning/metrics"
    ).strip()
    feature_format = str(getattr(cfg, "feature_format", "") or "parquet").strip().lower()
    cfg.feature_format = feature_format.lstrip(".") if feature_format else "parquet"
    if cfg.feature_format not in {"csv", "parquet"}:
        cfg.feature_format = "parquet"

    cfg.models = _normalize_deep_learning_model_list(getattr(cfg, "models", None))
    raw_ensemble_models = getattr(cfg, "ensemble_models", None)
    cfg.ensemble_models = (
        _normalize_deep_learning_model_list(raw_ensemble_models)
        if raw_ensemble_models
        else list(cfg.models)
    )
    cfg.freq = _normalize_timeframe_token(getattr(cfg, "freq", ""), "")
    cfg.target_suffix = str(getattr(cfg, "target_suffix", "") or "close").strip().lower()
    cfg.seq_len = max(1, _as_int(getattr(cfg, "seq_len", 120), 120))
    cfg.pred_len = max(1, _as_int(getattr(cfg, "pred_len", 30), 30))
    cfg.total_pred = max(cfg.pred_len, _as_int(getattr(cfg, "total_pred", 120), 120))
    cfg.gpus = str(getattr(cfg, "gpus", "[]") or "[]").strip()
    cfg.hpo_enabled = _as_bool(getattr(cfg, "hpo_enabled", False), False)
    cfg.hpo_trials = max(1, _as_int(getattr(cfg, "hpo_trials", 20), 20))
    cfg.gpu_parallel_jobs = max(1, _as_int(getattr(cfg, "gpu_parallel_jobs", 1), 1))
    cfg.gpu_parallel_min_free_vram_gb = max(
        0.0,
        _as_float(getattr(cfg, "gpu_parallel_min_free_vram_gb", 2.0), 2.0),
    )
    cfg.gpu_parallel_max_utilization_pct = max(
        1.0,
        min(
            100.0,
            _as_float(getattr(cfg, "gpu_parallel_max_utilization_pct", 85.0), 85.0),
        ),
    )
    cfg.hpo_top_trial_fraction = max(
        0.01,
        min(1.0, _as_float(getattr(cfg, "hpo_top_trial_fraction", 0.10), 0.10)),
    )
    cfg.hpo_min_top_trials = max(
        1,
        _as_int(getattr(cfg, "hpo_min_top_trials", 5), 5),
    )
    cfg.hpo_max_train_val_gap = max(
        0.0,
        _as_float(getattr(cfg, "hpo_max_train_val_gap", 0.15), 0.15),
    )
    cfg.hpo_boundary_tolerance = max(
        0.0,
        min(0.50, _as_float(getattr(cfg, "hpo_boundary_tolerance", 0.02), 0.02)),
    )
    cfg.chronological_sanity_check = _as_bool(
        getattr(cfg, "chronological_sanity_check", True),
        True,
    )
    cfg.chronological_sanity_val_fraction = max(
        0.01,
        min(
            0.90,
            _as_float(getattr(cfg, "chronological_sanity_val_fraction", 0.20), 0.20),
        ),
    )
    cfg.chronological_sanity_max_relative_drop = max(
        0.0,
        _as_float(
            getattr(cfg, "chronological_sanity_max_relative_drop", 0.25),
            0.25,
        ),
    )
    cfg.partial_hpo_trials = max(1, _as_int(getattr(cfg, "partial_hpo_trials", 5), 5))
    cfg.epochs = max(0, _as_int(getattr(cfg, "epochs", 0), 0))
    cfg.epochs_per_trial = max(0, _as_int(getattr(cfg, "epochs_per_trial", 0), 0))
    cfg.force_train = _as_bool(getattr(cfg, "force_train", False), False)
    cfg.resume = _as_bool(getattr(cfg, "resume", True), True)
    cfg.run_test_after_fit = _as_bool(getattr(cfg, "run_test_after_fit", False), False)
    cfg.no_upload = _as_bool(getattr(cfg, "no_upload", True), True)
    cfg.strict_source_models = _as_bool(getattr(cfg, "strict_source_models", True), True)
    cfg.min_model_coverage = max(1, min(len(cfg.models), _as_int(cfg.min_model_coverage, 2)))
    cfg.entry_threshold_bps = max(0.0, _as_float(cfg.entry_threshold_bps, 10.0))
    cfg.exit_threshold_bps = max(0.0, _as_float(cfg.exit_threshold_bps, 2.0))
    cfg.min_model_agreement = max(0.0, min(1.0, _as_float(cfg.min_model_agreement, 0.75)))
    cfg.max_dispersion_bps = max(0.0, _as_float(cfg.max_dispersion_bps, 80.0))
    cfg.min_confidence = max(0.0, min(1.0, _as_float(cfg.min_confidence, 0.0)))
    cfg.target_allocation = max(0.0, _as_float(cfg.target_allocation, 0.05))
    cfg.stop_loss_pct = max(0.0, _as_float(cfg.stop_loss_pct, 0.0))
    cfg.take_profit_pct = max(0.0, _as_float(cfg.take_profit_pct, 0.0))


def _normalize_strategy_quality_runtime_section(runtime: RuntimeConfig) -> None:
    cfg = runtime.strategy_quality
    cfg.enabled = _as_bool(getattr(cfg, "enabled", False), False)
    cfg.edge_gate_enabled = _as_bool(getattr(cfg, "edge_gate_enabled", True), True)
    cfg.min_expected_edge_bps = max(0.0, _as_float(cfg.min_expected_edge_bps, 8.0))
    cfg.edge_cost_buffer_bps = max(0.0, _as_float(cfg.edge_cost_buffer_bps, 3.0))
    cfg.allow_unknown_edge = _as_bool(getattr(cfg, "allow_unknown_edge", True), True)
    cfg.regime_router_enabled = _as_bool(getattr(cfg, "regime_router_enabled", True), True)
    cfg.regime_lookback_bars = max(5, _as_int(cfg.regime_lookback_bars, 60))
    cfg.trend_return_bps = max(0.0, _as_float(cfg.trend_return_bps, 35.0))
    cfg.panic_return_bps = max(0.0, _as_float(cfg.panic_return_bps, 120.0))
    cfg.low_liquidity_volume_ratio = max(
        0.0,
        min(1.0, _as_float(cfg.low_liquidity_volume_ratio, 0.20)),
    )
    cfg.position_sizing_enabled = _as_bool(
        getattr(cfg, "position_sizing_enabled", True),
        True,
    )
    cfg.target_vol_per_bar = max(0.0, _as_float(cfg.target_vol_per_bar, 0.006))
    cfg.min_position_scale = max(0.0, min(1.0, _as_float(cfg.min_position_scale, 0.20)))
    cfg.max_position_scale = max(
        cfg.min_position_scale,
        min(1.0, _as_float(cfg.max_position_scale, 1.0)),
    )
    cfg.turnover_budget_enabled = _as_bool(
        getattr(cfg, "turnover_budget_enabled", True),
        True,
    )
    cfg.max_daily_turnover_pct = max(0.0, _as_float(cfg.max_daily_turnover_pct, 3.0))
    cfg.exit_overlay_enabled = _as_bool(getattr(cfg, "exit_overlay_enabled", True), True)
    cfg.atr_window_bars = max(2, _as_int(cfg.atr_window_bars, 20))
    cfg.atr_stop_mult = max(0.0, _as_float(cfg.atr_stop_mult, 2.0))
    cfg.atr_take_profit_mult = max(0.0, _as_float(cfg.atr_take_profit_mult, 3.0))
    cfg.trailing_atr_mult = max(0.0, _as_float(cfg.trailing_atr_mult, 1.5))
    cfg.min_stop_bps = max(0.0, _as_float(cfg.min_stop_bps, 20.0))
    cfg.max_stop_bps = max(cfg.min_stop_bps, _as_float(cfg.max_stop_bps, 500.0))
    cfg.strategy_health_enabled = _as_bool(
        getattr(cfg, "strategy_health_enabled", True),
        True,
    )
    cfg.health_window_trades = max(1, _as_int(cfg.health_window_trades, 20))
    cfg.min_health_scale = max(0.0, min(1.0, _as_float(cfg.min_health_scale, 0.25)))
    cfg.loss_cooldown_bars = max(0, _as_int(cfg.loss_cooldown_bars, 12))
    cfg.profit_moonshot_conflict_cooldown_bars = max(
        0,
        _as_int(cfg.profit_moonshot_conflict_cooldown_bars, 3),
    )
    cfg.pair_min_correlation = max(0.0, min(1.0, _as_float(cfg.pair_min_correlation, 0.35)))


_VALID_ALLOCATION_METHODS: frozenset[str] = frozenset(
    {"legacy", "erc", "max_diversification", "mean_variance", "hrp"}
)


def _normalize_portfolio_runtime_section(runtime: RuntimeConfig) -> None:
    cfg = runtime.portfolio
    method = str(getattr(cfg, "allocation_method", "legacy") or "legacy").strip().lower()
    cfg.allocation_method = method if method in _VALID_ALLOCATION_METHODS else "legacy"
    cfg.erc_max_iter = max(1, _as_int(getattr(cfg, "erc_max_iter", 10000), 10000))
    cfg.erc_tol = max(0.0, _as_float(getattr(cfg, "erc_tol", 1e-10), 1e-10))
    cfg.mv_risk_aversion = max(0.0, _as_float(getattr(cfg, "mv_risk_aversion", 5.0), 5.0))
    cfg.pgd_max_iter = max(1, _as_int(getattr(cfg, "pgd_max_iter", 500), 500))
    cfg.pgd_step = max(0.0, _as_float(getattr(cfg, "pgd_step", 0.05), 0.05))
    cfg.hrp_corr_threshold = min(
        1.0, max(0.0, _as_float(getattr(cfg, "hrp_corr_threshold", 0.60), 0.60))
    )
    cfg.cov_window = max(0, _as_int(getattr(cfg, "cov_window", 0), 0))


def _normalize_research_runtime_section(runtime: RuntimeConfig) -> None:
    runtime.research.execution_attribution_enabled = _as_bool(
        getattr(runtime.research, "execution_attribution_enabled", False),
        False,
    )


def _normalize_promotion_gate_runtime_section(runtime: RuntimeConfig) -> None:
    runtime.promotion_gate.days = _as_int(runtime.promotion_gate.days, 14)
    runtime.promotion_gate.max_order_rejects = _as_int(runtime.promotion_gate.max_order_rejects, 0)
    runtime.promotion_gate.max_order_timeouts = _as_int(
        runtime.promotion_gate.max_order_timeouts, 0
    )
    runtime.promotion_gate.max_reconciliation_alerts = _as_int(
        runtime.promotion_gate.max_reconciliation_alerts,
        0,
    )
    runtime.promotion_gate.max_critical_errors = _as_int(
        runtime.promotion_gate.max_critical_errors, 0
    )
    runtime.promotion_gate.require_alpha_card = _as_bool(
        runtime.promotion_gate.require_alpha_card, False
    )

    normalized_profiles: dict[str, dict[str, Any]] = {}
    for name, profile in dict(runtime.promotion_gate.strategy_profiles or {}).items():
        if not isinstance(profile, dict):
            continue
        key = str(name).strip()
        if not key:
            continue
        normalized_profiles[key] = dict(profile)
    runtime.promotion_gate.strategy_profiles = normalized_profiles


def _normalize_runtime_config(
    runtime: RuntimeConfig,
    *,
    sections: dict[str, dict[str, Any]],
) -> None:
    storage_raw = sections["storage"]
    exec_raw = sections["execution"]
    live_raw = sections["live"]
    backtest_raw = sections["backtest"]
    market_window_raw = sections["market_window"]

    _normalize_storage_runtime_section(runtime, storage_raw=storage_raw)
    _normalize_data_runtime_section(runtime)
    _normalize_deep_learning_runtime_section(runtime)
    _normalize_strategy_quality_runtime_section(runtime)
    _normalize_portfolio_runtime_section(runtime)
    _normalize_trading_and_risk_runtime_section(runtime)
    _normalize_execution_runtime_section(runtime, exec_raw=exec_raw)
    _normalize_live_exchange_runtime_section(runtime)
    _normalize_backtest_risk_free_runtime_section(runtime)
    _normalize_live_runtime_section(runtime, live_raw=live_raw)
    _normalize_backtest_runtime_section(runtime, backtest_raw=backtest_raw)
    _normalize_optimization_runtime_section(runtime)
    _normalize_market_window_runtime_section(
        runtime,
        live_raw=live_raw,
        backtest_raw=backtest_raw,
        market_window_raw=market_window_raw,
    )
    _normalize_promotion_gate_runtime_section(runtime)
    _normalize_research_runtime_section(runtime)


def _build_runtime_config_tree(
    *,
    sections: dict[str, dict[str, Any]],
    env: Mapping[str, str],
) -> RuntimeConfig:
    promotion_kwargs = _coerce_dataclass_kwargs(sections["promotion_gate"], PromotionGateConfig)
    strategy_profiles = promotion_kwargs.pop("strategy_profiles", {})

    return RuntimeConfig(
        system=SystemConfig(**_coerce_dataclass_kwargs(sections["system"], SystemConfig)),
        trading=TradingConfig(**_coerce_dataclass_kwargs(sections["trading"], TradingConfig)),
        risk=RiskConfig(**_coerce_dataclass_kwargs(sections["risk"], RiskConfig)),
        execution=ExecutionConfig(
            **_coerce_dataclass_kwargs(sections["execution"], ExecutionConfig)
        ),
        storage=StorageConfig(**_coerce_dataclass_kwargs(sections["storage"], StorageConfig)),
        data=DataConfig(**_coerce_dataclass_kwargs(sections["data"], DataConfig)),
        deep_learning=DeepLearningRuntimeConfig(
            **_coerce_dataclass_kwargs(sections["deep_learning"], DeepLearningRuntimeConfig)
        ),
        strategy_quality=StrategyQualityConfig(
            **_coerce_dataclass_kwargs(sections["strategy_quality"], StrategyQualityConfig)
        ),
        portfolio=PortfolioConfig(
            **_coerce_dataclass_kwargs(sections["portfolio"], PortfolioConfig)
        ),
        backtest=BacktestRuntimeConfig(
            **{
                **_coerce_dataclass_kwargs(sections["backtest"], BacktestRuntimeConfig),
                "external": BacktestExternalConfig(
                    **_coerce_dataclass_kwargs(
                        sections["backtest_external"],
                        BacktestExternalConfig,
                    )
                ),
            }
        ),
        live=_build_live_runtime_config(
            live_raw=sections["live"],
            exchange_raw=sections["live_exchange"],
            live_external_raw=sections["live_external"],
            live_polymarket_raw=sections["live_polymarket"],
            env=env,
        ),
        optimization=OptimizationRuntimeConfig(
            **_coerce_dataclass_kwargs(
                sections["optimization"],
                OptimizationRuntimeConfig,
            )
        ),
        market_window=MarketWindowConfig(
            **_coerce_dataclass_kwargs(sections["market_window"], MarketWindowConfig)
        ),
        memory=MemoryConfig(**_coerce_dataclass_kwargs(sections["memory"], MemoryConfig)),
        validation=ValidationConfig(
            **_coerce_dataclass_kwargs(sections["validation"], ValidationConfig)
        ),
        research=ResearchConfig(
            **_coerce_dataclass_kwargs(sections["research"], ResearchConfig)
        ),
        promotion_gate=PromotionGateConfig(
            **promotion_kwargs,
            strategy_profiles=(
                dict(strategy_profiles) if isinstance(strategy_profiles, dict) else {}
            ),
        ),
    )


def build_runtime_config(data: dict[str, Any], env: Mapping[str, str]) -> RuntimeConfig:
    """Build a strongly typed runtime config from raw dict + environment."""
    mapped = apply_env_overrides(data, env)
    sections = _extract_runtime_sections(mapped)
    runtime = _build_runtime_config_tree(sections=sections, env=env)
    _normalize_runtime_config(runtime, sections=sections)
    return runtime


def load_runtime_config(
    config_path: str = "config.yaml", env: Mapping[str, str] | None = None
) -> RuntimeConfig:
    """Load `.env`, read YAML, apply overrides, and produce typed config."""
    effective_env: Mapping[str, str]
    if env is None:
        dotenv_path = Path(config_path).expanduser().resolve().parent / ".env"
        merged_env = dict(os.environ)
        if dotenv_path.exists():
            for key, value in dotenv_values(dotenv_path).items():
                if key and key not in merged_env:
                    merged_env[key] = "" if value is None else str(value)
        effective_env = merged_env
    else:
        effective_env = env
    raw = load_yaml_config(config_path=config_path)
    return build_runtime_config(raw, effective_env)
