"""Typed runtime config access for backtest/live modules."""

from __future__ import annotations

import json
import os
import re
from collections.abc import MutableMapping
from dataclasses import asdict
from typing import ClassVar

from lumina_quant.configuration.loader import load_runtime_config, load_yaml_config
from lumina_quant.configuration.validate import validate_runtime_config
from lumina_quant.live.readiness_policy import (
    effective_startup_reconciliation_hard_fail,
    real_mode_explicitly_enabled,
)

_AUTOSEEDED_DEFAULTS_ENV = "LQ__RUNTIME_AUTOSEEDED_DEFAULTS_JSON"
_AUTOSEEDED_PREEXISTING_KEYS_ENV = "LQ__RUNTIME_AUTOSEEDED_PREEXISTING_KEYS_JSON"


def load_config(config_path: str = "config.yaml") -> dict:
    """Load raw YAML config."""
    return load_yaml_config(config_path=config_path)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _bool_to_env(value: bool) -> str:
    return "1" if bool(value) else "0"


def _runtime_env_defaults(runtime) -> dict[str, str]:
    storage = runtime.storage
    execution = runtime.execution
    backtest = runtime.backtest
    live = runtime.live
    market_window = runtime.market_window

    defaults = {
        "LQ__STORAGE__BACKEND": str(storage.backend),
        "LQ__STORAGE__MARKET_DATA_PARQUET_PATH": str(
            storage.market_data_parquet_path or "data/market_parquet"
        ),
        "LQ__STORAGE__WAL_MAX_BYTES": str(int(getattr(storage, "wal_max_bytes", 268435456))),
        "LQ__STORAGE__WAL_COMPACT_ON_THRESHOLD": _bool_to_env(
            bool(getattr(storage, "wal_compact_on_threshold", True))
        ),
        "LQ__STORAGE__WAL_COMPACTION_INTERVAL_SECONDS": str(
            int(getattr(storage, "wal_compaction_interval_seconds", 3600))
        ),
        "LQ__STORAGE__COLLECTOR_PERIODIC_ENABLED": _bool_to_env(
            bool(getattr(storage, "collector_periodic_enabled", True))
        ),
        "LQ__STORAGE__COLLECTOR_POLL_SECONDS": str(
            int(getattr(storage, "collector_poll_seconds", 2))
        ),
        "LQ__STORAGE__COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS": str(
            int(getattr(storage, "collector_bootstrap_lookback_hours", 24))
        ),
        "LQ__STORAGE__MATERIALIZER_PERIODIC_ENABLED": _bool_to_env(
            bool(getattr(storage, "materializer_periodic_enabled", True))
        ),
        "LQ__STORAGE__MATERIALIZER_POLL_SECONDS": str(
            int(getattr(storage, "materializer_poll_seconds", 5))
        ),
        "LQ__STORAGE__MATERIALIZER_BASE_TIMEFRAME": str(
            getattr(storage, "materializer_base_timeframe", "1s") or "1s"
        ),
        "LQ__STORAGE__MATERIALIZER_REQUIRED_TIMEFRAMES": json.dumps(
            list(getattr(storage, "materializer_required_timeframes", ["1s"]))
        ),
        "LQ_GPU_MODE": str(getattr(execution, "gpu_mode", execution.compute_backend) or "gpu"),
        "LQ_GPU_VRAM_GB": str(float(getattr(execution, "gpu_vram_gb", 0.0))),
        "LQ__EXECUTION__GPU_MODE": str(
            getattr(execution, "gpu_mode", execution.compute_backend) or "gpu"
        ),
        "LQ__EXECUTION__GPU_VRAM_GB": str(float(getattr(execution, "gpu_vram_gb", 0.0))),
        "LQ__BACKTEST__CHUNK_DAYS": str(int(getattr(backtest, "chunk_days", 2))),
        "LQ__BACKTEST__CHUNK_WARMUP_BARS": str(int(getattr(backtest, "chunk_warmup_bars", 0))),
        "LQ__BACKTEST__SKIP_AHEAD_ENABLED": _bool_to_env(
            bool(getattr(backtest, "skip_ahead_enabled", True))
        ),
        "LQ__LIVE__POLL_SECONDS": str(
            int(getattr(live, "poll_seconds", getattr(live, "poll_interval", 20)))
        ),
        "LQ__LIVE__WINDOW_SECONDS": str(int(getattr(live, "window_seconds", 20))),
        "LQ__LIVE__MARKET_DATA_SOURCE": str(getattr(live, "market_data_source", "committed")),
        "LQ__LIVE__ORDER_STATE_SOURCE": str(getattr(live, "order_state_source", "polling")),
        "LQ__LIVE__SHADOW_LIVE_ENABLED": _bool_to_env(
            bool(getattr(live, "shadow_live_enabled", False))
        ),
        "LQ__LIVE__RECONCILIATION_POLL_FALLBACK_ENABLED": _bool_to_env(
            bool(getattr(live, "reconciliation_poll_fallback_enabled", True))
        ),
        "LQ__LIVE__BOOK_TICKER_ENABLED": _bool_to_env(
            bool(getattr(live, "book_ticker_enabled", False))
        ),
        "LQ__LIVE__DEFAULT_ORDER_TYPE": str(getattr(live, "default_order_type", "LMT")),
        "LQ__LIVE__ALLOW_MARKET_ORDERS": _bool_to_env(
            bool(getattr(live, "allow_market_orders", False))
        ),
        "LQ__LIVE__LIMIT_PRICE_MODE": str(getattr(live, "limit_price_mode", "one_tick_worse")),
        "LQ__LIVE__LIMIT_PRICE_OFFSET_TICKS": str(
            int(getattr(live, "limit_price_offset_ticks", 1))
        ),
        "LQ__LIVE__LIMIT_PRICE_TICK_FALLBACK": str(
            float(getattr(live, "limit_price_tick_fallback", 0.0))
        ),
        "LQ__LIVE__LIMIT_TIME_IN_FORCE": str(getattr(live, "limit_time_in_force", "GTC")),
        "LQ__LIVE__PROTECTIVE_ORDER_STYLE": str(getattr(live, "protective_order_style", "limit")),
        "LQ__LIVE__STARTUP_RECONCILIATION_HARD_FAIL": _bool_to_env(
            bool(getattr(live, "startup_reconciliation_hard_fail", False))
        ),
        "LQ__LIVE__MATERIALIZED_STALENESS_THRESHOLD_SECONDS": str(
            int(getattr(live, "materialized_staleness_threshold_seconds", 45))
        ),
        "LQ__LIVE__MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS": str(
            int(getattr(live, "materialized_staleness_alert_cooldown_seconds", 60))
        ),
        "LQ__BACKTEST__POLL_SECONDS": str(int(getattr(backtest, "poll_seconds", 20))),
        "LQ__BACKTEST__WINDOW_SECONDS": str(int(getattr(backtest, "window_seconds", 20))),
        "LQ__BACKTEST__DECISION_CADENCE_SECONDS": str(
            int(getattr(backtest, "decision_cadence_seconds", 20))
        ),
        "LQ__MARKET_WINDOW__PARITY_V2_ENABLED": _bool_to_env(
            bool(getattr(market_window, "parity_v2_enabled", False))
        ),
        "LQ__MARKET_WINDOW__METRICS_LOG_PATH": str(
            getattr(
                market_window,
                "metrics_log_path",
                "logs/live/market_window_metrics.ndjson",
            )
            or "logs/live/market_window_metrics.ndjson"
        ),
    }
    postgres_dsn = str(getattr(storage, "postgres_dsn", "") or "").strip()
    if postgres_dsn:
        defaults[
            str(getattr(storage, "postgres_dsn_env", "LQ_POSTGRES_DSN") or "LQ_POSTGRES_DSN")
        ] = postgres_dsn
    return defaults


def _seed_runtime_env_defaults(runtime, *, environ: MutableMapping[str, str] | None = None) -> None:
    target = os.environ if environ is None else environ
    for key, value in _runtime_env_defaults(runtime).items():
        target.setdefault(key, value)


def _current_config_path() -> str:
    return str(os.getenv("LQ_CONFIG_PATH", "config.yaml") or "config.yaml")


def _runtime_env_without_auto_seeded_defaults(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    effective = dict(source)
    persisted_seeded_defaults, persisted_preexisting_keys = _load_persisted_seed_metadata(source)
    seeded_defaults = dict(persisted_seeded_defaults)
    seeded_defaults.update(_SEEDED_RUNTIME_ENV_DEFAULTS)
    preexisting_keys = set(persisted_preexisting_keys)
    preexisting_keys.update(key for key in seeded_defaults if key in _ENV_BEFORE_RUNTIME_SEED)
    _strip_auto_seeded_runtime_env_defaults(
        effective,
        seeded_defaults=seeded_defaults,
        env_before_seed=preexisting_keys,
    )
    return effective


def _strip_auto_seeded_runtime_env_defaults(
    target: MutableMapping[str, str],
    *,
    seeded_defaults: MutableMapping[str, str] | dict[str, str],
    env_before_seed: MutableMapping[str, str] | dict[str, str] | set[str],
) -> None:
    preexisting_keys = (
        set(env_before_seed) if isinstance(env_before_seed, set) else set(env_before_seed.keys())
    )
    for key, value in seeded_defaults.items():
        if key in preexisting_keys:
            continue
        if target.get(key) == value:
            target.pop(key, None)


def _load_persisted_seed_metadata(
    source: MutableMapping[str, str] | dict[str, str],
) -> tuple[dict[str, str], set[str]]:
    persisted_defaults: dict[str, str] = {}
    defaults_raw = str(source.get(_AUTOSEEDED_DEFAULTS_ENV, "") or "").strip()
    if defaults_raw:
        try:
            parsed_defaults = json.loads(defaults_raw)
        except json.JSONDecodeError:
            parsed_defaults = {}
        if isinstance(parsed_defaults, dict):
            persisted_defaults = {str(key): str(value) for key, value in parsed_defaults.items()}

    persisted_preexisting_keys: set[str] = set()
    preexisting_raw = str(source.get(_AUTOSEEDED_PREEXISTING_KEYS_ENV, "") or "").strip()
    if preexisting_raw:
        try:
            parsed_preexisting = json.loads(preexisting_raw)
        except json.JSONDecodeError:
            parsed_preexisting = []
        if isinstance(parsed_preexisting, list):
            persisted_preexisting_keys = {
                str(value) for value in parsed_preexisting if isinstance(value, str)
            }
    return persisted_defaults, persisted_preexisting_keys


def _persist_seed_metadata(
    target: MutableMapping[str, str],
    *,
    seeded_defaults: dict[str, str],
    env_before_seed: MutableMapping[str, str] | dict[str, str],
) -> None:
    target[_AUTOSEEDED_DEFAULTS_ENV] = json.dumps(seeded_defaults, sort_keys=True)
    preexisting_keys = sorted(key for key in seeded_defaults if key in env_before_seed)
    target[_AUTOSEEDED_PREEXISTING_KEYS_ENV] = json.dumps(preexisting_keys)


def _clear_persisted_seed_metadata(target: MutableMapping[str, str]) -> None:
    target.pop(_AUTOSEEDED_DEFAULTS_ENV, None)
    target.pop(_AUTOSEEDED_PREEXISTING_KEYS_ENV, None)


def _refresh_seeded_runtime_env_defaults(
    runtime, *, environ: MutableMapping[str, str] | None = None
) -> None:
    target = os.environ if environ is None else environ
    next_defaults = _runtime_env_defaults(runtime)
    persisted_seeded_defaults, persisted_preexisting_keys = _load_persisted_seed_metadata(target)
    seeded_defaults = dict(persisted_seeded_defaults)
    seeded_defaults.update(_SEEDED_RUNTIME_ENV_DEFAULTS)
    preexisting_keys = set(persisted_preexisting_keys)
    preexisting_keys.update(key for key in seeded_defaults if key in _ENV_BEFORE_RUNTIME_SEED)
    _strip_auto_seeded_runtime_env_defaults(
        target,
        seeded_defaults=seeded_defaults,
        env_before_seed=preexisting_keys,
    )
    for key, value in next_defaults.items():
        if key in _ENV_BEFORE_RUNTIME_SEED:
            target.setdefault(key, value)
            continue
        target.setdefault(key, value)
    _SEEDED_RUNTIME_ENV_DEFAULTS.clear()
    _SEEDED_RUNTIME_ENV_DEFAULTS.update(next_defaults)
    _persist_seed_metadata(
        target,
        seeded_defaults=next_defaults,
        env_before_seed=_ENV_BEFORE_RUNTIME_SEED,
    )


def _load_runtime_snapshot(*, environ: MutableMapping[str, str] | None = None):
    return load_runtime_config(
        config_path=_current_config_path(),
        env=_runtime_env_without_auto_seeded_defaults(environ),
    )


def load_current_runtime_config():
    global _RUNTIME
    runtime = _load_runtime_snapshot()
    _RUNTIME = runtime
    return runtime


def seed_runtime_env_defaults(
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Explicitly seed runtime-derived environment defaults."""
    global _ENV_BEFORE_RUNTIME_SEED, _RUNTIME
    target = os.environ if environ is None else environ
    _ENV_BEFORE_RUNTIME_SEED = dict(target)
    runtime = _load_runtime_snapshot(environ=environ)
    _refresh_seeded_runtime_env_defaults(runtime, environ=environ)
    _RUNTIME = runtime
    return dict(_SEEDED_RUNTIME_ENV_DEFAULTS)


def current_market_data_runtime_settings() -> dict[str, object]:
    runtime = load_current_runtime_config()
    raw = load_yaml_config(config_path=_current_config_path())
    trading_raw = raw.get("trading", {}) if isinstance(raw.get("trading", {}), dict) else {}
    storage_raw = raw.get("storage", {}) if isinstance(raw.get("storage", {}), dict) else {}
    return {
        "symbols": list(trading_raw.get("symbols") or runtime.trading.symbols),
        "market_data_parquet_path": str(
            storage_raw.get("market_data_parquet_path")
            or runtime.storage.market_data_parquet_path
            or "data/market_parquet"
        ),
        "market_data_exchange": str(
            storage_raw.get("market_data_exchange")
            or runtime.storage.market_data_exchange
            or "binance"
        ),
    }


_PREVIOUS_ENV_BEFORE_RUNTIME_SEED = dict(globals().get("_ENV_BEFORE_RUNTIME_SEED", {}))
_PREVIOUS_SEEDED_RUNTIME_ENV_DEFAULTS = dict(globals().get("_SEEDED_RUNTIME_ENV_DEFAULTS", {}))
_strip_auto_seeded_runtime_env_defaults(
    os.environ,
    seeded_defaults=_PREVIOUS_SEEDED_RUNTIME_ENV_DEFAULTS,
    env_before_seed=_PREVIOUS_ENV_BEFORE_RUNTIME_SEED,
)
_ENV_BEFORE_RUNTIME_SEED = dict(os.environ)
_RUNTIME = None
_SEEDED_RUNTIME_ENV_DEFAULTS: dict[str, str] = {}
_APPLIED_CONFIG_KEYS: dict[type, set[str]] = {}


def _apply_config_values(config_cls: type, values: dict[str, object]) -> None:
    for key, value in values.items():
        setattr(config_cls, key, value)


def _apply_config_values_with_tracking(config_cls: type, values: dict[str, object]) -> None:
    _apply_config_values(config_cls, values)
    _APPLIED_CONFIG_KEYS.setdefault(config_cls, set()).update(values.keys())


def _clear_applied_config_values() -> None:
    for class_name in ("BaseConfig", "BacktestConfig", "LiveConfig", "OptimizationConfig"):
        config_cls = globals().get(class_name)
        if config_cls is None:
            continue
        for key in list(config_cls.__dict__):
            if key.isupper():
                delattr(config_cls, key)
    for config_cls, keys in _APPLIED_CONFIG_KEYS.items():
        for key in keys:
            if key in config_cls.__dict__:
                delattr(config_cls, key)
    _APPLIED_CONFIG_KEYS.clear()


def reset_runtime_config_cache(*, clear_seeded_defaults: bool = False) -> None:
    """Clear cached runtime-derived class snapshots so the next access reloads them lazily."""
    global _RUNTIME
    _RUNTIME = None
    _clear_applied_config_values()
    if clear_seeded_defaults:
        _SEEDED_RUNTIME_ENV_DEFAULTS.clear()
        _clear_persisted_seed_metadata(os.environ)


def clear_runtime_config_views() -> None:
    """Clear materialized config class attributes without touching runtime/env caches."""
    _clear_applied_config_values()


def reload_runtime_config(*, environ: MutableMapping[str, str] | None = None):
    """Refresh runtime-derived class snapshots from the current config/env sources."""
    global _RUNTIME
    runtime = _load_runtime_snapshot(environ=environ)
    _RUNTIME = runtime
    _clear_applied_config_values()
    _apply_config_values_with_tracking(BaseConfig, _base_config_values(runtime))
    _apply_config_values_with_tracking(BacktestConfig, _backtest_config_values(runtime))
    _apply_config_values_with_tracking(LiveConfig, _live_config_values(runtime))
    _apply_config_values_with_tracking(OptimizationConfig, _optimization_config_values(runtime))
    return runtime


def _base_config_values(runtime) -> dict[str, object]:
    trading = runtime.trading
    risk = runtime.risk
    execution = runtime.execution
    storage = runtime.storage
    backtest = runtime.backtest
    market_window = runtime.market_window
    timeframe = str(trading.timeframe)
    timeframes = list(getattr(trading, "timeframes", [])) or [timeframe]
    gpu_mode = str(getattr(execution, "gpu_mode", execution.compute_backend) or "gpu").lower()
    risk_free_annual = float(
        getattr(backtest, "risk_free_annual", getattr(backtest, "risk_free_rate", 0.0)) or 0.0
    )
    return {
        "LOG_LEVEL": runtime.system.log_level,
        "SYMBOLS": list(trading.symbols),
        "TIMEFRAME": timeframe,
        "TIMEFRAMES": timeframes,
        "INITIAL_CAPITAL": float(trading.initial_capital),
        "TARGET_ALLOCATION": float(trading.target_allocation),
        "TARGET_ALLOCATION_MODE": str(
            getattr(trading, "target_allocation_mode", "legacy_notional_cap")
            or "legacy_notional_cap"
        ),
        "MIN_TRADE_QTY": float(trading.min_trade_qty),
        "RISK_PER_TRADE": float(risk.risk_per_trade),
        "MAX_DAILY_LOSS_PCT": float(risk.max_daily_loss_pct),
        "MAX_TOTAL_MARGIN_PCT": float(risk.max_total_margin_pct),
        "MAX_SYMBOL_EXPOSURE_PCT": float(risk.max_symbol_exposure_pct),
        "MAX_ORDER_VALUE": float(risk.max_order_value),
        "MAX_ORDER_NOTIONAL_PCT": float(getattr(risk, "max_order_notional_pct", 0.0)),
        "MAX_TOTAL_NOTIONAL_PCT": float(getattr(risk, "max_total_notional_pct", 0.0)),
        "DEFAULT_STOP_LOSS_PCT": float(risk.default_stop_loss_pct),
        "MAX_INTRADAY_DRAWDOWN_PCT": float(risk.max_intraday_drawdown_pct),
        "MAX_ROLLING_LOSS_PCT_1H": float(risk.max_rolling_loss_pct_1h),
        "FREEZE_NEW_ENTRIES_ON_BREACH": bool(risk.freeze_new_entries_on_breach),
        "AUTO_FLATTEN_ON_BREACH": bool(risk.auto_flatten_on_breach),
        "MAKER_FEE_RATE": float(execution.maker_fee_rate),
        "TAKER_FEE_RATE": float(execution.taker_fee_rate),
        "SPREAD_RATE": float(execution.spread_rate),
        "SLIPPAGE_RATE": float(execution.slippage_rate),
        "FUNDING_RATE_PER_8H": float(execution.funding_rate_per_8h),
        "FUNDING_INTERVAL_HOURS": int(execution.funding_interval_hours),
        "MAINTENANCE_MARGIN_RATE": float(execution.maintenance_margin_rate),
        "LIQUIDATION_BUFFER_RATE": float(execution.liquidation_buffer_rate),
        "GPU_MODE": gpu_mode,
        "GPU_VRAM_GB": float(getattr(execution, "gpu_vram_gb", 0.0)),
        "COMPUTE_BACKEND": gpu_mode,
        "STORAGE_BACKEND": storage.backend,
        "STORAGE_MARKET_DATA_PARQUET_PATH": storage.market_data_parquet_path,
        "MARKET_DATA_PARQUET_PATH": storage.market_data_parquet_path,
        "MARKET_DATA_EXCHANGE": storage.market_data_exchange,
        "POSTGRES_DSN_ENV": str(getattr(storage, "postgres_dsn_env", "LQ_POSTGRES_DSN")),
        "POSTGRES_DSN": str(getattr(storage, "postgres_dsn", "") or ""),
        "STORAGE_EXPORT_CSV": bool(storage.export_csv),
        "WAL_MAX_BYTES": int(getattr(storage, "wal_max_bytes", 268435456)),
        "WAL_COMPACT_ON_THRESHOLD": bool(getattr(storage, "wal_compact_on_threshold", True)),
        "WAL_COMPACTION_INTERVAL_SECONDS": int(
            getattr(storage, "wal_compaction_interval_seconds", 3600)
        ),
        "COLLECTOR_PERIODIC_ENABLED": bool(getattr(storage, "collector_periodic_enabled", True)),
        "COLLECTOR_POLL_SECONDS": int(getattr(storage, "collector_poll_seconds", 2)),
        "COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS": int(
            getattr(storage, "collector_bootstrap_lookback_hours", 24)
        ),
        "MATERIALIZER_PERIODIC_ENABLED": bool(
            getattr(storage, "materializer_periodic_enabled", True)
        ),
        "MATERIALIZER_POLL_SECONDS": int(getattr(storage, "materializer_poll_seconds", 5)),
        "MATERIALIZER_BASE_TIMEFRAME": str(
            getattr(storage, "materializer_base_timeframe", "1s") or "1s"
        ).lower(),
        "MATERIALIZER_REQUIRED_TIMEFRAMES": list(
            getattr(storage, "materializer_required_timeframes", ["1s"])
        )
        or ["1s"],
        "MARKET_WINDOW_PARITY_V2_ENABLED": bool(getattr(market_window, "parity_v2_enabled", False)),
        "MARKET_WINDOW_METRICS_LOG_PATH": str(
            getattr(
                market_window,
                "metrics_log_path",
                "logs/live/market_window_metrics.ndjson",
            )
            or "logs/live/market_window_metrics.ndjson"
        ),
        "RISK_FREE_MODE": str(
            getattr(backtest, "risk_free_mode", "us_treasury_constant") or "us_treasury_constant"
        )
        .strip()
        .lower(),
        "RISK_FREE_TENOR": str(getattr(backtest, "risk_free_tenor", "3m") or "3m").strip().lower(),
        "RISK_FREE_ANNUAL": risk_free_annual,
        "RISK_FREE_SERIES_PATH": str(getattr(backtest, "risk_free_series_path", "") or ""),
        "SORTINO_TARGET_MODE": str(
            getattr(backtest, "sortino_target_mode", "same_as_rf") or "same_as_rf"
        )
        .strip()
        .lower(),
        "SORTINO_TARGET_ANNUAL": float(getattr(backtest, "sortino_target_annual", 0.0) or 0.0),
    }


def _backtest_config_values(runtime) -> dict[str, object]:
    backtest = runtime.backtest
    external = getattr(backtest, "external", None)
    poll_seconds = int(getattr(backtest, "poll_seconds", 20))
    window_seconds = int(getattr(backtest, "window_seconds", 20))
    decision_cadence_seconds = int(getattr(backtest, "decision_cadence_seconds", 20))
    risk_free_annual = float(getattr(backtest, "risk_free_annual", backtest.risk_free_rate))
    return {
        "START_DATE": backtest.start_date,
        "END_DATE": backtest.end_date,
        "MODE": str(getattr(backtest, "mode", "windowed") or "windowed"),
        "DATA_SOURCE": str(getattr(backtest, "data_source", "auto") or "auto").strip().lower(),
        "EXTERNAL_SOURCE_KIND": str(getattr(external, "source_kind", "csv") or "csv")
        .strip()
        .lower(),
        "EXTERNAL_DATA_ROOT": str(getattr(external, "root_path", "") or ""),
        "EXTERNAL_SYMBOL_MAP": dict(getattr(external, "symbol_map", {}) or {}),
        "COMMISSION_RATE": float(backtest.commission_rate),
        "SLIPPAGE_RATE": float(backtest.slippage_rate),
        "ANNUAL_PERIODS": int(backtest.annual_periods),
        "RISK_FREE_RATE": float(backtest.risk_free_rate),
        "RISK_FREE_MODE": str(
            getattr(backtest, "risk_free_mode", "us_treasury_constant") or "us_treasury_constant"
        )
        .strip()
        .lower(),
        "RISK_FREE_TENOR": str(getattr(backtest, "risk_free_tenor", "3m") or "3m").strip().lower(),
        "RISK_FREE_ANNUAL": risk_free_annual,
        "RISK_FREE_SERIES_PATH": str(getattr(backtest, "risk_free_series_path", "") or ""),
        "SORTINO_TARGET_MODE": str(
            getattr(backtest, "sortino_target_mode", "same_as_rf") or "same_as_rf"
        )
        .strip()
        .lower(),
        "SORTINO_TARGET_ANNUAL": float(
            getattr(backtest, "sortino_target_annual", risk_free_annual)
        ),
        "RANDOM_SEED": int(backtest.random_seed),
        "PERSIST_OUTPUT": bool(backtest.persist_output),
        "LEVERAGE": int(backtest.leverage),
        "MARGIN_MODE": str(getattr(backtest, "margin_mode", "isolated") or "isolated")
        .strip()
        .lower(),
        "POLL_SECONDS": poll_seconds,
        "WINDOW_SECONDS": window_seconds,
        "DECISION_CADENCE_SECONDS": decision_cadence_seconds,
        "BACKTEST_POLL_SECONDS": poll_seconds,
        "BACKTEST_WINDOW_SECONDS": window_seconds,
        "BACKTEST_DECISION_SECONDS": decision_cadence_seconds,
        "CHUNK_DAYS": int(getattr(backtest, "chunk_days", 2)),
        "CHUNK_WARMUP_BARS": int(getattr(backtest, "chunk_warmup_bars", 0)),
        "SKIP_AHEAD_ENABLED": bool(getattr(backtest, "skip_ahead_enabled", True)),
    }


def _live_exchange_values(runtime) -> dict[str, str | int]:
    exchange = runtime.live.exchange
    return {
        "driver": str(exchange.driver).lower(),
        "name": str(exchange.name).lower(),
        "market_type": str(exchange.market_type).lower(),
        "position_mode": str(exchange.position_mode).upper(),
        "margin_mode": str(exchange.margin_mode).lower(),
        "leverage": int(exchange.leverage),
    }


def _live_polymarket_values(runtime) -> dict[str, object]:
    polymarket = getattr(runtime.live, "polymarket", None)
    return {
        "POLYMARKET_HOST": str(getattr(polymarket, "host", "") or ""),
        "POLYMARKET_GAMMA_HOST": str(getattr(polymarket, "gamma_host", "") or ""),
        "POLYMARKET_DATA_HOST": str(getattr(polymarket, "data_host", "") or ""),
        "POLYMARKET_MARKET_WS_URL": str(getattr(polymarket, "market_ws_url", "") or ""),
        "POLYMARKET_USER_WS_URL": str(getattr(polymarket, "user_ws_url", "") or ""),
        "POLYMARKET_CHAIN_ID": int(getattr(polymarket, "chain_id", 137) or 137),
        "POLYMARKET_ASSET_IDS": list(getattr(polymarket, "asset_ids", []) or []),
        "POLYMARKET_PRIVATE_KEY_ENV": str(
            getattr(polymarket, "private_key_env", "POLYMARKET_PRIVATE_KEY")
            or "POLYMARKET_PRIVATE_KEY"
        ),
        "POLYMARKET_API_KEY_ENV": str(
            getattr(polymarket, "api_key_env", "POLYMARKET_API_KEY") or "POLYMARKET_API_KEY"
        ),
        "POLYMARKET_API_SECRET_ENV": str(
            getattr(polymarket, "api_secret_env", "POLYMARKET_API_SECRET")
            or "POLYMARKET_API_SECRET"
        ),
        "POLYMARKET_API_PASSPHRASE_ENV": str(
            getattr(polymarket, "api_passphrase_env", "POLYMARKET_API_PASSPHRASE")
            or "POLYMARKET_API_PASSPHRASE"
        ),
        "POLYMARKET_FUNDER": str(getattr(polymarket, "funder", "") or ""),
        "POLYMARKET_SIGNATURE_TYPE": int(getattr(polymarket, "signature_type", 0) or 0),
        "POLYMARKET_ALLOW_REAL_EXECUTION": bool(getattr(polymarket, "allow_real_execution", False)),
    }


def _live_config_values(runtime) -> dict[str, object]:
    live = runtime.live
    external = getattr(live, "external", None)
    exchange = _live_exchange_values(runtime)
    poll_seconds = int(getattr(live, "poll_seconds", live.poll_interval))
    window_seconds = int(getattr(live, "window_seconds", 20))
    values = {
        "BINANCE_API_KEY": live.api_key,
        "BINANCE_SECRET_KEY": live.secret_key,
        "TELEGRAM_BOT_TOKEN": live.telegram_bot_token,
        "TELEGRAM_CHAT_ID": live.telegram_chat_id,
        "MODE": str(live.mode).strip().lower(),
        "MARKET_DATA_SOURCE": str(getattr(live, "market_data_source", "committed")).strip().lower(),
        "ORDER_STATE_SOURCE": str(getattr(live, "order_state_source", "polling")).strip().lower(),
        "EXTERNAL_DATA_SOURCE_KIND": str(getattr(external, "source_kind", "jsonl") or "jsonl")
        .strip()
        .lower(),
        "EXTERNAL_DATA_PATH": str(getattr(external, "path", "") or ""),
        "EXTERNAL_DATA_SCHEMA": str(
            getattr(external, "schema", "market_window_v1") or "market_window_v1"
        )
        .strip()
        .lower(),
        "EXTERNAL_DATA_SYMBOL_MAP": dict(getattr(external, "symbol_map", {}) or {}),
        "EXTERNAL_DATA_POLL_SECONDS": int(getattr(external, "poll_seconds", 2) or 2),
        "EXTERNAL_DATA_ALLOW_STALE_SECONDS": int(
            getattr(external, "allow_stale_seconds", 45) or 45
        ),
        "SHADOW_LIVE_ENABLED": bool(getattr(live, "shadow_live_enabled", False)),
        "RECONCILIATION_POLL_FALLBACK_ENABLED": bool(
            getattr(live, "reconciliation_poll_fallback_enabled", True)
        ),
        "BOOK_TICKER_ENABLED": bool(getattr(live, "book_ticker_enabled", False)),
        "DEFAULT_ORDER_TYPE": str(getattr(live, "default_order_type", "LMT") or "LMT")
        .strip()
        .upper(),
        "ALLOW_MARKET_ORDERS": bool(getattr(live, "allow_market_orders", False)),
        "LIMIT_PRICE_MODE": str(
            getattr(live, "limit_price_mode", "one_tick_worse") or "one_tick_worse"
        )
        .strip()
        .lower(),
        "LIMIT_PRICE_OFFSET_TICKS": int(getattr(live, "limit_price_offset_ticks", 1) or 1),
        "LIMIT_PRICE_TICK_FALLBACK": float(getattr(live, "limit_price_tick_fallback", 0.0) or 0.0),
        "LIMIT_TIME_IN_FORCE": str(getattr(live, "limit_time_in_force", "GTC") or "GTC")
        .strip()
        .upper(),
        "PROTECTIVE_ORDER_STYLE": str(getattr(live, "protective_order_style", "limit") or "limit")
        .strip()
        .lower(),
        "STARTUP_RECONCILIATION_HARD_FAIL": effective_startup_reconciliation_hard_fail(
            mode=str(live.mode),
            configured=bool(getattr(live, "startup_reconciliation_hard_fail", False)),
        ),
        "MAIN_LOOP_ERROR_RETRY_LIMIT": int(getattr(live, "main_loop_error_retry_limit", 3) or 3),
        "MAIN_LOOP_ERROR_WINDOW_SECONDS": int(
            getattr(live, "main_loop_error_window_seconds", 60) or 60
        ),
        "IS_TESTNET": str(live.mode).strip().lower() != "real",
        "REQUIRE_REAL_ENABLE_FLAG": bool(live.require_real_enable_flag),
        "POLL_SECONDS": poll_seconds,
        "POLL_INTERVAL": poll_seconds,
        "LIVE_POLL_SECONDS": poll_seconds,
        "WINDOW_SECONDS": window_seconds,
        "INGEST_WINDOW_SECONDS": window_seconds,
        "DECISION_CADENCE_SECONDS": int(getattr(live, "decision_cadence_seconds", 20)),
        "MATERIALIZED_STALENESS_THRESHOLD_SECONDS": int(
            getattr(live, "materialized_staleness_threshold_seconds", 45)
        ),
        "MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS": int(
            getattr(live, "materialized_staleness_alert_cooldown_seconds", 60)
        ),
        "ORDER_TIMEOUT": int(live.order_timeout),
        "HEARTBEAT_INTERVAL_SEC": int(live.heartbeat_interval_sec),
        "RECONCILIATION_INTERVAL_SEC": int(live.reconciliation_interval_sec),
        "EXCHANGE": exchange,
        "EXCHANGE_ID": str(exchange["name"]),
        "MARKET_TYPE": str(exchange["market_type"]),
        "POSITION_MODE": str(exchange["position_mode"]),
        "MARGIN_MODE": str(exchange["margin_mode"]),
        "LEVERAGE": int(exchange["leverage"]),
        "SYMBOL_LIMITS": dict(live.symbol_limits),
        "MT5_MAGIC": int(live.mt5_magic),
        "MT5_DEVIATION": int(live.mt5_deviation),
        "MT5_BRIDGE_PYTHON": str(getattr(live, "mt5_bridge_python", "") or ""),
        "MT5_BRIDGE_SCRIPT": str(
            getattr(live, "mt5_bridge_script", "scripts/mt5_bridge_worker.py")
            or "scripts/mt5_bridge_worker.py"
        ),
        "MT5_BRIDGE_USE_WSLPATH": bool(getattr(live, "mt5_bridge_use_wslpath", True)),
    }
    values.update(_live_polymarket_values(runtime))
    return values


class _RuntimeConfigMeta(type):
    """Lazily bootstrap runtime-backed config classes on first attribute access."""

    def __getattr__(cls, name: str):
        if not name.startswith("_"):
            reload_runtime_config()
            try:
                return type.__getattribute__(cls, name)
            except AttributeError:
                pass
        raise AttributeError(f"{cls.__name__!s} has no attribute {name!r}")


class BaseConfig(metaclass=_RuntimeConfigMeta):
    """Shared configuration fields used by backtest and live modules."""


class BacktestConfig(BaseConfig):
    """Backtest configuration access."""


class LiveConfig(BaseConfig):
    """Live configuration access with runtime validation helpers."""

    EXCHANGE: ClassVar[dict[str, str | int]]

    @classmethod
    def _as_runtime(cls):
        runtime = load_runtime_config(
            config_path=_current_config_path(),
            env=_runtime_env_without_auto_seeded_defaults(),
        )
        runtime.live.mode = cls.MODE
        runtime.live.market_data_source = str(cls.MARKET_DATA_SOURCE)
        runtime.live.order_state_source = str(cls.ORDER_STATE_SOURCE)
        runtime.live.external.source_kind = str(cls.EXTERNAL_DATA_SOURCE_KIND)
        runtime.live.external.path = str(cls.EXTERNAL_DATA_PATH)
        runtime.live.external.schema = str(cls.EXTERNAL_DATA_SCHEMA)
        runtime.live.external.poll_seconds = int(cls.EXTERNAL_DATA_POLL_SECONDS)
        runtime.live.external.allow_stale_seconds = int(cls.EXTERNAL_DATA_ALLOW_STALE_SECONDS)
        runtime.live.external.symbol_map = dict(cls.EXTERNAL_DATA_SYMBOL_MAP)
        runtime.live.shadow_live_enabled = bool(cls.SHADOW_LIVE_ENABLED)
        runtime.live.reconciliation_poll_fallback_enabled = bool(
            cls.RECONCILIATION_POLL_FALLBACK_ENABLED
        )
        runtime.live.book_ticker_enabled = bool(cls.BOOK_TICKER_ENABLED)
        runtime.live.default_order_type = str(cls.DEFAULT_ORDER_TYPE)
        runtime.live.allow_market_orders = bool(cls.ALLOW_MARKET_ORDERS)
        runtime.live.limit_price_mode = str(cls.LIMIT_PRICE_MODE)
        runtime.live.limit_price_offset_ticks = int(cls.LIMIT_PRICE_OFFSET_TICKS)
        runtime.live.limit_price_tick_fallback = float(cls.LIMIT_PRICE_TICK_FALLBACK)
        runtime.live.limit_time_in_force = str(cls.LIMIT_TIME_IN_FORCE)
        runtime.live.protective_order_style = str(cls.PROTECTIVE_ORDER_STYLE)
        runtime.live.startup_reconciliation_hard_fail = bool(cls.STARTUP_RECONCILIATION_HARD_FAIL)
        runtime.live.main_loop_error_retry_limit = int(cls.MAIN_LOOP_ERROR_RETRY_LIMIT)
        runtime.live.main_loop_error_window_seconds = int(cls.MAIN_LOOP_ERROR_WINDOW_SECONDS)
        runtime.live.require_real_enable_flag = cls.REQUIRE_REAL_ENABLE_FLAG
        runtime.live.materialized_staleness_threshold_seconds = int(
            cls.MATERIALIZED_STALENESS_THRESHOLD_SECONDS
        )
        runtime.live.materialized_staleness_alert_cooldown_seconds = int(
            cls.MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS
        )
        runtime.live.api_key = cls.BINANCE_API_KEY
        runtime.live.secret_key = cls.BINANCE_SECRET_KEY
        runtime.live.exchange.driver = str(cls.EXCHANGE["driver"])
        runtime.live.exchange.name = str(cls.EXCHANGE["name"])
        runtime.live.exchange.market_type = str(cls.EXCHANGE["market_type"])
        runtime.live.exchange.position_mode = str(cls.EXCHANGE["position_mode"])
        runtime.live.exchange.margin_mode = str(cls.EXCHANGE["margin_mode"])
        runtime.live.exchange.leverage = int(cls.EXCHANGE["leverage"])
        runtime.live.mt5_bridge_python = str(cls.MT5_BRIDGE_PYTHON)
        runtime.live.mt5_bridge_script = str(cls.MT5_BRIDGE_SCRIPT)
        runtime.live.mt5_bridge_use_wslpath = bool(cls.MT5_BRIDGE_USE_WSLPATH)
        runtime.live.polymarket.host = str(cls.POLYMARKET_HOST)
        runtime.live.polymarket.gamma_host = str(cls.POLYMARKET_GAMMA_HOST)
        runtime.live.polymarket.data_host = str(cls.POLYMARKET_DATA_HOST)
        runtime.live.polymarket.market_ws_url = str(cls.POLYMARKET_MARKET_WS_URL)
        runtime.live.polymarket.user_ws_url = str(cls.POLYMARKET_USER_WS_URL)
        runtime.live.polymarket.chain_id = int(cls.POLYMARKET_CHAIN_ID)
        runtime.live.polymarket.asset_ids = list(cls.POLYMARKET_ASSET_IDS)
        runtime.live.polymarket.private_key_env = str(cls.POLYMARKET_PRIVATE_KEY_ENV)
        runtime.live.polymarket.api_key_env = str(cls.POLYMARKET_API_KEY_ENV)
        runtime.live.polymarket.api_secret_env = str(cls.POLYMARKET_API_SECRET_ENV)
        runtime.live.polymarket.api_passphrase_env = str(cls.POLYMARKET_API_PASSPHRASE_ENV)
        runtime.live.polymarket.funder = str(cls.POLYMARKET_FUNDER)
        runtime.live.polymarket.signature_type = int(cls.POLYMARKET_SIGNATURE_TYPE)
        runtime.live.polymarket.allow_real_execution = bool(cls.POLYMARKET_ALLOW_REAL_EXECUTION)
        runtime.storage.collector_periodic_enabled = bool(cls.COLLECTOR_PERIODIC_ENABLED)
        runtime.storage.collector_poll_seconds = int(cls.COLLECTOR_POLL_SECONDS)
        runtime.storage.collector_bootstrap_lookback_hours = int(
            cls.COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS
        )
        runtime.storage.materializer_periodic_enabled = bool(cls.MATERIALIZER_PERIODIC_ENABLED)
        runtime.storage.materializer_poll_seconds = int(cls.MATERIALIZER_POLL_SECONDS)
        runtime.storage.materializer_base_timeframe = str(cls.MATERIALIZER_BASE_TIMEFRAME)
        runtime.storage.materializer_required_timeframes = list(
            cls.MATERIALIZER_REQUIRED_TIMEFRAMES
        )
        runtime.market_window.parity_v2_enabled = bool(cls.MARKET_WINDOW_PARITY_V2_ENABLED)
        runtime.market_window.metrics_log_path = str(cls.MARKET_WINDOW_METRICS_LOG_PATH)
        runtime.trading.symbols = list(cls.SYMBOLS)
        runtime.trading.timeframe = str(cls.TIMEFRAME)
        runtime.trading.timeframes = list(getattr(cls, "TIMEFRAMES", [cls.TIMEFRAME]))
        runtime.trading.target_allocation = float(cls.TARGET_ALLOCATION)
        runtime.trading.target_allocation_mode = str(
            getattr(cls, "TARGET_ALLOCATION_MODE", "legacy_notional_cap") or "legacy_notional_cap"
        )
        runtime.risk.max_daily_loss_pct = cls.MAX_DAILY_LOSS_PCT
        runtime.risk.risk_per_trade = float(cls.RISK_PER_TRADE)
        runtime.risk.max_total_margin_pct = float(cls.MAX_TOTAL_MARGIN_PCT)
        runtime.risk.max_symbol_exposure_pct = float(cls.MAX_SYMBOL_EXPOSURE_PCT)
        runtime.risk.max_order_value = float(cls.MAX_ORDER_VALUE)
        runtime.risk.max_order_notional_pct = float(getattr(cls, "MAX_ORDER_NOTIONAL_PCT", 0.0))
        runtime.risk.max_total_notional_pct = float(getattr(cls, "MAX_TOTAL_NOTIONAL_PCT", 0.0))
        return runtime

    @classmethod
    def validate(cls):
        """Validate live config and enforce real-trading safety flag."""
        runtime = cls._as_runtime()
        validate_runtime_config(runtime, for_live=True)

        symbol_re = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
        for symbol in cls.SYMBOLS:
            if not symbol_re.match(symbol):
                raise ValueError(
                    f"Invalid symbol format '{symbol}'. Expected format like BTC/USDT."
                )

        if not real_mode_explicitly_enabled(
            mode=str(cls.MODE),
            require_real_enable_flag=bool(cls.REQUIRE_REAL_ENABLE_FLAG),
            env=os.environ,
        ):
            raise ValueError(
                "Real trading is blocked by default. "
                "Set LUMINA_ENABLE_LIVE_REAL=true to allow live real mode."
            )


class OptimizationConfig(metaclass=_RuntimeConfigMeta):
    """Optimization configuration access."""


def _optimization_config_values(runtime) -> dict[str, object]:
    optimization = runtime.optimization
    return {
        "METHOD": optimization.method,
        "STRATEGY_NAME": optimization.strategy,
        "OPTUNA_CONFIG": dict(optimization.optuna),
        "GRID_CONFIG": dict(optimization.grid),
        "WALK_FORWARD_FOLDS": int(optimization.walk_forward_folds),
        "OVERFIT_PENALTY": float(optimization.overfit_penalty),
        "MAX_WORKERS": int(optimization.max_workers),
        "PERSIST_BEST_PARAMS": bool(optimization.persist_best_params),
        "VALIDATION_DAYS": int(getattr(optimization, "validation_days", 30)),
        "OOS_DAYS": int(getattr(optimization, "oos_days", 30)),
    }


def export_runtime_dict() -> dict:
    """Export the loaded typed runtime as a plain dictionary."""
    return asdict(load_current_runtime_config())
