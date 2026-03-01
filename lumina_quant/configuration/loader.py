"""Configuration loader with env overrides and typed coercion."""

import json
import os
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from lumina_quant.configuration.schema import (
    BacktestRuntimeConfig,
    ExecutionConfig,
    LiveExchangeConfig,
    LiveRuntimeConfig,
    OptimizationRuntimeConfig,
    PromotionGateConfig,
    RiskConfig,
    RuntimeConfig,
    StorageConfig,
    SystemConfig,
    TradingConfig,
)

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
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_timeframe_token(value: Any, default: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return str(default)
    return token


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
        except (TypeError, ValueError):
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
    """Apply `LQ_` env overrides onto the config dictionary."""
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


def build_runtime_config(data: dict[str, Any], env: Mapping[str, str]) -> RuntimeConfig:
    """Build a strongly typed runtime config from raw dict + environment."""
    mapped = apply_env_overrides(data, env)

    system_raw = mapped.get("system", {}) if isinstance(mapped.get("system", {}), dict) else {}
    trading_raw = mapped.get("trading", {}) if isinstance(mapped.get("trading", {}), dict) else {}
    risk_raw = mapped.get("risk", {}) if isinstance(mapped.get("risk", {}), dict) else {}
    exec_raw = mapped.get("execution", {}) if isinstance(mapped.get("execution", {}), dict) else {}
    storage_raw = mapped.get("storage", {}) if isinstance(mapped.get("storage", {}), dict) else {}
    backtest_raw = (
        mapped.get("backtest", {}) if isinstance(mapped.get("backtest", {}), dict) else {}
    )
    live_raw = mapped.get("live", {}) if isinstance(mapped.get("live", {}), dict) else {}
    optimization_raw = (
        mapped.get("optimization", {}) if isinstance(mapped.get("optimization", {}), dict) else {}
    )
    promotion_raw = (
        mapped.get("promotion_gate", {})
        if isinstance(mapped.get("promotion_gate", {}), dict)
        else {}
    )
    exchange_raw = (
        live_raw.get("exchange", {}) if isinstance(live_raw.get("exchange", {}), dict) else {}
    )
    promotion_kwargs = _coerce_dataclass_kwargs(promotion_raw, PromotionGateConfig)
    strategy_profiles = promotion_kwargs.pop("strategy_profiles", {})

    live_kwargs = _coerce_dataclass_kwargs(live_raw, LiveRuntimeConfig)
    for reserved_key in (
        "exchange",
        "api_key",
        "secret_key",
        "telegram_bot_token",
        "telegram_chat_id",
    ):
        live_kwargs.pop(reserved_key, None)

    live = LiveRuntimeConfig(
        **live_kwargs,
        exchange=LiveExchangeConfig(
            **_coerce_dataclass_kwargs(exchange_raw, LiveExchangeConfig),
        ),
        api_key=str(
            env.get("BINANCE_API_KEY")
            or env.get("EXCHANGE_API_KEY")
            or live_raw.get("api_key")
            or ""
        ),
        secret_key=str(
            env.get("BINANCE_SECRET_KEY")
            or env.get("EXCHANGE_SECRET_KEY")
            or live_raw.get("secret_key")
            or ""
        ),
        telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN") or live_raw.get("telegram_bot_token"),
        telegram_chat_id=env.get("TELEGRAM_CHAT_ID") or live_raw.get("telegram_chat_id"),
    )
    if not live.mode:
        live.mode = "paper"

    runtime = RuntimeConfig(
        system=SystemConfig(**_coerce_dataclass_kwargs(system_raw, SystemConfig)),
        trading=TradingConfig(**_coerce_dataclass_kwargs(trading_raw, TradingConfig)),
        risk=RiskConfig(**_coerce_dataclass_kwargs(risk_raw, RiskConfig)),
        execution=ExecutionConfig(**_coerce_dataclass_kwargs(exec_raw, ExecutionConfig)),
        storage=StorageConfig(**_coerce_dataclass_kwargs(storage_raw, StorageConfig)),
        backtest=BacktestRuntimeConfig(
            **_coerce_dataclass_kwargs(backtest_raw, BacktestRuntimeConfig)
        ),
        live=live,
        optimization=OptimizationRuntimeConfig(
            **_coerce_dataclass_kwargs(optimization_raw, OptimizationRuntimeConfig)
        ),
        promotion_gate=PromotionGateConfig(
            **promotion_kwargs,
            strategy_profiles=(
                dict(strategy_profiles) if isinstance(strategy_profiles, dict) else {}
            ),
        ),
    )

    runtime.storage.market_data_parquet_path = _resolve_storage_path(
        runtime.storage.market_data_parquet_path
    )
    runtime.storage.wal_max_bytes = max(0, _as_int(runtime.storage.wal_max_bytes, 268435456))
    runtime.storage.wal_compact_on_threshold = _as_bool(
        runtime.storage.wal_compact_on_threshold,
        True,
    )
    runtime.storage.wal_compaction_interval_seconds = max(
        0,
        _as_int(runtime.storage.wal_compaction_interval_seconds, 3600),
    )

    # Safe type coercion for critical numeric fields.
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
    runtime.trading.min_trade_qty = _as_float(runtime.trading.min_trade_qty, 0.001)
    runtime.risk.risk_per_trade = _as_float(runtime.risk.risk_per_trade, 0.005)
    runtime.risk.max_daily_loss_pct = _as_float(runtime.risk.max_daily_loss_pct, 0.03)
    runtime.execution.slippage_rate = _as_float(runtime.execution.slippage_rate, 0.0005)
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

    raw_gpu_mode = exec_raw.get("gpu_mode") if isinstance(exec_raw, dict) else None
    raw_compute_backend = exec_raw.get("compute_backend") if isinstance(exec_raw, dict) else None
    normalized_gpu_mode = _normalize_backend_token(
        raw_gpu_mode if str(raw_gpu_mode or "").strip() else runtime.execution.gpu_mode,
        field_name="execution.gpu_mode",
    )
    normalized_compute_backend = _normalize_backend_token(
        raw_compute_backend if str(raw_compute_backend or "").strip() else runtime.execution.compute_backend,
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
    runtime.live.exchange.leverage = _as_int(runtime.live.exchange.leverage, 3)
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
        _as_int(
            runtime.live.decision_cadence_seconds,
            runtime.live.poll_seconds,
        ),
    )
    runtime.live.reconciliation_interval_sec = _as_int(runtime.live.reconciliation_interval_sec, 30)
    runtime.backtest.random_seed = _as_int(runtime.backtest.random_seed, 42)
    runtime.backtest.leverage = _as_int(runtime.backtest.leverage, 3)
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
    runtime.backtest.window_seconds = max(1, _as_int(backtest_window_raw, runtime.live.window_seconds))
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
    runtime.backtest.chunk_warmup_bars = max(
        0,
        _as_int(runtime.backtest.chunk_warmup_bars, 0),
    )
    runtime.backtest.skip_ahead_enabled = _as_bool(runtime.backtest.skip_ahead_enabled, True)
    runtime.optimization.walk_forward_folds = _as_int(runtime.optimization.walk_forward_folds, 3)
    runtime.optimization.overfit_penalty = _as_float(runtime.optimization.overfit_penalty, 0.5)
    runtime.optimization.max_workers = _as_int(runtime.optimization.max_workers, 4)
    runtime.optimization.persist_best_params = _as_bool(
        runtime.optimization.persist_best_params,
        False,
    )
    runtime.optimization.validation_days = max(
        0,
        _as_int(getattr(runtime.optimization, "validation_days", 30), 30),
    )
    runtime.optimization.oos_days = max(
        1,
        _as_int(getattr(runtime.optimization, "oos_days", 30), 30),
    )
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
        runtime.promotion_gate.require_alpha_card,
        False,
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
    return runtime


def load_runtime_config(
    config_path: str = "config.yaml", env: Mapping[str, str] | None = None
) -> RuntimeConfig:
    """Load `.env`, read YAML, apply overrides, and produce typed config."""
    load_dotenv()
    effective_env = env or os.environ
    raw = load_yaml_config(config_path=config_path)
    return build_runtime_config(raw, effective_env)
