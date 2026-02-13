"""Configuration loader with env overrides and legacy mapping."""

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import yaml
from dotenv import load_dotenv
from lumina_quant.configuration.schema import (
    BacktestRuntimeConfig,
    ExecutionConfig,
    LiveExchangeConfig,
    LiveRuntimeConfig,
    OptimizationRuntimeConfig,
    RiskConfig,
    RuntimeConfig,
    StorageConfig,
    SystemConfig,
    TradingConfig,
)

T = TypeVar("T")


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
        if not key.startswith("LQ_"):
            continue
        value = _parse_env_scalar(raw_value)
        if key.startswith("LQ__"):
            tokens = [token for token in key[4:].split("__") if token]
        else:
            tokens = [token for token in key[3:].split("__") if token]
        if not tokens:
            continue
        _set_nested_value(merged, tokens, value)
    return merged


def apply_legacy_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Map legacy keys into the current schema while preserving compatibility."""
    mapped = dict(data)
    live = mapped.get("live", {})
    if not isinstance(live, dict):
        live = {}
        mapped["live"] = live

    exchange = live.get("exchange", {})
    if isinstance(exchange, str):
        exchange = {"name": exchange}
    if not isinstance(exchange, dict):
        exchange = {}

    # Legacy flat exchange id mapping.
    if "exchange_id" in live and "name" not in exchange:
        exchange["name"] = str(live.get("exchange_id", "binance")).lower()
    if "exchange_driver" in live and "driver" not in exchange:
        exchange["driver"] = str(live.get("exchange_driver", "ccxt")).lower()
    live["exchange"] = exchange

    # Legacy testnet boolean -> mode string.
    if "mode" not in live and "testnet" in live:
        live["mode"] = "paper" if _as_bool(live.get("testnet"), True) else "real"
        warnings.warn(
            "live.testnet is deprecated; use live.mode: paper|real.",
            DeprecationWarning,
            stacklevel=2,
        )

    mapped["live"] = live
    return mapped


def _coerce_dataclass_kwargs(raw: dict[str, Any], model_cls: type[T]) -> dict[str, Any]:
    allowed = {item.name for item in fields(model_cls)}
    return {key: value for key, value in raw.items() if key in allowed}


def build_runtime_config(data: dict[str, Any], env: Mapping[str, str]) -> RuntimeConfig:
    """Build a strongly typed runtime config from raw dict + environment."""
    mapped = apply_legacy_mapping(apply_env_overrides(data, env))

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
    exchange_raw = (
        live_raw.get("exchange", {}) if isinstance(live_raw.get("exchange", {}), dict) else {}
    )

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
    # Normalize mode from legacy testnet when mode is still empty.
    if not live.mode:
        live.mode = "paper" if _as_bool(live.testnet, True) else "real"

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
    )

    # Safe type coercion for critical numeric fields.
    runtime.trading.initial_capital = _as_float(runtime.trading.initial_capital, 10000.0)
    runtime.trading.target_allocation = _as_float(runtime.trading.target_allocation, 0.1)
    runtime.trading.min_trade_qty = _as_float(runtime.trading.min_trade_qty, 0.001)
    runtime.risk.risk_per_trade = _as_float(runtime.risk.risk_per_trade, 0.005)
    runtime.risk.max_daily_loss_pct = _as_float(runtime.risk.max_daily_loss_pct, 0.03)
    runtime.execution.slippage_rate = _as_float(runtime.execution.slippage_rate, 0.0005)
    runtime.live.exchange.leverage = _as_int(runtime.live.exchange.leverage, 3)
    runtime.live.poll_interval = _as_int(runtime.live.poll_interval, 2)
    runtime.live.reconciliation_interval_sec = _as_int(runtime.live.reconciliation_interval_sec, 30)
    runtime.backtest.random_seed = _as_int(runtime.backtest.random_seed, 42)
    runtime.backtest.leverage = _as_int(runtime.backtest.leverage, 3)
    runtime.optimization.walk_forward_folds = _as_int(runtime.optimization.walk_forward_folds, 3)
    runtime.optimization.overfit_penalty = _as_float(runtime.optimization.overfit_penalty, 0.5)
    runtime.optimization.max_workers = _as_int(runtime.optimization.max_workers, 4)
    runtime.optimization.persist_best_params = _as_bool(
        runtime.optimization.persist_best_params,
        False,
    )
    return runtime


def load_runtime_config(
    config_path: str = "config.yaml", env: Mapping[str, str] | None = None
) -> RuntimeConfig:
    """Load `.env`, read YAML, apply overrides, and produce typed config."""
    load_dotenv()
    effective_env = env or os.environ
    raw = load_yaml_config(config_path=config_path)
    return build_runtime_config(raw, effective_env)
