"""Runtime configuration validation."""

from __future__ import annotations

import platform
import re
from collections.abc import Iterable

from lumina_quant.configuration.schema import RuntimeConfig

SYMBOL_RE = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")


def _validate_symbols(symbols: Iterable[str]) -> None:
    for symbol in symbols:
        if not SYMBOL_RE.match(symbol):
            raise ValueError(f"Invalid symbol format '{symbol}'. Expected format like BTC/USDT.")


def validate_runtime_config(runtime: RuntimeConfig, *, for_live: bool = False) -> None:
    """Validate runtime configuration invariants."""
    storage_backend = str(runtime.storage.backend or "").strip().lower()
    if storage_backend not in {"sqlite", "influxdb"}:
        raise ValueError("storage.backend must be one of: sqlite, influxdb.")

    if not runtime.trading.symbols:
        raise ValueError("No symbols configured in trading.symbols.")
    _validate_symbols(runtime.trading.symbols)

    mode = runtime.live.mode.strip().lower()
    if mode not in {"paper", "real"}:
        raise ValueError("live.mode must be one of: paper, real.")

    exchange = runtime.live.exchange
    if exchange.driver not in {"ccxt", "mt5"}:
        raise ValueError("live.exchange.driver must be 'ccxt' or 'mt5'.")
    if exchange.driver == "mt5":
        system_name = platform.system().lower()
        bridge_python = str(getattr(runtime.live, "mt5_bridge_python", "") or "").strip()
        if system_name != "windows" and not bridge_python:
            raise ValueError(
                "live.exchange.driver='mt5' on non-Windows requires live.mt5_bridge_python "
                "(or env LQ__LIVE__MT5_BRIDGE_PYTHON)."
            )
    if exchange.market_type not in {"spot", "future"}:
        raise ValueError("live.exchange.market_type must be 'spot' or 'future'.")
    if exchange.position_mode.upper() not in {"ONEWAY", "HEDGE"}:
        raise ValueError("live.exchange.position_mode must be ONEWAY or HEDGE.")
    if exchange.margin_mode not in {"isolated", "cross"}:
        raise ValueError("live.exchange.margin_mode must be isolated or cross.")
    if exchange.leverage < 1 or exchange.leverage > 3:
        raise ValueError("live.exchange.leverage must be in range [1, 3].")

    if runtime.risk.risk_per_trade <= 0 or runtime.risk.risk_per_trade > 0.05:
        raise ValueError("risk.risk_per_trade must be in (0, 0.05].")
    if runtime.risk.max_daily_loss_pct <= 0 or runtime.risk.max_daily_loss_pct > 1:
        raise ValueError("risk.max_daily_loss_pct must be in (0, 1].")
    if runtime.risk.max_intraday_drawdown_pct <= 0 or runtime.risk.max_intraday_drawdown_pct > 1:
        raise ValueError("risk.max_intraday_drawdown_pct must be in (0, 1].")
    if runtime.risk.max_rolling_loss_pct_1h <= 0 or runtime.risk.max_rolling_loss_pct_1h > 1:
        raise ValueError("risk.max_rolling_loss_pct_1h must be in (0, 1].")
    if runtime.execution.compute_backend != "cpu":
        raise ValueError("execution.compute_backend must be 'cpu'.")
    if runtime.backtest.leverage < 1 or runtime.backtest.leverage > 20:
        raise ValueError("backtest.leverage must be in range [1, 20].")
    if runtime.live.order_timeout < 1:
        raise ValueError("live.order_timeout must be >= 1.")
    if runtime.live.reconciliation_interval_sec < 1:
        raise ValueError("live.reconciliation_interval_sec must be >= 1.")
    if runtime.optimization.max_workers < 1:
        raise ValueError("optimization.max_workers must be >= 1.")

    if runtime.promotion_gate.days < 1:
        raise ValueError("promotion_gate.days must be >= 1.")
    if runtime.promotion_gate.max_order_rejects < 0:
        raise ValueError("promotion_gate.max_order_rejects must be >= 0.")
    if runtime.promotion_gate.max_order_timeouts < 0:
        raise ValueError("promotion_gate.max_order_timeouts must be >= 0.")
    if runtime.promotion_gate.max_reconciliation_alerts < 0:
        raise ValueError("promotion_gate.max_reconciliation_alerts must be >= 0.")
    if runtime.promotion_gate.max_critical_errors < 0:
        raise ValueError("promotion_gate.max_critical_errors must be >= 0.")

    allowed_profile_keys = {
        "days",
        "max_order_rejects",
        "max_order_timeouts",
        "max_reconciliation_alerts",
        "max_critical_errors",
        "require_alpha_card",
        "alpha_card_path",
    }
    for strategy_name, profile in runtime.promotion_gate.strategy_profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"promotion_gate.strategy_profiles.{strategy_name} must be a mapping.")
        unknown = set(profile.keys()) - allowed_profile_keys
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(
                "promotion_gate.strategy_profiles."
                f"{strategy_name} contains unsupported keys: {joined}"
            )

        days = profile.get("days")
        if days is not None and int(days) < 1:
            raise ValueError(f"promotion_gate.strategy_profiles.{strategy_name}.days must be >= 1.")
        for metric_key in (
            "max_order_rejects",
            "max_order_timeouts",
            "max_reconciliation_alerts",
            "max_critical_errors",
        ):
            value = profile.get(metric_key)
            if value is not None and int(value) < 0:
                raise ValueError(
                    f"promotion_gate.strategy_profiles.{strategy_name}.{metric_key} must be >= 0."
                )

    if for_live and (not runtime.live.api_key or not runtime.live.secret_key):
        raise ValueError(
            "API keys are missing. Set BINANCE_API_KEY and BINANCE_SECRET_KEY via environment."
        )
