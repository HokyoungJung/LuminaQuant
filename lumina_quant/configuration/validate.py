"""Runtime configuration validation."""

from __future__ import annotations

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
    if not runtime.trading.symbols:
        raise ValueError("No symbols configured in trading.symbols.")
    _validate_symbols(runtime.trading.symbols)

    mode = runtime.live.mode.strip().lower()
    if mode not in {"paper", "real"}:
        raise ValueError("live.mode must be one of: paper, real.")

    exchange = runtime.live.exchange
    if exchange.driver not in {"ccxt", "mt5"}:
        raise ValueError("live.exchange.driver must be 'ccxt' or 'mt5'.")
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

    if for_live and (not runtime.live.api_key or not runtime.live.secret_key):
        raise ValueError(
            "API keys are missing. Set BINANCE_API_KEY and BINANCE_SECRET_KEY via environment."
        )
