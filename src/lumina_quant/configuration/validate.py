"""Runtime configuration validation."""

from __future__ import annotations

import platform
import re
from collections.abc import Iterable

from lumina_quant.configuration.schema import RuntimeConfig

SYMBOL_RE = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
TIMEFRAME_RE = re.compile(r"^[1-9][0-9]*[smhdwM]$")


def _validate_symbols(symbols: Iterable[str]) -> None:
    for symbol in symbols:
        if not SYMBOL_RE.match(symbol):
            raise ValueError(f"Invalid symbol format '{symbol}'. Expected format like BTC/USDT.")


def _validate_timeframes(timeframes: Iterable[str]) -> None:
    for timeframe in timeframes:
        token = str(timeframe or "").strip()
        if not token or TIMEFRAME_RE.match(token) is None:
            raise ValueError(
                f"Invalid timeframe token '{timeframe}'. Expected format like 1m, 5m, 1h, 1d."
            )


def _validate_materializer_required_timeframes(runtime: RuntimeConfig) -> None:
    required = getattr(runtime.storage, "materializer_required_timeframes", None)
    if not isinstance(required, list) or not required:
        raise ValueError("storage.materializer_required_timeframes must be a non-empty list.")

    normalized: list[str] = []
    seen: set[str] = set()
    for token in required:
        raw_token = str(token or "").strip().lower()
        if TIMEFRAME_RE.fullmatch(raw_token) is None:
            raise ValueError(
                "storage.materializer_required_timeframes contains invalid timeframe "
                f"'{token}'. Expected format like 1s, 5m, 1h, 1d."
            )
        if raw_token in seen:
            continue
        normalized.append(raw_token)
        seen.add(raw_token)

    if "1s" not in normalized:
        raise ValueError("storage.materializer_required_timeframes must include '1s'.")

    trading_timeframes = {
        str(token).strip().lower() for token in getattr(runtime.trading, "timeframes", [])
    }
    for token in normalized:
        if token == "1s":
            continue
        if token not in trading_timeframes:
            raise ValueError(
                "storage.materializer_required_timeframes contains timeframe "
                f"'{token}' not present in trading.timeframes."
            )

    runtime.storage.materializer_required_timeframes = [
        "1s",
        *[token for token in normalized if token != "1s"],
    ]


def validate_runtime_config(runtime: RuntimeConfig, *, for_live: bool = False) -> None:
    """Validate runtime configuration invariants."""
    storage_backend = str(runtime.storage.backend or "").strip().lower()
    if storage_backend not in {"parquet-postgres", "parquet", "local"}:
        raise ValueError("storage.backend must be one of: parquet-postgres, parquet, local.")
    if int(getattr(runtime.storage, "wal_max_bytes", 0)) < 0:
        raise ValueError("storage.wal_max_bytes must be >= 0.")
    if not isinstance(getattr(runtime.storage, "wal_compact_on_threshold", True), bool):
        raise ValueError("storage.wal_compact_on_threshold must be boolean.")
    if int(getattr(runtime.storage, "wal_compaction_interval_seconds", 0)) < 0:
        raise ValueError("storage.wal_compaction_interval_seconds must be >= 0.")
    if int(getattr(runtime.storage, "collector_poll_seconds", 0)) < 1:
        raise ValueError("storage.collector_poll_seconds must be >= 1.")
    if int(getattr(runtime.storage, "collector_bootstrap_lookback_hours", 0)) < 1:
        raise ValueError("storage.collector_bootstrap_lookback_hours must be >= 1.")
    if int(getattr(runtime.storage, "materializer_poll_seconds", 0)) < 1:
        raise ValueError("storage.materializer_poll_seconds must be >= 1.")
    if str(getattr(runtime.storage, "materializer_base_timeframe", "")).strip().lower() != "1s":
        raise ValueError("storage.materializer_base_timeframe must be '1s'.")
    _validate_materializer_required_timeframes(runtime)

    if not runtime.trading.symbols:
        raise ValueError("No symbols configured in trading.symbols.")
    _validate_symbols(runtime.trading.symbols)
    if TIMEFRAME_RE.match(str(runtime.trading.timeframe or "").strip()) is None:
        raise ValueError("trading.timeframe must be a valid token like 1m, 5m, 1h, 1d.")
    _validate_timeframes(getattr(runtime.trading, "timeframes", []))
    configured_timeframes = {str(token).strip() for token in runtime.trading.timeframes}
    if str(runtime.trading.timeframe).strip() not in configured_timeframes:
        raise ValueError("trading.timeframe must be included in trading.timeframes.")

    mode = runtime.live.mode.strip().lower()
    if mode not in {"paper", "real"}:
        raise ValueError("live.mode must be one of: paper, real.")

    market_data_source = (
        str(getattr(runtime.live, "market_data_source", "committed")).strip().lower()
    )
    if market_data_source not in {"committed", "binance_live"}:
        raise ValueError("live.market_data_source must be one of: committed, binance_live.")

    order_state_source = str(getattr(runtime.live, "order_state_source", "polling")).strip().lower()
    if order_state_source not in {"polling", "user_stream"}:
        raise ValueError("live.order_state_source must be one of: polling, user_stream.")

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
    if market_data_source == "binance_live":
        if str(exchange.driver).lower() != "ccxt":
            raise ValueError(
                "live.market_data_source=binance_live requires live.exchange.driver='ccxt'."
            )
        if str(exchange.name).lower() != "binance":
            raise ValueError(
                "live.market_data_source=binance_live requires live.exchange.name='binance'."
            )
    if order_state_source == "user_stream" and market_data_source != "binance_live":
        raise ValueError(
            "live.order_state_source=user_stream requires live.market_data_source=binance_live."
        )
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
    if runtime.execution.compute_backend not in {"auto", "cpu", "gpu", "forced-gpu"}:
        raise ValueError("execution.compute_backend must be one of: auto, cpu, gpu, forced-gpu.")
    gpu_mode = str(
        getattr(runtime.execution, "gpu_mode", runtime.execution.compute_backend)
    ).strip()
    if gpu_mode not in {"auto", "cpu", "gpu", "forced-gpu"}:
        raise ValueError("execution.gpu_mode must be one of: auto, cpu, gpu, forced-gpu.")
    if float(getattr(runtime.execution, "gpu_vram_gb", 0.0)) < 0.0:
        raise ValueError("execution.gpu_vram_gb must be >= 0.")
    if runtime.backtest.leverage < 1 or runtime.backtest.leverage > 20:
        raise ValueError("backtest.leverage must be in range [1, 20].")
    if int(getattr(runtime.backtest, "poll_seconds", 1)) < 1:
        raise ValueError("backtest.poll_seconds must be >= 1.")
    if int(getattr(runtime.backtest, "window_seconds", 20)) < 1:
        raise ValueError("backtest.window_seconds must be >= 1.")
    if int(getattr(runtime.backtest, "decision_cadence_seconds", 20)) < 1:
        raise ValueError("backtest.decision_cadence_seconds must be >= 1.")
    if (
        getattr(runtime.backtest, "backtest_poll_seconds", None) is not None
        and int(getattr(runtime.backtest, "backtest_poll_seconds", 1)) < 1
    ):
        raise ValueError("backtest.backtest_poll_seconds must be >= 1.")
    if (
        getattr(runtime.backtest, "backtest_window_seconds", None) is not None
        and int(getattr(runtime.backtest, "backtest_window_seconds", 1)) < 1
    ):
        raise ValueError("backtest.backtest_window_seconds must be >= 1.")
    if (
        getattr(runtime.backtest, "backtest_decision_seconds", None) is not None
        and int(getattr(runtime.backtest, "backtest_decision_seconds", 1)) < 1
    ):
        raise ValueError("backtest.backtest_decision_seconds must be >= 1.")
    if int(getattr(runtime.backtest, "chunk_days", 7)) < 1:
        raise ValueError("backtest.chunk_days must be >= 1.")
    if int(getattr(runtime.backtest, "chunk_warmup_bars", 0)) < 0:
        raise ValueError("backtest.chunk_warmup_bars must be >= 0.")
    if str(getattr(runtime.backtest, "mode", "windowed")).strip().lower() not in {
        "windowed",
        "legacy_batch",
        "legacy_1s",
    }:
        raise ValueError("backtest.mode must be one of: windowed, legacy_batch, legacy_1s.")
    if int(getattr(runtime.live, "poll_seconds", runtime.live.poll_interval)) < 1:
        raise ValueError("live.poll_seconds must be >= 1.")
    if int(getattr(runtime.live, "window_seconds", 20)) < 1:
        raise ValueError("live.window_seconds must be >= 1.")
    if (
        getattr(runtime.live, "live_poll_seconds", None) is not None
        and int(getattr(runtime.live, "live_poll_seconds", 1)) < 1
    ):
        raise ValueError("live.live_poll_seconds must be >= 1.")
    if (
        getattr(runtime.live, "ingest_window_seconds", None) is not None
        and int(getattr(runtime.live, "ingest_window_seconds", 1)) < 1
    ):
        raise ValueError("live.ingest_window_seconds must be >= 1.")
    if int(getattr(runtime.live, "decision_cadence_seconds", 20)) < 1:
        raise ValueError("live.decision_cadence_seconds must be >= 1.")
    if int(getattr(runtime.live, "materialized_staleness_threshold_seconds", 45)) < 1:
        raise ValueError("live.materialized_staleness_threshold_seconds must be >= 1.")
    if int(getattr(runtime.live, "materialized_staleness_alert_cooldown_seconds", 60)) < 1:
        raise ValueError("live.materialized_staleness_alert_cooldown_seconds must be >= 1.")
    if runtime.live.order_timeout < 1:
        raise ValueError("live.order_timeout must be >= 1.")
    if runtime.live.reconciliation_interval_sec < 1:
        raise ValueError("live.reconciliation_interval_sec must be >= 1.")
    if runtime.optimization.max_workers < 1:
        raise ValueError("optimization.max_workers must be >= 1.")
    if int(getattr(runtime.optimization, "validation_days", 0)) < 0:
        raise ValueError("optimization.validation_days must be >= 0.")
    if int(getattr(runtime.optimization, "oos_days", 0)) < 1:
        raise ValueError("optimization.oos_days must be >= 1.")

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

    deprecated_live_parity = getattr(runtime.live, "market_window_parity_v2_enabled", None)
    deprecated_backtest_parity = getattr(
        runtime.backtest,
        "market_window_parity_v2_enabled",
        None,
    )
    if (
        deprecated_live_parity is not None
        and deprecated_backtest_parity is not None
        and bool(deprecated_live_parity) != bool(deprecated_backtest_parity)
    ):
        raise ValueError(
            "live.market_window_parity_v2_enabled and backtest.market_window_parity_v2_enabled "
            "must match; use market_window.parity_v2_enabled."
        )

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
