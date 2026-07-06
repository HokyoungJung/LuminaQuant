"""Runtime configuration validation."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Iterable

from lumina_quant.configuration.schema import RuntimeConfig
from lumina_quant.research_universe import BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED
from lumina_quant.core.order_policy import (
    SUPPORTED_LIMIT_PRICE_MODES,
    canonical_order_type,
    normalize_limit_price_mode,
)

SYMBOL_RE = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
TIMEFRAME_RE = re.compile(r"^[1-9][0-9]*[smhdwM]$")
MAX_LIVE_EXCHANGE_LEVERAGE = 20


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


def _validate_storage_runtime_invariants(runtime: RuntimeConfig) -> None:
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


def _validate_trading_runtime_invariants(runtime: RuntimeConfig) -> None:
    if not runtime.trading.symbols:
        raise ValueError("No symbols configured in trading.symbols.")
    _validate_symbols(runtime.trading.symbols)
    if TIMEFRAME_RE.match(str(runtime.trading.timeframe or "").strip()) is None:
        raise ValueError("trading.timeframe must be a valid token like 1m, 5m, 1h, 1d.")
    _validate_timeframes(getattr(runtime.trading, "timeframes", []))
    configured_timeframes = {str(token).strip() for token in runtime.trading.timeframes}
    if str(runtime.trading.timeframe).strip() not in configured_timeframes:
        raise ValueError("trading.timeframe must be included in trading.timeframes.")


def _validate_kill_switch_and_risk_envelope(runtime: RuntimeConfig) -> None:
    """Always-on kill-switch: reject any config that disables the risk envelope.

    Called first in validate_runtime_config so the full risk envelope is
    non-bypassable before any other check runs.  kill_switch_enabled=False is
    structurally rejected; all envelope bounds are enforced here so disabling
    any single component is equally rejected.

    Per-stage composition note (spec R7): stage selection is free — an operator
    may set go_live_stage='full' directly without traversing prior stages, provided
    each stage's own entry checks pass at startup.  What is mandatory is that
    the kill-switch envelope is intact for *any* selected stage.
    """
    # 1. kill_switch_enabled — structurally rejected when False
    if not getattr(runtime.live, "kill_switch_enabled", True):
        raise ValueError(
            "live.kill_switch_enabled cannot be set to false. "
            "The kill-switch is always-on and config-disable is rejected."
        )
    # 2. Daily loss cap — (0, 0.5]; zero disables the cap, >0.5 provides no protection
    max_daily = float(getattr(runtime.risk, "max_daily_loss_pct", 0.03))
    if max_daily <= 0 or max_daily > 0.5:
        raise ValueError(
            f"risk.max_daily_loss_pct={max_daily!r} must be in (0, 0.5]. "
            "Zero disables the cap; values above 0.5 provide no meaningful protection."
        )
    # 3. Per-symbol position size cap — (0, 1.0]
    max_pos = float(getattr(runtime.risk, "max_position_size_pct", 1.0))
    if max_pos <= 0 or max_pos > 1.0:
        raise ValueError(
            f"risk.max_position_size_pct={max_pos!r} must be in (0, 1.0]. "
            "Zero disables the cap; values above 1.0 exceed total capital."
        )
    # 4. Consecutive-loss halt — must be a positive integer (zero disables the trigger)
    halt = int(getattr(runtime.risk, "consecutive_loss_halt_count", 5))
    if halt <= 0:
        raise ValueError(
            f"risk.consecutive_loss_halt_count={halt!r} must be > 0. "
            "Zero disables the automatic consecutive-loss halt trigger."
        )
    # 5. freeze_new_entries_on_breach required in canary/full stages
    stage = str(getattr(runtime.live, "go_live_stage", "testnet")).strip().lower()
    if stage in {"canary", "full"} and not bool(
        getattr(runtime.risk, "freeze_new_entries_on_breach", True)
    ):
        raise ValueError(
            f"risk.freeze_new_entries_on_breach must be true when "
            f"live.go_live_stage='{stage}'. "
            "Canary and full stages require entry freeze on breach to contain runaway positions."
        )
    # 6. Structural prod-routing gate: canary/full stage MUST use mode='real'.
    # Stage determines the exchange endpoint (testnet=False for canary/full).
    # Combining stage=canary/full with mode=paper is an operator error that would
    # route paper-commission-simulated orders to the production REST endpoint.
    mode = str(getattr(runtime.live, "mode", "paper")).strip().lower()
    if stage in {"canary", "full"} and mode != "real":
        raise ValueError(
            f"live.go_live_stage='{stage}' requires live.mode='real'. "
            "Canary and full stages route orders to the production exchange endpoint; "
            "combining them with mode='paper' would place real orders while simulating paper commissions. "
            "Set live.mode='real' and set LUMINA_ENABLE_LIVE_REAL=true in the environment."
        )
    # 7. Per-order absolute value cap — must be positive (zero silently disables the cap
    # in RiskManager.check_order, bypassing all per-order notional limits)
    max_order_val = float(getattr(runtime.risk, "max_order_value", 5000.0))
    if max_order_val <= 0:
        raise ValueError(
            f"risk.max_order_value={max_order_val!r} must be > 0. "
            "Zero disables the per-order absolute value cap in the risk engine."
        )
    # 8. Per-symbol exposure cap — (0, 1.0]; zero silently disables symbol-level cap
    max_sym_exp = float(getattr(runtime.risk, "max_symbol_exposure_pct", 0.25))
    if max_sym_exp <= 0 or max_sym_exp > 1.0:
        raise ValueError(
            f"risk.max_symbol_exposure_pct={max_sym_exp!r} must be in (0, 1.0]. "
            "Zero disables the per-symbol exposure cap."
        )
    # 9. Total margin/notional cap — (0, 5.0]; zero silently disables portfolio-level cap
    max_total_margin = float(getattr(runtime.risk, "max_total_margin_pct", 0.5))
    if max_total_margin <= 0 or max_total_margin > 5.0:
        raise ValueError(
            f"risk.max_total_margin_pct={max_total_margin!r} must be in (0, 5.0]. "
            "Zero disables the total margin/notional cap."
        )


def _validate_live_mode_and_sources(runtime: RuntimeConfig) -> tuple[str, str]:
    mode = runtime.live.mode.strip().lower()
    if mode not in {"paper", "real"}:
        raise ValueError("live.mode must be one of: paper, real.")

    market_data_source = (
        str(getattr(runtime.live, "market_data_source", "committed")).strip().lower()
    )
    if market_data_source not in {"committed", "binance_futures", "external", "polymarket_live"}:
        raise ValueError(
            "live.market_data_source must be one of: committed, binance_futures, external, polymarket_live."
        )

    order_state_source = str(getattr(runtime.live, "order_state_source", "polling")).strip().lower()
    if order_state_source not in {"polling", "user_stream"}:
        raise ValueError("live.order_state_source must be one of: polling, user_stream.")

    return market_data_source, order_state_source


def _validate_live_exchange_runtime_invariants(
    runtime: RuntimeConfig, market_data_source: str, order_state_source: str
) -> None:
    exchange = runtime.live.exchange
    if exchange.driver not in {"binance_futures", "binance_native", "mt5", "polymarket"}:
        raise ValueError(
            "live.exchange.driver must be 'binance_futures', 'binance_native', 'mt5', or 'polymarket'."
        )
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
    if (
        str(exchange.driver).lower() in {"binance_futures", "binance_native"}
        and str(exchange.market_type).lower() != "future"
    ):
        raise ValueError(
            "Binance USDⓈ-M Futures integration supports live.exchange.market_type='future' only."
        )
    if market_data_source == "binance_futures":
        if str(exchange.driver).lower() not in {"binance_futures", "binance_native"}:
            raise ValueError(
                "live.market_data_source=binance_futures requires live.exchange.driver='binance_futures'."
            )
        if str(exchange.name).lower() != "binance":
            raise ValueError(
                "live.market_data_source=binance_futures requires live.exchange.name='binance'."
            )
        if str(exchange.market_type).lower() != "future":
            raise ValueError(
                "live.market_data_source=binance_futures requires live.exchange.market_type='future'."
            )
    if order_state_source == "user_stream" and market_data_source != "binance_futures":
        raise ValueError(
            "live.order_state_source=user_stream requires live.market_data_source=binance_futures."
        )
    if market_data_source == "external":
        external = runtime.live.external
        if str(getattr(external, "source_kind", "")).lower() not in {"jsonl", "parquet", "pipe"}:
            raise ValueError("live.external.source_kind must be one of: jsonl, parquet, pipe.")
        if not str(getattr(external, "path", "") or "").strip():
            raise ValueError("live.market_data_source=external requires live.external.path.")
        if str(getattr(external, "schema", "market_window_v1")).lower() not in {
            "market_window_v1",
            "ohlcv_1s_v1",
        }:
            raise ValueError("live.external.schema must be one of: market_window_v1, ohlcv_1s_v1.")
    if market_data_source == "polymarket_live":
        if str(exchange.driver).lower() != "polymarket":
            raise ValueError(
                "live.market_data_source=polymarket_live requires live.exchange.driver='polymarket'."
            )
        if order_state_source != "polling":
            raise ValueError(
                "live.market_data_source=polymarket_live currently requires live.order_state_source=polling."
            )
        asset_ids = list(getattr(runtime.live.polymarket, "asset_ids", []) or [])
        if not asset_ids:
            raise ValueError(
                "live.market_data_source=polymarket_live requires live.polymarket.asset_ids."
            )
        if runtime.live.mode == "real" and not bool(
            getattr(runtime.live.polymarket, "allow_real_execution", False)
        ):
            raise ValueError(
                "Polymarket Phase 1 does not support real execution. Use paper mode unless allow_real_execution is explicitly enabled for a later phase."
            )
    if exchange.position_mode.upper() not in {"ONEWAY", "HEDGE"}:
        raise ValueError("live.exchange.position_mode must be ONEWAY or HEDGE.")
    if exchange.margin_mode not in {"isolated", "cross"}:
        raise ValueError("live.exchange.margin_mode must be isolated or cross.")
    if exchange.leverage < 1 or exchange.leverage > MAX_LIVE_EXCHANGE_LEVERAGE:
        raise ValueError(
            f"live.exchange.leverage must be in range [1, {MAX_LIVE_EXCHANGE_LEVERAGE}]."
        )


def _validate_risk_and_execution_runtime_invariants(runtime: RuntimeConfig) -> None:
    if runtime.trading.target_allocation_mode not in {
        "legacy_notional_cap",
        "notional_cap",
        "notional_fraction",
        "notional_target_fraction",
        "isolated_margin_fraction",
        "margin_fraction",
        "isolated_margin",
    }:
        raise ValueError(
            "trading.target_allocation_mode must be one of: legacy_notional_cap, "
            "notional_fraction, isolated_margin_fraction."
        )
    if runtime.risk.risk_per_trade <= 0 or runtime.risk.risk_per_trade > 0.05:
        raise ValueError("risk.risk_per_trade must be in (0, 0.05].")
    if runtime.risk.max_daily_loss_pct <= 0 or runtime.risk.max_daily_loss_pct > 0.5:
        raise ValueError("risk.max_daily_loss_pct must be in (0, 0.5].")
    if runtime.risk.max_intraday_drawdown_pct <= 0 or runtime.risk.max_intraday_drawdown_pct > 1:
        raise ValueError("risk.max_intraday_drawdown_pct must be in (0, 1].")
    if runtime.risk.max_rolling_loss_pct_1h <= 0 or runtime.risk.max_rolling_loss_pct_1h > 1:
        raise ValueError("risk.max_rolling_loss_pct_1h must be in (0, 1].")
    if runtime.risk.max_order_notional_pct < 0:
        raise ValueError("risk.max_order_notional_pct must be >= 0.")
    if runtime.risk.max_total_notional_pct < 0:
        raise ValueError("risk.max_total_notional_pct must be >= 0.")
    # Real-money desk controls: bounds always enforced; 0 == disabled (legacy).
    if int(getattr(runtime.risk, "max_orders_per_minute", 0)) < 0:
        raise ValueError("risk.max_orders_per_minute must be >= 0.")
    if float(getattr(runtime.risk, "max_daily_notional_turnover_pct", 0.0)) < 0:
        raise ValueError("risk.max_daily_notional_turnover_pct must be >= 0.")
    if float(getattr(runtime.risk, "max_position_age_hours", 0.0)) < 0:
        raise ValueError("risk.max_position_age_hours must be >= 0.")
    if runtime.execution.compute_backend not in {"auto", "cpu", "gpu", "forced-gpu"}:
        raise ValueError("execution.compute_backend must be one of: auto, cpu, gpu, forced-gpu.")
    gpu_mode = str(
        getattr(runtime.execution, "gpu_mode", runtime.execution.compute_backend)
    ).strip()
    if gpu_mode not in {"auto", "cpu", "gpu", "forced-gpu"}:
        raise ValueError("execution.gpu_mode must be one of: auto, cpu, gpu, forced-gpu.")
    if float(getattr(runtime.execution, "gpu_vram_gb", 0.0)) < 0.0:
        raise ValueError("execution.gpu_vram_gb must be >= 0.")


def _validate_backtest_runtime_invariants(runtime: RuntimeConfig) -> None:
    if str(
        getattr(runtime.backtest, "risk_free_mode", "us_treasury_constant")
    ).strip().lower() not in {
        "zero",
        "us_treasury_constant",
        "us_treasury_series",
    }:
        raise ValueError(
            "backtest.risk_free_mode must be one of: zero, us_treasury_constant, us_treasury_series."
        )
    if str(getattr(runtime.backtest, "sortino_target_mode", "same_as_rf")).strip().lower() not in {
        "zero",
        "same_as_rf",
        "explicit",
    }:
        raise ValueError("backtest.sortino_target_mode must be one of: zero, same_as_rf, explicit.")
    if float(getattr(runtime.backtest, "risk_free_annual", 0.0)) <= -1.0:
        raise ValueError("backtest.risk_free_annual must be > -1.0.")
    if float(getattr(runtime.backtest, "sortino_target_annual", 0.0)) <= -1.0:
        raise ValueError("backtest.sortino_target_annual must be > -1.0.")
    if runtime.backtest.leverage < 1 or runtime.backtest.leverage > 20:
        raise ValueError("backtest.leverage must be in range [1, 20].")
    if str(getattr(runtime.backtest, "margin_mode", "isolated")).strip().lower() != "isolated":
        raise ValueError(
            "backtest.margin_mode currently supports isolated only because liquidation modeling is isolated-mode based."
        )
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


def _validate_live_and_optimization_runtime_invariants(runtime: RuntimeConfig) -> None:
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
    # Real-money live-safety controls: bounds always enforced; 0 == disabled (legacy).
    if float(getattr(runtime.live, "max_bbo_spread_bps_at_submit", 0.0)) < 0:
        raise ValueError("live.max_bbo_spread_bps_at_submit must be >= 0.")
    if float(getattr(runtime.live, "max_estimated_one_way_slippage_bps", 0.0)) < 0:
        raise ValueError("live.max_estimated_one_way_slippage_bps must be >= 0.")
    if float(getattr(runtime.live, "max_limit_price_band_pct", 0.0)) < 0:
        raise ValueError("live.max_limit_price_band_pct must be >= 0.")
    if float(getattr(runtime.live, "equity_reconciliation_interval_sec", 0.0)) < 0:
        raise ValueError("live.equity_reconciliation_interval_sec must be >= 0.")
    if float(getattr(runtime.live, "market_data_silence_timeout_sec", 0.0)) < 0:
        raise ValueError("live.market_data_silence_timeout_sec must be >= 0.")
    default_order_type = canonical_order_type(
        getattr(runtime.live, "default_order_type", "LMT"),
        default="LMT",
    )
    if default_order_type == "MKT" and not bool(
        getattr(runtime.live, "allow_market_orders", False)
    ):
        raise ValueError(
            "live.default_order_type=MKT requires live.allow_market_orders=true; "
            "live defaults must remain limit-first."
        )
    limit_price_mode = normalize_limit_price_mode(
        getattr(runtime.live, "limit_price_mode", "one_tick_worse")
    )
    if limit_price_mode not in SUPPORTED_LIMIT_PRICE_MODES:
        raise ValueError(
            "live.limit_price_mode must be one of: "
            + ", ".join(sorted(SUPPORTED_LIMIT_PRICE_MODES))
        )
    if int(getattr(runtime.live, "limit_price_offset_ticks", 1)) < 0:
        raise ValueError("live.limit_price_offset_ticks must be >= 0.")
    if float(getattr(runtime.live, "limit_price_tick_fallback", 0.0)) < 0.0:
        raise ValueError("live.limit_price_tick_fallback must be >= 0.")
    if not str(getattr(runtime.live, "limit_time_in_force", "GTC") or "").strip():
        raise ValueError("live.limit_time_in_force must be non-empty for limit orders.")
    protective_style = (
        str(getattr(runtime.live, "protective_order_style", "limit") or "limit").strip().lower()
    )
    if protective_style not in {"limit", "market"}:
        raise ValueError("live.protective_order_style must be 'limit' or 'market'.")
    if protective_style == "market" and not bool(
        getattr(runtime.live, "allow_market_orders", False)
    ):
        raise ValueError(
            "live.protective_order_style=market requires live.allow_market_orders=true; "
            "protective orders default to STOP/TAKE_PROFIT limit."
        )
    # Phase 5 go-live stage pipeline fields
    go_live_stage = str(getattr(runtime.live, "go_live_stage", "testnet")).strip().lower()
    if go_live_stage not in {"testnet", "shadow", "canary", "full"}:
        raise ValueError("live.go_live_stage must be one of: testnet, shadow, canary, full.")
    canary_frac = float(getattr(runtime.live, "canary_position_fraction", 0.10))
    if canary_frac <= 0 or canary_frac > 1.0:
        raise ValueError("live.canary_position_fraction must be in (0, 1.0].")
    parity_ratio = float(getattr(runtime.live, "shadow_parity_min_ratio", 0.99))
    if parity_ratio <= 0 or parity_ratio > 1.0:
        raise ValueError("live.shadow_parity_min_ratio must be in (0, 1.0].")
    window_bars = int(getattr(runtime.live, "shadow_parity_window_bars", 1000))
    if window_bars < 1:
        raise ValueError("live.shadow_parity_window_bars must be >= 1.")
    if runtime.optimization.max_workers < 1:
        raise ValueError("optimization.max_workers must be >= 1.")
    if int(getattr(runtime.optimization, "validation_days", 0)) < 0:
        raise ValueError("optimization.validation_days must be >= 0.")
    if int(getattr(runtime.optimization, "oos_days", 0)) < 1:
        raise ValueError("optimization.oos_days must be >= 1.")


def _validate_promotion_gate_runtime_invariants(runtime: RuntimeConfig) -> None:
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


def _validate_market_window_parity_invariants(runtime: RuntimeConfig) -> None:
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


def _validate_strategy_profile_invariants(runtime: RuntimeConfig) -> None:
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


def _validate_real_money_live_safety_gate(runtime: RuntimeConfig) -> None:
    """Fail-closed real-money live-safety gate (real_money_readiness audit 2026-07-06).

    Triggers only for a real-money Binance live path — ``live.mode == 'real'`` OR
    ``live.go_live_stage in {canary, full}`` (both of which already require
    ``mode == 'real'`` via the prod-routing gate in
    ``_validate_kill_switch_and_risk_envelope``). Paper / testnet / shadow / backtest
    validation is therefore unchanged and the shipped default config (paper + testnet)
    stays byte-identical.

    Scope note: these requirements are Binance-USDⓈ-M-specific (exchange-resident
    reduce-only algo stops, BBO/mark price bands, equity reconciliation). Non-Binance
    real paths are fenced elsewhere — Polymarket real execution is blocked in Phase 1
    by ``_validate_live_exchange_runtime_invariants`` and MT5 is a bridge — so this gate
    only applies to the ``binance_futures`` / ``binance_native`` drivers.
    """
    mode = str(getattr(runtime.live, "mode", "paper")).strip().lower()
    stage = str(getattr(runtime.live, "go_live_stage", "testnet")).strip().lower()
    if mode != "real" and stage not in {"canary", "full"}:
        return
    driver = str(getattr(runtime.live.exchange, "driver", "") or "").strip().lower()
    if driver not in {"binance_futures", "binance_native"}:
        return

    risk = runtime.risk
    live = runtime.live

    # C1 — the kill-switch must be able to actually de-risk (a reachable FLATTEN tier
    # AND market escalation), not FREEZE-only.
    has_flatten_tier = bool(getattr(risk, "auto_flatten_on_breach", False)) or (
        float(getattr(risk, "hard_drawdown_flatten_pct", 0.0)) > 0.0
    )
    if not (has_flatten_tier and bool(getattr(risk, "flatten_escalate_to_market", False))):
        raise ValueError(
            "Real-money live requires a reachable kill-switch FLATTEN tier: set "
            "risk.auto_flatten_on_breach=true OR risk.hard_drawdown_flatten_pct>0, AND "
            "risk.flatten_escalate_to_market=true so the kill-switch can force-close open "
            "positions instead of only freezing new entries."
        )

    # C2 — real-mode managed protective stops (exchange-resident STOP/TP).
    if not bool(getattr(live, "real_mode_managed_protective_stops", False)):
        raise ValueError(
            "Real-money live requires live.real_mode_managed_protective_stops=true so "
            "stop_loss/take_profit translate to Binance reduce-only algo orders "
            "(exchange-resident protection survives a process crash / disconnect)."
        )

    # C3 — fat-finger / price-band guard must be armed.
    band_pct = float(getattr(live, "max_limit_price_band_pct", 0.0))
    require_bbo = bool(getattr(live, "require_bbo_for_limit_orders", False))
    book_ticker = bool(getattr(live, "book_ticker_enabled", False))
    if not (band_pct > 0.0 or (require_bbo and book_ticker)):
        raise ValueError(
            "Real-money live requires a fat-finger price guard: set "
            "live.max_limit_price_band_pct>0, or set BOTH "
            "live.require_bbo_for_limit_orders=true and live.book_ticker_enabled=true so "
            "limit prices are validated against a live best-bid/offer."
        )

    # C4 — periodic equity/cash reconciliation.
    if float(getattr(live, "equity_reconciliation_interval_sec", 0.0)) <= 0.0:
        raise ValueError(
            "Real-money live requires live.equity_reconciliation_interval_sec>0 so live "
            "equity/cash is periodically re-synced against the exchange (funding drain and "
            "exchange-side liquidations must reach the equity kill-switches)."
        )

    # C6 — market-data-silence watchdog.
    if float(getattr(live, "market_data_silence_timeout_sec", 0.0)) <= 0.0:
        raise ValueError(
            "Real-money live requires live.market_data_silence_timeout_sec>0 so a quietly "
            "dead market-data feed is detected (alert + freeze) instead of marking equity at "
            "a frozen last price."
        )

    # D4 — live bar sanity gate must be armed (finite/positive OHLC, high>=low).
    if not bool(getattr(live, "bar_sanity_check", False)):
        raise ValueError(
            "Real-money live requires live.bar_sanity_check=true so incoming live bars are "
            "gated on finite/positive OHLC values and high>=low before being accepted."
        )

    # D2 — per-symbol dead-feed detection must be armed.
    if int(getattr(live, "stale_symbol_after_ms", 0) or 0) <= 0:
        raise ValueError(
            "Real-money live requires live.stale_symbol_after_ms>0 so a per-symbol dead "
            "market-data feed is detected instead of silently trading on frozen data."
        )

    # M6 — an explicit, small trading.symbols list (never inherit the research universe).
    symbols = list(getattr(runtime.trading, "symbols", []) or [])
    if not symbols:
        raise ValueError("Real-money live requires an explicit non-empty trading.symbols list.")
    if set(symbols) == set(BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED):
        raise ValueError(
            "Real-money live requires an explicit trading.symbols list; it currently equals "
            "the full research universe (inherited default). Pin a small explicit set of "
            "symbols to trade in real mode."
        )


def validate_runtime_config(runtime: RuntimeConfig, *, for_live: bool = False) -> None:
    """Validate runtime configuration invariants."""
    _validate_kill_switch_and_risk_envelope(runtime)
    _validate_storage_runtime_invariants(runtime)
    _validate_trading_runtime_invariants(runtime)
    market_data_source, order_state_source = _validate_live_mode_and_sources(runtime)
    _validate_backtest_runtime_invariants(runtime)
    _validate_live_exchange_runtime_invariants(
        runtime,
        market_data_source=market_data_source,
        order_state_source=order_state_source,
    )
    _validate_risk_and_execution_runtime_invariants(runtime)
    _validate_live_and_optimization_runtime_invariants(runtime)
    _validate_promotion_gate_runtime_invariants(runtime)
    _validate_market_window_parity_invariants(runtime)
    _validate_strategy_profile_invariants(runtime)
    _validate_real_money_live_safety_gate(runtime)

    if (
        for_live
        and str(runtime.live.mode).strip().lower() == "real"
        and getattr(runtime.live, "require_real_enable_flag", True)
        and os.environ.get("LUMINA_ENABLE_LIVE_REAL", "").strip().lower() != "true"
    ):
        raise ValueError(
            "Live real mode requires LUMINA_ENABLE_LIVE_REAL=true in the environment. "
            "Set this explicitly to confirm intent to place real orders."
        )

    if for_live:
        driver = str(getattr(runtime.live.exchange, "driver", "") or "").strip().lower()
        if driver in {"binance_futures", "binance_native"}:
            if not runtime.live.api_key or not runtime.live.secret_key:
                raise ValueError(
                    "API keys are missing for Binance driver. "
                    "Set BINANCE_API_KEY and BINANCE_SECRET_KEY via environment."
                )
        elif driver == "polymarket":
            private_key_env = str(
                getattr(runtime.live.polymarket, "private_key_env", "POLYMARKET_PRIVATE_KEY")
                or "POLYMARKET_PRIVATE_KEY"
            )
            if not os.environ.get(private_key_env, "").strip():
                raise ValueError(
                    f"Polymarket private key is missing. Set {private_key_env} in the environment."
                )
