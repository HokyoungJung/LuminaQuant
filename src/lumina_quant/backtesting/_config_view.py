"""BacktestConfigView — permanent engine-private uppercase-attr adapter.

Status: PERMANENT BY DESIGN (Phase 6, 2026-06-10).

Rationale: ~15 research scripts and the backtest engine internals (portfolio_backtest,
execution_sim) legitimately depend on the uppercase-attr surface.  Migrating all call
sites to RuntimeConfig direct access would be pure churn with no correctness benefit.
BacktestConfigView is intentional API — private to this package, engine-scoped.

Migration path (to remove entirely one day): replace all uppercase-attr access in
scripts and engine internals with typed RuntimeConfig dotpath access, then delete.
No timeline is committed; Phase 7 may audit but does NOT own the deletion.

configuration/__init__.py re-exports this class so research scripts can continue to
``from lumina_quant.configuration import BacktestConfigView`` without change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lumina_quant.configuration.schema import RuntimeConfig


_ALPHA_MAX_BACKTEST_CONFIG_TYPE_ID = (
    "lumina_quant.research.alpha_max_engine_runner",
    "AlphaMaxBacktestConfig",
)
_ALPHA_MAX_BACKTEST_CONFIG_TYPE: type[object] | None = None


def register_alpha_max_backtest_config_type(config_type: type[object]) -> None:
    """Register the exact frozen type without importing research from backtesting."""
    global _ALPHA_MAX_BACKTEST_CONFIG_TYPE
    if (
        type(config_type) is not type
        or (config_type.__module__, config_type.__qualname__) != _ALPHA_MAX_BACKTEST_CONFIG_TYPE_ID
        or (
            _ALPHA_MAX_BACKTEST_CONFIG_TYPE is not None
            and _ALPHA_MAX_BACKTEST_CONFIG_TYPE is not config_type
        )
    ):
        raise RuntimeError("alpha_max_backtest_config_type_registration_invalid")
    _ALPHA_MAX_BACKTEST_CONFIG_TYPE = config_type


def is_exact_alpha_max_backtest_config(config: object) -> bool:
    """Recognize the frozen research config without probing runtime fields."""
    return (
        _ALPHA_MAX_BACKTEST_CONFIG_TYPE is not None
        and type(config) is _ALPHA_MAX_BACKTEST_CONFIG_TYPE
    )


def wrapped_runtime_config(config: object) -> object | None:
    """Return ``BacktestConfigView._rt`` without probing the frozen alpha config."""
    if is_exact_alpha_max_backtest_config(config):
        return None
    return getattr(config, "_rt", None)


class BacktestConfigView:
    """Typed config bag for backtest engine — uppercase attrs from RuntimeConfig.

    Covers BaseConfig + BacktestConfig + LiveConfig attribute sets that
    portfolio_backtest.py, execution_sim.py, and backtest.py need.

    Stores ``._rt`` (the originating ``RuntimeConfig``) so engine components can call
    ``ExecutionModelConfig.from_runtime(config._rt)`` directly.
    """

    def __init__(self, runtime: RuntimeConfig) -> None:
        # Stored so ExecutionModelConfig.from_runtime() can be called by engine
        # components that receive a BacktestConfigView rather than a RuntimeConfig.
        self._rt: RuntimeConfig = runtime

        bt = runtime.backtest
        tr = runtime.trading
        rk = runtime.risk
        ex = runtime.execution
        st = runtime.storage
        lv = runtime.live
        mw = runtime.market_window
        sys_ = runtime.system
        sq = runtime.strategy_quality

        # System
        self.LOG_LEVEL = sys_.log_level

        # Trading
        self.SYMBOLS = list(tr.symbols)
        self.TIMEFRAME = str(tr.timeframe)
        self.TIMEFRAMES = list(tr.timeframes)
        self.INITIAL_CAPITAL = float(tr.initial_capital)
        self.TARGET_ALLOCATION = float(tr.target_allocation)
        self.TARGET_ALLOCATION_MODE = str(tr.target_allocation_mode)
        self.MIN_TRADE_QTY = float(tr.min_trade_qty)

        # Risk
        self.RISK_PER_TRADE = float(rk.risk_per_trade)
        self.MAX_DAILY_LOSS_PCT = float(rk.max_daily_loss_pct)
        self.MAX_TOTAL_MARGIN_PCT = float(rk.max_total_margin_pct)
        self.MAX_SYMBOL_EXPOSURE_PCT = float(rk.max_symbol_exposure_pct)
        self.MAX_ORDER_VALUE = float(rk.max_order_value)
        self.MAX_ORDER_NOTIONAL_PCT = float(rk.max_order_notional_pct)
        self.MAX_TOTAL_NOTIONAL_PCT = float(rk.max_total_notional_pct)
        self.DEFAULT_STOP_LOSS_PCT = float(rk.default_stop_loss_pct)
        self.MAX_INTRADAY_DRAWDOWN_PCT = float(rk.max_intraday_drawdown_pct)
        self.MAX_ROLLING_LOSS_PCT_1H = float(rk.max_rolling_loss_pct_1h)
        self.FREEZE_NEW_ENTRIES_ON_BREACH = bool(rk.freeze_new_entries_on_breach)
        self.AUTO_FLATTEN_ON_BREACH = bool(rk.auto_flatten_on_breach)
        self.MAX_POSITION_SIZE_PCT = float(rk.max_position_size_pct)
        self.CONSECUTIVE_LOSS_HALT_COUNT = int(rk.consecutive_loss_halt_count)
        self.HARD_DRAWDOWN_FLATTEN_PCT = float(rk.hard_drawdown_flatten_pct)
        self.ALLOW_METADATA_RISK_OVERRIDE = bool(rk.allow_metadata_risk_override)
        self.MAX_LEVERAGE = float(rk.max_leverage)

        # Execution
        self.MAKER_FEE_RATE = float(ex.maker_fee_rate)
        self.TAKER_FEE_RATE = float(ex.taker_fee_rate)
        self.SPREAD_RATE = float(ex.spread_rate)
        self.SLIPPAGE_RATE = float(bt.slippage_rate)  # backtest-specific
        self.FUNDING_RATE_PER_8H = float(ex.funding_rate_per_8h)
        self.FUNDING_INTERVAL_HOURS = int(ex.funding_interval_hours)
        self.MAINTENANCE_MARGIN_RATE = float(ex.maintenance_margin_rate)
        self.LIQUIDATION_BUFFER_RATE = float(ex.liquidation_buffer_rate)
        self.SIM_MAX_BAR_VOLUME_RATIO = float(ex.max_bar_volume_ratio)
        self.GPU_MODE = str(ex.gpu_mode)
        self.COMPUTE_BACKEND = str(ex.compute_backend)
        self.GPU_VRAM_GB = float(ex.gpu_vram_gb)

        # Storage
        self.STORAGE_BACKEND = str(st.backend)
        self.MARKET_DATA_PARQUET_PATH = str(st.market_data_parquet_path)
        self.STORAGE_MARKET_DATA_PARQUET_PATH = str(st.market_data_parquet_path)
        self.MARKET_DATA_EXCHANGE = str(st.market_data_exchange)
        self.POSTGRES_DSN = str(st.postgres_dsn)
        self.POSTGRES_DSN_ENV = str(st.postgres_dsn_env)
        self.WAL_MAX_BYTES = int(st.wal_max_bytes)
        self.WAL_COMPACT_ON_THRESHOLD = bool(st.wal_compact_on_threshold)
        self.WAL_COMPACTION_INTERVAL_SECONDS = int(st.wal_compaction_interval_seconds)
        self.COLLECTOR_PERIODIC_ENABLED = bool(st.collector_periodic_enabled)
        self.COLLECTOR_POLL_SECONDS = int(st.collector_poll_seconds)
        self.COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS = int(st.collector_bootstrap_lookback_hours)
        self.MATERIALIZER_PERIODIC_ENABLED = bool(st.materializer_periodic_enabled)
        self.MATERIALIZER_POLL_SECONDS = int(st.materializer_poll_seconds)
        self.MATERIALIZER_BASE_TIMEFRAME = str(st.materializer_base_timeframe)
        self.MATERIALIZER_REQUIRED_TIMEFRAMES = list(st.materializer_required_timeframes)

        # Backtest
        self.START_DATE = bt.start_date
        self.END_DATE = bt.end_date
        self.MODE = str(bt.mode)
        self.DATA_SOURCE = str(bt.data_source)
        self.COMMISSION_RATE = float(bt.commission_rate)
        self.ANNUAL_PERIODS = int(bt.annual_periods)
        self.RISK_FREE_RATE = float(bt.risk_free_rate)
        self.RISK_FREE_MODE = str(bt.risk_free_mode)
        self.RISK_FREE_TENOR = str(bt.risk_free_tenor)
        self.RISK_FREE_ANNUAL = float(bt.risk_free_annual)
        self.RISK_FREE_SERIES_PATH = str(bt.risk_free_series_path)
        self.SORTINO_TARGET_MODE = str(bt.sortino_target_mode)
        self.SORTINO_TARGET_ANNUAL = float(bt.sortino_target_annual)
        self.RANDOM_SEED = int(bt.random_seed)
        self.PERSIST_OUTPUT = bool(bt.persist_output)
        self.LEVERAGE = int(bt.leverage)
        self.MARGIN_MODE = str(bt.margin_mode)
        self.CHUNK_DAYS = int(bt.chunk_days)
        self.CHUNK_WARMUP_BARS = int(bt.chunk_warmup_bars)
        self.POLL_SECONDS = int(bt.poll_seconds)
        self.BACKTEST_POLL_SECONDS = int(bt.poll_seconds)
        self.WINDOW_SECONDS = int(bt.window_seconds)
        self.BACKTEST_WINDOW_SECONDS = int(bt.window_seconds)
        self.DECISION_CADENCE_SECONDS = int(bt.decision_cadence_seconds)
        self.BACKTEST_DECISION_SECONDS = int(bt.decision_cadence_seconds)
        self.SKIP_AHEAD_ENABLED = bool(bt.skip_ahead_enabled)
        self.EXTERNAL_DATA_ROOT = str(bt.external.root_path)
        self.EXTERNAL_SYMBOL_MAP = dict(bt.external.symbol_map)
        self.EXTERNAL_SOURCE_KIND = str(bt.external.source_kind)

        # Live — used by backtest CLI for auto data collection and live mode detection
        self.MARKET_TYPE = str(lv.exchange.market_type)
        self.IS_TESTNET = str(lv.mode).strip().lower() != "real"
        self.BINANCE_API_KEY = str(lv.api_key)
        self.BINANCE_SECRET_KEY = str(lv.secret_key)
        self.REQUIRE_REAL_ENABLE_FLAG = bool(lv.require_real_enable_flag)
        self.SYMBOL_LIMITS: dict = dict(lv.symbol_limits)
        # NOTE: DEFAULT_ORDER_TYPE intentionally omitted — backtest callers use
        # getattr(config, "DEFAULT_ORDER_TYPE", "MKT") so the "MKT" fallback is
        # picked up, matching the old BacktestConfig metaclass behaviour where
        # this attribute was never present on the class.

        # Market window
        self.MARKET_WINDOW_PARITY_V2_ENABLED = bool(mw.parity_v2_enabled)
        self.MARKET_WINDOW_METRICS_LOG_PATH = str(mw.metrics_log_path)

        # Strategy quality overlays
        self.STRATEGY_QUALITY_ENABLED = bool(sq.enabled)
        self.STRATEGY_QUALITY_EDGE_GATE_ENABLED = bool(sq.edge_gate_enabled)
        self.STRATEGY_QUALITY_MIN_EXPECTED_EDGE_BPS = float(sq.min_expected_edge_bps)
        self.STRATEGY_QUALITY_EDGE_COST_BUFFER_BPS = float(sq.edge_cost_buffer_bps)
        self.STRATEGY_QUALITY_ALLOW_UNKNOWN_EDGE = bool(sq.allow_unknown_edge)
        self.STRATEGY_QUALITY_REGIME_ROUTER_ENABLED = bool(sq.regime_router_enabled)
        self.STRATEGY_QUALITY_REGIME_LOOKBACK_BARS = int(sq.regime_lookback_bars)
        self.STRATEGY_QUALITY_TREND_RETURN_BPS = float(sq.trend_return_bps)
        self.STRATEGY_QUALITY_PANIC_RETURN_BPS = float(sq.panic_return_bps)
        self.STRATEGY_QUALITY_LOW_LIQUIDITY_VOLUME_RATIO = float(sq.low_liquidity_volume_ratio)
        self.STRATEGY_QUALITY_POSITION_SIZING_ENABLED = bool(sq.position_sizing_enabled)
        self.STRATEGY_QUALITY_TARGET_VOL_PER_BAR = float(sq.target_vol_per_bar)
        self.STRATEGY_QUALITY_MIN_POSITION_SCALE = float(sq.min_position_scale)
        self.STRATEGY_QUALITY_MAX_POSITION_SCALE = float(sq.max_position_scale)
        self.STRATEGY_QUALITY_TURNOVER_BUDGET_ENABLED = bool(sq.turnover_budget_enabled)
        self.STRATEGY_QUALITY_MAX_DAILY_TURNOVER_PCT = float(sq.max_daily_turnover_pct)
        self.STRATEGY_QUALITY_EXIT_OVERLAY_ENABLED = bool(sq.exit_overlay_enabled)
        self.STRATEGY_QUALITY_ATR_WINDOW_BARS = int(sq.atr_window_bars)
        self.STRATEGY_QUALITY_ATR_STOP_MULT = float(sq.atr_stop_mult)
        self.STRATEGY_QUALITY_ATR_TAKE_PROFIT_MULT = float(sq.atr_take_profit_mult)
        self.STRATEGY_QUALITY_TRAILING_ATR_MULT = float(sq.trailing_atr_mult)
        self.STRATEGY_QUALITY_MIN_STOP_BPS = float(sq.min_stop_bps)
        self.STRATEGY_QUALITY_MAX_STOP_BPS = float(sq.max_stop_bps)
        self.STRATEGY_QUALITY_STRATEGY_HEALTH_ENABLED = bool(sq.strategy_health_enabled)
        self.STRATEGY_QUALITY_HEALTH_WINDOW_TRADES = int(sq.health_window_trades)
        self.STRATEGY_QUALITY_MIN_HEALTH_SCALE = float(sq.min_health_scale)
        self.STRATEGY_QUALITY_LOSS_COOLDOWN_BARS = int(sq.loss_cooldown_bars)
        self.STRATEGY_QUALITY_PROFIT_MOONSHOT_CONFLICT_COOLDOWN_BARS = int(
            sq.profit_moonshot_conflict_cooldown_bars
        )
        self.STRATEGY_QUALITY_PAIR_MIN_CORRELATION = float(sq.pair_min_correlation)
