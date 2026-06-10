"""Engine config views — uppercase-attr adapters over RuntimeConfig (Phase 1 bridge).

The backtest engine (Backtest, Portfolio, SimulatedExecutionHandler) and the live
trader (LiveTrader, RiskManager) access uppercase config attributes inherited from
the old metaclass-backed XxxConfig classes.  These views are constructed explicitly
from a typed RuntimeConfig and expose the same uppercase interface — no metaclass,
no global state, no os.environ mutation.

Consumers and deletion schedule
---------------------------------
BacktestConfigView  — consumed by backtesting/backtest.py, cli/backtest.py
                      DELETION-GATE: Phase 4 (Backtest & validation engine overhaul)
LiveConfigView      — consumed by live/trader.py
                      DELETION-GATE: Phase 5 (Live trading path rewrite)
"""

from __future__ import annotations

from lumina_quant.configuration.schema import RuntimeConfig


# DELETION-GATE: Phase 4
class BacktestConfigView:
    """Typed config bag for backtest engine — uppercase attrs from RuntimeConfig.

    Covers BaseConfig + BacktestConfig + LiveConfig attribute sets that
    portfolio_backtest.py, execution_sim.py, and backtest.py need.

    DELETION-GATE: Phase 4 — migrate consumers to typed RuntimeConfig attrs directly.
    """

    def __init__(self, runtime: RuntimeConfig) -> None:
        bt = runtime.backtest
        tr = runtime.trading
        rk = runtime.risk
        ex = runtime.execution
        st = runtime.storage
        lv = runtime.live
        mw = runtime.market_window
        sys_ = runtime.system

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

        # Execution
        self.MAKER_FEE_RATE = float(ex.maker_fee_rate)
        self.TAKER_FEE_RATE = float(ex.taker_fee_rate)
        self.SPREAD_RATE = float(ex.spread_rate)
        self.SLIPPAGE_RATE = float(bt.slippage_rate)  # backtest-specific
        self.FUNDING_RATE_PER_8H = float(ex.funding_rate_per_8h)
        self.FUNDING_INTERVAL_HOURS = int(ex.funding_interval_hours)
        self.MAINTENANCE_MARGIN_RATE = float(ex.maintenance_margin_rate)
        self.LIQUIDATION_BUFFER_RATE = float(ex.liquidation_buffer_rate)
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


# DELETION-GATE: Phase 5
class LiveConfigView:
    """Typed config bag for live engine — uppercase attrs from RuntimeConfig.

    Covers BaseConfig + LiveConfig attribute sets that LiveTrader, RiskManager,
    and related components need.  Constructed once per run in cli/live.py and
    passed to LiveTrader; LiveTrader snapshots it via _snapshot_live_config.

    DELETION-GATE: Phase 5 — migrate consumers to typed RuntimeConfig attrs directly.
    """

    def __init__(self, runtime: RuntimeConfig) -> None:
        lv = runtime.live
        ex = lv.exchange
        pm = lv.polymarket
        ext = lv.external
        tr = runtime.trading
        rk = runtime.risk
        run_ex = runtime.execution
        st = runtime.storage
        mw = runtime.market_window
        bt = runtime.backtest
        sys_ = runtime.system

        # System
        self.LOG_LEVEL = str(sys_.log_level)

        # Trading (inherited from BaseConfig)
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

        # Execution
        self.MAKER_FEE_RATE = float(run_ex.maker_fee_rate)
        self.TAKER_FEE_RATE = float(run_ex.taker_fee_rate)
        self.SPREAD_RATE = float(run_ex.spread_rate)
        self.SLIPPAGE_RATE = float(run_ex.slippage_rate)
        self.FUNDING_RATE_PER_8H = float(run_ex.funding_rate_per_8h)
        self.FUNDING_INTERVAL_HOURS = int(run_ex.funding_interval_hours)
        self.MAINTENANCE_MARGIN_RATE = float(run_ex.maintenance_margin_rate)
        self.LIQUIDATION_BUFFER_RATE = float(run_ex.liquidation_buffer_rate)
        self.GPU_MODE = str(run_ex.gpu_mode)
        self.COMPUTE_BACKEND = str(run_ex.compute_backend)
        self.GPU_VRAM_GB = float(run_ex.gpu_vram_gb)

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

        # Risk-free (from backtest section — also referenced in live perf)
        self.RISK_FREE_MODE = str(bt.risk_free_mode)
        self.RISK_FREE_TENOR = str(bt.risk_free_tenor)
        self.RISK_FREE_ANNUAL = float(bt.risk_free_annual)
        self.RISK_FREE_SERIES_PATH = str(bt.risk_free_series_path)
        self.SORTINO_TARGET_MODE = str(bt.sortino_target_mode)
        self.SORTINO_TARGET_ANNUAL = float(bt.sortino_target_annual)

        # Market window
        self.MARKET_WINDOW_PARITY_V2_ENABLED = bool(mw.parity_v2_enabled)
        self.MARKET_WINDOW_METRICS_LOG_PATH = str(mw.metrics_log_path)

        # Live credentials
        self.BINANCE_API_KEY = str(lv.api_key)
        self.BINANCE_SECRET_KEY = str(lv.secret_key)
        self.TELEGRAM_BOT_TOKEN = lv.telegram_bot_token
        self.TELEGRAM_CHAT_ID = lv.telegram_chat_id

        # Live mode and sources
        self.MODE = str(lv.mode)
        self.IS_TESTNET = str(lv.mode).strip().lower() != "real"
        self.REQUIRE_REAL_ENABLE_FLAG = bool(lv.require_real_enable_flag)
        self.MARKET_DATA_SOURCE = str(lv.market_data_source)
        self.ORDER_STATE_SOURCE = str(lv.order_state_source)
        self.SHADOW_LIVE_ENABLED = bool(lv.shadow_live_enabled)
        self.RECONCILIATION_POLL_FALLBACK_ENABLED = bool(lv.reconciliation_poll_fallback_enabled)
        self.BOOK_TICKER_ENABLED = bool(lv.book_ticker_enabled)

        # Live order policy
        self.DEFAULT_ORDER_TYPE = str(lv.default_order_type)
        self.ALLOW_MARKET_ORDERS = bool(lv.allow_market_orders)
        self.LIMIT_PRICE_MODE = str(lv.limit_price_mode)
        self.LIMIT_PRICE_OFFSET_TICKS = int(lv.limit_price_offset_ticks)
        self.LIMIT_PRICE_TICK_FALLBACK = float(lv.limit_price_tick_fallback)
        self.LIMIT_TIME_IN_FORCE = str(lv.limit_time_in_force)
        self.PROTECTIVE_ORDER_STYLE = str(lv.protective_order_style)
        self.STARTUP_RECONCILIATION_HARD_FAIL = bool(lv.startup_reconciliation_hard_fail)
        self.MAIN_LOOP_ERROR_RETRY_LIMIT = int(lv.main_loop_error_retry_limit)
        self.MAIN_LOOP_ERROR_WINDOW_SECONDS = int(lv.main_loop_error_window_seconds)

        # Live timing
        self.POLL_SECONDS = int(lv.poll_seconds)
        self.POLL_INTERVAL = int(lv.poll_seconds)
        self.LIVE_POLL_SECONDS = int(lv.poll_seconds)
        self.WINDOW_SECONDS = int(lv.window_seconds)
        self.INGEST_WINDOW_SECONDS = int(lv.window_seconds)
        self.DECISION_CADENCE_SECONDS = int(lv.decision_cadence_seconds)
        self.MATERIALIZED_STALENESS_THRESHOLD_SECONDS = int(
            lv.materialized_staleness_threshold_seconds
        )
        self.MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = int(
            lv.materialized_staleness_alert_cooldown_seconds
        )
        self.ORDER_TIMEOUT = int(lv.order_timeout)
        self.HEARTBEAT_INTERVAL_SEC = int(lv.heartbeat_interval_sec)
        self.RECONCILIATION_INTERVAL_SEC = int(lv.reconciliation_interval_sec)

        # Exchange
        _exch_dict: dict = {
            "driver": str(ex.driver),
            "name": str(ex.name),
            "market_type": str(ex.market_type),
            "position_mode": str(ex.position_mode),
            "margin_mode": str(ex.margin_mode),
            "leverage": int(ex.leverage),
        }
        self.EXCHANGE = _exch_dict
        self.EXCHANGE_ID = str(ex.name)
        self.MARKET_TYPE = str(ex.market_type)
        self.POSITION_MODE = str(ex.position_mode)
        self.MARGIN_MODE = str(ex.margin_mode)
        self.LEVERAGE = int(ex.leverage)

        # External live data source
        self.EXTERNAL_DATA_SOURCE_KIND = str(ext.source_kind)
        self.EXTERNAL_DATA_PATH = str(ext.path)
        self.EXTERNAL_DATA_SCHEMA = str(ext.schema)
        self.EXTERNAL_DATA_SYMBOL_MAP = dict(ext.symbol_map)
        self.EXTERNAL_DATA_POLL_SECONDS = int(ext.poll_seconds)
        self.EXTERNAL_DATA_ALLOW_STALE_SECONDS = int(ext.allow_stale_seconds)

        # MT5
        self.MT5_MAGIC = int(lv.mt5_magic)
        self.MT5_DEVIATION = int(lv.mt5_deviation)
        self.MT5_BRIDGE_PYTHON = str(lv.mt5_bridge_python)
        self.MT5_BRIDGE_SCRIPT = str(lv.mt5_bridge_script)
        self.MT5_BRIDGE_USE_WSLPATH = bool(lv.mt5_bridge_use_wslpath)

        # Polymarket
        self.POLYMARKET_HOST = str(pm.host)
        self.POLYMARKET_GAMMA_HOST = str(pm.gamma_host)
        self.POLYMARKET_DATA_HOST = str(pm.data_host)
        self.POLYMARKET_MARKET_WS_URL = str(pm.market_ws_url)
        self.POLYMARKET_USER_WS_URL = str(pm.user_ws_url)
        self.POLYMARKET_CHAIN_ID = int(pm.chain_id)
        self.POLYMARKET_ASSET_IDS = list(pm.asset_ids)
        self.POLYMARKET_PRIVATE_KEY_ENV = str(pm.private_key_env)
        self.POLYMARKET_API_KEY_ENV = str(pm.api_key_env)
        self.POLYMARKET_API_SECRET_ENV = str(pm.api_secret_env)
        self.POLYMARKET_API_PASSPHRASE_ENV = str(pm.api_passphrase_env)
        self.POLYMARKET_FUNDER = str(pm.funder)
        self.POLYMARKET_SIGNATURE_TYPE = int(pm.signature_type)
        self.POLYMARKET_ALLOW_REAL_EXECUTION = bool(pm.allow_real_execution)

        # Symbol limits
        self.SYMBOL_LIMITS: dict = dict(lv.symbol_limits)
