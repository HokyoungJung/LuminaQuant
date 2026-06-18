import atexit
import os
import queue
import threading
import time
from copy import deepcopy
from types import SimpleNamespace

from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.configuration import get_default_runtime_config, validate_runtime_config
from lumina_quant.core.engine import TradingEngine
from lumina_quant.core.events import OrderEvent
from lumina_quant.core.market_window_contract import MarketWindowContractError
from lumina_quant.core.order_policy import (
    canonical_order_type,
    limit_price_for_direction,
    merge_order_policy_metadata,
    normalize_limit_price_mode,
    policy_order_type,
    price_tick_size_from_sources,
)
from lumina_quant.core.protocols import ExchangeInterface
from lumina_quant.exchanges import get_exchange
from lumina_quant.live.binance_market_stream import normalize_stream_symbol
from lumina_quant.live.readiness_policy import (
    DEFAULT_PREFLIGHT_STALE_MINUTES,
    LiveReadinessBlockedError,
    effective_startup_reconciliation_hard_fail,
    enforce_live_readiness_from_files,
)
from lumina_quant.live.binance_user_stream import BinanceUserStreamClient
from lumina_quant.live.recovery_reconciliation import RecoveryReconciliationService
from lumina_quant.risk_manager import RiskManager
from lumina_quant.runtime_cache import RuntimeCache
from lumina_quant.symbol_universe import resolve_available_symbols
from lumina_quant.utils.audit_store import AuditStore
from lumina_quant.utils.logging_utils import setup_logging
from lumina_quant.utils.notification import NotificationManager
from lumina_quant.utils.persistence import StateManager


class LiveDataFatalError(RuntimeError):
    """Raised when live data contract breach requires deterministic process exit."""


def _snapshot_live_config(base_config, *, symbols) -> SimpleNamespace:
    attrs: dict[str, object] = {}
    for name in dir(base_config):
        if name.startswith("_"):
            continue
        try:
            value = getattr(base_config, name)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            attrs[name] = deepcopy(value)
        except Exception:
            attrs[name] = value
    attrs["SYMBOLS"] = list(symbols or [])
    return SimpleNamespace(**attrs)


def _build_live_config_namespace(rt, *, symbols) -> SimpleNamespace:
    """Build the live config snapshot directly from RuntimeConfig.

    Replaces the two-step ``LiveConfigView(rt)`` → ``_snapshot_live_config(view, …)``
    pattern used prior to Phase 6 (views.py deletion gate).  All values are copies
    (primitive constructors or list/dict) so downstream deepcopy is not required.
    """
    lv = rt.live
    ex = lv.exchange
    pm = lv.polymarket
    ext = lv.external
    tr = rt.trading
    rk = rt.risk
    run_ex = rt.execution
    st = rt.storage
    mw = rt.market_window
    bt = rt.backtest
    sys_ = rt.system

    _stage = str(getattr(lv, "go_live_stage", "testnet") or "testnet").strip().lower()

    return SimpleNamespace(
        LOG_LEVEL=str(sys_.log_level),
        SYMBOLS=list(symbols or []),
        TIMEFRAME=str(tr.timeframe),
        TIMEFRAMES=list(tr.timeframes),
        INITIAL_CAPITAL=float(tr.initial_capital),
        TARGET_ALLOCATION=float(tr.target_allocation),
        TARGET_ALLOCATION_MODE=str(tr.target_allocation_mode),
        MIN_TRADE_QTY=float(tr.min_trade_qty),
        RISK_PER_TRADE=float(rk.risk_per_trade),
        MAX_DAILY_LOSS_PCT=float(rk.max_daily_loss_pct),
        MAX_TOTAL_MARGIN_PCT=float(rk.max_total_margin_pct),
        MAX_SYMBOL_EXPOSURE_PCT=float(rk.max_symbol_exposure_pct),
        MAX_ORDER_VALUE=float(rk.max_order_value),
        MAX_ORDER_NOTIONAL_PCT=float(rk.max_order_notional_pct),
        MAX_TOTAL_NOTIONAL_PCT=float(rk.max_total_notional_pct),
        DEFAULT_STOP_LOSS_PCT=float(rk.default_stop_loss_pct),
        MAX_INTRADAY_DRAWDOWN_PCT=float(rk.max_intraday_drawdown_pct),
        MAX_ROLLING_LOSS_PCT_1H=float(rk.max_rolling_loss_pct_1h),
        FREEZE_NEW_ENTRIES_ON_BREACH=bool(rk.freeze_new_entries_on_breach),
        AUTO_FLATTEN_ON_BREACH=bool(rk.auto_flatten_on_breach),
        MAX_POSITION_SIZE_PCT=float(rk.max_position_size_pct),
        CONSECUTIVE_LOSS_HALT_COUNT=int(rk.consecutive_loss_halt_count),
        HARD_DRAWDOWN_FLATTEN_PCT=float(rk.hard_drawdown_flatten_pct),
        ALLOW_METADATA_RISK_OVERRIDE=bool(rk.allow_metadata_risk_override),
        MAX_LEVERAGE=float(rk.max_leverage),
        MAKER_FEE_RATE=float(run_ex.maker_fee_rate),
        TAKER_FEE_RATE=float(run_ex.taker_fee_rate),
        SPREAD_RATE=float(run_ex.spread_rate),
        SLIPPAGE_RATE=float(run_ex.slippage_rate),
        FUNDING_RATE_PER_8H=float(run_ex.funding_rate_per_8h),
        FUNDING_INTERVAL_HOURS=int(run_ex.funding_interval_hours),
        MAINTENANCE_MARGIN_RATE=float(run_ex.maintenance_margin_rate),
        LIQUIDATION_BUFFER_RATE=float(run_ex.liquidation_buffer_rate),
        GPU_MODE=str(run_ex.gpu_mode),
        COMPUTE_BACKEND=str(run_ex.compute_backend),
        GPU_VRAM_GB=float(run_ex.gpu_vram_gb),
        STORAGE_BACKEND=str(st.backend),
        STORAGE_EXPORT_CSV=bool(st.export_csv),
        MARKET_DATA_PARQUET_PATH=str(st.market_data_parquet_path),
        STORAGE_MARKET_DATA_PARQUET_PATH=str(st.market_data_parquet_path),
        MARKET_DATA_EXCHANGE=str(st.market_data_exchange),
        POSTGRES_DSN=str(st.postgres_dsn),
        POSTGRES_DSN_ENV=str(st.postgres_dsn_env),
        WAL_MAX_BYTES=int(st.wal_max_bytes),
        WAL_COMPACT_ON_THRESHOLD=bool(st.wal_compact_on_threshold),
        WAL_COMPACTION_INTERVAL_SECONDS=int(st.wal_compaction_interval_seconds),
        COLLECTOR_PERIODIC_ENABLED=bool(st.collector_periodic_enabled),
        COLLECTOR_POLL_SECONDS=int(st.collector_poll_seconds),
        COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS=int(st.collector_bootstrap_lookback_hours),
        MATERIALIZER_PERIODIC_ENABLED=bool(st.materializer_periodic_enabled),
        MATERIALIZER_POLL_SECONDS=int(st.materializer_poll_seconds),
        MATERIALIZER_BASE_TIMEFRAME=str(st.materializer_base_timeframe),
        MATERIALIZER_REQUIRED_TIMEFRAMES=list(st.materializer_required_timeframes),
        RISK_FREE_MODE=str(bt.risk_free_mode),
        RISK_FREE_TENOR=str(bt.risk_free_tenor),
        RISK_FREE_ANNUAL=float(bt.risk_free_annual),
        RISK_FREE_SERIES_PATH=str(bt.risk_free_series_path),
        SORTINO_TARGET_MODE=str(bt.sortino_target_mode),
        SORTINO_TARGET_ANNUAL=float(bt.sortino_target_annual),
        MARKET_WINDOW_PARITY_V2_ENABLED=bool(mw.parity_v2_enabled),
        MARKET_WINDOW_METRICS_LOG_PATH=str(mw.metrics_log_path),
        BINANCE_API_KEY=str(lv.api_key),
        BINANCE_SECRET_KEY=str(lv.secret_key),
        TELEGRAM_BOT_TOKEN=lv.telegram_bot_token,
        TELEGRAM_CHAT_ID=lv.telegram_chat_id,
        MODE=str(lv.mode),
        IS_TESTNET=_stage in {"testnet", "shadow"},
        REQUIRE_REAL_ENABLE_FLAG=bool(lv.require_real_enable_flag),
        MARKET_DATA_SOURCE=str(lv.market_data_source),
        ORDER_STATE_SOURCE=str(lv.order_state_source),
        SHADOW_LIVE_ENABLED=bool(lv.shadow_live_enabled),
        RECONCILIATION_POLL_FALLBACK_ENABLED=bool(lv.reconciliation_poll_fallback_enabled),
        BOOK_TICKER_ENABLED=bool(lv.book_ticker_enabled),
        DEFAULT_ORDER_TYPE=str(lv.default_order_type),
        ALLOW_MARKET_ORDERS=bool(lv.allow_market_orders),
        LIMIT_PRICE_MODE=str(lv.limit_price_mode),
        LIMIT_PRICE_OFFSET_TICKS=int(lv.limit_price_offset_ticks),
        LIMIT_PRICE_TICK_FALLBACK=float(lv.limit_price_tick_fallback),
        LIMIT_TIME_IN_FORCE=str(lv.limit_time_in_force),
        PROTECTIVE_ORDER_STYLE=str(lv.protective_order_style),
        STARTUP_RECONCILIATION_HARD_FAIL=bool(lv.startup_reconciliation_hard_fail),
        MAIN_LOOP_ERROR_RETRY_LIMIT=int(lv.main_loop_error_retry_limit),
        MAIN_LOOP_ERROR_WINDOW_SECONDS=int(lv.main_loop_error_window_seconds),
        POLL_SECONDS=int(lv.poll_seconds),
        LIVE_POLL_SECONDS=int(lv.poll_seconds),
        WINDOW_SECONDS=int(lv.window_seconds),
        INGEST_WINDOW_SECONDS=int(lv.window_seconds),
        DECISION_CADENCE_SECONDS=int(lv.decision_cadence_seconds),
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS=int(lv.materialized_staleness_threshold_seconds),
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS=int(
            lv.materialized_staleness_alert_cooldown_seconds
        ),
        ORDER_TIMEOUT=int(lv.order_timeout),
        MAX_BBO_AGE_SECONDS=float(lv.max_bbo_age_seconds),
        HEARTBEAT_INTERVAL_SEC=int(lv.heartbeat_interval_sec),
        RECONCILIATION_INTERVAL_SEC=int(lv.reconciliation_interval_sec),
        EXCHANGE={
            "driver": str(ex.driver),
            "name": str(ex.name),
            "market_type": str(ex.market_type),
            "position_mode": str(ex.position_mode),
            "margin_mode": str(ex.margin_mode),
            "leverage": int(ex.leverage),
        },
        EXCHANGE_ID=str(ex.name),
        MARKET_TYPE=str(ex.market_type),
        POSITION_MODE=str(ex.position_mode),
        MARGIN_MODE=str(ex.margin_mode),
        LEVERAGE=int(ex.leverage),
        EXTERNAL_DATA_SOURCE_KIND=str(ext.source_kind),
        EXTERNAL_DATA_PATH=str(ext.path),
        EXTERNAL_DATA_SCHEMA=str(ext.schema),
        EXTERNAL_DATA_SYMBOL_MAP=dict(ext.symbol_map),
        EXTERNAL_DATA_POLL_SECONDS=int(ext.poll_seconds),
        EXTERNAL_DATA_ALLOW_STALE_SECONDS=int(ext.allow_stale_seconds),
        MT5_MAGIC=int(lv.mt5_magic),
        MT5_DEVIATION=int(lv.mt5_deviation),
        MT5_BRIDGE_PYTHON=str(lv.mt5_bridge_python),
        MT5_BRIDGE_SCRIPT=str(lv.mt5_bridge_script),
        MT5_BRIDGE_USE_WSLPATH=bool(lv.mt5_bridge_use_wslpath),
        POLYMARKET_HOST=str(pm.host),
        POLYMARKET_GAMMA_HOST=str(pm.gamma_host),
        POLYMARKET_DATA_HOST=str(pm.data_host),
        POLYMARKET_MARKET_WS_URL=str(pm.market_ws_url),
        POLYMARKET_USER_WS_URL=str(pm.user_ws_url),
        POLYMARKET_CHAIN_ID=int(pm.chain_id),
        POLYMARKET_ASSET_IDS=list(pm.asset_ids),
        POLYMARKET_PRIVATE_KEY_ENV=str(pm.private_key_env),
        POLYMARKET_API_KEY_ENV=str(pm.api_key_env),
        POLYMARKET_API_SECRET_ENV=str(pm.api_secret_env),
        POLYMARKET_API_PASSPHRASE_ENV=str(pm.api_passphrase_env),
        POLYMARKET_FUNDER=str(pm.funder),
        POLYMARKET_SIGNATURE_TYPE=int(pm.signature_type),
        POLYMARKET_ALLOW_REAL_EXECUTION=bool(pm.allow_real_execution),
        SYMBOL_LIMITS=dict(lv.symbol_limits),
        GO_LIVE_STAGE=_stage,
        KILL_SWITCH_ENABLED=bool(lv.kill_switch_enabled),
        CANARY_POSITION_FRACTION=float(lv.canary_position_fraction),
        SHADOW_PARITY_MIN_RATIO=float(lv.shadow_parity_min_ratio),
        SHADOW_PARITY_WINDOW_BARS=int(lv.shadow_parity_window_bars),
        EFFECTIVE_POSITION_FRACTION=(
            float(lv.canary_position_fraction) if _stage == "canary" else 1.0
        ),
        # Raw RuntimeConfig reference so LiveExecutionHandler can call
        # ExecutionModelConfig.from_runtime(config._rt, mode="live") directly.
        _rt=rt,
    )


# Module-level injection hook.  Production code keeps this as None (typed config
# path via get_default_runtime_config + validate_runtime_config is used instead).
# Tests may monkeypatch this to a fake config object to bypass validation:
#   monkeypatch.setattr("lumina_quant.live.trader.LiveConfig", fake_cfg)
LiveConfig = None


class LiveTrader(TradingEngine):
    """The LiveTrader engine."""

    def __init__(
        self,
        symbol_list,
        data_handler_cls,
        execution_handler_cls,
        portfolio_cls,
        strategy_cls,
        strategy_params=None,
        strategy_name=None,
        stop_file="",
        external_run_id="",
        runtime_config=None,
    ):
        self.logger = setup_logging("LiveTrader")
        self._audit_closed = True
        self._run_exit_status = "FAILED"
        self.run_id = None
        self.symbol_list = list(symbol_list or [])
        self.events = queue.Queue()
        if LiveConfig is not None:
            # Test-injection path: tests monkeypatch LiveConfig at module level to bypass
            # validate_runtime_config and supply a fake config object directly.
            self._config_source = LiveConfig
            self.config = LiveConfig
        else:
            _rt = runtime_config if runtime_config is not None else get_default_runtime_config()
            validate_runtime_config(_rt, for_live=True)
            self._config_source = _build_live_config_namespace(_rt, symbols=self.symbol_list)
            self.config = self._config_source
        default_strategy_name = getattr(strategy_cls, "__name__", strategy_cls.__class__.__name__)
        self.strategy_name = str(strategy_name or default_strategy_name)
        self.strategy_params = dict(strategy_params or {})
        self.stop_file = str(stop_file or "")
        self.external_run_id = str(external_run_id or "")
        self.state_manager = StateManager()
        self.risk_manager = RiskManager(self.config)
        self.heartbeat_interval_sec = max(1, int(self.config.HEARTBEAT_INTERVAL_SEC))
        self.reconciliation_interval_sec = max(
            5,
            int(self.config.RECONCILIATION_INTERVAL_SEC),
        )
        self._last_heartbeat_monotonic = time.monotonic()
        self._last_reconciliation_monotonic = 0.0
        self._last_equity_snapshot_monotonic = 0.0
        self._equity_snapshot_interval_sec = max(
            1,
            int(self.config.HEARTBEAT_INTERVAL_SEC),
        )
        self._last_drift_signature = ()
        self._last_dual_leg_signature = ()
        self._reconciliation_drift_events = 0
        self._flatten_inflight = False
        self._last_order_reconciliation_monotonic = 0.0
        self._order_reconciliation_interval_sec = max(
            5,
            int(self.config.RECONCILIATION_INTERVAL_SEC),
        )
        self.runtime_cache = RuntimeCache()
        # Reentrant lock protecting cross-thread reads/writes of the portfolio
        # position dicts (current_positions / current_position_legs).  The
        # user-stream thread (_apply_account_update) and the main loop
        # (_reconcile_positions / _sync_portfolio) mutate the same dicts and the
        # main loop reads them for sizing.  RuntimeCache has its own internal lock;
        # never held across blocking exchange I/O (snapshot first, mutate under lock).
        self._portfolio_lock = threading.RLock()
        self.order_state_source = (
            str(getattr(self.config, "ORDER_STATE_SOURCE", "polling") or "polling").strip().lower()
        )
        self.live_mode = str(getattr(self.config, "MODE", "paper") or "paper").strip().lower()
        self.market_data_source = (
            str(getattr(self.config, "MARKET_DATA_SOURCE", "committed") or "committed")
            .strip()
            .lower()
        )
        self.reconciliation_poll_fallback_enabled = bool(
            getattr(self.config, "RECONCILIATION_POLL_FALLBACK_ENABLED", True)
        )
        self.startup_reconciliation_hard_fail = effective_startup_reconciliation_hard_fail(
            mode=self.live_mode,
            configured=bool(getattr(self.config, "STARTUP_RECONCILIATION_HARD_FAIL", False)),
        )
        self.startup_reconciliation_timeout_seconds = max(
            10.0,
            float(getattr(self.config, "STARTUP_RECONCILIATION_TIMEOUT_SECONDS", 90.0) or 90.0),
        )
        self.user_stream_stale_timeout_seconds = max(
            10.0,
            float(getattr(self.config, "USER_STREAM_STALE_TIMEOUT_SECONDS", 45.0) or 45.0),
        )
        self.reconciliation_fallback_window_seconds = max(
            10.0,
            float(getattr(self.config, "RECONCILIATION_FALLBACK_WINDOW_SECONDS", 120.0) or 120.0),
        )
        self._startup_reconciliation_complete = False
        self._startup_state = "starting"
        self._startup_state_reason = ""
        self._startup_notification_sent = False
        self._user_stream_client: BinanceUserStreamClient | None = None
        self._user_stream_last_event_monotonic = time.monotonic()
        self._fallback_poll_until_monotonic = 0.0
        self._last_fallback_reason = ""
        self.main_loop_error_retry_limit = max(
            1,
            int(getattr(self.config, "MAIN_LOOP_ERROR_RETRY_LIMIT", 3) or 3),
        )
        self.main_loop_error_window_seconds = max(
            1.0,
            float(getattr(self.config, "MAIN_LOOP_ERROR_WINDOW_SECONDS", 60.0) or 60.0),
        )
        self._main_loop_error_timestamps: list[float] = []
        self.outbox_events: list[dict] = []
        self._readiness_preflight_stale_minutes = int(DEFAULT_PREFLIGHT_STALE_MINUTES)
        # Measured shadow signal-parity ratio fed to the readiness gate for the shadow
        # stage.  Fail-closed default None blocks shadow entry until an operator/ops
        # checklist injects an explicitly-measured ratio via set_measured_shadow_parity_ratio
        # (from ShadowLiveRunner.evaluate_signal_parity).  Ignored by canary/full stages.
        self._measured_shadow_parity_ratio: float | None = None

        # Initialize Notification Manager
        self.notifier = NotificationManager(
            self.config.TELEGRAM_BOT_TOKEN, self.config.TELEGRAM_CHAT_ID
        )

        try:
            # Initialize Exchange
            self.logger.info("Initializing Exchange...")
            self.exchange = get_exchange(self.config)
            self.symbol_list = self._filter_unavailable_symbols(self.symbol_list)
            if not self.symbol_list:
                raise RuntimeError(
                    "No tradable symbols are available on exchange after market sync."
                )
            self.config = _snapshot_live_config(self._config_source, symbols=self.symbol_list)
            self.risk_manager.config = self.config

            # Initialize Handlers with Exchange
            self.data_handler = data_handler_cls(
                self.events, self.symbol_list, self.config, self.exchange
            )
            self.execution_handler = execution_handler_cls(
                self.events, self.data_handler, self.config, self.exchange
            )
            if hasattr(self.execution_handler, "set_order_state_callback"):
                self.execution_handler.set_order_state_callback(self._on_order_state)

            self.portfolio = portfolio_cls(self.data_handler, self.events, time.time(), self.config)
            self.strategy = strategy_cls(self.data_handler, self.events, **self.strategy_params)
            try:
                configured_cadence = max(
                    1,
                    int(getattr(self.config, "DECISION_CADENCE_SECONDS", 20)),
                )
                if int(getattr(self.strategy, "decision_cadence_seconds", 0) or 0) <= 0:
                    self.strategy.decision_cadence_seconds = configured_cadence
            except Exception:
                pass
            self.materialized_staleness_threshold_seconds = max(
                1,
                int(
                    getattr(
                        self.config,
                        "MATERIALIZED_STALENESS_THRESHOLD_SECONDS",
                        45,
                    )
                    or 45
                ),
            )
            self.materialized_staleness_alert_cooldown_seconds = max(
                1,
                int(
                    getattr(
                        self.config,
                        "MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS",
                        60,
                    )
                    or 60
                ),
            )
            self._materialized_stale_block_active = False
            self._materialized_fresh_streak = 0
            self._materialized_last_alert_monotonic = 0.0
            self._hard_halt_active = False
            self._hard_halt_reason = ""

            self.audit_store: AuditStore = AuditStore(self.config.POSTGRES_DSN)
            self.run_id = self.audit_store.start_run(
                mode="live",
                metadata={
                    "symbols": self.symbol_list,
                    "exchange": self.config.EXCHANGE,
                    "mode": self.config.MODE,
                    "strategy": self.strategy_name,
                    "strategy_params": self.strategy_params,
                    "stop_file": self.stop_file,
                    "startup_state": self._startup_state,
                },
                run_id=self.external_run_id or None,
            )
            self._audit_closed = False
            atexit.register(self._close_audit_store)

            if self.order_state_source == "user_stream":
                self._start_user_stream()
            self._recovery_service = RecoveryReconciliationService(
                reconcile_positions=lambda: self._reconcile_positions(force=True),
                reconcile_orders=lambda: self._reconcile_orders(force=True),
                tracked_orders_count=lambda: len(
                    getattr(self.execution_handler, "tracked_orders", {}) or {}
                ),
                cache_open_orders_count=lambda: len(self.runtime_cache.open_orders or {}),
                exchange_open_orders_count=lambda: int(
                    getattr(self.execution_handler, "exchange_open_order_count", lambda: 0)()
                ),
                exchange_snapshot_ready=lambda: bool(
                    getattr(self.execution_handler, "exchange_open_snapshot_ready", lambda: False)()
                ),
                tracked_order_signature=lambda: tuple(
                    getattr(self.execution_handler, "tracked_order_signature", lambda: tuple())()
                ),
                cache_open_order_signature=self._cache_open_order_signature,
                exchange_open_order_signature=lambda: tuple(
                    getattr(
                        self.execution_handler, "exchange_open_order_signature", lambda: tuple()
                    )()
                ),
            )

            # Initialize Base Engine
            super().__init__(
                self.events,
                self.data_handler,
                self.strategy,
                self.portfolio,
                self.execution_handler,
            )

            # Load State
            self._load_state()
        except Exception as exc:
            self._set_startup_state("failed_init", error=exc)
            try:
                self._notify_startup_state()
            except Exception:
                pass
            self._close_audit_store(status="FAILED")
            raise

    @property
    def portfolio_lock(self) -> threading.RLock:
        """Lazy accessor for the portfolio-position lock.

        Lazily created so traders built via ``__new__`` in unit tests (which skip
        ``__init__``) still get a real lock instead of an AttributeError.
        """
        lock = getattr(self, "_portfolio_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._portfolio_lock = lock
        return lock

    def _filter_unavailable_symbols(self, symbols):
        requested = [str(symbol) for symbol in list(symbols or [])]

        load_markets = getattr(self.exchange, "load_markets", None)
        if not callable(load_markets):
            return requested

        try:
            markets = load_markets() or {}
        except Exception as exc:
            self.logger.warning(
                "Failed to load exchange markets for symbol availability check: %s", exc
            )
            return requested

        available = set(markets.keys()) if isinstance(markets, dict) else set()
        if not available:
            return requested

        resolved, dropped = resolve_available_symbols(requested, dict(markets))
        if dropped:
            warning = "Dropping unavailable symbols after exchange market sync: " + ", ".join(
                sorted(dropped)
            )
            self.logger.warning(warning)
            self.notifier.send_message(f"⚠️ {warning}")
        return resolved

    def _load_state(self):
        state = self.state_manager.load_state()
        if state:
            self.logger.info("Restoring state from disk...")
            try:
                if "portfolio" in state:
                    self.portfolio.set_state(state["portfolio"])
                if "strategy" in state:
                    self.strategy.set_state(state["strategy"])
                if isinstance(state.get("runtime_cache"), dict):
                    self.runtime_cache.restore(state["runtime_cache"])
                    if hasattr(self.execution_handler, "rehydrate_orders"):
                        self.execution_handler.rehydrate_orders(self.runtime_cache.open_orders)
                if isinstance(state.get("outbox_events"), list):
                    self.outbox_events = list(state["outbox_events"])
                self.logger.info("State restored.")
            except Exception as e:
                self.logger.error(f"Failed to restore state: {e}")

    def _save_state(self):
        try:
            state = {
                "portfolio": self.portfolio.get_state(),
                "strategy": self.strategy.get_state(),
                "runtime_cache": self.runtime_cache.snapshot(),
                "outbox_events": self.outbox_events[-2000:],
            }
            self.state_manager.save_state(state)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _is_stop_requested(self):
        if not self.stop_file:
            return False
        return os.path.exists(self.stop_file)

    def _close_audit_store(self, status=None):
        if self._audit_closed:
            return
        final_status = str(status or self._run_exit_status or "").strip().upper() or "FAILED"
        try:
            end_run = getattr(self.audit_store, "end_run", None)
            if self.run_id and callable(end_run):
                end_run(
                    self.run_id,
                    status=final_status,
                    metadata={
                        "startup_state": str(self._startup_state),
                        "startup_reason": str(self._startup_state_reason),
                    },
                )
        except Exception:
            pass
        try:
            close = getattr(self.audit_store, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        self._audit_closed = True

    def _set_startup_state(
        self, state: str, *, reason: str = "", error: Exception | None = None
    ) -> None:
        self._startup_state = str(state)
        self._startup_state_reason = str(reason or (str(error) if error is not None else ""))
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info(
                "LIVE_STARTUP_STATE state=%s reason=%s",
                self._startup_state,
                self._startup_state_reason or "n/a",
            )
        if self.run_id and not self._audit_closed:
            try:
                self.audit_store.log_risk_event(
                    self.run_id,
                    reason=f"STARTUP_STATE_{self._startup_state.upper()}",
                    details={
                        "reason": self._startup_state_reason,
                        "error": str(error) if error is not None else "",
                    },
                )
            except Exception:
                pass

    def _notify_startup_state(self) -> None:
        if self._startup_notification_sent:
            return
        if self._startup_state == "ready":
            message = (
                "🚀 **LuminaQuant Started**\n"
                f"Symbols: {self.symbol_list}\n"
                f"Strategy: {self.strategy_name}"
            )
        elif self._startup_state == "degraded":
            message = (
                "⚠️ **LuminaQuant Started in Degraded Mode**\n"
                f"Reason: {self._startup_state_reason or 'startup_reconciliation_timeout'}\n"
                f"Symbols: {self.symbol_list}\n"
                f"Strategy: {self.strategy_name}"
            )
        elif self._startup_state == "failed_init":
            message = (
                "🛑 **LuminaQuant Failed During Startup**\n"
                f"Reason: {self._startup_state_reason or 'initialization_error'}"
            )
        else:
            return
        self.notifier.send_message(message)
        self._startup_notification_sent = True

    def _start_user_stream(self) -> None:
        if self.order_state_source != "user_stream":
            return
        try:
            self._user_stream_client = BinanceUserStreamClient(
                exchange=self.exchange,
                market_type=str(getattr(self.config, "MARKET_TYPE", "future")),
            )
            self._user_stream_client.start(
                on_event=self._on_user_stream_event,
                on_error=self._on_user_stream_error,
            )
        except Exception as exc:
            self.logger.error("Failed to start user stream client: %s", exc)
            self.audit_store.log_risk_event(
                self.run_id,
                reason="USER_STREAM_START_ERROR",
                details={"error": str(exc)},
            )
            self._activate_fallback_polling("user_stream_start_error")
            if self.startup_reconciliation_hard_fail:
                raise

    def _on_user_stream_error(self, exc: Exception) -> None:
        self.logger.error("User stream error: %s", exc)
        self.audit_store.log_risk_event(
            self.run_id,
            reason="USER_STREAM_ERROR",
            details={"error": str(exc)},
        )
        # Stream connectivity degraded (reconnect imminent): drop any cached BBO so a
        # stale quote is not reused once the stream resumes.  The market-stream
        # reconnect is confined to the (unassigned) data handler's closures and is not
        # reachable from here, so the user-stream error is the closest available gap
        # signal owned by the trader.  invalidate_bbo_cache is a no-op safe to skip.
        self._invalidate_execution_bbo_cache()
        self._activate_fallback_polling("user_stream_error")

    def _invalidate_execution_bbo_cache(self) -> None:
        invalidate = getattr(self.execution_handler, "invalidate_bbo_cache", None)
        if callable(invalidate):
            try:
                invalidate(None)
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.error("Failed to invalidate BBO cache on stream gap: %s", exc)

    def _activate_fallback_polling(self, reason: str) -> None:
        if not self.reconciliation_poll_fallback_enabled:
            return
        now = time.monotonic()
        new_until = now + float(self.reconciliation_fallback_window_seconds)
        extended = new_until > float(self._fallback_poll_until_monotonic)
        self._fallback_poll_until_monotonic = max(
            float(self._fallback_poll_until_monotonic),
            float(new_until),
        )
        if extended and str(reason) != str(self._last_fallback_reason):
            self._last_fallback_reason = str(reason)
            self.audit_store.log_risk_event(
                self.run_id,
                reason="POLL_FALLBACK_ACTIVATED",
                details={
                    "trigger": str(reason),
                    "fallback_window_seconds": float(self.reconciliation_fallback_window_seconds),
                },
            )

    def _relinquish_fallback_polling(self) -> None:
        """Close any active fallback-poll window once authoritative user-stream
        order events resume, so the polling path stops mutating order state that
        the user-stream thread now owns (race avoidance, fix/audit-hardening).
        """
        if float(self._fallback_poll_until_monotonic) <= 0.0:
            return
        self._fallback_poll_until_monotonic = 0.0
        self._last_fallback_reason = ""

    def _is_fallback_polling_required(self) -> bool:
        if self.order_state_source != "user_stream":
            return True
        if not self.reconciliation_poll_fallback_enabled:
            return False
        now = time.monotonic()
        stale = (now - float(self._user_stream_last_event_monotonic)) > float(
            self.user_stream_stale_timeout_seconds
        )
        if stale:
            self._activate_fallback_polling("user_stream_stale")
        return now <= float(self._fallback_poll_until_monotonic)

    def _on_user_stream_event(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        self._user_stream_last_event_monotonic = time.monotonic()
        event_type = str(payload.get("event_type") or "").strip()
        self.runtime_cache.update_stream_state(
            {
                "last_event_type": event_type,
                "exchange_ts_ms": payload.get("exchange_ts_ms"),
                "received_at_ms": int(time.time() * 1000),
            }
        )
        self._append_outbox("user_stream", payload)
        if event_type == "executionReport" and hasattr(
            self.execution_handler, "ingest_user_stream_event"
        ):
            # Authoritative order event resumed: relinquish the fallback-poll window
            # so the polling path no longer mutates the same order state concurrently.
            self._relinquish_fallback_polling()
            self.execution_handler.ingest_user_stream_event(payload)
            return

        if event_type == "listenKeyExpired":
            self.audit_store.log_risk_event(
                self.run_id,
                reason="USER_STREAM_LISTEN_KEY_EXPIRED",
                details={"exchange_ts_ms": payload.get("exchange_ts_ms")},
            )
            # listenKeyExpired forces a stream reconnect (gap): drop the cached BBO so a
            # stale quote cannot be reused on resume.
            self._invalidate_execution_bbo_cache()
            self._activate_fallback_polling("listen_key_expired")
            return

        if event_type == "outboundAccountPosition":
            balances = list(payload.get("balances") or [])
            account_snapshot = {"balances": balances}
            self.runtime_cache.update_account(account_snapshot)
            return

        if event_type == "accountUpdate":
            self._apply_account_update(payload)
            return

        if event_type == "balanceUpdate":
            account_snapshot = dict(self.runtime_cache.account or {})
            updates = list(account_snapshot.get("balance_updates") or [])
            updates.append(
                {
                    "asset": payload.get("asset"),
                    "delta": payload.get("balance_delta"),
                    "clear_time_ms": payload.get("clear_time_ms"),
                    "exchange_ts_ms": payload.get("exchange_ts_ms"),
                }
            )
            if len(updates) > 200:
                updates = updates[-200:]
            account_snapshot["balance_updates"] = updates
            self.runtime_cache.update_account(account_snapshot)

    def _apply_account_update(self, payload: dict) -> None:
        balances = [dict(item or {}) for item in list(payload.get("balances") or [])]
        positions = [dict(item or {}) for item in list(payload.get("positions") or [])]

        account_snapshot = dict(self.runtime_cache.account or {})
        account_snapshot["balances"] = balances
        account_snapshot["reason"] = payload.get("reason")
        account_snapshot["exchange_ts_ms"] = payload.get("exchange_ts_ms")
        self.runtime_cache.update_account(account_snapshot)

        net_positions: dict[str, float] = {}
        legs: dict[str, dict[str, float]] = {}
        for item in positions:
            symbol = normalize_stream_symbol(str(item.get("symbol") or item.get("s") or ""))
            if not symbol:
                continue
            qty = float(item.get("position_amount") or item.get("pa") or 0.0)
            side = str(item.get("position_side") or item.get("ps") or "").upper()

            net_positions[symbol] = float(net_positions.get(symbol, 0.0)) + float(qty)
            current = legs.setdefault(symbol, {"LONG": 0.0, "SHORT": 0.0})
            if side == "LONG":
                current["LONG"] += abs(float(qty))
            elif side == "SHORT":
                current["SHORT"] += abs(float(qty))
            elif qty > 0:
                current["LONG"] += abs(float(qty))
            elif qty < 0:
                current["SHORT"] += abs(float(qty))

        if net_positions:
            self.runtime_cache.update_positions(net_positions)
            self.runtime_cache.update_position_legs(legs)
            portfolio = getattr(self, "portfolio", None)
            if portfolio is not None:
                with self.portfolio_lock:
                    portfolio.current_position_legs = dict(legs)
                    for symbol, qty in net_positions.items():
                        if symbol in portfolio.current_positions:
                            portfolio.current_positions[symbol] = float(qty)

    def _on_order_state(self, state_payload):
        self.runtime_cache.update_order_state(state_payload.get("order_id"), state_payload)
        self._append_outbox("order_state", state_payload)
        try:
            self.audit_store.log_order_state(self.run_id, state_payload)
        except Exception as exc:
            self.logger.error("Failed to persist order state event: %s", exc)
            return
        self._save_state()

        state = str(state_payload.get("state", "")).upper()
        if state not in {"REJECTED", "TIMEOUT", "CANCELED"}:
            return
        details = {
            "state": state,
            "symbol": state_payload.get("symbol"),
            "client_order_id": state_payload.get("client_order_id"),
            "exchange_order_id": state_payload.get("order_id"),
            "message": state_payload.get("message"),
            "metadata": state_payload.get("metadata") or {},
        }
        self.audit_store.log_risk_event(
            self.run_id,
            reason=f"ORDER_{state}",
            details=details,
        )
        msg = f"⚠️ **Order {state}** {details['symbol']} (client_id={details['client_order_id']})"
        self.notifier.send_message(msg)

    def on_fill(self, event):
        """Hook from TradingEngine to save state on fill."""
        self._append_outbox(
            "fill",
            {
                "symbol": event.symbol,
                "direction": event.direction,
                "quantity": event.quantity,
                "fill_cost": event.fill_cost,
                "status": event.status,
                "order_id": event.order_id,
                "timestamp_ns": getattr(event, "timestamp_ns", None),
                "sequence": getattr(event, "sequence", None),
            },
        )
        msg = f"✅ **FILL**: {event.direction} {event.quantity} {event.symbol} @ {event.fill_cost}"
        self.logger.info(msg)
        self.notifier.send_message(msg)
        self.audit_store.log_fill(self.run_id, event)
        self._save_state()

    def _cache_open_order_signature(self) -> tuple[tuple[str, ...], ...]:
        rows: list[tuple[str, ...]] = []
        for order_id, payload in sorted(dict(self.runtime_cache.open_orders or {}).items()):
            if not isinstance(payload, dict):
                continue
            rows.append(
                (
                    str(order_id),
                    str(payload.get("state") or ""),
                    str(round(float(payload.get("last_filled") or 0.0), 8)),
                    str(payload.get("client_order_id") or ""),
                )
            )
        return tuple(rows)

    def _best_available_position_legs(
        self, exchange_position_legs: dict[str, dict[str, float]] | None = None
    ) -> dict[str, dict[str, float]]:
        if isinstance(exchange_position_legs, dict) and exchange_position_legs:
            return dict(exchange_position_legs)
        cached_legs = dict(getattr(self.runtime_cache, "position_legs", {}) or {})
        if cached_legs:
            return cached_legs
        return dict(getattr(self.portfolio, "current_position_legs", {}) or {})

    def _append_outbox(self, event_type: str, payload: dict) -> None:
        item = {
            "event_type": str(event_type),
            "event_time": time.time(),
            "payload": dict(payload or {}),
        }
        self.outbox_events.append(item)
        if len(self.outbox_events) > 5000:
            self.outbox_events = self.outbox_events[-5000:]

    def _emit_heartbeat(self, force=False):
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_heartbeat_monotonic) < self.heartbeat_interval_sec:
            return
        self._last_heartbeat_monotonic = now_mono
        details = {"queue_size": self.events.qsize()}
        if hasattr(self.execution_handler, "tracked_orders"):
            details["tracked_orders"] = len(self.execution_handler.tracked_orders)
        details["reconciliation_drift_events"] = self._reconciliation_drift_events
        self.audit_store.log_heartbeat(self.run_id, status="ALIVE", details=details)

    def _reconcile_positions(self, force=False):
        now_mono = time.monotonic()
        if (
            not force
            and (now_mono - self._last_reconciliation_monotonic) < self.reconciliation_interval_sec
        ):
            return
        self._last_reconciliation_monotonic = now_mono

        try:
            exchange_positions = self.exchange.get_all_positions()
        except Exception as exc:
            self.logger.error("Reconciliation failed to fetch exchange positions: %s", exc)
            self.audit_store.log_risk_event(
                self.run_id,
                reason="RECONCILIATION_ERROR",
                details={"error": str(exc)},
            )
            return

        hedge_mode = str(getattr(self.config, "POSITION_MODE", "")).upper() == "HEDGE"
        exchange_position_legs = {}
        if hasattr(self.exchange, "get_all_position_legs"):
            try:
                exchange_position_legs = self.exchange.get_all_position_legs() or {}
            except Exception as exc:
                self.logger.error("Failed to fetch position legs for reconciliation: %s", exc)
                self.audit_store.log_risk_event(
                    self.run_id,
                    reason="POSITION_LEG_RECONCILIATION_ERROR",
                    details={"error": str(exc)},
                )

        self.runtime_cache.update_positions(exchange_positions)
        # Snapshot the local positions and apply leg mutations atomically under the
        # portfolio lock so the user-stream thread's _apply_account_update cannot
        # interleave.  Audit/notifier I/O below uses the snapshot, never the lock.
        with self.portfolio_lock:
            local = dict(self.portfolio.current_positions or {})
            if exchange_position_legs:
                effective_position_legs = self._best_available_position_legs(exchange_position_legs)
                self.runtime_cache.update_position_legs(effective_position_legs)
                self.portfolio.current_position_legs = dict(effective_position_legs)
            elif hedge_mode:
                effective_position_legs = self._best_available_position_legs()
            else:
                effective_position_legs = {}
                self.runtime_cache.update_position_legs({})
                self.portfolio.current_position_legs = {}

        if hedge_mode:
            dual_leg_signature = tuple(
                sorted(
                    (
                        symbol,
                        round(float(legs.get("LONG", 0.0)), 8),
                        round(float(legs.get("SHORT", 0.0)), 8),
                    )
                    for symbol, legs in effective_position_legs.items()
                    if float(legs.get("LONG", 0.0)) > 0.0 and float(legs.get("SHORT", 0.0)) > 0.0
                )
            )
            if dual_leg_signature and dual_leg_signature != self._last_dual_leg_signature:
                self._last_dual_leg_signature = dual_leg_signature
                self.audit_store.log_risk_event(
                    self.run_id,
                    reason="HEDGE_DUAL_LEG_DETECTED",
                    details={"legs": effective_position_legs},
                )
            if not dual_leg_signature:
                self._last_dual_leg_signature = ()

        drift = []
        for symbol in self.symbol_list:
            local_qty = float(local.get(symbol, 0.0))
            exchange_qty = float(exchange_positions.get(symbol, 0.0))
            delta = exchange_qty - local_qty
            if abs(delta) > 1e-9:
                drift.append(
                    {
                        "symbol": symbol,
                        "local_qty": local_qty,
                        "exchange_qty": exchange_qty,
                        "delta": delta,
                    }
                )

        signature = tuple(
            sorted(
                (
                    item["symbol"],
                    round(item["local_qty"], 8),
                    round(item["exchange_qty"], 8),
                )
                for item in drift
            )
        )
        if drift and signature != self._last_drift_signature:
            self._last_drift_signature = signature
            self._reconciliation_drift_events += 1
            self.audit_store.log_risk_event(
                self.run_id,
                reason="RECONCILIATION_DRIFT",
                details={
                    "drift_count": len(drift),
                    "drift": drift,
                },
            )
            self.notifier.send_message(
                f"⚠️ **Reconciliation Drift** detected for {len(drift)} symbol(s)."
            )
        if not drift:
            self._last_drift_signature = ()

    def _reconcile_orders(self, force=False):
        now_mono = time.monotonic()
        if (
            not force
            and (now_mono - self._last_order_reconciliation_monotonic)
            < self._order_reconciliation_interval_sec
        ):
            return
        self._last_order_reconciliation_monotonic = now_mono

        if hasattr(self.execution_handler, "reconcile_open_orders"):
            records = self.execution_handler.reconcile_open_orders() or []
            for item in records:
                try:
                    self.audit_store.log_order_reconciliation(self.run_id, item)
                except Exception as exc:
                    self.logger.error("Failed to persist order reconciliation event: %s", exc)

    def _set_trade_freeze(self, *, enabled, reason, details=None):
        self.portfolio.trading_frozen = bool(enabled)
        event_reason = "TRADE_FREEZE_ON" if enabled else "TRADE_FREEZE_OFF"
        self.audit_store.log_risk_event(
            self.run_id,
            reason=event_reason,
            details={"reason": reason, **dict(details or {})},
        )
        status = "ON" if enabled else "OFF"
        self.notifier.send_message(f"⚠️ **Trade Freeze {status}**: {reason}")

    def _market_spec_for_order(self, symbol):
        if hasattr(self.data_handler, "get_market_spec"):
            try:
                spec = self.data_handler.get_market_spec(symbol) or {}
                if isinstance(spec, dict) and spec:
                    return dict(spec)
            except Exception:
                pass
        if hasattr(self.exchange, "get_market_spec"):
            try:
                spec = self.exchange.get_market_spec(symbol) or {}
                if isinstance(spec, dict):
                    return dict(spec)
            except Exception:
                pass
        return {}

    def _latest_limit_reference_price(self, symbol):
        if hasattr(self.data_handler, "get_latest_bar_value"):
            try:
                price = float(self.data_handler.get_latest_bar_value(symbol, "close") or 0.0)
                if price > 0.0:
                    return price
            except Exception:
                pass
        return 0.0

    def _reduce_only_order_policy_fields(self, *, symbol, direction):
        requested = getattr(self.config, "DEFAULT_ORDER_TYPE", "MKT")
        order_type = policy_order_type(
            requested,
            default=getattr(self.config, "DEFAULT_ORDER_TYPE", "MKT"),
            allow_market_orders=bool(getattr(self.config, "ALLOW_MARKET_ORDERS", True)),
        )
        if canonical_order_type(order_type, default="LMT") != "LMT":
            return order_type, None, None, {}

        reference_price = self._latest_limit_reference_price(symbol)
        if reference_price <= 0.0:
            self.logger.error(
                "Refusing reduce-only LIMIT order for %s without a positive reference price.",
                symbol,
            )
            return None, None, None, {}
        market_spec = self._market_spec_for_order(symbol)
        tick_size = price_tick_size_from_sources(
            symbol,
            market_spec=market_spec,
            config=self.config,
        )
        mode = normalize_limit_price_mode(
            getattr(self.config, "LIMIT_PRICE_MODE", "one_tick_worse")
        )
        offset_ticks = int(getattr(self.config, "LIMIT_PRICE_OFFSET_TICKS", 1) or 0)
        price = limit_price_for_direction(
            reference_price=reference_price,
            direction=direction,
            tick_size=tick_size,
            mode=mode,
            offset_ticks=offset_ticks,
        )
        time_in_force = str(getattr(self.config, "LIMIT_TIME_IN_FORCE", "GTC") or "GTC").upper()
        metadata = merge_order_policy_metadata(
            None,
            {
                "source": "live_trader_reduce_only_flatten",
                "default_order_type": str(getattr(self.config, "DEFAULT_ORDER_TYPE", "MKT")),
                "resolved_order_type": "LMT",
                "direction": str(direction),
                "limit_price_mode": mode,
                "limit_price_offset_ticks": offset_ticks,
                "limit_reference_price": float(reference_price),
                "limit_price": float(price),
                "price_tick_size": float(tick_size or 0.0),
                "time_in_force": time_in_force,
            },
        )
        return order_type, price, time_in_force, metadata

    def _queue_reduce_only_order(self, *, symbol, quantity, direction, position_side):
        if float(quantity) <= 1e-12:
            return False
        order_type, price, time_in_force, metadata = self._reduce_only_order_policy_fields(
            symbol=symbol,
            direction=direction,
        )
        if not order_type:
            return False
        event = OrderEvent(
            symbol=symbol,
            order_type=order_type,
            quantity=abs(float(quantity)),
            direction=direction,
            price=price,
            position_side=position_side,
            reduce_only=True,
            time_in_force=time_in_force,
            metadata=metadata,
        )
        self.events.put(event)
        return True

    def _flatten_all_positions(self, *, reason, details=None):
        orders_sent = 0
        legs = {}
        hedge_mode = str(getattr(self.config, "POSITION_MODE", "")).upper() == "HEDGE"
        if hasattr(self.exchange, "get_all_position_legs"):
            try:
                legs = self.exchange.get_all_position_legs() or {}
            except Exception as exc:
                self.logger.error("Flatten could not fetch position legs: %s", exc)

        effective_legs = self._best_available_position_legs(
            legs if isinstance(legs, dict) else None
        )

        for symbol in self.symbol_list:
            payload = effective_legs.get(symbol) if isinstance(effective_legs, dict) else None
            if not isinstance(payload, dict):
                continue
            long_qty = float(payload.get("LONG", 0.0) or 0.0)
            short_qty = float(payload.get("SHORT", 0.0) or 0.0)
            if self._queue_reduce_only_order(
                symbol=symbol,
                quantity=long_qty,
                direction="SELL",
                position_side="LONG",
            ):
                orders_sent += 1
            if self._queue_reduce_only_order(
                symbol=symbol,
                quantity=short_qty,
                direction="BUY",
                position_side="SHORT",
            ):
                orders_sent += 1

        if orders_sent > 0:
            self.audit_store.log_risk_event(
                self.run_id,
                reason="FLATTEN_ALL_TRIGGERED",
                details={"reason": reason, "orders_sent": orders_sent, **dict(details or {})},
            )
            self.notifier.send_message(
                f"🛑 **Flatten All Triggered**: {reason} (orders={orders_sent})"
            )
            return orders_sent

        if hedge_mode:
            block_details = {"reason": reason, **dict(details or {})}
            self.audit_store.log_risk_event(
                self.run_id,
                reason="FLATTEN_ALL_BLOCKED_MISSING_LEGS",
                details=block_details,
            )
            self.notifier.send_message(
                f"⛔ **Flatten blocked**: {reason} (hedge legs unavailable; keeping freeze active)"
            )
            return 0

        for symbol in self.symbol_list:
            qty = float(self.portfolio.current_positions.get(symbol, 0.0))
            if self._queue_reduce_only_order(
                symbol=symbol,
                quantity=abs(qty),
                direction="SELL" if qty > 0 else "BUY",
                position_side="LONG" if qty > 0 else "SHORT",
            ):
                orders_sent += 1

        self.audit_store.log_risk_event(
            self.run_id,
            reason="FLATTEN_ALL_TRIGGERED",
            details={"reason": reason, "orders_sent": orders_sent, **dict(details or {})},
        )
        self.notifier.send_message(f"🛑 **Flatten All Triggered**: {reason} (orders={orders_sent})")
        return orders_sent

    def _evaluate_risk_guards(self):
        passed, reason, action, details = self.risk_manager.evaluate_portfolio_risk(self.portfolio)
        if passed:
            self._flatten_inflight = False
            if getattr(self.portfolio, "trading_frozen", False):
                self._set_trade_freeze(enabled=False, reason="risk_recovered", details=details)
            return

        if action == "FREEZE":
            if not getattr(self.portfolio, "trading_frozen", False):
                self._set_trade_freeze(enabled=True, reason=reason, details=details)
            return

        if action == "FLATTEN":
            if not getattr(self.portfolio, "trading_frozen", False):
                self._set_trade_freeze(enabled=True, reason=reason, details=details)
            if not self._flatten_inflight:
                orders_sent = self._flatten_all_positions(reason=reason, details=details)
                self._flatten_inflight = orders_sent > 0
            return

        self.audit_store.log_risk_event(
            self.run_id,
            reason="RISK_POLICY_BREACH",
            details={"reason": reason, **dict(details or {})},
        )

    def _sync_portfolio(self):
        """Syncs the internal portfolio state with the exchange state."""
        if isinstance(self.exchange, ExchangeInterface):
            self.logger.info("Syncing Portfolio with Exchange...")

            try:
                # 1. Sync Cash
                balance = self.exchange.get_balance("USDT")
                if balance > 0:
                    self.logger.info(f"Exchange USDT Balance: {balance}")
                    self.portfolio.current_holdings["cash"] = balance
                    self.runtime_cache.update_account({"cash": float(balance)})
                    # If total is 0 (first run), init with balance.
                    if self.portfolio.current_holdings["total"] == self.portfolio.initial_capital:
                        self.portfolio.initial_capital = balance

                # 2. Sync Positions
                exchange_positions = self.exchange.get_all_positions()
                self.logger.info(f"Exchange Positions: {exchange_positions}")

                with self.portfolio_lock:
                    for s, qty in exchange_positions.items():
                        if s in self.symbol_list:
                            self.portfolio.current_positions[s] = qty

                # 3. Recalculate Total Holdings (Cash + Position Value)
                total_equity = self.portfolio.current_holdings["cash"]
                for s in self.symbol_list:
                    qty = self.portfolio.current_positions.get(s, 0.0)
                    if qty != 0:
                        last_price = self.data_handler.get_latest_bar_value(s, "close")
                        # If price is 0 (no data yet), we can't value it perfectly.
                        if last_price > 0:
                            total_equity += qty * last_price

                self.portfolio.current_holdings["total"] = total_equity

                # 4. Reconcile Strategy State with Portfolio State
                if hasattr(self.strategy, "bought"):
                    for s in self.symbol_list:
                        position_qty = self.portfolio.current_positions.get(s, 0)
                        strategy_status = self.strategy.bought.get(s, "OUT")

                        if strategy_status != "OUT" and position_qty == 0:
                            msg = f"⚠️ **State Mismatch**: {s} Strategy={strategy_status}, Portfolio=0. Resetting Strategy to OUT."
                            self.logger.warning(msg)
                            self.notifier.send_message(msg)
                            self.strategy.bought[s] = "OUT"
                            self.audit_store.log_risk_event(
                                self.run_id,
                                reason="STATE_MISMATCH",
                                details={
                                    "symbol": s,
                                    "strategy_status": strategy_status,
                                    "position_qty": position_qty,
                                },
                            )
                        elif strategy_status == "OUT" and position_qty != 0:
                            msg = f"⚠️ **State Mismatch**: {s} Strategy=OUT, Portfolio={position_qty}. Syncing Strategy."
                            self.logger.warning(msg)
                            self.notifier.send_message(msg)
                            self.strategy.bought[s] = "LONG" if position_qty > 0 else "SHORT"
                            self.audit_store.log_risk_event(
                                self.run_id,
                                reason="STATE_MISMATCH",
                                details={
                                    "symbol": s,
                                    "strategy_status": strategy_status,
                                    "position_qty": position_qty,
                                },
                            )

                self.logger.info(f"Portfolio Sync Completed. Total Equity: {total_equity}")
            except Exception as e:
                self.logger.error(f"Portfolio Sync Failed: {e}")
                self.audit_store.log_risk_event(
                    self.run_id,
                    reason="EXCHANGE_SYNC_ERROR",
                    details={"error": str(e)},
                )

    def handle_market_event(self, event):
        """Override to save equity curve on every bar."""
        self.runtime_cache.update_market(
            event.symbol,
            {
                "time": event.time,
                "timestamp_ns": getattr(event, "timestamp_ns", None),
                "sequence": getattr(event, "sequence", None),
                "close": float(event.close),
                "volume": float(event.volume),
            },
        )
        super().handle_market_event(event)

        # Save Live Equity snapshot periodically to reduce per-bar overhead.
        now_mono = time.monotonic()
        should_snapshot = (
            now_mono - self._last_equity_snapshot_monotonic >= self._equity_snapshot_interval_sec
        )
        if should_snapshot:
            self._last_equity_snapshot_monotonic = now_mono
            self.portfolio.create_equity_curve_dataframe()
            if self.config.STORAGE_EXPORT_CSV:
                self.portfolio.save_equity_curve(os.path.join("data", "live_equity.csv"))
        self.audit_store.log_equity(
            self.run_id,
            timeindex=event.time,
            total=self.portfolio.current_holdings.get("total", 0.0),
            cash=self.portfolio.current_holdings.get("cash", 0.0),
            metadata={"symbol": event.symbol},
        )

    def handle_fill_event(self, event):
        """Override to save trades on every fill."""
        # Snapshot pre-fill portfolio state for realized-PnL computation.
        # Must happen before super() which calls update_positions_from_fill and
        # overwrites entry_prices.  The pre-fill QUANTITY (which tells us this is a
        # closing/reducing fill) is captured independently of the entry PRICE so a
        # failure to read the entry price still lets the kill-switch advance
        # conservatively below.
        _old_qty = 0.0
        _old_qty_ok = True
        try:
            with self.portfolio_lock:
                _old_qty = float((self.portfolio.current_positions or {}).get(event.symbol, 0.0))
        except Exception as exc:
            _old_qty_ok = False
            self.logger.error("Pre-fill position snapshot failed for %s: %s", event.symbol, exc)
            self._log_risk_event_safe(
                reason="FILL_POSITION_SNAPSHOT_ERROR",
                details={"symbol": getattr(event, "symbol", None), "error": str(exc)},
            )

        _entry_price = None
        _entry_price_ok = True
        try:
            _entry_price = dict(getattr(self.portfolio, "entry_prices", {}) or {}).get(event.symbol)
        except Exception as exc:
            # NEVER silent: a failed entry-price snapshot must not silently bypass the
            # consecutive-loss kill-switch.  Log + audit so a closing fill below can
            # still record a loss conservatively (fail-closed for the kill-switch).
            _entry_price_ok = False
            self.logger.error("Pre-fill entry-price snapshot failed for %s: %s", event.symbol, exc)
            self._log_risk_event_safe(
                reason="FILL_ENTRY_SNAPSHOT_ERROR",
                details={"symbol": getattr(event, "symbol", None), "error": str(exc)},
            )

        super().handle_fill_event(event)  # This calls self.on_fill(event) too

        # Wire Phase-5 consecutive-loss kill-switch: record_loss on every closing/
        # reducing fill so check_order can halt new entries after N consecutive losses.
        # Decoupled from the snapshot: a snapshot failure must not skip the loss.
        is_closing_sell = _old_qty > 1e-12 and event.direction == "SELL"
        is_closing_buy = _old_qty < -1e-12 and event.direction == "BUY"
        is_closing_fill = _old_qty_ok and (is_closing_sell or is_closing_buy)
        try:
            fill_qty = float(event.quantity or 0.0)
            fill_cost = float(event.fill_cost or 0.0)
            commission = float(event.commission or 0.0)
            if (
                is_closing_fill
                and _entry_price_ok
                and _entry_price is not None
                and fill_qty > 0.0
                and fill_cost > 0.0
            ):
                fill_price = fill_cost / fill_qty
                if is_closing_sell:
                    # Closing/reducing a long position
                    closed_qty = min(fill_qty, _old_qty)
                    realized_pnl = (fill_price - _entry_price) * closed_qty - commission
                    self.risk_manager.record_loss(realized_pnl=realized_pnl)
                else:
                    # Closing/reducing a short position
                    closed_qty = min(fill_qty, abs(_old_qty))
                    realized_pnl = (_entry_price - fill_price) * closed_qty - commission
                    self.risk_manager.record_loss(realized_pnl=realized_pnl)
            elif is_closing_fill and not _entry_price_ok:
                # The entry-price snapshot was unavailable for a closing/reducing
                # fill: record a conservative loss so the kill-switch advances rather
                # than being silently bypassed (realized PnL unknown -> treat as loss).
                self.risk_manager.record_loss(realized_pnl=-1.0)
                self.logger.warning(
                    "Recorded conservative loss for %s closing fill (entry snapshot unavailable).",
                    event.symbol,
                )
        except Exception as exc:
            # NEVER silent: a record_loss failure must be surfaced, not swallowed,
            # so the operator can see the kill-switch did not advance.
            self.logger.error("Consecutive-loss record_loss failed for %s: %s", event.symbol, exc)
            self._log_risk_event_safe(
                reason="RECORD_LOSS_ERROR",
                details={"symbol": getattr(event, "symbol", None), "error": str(exc)},
            )

        # Save Live Trades
        if self.config.STORAGE_EXPORT_CSV:
            self.portfolio.output_trade_log(os.path.join("data", "live_trades.csv"))

    def _log_risk_event_safe(self, *, reason: str, details: dict) -> None:
        """Best-effort audit log that never raises (used inside fill error paths)."""
        try:
            self.audit_store.log_risk_event(self.run_id, reason=reason, details=details)
        except Exception:  # pragma: no cover - defensive audit guard
            pass

    def _resolve_market_window_staleness(self, event) -> tuple[bool, int]:
        threshold_ms = int(self.materialized_staleness_threshold_seconds) * 1000
        lag_ms = int(getattr(event, "lag_ms", 0) or 0)
        watermark_ms = getattr(event, "event_time_watermark_ms", None)
        if lag_ms <= 0 and watermark_ms is not None:
            try:
                lag_ms = max(0, int(time.time() * 1000) - int(watermark_ms))
            except Exception:
                lag_ms = 0
        stale_flag = bool(getattr(event, "is_stale", False))
        return (stale_flag or lag_ms > threshold_ms), int(lag_ms)

    def _handle_market_window_staleness(self, event) -> bool:
        is_stale, lag_ms = self._resolve_market_window_staleness(event)
        commit_id = str(getattr(event, "commit_id", "") or "")
        details = {
            "symbol": ",".join(sorted((event.bars_1s or {}).keys())),
            "timeframe": "1s",
            "lag_ms": int(lag_ms),
            "commit_id": commit_id,
        }
        if is_stale:
            self._materialized_stale_block_active = True
            self._materialized_fresh_streak = 0
            self.logger.error("LIVE_DATA_STALE %s", details)
            now_mono = time.monotonic()
            if (
                now_mono - self._materialized_last_alert_monotonic
                >= self.materialized_staleness_alert_cooldown_seconds
            ):
                self._materialized_last_alert_monotonic = now_mono
                self.notifier.send_message(
                    "⛔ **Live data stale** "
                    f"(lag_ms={lag_ms}, commit_id={commit_id or 'n/a'}). "
                    "Decisions and new orders are blocked until feed recovers."
                )
            return True

        if not self._materialized_stale_block_active:
            return False

        self._materialized_fresh_streak += 1
        if int(self._materialized_fresh_streak) < 2:
            return True

        self._materialized_stale_block_active = False
        self._materialized_fresh_streak = 0
        self.logger.info("LIVE_DATA_RECOVERED %s", details)
        self.notifier.send_message(
            "✅ **Live data recovered**. Trading resumed after two fresh windows."
        )
        return False

    def _consume_data_fatal(self):
        consume = getattr(self.data_handler, "consume_fatal_error", None)
        if not callable(consume):
            return None
        return consume()

    def _trigger_hard_halt(self, *, reason: str, error: Exception) -> None:
        if self._hard_halt_active:
            return
        self._hard_halt_active = True
        self._hard_halt_reason = str(reason)
        self.portfolio.trading_frozen = True
        details = {
            "reason": str(reason),
            "error_type": error.__class__.__name__,
            "error": str(error),
        }
        log_critical = getattr(self.logger, "critical", None) or getattr(self.logger, "error", None)
        if callable(log_critical):
            log_critical("LIVE_DATA_FAIL_FAST %s", details)
        self.audit_store.log_risk_event(
            self.run_id,
            reason="LIVE_DATA_FAIL_FAST",
            details=details,
        )
        self.notifier.send_message(
            "🛑 **Live data fail-fast triggered**. "
            f"reason={reason}. Strategy/order paths are halted until restart."
        )

    def _register_main_loop_error(self, error: Exception) -> bool:
        now = time.monotonic()
        window = float(self.main_loop_error_window_seconds)
        self._main_loop_error_timestamps = [
            stamp
            for stamp in list(self._main_loop_error_timestamps)
            if (now - float(stamp)) <= window
        ]
        self._main_loop_error_timestamps.append(now)
        error_count = len(self._main_loop_error_timestamps)
        self.audit_store.log_risk_event(
            self.run_id,
            reason="MAIN_LOOP_ERROR",
            details={
                "error": str(error),
                "error_type": error.__class__.__name__,
                "error_count_within_window": int(error_count),
                "error_window_seconds": float(window),
                "retry_limit": int(self.main_loop_error_retry_limit),
            },
        )
        if error_count < int(self.main_loop_error_retry_limit):
            return False
        self._trigger_hard_halt(reason="main_loop_error_threshold", error=error)
        return True

    def _ordered_shutdown(self) -> None:
        self.data_handler.continue_backtest = False
        if hasattr(self.data_handler, "ws_running"):
            self.data_handler.ws_running = False
        if self._user_stream_client is not None:
            try:
                self._user_stream_client.stop(join_timeout=5.0)
            except Exception:
                pass
        shutdown = getattr(self.data_handler, "shutdown", None)
        if callable(shutdown):
            shutdown(join_timeout=5.0)

    def _run_startup_reconciliation_gate(self) -> None:
        if self.order_state_source != "user_stream":
            self._startup_reconciliation_complete = True
            self._set_startup_state("ready")
            return

        self.portfolio.trading_frozen = True
        outcome = self._recovery_service.startup_converge(
            timeout_seconds=float(self.startup_reconciliation_timeout_seconds)
        )
        if outcome.converged:
            self._startup_reconciliation_complete = True
            self.portfolio.trading_frozen = False
            self._set_startup_state("ready")
            self.audit_store.log_risk_event(
                self.run_id,
                reason="STARTUP_RECONCILIATION_COMPLETE",
                details={
                    "tracked_orders": outcome.tracked_orders,
                    "cache_open_orders": outcome.cache_open_orders,
                    "exchange_open_orders": outcome.exchange_open_orders,
                    "exchange_snapshot_ready": outcome.exchange_snapshot_ready,
                    "tracked_cache_signature_match": outcome.tracked_cache_signature_match,
                    "exchange_signature_match": outcome.exchange_signature_match,
                    "elapsed_seconds": round(float(outcome.elapsed_seconds), 3),
                },
            )
            return

        self.audit_store.log_risk_event(
            self.run_id,
            reason="STARTUP_RECONCILIATION_TIMEOUT",
            details={
                "timeout_seconds": float(self.startup_reconciliation_timeout_seconds),
                "tracked_orders": outcome.tracked_orders,
                "cache_open_orders": outcome.cache_open_orders,
                "exchange_open_orders": outcome.exchange_open_orders,
                "exchange_snapshot_ready": outcome.exchange_snapshot_ready,
                "tracked_cache_signature_match": outcome.tracked_cache_signature_match,
                "exchange_signature_match": outcome.exchange_signature_match,
            },
        )
        self._activate_fallback_polling("startup_reconciliation_timeout")
        if self.startup_reconciliation_hard_fail:
            raise LiveDataFatalError("startup_reconciliation_timeout")
        self._startup_reconciliation_complete = True
        self.portfolio.trading_frozen = False
        self._set_startup_state("degraded", reason="startup_reconciliation_timeout")
        self.audit_store.log_risk_event(
            self.run_id,
            reason="STARTUP_RECONCILIATION_DEGRADED",
            details={
                "mode": "fallback_polling",
                "fallback_window_seconds": float(self.reconciliation_fallback_window_seconds),
            },
        )

    def set_measured_shadow_parity_ratio(self, ratio: float | None) -> None:
        """Inject an explicitly-measured shadow signal-parity ratio.

        Ops/go-live-checklist tooling calls this (with the value from
        ``ShadowLiveRunner.evaluate_signal_parity``) before ``run()`` so the shadow
        stage readiness gate can validate ``ratio >= shadow_parity_min_ratio``.  Passing
        None (the default) keeps the shadow gate fail-closed.
        """
        self._measured_shadow_parity_ratio = None if ratio is None else float(ratio)

    def _run_live_readiness_gate(self) -> None:
        go_live_stage = (
            str(getattr(self.config, "GO_LIVE_STAGE", "testnet") or "testnet").strip().lower()
        )
        # testnet is the only stage with no artifact/env constraints, and only when not
        # running real mode.  shadow, canary, and full stages — and ANY real-mode run —
        # must pass their per-stage entry checks (spec R7: stage selection is free
        # composition; the selected stage's gate is mandatory, and real mode is always
        # gated so a real run can never start without the readiness chain).
        if go_live_stage == "testnet" and self.live_mode != "real":
            return
        payload = enforce_live_readiness_from_files(
            mode=self.live_mode,
            stale_minutes=int(self._readiness_preflight_stale_minutes),
            env=os.environ,
            go_live_stage=go_live_stage,
            # Measured shadow parity ratio (fail-closed None by default → shadow stage
            # stays blocked until an operator injects an explicitly-measured ratio via
            # set_measured_shadow_parity_ratio). canary/full stages ignore this field.
            shadow_parity_ratio=self._measured_shadow_parity_ratio,
        )
        self.audit_store.log_risk_event(
            self.run_id,
            reason="LIVE_READINESS_VERIFIED",
            details={
                "recommended_action": payload.get("recommended_action"),
                "status": dict(payload.get("status") or {}),
            },
        )

    def run(self):
        """Main Live Trading Loop."""
        self.logger.info(f"Starting Live Trading on {self.symbol_list}...")
        try:
            self._run_live_readiness_gate()
            self._sync_portfolio()
            self._run_startup_reconciliation_gate()
            self._run_exit_status = "STOPPED"
            self._notify_startup_state()
        except LiveReadinessBlockedError as exc:
            self._set_startup_state("failed_init", error=exc)
            self._notify_startup_state()
            self._run_exit_status = "FAILED"
            self.audit_store.log_risk_event(
                self.run_id,
                reason="LIVE_READINESS_BLOCKED",
                details={
                    "mode": self.live_mode,
                    "recommended_action": exc.recommended_action,
                },
            )
            self._close_audit_store(status="FAILED")
            raise
        except Exception as exc:
            self._set_startup_state("failed_init", error=exc)
            self._notify_startup_state()
            self._run_exit_status = "FAILED"
            self._close_audit_store(status="FAILED")
            raise

        while True:
            try:
                fatal_exc = self._consume_data_fatal()
                if fatal_exc is not None:
                    reason = "live_data_contract_breach"
                    if isinstance(fatal_exc, RawFirstDataMissingError):
                        reason = "missing_committed_data"
                    elif isinstance(fatal_exc, MarketWindowContractError):
                        reason = "market_window_contract_error"
                    self._trigger_hard_halt(reason=reason, error=fatal_exc)
                    self._save_state()
                    self._ordered_shutdown()
                    self._run_exit_status = "FAILED"
                    self._close_audit_store(status="FAILED")
                    raise LiveDataFatalError(str(fatal_exc)) from fatal_exc

                if self._is_stop_requested():
                    self.logger.warning("Stop file detected. Shutting down live trader gracefully.")
                    self.notifier.send_message(
                        "⏹️ **Stop Requested** via control file. Stopping trader."
                    )
                    self._save_state()
                    self._ordered_shutdown()
                    self._run_exit_status = "STOPPED"
                    self._close_audit_store(status="STOPPED")
                    break

                self._emit_heartbeat(force=False)
                if self._is_fallback_polling_required():
                    self._reconcile_positions(force=False)
                    self._reconcile_orders(force=False)
                self._evaluate_risk_guards()
                # In Live mode, data_handler is threaded and pushes events autonomously.
                # We just blocking-wait for events.
                event = self.events.get(True, timeout=10)  # Wait up to 10s

                # Add logging for events if needed (optional overlay on process_event)
                # But engine.py handles basic routing.
                # We might want logging wrappers?
                # For now, rely on engine.
                # But wait, engine doesn't log!
                # I should override handle_xxx methods or add logging to engine.
                # To be Clean OOP, let's keep engine distinct and perhaps add logging there or here.
                # simpler: just log here then call process.

                if event is not None:
                    if event.type == "MARKET_WINDOW" and self._handle_market_window_staleness(
                        event
                    ):
                        continue
                    if self._hard_halt_active and event.type in {"SIGNAL", "ORDER"}:
                        self.logger.error(
                            "EVENT_BLOCKED_HARD_HALT type=%s reason=%s",
                            event.type,
                            self._hard_halt_reason,
                        )
                        continue
                    if event.type == "MARKET":
                        self.logger.debug(f"Market Event: {event.symbol}")
                    elif event.type == "SIGNAL":
                        self.logger.info(f"Signal Event: {event.signal_type}")
                    if event.type == "ORDER":
                        if (
                            self.order_state_source == "user_stream"
                            and not self._startup_reconciliation_complete
                        ):
                            self.logger.error("ORDER_BLOCKED_STARTUP_RECONCILIATION_PENDING")
                            continue
                        if self._materialized_stale_block_active:
                            self.logger.error(
                                "ORDER_BLOCKED_STALE_FEED symbol=%s direction=%s",
                                event.symbol,
                                event.direction,
                            )
                            self.audit_store.log_risk_event(
                                self.run_id,
                                reason="ORDER_BLOCKED_STALE_WINDOW",
                                details={
                                    "symbol": event.symbol,
                                    "direction": event.direction,
                                    "quantity": event.quantity,
                                },
                            )
                            continue
                        self.logger.info(f"Order Event: {event.direction}")
                        # RISK CHECK
                        current_price = self.data_handler.get_latest_bar_value(
                            event.symbol, "close"
                        )
                        passed, reason = self.risk_manager.check_order(
                            event, current_price, portfolio=self.portfolio
                        )
                        if not passed:
                            msg = f"⛔ **Risk Reject**: {reason}"
                            self.logger.warning(msg)
                            self.notifier.send_message(msg)
                            self.audit_store.log_risk_event(
                                self.run_id,
                                reason="ORDER_REJECTED",
                                details={
                                    "symbol": event.symbol,
                                    "direction": event.direction,
                                    "quantity": event.quantity,
                                    "reason": reason,
                                },
                            )
                            continue  # Skip processing this event
                        self.audit_store.log_order(self.run_id, event, status="NEW")

                    self.process_event(event)

            except queue.Empty:
                # Heartbeat
                self._emit_heartbeat(force=True)
                should_poll_fallback = self._is_fallback_polling_required()
                if should_poll_fallback:
                    self._reconcile_positions(force=False)
                    self._reconcile_orders(force=False)
                self._evaluate_risk_guards()
                if hasattr(self.execution_handler, "check_open_orders") and should_poll_fallback:
                    self.execution_handler.check_open_orders(None)
            except KeyboardInterrupt:
                self.logger.info("Stopping Live Trader...")
                self._save_state()  # Save on exit
                self._ordered_shutdown()
                self._run_exit_status = "STOPPED"
                self._close_audit_store(status="STOPPED")
                break
            except LiveDataFatalError:
                self._run_exit_status = "FAILED"
                raise
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self._save_state()  # Save on crash attempt
                if self._register_main_loop_error(e):
                    self._run_exit_status = "FAILED"
                    self._ordered_shutdown()
                    self._close_audit_store(status="FAILED")
                    raise LiveDataFatalError(str(e)) from e
                time.sleep(1.0)

    def __del__(self):
        self._close_audit_store(status="STOPPED")


__all__ = ["LiveDataFatalError", "LiveTrader"]
