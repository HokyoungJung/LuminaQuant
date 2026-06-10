import argparse
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.chunked_runner import run_backtest_chunked
from lumina_quant.backtesting.cli_contract import (
    RawFirstDataMissingError,
    RawFirstManifestInvalidError,
    RawFirstStaleWindowError,
    normalize_data_mode,
    raw_first_exit_code,
    resolve_data_contract,
)
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.core.plugin_registry import (
    import_private_strategy_registry,
    load_strategy_registry,
)
from lumina_quant.configuration import (
    get_default_runtime_config,
)
from lumina_quant.live_selection import (
    normalize_portfolio_mode_reference,
    resolve_portfolio_mode_runtime_config,
    supports_live_portfolio_mode,
)
from lumina_quant.market_data import (
    load_data_dict_from_db,
    load_data_dict_from_external_root,
    load_data_dict_from_parquet,
    normalize_timeframe_token,
)
from lumina_quant.storage.parquet import is_parquet_market_data_store
from lumina_quant.strategies.artifact_portfolio_mode import ArtifactPortfolioModeStrategy
from lumina_quant.utils.audit_store import AuditStore

_strategy_registry = None


@dataclass(slots=True)
class BacktestStrategySetup:
    strategy_cls: type
    strategy_name: str
    strategy_params: dict[str, Any]
    symbol_list: list[str] | None = None
    portfolio_mode: str | None = None


def _get_strategy_registry():
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = load_strategy_registry(import_private_strategy_registry)
    return _strategy_registry


# ==========================================
# CONFIGURATION FROM YAML
# ==========================================


# 3. Data Settings
CSV_DIR = "data"
# Compatibility shims for tests/legacy monkeypatching. Keep None by default so
# runtime settings still resolve dynamically at call time.
SYMBOL_LIST = None
START_DATE = None
END_DATE = None
MARKET_DB_PATH = None
MARKET_DB_EXCHANGE = None
MARKET_DB_BACKEND = None
BT_CHUNK_DAYS = None
BT_CHUNK_WARMUP_BARS = None
BACKTEST_POLL_SECONDS = None
BACKTEST_WINDOW_SECONDS = None
BACKTEST_DECISION_CADENCE_SECONDS = None
BACKTEST_AUDIT_SNAPSHOT_SECONDS = None


def _portfolio_mode_from_requested_strategy(requested_strategy_name: str) -> str:
    token = normalize_portfolio_mode_reference(requested_strategy_name)
    if supports_live_portfolio_mode(token):
        return token
    return ""


def _resolve_strategy_setup_detail(
    *,
    log: bool,
    portfolio_mode: str | None = None,
) -> BacktestStrategySetup:
    requested_portfolio_mode = str(
        portfolio_mode or os.getenv("LQ_BACKTEST_PORTFOLIO_MODE", "") or ""
    ).strip()
    requested_strategy_name = str(get_default_runtime_config().optimization.strategy or "").strip()
    if not requested_portfolio_mode:
        requested_portfolio_mode = _portfolio_mode_from_requested_strategy(requested_strategy_name)
    if requested_portfolio_mode:
        runtime_config = resolve_portfolio_mode_runtime_config(requested_portfolio_mode)
        if log:
            print(
                "[OK] Backtest portfolio mode resolved through live runtime: "
                f"{runtime_config['strategy_name']} "
                f"symbols={runtime_config['symbols']}"
            )
        return BacktestStrategySetup(
            strategy_cls=ArtifactPortfolioModeStrategy,
            strategy_name=str(runtime_config["strategy_name"]),
            strategy_params=dict(runtime_config["strategy_params"]),
            symbol_list=list(runtime_config["symbols"]),
            portfolio_mode=str(runtime_config["portfolio_mode"]),
        )

    strategy_registry = _get_strategy_registry()
    strategy_cls = strategy_registry.resolve_strategy_class(
        requested_strategy_name,
        default_name=strategy_registry.DEFAULT_STRATEGY_NAME,
    )
    strategy_name = strategy_cls.__name__
    strategy_params = strategy_registry.get_default_strategy_params(strategy_name)

    param_path = os.path.join("best_optimized_parameters", strategy_name, "best_params.json")
    meta_path = os.path.join("best_optimized_parameters", strategy_name, "best_params.meta.json")

    if os.path.exists(param_path):
        try:
            with open(param_path) as handle:
                strategy_params = json.load(handle)
            if log:
                print(f"[OK] Loaded Optimized Params from {param_path}")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as meta_file:
                        meta = json.load(meta_file)
                    selection_basis = str(meta.get("selection_basis", "")).strip().lower()
                    if log:
                        if selection_basis == "validation_only":
                            print(
                                "[INFO] Parameter provenance: validation-only selection with locked OOS holdout."
                            )
                        else:
                            print(
                                "[WARN] Parameter provenance metadata exists but selection basis is not "
                                "'validation_only'."
                            )
                except Exception as exc:
                    if log:
                        print(f"[WARN] Failed to parse params metadata: {exc}")
            elif log:
                print(
                    "[WARN] No parameter provenance metadata file found. "
                    "Consider re-running optimize.py with strict OOS settings."
                )
        except Exception as exc:
            if log:
                print(f"[WARN] Failed to load optimized params: {exc}")
    elif log:
        print(f"[INFO] Optimized params not found at {param_path}. Using Defaults.")

    return BacktestStrategySetup(
        strategy_cls=strategy_cls,
        strategy_name=strategy_name,
        strategy_params=strategy_registry.resolve_strategy_params(strategy_name, strategy_params),
    )


def _resolve_strategy_setup(*, log: bool) -> tuple[type, dict[str, Any]]:
    setup = _resolve_strategy_setup_detail(log=log)
    return setup.strategy_cls, setup.strategy_params


def _auto_collect_market_data(*args, **kwargs):
    try:
        from lumina_quant.data_collector import auto_collect_market_data as collector
    except Exception as exc:
        raise RuntimeError("auto_collect_market_data is unavailable in this distribution.") from exc
    return collector(*args, **kwargs)


def _normalize_timeframe_or_default(value, default):
    token = str(value or "").strip()
    if not token:
        return str(default)
    try:
        return normalize_timeframe_token(token)
    except Exception:
        return str(default)


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_optional_bool(name):
    raw = os.getenv(name)
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None
    return token.lower() not in {"0", "false", "no", "off"}


def _env_int(name, default):
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _normalize_backtest_mode(value: str | None, default: str = "windowed") -> str:
    token = str(value or default).strip().lower()
    if token in {"windowed", "legacy_batch", "legacy_1s"}:
        return token
    return str(default)


def _normalize_data_mode(value: str | None, default: str = "raw-first") -> str:
    try:
        return normalize_data_mode(value, default=default)
    except RawFirstDataMissingError:
        return str(default)


def _parse_config_date(value, default):
    try:
        if value:
            return datetime.strptime(value, "%Y-%m-%d")
    except Exception:
        return default
    return default


def _current_backtest_runtime_settings() -> dict[str, Any]:
    symbol_list_override = globals().get("SYMBOL_LIST")
    start_date_override = globals().get("START_DATE")
    end_date_override = globals().get("END_DATE")
    market_db_path_override = globals().get("MARKET_DB_PATH")
    market_db_exchange_override = globals().get("MARKET_DB_EXCHANGE")
    market_db_backend_override = globals().get("MARKET_DB_BACKEND")
    bt_chunk_days_override = globals().get("BT_CHUNK_DAYS")
    bt_chunk_warmup_override = globals().get("BT_CHUNK_WARMUP_BARS")
    backtest_poll_override = globals().get("BACKTEST_POLL_SECONDS")
    backtest_window_override = globals().get("BACKTEST_WINDOW_SECONDS")
    backtest_decision_override = globals().get("BACKTEST_DECISION_CADENCE_SECONDS")
    backtest_audit_snapshot_override = globals().get("BACKTEST_AUDIT_SNAPSHOT_SECONDS")
    # Load typed runtime config once — tests may override globals above for monkeypatching.
    _rt = get_default_runtime_config()
    _bt = _rt.backtest
    start_date = (
        start_date_override
        if start_date_override is not None
        else _parse_config_date(_bt.start_date, datetime(2024, 1, 1))
    )
    end_date = (
        end_date_override
        if end_date_override is not None
        else _parse_config_date(_bt.end_date, None)
    )
    settings = {
        "symbol_list": list(symbol_list_override)
        if symbol_list_override is not None
        else list(_rt.trading.symbols),
        "start_date": start_date,
        "end_date": end_date,
        "market_db_path": (
            str(market_db_path_override)
            if market_db_path_override is not None
            else str(_rt.storage.market_data_parquet_path)
        ),
        "market_db_exchange": (
            str(market_db_exchange_override)
            if market_db_exchange_override is not None
            else str(_rt.storage.market_data_exchange)
        ),
        "market_db_backend": (
            str(market_db_backend_override)
            if market_db_backend_override is not None
            else str(_rt.storage.backend)
        ),
        "base_timeframe": _normalize_timeframe_or_default(
            os.getenv("LQ_BASE_TIMEFRAME", "1s"), "1s"
        ),
        "auto_collect_db": _env_bool("LQ_AUTO_COLLECT_DB", False),
        "bt_chunk_days": (
            max(1, int(bt_chunk_days_override))
            if bt_chunk_days_override is not None
            else max(
                1,
                _env_int("LQ__BACKTEST__CHUNK_DAYS", int(_bt.chunk_days)),
            )
        ),
        "bt_chunk_warmup_bars": (
            max(0, int(bt_chunk_warmup_override))
            if bt_chunk_warmup_override is not None
            else max(
                0,
                _env_int("LQ__BACKTEST__CHUNK_WARMUP_BARS", int(_bt.chunk_warmup_bars)),
            )
        ),
        "backtest_poll_seconds": (
            max(1, int(backtest_poll_override))
            if backtest_poll_override is not None
            else max(
                1,
                _env_int("LQ__BACKTEST__POLL_SECONDS", int(_bt.poll_seconds)),
            )
        ),
        "backtest_window_seconds": (
            max(1, int(backtest_window_override))
            if backtest_window_override is not None
            else max(
                1,
                _env_int("LQ__BACKTEST__WINDOW_SECONDS", int(_bt.window_seconds)),
            )
        ),
        "backtest_decision_cadence_seconds": (
            max(1, int(backtest_decision_override))
            if backtest_decision_override is not None
            else max(
                1,
                _env_int(
                    "LQ__BACKTEST__DECISION_CADENCE_SECONDS",
                    int(_bt.decision_cadence_seconds),
                ),
            )
        ),
        "backtest_audit_snapshot_seconds": (
            max(1, int(backtest_audit_snapshot_override))
            if backtest_audit_snapshot_override is not None
            else max(
                1,
                _env_int(
                    "LQ__BACKTEST__AUDIT_SNAPSHOT_SECONDS",
                    int(os.getenv("LQ_BACKTEST_AUDIT_SNAPSHOT_SECONDS", "60") or "60"),
                ),
            )
        ),
        "backtest_mode": _normalize_backtest_mode(
            os.getenv("LQ_BACKTEST_MODE", str(_bt.mode)),
            default="windowed",
        ),
        "data_mode": _normalize_data_mode(
            os.getenv("LQ_DATA_MODE", "raw-first"),
            default="raw-first",
        ),
    }
    os.environ.setdefault(
        "LQ__BACKTEST__DECISION_CADENCE_SECONDS",
        str(int(settings["backtest_decision_cadence_seconds"])),
    )
    return settings


def _enforce_1s_base_timeframe(value: str) -> str:
    token = _normalize_timeframe_or_default(value, "1s")
    if token != "1s":
        print(
            f"[WARN] base_timeframe '{token}' overridden to '1s' for intrabar backtest execution."
        )
    return "1s"


def _create_audit_store():
    try:
        return AuditStore(get_default_runtime_config().storage.postgres_dsn), True
    except Exception as exc:
        message = str(exc).strip()
        if not message:
            message = exc.__class__.__name__
        print(
            "[WARN] PostgreSQL audit store unavailable "
            f"({message}). Backtest will run without PostgreSQL audit persistence."
        )
        return None, False


# ==========================================
# EXECUTION (Do not modify generally)
# ==========================================


def _load_data_dict(
    data_source,
    market_db_path,
    market_exchange,
    *,
    base_timeframe,
    external_data_root=None,
    external_symbol_map=None,
    data_mode="legacy",
    backtest_mode="windowed",
    auto_collect_db=True,
    symbol_list=None,
):
    settings = _current_backtest_runtime_settings()
    symbol_list = list(symbol_list) if symbol_list is not None else list(settings["symbol_list"])
    start_date = settings["start_date"]
    end_date = settings["end_date"]
    market_db_backend = str(settings["market_db_backend"])
    bt_chunk_days = int(settings["bt_chunk_days"])
    bt_chunk_warmup_bars = int(settings["bt_chunk_warmup_bars"])
    contract = resolve_data_contract(
        data_mode=str(data_mode),
        backtest_mode=str(backtest_mode),
        data_source=str(data_source),
        default_backtest_mode="windowed",
        default_data_source="auto",
    )
    source = str(contract.data_source).strip().lower()
    if source == "csv":
        return None
    if source == "external":
        data_dict = load_data_dict_from_external_root(
            str(external_data_root or market_db_path),
            symbol_list=symbol_list,
            symbol_map=dict(external_symbol_map or {}),
            start_date=start_date,
            end_date=end_date,
        )
        if data_dict:
            return data_dict
        raise RuntimeError(
            "No market data found in external root for requested symbols. "
            "Provide canonical CSV/parquet OHLCV files."
        )

    use_parquet = is_parquet_market_data_store(
        str(market_db_path),
        backend=market_db_backend,
    )
    if contract.data_mode == "raw-first" and not use_parquet:
        raise RawFirstDataMissingError(
            "Raw-first requires parquet manifest store. Configure --market-db-path to parquet root."
        )

    if source in {"auto", "db"} and auto_collect_db and not use_parquet:
        _lv_rt = get_default_runtime_config().live
        try:
            sync_rows = _auto_collect_market_data(
                symbol_list=symbol_list,
                timeframe=str(base_timeframe),
                db_path=str(market_db_path),
                exchange_id=str(market_exchange),
                market_type=str(_lv_rt.exchange.market_type),
                since_dt=start_date,
                until_dt=end_date,
                api_key=str(_lv_rt.api_key or ""),
                secret_key=str(_lv_rt.secret_key or ""),
                testnet=bool(str(_lv_rt.mode).strip().lower() != "real"),
                limit=1000,
                max_batches=100000,
                retries=3,
                base_wait_sec=0.5,
            )

            def _safe_int(value):
                try:
                    return int(value)
                except Exception:
                    return 0

            upserted = sum(_safe_int(item.get("upserted_rows", 0)) for item in sync_rows)
            fetched = sum(_safe_int(item.get("fetched_rows", 0)) for item in sync_rows)
            print(
                f"[INFO] Auto collector checked DB coverage for {len(sync_rows)} symbols "
                f"(fetched={fetched}, upserted={upserted})."
            )
        except Exception as exc:
            if source == "db":
                raise RuntimeError(f"DB auto-collect failed: {exc}") from exc
            print(f"[WARN] DB auto-collect failed; continuing with fallback behavior: {exc}")
    elif source in {"auto", "db"} and auto_collect_db and use_parquet:
        print("[INFO] Auto collector skipped for parquet market-data backend.")

    if use_parquet:
        data_dict = load_data_dict_from_parquet(
            str(market_db_path),
            exchange=str(market_exchange),
            symbol_list=symbol_list,
            timeframe=str(base_timeframe),
            start_date=start_date,
            end_date=end_date,
            chunk_days=bt_chunk_days,
            warmup_bars=bt_chunk_warmup_bars,
            data_mode=str(contract.data_mode),
        )
    else:
        data_dict = load_data_dict_from_db(
            market_db_path,
            exchange=market_exchange,
            symbol_list=symbol_list,
            timeframe=str(base_timeframe),
            start_date=start_date,
            end_date=end_date,
            backend=market_db_backend,
        )
    if data_dict:
        missing = [symbol for symbol in symbol_list if symbol not in data_dict]
        print(
            f"[INFO] Loaded {len(data_dict)}/{len(symbol_list)} symbols from DB "
            f"{market_db_path} (exchange={market_exchange}, timeframe={base_timeframe})."
        )
        if missing:
            if contract.data_mode == "raw-first":
                raise RawFirstDataMissingError(
                    "Raw-first committed data missing for symbols: " + ", ".join(missing)
                )
            print(f"[WARN] Symbols still missing in DB after load: {missing}")
        return data_dict
    if contract.data_mode == "raw-first":
        raise RawFirstDataMissingError(
            "No committed manifest data found for raw-first mode. "
            "Run materializer and verify manifest commits."
        )
    if source == "db":
        raise RuntimeError(
            "No market data found in DB for requested symbols/timeframe. "
            "Run scripts/sync_binance_ohlcv.py first or switch to --data-source csv."
        )
    return None


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_float_or(value, default):
    parsed = _safe_float(value)
    if parsed is None:
        return float(default)
    return float(parsed)


def _persist_backtest_audit_rows(audit_store, run_id, backtest, *, low_memory=False):
    equity_rows = 0
    fill_rows = 0

    try:
        if bool(low_memory):
            points = list(getattr(backtest.portfolio, "_equity_points", []) or [])
            snapshot_interval = max(
                1.0, float(_current_backtest_runtime_settings()["backtest_audit_snapshot_seconds"])
            )
            last_snapshot_ts = None
            for ts_seconds, total_value in points:
                total = _safe_float(total_value)
                if total is None:
                    continue
                if (
                    last_snapshot_ts is not None
                    and (float(ts_seconds) - float(last_snapshot_ts)) < snapshot_interval
                ):
                    continue
                last_snapshot_ts = float(ts_seconds)
                try:
                    timeindex = datetime.fromtimestamp(float(ts_seconds), tz=UTC)
                except Exception:
                    timeindex = None
                audit_store.log_equity(
                    run_id,
                    timeindex=timeindex,
                    total=total,
                    cash=None,
                    metadata={
                        "source": "low_memory_equity_point",
                        "snapshot_interval_seconds": int(snapshot_interval),
                    },
                )
                equity_rows += 1
        else:
            equity_curve = getattr(backtest.portfolio, "equity_curve", None)
            if equity_curve is None:
                backtest.portfolio.create_equity_curve_dataframe()
                equity_curve = getattr(backtest.portfolio, "equity_curve", None)
            if equity_curve is not None:
                for row in equity_curve.iter_rows(named=True):
                    total = _safe_float(row.get("total"))
                    if total is None:
                        continue
                    cash = _safe_float(row.get("cash"))
                    metadata = {}
                    benchmark = _safe_float(row.get("benchmark_price"))
                    funding = _safe_float(row.get("funding"))
                    if benchmark is not None:
                        metadata["benchmark_price"] = benchmark
                    if funding is not None:
                        metadata["funding"] = funding
                    audit_store.log_equity(
                        run_id,
                        timeindex=row.get("datetime"),
                        total=total,
                        cash=cash,
                        metadata=metadata,
                    )
                    equity_rows += 1
    except Exception as exc:
        print(f"[WARN] Failed to persist equity rows to audit DB: {exc}")

    try:
        trades = list(getattr(backtest.portfolio, "trades", []) or [])
        for idx, trade in enumerate(trades):
            symbol = str(trade.get("symbol") or "").strip()
            side = str(trade.get("direction") or "").strip().upper()
            quantity = _safe_float(trade.get("quantity"))
            if not symbol or side not in {"BUY", "SELL"} or quantity is None or quantity <= 0.0:
                continue
            fill_event = SimpleNamespace(
                timeindex=trade.get("datetime"),
                symbol=symbol,
                direction=side,
                quantity=quantity,
                fill_cost=_safe_float(trade.get("fill_cost")),
                commission=_safe_float(trade.get("commission")) or 0.0,
                client_order_id=f"bt-{run_id}-{idx:06d}",
                order_id=None,
                status="FILLED",
                metadata={"source": "backtest_trade_log"},
            )
            audit_store.log_fill(run_id, fill_event)
            fill_rows += 1
    except Exception as exc:
        print(f"[WARN] Failed to persist fill rows to audit DB: {exc}")

    return {"equity_rows": equity_rows, "fill_rows": fill_rows}


def _is_year_scale_window(start_date, end_date, *, threshold_days=30) -> bool:
    start = start_date if isinstance(start_date, datetime) else None
    end = end_date if isinstance(end_date, datetime) else None
    if start is None:
        return False
    if end is None:
        end = datetime.now(tz=UTC).replace(tzinfo=None)
    if end < start:
        return False
    return (end - start).days >= int(threshold_days)


def _resolve_execution_profile(
    *, low_memory=None, persist_output=None, start_date=None, end_date=None
):
    env_low_memory = _env_optional_bool("LQ_BACKTEST_LOW_MEMORY")
    if low_memory is not None:
        resolved_low_memory = bool(low_memory)
    elif env_low_memory is not None:
        resolved_low_memory = bool(env_low_memory)
    else:
        resolved_low_memory = _is_year_scale_window(start_date, end_date, threshold_days=30)
    resolved_persist_output = persist_output
    if resolved_persist_output is None:
        resolved_persist_output = _env_optional_bool("LQ_BACKTEST_PERSIST_OUTPUT")
    if resolved_persist_output is None:
        resolved_persist_output = (
            False
            if resolved_low_memory
            else bool(get_default_runtime_config().backtest.persist_output)
        )
    return {
        "low_memory": bool(resolved_low_memory),
        "record_history": not bool(resolved_low_memory),
        "track_metrics": True,
        "record_trades": not bool(resolved_low_memory),
        "persist_output": bool(resolved_persist_output),
    }


def _print_low_memory_stats(backtest):
    fast_stats = {}
    try:
        fast_stats = dict(backtest.portfolio.output_summary_stats_fast() or {})
    except Exception as exc:
        fast_stats = {"status": f"error: {exc}"}

    final_equity = _safe_float(getattr(backtest.portfolio, "current_holdings", {}).get("total"))
    trade_count = int(getattr(backtest.portfolio, "trade_count", 0))
    print("[INFO] Low-memory mode enabled (history/trade logs disabled).")
    print(
        "[INFO] Backtest summary: "
        f"final_equity={final_equity if final_equity is not None else 0.0:.4f}, "
        f"trade_count={trade_count}, "
        f"sharpe={_safe_float_or(fast_stats.get('sharpe'), 0.0):.4f}, "
        f"cagr={_safe_float_or(fast_stats.get('cagr'), 0.0):.6f}, "
        f"max_drawdown={_safe_float_or(fast_stats.get('max_drawdown'), 0.0):.6f}"
    )
    return fast_stats


def _persist_low_memory_outputs(backtest, persist_output):
    if not bool(persist_output):
        return
    backtest.portfolio.create_equity_curve_dataframe()
    backtest.portfolio.output_trade_log(os.path.join("data", "trades.csv"))
    backtest.portfolio.save_equity_curve(os.path.join("data", "equity.csv"))


def run(
    data_source="auto",
    data_mode=None,
    market_db_path=None,
    market_exchange=None,
    base_timeframe=None,
    external_data_root="",
    auto_collect_db=None,
    run_id="",
    low_memory=None,
    persist_output=None,
    backtest_mode=None,
    portfolio_mode=None,
):
    settings = _current_backtest_runtime_settings()
    symbol_list = list(settings["symbol_list"])
    start_date = settings["start_date"]
    end_date = settings["end_date"]
    market_db_path = (
        str(settings["market_db_path"]) if market_db_path is None else str(market_db_path)
    )
    market_exchange = (
        str(settings["market_db_exchange"]) if market_exchange is None else str(market_exchange)
    )
    market_db_backend = str(settings["market_db_backend"])
    base_timeframe = (
        str(settings["base_timeframe"]) if base_timeframe is None else str(base_timeframe)
    )
    auto_collect_db = (
        bool(settings["auto_collect_db"]) if auto_collect_db is None else bool(auto_collect_db)
    )
    data_mode = str(settings["data_mode"]) if data_mode is None else str(data_mode)
    backtest_mode = str(settings["backtest_mode"]) if backtest_mode is None else str(backtest_mode)
    bt_chunk_days = int(settings["bt_chunk_days"])
    bt_chunk_warmup_bars = int(settings["bt_chunk_warmup_bars"])
    backtest_poll_seconds = int(settings["backtest_poll_seconds"])
    backtest_window_seconds = int(settings["backtest_window_seconds"])
    backtest_decision_cadence_seconds = int(settings["backtest_decision_cadence_seconds"])
    strategy_setup = _resolve_strategy_setup_detail(log=True, portfolio_mode=portfolio_mode)
    strategy_cls = strategy_setup.strategy_cls
    strategy_params = dict(strategy_setup.strategy_params)
    strategy_runtime_name = str(strategy_setup.strategy_name)
    if strategy_setup.symbol_list is not None:
        symbol_list = list(strategy_setup.symbol_list)

    print("------------------------------------------------")
    print(f"Running Backtest for {symbol_list}")
    print(f"Strategy: {strategy_runtime_name}")
    print(f"Params: {strategy_params}")
    print("------------------------------------------------")

    backtest_run_id = str(run_id or "").strip() or str(uuid.uuid4())
    timeframe_token = _enforce_1s_base_timeframe(str(base_timeframe))
    audit_store, audit_enabled = _create_audit_store()
    execution_profile = _resolve_execution_profile(
        low_memory=low_memory,
        persist_output=persist_output,
        start_date=start_date,
        end_date=end_date,
    )
    contract = resolve_data_contract(
        data_mode=str(data_mode),
        backtest_mode=str(backtest_mode),
        data_source=str(data_source),
        default_backtest_mode="windowed",
        default_data_source="auto",
    )
    resolved_backtest_mode = str(contract.backtest_mode)
    resolved_data_source = str(contract.data_source)
    resolved_data_mode = str(contract.data_mode)
    selected_data_handler_cls = (
        HistoricParquetWindowedDataHandler
        if resolved_backtest_mode == "windowed"
        else HistoricCSVDataHandler
    )
    selected_data_handler_kwargs = (
        {
            "backtest_poll_seconds": backtest_poll_seconds,
            "backtest_window_seconds": backtest_window_seconds,
        }
        if resolved_backtest_mode == "windowed"
        else {}
    )
    if audit_store is not None:
        audit_store.start_run(
            mode="backtest",
            metadata={
                "symbols": symbol_list,
                "strategy": strategy_runtime_name,
                "params": strategy_params,
                "portfolio_mode": strategy_setup.portfolio_mode,
                "data_source": str(resolved_data_source),
                "data_mode": str(resolved_data_mode),
                "market_db_path": str(market_db_path),
                "external_data_root": str(external_data_root or ""),
                "market_exchange": str(market_exchange),
                "base_timeframe": str(timeframe_token),
                "strategy_timeframe": str(get_default_runtime_config().trading.timeframe),
                "auto_collect_db": bool(auto_collect_db),
                "backtest_poll_seconds": backtest_poll_seconds,
                "backtest_window_seconds": backtest_window_seconds,
                "backtest_decision_cadence_seconds": backtest_decision_cadence_seconds,
                "backtest_mode": str(resolved_backtest_mode),
                "chunk_days": bt_chunk_days,
                "chunk_warmup_bars": bt_chunk_warmup_bars,
                **execution_profile,
            },
            run_id=backtest_run_id,
        )
    elif not bool(audit_enabled):
        print("[INFO] Audit persistence is disabled for this backtest run.")

    try:
        use_parquet = is_parquet_market_data_store(
            str(market_db_path),
            backend=market_db_backend,
        )
        source_token = str(resolved_data_source).strip().lower()
        use_chunked_runner = bool(use_parquet and source_token in {"auto", "db"})
        data_dict = None

        if not use_chunked_runner:
            data_dict = _load_data_dict(
                resolved_data_source,
                market_db_path,
                market_exchange,
                base_timeframe=str(timeframe_token),
                external_data_root=str(external_data_root or ""),
                external_symbol_map=dict(
                    get_default_runtime_config().backtest.external.symbol_map or {}
                ),
                data_mode=resolved_data_mode,
                backtest_mode=resolved_backtest_mode,
                auto_collect_db=bool(auto_collect_db),
                symbol_list=symbol_list,
            )

        if use_chunked_runner:
            if end_date is None:
                raise RuntimeError(
                    "Chunked backtest requires an explicit END_DATE when using parquet backend."
                )

            def _chunk_loader(chunk_start, chunk_end):
                return load_data_dict_from_parquet(
                    str(market_db_path),
                    exchange=str(market_exchange),
                    symbol_list=symbol_list,
                    timeframe=str(timeframe_token),
                    start_date=chunk_start,
                    end_date=chunk_end,
                    chunk_days=bt_chunk_days,
                    warmup_bars=bt_chunk_warmup_bars,
                    data_mode=resolved_data_mode,
                )

            backtest = run_backtest_chunked(
                csv_dir=CSV_DIR,
                symbol_list=symbol_list,
                start_date=start_date,
                end_date=end_date,
                strategy_cls=strategy_cls,
                strategy_params=strategy_params,
                data_loader=_chunk_loader,
                chunk_days=bt_chunk_days,
                strategy_timeframe=str(get_default_runtime_config().trading.timeframe),
                data_handler_cls=selected_data_handler_cls,
                execution_handler_cls=SimulatedExecutionHandler,
                portfolio_cls=Portfolio,
                backtest_mode=str(resolved_backtest_mode),
                data_handler_kwargs=selected_data_handler_kwargs,
                record_history=bool(execution_profile["record_history"]),
                track_metrics=bool(execution_profile["track_metrics"]),
                record_trades=bool(execution_profile["record_trades"]),
            )
            if bool(execution_profile["low_memory"]):
                _persist_low_memory_outputs(backtest, execution_profile["persist_output"])
                _print_low_memory_stats(backtest)
            else:
                backtest._output_performance(
                    persist_output=bool(execution_profile["persist_output"]),
                    verbose=True,
                )
        else:
            backtest = Backtest(
                csv_dir=CSV_DIR,
                symbol_list=symbol_list,
                start_date=start_date,
                end_date=end_date,
                data_handler_cls=selected_data_handler_cls,
                execution_handler_cls=SimulatedExecutionHandler,
                portfolio_cls=Portfolio,
                strategy_cls=strategy_cls,
                strategy_params=strategy_params,
                data_dict=data_dict,
                data_handler_kwargs=selected_data_handler_kwargs,
                record_history=bool(execution_profile["record_history"]),
                track_metrics=bool(execution_profile["track_metrics"]),
                record_trades=bool(execution_profile["record_trades"]),
                strategy_timeframe=str(get_default_runtime_config().trading.timeframe),
            )
            if bool(execution_profile["low_memory"]):
                backtest.simulate_trading(output=False)
                _persist_low_memory_outputs(backtest, execution_profile["persist_output"])
                _print_low_memory_stats(backtest)
            else:
                backtest.simulate_trading(
                    persist_output=bool(execution_profile["persist_output"]),
                )

        persisted_counts = {"equity_rows": 0, "fill_rows": 0}
        if audit_store is not None:
            persisted_counts = _persist_backtest_audit_rows(
                audit_store,
                backtest_run_id,
                backtest,
                low_memory=bool(execution_profile["low_memory"]),
            )
        summary_metadata = {}
        if bool(execution_profile["low_memory"]):
            try:
                summary_metadata["summary_stats_fast"] = dict(
                    backtest.portfolio.output_summary_stats_fast() or {}
                )
            except Exception:
                summary_metadata["summary_stats_fast"] = {}
        if audit_store is not None:
            audit_store.end_run(
                backtest_run_id,
                status="COMPLETED",
                metadata={
                    "final_equity": float(backtest.portfolio.current_holdings.get("total", 0.0)),
                    **persisted_counts,
                    **summary_metadata,
                },
            )
    except Exception as exc:
        if audit_store is not None:
            audit_store.end_run(
                backtest_run_id,
                status="FAILED",
                metadata={"error": str(exc)},
            )
        raise
    finally:
        if audit_store is not None:
            audit_store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LuminaQuant backtest.")
    settings = _current_backtest_runtime_settings()
    parser.add_argument(
        "--data-mode",
        choices=["raw-first", "legacy"],
        default=str(settings["data_mode"]),
        help="Data contract mode. raw-first requires committed manifest parquet windows.",
    )
    parser.add_argument(
        "--data-source",
        choices=["auto", "csv", "db", "external"],
        default=str(get_default_runtime_config().backtest.data_source or "auto"),
        help="Market data source (auto: DB first then CSV fallback; external: user-managed CSV/parquet root).",
    )
    parser.add_argument(
        "--market-db-path",
        default=str(settings["market_db_path"]),
        help="Market data parquet root path.",
    )
    parser.add_argument(
        "--external-data-root",
        default=str(get_default_runtime_config().backtest.external.root_path or ""),
        help="External market-data root for --data-source external (canonical CSV/parquet OHLCV).",
    )
    parser.add_argument(
        "--market-exchange",
        default=str(settings["market_db_exchange"]),
        help="Exchange key used in OHLCV DB rows.",
    )
    parser.add_argument(
        "--base-timeframe",
        default=str(settings["base_timeframe"]),
        help="Collection/backtest source timeframe. Use the minimum resolution (recommended: 1s).",
    )
    parser.add_argument(
        "--backtest-mode",
        choices=["windowed", "legacy_batch", "legacy_1s"],
        default=str(settings["backtest_mode"]),
        help="Backtest event model: windowed (default) or legacy modes.",
    )
    parser.add_argument(
        "--portfolio-mode",
        default=str(os.getenv("LQ_BACKTEST_PORTFOLIO_MODE", "") or ""),
        help=(
            "Run a live portfolio mode through the event-driven backtest path "
            "(for example production_guarded_state_vwap_pair_mode)."
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional external run_id for audit trail correlation.",
    )
    parser.add_argument(
        "--no-auto-collect-db",
        action="store_true",
        help="Disable automatic DB market-data collection before loading.",
    )
    low_memory_group = parser.add_mutually_exclusive_group()
    low_memory_group.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        help=(
            "Use low-memory execution profile (record_history=False, record_trades=False, "
            "track_metrics=True)."
        ),
    )
    low_memory_group.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Explicitly disable low-memory profile even if LQ_BACKTEST_LOW_MEMORY is set.",
    )
    parser.set_defaults(low_memory=None)
    persist_group = parser.add_mutually_exclusive_group()
    persist_group.add_argument(
        "--persist-output",
        dest="persist_output",
        action="store_true",
        help="Force writing CSV outputs (equity/trades).",
    )
    persist_group.add_argument(
        "--no-persist-output",
        dest="persist_output",
        action="store_false",
        help="Force disabling CSV outputs (equity/trades).",
    )
    parser.set_defaults(persist_output=None)
    args = parser.parse_args(argv)
    try:
        run(
            data_source=args.data_source,
            data_mode=args.data_mode,
            market_db_path=args.market_db_path,
            market_exchange=args.market_exchange,
            base_timeframe=_normalize_timeframe_or_default(args.base_timeframe, "1s"),
            external_data_root=args.external_data_root,
            auto_collect_db=(
                not bool(args.no_auto_collect_db) and bool(settings["auto_collect_db"])
            ),
            run_id=args.run_id,
            low_memory=args.low_memory,
            persist_output=args.persist_output,
            backtest_mode=args.backtest_mode,
            portfolio_mode=args.portfolio_mode,
        )
    except (
        RawFirstDataMissingError,
        RawFirstManifestInvalidError,
        RawFirstStaleWindowError,
    ) as exc:
        code = raw_first_exit_code(exc)
        if code is None:
            raise
        return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
