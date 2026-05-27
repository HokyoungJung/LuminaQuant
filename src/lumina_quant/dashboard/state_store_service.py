"""Dashboard data/state access helpers retained from the retired legacy app."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class StateCursor:
    """Thin adapter that normalizes cursor parameter handling."""

    def __init__(self, cursor: Any):
        self._cursor = cursor

    def execute(self, query: Any, params: Any = None) -> StateCursor:
        self._cursor.execute(str(query), tuple(params or ()))
        return self

    def __enter__(self) -> StateCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def close(self) -> None:
        self._cursor.close()

    @property
    def description(self):
        return self._cursor.description


class StateConnection:
    """Connection adapter with sqlite-like helpers used by the dashboard."""

    def __init__(self, connection: Any):
        self._conn = connection

    def cursor(self) -> StateCursor:
        return StateCursor(self._conn.cursor())

    def execute(self, query: Any, params: Any = None) -> StateCursor:
        cursor = self.cursor()
        cursor.execute(query, params)
        return cursor

    def executescript(self, script: str) -> None:
        with self._conn.cursor() as cursor:
            for statement in str(script).split(";"):
                payload = statement.strip()
                if not payload:
                    continue
                cursor.execute(payload)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def resolve_postgres_dsn(dsn: str | None = None, *, base_config: Any = None) -> str:
    """Resolve dashboard Postgres DSN from explicit value, env, or config."""
    token = str(
        dsn or os.getenv("LQ_POSTGRES_DSN") or getattr(base_config, "POSTGRES_DSN", "") or ""
    ).strip()
    return token


def connect_state_store(
    dsn: str,
    *,
    resolve_postgres_dsn=resolve_postgres_dsn,
) -> StateConnection:
    """Open a normalized dashboard state-store connection."""
    resolved = resolve_postgres_dsn(dsn)
    if not resolved:
        raise RuntimeError("Postgres DSN is required.")
    from lumina_quant.postgres_state import _connect_postgres

    return StateConnection(_connect_postgres(resolved))


def execute_query(
    dsn: str,
    query: str,
    params: Any = None,
    *,
    connect_state_store=connect_state_store,
    logger: logging.Logger | None = None,
):
    """Execute a query and tolerate fetchall failures with an empty result."""
    active_logger = logger or logging.getLogger(__name__)
    conn = connect_state_store(dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params or ()))
            try:
                rows = cursor.fetchall()
            except Exception:
                active_logger.warning(
                    "Dashboard query helper fell back to an empty result set after fetchall failed."
                )
                rows = []
        conn.commit()
        return rows
    finally:
        conn.close()


def read_sql_query(
    dsn: str,
    query: str,
    params: Any = None,
    *,
    connect_state_store=connect_state_store,
):
    """Run ``pandas.read_sql_query`` against the normalized state store."""
    conn = connect_state_store(dsn)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def load_runs_frame(
    dsn: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    limit: int = 300,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT
                r.run_id,
                r.mode,
                r.started_at,
                r.ended_at,
                r.status,
                r.metadata_json AS metadata,
                COALESCE(
                    (r.metadata_json ->> 'strategy'),
                    (
                        SELECT w.strategy
                        FROM workflow_jobs w
                        WHERE w.run_id = r.run_id
                        ORDER BY w.started_at DESC
                        LIMIT 1
                    )
                ) AS strategy,
                COALESCE(eq.equity_rows, 0) AS equity_rows,
                COALESCE(fl.fill_rows, 0) AS fill_rows,
                COALESCE(od.order_rows, 0) AS order_rows,
                COALESCE(rk.risk_rows, 0) AS risk_rows,
                COALESCE(hb.hb_rows, 0) AS hb_rows,
                eq.last_equity_at,
                hb.last_heartbeat_at
            FROM runs r
            LEFT JOIN (
                SELECT run_id, COUNT(*) AS equity_rows, MAX(timeindex) AS last_equity_at
                FROM equity
                GROUP BY run_id
            ) eq ON eq.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS fill_rows FROM fills GROUP BY run_id) fl ON fl.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS order_rows FROM orders GROUP BY run_id) od ON od.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS risk_rows FROM risk_events GROUP BY run_id) rk ON rk.run_id = r.run_id
            LEFT JOIN (
                SELECT run_id, COUNT(*) AS hb_rows, MAX(heartbeat_time) AS last_heartbeat_at
                FROM heartbeats
                GROUP BY run_id
            ) hb ON hb.run_id = r.run_id
            ORDER BY r.started_at DESC
            LIMIT {int(max(1, limit))}
            """,
            conn,
        )
        for column in ("started_at", "ended_at", "last_equity_at", "last_heartbeat_at"):
            df = coerce_datetime(df, column)
        return df
    finally:
        conn.close()


def load_equity_state_frame(
    dsn: str,
    run_id: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT timeindex AS datetime, total, cash, metadata_json AS metadata
            FROM (
                SELECT id, timeindex, total, cash, metadata_json
                FROM equity
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return coerce_datetime(df, "datetime")
    finally:
        conn.close()


def load_metrics_state_frame(
    dsn: str,
    run_id: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    parse_json_dict: Callable[[Any], dict[str, Any]],
    max_points: int,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                timeindex AS datetime,
                total,
                cash,
                metadata_json AS metadata
            FROM (
                SELECT id, timeindex, total, cash, metadata_json
                FROM equity
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        df = coerce_datetime(df, "datetime")
        if df.empty:
            return df

        benchmark: list[Any] = []
        funding: list[Any] = []
        symbol: list[Any] = []
        for meta in df["metadata"].tolist() if "metadata" in df.columns else []:
            info = parse_json_dict(meta)
            benchmark.append(info.get("benchmark_price"))
            funding.append(info.get("funding_total"))
            symbol.append(info.get("symbol"))

        if benchmark:
            df["benchmark_price"] = pd.to_numeric(pd.Series(benchmark), errors="coerce")
        if funding:
            df["funding"] = pd.to_numeric(pd.Series(funding), errors="coerce").fillna(0.0)
        if symbol:
            df["event_symbol"] = pd.Series(symbol)
        return df
    finally:
        conn.close()


def load_fills_state_frame(
    dsn: str,
    run_id: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                fill_time AS datetime,
                symbol,
                side AS direction,
                quantity,
                fill_cost,
                commission,
                fill_price AS price,
                status,
                metadata_json AS metadata,
                exchange_order_id,
                client_order_id
            FROM (
                SELECT id, fill_time, symbol, side, quantity, fill_cost, commission,
                       fill_price, status, metadata_json, exchange_order_id, client_order_id
                FROM fills
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        if not df.empty and "direction" in df.columns:
            df["direction"] = (
                df["direction"]
                .fillna("")
                .astype(str)
                .str.upper()
                .map({"BUY": "BUY", "SELL": "SELL", "BUY_LONG": "BUY", "SELL_SHORT": "SELL"})
                .fillna(df["direction"])
            )
        return coerce_datetime(df, "datetime")
    finally:
        conn.close()


def load_orders_state_frame(
    dsn: str,
    run_id: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT created_at, symbol, side, order_type, quantity, price, status,
                   client_order_id, exchange_order_id, metadata_json AS metadata
            FROM (
                SELECT id, created_at, symbol, side, order_type, quantity, price, status,
                       client_order_id, exchange_order_id, metadata_json
                FROM orders
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return coerce_datetime(df, "created_at")
    finally:
        conn.close()


def load_risk_events_state_frame(
    dsn: str,
    run_id: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT event_time, reason, details_json AS details
            FROM (
                SELECT id, event_time, reason, details_json
                FROM risk_events
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return coerce_datetime(df, "event_time")
    finally:
        conn.close()


def load_heartbeats_state_frame(
    dsn: str,
    run_id: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT heartbeat_time, status, details_json AS details
            FROM (
                SELECT id, heartbeat_time, status, details_json
                FROM heartbeats
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return coerce_datetime(df, "heartbeat_time")
    finally:
        conn.close()


def load_order_states_state_frame(
    dsn: str,
    run_id: str,
    *,
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT event_time, symbol, client_order_id, exchange_order_id, state, message, details_json AS details
            FROM (
                SELECT id, event_time, symbol, client_order_id, exchange_order_id, state, message, details_json
                FROM order_state_events
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return coerce_datetime(df, "event_time")
    finally:
        conn.close()


def load_optimization_results_state_frame(
    dsn: str,
    *,
    resolve_postgres_dsn: Callable[[str | None], str],
    connect_state_store=connect_state_store,
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    parse_json_dict: Callable[[Any], dict[str, Any]],
    max_points: int,
) -> pd.DataFrame:
    if not resolve_postgres_dsn(dsn):
        return pd.DataFrame()
    conn = connect_state_store(dsn)
    try:
        df = pd.read_sql_query(
            """
            SELECT id, run_id, stage, created_at, params_json, sharpe, cagr, mdd,
                   train_sharpe, robustness_score, extra_json
            FROM (
                SELECT id, run_id, stage, created_at, params_json, sharpe, cagr, mdd,
                       train_sharpe, robustness_score, extra_json
                FROM optimization_results
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[int(max(1, max_points))],
        )
        if df.empty:
            return df
        df = coerce_datetime(df, "created_at")
        df["params"] = df["params_json"].apply(parse_json_dict)
        df["extra"] = df["extra_json"].apply(parse_json_dict)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def load_market_ohlcv_frame(
    root_path: str,
    symbol: str,
    timeframe: str,
    exchange_id: str,
    *,
    normalize_symbol: Callable[[str], str],
    resolve_dashboard_market_timeframe: Callable[[str], tuple[str, bool]],
    timeframe_to_milliseconds: Callable[[str], int],
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
    max_points: int,
    parquet_repo_cls: type[Any] | None = None,
    utc_now: Callable[[], datetime] | None = None,
) -> pd.DataFrame:
    resolved_root = str(root_path or "").strip()
    if not resolved_root:
        return pd.DataFrame()

    if parquet_repo_cls is None:
        from lumina_quant.parquet_market_data import ParquetMarketDataRepository

        parquet_repo_cls = ParquetMarketDataRepository

    symbol_token = normalize_symbol(symbol)
    timeframe_token, _ = resolve_dashboard_market_timeframe(timeframe)
    now_fn = utc_now or (lambda: datetime.now(UTC).replace(tzinfo=None))
    try:
        repo = parquet_repo_cls(resolved_root)
        interval_ms = max(1, int(timeframe_to_milliseconds(timeframe_token)))
        end_dt = now_fn()
        start_dt = end_dt - timedelta(milliseconds=interval_ms * int(max(2, max_points) * 2))
        frame = repo.load_ohlcv(
            exchange=str(exchange_id).strip().lower(),
            symbol=symbol_token,
            timeframe=timeframe_token,
            start_date=start_dt,
            end_date=end_dt,
        )
        if frame.is_empty():
            return pd.DataFrame()
        if frame.height > max_points:
            frame = frame.tail(int(max_points))
        return coerce_datetime(frame.to_pandas(), "datetime")
    except Exception:
        return pd.DataFrame()


__all__ = [
    "StateConnection",
    "StateCursor",
    "connect_state_store",
    "execute_query",
    "load_equity_state_frame",
    "load_fills_state_frame",
    "load_heartbeats_state_frame",
    "load_market_ohlcv_frame",
    "load_metrics_state_frame",
    "load_optimization_results_state_frame",
    "load_order_states_state_frame",
    "load_orders_state_frame",
    "load_risk_events_state_frame",
    "load_runs_frame",
    "read_sql_query",
    "resolve_postgres_dsn",
]
