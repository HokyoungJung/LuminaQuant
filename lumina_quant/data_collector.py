"""Automatic market-data collection helpers for DB-backed workflows."""

from __future__ import annotations

from datetime import UTC, datetime

from lumina_quant.data_sync import (
    create_binance_exchange,
    ensure_market_data_coverage,
    parse_timestamp_input,
    sync_futures_feature_points,
)


def _datetime_to_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def auto_collect_market_data(
    *,
    symbol_list: list[str],
    timeframe: str,
    db_path: str,
    exchange_id: str,
    market_type: str,
    since_dt: datetime | None,
    until_dt: datetime | None,
    api_key: str = "",
    secret_key: str = "",
    testnet: bool = False,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
    force_full: bool = False,
    backend: str | None = None,
    **legacy: object,
) -> list[dict[str, int | str | None]]:
    """Ensure requested OHLCV coverage exists in parquet storage and return summary."""
    _ = legacy
    exchange = create_binance_exchange(
        api_key=api_key,
        secret_key=secret_key,
        market_type=market_type,
        testnet=testnet,
    )
    try:
        stats = ensure_market_data_coverage(
            exchange=exchange,
            db_path=db_path,
            exchange_id=exchange_id,
            symbol_list=symbol_list,
            timeframe=timeframe,
            since_ms=_datetime_to_ms(since_dt),
            until_ms=_datetime_to_ms(until_dt),
            force_full=bool(force_full),
            limit=max(1, int(limit)),
            max_batches=max(1, int(max_batches)),
            retries=max(0, int(retries)),
            base_wait_sec=float(base_wait_sec),
            backend=backend,
        )
    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            close_fn()

    return [
        {
            "symbol": item.symbol,
            "fetched_rows": int(item.fetched_rows),
            "upserted_rows": int(item.upserted_rows),
            "first_timestamp_ms": item.first_timestamp_ms,
            "last_timestamp_ms": item.last_timestamp_ms,
        }
        for item in stats
    ]


def collect_strategy_support_data(
    *,
    db_path: str,
    exchange_id: str,
    symbol_list: list[str],
    since: str | int | float,
    until: str | int | float | None = None,
    mark_index_interval: str = "1m",
    open_interest_period: str = "5m",
    retries: int = 3,
    execute: bool = False,
    backend: str | None = None,
    **legacy: object,
) -> dict[str, object]:
    """Prepare or execute strategy-support data collection.

    Defaults to plan-only mode so no network data is fetched unless execute=True.
    """
    _ = legacy
    since_ms = parse_timestamp_input(since)
    until_ms = parse_timestamp_input(until)
    if since_ms is None:
        raise ValueError("since must resolve to a valid timestamp")
    effective_until = (
        int(until_ms) if until_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    )

    plan: dict[str, object] = {
        "db_path": str(db_path),
        "exchange_id": str(exchange_id),
        "symbol_list": list(symbol_list),
        "since_ms": int(since_ms),
        "until_ms": int(effective_until),
        "mark_index_interval": str(mark_index_interval),
        "open_interest_period": str(open_interest_period),
        "backend": str(backend or ""),
        "execute": bool(execute),
        "features": [
            "funding_rate",
            "funding_mark_price",
            "mark_price",
            "index_price",
            "open_interest",
            "liquidation_long_qty",
            "liquidation_short_qty",
            "liquidation_long_notional",
            "liquidation_short_notional",
        ],
    }

    if not execute:
        plan["status"] = "planned_only"
        plan["upserted_rows"] = 0
        return plan

    stats = sync_futures_feature_points(
        db_path=str(db_path),
        exchange_id=str(exchange_id),
        symbol_list=list(symbol_list),
        since_ms=int(since_ms),
        until_ms=int(effective_until),
        mark_index_interval=str(mark_index_interval),
        open_interest_period=str(open_interest_period),
        retries=max(0, int(retries)),
        backend=backend,
    )
    upserted_rows = sum(int(item.upserted_rows) for item in stats)
    plan["status"] = "executed"
    plan["upserted_rows"] = int(upserted_rows)
    plan["per_symbol"] = [
        {
            "symbol": row.symbol,
            "upserted_rows": int(row.upserted_rows),
            "first_timestamp_ms": row.first_timestamp_ms,
            "last_timestamp_ms": row.last_timestamp_ms,
        }
        for row in stats
    ]
    return plan
