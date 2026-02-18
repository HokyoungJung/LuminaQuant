"""Automatic market-data collection helpers for DB-backed workflows."""

from __future__ import annotations

from datetime import UTC, datetime

from lumina_quant.data_sync import create_binance_exchange, ensure_market_data_coverage


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
) -> list[dict[str, int | str | None]]:
    """Ensure requested OHLCV coverage exists in SQLite and return sync summary."""
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
            limit=max(1, int(limit)),
            max_batches=max(1, int(max_batches)),
            retries=max(0, int(retries)),
            base_wait_sec=float(base_wait_sec),
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
