"""Automatic market-data collection helpers for DB-backed workflows."""

from __future__ import annotations

from datetime import UTC, datetime

from lumina_quant.data_sync import (
    create_binance_futures_client,
    ensure_market_data_coverage,
    parse_timestamp_input,
    sync_symbol_aggtrades_raw,
    sync_futures_feature_points,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository, normalize_symbol


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
    exchange = create_binance_futures_client(
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
    feature_profile: str = "full",
    include_funding: bool | None = None,
    include_mark_index: bool | None = None,
    include_open_interest: bool | None = None,
    include_liquidations: bool | None = None,
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
    profile_token = str(feature_profile or "full").strip().lower().replace("-", "_")
    if profile_token in {"", "default"}:
        profile_token = "full"
    if profile_token not in {"full", "strategy_used"}:
        raise ValueError("feature_profile must be one of: full, strategy_used")

    resolved_include_funding = bool(include_funding) if include_funding is not None else True
    resolved_include_mark_index = (
        bool(include_mark_index) if include_mark_index is not None else profile_token == "full"
    )
    resolved_include_open_interest = (
        bool(include_open_interest) if include_open_interest is not None else True
    )
    resolved_include_liquidations = (
        bool(include_liquidations) if include_liquidations is not None else True
    )

    features: list[str] = []
    if resolved_include_funding:
        features.extend(
            [
                "funding_rate",
                "funding_mark_price",
                "funding_fee_rate",
                "funding_fee_quote_per_unit",
            ]
        )
    if resolved_include_mark_index:
        features.extend(["mark_price", "index_price"])
    if resolved_include_open_interest:
        features.append("open_interest")
    if resolved_include_liquidations:
        features.extend(
            [
                "liquidation_long_qty",
                "liquidation_short_qty",
                "liquidation_long_notional",
                "liquidation_short_notional",
            ]
        )

    plan: dict[str, object] = {
        "db_path": str(db_path),
        "exchange_id": str(exchange_id),
        "symbol_list": list(symbol_list),
        "since_ms": int(since_ms),
        "until_ms": int(effective_until),
        "mark_index_interval": str(mark_index_interval),
        "open_interest_period": str(open_interest_period),
        "feature_profile": profile_token,
        "feature_groups": {
            "funding": bool(resolved_include_funding),
            "mark_index": bool(resolved_include_mark_index),
            "open_interest": bool(resolved_include_open_interest),
            "liquidations": bool(resolved_include_liquidations),
        },
        "backend": str(backend or ""),
        "execute": bool(execute),
        "features": features,
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
        include_funding=bool(resolved_include_funding),
        include_mark_index=bool(resolved_include_mark_index),
        include_open_interest=bool(resolved_include_open_interest),
        include_liquidations=bool(resolved_include_liquidations),
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


def collect_binance_aggtrades_raw(
    *,
    db_path: str,
    exchange_id: str,
    symbol: str,
    market_type: str = "future",
    api_key: str = "",
    secret_key: str = "",
    testnet: bool = False,
    since_ms: int | None = None,
    until_ms: int | None = None,
    bootstrap_lookback_hours: int = 24,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
) -> dict[str, int | str | None]:
    """Collect Binance aggTrades as raw canonical source with checkpoint resume."""
    repo = ParquetMarketDataRepository(str(db_path))
    normalized_symbol = normalize_symbol(symbol)
    exchange_token = str(exchange_id or "binance").strip().lower()
    checkpoint = repo.read_raw_checkpoint(exchange=exchange_token, symbol=normalized_symbol)
    checkpoint_cursor = checkpoint.get("last_timestamp_ms")
    if checkpoint_cursor is not None:
        try:
            checkpoint_cursor = int(checkpoint_cursor)
        except Exception:
            checkpoint_cursor = None

    end_cursor = (
        int(until_ms) if until_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    )
    start_cursor = int(since_ms) if since_ms is not None else None
    if start_cursor is None:
        if checkpoint_cursor is not None:
            start_cursor = int(checkpoint_cursor + 1)
        else:
            lookback_ms = max(1, int(bootstrap_lookback_hours)) * 60 * 60 * 1000
            start_cursor = max(0, int(end_cursor) - int(lookback_ms))

    exchange = create_binance_futures_client(
        api_key=api_key,
        secret_key=secret_key,
        market_type=market_type,
        testnet=testnet,
    )
    try:
        result = sync_symbol_aggtrades_raw(
            exchange=exchange,
            db_path=str(db_path),
            exchange_id=exchange_token,
            symbol=normalized_symbol,
            start_ms=int(start_cursor),
            end_ms=int(end_cursor),
            limit=max(1, int(limit)),
            max_batches=max(1, int(max_batches)),
            retries=max(0, int(retries)),
            base_wait_sec=float(base_wait_sec),
            resume_from_checkpoint=True,
        )
    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            close_fn()

    return {
        "symbol": normalized_symbol,
        "exchange": exchange_token,
        "start_cursor_ms": int(start_cursor),
        "end_cursor_ms": int(end_cursor),
        "fetched_rows": int(result.fetched_rows),
        "upserted_rows": int(result.upserted_rows),
        "last_timestamp_ms": result.last_timestamp_ms,
        "last_trade_id": result.checkpoint_trade_id,
        "last_agg_trade_id": result.checkpoint_trade_id,
    }
