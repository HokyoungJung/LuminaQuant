"""Shared MARKET_WINDOW parity contract builder for live/backtest."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.core.events import MarketWindowEvent


class MarketWindowContractError(RuntimeError):
    """Raised when MARKET_WINDOW payload fails strict parity contract."""


def _to_epoch_ms(value: Any, *, field_name: str) -> int:
    if value is None:
        raise MarketWindowContractError(f"{field_name} must be present.")
    if isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            return int(numeric * 1000)
        return int(numeric)
    if isinstance(value, datetime):
        parsed = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)

    ts_fn = getattr(value, "timestamp", None)
    if callable(ts_fn):
        try:
            ts = ts_fn()
        except Exception as exc:  # pragma: no cover - defensive
            raise MarketWindowContractError(
                f"{field_name} timestamp conversion failed: {exc}"
            ) from exc
        if isinstance(ts, (int, float)):
            return _to_epoch_ms(float(ts), field_name=field_name)

    raise MarketWindowContractError(
        f"{field_name} must be epoch-ms compatible int/float/datetime. Received: {type(value)!r}"
    )


def _normalize_row(row: Any, *, symbol: str) -> tuple[int, float, float, float, float, float]:
    if not isinstance(row, (tuple, list)) or len(row) < 6:
        raise MarketWindowContractError(
            f"bars_1s[{symbol}] rows must contain 6 items: (time, open, high, low, close, volume)."
        )
    return (
        _to_epoch_ms(row[0], field_name=f"bars_1s[{symbol}].time"),
        float(row[1]),
        float(row[2]),
        float(row[3]),
        float(row[4]),
        float(row[5]),
    )


def normalize_bars_1s(
    bars_1s: dict[str, tuple[Any, ...]] | dict[str, list[Any]] | None,
) -> dict[str, tuple[tuple[int, float, float, float, float, float], ...]]:
    if not isinstance(bars_1s, dict):
        raise MarketWindowContractError("bars_1s must be a dict[str, rows].")

    normalized: dict[str, tuple[tuple[int, float, float, float, float, float], ...]] = {}
    for symbol, rows in bars_1s.items():
        symbol_key = str(symbol)
        if rows is None:
            normalized[symbol_key] = tuple()
            continue
        if not isinstance(rows, (tuple, list)):
            raise MarketWindowContractError(
                f"bars_1s[{symbol_key}] must be tuple/list of OHLCV rows."
            )
        fast_rows = _normalized_sorted_rows_fast_path(rows)
        if fast_rows is not None:
            normalized[symbol_key] = fast_rows
            continue
        converted = [_normalize_row(row, symbol=symbol_key) for row in rows]
        converted.sort(key=lambda item: item[0])
        normalized[symbol_key] = tuple(converted)
    return normalized


def _reuse_normalized_bars_1s(
    bars_1s: dict[str, tuple[Any, ...]] | dict[str, list[Any]] | None,
) -> dict[str, tuple[tuple[int, float, float, float, float, float], ...]]:
    """Reuse trusted canonical MARKET_WINDOW rows without per-row validation.

    This is only for internal producers that already materialize rows as
    `(epoch_ms:int, open:float, high:float, low:float, close:float, volume:float)`
    in ascending order. External/user payloads must continue through
    `normalize_bars_1s`.
    """
    if not isinstance(bars_1s, dict):
        raise MarketWindowContractError("bars_1s must be a dict[str, rows].")

    normalized: dict[str, tuple[tuple[int, float, float, float, float, float], ...]] = {}
    for symbol, rows in bars_1s.items():
        symbol_key = str(symbol)
        if rows is None:
            normalized[symbol_key] = tuple()
            continue
        if isinstance(rows, tuple):
            normalized[symbol_key] = rows
            continue
        if isinstance(rows, list):
            normalized[symbol_key] = tuple(rows)
            continue
        raise MarketWindowContractError(f"bars_1s[{symbol_key}] must be tuple/list of OHLCV rows.")
    return normalized


def _normalized_sorted_rows_fast_path(
    rows: tuple[Any, ...] | list[Any],
) -> tuple[tuple[int, float, float, float, float, float], ...] | None:
    """Reuse already-normalized MARKET_WINDOW rows without per-tick casting/sorting.

    `build_market_window_event` is called for every backtest poll. Backtest
    window handlers already know their 1s rows are in ascending epoch-ms order,
    while live/external callers may still need the slower validation path. This
    helper only accepts rows that are already exactly in the canonical contract
    representation, so the returned payload is byte-for-byte equivalent to the
    generic normalize/sort path.
    """
    if not rows:
        return tuple()

    out: list[tuple[int, float, float, float, float, float]] = []
    previous_ts: int | None = None
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) < 6:
            return None
        ts = row[0]
        if not isinstance(ts, int) or abs(ts) < 100_000_000_000:
            return None
        if previous_ts is not None and int(ts) < previous_ts:
            return None
        open_, high, low, close, volume = row[1], row[2], row[3], row[4], row[5]
        if not all(isinstance(value, float) for value in (open_, high, low, close, volume)):
            return None
        out.append((int(ts), open_, high, low, close, volume))
        previous_ts = int(ts)
    return tuple(out)


def validate_market_window_event_schema(
    event: MarketWindowEvent,
    *,
    bars_1s_already_normalized: bool = False,
) -> None:
    if str(getattr(event, "type", "")) != "MARKET_WINDOW":
        raise MarketWindowContractError("type must equal 'MARKET_WINDOW'.")
    if not isinstance(getattr(event, "time", None), int):
        raise MarketWindowContractError("time must be int epoch ms.")
    if not isinstance(getattr(event, "window_seconds", None), int):
        raise MarketWindowContractError("window_seconds must be int.")
    if int(event.window_seconds) < 1:
        raise MarketWindowContractError("window_seconds must be >= 1.")

    if bool(bars_1s_already_normalized):
        if not isinstance(getattr(event, "bars_1s", None), dict):
            raise MarketWindowContractError("bars_1s must be a dict[str, rows].")
    else:
        normalized = normalize_bars_1s(getattr(event, "bars_1s", None))
        event.bars_1s = normalized

    watermark = getattr(event, "event_time_watermark_ms", None)
    if watermark is not None and not isinstance(watermark, int):
        raise MarketWindowContractError("event_time_watermark_ms must be int | None.")

    commit_id = getattr(event, "commit_id", None)
    if commit_id is not None and not isinstance(commit_id, str):
        raise MarketWindowContractError("commit_id must be str | None.")

    lag_ms = getattr(event, "lag_ms", None)
    if lag_ms is not None and not isinstance(lag_ms, int):
        raise MarketWindowContractError("lag_ms must be int | None.")

    if not isinstance(getattr(event, "is_stale", None), bool):
        raise MarketWindowContractError("is_stale must be bool.")

    timestamp_ns = getattr(event, "timestamp_ns", None)
    if timestamp_ns is not None and not isinstance(timestamp_ns, int):
        raise MarketWindowContractError("timestamp_ns must be int | None.")

    sequence = getattr(event, "sequence", None)
    if sequence is not None and not isinstance(sequence, int):
        raise MarketWindowContractError("sequence must be int | None.")


def serialize_market_window_event(event: MarketWindowEvent) -> bytes:
    payload = market_window_event_payload(event)
    payload["bars_1s"] = {
        str(symbol): [
            [
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ]
            for row in rows
        ]
        for symbol, rows in dict(event.bars_1s).items()
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def market_window_event_payload(event: MarketWindowEvent) -> dict[str, Any]:
    payload = {
        "time": int(event.time),
        "window_seconds": int(event.window_seconds),
        "bars_1s": {
            str(symbol): tuple(
                (
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                )
                for row in rows
            )
            for symbol, rows in dict(event.bars_1s).items()
        },
        "event_time_watermark_ms": (
            int(event.event_time_watermark_ms)
            if event.event_time_watermark_ms is not None
            else None
        ),
        "commit_id": str(event.commit_id) if event.commit_id is not None else None,
        "lag_ms": int(event.lag_ms) if event.lag_ms is not None else None,
        "is_stale": bool(event.is_stale),
        "timestamp_ns": int(event.timestamp_ns) if event.timestamp_ns is not None else None,
        "sequence": int(event.sequence) if event.sequence is not None else None,
        "type": "MARKET_WINDOW",
    }
    return payload


def emit_market_window_metrics(
    event: MarketWindowEvent,
    *,
    parity_v2_enabled: bool,
    fail_fast_incident: bool,
    metrics_log_path: str,
) -> None:
    payload_bytes = len(serialize_market_window_event(event))
    queue_lag_ms = int(getattr(event, "lag_ms", 0) or 0)
    timestamp_ms = int(getattr(event, "time", 0) or 0)

    log_payload = {
        "timestamp_ms": int(timestamp_ms),
        "payload_bytes": int(payload_bytes),
        "queue_lag_ms": int(queue_lag_ms),
        "parity_v2_enabled": bool(parity_v2_enabled),
        "fail_fast_incident": bool(fail_fast_incident),
    }

    log_path = Path(str(metrics_log_path or "logs/live/market_window_metrics.ndjson"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_payload, ensure_ascii=False) + "\n")


def append_market_window_metrics(
    *,
    event: MarketWindowEvent,
    metrics_path: str,
    parity_v2_enabled: bool,
    fail_fast_incident: bool = False,
) -> None:
    """Backward-compatible alias for rollout metric append helper."""
    emit_market_window_metrics(
        event,
        parity_v2_enabled=bool(parity_v2_enabled),
        fail_fast_incident=bool(fail_fast_incident),
        metrics_log_path=str(metrics_path),
    )


def build_market_window_event(
    *,
    time: Any,
    window_seconds: int,
    bars_1s: dict[str, tuple[Any, ...]] | dict[str, list[Any]],
    event_time_watermark_ms: Any = None,
    commit_id: str | None = None,
    lag_ms: int | None = None,
    is_stale: bool = False,
    timestamp_ns: int | None = None,
    sequence: int | None = None,
    parity_v2_enabled: bool = False,
    metrics_log_path: str = "logs/live/market_window_metrics.ndjson",
    emit_metrics: bool = False,
    bars_1s_already_normalized: bool = False,
) -> MarketWindowEvent:
    time_ms = _to_epoch_ms(time, field_name="time")
    normalized_rows = (
        _reuse_normalized_bars_1s(bars_1s)
        if bool(bars_1s_already_normalized)
        else normalize_bars_1s(bars_1s)
    )

    watermark_ms = (
        _to_epoch_ms(event_time_watermark_ms, field_name="event_time_watermark_ms")
        if event_time_watermark_ms is not None
        else None
    )
    commit_value = str(commit_id) if commit_id is not None else None
    lag_value = int(lag_ms) if lag_ms is not None else None
    stale_value = bool(is_stale)
    ts_ns = int(timestamp_ns) if timestamp_ns is not None else None
    seq = int(sequence) if sequence is not None else None

    event = MarketWindowEvent(
        time=int(time_ms),
        window_seconds=max(1, int(window_seconds)),
        bars_1s=normalized_rows,
        event_time_watermark_ms=watermark_ms,
        commit_id=commit_value,
        lag_ms=lag_value,
        is_stale=stale_value,
        timestamp_ns=ts_ns,
        sequence=seq,
    )
    validate_market_window_event_schema(event, bars_1s_already_normalized=True)

    if emit_metrics and bool(parity_v2_enabled):
        emit_market_window_metrics(
            event,
            parity_v2_enabled=True,
            fail_fast_incident=False,
            metrics_log_path=str(metrics_log_path),
        )
    return event


__all__ = [
    "MarketWindowContractError",
    "append_market_window_metrics",
    "build_market_window_event",
    "emit_market_window_metrics",
    "market_window_event_payload",
    "normalize_bars_1s",
    "serialize_market_window_event",
    "validate_market_window_event_schema",
]
