"""Incremental timeframe aggregation from 1-second OHLCV bars."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

from lumina_quant.core.events import MarketEvent
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds

_DEFAULT_TIMEFRAMES = ("20s", "1m", "5m", "15m", "1h", "4h", "1d")
_DEFAULT_LOOKBACK = 4096
_DEFAULT_1S_LOOKBACK = 20_000


class TimeframeAggregator:
    """Build and retain rolling OHLCV bars for multiple timeframes."""

    def __init__(
        self,
        *,
        timeframes: list[str] | tuple[str, ...] | None = None,
        lookbacks: dict[str, int] | None = None,
    ) -> None:
        requested = tuple(timeframes or _DEFAULT_TIMEFRAMES)
        normalized = [normalize_timeframe_token(tf) for tf in requested]
        if "1s" not in normalized:
            normalized.insert(0, "1s")

        self._timeframes: tuple[str, ...] = tuple(dict.fromkeys(normalized))
        self._timeframe_ms: dict[str, int] = {
            tf: int(timeframe_to_milliseconds(tf)) for tf in self._timeframes
        }

        raw_lookbacks = dict(lookbacks or {})
        self._lookbacks: dict[str, int] = {}
        for tf in self._timeframes:
            if tf == "1s":
                default = _DEFAULT_1S_LOOKBACK
            else:
                default = _DEFAULT_LOOKBACK
            try:
                value = int(raw_lookbacks.get(tf, default))
            except Exception:
                value = default
            self._lookbacks[tf] = max(16, value)

        self._history: dict[str, dict[str, deque[tuple[Any, float, float, float, float, float]]]] = (
            defaultdict(dict)
        )
        self._working: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._last_seen_ms: dict[str, int] = {}

    @staticmethod
    def _coerce_timestamp_ms(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            ts = int(float(value))
            if abs(ts) < 100_000_000_000:
                ts *= 1000
            return ts
        ts_fn = getattr(value, "timestamp", None)
        if callable(ts_fn):
            try:
                ts_value = ts_fn()
                if isinstance(ts_value, (int, float)):
                    return int(float(ts_value) * 1000)
            except Exception:
                return None
        return None

    @staticmethod
    def _bucket_time(bucket_ms: int) -> datetime:
        return datetime.fromtimestamp(float(bucket_ms) / 1000.0, tz=UTC).replace(tzinfo=None)

    def _ensure_history(self, symbol: str, timeframe: str) -> deque:
        bucket = self._history[symbol]
        if timeframe not in bucket:
            bucket[timeframe] = deque(maxlen=int(self._lookbacks.get(timeframe, _DEFAULT_LOOKBACK)))
        return bucket[timeframe]

    @staticmethod
    def _coerce_bar(symbol: str, bar: Any) -> tuple[int, tuple[Any, float, float, float, float, float]] | None:
        if isinstance(bar, MarketEvent):
            ts_ms = TimeframeAggregator._coerce_timestamp_ms(getattr(bar, "time", None))
            if ts_ms is None:
                return None
            return ts_ms, (
                getattr(bar, "time", None),
                float(getattr(bar, "open", 0.0)),
                float(getattr(bar, "high", 0.0)),
                float(getattr(bar, "low", 0.0)),
                float(getattr(bar, "close", 0.0)),
                float(getattr(bar, "volume", 0.0)),
            )

        if isinstance(bar, dict):
            ts_ms = TimeframeAggregator._coerce_timestamp_ms(bar.get("time") or bar.get("datetime"))
            if ts_ms is None:
                return None
            return ts_ms, (
                bar.get("time") or bar.get("datetime"),
                float(bar.get("open", 0.0)),
                float(bar.get("high", 0.0)),
                float(bar.get("low", 0.0)),
                float(bar.get("close", 0.0)),
                float(bar.get("volume", 0.0)),
            )

        if isinstance(bar, (tuple, list)) and len(bar) >= 6:
            ts_ms = TimeframeAggregator._coerce_timestamp_ms(bar[0])
            if ts_ms is None:
                return None
            return ts_ms, (
                bar[0],
                float(bar[1]),
                float(bar[2]),
                float(bar[3]),
                float(bar[4]),
                float(bar[5]),
            )

        _ = symbol
        return None

    @staticmethod
    def _to_bar_tuple(working: dict[str, Any]) -> tuple[Any, float, float, float, float, float]:
        return (
            working["time"],
            float(working["open"]),
            float(working["high"]),
            float(working["low"]),
            float(working["close"]),
            float(working["volume"]),
        )

    def _update_aggregated_timeframe(
        self,
        *,
        symbol: str,
        timeframe: str,
        tf_ms: int,
        ts_ms: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
    ) -> None:
        bucket_ms = (int(ts_ms) // int(tf_ms)) * int(tf_ms)
        working = self._working[symbol].get(timeframe)
        if working is None:
            self._working[symbol][timeframe] = {
                "bucket_ms": bucket_ms,
                "time": self._bucket_time(bucket_ms),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
            }
            return

        if int(working.get("bucket_ms", bucket_ms)) != bucket_ms:
            self._ensure_history(symbol, timeframe).append(self._to_bar_tuple(working))
            self._working[symbol][timeframe] = {
                "bucket_ms": bucket_ms,
                "time": self._bucket_time(bucket_ms),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
            }
            return

        working["high"] = max(float(working["high"]), float(high_price))
        working["low"] = min(float(working["low"]), float(low_price))
        working["close"] = float(close_price)
        working["volume"] = float(working["volume"]) + float(volume)

    def update_from_1s_batch(
        self,
        symbol_or_bars: str | dict[str, tuple[Any, ...] | list[Any]],
        rows_1s: tuple[Any, ...] | list[Any] | None = None,
    ) -> None:
        """Update internal timeframe state from 1-second bars.

        The input may contain overlapping windows; bars at or before the last
        processed timestamp per symbol are ignored to keep updates incremental.
        """
        if isinstance(symbol_or_bars, str):
            bars_1s: dict[str, tuple[Any, ...] | list[Any]] = {str(symbol_or_bars): tuple(rows_1s or ())}
        else:
            bars_1s = dict(symbol_or_bars or {})

        for symbol, rows in bars_1s.items():
            if not rows:
                continue
            normalized: list[tuple[int, tuple[Any, float, float, float, float, float]]] = []
            for row in rows:
                parsed = self._coerce_bar(str(symbol), row)
                if parsed is not None:
                    normalized.append(parsed)
            if not normalized:
                continue

            normalized.sort(key=lambda item: item[0])
            last_seen = self._last_seen_ms.get(str(symbol))
            for ts_ms, bar in normalized:
                if last_seen is not None and int(ts_ms) <= int(last_seen):
                    continue

                last_seen = int(ts_ms)
                self._last_seen_ms[str(symbol)] = int(ts_ms)
                self._ensure_history(str(symbol), "1s").append(bar)

                _, open_price, high_price, low_price, close_price, volume = bar
                for timeframe, tf_ms in self._timeframe_ms.items():
                    if timeframe == "1s":
                        continue
                    self._update_aggregated_timeframe(
                        symbol=str(symbol),
                        timeframe=str(timeframe),
                        tf_ms=int(tf_ms),
                        ts_ms=int(ts_ms),
                        open_price=float(open_price),
                        high_price=float(high_price),
                        low_price=float(low_price),
                        close_price=float(close_price),
                        volume=float(volume),
                    )

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 1,
        *,
        n: int | None = None,
    ) -> list[tuple[Any, float, float, float, float, float]]:
        """Return up to `n` bars for symbol/timeframe including active bar."""
        token = normalize_timeframe_token(timeframe)
        history = list(self._history.get(str(symbol), {}).get(token, ()))
        working = self._working.get(str(symbol), {}).get(token)
        if working is not None:
            history.append(self._to_bar_tuple(working))
        effective_n = int(lookback_bars if n is None else n)
        if effective_n <= 0:
            return []
        return history[-effective_n:]

    def get_last_bar(self, symbol: str, timeframe: str) -> tuple[Any, float, float, float, float, float] | None:
        bars = self.get_bars(symbol=str(symbol), timeframe=str(timeframe), n=1)
        return bars[-1] if bars else None

    def get_state(self) -> dict[str, Any]:
        """Capture aggregator state for chunk-boundary continuity."""
        history_state: dict[str, dict[str, list[Any]]] = {}
        for symbol, by_tf in self._history.items():
            history_state[str(symbol)] = {tf: list(values) for tf, values in by_tf.items()}
        working_state: dict[str, dict[str, dict[str, Any]]] = {}
        for symbol, by_tf in self._working.items():
            working_state[str(symbol)] = {tf: dict(values) for tf, values in by_tf.items()}
        return {
            "timeframes": list(self._timeframes),
            "lookbacks": dict(self._lookbacks),
            "history": history_state,
            "working": working_state,
            "last_seen_ms": dict(self._last_seen_ms),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore aggregator state from `get_state()` output."""
        if not isinstance(state, dict):
            return

        self._history = defaultdict(dict)
        history_raw = state.get("history", {})
        if isinstance(history_raw, dict):
            for symbol, by_tf in history_raw.items():
                if not isinstance(by_tf, dict):
                    continue
                for timeframe, values in by_tf.items():
                    token = normalize_timeframe_token(str(timeframe))
                    history_deque = deque(
                        list(values or []),
                        maxlen=int(self._lookbacks.get(token, _DEFAULT_LOOKBACK)),
                    )
                    self._history[str(symbol)][token] = history_deque

        self._working = defaultdict(dict)
        working_raw = state.get("working", {})
        if isinstance(working_raw, dict):
            for symbol, by_tf in working_raw.items():
                if not isinstance(by_tf, dict):
                    continue
                for timeframe, values in by_tf.items():
                    if isinstance(values, dict):
                        token = normalize_timeframe_token(str(timeframe))
                        self._working[str(symbol)][token] = dict(values)

        self._last_seen_ms = {}
        last_seen = state.get("last_seen_ms", {})
        if isinstance(last_seen, dict):
            for symbol, value in last_seen.items():
                try:
                    self._last_seen_ms[str(symbol)] = int(value)
                except Exception:
                    continue
