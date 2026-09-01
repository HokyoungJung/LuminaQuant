"""In-memory rolling 1s window aggregation for live Binance trade streams."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.core.market_window_contract import build_market_window_event


@dataclass(slots=True)
class NormalizedTradeTick:
    """Normalized market trade tick used by live rolling aggregation."""

    symbol: str
    exchange_ts_ms: int
    price: float
    quantity: float
    event_id: str
    receive_ts_ms: int


class RollingWindowAggregator:
    """Aggregate trade ticks into cadence-aligned MARKET_WINDOW events."""

    def __init__(
        self,
        *,
        symbol_list: list[str],
        window_seconds: int,
        max_lateness_ms: int = 1500,
        max_history_seconds: int | None = None,
        dedupe_ttl_ms: int = 600_000,
        dedupe_max_keys: int = 100_000,
        stale_symbol_after_ms: int = 0,
        bar_sanity_check: bool = False,
    ) -> None:
        self.symbol_list = [str(symbol) for symbol in list(symbol_list or [])]
        self.window_seconds = max(1, int(window_seconds))
        self.max_lateness_ms = max(0, int(max_lateness_ms))
        history_default = int(self.window_seconds) + 120
        self.max_history_seconds = max(history_default, int(max_history_seconds or history_default))
        self.dedupe_ttl_ms = max(10_000, int(dedupe_ttl_ms))
        self.dedupe_max_keys = max(1000, int(dedupe_max_keys))
        # 0 disables per-symbol staleness detection (default -> legacy behavior).
        self.stale_symbol_after_ms = max(0, int(stale_symbol_after_ms or 0))
        # Default OFF -> emitted events are byte-identical. When enabled, a bar
        # failing OHLCV sanity raises at the seam instead of reaching strategies.
        self.bar_sanity_check = bool(bar_sanity_check)

        self._bars: dict[str, dict[int, list[float]]] = {symbol: {} for symbol in self.symbol_list}
        self._dedupe: OrderedDict[str, int] = OrderedDict()
        self._max_seen_ts_ms: int | None = None
        self._last_emitted_sec_ms: int | None = None
        # Exchange ts of the last *real* trade seen per symbol (flat-fill rows do
        # not update this), so a single dead/suspended symbol whose window keeps
        # getting synthesized at the last close can be detected even while the
        # global watermark advances off surviving symbols.
        self._last_real_trade_ms: dict[str, int] = {}

    def ingest(self, tick: NormalizedTradeTick) -> list[MarketWindowEvent]:
        """Ingest one normalized trade and emit zero or more MARKET_WINDOW events."""
        symbol = str(tick.symbol)
        if symbol not in self._bars:
            self._bars[symbol] = {}
            self.symbol_list.append(symbol)

        ts_ms = int(tick.exchange_ts_ms)
        if ts_ms <= 0:
            return []
        if float(tick.price) <= 0.0:
            return []
        qty = max(0.0, float(tick.quantity))

        # A real trade (even a duplicate) proves the symbol is still alive.
        prev_real = self._last_real_trade_ms.get(symbol)
        if prev_real is None or ts_ms > int(prev_real):
            self._last_real_trade_ms[symbol] = int(ts_ms)

        if self._is_duplicate(tick.event_id, ts_ms):
            return self.flush_until(now_ms=int(tick.receive_ts_ms or int(time.time() * 1000)))

        if self._max_seen_ts_ms is None or ts_ms > self._max_seen_ts_ms:
            self._max_seen_ts_ms = int(ts_ms)

        bucket_sec_ms = (ts_ms // 1000) * 1000
        symbol_bars = self._bars[symbol]
        bar = symbol_bars.get(bucket_sec_ms)
        if bar is None:
            bar = [
                float(tick.price),
                float(tick.price),
                float(tick.price),
                float(tick.price),
                float(qty),
            ]
            symbol_bars[bucket_sec_ms] = bar
        else:
            bar[1] = max(float(bar[1]), float(tick.price))
            bar[2] = min(float(bar[2]), float(tick.price))
            bar[3] = float(tick.price)
            bar[4] = float(bar[4]) + float(qty)

        self._prune_history(anchor_sec_ms=bucket_sec_ms, symbol=symbol)
        return self.flush_until(now_ms=int(tick.receive_ts_ms or int(time.time() * 1000)))

    def flush_until(self, *, now_ms: int | None = None) -> list[MarketWindowEvent]:
        """Flush all cadence windows currently closed by the lateness watermark."""
        if self._max_seen_ts_ms is None:
            return []

        watermark_ms = int(self._max_seen_ts_ms) - int(self.max_lateness_ms)
        watermark_sec_ms = (int(watermark_ms) // 1000) * 1000
        if self._last_emitted_sec_ms is None:
            earliest = self._earliest_bar_second()
            self._last_emitted_sec_ms = int(
                (earliest if earliest is not None else watermark_sec_ms) - 1000
            )

        now_value_ms = int(now_ms if now_ms is not None else int(time.time() * 1000))
        events: list[MarketWindowEvent] = []
        while int(self._last_emitted_sec_ms + 1000) <= int(watermark_sec_ms):
            emit_sec_ms = int(self._last_emitted_sec_ms + 1000)
            bars_1s = self._build_window_snapshot(emit_sec_ms)
            lag_ms = max(0, int(now_value_ms) - int(emit_sec_ms))
            event = build_market_window_event(
                time=int(emit_sec_ms),
                window_seconds=int(self.window_seconds),
                bars_1s=bars_1s,
                event_time_watermark_ms=int(emit_sec_ms),
                commit_id=None,
                lag_ms=int(lag_ms),
                is_stale=False,
                emit_metrics=False,
                bars_1s_already_normalized=True,
                sanity_check=self.bar_sanity_check,
            )
            events.append(event)
            self._last_emitted_sec_ms = int(emit_sec_ms)
        return events

    def _build_window_snapshot(
        self,
        emit_sec_ms: int,
    ) -> dict[str, tuple[tuple[int, float, float, float, float, float], ...]]:
        out: dict[str, tuple[tuple[int, float, float, float, float, float], ...]] = {}
        start_sec_ms = int(emit_sec_ms) - (int(self.window_seconds) - 1) * 1000
        for symbol in self.symbol_list:
            symbol_bars = self._bars.get(symbol, {})
            rows: list[tuple[int, float, float, float, float, float]] = []
            prev_close = self._last_close_before(symbol, start_sec_ms)
            sec = int(start_sec_ms)
            while sec <= int(emit_sec_ms):
                bar = symbol_bars.get(sec)
                if bar is not None:
                    row = (
                        int(sec),
                        float(bar[0]),
                        float(bar[1]),
                        float(bar[2]),
                        float(bar[3]),
                        float(bar[4]),
                    )
                    prev_close = float(row[4])
                elif prev_close is not None:
                    row = (
                        int(sec),
                        float(prev_close),
                        float(prev_close),
                        float(prev_close),
                        float(prev_close),
                        0.0,
                    )
                else:
                    sec += 1000
                    continue
                rows.append(row)
                sec += 1000
            out[str(symbol)] = tuple(rows)
        return out

    def _last_close_before(self, symbol: str, sec_ms: int) -> float | None:
        symbol_bars = self._bars.get(symbol, {})
        if not symbol_bars:
            return None
        nearest = max(
            (int(timestamp) for timestamp in symbol_bars if int(timestamp) < int(sec_ms)),
            default=None,
        )
        if nearest is None:
            return None
        bar = symbol_bars.get(nearest)
        if not bar:
            return None
        return float(bar[3])

    def _earliest_bar_second(self) -> int | None:
        earliest: int | None = None
        for bars in self._bars.values():
            if not bars:
                continue
            candidate = min(int(ts) for ts in bars)
            if earliest is None or int(candidate) < int(earliest):
                earliest = int(candidate)
        return earliest

    def _prune_history(self, *, anchor_sec_ms: int, symbol: str | None = None) -> None:
        lower_bound = int(anchor_sec_ms) - (int(self.max_history_seconds) * 1000)
        targets = (
            [(str(symbol), self._bars.get(str(symbol), {}))]
            if symbol is not None
            else list(self._bars.items())
        )
        for symbol_key, bars in targets:
            stale_seconds = [timestamp for timestamp in bars if int(timestamp) < int(lower_bound)]
            for stale in stale_seconds:
                bars.pop(stale, None)
            self._bars[symbol_key] = bars

    def last_real_trade_ms(self, symbol: str) -> int | None:
        """Exchange ts of the last real trade seen for ``symbol`` (or None)."""
        value = self._last_real_trade_ms.get(str(symbol))
        return int(value) if value is not None else None

    def symbol_trade_age_ms(self, symbol: str, *, reference_ms: int | None = None) -> int | None:
        """Age (ms) of the last real trade for ``symbol`` vs the event watermark.

        Returns ``None`` when the symbol has never traded or no watermark exists.
        The reference defaults to the global max-seen exchange ts so per-symbol
        liveness is measured on exchange time, immune to local clock skew.
        """
        if reference_ms is not None:
            reference = int(reference_ms)
        elif self._max_seen_ts_ms is not None:
            reference = int(self._max_seen_ts_ms)
        else:
            return None
        last = self._last_real_trade_ms.get(str(symbol))
        if last is None:
            return None
        return max(0, int(reference) - int(last))

    def stale_symbols(
        self, *, reference_ms: int | None = None, threshold_ms: int | None = None
    ) -> list[str]:
        """Return symbols whose last real trade is older than the stale threshold.

        A symbol that has never delivered a tick counts as stale once a reference
        watermark exists (covers a typo'd/never-subscribed symbol). Returns an
        empty list when the threshold is disabled (``<= 0``), preserving legacy
        behavior at defaults.
        """
        threshold = (
            int(threshold_ms) if threshold_ms is not None else int(self.stale_symbol_after_ms)
        )
        if threshold <= 0:
            return []
        if reference_ms is not None:
            reference = int(reference_ms)
        elif self._max_seen_ts_ms is not None:
            reference = int(self._max_seen_ts_ms)
        else:
            return []
        stale: list[str] = []
        for symbol in self.symbol_list:
            last = self._last_real_trade_ms.get(str(symbol))
            if last is None or (int(reference) - int(last)) > threshold:
                stale.append(str(symbol))
        return stale

    def _is_duplicate(self, event_id: str, ts_ms: int) -> bool:
        token = str(event_id or "")
        if not token:
            return False
        if token in self._dedupe:
            return True
        self._dedupe[token] = int(ts_ms)
        cutoff = int(ts_ms) - int(self.dedupe_ttl_ms)
        while self._dedupe:
            first_key = next(iter(self._dedupe))
            first_ts = int(self._dedupe[first_key])
            if len(self._dedupe) > int(self.dedupe_max_keys) or first_ts < cutoff:
                self._dedupe.popitem(last=False)
                continue
            break
        return False


__all__ = ["NormalizedTradeTick", "RollingWindowAggregator"]
