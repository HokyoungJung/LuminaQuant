"""Committed materialized market-window reader for live runtime."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import polars as pl
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError, RawFirstStaleWindowError
from lumina_quant.core.market_window_contract import (
    MarketWindowContractError,
    build_market_window_event,
)
from lumina_quant.data.feature_points import FeaturePointLookup
from lumina_quant.storage.parquet import ParquetMarketDataRepository


@dataclass(slots=True)
class MaterializedSnapshot:
    """Latest committed market-window snapshot."""

    event_time_ms: int
    event_time_watermark_ms: int | None
    bars_1s: dict[str, tuple[tuple[int, float, float, float, float, float], ...]]
    commit_id: str | None
    lag_ms: int | None
    is_stale: bool


class MaterializedWindowReader:
    """Read committed 1s windows from manifest-gated parquet storage."""

    def __init__(
        self,
        *,
        root_path: str,
        exchange: str,
        symbol_list: list[str],
        window_seconds: int,
        staleness_threshold_seconds: int,
    ) -> None:
        self.repo = ParquetMarketDataRepository(root_path)
        self.exchange = str(exchange)
        self.symbol_list = list(symbol_list)
        self.window_seconds = max(1, int(window_seconds))
        self.staleness_threshold_seconds = max(1, int(staleness_threshold_seconds))

    @staticmethod
    def _frame_to_rows(
        frame: pl.DataFrame,
        *,
        window_seconds: int,
    ) -> tuple[tuple[int, float, float, float, float, float], ...]:
        if frame.is_empty():
            return tuple()
        selected = (
            frame.select(
                [
                    pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )
            .sort("timestamp_ms")
            .tail(max(1, int(window_seconds)))
        )
        rows: list[tuple[int, float, float, float, float, float]] = []
        for row in selected.iter_rows(named=False):
            rows.append(
                (
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                )
            )
        return tuple(rows)

    def read_snapshot(self) -> MaterializedSnapshot:
        now_ms = int(time.time() * 1000)
        bars_1s: dict[str, tuple[tuple[int, float, float, float, float, float], ...]] = {}
        watermark_ms: int | None = None
        commit_id: str | None = None
        lag_ms: int | None = None
        stale_error: RawFirstStaleWindowError | None = None

        for symbol in self.symbol_list:
            try:
                frame = self.repo.load_committed_ohlcv_chunked(
                    exchange=self.exchange,
                    symbol=str(symbol),
                    timeframe="1s",
                    start_date=None,
                    end_date=None,
                    chunk_days=1,
                    warmup_bars=0,
                    staleness_threshold_seconds=self.staleness_threshold_seconds,
                )
            except RawFirstStaleWindowError as exc:
                stale_error = exc
                frame = self.repo.load_committed_ohlcv_chunked(
                    exchange=self.exchange,
                    symbol=str(symbol),
                    timeframe="1s",
                    start_date=None,
                    end_date=None,
                    chunk_days=1,
                    warmup_bars=0,
                    staleness_threshold_seconds=None,
                )

            rows = self._frame_to_rows(frame, window_seconds=self.window_seconds)
            if not rows:
                raise RawFirstDataMissingError(
                    f"Committed 1s data missing for {self.exchange}:{symbol}."
                )
            bars_1s[str(symbol)] = rows
            symbol_watermark = int(rows[-1][0])
            if watermark_ms is None or symbol_watermark >= watermark_ms:
                watermark_ms = symbol_watermark

        if watermark_ms is not None:
            lag_ms = max(0, int(now_ms - int(watermark_ms)))

        is_stale = bool(stale_error)
        if stale_error is not None:
            commit_id = stale_error.commit_id
            if stale_error.lag_ms is not None:
                lag_ms = int(stale_error.lag_ms)

        return MaterializedSnapshot(
            event_time_ms=int(watermark_ms or now_ms),
            event_time_watermark_ms=watermark_ms,
            bars_1s=bars_1s,
            commit_id=commit_id,
            lag_ms=lag_ms,
            is_stale=is_stale,
        )


class CommittedWindowDataHandler:
    """Base live data handler that emits MARKET_WINDOW from committed materialized data."""

    def __init__(self, events, symbol_list, config, exchange=None):
        self.events = events
        self.symbol_list = list(symbol_list)
        self.config = config
        self.exchange = exchange
        self.continue_backtest = True
        self.latest_symbol_data = {s: deque(maxlen=500) for s in self.symbol_list}
        self.lock = threading.Lock()
        self.col_idx = {
            "datetime": 0,
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
        }

        self._poll_seconds = max(
            1.0,
            float(
                getattr(
                    self.config,
                    "LIVE_POLL_SECONDS",
                    getattr(self.config, "POLL_SECONDS", 20),
                )
                or 20
            ),
        )
        self._window_seconds = max(
            1,
            int(
                getattr(
                    self.config,
                    "INGEST_WINDOW_SECONDS",
                    getattr(self.config, "WINDOW_SECONDS", 20),
                )
                or 20
            ),
        )
        self._staleness_threshold_seconds = max(
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
        self._parity_v2_enabled = bool(
            getattr(self.config, "MARKET_WINDOW_PARITY_V2_ENABLED", False)
        )
        self._metrics_log_path = str(
            getattr(
                self.config,
                "MARKET_WINDOW_METRICS_LOG_PATH",
                "logs/live/market_window_metrics.ndjson",
            )
            or "logs/live/market_window_metrics.ndjson"
        )

        self._reader = MaterializedWindowReader(
            root_path=str(getattr(self.config, "MARKET_DATA_PARQUET_PATH", "data/market_parquet")),
            exchange=str(getattr(self.config, "MARKET_DATA_EXCHANGE", "binance")),
            symbol_list=self.symbol_list,
            window_seconds=self._window_seconds,
            staleness_threshold_seconds=self._staleness_threshold_seconds,
        )
        self._feature_lookup = FeaturePointLookup(
            db_path=str(getattr(self.config, "MARKET_DATA_PARQUET_PATH", "data/market_parquet")),
            exchange=str(getattr(self.config, "MARKET_DATA_EXCHANGE", "binance")),
        )

        self._fatal_channel: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self._fatal_error: BaseException | None = None
        self._shutdown = threading.Event()

        self.polling_thread = threading.Thread(target=self._poll_market_data, daemon=False)
        self.polling_thread.start()

    def _publish_fatal(self, exc: BaseException) -> None:
        if self._fatal_error is not None:
            return
        self._fatal_error = exc
        self.continue_backtest = False
        self._shutdown.set()
        try:
            self._fatal_channel.put_nowait(exc)
        except queue.Full:
            pass

    def poll_fatal_error(self) -> BaseException | None:
        try:
            return self._fatal_channel.get_nowait()
        except queue.Empty:
            return None

    def consume_fatal_error(self) -> BaseException | None:
        """Backward-compatible alias for live trader fatal polling."""
        return self.poll_fatal_error()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self.continue_backtest = False
        self._shutdown.set()
        thread = getattr(self, "polling_thread", None)
        if thread is not None and getattr(thread, "is_alive", lambda: False)():
            thread.join(timeout=max(0.1, float(join_timeout)))

    def shutdown(self, join_timeout: float = 5.0) -> None:
        """Backward-compatible alias for ordered shutdown path."""
        self.stop(join_timeout=join_timeout)

    def _poll_market_data(self) -> None:
        while self.continue_backtest and not self._shutdown.is_set():
            try:
                snapshot = self._reader.read_snapshot()
                with self.lock:
                    for symbol, rows in snapshot.bars_1s.items():
                        self.latest_symbol_data[str(symbol)].clear()
                        self.latest_symbol_data[str(symbol)].extend(rows)

                event = build_market_window_event(
                    time=int(snapshot.event_time_ms),
                    window_seconds=int(self._window_seconds),
                    bars_1s=snapshot.bars_1s,
                    event_time_watermark_ms=snapshot.event_time_watermark_ms,
                    commit_id=snapshot.commit_id,
                    lag_ms=snapshot.lag_ms,
                    is_stale=snapshot.is_stale,
                    parity_v2_enabled=self._parity_v2_enabled,
                    metrics_log_path=self._metrics_log_path,
                    emit_metrics=True,
                )
                self.events.put(event)
            except (RawFirstDataMissingError, MarketWindowContractError) as exc:
                self._publish_fatal(exc)
                return
            except Exception as exc:  # pragma: no cover - defensive unexpected fatal
                self._publish_fatal(exc)
                return

            if self.continue_backtest and not self._shutdown.is_set():
                time.sleep(self._poll_seconds)

    def update_bars(self) -> None:
        return

    def get_latest_bar(self, symbol):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            return data[-1] if data else None

    def get_latest_bars(self, symbol, N=1):
        with self.lock:
            return list(self.latest_symbol_data.get(symbol, deque()))[-N:]

    def get_latest_bar_datetime(self, symbol):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            if not data:
                return None
            return data[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            if data:
                idx = self.col_idx.get(val_type)
                if idx is not None:
                    return float(data[-1][idx])
                timestamp_ms = int(data[-1][0])
            else:
                timestamp_ms = None
        feature_value = self._feature_lookup.get_latest(
            str(symbol),
            str(val_type),
            timestamp_ms=timestamp_ms,
        )
        if feature_value is not None:
            return float(feature_value)
        return 0.0

    def get_latest_bars_values(self, symbol, val_type, N=1):
        with self.lock:
            data = list(self.latest_symbol_data.get(symbol, deque()))[-N:]
            idx = self.col_idx.get(val_type)
            if idx is None:
                return []
            return [float(row[idx]) for row in data]

    def get_latest_feature_value(self, symbol, field):
        with self.lock:
            data = self.latest_symbol_data.get(symbol) or deque()
            timestamp_ms = int(data[-1][0]) if data else None
        return self._feature_lookup.get_latest(
            str(symbol),
            str(field),
            timestamp_ms=timestamp_ms,
        )

    def get_market_spec(self, symbol):
        if self.exchange and hasattr(self.exchange, "get_market_spec"):
            return self.exchange.get_market_spec(symbol)
        return {}


__all__ = ["CommittedWindowDataHandler", "MaterializedSnapshot", "MaterializedWindowReader"]
