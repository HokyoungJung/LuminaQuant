"""External live data handler for canonical MARKET_WINDOW / 1s OHLCV sources."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.core.market_window_contract import build_market_window_event
from lumina_quant.market_data import external_symbol_candidate_paths


class ExternalWindowDataHandler:
    """Read canonical external live inputs and emit MARKET_WINDOW events."""

    def __init__(self, events, symbol_list, config, exchange=None):
        self.events = events
        self.symbol_list = [str(symbol) for symbol in list(symbol_list or [])]
        self.config = config
        self.exchange = exchange
        self.continue_backtest = True
        self.lock = threading.Lock()
        self._shutdown = threading.Event()
        self._fatal_channel: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self._fatal_error: BaseException | None = None
        self.latest_symbol_data = {symbol: deque(maxlen=500) for symbol in self.symbol_list}
        self.col_idx = {
            "datetime": 0,
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
        }
        self._source_kind = (
            str(getattr(config, "EXTERNAL_DATA_SOURCE_KIND", "jsonl") or "jsonl").strip().lower()
        )
        self._path = Path(str(getattr(config, "EXTERNAL_DATA_PATH", "") or "")).expanduser()
        self._schema = (
            str(getattr(config, "EXTERNAL_DATA_SCHEMA", "market_window_v1") or "market_window_v1")
            .strip()
            .lower()
        )
        self._poll_seconds = max(1.0, float(getattr(config, "EXTERNAL_DATA_POLL_SECONDS", 2) or 2))
        self._allow_stale_ms = (
            max(1, int(getattr(config, "EXTERNAL_DATA_ALLOW_STALE_SECONDS", 45) or 45)) * 1000
        )
        self._symbol_map = dict(getattr(config, "EXTERNAL_DATA_SYMBOL_MAP", {}) or {})
        self._jsonl_offset = 0
        self._thread = threading.Thread(target=self._run_loop, daemon=False)
        self._thread.start()

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

    def consume_fatal_error(self) -> BaseException | None:
        try:
            return self._fatal_channel.get_nowait()
        except queue.Empty:
            return None

    def poll_fatal_error(self) -> BaseException | None:
        return self.consume_fatal_error()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self.continue_backtest = False
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, float(join_timeout)))

    def shutdown(self, join_timeout: float = 5.0) -> None:
        self.stop(join_timeout=join_timeout)

    def _run_loop(self) -> None:
        try:
            if self._source_kind == "parquet":
                self._run_parquet_loop()
            elif self._source_kind == "pipe":
                self._run_pipe_loop()
            else:
                self._run_jsonl_loop()
        except Exception as exc:  # pragma: no cover - defensive
            self._publish_fatal(exc)

    def _update_latest_rows(
        self, bars_1s: dict[str, tuple[tuple[int, float, float, float, float, float], ...]]
    ) -> None:
        with self.lock:
            for symbol, rows in bars_1s.items():
                key = str(symbol)
                if key not in self.latest_symbol_data:
                    self.latest_symbol_data[key] = deque(maxlen=500)
                self.latest_symbol_data[key].clear()
                self.latest_symbol_data[key].extend(rows)

    def _emit_payload(self, payload: dict[str, Any]) -> None:
        event_time = int(payload.get("time") or 0)
        window_seconds = int(payload.get("window_seconds") or 20)
        watermark = int(payload.get("event_time_watermark_ms") or event_time)
        lag_ms = int(payload.get("lag_ms") or max(0, int(time.time() * 1000) - int(watermark)))
        is_stale = bool(payload.get("is_stale", False) or lag_ms > self._allow_stale_ms)
        event = build_market_window_event(
            time=event_time,
            window_seconds=window_seconds,
            bars_1s=dict(payload.get("bars_1s") or {}),
            event_time_watermark_ms=watermark,
            commit_id=str(payload.get("commit_id") or "") or None,
            lag_ms=lag_ms,
            is_stale=is_stale,
            emit_metrics=False,
        )
        self._update_latest_rows(dict(event.bars_1s))
        self.events.put(event)

    def _run_jsonl_loop(self) -> None:
        while self.continue_backtest and not self._shutdown.is_set():
            if not self._path.exists():
                time.sleep(self._poll_seconds)
                continue
            with self._path.open("r", encoding="utf-8") as handle:
                handle.seek(self._jsonl_offset)
                for line in handle:
                    token = str(line).strip()
                    if not token:
                        continue
                    payload = json.loads(token)
                    if isinstance(payload, dict):
                        self._emit_payload(payload)
                self._jsonl_offset = handle.tell()
            time.sleep(self._poll_seconds)

    def _run_pipe_loop(self) -> None:
        while self.continue_backtest and not self._shutdown.is_set():
            if not self._path.exists():
                time.sleep(self._poll_seconds)
                continue
            with self._path.open("r", encoding="utf-8") as handle:
                while self.continue_backtest and not self._shutdown.is_set():
                    line = handle.readline()
                    if not line:
                        time.sleep(self._poll_seconds)
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        self._emit_payload(payload)

    def _resolve_parquet_path(self, symbol: str) -> Path:
        if self._path.is_file():
            if len(self.symbol_list) > 1:
                raise RuntimeError(
                    "Single-file external live parquet mode only supports one symbol. Use a directory root for multi-symbol external data."
                )
            return self._path
        for candidate in external_symbol_candidate_paths(
            self._path,
            symbol,
            symbol_map=self._symbol_map,
            include_csv=False,
            include_parquet=True,
        ):
            path = Path(candidate)
            if path.exists():
                return path
        raise FileNotFoundError(f"External parquet data not found for {symbol} under {self._path}")

    def _run_parquet_loop(self) -> None:
        while self.continue_backtest and not self._shutdown.is_set():
            payload: dict[str, Any] = {
                "time": int(time.time() * 1000),
                "window_seconds": int(getattr(self.config, "INGEST_WINDOW_SECONDS", 20) or 20),
                "bars_1s": {},
                "commit_id": "external-parquet",
                "lag_ms": 0,
                "is_stale": False,
            }
            for symbol in self.symbol_list:
                path = self._resolve_parquet_path(symbol)
                frame = (
                    pl.read_parquet(path)
                    .select(["datetime", "open", "high", "low", "close", "volume"])
                    .sort("datetime")
                    .tail(max(1, int(payload["window_seconds"])))
                )
                rows = tuple(
                    (
                        int(row[0].timestamp() * 1000),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    )
                    for row in frame.iter_rows(named=False)
                )
                payload["bars_1s"][symbol] = rows
                if rows:
                    payload["event_time_watermark_ms"] = int(rows[-1][0])
                    payload["time"] = int(rows[-1][0])
            self._emit_payload(payload)
            time.sleep(self._poll_seconds)


__all__ = ["ExternalWindowDataHandler"]
