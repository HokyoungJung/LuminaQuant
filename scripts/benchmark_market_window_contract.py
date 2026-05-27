#!/usr/bin/env python3
"""Benchmark live MARKET_WINDOW construction and rolling aggregation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.core.market_window_contract import (
    build_market_window_event,
    market_window_event_payload,
)
from lumina_quant.live.market_window_rolling import NormalizedTradeTick, RollingWindowAggregator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "var"
    / "reports"
    / "native_acceleration_20260527"
    / "market_window_contract_benchmark_latest.json"
)


@dataclass(frozen=True)
class EventBuilderTiming:
    generic_seconds_per_eval: float
    trusted_seconds_per_eval: float
    trusted_speedup: float
    payloads_match: bool


@dataclass(frozen=True)
class AggregatorTiming:
    ticks: int
    symbols: int
    window_seconds: int
    elapsed_seconds: float
    ticks_per_second: float
    emitted_events: int


@dataclass(frozen=True)
class BenchmarkResult:
    generated_at_utc: str
    status: str
    builder: EventBuilderTiming
    aggregator: AggregatorTiming
    memory_note: str


def generate_normalized_bars(
    *,
    symbols: int,
    window_seconds: int,
    base_ms: int = 1_700_000_000_000,
) -> dict[str, tuple[tuple[int, float, float, float, float, float], ...]]:
    symbol_count = max(1, int(symbols))
    window = max(1, int(window_seconds))
    out: dict[str, tuple[tuple[int, float, float, float, float, float], ...]] = {}
    for symbol_idx in range(symbol_count):
        rows: list[tuple[int, float, float, float, float, float]] = []
        price_base = 100.0 + float(symbol_idx)
        for offset in range(window):
            ts_ms = int(base_ms + offset * 1000)
            close = price_base + float(offset) * 0.01
            rows.append(
                (
                    ts_ms,
                    close - 0.005,
                    close + 0.010,
                    close - 0.010,
                    close,
                    1.0 + float(offset % 7),
                )
            )
        out[f"S{symbol_idx}/USDT"] = tuple(rows)
    return out


def _time_event_builder(
    *,
    bars_1s: dict[str, tuple[tuple[int, float, float, float, float, float], ...]],
    evals: int,
) -> EventBuilderTiming:
    eval_count = max(1, int(evals))
    event_time = max(row[-1][0] for row in bars_1s.values() if row)
    window_seconds = max(len(rows) for rows in bars_1s.values())

    generic_event = build_market_window_event(
        time=event_time,
        window_seconds=window_seconds,
        bars_1s=bars_1s,
        event_time_watermark_ms=event_time,
        lag_ms=0,
        is_stale=False,
    )
    trusted_event = build_market_window_event(
        time=event_time,
        window_seconds=window_seconds,
        bars_1s=bars_1s,
        event_time_watermark_ms=event_time,
        lag_ms=0,
        is_stale=False,
        bars_1s_already_normalized=True,
    )
    payloads_match = market_window_event_payload(generic_event) == market_window_event_payload(
        trusted_event
    )

    started = time.perf_counter()
    for _ in range(eval_count):
        build_market_window_event(
            time=event_time,
            window_seconds=window_seconds,
            bars_1s=bars_1s,
            event_time_watermark_ms=event_time,
            lag_ms=0,
            is_stale=False,
        )
    generic_seconds = (time.perf_counter() - started) / float(eval_count)

    started = time.perf_counter()
    for _ in range(eval_count):
        build_market_window_event(
            time=event_time,
            window_seconds=window_seconds,
            bars_1s=bars_1s,
            event_time_watermark_ms=event_time,
            lag_ms=0,
            is_stale=False,
            bars_1s_already_normalized=True,
        )
    trusted_seconds = (time.perf_counter() - started) / float(eval_count)

    return EventBuilderTiming(
        generic_seconds_per_eval=float(generic_seconds),
        trusted_seconds_per_eval=float(trusted_seconds),
        trusted_speedup=float(generic_seconds / trusted_seconds) if trusted_seconds > 0 else 0.0,
        payloads_match=bool(payloads_match),
    )


def _time_aggregator(*, ticks: int, symbols: int, window_seconds: int) -> AggregatorTiming:
    tick_count = max(1, int(ticks))
    symbol_count = max(1, int(symbols))
    window = max(1, int(window_seconds))
    symbol_list = [f"S{idx}/USDT" for idx in range(symbol_count)]
    aggregator = RollingWindowAggregator(
        symbol_list=symbol_list,
        window_seconds=window,
        max_lateness_ms=0,
    )
    base_ms = 1_700_000_000_000
    emitted_events = 0
    started = time.perf_counter()
    for idx in range(tick_count):
        symbol = symbol_list[idx % symbol_count]
        exchange_ts_ms = int(base_ms + (idx // symbol_count) * 1000)
        emitted_events += len(
            aggregator.ingest(
                NormalizedTradeTick(
                    symbol=symbol,
                    exchange_ts_ms=exchange_ts_ms,
                    price=100.0 + float(idx % 100) * 0.01,
                    quantity=1.0,
                    event_id=f"{symbol}:{idx}",
                    receive_ts_ms=exchange_ts_ms + 1,
                )
            )
        )
    elapsed = time.perf_counter() - started
    return AggregatorTiming(
        ticks=tick_count,
        symbols=symbol_count,
        window_seconds=window,
        elapsed_seconds=float(elapsed),
        ticks_per_second=float(tick_count / elapsed) if elapsed > 0 else 0.0,
        emitted_events=int(emitted_events),
    )


def run_benchmark(
    *,
    symbols: int,
    window_seconds: int,
    ticks: int,
    evals: int,
) -> BenchmarkResult:
    bars_1s = generate_normalized_bars(symbols=symbols, window_seconds=window_seconds)
    builder = _time_event_builder(bars_1s=bars_1s, evals=evals)
    aggregator = _time_aggregator(
        ticks=ticks,
        symbols=symbols,
        window_seconds=window_seconds,
    )
    return BenchmarkResult(
        generated_at_utc=datetime.now(UTC).isoformat(),
        status="pass" if builder.payloads_match else "payload_mismatch",
        builder=builder,
        aggregator=aggregator,
        memory_note="Synthetic MARKET_WINDOW benchmark; inputs bounded for <8GB sessions.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=5)
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--ticks", type=int, default=5_000)
    parser.add_argument("--evals", type=int, default=500)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_benchmark(
        symbols=args.symbols,
        window_seconds=args.window_seconds,
        ticks=args.ticks,
        evals=args.evals,
    )
    payload = asdict(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
