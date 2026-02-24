"""Benchmark CPU vs GPU on weekly-chunked Polars bucket resampling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter

import polars as pl
from lumina_quant.compute_engine import GPUNotAvailableError, collect, select_engine


@dataclass(slots=True)
class BenchResult:
    engine: str
    seconds: float
    rows: int


def _build_synthetic_week(start: datetime, *, seconds: int = 604_800) -> pl.DataFrame:
    end = start + timedelta(seconds=max(seconds - 1, 0))
    return pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                start=start,
                end=end,
                interval="1s",
                eager=True,
            ),
            "open": [100.0 + (idx % 300) * 0.01 for idx in range(seconds)],
            "high": [100.1 + (idx % 300) * 0.01 for idx in range(seconds)],
            "low": [99.9 + (idx % 300) * 0.01 for idx in range(seconds)],
            "close": [100.0 + (idx % 300) * 0.01 for idx in range(seconds)],
            "volume": [1.0 + (idx % 10) for idx in range(seconds)],
        }
    )


def _resample_bucket_weekly(frame: pl.DataFrame, *, bucket_ms: int) -> pl.LazyFrame:
    lazy_frame = frame.lazy().with_columns(
        [
            pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"),
            pl.col("datetime").dt.truncate("1w").alias("week_start"),
        ]
    )

    return (
        lazy_frame.with_columns(((pl.col("timestamp_ms") // bucket_ms) * bucket_ms).alias("bucket_ms"))
        .group_by(["week_start", "bucket_ms"])
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .sort(["week_start", "bucket_ms"])
        .with_columns(pl.from_epoch("bucket_ms", time_unit="ms").alias("datetime"))
        .select(["datetime", "open", "high", "low", "close", "volume"])
    )


def _run_once(frame: pl.DataFrame, *, mode: str, device: str | None, bucket_ms: int) -> BenchResult:
    lazy_frame = _resample_bucket_weekly(frame, bucket_ms=bucket_ms)
    engine = select_engine(mode=mode, device=device)
    t0 = perf_counter()
    out = collect(lazy_frame, engine=engine)
    elapsed = perf_counter() - t0
    return BenchResult(engine=engine.resolved_engine, seconds=elapsed, rows=int(out.height))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU weekly chunked resampling.")
    parser.add_argument("--bucket-ms", type=int, default=60_000, help="Resample bucket width in ms.")
    parser.add_argument(
        "--seconds",
        type=int,
        default=604_800,
        help="Synthetic 1s row-count to benchmark (default: one week).",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-mode", default="auto", help="GPU mode override for GPU benchmark.")
    parser.add_argument("--gpu-device", default="", help="GPU device override (e.g. 0, cuda:0).")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> int:
    args = parse_args()
    frame = _build_synthetic_week(datetime(2025, 1, 6, tzinfo=UTC), seconds=int(args.seconds))
    bucket_ms = int(args.bucket_ms)
    gpu_device = str(args.gpu_device or "").strip() or None

    cpu_times: list[float] = []
    gpu_times: list[float] = []
    rows = 0

    for _ in range(int(args.runs)):
        cpu_result = _run_once(frame, mode="cpu", device=gpu_device, bucket_ms=bucket_ms)
        cpu_times.append(cpu_result.seconds)
        rows = cpu_result.rows

    for _ in range(int(args.runs)):
        try:
            gpu_result = _run_once(
                frame,
                mode=str(args.gpu_mode),
                device=gpu_device,
                bucket_ms=bucket_ms,
            )
        except GPUNotAvailableError as exc:
            print(f"GPU benchmark skipped: {exc}")
            break
        gpu_times.append(gpu_result.seconds)

    print(f"rows={rows}")
    print(f"cpu_mean_s={_mean(cpu_times):.6f}")
    if gpu_times:
        print(f"gpu_mean_s={_mean(gpu_times):.6f}")
        speedup = _mean(cpu_times) / max(_mean(gpu_times), 1e-9)
        print(f"speedup={speedup:.3f}x")
    else:
        print("gpu_mean_s=NA")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
