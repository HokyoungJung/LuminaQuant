"""Benchmark one-time dataset build and freezing throughput."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import polars as pl
from lumina_quant.optimization.frozen_dataset import build_frozen_dataset


def _synthetic_frame(rows: int, start: datetime) -> pl.DataFrame:
    timestamps = [start + timedelta(seconds=i) for i in range(rows)]
    base = [100.0 + (i * 0.01) for i in range(rows)]
    return pl.DataFrame(
        {
            "datetime": timestamps,
            "open": base,
            "high": [value + 0.1 for value in base],
            "low": [value - 0.1 for value in base],
            "close": base,
            "volume": [1000.0 for _ in range(rows)],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen dataset build stage")
    parser.add_argument("--symbols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=50_000)
    args = parser.parse_args()

    start = datetime(2024, 1, 1)
    data_dict = {
        f"SYM{i}/USDT": _synthetic_frame(int(args.rows), start)
        for i in range(max(1, int(args.symbols)))
    }

    t0 = time.perf_counter()
    frozen = build_frozen_dataset(data_dict)
    elapsed = max(1e-9, time.perf_counter() - t0)
    rows_total = max(1, int(frozen.close.shape[0]))

    print("benchmark_dataset_build")
    print(f"symbols={int(args.symbols)} rows_per_symbol={int(args.rows)}")
    print(f"elapsed_sec={elapsed:.6f}")
    print(f"rows_total={rows_total}")
    print(f"rows_per_sec={rows_total / elapsed:.2f}")


if __name__ == "__main__":
    main()
