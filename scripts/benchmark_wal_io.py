"""Benchmark custom binary WAL write and scan throughput."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from lumina_quant.storage.wal_binary import BinaryWAL


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark WAL append/scan performance")
    parser.add_argument("--records", type=int, default=1_000_000, help="Number of records to append")
    parser.add_argument("--batch", type=int, default=60, help="Records per append batch")
    parser.add_argument("--path", default="data/market_ohlcv_1s/bench/BTCUSDT/wal.bin", help="WAL file path")
    args = parser.parse_args()

    total_records = max(1, int(args.records))
    batch_size = max(1, int(args.batch))
    wal_path = Path(args.path)
    wal_path.parent.mkdir(parents=True, exist_ok=True)

    wal = BinaryWAL(wal_path, fsync_every_n_batches=1, auto_repair=True)
    wal.truncate()

    start_ts = 1_700_000_000_000
    write_start = perf_counter()
    produced = 0
    while produced < total_records:
        current_batch = min(batch_size, total_records - produced)
        rows = []
        for i in range(current_batch):
            idx = produced + i
            rows.append(
                {
                    "datetime": start_ts + idx * 1000,
                    "open": 100.0 + idx * 0.001,
                    "high": 100.2 + idx * 0.001,
                    "low": 99.8 + idx * 0.001,
                    "close": 100.1 + idx * 0.001,
                    "volume": 1.0 + (idx % 10),
                }
            )
        produced += wal.append(rows)
    write_elapsed = perf_counter() - write_start

    scan_start = perf_counter()
    scanned = sum(1 for _ in wal.iter_all())
    scan_elapsed = perf_counter() - scan_start

    file_size = wal.size_bytes()
    write_rate = produced / max(write_elapsed, 1e-9)
    scan_rate = scanned / max(scan_elapsed, 1e-9)

    print("=== WAL I/O Benchmark ===")
    print(f"path={wal_path}")
    print(f"records={produced} file_size={file_size} bytes")
    print(f"write_elapsed={write_elapsed:.4f}s write_rate={write_rate:,.0f} rec/s")
    print(f"scan_elapsed={scan_elapsed:.4f}s scan_rate={scan_rate:,.0f} rec/s")


if __name__ == "__main__":
    main()
