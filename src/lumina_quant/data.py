"""Local sample-data loader.

The public repository intentionally has no network data collection code. All
examples load deterministic CSV files from disk.
"""

from __future__ import annotations

import csv
from pathlib import Path

from lumina_quant.models import Bar

_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def load_ohlcv_csv(path: str | Path) -> list[Bar]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column for column in _REQUIRED_COLUMNS if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        bars = [
            Bar(
                timestamp=str(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in reader
        ]
    if not bars:
        raise ValueError(f"no bars loaded from {csv_path}")
    return bars
