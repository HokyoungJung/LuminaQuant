from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.optimization.frozen_dataset import build_frozen_dataset


def test_build_frozen_dataset_basic_shapes():
    start = datetime(2024, 1, 1)
    frame_a = pl.DataFrame(
        {
            "datetime": [start + timedelta(minutes=i) for i in range(5)],
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 10, 10, 10, 10],
        }
    )
    frame_b = pl.DataFrame(
        {
            "datetime": [start + timedelta(minutes=i) for i in range(5)],
            "open": [2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6],
            "low": [2, 3, 4, 5, 6],
            "close": [2, 3, 4, 5, 6],
            "volume": [20, 20, 20, 20, 20],
        }
    )

    frozen = build_frozen_dataset({"BTC/USDT": frame_a, "ETH/USDT": frame_b})
    assert frozen.close.shape[0] == 10
    assert frozen.timestamp_ns.shape[0] == 10
    assert "BTC/USDT" in frozen.symbol_index
    assert "ETH/USDT" in frozen.symbol_index
    assert frozen.split_ranges["all"] == (0, 10)
