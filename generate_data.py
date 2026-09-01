import os
from datetime import datetime, timedelta

import numpy as np
import polars as pl


def generate_random_data(symbol, days=100, seed=None):
    """Generate a deterministic synthetic OHLCV CSV for ``symbol``.

    A ``seed`` makes the output reproducible — required so the CI perf/8GB smoke
    gate and any local benchmark exercise the SAME data every run (an unseeded
    random walk was flaky: some draws produced zero RSI signals and tripped the
    benchmark's signals>0 honesty assertion).
    """
    rng = np.random.default_rng(seed)
    # Start at the conventional backtest window start (config backtest.start_date
    # default = 2024-01-01) so the full generated series lies inside the benchmark
    # window — otherwise only the tail bars are in-window and may miss every RSI
    # threshold crossing, producing zero signals.
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # Generate random walk
    n = len(dates)
    returns = rng.standard_normal(n) * 0.02  # 2% daily vol
    price_path = 100 * np.cumprod(1 + returns)

    # Create OHLCV
    opens = price_path
    highs = opens * (1 + np.abs(rng.standard_normal(n) * 0.01))
    lows = opens * (1 - np.abs(rng.standard_normal(n) * 0.01))
    closes = lows + (highs - lows) * rng.random(n)
    volumes = rng.integers(100, 1000, size=n)

    # Create Polars DataFrame
    df = pl.DataFrame(
        {
            "datetime": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    # Ensure correct types
    df = df.with_columns([pl.col("datetime").cast(pl.Datetime)])

    if not os.path.exists("data"):
        os.makedirs("data")

    df.write_csv(f"data/{symbol}.csv")
    print(f"Generated data/{symbol}.csv")


if __name__ == "__main__":
    # Fixed per-symbol seeds → reproducible data across machines and CI runs.
    generate_random_data("BTCUSDT", 1000, seed=42)
    generate_random_data("ETHUSDT", 1000, seed=43)
