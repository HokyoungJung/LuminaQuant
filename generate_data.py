import os
from datetime import datetime, timedelta

import numpy as np
import polars as pl


def generate_random_data(symbol, days=100):
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # Generate random walk
    n = len(dates)
    returns = np.random.randn(n) * 0.02  # 2% daily vol
    price_path = 100 * np.cumprod(1 + returns)

    # Create OHLCV
    opens = price_path
    highs = opens * (1 + np.abs(np.random.randn(n) * 0.01))
    lows = opens * (1 - np.abs(np.random.randn(n) * 0.01))
    closes = lows + (highs - lows) * np.random.rand(n)
    volumes = np.random.randint(100, 1000, size=n)

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
    generate_random_data("BTCUSDT", 1000)
    generate_random_data("ETHUSDT", 1000)
