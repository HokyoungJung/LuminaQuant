"""Run a no-infra minimum viable backtest for first-time users."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _write_synthetic_ohlcv_csv(path: Path, *, days: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    returns = rng.normal(loc=0.0, scale=0.015, size=days)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(loc=0.0, scale=0.002, size=days))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(loc=0.0, scale=0.003, size=days)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(loc=0.0, scale=0.003, size=days)))
    volume = rng.integers(low=100, high=5000, size=days)

    frame = pl.DataFrame(
        {
            "datetime": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime))

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


def ensure_sample_data(*, data_dir: Path, days: int) -> None:
    targets = (
        ("BTCUSDT.csv", 42),
        ("ETHUSDT.csv", 84),
    )
    for filename, seed in targets:
        path = data_dir / filename
        if path.exists():
            continue
        _write_synthetic_ohlcv_csv(path, days=days, seed=seed)
        print(f"[INFO] Generated synthetic sample data: {path}")


def build_demo_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.update(
        {
            "LQ__TRADING__SYMBOLS": '["BTC/USDT","ETH/USDT"]',
            "LQ__STORAGE__BACKEND": "local",
            "LQ__BACKTEST__START_DATE": "2022-01-01",
            "LQ__BACKTEST__END_DATE": "2022-04-30",
            "LQ_AUTO_COLLECT_DB": "0",
            "LQ_BACKTEST_LOW_MEMORY": "1",
            "LQ_BACKTEST_PERSIST_OUTPUT": "0",
        }
    )
    return env


def run_minimum_viable_backtest(*, days: int) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_sample_data(data_dir=project_root / "data", days=max(30, int(days)))
    env = build_demo_env()
    cmd = [
        sys.executable,
        "-m",
        "lumina_quant.cli.main",
        "backtest",
        "--data-source",
        "csv",
        "--no-auto-collect-db",
        "--run-id",
        "minimum-viable-run",
    ]
    print(f"[INFO] Running minimum viable backtest: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(project_root), env=env, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a no-infra minimum viable LuminaQuant backtest."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=120,
        help="Synthetic dataset length in days (minimum 30).",
    )
    args = parser.parse_args()
    return run_minimum_viable_backtest(days=int(args.days))


if __name__ == "__main__":
    raise SystemExit(main())
