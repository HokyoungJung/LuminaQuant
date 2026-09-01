from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.backtesting.liquidity_metrics import compute_liquidity_metrics
from lumina_quant.backtesting.timeframe_panel import build_timeframe_panel_from_frames


def _make_frame(start: datetime, base_price: float) -> pl.DataFrame:
    rows = []
    for idx in range(120):
        ts = start + timedelta(seconds=idx)
        open_px = base_price + idx * 0.1
        close_px = open_px + 0.05
        rows.append(
            {
                "datetime": ts,
                "open": open_px,
                "high": close_px + 0.1,
                "low": open_px - 0.1,
                "close": close_px,
                "volume": 100.0 + idx,
            }
        )
    return pl.DataFrame(rows)


def test_timeframe_panel_resampling_and_liquidity_metrics():
    start = datetime(2026, 1, 1, 0, 0, 0)
    frames = {
        "BTC/USDT": _make_frame(start, 100.0),
        "ETH/USDT": _make_frame(start, 200.0),
    }

    panels = build_timeframe_panel_from_frames(frames, ["1m"])
    panel = panels["1m"]

    assert not panel.is_empty()
    assert panel.select("asset").n_unique() == 2
    assert panel.filter(pl.col("asset") == "BTC/USDT").height == 2

    btc_first = panel.filter(pl.col("asset") == "BTC/USDT").sort("datetime").row(0, named=True)
    assert abs(btc_first["open"] - 100.0) < 1e-9
    assert abs(btc_first["close"] - (100.0 + 59 * 0.1 + 0.05)) < 1e-9

    liquidity = compute_liquidity_metrics(panel, rolling_window=2)
    assert {"adv", "adtv", "sigma"}.issubset(set(liquidity.columns))
    assert liquidity.filter(pl.col("asset") == "BTC/USDT")["adv"].to_list()[0] > 0.0
