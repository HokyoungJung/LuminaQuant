from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from lumina_quant.strategies.plugins.trend_momentum import TrendMomentumPlugin
from lumina_quant.strategies.plugins.xs_mean_reversion import CrossSectionalMeanReversionPlugin


def _panel(timeframe: str, bars: int = 40) -> pl.DataFrame:
    rows = []
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    start = datetime(2026, 1, 1)
    for idx in range(bars):
        for a_idx, asset in enumerate(assets):
            px = 100.0 + a_idx * 5 + idx * (0.1 + 0.02 * a_idx)
            rows.append(
                {
                    "datetime": start + timedelta(minutes=idx),
                    "asset": asset,
                    "timeframe": timeframe,
                    "open": px,
                    "high": px + 0.4,
                    "low": px - 0.4,
                    "close": px + 0.1,
                    "volume": 1000 + 10 * a_idx,
                }
            )
    return pl.DataFrame(rows)


def _run_plugin(plugin, frame: pl.DataFrame):
    feat = plugin.compute_features(frame, {"lookback_bars": 5, "top_n": 1, "bottom_n": 1})
    sig = plugin.compute_signal(feat, {})
    tgt = plugin.signal_to_targets(sig, {"top_n": 1, "bottom_n": 1, "allow_short": True})
    assert not tgt.is_empty()
    assert {"datetime", "asset", "target_weight"}.issubset(set(tgt.columns))
    return tgt


def test_plugins_run_across_timeframes_without_code_changes():
    for plugin in (TrendMomentumPlugin(), CrossSectionalMeanReversionPlugin()):
        t1 = _run_plugin(plugin, _panel("1m"))
        t2 = _run_plugin(plugin, _panel("1h"))
        assert t1.select("asset").n_unique() == t2.select("asset").n_unique()
