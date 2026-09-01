from __future__ import annotations

import numpy as np
import polars as pl
from lumina_quant.indicators.rare_event import (
    load_close_tail_from_lazy,
    rare_event_scores_from_frame,
    rare_event_scores_latest,
)


def test_rare_event_scores_are_bounded_to_unit_interval():
    closes = np.linspace(100.0, 140.0, 480)
    closes[-5:] = np.asarray([136.0, 129.0, 123.0, 116.0, 110.0], dtype=np.float64)

    scores = rare_event_scores_latest(
        closes,
        lookbacks=(1, 2, 3, 4, 5),
        return_factor=1.0,
        trend_rolling_window=20,
        local_extremum_window=120,
        max_points=512,
    )
    assert scores is not None

    values = [
        scores.rare_return_score,
        scores.rare_streak_score,
        scores.trend_break_score,
        scores.local_extremum_score,
        scores.composite_score,
    ]
    assert all(0.0 <= float(v) <= 1.0 for v in values)


def test_lazy_close_tail_loader_keeps_projection_small():
    rows = 1200
    frame = pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                start=pl.datetime(2026, 1, 1),
                end=pl.datetime(2026, 1, 1, 0, 19, 59),
                interval="1s",
                eager=True,
            )[:rows],
            "open": np.linspace(100.0, 120.0, rows),
            "high": np.linspace(100.2, 120.2, rows),
            "low": np.linspace(99.8, 119.8, rows),
            "close": np.linspace(100.0, 130.0, rows),
            "volume": np.linspace(1000.0, 1500.0, rows),
            "noise_col": np.arange(rows, dtype=np.int64),
        }
    )

    tail = load_close_tail_from_lazy(frame.lazy(), close_column="close", max_points=256, mode="cpu")
    assert tail.shape[0] == 256
    expected = frame.get_column("close").to_numpy()[-256:]
    np.testing.assert_allclose(tail, expected)

    scores = rare_event_scores_from_frame(
        frame.lazy(),
        close_column="close",
        max_points=256,
        mode="cpu",
        kwargs={"local_extremum_window": 80},
    )
    assert scores is not None
    assert 0.0 <= scores.composite_score <= 1.0
