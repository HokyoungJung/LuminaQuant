from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/research/screen_profit_moonshot_external_alpha.py"
)
spec = importlib.util.spec_from_file_location("screen_profit_moonshot_external_alpha", MODULE_PATH)
assert spec is not None and spec.loader is not None
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)


def _candidate(
    train_edge, val_edge, oos_edge, *, train_count=100, val_count=30, oos_count=50, oos_hit=0.52
):
    return {
        "splits": [
            {
                "split": "train",
                "count": train_count,
                "mean_after_cost": train_edge,
                "hit_rate": 0.53,
            },
            {
                "split": "val",
                "count": val_count,
                "mean_after_cost": val_edge,
                "hit_rate": 0.51,
            },
            {
                "split": "oos",
                "count": oos_count,
                "mean_after_cost": oos_edge,
                "hit_rate": oos_hit,
            },
        ]
    }


def test_metrics_apply_roundtrip_cost_and_hit_rate():
    metrics = screen._metrics([0.01, -0.002, 0.004], roundtrip_cost=0.0018)

    assert metrics["count"] == 3
    assert metrics["hit_rate"] == 2 / 3
    assert metrics["mean_after_cost"] == ((0.01 - 0.002 + 0.004) / 3) - 0.0018
    assert metrics["t_stat"] > 0


def test_funding_screen_rejects_train_val_positive_when_oos_loses_after_cost():
    candidate = _candidate(0.0010, 0.0020, -0.0030, train_count=50, val_count=10, oos_count=8)

    assert screen._funding_candidate_rejected_reason(candidate) == "oos_post_cost_edge_non_positive"


def test_flow_imbalance_is_null_when_no_real_taker_flow():
    frame = pl.DataFrame({"buy_sum": [0.0, 10.0], "sell_sum": [0.0, 30.0]}).with_columns(
        screen._flow_imbalance_expr()
    )

    assert frame["flow"].to_list() == [None, -0.5]


def test_stale_bounded_forward_fill_drops_values_after_staleness_limit():
    stale_ms = screen.FEATURE_POINT_MAX_STALE_MS + 1
    frame = screen._stale_bounded_forward_fill(
        pl.DataFrame(
            {
                "timestamp_ms": [0, stale_ms],
                "mark_price": [100.0, None],
                "funding_rate": [0.001, None],
            }
        ),
        ("mark_price", "funding_rate"),
    )

    assert frame["mark_price"].to_list() == [100.0, None]
    assert frame["funding_rate"].to_list() == [0.001, None]


def test_leadlag_screen_requires_oos_sample_and_positive_hit_rate():
    too_sparse = _candidate(0.0005, 0.0007, 0.0006, oos_count=12, oos_hit=0.70)
    weak_hit = _candidate(0.0005, 0.0007, 0.0006, oos_count=50, oos_hit=0.49)
    survivor = _candidate(0.0005, 0.0007, 0.0006, oos_count=50, oos_hit=0.54)

    assert (
        screen._leadlag_rejected_reason(too_sparse)
        == "insufficient_split_events_for_live_equivalent_followup"
    )
    assert screen._leadlag_rejected_reason(weak_hit) == "oos_hit_rate_below_half"
    assert screen._leadlag_rejected_reason(survivor) is None
    assert screen._leadlag_candidate_score(survivor) > 0
