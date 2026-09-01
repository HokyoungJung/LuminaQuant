from __future__ import annotations

import polars as pl
import pytest

from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.strategies import get_live_strategy_names, get_strategy_map, get_strategy_tier
from lumina_quant.strategies.dacapogo_daily_source import (
    COST,
    DacapogoDailySourceStrategy,
    backtest_daily,
    daily_candidates,
)


def _frame(rows: list[tuple[str, str, float, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["market", "date", "value", "open", "high", "close"],
        orient="row",
    )


def test_selection_is_causal_and_uses_explicit_previous_day_value():
    rows = []
    for index in range(11):
        market = f"M{index:02d}"
        rows.append((market, "2026-01-01", 11.0 - index, 100.0, 100.0, 100.0))
        rows.append((market, "2026-01-02", 1_000.0 if index == 10 else 1.0, 100.0, 104.0, 104.0))
    selected = daily_candidates(_frame(rows))

    assert set(selected["market"]) == {f"M{index:02d}" for index in range(10)}
    assert selected["date"].unique().to_list() == ["2026-01-02"]
    with pytest.raises(ValueError, match="value"):
        daily_candidates(_frame(rows).drop("value"))


def test_trigger_is_inclusive_at_four_percent():
    data = _frame(
        [
            ("BTC", "2026-01-01", 1.0, 100.0, 100.0, 100.0),
            ("BTC", "2026-01-02", 1.0, 100.0, 104.0, 104.0),
        ]
    )
    trades, _ = backtest_daily(data)

    assert trades.height == 1
    assert trades[0, "entry"] == pytest.approx(104.0)
    assert trades[0, "ret_opt"] == pytest.approx(-COST)


def test_golden_return_branches_cost_and_fixed_topk_cash_divisor():
    rows = []
    cases = {
        "TP_FIRST": (105.04, 102.0),
        "FLOOR": (104.0, 103.48),
        "CLOSE": (104.0, 104.208),
    }
    for market, (high, close) in cases.items():
        rows.append((market, "2026-01-01", 10.0, 100.0, 100.0, 100.0))
        rows.append((market, "2026-01-02", 10.0, 100.0, high, close))

    trades, daily = backtest_daily(_frame(rows))
    returns = {row["market"]: (row["ret_opt"], row["ret_pess"]) for row in trades.to_dicts()}
    assert returns["TP_FIRST"] == pytest.approx((0.008 - COST, -0.005 - COST))
    assert returns["FLOOR"] == pytest.approx((-0.005 - COST, -0.005 - COST))
    assert returns["CLOSE"] == pytest.approx((0.002 - COST, 0.002 - COST))
    assert daily[0, "ret_opt"] == pytest.approx(sum(item[0] for item in returns.values()) / 10)
    assert daily[0, "ret_pess"] == pytest.approx(sum(item[1] for item in returns.values()) / 10)


def test_polars_batch_registry_and_dedicated_runner_contract():
    assert DacapogoDailySourceStrategy.required_timeframes == ("1d",)
    assert DacapogoDailySourceStrategy.decision_cadence_seconds == 86_400
    assert DacapogoDailySourceStrategy.required_features == (
        "market",
        "date",
        "value",
        "open",
        "high",
        "close",
    )
    assert DacapogoDailySourceStrategy.runner_kind == "dedicated_dacapogo_daily_research"
    assert (
        GLOBAL_REGISTRY.get("strategy", "DacapogoDailySourceStrategy")
        is DacapogoDailySourceStrategy
    )
    assert (
        GLOBAL_REGISTRY.get_interface("strategy", "DacapogoDailySourceStrategy") == "polars_batch"
    )
    assert get_strategy_tier("DacapogoDailySourceStrategy") == "research_only"
    assert "DacapogoDailySourceStrategy" not in get_strategy_map()
    assert "DacapogoDailySourceStrategy" not in get_live_strategy_names(include_opt_in=True)
    strategy = DacapogoDailySourceStrategy()
    trades, daily = strategy.run(
        _frame(
            [
                ("BTC", "2026-01-01", 1.0, 100.0, 100.0, 100.0),
                ("BTC", "2026-01-02", 1.0, 100.0, 104.0, 104.0),
            ]
        )
    )
    assert (trades.height, daily.height) == (1, 1)
