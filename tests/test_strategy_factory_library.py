from __future__ import annotations

from lumina_quant.strategy_factory import (
    build_binance_futures_candidates,
    candidate_identity,
    select_diversified_shortlist,
)
from strategies.factory_candidate_set import build_candidate_set, summarize_candidate_set
from strategies.registry import get_strategy_names


def test_factory_candidate_set_is_large_and_diverse():
    candidates = build_candidate_set(
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
        timeframes=["1s", "1m", "1h"],
    )
    summary = summarize_candidate_set(candidates)

    assert len(candidates) >= 100
    assert "RegimeBreakoutCandidateStrategy" in summary["strategies"]
    assert "VolatilityCompressionReversionStrategy" in summary["strategies"]
    assert "trend_breakout" in summary["families"]
    assert "mean_reversion" in summary["families"]


def test_strategy_factory_library_builds_candidates_and_shortlist():
    rows = build_binance_futures_candidates(
        timeframes=["1m"],
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
    )
    assert len(rows) > 20

    mock_candidates = []
    for idx, item in enumerate(rows[:40]):
        as_dict = item.to_dict()
        as_dict["name"] = f"{as_dict['name']}_{idx}"
        as_dict["strategy_timeframe"] = as_dict["timeframe"]
        as_dict["hurdle_fields"] = {"oos": {"pass": True, "score": float(100 - idx)}}
        as_dict["identity"] = candidate_identity(as_dict)
        mock_candidates.append(as_dict)

    shortlist = select_diversified_shortlist(
        mock_candidates,
        mode="oos",
        max_total=12,
        max_per_family=6,
        max_per_timeframe=6,
    )
    assert 1 <= len(shortlist) <= 12
    assert all("shortlist_score" in row for row in shortlist)


def test_registry_exposes_new_candidate_strategies():
    names = get_strategy_names()
    assert "RegimeBreakoutCandidateStrategy" in names
    assert "VolatilityCompressionReversionStrategy" in names
