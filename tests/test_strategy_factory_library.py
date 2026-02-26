from __future__ import annotations

from lumina_quant.strategies.factory_candidate_set import (
    build_candidate_set,
    summarize_candidate_set,
)
from lumina_quant.strategies.registry import get_strategy_names
from lumina_quant.strategy_factory import (
    build_binance_futures_candidates,
    build_single_asset_portfolio_sets,
    candidate_identity,
    select_diversified_shortlist,
)


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


def test_shortlist_filters_weak_single_and_assigns_weights():
    candidates = [
        {
            "name": "rsi_1m_candidate",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 2.0}},
            "oos": {"return": 0.02, "sharpe": 1.0, "mdd": 0.08, "trades": 19},
            "params": {"rsi_period": 14},
        },
        {
            "name": "pair_z_1m_candidate",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 5.5}},
            "oos": {"return": 0.11, "sharpe": 1.4, "mdd": 0.05, "trades": 30},
            "params": {"entry_z": 2.0},
        },
        {
            "name": "topcap_tsmom_1h_candidate",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 4.2}},
            "oos": {"return": 0.09, "sharpe": 1.1, "mdd": 0.06, "trades": 24},
            "params": {"lookback_bars": 16},
        },
    ]

    shortlist = select_diversified_shortlist(
        candidates,
        mode="oos",
        max_total=5,
        max_per_family=5,
        max_per_timeframe=5,
        single_min_score=0.0,
        single_min_return=0.0,
        single_min_sharpe=0.7,
        single_min_trades=20,
        drop_single_without_metrics=True,
        include_weights=True,
    )

    # Weak single strategy is filtered out; mixed strategies remain and get normalized weights.
    assert len(shortlist) == 2
    assert all(float(row.get("portfolio_weight", 0.0)) > 0.0 for row in shortlist)
    total_weight = sum(float(row.get("portfolio_weight", 0.0)) for row in shortlist)
    assert abs(total_weight - 1.0) < 1e-9


def test_build_single_asset_portfolio_sets_from_shortlist():
    shortlist = [
        {
            "name": "rsi_btc",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "shortlist_score": 5.0,
            "oos": {"return": 0.1, "sharpe": 1.2},
        },
        {
            "name": "ma_btc_alt",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "shortlist_score": 4.5,
            "oos": {"return": 0.08, "sharpe": 1.0},
        },
        {
            "name": "rsi_eth",
            "strategy_timeframe": "1m",
            "symbols": ["ETH/USDT"],
            "shortlist_score": 4.8,
            "oos": {"return": 0.09, "sharpe": 1.1},
        },
        {
            "name": "pair_btc_eth",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "shortlist_score": 6.0,
            "oos": {"return": 0.12, "sharpe": 1.5},
        },
    ]
    sets = build_single_asset_portfolio_sets(shortlist, mode="oos", max_per_asset=2, max_sets=4)
    assert len(sets) >= 1
    top = sets[0]
    assert top["member_count"] == 2
    assert abs(sum(float(row.get("portfolio_weight", 0.0)) for row in top["members"]) - 1.0) < 1e-9
