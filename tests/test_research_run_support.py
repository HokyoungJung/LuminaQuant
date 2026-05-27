from __future__ import annotations

from lumina_quant.strategy_factory import research_run_support as support


def _candidate(
    name: str,
    strategy_class: str,
    family: str,
    *,
    symbols: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "candidate_id": name,
        "strategy_class": strategy_class,
        "family": family,
        "strategy_timeframe": "1h",
        "symbols": symbols or ["BTC/USDT"],
        "params": {},
    }


def test_adapt_candidate_inputs_round_robins_across_families_when_limited():
    candidates = [
        _candidate("trend_a", "CompositeTrendStrategy", "trend"),
        _candidate("trend_b", "CompositeTrendStrategy", "trend"),
        _candidate("trend_c", "CompositeTrendStrategy", "trend"),
        _candidate(
            "carry_a",
            "PerpCrowdingCarryStrategy",
            "carry",
            symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        ),
        _candidate(
            "cross_a",
            "LastDayLiquidityRegimeStrategy",
            "cross_sectional",
            symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
        ),
    ]

    adapted = support._adapt_candidate_inputs(candidates, max_candidates=3)

    assert [row["candidate_id"] for row in adapted] == ["trend_a", "carry_a", "cross_a"]


def test_adapt_candidate_inputs_preserves_input_order_when_unbounded():
    candidates = [
        _candidate("trend_a", "CompositeTrendStrategy", "trend"),
        _candidate("trend_b", "CompositeTrendStrategy", "trend"),
        _candidate("carry_a", "PerpCrowdingCarryStrategy", "carry"),
    ]

    adapted = support._adapt_candidate_inputs(candidates, max_candidates=0)

    assert [row["candidate_id"] for row in adapted] == ["trend_a", "trend_b", "carry_a"]
