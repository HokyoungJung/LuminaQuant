from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from lumina_quant.strategy_factory import research_entrypoints
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


def test_research_config_to_overrides_routes_unmapped_registered_strategies():
    runtime = SimpleNamespace(research=SimpleNamespace(route_unmapped_registered_strategies=True))

    overrides = support.research_config_to_overrides(runtime)

    assert overrides["score_config_research"] == {"route_unmapped_registered_strategies": True}


def test_research_config_costs_must_be_finite_and_non_negative():
    for multiplier, override in (
        (math.nan, None),
        (-1.0, None),
        (1.0, math.inf),
        (1.0, -0.1),
    ):
        runtime = SimpleNamespace(
            research=SimpleNamespace(
                cost_rate_multiplier=multiplier,
                cost_rate_bps_override=override,
            )
        )
        with pytest.raises(ValueError):
            support.research_config_to_overrides(runtime)


def test_runtime_research_costs_override_score_json_costs_only():
    merged = research_entrypoints._merge_score_config_with_research_overrides(
        {
            "research": {
                "cost_rate_multiplier": 0.5,
                "cost_rate_bps_override": 1.0,
                "caller_owned": "preserved",
            }
        },
        score_config_research_overrides={
            "cost_rate_multiplier": 2.0,
            "cost_rate_bps_override": 20.0,
            "profile_owned": "caller-wins",
        },
        deflation_kwargs={},
    )

    assert merged == {
        "research": {
            "cost_rate_multiplier": 2.0,
            "cost_rate_bps_override": 20.0,
            "caller_owned": "preserved",
            "profile_owned": "caller-wins",
        }
    }
