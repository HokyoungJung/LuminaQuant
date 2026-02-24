from __future__ import annotations

from pathlib import Path

from lumina_quant.strategy_factory.pipeline import build_shortlist_payload


def test_build_shortlist_payload_applies_filters_weights_and_sets():
    report = {
        "selected_team": [
            {
                "name": "weak_rsi_btc",
                "strategy_timeframe": "1m",
                "symbols": ["BTC/USDT"],
                "hurdle_fields": {"oos": {"pass": True, "score": 1.0}},
                "oos": {"return": 0.01, "sharpe": 1.0, "mdd": 0.06, "trades": 19},
                "params": {"rsi_period": 14},
            },
            {
                "name": "strong_rsi_btc",
                "strategy_timeframe": "1m",
                "symbols": ["BTC/USDT"],
                "hurdle_fields": {"oos": {"pass": True, "score": 3.2}},
                "oos": {"return": 0.07, "sharpe": 1.4, "mdd": 0.05, "trades": 28},
                "params": {"rsi_period": 12},
            },
            {
                "name": "strong_rsi_eth",
                "strategy_timeframe": "5m",
                "symbols": ["ETH/USDT"],
                "hurdle_fields": {"oos": {"pass": True, "score": 2.9}},
                "oos": {"return": 0.05, "sharpe": 1.2, "mdd": 0.04, "trades": 21},
                "params": {"rsi_period": 16},
            },
            {
                "name": "pair_btc_eth",
                "strategy_timeframe": "1m",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "hurdle_fields": {"oos": {"pass": True, "score": 3.4}},
                "oos": {"return": 0.08, "sharpe": 1.5, "mdd": 0.05, "trades": 26},
                "params": {"entry_z": 2.0},
            },
            {
                "name": "topcap_multi",
                "strategy_timeframe": "1h",
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "hurdle_fields": {"oos": {"pass": True, "score": 3.1}},
                "oos": {"return": 0.06, "sharpe": 1.1, "mdd": 0.07, "trades": 19},
                "params": {"lookback_bars": 20},
            },
        ]
    }

    payload = build_shortlist_payload(
        report=report,
        mode="oos",
        shortlist_max_total=10,
        shortlist_max_per_family=10,
        shortlist_max_per_timeframe=10,
        single_min_score=0.0,
        drop_single_without_metrics=True,
        single_min_return=0.0,
        single_min_sharpe=0.0,
        single_min_trades=20,
        allow_multi_asset=False,
        include_weights=True,
        weight_temperature=0.35,
        max_weight=0.5,
        set_max_per_asset=2,
        set_max_sets=8,
        manifest_path=Path("reports/strategy_factory_candidates_test.json"),
        research_report_path=Path("reports/strategy_factory_report_test.json"),
    )

    shortlist = list(payload.get("shortlist") or [])
    assert len(shortlist) == 3
    assert all(str(row.get("mix_type")) != "multi" for row in shortlist)
    assert "weak_rsi_btc" not in {str(row.get("name")) for row in shortlist}

    total_weight = sum(float(row.get("portfolio_weight", 0.0)) for row in shortlist)
    assert abs(total_weight - 1.0) < 1e-9

    portfolio_sets = list(payload.get("portfolio_sets") or [])
    assert payload.get("portfolio_set_count") == len(portfolio_sets)
    assert len(portfolio_sets) >= 1

    top_set = portfolio_sets[0]
    assert top_set.get("set_id") == "single_asset_top_set"
    assert int(top_set.get("member_count", 0)) == 2
    member_weights = [float(row.get("portfolio_weight", 0.0)) for row in list(top_set.get("members") or [])]
    assert abs(sum(member_weights) - 1.0) < 1e-9
