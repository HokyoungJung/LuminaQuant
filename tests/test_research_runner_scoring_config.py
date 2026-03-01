from __future__ import annotations

import numpy as np
from lumina_quant.strategy_factory import research_runner


def test_run_candidate_research_scoring_config_defaults_and_override(monkeypatch):
    candidates = [
        {
            "candidate_id": "cand-a",
            "name": "candidate-a",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1m",
            "timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
        {
            "candidate_id": "cand-b",
            "name": "candidate-b",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1m",
            "timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
    ]

    metric_by_id = {
        "cand-a": {"sharpe": 1.8, "return": 0.02, "deflated_sharpe": 0.5},
        "cand-b": {"sharpe": 0.2, "return": 0.08, "deflated_sharpe": 0.5},
    }

    def _mock_load_bundle_cache(*, symbols, timeframes):
        _ = symbols, timeframes
        return {}, {"parquet": [], "csv": [], "synthetic": []}

    def _mock_benchmark_cache(cache, timeframes):
        _ = cache, timeframes
        return {"1m": np.zeros(24, dtype=float)}

    def _mock_evaluate_candidate(candidate, *, cache, benchmark_cache, candidate_count, scoring_config=None):
        _ = cache, benchmark_cache, candidate_count, scoring_config
        cid = str(candidate.get("candidate_id"))
        base = metric_by_id[cid]
        metrics = {
            "return": float(base["return"]),
            "total_return": float(base["return"]),
            "cagr": float(base["return"]),
            "sharpe": float(base["sharpe"]),
            "sortino": float(base["sharpe"]),
            "calmar": float(base["return"]),
            "mdd": 0.1,
            "max_drawdown": 0.1,
            "turnover": 1.0,
            "trades": 20.0,
            "trade_count": 20.0,
            "win_rate": 0.5,
            "avg_trade": 0.01,
            "exposure": 0.5,
            "volatility": 0.2,
            "stability": 0.1,
            "rolling_sharpe_min": 0.0,
            "worst_month": -0.05,
            "benchmark_corr": 0.0,
            "deflated_sharpe": float(base["deflated_sharpe"]),
            "pbo": 0.1,
            "spa_pvalue": 0.2,
        }
        return {
            "candidate": candidate,
            "returns": np.linspace(0.001, 0.002, 24),
            "turnover": np.full(24, 1.0),
            "exposure": np.full(24, 0.2),
            "train": dict(metrics),
            "val": dict(metrics),
            "oos": dict(metrics),
            "oos_cost_stress": {"x2": {"sharpe": 0.5, "return": 0.01}, "x3": {"sharpe": 0.2, "return": 0.005}},
            "hurdle_fields": {"train": {"pass": True, "score": 1.0}, "val": {"pass": True, "score": 1.0}, "oos": {"pass": True, "score": 1.0}},
            "pass": True,
            "hard_reject_reasons": {},
            "metadata": {"cost_rate": 0.0005},
        }

    monkeypatch.setattr(research_runner, "_load_bundle_cache", _mock_load_bundle_cache)
    monkeypatch.setattr(research_runner, "_benchmark_cache", _mock_benchmark_cache)
    monkeypatch.setattr(research_runner, "_evaluate_candidate", _mock_evaluate_candidate)

    report_default = research_runner.run_candidate_research(
        candidates=candidates,
        base_timeframe="1s",
        strategy_timeframes=["1m"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
    )
    rows_default = list(report_default.get("candidates") or [])
    assert rows_default
    assert rows_default[0]["candidate_id"] == "cand-a"
    assert report_default["scoring_config"]["candidate_rank_score_weights"]["sharpe_weight"] == 2.8

    report_override = research_runner.run_candidate_research(
        candidates=candidates,
        base_timeframe="1s",
        strategy_timeframes=["1m"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
        score_config={
            "candidate_rank_score_weights": {
                "sharpe_weight": 0.1,
                "return_weight": 80.0,
            }
        },
    )
    rows_override = list(report_override.get("candidates") or [])
    assert rows_override
    assert rows_override[0]["candidate_id"] == "cand-b"
