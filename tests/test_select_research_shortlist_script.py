from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "select_research_shortlist.py"
    spec = importlib.util.spec_from_file_location("select_research_shortlist_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load select_research_shortlist module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_candidate_score_uses_overridden_weights():
    row = {
        "hurdle_fields": {"oos": {"pass": True, "score": 10.0}},
        "oos": {"return": 0.1, "sharpe": 1.0, "mdd": 0.0, "trades": 10},
    }
    cfg = MODULE._resolve_shortlist_score_config(
        {
            "strategy_shortlist": {
                "candidate_score": {
                    "sharpe_weight": 1.0,
                    "return_weight": 10.0,
                    "drawdown_penalty": 0.0,
                    "trade_bonus_cap": 0.0,
                    "trade_bonus_per_trade": 0.0,
                }
            }
        }
    )
    score = MODULE._candidate_score(
        row,
        mode="oos",
        require_pass=False,
        score_config=cfg,
    )
    assert abs(float(score) - 12.0) < 1e-12


def test_apply_portfolio_weights_honors_risk_penalty_override():
    rows = [
        {"selection_score": 1.0, "oos": {"mdd": 0.1}},
        {"selection_score": 1.0, "oos": {"mdd": 0.5}},
    ]

    flat_cfg = MODULE._resolve_shortlist_score_config(
        {"strategy_shortlist": {"portfolio_weights": {"mdd_risk_penalty_coeff": 0.0}}}
    )
    weighted_flat = MODULE._apply_portfolio_weights(
        [dict(row) for row in rows],
        score_config=flat_cfg,
    )
    w_flat = [float(item.get("portfolio_weight", 0.0)) for item in weighted_flat]
    assert abs(sum(w_flat) - 1.0) < 1e-12
    assert abs(w_flat[0] - w_flat[1]) < 1e-12

    steep_cfg = MODULE._resolve_shortlist_score_config(
        {"strategy_shortlist": {"portfolio_weights": {"mdd_risk_penalty_coeff": 10.0}}}
    )
    weighted_steep = MODULE._apply_portfolio_weights(
        [dict(row) for row in rows],
        score_config=steep_cfg,
    )
    by_mdd = sorted(
        weighted_steep,
        key=lambda item: float((item.get("oos") or {}).get("mdd", 0.0)),
    )
    assert float(by_mdd[0].get("portfolio_weight", 0.0)) > float(by_mdd[1].get("portfolio_weight", 0.0))
