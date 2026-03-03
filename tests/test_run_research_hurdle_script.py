from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_research_hurdle.py"
    spec = importlib.util.spec_from_file_location("run_research_hurdle_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_research_hurdle module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_hurdle_score_from_row_uses_configurable_fallbacks():
    row = {"hurdle_fields": {}}
    cfg = MODULE._resolve_factory_score_config(
        {
            "futures_strategy_factory": {
                "hurdle": {
                    "missing_score_fallback": -123.0,
                    "missing_excess_fallback": -7.0,
                }
            }
        }
    )
    score, excess, passed = MODULE._hurdle_score_from_row(row, "oos", score_config=cfg)
    assert float(score) == -123.0
    assert float(excess) == -7.0
    assert passed is False


def test_normalize_strategy_row_applies_failed_candidate_penalty_from_config():
    row = {
        "name": "sample_strategy",
        "symbols": ["BTC/USDT"],
        "params": {"x": 1},
        "hurdle_fields": {"oos": {"score": 1.0, "excess_return": 2.0, "pass": False}},
    }
    cfg = MODULE._resolve_factory_score_config(
        {"futures_strategy_factory": {"hurdle": {"failed_candidate_base_penalty": -50.0}}}
    )
    normalized = MODULE._normalize_strategy_row(
        row,
        timeframe="1h",
        source_report=Path("reports/mock.json"),
        mode="oos",
        source="unit_test",
        score_config=cfg,
    )
    assert normalized is not None
    assert float(normalized.get("base_score", 0.0)) == -48.0


def test_regime_bias_uses_configurable_thresholds_and_bonuses():
    row = {"family": "trend_overlay", "symbols": ["BTC/USDT"]}
    snapshot = {
        "BTC/USDT": {
            "trend_efficiency": 0.6,
            "normalized_true_range": 0.02,
            "volume_shock_zscore": 2.0,
        }
    }
    cfg = MODULE._resolve_factory_score_config(
        {
            "futures_strategy_factory": {
                "regime_bias": {
                    "trend_overlay": {
                        "eff_min": 0.0,
                        "eff_bonus": 1.0,
                        "ntr_min": 0.0,
                        "ntr_bonus": 1.0,
                        "vshock_min": 0.0,
                        "vshock_bonus": 1.0,
                    }
                }
            }
        }
    )
    bias = MODULE._regime_bias(row, snapshot, score_config=cfg)
    assert float(bias) == 3.0
