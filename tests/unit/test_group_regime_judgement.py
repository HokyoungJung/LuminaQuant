import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "research" / "run_group_regime_judgement.py"
)
SPEC = importlib.util.spec_from_file_location("run_group_regime_judgement", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_select_rules_prefers_consistent_feature() -> None:
    dates = pd.date_range("2025-01-01", periods=80, tz="UTC", freq="D")
    pattern = ([True, True, False, False] * 20)[:80]
    frame = pd.DataFrame(
        {
            "date": dates,
            "split_group": ["train"] * 50 + ["val"] * 20 + ["oos"] * 10,
            "autoresearch_leading_5d": pattern,
            "autoresearch_leading_20d": pattern,
            "autoresearch_vol_higher_20d": pattern,
            "autoresearch_drawdown_worse_20d": [not flag for flag in pattern],
            "relative_hit_rate_ge_55": pattern,
            "incumbent_ret_5d": [-0.01 if flag else 0.03 for flag in pattern],
            "incumbent_ret_20d": [-0.02 if flag else 0.05 for flag in pattern],
            "autoresearch_ret_5d": [0.04 if flag else -0.02 for flag in pattern],
            "autoresearch_ret_20d": [0.07 if flag else -0.03 for flag in pattern],
            "rel_ret_5d": [0.05 if flag else -0.05 for flag in pattern],
            "rel_ret_20d": [0.09 if flag else -0.08 for flag in pattern],
            "incumbent_vol_20d": [0.12 if flag else 0.09 for flag in pattern],
            "autoresearch_vol_20d": [0.18 if flag else 0.08 for flag in pattern],
            "rel_vol_ratio_20d": [1.5 if flag else 0.9 for flag in pattern],
            "incumbent_drawdown_20d": [-0.04 if flag else -0.02 for flag in pattern],
            "autoresearch_drawdown_20d": [-0.02 if flag else -0.05 for flag in pattern],
            "rel_drawdown_20d": [0.02 if flag else -0.03 for flag in pattern],
            "relative_hit_rate_20d": [0.7 if flag else 0.3 for flag in pattern],
        }
    )
    frame["forward_5d_incumbent"] = [
        -0.01 if flag else 0.03 for flag in frame["autoresearch_leading_5d"]
    ]
    frame["forward_5d_autoresearch"] = [
        0.04 if flag else -0.02 for flag in frame["autoresearch_leading_5d"]
    ]
    frame["forward_5d_rel"] = frame["forward_5d_autoresearch"] - frame["forward_5d_incumbent"]

    selected, diagnostics = MODULE._select_rules(frame, horizon_days=5)

    assert selected
    by_side = diagnostics["selected_by_side"]
    assert by_side["autoresearch"]
    assert any(rule["rule_id"] == "autoresearch_leading_5d" for rule in by_side["autoresearch"])
    assert any(rule["favored_group"] == "incumbent" for rule in selected)


def test_current_judgement_uses_active_rule_scores() -> None:
    latest_row = pd.Series(
        {
            "date": pd.Timestamp("2026-03-07", tz="UTC"),
            "incumbent_ret_5d": -0.01,
            "incumbent_ret_20d": -0.03,
            "autoresearch_ret_5d": 0.04,
            "autoresearch_ret_20d": 0.08,
            "rel_ret_5d": 0.05,
            "rel_ret_20d": 0.11,
            "incumbent_vol_20d": 0.12,
            "autoresearch_vol_20d": 0.18,
            "rel_vol_ratio_20d": 1.5,
            "incumbent_drawdown_20d": -0.04,
            "autoresearch_drawdown_20d": -0.02,
            "rel_drawdown_20d": 0.02,
            "relative_hit_rate_20d": 0.7,
            "autoresearch_leading_5d": True,
            "autoresearch_leading_20d": True,
            "autoresearch_vol_higher_20d": True,
            "autoresearch_drawdown_worse_20d": False,
            "relative_hit_rate_ge_55": True,
        }
    )
    selected_rules = [
        {
            "rule_id": "autoresearch_leading_5d",
            "label": "55/45 trailing 5-day return above incumbent",
            "family": "bool",
            "feature_names": ("autoresearch_leading_5d",),
            "threshold": None,
            "comparator": None,
            "polarity": "normal",
            "favored_group": "autoresearch",
            "score": 1.5,
        },
        {
            "rule_id": "not_autoresearch_drawdown_worse_20d",
            "label": "NOT 55/45 trailing 20-day drawdown worse than incumbent",
            "family": "bool",
            "feature_names": ("autoresearch_drawdown_worse_20d",),
            "threshold": None,
            "comparator": None,
            "polarity": "negated",
            "favored_group": "incumbent",
            "score": 0.4,
        },
    ]

    judgement = MODULE._current_judgement(latest_row=latest_row, selected_rules=selected_rules)

    assert judgement["favored_group"] == "autoresearch"
    assert judgement["autoresearch_score"] == 1.5
    assert judgement["incumbent_score"] == 0.4
    assert len(judgement["active_rules"]) == 2
