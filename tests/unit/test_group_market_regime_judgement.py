import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_group_market_regime_judgement.py"
)
SPEC = importlib.util.spec_from_file_location("run_group_market_regime_judgement", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_daily_market_feature_frame_builds_expected_flags() -> None:
    idx = pd.date_range("2025-01-01", periods=400, freq="30min", tz="UTC")

    def make_symbol(symbol: str, values: list[float]) -> pd.DataFrame:
        frame = pd.DataFrame({"datetime": idx, "close": values})
        frame["symbol"] = symbol
        frame["date"] = frame["datetime"].dt.floor("D")
        return frame

    up = [100.0 + (0.1 * i) for i in range(len(idx))]
    flat = [50.0 + (0.01 * i) for i in range(len(idx))]
    frames = [
        make_symbol("BTC/USDT", up),
        make_symbol("ETH/USDT", up),
        make_symbol("BNB/USDT", up),
        make_symbol("SOL/USDT", flat),
        make_symbol("TRX/USDT", flat),
    ]
    out = MODULE._daily_market_feature_frame(frames)
    latest = out.iloc[-1]

    assert bool(latest["btc_above_ma192"]) is True
    assert bool(latest["btc_above_ma336"]) is True
    assert bool(latest["btc_trend_gap_336_pos"]) is True
    assert bool(latest["breadth_ma96_ge_60"]) is True
    assert bool(latest["breadth_ma192_ge_60"]) is True
    assert bool(latest["breadth_ma96_ge_40"]) is True
    assert bool(latest["basket_ret96_top3_pos"]) is True
    assert bool(latest["btc_trend_accel_pos"]) is True
    assert bool(latest["breadth_expanding"]) is False
    assert isinstance(bool(latest["basket_ret_dispersion_compressed"]), bool)
    assert float(latest["btc_trend_accel"]) > 0.0
    assert isinstance(float(latest["basket_ret96_top3_mean"]), float)
    assert isinstance(float(latest["basket_ret96_dispersion"]), float)
    assert isinstance(float(latest["basket_vol_ratio"]), float)


def test_current_judgement_uses_market_rules() -> None:
    latest_row = pd.Series(
        {
            "date": pd.Timestamp("2026-03-07", tz="UTC"),
            "btc_above_ma192": True,
            "btc_above_ma336": True,
            "btc_trend_gap_336_pos": True,
            "breadth_ma96_ge_60": True,
            "breadth_ma192_ge_60": False,
            "breadth_ma96_ge_40": True,
            "basket_ret96_pos": True,
            "basket_ret96_top3_pos": True,
            "basket_vol_ratio_moderate": True,
            "btc_trend_accel_pos": True,
            "breadth_expanding": True,
            "basket_ret_dispersion_compressed": True,
            "btc_close": 1.0,
            "btc_ma192": 0.9,
            "btc_ma336": 0.8,
            "btc_trend_gap_192": 0.11,
            "btc_trend_gap_336": 0.25,
            "btc_trend_accel": 0.125,
            "breadth_ma96": 0.8,
            "breadth_ma192": 0.4,
            "breadth_delta": 0.4,
            "basket_ret96": 0.03,
            "basket_ret96_top3_mean": 0.04,
            "basket_ret96_dispersion": 0.01,
            "basket_vol_ratio": 1.1,
        }
    )
    selected_rules = [
        {
            "rule_id": "btc_above_ma192",
            "label": "BTC above 4-day trend",
            "family": "bool",
            "feature_names": ("btc_above_ma192",),
            "threshold": None,
            "comparator": None,
            "polarity": "normal",
            "favored_group": "autoresearch",
            "score": 1.2,
        },
        {
            "rule_id": "not_breadth_ma192_ge_60",
            "label": "NOT 4-day breadth at or above 60%",
            "family": "bool",
            "feature_names": ("breadth_ma192_ge_60",),
            "threshold": None,
            "comparator": None,
            "polarity": "negated",
            "favored_group": "incumbent",
            "score": 0.3,
        },
    ]
    judgement = MODULE._current_judgement(latest_row=latest_row, selected_rules=selected_rules)
    assert judgement["favored_group"] == "autoresearch"
    assert judgement["autoresearch_score"] == 1.2
    assert judgement["incumbent_score"] == 0.3


def test_oos_confirmation_override_triggers_only_for_small_train_val_mismatch() -> None:
    assert (
        MODULE._should_use_oos_confirmation_override(
            combined_mean=-0.0003,
            oos_mean=0.0015,
            oos_count=6,
        )
        is True
    )
    assert (
        MODULE._should_use_oos_confirmation_override(
            combined_mean=-0.0020,
            oos_mean=0.0015,
            oos_count=6,
        )
        is False
    )
    assert (
        MODULE._should_use_oos_confirmation_override(
            combined_mean=-0.0003,
            oos_mean=0.0004,
            oos_count=6,
        )
        is False
    )
