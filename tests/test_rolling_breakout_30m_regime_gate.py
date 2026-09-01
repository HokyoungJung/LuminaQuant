from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "rolling_breakout_30m_regime_gate.py"
SPEC = importlib.util.spec_from_file_location("rolling_breakout_30m_regime_gate", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load rolling_breakout_30m_regime_gate module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TARGET_CANDIDATE = MODULE.TARGET_CANDIDATE
build_rolling_breakout_30m_gate = MODULE.build_rolling_breakout_30m_gate
write_rolling_breakout_30m_gate = MODULE.write_rolling_breakout_30m_gate


def _call_gate(
    *,
    decision: dict[str, object] | None = None,
    feature_frame: pd.DataFrame | None = None,
    evaluation_payload: dict[str, object] | None = None,
    extra_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    params = inspect.signature(build_rolling_breakout_30m_gate).parameters
    candidate_kwargs: dict[str, object] = {
        "decision": decision if decision is not None else _decision_payload(),
        "feature_frame": feature_frame if feature_frame is not None else _feature_frame(),
        "evaluation_payload": (
            evaluation_payload if evaluation_payload is not None else _evaluation_payload()
        ),
    }
    if extra_kwargs:
        candidate_kwargs.update(extra_kwargs)
    return build_rolling_breakout_30m_gate(
        **{name: value for name, value in candidate_kwargs.items() if name in params}
    )


def _decision_payload() -> dict[str, object]:
    return {
        "timeframe_rows": [
            {
                "timeframe": "30m",
                "windows": {
                    "train_start": "2025-01-01T00:00:00Z",
                    "train_end_exclusive": "2025-01-03T00:00:00Z",
                    "val_start": "2025-01-03T00:00:00Z",
                    "val_end_exclusive": "2025-01-05T00:00:00Z",
                    "actual_oos_end_exclusive": "2025-01-08T00:00:00Z",
                },
                "best_row": {
                    "candidate_id": "cand-rolling",
                    "name": TARGET_CANDIDATE,
                    "strategy_class": "RollingBreakoutStrategy",
                    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
                    "params": {
                        "lookback_bars": 64,
                        "breakout_buffer": 0.002,
                        "atr_window": 21,
                        "atr_stop_multiplier": 2.8,
                        "stop_loss_pct": 0.03,
                        "allow_short": True,
                    },
                    "metadata": {"timeframe": "30m"},
                    "train": {"return": -0.016, "sharpe": -1.1},
                    "val": {"return": 0.015, "sharpe": 1.9},
                    "oos": {"return": 0.028, "sharpe": 1.0},
                },
            }
        ]
    }


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-01-01T00:00:00Z",
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": False,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "basket_vol_ratio": 1.1,
            },
            {
                "date": "2025-01-02T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": False,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "basket_vol_ratio": 1.1,
            },
            {
                "date": "2025-01-03T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2025-01-04T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": True,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2025-01-05T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": True,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2025-01-06T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2025-01-07T00:00:00Z",
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": False,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "basket_vol_ratio": 1.1,
            },
        ]
    )


def _evaluation_payload() -> dict[str, object]:
    timestamps = pd.date_range("2025-01-01", periods=7, freq="D", tz="UTC")
    return {
        "timestamps": list(timestamps),
        "returns_raw": [-0.0100, -0.0060, 0.0080, 0.0070, 0.0300, 0.0200, -0.0200],
        "turnover": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        "exposure": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "benchmark_returns": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cost_rate": 0.0005,
        "split_masks": {
            "train": [True, True, False, False, False, False, False],
            "val": [False, False, True, True, False, False, False],
            "oos": [False, False, False, False, True, True, True],
        },
    }


def _train_val_only_fixture() -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    decision = {
        "timeframe_rows": [
            {
                "timeframe": "30m",
                "windows": {
                    "train_start": "2025-01-01T00:00:00Z",
                    "train_end_exclusive": "2025-01-05T00:00:00Z",
                    "val_start": "2025-01-05T00:00:00Z",
                    "val_end_exclusive": "2025-01-09T00:00:00Z",
                    "actual_oos_end_exclusive": "2025-01-13T00:00:00Z",
                },
                "best_row": {
                    "candidate_id": "cand-rolling",
                    "name": TARGET_CANDIDATE,
                    "strategy_class": "RollingBreakoutStrategy",
                    "symbols": ["BTC/USDT"],
                    "metadata": {"timeframe": "30m"},
                    "train": {"return": -0.05, "sharpe": -0.5},
                    "val": {"return": 0.03, "sharpe": 1.2},
                    "oos": {"return": -0.3, "sharpe": -1.0},
                },
            }
        ]
    }
    dates = pd.date_range("2025-01-01", periods=12, freq="D", tz="UTC")
    feature_rows = []
    active_days = {2, 4, 5, 6}
    for idx, day in enumerate(dates):
        feature_rows.append(
            {
                "date": day,
                "btc_above_ma192": idx in active_days,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": False,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": False,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.1,
                "breadth_ma192": 0.1,
                "basket_vol_ratio": 1.0,
            }
        )
    feature_frame = pd.DataFrame(feature_rows)
    returns_raw = [
        0.0,
        0.0,
        0.0,
        -0.05,
        0.0,
        0.015,
        0.02,
        0.01,
        0.0,
        -0.15,
        -0.12,
        -0.08,
    ]
    evaluation_payload = {
        "timestamps": list(dates),
        "returns_raw": returns_raw,
        "turnover": [0.02] * len(dates),
        "exposure": [1.0] * len(dates),
        "benchmark_returns": [0.0] * len(dates),
        "cost_rate": 0.0005,
        "split_masks": {
            "train": [idx < 4 for idx in range(len(dates))],
            "val": [4 <= idx < 8 for idx in range(len(dates))],
            "oos": [8 <= idx < len(dates) for idx in range(len(dates))],
        },
    }
    return decision, feature_frame, evaluation_payload


def test_build_rolling_breakout_gate_uses_lagged_ex_ante_rule():
    gate = build_rolling_breakout_30m_gate(
        _decision_payload(),
        feature_frame=_feature_frame(),
        evaluation_payload=_evaluation_payload(),
    )

    selected = dict(gate["selected_rule"])
    assert selected["signal_lag_days"] == 1
    assert float(selected["metrics"]["oos"]["return"]) > 0.0
    assert float(selected["metrics"]["oos"]["trade_count"]) >= 2.0
    assert "hard_reject_reasons" in selected

    by_rule = {row["rule_id"]: row for row in gate["evaluated_rules"]}
    focused = by_rule["btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos"]
    broad = by_rule["btc_above_ma192"]
    assert focused["signal_lag_days"] == 1
    assert float(focused["metrics"]["oos"]["return"]) > float(broad["metrics"]["oos"]["return"])
    assert focused["metrics"]["oos"]["gate_days"] == 2


def test_write_rolling_breakout_30m_gate_writes_files(tmp_path: Path):
    result = write_rolling_breakout_30m_gate(
        _decision_payload(),
        report_root=tmp_path,
        feature_frame=_feature_frame(),
        evaluation_payload=_evaluation_payload(),
    )

    json_path = Path(result["json_path"])
    md_path = Path(result["md_path"])
    assert json_path.exists()
    assert md_path.exists()
    text = json_path.read_text(encoding="utf-8")
    assert TARGET_CANDIDATE in text
    assert "selected_rule" in text


def test_train_val_only_selection_ignores_oos_mutations():
    baseline = _call_gate(extra_kwargs={"selection_basis": "train_val_only"})
    degraded_eval = _evaluation_payload()
    degraded_eval["returns_raw"] = [*degraded_eval["returns_raw"][:-1], -2.0]
    degraded = _call_gate(
        evaluation_payload=degraded_eval,
        extra_kwargs={"selection_basis": "train_val_only"},
    )

    assert degraded["selected_rule"]["rule_id"] == baseline["selected_rule"]["rule_id"]
    assert degraded["selected_rule"]["score"] == pytest.approx(
        baseline["selected_rule"]["score"],
        abs=1e-12,
    )


def test_train_val_only_survival_uses_declared_thresholds():
    decision, feature_frame, evaluation_payload = _train_val_only_fixture()
    gate = _call_gate(
        decision=decision,
        feature_frame=feature_frame,
        evaluation_payload=evaluation_payload,
        extra_kwargs={"selection_basis": "train_val_only"},
    )

    selected_rule = dict(gate.get("selected_rule") or {})
    assert gate.get("selection_basis") == "train_val_only"
    assert gate.get("survives_train_val") is True
    assert selected_rule.get("rule_id") == "btc_above_ma192"
    assert selected_rule.get("survives_train_val") is True
    assert gate.get("recommended_action") == "activate_conditionally"
