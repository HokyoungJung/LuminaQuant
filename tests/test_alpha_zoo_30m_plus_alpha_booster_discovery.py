from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_30m_plus_alpha_booster_discovery.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_30m_plus_alpha_booster_discovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "train_return": 0.20,
        "validation_return": 0.12,
        "locked_oos_return": 0.04,
        "validation_mdd": 0.05,
        "validation_min_half_return": 0.01,
        "validation_trade_event_count": 40,
        "locked_oos_trade_event_count": 25,
        "train_return_per_turnover_proxy_bps": 30.0,
        "validation_return_per_turnover_proxy_bps": 35.0,
        "locked_oos_return_per_turnover_proxy_bps": 25.0,
        "paper_candidate_gate_pass": True,
        "decision": "paper_testnet_candidate_after_fill_preflight",
    }
    row.update(overrides)
    return row


def test_booster_score_ignores_locked_oos() -> None:
    base = _row(locked_oos_return=0.50, locked_oos_return_per_turnover_proxy_bps=100.0)
    changed = _row(locked_oos_return=-0.50, locked_oos_return_per_turnover_proxy_bps=-100.0)

    assert MODULE._booster_score(base) == MODULE._booster_score(changed)


def test_booster_report_target_is_post_freeze_gate_only() -> None:
    strong = MODULE._apply_booster_targets(_row())
    weak_oos = MODULE._apply_booster_targets(_row(locked_oos_return=0.01))

    assert strong["booster_target_gate_pass"] is True
    assert strong["decision"] == "paper_testnet_booster_candidate_after_fill_preflight"
    assert weak_oos["booster_target_gate_pass"] is False
    assert any("locked_oos_return" in reason for reason in weak_oos["booster_target_reasons"])
    assert MODULE._booster_score(strong) == MODULE._booster_score(weak_oos)


def test_trailing_state_signal_respects_30m_plus_runner_safety() -> None:
    close = pd.Series([100.0, 102.0, 104.0, 103.0, 106.0, 105.0])
    atr = pd.Series([1.0] * len(close))
    long_entry = pd.Series([False, True, False, False, False, False])
    short_entry = pd.Series([False] * len(close))
    long_exit = pd.Series([False, False, False, True, False, False])
    short_exit = pd.Series([False] * len(close))

    signal = MODULE._trailing_state_signal(
        close,
        long_entry,
        short_entry,
        long_exit,
        short_exit,
        atr,
        side="long_short",
        min_hold_bars=1,
        cooldown_bars=1,
        trail_atr_mult=2.0,
    )

    assert signal.tolist()[1] == 1.0
    assert set(np.unique(signal)).issubset({0.0, 1.0})
    assert MODULE.BAR_CONSTRUCTION == "native_1s_to_30m_base_then_requested_timeframe"


def test_parse_args_defaults_are_paper_only_30m_plus() -> None:
    args = MODULE.parse_args([])
    assert all(MODULE.feedback._timeframe_hours(tf) >= 0.5 for tf in args.timeframes.split(","))
    assert MODULE.STRATEGY_SCOPE == "single_symbol_only"
    assert MODULE.BOOSTER_TARGETS["preferred_min_validation_return"] == 0.10
