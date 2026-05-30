from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/research/run_alpha_zoo_69_asset_optuna_hybrid_refit.py"
spec = importlib.util.spec_from_file_location(
    "run_alpha_zoo_69_asset_optuna_hybrid_refit", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_gate_candidate_keeps_real_money_false_and_flags_validation_spike() -> None:
    row = {
        "train_return": 0.03,
        "validation_return": 0.08,
        "validation_mdd": 0.05,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 40,
        "train_return_per_turnover_proxy_bps": 20.0,
        "validation_return_per_turnover_proxy_bps": 20.0,
    }

    gated = MODULE.gate_candidate(row)

    assert gated["ready_for_real"] is False
    assert gated["real_money_execution"] is False
    assert gated["real_execution_allowed"] is False
    assert gated["ready_for_paper"] is False
    assert (
        "train_return_below_validation_return_possible_validation_spike"
        in gated["rejection_reasons"]
    )


def test_concentration_metrics_detect_top_symbol_share() -> None:
    streams = [
        MODULE.CandidateStream(
            row={
                "model_id": "a",
                "symbol": "ETHUSDT",
                "asset_group": "crypto_core",
                "family": "trend",
                "timeframe": "1h",
                "notional_fraction": 1.0,
            },
            returns=pd.Series([0.0]),
            position=pd.Series([0.0]),
        ),
        MODULE.CandidateStream(
            row={
                "model_id": "b",
                "symbol": "SOLUSDT",
                "asset_group": "crypto_core",
                "family": "trend",
                "timeframe": "1h",
                "notional_fraction": 1.0,
            },
            returns=pd.Series([0.0]),
            position=pd.Series([0.0]),
        ),
    ]

    metrics = MODULE.concentration_metrics(streams, np.array([0.8, 0.2]))

    assert metrics["top_symbol"] == "ETHUSDT"
    assert metrics["top_symbol_share"] == 0.8
    assert "top_symbol_share_above_35pct" in metrics["concentration_flags"]


def test_standard_split_payload_disables_locked_oos_for_live_refit() -> None:
    windows = MODULE.build_standard_split_windows(
        data_end_utc="2026-05-30T04:40:00Z", validation_weeks=8, bar_minutes=60
    )

    payload = windows.as_payload()

    assert payload["train"]["enabled"] is True
    assert payload["validation"]["enabled"] is True
    assert payload["locked_oos"] == {
        "start": None,
        "end": None,
        "role": "disabled_for_live_final_refit_no_test_set_reserved",
        "enabled": False,
    }


def test_simulate_symbol_uses_integer_leverage_and_costs() -> None:
    bars = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=4, freq="h"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    signal = np.array([0.0, 1.0, 1.0, 0.0])

    sim = MODULE.simulate_symbol(bars, signal, integer_leverage=2, allocation_fraction=0.1)

    assert sim.returns.shape == (4,)
    assert np.isfinite(sim.returns).all()
    assert sim.position.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_selected_hybrid_gate_blocks_negative_train_even_with_positive_validation() -> None:
    gate = MODULE.gate_selected_hybrid(
        {
            "train_total_return": -0.01,
            "validation_total_return": 0.05,
            "validation_max_drawdown": 0.04,
            "train_component_trade_event_count": 100,
            "validation_component_trade_event_count": 40,
            "train_return_per_turnover_proxy_bps": 20.0,
            "validation_return_per_turnover_proxy_bps": 20.0,
        }
    )

    assert gate["ready_for_paper"] is False
    assert gate["ready_for_real"] is False
    assert "hybrid_train_return_not_positive" in gate["rejection_reasons"]


def test_selected_hybrid_gate_requires_component_trade_events_and_efficiency() -> None:
    gate = MODULE.gate_selected_hybrid(
        {
            "train_total_return": 0.05,
            "validation_total_return": 0.04,
            "validation_max_drawdown": 0.04,
            "train_trade_event_count": 100,
            "validation_trade_event_count": 3,
            "train_component_trade_event_count": 120,
            "validation_component_trade_event_count": 45,
            "train_return_per_turnover_proxy_bps": 12.0,
            "validation_return_per_turnover_proxy_bps": 9.9,
        }
    )

    assert gate["ready_for_paper"] is False
    assert "hybrid_validation_trade_event_count_below_30" not in gate["rejection_reasons"]
    assert any(
        reason.startswith("hybrid_validation_return_per_turnover_proxy_bps_9.900")
        for reason in gate["rejection_reasons"]
    )
