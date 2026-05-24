from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_pnl_correlation_decision.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_pnl_correlation_decision", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_id": "candidate-a",
        "source_artifact_kind": MODULE.SOURCE_KIND_DEBOUNCED_REPAIR,
        "monitoring_status": MODULE.PAPER_STATUS,
        "symbol": "SOLUSDT",
        "timeframe": "1h",
        "family": "debounced_momentum_hysteresis_efficiency_repair",
        "side": "short_only",
        "notional_fraction": 0.45,
        "train_return": 0.20,
        "validation_return": 0.08,
        "locked_oos_return": 0.03,
        "validation_mdd": 0.05,
        "train_trade_event_count": 120,
        "validation_trade_event_count": 40,
        "locked_oos_trade_event_count": 25,
        "train_return_per_turnover_proxy_bps": 25.0,
        "validation_return_per_turnover_proxy_bps": 35.0,
        "locked_oos_return_per_turnover_proxy_bps": 20.0,
        "paper_candidate_gate_pass": True,
        "primary_10bps_promotion_gate_pass": True,
        "ready_for_paper": True,
        "ready_for_real": False,
        "real_money_execution": False,
    }
    row.update(overrides)
    return row


def _capture(model_id: str, returns: list[float], datetimes: list[str]) -> object:
    return MODULE.CapturedPnl(
        model_id=model_id,
        source_artifact_kind=MODULE.SOURCE_KIND_DEBOUNCED_REPAIR,
        datetimes=tuple(pd.to_datetime(datetimes).tolist()),
        returns=np.asarray(returns, dtype=float),
        position=np.ones(len(returns), dtype=float),
        timeframe="1h",
    )


def test_train_validation_score_ignores_locked_oos_return_when_no_explicit_score() -> None:
    high_oos = _row(locked_oos_return=0.90, locked_oos_return_per_turnover_proxy_bps=200.0)
    low_oos = _row(locked_oos_return=-0.90, locked_oos_return_per_turnover_proxy_bps=-200.0)

    assert MODULE._monitoring_score_train_validation_only(high_oos) == MODULE._monitoring_score_train_validation_only(
        low_oos
    )


def test_aligned_pnl_frame_fills_missing_strategy_bars_with_zero() -> None:
    captures = {
        "a": _capture("a", [0.01, 0.02], ["2025-01-01 00:00:00", "2026-01-01 00:00:00"]),
        "b": _capture("b", [0.03], ["2026-01-01 01:00:00"]),
    }

    frame = MODULE._aligned_pnl_frame(captures, ["a", "b"], split="train_validation")

    assert list(frame.columns) == ["a", "b"]
    assert frame.loc[pd.Timestamp("2025-01-01 00:00:00"), "b"] == 0.0
    assert frame.loc[pd.Timestamp("2026-01-01 01:00:00"), "a"] == 0.0


def test_greedy_selection_rejects_high_train_validation_correlation_duplicate() -> None:
    ids = ["a", "b", "c"]
    train_validation_corr = pd.DataFrame(
        [[1.0, 0.91, 0.20], [0.91, 1.0, 0.10], [0.20, 0.10, 1.0]],
        index=ids,
        columns=ids,
    )
    validation_corr = pd.DataFrame(
        [[1.0, 0.60, 0.10], [0.60, 1.0, 0.20], [0.10, 0.20, 1.0]],
        index=ids,
        columns=ids,
    )
    rows = [
        _row(model_id="a", monitoring_score_train_validation_only=3.0),
        _row(model_id="b", monitoring_score_train_validation_only=2.0),
        _row(model_id="c", monitoring_score_train_validation_only=1.0),
    ]

    decisions = MODULE.greedy_correlation_selection(
        rows,
        train_validation_corr=train_validation_corr,
        validation_corr=validation_corr,
    )

    by_id = {row["model_id"]: row for row in decisions}
    assert by_id["a"]["correlation_decision"] == "selected_corr_diversified_paper_monitor"
    assert by_id["b"]["correlation_decision"] == "rejected_high_pnl_correlation_duplicate"
    assert by_id["c"]["correlation_decision"] == "selected_corr_diversified_paper_monitor"
    assert by_id["b"]["ready_for_real"] is False
    assert by_id["b"]["real_money_execution"] is False


def test_build_payload_keeps_locked_oos_report_only_and_real_money_disabled(monkeypatch: object, tmp_path: Path) -> None:
    rows = [
        _row(model_id="a", monitoring_score_train_validation_only=2.0),
        _row(model_id="b", monitoring_score_train_validation_only=1.0, symbol="ETHUSDT"),
    ]
    monitoring_payload = {
        "ready_for_real": False,
        "real_money_execution": False,
        "selection_policy": {"uses_locked_oos_for_discovery": False},
        "monitoring_rows": rows,
        "source_artifacts": [],
    }
    captures = {
        "a": _capture(
            "a",
            [0.01, 0.02, 0.01, 0.003],
            ["2025-01-01 00:00:00", "2026-01-01 00:00:00", "2026-01-02 00:00:00", "2026-04-01 00:00:00"],
        ),
        "b": _capture(
            "b",
            [0.00, -0.01, 0.02, 0.004],
            ["2025-01-01 00:00:00", "2026-01-01 00:00:00", "2026-01-02 00:00:00", "2026-04-01 00:00:00"],
        ),
    }

    def fake_capture(*_: object, **__: object) -> dict[str, object]:
        return captures

    monkeypatch.setattr(MODULE, "capture_pnl_series", fake_capture)

    payload = MODULE.build_payload_from_monitoring(
        monitoring_payload,
        output_dir=tmp_path,
        monitoring_artifact_path=tmp_path / "monitoring.json",
        data_root=tmp_path,
        feature_root=tmp_path,
        write_outputs=False,
    )

    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["selection_policy"]["uses_locked_oos_for_selection"] is False
    assert payload["correlation_decision_summary"]["captured_paper_pnl_candidate_count"] == 2
    assert payload["selected_corr_diversified_candidates"]
