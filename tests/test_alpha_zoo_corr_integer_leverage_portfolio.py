from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_corr_integer_leverage_portfolio.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_corr_integer_leverage_portfolio", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _dates() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
                "2026-04-01 00:00:00",
                "2026-04-01 01:00:00",
            ]
        )
    )


def _bars(close: list[float]) -> pd.DataFrame:
    idx = _dates()
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": [value * 1.001 for value in close],
            "low": [value * 0.999 for value in close],
            "close": close,
            "volume": [1.0] * len(close),
        }
    )


def _replay(
    *,
    model_id: str = "candidate-a",
    symbol: str = "SOLUSDT",
    allocation_fraction: float = 0.2,
    close: list[float] | None = None,
) -> object:
    frame = _bars(close or [100.0, 103.0, 103.0, 105.0, 105.0, 106.0])
    return MODULE.CandidateReplay(
        model_id=model_id,
        source_artifact_kind=MODULE.corr.SOURCE_KIND_DEBOUNCED_REPAIR,
        symbol=symbol,
        timeframe="1h",
        allocation_fraction=allocation_fraction,
        datetimes=pd.DatetimeIndex(frame["datetime"]),
        signal=np.ones(len(frame), dtype=float),
        close=frame["close"].to_numpy(dtype=float),
        high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float),
    )


def _selected_row(model_id: str, symbol: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "source_artifact_kind": MODULE.corr.SOURCE_KIND_DEBOUNCED_REPAIR,
        "correlation_decision": "selected_corr_diversified_paper_monitor",
        "selection_rank": 1,
        "symbol": symbol,
        "timeframe": "1h",
        "allocation_fraction": 0.2,
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _safe_correlation_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "ready_for_real": False,
        "real_money_execution": False,
        "selection_policy": {
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
        },
        "correlation_decision_rows": rows,
    }


def test_simulate_candidate_requires_positive_integer_leverage() -> None:
    replay = _replay(allocation_fraction=0.1)

    with pytest.raises(ValueError, match="positive integer"):
        MODULE.simulate_candidate_with_integer_leverage(replay, integer_leverage=0)

    sim = MODULE.simulate_candidate_with_integer_leverage(replay, integer_leverage=3)

    assert sim.integer_leverage == 3
    assert sim.notional_fraction == pytest.approx(0.3)
    assert sim.returns[0] == pytest.approx((0.3 * 0.03) - (0.001 * 0.3 / 2.0))
    assert not sim.liquidation_flags.any()
    assert not sim.account_wipeout_flags.any()


def test_train_validation_score_ignores_locked_oos_metrics() -> None:
    base = {
        "gross_notional_fraction": 1.0,
        "split_metrics": {
            "train": {"total_return": 0.12, "max_drawdown": 0.02, "return_per_turnover_proxy_bps": 30.0},
            "validation": {"total_return": 0.08, "max_drawdown": 0.03, "return_per_turnover_proxy_bps": 40.0},
            "locked_oos": {"total_return": -0.50, "max_drawdown": 0.80, "return_per_turnover_proxy_bps": -100.0},
        },
    }
    better_oos = {
        **base,
        "split_metrics": {
            **base["split_metrics"],
            "locked_oos": {"total_return": 0.90, "max_drawdown": 0.01, "return_per_turnover_proxy_bps": 300.0},
        },
    }

    assert MODULE._train_validation_score(base) == MODULE._train_validation_score(better_oos)


def test_search_integer_asset_leverage_profiles_emits_integer_asset_maps(monkeypatch: object) -> None:
    monkeypatch.setattr(MODULE, "MIN_TRAIN_TRADE_EVENTS", 1)
    monkeypatch.setattr(MODULE, "MIN_VALIDATION_TRADE_EVENTS", 1)
    monkeypatch.setattr(MODULE, "MIN_LOCKED_OOS_TRADE_EVENTS", 1)
    replays = [
        _replay(model_id="sol", symbol="SOLUSDT", allocation_fraction=0.2),
        _replay(model_id="eth", symbol="ETHUSDT", allocation_fraction=0.2),
    ]

    results = MODULE.search_integer_asset_leverage_profiles(replays, leverage_min=1, leverage_max=12)
    balanced = results["balanced_mdd12_gross5"]

    assert balanced["train_validation_rejection_reasons"] == []
    assert balanced["locked_oos_report_only_gate_reasons"] == []
    assert set(balanced["leverage_by_asset"]) == {"ETHUSDT", "SOLUSDT"}
    assert all(isinstance(value, int) for value in balanced["leverage_by_asset"].values())
    assert balanced["split_metrics"]["validation"]["total_return"] > 0.02


def test_build_payload_keeps_real_money_disabled_and_oos_report_only(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr(MODULE, "MIN_TRAIN_TRADE_EVENTS", 1)
    monkeypatch.setattr(MODULE, "MIN_VALIDATION_TRADE_EVENTS", 1)
    monkeypatch.setattr(MODULE, "MIN_LOCKED_OOS_TRADE_EVENTS", 1)
    rows = [_selected_row("sol", "SOLUSDT"), _selected_row("eth", "ETHUSDT")]
    correlation_payload = _safe_correlation_payload(rows)
    monitoring_payload = {"ready_for_real": False, "real_money_execution": False}
    dates = tuple(_dates().tolist())
    captures = {
        model_id: MODULE.corr.CapturedPnl(
            model_id=model_id,
            source_artifact_kind=MODULE.corr.SOURCE_KIND_DEBOUNCED_REPAIR,
            datetimes=dates,
            returns=np.zeros(len(dates), dtype=float),
            position=np.ones(len(dates), dtype=float),
            timeframe="1h",
        )
        for model_id in ("sol", "eth")
    }

    def fake_capture(*_: object, **__: object) -> dict[str, object]:
        return captures

    def fake_load_bars(selected_rows: object, **_: object) -> dict[tuple[str, str, str], pd.DataFrame]:
        del selected_rows
        return {
            (MODULE.corr.SOURCE_KIND_DEBOUNCED_REPAIR, "SOLUSDT", "1h"): _bars(
                [100.0, 103.0, 103.0, 105.0, 105.0, 106.0]
            ),
            (MODULE.corr.SOURCE_KIND_DEBOUNCED_REPAIR, "ETHUSDT", "1h"): _bars(
                [200.0, 206.0, 206.0, 210.0, 210.0, 212.0]
            ),
        }

    monkeypatch.setattr(MODULE.corr, "capture_pnl_series", fake_capture)
    monkeypatch.setattr(MODULE, "_load_bars_for_rows", fake_load_bars)

    payload = MODULE.build_payload_from_inputs(
        correlation_payload=correlation_payload,
        monitoring_payload=monitoring_payload,
        output_dir=tmp_path,
        correlation_artifact_path=tmp_path / "corr.json",
        monitoring_artifact_path=tmp_path / "monitoring.json",
        data_root=tmp_path,
        feature_root=tmp_path,
        write_outputs=False,
    )

    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["real_execution_allowed"] is False
    assert payload["selection_policy"]["uses_locked_oos_for_selection"] is False
    assert payload["selection_policy"]["uses_locked_oos_for_objective"] is False
    assert payload["selected_profile"]["ready_for_real"] is False
    assert payload["selected_profile"]["real_money_execution"] is False
