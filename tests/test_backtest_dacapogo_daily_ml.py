from __future__ import annotations

import fcntl
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts.research import backtest_dacapogo_daily_ml as ml


def _panel(days: int = 12) -> pl.DataFrame:
    start = date(2025, 1, 1)
    return pl.DataFrame(
        {
            "market": ["BTCUSDT"] * days,
            "date": [start + timedelta(days=i) for i in range(days)],
            "open": [100.0 + i for i in range(days)],
            "high": [102.0 + i for i in range(days)],
            "low": [99.0 + i for i in range(days)],
            "close": [101.0 + i for i in range(days)],
            "value": [1_000.0 + 10 * i for i in range(days)],
        }
    )


def test_current_day_mutation_does_not_change_features() -> None:
    panel = _panel()
    before = ml.build_daily_features(panel).row(-1, named=True)
    mutated = panel.with_row_index().with_columns(
        pl.when(pl.col("index") == panel.height - 1)
        .then(pl.lit(9_999_999.0))
        .otherwise(pl.col(column))
        .alias(column)
        for column in ("open", "high", "low", "close", "value")
    )
    after = ml.build_daily_features(mutated).row(-1, named=True)
    assert {name: before[name] for name in ml.FEATURES} == {
        name: after[name] for name in ml.FEATURES
    }
    assert before["momentum_3"] == panel[-2, "close"] / panel[-4, "close"] - 1
    assert before["momentum_7"] == panel[-2, "close"] / panel[-8, "close"] - 1
    zero_base = panel.with_row_index().with_columns(
        pl.when(pl.col("index") == 0).then(0.0).otherwise(pl.col("value")).alias("value")
    )
    assert ml.build_daily_features(zero_base)[2, "value_change"] == 0.0


def test_panel_cache_rejects_gaps_and_truncated_symbol_history() -> None:
    btc = _panel()
    panel = pl.concat([btc, btc.with_columns(pl.lit("ETHUSDT").alias("market"))])
    audits = [
        {
            "symbol": symbol,
            "requested_start": str(btc[0, "date"]),
            "start": str(btc[0, "date"]),
            "end": str(btc[-1, "date"]),
            "days": btc.height,
        }
        for symbol in ("BTCUSDT", "ETHUSDT")
    ]
    ml._validate_panel(panel, ("BTCUSDT", "ETHUSDT"), btc[0, "date"], btc[-1, "date"], audits)
    with pytest.raises(ValueError, match="date gaps"):
        ml._validate_panel(
            panel.filter(~((pl.col("market") == "ETHUSDT") & (pl.col("date") == btc[5, "date"]))),
            ("BTCUSDT", "ETHUSDT"),
            btc[0, "date"],
            btc[-1, "date"],
            audits,
        )
    with pytest.raises(ValueError, match="per-symbol coverage"):
        ml._validate_panel(
            panel.filter(~((pl.col("market") == "ETHUSDT") & (pl.col("date") == btc[0, "date"]))),
            ("BTCUSDT", "ETHUSDT"),
            btc[0, "date"],
            btc[-1, "date"],
            audits,
        )


def test_source_artifact_integrity_is_required(tmp_path) -> None:
    trades_path = tmp_path / "trades.csv"
    execution_path = tmp_path / "execution_trades.csv"
    panel_path = tmp_path / "daily_panel.parquet"
    manifest_path = tmp_path / "daily_panel.parquet.manifest.json"
    pl.DataFrame({"market": ["BTCUSDT"], "date": [date(2025, 1, 2)]}).write_csv(trades_path)
    pl.DataFrame(
        {
            "market": ["BTCUSDT"],
            "date": [date(2025, 1, 2)],
            "scenario": [ml.TARGET_SCENARIO],
            "leverage": [1],
            "entry_time": [datetime(2025, 1, 2, 1)],
            "slot_return": [0.01],
            "liquidated": [False],
            "mark_liquidation_breach": [False],
            "ambiguous_minute": [False],
        }
    ).write_csv(execution_path)
    _panel().write_parquet(panel_path)
    manifest_path.write_text(
        json.dumps({"file": ml._file_identity(panel_path), "audits": []}), encoding="utf-8"
    )
    summary = {
        "source": {"file_sha256": ml.SOURCE_FILE_SHA256},
        "rules": {"topk": ml.TOPK, "round_trip_cost": ml.COST},
        "data": {
            "symbols": ["BTCUSDT"],
            "start": "2025-01-02",
            "end": "2025-01-02",
        },
        "execution": {
            "scenarios": {ml.TARGET_SCENARIO: "test"},
            "leverages": [1],
            "audited_trigger_symbol_days": 1,
        },
        "artifacts": {
            "trades.csv": ml._file_identity(trades_path),
            "execution_trades.csv": ml._file_identity(execution_path),
            "daily_panel.parquet": ml._file_identity(panel_path),
            "daily_panel.parquet.manifest.json": ml._file_identity(manifest_path),
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    ml._read_inputs(tmp_path)
    manifest_text = manifest_path.read_text()
    manifest_path.write_text(manifest_text + "\n")
    with pytest.raises(ValueError, match="integrity metadata"):
        ml._read_inputs(tmp_path)
    manifest_path.write_text(manifest_text)
    execution_path.write_text(execution_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity metadata"):
        ml._read_inputs(tmp_path)


def test_source_reader_uses_shared_publication_lock(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(ml.fcntl, "flock", lambda _handle, mode: calls.append(mode))
    expected = ({"sealed": True}, pl.DataFrame(), {})
    monkeypatch.setattr(ml, "_read_inputs_unlocked", lambda _source: expected)
    assert ml._read_inputs(tmp_path) == expected
    assert calls == [fcntl.LOCK_SH]


def test_true_forward_date_is_after_data_and_generation_day() -> None:
    assert ml._true_forward_start(date(2026, 8, 11), date(2026, 8, 12)) == date(2026, 8, 13)


def test_ridge_uses_train_only_standardization_and_prediction() -> None:
    train_x = np.array([[1.0], [2.0], [3.0]])
    train_y = np.array([2.0, 4.0, 6.0])
    predicted = ml.ridge_predict(train_x, train_y, np.array([[4.0], [4000.0]]), 0.0)
    np.testing.assert_allclose(predicted, [8.0, 8000.0], atol=1e-10)
    assert ml.ridge_predict(train_x, train_y, np.array([[4.0]]), 1.0)[0] < 8.0
    tiny_x = np.array([[0.0], [1e-13], [2e-13]])
    tiny_y = np.array([1.0, 4.0, 9.0])
    mean, scale = tiny_x.mean(axis=0), tiny_x.std(axis=0)
    z = (tiny_x - mean) / scale
    expected = tiny_y.mean() + ((np.array([[3e-13]]) - mean) / scale) @ np.linalg.solve(
        z.T @ z + np.eye(1), z.T @ (tiny_y - tiny_y.mean())
    )
    np.testing.assert_allclose(ml.ridge_predict(tiny_x, tiny_y, np.array([[3e-13]]), 1.0), expected)


def test_select_ranks_prediction_before_market_and_ignores_entry_time() -> None:
    day = date(2025, 1, 2)
    rows = pl.DataFrame(
        {
            "market": ["CUSDT", "BUSDT", "AUSDT"],
            "date": [day] * 3,
            "entry_time": [
                datetime(2025, 1, 2, 0, 1),
                datetime(2025, 1, 2, 0, 3),
                datetime(2025, 1, 2, 0, 2),
            ],
        }
    )
    assert ml._select(rows, np.array([0.1, 0.5, 0.5]), 2) == {
        ("AUSDT", day),
        ("BUSDT", day),
    }


def test_fold_excludes_oos_and_ineligible_research_best_locks_cash(monkeypatch) -> None:
    folds = ml.make_folds(date(2025, 1, 1), date(2025, 8, 31))
    assert folds[0].train_end < folds[0].validation_start
    assert folds[0].validation_end < folds[0].embargo < folds[0].oos_start
    assert (folds[0].oos_start - folds[0].train_start).days == 211

    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(242)]
    rows = pl.DataFrame(
        {
            "market": ["BTCUSDT"] * len(days),
            "date": days,
            "entry_time": [datetime.combine(day, datetime.min.time()) for day in days],
            "target": [1.0] * len(days),
            **{feature: [float(i + 1) for i in range(len(days))] for feature in ml.FEATURES},
        }
    )
    execution = pl.DataFrame(
        {
            "market": ["BTCUSDT", "BTCUSDT"] * len(days),
            "date": [day for day in days for _ in range(2)],
            "scenario": list(ml.GATE_SCENARIOS) * len(days),
            "leverage": [1, 1] * len(days),
            "slot_return": [0.02, -0.01] * len(days),
        }
    )
    seen_train_max: list[float] = []
    original = ml.ridge_predict

    def recording_ridge(train_x, train_y, predict_x, alpha):
        seen_train_max.append(float(train_x[:, 0].max()))
        predictions = original(train_x, train_y, predict_x, alpha)
        return predictions if alpha == 1.0 else -np.ones(len(predict_x))

    monkeypatch.setattr(ml, "ridge_predict", recording_ridge)
    selections, _, locked = ml.evaluate_folds(rows, execution, folds[:1])
    assert selections and {row["research_replay_action"] for row in selections} == {"cash"}
    assert not any(row.get("promotion_eligible", False) for row in selections)
    assert {row["alpha"] for row in selections} == {1.0}
    assert not locked
    oos_first_feature = float(rows.filter(pl.col("date") == folds[0].oos_start)[ml.FEATURES[0]][0])
    assert max(seen_train_max) < oos_first_feature
