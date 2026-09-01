from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from scripts.research import backtest_dacapogo_daily_v2 as model


def candle(day: str, market: str = "AUSDT", close: float = 101.0, value: float = 1_000.0):
    return {
        "date": day,
        "market": market,
        "open": 100.0,
        "high": max(100.0, close) + 1,
        "low": min(100.0, close) - 1,
        "close": close,
        "value": value,
    }


def example(target: str, market: str, x: float, gross: float | None):
    day = date.fromisoformat(target)
    return {
        "date": target,
        "target_date": target,
        "decision_date": (day - timedelta(days=1)).isoformat(),
        "source_date": (day - timedelta(days=2)).isoformat(),
        "market": market,
        "x": [x] * len(model.FEATURE_NAMES),
        "gross_return": gross,
        "open": 100.0 if gross is not None else None,
        "close": 100.0 * (1 + gross) if gross is not None else None,
    }


def test_features_are_prior_only_and_require_adjacent_close():
    rows = [candle(f"2026-01-{day:02d}", close=100 + day, value=day * 100) for day in range(1, 7)]
    before = next(row for row in model.build_examples(rows) if row["date"] == "2026-01-05")
    changed = [dict(row) for row in rows]
    changed[3].update(open=1.0, high=1_000.0, low=0.5, close=900.0, value=1e12)
    changed[4].update(open=2.0, high=2_000.0, low=1.0, close=1_800.0, value=2e12)
    after = next(row for row in model.build_examples(changed) if row["date"] == "2026-01-05")
    assert before["source_date"] == "2026-01-03"
    assert before["decision_date"] == "2026-01-04"
    assert np.allclose(before["x"], after["x"])
    gapped = [candle("2026-01-01"), candle("2026-01-03"), candle("2026-01-04")]
    assert all(row["source_date"] != "2026-01-03" for row in model.build_examples(gapped))


def test_ols_uses_training_preprocessing_only():
    x = np.array([[0.0] * 5, [2.0, 4.0, 6.0, 8.0, 10.0]])
    fitted = model.fit_ols(x, [0.0, 1.0])
    assert np.allclose(fitted.mean, x.mean(axis=0))
    assert np.allclose(fitted.scale, x.std(axis=0))


def test_walk_forward_maturity_and_fixed_cash_slots(monkeypatch: pytest.MonkeyPatch):
    class Score:
        def predict(self, x):
            values = np.asarray(x, dtype=float)
            return values[:, 0] if values.ndim == 2 else values[0]

    monkeypatch.setattr(model, "fit_ols", lambda x, y: Score())
    rows = [
        example("2026-01-02", "TRAIN", 0.0, 0.0),
        example("2026-01-03", "IMMATURE", 0.0, 0.9),
        example("2026-01-04", "A", 1.0, 0.10),
        example("2026-01-04", "B", -1.0, 0.10),
        example("2026-01-04", "C", 1.0, None),
    ]
    days, trades = model.walk_forward(rows, "2026-01-04", "2026-01-04", cost=0.0)
    assert days[0]["eligible_slots"] == 3 and days[0]["selected_slots"] == 2
    assert days[0]["filled_slots"] == 1 and days[0]["missing_trade_slots"] == 1
    assert np.isclose(days[0]["ml_1x"], 0.10 / 3)
    assert len(trades) == 1 and trades[0]["slot_weight"] == 1 / 3


def test_trade_prediction_excludes_label_maturing_on_decision_date(
    monkeypatch: pytest.MonkeyPatch,
):
    class MeanLabel:
        def __init__(self, labels):
            self.prediction = float(np.mean(labels))

        def predict(self, x):
            values = np.asarray(x)
            return np.full(len(values), self.prediction) if values.ndim == 2 else self.prediction

    monkeypatch.setattr(model, "fit_ols", lambda x, y: MeanLabel(y))
    rows = [
        example("2026-01-02", "TRAIN", 0.0, 0.10),
        example("2026-01-03", "IMMATURE", 0.0, 0.20),
        example("2026-01-04", "TRADE", 1.0, 0.05),
    ]

    def prediction(values):
        return model.walk_forward(values, "2026-01-04", "2026-01-04")[1][0][
            "predicted_gross_return"
        ]

    first = prediction(rows)
    changed = [dict(row) for row in rows]
    changed[1]["gross_return"] = 0.90
    assert prediction(changed) == first
    changed[0]["gross_return"] = 0.30
    assert prediction(changed) != first


def test_walk_forward_is_deterministic():
    rows = [
        example("2026-01-02", "TRAIN", -1.0, -0.02),
        example("2026-01-04", "A", 1.0, 0.03),
        example("2026-01-04", "B", 0.5, 0.04),
    ]
    assert model.walk_forward(rows, "2026-01-04", "2026-01-04") == model.walk_forward(
        list(reversed(rows)), "2026-01-04", "2026-01-04"
    )


def test_bisected_training_matches_source_style_daily_scan(monkeypatch: pytest.MonkeyPatch):
    rows = [
        example("2026-01-02", "TRAIN-A", -1.0, -0.02),
        example("2026-01-03", "TRAIN-B", 0.0, 0.01),
        example("2026-01-04", "A", 1.0, 0.03),
        example("2026-01-04", "B", 0.5, 0.04),
        example("2026-01-05", "C", 0.2, 0.02),
    ]
    optimized = model.walk_forward(rows, "2026-01-04", "2026-01-05")
    monkeypatch.setattr(
        model,
        "bisect_left",
        lambda dates, decision: sum(day < decision for day in dates),
    )
    assert model.walk_forward(rows, "2026-01-04", "2026-01-05") == optimized


def test_seen_through_requires_iso_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        model,
        "_load_panel_snapshot",
        lambda *_args, **_kwargs: (
            [{**candle("2026-08-07"), "date": "2026-08-07"}],
            {},
            {},
        ),
    )
    args = model.build_arg_parser().parse_args(
        [
            "--panel-cache",
            str(tmp_path / "panel"),
            "--panel-manifest",
            str(tmp_path / "manifest"),
            "--output-dir",
            str(tmp_path),
            "--seen-through",
            "not-a-date",
        ]
    )
    with pytest.raises(ValueError, match="ISO date"):
        model.run(args)


def test_complete_week_gate_inputs_ignore_partial_weeks():
    start = date(2026, 4, 5)  # Sunday partial boundary, then one complete week.
    rows = []
    for offset in range(9):
        day = start + timedelta(days=offset)
        rows.append({"date": day.isoformat(), "ml_2x": 0.01 if day.weekday() == 0 else 0.0})
    median, weeks = model._weekly_median(rows)
    assert len(weeks) == 1 and median > 0


def test_input_identity_detects_tampering(tmp_path: Path):
    path = tmp_path / "panel.parquet"
    path.write_bytes(b"sealed")
    manifest = tmp_path / "panel.manifest.json"
    manifest.write_text(json.dumps({"file": model._file_identity(path), "audits": []}))
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity manifest"):
        model.load_panel(path, manifest)


def test_atomic_json_allows_concurrent_writers(tmp_path: Path):
    path = tmp_path / "result.json"
    values = [{"writer": 1}, {"writer": 2}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda value: model._atomic_json(path, value), values))
    assert json.loads(path.read_text()) in values


def test_output_manifest_seals_expected_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = [example("2026-01-02", "TRAIN", 0.0, 0.01)]
    daily = [
        {
            "date": "2026-04-01",
            "decision_date": "2026-03-31",
            "eligible_slots": 1,
            "baseline_filled_slots": 1,
            "selected_slots": 0,
            "filled_slots": 0,
            "missing_trade_slots": 0,
            "baseline_exposure": 1.0,
            "ml_exposure": 0.0,
            "cash_exposure": 0.0,
            "baseline_1x": 0.0,
            "baseline_2x": 0.0,
            "ml_1x": 0.0,
            "ml_2x": 0.0,
            "cash": 0.0,
        }
    ] * 112
    for offset, row in enumerate(daily):
        row["date"] = (date(2026, 4, 1) + timedelta(days=offset)).isoformat()
        row["decision_date"] = (date.fromisoformat(row["date"]) - timedelta(days=1)).isoformat()
    panel = tmp_path / "panel.parquet"
    panel.write_bytes(b"panel")
    manifest_in = tmp_path / "panel.manifest.json"
    manifest_in.write_bytes(b"manifest")
    already_locked: list[bool] = []

    def panel_snapshot(*_args, **kwargs):
        already_locked.append(bool(kwargs.get("already_locked")))
        return (
            [{**candle("2026-08-07"), "date": "2026-08-07"}],
            {},
            {
                panel.name: model._file_identity(panel),
                manifest_in.name: model._file_identity(manifest_in),
            },
        )

    monkeypatch.setattr(
        model,
        "_load_panel_snapshot",
        panel_snapshot,
    )
    monkeypatch.setattr(model, "build_examples", lambda _: rows)
    monkeypatch.setattr(model, "walk_forward", lambda *_: (daily, []))
    monkeypatch.setattr(
        model, "fit_ols", lambda *_: model.OLSModel(np.zeros(5), np.ones(5), np.zeros(6))
    )
    args = model.build_arg_parser().parse_args(
        [
            "--panel-cache",
            str(panel),
            "--panel-manifest",
            str(manifest_in),
            "--output-dir",
            str(tmp_path),
        ]
    )
    model.run(args)
    assert already_locked == [True]
    summary = json.loads((tmp_path / "dacapogo_binance_daily_v2_summary.json").read_text())
    assert summary["strategy_tier"] == "research_only"
    assert summary["promotion_eligible"] is False and summary["deploy_action"] == "cash"
    assert summary["parity"]["research_replay_action"] == "cash"
    assert "locked_action" not in summary["parity"]
    assert "deployed_result" not in summary["parity"]
    assert (
        summary["runtime"]["provenance"]["source_files"][str(model.Path(model.__file__).resolve())][
            "sha256"
        ]
        == hashlib.sha256(model.Path(model.__file__).read_bytes()).hexdigest()
    )
    lock = json.loads((tmp_path / "dacapogo_binance_daily_v2_lock.json").read_text())
    assert lock["promotion_eligible"] is False and lock["deploy_action"] == "cash"
    assert lock["research_replay_action"] == "cash"
    manifest = json.loads((tmp_path / "dacapogo_binance_daily_v2_manifest.json").read_text())
    assert set(manifest["files"]) == {
        "dacapogo_binance_daily_v2_lock.json",
        "dacapogo_binance_daily_v2_daily.csv",
        "dacapogo_binance_daily_v2_trades.csv",
        "dacapogo_binance_daily_v2_summary.json",
    }
    for name, identity in manifest["files"].items():
        target = tmp_path / name
        assert target.stat().st_size == identity["bytes"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == identity["sha256"]
