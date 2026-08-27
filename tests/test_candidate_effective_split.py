from __future__ import annotations

from typing import Any

import numpy as np

from lumina_quant.strategy_factory import research_runner


def _metrics(value: float) -> dict[str, float]:
    return {
        "total_return": value,
        "return": value,
        "sharpe": value,
        "deflated_sharpe": value,
        "pbo": 0.1,
        "turnover": 0.1,
        "mdd": 0.01,
        "max_drawdown": 0.01,
        "trade_count": 10.0,
    }


def test_run_level_policy_overrides_candidate_split_downgrade() -> None:
    candidate = {
        "strategy_timeframe": "1h",
        "effective_split": {
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-01-10T00:00:00Z",
            "val_start": "2026-01-11T00:00:00Z",
            "val_end": "2026-01-20T00:00:00Z",
            "oos_start": "2026-01-21T00:00:00Z",
            "oos_end": "2026-01-31T00:00:00Z",
            "use_lockbox_split": False,
            "purge_embargo_bars": 0,
        },
    }
    resolved = research_runner._candidate_effective_split(
        candidate,
        {"use_lockbox_split": True, "purge_embargo_bars": 2},
        timeframe="1h",
    )

    assert resolved is not None
    assert resolved["use_lockbox_split"] is True
    assert resolved["purge_embargo_bars"] == 2


def test_evaluate_candidate_uses_candidate_effective_split(monkeypatch):
    captured: dict[str, Any] = {}
    candidate_split = {
        "train_start": "2026-05-15T00:00:00Z",
        "train_end": "2026-05-25T00:00:00Z",
        "val_start": "2026-05-25T00:00:00Z",
        "val_end": "2026-05-31T00:00:00Z",
        "oos_start": "2026-05-31T00:00:00Z",
        "oos_end": "2026-06-06T23:59:59.999000Z",
        "strategy_timeframe": "30m",
        "mode": "candidate_data_window",
    }
    candidate = {
        "candidate_id": "late-tradfi",
        "name": "late-tradfi",
        "strategy_class": "CompositeTrendStrategy",
        "strategy_timeframe": "30m",
        "symbols": ["ORCL/USDT"],
        "metadata": {"effective_split": candidate_split},
    }

    timestamps = np.array(
        [
            np.datetime64("2026-05-15T00:00:00", "ms"),
            np.datetime64("2026-05-26T00:00:00", "ms"),
            np.datetime64("2026-06-02T00:00:00", "ms"),
        ],
        dtype="datetime64[ms]",
    )

    monkeypatch.setattr(
        research_runner,
        "_load_candidate_signal_payload",
        lambda *args, **kwargs: research_runner._CandidateSignalPayload(
            symbols=["ORCL/USDT"],
            timeframe="30m",
            timestamps=timestamps,
            returns_raw=np.array([0.01, 0.02, 0.03], dtype=float),
            returns=np.array([0.01, 0.02, 0.03], dtype=float),
            turnover=np.array([0.1, 0.1, 0.1], dtype=float),
            exposure=np.array([1.0, 1.0, 1.0], dtype=float),
            meta={},
            cost_rate=0.0005,
        ),
    )

    def _metric_payload(signal_payload, *, benchmark_cache, candidate_count, split):
        captured["split"] = dict(split or {})
        return research_runner._CandidateMetricPayload(
            train_metrics=_metrics(0.1),
            val_metrics=_metrics(0.2),
            oos_metrics=_metrics(0.3),
            oos_stress_x2={"sharpe": 0.2, "return": 0.2},
            oos_stress_x3={"sharpe": 0.1, "return": 0.1},
        )

    monkeypatch.setattr(research_runner, "_evaluate_candidate_metric_payload", _metric_payload)
    monkeypatch.setattr(
        research_runner,
        "_evaluate_candidate_hurdles",
        lambda *args, **kwargs: ({}, True, {}),
    )

    result = research_runner._evaluate_candidate(
        candidate,
        cache={},
        feature_cache=None,
        aligned_cache={},
        benchmark_cache={},
        candidate_count=1,
        split={
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-04-01T00:00:00Z",
            "val_start": "2026-04-01T00:00:00Z",
            "val_end": "2026-05-01T00:00:00Z",
            "oos_start": "2026-05-01T00:00:00Z",
            "oos_end": "2026-06-06T23:59:59.999000Z",
            "strategy_timeframe": "30m",
        },
    )

    assert captured["split"]["mode"] == "candidate_data_window"
    assert captured["split"]["train_start"] == "2026-05-15T00:00:00Z"
    assert result["metadata"]["effective_split"]["train_start"] == "2026-05-15T00:00:00Z"


def test_insufficient_candidate_result_keeps_candidate_effective_split(monkeypatch):
    candidate_split = {
        "train_start": "2026-04-06T00:00:00Z",
        "train_end": "2026-05-13T04:47:59.999400Z",
        "val_start": "2026-05-13T04:47:59.999400Z",
        "val_end": "2026-05-25T14:23:59.999200Z",
        "oos_start": "2026-05-25T14:23:59.999200Z",
        "oos_end": "2026-06-06T23:59:59.999000Z",
        "strategy_timeframe": "1d",
        "mode": "candidate_data_window",
    }
    candidate = {
        "candidate_id": "late-tradfi-short-window",
        "name": "late-tradfi-short-window",
        "strategy_class": "AbnormalReturnContinuationStrategy",
        "strategy_timeframe": "1d",
        "symbols": ["AAPL/USDT"],
        "effective_split": candidate_split,
    }

    monkeypatch.setattr(
        research_runner,
        "_load_candidate_signal_payload",
        lambda *args, **kwargs: None,
    )

    result = research_runner._evaluate_candidate(
        candidate,
        cache={},
        feature_cache=None,
        aligned_cache={},
        benchmark_cache={},
        candidate_count=1,
        split={
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-04-01T00:00:00Z",
            "val_start": "2026-04-01T00:00:00Z",
            "val_end": "2026-05-01T00:00:00Z",
            "oos_start": "2026-05-01T00:00:00Z",
            "oos_end": "2026-06-06T23:59:59.999000Z",
            "strategy_timeframe": "1d",
        },
    )

    assert result["error"] == "insufficient_data"
    assert result["metadata"]["effective_split"]["mode"] == "candidate_data_window"
    assert result["metadata"]["effective_split"]["train_start"] == "2026-04-06T00:00:00Z"


def test_report_builder_return_streams_use_candidate_effective_split():
    builder = research_runner._research_report_builder()
    timestamps = np.array(
        [np.datetime64(f"2026-01-{day:02d}T00:00:00", "ms") for day in range(1, 11)],
        dtype="datetime64[ms]",
    )
    result = {
        "candidate": {
            "candidate_id": "demo",
            "name": "demo",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1d",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
        "timestamps": timestamps,
        "returns": np.arange(10, dtype=float),
        "train": _metrics(0.1),
        "val": _metrics(0.2),
        "oos": _metrics(0.3),
        "hurdle_fields": {},
        "oos_cost_stress": {},
        "hard_reject_reasons": {},
        "pass": True,
        "metadata": {
            "cost_rate": 0.0005,
            "effective_split": {
                "train_start": "2026-01-05T00:00:00Z",
                "train_end": "2026-01-06T00:00:00Z",
                "val_start": "2026-01-07T00:00:00Z",
                "val_end": "2026-01-08T00:00:00Z",
                "oos_start": "2026-01-09T00:00:00Z",
                "oos_end": "2026-01-10T23:59:59.999000Z",
                "strategy_timeframe": "1d",
                "mode": "candidate_data_window",
            },
        },
    }

    payload = builder.successful_candidate_report_payload(
        result=result,
        resolved_split={
            "train_start": "2026-01-01T00:00:00Z",
            "train_end": "2026-01-03T00:00:00Z",
            "val_start": "2026-01-04T00:00:00Z",
            "val_end": "2026-01-06T00:00:00Z",
            "oos_start": "2026-01-07T00:00:00Z",
            "oos_end": "2026-01-10T23:59:59.999000Z",
            "strategy_timeframe": "1d",
        },
        resolved_scoring_config={},
    )

    assert [point["t"] for point in payload["return_streams"]["train"]] == [
        int(np.datetime64("2026-01-05T00:00:00", "ms").astype(np.int64)),
        int(np.datetime64("2026-01-06T00:00:00", "ms").astype(np.int64)),
    ]
    assert payload["return_streams"]["oos"][0]["v"] == 8.0
    assert payload["effective_split"]["mode"] == "candidate_data_window"
