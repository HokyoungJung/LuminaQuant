from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_saved_incumbent_portfolio.py"
SPEC = importlib.util.spec_from_file_location("validate_saved_incumbent_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_validation_split_uses_canonical_boundaries() -> None:
    split = MODULE.build_validation_split(MODULE.parse_utc("2026-03-17T23:59:59Z"))
    assert split["train_start"] == "2025-01-01T00:00:00Z"
    assert split["val_start"] == "2026-01-01T00:00:00Z"
    assert split["oos_start"] == "2026-02-01T00:00:00Z"
    assert split["oos_end"] == "2026-03-17T23:59:59Z"


def test_evaluate_saved_weight_portfolio_separates_weighted_component_summaries() -> None:
    rows = [
        {
            "name": "a",
            "strategy_class": "A",
            "strategy_timeframe": "1h",
            "candidate_id": "a",
            "_saved_weight": 0.6,
            "oos": {"total_return": 0.1, "sharpe": 1.0, "max_drawdown": 0.05, "turnover": 0.2},
            "train": {"trade_count": 1, "turnover": 0.1, "benchmark_corr": 0.2},
            "val": {"trade_count": 1, "turnover": 0.1, "benchmark_corr": 0.2},
            "return_streams": {
                "train": [{"datetime": "2026-01-01T00:00:00Z", "v": 0.01}],
                "val": [{"datetime": "2026-02-01T00:00:00Z", "v": 0.02}],
                "oos": [
                    {"datetime": "2026-03-01T00:00:00Z", "v": 0.03},
                    {"datetime": "2026-03-02T00:00:00Z", "v": -0.01},
                ],
            },
            "metadata": {"cost_rate": 0.0005},
        },
        {
            "name": "b",
            "strategy_class": "B",
            "strategy_timeframe": "30m",
            "candidate_id": "b",
            "_saved_weight": 0.4,
            "oos": {"total_return": 0.05, "sharpe": 0.5, "max_drawdown": 0.02, "turnover": 0.1},
            "train": {"trade_count": 2, "turnover": 0.2, "benchmark_corr": 0.1},
            "val": {"trade_count": 2, "turnover": 0.2, "benchmark_corr": 0.1},
            "return_streams": {
                "train": [{"datetime": "2026-01-01T00:00:00Z", "v": 0.02}],
                "val": [{"datetime": "2026-02-01T00:00:00Z", "v": 0.01}],
                "oos": [
                    {"datetime": "2026-03-01T00:00:00Z", "v": 0.01},
                    {"datetime": "2026-03-02T00:00:00Z", "v": 0.00},
                ],
            },
            "metadata": {"cost_rate": 0.001},
        },
    ]

    payload = MODULE.evaluate_saved_weight_portfolio(rows)
    oos = payload["portfolio_metrics"]["oos"]
    weighted = payload["weighted_component_summaries"]["oos"]
    assert payload["component_rows"][0]["saved_weight"] == 0.6
    assert len(payload["oos_monthly_returns"]) == 1
    assert oos["total_return"] != 0.0
    assert "turnover" not in oos
    assert weighted["turnover"] > 0.0


def test_latest_common_complete_time_uses_timeframe_complete_buckets() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-03-19T09:30:29Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-03-19T09:30:29Z"},
        ],
        "feature_results": [
            {"symbol": "BTC/USDT", "last_timestamp_utc": "2026-03-19T08:59:00Z"},
        ],
    }

    anchored_end, evidence = MODULE._latest_common_complete_time(
        refresh_payload=refresh_payload,
        required_pairs=[("BTC/USDT", "1m"), ("ETH/USDT", "4h")],
        feature_symbols=["BTC/USDT"],
    )

    assert MODULE.iso_utc(anchored_end) == "2026-03-19T04:00:00Z"
    assert any(item["symbol"] == "ETH/USDT" and item["timeframe"] == "4h" for item in evidence)


def test_latest_common_complete_time_accepts_refresh_symbol_aliases() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "XAU/USD", "after_ohlcv_max_utc": "2026-03-19T09:30:29Z"},
        ],
        "feature_results": [
            {"symbol": "XAU/USD", "last_timestamp_utc": "2026-03-19T09:00:00Z"},
        ],
    }

    anchored_end, evidence = MODULE._latest_common_complete_time(
        refresh_payload=refresh_payload,
        required_pairs=[("XAU/USDT", "1m")],
        feature_symbols=["XAU/USDT"],
    )

    assert MODULE.iso_utc(anchored_end) == "2026-03-19T09:00:00Z"
    assert evidence[0]["symbol"] == "XAU/USDT"


def test_latest_common_complete_time_accepts_latest_universe_results_schema() -> None:
    refresh_payload = {
        "results": [
            {"symbol": "BNB/USDT", "after_ohlcv_max_utc": "2026-03-28T14:33:47Z"},
        ],
    }

    anchored_end, evidence = MODULE._latest_common_complete_time(
        refresh_payload=refresh_payload,
        required_pairs=[("BNB/USDT", "1h")],
        feature_symbols=[],
    )

    assert MODULE.iso_utc(anchored_end) == "2026-03-28T13:00:00Z"
    assert evidence[0]["symbol"] == "BNB/USDT"


def test_latest_common_complete_time_uses_support_inventory_when_feature_results_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    support_path = tmp_path / "support_inventory.json"
    support_path.write_text(
        """
        {
          "symbols": [
            {"symbol": "BNBUSDT", "last_timestamp_utc": "2026-03-28T14:33:00+00:00"}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "DEFAULT_SUPPORT_INVENTORY_JSON", support_path)
    refresh_payload = {
        "results": [
            {"symbol": "BNB/USDT", "after_ohlcv_max_utc": "2026-03-28T14:33:47Z"},
        ],
    }

    anchored_end, evidence = MODULE._latest_common_complete_time(
        refresh_payload=refresh_payload,
        required_pairs=[("BNB/USDT", "1h")],
        feature_symbols=["BNB/USDT"],
    )

    assert MODULE.iso_utc(anchored_end) == "2026-03-28T13:00:00Z"
    assert any(item["source"] == "support_inventory" for item in evidence)


def test_run_strict_research_rejects_synthetic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "run_candidate_research",
        lambda **_kwargs: {"data_sources": {"synthetic": ["BTC/USDT@1h"]}, "candidates": []},
    )

    with pytest.raises(RuntimeError, match="synthetic fallback"):
        MODULE._run_strict_research(
            candidates=[
                {"candidate_id": "c1", "symbols": ["BTC/USDT"], "strategy_timeframe": "1h"}
            ],
            strategy_timeframes=["1h"],
            symbol_universe=["BTC/USDT"],
            split={
                "train_start": "2025-01-01T00:00:00Z",
                "train_end": "2025-12-31T23:59:59Z",
                "val_start": "2026-01-01T00:00:00Z",
                "val_end": "2026-01-31T23:59:59Z",
                "oos_start": "2026-02-01T00:00:00Z",
                "oos_end": "2026-03-01T00:00:00Z",
            },
            min_bundle_bars=1,
        )


def test_run_strict_research_executes_candidates_sequentially_with_candidate_specific_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_candidate_research(**kwargs):
        calls.append(kwargs)
        candidate = kwargs["candidates"][0]
        return {
            "generated_at": "2026-03-28T00:00:00Z",
            "data_sources": {"raw-first": [candidate["name"]]},
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "name": candidate["name"],
                }
            ],
        }

    monkeypatch.setattr(MODULE, "run_candidate_research", _fake_run_candidate_research)
    monkeypatch.setattr(MODULE, "DEFAULT_STRICT_VALIDATION_CACHE_DIR", tmp_path)

    report = MODULE._run_strict_research(
        candidates=[
            {
                "candidate_id": "pair",
                "name": "pair",
                "strategy_timeframe": "1h",
                "symbols": ["BNB/USDT", "TRX/USDT"],
            },
            {
                "candidate_id": "trend",
                "name": "trend",
                "strategy_timeframe": "30m",
                "symbols": ["BTC/USDT", "ETH/USDT"],
            },
        ],
        strategy_timeframes=["1h", "30m"],
        symbol_universe=["BTC/USDT", "ETH/USDT", "BNB/USDT", "TRX/USDT"],
        split={
            "train_start": "2025-01-01T00:00:00Z",
            "train_end": "2025-12-31T23:59:59Z",
            "val_start": "2026-01-01T00:00:00Z",
            "val_end": "2026-01-31T23:59:59Z",
            "oos_start": "2026-02-01T00:00:00Z",
            "oos_end": "2026-03-01T00:00:00Z",
        },
        min_bundle_bars=1,
    )

    assert [call["max_candidates"] for call in calls] == [1, 1]
    assert [call["data_mode"] for call in calls] == [MODULE.STRICT_VALIDATION_DATA_MODE] * 2
    assert calls[0]["strategy_timeframes"] == ["1h"]
    assert calls[0]["symbol_universe"] == ["BNB/USDT", "TRX/USDT"]
    assert calls[1]["strategy_timeframes"] == ["30m"]
    assert calls[1]["symbol_universe"] == ["BTC/USDT", "ETH/USDT"]
    assert [row["candidate_id"] for row in report["candidates"]] == ["pair", "trend"]


def test_build_latest_anchored_split_trims_from_left_when_latest_anchor_moves_forward() -> None:
    saved_oos_end = MODULE.parse_utc("2026-03-17T23:59:59Z")
    latest_common_end = MODULE.parse_utc("2026-03-19T23:59:59Z")
    assert saved_oos_end is not None and latest_common_end is not None

    shifted = MODULE.build_latest_anchored_split(
        saved_oos_end=saved_oos_end,
        anchored_oos_end=latest_common_end,
    ).as_dict()

    assert shifted["train_start"] == "2025-01-03T00:00:00Z"
    assert shifted["val_start"] == "2026-01-03T00:00:00Z"
    assert shifted["oos_start"] == "2026-02-03T00:00:00Z"
    assert shifted["oos_end"] == "2026-03-19T23:59:59Z"


def test_run_strict_research_reuses_disk_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_candidate_research(**kwargs):
        calls.append(kwargs)
        candidate = kwargs["candidates"][0]
        return {
            "generated_at": "2026-03-28T00:00:00Z",
            "data_sources": {"raw-first": [candidate["name"]]},
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "name": candidate["name"],
                }
            ],
        }

    monkeypatch.setattr(MODULE, "run_candidate_research", _fake_run_candidate_research)
    monkeypatch.setattr(MODULE, "DEFAULT_STRICT_VALIDATION_CACHE_DIR", tmp_path)

    kwargs = dict(
        candidates=[
            {
                "candidate_id": "pair",
                "name": "pair",
                "strategy_timeframe": "1h",
                "symbols": ["BNB/USDT", "TRX/USDT"],
            }
        ],
        strategy_timeframes=["1h"],
        symbol_universe=["BNB/USDT", "TRX/USDT"],
        split={
            "train_start": "2025-01-01T00:00:00Z",
            "train_end": "2025-12-31T23:59:59Z",
            "val_start": "2026-01-01T00:00:00Z",
            "val_end": "2026-01-31T23:59:59Z",
            "oos_start": "2026-02-01T00:00:00Z",
            "oos_end": "2026-03-01T00:00:00Z",
        },
        min_bundle_bars=1,
    )

    first = MODULE._run_strict_research(**kwargs)
    second = MODULE._run_strict_research(**kwargs)

    assert len(calls) == 1
    assert first == second
    assert list(tmp_path.glob("*.json"))


def test_run_strict_research_does_not_reuse_cache_when_leverage_changes(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_candidate_research(**kwargs):
        calls.append(kwargs)
        candidate = kwargs["candidates"][0]
        return {
            "generated_at": "2026-03-28T00:00:00Z",
            "data_sources": {"raw-first": [candidate["name"]]},
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "name": candidate["name"],
                    "params": dict(candidate.get("params") or {}),
                    "leverage": candidate.get("leverage"),
                }
            ],
        }

    monkeypatch.setattr(MODULE, "run_candidate_research", _fake_run_candidate_research)
    monkeypatch.setattr(MODULE, "DEFAULT_STRICT_VALIDATION_CACHE_DIR", tmp_path)

    base_kwargs = dict(
        strategy_timeframes=["1h"],
        symbol_universe=["BNB/USDT", "TRX/USDT"],
        split={
            "train_start": "2025-01-01T00:00:00Z",
            "train_end": "2025-12-31T23:59:59Z",
            "val_start": "2026-01-01T00:00:00Z",
            "val_end": "2026-01-31T23:59:59Z",
            "oos_start": "2026-02-01T00:00:00Z",
            "oos_end": "2026-03-01T00:00:00Z",
        },
        min_bundle_bars=1,
    )

    first = MODULE._run_strict_research(
        candidates=[
            {
                "candidate_id": "pair",
                "name": "pair",
                "strategy_timeframe": "1h",
                "symbols": ["BNB/USDT", "TRX/USDT"],
                "strategy_class": "PairSpreadZScoreStrategy",
                "leverage": 1,
                "params": {"lookback_window": 96, "leverage": 1},
            }
        ],
        **base_kwargs,
    )
    second = MODULE._run_strict_research(
        candidates=[
            {
                "candidate_id": "pair",
                "name": "pair",
                "strategy_timeframe": "1h",
                "symbols": ["BNB/USDT", "TRX/USDT"],
                "strategy_class": "PairSpreadZScoreStrategy",
                "leverage": 3,
                "params": {"lookback_window": 96, "leverage": 3},
            }
        ],
        **base_kwargs,
    )

    assert len(calls) == 2
    assert first != second
