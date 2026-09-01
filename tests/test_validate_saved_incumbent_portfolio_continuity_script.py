from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_saved_incumbent_portfolio_continuity.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_saved_incumbent_portfolio_continuity", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio_continuity module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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
    monkeypatch: pytest.MonkeyPatch,
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


def test_run_strict_research_executes_candidates_sequentially_with_candidate_specific_scope(
    monkeypatch: pytest.MonkeyPatch,
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


def test_run_strict_research_reuses_disk_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
