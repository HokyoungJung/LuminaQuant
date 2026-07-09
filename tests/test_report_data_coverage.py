"""Tests for scripts/research/report_data_coverage.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "research" / "report_data_coverage.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("report_data_coverage", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-inspection can resolve the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rdc = _load_module()

_SCHEMA = {
    "datetime": pl.Datetime("ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def _write_monthly_1s(root: Path, *, exchange: str, token: str, month: str, bars: int) -> None:
    """Write a monthly 1s parquet in the canonical market_ohlcv_1s layout."""
    start = datetime(int(month[:4]), int(month[5:7]), 1)
    rows = [
        {
            "datetime": start + timedelta(seconds=i),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
        for i in range(bars)
    ]
    symbol_dir = root / "market_ohlcv_1s" / exchange.lower() / token
    symbol_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_SCHEMA).write_parquet(symbol_dir / f"{month}.parquet")


def _write_partitioned(
    root: Path, *, exchange: str, token: str, timeframe: str, day: str, bars: int
) -> None:
    """Write a partitioned timeframe parquet (exchange=/symbol=/timeframe=/date=)."""
    start = datetime.fromisoformat(day)
    step = timedelta(minutes=1 if timeframe == "1m" else 60)
    rows = [
        {
            "datetime": start + step * i,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
        for i in range(bars)
    ]
    part_dir = (
        root
        / f"exchange={exchange.lower()}"
        / f"symbol={token}"
        / f"timeframe={timeframe}"
        / f"date={day}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_SCHEMA).write_parquet(part_dir / "part-0.parquet")


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Path:
    root = tmp_path / "market_parquet"
    # BTC has a 1s base (2 months) + a partitioned 1m series.
    _write_monthly_1s(root, exchange="binance", token="BTCUSDT", month="2026-01", bars=600)
    _write_monthly_1s(root, exchange="binance", token="BTCUSDT", month="2026-02", bars=600)
    _write_partitioned(
        root, exchange="binance", token="BTCUSDT", timeframe="1m", day="2026-01-01", bars=500
    )
    # ETH has only a short partitioned 1h series (below the 360-bar floor).
    _write_partitioned(
        root, exchange="binance", token="ETHUSDT", timeframe="1h", day="2026-01-01", bars=50
    )
    return root


# --------------------------------------------------------------------------- #
# enumeration + scan-mode coverage
# --------------------------------------------------------------------------- #
def test_enumerate_symbols_discovers_layouts(synthetic_store: Path) -> None:
    inventories = rdc.enumerate_symbols(synthetic_store, "binance")
    assert set(inventories) == {"BTC/USDT", "ETH/USDT"}

    btc = inventories["BTC/USDT"]
    assert btc.monthly_1s_dir is not None
    assert "1m" in btc.partitioned_tfs
    assert btc.has_base() is True

    eth = inventories["ETH/USDT"]
    assert eth.monthly_1s_dir is None
    assert set(eth.partitioned_tfs) == {"1h"}
    assert eth.has_base() is False


def test_scan_coverage_rows_report_physical_footprint(synthetic_store: Path) -> None:
    inventories = rdc.enumerate_symbols(synthetic_store, "binance")
    rows = rdc.scan_coverage_rows(inventories, exchange="binance", timeframes=None, min_bars=360)
    index = {(row.symbol, row.timeframe): row for row in rows}

    # BTC: 1s base (1200 bars across 2 months) + 1m partitioned (500 bars).
    assert index[("BTC/USDT", "1s")].bar_count == 1200
    assert index[("BTC/USDT", "1s")].source == "monthly_1s"
    assert index[("BTC/USDT", "1m")].bar_count == 500
    assert index[("BTC/USDT", "1m")].sufficient is True

    # ETH: only a 50-bar 1h series -> below the floor.
    eth_1h = index[("ETH/USDT", "1h")]
    assert eth_1h.bar_count == 50
    assert eth_1h.sufficient is False
    assert eth_1h.first_ts == "2026-01-01T00:00:00Z"

    # Deterministic ordering.
    assert rows == sorted(rows, key=rdc.CoverageRow.sort_key)


def test_timeframe_filter_and_gap_ratio(synthetic_store: Path) -> None:
    inventories = rdc.enumerate_symbols(synthetic_store, "binance")
    rows = rdc.scan_coverage_rows(inventories, exchange="binance", timeframes=["1m"], min_bars=360)
    assert {row.timeframe for row in rows} == {"1m"}
    # 500 contiguous 1-minute bars -> gap ratio ~0.
    btc = next(row for row in rows if row.symbol == "BTC/USDT")
    assert btc.gap_ratio == 0.0


# --------------------------------------------------------------------------- #
# registry mode reuses the repo loader
# --------------------------------------------------------------------------- #
def test_registry_mode_resamples_via_repo(synthetic_store: Path) -> None:
    inventories = rdc.enumerate_symbols(synthetic_store, "binance")
    rows = rdc.registry_coverage_rows(
        synthetic_store,
        inventories,
        exchange="binance",
        timeframes=["1m", "5m"],
        min_bars=1,
    )
    index = {(row.symbol, row.timeframe): row for row in rows}
    # BTC 1s base (1200 s across two month-starts) resamples to a handful of
    # 1m/5m buckets via the repository's own load path.
    assert index[("BTC/USDT", "1m")].bar_count > 0
    assert index[("BTC/USDT", "1m")].source == "loader"
    # ETH has no 1s base -> the loader yields nothing.
    assert index[("ETH/USDT", "1m")].bar_count == 0
    assert index[("ETH/USDT", "1m")].sufficient is False


# --------------------------------------------------------------------------- #
# --check-manifest dry-run
# --------------------------------------------------------------------------- #
def _write_manifest(path: Path) -> None:
    payload = {
        "artifact_kind": "candidate_manifest",
        "candidates": [
            {"name": "btc_1h", "strategy_timeframe": "1h", "symbols": ["BTC/USDT"]},
            {"name": "eth_1h", "strategy_timeframe": "1h", "symbols": ["ETHUSDT"]},
            {
                "name": "pair_1h",
                "timeframe": "1h",
                "symbols": ["BTC/USDT", "DOGE/USDT"],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_manifest_scan_mode(synthetic_store: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    inventories = rdc.enumerate_symbols(synthetic_store, "binance")

    result = rdc.check_manifest(
        manifest,
        mode="scan",
        root=synthetic_store,
        exchange="binance",
        inventories=inventories,
        min_bars=360,
    )
    rows = {row["name"]: row for row in result["rows"]}

    # BTC@1h derivable from its 1s base -> present (scan presence-only).
    assert rows["btc_1h"]["evaluable"] is True
    assert rows["btc_1h"]["missing_symbols"] == []
    # ETH@1h physically present but only 50 bars (< 360) -> missing.
    assert rows["eth_1h"]["missing_symbols"] == ["ETH/USDT"]
    assert rows["eth_1h"]["evaluable"] is False
    # DOGE absent from the store entirely -> the pair row is not evaluable.
    assert rows["pair_1h"]["missing_symbols"] == ["DOGE/USDT"]
    assert result["candidate_count"] == 3
    assert result["evaluable_count"] == 1
    assert result["insufficient_count"] == 2


def test_check_manifest_registry_mode_counts_bars(synthetic_store: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    inventories = rdc.enumerate_symbols(synthetic_store, "binance")

    result = rdc.check_manifest(
        manifest,
        mode="registry",
        root=synthetic_store,
        exchange="binance",
        inventories=inventories,
        min_bars=1,
    )
    rows = {row["name"]: row for row in result["rows"]}
    # With min_bars=1 and a real 1s base, BTC@1h resamples to >=1 bar.
    assert rows["btc_1h"]["evaluable"] is True
    # ETH has no 1s base, so the loader cannot produce a 1h bundle.
    assert rows["eth_1h"]["missing_symbols"] == ["ETH/USDT"]
    # DOGE is absent everywhere.
    assert "DOGE/USDT" in rows["pair_1h"]["missing_symbols"]


def test_candidate_symbols_and_timeframe_canonicalizes() -> None:
    symbols, tf = rdc.candidate_symbols_and_timeframe(
        {"symbols": ["BTCUSDT", "eth-usdt", "BTCUSDT"], "strategy_timeframe": "4H"}
    )
    assert symbols == ["BTC/USDT", "ETH/USDT"]  # canonical + de-duplicated
    assert tf == "4h"


# --------------------------------------------------------------------------- #
# never-raise / report shape / CLI
# --------------------------------------------------------------------------- #
def test_missing_store_never_raises(tmp_path: Path) -> None:
    report = rdc.build_report(
        root=tmp_path / "does_not_exist",
        exchange="binance",
        mode="registry",
        symbols=None,
        timeframes=["1h"],
        min_bars=360,
        manifest_path=None,
    )
    assert report["store_detected"] is False
    assert report["coverage"] == []
    assert report["summary"]["symbols"] == 0
    # Markdown rendering also must not raise on an empty report.
    assert "No parquet market-data store detected" in rdc.render_markdown(report)


def test_main_writes_json_and_markdown(synthetic_store: Path, tmp_path: Path, capsys) -> None:
    json_out = tmp_path / "coverage.json"
    rc = rdc.main(
        [
            "--scan-dir",
            str(synthetic_store),
            "--exchange",
            "binance",
            "--json",
            str(json_out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "# Data coverage report (scan)" in captured.out
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "data_coverage_report"
    assert payload["mode"] == "scan"
    assert payload["summary"]["symbols"] == 2
