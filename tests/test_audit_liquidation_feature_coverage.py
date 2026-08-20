"""Tests for scripts/research/audit_liquidation_feature_coverage.py.

A tiny synthetic feature-points parquet store is written in-test to ``tmp_path``
through the repository's own upsert machinery (the exact on-disk layout the
read-only loader consumes), then the audit CLI is driven end-to-end via
``main([...])``.  Everything is closed-form and deterministic (no RNG, no wall
clock in the artifact).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.market_data import upsert_futures_feature_points_rows

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research"
    / "audit_liquidation_feature_coverage.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "audit_liquidation_feature_coverage", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()

_START = "2025-06-01"
_END = "2025-06-05"


def _ts(day: int, hour: int = 12) -> int:
    """Epoch ms for 2025-06-<day> at <hour>:00 UTC."""
    return int(datetime(2025, 6, day, hour, tzinfo=UTC).timestamp() * 1000)


def _seed_store(root: Path) -> None:
    """BTCUSDT: dense OI (5/5 days) + liquidation pairs on 4/5 days.
    TSLAUSDT: OI on 3/5 days, liquidation columns always null."""
    btc_rows = []
    for day in (1, 2, 3, 4, 5):
        row: dict[str, Any] = {
            "timestamp_ms": _ts(day),
            "open_interest": 1_000.0 + day,
            "mark_price": 100.0 + day,
        }
        if day <= 4:
            row["liquidation_long_notional"] = 10.0 * day
            row["liquidation_short_notional"] = 5.0 * day
        btc_rows.append(row)
    upsert_futures_feature_points_rows(
        str(root), exchange="binance", symbol="BTCUSDT", rows=btc_rows
    )

    tsla_rows = [
        {"timestamp_ms": _ts(day), "open_interest": 2_000.0 + day, "mark_price": 200.0}
        for day in (1, 3, 5)
    ]
    upsert_futures_feature_points_rows(
        str(root), exchange="binance", symbol="TSLAUSDT", rows=tsla_rows
    )


def _run(root: Path, out: Path, *extra: str) -> dict[str, Any]:
    rc = audit.main(
        [
            "--data-root",
            str(root),
            "--symbols",
            "BTC/USDT",
            "TSLAUSDT",
            "--start",
            _START,
            "--end",
            _END,
            "--json",
            str(out),
            *extra,
        ]
    )
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# happy path: per-symbol day counts, per-group shares, kill gates
# --------------------------------------------------------------------------- #


def test_symbol_day_coverage_and_group_gates(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _seed_store(root)
    report = _run(root, tmp_path / "report.json")

    assert report["status"] == "ok"
    assert report["days_in_range"] == 5
    assert report["symbol_count"] == 2

    rows = {row["symbol"]: row for row in report["symbols"]}
    btc = rows["BTCUSDT"]
    assert btc["group"] == "core_crypto"
    assert btc["days_with_open_interest"] == 5
    assert btc["days_with_liquidation_long"] == 4
    assert btc["days_with_liquidation_short"] == 4
    assert btc["days_with_liquidation_both"] == 4
    assert btc["open_interest_share"] == 1.0
    assert btc["liquidation_both_share"] == 0.8

    tsla = rows["TSLAUSDT"]
    assert tsla["group"] == "tradfi_perp"
    assert tsla["days_with_open_interest"] == 3
    assert tsla["days_with_liquidation_both"] == 0
    assert tsla["open_interest_share"] == 0.6

    groups = report["groups"]
    core = groups["core_crypto"]
    assert core["symbol_days"] == 5
    assert core["open_interest_gate"]["passed"] is True  # 1.0 >= 0.90
    assert core["liquidation_gate"]["passed"] is True  # 0.8 >= 0.80
    tradfi = groups["tradfi_perp"]
    assert tradfi["open_interest_gate"]["passed"] is False  # 0.6 < 0.90
    assert tradfi["liquidation_gate"]["passed"] is False  # 0.0 < 0.80

    summary = report["summary"]
    assert summary["groups_passing_oi_gate"] == 1
    assert summary["groups_passing_liquidation_gate"] == 1


def test_coverage_floors_are_overridable_kill_knobs(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _seed_store(root)
    report = _run(
        root,
        tmp_path / "report.json",
        "--oi-coverage-floor",
        "0.5",
        "--liquidation-coverage-floor",
        "0.9",
    )
    groups = report["groups"]
    assert groups["tradfi_perp"]["open_interest_gate"]["passed"] is True  # 0.6 >= 0.5
    assert groups["core_crypto"]["liquidation_gate"]["passed"] is False  # 0.8 < 0.9


# --------------------------------------------------------------------------- #
# zero-data run fails closed as insufficient_data JSON, never raises
# --------------------------------------------------------------------------- #


def test_zero_data_run_fails_closed_without_raising(tmp_path: Path) -> None:
    empty_root = tmp_path / "does-not-exist"
    out = tmp_path / "empty.json"
    report = _run(empty_root, out)
    assert report["status"] == "insufficient_data"
    assert report["summary"]["total_feature_rows"] == 0
    for group in report["groups"].values():
        assert group["open_interest_gate"]["passed"] is False
        assert group["liquidation_gate"]["passed"] is False
    # per-symbol rows still enumerate with zero coverage (fail-closed, visible)
    assert {row["symbol"] for row in report["symbols"]} == {"BTCUSDT", "TSLAUSDT"}
    assert all(row["feature_rows"] == 0 for row in report["symbols"])


# --------------------------------------------------------------------------- #
# determinism: identical inputs -> byte-identical artifact
# --------------------------------------------------------------------------- #


def test_artifact_is_byte_identical_across_runs(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _seed_store(root)
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    _run(root, out_a)
    _run(root, out_b)
    assert out_a.read_bytes() == out_b.read_bytes()


# --------------------------------------------------------------------------- #
# hygiene: symbol dedup/canonicalization + out-of-range points ignored
# --------------------------------------------------------------------------- #


def test_symbols_deduplicate_on_compact_token_and_range_is_respected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    _seed_store(root)
    # A point OUTSIDE the audit window must not count toward coverage.
    upsert_futures_feature_points_rows(
        str(root),
        exchange="binance",
        symbol="BTCUSDT",
        rows=[
            {
                "timestamp_ms": int(datetime(2025, 6, 7, 12, tzinfo=UTC).timestamp() * 1000),
                "open_interest": 9_999.0,
                "liquidation_long_notional": 1.0,
                "liquidation_short_notional": 1.0,
            }
        ],
    )
    out = tmp_path / "dedup.json"
    rc = audit.main(
        [
            "--data-root",
            str(root),
            "--symbols",
            "BTC/USDT",
            "BTCUSDT",
            "btc-usdt",
            "--start",
            _START,
            "--end",
            _END,
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["symbol_count"] == 1
    row = report["symbols"][0]
    assert row["symbol"] == "BTCUSDT"
    assert row["days_with_open_interest"] == 5  # 2025-06-07 ignored
    assert row["days_with_liquidation_both"] == 4
