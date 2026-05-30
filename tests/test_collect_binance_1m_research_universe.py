from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/collect_binance_1m_research_universe.py"
    spec = importlib.util.spec_from_file_location("collect_binance_1m_research_universe", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_data_vision_specs_use_monthly_files_when_period_is_complete() -> None:
    since_ms = MODULE.parse_utc_ms("2025-01-01T00:00:00Z")
    until_ms = MODULE.parse_utc_ms("2025-03-02T23:59:59Z")
    until_ms = (
        (until_ms // MODULE.KLINE_INTERVAL_MS) * MODULE.KLINE_INTERVAL_MS
        + MODULE.KLINE_INTERVAL_MS
        - 1
    )

    specs = list(MODULE.iter_data_vision_specs(since_ms, until_ms))

    assert specs[:2] == [("monthly", "2025-01"), ("monthly", "2025-02")]
    assert specs[-2:] == [("daily", "2025-03-01"), ("daily", "2025-03-02")]


def test_data_vision_url_is_official_usd_m_futures_kline_path() -> None:
    assert MODULE.data_vision_url("BTCUSDT", period="monthly", token="2025-01") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1m/BTCUSDT-1m-2025-01.zip"
    )


def test_symbol_plan_keeps_inclusive_until_minute(tmp_path: Path) -> None:
    since_ms = MODULE.parse_utc_ms("2026-05-28T00:00:00Z")
    until_ms = MODULE.parse_utc_ms("2026-05-28T23:59:59Z")
    until_ms = (
        (until_ms // MODULE.KLINE_INTERVAL_MS) * MODULE.KLINE_INTERVAL_MS
        + MODULE.KLINE_INTERVAL_MS
        - 1
    )

    [plan] = MODULE.make_symbol_plans(
        ["BTCUSDT"],
        exchange_info={},
        since_ms=since_ms,
        until_ms=until_ms,
        db_path=tmp_path,
        exchange="binance",
        resume=True,
    )

    assert plan.start_ms == since_ms
    assert plan.end_ms == until_ms


def test_binance_ban_until_regex_extracts_epoch_milliseconds() -> None:
    match = MODULE.BINANCE_BAN_UNTIL_RE.search(
        "Way too many requests; IP(127.0.0.1) banned until 1780116708940."
    )

    assert match is not None
    assert match.group("until_ms") == "1780116708940"
