from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
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


def test_default_universe_appends_new_tradfi_discovery() -> None:
    exchange_info = {
        "NEWEQUSDT": {
            "symbol": "NEWEQUSDT",
            "contractType": "TRADIFI_PERPETUAL",
            "quoteAsset": "USDT",
            "status": "TRADING",
        }
    }

    static_symbols = MODULE.resolve_default_symbols(
        exchange_info=exchange_info,
        universe_source="static",
    )
    expanded_symbols = MODULE.resolve_default_symbols(
        exchange_info=exchange_info,
        universe_source="static-plus-fapi-tradfi",
    )
    discovery = MODULE.universe_discovery_payload(
        exchange_info=exchange_info,
        universe_source="static-plus-fapi-tradfi",
        symbols=expanded_symbols,
        explicit_symbols=False,
    )

    assert "NEWEQUSDT" not in static_symbols
    assert "NEWEQUSDT" in expanded_symbols
    assert discovery["new_tradfi_since_static_snapshot_symbols"] == ["NEWEQUSDT"]
    assert discovery["selected_new_tradfi_symbols"] == ["NEWEQUSDT"]


def test_rows_to_frame_preserves_official_kline_taker_flow() -> None:
    frame = MODULE.rows_to_frame(
        [
            [
                1_786_060_800_000,
                "100",
                "101",
                "99",
                "100.5",
                "12",
                1_786_060_859_999,
                "1206",
                42,
                "7",
                "704",
                "0",
            ]
        ]
    )

    assert frame.row(0, named=True) == {
        "timestamp_ms": 1_786_060_800_000,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 12.0,
        "quote_volume": 1206.0,
        "taker_buy_base_volume": 7.0,
        "taker_buy_quote_volume": 704.0,
        "taker_sell_base_volume": 5.0,
        "taker_sell_quote_volume": 502.0,
    }
    assert MODULE.taker_feature_rows(frame) == [
        {
            "timestamp_ms": 1_786_060_800_000,
            "taker_buy_base_volume": 7.0,
            "taker_sell_base_volume": 5.0,
            "taker_buy_quote_volume": 704.0,
            "taker_sell_quote_volume": 502.0,
            "source": "binance_futures_kline",
        }
    ]


def test_data_vision_zip_preserves_official_taker_flow_columns() -> None:
    csv_bytes = (
        b"open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        b"taker_buy_volume,taker_buy_quote_volume,ignore\n"
        b"1786060800000,100,101,99,100.5,12,1786060859999,1206,42,7,704,0\n"
        b"1786060860000,100.5,102,100,101,9,1786060919999,910,38,4,405,0\n"
    )
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("BTCUSDT-1m.csv", csv_bytes)

    frame = MODULE.data_vision_zip_to_frame(
        archive_bytes.getvalue(),
        since_ms=1_786_060_860_000,
        until_ms=1_786_060_919_999,
    )

    assert frame.to_dicts() == [
        {
            "timestamp_ms": 1_786_060_860_000,
            "open": 100.5,
            "high": 102.0,
            "low": 100.0,
            "close": 101.0,
            "volume": 9.0,
            "quote_volume": 910.0,
            "taker_buy_base_volume": 4.0,
            "taker_buy_quote_volume": 405.0,
            "taker_sell_base_volume": 5.0,
            "taker_sell_quote_volume": 505.0,
        }
    ]
