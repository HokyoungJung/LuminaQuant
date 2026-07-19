from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import polars as pl

from lumina_quant import data_sync
from lumina_quant.data_collector import collect_binance_aggtrades_raw
from lumina_quant.storage.parquet import ParquetMarketDataRepository
from lumina_quant.storage.parquet.ohlcv_repo import RawPartitionBusyError


class _ExchangeStub:
    def close(self):
        return None


class _AggTradesExchange:
    def __init__(self, rows):
        self.rows = rows

    def agg_trades(self, **_kwargs):
        return self.rows


def _native_aggtrade(**overrides):
    row = {
        "a": 7,
        "T": 1_700_000_000_000,
        "p": "100.25",
        "q": "0.5",
        "m": False,
    }
    row.update(overrides)
    return row


ARCHIVE_MEMBER_NAME = "BTCUSDT-aggTrades-2025-01-01.csv"


def _zip_csv(
    payload: str,
    *,
    name: str = ARCHIVE_MEMBER_NAME,
    mode: int | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if mode is None:
            zf.writestr(name, payload)
        else:
            member = zipfile.ZipInfo(name)
            member.external_attr = mode << 16
            zf.writestr(member, payload)
    return buffer.getvalue()


def _zip_entries(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def _archive_rows(
    archive: bytes,
    *,
    expected_member_name: str = ARCHIVE_MEMBER_NAME,
    chunk_rows: int | None = None,
) -> list[dict]:
    return [
        row
        for chunk in data_sync._iter_archive_rows_to_raw_aggtrades(
            archive,
            expected_member_name=expected_member_name,
            cursor_ms=0,
            until_ms=2_000_000_000_000,
            chunk_rows=chunk_rows,
        )
        for row in chunk
    ]


@pytest.mark.parametrize("header", [False, True])
def test_archive_rows_accept_exact_optional_binance_header(header):
    payload = "1,100.0,0.5,10,11,1735689600000,False"
    if header:
        payload = (
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
            "transact_time,is_buyer_maker\n"
            f"{payload}"
        )

    rows = _archive_rows(_zip_csv(payload))

    assert rows == [
        {
            "agg_trade_id": 1,
            "timestamp_ms": 1_735_689_600_000,
            "price": 100.0,
            "quantity": 0.5,
            "is_buyer_maker": False,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        "1,100,0.5,10,11,1735689600000",
        "1,100,0.5,10,11,1735689600000,",
        "-1,100,0.5,10,11,1735689600000,true",
        "1,100,0.5,10,11,1735689600000,true,extra",
        "1,NaN,0.5,10,11,1735689600000,true",
        "1,Infinity,0.5,10,11,1735689600000,true",
        "1,100,0.5,11,10,1735689600000,true",
        "1,100,0.5,10,11,1735689600000,yes",
    ],
)
def test_archive_rows_reject_malformed_values(payload):
    with pytest.raises(ValueError):
        _archive_rows(_zip_csv(payload))


@pytest.mark.parametrize(
    "member_name",
    [
        "ETHUSDT-aggTrades-2025-01-01.csv",
        "BTCUSDT-aggTrades-2025-01-02.csv",
        "payload.csv",
        "BTCUSDT-aggTrades-2025-1-01.csv",
        f"nested/{ARCHIVE_MEMBER_NAME}",
    ],
)
def test_archive_rows_require_exact_canonical_member_identity(member_name):
    with pytest.raises(ValueError):
        _archive_rows(
            _zip_csv(
                "1,100,0.5,10,11,1735689600000,true",
                name=member_name,
            )
        )


def test_archive_rows_reject_noncanonical_expected_member_identity():
    with pytest.raises(ValueError):
        _archive_rows(
            _zip_csv(
                "1,100,0.5,10,11,1735689600000,true",
                name="payload.csv",
            ),
            expected_member_name="payload.csv",
        )


@pytest.mark.parametrize(
    "archive",
    [
        _zip_entries(
            {
                ARCHIVE_MEMBER_NAME: "1,100,0.5,10,11,1735689600000,true",
                "extra.csv": "2,100,0.5,12,13,1735689600001,false",
            }
        ),
        _zip_csv(
            "1,100,0.5,10,11,1735689600000,true",
            name=f"{ARCHIVE_MEMBER_NAME}/",
        ),
        _zip_csv(
            "1,100,0.5,10,11,1735689600000,true",
            mode=stat.S_IFLNK | 0o777,
        ),
        _zip_csv(
            "1,100,0.5,10,11,1735689600000,true",
            mode=stat.S_IFCHR | 0o600,
        ),
    ],
)
def test_archive_rows_require_one_expected_regular_member(archive):
    with pytest.raises(ValueError):
        _archive_rows(archive)


@pytest.mark.parametrize(
    "payload",
    [
        "\n".join(
            [
                "1,100,0.5,10,11,1735689600000,true",
                "1,100,0.5,12,13,1735689600001,false",
            ]
        ),
        "\n".join(
            [
                "2,100,0.5,10,11,1735689600000,true",
                "1,100,0.5,12,13,1735689600001,false",
            ]
        ),
    ],
)
def test_archive_rows_reject_duplicate_or_decreasing_aggregate_ids(payload):
    with pytest.raises(ValueError):
        _archive_rows(_zip_csv(payload))


def test_archive_rows_reject_decreasing_timestamp_across_chunk_boundary():
    archive = _zip_csv(
        "\n".join(
            [
                "1,100,0.5,10,11,1735689600000,true",
                "2,100,0.5,12,13,1735689600002,false",
                "3,100,0.5,14,15,1735689600001,true",
            ]
        )
    )

    with pytest.raises(ValueError):
        _archive_rows(archive, chunk_rows=2)


@pytest.mark.parametrize(
    "payload",
    [
        '1,100,0.5,10,11,1735689600000,"true',
        "\n".join(
            [
                "1,100,0.5,10,11,1735689600000,true",
                "2,100,0.5,12,13,1735689600001,false",
                '3,100,0.5,14,15,1735689600002,"true',
            ]
        ),
    ],
)
def test_archive_rows_normalize_strict_csv_errors_before_any_output(payload):
    rows = data_sync._iter_archive_rows_to_raw_aggtrades(
        _zip_csv(payload),
        expected_member_name=ARCHIVE_MEMBER_NAME,
        cursor_ms=0,
        until_ms=2_000_000_000_000,
        chunk_rows=2,
    )
    with pytest.raises(ValueError):
        next(rows)


def test_archive_rows_reject_empty_archive_and_out_of_window_malformed_row():
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w", compression=zipfile.ZIP_DEFLATED):
        pass
    with pytest.raises(ValueError):
        _archive_rows(empty.getvalue())
    with pytest.raises(ValueError):
        _archive_rows(
            _zip_csv(
                "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
                "transact_time,is_buyer_maker"
            )
        )
    with pytest.raises(ValueError):
        list(
            data_sync._iter_archive_rows_to_raw_aggtrades(
                _zip_csv(
                    "\n".join(
                        [
                            "1,100,0.5,10,11,1735689600000,true",
                            "2,NaN,0.5,12,13,1735689700000,false",
                        ]
                    )
                ),
                expected_member_name=ARCHIVE_MEMBER_NAME,
                cursor_ms=0,
                until_ms=1,
                chunk_rows=1,
            )
        )


def test_archive_rows_reject_timestamp_outside_named_archive_day():
    archive = _zip_csv(
        "1,100,0.5,10,11,1735776000000,true",
        name="BTCUSDT-aggTrades-2025-01-01.csv",
    )

    with pytest.raises(ValueError):
        _archive_rows(archive)


def test_archive_day_bounds_include_final_millisecond_without_overlap():
    first_start, first_end = data_sync._day_bounds_ms(date(2025, 1, 1))
    second_start, second_end = data_sync._day_bounds_ms(date(2025, 1, 2))
    first_rows = data_sync._archive_rows_to_raw_aggtrades(
        _zip_csv(f"1,100.0,0.5,1,1,{first_end},false"),
        expected_member_name=ARCHIVE_MEMBER_NAME,
        cursor_ms=first_start,
        until_ms=first_end,
    )
    second_rows = data_sync._archive_rows_to_raw_aggtrades(
        _zip_csv(
            f"2,100.0,0.5,2,2,{second_start},true",
            name="BTCUSDT-aggTrades-2025-01-02.csv",
        ),
        expected_member_name="BTCUSDT-aggTrades-2025-01-02.csv",
        cursor_ms=second_start,
        until_ms=second_end,
    )
    assert first_end == first_start + 86_399_999
    assert second_start == first_end + 1
    assert [row["timestamp_ms"] for row in first_rows] == [first_end]
    assert [row["timestamp_ms"] for row in second_rows] == [second_start]


def test_collect_binance_aggtrades_raw_checkpoint_resume(tmp_path, monkeypatch):
    calls: list[int] = []
    state = {"last": 0}

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        start_ms = int(kwargs["start_ms"])
        calls.append(start_ms)
        if start_ms == 0:
            state["last"] = 1_700_000_001_000
            return SimpleNamespace(
                fetched_rows=2,
                upserted_rows=2,
                first_timestamp_ms=1_700_000_000_000,
                last_timestamp_ms=1_700_000_001_000,
                checkpoint_timestamp_ms=1_700_000_001_000,
                checkpoint_trade_id=2,
            )
        state["last"] = 1_700_000_002_000
        return SimpleNamespace(
            fetched_rows=1,
            upserted_rows=1,
            first_timestamp_ms=1_700_000_002_000,
            last_timestamp_ms=1_700_000_002_000,
            checkpoint_timestamp_ms=1_700_000_002_000,
            checkpoint_trade_id=3,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    first = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=0,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    checkpoint_row = {
        "agg_trade_id": 2,
        "timestamp_ms": state["last"],
        "price": 100.0,
        "quantity": 0.5,
        "is_buyer_maker": False,
    }
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[checkpoint_row])
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload={
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "last_timestamp_ms": state["last"],
            "last_trade_id": 2,
            "observed_until_ms": state["last"],
            "updated_at_utc": "2025-01-01T00:00:00+00:00",
            "batch_rows": 1,
            "last_row": checkpoint_row,
            "last_row_sha256": _checkpoint_fixture_digest(checkpoint_row),
        },
    )
    second = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    assert calls == [0, 1_700_000_001_001]
    assert int(first["fetched_rows"]) == 2
    assert int(second["fetched_rows"]) == 1
    assert int(second["last_trade_id"]) == 3


def test_sync_symbol_aggtrades_raw_streams_archive_chunks(tmp_path, monkeypatch):
    start_ms = 1_735_689_600_000  # 2025-01-01T00:00:00Z
    row_count = 1_001
    end_ms = start_ms + ((row_count - 1) * 1_000)
    csv_payload = "\n".join(
        f"{1000 + idx},{100.0 + idx},0.5,{2000 + idx},{2000 + idx},{start_ms + idx * 1000},{str(idx % 2 == 0).lower()}"
        for idx in range(row_count)
    )
    archive = _zip_csv(csv_payload)

    monkeypatch.setattr(data_sync, "_download_zip_bytes", lambda *_args, **_kwargs: archive)
    monkeypatch.setattr(data_sync, "_now_ms", lambda: end_ms + (3 * 86_400_000))
    monkeypatch.setenv("LQ_RAW_ARCHIVE_CHUNK_ROWS", "1000")
    monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "128")
    monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "false")

    stats = data_sync.sync_symbol_aggtrades_raw(
        exchange=_ExchangeStub(),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=start_ms,
        end_ms=end_ms,
        max_batches=10,
        resume_from_checkpoint=False,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert stats.fetched_rows == row_count
    assert stats.upserted_rows == row_count
    assert raw.height == row_count
    assert raw["agg_trade_id"].head(3).to_list() == [1000, 1001, 1002]
    part_root = tmp_path / "market_data_raw_aggtrades/binance/BTCUSDT/date=2025-01-01"
    assert len(list(part_root.glob("part-*.parquet"))) == 2


@pytest.mark.parametrize("value", ["0", "999", "1000001", "-1", "1.5", "invalid"])
def test_raw_archive_chunk_rows_rejects_present_invalid_configuration(monkeypatch, value):
    monkeypatch.setenv("LQ_RAW_ARCHIVE_CHUNK_ROWS", value)

    with pytest.raises(ValueError, match="LQ_RAW_ARCHIVE_CHUNK_ROWS"):
        data_sync._raw_archive_chunk_rows()


def test_sync_archive_gap_starts_live_coverage_at_exact_gap_cursor(tmp_path, monkeypatch):
    start_ms = 1_735_689_600_000
    downloaded_days: list[date] = []
    live_requests: list[dict] = []

    def _missing_archive(url, **_kwargs):
        archive_date_parts = url.removesuffix(".zip").rsplit("-", 3)[-3:]
        downloaded_days.append(date.fromisoformat("-".join(archive_date_parts)))
        return None

    class _LiveAtGapCursor:
        def agg_trades(self, **kwargs):
            live_requests.append(kwargs)
            return [_native_aggtrade(a=101, T=start_ms)]

    monkeypatch.setattr(data_sync, "_download_zip_bytes", _missing_archive)
    monkeypatch.setattr(data_sync, "_now_ms", lambda: start_ms + (3 * 86_400_000))

    stats = data_sync.sync_symbol_aggtrades_raw(
        exchange=_LiveAtGapCursor(),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=start_ms,
        end_ms=start_ms,
        retries=0,
        resume_from_checkpoint=False,
    )

    assert downloaded_days == [date(2025, 1, 1)]
    assert live_requests == [
        {
            "symbol": "BTC/USDT",
            "start_time": start_ms,
            "end_time": start_ms,
            "from_id": None,
            "limit": 1000,
        }
    ]
    assert stats.fetched_rows == 1
    assert stats.upserted_rows == 1


def test_sync_rejects_incomplete_future_live_coverage_without_checkpoint(tmp_path, monkeypatch):
    start_ms = 1_700_000_000_000
    repo = ParquetMarketDataRepository(str(tmp_path))
    requests: list[dict] = []

    class _EmptyLiveCoverage:
        def agg_trades(self, **kwargs):
            requests.append(kwargs)
            return []

    monkeypatch.setattr(data_sync, "_now_ms", lambda: start_ms)
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()
    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root)

    with pytest.raises(ValueError, match="Incomplete aggTrade continuity"):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_EmptyLiveCoverage(),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=start_ms,
            end_ms=start_ms + 1,
            max_batches=2,
            retries=0,
            resume_from_checkpoint=False,
        )
    assert _raw_stream_tree_snapshot(stream_root) == before

    assert requests
    assert repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT") == {}
    assert not repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT").exists()
    assert repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT").is_empty()


def test_sync_invalid_native_live_tokens_fail_before_raw_mutation(tmp_path, monkeypatch):
    start_ms = 1_700_000_000_000
    mutations: list[str] = []
    monkeypatch.setattr(data_sync, "_now_ms", lambda: start_ms)
    for field, invalid in (
        ("a", True),
        ("a", -1),
        ("a", " 7"),
        ("a", "07"),
        ("a", 7.0),
        ("T", True),
        ("T", -1),
        ("T", f" {start_ms}"),
        ("T", f"0{start_ms}"),
        ("T", float(start_ms)),
        ("p", " 100.25"),
        ("p", "+100.25"),
        ("p", "1e2"),
        ("p", "NaN"),
        ("p", "100."),
        ("q", "Infinity"),
        ("q", "01.0"),
    ):
        monkeypatch.setattr(
            ParquetMarketDataRepository,
            "append_raw_aggtrades",
            lambda *_args, **_kwargs: mutations.append("append"),
        )
        monkeypatch.setattr(
            ParquetMarketDataRepository,
            "write_raw_checkpoint",
            lambda *_args, **_kwargs: mutations.append("checkpoint"),
        )
        monkeypatch.setattr(
            ParquetMarketDataRepository,
            "append_raw_wal_record",
            lambda *_args, **_kwargs: mutations.append("wal"),
        )
        with pytest.raises(ValueError):
            data_sync.sync_symbol_aggtrades_raw(
                exchange=_AggTradesExchange([_native_aggtrade(**{field: invalid})]),
                db_path=str(tmp_path),
                exchange_id="binance",
                symbol="BTC/USDT",
                start_ms=start_ms,
                end_ms=start_ms,
                retries=0,
                resume_from_checkpoint=False,
            )
    assert mutations == []


def test_normalize_aggtrade_row_accepts_only_documented_generic_types():
    assert data_sync.normalize_aggtrade_row(
        {
            "id": 7,
            "timestamp": 1_700_000_000_000,
            "price": 100,
            "amount": 0.5,
            "maker": False,
        }
    ) == {
        "agg_trade_id": 7,
        "timestamp_ms": 1_700_000_000_000,
        "price": 100.0,
        "quantity": 0.5,
        "is_buyer_maker": False,
    }

    assert data_sync.normalize_aggtrade_row(
        {
            "id": "7",
            "timestamp": "1700000000000",
            "price": "100.0",
            "amount": "0.5",
            "maker": False,
        }
    ) == {
        "agg_trade_id": 7,
        "timestamp_ms": 1_700_000_000_000,
        "price": 100.0,
        "quantity": 0.5,
        "is_buyer_maker": False,
    }

    for field, invalid in (
        ("id", True),
        ("id", " 7"),
        ("id", "+7"),
        ("id", "7e0"),
        ("timestamp", True),
        ("timestamp", " 1700000000000"),
        ("timestamp", "+1700000000000"),
        ("timestamp", "1.7e12"),
        ("price", True),
        ("price", " 100"),
        ("price", "+100"),
        ("price", "1e2"),
        ("price", float("nan")),
        ("amount", "Infinity"),
        ("amount", float("inf")),
        ("amount", SimpleNamespace(value=0.5)),
        ("maker", 0),
    ):
        payload = {
            "id": 7,
            "timestamp": 1_700_000_000_000,
            "price": 100,
            "amount": 0.5,
            "maker": False,
        }
        payload[field] = invalid
        with pytest.raises(ValueError):
            data_sync.normalize_aggtrade_row(payload)


def test_fetch_trades_body_type_error_is_not_a_compatibility_fallback():
    calls: list[dict | None] = []

    class _BodyTypeError:
        def fetch_trades(self, _symbol, *, since, limit, params=None):
            calls.append(params)
            raise TypeError("exchange body failure")

    with pytest.raises(TypeError, match="exchange body failure"):
        data_sync._fetch_trades_with_retry(
            _BodyTypeError(),
            "BTC/USDT",
            since_ms=1_700_000_000_000,
            from_id=12,
            until_ms=1_700_000_000_100,
            limit=2,
            retries=0,
            base_wait_sec=0,
        )
    assert calls == [{"fromId": 12}]


def test_sync_crash_after_full_page_checkpoints_proven_boundary_and_resumes_from_id(
    tmp_path, monkeypatch
):
    start_ms = 1_700_000_000_000
    until_ms = start_ms + 100
    now_ms = [start_ms]
    monkeypatch.setattr(data_sync, "_now_ms", lambda: now_ms[0])
    first_page = [_native_aggtrade(a=10, T=start_ms), _native_aggtrade(a=11, T=start_ms)]
    first_requests: list[dict] = []

    class _CrashAfterFirstPage:
        def agg_trades(self, **kwargs):
            first_requests.append(kwargs)
            if len(first_requests) == 1:
                return first_page
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_CrashAfterFirstPage(),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=start_ms,
            end_ms=until_ms,
            limit=2,
            retries=0,
            max_batches=4,
            resume_from_checkpoint=False,
        )

    repo = ParquetMarketDataRepository(str(tmp_path))
    checkpoint = repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")
    assert checkpoint["last_trade_id"] == 11
    assert checkpoint["observed_until_ms"] == start_ms

    now_ms[0] = until_ms
    resume_requests: list[dict] = []

    class _ResumeAtCompoundCursor:
        def agg_trades(self, **kwargs):
            resume_requests.append(kwargs)
            return [_native_aggtrade(a=12, T=start_ms)]

    data_sync.sync_symbol_aggtrades_raw(
        exchange=_ResumeAtCompoundCursor(),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=start_ms,
        end_ms=until_ms,
        limit=2,
        retries=0,
        max_batches=4,
    )

    checkpoint = repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")
    assert resume_requests == [
        {
            "symbol": "BTC/USDT",
            "start_time": None,
            "end_time": None,
            "from_id": 12,
            "limit": 2,
        }
    ]
    assert repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT").to_dicts() == [
        {
            "agg_trade_id": 10,
            "timestamp_ms": start_ms,
            "price": 100.25,
            "quantity": 0.5,
            "is_buyer_maker": False,
        },
        {
            "agg_trade_id": 11,
            "timestamp_ms": start_ms,
            "price": 100.25,
            "quantity": 0.5,
            "is_buyer_maker": False,
        },
        {
            "agg_trade_id": 12,
            "timestamp_ms": start_ms,
            "price": 100.25,
            "quantity": 0.5,
            "is_buyer_maker": False,
        },
    ]
    assert checkpoint["observed_until_ms"] == until_ms


def test_sync_terminal_finalization_advances_observed_until_only_after_coverage(
    tmp_path, monkeypatch
):
    start_ms = 1_700_000_000_000
    until_ms = start_ms + 100
    monkeypatch.setattr(data_sync, "_now_ms", lambda: until_ms)

    data_sync.sync_symbol_aggtrades_raw(
        exchange=_AggTradesExchange([_native_aggtrade(a=10, T=start_ms)]),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=start_ms,
        end_ms=until_ms,
        limit=2,
        retries=0,
        resume_from_checkpoint=False,
    )

    checkpoint = ParquetMarketDataRepository(str(tmp_path)).read_raw_checkpoint(
        exchange="binance", symbol="BTC/USDT"
    )
    assert checkpoint["last_timestamp_ms"] == start_ms
    assert checkpoint["observed_until_ms"] == until_ms


def test_sync_same_millisecond_pages_advance_by_aggregate_id_and_return_checkpoint_stats(
    tmp_path, monkeypatch
):
    timestamp_ms = 1_700_000_000_000
    requests: list[dict] = []
    pages = [
        [
            _native_aggtrade(a=10, T=timestamp_ms),
            _native_aggtrade(a=11, T=timestamp_ms),
        ],
        [_native_aggtrade(a=12, T=timestamp_ms)],
    ]

    class _SameMillisecondPages:
        def agg_trades(self, **kwargs):
            requests.append(kwargs)
            return pages.pop(0)

    monkeypatch.setattr(data_sync, "_now_ms", lambda: timestamp_ms)

    stats = data_sync.sync_symbol_aggtrades_raw(
        exchange=_SameMillisecondPages(),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=timestamp_ms,
        end_ms=timestamp_ms,
        limit=2,
        retries=0,
        resume_from_checkpoint=False,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    checkpoint = repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")
    persisted = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert requests == [
        {
            "symbol": "BTC/USDT",
            "start_time": timestamp_ms,
            "end_time": timestamp_ms,
            "from_id": None,
            "limit": 2,
        },
        {
            "symbol": "BTC/USDT",
            "start_time": None,
            "end_time": None,
            "from_id": 12,
            "limit": 2,
        },
    ]
    assert persisted["agg_trade_id"].to_list() == [10, 11, 12]
    assert stats.fetched_rows == stats.upserted_rows == persisted.height == 3
    assert stats.first_timestamp_ms == timestamp_ms
    assert stats.last_timestamp_ms == checkpoint["last_row"]["timestamp_ms"]
    assert stats.checkpoint_timestamp_ms == checkpoint["last_timestamp_ms"]
    assert stats.checkpoint_trade_id == checkpoint["last_trade_id"]


def test_sync_rejects_malformed_archive_before_committing(tmp_path, monkeypatch):
    start_ms = 1_735_689_600_000
    archive = _zip_csv(
        "\n".join(
            [
                f"1,100,0.5,10,11,{start_ms},true",
                f"2,100,0.5,12,13,{start_ms + 1},false",
                f'3,100,0.5,14,15,{start_ms + 2},"true',
            ]
        )
    )
    append_calls: list[list[dict]] = []

    def _append_raw_aggtrades(_self, **kwargs):
        append_calls.append(kwargs["rows"])
        return len(kwargs["rows"])

    monkeypatch.setattr(data_sync, "_download_zip_bytes", lambda *_args, **_kwargs: archive)
    monkeypatch.setattr(data_sync, "_now_ms", lambda: start_ms + (3 * 86_400_000))
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_aggtrades",
        _append_raw_aggtrades,
    )

    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_ExchangeStub(),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=start_ms,
            end_ms=start_ms + 2,
            max_batches=10,
            resume_from_checkpoint=False,
        )

    assert append_calls == []


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            _native_aggtrade(),
            {
                "agg_trade_id": 7,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.25,
                "quantity": 0.5,
                "is_buyer_maker": False,
            },
        ),
        (
            {
                "id": "8",
                "timestamp": 1_700_000_000_001,
                "price": "101.5",
                "amount": "0.25",
                "info": {
                    "a": 8,
                    "T": 1_700_000_000_001,
                    "p": "101.5",
                    "q": "0.25",
                    "m": True,
                },
            },
            {
                "agg_trade_id": 8,
                "timestamp_ms": 1_700_000_000_001,
                "price": 101.5,
                "quantity": 0.25,
                "is_buyer_maker": True,
            },
        ),
    ],
)
def test_normalize_aggtrade_row_accepts_native_and_ccxt_shapes(payload, expected):
    assert data_sync.normalize_aggtrade_row(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        _native_aggtrade(a=None),
        _native_aggtrade(a="invalid"),
        _native_aggtrade(a=True),
        _native_aggtrade(a=-1),
        _native_aggtrade(T=None),
        _native_aggtrade(T="invalid"),
        _native_aggtrade(T=True),
        _native_aggtrade(T=0),
        _native_aggtrade(p="NaN"),
        _native_aggtrade(p="Infinity"),
        _native_aggtrade(p=0),
        _native_aggtrade(p=-1),
        _native_aggtrade(q="NaN"),
        _native_aggtrade(q="Infinity"),
        _native_aggtrade(q=0),
        _native_aggtrade(q=-1),
        _native_aggtrade(m=None, side="sell"),
        _native_aggtrade(m="false"),
        _native_aggtrade(m=0),
    ],
)
def test_normalize_aggtrade_row_rejects_invalid_live_invariants(payload):
    with pytest.raises(ValueError):
        data_sync.normalize_aggtrade_row(payload)


def test_fetch_aggtrades_batch_rejects_mixed_valid_and_invalid_rows():
    exchange = _AggTradesExchange([_native_aggtrade(), _native_aggtrade(a=None)])

    with pytest.raises(ValueError):
        data_sync.fetch_aggtrades_batch(
            exchange=exchange,
            symbol="BTC/USDT",
            since_ms=1_700_000_000_000,
            retries=0,
        )


@pytest.mark.parametrize(
    "rows",
    [
        [_native_aggtrade(), _native_aggtrade()],
        [_native_aggtrade(), _native_aggtrade(a=6, T=1_700_000_000_001)],
        [_native_aggtrade(), _native_aggtrade(a=8, T=1_699_999_999_999)],
    ],
)
def test_fetch_aggtrades_batch_rejects_invalid_live_identity_order(rows):
    with pytest.raises(ValueError):
        data_sync.fetch_aggtrades_batch(
            exchange=_AggTradesExchange(rows),
            symbol="BTC/USDT",
            since_ms=1_700_000_000_000,
            retries=0,
        )


def test_sync_rejects_live_malformed_batch_before_append_wal_or_checkpoint(tmp_path, monkeypatch):
    append_calls: list[dict] = []
    checkpoint_calls: list[dict] = []
    wal_calls: list[dict] = []
    exchange = _AggTradesExchange([_native_aggtrade(), _native_aggtrade(m="false")])

    monkeypatch.setattr(data_sync, "_now_ms", lambda: 1_700_000_000_000)

    def _append(_self, **kwargs):
        append_calls.append(kwargs)
        return len(kwargs["rows"])

    def _checkpoint(_self, **kwargs):
        checkpoint_calls.append(kwargs)

    def _wal(_self, **kwargs):
        wal_calls.append(kwargs)

    monkeypatch.setattr(ParquetMarketDataRepository, "append_raw_aggtrades", _append)
    monkeypatch.setattr(ParquetMarketDataRepository, "write_raw_checkpoint", _checkpoint)
    monkeypatch.setattr(ParquetMarketDataRepository, "append_raw_wal_record", _wal)

    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=exchange,
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=1_700_000_000_000,
            end_ms=1_700_000_000_001,
            retries=0,
            resume_from_checkpoint=False,
        )

    assert append_calls == []
    assert checkpoint_calls == []
    assert wal_calls == []


_CHECKPOINT_LAST_ROW_FIELDS = (
    "agg_trade_id",
    "timestamp_ms",
    "price",
    "quantity",
    "is_buyer_maker",
)


def _checkpoint_fixture_digest(row: dict) -> str:
    """Hash the documented raw-row fields without sharing the production helper."""
    canonical_row = {field: row[field] for field in _CHECKPOINT_LAST_ROW_FIELDS}
    encoded = json.dumps(
        canonical_row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_payload(row: dict, **overrides) -> dict:
    payload = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "last_timestamp_ms": row["timestamp_ms"],
        "last_trade_id": row["agg_trade_id"],
        "observed_until_ms": row["timestamp_ms"],
        "updated_at_utc": "2025-01-01T00:00:00+00:00",
        "batch_rows": 1,
        "last_row": row,
        "last_row_sha256": _checkpoint_fixture_digest(row),
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_read_raw_checkpoint_rejects_present_malformed_or_non_object(tmp_path, contents):
    repo = ParquetMarketDataRepository(str(tmp_path))
    path = repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT")
    path.parent.mkdir(parents=True)
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")


def test_read_raw_checkpoint_rejects_present_empty_object(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    path = repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT")
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")


@pytest.mark.parametrize(
    "field, value",
    [
        ("last_timestamp_ms", True),
        ("last_trade_id", "7"),
        ("observed_until_ms", 0),
        ("batch_rows", 0),
        ("updated_at_utc", "not-a-time"),
        ("exchange", "other"),
        ("symbol", "ETH/USDT"),
        ("last_row_sha256", "not-a-digest"),
    ],
)
def test_sync_rejects_malformed_checkpoint_without_side_effects(
    tmp_path, monkeypatch, field, value
):
    row = data_sync.normalize_aggtrade_row(_native_aggtrade())
    repo = ParquetMarketDataRepository(str(tmp_path))
    path = repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            _checkpoint_payload(row, **{field: value}), sort_keys=True, separators=(",", ":")
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_aggtrades",
        lambda *_args, **_kwargs: calls.append("append"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "write_raw_checkpoint",
        lambda *_args, **_kwargs: calls.append("checkpoint"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_wal_record",
        lambda *_args, **_kwargs: calls.append("wal"),
    )

    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_AggTradesExchange([]),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=row["timestamp_ms"],
            end_ms=row["timestamp_ms"] + 1,
            retries=0,
        )
    assert calls == []


def test_read_raw_checkpoint_rejects_stale_digest_after_non_identity_row_change(tmp_path):
    row = data_sync.normalize_aggtrade_row(_native_aggtrade())
    repo = ParquetMarketDataRepository(str(tmp_path))
    path = repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            _checkpoint_payload(
                {**row, "price": row["price"] + 1.0},
                last_row_sha256=_checkpoint_fixture_digest(row),
            ),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()
    before = _raw_stream_tree_snapshot(tmp_path / "market_data_raw_aggtrades")

    with pytest.raises(ValueError, match="Raw aggTrades checkpoint is malformed"):
        repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")

    assert _raw_stream_tree_snapshot(tmp_path / "market_data_raw_aggtrades") == before


@pytest.mark.parametrize(
    "payload",
    [
        _native_aggtrade(id=8),
        _native_aggtrade(timestamp=1_700_000_000_001),
        _native_aggtrade(price="101"),
        _native_aggtrade(amount="0.6"),
        _native_aggtrade(maker=True),
        {**_native_aggtrade(), "info": {**_native_aggtrade(), "a": 8}},
    ],
)
def test_normalize_aggtrade_row_rejects_conflicting_aliases(payload):
    with pytest.raises(ValueError):
        data_sync.normalize_aggtrade_row(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"a": 7},
        {"id": "7", "timestamp": 1_700_000_000_000, "price": "1", "amount": "1", "m": False},
        _native_aggtrade(a="7.0"),
        _native_aggtrade(T=1_700_000_000_000.5),
    ],
)
def test_normalize_aggtrade_row_rejects_incomplete_native_or_fractional_identity(payload):
    with pytest.raises(ValueError):
        data_sync.normalize_aggtrade_row(payload)


@pytest.mark.parametrize(
    "rows",
    [
        [_native_aggtrade(), _native_aggtrade()],
        [_native_aggtrade(), _native_aggtrade(p="101")],
        [_native_aggtrade(), _native_aggtrade(a=6, T=1_700_000_000_001)],
        [_native_aggtrade(), _native_aggtrade(a=8, T=1_699_999_999_999)],
    ],
)
def test_sync_rejects_invalid_live_order_without_side_effects(tmp_path, monkeypatch, rows):
    calls: list[str] = []
    monkeypatch.setattr(data_sync, "_now_ms", lambda: 1_700_000_000_000)
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_aggtrades",
        lambda *_args, **_kwargs: calls.append("append"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "write_raw_checkpoint",
        lambda *_args, **_kwargs: calls.append("checkpoint"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_wal_record",
        lambda *_args, **_kwargs: calls.append("wal"),
    )

    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_AggTradesExchange(rows),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=1_700_000_000_000,
            end_ms=1_700_000_000_001,
            retries=0,
            resume_from_checkpoint=False,
        )
    assert calls == []


def test_sync_accepts_exact_authenticated_checkpoint_overlap(tmp_path, monkeypatch):
    overlap = data_sync.normalize_aggtrade_row(_native_aggtrade())
    next_row = _native_aggtrade(a=8, T=1_700_000_000_001)
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[overlap],
    )
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload=_checkpoint_payload(overlap),
    )
    appended: list[list[dict]] = []
    monkeypatch.setattr(data_sync, "_now_ms", lambda: overlap["timestamp_ms"])
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_aggtrades",
        lambda _self, **kwargs: appended.append(kwargs["rows"]) or len(kwargs["rows"]),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository, "write_raw_checkpoint", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository, "append_raw_wal_record", lambda *_args, **_kwargs: None
    )

    data_sync.sync_symbol_aggtrades_raw(
        exchange=_AggTradesExchange([_native_aggtrade(), next_row]),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=overlap["timestamp_ms"],
        end_ms=next_row["T"],
        retries=0,
    )
    assert appended == [[data_sync.normalize_aggtrade_row(next_row)]]


@pytest.mark.parametrize("case", ["exact", "conflict", "higher_id", "next_timestamp"])
def test_sync_archive_restart_requires_exact_checkpoint_identity_before_advancing(
    tmp_path, monkeypatch, case
):
    timestamp_ms = 1_735_689_600_000
    stored_rows = [
        data_sync.normalize_aggtrade_row(_native_aggtrade(a=1, T=timestamp_ms, p="100.0")),
        data_sync.normalize_aggtrade_row(_native_aggtrade(a=2, T=timestamp_ms, p="100.1")),
        data_sync.normalize_aggtrade_row(_native_aggtrade(a=3, T=timestamp_ms, p="100.2")),
    ]
    checkpoint_row = stored_rows[-1]
    archive_rows = [
        _native_aggtrade(a=1, T=timestamp_ms, p="100.0"),
        _native_aggtrade(a=2, T=timestamp_ms, p="100.1"),
    ]
    if case == "exact":
        archive_rows.extend(
            [
                _native_aggtrade(a=3, T=timestamp_ms, p="100.2"),
                _native_aggtrade(a=4, T=timestamp_ms + 1, p="100.3"),
            ]
        )
    elif case == "conflict":
        archive_rows.append(_native_aggtrade(a=3, T=timestamp_ms, p="101.0"))
    elif case == "higher_id":
        archive_rows.append(_native_aggtrade(a=4, T=timestamp_ms, p="100.3"))
    else:
        archive_rows.append(_native_aggtrade(a=4, T=timestamp_ms + 1, p="100.3"))
    archive = _zip_csv(
        "\n".join(
            f"{row['a']},{row['p']},{row['q']},0,0,{row['T']},{str(row['m']).lower()}"
            for row in archive_rows
        )
    )
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=stored_rows)
    repo.write_raw_checkpoint(
        exchange="binance", symbol="BTC/USDT", payload=_checkpoint_payload(checkpoint_row)
    )
    stream_root = tmp_path / "market_data_raw_aggtrades"
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()
    before = _raw_stream_tree_snapshot(stream_root)
    mutations: list[str] = []
    if case != "exact":
        original_append = ParquetMarketDataRepository.append_raw_aggtrades
        original_checkpoint = ParquetMarketDataRepository.write_raw_checkpoint
        original_wal = ParquetMarketDataRepository.append_raw_wal_record

        def _append(*args, **kwargs):
            mutations.append("append")
            return original_append(*args, **kwargs)

        def _checkpoint(*args, **kwargs):
            mutations.append("checkpoint")
            return original_checkpoint(*args, **kwargs)

        def _wal(*args, **kwargs):
            mutations.append("wal")
            return original_wal(*args, **kwargs)

        monkeypatch.setattr(ParquetMarketDataRepository, "append_raw_aggtrades", _append)
        monkeypatch.setattr(ParquetMarketDataRepository, "write_raw_checkpoint", _checkpoint)
        monkeypatch.setattr(ParquetMarketDataRepository, "append_raw_wal_record", _wal)
    monkeypatch.setattr(data_sync, "_download_zip_bytes", lambda *_args, **_kwargs: archive)
    monkeypatch.setattr(data_sync, "_now_ms", lambda: timestamp_ms + (3 * 86_400_000))
    monkeypatch.setattr(data_sync, "_raw_archive_chunk_rows", lambda: 2)

    if case == "exact":
        stats = data_sync.sync_symbol_aggtrades_raw(
            exchange=_ExchangeStub(),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=timestamp_ms,
            end_ms=timestamp_ms + 1,
            max_batches=4,
            retries=0,
        )
        assert stats.upserted_rows == 1
        assert repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT").to_dicts() == [
            *stored_rows,
            data_sync.normalize_aggtrade_row(_native_aggtrade(a=4, T=timestamp_ms + 1, p="100.3")),
        ]
    else:
        with pytest.raises(ValueError):
            data_sync.sync_symbol_aggtrades_raw(
                exchange=_ExchangeStub(),
                db_path=str(tmp_path),
                exchange_id="binance",
                symbol="BTC/USDT",
                start_ms=timestamp_ms,
                end_ms=timestamp_ms + 1,
                max_batches=4,
                retries=0,
            )
        assert _raw_stream_tree_snapshot(stream_root) == before
        assert mutations == []


def test_sync_checkpoint_recovery_does_not_load_full_raw_history(tmp_path, monkeypatch):
    row = data_sync.normalize_aggtrade_row(_native_aggtrade())
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
    repo.write_raw_checkpoint(
        exchange="binance", symbol="BTC/USDT", payload=_checkpoint_payload(row)
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "load_raw_aggtrades",
        lambda *_args, **_kwargs: pytest.fail("full raw history load must not be used"),
    )
    monkeypatch.setattr(data_sync, "_now_ms", lambda: row["timestamp_ms"])
    data_sync.sync_symbol_aggtrades_raw(
        exchange=_AggTradesExchange([]),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=row["timestamp_ms"],
        end_ms=row["timestamp_ms"],
        retries=0,
    )


def test_sync_recovers_parquet_ahead_checkpoint_with_wal(tmp_path, monkeypatch):
    first = data_sync.normalize_aggtrade_row(_native_aggtrade(a=1))
    tail = data_sync.normalize_aggtrade_row(_native_aggtrade(a=2, T=1_700_000_000_001))
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first, tail])
    repo.write_raw_checkpoint(
        exchange="binance", symbol="BTC/USDT", payload=_checkpoint_payload(first)
    )
    monkeypatch.setattr(data_sync, "_now_ms", lambda: tail["timestamp_ms"])
    data_sync.sync_symbol_aggtrades_raw(
        exchange=_AggTradesExchange([]),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=first["timestamp_ms"],
        end_ms=tail["timestamp_ms"],
        retries=0,
    )
    assert repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")["last_row"] == tail
    assert "aggtrades_raw_checkpoint_recovery" in repo.raw_wal_path(
        exchange="binance", symbol="BTC/USDT"
    ).read_text(encoding="utf-8")


def test_sync_recovers_first_raw_commit_with_checkpoint_and_wal(tmp_path, monkeypatch):
    row = data_sync.normalize_aggtrade_row(_native_aggtrade(a=1))
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
    monkeypatch.setattr(data_sync, "_now_ms", lambda: row["timestamp_ms"])
    data_sync.sync_symbol_aggtrades_raw(
        exchange=_AggTradesExchange([]),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=row["timestamp_ms"],
        end_ms=row["timestamp_ms"],
        retries=0,
    )
    assert repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")["last_row"] == row
    assert "aggtrades_raw_first_commit_recovery" in repo.raw_wal_path(
        exchange="binance", symbol="BTC/USDT"
    ).read_text(encoding="utf-8")


def test_sync_rejects_equal_checkpoint_id_at_later_timestamp_without_side_effects(
    tmp_path, monkeypatch
):
    checkpoint_row = data_sync.normalize_aggtrade_row(_native_aggtrade())
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[checkpoint_row])
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload=_checkpoint_payload(checkpoint_row),
    )
    calls: list[str] = []
    monkeypatch.setattr(data_sync, "_now_ms", lambda: checkpoint_row["timestamp_ms"])
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_aggtrades",
        lambda *_args, **_kwargs: calls.append("append"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "write_raw_checkpoint",
        lambda *_args, **_kwargs: calls.append("checkpoint"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_wal_record",
        lambda *_args, **_kwargs: calls.append("wal"),
    )

    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_AggTradesExchange([_native_aggtrade(T=checkpoint_row["timestamp_ms"] + 1)]),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=checkpoint_row["timestamp_ms"],
            end_ms=checkpoint_row["timestamp_ms"] + 1,
            retries=0,
        )

    assert calls == []


def test_sync_rejects_conflicting_checkpoint_overlap_without_side_effects(tmp_path, monkeypatch):
    overlap = data_sync.normalize_aggtrade_row(_native_aggtrade())
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[overlap])
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload=_checkpoint_payload(overlap),
    )
    calls: list[str] = []
    monkeypatch.setattr(data_sync, "_now_ms", lambda: overlap["timestamp_ms"])
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_aggtrades",
        lambda *_args, **_kwargs: calls.append("append"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "write_raw_checkpoint",
        lambda *_args, **_kwargs: calls.append("checkpoint"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_wal_record",
        lambda *_args, **_kwargs: calls.append("wal"),
    )

    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_AggTradesExchange([_native_aggtrade(p="101")]),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=overlap["timestamp_ms"],
            end_ms=overlap["timestamp_ms"],
            retries=0,
        )
    assert calls == []


def test_collect_binance_aggtrades_raw_bootstrap_lookback_used_without_checkpoint(
    tmp_path, monkeypatch
):
    observed_since: list[int] = []

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        observed_since.append(int(kwargs["start_ms"]))
        return SimpleNamespace(
            fetched_rows=0,
            upserted_rows=0,
            first_timestamp_ms=None,
            last_timestamp_ms=None,
            checkpoint_timestamp_ms=None,
            checkpoint_trade_id=None,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    until_ms = 1_700_000_010_000
    result = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=until_ms,
        bootstrap_lookback_hours=2,
        limit=1000,
        max_batches=10,
    )

    expected_since = until_ms - (2 * 60 * 60 * 1000)
    assert observed_since == [expected_since]
    assert int(result["start_cursor_ms"]) == expected_since


def test_collect_binance_aggtrades_raw_rejects_missing_inventory_before_corrupt_part(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**_kwargs):
        repo = ParquetMarketDataRepository(str(tmp_path))
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    "agg_trade_id": 10,
                    "timestamp_ms": 1_700_000_100_000,
                    "price": 100.0,
                    "quantity": 0.1,
                    "is_buyer_maker": False,
                }
            ],
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    corrupt_dir = tmp_path / "market_data_raw_aggtrades" / "binance" / "BTCUSDT" / "date=2023-11-14"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    corrupt_path = corrupt_dir / "part-0000.parquet"
    corrupt_bytes = b"not-a-parquet-file"
    corrupt_path.write_bytes(corrupt_bytes)
    repo = ParquetMarketDataRepository(str(tmp_path))
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()

    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
    with pytest.raises(ValueError, match="inventory"):
        collect_binance_aggtrades_raw(
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            since_ms=1_700_000_100_000,
            until_ms=1_700_000_100_000,
            limit=1000,
            max_batches=10,
        )

    assert (
        _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
        == before
    )


def test_append_raw_aggtrades_rejects_missing_inventory_before_corrupt_part_without_mutation(
    tmp_path, monkeypatch
):
    repo = ParquetMarketDataRepository(str(tmp_path))
    part_path = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    )
    part_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_bytes = b"not-a-real-parquet-file"
    part_path.write_bytes(corrupt_bytes)

    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()
    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "_raw_read_parquet",
        lambda *_a, **_k: pytest.fail("uninventoried part must not be read"),
    )
    with pytest.raises(ValueError, match="inventory"):
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    "agg_trade_id": 1,
                    "timestamp_ms": 1_735_689_600_000,
                    "price": 100.0,
                    "quantity": 0.1,
                    "is_buyer_maker": False,
                }
            ],
        )

    assert (
        _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
        == before
    )


def test_append_raw_aggtrades_uses_incremental_parts_for_append_only_batches(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))

    first_written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    second_written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    part_files = sorted(path.name for path in part_dir.glob("part-*.parquet"))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert first_written == 1
    assert second_written == 1
    assert part_files == ["part-0000.parquet", "part-0001.parquet"]
    assert raw.height == 2


def test_append_raw_aggtrades_compacts_back_to_single_part_when_batches_overlap(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))

    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
            {
                "agg_trade_id": 3,
                "timestamp_ms": 1_735_689_602_000,
                "price": 102.0,
                "quantity": 0.3,
                "is_buyer_maker": False,
            },
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    part_files = sorted(path.name for path in part_dir.glob("part-*.parquet"))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert written == 2
    assert part_files == ["part-0000.parquet"]
    assert raw.height == 3


def test_load_raw_aggtrades_rejects_divergent_inventory_before_corrupt_extra_part(
    tmp_path, monkeypatch
):
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    corrupt_path = part_dir / "part-0002.parquet"
    corrupt_path.write_bytes(b"not-a-real-parquet-file")

    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "_raw_read_parquet",
        lambda *_a, **_k: pytest.fail("uninventoried part must not be read"),
    )
    with pytest.raises(ValueError, match="inventory"):
        repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert (
        _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
        == before
    )


def test_append_raw_aggtrades_auto_compacts_when_part_threshold_exceeded(tmp_path, monkeypatch):
    repo = ParquetMarketDataRepository(str(tmp_path))
    monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "2")
    monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "true")

    for offset in range(3):
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    "agg_trade_id": offset + 1,
                    "timestamp_ms": 1_735_689_600_000 + offset,
                    "price": 100.0 + offset,
                    "quantity": 0.1,
                    "is_buyer_maker": bool(offset % 2),
                }
            ],
        )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")
    meta = json.loads(repo._raw_meta_path(exchange="binance", symbol="BTC/USDT").read_text())

    assert sorted(path.name for path in part_dir.glob("part-*.parquet")) == ["part-0000.parquet"]
    assert raw.height == 3
    assert meta["raw_compaction_required"] is False
    assert meta["last_raw_compaction_partition"] == "date=2025-01-01"


def test_append_raw_aggtrades_marks_meta_when_threshold_exceeded_but_auto_compact_disabled(
    tmp_path, monkeypatch
):
    repo = ParquetMarketDataRepository(str(tmp_path))
    monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "1")
    monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "false")

    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    meta = json.loads(repo._raw_meta_path(exchange="binance", symbol="BTC/USDT").read_text())

    assert sorted(path.name for path in part_dir.glob("part-*.parquet")) == [
        "part-0000.parquet",
        "part-0001.parquet",
    ]
    assert meta["raw_compaction_required"] is True
    assert meta["last_raw_part_count"] == 2


def test_append_raw_aggtrades_raises_when_raw_stream_lease_is_held(tmp_path, monkeypatch):
    repo = ParquetMarketDataRepository(str(tmp_path))
    holder = ParquetMarketDataRepository(str(tmp_path))
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_POLL_SECONDS", "0.01")
    lease = holder.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")

    try:
        with pytest.raises(RawPartitionBusyError):
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[
                    {
                        "agg_trade_id": 1,
                        "timestamp_ms": 1_735_689_600_000,
                        "price": 100.0,
                        "quantity": 0.1,
                        "is_buyer_maker": False,
                    }
                ],
            )
    finally:
        lease.release()


def test_raw_validator_accepts_only_adjacent_identical_duplicates(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    row = {
        "agg_trade_id": 1,
        "timestamp_ms": 1_700_000_000_000,
        "price": 100.0,
        "quantity": 0.5,
        "is_buyer_maker": False,
    }
    frame = pl.DataFrame(
        [row, row],
        schema={
            "agg_trade_id": pl.Int64,
            "timestamp_ms": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "is_buyer_maker": pl.Boolean,
        },
    )
    assert repo._ensure_raw_aggtrades_frame(frame).height == 1
    repeated = pl.DataFrame([row, {**row, "agg_trade_id": 2}, row], schema=frame.schema)
    with pytest.raises(ValueError):
        repo._ensure_raw_aggtrades_frame(repeated)


def test_raw_validator_rejects_invalid_storage_values(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    frame = pl.DataFrame(
        [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1,
                "price": float("nan"),
                "quantity": 0.0,
                "is_buyer_maker": False,
            }
        ],
        schema={
            "agg_trade_id": pl.Int64,
            "timestamp_ms": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "is_buyer_maker": pl.Boolean,
        },
    )
    with pytest.raises(ValueError):
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=frame)


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame(
            [
                {
                    "agg_trade_id": 1,
                    "timestamp_ms": 1,
                    "price": 1.0,
                    "quantity": 1.0,
                    "is_buyer_maker": None,
                }
            ],
            schema={
                "agg_trade_id": pl.Int64,
                "timestamp_ms": pl.Int64,
                "price": pl.Float64,
                "quantity": pl.Float64,
                "is_buyer_maker": pl.Boolean,
            },
        ),
        pl.DataFrame(
            [
                {
                    "agg_trade_id": 1,
                    "timestamp_ms": 1,
                    "price": 1,
                    "quantity": 1.0,
                    "is_buyer_maker": False,
                }
            ],
            schema={
                "agg_trade_id": pl.Int64,
                "timestamp_ms": pl.Int64,
                "price": pl.Int64,
                "quantity": pl.Float64,
                "is_buyer_maker": pl.Boolean,
            },
        ),
    ],
)
def test_raw_validator_rejects_null_and_wrong_storage_types(tmp_path, frame):
    repo = ParquetMarketDataRepository(str(tmp_path))
    with pytest.raises(ValueError):
        repo._ensure_raw_aggtrades_frame(frame)


def test_raw_validator_rejects_adjacent_conflicting_duplicate(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    row = {
        "agg_trade_id": 1,
        "timestamp_ms": 1,
        "price": 1.0,
        "quantity": 1.0,
        "is_buyer_maker": False,
    }
    frame = pl.DataFrame(
        [row, {**row, "price": 2.0}],
        schema={
            "agg_trade_id": pl.Int64,
            "timestamp_ms": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "is_buyer_maker": pl.Boolean,
        },
    )
    with pytest.raises(ValueError):
        repo._ensure_raw_aggtrades_frame(frame)


@pytest.mark.parametrize("kind", ["ahead", "conflict"])
def test_sync_checkpoint_binding_rejections_do_not_mutate(tmp_path, monkeypatch, kind):
    raw = data_sync.normalize_aggtrade_row(_native_aggtrade(a=1))
    checkpoint_row = raw
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[raw])
    if kind == "ahead":
        checkpoint_row = data_sync.normalize_aggtrade_row(
            _native_aggtrade(a=2, T=raw["timestamp_ms"] + 1)
        )
    elif kind == "conflict":
        checkpoint_row = {**raw, "price": 101.0}
    checkpoint_path = repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(_checkpoint_payload(checkpoint_row), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    stream_root = tmp_path / "market_data_raw_aggtrades"
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()
    before = _raw_stream_tree_snapshot(stream_root)
    mutations: list[str] = []
    original_append = ParquetMarketDataRepository.append_raw_aggtrades
    original_checkpoint = ParquetMarketDataRepository.write_raw_checkpoint
    original_wal = ParquetMarketDataRepository.append_raw_wal_record

    def _append(*args, **kwargs):
        mutations.append("append")
        return original_append(*args, **kwargs)

    def _checkpoint(*args, **kwargs):
        mutations.append("checkpoint")
        return original_checkpoint(*args, **kwargs)

    def _wal(*args, **kwargs):
        mutations.append("wal")
        return original_wal(*args, **kwargs)

    monkeypatch.setattr(ParquetMarketDataRepository, "append_raw_aggtrades", _append)
    monkeypatch.setattr(ParquetMarketDataRepository, "write_raw_checkpoint", _checkpoint)
    monkeypatch.setattr(ParquetMarketDataRepository, "append_raw_wal_record", _wal)
    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_AggTradesExchange([]),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=raw["timestamp_ms"],
            end_ms=raw["timestamp_ms"],
            retries=0,
        )
    assert mutations == []
    assert _raw_stream_tree_snapshot(stream_root) == before


def test_sync_missing_checkpoint_binding_is_exact_and_non_mutating(tmp_path, monkeypatch):
    raw = data_sync.normalize_aggtrade_row(_native_aggtrade(a=1))
    repo = ParquetMarketDataRepository(str(tmp_path))
    checkpoint_path = repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(_checkpoint_payload(raw), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()
    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root)
    mutations: list[str] = []
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "write_raw_checkpoint",
        lambda *_a, **_k: mutations.append("checkpoint"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_wal_record",
        lambda *_a, **_k: mutations.append("wal"),
    )
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "append_raw_aggtrades",
        lambda *_a, **_k: mutations.append("append"),
    )

    with pytest.raises(
        ValueError, match="Raw aggTrades checkpoint is not bound to persisted raw parquet"
    ):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_AggTradesExchange([]),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=raw["timestamp_ms"],
            end_ms=raw["timestamp_ms"],
            retries=0,
        )

    assert mutations == []
    assert _raw_stream_tree_snapshot(stream_root) == before


def test_compaction_invalid_part_preserves_bytes(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    row = data_sync.normalize_aggtrade_row(_native_aggtrade())
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
    part = repo.raw_partition_path(
        exchange="binance", symbol="BTC/USDT", partition_date="2023-11-14"
    )
    original = part.read_bytes()
    part.write_bytes(b"not parquet")
    corrupt = part.read_bytes()
    with pytest.raises(ValueError, match="Raw aggTrades inventory file identity diverges"):
        repo._compact_raw_partition(
            exchange="binance", symbol="BTC/USDT", partition_root=part.parent
        )
    assert part.read_bytes() == corrupt
    assert original != corrupt


def test_symbol_stream_lease_contention_release_and_retry(tmp_path, monkeypatch):
    first = ParquetMarketDataRepository(str(tmp_path))
    second = ParquetMarketDataRepository(str(tmp_path))
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_POLL_SECONDS", "0.001")
    lease = first.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    try:
        with pytest.raises(RawPartitionBusyError):
            second.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    finally:
        lease.release()
    retry = second.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    retry.release()


def test_sync_failure_releases_symbol_stream_lease(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    path = repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT")
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        data_sync.sync_symbol_aggtrades_raw(
            exchange=_AggTradesExchange([]),
            db_path=str(tmp_path),
            exchange_id="binance",
            symbol="BTC/USDT",
            start_ms=1,
            end_ms=1,
            retries=0,
        )
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()


def _raw_stream_tree_snapshot(root: Path, *, exclude: set[str] | None = None):
    """Capture non-following identity and content evidence for fail-closed tests."""
    try:
        os.lstat(root)
    except FileNotFoundError:
        return None

    excluded = exclude or set()
    snapshot = {}
    pending = [root]
    while pending:
        path = pending.pop()
        relative = "." if path == root else path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        metadata = os.lstat(path)
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if stat.S_ISLNK(metadata.st_mode):
            snapshot[relative] = ("symlink", identity, os.readlink(path))
        elif stat.S_ISDIR(metadata.st_mode):
            snapshot[relative] = ("directory", identity)
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)
        elif stat.S_ISREG(metadata.st_mode):
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                digest = hashlib.sha256()
                while block := os.read(fd, 1 << 20):
                    digest.update(block)
            finally:
                os.close(fd)
            snapshot[relative] = ("regular", identity, digest.hexdigest())
        else:
            snapshot[relative] = ("special", identity)
    return dict(sorted(snapshot.items()))


@pytest.mark.parametrize(
    ("exchange", "symbol"),
    [
        ("..", "BTC/USDT"),
        ("/absolute", "BTC/USDT"),
        ("binance/unsafe", "BTC/USDT"),
        ("binance", ".."),
        ("binance", "/absolute"),
        ("binance", "BTC/USDT/unsafe"),
    ],
)
def test_append_raw_aggtrades_rejects_unsafe_stream_components_without_mutation(
    tmp_path, exchange, symbol
):
    repo = ParquetMarketDataRepository(str(tmp_path))
    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root)

    with pytest.raises(ValueError):
        repo.append_raw_aggtrades(
            exchange=exchange,
            symbol=symbol,
            rows=[
                {
                    "agg_trade_id": 1,
                    "timestamp_ms": 1_735_689_600_000,
                    "price": 100.0,
                    "quantity": 0.1,
                    "is_buyer_maker": False,
                }
            ],
        )

    assert _raw_stream_tree_snapshot(stream_root) == before


def test_append_raw_aggtrades_rejects_multi_date_batch_before_part_publication(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    rows = [
        {
            "agg_trade_id": 1,
            "timestamp_ms": 1_735_689_600_000,
            "price": 100.0,
            "quantity": 0.1,
            "is_buyer_maker": False,
        },
        {
            "agg_trade_id": 2,
            "timestamp_ms": 1_735_776_000_000,
            "price": 101.0,
            "quantity": 0.1,
            "is_buyer_maker": True,
        },
    ]

    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root)
    with pytest.raises(ValueError, match="exactly one UTC date"):
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=rows)

    assert _raw_stream_tree_snapshot(stream_root) == before


def test_append_raw_aggtrades_rejects_cross_date_id_timestamp_regression_without_mutation(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    first = {
        "agg_trade_id": 2,
        "timestamp_ms": 1_735_689_600_000,
        "price": 100.0,
        "quantity": 0.1,
        "is_buyer_maker": False,
    }
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
    original = repo.raw_partition_path(
        exchange="binance", symbol="BTC/USDT", partition_date="2025-01-01"
    ).read_bytes()

    with pytest.raises(ValueError, match="timestamp regresses"):
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    **first,
                    "agg_trade_id": 1,
                    "timestamp_ms": 1_735_776_000_000,
                }
            ],
        )

    assert (
        repo.raw_partition_path(
            exchange="binance", symbol="BTC/USDT", partition_date="2025-01-01"
        ).read_bytes()
        == original
    )
    assert not (
        tmp_path / "market_data_raw_aggtrades" / "binance" / "BTCUSDT" / "date=2025-01-02"
    ).exists()


def test_append_raw_aggtrades_rejects_exact_cross_date_conflict_without_mutation(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    row = {
        "agg_trade_id": 1,
        "timestamp_ms": 1_735_689_600_000,
        "price": 100.0,
        "quantity": 0.1,
        "is_buyer_maker": False,
    }
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
    stream_root = tmp_path / "market_data_raw_aggtrades"
    before = _raw_stream_tree_snapshot(stream_root)

    with pytest.raises(ValueError, match="duplicate aggregate ID conflicts"):
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    **row,
                    "timestamp_ms": 1_735_776_000_000,
                    "price": 101.0,
                }
            ],
        )

    assert _raw_stream_tree_snapshot(stream_root) == before
    assert not (
        tmp_path / "market_data_raw_aggtrades" / "binance" / "BTCUSDT" / "date=2025-01-02"
    ).exists()


def test_preflight_raw_aggtrades_rejects_later_corrupt_date_without_first_date_mutation(
    tmp_path,
):
    repo = ParquetMarketDataRepository(str(tmp_path))
    first = {
        "agg_trade_id": 1,
        "timestamp_ms": 1_735_689_600_000,
        "price": 100.0,
        "quantity": 0.1,
        "is_buyer_maker": False,
    }
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
    stream_root = tmp_path / "market_data_raw_aggtrades"
    corrupt_path = repo.raw_partition_path(
        exchange="binance", symbol="BTC/USDT", partition_date="2025-01-02"
    )
    corrupt_path.parent.mkdir(parents=True)
    corrupt_bytes = b"not-a-real-parquet-file"
    corrupt_path.write_bytes(corrupt_bytes)
    before = _raw_stream_tree_snapshot(stream_root)

    with pytest.raises(ValueError, match="inventory"):
        repo.preflight_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {**first, "agg_trade_id": 2, "timestamp_ms": 1_735_689_600_001},
                {**first, "agg_trade_id": 3, "timestamp_ms": 1_735_776_000_000},
            ],
        )
    assert _raw_stream_tree_snapshot(stream_root) == before


def test_sync_recovers_checkpointless_multi_part_tail_without_full_history_load(
    tmp_path, monkeypatch
):
    first = data_sync.normalize_aggtrade_row(_native_aggtrade(a=1, T=1_735_689_600_000))
    tail = data_sync.normalize_aggtrade_row(_native_aggtrade(a=2, T=1_735_776_000_000))
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
    repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[tail])
    assert not repo.raw_checkpoint_path(exchange="binance", symbol="BTC/USDT").exists()
    monkeypatch.setattr(
        ParquetMarketDataRepository,
        "load_raw_aggtrades",
        lambda *_args, **_kwargs: pytest.fail("full raw history load must not be used"),
    )
    monkeypatch.setattr(data_sync, "_now_ms", lambda: tail["timestamp_ms"])

    data_sync.sync_symbol_aggtrades_raw(
        exchange=_AggTradesExchange([]),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=first["timestamp_ms"],
        end_ms=tail["timestamp_ms"],
        retries=0,
    )

    assert repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")["last_row"] == tail
    assert "aggtrades_raw_first_commit_recovery" in repo.raw_wal_path(
        exchange="binance", symbol="BTC/USDT"
    ).read_text(encoding="utf-8")


def _seed_authenticated_raw_history(repo, *, symbol: str, parts: int) -> list[dict]:
    start_ms = 1_735_689_600_000
    rows = []
    for offset in range(parts):
        row = {
            "agg_trade_id": offset + 1,
            "timestamp_ms": start_ms + offset * 86_400_000,
            "price": 100.0 + offset,
            "quantity": 0.1,
            "is_buyer_maker": bool(offset % 2),
        }
        repo.append_raw_aggtrades(exchange="binance", symbol=symbol, rows=[row])
        rows.append(row)
    return rows


def test_raw_stream_lease_spans_recovery_inventory_bounds_and_load(tmp_path, monkeypatch):
    first = ParquetMarketDataRepository(str(tmp_path))
    second = ParquetMarketDataRepository(str(tmp_path))
    rows = _seed_authenticated_raw_history(first, symbol="BTC/USDT", parts=2)
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_POLL_SECONDS", "0.01")
    lease = first.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")

    try:
        first.recover_raw_stream(exchange="binance", symbol="BTC/USDT", lease=lease)
        bounds = first.read_raw_recovery_bounds(
            exchange="binance",
            symbol="BTC/USDT",
            checkpoint_last_row=rows[0],
            lease=lease,
        )
        loaded = first.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT", lease=lease)
        with pytest.raises(RawPartitionBusyError):
            second.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[
                    {
                        "agg_trade_id": 3,
                        "timestamp_ms": rows[-1]["timestamp_ms"] + 1,
                        "price": 102.0,
                        "quantity": 0.1,
                        "is_buyer_maker": False,
                    }
                ],
            )
    finally:
        lease.release()

    assert bounds.to_dicts() == rows
    assert loaded.to_dicts() == rows
    assert (
        second.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    "agg_trade_id": 3,
                    "timestamp_ms": rows[-1]["timestamp_ms"] + 1,
                    "price": 102.0,
                    "quantity": 0.1,
                    "is_buyer_maker": False,
                }
            ],
        )
        == 1
    )


def test_strict_tail_append_never_reads_or_hashes_authenticated_history(tmp_path, monkeypatch):
    original_read = ParquetMarketDataRepository._raw_read_parquet
    original_digest = ParquetMarketDataRepository._raw_file_digest
    original_names = ParquetMarketDataRepository._raw_dir_names
    active: dict[str, object] = {}
    namespace_scans: dict[int, int] = {}

    def _read(self, path):
        if path in active["historical_parts"]:
            active["reads"].append(path)
        return original_read(self, path)

    def _digest(self, path):
        if path in active["historical_parts"]:
            active["digests"].append(path)
        return original_digest(self, path)

    def _names(self, fd):
        active["namespace_enumerations"].append(fd)
        return original_names(fd)

    monkeypatch.setattr(ParquetMarketDataRepository, "_raw_read_parquet", _read)
    monkeypatch.setattr(ParquetMarketDataRepository, "_raw_file_digest", _digest)
    monkeypatch.setattr(ParquetMarketDataRepository, "_raw_dir_names", _names)

    for symbol, parts in (("BTC/USDT", 2), ("ETH/USDT", 12)):
        repo = ParquetMarketDataRepository(str(tmp_path))
        active = {
            "historical_parts": set(),
            "reads": [],
            "digests": [],
            "namespace_enumerations": [],
        }
        history = _seed_authenticated_raw_history(repo, symbol=symbol, parts=parts)
        active = {
            "historical_parts": {
                repo.raw_partition_path(
                    exchange="binance",
                    symbol=symbol,
                    partition_date=datetime.fromtimestamp(row["timestamp_ms"] / 1000, UTC)
                    .date()
                    .isoformat(),
                )
                for row in history
            },
            "reads": [],
            "digests": [],
            "namespace_enumerations": [],
        }
        tail = history[-1]
        assert (
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol=symbol,
                rows=[
                    {
                        "agg_trade_id": parts + 1,
                        "timestamp_ms": tail["timestamp_ms"] + 86_400_000,
                        "price": 200.0,
                        "quantity": 0.2,
                        "is_buyer_maker": True,
                    }
                ],
            )
            == 1
        )

        assert active["reads"] == []
        assert active["digests"] == []
        assert active["namespace_enumerations"]
        namespace_scans[parts] = len(active["namespace_enumerations"])

    # Allow a conservative affine number of namespace enumerations: fixed
    # setup/preflight work plus at most ten per authenticated part. The
    # small-to-large marginal bound rejects history-rescan growth without
    # freezing harmless directory-operation details.
    per_part_scan_upper_bound = 10
    fixed_scan_upper_bound = 32
    assert namespace_scans[2] <= (per_part_scan_upper_bound * 2) + fixed_scan_upper_bound
    assert namespace_scans[12] <= (per_part_scan_upper_bound * 12) + fixed_scan_upper_bound
    assert namespace_scans[12] - namespace_scans[2] <= per_part_scan_upper_bound * (12 - 2)


@pytest.mark.parametrize("parts", [2, 12])
def test_checkpoint_recovery_reads_only_bound_point_and_tail_parts(tmp_path, monkeypatch, parts):
    repo = ParquetMarketDataRepository(str(tmp_path))
    history = _seed_authenticated_raw_history(repo, symbol="BTC/USDT", parts=parts)
    authenticated_reads: list[tuple[Path, str]] = []
    digests: list[Path] = []
    original_authenticated_read = ParquetMarketDataRepository._raw_read_authenticated_parquet
    original_digest = ParquetMarketDataRepository._raw_file_digest

    def _authenticated_read(self, path, entry):
        authenticated_reads.append((path, entry["name"]))
        return original_authenticated_read(self, path, entry)

    def _digest(self, path):
        digests.append(path)
        return original_digest(self, path)

    monkeypatch.setattr(
        ParquetMarketDataRepository, "_raw_read_authenticated_parquet", _authenticated_read
    )
    monkeypatch.setattr(ParquetMarketDataRepository, "_raw_file_digest", _digest)
    bounds = repo.read_raw_recovery_bounds(
        exchange="binance", symbol="BTC/USDT", checkpoint_last_row=history[0]
    )
    checkpointless = repo.read_raw_recovery_bounds(
        exchange="binance", symbol="BTC/USDT", checkpoint_last_row=None
    )

    assert bounds.to_dicts() == [history[0], history[-1]]
    assert checkpointless.to_dicts() == [history[-1]]
    checkpoint_part = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date=datetime.fromtimestamp(history[0]["timestamp_ms"] / 1000, UTC)
        .date()
        .isoformat(),
    )
    tail_part = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date=datetime.fromtimestamp(history[-1]["timestamp_ms"] / 1000, UTC)
        .date()
        .isoformat(),
    )
    assert authenticated_reads == [
        (checkpoint_part, checkpoint_part.relative_to(checkpoint_part.parents[1]).as_posix()),
        (tail_part, tail_part.relative_to(tail_part.parents[1]).as_posix()),
        (tail_part, tail_part.relative_to(tail_part.parents[1]).as_posix()),
    ]
    assert digests == []


@pytest.mark.parametrize("selected", ["checkpoint", "tail"])
def test_checkpoint_recovery_authenticates_selected_parts_without_reading_middle_history(
    tmp_path, monkeypatch, selected
):
    repo = ParquetMarketDataRepository(str(tmp_path))
    history = _seed_authenticated_raw_history(repo, symbol="BTC/USDT", parts=3)
    stream_root = tmp_path / "market_data_raw_aggtrades"
    symbol_root = stream_root / "binance" / "BTCUSDT"
    part_paths = [
        repo.raw_partition_path(
            exchange="binance",
            symbol="BTC/USDT",
            partition_date=datetime.fromtimestamp(row["timestamp_ms"] / 1000, UTC)
            .date()
            .isoformat(),
        )
        for row in history
    ]
    inventory_path = symbol_root / ".raw-inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    target = part_paths[0] if selected == "checkpoint" else part_paths[-1]
    middle = part_paths[1]
    bad_digests = {
        target.relative_to(symbol_root).as_posix(): "0" * 64,
        middle.relative_to(symbol_root).as_posix(): "1" * 64,
    }
    for entry in inventory["parts"]:
        if entry["name"] in bad_digests:
            entry["content_sha256"] = bad_digests[entry["name"]]
    inventory_body = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    inventory["inventory_sha256"] = hashlib.sha256(
        json.dumps(inventory_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    lease = repo.acquire_raw_symbol_stream_lease(exchange="binance", symbol="BTC/USDT")
    lease.release()
    before = _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
    reads: list[Path] = []
    digests: list[Path] = []
    original_authenticated_read = ParquetMarketDataRepository._raw_read_authenticated_parquet
    original_digest = ParquetMarketDataRepository._raw_file_digest

    def _authenticated_read(self, path, entry):
        reads.append(path)
        return original_authenticated_read(self, path, entry)

    def _digest(self, path):
        digests.append(path)
        return original_digest(self, path)

    monkeypatch.setattr(
        ParquetMarketDataRepository, "_raw_read_authenticated_parquet", _authenticated_read
    )
    monkeypatch.setattr(ParquetMarketDataRepository, "_raw_file_digest", _digest)

    with pytest.raises(ValueError, match="bytes do not match inventory"):
        repo.read_raw_recovery_bounds(
            exchange="binance",
            symbol="BTC/USDT",
            checkpoint_last_row=history[0],
        )

    assert target in reads
    assert middle not in reads
    assert middle not in digests
    assert (
        _raw_stream_tree_snapshot(stream_root, exclude={"binance/BTCUSDT/.raw-stream.lock"})
        == before
    )
