from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

import pytest

from lumina_quant.storage.parquet import ParquetMarketDataRepository

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "refresh_final_portfolio_validation_data.py"
SPEC = importlib.util.spec_from_file_location(
    "refresh_final_portfolio_validation_data", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load refresh_final_portfolio_validation_data module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_portfolio_symbols_preserves_saved_weight_order(tmp_path: Path) -> None:
    payload = {
        "weights": [
            {"symbols": ["BNB/USDT", "TRX/USDT"]},
            {"symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"]},
        ]
    }
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert MODULE.load_portfolio_symbols(path) == ["BNB/USDT", "TRX/USDT", "BTC/USDT", "ETH/USDT"]


def test_load_feature_symbols_filters_to_required_strategy_classes(tmp_path: Path) -> None:
    payload = {
        "selected_team": [
            {"strategy_class": "CompositeTrendStrategy", "symbols": ["BTC/USDT", "ETH/USDT"]},
            {
                "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                "symbols": ["BTC/USDT", "BNB/USDT"],
            },
            {"strategy_class": "PerpCrowdingCarryStrategy", "symbols": ["SOL/USDT"]},
        ]
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert MODULE.load_feature_symbols(path) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_latest_runtime_tail_uses_runtime_second_not_previous_day() -> None:
    now = MODULE.parse_utc("2026-03-19T09:30:29.591000Z")
    assert MODULE.iso_utc(MODULE.latest_runtime_tail_utc(now)) == "2026-03-19T09:30:29Z"


def test_iso_utc_treats_naive_datetime_as_utc() -> None:
    assert MODULE.iso_utc(datetime(2026, 3, 18, 23, 59, 58)) == "2026-03-18T23:59:58Z"


def _build_archive_zip(rows: list[tuple[int, float, float, int, bool]]) -> bytes:
    payload = "\n".join(
        f"{agg_trade_id},{price},{quantity},0,0,{timestamp_ms},{str(is_buyer_maker).lower()},true"
        for agg_trade_id, price, quantity, timestamp_ms, is_buyer_maker in rows
    )
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BTCUSDT-aggTrades-2025-01-01.csv", payload)
    return blob.getvalue()


def test_refresh_symbol_raw_first_ohlcv_derives_from_stored_raw_aggtrades(
    tmp_path: Path, monkeypatch
) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    cutoff_dt = MODULE.parse_utc("2025-01-01T00:00:02Z")
    floor_dt = MODULE.parse_utc("2025-01-01T00:00:00Z")
    assert cutoff_dt is not None
    assert floor_dt is not None

    archive_zip = _build_archive_zip(
        [
            (1, 100.0, 0.1, 1_735_689_600_000, False),
            (2, 101.0, 0.2, 1_735_689_600_500, True),
            (3, 102.0, 0.3, 1_735_689_601_500, False),
        ]
    )

    monkeypatch.setattr(MODULE, "_download_zip_bytes", lambda *args, **kwargs: archive_zip)
    monkeypatch.setattr(
        MODULE, "_binance_archive_url", lambda *args, **kwargs: "https://example.test"
    )
    monkeypatch.setattr(
        MODULE,
        "_collect_live_raw_rows",
        lambda **kwargs: [],
    )

    result = MODULE.refresh_symbol_raw_first_ohlcv(
        repo=repo,
        symbol="BTC/USDT",
        db_path=str(tmp_path),
        exchange_id="binance",
        cutoff_dt=cutoff_dt,
        floor_dt=floor_dt,
        guard=None,
    )

    raw = repo.load_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-01T00:00:02Z",
    )
    ohlcv = repo.load_ohlcv(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-01T00:00:02Z",
    )

    assert raw.height == 3
    assert ohlcv.height == 2
    assert result.after_raw_agg_trade_utc == "2025-01-01T00:00:01.500000Z"
    assert result.after_ohlcv_max_utc == "2025-01-01T00:00:01Z"
    assert result.archive_days_missing == 0
    assert result.source_mix == "archive_only"
    assert result.live_tail_status == "empty"
    assert result.stage_timings_seconds["archive_download"] >= 0.0
    assert result.stage_timings_seconds["archive_parse"] >= 0.0
    assert result.stage_timings_seconds["total_refresh"] >= 0.0
    assert result.live_raw_rows_upserted == 0
    assert result.derived_ohlcv_rows_upserted >= 2


def test_collect_live_raw_rows_reduces_limit_after_rate_limit(monkeypatch) -> None:
    class _Exchange:
        def close(self):
            return None

    calls: list[int] = []
    state = {"attempt": 0}

    monkeypatch.setattr(MODULE, "create_binance_futures_client", lambda **kwargs: _Exchange())
    monkeypatch.setattr(MODULE.time, "sleep", lambda *_args, **_kwargs: None)

    def _fetch(*, exchange, symbol, since_ms, limit, retries, base_wait_sec):
        _ = exchange, symbol, since_ms, retries, base_wait_sec
        calls.append(int(limit))
        state["attempt"] += 1
        if state["attempt"] == 1:
            raise RuntimeError("Too Many Requests")
        return [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ]

    monkeypatch.setattr(MODULE, "fetch_aggtrades_batch", _fetch)

    rows = MODULE._collect_live_raw_rows(
        symbol="BTC/USDT",
        start_ms=1_735_689_600_000,
        end_ms=1_735_689_600_000,
        limit=1000,
        pause_sec=0.0,
    )

    assert calls[:2] == [1000, 500]
    assert len(rows) == 1


def test_collect_live_raw_rows_scans_across_sparse_hour_windows(monkeypatch) -> None:
    class _Exchange:
        def close(self):
            return None

    calls: list[int] = []
    hour_ms = 3_600_000

    monkeypatch.setattr(MODULE, "create_binance_futures_client", lambda **kwargs: _Exchange())
    monkeypatch.setattr(MODULE.time, "sleep", lambda *_args, **_kwargs: None)

    def _fetch(*, exchange, symbol, since_ms, limit, retries, base_wait_sec):
        _ = exchange, symbol, limit, retries, base_wait_sec
        calls.append(int(since_ms))
        if int(since_ms) == 0:
            return [
                {
                    "agg_trade_id": 10,
                    "timestamp_ms": 1_000,
                    "price": 100.0,
                    "quantity": 0.1,
                    "is_buyer_maker": False,
                }
            ]
        if int(since_ms) == hour_ms:
            return [
                {
                    "agg_trade_id": 20,
                    "timestamp_ms": hour_ms + 2_000,
                    "price": 101.0,
                    "quantity": 0.2,
                    "is_buyer_maker": True,
                }
            ]
        return []

    monkeypatch.setattr(MODULE, "fetch_aggtrades_batch", _fetch)

    rows = MODULE._collect_live_raw_rows(
        symbol="XAG/USDT",
        start_ms=0,
        end_ms=(2 * hour_ms) - 1,
        limit=1000,
        pause_sec=0.0,
    )

    assert calls[:2] == [0, hour_ms]
    assert [row["agg_trade_id"] for row in rows] == [10, 20]


def test_collect_live_raw_rows_skips_empty_windows_before_later_trades(monkeypatch) -> None:
    class _Exchange:
        def close(self):
            return None

    calls: list[int] = []
    hour_ms = 3_600_000

    monkeypatch.setattr(MODULE, "create_binance_futures_client", lambda **kwargs: _Exchange())
    monkeypatch.setattr(MODULE.time, "sleep", lambda *_args, **_kwargs: None)

    def _fetch(*, exchange, symbol, since_ms, limit, retries, base_wait_sec):
        _ = exchange, symbol, limit, retries, base_wait_sec
        calls.append(int(since_ms))
        if int(since_ms) == 0:
            return []
        if int(since_ms) == hour_ms:
            return [
                {
                    "agg_trade_id": 30,
                    "timestamp_ms": hour_ms + 500,
                    "price": 102.0,
                    "quantity": 0.3,
                    "is_buyer_maker": False,
                }
            ]
        return []

    monkeypatch.setattr(MODULE, "fetch_aggtrades_batch", _fetch)

    rows = MODULE._collect_live_raw_rows(
        symbol="XPT/USDT",
        start_ms=0,
        end_ms=(2 * hour_ms) - 1,
        limit=1000,
        pause_sec=0.0,
    )

    assert calls[:2] == [0, hour_ms]
    assert [row["agg_trade_id"] for row in rows] == [30]


def test_live_raw_batch_pause_seconds_defaults_to_zero(monkeypatch) -> None:
    monkeypatch.delenv("LQ_LIVE_RAW_BATCH_PAUSE_SEC", raising=False)

    assert MODULE._live_raw_batch_pause_seconds() == 0.0


def test_current_utc_day_skips_archive_probe(monkeypatch, tmp_path: Path) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    cutoff_dt = MODULE.parse_utc("2026-03-28T00:10:00Z")
    floor_dt = MODULE.parse_utc("2026-03-28T00:00:00Z")
    assert cutoff_dt is not None
    assert floor_dt is not None

    monkeypatch.setattr(MODULE, "_utc_today", lambda: date(2026, 3, 28))
    monkeypatch.setattr(
        MODULE,
        "_download_zip_bytes",
        lambda *args, **kwargs: pytest.fail("archive probe should be skipped"),
    )

    live_calls: list[dict[str, int]] = []

    def _collect_live_raw_rows(**kwargs):
        live_calls.append({"start_ms": int(kwargs["start_ms"]), "end_ms": int(kwargs["end_ms"])})
        return [
            {
                "agg_trade_id": 1,
                "timestamp_ms": int(kwargs["start_ms"]),
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ]

    monkeypatch.setattr(MODULE, "_collect_live_raw_rows", _collect_live_raw_rows)

    result = MODULE.refresh_symbol_raw_first_ohlcv(
        repo=repo,
        symbol="BTC/USDT",
        db_path=str(tmp_path),
        exchange_id="binance",
        cutoff_dt=cutoff_dt,
        floor_dt=floor_dt,
        guard=None,
    )

    assert result.source_mix == "live_only_recent_archive_cutover"
    assert result.archive_days_missing == 0
    assert result.live_tail_status == "fetched"
    assert live_calls and live_calls[0]["start_ms"] == int(floor_dt.timestamp() * 1000)


def test_unsupported_live_symbol_skips_live_tail_without_failing(
    monkeypatch, tmp_path: Path
) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    cutoff_dt = MODULE.parse_utc("2026-03-28T00:10:00Z")
    floor_dt = MODULE.parse_utc("2026-03-28T00:00:00Z")
    assert cutoff_dt is not None
    assert floor_dt is not None

    monkeypatch.setattr(MODULE, "_utc_today", lambda: date(2026, 3, 28))
    monkeypatch.setattr(
        MODULE,
        "_download_zip_bytes",
        lambda *args, **kwargs: pytest.fail("archive probe should be skipped"),
    )
    monkeypatch.setattr(MODULE, "_supports_live_raw_symbol", lambda _symbol: False)
    monkeypatch.setattr(
        MODULE,
        "_collect_live_raw_rows",
        lambda **_kwargs: pytest.fail("live fetch should be skipped"),
    )

    result = MODULE.refresh_symbol_raw_first_ohlcv(
        repo=repo,
        symbol="XAU/USD",
        db_path=str(tmp_path),
        exchange_id="binance",
        cutoff_dt=cutoff_dt,
        floor_dt=floor_dt,
        guard=None,
    )

    assert result.symbol == "XAU/USDT"
    assert result.source_mix == "noop_recent_archive_cutover"
    assert result.live_tail_status == "skipped_unsupported_symbol"
    assert result.live_raw_rows_fetched == 0
    assert result.live_raw_rows_upserted == 0


def test_invalid_symbol_error_from_live_fetch_falls_back_to_skip(
    monkeypatch, tmp_path: Path
) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    cutoff_dt = MODULE.parse_utc("2026-03-28T00:10:00Z")
    floor_dt = MODULE.parse_utc("2026-03-28T00:00:00Z")
    assert cutoff_dt is not None
    assert floor_dt is not None

    monkeypatch.setattr(MODULE, "_utc_today", lambda: date(2026, 3, 28))
    monkeypatch.setattr(MODULE, "_supports_live_raw_symbol", lambda _symbol: True)
    monkeypatch.setattr(
        MODULE,
        "_download_zip_bytes",
        lambda *args, **kwargs: pytest.fail("archive probe should be skipped"),
    )

    def _raise_invalid(**_kwargs):
        raise MODULE.LiveRawSymbolUnsupportedError("invalid symbol")

    monkeypatch.setattr(MODULE, "_collect_live_raw_rows", _raise_invalid)

    result = MODULE.refresh_symbol_raw_first_ohlcv(
        repo=repo,
        symbol="XPD/USD",
        db_path=str(tmp_path),
        exchange_id="binance",
        cutoff_dt=cutoff_dt,
        floor_dt=floor_dt,
        guard=None,
    )

    assert result.symbol == "XPD/USDT"
    assert result.source_mix == "noop_recent_archive_cutover"
    assert result.live_tail_status == "skipped_unsupported_symbol"
    assert result.live_raw_rows_fetched == 0


def test_prioritize_symbols_keeps_requested_cores_first() -> None:
    ordered = MODULE.prioritize_symbols(
        ["DOGE/USDT", "BNB/USDT", "BTC/USDT", "SOL/USDT"],
        priority_symbols=["BTC/USDT", "SOL/USDT", "ETH/USDT"],
    )

    assert ordered == ["BTC/USDT", "SOL/USDT", "DOGE/USDT", "BNB/USDT"]


def test_parse_symbol_tokens_canonicalizes_research_aliases() -> None:
    ordered = MODULE.parse_symbol_tokens("XAU/USD,XAU/USDT,BTCUSDT")

    assert ordered == ["XAU/USDT", "BTC/USDT"]


def test_order_symbols_for_parallel_refresh_dedupes_aliases(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "_supports_live_raw_symbol", lambda _symbol: True)

    ordered = MODULE._order_symbols_for_parallel_refresh(
        ["XAU/USD", "BTC/USDT", "XAU/USDT"],
        previous_costs={"BTC/USDT": 10.0, "XAU/USDT": 1.0},
    )

    assert ordered == ["BTC/USDT", "XAU/USDT"]


def test_order_symbols_for_parallel_refresh_uses_previous_costs_and_live_support(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_supports_live_raw_symbol",
        lambda symbol: symbol not in {"XAU/USDT"},
    )

    ordered = MODULE._order_symbols_for_parallel_refresh(
        ["ADA/USDT", "BTC/USDT", "XAU/USD", "ETH/USDT"],
        previous_costs={"ETH/USDT": 50.0, "BTC/USDT": 100.0, "ADA/USDT": 10.0},
    )

    assert ordered == ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XAU/USDT"]


def test_build_source_skew_summary_flags_unsupported_and_slowest_symbols() -> None:
    summary = MODULE._build_source_skew_summary(
        [
            MODULE.OhlcvRefreshResult(
                symbol="BTC/USDT",
                before_ohlcv_max_utc=None,
                after_ohlcv_max_utc=None,
                before_raw_agg_trade_utc=None,
                after_raw_agg_trade_utc=None,
                resume_start_utc=None,
                cutoff_utc="2026-03-28T04:00:00Z",
                archive_days_considered=0,
                archive_days_downloaded=0,
                archive_days_missing=0,
                archive_raw_rows_fetched=0,
                archive_raw_rows_upserted=0,
                live_raw_rows_fetched=1,
                live_raw_rows_upserted=1,
                live_tail_status="fetched",
                derived_ohlcv_rows_upserted=1,
                source_mix="live_only",
                stage_timings_seconds={"live_fetch": 12.0, "total_refresh": 13.0},
            ),
            MODULE.OhlcvRefreshResult(
                symbol="XAU/USDT",
                before_ohlcv_max_utc=None,
                after_ohlcv_max_utc=None,
                before_raw_agg_trade_utc=None,
                after_raw_agg_trade_utc=None,
                resume_start_utc=None,
                cutoff_utc="2026-03-28T04:00:00Z",
                archive_days_considered=0,
                archive_days_downloaded=0,
                archive_days_missing=1,
                archive_raw_rows_fetched=0,
                archive_raw_rows_upserted=0,
                live_raw_rows_fetched=0,
                live_raw_rows_upserted=0,
                live_tail_status="skipped_unsupported_symbol",
                derived_ohlcv_rows_upserted=0,
                source_mix="noop_recent_archive_cutover",
                stage_timings_seconds={"total_refresh": 0.5},
            ),
        ]
    )

    assert summary["unsupported_live_tail_symbols"] == ["XAU/USDT"]
    assert summary["symbols_with_live_tail"] == ["BTC/USDT"]
    assert summary["top_live_fetch_seconds"][0] == {"symbol": "BTC/USDT", "seconds": 12.0}


def test_estimate_parallel_workers_respects_memory_budget() -> None:
    workers = MODULE.estimate_parallel_workers(
        symbol_count=14,
        memory_budget_bytes=8 * 1024 * 1024 * 1024,
        reserve_memory_bytes=2 * 1024 * 1024 * 1024,
        per_worker_memory_bytes=int(1.5 * 1024 * 1024 * 1024),
        max_workers=8,
    )

    assert workers == 2


def test_resolve_effective_memory_budget_bytes_clamps_to_safe_session_cap(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "resolve_memory_budget_bytes", lambda: 32 * 1024 * 1024 * 1024)

    effective, system_budget = MODULE.resolve_effective_memory_budget_bytes(12 * 1024 * 1024 * 1024)

    assert effective == MODULE.DEFAULT_HEAVY_RUN_MEMORY_BUDGET_BYTES
    assert system_budget == 32 * 1024 * 1024 * 1024


def test_recent_archive_404_cuts_over_to_live_tail(monkeypatch, tmp_path: Path) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    cutoff_dt = MODULE.parse_utc("2025-01-03T00:00:02Z")
    floor_dt = MODULE.parse_utc("2025-01-03T00:00:00Z")
    assert cutoff_dt is not None
    assert floor_dt is not None

    monkeypatch.setenv("LQ_RECENT_ARCHIVE_LIVE_CUTOVER_DAYS", "3")
    monkeypatch.setenv("LQ_ARCHIVE_MISS_STREAK_FOR_LIVE_CUTOVER", "1")
    monkeypatch.setattr(MODULE, "_download_zip_bytes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        MODULE, "_binance_archive_url", lambda *args, **kwargs: "https://example.test"
    )

    live_calls: list[dict[str, int]] = []

    def _collect_live_raw_rows(**kwargs):
        live_calls.append({"start_ms": int(kwargs["start_ms"]), "end_ms": int(kwargs["end_ms"])})
        return [
            {
                "agg_trade_id": 7,
                "timestamp_ms": int(kwargs["start_ms"]),
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ]

    monkeypatch.setattr(MODULE, "_collect_live_raw_rows", _collect_live_raw_rows)

    result = MODULE.refresh_symbol_raw_first_ohlcv(
        repo=repo,
        symbol="BTC/USDT",
        db_path=str(tmp_path),
        exchange_id="binance",
        cutoff_dt=cutoff_dt,
        floor_dt=floor_dt,
        guard=None,
    )

    assert result.archive_days_missing == 1
    assert result.source_mix == "live_only_recent_archive_cutover"
    assert result.live_tail_status == "fetched"
    assert live_calls
    assert live_calls[0]["start_ms"] == int(floor_dt.timestamp() * 1000)


def test_refresh_payload_reports_backend(monkeypatch, tmp_path: Path) -> None:
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    rss_log = tmp_path / "rss.jsonl"
    inventory_json = tmp_path / "inventory.json"
    inventory_csv = tmp_path / "inventory.csv"
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"selected_team": []}), encoding="utf-8")

    class _Lease:
        lock_path = tmp_path / "session_memory_budget.lock"

        def release(self) -> None:
            return None

    monkeypatch.setattr(MODULE, "load_portfolio_symbols", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(MODULE, "load_feature_symbols", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(MODULE, "acquire_session_memory_lease", lambda **_kwargs: _Lease())
    monkeypatch.setattr(
        MODULE, "resolve_raw_aggtrades_backend_name", lambda *_args, **_kwargs: "python"
    )
    monkeypatch.setattr(
        MODULE,
        "raw_first_backend_diagnostics",
        lambda *_args, **_kwargs: {
            "requested_backend": "auto",
            "resolved_backend": "python",
            "description": "python",
            "native_library_path": None,
            "native_load_error": "test missing library",
            "auto_fallback_warning_count": 1,
            "auto_fallback_warning_reasons": ["test missing library"],
        },
    )

    exit_code = MODULE.main(
        [
            "--bundle-path",
            str(bundle),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--rss-log",
            str(rss_log),
            "--support-inventory-json",
            str(inventory_json),
            "--support-inventory-csv",
            str(inventory_csv),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["aggregation_backend_requested"] in {"auto", "python", "rust"}
    assert payload["aggregation_backend_resolved"] == "python"
    assert payload["aggregation_backend_diagnostics"]["native_load_error"] == "test missing library"


def test_refresh_payload_blocks_when_session_memory_lease_is_active(
    monkeypatch, tmp_path: Path
) -> None:
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    rss_log = tmp_path / "rss.jsonl"
    inventory_json = tmp_path / "inventory.json"
    inventory_csv = tmp_path / "inventory.csv"
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"selected_team": []}), encoding="utf-8")

    class _BusyLeaseError(MODULE.HeavyRunActiveError):
        def __init__(self) -> None:
            super().__init__(
                lock_path=tmp_path / "session_memory_budget.lock",
                active_payload={"pid": 4321, "run_id": "active-refresh"},
            )

    monkeypatch.setattr(MODULE, "load_portfolio_symbols", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(MODULE, "load_feature_symbols", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        MODULE,
        "acquire_session_memory_lease",
        lambda **_kwargs: (_ for _ in ()).throw(_BusyLeaseError()),
    )

    exit_code = MODULE.main(
        [
            "--bundle-path",
            str(bundle),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--rss-log",
            str(rss_log),
            "--support-inventory-json",
            str(inventory_json),
            "--support-inventory-csv",
            str(inventory_csv),
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "blocked_active_run"
    assert payload["session_memory_lease"]["status"] == "blocked"


def test_raw_checkpoint_utc_reads_incremental_raw_part_files(tmp_path: Path) -> None:
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
                "timestamp_ms": 1_735_689_601_500,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    latest = MODULE._raw_checkpoint_utc(
        repo,
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
    )

    assert latest is not None
    assert MODULE.iso_utc(latest) == "2025-01-01T00:00:01.500000Z"
