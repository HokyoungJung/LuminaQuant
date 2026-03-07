from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.storage.parquet import ParquetMarketDataRepository
from lumina_quant.services.materialize_from_raw import MaterializedCommit

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "materialize_market_windows.py"
_SPEC = importlib.util.spec_from_file_location("materialize_script_module", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load materialize script module from {_SCRIPT_PATH}")
materialize_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(materialize_script)


def test_materializer_periodic_loop_resolves_required_timeframes(monkeypatch):
    captured: dict[str, object] = {}

    def _bundle(**kwargs):
        captured.update(kwargs)
        return {
            "1s": [
                MaterializedCommit(
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1s",
                    partition="2026-01-01",
                    commit_id="c-1s",
                    row_count=1,
                    canonical_row_checksum="abc",
                    manifest_path="manifest-1s.json",
                )
            ],
            "5m": [
                MaterializedCommit(
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="5m",
                    partition="2026-01-01",
                    commit_id="c-5m",
                    row_count=1,
                    canonical_row_checksum="def",
                    manifest_path="manifest-5m.json",
                )
            ],
            "1h": [
                MaterializedCommit(
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1h",
                    partition="2026-01-01",
                    commit_id="c-1h",
                    row_count=1,
                    canonical_row_checksum="ghi",
                    manifest_path="manifest-1h.json",
                )
            ],
        }

    monkeypatch.setattr(materialize_script, "materialize_raw_aggtrades_bundle", _bundle)

    cycles = materialize_script.run_materializer_periodic_loop(
        db_path="data/market_parquet",
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=materialize_script._parse_timeframes("5m,1s,1h,5m"),
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
        periodic_enabled=False,
        poll_seconds=5,
    )

    assert len(cycles) == 1
    assert captured["timeframes"] == ["1s", "5m", "1h"]


def test_materializer_skips_commit_publication_when_bundle_incomplete(monkeypatch):
    def _bundle(**kwargs):
        _ = kwargs
        raise RawFirstDataMissingError("missing BTC/USDT:5m")

    monkeypatch.setattr(materialize_script, "materialize_raw_aggtrades_bundle", _bundle)

    payload = materialize_script.run_materializer_cycle(
        db_path="data/market_parquet",
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=["1s", "5m"],
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
    )

    assert payload["success"] is False
    symbol_rows = payload["symbols"]
    assert isinstance(symbol_rows, list) and len(symbol_rows) == 1
    assert symbol_rows[0]["status"] == "skipped_incomplete_required_timeframes"
    assert "missing BTC/USDT:5m" in str(symbol_rows[0]["error"])


def _manifest_payload(*, anchor_dt: datetime) -> dict[str, object]:
    checkpoint_end_ms = int(anchor_dt.timestamp() * 1000)
    watermark_ms = (checkpoint_end_ms // 1000) * 1000
    return {
        "manifest_version": 1,
        "commit_id": "commit-1s",
        "symbol": "BTC/USDT",
        "timeframe": "1s",
        "partition": "market_data_materialized/binance/BTC_USDT/timeframe=1s/date=2026-03-07",
        "window_start_ms": watermark_ms,
        "window_end_ms": watermark_ms,
        "event_time_watermark_ms": watermark_ms,
        "source_checkpoint_start": checkpoint_end_ms,
        "source_checkpoint_end": checkpoint_end_ms,
        "row_count": 1,
        "canonical_row_checksum": "abc",
        "data_files": ["commit=commit-1s/part-0000.parquet"],
        "bundle_boundary_id": "bundle-1",
        "created_at_utc": anchor_dt.isoformat(),
        "producer": "test",
        "status": "committed",
    }


def _bundle_result_for_timeframes(timeframes: list[str]) -> dict[str, list[MaterializedCommit]]:
    return {
        timeframe: [
            MaterializedCommit(
                exchange="binance",
                symbol="BTC/USDT",
                timeframe=str(timeframe),
                partition="2026-03-07",
                commit_id=f"commit-{timeframe}",
                row_count=1,
                canonical_row_checksum=f"checksum-{timeframe}",
                manifest_path=f"manifest-{timeframe}.json",
            )
        ]
        for timeframe in list(timeframes)
    }


def test_materializer_cycle_uses_partition_safe_incremental_window(tmp_path, monkeypatch):
    repo = ParquetMarketDataRepository(str(tmp_path))
    anchor_dt = datetime(2026, 3, 7, 10, 3, 17, 900_000, tzinfo=UTC)
    repo.write_materialized_manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date="2026-03-07",
        payload=_manifest_payload(anchor_dt=anchor_dt),
    )

    captured: dict[str, object] = {}

    def _bundle(**kwargs):
        captured.update(kwargs)
        return _bundle_result_for_timeframes(["1s", "5m", "1d"])

    monkeypatch.setattr(materialize_script, "materialize_raw_aggtrades_bundle", _bundle)

    payload = materialize_script.run_materializer_cycle(
        db_path=str(tmp_path),
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=["1s", "5m"],
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
    )

    assert payload["success"] is True
    assert captured["start_date"] == "2026-03-07T00:00:00Z"
    assert captured["end_date"] is None


def test_materializer_cycle_respects_explicit_window_over_incremental_state(tmp_path, monkeypatch):
    repo = ParquetMarketDataRepository(str(tmp_path))
    anchor_dt = datetime(2026, 3, 7, 10, 3, 17, 900_000, tzinfo=UTC)
    repo.write_materialized_manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date="2026-03-07",
        payload=_manifest_payload(anchor_dt=anchor_dt),
    )

    captured: dict[str, object] = {}

    def _bundle(**kwargs):
        captured.update(kwargs)
        return _bundle_result_for_timeframes(["1s", "5m"])

    monkeypatch.setattr(materialize_script, "materialize_raw_aggtrades_bundle", _bundle)

    payload = materialize_script.run_materializer_cycle(
        db_path=str(tmp_path),
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=["1s", "5m"],
        base_timeframe="1s",
        start_date="2026-03-01T00:00:00Z",
        end_date=None,
        producer="test",
    )

    assert payload["success"] is True
    assert captured["start_date"] == "2026-03-01T00:00:00Z"


def test_materializer_cycle_supports_full_rebuild_mode(tmp_path, monkeypatch):
    repo = ParquetMarketDataRepository(str(tmp_path))
    anchor_dt = datetime(2026, 3, 7, 10, 3, 17, 900_000, tzinfo=UTC)
    repo.write_materialized_manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date="2026-03-07",
        payload=_manifest_payload(anchor_dt=anchor_dt),
    )

    captured: dict[str, object] = {}

    def _bundle(**kwargs):
        captured.update(kwargs)
        return _bundle_result_for_timeframes(["1s", "5m"])

    monkeypatch.setattr(materialize_script, "materialize_raw_aggtrades_bundle", _bundle)

    payload = materialize_script.run_materializer_cycle(
        db_path=str(tmp_path),
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=["1s", "5m"],
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
        incremental_window=False,
    )

    assert payload["success"] is True
    assert captured["start_date"] is None
    assert captured["end_date"] is None


def test_materializer_incremental_cycle_keeps_earlier_same_day_rows(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": int(datetime(2026, 3, 7, 0, 1, 0, 100_000, tzinfo=UTC).timestamp() * 1000),
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": int(datetime(2026, 3, 7, 10, 0, 0, 100_000, tzinfo=UTC).timestamp() * 1000),
                "price": 110.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
        ],
    )

    first = materialize_script.run_materializer_cycle(
        db_path=str(tmp_path),
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=["1s", "5m"],
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
    )
    assert first["success"] is True

    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 3,
                "timestamp_ms": int(datetime(2026, 3, 7, 10, 5, 0, 100_000, tzinfo=UTC).timestamp() * 1000),
                "price": 120.0,
                "quantity": 0.3,
                "is_buyer_maker": False,
            }
        ],
    )

    second = materialize_script.run_materializer_cycle(
        db_path=str(tmp_path),
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=["1s", "5m"],
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
    )
    assert second["success"] is True

    frame_1s = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
    )
    timestamps = {value.isoformat() for value in frame_1s["datetime"].to_list()}
    assert datetime(2026, 3, 7, 0, 1, 0).isoformat() in timestamps
    assert datetime(2026, 3, 7, 10, 5, 0).isoformat() in timestamps
