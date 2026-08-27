from __future__ import annotations

import argparse
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from lumina_quant.storage.wal.binary import BinaryWAL, WALRecord
from scripts.research import integrate_alpha_max_direct_db as subject


def _datetime(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).replace(tzinfo=None)


def _raw_frame(timestamp_ms: list[int], values: list[float] | None = None) -> pl.DataFrame:
    prices = values or [100.0] * len(timestamp_ms)
    return pl.DataFrame(
        {
            "datetime": [_datetime(value) for value in timestamp_ms],
            "open": prices,
            "high": [value + 1.0 for value in prices],
            "low": [value - 1.0 for value in prices],
            "close": [value + 0.5 for value in prices],
            "volume": [1.0] * len(timestamp_ms),
        },
        schema=subject.RAW_SCHEMA,
    )


def _funding_frame(timestamp_ms: list[int], rates: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp_ms": timestamp_ms,
            "source_timestamp_ms": timestamp_ms,
            "exchange": ["binance"] * len(timestamp_ms),
            "symbol": ["BTCUSDT"] * len(timestamp_ms),
            "funding_rate": rates,
        },
        schema={
            "timestamp_ms": pl.Int64,
            "source_timestamp_ms": pl.Int64,
            "exchange": pl.String,
            "symbol": pl.String,
            "funding_rate": pl.Float64,
        },
    )


def test_partial_raw_merge_preserves_rows_outside_contract(tmp_path: Path) -> None:
    nominal_start = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    nominal_end = int(datetime(2024, 2, 1, tzinfo=UTC).timestamp() * 1000)
    start_ms = nominal_start + 86_400_000
    end_ms = start_ms + 3_000
    source = tmp_path / "source.parquet"
    target = tmp_path / "target.parquet"
    _raw_frame([start_ms, start_ms + 1_000, start_ms + 2_000], [10.0, 11.0, 12.0]).write_parquet(
        source
    )
    _raw_frame(
        [start_ms - 1_000, start_ms + 1_000, end_ms + 1_000],
        [1.0, 999.0, 2.0],
    ).write_parquet(target)

    receipt = subject.merge_raw_contract_partition(
        source,
        target,
        symbol="BTCUSDT",
        month="2024-01",
        start_ms=start_ms,
        end_ms=end_ms,
        nominal_start_ms=nominal_start,
        nominal_end_ms=nominal_end,
    )

    merged = pl.read_parquet(target)
    assert merged.get_column("datetime").dt.epoch("ms").to_list() == [
        start_ms - 1_000,
        start_ms,
        start_ms + 1_000,
        start_ms + 2_000,
        end_ms + 1_000,
    ]
    assert merged.get_column("open").to_list() == [1.0, 10.0, 11.0, 12.0, 2.0]
    assert receipt["source_rows"] == 3
    assert receipt["preserved_rows"] == 2


def test_funding_publication_uses_canonical_feature_merge(tmp_path: Path) -> None:
    day_start = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    repository = subject.FeatureMarketDataRepository(str(candidate))
    repository.upsert_futures_feature_points(
        exchange="binance",
        symbol="BTCUSDT",
        rows=[
            {"timestamp_ms": day_start + 1_000, "funding_rate": -0.1},
            {"timestamp_ms": day_start + 8_000, "funding_rate": 9.9},
            {"timestamp_ms": day_start + 24_000, "funding_rate": 0.3},
        ],
    )
    subject.scrub_funding_features(
        candidate,
        symbol="BTCUSDT",
        start_ms=day_start + 8_000,
        end_ms=day_start + 24_000,
    )

    relative = Path(
        "feature_points/exchange=binance/symbol=BTCUSDT/date=2024-01-01/funding.parquet"
    )
    source = tmp_path / "source" / relative
    source.parent.mkdir(parents=True)
    _funding_frame([day_start + 8_000, day_start + 16_000], [0.1, 0.2]).write_parquet(source)
    report = tmp_path / "report"
    provenance = subject.acquisition.partition_path(report, relative.as_posix())
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps(
            {
                "schema": "alpha_max_partition_receipt.v2",
                "path": relative.as_posix(),
                "output_sha256": subject.file_sha256(source),
                "rows": 2,
            }
        )
    )

    receipt = subject.publish_funding_contract_file(
        source,
        relative=relative,
        source_report=report,
        repository=repository,
    )

    merged = repository.load_futures_feature_points(
        exchange="binance",
        symbol="BTCUSDT",
        start_date=day_start,
        end_date=day_start + 30_000,
    )
    assert merged.get_column("timestamp_ms").to_list() == [
        day_start + 1_000,
        day_start + 8_000,
        day_start + 16_000,
        day_start + 24_000,
    ]
    assert merged.get_column("funding_rate").to_list() == [-0.1, 0.1, 0.2, 0.3]
    assert receipt["source_rows"] == 2
    assert receipt["copy_mode"] == "canonical-feature-merge"


def test_scrub_funding_features_preserves_other_fields_and_outside_rows(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "feature_points"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "date=1970-01-01"
        / "compact.parquet"
    )
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp_ms": [-1, 0, 1],
            "funding_rate": [0.1, 0.2, 0.3],
            "funding_mark_price": [100.0, 101.0, 102.0],
            "mark_price": [99.0, 100.0, 101.0],
        },
        schema={
            "timestamp_ms": pl.Int64,
            "funding_rate": pl.Float64,
            "funding_mark_price": pl.Float64,
            "mark_price": pl.Float64,
        },
    ).write_parquet(path)

    receipt = subject.scrub_funding_features(
        tmp_path,
        symbol="BTCUSDT",
        start_ms=0,
        end_ms=1,
    )

    rewritten = pl.read_parquet(path)
    assert rewritten.get_column("funding_rate").to_list() == [0.1, None, 0.3]
    assert rewritten.get_column("funding_mark_price").to_list() == [100.0, None, 102.0]
    assert rewritten.get_column("mark_price").to_list() == [99.0, 100.0, 101.0]
    assert receipt == {"files_seen": 1, "files_rewritten": 1, "values_cleared": 2}


def test_wal_filter_retains_records_outside_contract(tmp_path: Path) -> None:
    symbol_root = tmp_path / "market_ohlcv_1s" / "binance" / "BTCUSDT"
    wal_path = symbol_root / "wal.bin"
    wal = BinaryWAL(wal_path, auto_repair=False)
    wal.append(
        [WALRecord(timestamp, 1.0, 2.0, 0.5, 1.5, 1.0) for timestamp in (1_000, 2_000, 3_000)]
    )
    (symbol_root / "compaction.meta.json").write_text(json.dumps({"wal_offset": 192}))

    receipt = subject.filter_wal_contract(
        tmp_path,
        symbol="BTCUSDT",
        start_ms=2_000,
        end_ms=3_000,
    )

    assert [record.ts_ms for record in BinaryWAL(wal_path, auto_repair=False).iter_all()] == [
        1_000,
        3_000,
    ]
    assert receipt == {"records_before": 3, "records_removed": 1, "records_after": 2}
    metadata = json.loads((symbol_root / "compaction.meta.json").read_text())
    assert metadata["wal_offset"] == 0
    assert metadata["g003_contract_rows_removed_from_wal"] == 1


def test_rename_exchange_swaps_complete_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "marker").write_text("old")
    (second / "marker").write_text("new")

    subject.rename_exchange(first, second)

    assert (first / "marker").read_text() == "new"
    assert (second / "marker").read_text() == "old"


def test_deep_audited_raw_seal_blocks_later_wal_mutation(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    target = candidate / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2024-01.parquet"
    source = tmp_path / "source" / target.relative_to(candidate)
    target.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    start_ms = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    frame = _raw_frame([start_ms, start_ms + 1_000, start_ms + 2_000])
    frame.write_parquet(target)
    frame.write_parquet(source)
    relative = target.relative_to(candidate)
    report = tmp_path / "report"
    provenance = subject.acquisition.partition_path(report, relative.as_posix())
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps(
            {
                "schema": "alpha_max_partition_receipt.v2",
                "path": relative.as_posix(),
                "output_sha256": subject.file_sha256(source),
                "rows": 3,
            }
        )
    )
    audit_item = subject.audit_raw_partition(
        target,
        symbol="BTCUSDT",
        month="2024-01",
        start_ms=start_ms,
        end_ms=start_ms + 3_000,
        deep=True,
    )
    prepared = {
        "raw_partitions": [
            {
                "symbol": "BTCUSDT",
                "month": "2024-01",
                "source": str(source),
                "target": str(target),
                "source_rows": 3,
                "sha256": subject.file_sha256(target),
                "byte_count": target.stat().st_size,
            }
        ]
    }

    seals = subject.seal_candidate_raw_partitions(
        prepared=prepared,
        candidate_audit={"raw": {"items": [audit_item]}},
        candidate_root=candidate,
        source_report=report,
    )

    assert len(seals) == 1
    assert target.with_suffix(".seal.json").is_file()
    repository = subject.ParquetMarketDataRepository(candidate)
    parsed_seal = repository._read_canonical_json(target.with_suffix(".seal.json"))
    assert parsed_seal["sha256"] == subject.file_sha256(target)
    assert (
        repository.publish_sealed_monthly_partition(
            exchange="binance",
            symbol="BTCUSDT",
            month="2024-01",
            source=target,
            expected_sha256=subject.file_sha256(target),
            expected_byte_count=target.stat().st_size,
            expected_row_count=3,
            provenance_receipt_sha256=subject.file_sha256(provenance),
        )
        == target
    )
    with pytest.raises(Exception, match="sealed monthly partition"):
        repository.upsert_1s(
            exchange="binance",
            symbol="BTCUSDT",
            rows=_raw_frame([start_ms + 3_000]),
        )


def _exact_audit(*, deep: bool = True) -> dict[str, object]:
    return {
        "status": "complete" if deep else "inventory-complete",
        "audit_mode": "deep" if deep else "inventory",
        "raw": {
            "target_partitions": subject.EXPECTED_RAW_PARTITIONS,
            "complete_partitions": subject.EXPECTED_RAW_PARTITIONS,
            "target_rows": subject.EXPECTED_RAW_ROWS,
            "complete_rows": subject.EXPECTED_RAW_ROWS,
        },
        "funding": {
            "target_rows": subject.EXPECTED_FUNDING_ROWS,
            "complete_rows": subject.EXPECTED_FUNDING_ROWS,
            "missing_rows": 0,
            "extra_rows_in_window": 0,
            "duplicate_rows_in_window": 0,
            "jitter_violation_rows": 0,
            "error_count": 0,
        },
    }


def test_activation_audit_failure_rolls_back_directory_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    source = tmp_path / "source"
    report = tmp_path / "report"
    receipts = tmp_path / "receipts"
    canonical.mkdir()
    source.mkdir()
    report.mkdir()
    (canonical / "marker").write_text("old")
    (report / "source_eligible_receipt.json").write_text("{}")
    (report / "source_manifest.json").write_text("{}")
    candidate = tmp_path / ".canonical.g003-candidate1"
    args = argparse.Namespace(
        contract=tmp_path / "contract.json",
        source_root=source,
        source_report=report,
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
        reserve_bytes=0,
    )
    canonical_json_reader = subject.ParquetMarketDataRepository._read_canonical_json

    class Repository:
        def __init__(self, _root: Path) -> None:
            pass

        def generation_lock(self, **_kwargs):
            return contextlib.nullcontext()

        _read_canonical_json = canonical_json_reader

    def prepare(**kwargs):
        subject._create_owned_candidate(
            candidate,
            preparation=kwargs["preparation"],
            preparation_intent_path=kwargs["preparation_intent_path"],
        )
        (candidate / "marker").write_text("new")
        return {
            "prepared": True,
            "raw_partition_count": subject.EXPECTED_RAW_PARTITIONS,
            "funding_partition_count": subject.EXPECTED_FUNDING_PARTITIONS,
        }

    audits = iter([_exact_audit(), {"status": "incomplete"}])
    monkeypatch.setattr(subject, "ParquetMarketDataRepository", Repository)
    monkeypatch.setattr(subject, "capacity_audit", lambda *_args, **_kwargs: {"passes": True})
    monkeypatch.setattr(subject, "prepare_candidate", prepare)
    monkeypatch.setattr(
        subject,
        "seal_candidate_raw_partitions",
        lambda **_kwargs: [{}] * subject.EXPECTED_RAW_PARTITIONS,
    )
    monkeypatch.setattr(subject, "sync_filesystem", lambda _path: None)
    monkeypatch.setattr(subject, "verify_raw_seal_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "audit_contract", lambda **_kwargs: next(audits))

    with pytest.raises(ValueError, match="active canonical contract audit failed"):
        subject._integrate_verified(args, contract={"records": []}, candidate_root=candidate)

    assert (canonical / "marker").read_text() == "old"
    assert (candidate / "marker").read_text() == "new"
    assert not (candidate / subject.CANDIDATE_OWNER_FILENAME).exists()
    failure = json.loads((receipts / "activation_failure.json").read_text())
    assert failure["exchanged"] is True
    assert failure["rollback_complete"] is True


def test_interrupted_preparation_is_proven_discarded_and_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    source = tmp_path / "source"
    report = tmp_path / "report"
    receipts = tmp_path / "receipts"
    canonical.mkdir()
    source.mkdir()
    report.mkdir()
    receipts.mkdir()
    (canonical / "marker").write_text("old")
    eligible_path = report / "source_eligible_receipt.json"
    manifest_path = report / "source_manifest.json"
    eligible_path.write_text("{}")
    manifest_path.write_text("{}")
    candidate = tmp_path / ".canonical.g003-candidate1"
    args = argparse.Namespace(
        contract=tmp_path / "contract.json",
        source_root=source,
        source_report=report,
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
        reserve_bytes=0,
    )
    source_receipt = {
        "eligible_path": str(eligible_path),
        "eligible_sha256": subject.file_sha256(eligible_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": subject.file_sha256(manifest_path),
    }
    old_preparation = subject._create_preparation_intent(
        args,
        candidate_root=candidate,
        canonical_before=subject.directory_identity(canonical),
        source_receipt=source_receipt,
    )
    subject._create_owned_candidate(
        candidate,
        preparation=old_preparation,
        preparation_intent_path=receipts / subject.PREPARATION_INTENT_FILENAME,
    )
    (candidate / "partial").write_text("must-not-be-reused")
    (candidate / "nested").mkdir()
    (candidate / "nested" / "payload").write_text("must-not-be-reused")
    (receipts / "candidate_inventory.json").write_text('{"stale":true}')
    fresh_nonces: list[str] = []
    canonical_json_reader = subject.ParquetMarketDataRepository._read_canonical_json

    class Repository:
        def __init__(self, _root: Path) -> None:
            pass

        def generation_lock(self, **_kwargs):
            return contextlib.nullcontext()

        _read_canonical_json = canonical_json_reader

    def prepare(**kwargs):
        assert not candidate.exists()
        fresh_nonces.append(kwargs["preparation"]["candidate_nonce"])
        subject._create_owned_candidate(
            candidate,
            preparation=kwargs["preparation"],
            preparation_intent_path=kwargs["preparation_intent_path"],
        )
        (candidate / "marker").write_text("fresh")
        return {
            "raw_partition_count": subject.EXPECTED_RAW_PARTITIONS,
            "funding_partition_count": subject.EXPECTED_FUNDING_PARTITIONS,
        }

    audits = iter([_exact_audit(), _exact_audit(deep=False)])
    monkeypatch.setattr(subject, "ParquetMarketDataRepository", Repository)
    monkeypatch.setattr(subject, "capacity_audit", lambda *_args, **_kwargs: {"passes": True})
    monkeypatch.setattr(subject, "prepare_candidate", prepare)
    monkeypatch.setattr(
        subject,
        "seal_candidate_raw_partitions",
        lambda **_kwargs: [{}] * subject.EXPECTED_RAW_PARTITIONS,
    )
    monkeypatch.setattr(subject, "sync_filesystem", lambda _path: None)
    monkeypatch.setattr(subject, "verify_raw_seal_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "audit_contract", lambda **_kwargs: next(audits))

    activation = subject._integrate_verified(
        args,
        contract={"records": []},
        candidate_root=candidate,
    )

    assert activation["raw_rows"] == subject.EXPECTED_RAW_ROWS
    assert (canonical / "marker").read_text() == "fresh"
    assert not (canonical / "partial").exists()
    assert fresh_nonces and fresh_nonces[0] != old_preparation["candidate_nonce"]
    recovery = subject.ParquetMarketDataRepository._read_canonical_json(
        receipts / "preparation_recovery.json"
    )
    assert recovery["candidate_nonce"] == old_preparation["candidate_nonce"]
    assert recovery["rebuild_required"] is True
    assert "candidate_inventory.json" in recovery["removed_stale_artifacts"]
    quarantine = Path(recovery["quarantine_path"])
    assert recovery["quarantine_retained"] is True
    assert {path.name for path in quarantine.iterdir()} == {subject.CANDIDATE_OWNER_FILENAME}
    assert not (canonical / subject.CANDIDATE_OWNER_FILENAME).exists()
    assert not (candidate / subject.CANDIDATE_OWNER_FILENAME).exists()


def test_unowned_preseeded_candidate_is_rejected_without_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    source = tmp_path / "source"
    report = tmp_path / "report"
    receipts = tmp_path / "receipts"
    candidate = tmp_path / ".canonical.g003-candidate1"
    for path in (canonical, source, report, candidate):
        path.mkdir()
    (canonical / "marker").write_text("old")
    (candidate / "attacker").write_text("preserve")
    (report / "source_eligible_receipt.json").write_text("{}")
    (report / "source_manifest.json").write_text("{}")
    args = argparse.Namespace(
        contract=tmp_path / "contract.json",
        source_root=source,
        source_report=report,
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
        reserve_bytes=0,
    )

    class Repository:
        def __init__(self, _root: Path) -> None:
            pass

        def generation_lock(self, **_kwargs):
            return contextlib.nullcontext()

    monkeypatch.setattr(subject, "ParquetMarketDataRepository", Repository)

    with pytest.raises(ValueError, match="proven preparation intent"):
        subject._integrate_verified(args, contract={"records": []}, candidate_root=candidate)

    assert (candidate / "attacker").read_text() == "preserve"
    assert (canonical / "marker").read_text() == "old"


def test_candidate_final_name_is_published_only_after_owner_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    receipts = tmp_path / "receipts"
    candidate = tmp_path / "candidate"
    canonical.mkdir()
    receipts.mkdir()
    args = argparse.Namespace(
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
    )
    source_receipt = {
        "eligible_path": "/source/source_eligible_receipt.json",
        "eligible_sha256": "a" * 64,
        "manifest_path": "/source/source_manifest.json",
        "manifest_sha256": "b" * 64,
    }
    preparation = subject._create_preparation_intent(
        args,
        candidate_root=candidate,
        canonical_before=subject.directory_identity(canonical),
        source_receipt=source_receipt,
    )
    intent_path = receipts / subject.PREPARATION_INTENT_FILENAME
    real_write_owner = subject._write_candidate_owner

    def fail_before_owner(*_args, **_kwargs):
        raise OSError("injected owner write failure")

    monkeypatch.setattr(subject, "_write_candidate_owner", fail_before_owner)
    with pytest.raises(OSError, match="injected owner write failure"):
        subject._create_owned_candidate(
            candidate,
            preparation=preparation,
            preparation_intent_path=intent_path,
        )

    assert not candidate.exists()
    assert list(tmp_path.glob("candidate.*.g003-preparing"))
    monkeypatch.setattr(subject, "_write_candidate_owner", real_write_owner)

    subject._create_owned_candidate(
        candidate,
        preparation=preparation,
        preparation_intent_path=intent_path,
    )

    owner = subject._validate_candidate_owner(
        candidate,
        candidate_root=candidate,
        preparation=preparation,
        preparation_intent_path=intent_path,
    )
    assert owner["candidate_nonce"] == preparation["candidate_nonce"]


def test_quarantine_path_swap_aborts_without_deleting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    receipts = tmp_path / "receipts"
    candidate = tmp_path / "candidate"
    canonical.mkdir()
    receipts.mkdir()
    args = argparse.Namespace(
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
    )
    source_receipt = {
        "eligible_path": "/source/source_eligible_receipt.json",
        "eligible_sha256": "a" * 64,
        "manifest_path": "/source/source_manifest.json",
        "manifest_sha256": "b" * 64,
    }
    canonical_before = subject.directory_identity(canonical)
    preparation = subject._create_preparation_intent(
        args,
        candidate_root=candidate,
        canonical_before=canonical_before,
        source_receipt=source_receipt,
    )
    intent_path = receipts / subject.PREPARATION_INTENT_FILENAME
    subject._create_owned_candidate(
        candidate,
        preparation=preparation,
        preparation_intent_path=intent_path,
    )
    (candidate / "partial").write_text("owned")
    real_retire = subject._retire_owned_quarantine
    swapped: dict[str, Path] = {}

    def swap_before_retirement(
        quarantine: Path,
        *,
        expected_identity: dict[str, int],
        expected_marker_sha256: str,
    ) -> None:
        moved = quarantine.with_name(f"{quarantine.name}.proven")
        quarantine.rename(moved)
        quarantine.mkdir()
        (quarantine / "user-work").write_text("preserve")
        swapped.update({"replacement": quarantine, "proven": moved})
        real_retire(
            quarantine,
            expected_identity=expected_identity,
            expected_marker_sha256=expected_marker_sha256,
        )

    monkeypatch.setattr(subject, "_retire_owned_quarantine", swap_before_retirement)

    with pytest.raises(ValueError, match="quarantine changed before descriptor pin"):
        subject._discard_interrupted_preparation(
            args,
            candidate_root=candidate,
            canonical_before=canonical_before,
            source_receipt=source_receipt,
        )

    assert (swapped["replacement"] / "user-work").read_text() == "preserve"
    assert (swapped["proven"] / "partial").read_text() == "owned"
    assert (swapped["proven"] / subject.CANDIDATE_OWNER_FILENAME).is_file()


def test_clone_fallback_copies_children_without_changing_owned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    receipts = tmp_path / "receipts"
    candidate = tmp_path / "candidate"
    canonical.mkdir()
    receipts.mkdir()
    (canonical / "payload").write_text("old")
    args = argparse.Namespace(
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
    )
    source_receipt = {
        "eligible_path": "/source/source_eligible_receipt.json",
        "eligible_sha256": "a" * 64,
        "manifest_path": "/source/source_manifest.json",
        "manifest_sha256": "b" * 64,
    }
    preparation = subject._create_preparation_intent(
        args,
        candidate_root=candidate,
        canonical_before=subject.directory_identity(canonical),
        source_receipt=source_receipt,
    )
    intent_path = receipts / subject.PREPARATION_INTENT_FILENAME
    monkeypatch.setattr(
        subject,
        "_supports_tree_reflink",
        lambda _candidate: (False, "unsupported"),
    )

    assert (
        subject.clone_canonical_tree(
            canonical,
            candidate,
            preparation=preparation,
            preparation_intent_path=intent_path,
        )
        == "copy-fallback"
    )
    assert subject.stat.S_IMODE(candidate.stat().st_mode) == 0o700
    assert (candidate / "payload").read_text() == "old"
    command = subject._candidate_copy_command(canonical, candidate, "always")
    assert f"{canonical}/." not in command
    assert str(canonical / "payload") in command
    subject._validate_candidate_owner(
        candidate,
        candidate_root=candidate,
        preparation=preparation,
        preparation_intent_path=intent_path,
    )


def test_reflink_probe_fails_closed_and_removes_owned_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    (candidate / subject.CANDIDATE_OWNER_FILENAME).write_text("owner")

    def unsupported(*_args):
        raise OSError(subject.errno.EOPNOTSUPP, "unsupported")

    monkeypatch.setattr(subject.fcntl, "ioctl", unsupported)

    supported, diagnostic = subject._supports_tree_reflink(candidate)

    assert supported is False
    assert "unsupported" in diagnostic
    assert [path.name for path in candidate.iterdir()] == [subject.CANDIDATE_OWNER_FILENAME]


def test_reflink_probe_rejects_false_success_and_removes_owned_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    (candidate / subject.CANDIDATE_OWNER_FILENAME).write_text("owner")
    monkeypatch.setattr(subject.fcntl, "ioctl", lambda *_args: None)

    with pytest.raises(ValueError, match="reflink probe content mismatch"):
        subject._supports_tree_reflink(candidate)

    assert [path.name for path in candidate.iterdir()] == [subject.CANDIDATE_OWNER_FILENAME]


def test_clone_fallback_path_swap_aborts_before_recursive_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    receipts = tmp_path / "receipts"
    candidate = tmp_path / "candidate"
    canonical.mkdir()
    receipts.mkdir()
    (canonical / "marker").write_text("old")
    args = argparse.Namespace(
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
    )
    source_receipt = {
        "eligible_path": "/source/source_eligible_receipt.json",
        "eligible_sha256": "a" * 64,
        "manifest_path": "/source/source_manifest.json",
        "manifest_sha256": "b" * 64,
    }
    preparation = subject._create_preparation_intent(
        args,
        candidate_root=candidate,
        canonical_before=subject.directory_identity(canonical),
        source_receipt=source_receipt,
    )
    intent_path = receipts / subject.PREPARATION_INTENT_FILENAME
    calls = 0

    def fail_reflink(*args, **_kwargs):
        nonlocal calls
        calls += 1
        return subject.subprocess.CompletedProcess(args, 1, "", "forced reflink failure")

    real_retire = subject._retire_owned_quarantine
    swapped: dict[str, Path] = {}

    def swap_before_retirement(
        quarantine: Path,
        *,
        expected_identity: dict[str, int],
        expected_marker_sha256: str,
    ) -> None:
        moved = quarantine.with_name(f"{quarantine.name}.proven")
        quarantine.rename(moved)
        quarantine.mkdir()
        (quarantine / "user-work").write_text("preserve")
        swapped.update({"replacement": quarantine, "proven": moved})
        real_retire(
            quarantine,
            expected_identity=expected_identity,
            expected_marker_sha256=expected_marker_sha256,
        )

    monkeypatch.setattr(subject.subprocess, "run", fail_reflink)
    monkeypatch.setattr(
        subject,
        "_supports_tree_reflink",
        lambda _candidate: (True, ""),
    )
    monkeypatch.setattr(subject, "_retire_owned_quarantine", swap_before_retirement)

    with pytest.raises(ValueError, match="quarantine changed before descriptor pin"):
        subject.clone_canonical_tree(
            canonical,
            candidate,
            preparation=preparation,
            preparation_intent_path=intent_path,
        )

    assert calls == 1
    assert (swapped["replacement"] / "user-work").read_text() == "preserve"
    assert (swapped["proven"] / subject.CANDIDATE_OWNER_FILENAME).is_file()


@pytest.mark.parametrize("failure_name", ["active_inventory.json", "activation_receipt.json"])
def test_receipt_write_failure_rolls_back_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_name: str,
) -> None:
    canonical = tmp_path / "canonical"
    candidate = tmp_path / "candidate"
    receipts = tmp_path / "receipts"
    canonical.mkdir()
    candidate.mkdir()
    receipts.mkdir()
    (canonical / "marker").write_text("old")
    (candidate / "marker").write_text("new")
    candidate_audit_path = receipts / "candidate_inventory.json"
    intent_path = receipts / "activation_intent.json"
    candidate_audit_path.write_text("{}")
    intent_path.write_text("{}")
    args = argparse.Namespace(
        contract=tmp_path / "contract.json",
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
    )
    canonical_before = subject.directory_identity(canonical)
    candidate_identity = subject.directory_identity(candidate)
    real_atomic_json = subject.atomic_json

    def failing_atomic_json(path: Path, payload: dict) -> None:
        if path.name == failure_name:
            raise OSError(f"injected {failure_name} failure")
        real_atomic_json(path, payload)

    monkeypatch.setattr(subject, "sync_filesystem", lambda _path: None)
    monkeypatch.setattr(subject, "audit_contract", lambda **_kwargs: _exact_audit(deep=False))
    monkeypatch.setattr(subject, "atomic_json", failing_atomic_json)

    with pytest.raises(OSError, match="injected"):
        subject._activate_candidate_transaction(
            args,
            candidate_root=candidate,
            canonical_before=canonical_before,
            candidate_identity=candidate_identity,
            source_receipt={"eligible_sha256": "a", "manifest_sha256": "b"},
            capacity={"passes": True},
            free_after_prepare=1,
            candidate_audit_path=candidate_audit_path,
            intent_path=intent_path,
        )

    assert (canonical / "marker").read_text() == "old"
    assert (candidate / "marker").read_text() == "new"
    failure = json.loads((receipts / "activation_failure.json").read_text())
    assert failure["rollback_complete"] is True


@pytest.mark.parametrize("already_exchanged", [False, True])
def test_interrupted_intent_resumes_pre_or_post_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_exchanged: bool,
) -> None:
    canonical = tmp_path / "canonical"
    candidate = tmp_path / "candidate"
    receipts = tmp_path / "receipts"
    canonical.mkdir()
    candidate.mkdir()
    receipts.mkdir()
    (canonical / "marker").write_text("old")
    (candidate / "marker").write_text("new")
    args = argparse.Namespace(
        contract=tmp_path / "contract.json",
        canonical_root=canonical,
        receipt_dir=receipts,
        candidate_id="candidate1",
    )
    source_receipt = {
        "eligible_path": "/source/source_eligible_receipt.json",
        "eligible_sha256": "a" * 64,
        "manifest_path": "/source/source_manifest.json",
        "manifest_sha256": "b" * 64,
    }
    canonical_before = subject.directory_identity(canonical)
    preparation = {
        "artifact_kind": "alpha_max_direct_db_preparation_intent",
        "schema": "alpha_max_direct_db_preparation_intent.v1",
        "phase": "building",
        "candidate_id": args.candidate_id,
        "candidate_nonce": "c" * 32,
        "canonical_root": str(canonical),
        "candidate_root": str(candidate),
        "canonical_before": canonical_before,
        "contract_sha256": subject.CONTRACT_SHA256,
        "source": source_receipt,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    preparation_intent_path = receipts / subject.PREPARATION_INTENT_FILENAME
    subject.atomic_json(preparation_intent_path, preparation, canonical=True)
    subject._write_candidate_owner(
        candidate,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    candidate_identity = subject.directory_identity(candidate)
    candidate_audit_path = receipts / "candidate_inventory.json"
    candidate_audit_path.write_text(json.dumps(_exact_audit()))
    prepared = {
        "raw_partition_count": subject.EXPECTED_RAW_PARTITIONS,
        "funding_partition_count": subject.EXPECTED_FUNDING_PARTITIONS,
        "raw_seal_count": subject.EXPECTED_RAW_PARTITIONS,
    }
    subject._write_preparation_ready(
        args,
        candidate_root=candidate,
        candidate_identity=candidate_identity,
        preparation=preparation,
        prepared=prepared,
        candidate_audit_path=candidate_audit_path,
    )
    preparation_ready_path = receipts / subject.PREPARATION_READY_FILENAME
    intent = {
        "artifact_kind": "alpha_max_direct_db_activation_intent",
        "candidate_id": args.candidate_id,
        "candidate_nonce": preparation["candidate_nonce"],
        "canonical_root": str(canonical),
        "candidate_root": str(candidate),
        "canonical_before": canonical_before,
        "candidate_before": candidate_identity,
        "contract_sha256": subject.CONTRACT_SHA256,
        "source": source_receipt,
        "capacity": {"passes": True},
        "free_after_prepare": 1,
        "prepared": prepared,
        "preparation_intent_sha256": subject.file_sha256(preparation_intent_path),
        "preparation_ready_sha256": subject.file_sha256(preparation_ready_path),
        "candidate_audit_sha256": subject.file_sha256(candidate_audit_path),
    }
    subject.atomic_json(receipts / "activation_intent.json", intent)
    if already_exchanged:
        subject.rename_exchange(canonical, candidate)

    audits = iter([_exact_audit(), _exact_audit(deep=False)])
    monkeypatch.setattr(subject, "audit_contract", lambda **_kwargs: next(audits))
    monkeypatch.setattr(subject, "verify_raw_seal_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "sync_filesystem", lambda _path: None)

    activation = subject._recover_existing_candidate(
        args,
        candidate_root=candidate,
        source_receipt=source_receipt,
    )

    assert activation["raw_rows"] == subject.EXPECTED_RAW_ROWS
    assert (canonical / "marker").read_text() == "new"
    assert (candidate / "marker").read_text() == "old"
    assert not (canonical / subject.CANDIDATE_OWNER_FILENAME).exists()
    assert not (candidate / subject.CANDIDATE_OWNER_FILENAME).exists()
    assert (
        subject._recover_existing_candidate(
            args,
            candidate_root=candidate,
            source_receipt=source_receipt,
        )
        == activation
    )
