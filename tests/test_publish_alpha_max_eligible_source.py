from __future__ import annotations

import importlib.util
import argparse
import base64
import hashlib
import json
import math
from datetime import datetime
import threading
import time
from pathlib import Path

import pytest
import polars as pl
from lumina_quant import market_data
from lumina_quant.storage.parquet import ParquetMarketDataRepository
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


_SCRIPT = Path(__file__).parents[1] / "scripts/research/publish_alpha_max_eligible_source.py"
_spec = importlib.util.spec_from_file_location("publish_alpha_max_eligible_source", _SCRIPT)
assert _spec is not None and _spec.loader is not None
subject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(subject)


def test_canonical_bytes_are_stable():
    assert subject.canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'


def test_clean_absolute_path_rejects_relative_and_traversal():
    with pytest.raises(subject.PublicationError):
        subject._absolute("relative", "source_root")
    with pytest.raises(subject.PublicationError):
        subject._absolute("/tmp/../unsafe", "source_root")


def test_regular_json_reads_are_bounded_and_reject_hardlinks(tmp_path):
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (subject._MAX_JSON + 1))
    with pytest.raises(subject.PublicationError, match="bounded"):
        subject._regular_bytes(oversized, "oversized")
    private = tmp_path / "private.json"
    private.write_bytes(b"{}")
    linked = tmp_path / "linked.json"
    linked.hardlink_to(private)
    with pytest.raises(subject.PublicationError, match="private"):
        subject._regular_bytes(private, "linked")


def test_conflict_public_key_rejects_path_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key = tmp_path / "authority.pub"
    replacement = tmp_path / "replacement.pub"
    key.write_bytes(b"a" * 32)
    replacement.write_bytes(b"b" * 32)
    key.chmod(0o400)
    replacement.chmod(0o400)
    original_revalidate = subject._revalidate
    swapped = False

    def swap_before_revalidate(path, fd, expected, label):
        nonlocal swapped
        if not swapped:
            replacement.replace(path)
            swapped = True
        original_revalidate(path, fd, expected, label)

    monkeypatch.setattr(subject, "_revalidate", swap_before_revalidate)
    with pytest.raises(subject.PublicationError, match="identity drift"):
        subject._conflict_public_key(key)


def test_write_noreplace_has_exact_idempotent_readback(tmp_path):
    path = tmp_path / "receipt.json"
    value = {"schema": "test", "count": 1}
    subject._write_noreplace(path, value)
    assert path.read_bytes() == subject.canonical_bytes(value)
    subject._write_noreplace(path, value)
    with pytest.raises(subject.PublicationError, match="conflict"):
        subject._write_noreplace(path, {"schema": "test", "count": 2})


def test_clone_root_revalidates_bound_hardlinks_and_rejects_extras(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "nested").mkdir()
    source = old / "nested" / "data.bin"
    source.write_bytes(b"source")
    lock_parent = old / "feature_points" / "exchange=binance" / "symbol=BTCUSDT" / "date=2026-04-01"
    lock_parent.mkdir(parents=True)
    coordination_lock = lock_parent / ".writer.lock"
    coordination_lock.touch(mode=0o600)
    ordinary_lock = old / "nested" / ".writer.lock"
    ordinary_lock.touch(mode=0o600)
    expected = subject._inventory(old)
    candidate = tmp_path / "candidate"
    subject._clone_root(old, candidate, expected)
    assert (source.stat().st_dev, source.stat().st_ino) == (
        (candidate / "nested" / "data.bin").stat().st_dev,
        (candidate / "nested" / "data.bin").stat().st_ino,
    )
    assert source.stat().st_nlink == 2
    assert coordination_lock.stat().st_nlink == 1
    assert not (
        candidate
        / "feature_points"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "date=2026-04-01"
        / ".writer.lock"
    ).exists()
    assert (candidate / "nested" / ".writer.lock").exists()
    assert ordinary_lock.stat().st_nlink == 2
    (candidate / "extra").write_bytes(b"x")
    with pytest.raises(subject.PublicationError, match="unbound"):
        subject._remove_bound_tree(
            candidate, (candidate.stat().st_dev, candidate.stat().st_ino), expected
        )


def test_remove_bound_tree_resumes_from_authenticated_subset(tmp_path: Path):
    retired = tmp_path / "retired"
    retired.mkdir()
    first = retired / "first.bin"
    first.write_bytes(b"first")
    (retired / "second.bin").write_bytes(b"second")
    expected = subject._inventory(retired)
    identity = (retired.stat().st_dev, retired.stat().st_ino)
    first.unlink()

    subject._remove_bound_tree(retired, identity, expected)

    assert not retired.exists()


def test_bounded_feature_writer_does_not_create_oversized_output(tmp_path: Path):
    frame = pl.DataFrame({"timestamp_ms": [1], "funding_rate": [0.001]})
    output = tmp_path / "compact.parquet"

    with pytest.raises(ValueError, match="publication quota"):
        market_data._atomic_feature_write(
            tmp_path,
            output,
            frame,
            max_output_bytes=1,
        )

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_publication_admission_reserves_full_merged_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    candidate = tmp_path / "candidate"
    target = candidate / "market_ohlcv_1s/binance/BTCUSDT/2026-04.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * 101)
    item = {
        "relative": "market_ohlcv_1s/binance/BTCUSDT/2026-04.parquet",
        "source": tmp_path / "source.parquet",
    }
    monkeypatch.setitem(subject._STORAGE, "host_reserve_bytes", 1_000)
    monkeypatch.setattr(subject, "_parquet_materialized_bytes", lambda _paths: 500)
    output_bound = 500 + subject._PARQUET_ENCODING_RESERVE_BYTES
    required = 1_000 + 211 + output_bound + subject._PUBLICATION_CONTROL_RESERVE_BYTES
    monkeypatch.setattr(subject, "_host_free_bytes", lambda: required)
    admission = subject._publication_admission(candidate, item, 211)
    assert admission["required_free_bytes"] == required
    assert admission["temporary_merged_output_upper_bound"] == output_bound
    assert admission["immutable_source_pin_bytes"] == 211

    monkeypatch.setattr(subject, "_host_free_bytes", lambda: required - 1)
    with pytest.raises(subject.PublicationError, match="publication peak"):
        subject._publication_admission(candidate, item, 211)


def test_prepared_resume_accepts_only_authenticated_hardlink_break(tmp_path: Path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "partition").mkdir()
    original = old / "partition/data.parquet"
    original.write_bytes(b"old")
    untouched = old / "untouched.bin"
    untouched.write_bytes(b"same")
    inventory = subject._inventory(old)
    candidate = tmp_path / "candidate"
    subject._clone_root(old, candidate, inventory)

    replacement = candidate / "partition/replacement.tmp"
    replacement.write_bytes(b"new")
    replacement.replace(candidate / "partition/data.parquet")
    published = [
        {
            "target": "partition/data.parquet",
            "authorized_predecessor_detachments": ["partition/data.parquet"],
        }
    ]
    detached = subject._validate_cloned_predecessor(old, candidate, inventory, published)
    assert detached == ["partition/data.parquet"]
    assert original.stat().st_nlink == 1
    assert untouched.stat().st_nlink == 2

    (candidate / "untouched.bin").unlink()
    with pytest.raises(subject.PublicationError, match="unauthenticated"):
        subject._validate_cloned_predecessor(old, candidate, inventory, published)


def test_source_revalidation_rejects_in_place_mutation(tmp_path: Path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source-v1")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    size, _ = subject._hash_source(source, expected)
    assert size == len(b"source-v1")

    source.write_bytes(b"source-v2")
    with pytest.raises(subject.PublicationError, match="hash mismatch"):
        subject._hash_source(source, expected)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_main_atomically_activates_shared_root_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bootstrap: bool
) -> None:
    canonical = tmp_path / "market_parquet"
    sentinel = canonical / "sentinel.bin"
    old_sentinel_identity = None
    if not bootstrap:
        canonical.mkdir()
        sentinel.write_bytes(b"old-view")
        old_sentinel_identity = (sentinel.stat().st_dev, sentinel.stat().st_ino)

    source_root = tmp_path / "source"
    report = tmp_path / "report"
    source = source_root / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2026-04.parquet"
    source.parent.mkdir(parents=True)
    report.mkdir()
    frame = pl.DataFrame(
        {
            "datetime": [
                datetime(2026, 4, 1, 0, 0),
                datetime(2026, 4, 1, 0, 0, 1),
            ],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1.0, 2.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    frame.write_parquet(source)
    source_before = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)

    terminal_receipt = tmp_path / "terminal-receipt.json"
    authority_key = tmp_path / "authority.pub"
    terminal_receipt.write_bytes(b"terminal\n")
    authority_key.write_bytes(b"k" * 32)
    request_id = "1" * 64
    relative = "market_ohlcv_1s/binance/BTCUSDT/2026-04.parquet"
    source_sha = hashlib.sha256(source_before).hexdigest()
    parts = [
        {
            "relative": relative,
            "source": source,
            "sha256": source_sha,
            "rows": frame.height,
            "provenance": "2" * 64,
        }
    ]
    hashes = {
        "manifest": "3" * 64,
        "eligible": "4" * 64,
        "journal": "5" * 64,
        "contract": "6" * 64,
    }
    listing = [{"symbol": "BTCUSDT"}]
    monkeypatch.setattr(
        subject,
        "_terminal",
        lambda _args: ({"request_id": request_id}, "7" * 64, {}),
    )
    monkeypatch.setattr(
        subject,
        "_partitions",
        lambda _source, _report, _bound: (parts, hashes, listing),
    )
    monkeypatch.setattr(
        subject,
        "_STORAGE",
        {
            "host_reserve_path": str(tmp_path),
            "host_reserve_bytes": 0,
            "max_live_archives": 1,
            "archive_retention": "retired_after_double_derivation",
        },
    )
    argv = [
        "--source-root",
        str(source_root),
        "--source-report",
        str(report),
        "--terminal-receipt",
        str(terminal_receipt),
        "--authority-public-key",
        str(authority_key),
        "--canonical-root",
        str(canonical),
    ]

    validate_commit = subject._validate_prepared_commit
    validation_calls = 0

    def fail_after_exchange(*args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        result = validate_commit(*args, **kwargs)
        if validation_calls == 2:
            raise subject.PublicationError("injected post-exchange failure")
        return result

    monkeypatch.setattr(subject, "_validate_prepared_commit", fail_after_exchange)
    with pytest.raises(subject.PublicationError, match="post-exchange"):
        subject.main(argv)
    if bootstrap:
        assert not canonical.exists()
        assert not canonical.is_symlink()
    else:
        assert not canonical.is_symlink()
        assert sentinel.read_bytes() == b"old-view"
    monkeypatch.setattr(subject, "_validate_prepared_commit", validate_commit)

    assert subject.main(argv) == 0
    assert canonical.is_symlink()
    if bootstrap:
        assert not sentinel.exists()
    else:
        assert sentinel.read_bytes() == b"old-view"
        assert sentinel.stat().st_nlink == 1
        assert sentinel.stat().st_ino == old_sentinel_identity[1]
    target = canonical / relative
    assert target.exists()
    assert (target.stat().st_dev, target.stat().st_ino) != source_identity
    assert source.read_bytes() == source_before
    assert source.stat().st_nlink == 1
    assert pl.read_parquet(target).equals(frame)
    commit, _ = subject._json(canonical / "commit.json", "commit")
    listing_value, listing_bytes = subject._json(
        canonical / ".alpha_max_publication" / request_id / "listing_records.json",
        "listing",
    )
    assert listing_value["records"] == listing
    assert hashlib.sha256(listing_bytes).hexdigest() == commit["listing_metadata_sha256"]

    active_lock = (
        canonical / "market_data_raw_aggtrades" / "binance" / "BTCUSDT" / ".raw-stream.lock"
    )
    active_lock.parent.mkdir(parents=True, exist_ok=True)
    active_lock.touch(mode=0o600)
    assert subject.main(argv) == 0
    assert target.exists()
    assert source.read_bytes() == source_before


def test_reader_pins_complete_generation_until_exchange(tmp_path: Path):
    canonical = tmp_path / "market_parquet"
    canonical.mkdir()
    (canonical / "view.txt").write_text("old", encoding="utf-8")
    generations = tmp_path / ".market_parquet.generations"
    new = generations / "new"
    new.mkdir(parents=True)
    (new / "view.txt").write_text("new", encoding="utf-8")
    temporary = tmp_path / ".market_parquet.swap"
    temporary.symlink_to(".market_parquet.generations/new")

    repo = subject.ParquetMarketDataRepository(canonical)
    reader_entered = threading.Event()
    release_reader = threading.Event()
    exchange_acquired = threading.Event()
    failures: list[BaseException] = []
    observed: list[tuple[str, str]] = []

    def read_generation():
        try:
            with repo.generation_lock(exclusive=False):
                observed.append(
                    (
                        repo.root_path.name,
                        (repo.root_path / "view.txt").read_text(encoding="utf-8"),
                    )
                )
                reader_entered.set()
                assert release_reader.wait(2)
                observed.append(
                    (
                        repo.root_path.name,
                        (repo.root_path / "view.txt").read_text(encoding="utf-8"),
                    )
                )
        except BaseException as exc:
            failures.append(exc)

    def exchange_generation():
        try:
            with repo.generation_lock(exclusive=True):
                exchange_acquired.set()
                subject._rename_exchange(temporary, canonical)
        except BaseException as exc:
            failures.append(exc)

    reader = threading.Thread(target=read_generation)
    publisher = threading.Thread(target=exchange_generation)
    reader.start()
    assert reader_entered.wait(2)
    publisher.start()
    time.sleep(0.05)
    assert not exchange_acquired.is_set()
    release_reader.set()
    reader.join(2)
    publisher.join(2)
    assert not failures
    assert observed == [("market_parquet", "old"), ("market_parquet", "old")]

    with repo.generation_lock(exclusive=False):
        assert repo.root_path == new
        assert (repo.root_path / "view.txt").read_text(encoding="utf-8") == "new"


def test_raw_lease_cross_context_release_cleans_global_lock(tmp_path: Path):
    root = tmp_path / "market_parquet"
    repo = subject.ParquetMarketDataRepository(root)
    lease = repo.acquire_raw_symbol_stream_lease(
        exchange="binance",
        symbol="BTCUSDT",
    )
    failures: list[BaseException] = []

    def release():
        try:
            lease.release()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=release)
    thread.start()
    thread.join(2)

    assert not thread.is_alive()
    assert not failures
    with subject.ParquetMarketDataRepository(root).generation_lock(
        exclusive=True,
        timeout_seconds=0.2,
    ):
        pass


def test_writer_waits_for_global_publication_lock(tmp_path: Path):
    root = tmp_path / "market_parquet"
    root.mkdir()
    repo = subject.ParquetMarketDataRepository(root)
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 4, 1)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    completed = threading.Event()
    failures: list[BaseException] = []

    def write():
        try:
            repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=frame)
            completed.set()
        except BaseException as exc:
            failures.append(exc)

    with repo.generation_lock(exclusive=True):
        writer = threading.Thread(target=write)
        writer.start()
        time.sleep(0.05)
        assert not completed.is_set()
    writer.join(2)
    assert completed.is_set()
    assert not failures
    assert repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1s").height == 1


def test_market_loader_does_not_fallback_to_empty_on_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "market_parquet"
    root.mkdir()
    repo = subject.MarketDataRepository(str(root))

    def fail_read(**_kwargs):
        raise RuntimeError("injected generation read failure")

    monkeypatch.setattr(repo._parquet_repo, "load_ohlcv", fail_read)
    with pytest.raises(RuntimeError, match="generation read failure"):
        repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1s")


def test_authorized_effects_are_exact_and_preserve_nonoverlapping_rows(tmp_path: Path):
    repository = ParquetMarketDataRepository(tmp_path / "canonical")
    existing = pl.DataFrame(
        {
            "datetime": [datetime(2026, 4, 1), datetime(2026, 4, 1, 0, 0, 1)],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    incoming = pl.DataFrame(
        {
            "datetime": [datetime(2026, 4, 1), datetime(2026, 4, 1, 0, 0, 2)],
            "open": [9.0, 3.0],
            "high": [10.0, 4.0],
            "low": [8.0, 2.5],
            "close": [9.5, 3.5],
            "volume": [90.0, 30.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    effects = repository._signed_month_conflict_effects(existing, incoming)

    assert effects["conflict_rows"] == 1
    assert effects["canonical_only_rows"] == 1
    assert effects["source_only_rows"] == 1
    assert len(effects["conflict_sha256"]) == 64
    assert len(effects["canonical_only_sha256"]) == 64
    assert len(effects["source_only_sha256"]) == 64


def test_v3_replay_requires_paired_conflict_authorization(tmp_path: Path):
    args = type(
        "Args",
        (),
        {
            "conflict_authorization_receipt": None,
            "conflict_authority_public_key": None,
        },
    )()
    final = {
        "schema": "alpha_max_canonical_publication_receipt.v3",
        "predecessor_path": str(tmp_path),
        "old_inventory": [],
    }
    with pytest.raises(subject.PublicationError, match="requires conflict authorization"):
        subject._replay_conflict_authorization(
            args,
            final=final,
            terminal_sha256="0" * 64,
            request_id="1" * 64,
            hashes={"manifest": "2" * 64, "eligible": "3" * 64},
            parts=[],
        )


def test_conflict_authorization_rejects_wrong_effects_and_key(tmp_path: Path):
    canonical = tmp_path / "canonical"
    predecessor = tmp_path / "predecessor"
    relative = "market_ohlcv_1s/binance/BTCUSDT/2026-04.parquet"
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 4, 1)],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    target = predecessor / relative
    source = tmp_path / "source.parquet"
    target.parent.mkdir(parents=True)
    frame.write_parquet(target)
    changed = frame.with_columns(pl.lit(9.0).alias("open"))
    changed.write_parquet(source)
    repository = ParquetMarketDataRepository(tmp_path / "repository")
    effects = repository._signed_month_conflict_effects(frame, changed)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    parts = [
        {
            "relative": relative,
            "source": source,
            "sha256": source_sha,
            "rows": 1,
            "provenance": "2" * 64,
        }
    ]
    private = Ed25519PrivateKey.generate()
    public_path = tmp_path / "authority.pub"
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    public_path.write_bytes(public)
    public_path.chmod(0o400)
    receipt_path = tmp_path / "authorization.json"
    message = {
        "schema": "alpha_max_canonical_conflict_authorization_message.v1",
        "scope": "canonical_conflict_reconciliation",
        "decision": "approve_exact_effects",
        "canonical_root": str(canonical),
        "acquisition_request_id": "1" * 64,
        "terminal_receipt_sha256": "0" * 64,
        "source_manifest_sha256": "3" * 64,
        "source_eligible_receipt_sha256": "4" * 64,
        "predecessor_path": str(predecessor),
        "predecessor_identity": [predecessor.stat().st_dev, predecessor.stat().st_ino],
        "predecessor_inventory_sha256": subject._inventory_digest(subject._inventory(predecessor)),
        "entries": [
            {
                "relative": relative,
                "source_sha256": source_sha,
                "source_byte_count": source.stat().st_size,
                "source_row_count": 1,
                "provenance_receipt_sha256": "2" * 64,
                "predecessor_identity": list(subject._identity(target.stat())),
                "predecessor_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "predecessor_byte_count": target.stat().st_size,
                "predecessor_row_count": 1,
                "effects": effects,
            }
        ],
    }
    unsigned = {
        "schema": "alpha_max_canonical_conflict_authorization_receipt.v1",
        "type": "canonical_conflict_authorization",
        "authority_key_id": hashlib.sha256(public).hexdigest(),
        "message": message,
    }
    receipt = {
        **unsigned,
        "signature": base64.b64encode(
            private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v1\0"
                + subject.canonical_bytes(unsigned)
            )
        ).decode(),
    }
    receipt_path.write_bytes(subject.canonical_bytes(receipt))
    receipt_path.chmod(0o600)
    args = argparse.Namespace(
        canonical_root=canonical,
        conflict_authorization_receipt=receipt_path,
        conflict_authority_public_key=public_path,
    )
    hashes = {"manifest": "3" * 64, "eligible": "4" * 64}
    verified = subject._conflict_authorization(
        args,
        terminal_sha256="0" * 64,
        request_id="1" * 64,
        hashes=hashes,
        predecessor=predecessor,
        predecessor_inventory=subject._inventory(predecessor),
        parts=parts,
    )
    assert verified is not None
    wrong_public_path = tmp_path / "wrong-authority.pub"
    wrong_public_path.write_bytes(
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    wrong_public_path.chmod(0o400)
    args.conflict_authority_public_key = wrong_public_path
    with pytest.raises(subject.PublicationError, match="schema"):
        subject._conflict_authorization(
            args,
            terminal_sha256="0" * 64,
            request_id="1" * 64,
            hashes=hashes,
            predecessor=predecessor,
            predecessor_inventory=subject._inventory(predecessor),
            parts=parts,
        )
    args.conflict_authority_public_key = public_path
    message["entries"][0]["effects"]["conflict_rows"] = 2
    receipt_path.write_bytes(subject.canonical_bytes({**receipt, "message": message}))
    with pytest.raises(subject.PublicationError, match="signature"):
        subject._conflict_authorization(
            args,
            terminal_sha256="0" * 64,
            request_id="1" * 64,
            hashes=hashes,
            predecessor=predecessor,
            predecessor_inventory=subject._inventory(predecessor),
            parts=parts,
        )
    boolean_message = json.loads(json.dumps(message))
    boolean_message["entries"][0]["effects"]["conflict_rows"] = effects["conflict_rows"]
    boolean_message["entries"][0]["source_row_count"] = True
    boolean_unsigned = {
        "schema": "alpha_max_canonical_conflict_authorization_receipt.v1",
        "type": "canonical_conflict_authorization",
        "authority_key_id": hashlib.sha256(public).hexdigest(),
        "message": boolean_message,
    }
    boolean_receipt = {
        **boolean_unsigned,
        "signature": base64.b64encode(
            private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v1\0"
                + subject.canonical_bytes(boolean_unsigned)
            )
        ).decode(),
    }
    receipt_path.write_bytes(subject.canonical_bytes(boolean_receipt))
    with pytest.raises(subject.PublicationError, match="source binding"):
        subject._conflict_authorization(
            args,
            terminal_sha256="0" * 64,
            request_id="1" * 64,
            hashes=hashes,
            predecessor=predecessor,
            predecessor_inventory=subject._inventory(predecessor),
            parts=parts,
        )


def test_signed_month_authorization_reconciles_only_exact_effects(tmp_path: Path):
    old = tmp_path / "old"
    candidate = tmp_path / "candidate"
    source = tmp_path / "source.parquet"
    relative = Path("market_ohlcv_1s/binance/BTCUSDT/2026-04.parquet")
    existing = pl.DataFrame(
        {
            "datetime": [
                datetime(2026, 4, 1),
                datetime(2026, 4, 1, 0, 0, 1),
                datetime(2026, 4, 1, 0, 0, 3),
            ],
            "open": [1.0, 2.0, -0.0],
            "high": [2.0, 3.0, 1.0],
            "low": [0.5, 1.5, -1.0],
            "close": [1.5, 2.5, -0.0],
            "volume": [10.0, 20.0, 40.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    incoming = pl.DataFrame(
        {
            "datetime": [
                datetime(2026, 4, 1),
                datetime(2026, 4, 1, 0, 0, 2),
                datetime(2026, 4, 1, 0, 0, 3),
            ],
            "open": [9.0, 3.0, 0.0],
            "high": [10.0, 4.0, 1.0],
            "low": [8.0, 2.5, -1.0],
            "close": [9.5, 3.5, 0.0],
            "volume": [90.0, 30.0, 40.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    original = old / relative
    target = candidate / relative
    original.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    existing.write_parquet(original)
    predecessor_identity = list(subject._identity(original.stat()))
    predecessor_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    predecessor_bytes = original.stat().st_size
    target.hardlink_to(original)
    incoming.write_parquet(source)
    repository = ParquetMarketDataRepository(candidate)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    kwargs = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "month": "2026-04",
        "source": source,
        "expected_sha256": source_sha,
        "expected_byte_count": source.stat().st_size,
        "expected_row_count": incoming.height,
        "provenance_receipt_sha256": "1" * 64,
        "max_output_bytes": 1_000_000,
    }

    with pytest.raises(
        Exception,
        match="Signed monthly source conflicts with canonical OHLCV values",
    ):
        repository.merge_signed_month_into_candidate(**kwargs)

    effects = repository._signed_month_conflict_effects(existing, incoming)
    with pytest.raises(TypeError, match="conflict_effects"):
        repository.merge_signed_month_into_candidate(
            **kwargs,
            conflict_effects=effects,
        )
    repair_private = Ed25519PrivateKey.generate()
    repair_public = repair_private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    def signed_receipt(entry):
        message = {
            "schema": "alpha_max_canonical_conflict_authorization_message.v1",
            "scope": "canonical_conflict_reconciliation",
            "decision": "approve_exact_effects",
            "canonical_root": str(candidate),
            "acquisition_request_id": "2" * 64,
            "terminal_receipt_sha256": "3" * 64,
            "source_manifest_sha256": "4" * 64,
            "source_eligible_receipt_sha256": "5" * 64,
            "predecessor_path": str(old),
            "predecessor_identity": [old.stat().st_dev, old.stat().st_ino],
            "predecessor_inventory_sha256": "6" * 64,
            "entries": [entry],
        }
        unsigned = {
            "schema": "alpha_max_canonical_conflict_authorization_receipt.v1",
            "type": "canonical_conflict_authorization",
            "authority_key_id": hashlib.sha256(repair_public).hexdigest(),
            "message": message,
        }
        return {
            **unsigned,
            "signature": base64.b64encode(
                repair_private.sign(
                    b"luminaquant.alpha_max.canonical_conflict_authorization.v1\0"
                    + subject.canonical_bytes(unsigned)
                )
            ).decode(),
        }

    entry = {
        "relative": str(relative),
        "source_sha256": source_sha,
        "source_byte_count": source.stat().st_size,
        "source_row_count": incoming.height,
        "provenance_receipt_sha256": "1" * 64,
        "predecessor_identity": predecessor_identity,
        "predecessor_sha256": predecessor_sha,
        "predecessor_byte_count": predecessor_bytes,
        "predecessor_row_count": existing.height,
        "effects": effects,
    }
    wrong_entry = {
        **entry,
        "effects": {**effects, "conflict_rows": effects["conflict_rows"] + 1},
    }
    with pytest.raises(Exception, match="authorization effects do not match"):
        repository.reconcile_authorized_signed_month_into_candidate(
            **kwargs,
            conflict_authorization_receipt=signed_receipt(wrong_entry),
            conflict_authority_public_key=repair_public,
            authorization_entry=wrong_entry,
        )
    boolean_entry = {
        **entry,
        "effects": {**effects, "source_only_rows": True},
    }
    with pytest.raises(Exception, match="authorization entry is invalid"):
        repository.reconcile_authorized_signed_month_into_candidate(
            **kwargs,
            conflict_authorization_receipt=signed_receipt(boolean_entry),
            conflict_authority_public_key=repair_public,
            authorization_entry=boolean_entry,
        )
    for field, value in (
        ("source_row_count", True),
        ("predecessor_row_count", float(existing.height)),
    ):
        scalar_entry = {**entry, field: value}
        with pytest.raises(Exception, match="authorization entry is invalid"):
            repository.reconcile_authorized_signed_month_into_candidate(
                **kwargs,
                conflict_authorization_receipt=signed_receipt(scalar_entry),
                conflict_authority_public_key=repair_public,
                authorization_entry=scalar_entry,
            )

    result = repository.reconcile_authorized_signed_month_into_candidate(
        **kwargs,
        conflict_authorization_receipt=signed_receipt(entry),
        conflict_authority_public_key=repair_public,
        authorization_entry=entry,
    )
    actual = pl.read_parquet(result).sort("datetime")
    assert actual.get_column("datetime").to_list() == [
        datetime(2026, 4, 1),
        datetime(2026, 4, 1, 0, 0, 1),
        datetime(2026, 4, 1, 0, 0, 2),
        datetime(2026, 4, 1, 0, 0, 3),
    ]
    assert actual.get_column("open").to_list()[:3] == [9.0, 2.0, 3.0]
    assert math.copysign(1.0, actual.get_column("open")[-1]) == -1.0
    assert actual.get_column("volume").to_list() == [90.0, 20.0, 30.0, 40.0]
    assert original.stat().st_nlink == 1
    assert result.stat().st_nlink == 1
    assert result.stat().st_ino != source.stat().st_ino
    assert result.with_suffix(".seal.json").exists()


def test_signed_month_authorization_rejects_wal_before_missing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = ParquetMarketDataRepository(tmp_path / "candidate")
    source = tmp_path / "source.parquet"
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 4, 1)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    frame.write_parquet(source)
    monkeypatch.setattr(repository, "_load_wal_frame", lambda **_kwargs: frame)

    with pytest.raises(Exception, match="WAL overlaps"):
        repository.merge_signed_month_into_candidate(
            exchange="binance",
            symbol="BTCUSDT",
            month="2026-04",
            source=source,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_byte_count=source.stat().st_size,
            expected_row_count=1,
            provenance_receipt_sha256="1" * 64,
        )


def test_main_authorized_reconciliation_is_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    canonical = tmp_path / "market_parquet"
    source_root = tmp_path / "source"
    report = tmp_path / "report"
    relative = "market_ohlcv_1s/binance/BTCUSDT/2026-04.parquet"
    target = canonical / relative
    source = source_root / relative
    target.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    report.mkdir()
    existing = pl.DataFrame(
        {
            "datetime": [
                datetime(2026, 4, 1),
                datetime(2026, 4, 1, 0, 0, 1),
            ],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    incoming = pl.DataFrame(
        {
            "datetime": [
                datetime(2026, 4, 1),
                datetime(2026, 4, 1, 0, 0, 2),
            ],
            "open": [9.0, 3.0],
            "high": [10.0, 4.0],
            "low": [8.0, 2.5],
            "close": [9.5, 3.5],
            "volume": [90.0, 30.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    existing.write_parquet(target)
    incoming.write_parquet(source)
    source_before = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    request_id = "1" * 64
    source_sha = hashlib.sha256(source_before).hexdigest()
    parts = [
        {
            "relative": relative,
            "source": source,
            "sha256": source_sha,
            "rows": incoming.height,
            "provenance": "2" * 64,
        }
    ]
    hashes = {
        "manifest": "3" * 64,
        "eligible": "4" * 64,
        "journal": "5" * 64,
        "contract": "6" * 64,
    }
    listing = [{"symbol": "BTCUSDT"}]
    monkeypatch.setattr(
        subject,
        "_terminal",
        lambda _args: ({"request_id": request_id}, "7" * 64, {}),
    )
    monkeypatch.setattr(
        subject,
        "_partitions",
        lambda _source, _report, _bound: (parts, hashes, listing),
    )
    monkeypatch.setattr(
        subject,
        "_STORAGE",
        {
            "host_reserve_path": str(tmp_path),
            "host_reserve_bytes": 0,
            "max_live_archives": 1,
            "archive_retention": "retired_after_double_derivation",
        },
    )

    terminal_receipt = tmp_path / "terminal-receipt.json"
    authority_key = tmp_path / "authority.pub"
    terminal_receipt.write_bytes(b"terminal\n")
    authority_key.write_bytes(b"k" * 32)
    repair_private = Ed25519PrivateKey.generate()
    repair_public = repair_private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    repair_public_path = tmp_path / "conflict-authority.pub"
    repair_public_path.write_bytes(repair_public)
    repair_public_path.chmod(0o400)
    repair_receipt_path = tmp_path / "conflict-authorization.json"
    repository = ParquetMarketDataRepository(canonical)
    effects = repository._signed_month_conflict_effects(existing, incoming)
    old_inventory = subject._inventory(canonical)
    message = {
        "schema": "alpha_max_canonical_conflict_authorization_message.v1",
        "scope": "canonical_conflict_reconciliation",
        "decision": "approve_exact_effects",
        "canonical_root": str(canonical),
        "acquisition_request_id": request_id,
        "terminal_receipt_sha256": hashlib.sha256(terminal_receipt.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashes["manifest"],
        "source_eligible_receipt_sha256": hashes["eligible"],
        "predecessor_path": str(canonical),
        "predecessor_identity": [canonical.stat().st_dev, canonical.stat().st_ino],
        "predecessor_inventory_sha256": subject._inventory_digest(old_inventory),
        "entries": [
            {
                "relative": relative,
                "source_sha256": source_sha,
                "source_byte_count": source.stat().st_size,
                "source_row_count": incoming.height,
                "provenance_receipt_sha256": "2" * 64,
                "predecessor_identity": list(subject._identity(target.stat())),
                "predecessor_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "predecessor_byte_count": target.stat().st_size,
                "predecessor_row_count": existing.height,
                "effects": effects,
            }
        ],
    }
    unsigned = {
        "schema": "alpha_max_canonical_conflict_authorization_receipt.v1",
        "type": "canonical_conflict_authorization",
        "authority_key_id": hashlib.sha256(repair_public).hexdigest(),
        "message": message,
    }
    repair_receipt = {
        **unsigned,
        "signature": base64.b64encode(
            repair_private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v1\0"
                + subject.canonical_bytes(unsigned)
            )
        ).decode(),
    }
    repair_receipt_path.write_bytes(subject.canonical_bytes(repair_receipt))
    repair_receipt_path.chmod(0o600)
    argv = [
        "--source-root",
        str(source_root),
        "--source-report",
        str(report),
        "--terminal-receipt",
        str(terminal_receipt),
        "--authority-public-key",
        str(authority_key),
        "--canonical-root",
        str(canonical),
        "--conflict-authorization-receipt",
        str(repair_receipt_path),
        "--conflict-authority-public-key",
        str(repair_public_path),
    ]

    assert subject.main(argv) == 0
    assert canonical.is_symlink()
    actual = pl.read_parquet(canonical / relative).sort("datetime")
    assert actual.get_column("open").to_list() == [9.0, 2.0, 3.0]
    assert actual.get_column("volume").to_list() == [90.0, 20.0, 30.0]
    assert source.read_bytes() == source_before
    assert (canonical / relative).stat().st_ino != source_identity[1]
    commit, _ = subject._json(canonical / "commit.json", "commit")
    assert commit["schema"] == "alpha_max_canonical_publication_receipt.v3"
    assert commit["conflict_authority_key_id"] != commit["authority_key_id"]
    assert commit["authorized_conflict_partition_count"] == 1
    assert commit["authorized_replaced_row_count"] == 1
    assert commit["partitions"][0]["publication_mode"] == "authorized_reconciliation"
    assert commit["partitions"][0]["conflict_effects"] == effects

    assert subject.main(argv) == 0
    stale_unsigned = json.loads(json.dumps(unsigned))
    stale_unsigned["message"]["acquisition_request_id"] = "9" * 64
    stale_receipt = {
        **stale_unsigned,
        "signature": base64.b64encode(
            repair_private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v1\0"
                + subject.canonical_bytes(stale_unsigned)
            )
        ).decode(),
    }
    stale_receipt_bytes = subject.canonical_bytes(stale_receipt)
    repair_receipt_path.write_bytes(stale_receipt_bytes)
    stale_final = json.loads(json.dumps(commit))
    stale_final["conflict_authorization_receipt_sha256"] = hashlib.sha256(
        stale_receipt_bytes
    ).hexdigest()
    stale_final["conflict_authorization_message_sha256"] = hashlib.sha256(
        subject.canonical_bytes(stale_unsigned["message"])
    ).hexdigest()
    replay_args = argparse.Namespace(
        canonical_root=canonical,
        conflict_authorization_receipt=repair_receipt_path,
        conflict_authority_public_key=repair_public_path,
    )
    with pytest.raises(subject.PublicationError, match="binding mismatch"):
        subject._replay_conflict_authorization(
            replay_args,
            final=stale_final,
            terminal_sha256=hashlib.sha256(terminal_receipt.read_bytes()).hexdigest(),
            request_id=request_id,
            hashes=hashes,
            parts=parts,
        )
    repair_receipt_path.write_bytes(subject.canonical_bytes(repair_receipt))
    with pytest.raises(subject.PublicationError, match="V3 replay requires"):
        subject.main(argv[:-4])
    tampered = json.loads(json.dumps(commit))
    tampered["partitions"][0]["conflict_effects"]["source_only_sha256"] = "f" * 64
    publication = canonical / ".alpha_max_publication" / request_id
    partition_receipt = (
        publication / "partitions" / (hashlib.sha256(relative.encode()).hexdigest() + ".json")
    )
    partition_receipt.write_bytes(subject.canonical_bytes(tampered["partitions"][0]))
    (publication / "canonical_publication_receipt.json").write_bytes(
        subject.canonical_bytes(tampered)
    )
    (canonical / "commit.json").write_bytes(subject.canonical_bytes(tampered))
    with pytest.raises(
        subject.PublicationError, match="prepared conflict authorization binding mismatch"
    ):
        subject.main(argv)
