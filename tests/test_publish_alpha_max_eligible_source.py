from __future__ import annotations

import importlib.util
import hashlib
from datetime import datetime
import threading
import time
from pathlib import Path

import pytest
import polars as pl
from lumina_quant import market_data


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
