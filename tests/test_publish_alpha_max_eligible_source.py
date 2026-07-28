from __future__ import annotations

import importlib.util
import hashlib
from datetime import datetime
from pathlib import Path

import pytest
import polars as pl


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
    expected = subject._inventory(old)
    candidate = tmp_path / "candidate"
    subject._clone_root(old, candidate, expected)
    assert (source.stat().st_dev, source.stat().st_ino) == (
        (candidate / "nested" / "data.bin").stat().st_dev,
        (candidate / "nested" / "data.bin").stat().st_ino,
    )
    assert source.stat().st_nlink == 2
    (candidate / "extra").write_bytes(b"x")
    with pytest.raises(subject.PublicationError, match="extra"):
        subject._remove_bound_tree(
            candidate, (candidate.stat().st_dev, candidate.stat().st_ino), expected
        )


def test_main_atomically_activates_shared_root_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "market_parquet"
    canonical.mkdir()
    sentinel = canonical / "sentinel.bin"
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
    assert not canonical.is_symlink()
    assert (canonical / "sentinel.bin").read_bytes() == b"old-view"
    monkeypatch.setattr(subject, "_validate_prepared_commit", validate_commit)

    assert subject.main(argv) == 0
    assert canonical.is_symlink()
    assert (canonical / "sentinel.bin").read_bytes() == b"old-view"
    assert (canonical / "sentinel.bin").stat().st_nlink == 1
    assert (canonical / "sentinel.bin").stat().st_ino == old_sentinel_identity[1]
    target = canonical / relative
    assert target.exists()
    assert (target.stat().st_dev, target.stat().st_ino) != source_identity
    assert source.read_bytes() == source_before
    assert source.stat().st_nlink == 1
    assert pl.read_parquet(target).equals(frame)

    assert subject.main(argv) == 0
    assert target.exists()
    assert source.read_bytes() == source_before
