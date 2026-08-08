from __future__ import annotations

import importlib.util
import os
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
from lumina_quant.alpha_max_terminal_policy import public_key_id, sign_message


_SCRIPT = Path(__file__).parents[1] / "scripts/research/publish_alpha_max_eligible_source.py"
_spec = importlib.util.spec_from_file_location("publish_alpha_max_eligible_source", _SCRIPT)
assert _spec is not None and _spec.loader is not None
subject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(subject)


def test_canonical_bytes_are_stable():
    assert subject.canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'


def test_current_approval_v3_binds_ignored_runtime_and_execution_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository_root = Path(subject.__file__).resolve().parents[2]
    accepted = subject._APPROVAL_ACCEPTED_ALPHA_COMMIT
    head = "c" * 40
    porcelain = b"porcelain"
    commit_overlay = b"commit-overlay"
    worktree_overlay = b"worktree-overlay"
    source_inventory = b"source-inventory"
    ignored_inventory = b"ignored-inventory"
    accepted_porcelain = b"accepted-porcelain"
    accepted_source_inventory = b"accepted-source-inventory"
    accepted_ignored_inventory = b"accepted-ignored-inventory"
    accepted_drift = {"tracked": False, "ignored": False}
    runtime_roots = {
        "current_python": Path("/runtime/current"),
        "accepted_python": Path("/runtime/accepted"),
        "base_python": Path("/runtime/base"),
    }
    runtime_raw = {name: str(root).encode() for name, root in runtime_roots.items()}
    execution_alias = {
        "path": "/execution-alias",
        "target": "/recovery",
        "st_dev": 1,
        "st_ino": 2,
        "st_uid": os.getuid(),
        "st_gid": os.getgid(),
        "mode": 0o777,
        "nlink": 1,
    }

    def fake_git(root: Path, *arguments: str) -> bytes:
        is_accepted = root == subject._APPROVAL_ACCEPTED_SOURCE_ROOT
        if arguments == ("rev-parse", "HEAD"):
            return ((accepted if is_accepted else head) + "\n").encode()
        if arguments[0] == "status":
            if is_accepted:
                return accepted_porcelain + (b"-drift" if accepted_drift["tracked"] else b"")
            return porcelain
        if arguments[0] == "diff" and f"{accepted}..HEAD" in arguments:
            return commit_overlay
        if arguments[0] == "diff":
            return worktree_overlay
        raise AssertionError(arguments)

    monkeypatch.setattr(subject, "_APPROVAL_RUNTIME_ROOTS", runtime_roots)
    monkeypatch.setattr(subject, "_approval_git", fake_git)
    monkeypatch.setattr(
        subject,
        "_approval_source_inventory",
        lambda root: (
            accepted_source_inventory
            + (
                b"-drift"
                if root == subject._APPROVAL_ACCEPTED_SOURCE_ROOT and accepted_drift["tracked"]
                else b""
            )
            if root == subject._APPROVAL_ACCEPTED_SOURCE_ROOT
            else source_inventory
        ),
    )
    monkeypatch.setattr(
        subject,
        "_approval_ignored_source_inventory",
        lambda root: (
            accepted_ignored_inventory + (b"-drift" if accepted_drift["ignored"] else b"")
            if root == subject._APPROVAL_ACCEPTED_SOURCE_ROOT
            else ignored_inventory
        ),
    )
    monkeypatch.setattr(
        subject, "_approval_runtime_inventory", lambda root: runtime_raw[root.name + "_python"]
    )
    monkeypatch.setattr(subject, "_approval_execution_alias", lambda: execution_alias)

    record = subject._approval_record
    approval = {
        "schema": "alpha_max_v8_current_state_approval.v3",
        "repository_root": str(repository_root),
        "head": head,
        "accepted_alpha_commit": accepted,
        "baseline_ancestor": subject._APPROVAL_BASELINE_ANCESTOR,
        "verdict": "PASS_REVIEWED_OVERLAY",
        "porcelain": record(porcelain),
        "commit_overlay": record(commit_overlay),
        "worktree_overlay": record(worktree_overlay),
        "source_inventory": record(source_inventory),
        "ignored_source_inventory": record(ignored_inventory),
        "runtime_inventories": {name: record(value) for name, value in runtime_raw.items()},
        "execution_alias": execution_alias,
        "run_id": subject._CONFLICT_RUN_ID,
        "request_ids": {
            "acquisition": subject._ACQUISITION_REQUEST_ID,
            "phase_preparation": subject._PHASE_REQUEST_ID,
            "one_touch": subject._ONE_TOUCH_REQUEST_ID,
        },
        "absent_recovery_artifacts": subject._APPROVAL_ABSENT_RECOVERY_ARTIFACTS,
        "accepted_source_state": {
            "root": str(subject._APPROVAL_ACCEPTED_SOURCE_ROOT),
            "head": accepted,
            "porcelain": record(accepted_porcelain),
            "source_inventory": record(accepted_source_inventory),
            "ignored_source_inventory": record(accepted_ignored_inventory),
        },
    }
    path = tmp_path / subject._CURRENT_APPROVAL_LEAF
    path.write_bytes(subject.canonical_bytes(approval))
    loaded, digest = subject._approval(path)
    assert loaded == approval
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    approval["ignored_source_inventory"] = {"sha256": "0" * 64, "entries": []}
    path.write_bytes(subject.canonical_bytes(approval))
    with pytest.raises(subject.PublicationError, match="binding is invalid"):
        subject._approval(path)
    approval["ignored_source_inventory"] = record(ignored_inventory)
    approval["accepted_alpha_commit"] = "a" * 40
    path.write_bytes(subject.canonical_bytes(approval))
    with pytest.raises(subject.PublicationError, match="binding is invalid"):
        subject._approval(path)
    approval["accepted_alpha_commit"] = accepted
    approval["baseline_ancestor"] = "b" * 40
    path.write_bytes(subject.canonical_bytes(approval))
    with pytest.raises(subject.PublicationError, match="binding is invalid"):
        subject._approval(path)
    approval["baseline_ancestor"] = subject._APPROVAL_BASELINE_ANCESTOR
    approval["absent_recovery_artifacts"] = {"control_root": "/tmp/absent-control"}
    path.write_bytes(subject.canonical_bytes(approval))
    with pytest.raises(subject.PublicationError, match="binding is invalid"):
        subject._approval(path)
    approval["absent_recovery_artifacts"] = subject._APPROVAL_ABSENT_RECOVERY_ARTIFACTS
    accepted_drift["tracked"] = True
    path.write_bytes(subject.canonical_bytes(approval))
    with pytest.raises(subject.PublicationError, match="repository binding drift"):
        subject._approval(path)
    accepted_drift["tracked"] = False
    accepted_drift["ignored"] = True
    with pytest.raises(subject.PublicationError, match="repository binding drift"):
        subject._approval(path)


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
    monkeypatch.setattr(subject, "_host_free_bytes", lambda: required + 1)
    admission = subject._publication_admission(candidate, item, 211)
    assert admission["required_free_bytes"] == required
    assert admission["temporary_merged_output_upper_bound"] == output_bound
    assert admission["immutable_source_pin_bytes"] == 211

    monkeypatch.setattr(subject, "_host_free_bytes", lambda: required)
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


def test_main_rejects_legacy_unsigned_invocation_before_mutation(tmp_path: Path) -> None:
    canonical = tmp_path / "market_parquet"
    source_root = tmp_path / "source"
    report = tmp_path / "report"
    source = source_root / "source.parquet"
    terminal_receipt = tmp_path / "terminal-receipt.json"
    authority_key = tmp_path / "authority.pub"
    source_root.mkdir()
    report.mkdir()
    source.write_bytes(b"source")
    terminal_receipt.write_bytes(b"terminal")
    authority_key.write_bytes(b"authority")
    source_before = source.read_bytes()

    legacy_argv = [
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

    with pytest.raises(SystemExit):
        subject.main(legacy_argv)

    assert not canonical.exists()
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
        "schema": "alpha_max_canonical_conflict_authorization_message.v2",
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
        "fresh_acquisition_audit_receipt_sha256": "5" * 64,
        "acquisition_run_id": "69ec878bb92644c6963d25ccebd1a11242801e8d5feeaaed144e5256037baafd",
        "composite_telemetry_sha256": "6" * 64,
        "wal_transition_receipt_sha256": "7" * 64,
        "wal_post_transition_inventory_sha256": subject._inventory_digest(
            subject._inventory(predecessor)
        ),
        "signed_terminal_request_sha256": "8" * 64,
        "approval_sha256": "9" * 64,
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
        "schema": "alpha_max_canonical_conflict_authorization_receipt.v2",
        "type": "canonical_conflict_authorization",
        "authority_key_id": hashlib.sha256(public).hexdigest(),
        "message": message,
    }
    receipt = {
        **unsigned,
        "signature": base64.b64encode(
            private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v2\0"
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
    controls = {
        field: message[field]
        for field in (
            "fresh_acquisition_audit_receipt_sha256",
            "acquisition_run_id",
            "composite_telemetry_sha256",
            "wal_transition_receipt_sha256",
            "wal_post_transition_inventory_sha256",
            "signed_terminal_request_sha256",
            "approval_sha256",
        )
    }
    verified = subject._conflict_authorization(
        args,
        terminal_sha256="0" * 64,
        request_id="1" * 64,
        hashes=hashes,
        predecessor=predecessor,
        predecessor_inventory=subject._inventory(predecessor),
        parts=parts,
        controls=controls,
        terminal_public_key=public,
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
    with pytest.raises(subject.PublicationError, match="equal terminal authority"):
        subject._conflict_authorization(
            args,
            terminal_sha256="0" * 64,
            request_id="1" * 64,
            hashes=hashes,
            predecessor=predecessor,
            predecessor_inventory=subject._inventory(predecessor),
            parts=parts,
            controls=controls,
            terminal_public_key=public,
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
            controls=controls,
            terminal_public_key=public,
        )
    boolean_message = json.loads(json.dumps(message))
    boolean_message["entries"][0]["effects"]["conflict_rows"] = effects["conflict_rows"]
    boolean_message["entries"][0]["source_row_count"] = True
    boolean_unsigned = {
        "schema": "alpha_max_canonical_conflict_authorization_receipt.v2",
        "type": "canonical_conflict_authorization",
        "authority_key_id": hashlib.sha256(public).hexdigest(),
        "message": boolean_message,
    }
    boolean_receipt = {
        **boolean_unsigned,
        "signature": base64.b64encode(
            private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v2\0"
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
            controls=controls,
            terminal_public_key=public,
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
    expected_run_id = "69ec878bb92644c6963d25ccebd1a11242801e8d5feeaaed144e5256037baafd"
    expected_approval_sha256 = "c" * 64
    authorization_context = {
        "expected_run_id": expected_run_id,
        "expected_approval_sha256": expected_approval_sha256,
    }

    def signed_receipt(entry, *, run_id=expected_run_id, approval_digest=expected_approval_sha256):
        message = {
            "schema": "alpha_max_canonical_conflict_authorization_message.v2",
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
            "fresh_acquisition_audit_receipt_sha256": "7" * 64,
            "acquisition_run_id": run_id,
            "composite_telemetry_sha256": "8" * 64,
            "wal_transition_receipt_sha256": "9" * 64,
            "wal_post_transition_inventory_sha256": "a" * 64,
            "signed_terminal_request_sha256": "b" * 64,
            "approval_sha256": approval_digest,
            "entries": [entry],
        }
        unsigned = {
            "schema": "alpha_max_canonical_conflict_authorization_receipt.v2",
            "type": "canonical_conflict_authorization",
            "authority_key_id": hashlib.sha256(repair_public).hexdigest(),
            "message": message,
        }
        return {
            **unsigned,
            "signature": base64.b64encode(
                repair_private.sign(
                    b"luminaquant.alpha_max.canonical_conflict_authorization.v2\0"
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
    with pytest.raises(Exception, match="acquisition run ID does not match trusted context"):
        repository.reconcile_authorized_signed_month_into_candidate(
            **kwargs,
            **authorization_context,
            conflict_authorization_receipt=signed_receipt(entry, run_id="d" * 64),
            conflict_authority_public_key=repair_public,
            authorization_entry=entry,
        )
    with pytest.raises(Exception, match="approval digest does not match trusted context"):
        repository.reconcile_authorized_signed_month_into_candidate(
            **kwargs,
            **authorization_context,
            conflict_authorization_receipt=signed_receipt(entry, approval_digest="e" * 64),
            conflict_authority_public_key=repair_public,
            authorization_entry=entry,
        )

    wrong_entry = {
        **entry,
        "effects": {**effects, "conflict_rows": effects["conflict_rows"] + 1},
    }
    with pytest.raises(Exception, match="authorization effects do not match"):
        repository.reconcile_authorized_signed_month_into_candidate(
            **kwargs,
            **authorization_context,
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
            **authorization_context,
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
                **authorization_context,
                conflict_authorization_receipt=signed_receipt(scalar_entry),
                conflict_authority_public_key=repair_public,
                authorization_entry=scalar_entry,
            )

    result = repository.reconcile_authorized_signed_month_into_candidate(
        **kwargs,
        **authorization_context,
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


@pytest.mark.parametrize(
    "scenario",
    (
        "happy",
        "activated_only",
        "preopen_marker",
        "preopen_dead",
        "preopen_symlink_restart",
        "alias_terminal_publisher",
        "alias_terminal_observer",
        "alias_publisher_observer",
    ),
)
def test_main_authorized_reconciliation_is_atomic_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
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
    predecessor_root = canonical
    if scenario == "preopen_symlink_restart":
        generations = canonical.parent / f".{canonical.name}.generations"
        generations.mkdir(mode=0o700)
        predecessor_root = generations / "predecessor"
        canonical.rename(predecessor_root)
        canonical.symlink_to(f"{generations.name}/{predecessor_root.name}")
    source_before = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    request_id = subject._ACQUISITION_REQUEST_ID
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
    terminal_private = Ed25519PrivateKey.generate()
    terminal_public = terminal_private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    terminal_request_sha256 = "8" * 64
    terminal_receipt = tmp_path / "terminal-receipt.json"
    authority_key = tmp_path / "authority.pub"
    terminal_receipt.write_bytes(b"terminal\n")
    authority_key.write_bytes(terminal_public)
    authority_key.chmod(0o400)
    publisher_private = (
        terminal_private if scenario == "alias_terminal_publisher" else Ed25519PrivateKey.generate()
    )
    publisher_key_path = tmp_path / "publisher-key.bin"
    publisher_key_path.write_bytes(
        publisher_private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    publisher_key_path.chmod(0o400)
    publisher_key_fd = os.open(publisher_key_path, os.O_RDONLY)
    monkeypatch.setattr(
        subject,
        "_terminal",
        lambda _args: (
            {"request_id": request_id, "request_sha256": terminal_request_sha256},
            hashlib.sha256(terminal_public).hexdigest(),
            {},
            hashlib.sha256(terminal_receipt.read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        subject,
        "_partitions",
        lambda _source, _report, _bound: (parts, hashes, [{"symbol": "BTCUSDT"}]),
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
    repair_public_path = authority_key
    repair_receipt_path = tmp_path / "conflict-authorization.json"
    repository = ParquetMarketDataRepository(canonical)
    effects = repository._signed_month_conflict_effects(existing, incoming)
    old_inventory = subject._inventory(predecessor_root)
    old_digest = subject._inventory_digest(old_inventory)
    fresh_path = tmp_path / "fresh-audit.json"
    wal_path = tmp_path / "wal-transition.json"
    fresh = {
        "schema": "luminaquant_fresh_acquisition_audit.v1",
        "run_id": "69ec878bb92644c6963d25ccebd1a11242801e8d5feeaaed144e5256037baafd",
        "request_id": request_id,
        "sealed": True,
        "outcome": "pass",
        "terminal_receipt_sha256": hashlib.sha256(terminal_receipt.read_bytes()).hexdigest(),
        "authority_key_id": hashlib.sha256(terminal_public).hexdigest(),
        "source_root": str(source_root),
        "source_report": str(report),
        "signed_terminal_request_sha256": terminal_request_sha256,
        "composite_telemetry": {"sha256": "4" * 64},
        "approval_sha256": "c" * 64,
    }
    wal = {
        "schema": "luminaquant.canonical_wal_transition.v1",
        "mode": "execute",
        "compaction_complete": True,
        "run_id": fresh["run_id"],
        "request_id": request_id,
        "canonical_root": str(canonical),
        "post_transition_inventory": old_inventory,
        "post_transition_inventory_sha256": old_digest,
        "approval_sha256": fresh["approval_sha256"],
    }
    fresh_path.write_bytes(subject.canonical_bytes(fresh))
    wal_path.write_bytes(subject.canonical_bytes(wal))
    fresh_path.chmod(0o600)
    wal_path.chmod(0o600)
    message = {
        "schema": "alpha_max_canonical_conflict_authorization_message.v2",
        "scope": "canonical_conflict_reconciliation",
        "decision": "approve_exact_effects",
        "canonical_root": str(canonical),
        "acquisition_request_id": request_id,
        "terminal_receipt_sha256": fresh["terminal_receipt_sha256"],
        "source_manifest_sha256": hashes["manifest"],
        "source_eligible_receipt_sha256": hashes["eligible"],
        "predecessor_path": str(predecessor_root),
        "predecessor_identity": [
            predecessor_root.stat().st_dev,
            predecessor_root.stat().st_ino,
        ],
        "predecessor_inventory_sha256": old_digest,
        "fresh_acquisition_audit_receipt_sha256": hashlib.sha256(
            fresh_path.read_bytes()
        ).hexdigest(),
        "acquisition_run_id": fresh["run_id"],
        "composite_telemetry_sha256": fresh["composite_telemetry"]["sha256"],
        "wal_transition_receipt_sha256": hashlib.sha256(wal_path.read_bytes()).hexdigest(),
        "wal_post_transition_inventory_sha256": old_digest,
        "signed_terminal_request_sha256": terminal_request_sha256,
        "approval_sha256": fresh["approval_sha256"],
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
        "schema": "alpha_max_canonical_conflict_authorization_receipt.v2",
        "type": "canonical_conflict_authorization",
        "authority_key_id": hashlib.sha256(terminal_public).hexdigest(),
        "message": message,
    }
    repair_receipt = {
        **unsigned,
        "signature": base64.b64encode(
            terminal_private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v2\0"
                + subject.canonical_bytes(unsigned)
            )
        ).decode(),
    }
    repair_receipt_path.write_bytes(subject.canonical_bytes(repair_receipt))
    repair_receipt_path.chmod(0o600)
    approval_path = tmp_path / "current-state-approval-v10.json"
    approval_path.write_bytes(b"approval\n")
    monkeypatch.setattr(subject, "_approval", lambda _path: ({}, fresh["approval_sha256"]))
    observer_private = (
        terminal_private
        if scenario == "alias_terminal_observer"
        else publisher_private
        if scenario == "alias_publisher_observer"
        else Ed25519PrivateKey.generate()
    )
    observer_public_path = tmp_path / "observer.pub"
    observer_public_path.write_bytes(
        observer_private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    )
    observer_public_path.chmod(0o400)
    control = canonical.parent / f".{canonical.name}.transactions" / request_id
    control.mkdir(mode=0o700, parents=True)
    old_observation = {
        "sequence": 0,
        "timestamp_ns": time.time_ns(),
        "generation": predecessor_root.name,
        "identity": subject._topology_identity(predecessor_root),
        "inventory_sha256": subject._canonical_inventory(predecessor_root)["inventory_sha256"],
        "representatives": [{"path": relative, "sha256": "d" * 64, "bytes": 1}],
        "public_loader_sha256": "e" * 64,
    }
    observer_ready = sign_message(
        "publication_observer_ready",
        {
            "schema": "alpha_max_publication_observer_ready.v1",
            "kind": "publication_observer_ready",
            "run_id": fresh["run_id"],
            "request_id": request_id,
            "approval_leaf": "current-state-approval-v10.json",
            "approval_sha256": fresh["approval_sha256"],
            "canonical_root": str(canonical),
            "query_spec_sha256": "f" * 64,
            "observer_key_id": public_key_id(observer_private.public_key()),
            "observer_pid": os.getpid(),
            "observer_uid": os.getuid(),
            "observer_start_ticks": subject._process_start_ticks(os.getpid()),
            "observed_ns": old_observation["timestamp_ns"],
            "old_identity": old_observation["identity"],
            "old_inventory_sha256": old_observation["inventory_sha256"],
            "old_loader_sha256": old_observation["public_loader_sha256"],
            "old_observation": old_observation,
        },
        observer_private,
    )
    observer_ready_path = control / "observer-ready.json"
    observer_ready_path.write_bytes(subject.canonical_bytes(observer_ready))
    observer_ready_path.chmod(0o600)
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
        "--publisher-key-fd",
        str(publisher_key_fd),
        "--current-state-approval",
        str(approval_path),
        "--conflict-authorization-receipt",
        str(repair_receipt_path),
        "--conflict-authority-public-key",
        str(repair_public_path),
        "--fresh-acquisition-audit-receipt",
        str(fresh_path),
        "--wal-transition-receipt",
        str(wal_path),
        "--observer-ready-receipt",
        str(observer_ready_path),
        "--observer-public-key",
        str(observer_public_path),
    ]

    def publish(arguments: list[str] = argv) -> int:
        os.lseek(publisher_key_fd, 0, os.SEEK_SET)
        return subject.main(arguments)

    if scenario.startswith("alias_"):
        with pytest.raises(subject.PublicationError, match="distinct"):
            publish()
        assert not canonical.is_symlink()
        assert not (canonical.parent / f".{canonical.name}.generations" / request_id).exists()
        return

    if scenario == "activated_only":
        original_write = subject._write_noreplace

        def crash_before_open(path: Path, value: dict[str, object]) -> None:
            if path.name == "rollback-window-open.json":
                raise RuntimeError("crash before open receipt")
            original_write(path, value)

        monkeypatch.setattr(subject, "_write_noreplace", crash_before_open)
        with pytest.raises(RuntimeError, match="crash before open"):
            publish()
        assert canonical.is_symlink()
        assert (control / "activated.json").exists()
        assert not (control / "rollback-window-open.json").exists()
        monkeypatch.setattr(subject, "_write_noreplace", original_write)
    elif scenario in {"preopen_marker", "preopen_dead", "preopen_symlink_restart"}:
        original_window_writer = subject._write_open_window_receipts

        def crash_after_exchange(**_kwargs: object) -> None:
            raise RuntimeError("crash after exchange")

        monkeypatch.setattr(subject, "_write_open_window_receipts", crash_after_exchange)
        with pytest.raises(RuntimeError, match="crash after exchange"):
            publish()
        monkeypatch.setattr(subject, "_write_open_window_receipts", original_window_writer)
        assert canonical.is_symlink()
        assert not (control / "activated.json").exists()
        if scenario == "preopen_marker":
            failure_message = {
                "kind": "observer_failure",
                "outcome": "FAIL",
                "run_id": subject._CONFLICT_RUN_ID,
                "request_id": request_id,
                "approval_leaf": subject._CURRENT_APPROVAL_LEAF,
                "approval_sha256": fresh["approval_sha256"],
                "state": "W1_ACTIVATED_OPEN",
                "frozen_observation": {},
                "error": "observer failed",
                "timestamp_ns": time.time_ns(),
            }
            failure_unsigned = {
                "schema": subject._PUBLICATION_STAGE_ENVELOPE_SCHEMA,
                "kind": "observer",
                "authority_key_id": public_key_id(observer_private.public_key()),
                "message": failure_message,
            }
            subject._write_noreplace(
                control / "observer-failure-intent.json",
                {
                    **failure_unsigned,
                    "signature": base64.b64encode(
                        observer_private.sign(
                            subject._PUBLICATION_STAGE_ENVELOPE_DOMAIN
                            + subject.canonical_bytes(failure_unsigned)
                        )
                    ).decode("ascii"),
                },
            )
        else:

            def observer_dead(_ready: dict[str, object]) -> None:
                raise subject.PublicationError("bound observer process is not alive")

            monkeypatch.setattr(subject, "_observer_process_alive", observer_dead)
        if scenario == "preopen_symlink_restart":
            original_write = subject._write_noreplace

            def crash_after_rollback_exchange(path: Path, value: dict[str, object]) -> None:
                if path.name == "pre-open-rollback.json":
                    raise RuntimeError("crash after rollback exchange")
                original_write(path, value)

            monkeypatch.setattr(subject, "_write_noreplace", crash_after_rollback_exchange)
            with pytest.raises(RuntimeError, match="crash after rollback exchange"):
                publish()
            monkeypatch.setattr(subject, "_write_noreplace", original_write)
            activation_intent = subject._verify_stage_envelope(
                control / "activation-intent.json", publisher_private, "activation"
            )
            swap, _ = subject._json(control / "swap.json", "swap")
            temporary = canonical.parent / swap["temporary"]
            assert (
                subject._topology_identity(canonical) == activation_intent["expected_old_identity"]
            )
            assert (
                subject._topology_identity(temporary) == activation_intent["expected_swap_identity"]
            )
        with pytest.raises(subject.PublicationError, match="observer failure was rolled back"):
            publish()
        assert canonical.is_symlink() is (scenario == "preopen_symlink_restart")
        assert pl.read_parquet(canonical / relative).sort("datetime").get_column(
            "open"
        ).to_list() == [1.0, 2.0]
        rollback = subject._verify_stage_envelope(
            control / "pre-open-rollback.json",
            publisher_private,
            "activation",
        )
        assert rollback["phase"] == "rolled_back"
        assert rollback["request_id"] == request_id
        assert rollback["approval_sha256"] == fresh["approval_sha256"]
        assert (
            rollback["activation_intent_sha256"]
            == hashlib.sha256((control / "activation-intent.json").read_bytes()).hexdigest()
        )
        assert (
            rollback["observer_ready_sha256"]
            == hashlib.sha256(observer_ready_path.read_bytes()).hexdigest()
        )
        assert rollback["failure_reason"] == (
            "observer_failure_evidence"
            if scenario == "preopen_marker"
            else "observer_process_unavailable"
        )
        assert source.read_bytes() == source_before
        return

    assert publish() == 0
    assert canonical.is_symlink()
    actual = pl.read_parquet(canonical / relative).sort("datetime")
    assert actual.get_column("open").to_list() == [9.0, 2.0, 3.0]
    assert actual.get_column("volume").to_list() == [90.0, 20.0, 30.0]
    assert source.read_bytes() == source_before
    assert (canonical / relative).stat().st_ino != source_identity[1]
    commit, _ = subject._json(canonical / "commit.json", "commit")
    assert commit["schema"] == "alpha_max_canonical_publication_receipt.v4"
    assert commit["conflict_authority_key_id"] == commit["authority_key_id"]
    assert commit["authorized_conflict_partition_count"] == 1
    assert commit["authorized_replaced_row_count"] == 1
    assert commit["partitions"][0]["publication_mode"] == "authorized_reconciliation"
    assert commit["partitions"][0]["conflict_effects"] == effects
    assert commit["partitions"][0]["merged_targets"] == {
        "predecessor_rows": 2,
        "source_rows": 2,
        "equal_rows": 0,
        "conflict_rows": 1,
        "canonical_only_rows": 1,
        "source_only_rows": 1,
        "target_rows": 3,
    }
    assert commit["source_input"]["rows"] == 2
    assert commit["canonical_before"]["rows"] == 2
    assert commit["canonical_candidate"]["rows"] >= 3
    merged = commit["partitions"][0]["merged_targets"]
    assert merged["equal_rows"] + merged["conflict_rows"] == 1
    assert merged["target_rows"] == merged["source_rows"] + merged["canonical_only_rows"]
    assert merged["target_rows"] == merged["predecessor_rows"] + merged["source_only_rows"]
    window_envelope, _ = subject._json(
        canonical.parent
        / f".{canonical.name}.transactions"
        / request_id
        / "rollback-window-open.json",
        "window",
    )
    window = window_envelope["message"]
    assert window["terminal_receipt_sha256"] == fresh["terminal_receipt_sha256"]
    assert window["approval_sha256"] == fresh["approval_sha256"]
    assert window["swap"] == window["swap_temporary_path"].split("/")[-1]
    assert len(window["swap_receipt_sha256"]) == 64
    assert set(window) == {
        "kind",
        "schema",
        "phase",
        "request_id",
        "run_id",
        "acquisition_request_id",
        "approval_leaf",
        "approval_sha256",
        "authority_key_id",
        "terminal_receipt_sha256",
        "observer_key_id",
        "observer_ready_sha256",
        "observer_query_spec_sha256",
        "candidate",
        "candidate_leaf",
        "predecessor",
        "swap",
        "swap_receipt_sha256",
        "swap_temporary_path",
        "candidate_identity",
        "predecessor_identity",
        "swap_identity",
        "pre_exchange_predecessor_identity",
        "post_exchange_candidate_identity",
        "post_exchange_predecessor_identity",
        "canonical_logical_root_identity",
        "canonical_resolved_root",
        "candidate_inventory_sha256",
        "predecessor_inventory_sha256",
    }
    assert window["candidate"] == str(canonical.resolve())
    assert window["candidate_leaf"] == request_id
    assert window["predecessor"] == window["swap_temporary_path"]
    assert window["candidate_identity"][2] == "directory"
    assert window["canonical_logical_root_identity"][2:] == [
        "symlink",
        f".{canonical.name}.generations/{request_id}",
    ]

    assert publish() == 0
    replay_envelope, _ = subject._json(
        canonical.parent / f".{canonical.name}.transactions" / request_id / "replay-verified.json",
        "replay",
    )
    replay = replay_envelope["message"]
    assert set(replay) == set(window)
    stage_fields = {"schema", "phase", "kind"}
    assert {key: value for key, value in replay.items() if key not in stage_fields} == {
        key: value for key, value in window.items() if key not in stage_fields
    }
    stale_unsigned = json.loads(json.dumps(unsigned))
    stale_unsigned["message"]["acquisition_request_id"] = "9" * 64
    stale_receipt = {
        **stale_unsigned,
        "signature": base64.b64encode(
            terminal_private.sign(
                b"luminaquant.alpha_max.canonical_conflict_authorization.v2\0"
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
        fresh_acquisition_audit_receipt=fresh_path,
        wal_transition_receipt=wal_path,
        source_root=source_root,
        source_report=report,
        authority_public_key=authority_key,
    )
    with pytest.raises(subject.PublicationError, match="binding mismatch"):
        subject._replay_conflict_authorization(
            replay_args,
            final=stale_final,
            terminal_sha256=hashlib.sha256(terminal_receipt.read_bytes()).hexdigest(),
            request_id=request_id,
            hashes=hashes,
            parts=parts,
            terminal={"request_id": request_id, "request_sha256": terminal_request_sha256},
            key_id=hashlib.sha256(terminal_public).hexdigest(),
            terminal_public_key=terminal_public,
        )
    repair_receipt_path.write_bytes(subject.canonical_bytes(repair_receipt))
    incomplete_controls = argv.copy()
    wal_index = incomplete_controls.index("--wal-transition-receipt")
    del incomplete_controls[wal_index : wal_index + 2]
    with pytest.raises(SystemExit):
        publish(incomplete_controls)
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
        publish()


def test_publication_stage_envelope_is_domain_signed_and_tamper_evident(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    envelope = subject._stage_envelope(key, "activation", {"schema": "test", "phase": "activated"})
    path = tmp_path / "activated.json"
    path.write_bytes(subject.canonical_bytes(envelope))
    assert subject._verify_stage_envelope(path, key, "activation")["kind"] == "activation"
    envelope["message"]["phase"] = "tampered"
    path.write_bytes(subject.canonical_bytes(envelope))
    with pytest.raises(subject.PublicationError):
        subject._verify_stage_envelope(path, key, "activation")


def test_publication_admission_rejects_capacity_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    item = {
        "relative": "market_ohlcv_1s/binance/BTC_USDT/2024-01.parquet",
        "source": tmp_path / "source",
    }
    item["source"].write_bytes(b"x")
    monkeypatch.setattr(subject, "_parquet_materialized_bytes", lambda _: 1)
    required = (
        subject._STORAGE["host_reserve_bytes"]
        + 1
        + (1 + subject._PARQUET_ENCODING_RESERVE_BYTES)
        + subject._PUBLICATION_CONTROL_RESERVE_BYTES
    )
    monkeypatch.setattr(subject, "_host_free_bytes", lambda: required)
    with pytest.raises(subject.PublicationError):
        subject._publication_admission(candidate, item, 1)


@pytest.mark.parametrize("boundary", ("write", "file_fsync", "install", "directory_fsync"))
def test_write_noreplace_recovers_from_each_atomic_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
):
    path = tmp_path / "receipt.json"
    value = {"schema": "test", "count": 1}
    original = getattr(
        subject,
        {
            "write": "_receipt_write",
            "file_fsync": "_receipt_fsync",
            "install": "_receipt_install_noreplace",
            "directory_fsync": "_fsync_dir",
        }[boundary],
    )

    def crash(*args, **kwargs):
        raise RuntimeError(boundary)

    monkeypatch.setattr(
        subject,
        {
            "write": "_receipt_write",
            "file_fsync": "_receipt_fsync",
            "install": "_receipt_install_noreplace",
            "directory_fsync": "_fsync_dir",
        }[boundary],
        crash,
    )
    with pytest.raises(RuntimeError, match=boundary):
        subject._write_noreplace(path, value)
    monkeypatch.setattr(
        subject,
        {
            "write": "_receipt_write",
            "file_fsync": "_receipt_fsync",
            "install": "_receipt_install_noreplace",
            "directory_fsync": "_fsync_dir",
        }[boundary],
        original,
    )
    subject._write_noreplace(path, value)
    assert path.read_bytes() == subject.canonical_bytes(value)
    assert not list(tmp_path.glob(".receipt.json.tmp-*"))
