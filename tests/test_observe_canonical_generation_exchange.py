from __future__ import annotations

import importlib.util
import base64
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest
import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from lumina_quant.market_data import MarketDataRepository, upsert_futures_feature_points_rows
from lumina_quant.storage.parquet.ohlcv_repo import ParquetMarketDataRepository

MODULE = Path(__file__).parents[1] / "scripts/research/observe_canonical_generation_exchange.py"
spec = importlib.util.spec_from_file_location("generation_observer", MODULE)
observer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(observer)

PUBLISHER_MODULE = (
    Path(__file__).parents[1] / "scripts/research/publish_alpha_max_eligible_source.py"
)
publisher_spec = importlib.util.spec_from_file_location("generation_publisher", PUBLISHER_MODULE)
publisher = importlib.util.module_from_spec(publisher_spec)
assert publisher_spec.loader is not None
publisher_spec.loader.exec_module(publisher)

MANAGER_MODULE = (
    Path(__file__).parents[1] / "scripts/research/manage_alpha_max_publication_window.py"
)
manager_spec = importlib.util.spec_from_file_location("generation_manager", MANAGER_MODULE)
manager = importlib.util.module_from_spec(manager_spec)
assert manager_spec.loader is not None
manager_spec.loader.exec_module(manager)


def generation(tmp_path: Path, name: str, content: bytes = b"data") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "market.parquet").write_bytes(content)
    return root


def test_observes_complete_old_new_and_rollback_generations(tmp_path: Path) -> None:
    old, new = generation(tmp_path, "old"), generation(tmp_path, "new", b"new")
    active = tmp_path / "canonical"
    os.symlink(old, active)
    first = observer.sample(active, 0, observer.inventory(old))
    active.unlink()
    os.symlink(new, active)
    second = observer.sample(active, 1, observer.inventory(new))
    active.unlink()
    os.symlink(old, active)
    rollback = observer.sample(active, 2, observer.inventory(old))
    assert [first["generation"], second["generation"], rollback["generation"]] == [
        "old",
        "new",
        "old",
    ]
    assert first["inventory_sha256"] != second["inventory_sha256"]


def test_empty_generation_is_rejected_and_physical_root_is_supported(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    active = tmp_path / "canonical"
    os.symlink(empty, active)
    with pytest.raises(observer.ObserverError, match="empty"):
        observer.sample(active, 0, "0" * 64)
    active.unlink()
    active.mkdir()
    (active / "market.parquet").write_bytes(b"data")
    observation = observer.sample(active, 0, observer.inventory(active))
    assert observation["generation"] == "canonical"


def test_representative_sentinel_rejects_old_view_drift(tmp_path: Path) -> None:
    root = generation(tmp_path, "old")
    observation = observer.sample_physical(root, 0)
    (root / "market.parquet").write_bytes(b"changed")
    with pytest.raises(observer.ObserverError, match="changed after readiness"):
        observer._verify_representatives(root, observation["representatives"])


def test_publisher_envelope_rejects_plain_tampered_and_wrong_key() -> None:
    signer = Ed25519PrivateKey.generate()
    verifier = Ed25519PrivateKey.generate().public_key()
    inner = dict.fromkeys(observer.ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS, "bound")
    inner.update(
        {
            "schema": "alpha_max_publication_activation.v1",
            "kind": "activation",
        }
    )
    public = signer.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    unsigned = {
        "schema": observer.STAGE_ENVELOPE_SCHEMA,
        "kind": "activation",
        "authority_key_id": observer.digest(public),
        "message": inner,
    }
    envelope = {
        **unsigned,
        "signature": base64.b64encode(
            signer.sign(observer.DOMAIN + observer.canonical_bytes(unsigned))
        ).decode("ascii"),
    }
    raw = observer.canonical_bytes(envelope)
    assert (
        observer._publisher_envelope(
            envelope,
            raw,
            expected_kind="activation",
            expected_schema="alpha_max_publication_activation.v1",
            public_key=signer.public_key(),
            key_id=envelope["authority_key_id"],
            name="activated.json",
        )
        == inner
    )
    with pytest.raises(observer.ObserverError, match="signature"):
        observer._publisher_envelope(
            envelope,
            raw,
            expected_kind="activation",
            expected_schema="alpha_max_publication_activation.v1",
            public_key=verifier,
            key_id=envelope["authority_key_id"],
            name="activated.json",
        )
    tampered = {**envelope, "message": {**inner, "phase": "tampered"}}
    with pytest.raises(observer.ObserverError, match="signature"):
        observer._publisher_envelope(
            tampered,
            observer.canonical_bytes(tampered),
            expected_kind="activation",
            expected_schema="alpha_max_publication_activation.v1",
            public_key=signer.public_key(),
            key_id=envelope["authority_key_id"],
            name="activated.json",
        )
    with pytest.raises(observer.ObserverError, match="fields"):
        observer._publisher_envelope(
            inner,
            observer.canonical_bytes(inner),
            expected_kind="activation",
            expected_schema="alpha_max_publication_activation.v1",
            public_key=signer.public_key(),
            key_id=envelope["authority_key_id"],
            name="activated.json",
        )


def test_observer_rejects_private_key_aliasing_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(observer, "_validate_paths", lambda *_args: None)
    monkeypatch.setattr(observer, "_query", lambda _raw: {})
    monkeypatch.setattr(observer, "_private_key", lambda _fd: key)
    monkeypatch.setattr(observer, "_publisher_public_key", lambda _path: key.public_key())

    with pytest.raises(observer.ObserverError, match="distinct from publisher"):
        observer.main(
            [
                "--canonical-root",
                str(tmp_path / "canonical"),
                "--transaction-root",
                str(tmp_path / "transaction"),
                "--request-id",
                observer.ACQUISITION_REQUEST_ID,
                "--publisher-public-key",
                str(tmp_path / "publisher.pub"),
                "--approval-sha256",
                "b" * 64,
                "--observer-key-fd",
                "0",
                "--query-spec",
                "{}",
            ]
        )


def test_observer_rejects_foreign_acquisition_request(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    transaction = tmp_path / ".canonical.transactions" / ("a" * 64)
    with pytest.raises(observer.ObserverError, match="binding"):
        observer._validate_paths(canonical.absolute(), transaction.absolute(), "a" * 64)


def test_terminal_is_signed_and_noclobber(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "observer.key"
    from cryptography.hazmat.primitives import serialization

    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    key_fd = os.open(key_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        receipt = observer.sign_terminal(key_fd, {"run_id": observer.RUN_ID, "outcome": "PASS"})
    finally:
        os.close(key_fd)
    assert receipt["authority_key_id"]
    output = tmp_path / "observer-terminal-pass.json"
    observer.write_noreplace(output, receipt)
    observer.write_noreplace(output, receipt)
    conflicting = dict(receipt)
    conflicting["signature"] = "different"
    with pytest.raises(observer.ObserverError, match="conflicts"):
        observer.write_noreplace(output, conflicting)


@pytest.mark.parametrize(
    "boundary", ("write", "file_fsync", "install", "install_observed_error", "parent_fsync")
)
def test_receipt_persistence_recovers_each_crash_boundary_without_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    output = tmp_path / "observer-ready.json"
    receipt = {"schema": "receipt.v1", "outcome": "PASS", "value": [1, 2, 3]}
    real_fsync = observer.os.fsync
    real_rename = observer._rename_noreplace

    with monkeypatch.context() as fault:
        if boundary == "write":

            def fail_write(fd: int, data: object) -> int:
                raise OSError("injected write failure")

            fault.setattr(observer.os, "write", fail_write)
        elif boundary == "file_fsync":

            def fail_file_fsync(fd: int) -> None:
                if stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError("injected file fsync failure")
                real_fsync(fd)

            fault.setattr(observer.os, "fsync", fail_file_fsync)
        elif boundary == "install":

            def fail_install(source: Path, destination: Path) -> None:
                raise OSError("injected install failure")

            fault.setattr(observer, "_rename_noreplace", fail_install)
        elif boundary == "install_observed_error":

            def install_then_fail(source: Path, destination: Path) -> None:
                real_rename(source, destination)
                raise OSError("injected install observed error failure")

            fault.setattr(observer, "_rename_noreplace", install_then_fail)
        else:
            file_synced = False

            def fail_parent_fsync(fd: int) -> None:
                nonlocal file_synced
                if stat.S_ISREG(os.fstat(fd).st_mode):
                    file_synced = True
                    real_fsync(fd)
                    return
                assert file_synced
                raise OSError("injected parent fsync failure")

            fault.setattr(observer.os, "fsync", fail_parent_fsync)

        with pytest.raises(OSError, match=f"injected {boundary.replace('_', ' ')} failure"):
            observer.write_noreplace(output, receipt)

    if boundary in ("install_observed_error", "parent_fsync"):
        assert output.read_bytes() == observer.canonical_bytes(receipt)
    else:
        assert not output.exists()

    observer.write_noreplace(output, receipt)

    assert output.read_bytes() == observer.canonical_bytes(receipt)
    assert list(tmp_path.iterdir()) == [output]
    conflicting = dict(receipt)
    conflicting["outcome"] = "FAIL"
    with pytest.raises(observer.ObserverError, match="conflicts"):
        observer.write_noreplace(output, conflicting)


def test_query_contract_rejects_missing_feature_or_digest() -> None:
    query = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "aggregate_timeframe": "1m",
        "feature_name": "funding_rate",
        "old_result_sha256": "a" * 64,
        "new_result_sha256": "b" * 64,
    }
    assert observer._query(__import__("json").dumps(query)) == query
    query.pop("feature_name")
    with pytest.raises(observer.ObserverError, match="fields"):
        observer._query(__import__("json").dumps(query))


def test_facade_digest_uses_1s_aggregate_and_feature_query() -> None:
    calls: list[str] = []

    class Frame:
        columns = ["timestamp_ms", "funding_rate"]

        def is_empty(self) -> bool:
            return False

        def to_dicts(self) -> list[dict[str, int]]:
            return [{"timestamp_ms": 1, "funding_rate": 2}]

        def write_json(self) -> str:
            return '[{"timestamp_ms":1,"funding_rate":2}]'

        def select(self, *_: str) -> Frame:
            return self

    class Facade:
        def load_ohlcv(self, *, timeframe: str, **_: object) -> Frame:
            calls.append(timeframe)
            return Frame()

        def load_futures_feature_points(self, **_: object) -> Frame:
            calls.append("features")
            return Frame()

    query = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "aggregate_timeframe": "1m",
        "feature_name": "funding_rate",
        "old_result_sha256": "a" * 64,
        "new_result_sha256": "b" * 64,
    }
    assert len(observer.facade_digest(Facade(), query)) == 64
    assert calls == ["1s", "1m", "features"]


def _seed_public_generation(root: Path, *, close_offset: float, funding_rate: float) -> None:
    frame = pl.DataFrame(
        {
            "datetime": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:59Z",
                "2026-01-01T00:01:00Z",
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [
                100.5 + close_offset,
                101.5 + close_offset,
                102.5 + close_offset,
                103.5 + close_offset,
            ],
            "volume": [1.0, 2.0, 3.0, 4.0],
        }
    )
    repo = ParquetMarketDataRepository(root)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=frame)
    upsert_futures_feature_points_rows(
        str(root),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[{"timestamp_ms": 1_767_225_600_000, "funding_rate": funding_rate}],
    )


@pytest.mark.parametrize("expected_outcome", ("PASS", "FAIL"))
def test_real_observer_spans_publisher_exchange_and_waits_for_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_outcome: str,
) -> None:
    request_id = observer.ACQUISITION_REQUEST_ID
    approval_sha256 = "b" * 64
    generations = tmp_path / ".canonical.generations"
    generations.mkdir()
    predecessor = generations / "predecessor"
    candidate = generations / request_id
    predecessor.mkdir()
    candidate.mkdir()
    _seed_public_generation(predecessor, close_offset=0.0, funding_rate=0.0001)
    _seed_public_generation(candidate, close_offset=10.0, funding_rate=0.0002)

    canonical = tmp_path / "canonical"
    os.symlink(f"{generations.name}/{predecessor.name}", canonical)
    temporary = tmp_path / ".canonical.integration.swap"
    os.symlink(f"{generations.name}/{candidate.name}", temporary)
    transaction = tmp_path / ".canonical.transactions" / request_id
    transaction.mkdir(parents=True)
    transaction.chmod(0o700)

    key = Ed25519PrivateKey.generate()
    private_key = tmp_path / "observer.seed"
    private_key.write_bytes(
        key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    private_key.chmod(0o400)
    observer_public_key = tmp_path / "observer.public"
    observer_public_key.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    observer_public_key.chmod(0o444)
    publisher_key = Ed25519PrivateKey.generate()
    publisher_public_key = tmp_path / "publisher.public"
    publisher_public_key.write_bytes(
        publisher_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    )
    publisher_public_key.chmod(0o444)
    authority_key = Ed25519PrivateKey.generate()
    authority_private_key = tmp_path / "authority.seed"
    authority_public_key = tmp_path / "authority.public"
    authority_private_key.write_bytes(
        authority_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    authority_private_key.chmod(0o400)
    authority_public_key.write_bytes(
        authority_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    )
    authority_public_key.chmod(0o444)
    observer_key_id = observer.public_key_id(key.public_key())
    query = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "start": "2026-01-01",
        "end": "2026-01-02",
        "aggregate_timeframe": "1m",
        "feature_name": "funding_rate",
        "old_result_sha256": "0" * 64,
        "new_result_sha256": "0" * 64,
    }
    query["old_result_sha256"] = observer.facade_digest(
        MarketDataRepository(str(predecessor)), query
    )
    actual_new_sha256 = observer.facade_digest(MarketDataRepository(str(candidate)), query)
    query["new_result_sha256"] = actual_new_sha256 if expected_outcome == "PASS" else "f" * 64
    assert query["old_result_sha256"] != actual_new_sha256

    new_view_read = threading.Event()
    original_facade_digest = observer.facade_digest

    def tracked_facade_digest(repository: MarketDataRepository, value: dict[str, object]) -> str:
        result = original_facade_digest(repository, value)
        if result != query["old_result_sha256"]:
            new_view_read.set()
        return result

    monkeypatch.setattr(observer, "facade_digest", tracked_facade_digest)
    original_inventory = observer.inventory
    inventory_calls: dict[str, int] = {}

    def tracked_inventory(root: Path) -> str:
        path = str(root)
        inventory_calls[path] = inventory_calls.get(path, 0) + 1
        return original_inventory(root)

    monkeypatch.setattr(observer, "inventory", tracked_inventory)
    results: list[int] = []
    failures: list[BaseException] = []
    key_fd = os.open(private_key, os.O_RDONLY | os.O_NOFOLLOW)

    def run_observer() -> None:
        try:
            results.append(
                observer.main(
                    [
                        "--canonical-root",
                        str(canonical),
                        "--transaction-root",
                        str(transaction),
                        "--request-id",
                        request_id,
                        "--publisher-public-key",
                        str(publisher_public_key),
                        "--approval-sha256",
                        approval_sha256,
                        "--observer-key-fd",
                        str(key_fd),
                        "--query-spec",
                        json.dumps(query, sort_keys=True),
                        "--timeout-seconds",
                        "10",
                        "--interval-seconds",
                        "0.01",
                    ]
                )
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run_observer, daemon=True)
    thread.start()
    ready_path = transaction / "observer-ready.json"
    deadline = time.monotonic() + 5
    while not ready_path.exists() and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready_path.exists(), failures
    ready, ready_raw = observer._read_json(ready_path, "observer readiness")

    swap = {
        "request_id": request_id,
        "phase": "swap_ready",
        "candidate": candidate.name,
        "temporary": temporary.name,
        "mode": "exchange",
    }
    final = {
        "canonical_root": str(canonical),
        "predecessor_path": str(predecessor),
        "predecessor_identity": [
            os.lstat(predecessor).st_dev,
            os.lstat(predecessor).st_ino,
        ],
        "approval_sha256": approval_sha256,
        "observer_key_id": observer_key_id,
        "observer_ready_sha256": observer.digest(ready_raw),
        "observer_query_spec_sha256": observer.digest(observer.canonical_bytes(query)),
    }
    authority_key_id = observer.public_key_id(authority_key.public_key())
    terminal_sha256 = "8" * 64
    writer = ParquetMarketDataRepository(canonical)
    with writer.generation_lock(exclusive=True):
        publisher._rename_exchange(temporary, canonical)
        publisher._fsync_dir(canonical.parent)
        publisher._write_open_window_receipts(
            control=transaction,
            request_id=request_id,
            publisher_key=publisher_key,
            key_id=authority_key_id,
            terminal_sha256=terminal_sha256,
            candidate=candidate,
            final=final,
            swap=swap,
            temporary=temporary,
            replay=expected_outcome == "FAIL",
        )

    assert new_view_read.wait(timeout=5), failures
    if expected_outcome == "PASS":
        assert thread.is_alive()
        assert not (transaction / "observer-terminal-pass.json").exists()
        with writer.generation_lock(exclusive=True):
            publisher._write_open_window_receipts(
                control=transaction,
                request_id=request_id,
                publisher_key=publisher_key,
                key_id=authority_key_id,
                terminal_sha256=terminal_sha256,
                candidate=candidate,
                final=final,
                swap=swap,
                temporary=temporary,
                replay=True,
            )
    thread.join(timeout=5)
    os.close(key_fd)

    assert not thread.is_alive()
    assert failures == []
    assert results == [0 if expected_outcome == "PASS" else 1]
    assert inventory_calls.get(str(predecessor)) == 1
    evidence_path = transaction / (
        "observer-observations-pass.json"
        if expected_outcome == "PASS"
        else "observer-observations-fail.json"
    )
    terminal_path = transaction / (
        "observer-terminal-pass.json"
        if expected_outcome == "PASS"
        else "observer-terminal-fail.json"
    )
    evidence, _ = observer._read_json(evidence_path, "observer evidence")
    terminal, terminal_raw = observer._read_json(terminal_path, "observer terminal")
    assert evidence["outcome"] == expected_outcome
    assert terminal["message"]["state"] == "W2_REPLAY_OK"
    assert terminal["message"]["stage"] == "observer"
    assert terminal["authority_key_id"] == observer_key_id
    assert terminal["authority_key_id"] != authority_key_id
    assert ready["observer_pid"] == os.getpid()

    if expected_outcome == "PASS":
        assert evidence["old_before_new"] is True
        assert len(evidence["observations"]) == 2
        assert evidence["replay_receipt_sha256"] == observer.digest(
            (transaction / "replay-verified.json").read_bytes()
        )
        return

    assert terminal["message"]["failure_reason"] == "observer_failed"
    assert terminal["message"]["failure_payload_sha256"] == terminal["message"]["evidence_sha256"]
    context = {
        **publisher._verify_stage_envelope(
            transaction / "activated.json", publisher_key, "activation"
        ),
        "_authority_public_key": str(authority_public_key),
        "_observer_public_key": str(observer_public_key),
        "_publisher_public_key": str(publisher_public_key),
        "_canonical_root": str(canonical),
    }
    assert manager.state(transaction, context) == "W2_REPLAY_OK"
    guard_timestamp_ns = max(time.time_ns(), terminal["message"]["timestamp_ns"] + 1)
    guard_message = {
        "kind": "guard_capacity",
        "outcome": "PASS",
        "run_id": manager.RUN_ID,
        "request_id": request_id,
        "approval_leaf": manager.APPROVAL_LEAF,
        "approval_sha256": approval_sha256,
        "state": "W2_REPLAY_OK",
        "stage": manager.STATE_STAGE["W2_REPLAY_OK"],
        "timestamp_ns": guard_timestamp_ns,
        "identities": manager._context_identities(context),
        "inventories": manager._context_inventories(context),
        "evidence_sha256": "0" * 64,
        "capacity_path": manager.CAPACITY_PATH,
        "free_bytes": manager.HOST_RESERVE_BYTES + 2,
        "required_bytes": manager.HOST_RESERVE_BYTES + 1,
        "sample_sequence": 1,
    }
    guard_fd = os.open(authority_private_key, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        guard = manager.seal_stage_envelope(guard_fd, guard_message)
    finally:
        os.close(guard_fd)
    guard_path = transaction / "observer-rollback-guard.json"
    manager.write_noreplace(guard_path, guard)
    issued_ns = max(time.time_ns(), guard_timestamp_ns + 1)
    authorization_message = {
        "kind": "rollback",
        "run_id": manager.RUN_ID,
        "request_id": request_id,
        "approval_leaf": manager.APPROVAL_LEAF,
        "approval_sha256": approval_sha256,
        "state": "W2_REPLAY_OK",
        "issued_ns": issued_ns,
        "expires_ns": issued_ns + 60 * 1_000_000_000,
        "context_sha256": manager._context_hash(context),
        "identities": manager._context_identities(context),
        "inventories": manager._context_inventories(context),
        "reason": "observer_failed",
        "failure_evidence_kind": "observer",
        "failure_evidence_sha256": manager.sha(terminal_raw),
    }
    authority_fd = os.open(authority_private_key, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        authorization = manager.seal_action_envelope(authority_fd, authorization_message)
    finally:
        os.close(authority_fd)
    authorization_path = transaction / "observer-rollback-authorization.json"
    manager.write_noreplace(authorization_path, authorization)
    assert (
        manager.decide(
            type(
                "Args",
                (),
                {
                    "authorization": str(authorization_path),
                    "failure_evidence": str(terminal_path),
                    "guard_receipt": str(guard_path),
                },
            )(),
            transaction,
            "rollback",
            context,
        )
        == 0
    )
    assert manager.state(transaction, context) == "W9_ROLLED_BACK"
    assert canonical.resolve(strict=True) == predecessor
