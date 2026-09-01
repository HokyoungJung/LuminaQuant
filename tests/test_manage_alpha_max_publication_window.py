from __future__ import annotations

import importlib.util
import hashlib
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from types import SimpleNamespace
import time

MODULE = Path(__file__).parents[1] / "scripts/research/manage_alpha_max_publication_window.py"
spec = importlib.util.spec_from_file_location("publication_window", MODULE)
window = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(window)


@pytest.fixture(autouse=True)
def _stable_test_clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    now_ns = time.time_ns()
    monkeypatch.setattr(window.time, "time_ns", lambda: now_ns)
    monkeypatch.setattr(window, "CAPACITY_PATH", str(tmp_path))


def put(root: Path, name: str) -> None:
    window.write_noreplace(root / name, {"schema": "test", "name": name})


def test_success_prefix_derives_each_state_and_rejects_incomplete_pair_or_gap(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert window.state(empty) == "W0_PRE_ACTIVATION"

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    put(incomplete, "activated.json")
    with pytest.raises(window.WindowError, match="incomplete"):
        window.state(incomplete)

    expected_prefixes = (
        (2, "W1_ACTIVATED_OPEN"),
        (3, "W2_REPLAY_OK"),
        (4, "W3_OBSERVER_OK"),
        (5, "W4_POSTVERIFY_OK"),
        (6, "W5_BACKTEST_OK"),
        (7, "W5_BACKTEST_OK"),
        (8, "W6_READY_TO_FINALIZE"),
    )
    for prefix, expected in expected_prefixes:
        root = tmp_path / str(prefix)
        root.mkdir()
        for name in window.PASS_NAMES[:prefix]:
            put(root, name)
        assert window.state(root) == expected

    gap = tmp_path / "gap"
    gap.mkdir()
    put(gap, "activated.json")
    put(gap, "observer-terminal-pass.json")
    with pytest.raises(window.WindowError, match="contiguous"):
        window.state(gap)


@pytest.mark.parametrize("state,reasons", list(window.REASONS.items()))
def test_failure_reasons_are_closed_and_state_local(state: str, reasons: set[str]) -> None:
    for reason in reasons:
        assert reason in window.REASONS[state]
        assert all(
            reason not in allowed for other, allowed in window.REASONS.items() if other != state
        )


def test_canonical_json_receipt_is_noclobber_and_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"a": 1}
    window.write_noreplace(path, value)
    window.write_noreplace(path, value)
    with pytest.raises(window.WindowError, match="conflicts"):
        window.write_noreplace(path, {"a": 2})


def test_terminal_repeat_requires_exact_action_and_authorization(tmp_path: Path) -> None:
    root = tmp_path / "transaction"
    root.mkdir()
    window.write_noreplace(
        root / "rolled-back.json",
        {
            "schema": "alpha_max_window_terminal.v1",
            "action": "rollback",
            "state": "W9_ROLLED_BACK",
            "authorization_sha256": "a" * 64,
        },
    )
    receipt, _ = window.read_json(root / "rolled-back.json", "terminal")
    assert receipt["state"] == "W9_ROLLED_BACK"
    assert receipt["action"] != "finalize"


def test_w6_allows_only_late_infrastructure_failure() -> None:
    assert window.REASONS["W6_READY_TO_FINALIZE"] == {"late_infrastructure_failure"}
    assert "final_audit_capacity_failed" not in window.REASONS["W6_READY_TO_FINALIZE"]


@pytest.fixture
def authority(tmp_path: Path) -> tuple[Path, Path]:
    key = Ed25519PrivateKey.generate()
    private, public = tmp_path / "authority.seed", tmp_path / "authority.pub"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    public.chmod(0o444)
    return private, public


def _stage_message(
    *,
    kind: str,
    state: str,
    outcome: str = "PASS",
    timestamp_ns: int | None = None,
    **extra: object,
) -> dict[str, object]:
    message: dict[str, object] = {
        "kind": kind,
        "outcome": outcome,
        "run_id": window.RUN_ID,
        "request_id": "a" * 64,
        "approval_leaf": window.APPROVAL_LEAF,
        "approval_sha256": "b" * 64,
        "state": state,
        "stage": kind,
        "timestamp_ns": time.time_ns() if timestamp_ns is None else timestamp_ns,
        "identities": {},
        "inventories": {},
        "evidence_sha256": "c" * 64,
    }
    message.update(extra)
    if outcome == "FAIL":
        message.setdefault(
            "failure_reason",
            {
                "replay": "replay_failed",
                "observer": "observer_failed",
                "postverify": "postverify_failed",
                "deterministic_backtest": "deterministic_backtest_failed",
                "final_audit": "final_audit_inventory_failed",
                "unit_exit": "replay_crashed",
                "integrity": "immediate_postactivation_integrity_failure",
            }.get(kind, "late_infrastructure_failure"),
        )
        message.setdefault("failure_payload_sha256", message["evidence_sha256"])
    return message


def _seal_stage(tmp_path: Path, authority: tuple[Path, Path], name: str, **kwargs: object) -> Path:
    private, _ = authority
    fd = os.open(private, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        sealed = window.seal_stage_envelope(fd, _stage_message(**kwargs))
    finally:
        os.close(fd)
    path = tmp_path / name
    window.write_noreplace(path, sealed)
    return path


def _seal_publisher_receipt(
    root: Path,
    publisher: tuple[Path, Path],
    name: str,
    message: dict[str, object],
) -> Path:
    fd = os.open(publisher[0], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        sealed = window._seal(
            schema="alpha_max_publication_stage_envelope.v1",
            domain=window.STAGE_DOMAIN,
            private_key_fd=fd,
            message=message,
        )
    finally:
        os.close(fd)
    path = root / name
    window.write_noreplace(path, sealed)
    return path


@pytest.mark.parametrize("state,reasons", list(window.REASONS.items()))
def test_all_rollback_reasons_have_only_their_approved_evidence_kinds(
    state: str, reasons: set[str]
) -> None:
    for reason in reasons:
        accepted = window.REASON_EVIDENCE_KINDS[reason]
        assert accepted
        for other_state, other_reasons in window.REASONS.items():
            if other_state != state:
                assert reason not in other_reasons


def test_stage_envelope_rejects_tamper_cross_kind_and_state(
    tmp_path: Path, authority: tuple[Path, Path]
) -> None:
    path = _seal_stage(
        tmp_path,
        authority,
        "guard.json",
        kind="guard_capacity",
        state="W1_ACTIVATED_OPEN",
        capacity_path=window.CAPACITY_PATH,
        free_bytes=window.HOST_RESERVE_BYTES + 2,
        required_bytes=window.HOST_RESERVE_BYTES + 1,
        sample_sequence=1,
    )
    assert (
        window.verify_stage_envelope(
            path, authority[1], kind="guard_capacity", request_id="a" * 64, approval_sha256="b" * 64
        )["outcome"]
        == "PASS"
    )
    with pytest.raises(window.WindowError):
        window.verify_stage_envelope(
            path, authority[1], kind="cgroup", request_id="a" * 64, approval_sha256="b" * 64
        )
    receipt, _ = window.read_json(path, "guard")
    receipt["message"]["state"] = "W2_REPLAY_OK"
    path.unlink()
    window.write_noreplace(path, receipt)
    with pytest.raises(window.WindowError, match="signature"):
        window.verify_stage_envelope(
            path, authority[1], kind="guard_capacity", request_id="a" * 64, approval_sha256="b" * 64
        )


def test_guard_rejects_fail_stale_under_capacity_and_current_statvfs(
    tmp_path: Path, authority: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    context = {
        "_authority_public_key": str(authority[1]),
        "request_id": "a" * 64,
        "approval_sha256": "b" * 64,
        **{key: {} for key in ()},
    }
    context.update(
        {
            "candidate_identity": [],
            "predecessor_identity": [],
            "swap_identity": [],
            "candidate_inventory_sha256": "d" * 64,
            "predecessor_inventory_sha256": "e" * 64,
        }
    )
    context_identities = window._context_identities(context)
    context_inventories = window._context_inventories(context)
    monkeypatch.setattr(
        window.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=window.HOST_RESERVE_BYTES + 2, f_frsize=1),
    )
    for name, outcome, timestamp, free in (
        ("fail", "FAIL", time.time_ns(), window.HOST_RESERVE_BYTES + 2),
        (
            "stale",
            "PASS",
            time.time_ns() - window.MAX_GUARD_AGE_NS - 1,
            window.HOST_RESERVE_BYTES + 2,
        ),
        ("under", "PASS", time.time_ns(), window.HOST_RESERVE_BYTES + 1),
    ):
        path = _seal_stage(
            tmp_path,
            authority,
            name,
            kind="guard_capacity",
            state="W1_ACTIVATED_OPEN",
            outcome=outcome,
            timestamp_ns=timestamp,
            identities=context_identities,
            inventories=context_inventories,
            capacity_path=window.CAPACITY_PATH,
            free_bytes=free,
            required_bytes=window.HOST_RESERVE_BYTES + 1,
            sample_sequence=1,
        )
        with pytest.raises(window.WindowError):
            window._verify_guard(path, context, "W1_ACTIVATED_OPEN")
    valid = _seal_stage(
        tmp_path,
        authority,
        "valid",
        kind="guard_capacity",
        state="W1_ACTIVATED_OPEN",
        identities=context_identities,
        inventories=context_inventories,
        capacity_path=window.CAPACITY_PATH,
        free_bytes=window.HOST_RESERVE_BYTES + 2,
        required_bytes=window.HOST_RESERVE_BYTES + 1,
        sample_sequence=1,
    )
    monkeypatch.setattr(
        window.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=window.HOST_RESERVE_BYTES + 1, f_frsize=1),
    )
    with pytest.raises(window.WindowError, match="current host capacity"):
        window._verify_guard(valid, context, "W1_ACTIVATED_OPEN")


def test_open_topology_rejects_identity_and_inventory_drift_before_exchange(tmp_path: Path) -> None:
    generations = tmp_path / ".canonical.generations"
    generations.mkdir()
    candidate, predecessor = generations / ("a" * 64), generations / "old"
    candidate.mkdir()
    predecessor.mkdir()
    (candidate / "bars.parquet").write_bytes(b"candidate")
    (predecessor / "bars.parquet").write_bytes(b"predecessor")
    canonical = tmp_path / "canonical"
    os.symlink(f"{generations.name}/{candidate.name}", canonical)
    swap = tmp_path / ".canonical.swap"
    os.symlink(f"{generations.name}/{candidate.name}", swap)
    context = {
        "candidate": str(candidate),
        "candidate_leaf": candidate.name,
        "predecessor": str(predecessor),
        "swap": swap.name,
        "swap_temporary_path": str(swap),
        "canonical_resolved_root": str(candidate),
        "candidate_identity": window.identity(candidate),
        "predecessor_identity": window.identity(predecessor),
        "swap_identity": window.identity(swap),
        "canonical_logical_root_identity": window.identity(canonical),
        "candidate_inventory_sha256": window.inventory(candidate),
        "predecessor_inventory_sha256": window.inventory(predecessor),
    }
    window._verify_open_topology(context, canonical, phase="before")
    (predecessor / "unexpected.parquet").write_bytes(b"drift")
    with pytest.raises(window.WindowError, match="topology"):
        window._verify_open_topology(context, canonical, phase="before")


def test_publisher_topology_identity_supports_directory_and_symlink(tmp_path: Path) -> None:
    directory = tmp_path / "generation"
    directory.mkdir()
    link = tmp_path / "canonical"
    os.symlink(directory.name, link)
    assert window.identity(directory)[2] == "directory"
    assert window.identity(link) == [
        os.lstat(link).st_dev,
        os.lstat(link).st_ino,
        "symlink",
        directory.name,
    ]
    assert window.identity(link) == window.identity(link)


def _make_keypair(root: Path, name: str) -> tuple[Path, Path]:
    key = Ed25519PrivateKey.generate()
    private, public = root / f"{name}.seed", root / f"{name}.pub"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    private.chmod(0o400)
    public.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    public.chmod(0o444)
    return private, public


def _pass_context_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    terminal_message: dict[str, object] | None = None,
    key_ids: dict[str, str] | None = None,
) -> tuple[dict[str, object], SimpleNamespace]:
    ids = key_ids or {
        "authority": "1" * 64,
        "observer": "2" * 64,
        "publisher": "3" * 64,
    }
    message = terminal_message or {
        "scope": "acquisition",
        "request_id": window.ACQUISITION_REQUEST,
        "terminal_state": {"kind": "SUCCEEDED"},
        "target_results": [{"return_code": 0}, {"return_code": 0}],
    }
    terminal = SimpleNamespace(
        message=message,
        key_id=ids["authority"],
        receipt_sha256="4" * 64,
    )
    activation = dict.fromkeys(window.ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS | {"kind"})
    activation.update(
        {
            "schema": window.PASS_SCHEMAS[0],
            "kind": "activation",
            "phase": "activated",
            "request_id": "a" * 64,
            "run_id": window.RUN_ID,
            "acquisition_request_id": window.ACQUISITION_REQUEST,
            "approval_leaf": window.APPROVAL_LEAF,
            "terminal_receipt_sha256": terminal.receipt_sha256,
            "authority_key_id": terminal.key_id,
            "observer_key_id": ids["observer"],
        }
    )
    monkeypatch.setattr(window, "verify_signed_receipt", lambda *_args: terminal)
    monkeypatch.setattr(
        window,
        "_public_key",
        lambda path, _label: (
            object(),
            ids[
                "authority"
                if path.name == "authority.pub"
                else "observer"
                if path.name == "observer.pub"
                else "publisher"
            ],
        ),
    )
    monkeypatch.setattr(window, "_verify_publisher_receipt", lambda *_args, **_kwargs: activation)
    monkeypatch.setattr(window, "_validate_context_paths", lambda *_args: None)
    return activation, terminal


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("scope", "one_touch"),
        ("request_id", "f" * 64),
        ("terminal_state", {"kind": "FAILED", "failed_command_index": 0}),
        ("target_results", [{"return_code": 0}, {"return_code": 1}]),
    ),
)
def test_pass_context_rejects_wrong_or_failed_terminal_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    message: dict[str, object] = {
        "scope": "acquisition",
        "request_id": window.ACQUISITION_REQUEST,
        "terminal_state": {"kind": "SUCCEEDED"},
        "target_results": [{"return_code": 0}, {"return_code": 0}],
    }
    message[field] = value
    _pass_context_case(monkeypatch, terminal_message=message)

    with pytest.raises(window.WindowError, match="does not authorize successful acquisition"):
        window.pass_context(
            tmp_path / ("a" * 64),
            tmp_path / "canonical",
            tmp_path / "terminal.json",
            tmp_path / "authority.pub",
            tmp_path / "observer.pub",
            tmp_path / "publisher.pub",
        )


@pytest.mark.parametrize(
    "aliased",
    (("authority", "observer"), ("authority", "publisher"), ("observer", "publisher")),
)
def test_pass_context_rejects_every_signing_key_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aliased: tuple[str, str],
) -> None:
    key_ids = {
        "authority": "1" * 64,
        "observer": "2" * 64,
        "publisher": "3" * 64,
    }
    key_ids[aliased[1]] = key_ids[aliased[0]]
    _pass_context_case(monkeypatch, key_ids=key_ids)

    with pytest.raises(window.WindowError, match="keys must be distinct"):
        window.pass_context(
            tmp_path / ("a" * 64),
            tmp_path / "canonical",
            tmp_path / "terminal.json",
            tmp_path / "authority.pub",
            tmp_path / "observer.pub",
            tmp_path / "publisher.pub",
        )


def _finalize_decide(args: SimpleNamespace, root: Path, context: dict[str, object]) -> int:
    authority_key_fd = os.open(str(context["_authority_private_key"]), os.O_RDONLY)
    args.authority_key_fd = authority_key_fd
    try:
        return window.decide(args, root, "finalize", context)
    finally:
        args.authority_key_fd = None
        os.close(authority_key_fd)


def _window_harness(tmp_path: Path, state: str) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    authority_pair = _make_keypair(tmp_path, "window-authority")
    publisher_pair = _make_keypair(tmp_path, "window-publisher")
    observer_pair = _make_keypair(tmp_path, "window-observer")
    request_id = "a" * 64
    approval_sha256 = "b" * 64
    generations = tmp_path / ".canonical.generations"
    generations.mkdir()
    candidate = generations / request_id
    predecessor = generations / "predecessor"
    candidate.mkdir()
    predecessor.mkdir()
    (candidate / "market_ohlcv_1s").mkdir()
    (candidate / "market_ohlcv_1s" / "candidate.parquet").write_bytes(b"candidate")
    (predecessor / "market_ohlcv_1s").mkdir()
    (predecessor / "market_ohlcv_1s" / "predecessor.parquet").write_bytes(b"predecessor")
    canonical = tmp_path / "canonical"
    os.symlink(f"{generations.name}/{candidate.name}", canonical)
    swap = tmp_path / ".canonical.bound.swap"
    os.symlink(f"{generations.name}/{predecessor.name}", swap)
    root = tmp_path / ".canonical.transactions" / request_id
    root.mkdir(parents=True)
    authority_key_id = hashlib.sha256(authority_pair[1].read_bytes()).hexdigest()
    observer_key_id = hashlib.sha256(observer_pair[1].read_bytes()).hexdigest()
    context = {
        "request_id": request_id,
        "run_id": window.RUN_ID,
        "acquisition_request_id": window.ACQUISITION_REQUEST,
        "approval_leaf": window.APPROVAL_LEAF,
        "approval_sha256": approval_sha256,
        "authority_key_id": authority_key_id,
        "terminal_receipt_sha256": "c" * 64,
        "observer_key_id": observer_key_id,
        "observer_ready_sha256": "d" * 64,
        "observer_query_spec_sha256": "e" * 64,
        "candidate": str(candidate),
        "candidate_leaf": candidate.name,
        "predecessor": str(predecessor),
        "swap": swap.name,
        "swap_receipt_sha256": "f" * 64,
        "swap_temporary_path": str(swap),
        "candidate_identity": window.identity(candidate),
        "predecessor_identity": window.identity(predecessor),
        "swap_identity": window.identity(swap),
        "pre_exchange_predecessor_identity": window.identity(predecessor),
        "post_exchange_candidate_identity": window.identity(candidate),
        "post_exchange_predecessor_identity": window.identity(predecessor),
        "canonical_logical_root_identity": window.identity(canonical),
        "canonical_resolved_root": str(candidate),
        "candidate_inventory_sha256": window.inventory(candidate),
        "predecessor_inventory_sha256": window.inventory(predecessor),
        "_authority_private_key": str(authority_pair[0]),
        "_authority_public_key": str(authority_pair[1]),
        "_observer_public_key": str(observer_pair[1]),
        "_publisher_public_key": str(publisher_pair[1]),
        "_canonical_root": str(canonical),
    }
    binding = {
        field: context[field] for field in window.ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS
    }
    prefix = window.STATES.index(state) + 1
    for index, name, schema, phase, kind in (
        (0, "activated.json", "alpha_max_publication_activation.v1", "activated", "activation"),
        (
            1,
            "rollback-window-open.json",
            "alpha_max_publication_rollback_window.v1",
            "open",
            "open_window",
        ),
        (2, "replay-verified.json", "alpha_max_publication_replay.v1", "pass", "replay"),
    ):
        if index < prefix:
            _seal_publisher_receipt(
                root,
                publisher_pair,
                name,
                {**binding, "schema": schema, "phase": phase, "kind": kind},
            )

    pass_timestamps: dict[str, int] = {}
    for index in range(3, prefix):
        kind, stage_state = window.PASS_STAGE_KINDS[index]
        evidence_sha256 = "1" * 64
        private_pair = authority_pair
        if kind == "observer":
            private_pair = observer_pair
            observation = {
                "schema": "alpha_max_canonical_exchange_observations.v2",
                "outcome": "PASS",
                "old_before_new": True,
                "observer_ready_sha256": context["observer_ready_sha256"],
            }
            window.write_noreplace(root / "observer-observations-pass.json", observation)
            _, observation_raw = window.read_json(
                root / "observer-observations-pass.json", "observer observations"
            )
            evidence_sha256 = window.sha(observation_raw)
        timestamp_ns = time.time_ns()
        pass_timestamps[kind] = timestamp_ns
        _seal_stage(
            root,
            private_pair,
            window.PASS_NAMES[index],
            kind=kind,
            state=stage_state,
            stage=kind,
            timestamp_ns=timestamp_ns,
            identities=window._context_identities(context),
            inventories=window._context_inventories(context),
            evidence_sha256=evidence_sha256,
        )
    capacity = os.statvfs(window.CAPACITY_PATH)
    decision_guard = _seal_stage(
        root,
        authority_pair,
        "decision-guard.json",
        kind="guard_capacity",
        state=state,
        stage=window.STATE_STAGE[state],
        timestamp_ns=time.time_ns(),
        identities=window._context_identities(context),
        inventories=window._context_inventories(context),
        evidence_sha256="0" * 64,
        capacity_path=window.CAPACITY_PATH,
        free_bytes=capacity.f_bavail * capacity.f_frsize,
        required_bytes=window.HOST_RESERVE_BYTES,
        sample_sequence=1,
    )
    context["_decision_guard_receipt"] = str(decision_guard)
    if state == "W6_READY_TO_FINALIZE":
        capacity = os.statvfs(window.CAPACITY_PATH)
        free_bytes = capacity.f_bavail * capacity.f_frsize
        guard_timestamp_ns = time.time_ns() - 1_000
        guard_paths = []
        for sequence in (10, 11):
            guard_paths.append(
                _seal_stage(
                    root,
                    authority_pair,
                    f"admission-guard-{sequence}.json",
                    kind="guard_capacity",
                    state="W5_BACKTEST_OK",
                    stage="decision_admission",
                    timestamp_ns=guard_timestamp_ns + sequence,
                    identities=window._context_identities(context),
                    inventories=window._context_inventories(context),
                    evidence_sha256=f"{sequence:x}".rjust(64, "0"),
                    capacity_path=window.CAPACITY_PATH,
                    free_bytes=free_bytes,
                    required_bytes=window.HOST_RESERVE_BYTES,
                    sample_sequence=sequence,
                )
            )
        authority_fd = os.open(authority_pair[0], os.O_RDONLY | os.O_NOFOLLOW)
        try:
            window.admit_finalize(
                SimpleNamespace(
                    authority_key_fd=authority_fd,
                    previous_guard_receipt=str(guard_paths[0]),
                    guard_receipt=str(guard_paths[1]),
                ),
                root,
                context,
            )
        finally:
            os.close(authority_fd)

    assert window.state(root, context) == state
    return SimpleNamespace(
        root=root,
        context=context,
        authority=authority_pair,
        observer=observer_pair,
        publisher=publisher_pair,
        canonical=canonical,
        candidate=candidate,
        predecessor=predecessor,
        swap=swap,
        pass_timestamps=pass_timestamps,
    )


def _complete_final_audit_prefix(harness: SimpleNamespace) -> None:
    kind, stage_state = window.PASS_STAGE_KINDS[6]
    _seal_stage(
        harness.root,
        harness.authority,
        window.PASS_NAMES[6],
        kind=kind,
        state=stage_state,
        stage=kind,
        timestamp_ns=time.time_ns(),
        identities=window._context_identities(harness.context),
        inventories=window._context_inventories(harness.context),
        evidence_sha256="8" * 64,
    )


def _finalize_admission_guards(
    harness: SimpleNamespace,
    *,
    current_sequence: int = 11,
    current_timestamp_offset: int = 1,
) -> tuple[Path, Path]:
    capacity = os.statvfs(window.CAPACITY_PATH)
    free_bytes = capacity.f_bavail * capacity.f_frsize
    timestamp_ns = time.time_ns()
    paths = []
    for name, sequence, sample_timestamp in (
        ("test-finalize-previous-guard.json", 10, timestamp_ns),
        (
            "test-finalize-guard.json",
            current_sequence,
            timestamp_ns + current_timestamp_offset,
        ),
    ):
        paths.append(
            _seal_stage(
                harness.root,
                harness.authority,
                name,
                kind="guard_capacity",
                state="W5_BACKTEST_OK",
                stage="decision_admission",
                timestamp_ns=sample_timestamp,
                identities=window._context_identities(harness.context),
                inventories=window._context_inventories(harness.context),
                evidence_sha256=f"{sequence:x}".rjust(64, "0"),
                capacity_path=window.CAPACITY_PATH,
                free_bytes=free_bytes,
                required_bytes=window.HOST_RESERVE_BYTES,
                sample_sequence=sequence,
            )
        )
    return paths[0], paths[1]


def _admit_finalize(
    harness: SimpleNamespace,
    previous: Path,
    current: Path,
    *,
    key_pair: tuple[Path, Path] | None = None,
) -> int:
    authority_fd = os.open((key_pair or harness.authority)[0], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        return window.admit_finalize(
            SimpleNamespace(
                authority_key_fd=authority_fd,
                previous_guard_receipt=str(previous),
                guard_receipt=str(current),
            ),
            harness.root,
            harness.context,
        )
    finally:
        os.close(authority_fd)


def test_final_audit_alone_stays_w5_until_guard_admission_is_sealed(tmp_path: Path) -> None:
    harness = _window_harness(tmp_path, "W5_BACKTEST_OK")
    _complete_final_audit_prefix(harness)
    assert (harness.root / "final-audit-pass.json").is_file()
    assert not (harness.root / "ready-to-finalize.json").exists()
    assert window.state(harness.root, harness.context) == "W5_BACKTEST_OK"

    previous, current = _finalize_admission_guards(harness)
    assert _admit_finalize(harness, previous, current) == 0
    assert window.state(harness.root, harness.context) == "W6_READY_TO_FINALIZE"
    ready = window._verify_ready_to_finalize(harness.root, harness.context)
    assert ready["guard_no_gap"] is True

    _, previous_raw = window.read_json(previous, "previous guard")
    _, current_raw = window.read_json(current, "current guard")
    assert (harness.root / "ready-to-finalize-previous-guard.json").read_bytes() == previous_raw
    assert (harness.root / "ready-to-finalize-guard.json").read_bytes() == current_raw
    artifacts = {
        name: (harness.root / name).read_bytes()
        for name in (
            "ready-to-finalize-previous-guard.json",
            "ready-to-finalize-guard.json",
            "ready-to-finalize.json",
        )
    }
    assert _admit_finalize(harness, previous, current) == 0
    assert {name: (harness.root / name).read_bytes() for name in artifacts} == artifacts


@pytest.mark.parametrize("mode", ("gap", "equal_timestamp", "wrong_key", "tampered_guard"))
def test_finalize_admission_rejects_invalid_inputs_before_ready_mutation(
    tmp_path: Path, mode: str
) -> None:
    harness = _window_harness(tmp_path / mode, "W5_BACKTEST_OK")
    _complete_final_audit_prefix(harness)
    previous, current = _finalize_admission_guards(
        harness,
        current_sequence=12 if mode == "gap" else 11,
        current_timestamp_offset=0 if mode == "equal_timestamp" else 1,
    )
    key_pair = harness.authority
    if mode == "wrong_key":
        key_pair = _make_keypair(tmp_path / mode, "wrong-authority")
    elif mode == "tampered_guard":
        envelope, _ = window.read_json(current, "current guard")
        envelope["message"]["sample_sequence"] += 1
        current.unlink()
        window.write_noreplace(current, envelope)

    with pytest.raises(window.WindowError):
        _admit_finalize(harness, previous, current, key_pair=key_pair)
    assert window.state(harness.root, harness.context) == "W5_BACKTEST_OK"
    assert not any(
        (harness.root / name).exists()
        for name in (
            "ready-to-finalize-previous-guard.json",
            "ready-to-finalize-guard.json",
            "ready-to-finalize.json",
        )
    )


@pytest.mark.parametrize("tamper", ("outer", "message"))
def test_ready_to_finalize_verifier_rejects_outer_and_message_tamper(
    tmp_path: Path, tamper: str
) -> None:
    harness = _window_harness(tmp_path / tamper, "W5_BACKTEST_OK")
    _complete_final_audit_prefix(harness)
    previous, current = _finalize_admission_guards(harness)
    assert _admit_finalize(harness, previous, current) == 0
    path = harness.root / "ready-to-finalize.json"
    envelope, _ = window.read_json(path, "ready-to-finalize")
    if tamper == "outer":
        envelope["authority_key_id"] = "0" * 64
    else:
        envelope["message"]["guard_no_gap"] = False
    path.unlink()
    window.write_noreplace(path, envelope)

    with pytest.raises(window.WindowError):
        window._verify_ready_to_finalize(harness.root, harness.context)


def test_window_harness_publisher_receipts_use_a_distinct_key_and_reject_tampering(
    tmp_path: Path,
) -> None:
    harness = _window_harness(tmp_path, "W2_REPLAY_OK")
    assert harness.context["_publisher_public_key"] == str(harness.publisher[1])
    assert harness.publisher[1].read_bytes() != harness.authority[1].read_bytes()

    for name, kind in (
        ("activated.json", "activation"),
        ("rollback-window-open.json", "open_window"),
        ("replay-verified.json", "replay"),
    ):
        receipt, _ = window.read_json(harness.root / name, "publisher receipt")
        assert set(receipt) == window.ENVELOPE_FIELDS
        assert (
            receipt["authority_key_id"]
            == hashlib.sha256(harness.publisher[1].read_bytes()).hexdigest()
        )
        assert (
            window._verify_publisher_receipt(harness.root / name, harness.publisher[1], kind=kind)
            == receipt["message"]
        )

    receipt, _ = window.read_json(harness.root / "replay-verified.json", "publisher receipt")
    receipt["message"]["phase"] = "tampered"
    (harness.root / "replay-verified.json").unlink()
    window.write_noreplace(harness.root / "replay-verified.json", receipt)
    with pytest.raises(window.WindowError, match="signature"):
        window.state(harness.root, harness.context)


def _seal_action(
    harness: SimpleNamespace,
    kind: str,
    name: str,
    *,
    issued_ns: int | None = None,
    **values: object,
) -> Path:
    issued = time.time_ns() if issued_ns is None else issued_ns
    message: dict[str, object] = {
        "kind": kind,
        "run_id": window.RUN_ID,
        "request_id": harness.context["request_id"],
        "approval_leaf": window.APPROVAL_LEAF,
        "approval_sha256": harness.context["approval_sha256"],
        "state": window._success_state(harness.root, harness.context),
        "issued_ns": issued,
        "expires_ns": issued + 60 * 1_000_000_000,
        "context_sha256": window._context_hash(harness.context),
        "identities": window._context_identities(harness.context),
        "inventories": window._context_inventories(harness.context),
    }
    message.update(values)
    fd = os.open(harness.authority[0], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        envelope = window.seal_action_envelope(fd, message)
    finally:
        os.close(fd)
    path = harness.root / name
    window.write_noreplace(path, envelope)
    return path


def _failure_evidence(
    harness: SimpleNamespace,
    state: str,
    kind: str,
    *,
    timestamp_ns: int,
    name: str,
    reason: str | None = None,
) -> Path:
    pair = harness.observer if kind == "observer" else harness.authority
    extra: dict[str, object] = {}
    if kind == "guard_capacity":
        extra = {
            "capacity_path": window.CAPACITY_PATH,
            "free_bytes": window.HOST_RESERVE_BYTES,
            "required_bytes": window.HOST_RESERVE_BYTES,
            "sample_sequence": 1,
        }
    if reason is not None:
        extra["failure_reason"] = reason
    return _seal_stage(
        harness.root,
        pair,
        name,
        kind=kind,
        state=state,
        stage=window.STATE_STAGE[state],
        outcome="FAIL",
        timestamp_ns=timestamp_ns,
        identities=window._context_identities(harness.context),
        inventories=window._context_inventories(harness.context),
        evidence_sha256="2" * 64,
        **extra,
    )


_EVIDENCE_PREFERENCE = (
    "replay",
    "observer",
    "postverify",
    "deterministic_backtest",
    "final_audit",
    "integrity",
    "unit_exit",
    "guard_capacity",
)
_REASON_CASES = tuple(
    (state, reason) for state, reasons in window.REASONS.items() for reason in sorted(reasons)
)
_DIRECT_REASON_CASES = tuple(
    (state, reason, next(kind for kind in _EVIDENCE_PREFERENCE if kind in kinds))
    for state, reason in _REASON_CASES
    if (kinds := window.REASON_EVIDENCE_KINDS[reason]) != {"inspection"}
)


@pytest.mark.parametrize(("state", "reason", "kind"), _DIRECT_REASON_CASES)
def test_each_direct_failure_reason_rolls_back_only_from_its_exact_state(
    tmp_path: Path,
    state: str,
    reason: str,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now_ns = time.time_ns()
    monkeypatch.setattr(window.time, "time_ns", lambda: fixed_now_ns)
    harness = _window_harness(tmp_path, state)
    issued_ns = time.time_ns() + (2 if state == "W6_READY_TO_FINALIZE" else 0)
    failure = _failure_evidence(
        harness,
        state,
        kind,
        reason=reason,
        timestamp_ns=issued_ns - 1,
        name=f"{reason}-failure.json",
    )
    _, failure_raw = window.read_json(failure, "failure")
    authorization = _seal_action(
        harness,
        "rollback",
        f"{reason}-rollback.json",
        issued_ns=issued_ns,
        reason=reason,
        failure_evidence_kind=kind,
        failure_evidence_sha256=window.sha(failure_raw),
    )

    assert (
        window.decide(
            SimpleNamespace(
                authorization=str(authorization),
                failure_evidence=str(failure),
            ),
            harness.root,
            "rollback",
            harness.context,
        )
        == 0
    )
    assert window.state(harness.root, harness.context) == "W9_ROLLED_BACK"
    assert harness.canonical.resolve(strict=True) == harness.predecessor
    assert harness.candidate.is_dir()


@pytest.mark.parametrize(("owner_state", "reason"), _REASON_CASES)
def test_each_failure_reason_is_rejected_in_every_other_success_state_before_mutation(
    tmp_path: Path,
    owner_state: str,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now_ns = time.time_ns()
    monkeypatch.setattr(window.time, "time_ns", lambda: fixed_now_ns)
    kind = sorted(window.REASON_EVIDENCE_KINDS[reason])[0]
    for other_state in (state for state in window.REASONS if state != owner_state):
        harness = _window_harness(tmp_path / other_state, other_state)
        authorization = _seal_action(
            harness,
            "rollback",
            f"{reason}-wrong-state.json",
            reason=reason,
            failure_evidence_kind=kind,
            failure_evidence_sha256="3" * 64,
        )
        before_target = os.readlink(harness.canonical)
        with pytest.raises(window.WindowError, match="reason is invalid"):
            window.decide(
                SimpleNamespace(
                    authorization=str(authorization),
                    failure_evidence=None,
                ),
                harness.root,
                "rollback",
                harness.context,
            )
        assert os.readlink(harness.canonical) == before_target
        assert not (harness.root / "window-decision-intent.json").exists()


def test_inspection_binds_failed_unit_cgroup_guard_and_authorizes_w1_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _window_harness(tmp_path, "W1_ACTIVATED_OPEN")
    identities = window._context_identities(harness.context)
    inventories = window._context_inventories(harness.context)
    unit = _seal_stage(
        harness.root,
        harness.authority,
        "unit-exit.json",
        kind="unit_exit",
        state="W1_ACTIVATED_OPEN",
        stage="replay",
        outcome="FAIL",
        identities=identities,
        inventories=inventories,
    )
    cgroup = _seal_stage(
        harness.root,
        harness.authority,
        "cgroup.json",
        kind="cgroup",
        state="W1_ACTIVATED_OPEN",
        stage="replay",
        outcome="PASS",
        identities=identities,
        inventories=inventories,
    )
    guard = _seal_stage(
        harness.root,
        harness.authority,
        "guard.json",
        kind="guard_capacity",
        state="W1_ACTIVATED_OPEN",
        identities=identities,
        inventories=inventories,
        capacity_path=window.CAPACITY_PATH,
        free_bytes=window.HOST_RESERVE_BYTES + 2,
        required_bytes=window.HOST_RESERVE_BYTES + 1,
        sample_sequence=1,
    )
    _, unit_raw = window.read_json(unit, "unit")
    _, cgroup_raw = window.read_json(cgroup, "cgroup")
    _, guard_raw = window.read_json(guard, "guard")
    inspection_authorization = _seal_action(
        harness,
        "inspection",
        "inspection.json",
        next_pass="replay-verified.json",
        unit_exit_sha256=window.sha(unit_raw),
        cgroup_sha256=window.sha(cgroup_raw),
        guard_receipt_sha256=window.sha(guard_raw),
    )
    monkeypatch.setattr(
        window.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=window.HOST_RESERVE_BYTES + 2, f_frsize=1),
    )
    assert (
        window.inspect(
            SimpleNamespace(
                for_rollback=True,
                authorization=str(inspection_authorization),
                unit_exit_evidence=str(unit),
                cgroup_evidence=str(cgroup),
                guard_receipt=str(guard),
            ),
            harness.root,
            harness.context,
        )
        == 0
    )
    inspection = harness.root / "recovery-inspection-W1_ACTIVATED_OPEN.json"
    _, inspection_raw = window.read_json(inspection, "inspection")
    rollback_authorization = _seal_action(
        harness,
        "rollback",
        "rollback-from-inspection.json",
        reason="replay_not_started_after_activation",
        failure_evidence_kind="inspection",
        failure_evidence_sha256=window.sha(inspection_raw),
    )
    assert (
        window.decide(
            SimpleNamespace(
                authorization=str(rollback_authorization),
                failure_evidence=str(inspection),
            ),
            harness.root,
            "rollback",
            harness.context,
        )
        == 0
    )
    assert window.state(harness.root, harness.context) == "W9_ROLLED_BACK"


@pytest.mark.parametrize(
    "boundary",
    ("after_intent", "after_exchange", "before_exchange_receipt", "before_terminal"),
)
def test_rollback_crash_replays_exact_action_at_each_durable_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    harness = _window_harness(tmp_path, "W1_ACTIVATED_OPEN")
    issued_ns = time.time_ns()
    monkeypatch.setattr(window.time, "time_ns", lambda: issued_ns)
    failure = _failure_evidence(
        harness,
        "W1_ACTIVATED_OPEN",
        "replay",
        timestamp_ns=issued_ns - 1,
        name="replay-failure.json",
    )
    _, failure_raw = window.read_json(failure, "failure")
    authorization = _seal_action(
        harness,
        "rollback",
        "rollback.json",
        issued_ns=issued_ns,
        reason="replay_failed",
        failure_evidence_kind="replay",
        failure_evidence_sha256=window.sha(failure_raw),
    )
    args = SimpleNamespace(
        authorization=str(authorization),
        failure_evidence=str(failure),
    )
    original_write = window.write_noreplace
    original_exchange = window._rename_exchange

    if boundary == "after_intent":

        def crash_write(path: Path, value: dict[str, object]) -> None:
            if path.name == "W7_ROLLING_BACK.json":
                raise RuntimeError("crash after intent")
            original_write(path, value)

        monkeypatch.setattr(window, "write_noreplace", crash_write)
    elif boundary == "after_exchange":

        def crash_exchange(left: Path, right: Path) -> None:
            original_exchange(left, right)
            raise RuntimeError("crash after exchange")

        monkeypatch.setattr(window, "_rename_exchange", crash_exchange)
    else:
        crash_name = (
            "rollback-exchange-fsynced.json"
            if boundary == "before_exchange_receipt"
            else "rolled-back.json"
        )

        def crash_write(path: Path, value: dict[str, object]) -> None:
            if path.name == crash_name:
                raise RuntimeError(f"crash before {crash_name}")
            original_write(path, value)

        monkeypatch.setattr(window, "write_noreplace", crash_write)

    with pytest.raises(RuntimeError, match="crash"):
        window.decide(args, harness.root, "rollback", harness.context)
    assert window.state(harness.root, harness.context) == "W7_ROLLING_BACK"

    monkeypatch.setattr(window, "write_noreplace", original_write)
    monkeypatch.setattr(window, "_rename_exchange", original_exchange)
    authorization.unlink()
    failure.unlink()
    assert not authorization.exists()
    assert not failure.exists()
    assert window.decide(args, harness.root, "rollback", harness.context) == 0
    assert window.state(harness.root, harness.context) == "W9_ROLLED_BACK"
    assert harness.canonical.resolve(strict=True) == harness.predecessor

    before = sorted(str(path.relative_to(harness.root)) for path in harness.root.rglob("*"))
    assert window.decide(args, harness.root, "rollback", harness.context) == 0
    after = sorted(str(path.relative_to(harness.root)) for path in harness.root.rglob("*"))
    assert after == before


def test_decision_replay_rejects_changed_external_inputs_but_uses_local_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _window_harness(tmp_path, "W1_ACTIVATED_OPEN")
    issued_ns = time.time_ns()
    failure = _failure_evidence(
        harness,
        "W1_ACTIVATED_OPEN",
        "replay",
        timestamp_ns=issued_ns - 1,
        name="replay-failure.json",
    )
    failure_envelope, failure_raw = window.read_json(failure, "failure")
    authorization = _seal_action(
        harness,
        "rollback",
        "rollback.json",
        issued_ns=issued_ns,
        reason="replay_failed",
        failure_evidence_kind="replay",
        failure_evidence_sha256=window.sha(failure_raw),
    )
    authorization_envelope, _ = window.read_json(authorization, "authorization")
    args = SimpleNamespace(
        authorization=str(authorization),
        failure_evidence=str(failure),
    )
    original_write = window.write_noreplace

    def crash_after_intent(path: Path, value: dict[str, object]) -> None:
        if path.name == "W7_ROLLING_BACK.json":
            raise RuntimeError("crash after intent")
        original_write(path, value)

    monkeypatch.setattr(window, "write_noreplace", crash_after_intent)
    with pytest.raises(RuntimeError, match="crash after intent"):
        window.decide(args, harness.root, "rollback", harness.context)
    monkeypatch.setattr(window, "write_noreplace", original_write)
    assert window.state(harness.root, harness.context) == "W7_ROLLING_BACK"

    authorization_envelope["message"]["issued_ns"] += 1
    authorization.unlink()
    window.write_noreplace(authorization, authorization_envelope)
    with pytest.raises(window.WindowError):
        window.decide(args, harness.root, "rollback", harness.context)

    authorization.unlink()
    failure_envelope["message"]["evidence_sha256"] = "9" * 64
    failure.unlink()
    window.write_noreplace(failure, failure_envelope)
    with pytest.raises(window.WindowError, match="changed failure evidence"):
        window.decide(args, harness.root, "rollback", harness.context)

    failure.unlink()
    assert window.decide(args, harness.root, "rollback", harness.context) == 0
    assert window.state(harness.root, harness.context) == "W9_ROLLED_BACK"


@pytest.mark.parametrize(
    "boundary",
    (
        "after_intent",
        "after_quarantine",
        "after_removal",
        "before_cleanup_receipt",
        "before_completion",
        "before_terminal",
        "before_bundle",
    ),
)
def test_finalize_crash_replays_cleanup_to_w10_without_reintroducing_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    fixed_now_ns = time.time_ns()
    monkeypatch.setattr(window.time, "time_ns", lambda: fixed_now_ns)
    harness = _window_harness(tmp_path, "W6_READY_TO_FINALIZE")
    authorization = _seal_action(harness, "finalize", "finalize.json")
    args = SimpleNamespace(authorization=str(authorization), failure_evidence=None)
    original_write = window.write_noreplace
    original_rename = window._rename_noreplace
    original_remove = window._remove_manifest_subset

    if boundary == "after_intent":

        def crash_write(path: Path, value: dict[str, object]) -> None:
            if path.name == "W8_FINALIZING.json":
                raise RuntimeError("crash after intent")
            original_write(path, value)

        monkeypatch.setattr(window, "write_noreplace", crash_write)
    elif boundary == "after_quarantine":

        def crash_rename(source: Path, destination: Path) -> None:
            original_rename(source, destination)
            raise RuntimeError("crash after quarantine")

        monkeypatch.setattr(window, "_rename_noreplace", crash_rename)
    elif boundary == "after_removal":

        def crash_remove(root: Path, manifest: dict[str, object]) -> None:
            original_remove(root, manifest)
            raise RuntimeError("crash after removal")

        monkeypatch.setattr(window, "_remove_manifest_subset", crash_remove)
    else:
        crash_name = {
            "before_cleanup_receipt": "predecessor-cleanup-fsynced.json",
            "before_completion": "completed.json",
            "before_terminal": "finalized.json",
            "before_bundle": "canonical-finalize-bundle.json",
        }[boundary]

        def crash_write(path: Path, value: dict[str, object]) -> None:
            if path.name == crash_name:
                raise RuntimeError(f"crash before {crash_name}")
            original_write(path, value)

        monkeypatch.setattr(window, "write_noreplace", crash_write)

    with pytest.raises(RuntimeError, match="crash"):
        _finalize_decide(args, harness.root, harness.context)
    assert window.state(harness.root, harness.context) == "W8_FINALIZING"

    monkeypatch.setattr(window, "write_noreplace", original_write)
    monkeypatch.setattr(window, "_rename_noreplace", original_rename)
    monkeypatch.setattr(window, "_remove_manifest_subset", original_remove)
    assert _finalize_decide(args, harness.root, harness.context) == 0
    assert window.state(harness.root, harness.context) == "W10_FINALIZED"
    assert harness.canonical.resolve(strict=True) == harness.candidate
    assert not harness.predecessor.exists()
    assert not os.path.lexists(harness.swap)

    before = sorted(str(path.relative_to(harness.root)) for path in harness.root.rglob("*"))
    assert _finalize_decide(args, harness.root, harness.context) == 0
    after = sorted(str(path.relative_to(harness.root)) for path in harness.root.rglob("*"))
    assert after == before

    changed_authorization = _seal_action(
        harness,
        "finalize",
        "changed-finalize.json",
        issued_ns=time.time_ns() + 1,
    )
    with pytest.raises(window.WindowError, match="conflicts with decision intent"):
        _finalize_decide(
            SimpleNamespace(
                authorization=str(changed_authorization),
                failure_evidence=None,
            ),
            harness.root,
            harness.context,
        )
    opposite = _seal_action(
        harness,
        "rollback",
        "opposite-rollback.json",
        reason="late_infrastructure_failure",
        failure_evidence_kind="integrity",
        failure_evidence_sha256="4" * 64,
    )
    with pytest.raises(window.WindowError, match="opposite"):
        window.decide(
            SimpleNamespace(
                authorization=str(opposite),
                failure_evidence=None,
            ),
            harness.root,
            "rollback",
            harness.context,
        )


@pytest.mark.parametrize("drift_target", ("candidate", "predecessor"))
def test_identity_or_inventory_drift_rejects_before_decision_intent(
    tmp_path: Path,
    drift_target: str,
) -> None:
    harness = _window_harness(tmp_path, "W1_ACTIVATED_OPEN")
    issued_ns = time.time_ns()
    failure = _failure_evidence(
        harness,
        "W1_ACTIVATED_OPEN",
        "replay",
        timestamp_ns=issued_ns - 1,
        name="replay-failure.json",
    )
    _, failure_raw = window.read_json(failure, "failure")
    authorization = _seal_action(
        harness,
        "rollback",
        "rollback.json",
        issued_ns=issued_ns,
        reason="replay_failed",
        failure_evidence_kind="replay",
        failure_evidence_sha256=window.sha(failure_raw),
    )
    target = getattr(harness, drift_target)
    (target / "unauthorized.parquet").write_bytes(b"drift")
    before_target = os.readlink(harness.canonical)

    with pytest.raises(window.WindowError, match="changed"):
        window.decide(
            SimpleNamespace(
                authorization=str(authorization),
                failure_evidence=str(failure),
            ),
            harness.root,
            "rollback",
            harness.context,
        )
    assert os.readlink(harness.canonical) == before_target
    assert not (harness.root / "window-decision-intent.json").exists()


def test_wrong_action_context_and_expired_authorization_reject_before_mutation(
    tmp_path: Path,
) -> None:
    harness = _window_harness(tmp_path, "W1_ACTIVATED_OPEN")
    issued_ns = time.time_ns()
    failure = _failure_evidence(
        harness,
        "W1_ACTIVATED_OPEN",
        "replay",
        timestamp_ns=issued_ns - 1,
        name="replay-failure.json",
    )
    _, failure_raw = window.read_json(failure, "failure")
    common = {
        "reason": "replay_failed",
        "failure_evidence_kind": "replay",
        "failure_evidence_sha256": window.sha(failure_raw),
    }
    wrong_context = _seal_action(
        harness,
        "rollback",
        "wrong-context.json",
        issued_ns=issued_ns,
        context_sha256="0" * 64,
        **common,
    )
    with pytest.raises(window.WindowError, match="context"):
        window.decide(
            SimpleNamespace(
                authorization=str(wrong_context),
                failure_evidence=str(failure),
            ),
            harness.root,
            "rollback",
            harness.context,
        )
    expired_issued = issued_ns - 120 * 1_000_000_000
    expired_failure = _failure_evidence(
        harness,
        "W1_ACTIVATED_OPEN",
        "replay",
        timestamp_ns=expired_issued - 1,
        name="expired-replay-failure.json",
    )
    _, expired_raw = window.read_json(expired_failure, "expired failure")
    expired = _seal_action(
        harness,
        "rollback",
        "expired.json",
        issued_ns=expired_issued,
        reason="replay_failed",
        failure_evidence_kind="replay",
        failure_evidence_sha256=window.sha(expired_raw),
    )
    with pytest.raises(window.WindowError, match="not current"):
        window.decide(
            SimpleNamespace(
                authorization=str(expired),
                failure_evidence=str(expired_failure),
            ),
            harness.root,
            "rollback",
            harness.context,
        )
    assert harness.canonical.resolve(strict=True) == harness.candidate
    assert not (harness.root / "window-decision-intent.json").exists()


@pytest.mark.parametrize("boundary", ("write", "file_fsync", "install", "directory_fsync"))
def test_write_noreplace_recovers_from_each_atomic_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
):
    path = tmp_path / "receipt.json"
    value = {"schema": "test", "count": 1}
    names = {
        "write": "_receipt_write",
        "file_fsync": "_receipt_fsync",
        "install": "_receipt_install_noreplace",
        "directory_fsync": "_fsync_dir",
    }
    original = getattr(window, names[boundary])

    def crash(*args, **kwargs):
        raise RuntimeError(boundary)

    monkeypatch.setattr(window, names[boundary], crash)
    with pytest.raises(RuntimeError, match=boundary):
        window.write_noreplace(path, value)
    monkeypatch.setattr(window, names[boundary], original)
    window.write_noreplace(path, value)
    assert path.read_bytes() == window.canonical_bytes(value)
    assert not list(tmp_path.glob(".receipt.json.tmp-*"))


def test_decision_localization_replays_after_guard_authorization_and_failure_are_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    now = time.time_ns()
    monkeypatch.setattr(window.time, "time_ns", lambda: now)
    harness = _window_harness(tmp_path, "W1_ACTIVATED_OPEN")
    failure = _failure_evidence(
        harness,
        "W1_ACTIVATED_OPEN",
        "replay",
        reason="replay_failed",
        timestamp_ns=now - 1,
        name="localized-failure.json",
    )
    _, failure_raw = window.read_json(failure, "failure")
    authorization = _seal_action(
        harness,
        "rollback",
        "localized-authorization.json",
        issued_ns=now,
        reason="replay_failed",
        failure_evidence_kind="replay",
        failure_evidence_sha256=window.sha(failure_raw),
    )
    guard = Path(harness.context["_decision_guard_receipt"])
    args = SimpleNamespace(
        authorization=str(authorization),
        failure_evidence=str(failure),
        guard_receipt=str(guard),
    )
    assert window.decide(args, harness.root, "rollback", harness.context) == 0
    authorization.unlink()
    failure.unlink()
    guard.unlink()
    assert window.decide(args, harness.root, "rollback", harness.context) == 0
