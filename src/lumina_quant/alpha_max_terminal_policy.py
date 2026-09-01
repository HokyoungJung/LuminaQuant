"""Fail-closed schema and argv authority for Alpha-Max terminal scopes."""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import math
import os
import re
import socket
import stat
import resource
import struct
import urllib.parse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUEST_ID = re.compile(r"^[0-9a-f]{64}$")
_POLICY_SCHEMA = "alpha_max_terminal_authority_policy.v3"
_CHECKPOINT_SCHEMA = "alpha_max_terminal_checkpoint.v1"
_ENVELOPE_SCHEMA = "alpha_max_terminal_launch_envelope.v3"
_SCOPES = ("acquisition", "phase_preparation", "one_touch")
_RECEIPT_PREREQUISITE_KINDS = {
    "acquisition": ("checkpoint_pin", "alignment_receipt"),
    "phase_preparation": (
        "checkpoint_pin",
        "alignment_receipt",
        "source_eligible_receipt",
        "source_manifest",
        "source_journal",
        "canonical_finalize_receipt",
    ),
    "one_touch": (
        "checkpoint_pin",
        "alignment_receipt",
        "phase_handoff_receipt",
        "preparation_manifest",
    ),
}
_RESULT_ARTIFACT_KINDS = {
    "acquisition": (
        (("source_eligible_receipt", "source_manifest", "source_journal"), ()),
        (("source_eligible_receipt", "source_manifest", "source_journal"), ()),
    ),
    "phase_preparation": ((("phase_handoff_receipt", "preparation_manifest"), ()),),
    "one_touch": (
        (
            (
                "prelock_readback",
                "prelock_observability",
                "prelock_inventory_before",
                "input_inventory_before",
            ),
            ("prelock_bundle",),
        ),
        (
            (
                "historical_readback",
                "historical_observability",
                "prelock_inventory_after",
                "input_inventory_after",
            ),
            ("historical_bundle",),
        ),
    ),
}


def scope_contract(
    scope: str,
) -> tuple[tuple[str, ...], tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]]:
    """Return the immutable prerequisite and successful-result contract for one scope."""
    if scope not in _SCOPES:
        raise TerminalPolicyError("unknown scope")
    return _RECEIPT_PREREQUISITE_KINDS[scope], _RESULT_ARTIFACT_KINDS[scope]


ALPHA_MAX_PUBLICATION_OBSERVER_READY_SCHEMA = "alpha_max_publication_observer_ready.v1"
ALPHA_MAX_PUBLICATION_OBSERVER_READY_UNSIGNED_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "run_id",
        "request_id",
        "approval_leaf",
        "approval_sha256",
        "canonical_root",
        "query_spec_sha256",
        "observer_key_id",
        "observer_pid",
        "observer_uid",
        "observer_start_ticks",
        "observed_ns",
        "old_identity",
        "old_inventory_sha256",
        "old_loader_sha256",
        "old_observation",
    }
)
ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS = frozenset(
    {
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
)
ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS = ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS | {
    "schema",
    "phase",
}


_ALPHA_MAX_EXCHANGE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ALPHA_MAX_SYMBOL_TOKEN = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")
_ALPHA_MAX_MONTH_TOKEN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_ALPHA_MAX_DATE_TOKEN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_alpha_max_coordination_lock(path: str) -> bool:
    """Return whether a canonical-relative path is an approved mutable writer lock."""
    parts = PurePosixPath(path).parts
    exchange = _ALPHA_MAX_EXCHANGE_TOKEN
    symbol = _ALPHA_MAX_SYMBOL_TOKEN
    month = _ALPHA_MAX_MONTH_TOKEN
    date = _ALPHA_MAX_DATE_TOKEN

    def valid_date(value: str) -> bool:
        if not date.fullmatch(value):
            return False
        try:
            return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
        except ValueError:
            return False

    if parts == (".bootstrap-incomplete",):
        return True
    if (
        len(parts) == 4
        and parts[0] == "market_ohlcv_1s"
        and exchange.fullmatch(parts[1])
        and symbol.fullmatch(parts[2])
        and month.fullmatch(parts[3][:-5])
        and parts[3].endswith(".lock")
    ):
        return True
    if (
        len(parts) == 4
        and parts[0] == "market_data_raw_aggtrades"
        and exchange.fullmatch(parts[1])
        and symbol.fullmatch(parts[2])
        and parts[3] == ".raw-stream.lock"
    ):
        return True
    if (
        len(parts) == 5
        and parts[0] == "market_data_raw_aggtrades"
        and exchange.fullmatch(parts[1])
        and symbol.fullmatch(parts[2])
        and parts[3].startswith("date=")
        and valid_date(parts[3][5:])
        and parts[4] == ".raw-partition.lock"
    ):
        return True
    return (
        len(parts) == 5
        and parts[0] == "feature_points"
        and parts[1].startswith("exchange=")
        and exchange.fullmatch(parts[1][9:])
        and parts[2].startswith("symbol=")
        and symbol.fullmatch(parts[2][7:])
        and parts[3].startswith("date=")
        and valid_date(parts[3][5:])
        and parts[4] == ".writer.lock"
    )


def stable_alpha_max_canonical_inventory(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a full tree inventory to immutable, data-bearing entries."""
    included_files = {
        record["path"]
        for record in records
        if record.get("kind") == "file"
        and record.get("path") != "commit.json"
        and not str(record.get("path", "")).startswith(".alpha_max_publication/")
        and not is_alpha_max_coordination_lock(str(record.get("path", "")))
    }
    included_directories = {
        "/".join(PurePosixPath(path).parts[:index])
        for path in included_files
        for index in range(1, len(PurePosixPath(path).parts))
    }
    return [
        {key: value for key, value in record.items() if key not in {"ctime_ns", "nlink"}}
        for record in records
        if (record.get("kind") == "file" and record.get("path") in included_files)
        or (record.get("kind") == "dir" and record.get("path") in included_directories)
    ]


def _canonical_regular_file_record(
    directory_fd: int, name: str, before: os.stat_result, common: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """Hash and retain a regular file until the full generation snapshot completes."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
    except OSError as exc:
        raise TerminalPolicyError("canonical generation changed during inventory") from exc
    try:
        opened = os.fstat(fd)
        if not _same_stat(before, opened):
            raise TerminalPolicyError("canonical generation changed during inventory")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        try:
            pathname_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise TerminalPolicyError("canonical generation changed during inventory") from exc
        if not _same_stat(before, after) or not _same_stat(before, pathname_after):
            raise TerminalPolicyError("canonical generation changed during inventory")
        return (
            {
                **common,
                "kind": "file",
                "size": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "nlink": before.st_nlink,
                "sha256": digest.hexdigest(),
            },
            fd,
        )
    except BaseException:
        os.close(fd)
        raise


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
        stat.S_IFMT(right.st_mode),
    )


def alpha_max_canonical_inventory_records(root: Path) -> list[dict[str, Any]]:
    """Scan one pinned physical generation without mixing exchanged directories."""
    retained_file_fds: list[tuple[int, os.stat_result, str]] = []
    retained_directory_fds: list[tuple[int, int, str, os.stat_result]] = []
    try:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        descriptor_budget = 131_072 if soft_limit == resource.RLIM_INFINITY else soft_limit - 256
    except (OSError, ValueError) as exc:
        raise TerminalPolicyError("cannot establish canonical inventory descriptor budget") from exc
    if descriptor_budget <= 1:
        raise TerminalPolicyError("canonical inventory descriptor budget is exhausted")

    def reserve_descriptor() -> None:
        if 1 + len(retained_file_fds) + len(retained_directory_fds) >= descriptor_budget:
            raise TerminalPolicyError("canonical inventory descriptor budget is exhausted")

    try:
        named_root = os.lstat(root)
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise TerminalPolicyError("canonical generation is unavailable") from exc
    try:
        opened_root = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(named_root.st_mode)
            or stat.S_ISLNK(named_root.st_mode)
            or not _same_stat(named_root, opened_root)
        ):
            raise TerminalPolicyError("canonical generation is not a physical directory")
        records: list[dict[str, Any]] = []

        def visit(directory_fd: int, relative: str) -> None:
            before = os.fstat(directory_fd)
            try:
                names = sorted(os.listdir(directory_fd))
            except OSError as exc:
                raise TerminalPolicyError("canonical generation cannot be inventoried") from exc
            if not _same_stat(before, os.fstat(directory_fd)):
                raise TerminalPolicyError("canonical generation changed during inventory")
            for entry_name in names:
                name = f"{relative}/{entry_name}" if relative else entry_name
                try:
                    info = os.stat(entry_name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise TerminalPolicyError(
                        "canonical generation changed during inventory"
                    ) from exc
                common = {
                    "path": name,
                    "mode": stat.S_IMODE(info.st_mode),
                    "dev": info.st_dev,
                    "ino": info.st_ino,
                    "ctime_ns": info.st_ctime_ns,
                }
                if stat.S_ISDIR(info.st_mode):
                    reserve_descriptor()
                    try:
                        child_fd = os.open(
                            entry_name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory_fd,
                        )
                    except OSError as exc:
                        raise TerminalPolicyError(
                            "canonical generation changed during inventory"
                        ) from exc
                    try:
                        if not _same_stat(info, os.fstat(child_fd)):
                            raise TerminalPolicyError(
                                "canonical generation changed during inventory"
                            )
                        retained_directory_fds.append((child_fd, directory_fd, entry_name, info))
                        records.append({**common, "kind": "dir"})
                        visit(child_fd, name)
                    except BaseException:
                        if not any(item[0] == child_fd for item in retained_directory_fds):
                            os.close(child_fd)
                        raise
                elif stat.S_ISREG(info.st_mode):
                    reserve_descriptor()
                    record, retained_fd = _canonical_regular_file_record(
                        directory_fd, entry_name, info, common
                    )
                    transferred = False
                    try:
                        included = (
                            name != "commit.json"
                            and not name.startswith(".alpha_max_publication/")
                            and not is_alpha_max_coordination_lock(name)
                        )
                        if included:
                            retained_file_fds.append((retained_fd, info, record["sha256"]))
                            transferred = True
                        else:
                            os.close(retained_fd)
                            transferred = True
                        records.append(record)
                    except BaseException:
                        if not transferred:
                            os.close(retained_fd)
                        raise
                else:
                    raise TerminalPolicyError("canonical generation contains an unsafe artifact")
            if not _same_stat(before, os.fstat(directory_fd)):
                raise TerminalPolicyError("canonical generation changed during inventory")

        visit(root_fd, "")
        try:
            for fd, before, expected_digest in retained_file_fds:
                if not _same_stat(before, os.fstat(fd)):
                    raise TerminalPolicyError("canonical generation changed during inventory")
                os.lseek(fd, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                if digest.hexdigest() != expected_digest or not _same_stat(before, os.fstat(fd)):
                    raise TerminalPolicyError("canonical generation changed during inventory")
            for fd, parent_fd, entry_name, before in reversed(retained_directory_fds):
                if not _same_stat(before, os.fstat(fd)) or not _same_stat(
                    before, os.stat(entry_name, dir_fd=parent_fd, follow_symlinks=False)
                ):
                    raise TerminalPolicyError("canonical generation changed during inventory")
        except OSError as exc:
            raise TerminalPolicyError("canonical generation changed during inventory") from exc
        try:
            pathname_root = os.lstat(root)
        except OSError as exc:
            raise TerminalPolicyError("canonical generation changed during inventory") from exc
        if not _same_stat(opened_root, os.fstat(root_fd)) or not _same_stat(
            opened_root, pathname_root
        ):
            raise TerminalPolicyError("canonical generation changed during inventory")
        return stable_alpha_max_canonical_inventory(records)
    finally:
        for fd, _parent_fd, _entry_name, _before in reversed(retained_directory_fds):
            os.close(fd)
        for fd, _before, _expected_digest in retained_file_fds:
            os.close(fd)
        os.close(root_fd)


def alpha_max_canonical_inventory_sha256(root: Path) -> str:
    """Digest one physical generation using publication-window inventory semantics."""
    return hashlib.sha256(canonical_bytes(alpha_max_canonical_inventory_records(root))).hexdigest()


_FORBIDDEN_ROOTS = (
    "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source",
    "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc",
)
_ENVIRONMENT_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "TZ",
)
_FILE_ROLES = (
    "policy_json",
    "policy_module",
    "authority_script",
    "observer_script",
    "key_creator",
    "acquirer",
    "phase_wrapper",
    "runbook",
    "alpha_uv_lock",
    "alignment_receipt",
    "portfolio",
    "contract_manifest",
    "availability_evidence",
    "preparer",
    "prelock_script",
    "historical_script",
    "process_boundary",
)
_PHASES = ("warmup", "train", "purge", "validation", "embargo", "historical_exposed_evaluation")
_SYMBOLS = (
    "ADAUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "TONUSDT",
    "TRXUSDT",
    "XRPUSDT",
)
_PHASE_INTERVALS = (
    ("warmup", "2022-12-31T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("train", "2024-01-01T00:00:00Z", "2025-06-01T00:00:00Z"),
    ("purge", "2025-06-01T00:00:00Z", "2025-06-08T00:00:00Z"),
    ("validation", "2025-06-08T00:00:00Z", "2025-08-31T00:00:00Z"),
    ("embargo", "2025-08-31T00:00:00Z", "2025-09-07T00:00:00Z"),
    (
        "historical_exposed_evaluation",
        "2025-09-07T00:00:00Z",
        "2026-07-01T00:00:00Z",
    ),
)


class TerminalPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class DirectoryIdentity:
    path: str
    st_dev: int
    st_ino: int
    st_uid: int
    st_gid: int
    mode: int


@dataclass(frozen=True)
class FileIdentity:
    path: str
    sha256: str
    byte_count: int
    st_dev: int
    st_ino: int
    st_uid: int
    st_gid: int
    mode: int
    nlink: int


@dataclass(frozen=True)
class AbsentOutput:
    path: str
    parent: DirectoryIdentity
    leaf: str
    must_be_absent: bool


@dataclass(frozen=True)
class Environment:
    HOME: str
    LANG: str
    LC_ALL: str
    PATH: str
    PYTHONHASHSEED: str
    PYTHONNOUSERSITE: str
    PYTHONDONTWRITEBYTECODE: str
    TZ: str

    def to_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in _ENVIRONMENT_KEYS}


@dataclass(frozen=True)
class PublicationPaths:
    claim: str
    journal: str
    stdout: tuple[str, ...]
    stderr: tuple[str, ...]
    receipt: str


@dataclass(frozen=True)
class PrerequisiteRecord:
    kind: str
    path: str
    sha256: str
    byte_count: int
    st_dev: int
    st_ino: int
    mode: int
    nlink: int


@dataclass(frozen=True)
class RepositoryBinding:
    role: str
    root: DirectoryIdentity
    head: str
    clean_receipt: FileIdentity


@dataclass(frozen=True)
class FileBinding:
    role: str
    file: FileIdentity


@dataclass(frozen=True)
class InterpreterBinding:
    role: str
    file: FileIdentity
    package_freeze: FileIdentity


@dataclass(frozen=True)
class KeyBinding:
    key_id: str
    public_key_b64: str
    public_key_sha256: str


@dataclass(frozen=True)
class ObserverKeyBinding:
    scope: str
    key_id: str
    public_key_b64: str
    public_key_sha256: str


@dataclass(frozen=True)
class TerminalPolicy:
    schema: str
    accepted_alpha_commit: str
    baseline_ancestor: str
    pins: dict[str, str]
    scope_order: tuple[str, ...]
    source_sha256: str


@dataclass(frozen=True)
class CheckpointPin:
    schema: str
    accepted_alpha_commit: str
    baseline_ancestor: str
    runbook_sha256: str
    uv_lock_sha256: str
    alignment_receipt_sha256: str
    portfolio_sha256: str
    contract_sha256: str
    availability_sha256: str
    preparer_sha256: str
    prelock_sha256: str
    historical_sha256: str
    process_boundary_sha256: str
    acquirer_sha256: str
    phase_wrapper_sha256: str
    authority_manifest_sha256: str
    source_path: str = dataclass_field(repr=False, compare=False)
    source_identity: FileIdentity = dataclass_field(repr=False, compare=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


@dataclass(frozen=True)
class LaunchEnvelope:
    schema: str
    policy_sha256: str
    current_head: str
    accepted_alpha_commit: str
    baseline_ancestor: str
    repositories: tuple[RepositoryBinding, ...]
    files: tuple[FileBinding, ...]
    interpreters: tuple[InterpreterBinding, ...]
    authority_key: KeyBinding
    observer_keys: tuple[ObserverKeyBinding, ...]
    forbidden_roots: tuple[str, ...]
    scope_order: tuple[str, ...]

    def file(self, role: str) -> FileIdentity:
        try:
            return next(binding.file for binding in self.files if binding.role == role)
        except StopIteration as exc:
            raise TerminalPolicyError(f"missing envelope file role: {role}") from exc

    def observer_key(self, scope: str) -> ObserverKeyBinding:
        try:
            return next(binding for binding in self.observer_keys if binding.scope == scope)
        except StopIteration as exc:
            raise TerminalPolicyError(f"missing observer key scope: {scope}") from exc

    @property
    def observer_source_sha256(self) -> str:
        return self.file("observer_script").sha256

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


@dataclass(frozen=True)
class AcquisitionRecords:
    acquirer: FileIdentity
    contract_manifest: FileIdentity
    availability_evidence: FileIdentity
    source_root: AbsentOutput
    report_root: AbsentOutput


@dataclass(frozen=True)
class PhaseRecords:
    phase_wrapper: FileIdentity
    acquirer: FileIdentity
    source_root: DirectoryIdentity
    source_report: DirectoryIdentity
    contract_manifest: FileIdentity
    availability_evidence: FileIdentity
    preparer: FileIdentity
    phase_output: AbsentOutput


@dataclass(frozen=True)
class OneTouchRecords:
    portfolio: FileIdentity
    contract_manifest: FileIdentity
    prelock_script: FileIdentity
    historical_script: FileIdentity
    phase_output: DirectoryIdentity
    prelock_output: AbsentOutput
    historical_output: AbsentOutput


ScopeRecords = AcquisitionRecords | PhaseRecords | OneTouchRecords


@dataclass(frozen=True)
class ScopeRequest:
    schema: str
    request_id: str
    scope: str
    checkpoint_pin_sha256: str
    interpreter: FileIdentity
    repository_root: DirectoryIdentity
    evidence_root: DirectoryIdentity
    authority_socket: str
    environment: Environment
    forbidden_roots: tuple[str, ...]
    publication: PublicationPaths
    prerequisites: tuple[PrerequisiteRecord, ...]
    records: ScopeRecords

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


@dataclass(frozen=True)
class ValidatedArtifact:
    kind: str
    path: str
    sha256: str
    byte_count: int
    st_dev: int
    st_ino: int
    mode: int
    nlink: int


@dataclass(frozen=True)
class SealedArtifact:
    kind: str
    path: str
    sha256: str
    byte_count: int
    st_dev: int
    st_ino: int
    mode: int
    nlink: int
    sealed_payload_sha256: str
    canonical_inventory_sha256: str
    readback_sha256: str


@dataclass(frozen=True)
class CommandEvidence:
    """Filesystem evidence required to advance one authenticated child."""

    command_index: int
    state: str
    snapshot_sha256: str
    root_snapshot_sha256s: tuple[str, ...]
    validated_artifacts: tuple[ValidatedArtifact, ...]
    sealed_artifacts: tuple[SealedArtifact, ...]


@dataclass(frozen=True)
class TargetResult:
    command_index: int
    argv_sha256: str
    environment_sha256: str
    return_code: int
    stdout: ValidatedArtifact
    stderr: ValidatedArtifact
    validated_artifacts: tuple[ValidatedArtifact, ...]
    sealed_artifacts: tuple[SealedArtifact, ...]
    completed_utc: str


@dataclass(frozen=True)
class PrelaunchSnapshot:
    files: tuple[FileIdentity, ...]
    directories: tuple[DirectoryIdentity, ...]
    outputs: tuple[AbsentOutput, ...]


@dataclass(frozen=True)
class CommandPreflight:
    command_index: int
    argv: tuple[str, ...]
    argv_sha256: str
    environment_sha256: str


@dataclass(frozen=True)
class VerifiedTerminalReceipt:
    message: dict[str, Any]
    key_id: str
    authorization: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    receipt_sha256: str


def _plain(value: Any) -> Any:
    if isinstance(value, CheckpointPin):
        return _plain(
            {
                name: getattr(value, name)
                for name in CheckpointPin.__dataclass_fields__
                if name not in {"source_path", "source_identity"}
            }
        )
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def plain(value: Any) -> Any:
    """Return the canonical JSON-compatible representation of a typed record."""
    return _plain(value)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def signing_preimage(message_type: str, unsigned_message: Any) -> bytes:
    if not isinstance(message_type, str) or not message_type:
        raise TerminalPolicyError("message type is required")
    return (
        b"luminaquant.alpha_max.terminal.v1/"
        + message_type.encode("utf-8")
        + b"\0"
        + canonical_bytes(unsigned_message)
    )


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise TerminalPolicyError(f"{label} has unexpected fields")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise TerminalPolicyError(f"invalid {label}")
    return value


def _absolute(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "//" in value
        or "/../" in value
        or value.endswith("/..")
        or "/./" in value
        or value.endswith("/.")
    ):
        raise TerminalPolicyError(f"invalid {label}")
    return value


def validate_lexical_control_path(path: Path | str) -> str:
    """Reject a control path that lexically names a quarantined root."""
    value = _absolute(os.fspath(path), "control path")
    if str(Path(value)) != value:
        raise TerminalPolicyError("control path is not normalized")
    if any(value == root or value.startswith(root + "/") for root in _FORBIDDEN_ROOTS):
        raise TerminalPolicyError("control path is under a forbidden root")
    return value


def _under_root(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def validate_sha256(value: Any, label: str) -> str:
    """Validate one canonical lowercase SHA-256 digest."""
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise TerminalPolicyError(f"invalid {label}")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise TerminalPolicyError(f"invalid {label}")
    return value


def _directory(value: Any) -> DirectoryIdentity:
    value = _exact(
        value, {"path", "st_dev", "st_ino", "st_uid", "st_gid", "mode"}, "directory identity"
    )
    return DirectoryIdentity(
        _absolute(value["path"], "directory path"),
        *(
            _integer(value[key], f"directory {key}")
            for key in ("st_dev", "st_ino", "st_uid", "st_gid", "mode")
        ),
    )


def _file(value: Any) -> FileIdentity:
    value = _exact(
        value,
        {"path", "sha256", "byte_count", "st_dev", "st_ino", "st_uid", "st_gid", "mode", "nlink"},
        "file identity",
    )
    return FileIdentity(
        _absolute(value["path"], "file path"),
        validate_sha256(value["sha256"], "file sha256"),
        *(
            _integer(value[key], f"file {key}")
            for key in ("byte_count", "st_dev", "st_ino", "st_uid", "st_gid", "mode", "nlink")
        ),
    )


def _absent(value: Any) -> AbsentOutput:
    value = _exact(value, {"path", "parent", "leaf", "must_be_absent"}, "absent output")
    parent = _directory(value["parent"])
    if (
        not isinstance(value["leaf"], str)
        or not value["leaf"]
        or "/" in value["leaf"]
        or value["must_be_absent"] is not True
        or _absolute(value["path"], "absent output path")
        != f"{parent.path.rstrip('/')}/{value['leaf']}"
        or parent.st_uid != os.getuid()
        or parent.mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise TerminalPolicyError("invalid absent output")
    return AbsentOutput(value["path"], parent, value["leaf"], True)


def _environment(value: Any, *, evidence_root: str) -> Environment:
    value = _exact(value, set(_ENVIRONMENT_KEYS), "environment")
    if any(not isinstance(value[key], str) for key in _ENVIRONMENT_KEYS):
        raise TerminalPolicyError("environment values must be strings")
    if (
        value["HOME"] != evidence_root
        or value["LANG"] != "C.UTF-8"
        or value["LC_ALL"] != "C.UTF-8"
        or value["PATH"] != "/usr/bin:/bin"
        or value["PYTHONHASHSEED"] != "0"
        or value["PYTHONNOUSERSITE"] != "1"
        or value["PYTHONDONTWRITEBYTECODE"] != "1"
        or value["TZ"] != "UTC"
    ):
        raise TerminalPolicyError("environment determinism mismatch")
    return Environment(**value)


def _key(value: Any, *, scope: str | None = None) -> KeyBinding | ObserverKeyBinding:
    fields = {"key_id", "public_key_b64", "public_key_sha256"} | (
        {"scope"} if scope is not None else set()
    )
    value = _exact(value, fields, "key binding")
    if scope is not None and value["scope"] != scope:
        raise TerminalPolicyError("observer key scope mismatch")
    validate_sha256(value["key_id"], "key id")
    validate_sha256(value["public_key_sha256"], "public key sha256")
    try:
        raw = base64.b64decode(value["public_key_b64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise TerminalPolicyError("invalid public key") from exc
    if (
        len(raw) != 32
        or hashlib.sha256(raw).hexdigest() != value["key_id"]
        or value["key_id"] != value["public_key_sha256"]
    ):
        raise TerminalPolicyError("public key identity mismatch")
    if scope is None:
        return KeyBinding(value["key_id"], value["public_key_b64"], value["public_key_sha256"])
    return ObserverKeyBinding(
        scope, value["key_id"], value["public_key_b64"], value["public_key_sha256"]
    )


def _nonoverlap(paths: tuple[str, ...], label: str) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if (
                left == right
                or left.startswith(right.rstrip("/") + "/")
                or right.startswith(left.rstrip("/") + "/")
            ):
                raise TerminalPolicyError(f"overlapping {label}")


def _w10_topology_identity(info: os.stat_result, target: str | None = None) -> list[Any]:
    if (
        type(info.st_dev) is not int
        or type(info.st_ino) is not int
        or info.st_dev < 0
        or info.st_ino < 0
    ):
        raise TerminalPolicyError("invalid W10 canonical identity")
    if stat.S_ISDIR(info.st_mode):
        if target is not None:
            raise TerminalPolicyError("invalid W10 canonical identity")
        return [info.st_dev, info.st_ino, "directory"]
    if not stat.S_ISLNK(info.st_mode) or not isinstance(target, str) or not target:
        raise TerminalPolicyError("invalid W10 canonical identity")
    target_path = PurePosixPath(target)
    if target_path.is_absolute() or any(part in {"", ".", ".."} for part in target_path.parts):
        raise TerminalPolicyError("invalid W10 canonical identity")
    return [info.st_dev, info.st_ino, "symlink", target]


def _w10_canonical_root(bundle_path: Path) -> Path:
    transaction_root = bundle_path.parent
    transactions = transaction_root.parent
    suffix = ".transactions"
    if (
        not _REQUEST_ID.fullmatch(transaction_root.name)
        or not transactions.name.startswith(".")
        or not transactions.name.endswith(suffix)
    ):
        raise TerminalPolicyError("W10 canonical finalize bundle path mismatch")
    canonical_name = transactions.name[1 : -len(suffix)]
    if not canonical_name or "/" in canonical_name or canonical_name in {".", ".."}:
        raise TerminalPolicyError("W10 canonical finalize bundle path mismatch")
    return transactions.parent / canonical_name


def _sample_w10_canonical_identity(bundle_path: Path) -> dict[str, list[Any]]:
    logical_root = _w10_canonical_root(bundle_path)
    try:
        logical_before = os.lstat(logical_root)
        logical_target = os.readlink(logical_root) if stat.S_ISLNK(logical_before.st_mode) else None
        logical_identity = _w10_topology_identity(logical_before, logical_target)
        active_path = logical_root.resolve(strict=True)
        active_before = os.lstat(active_path)
        active_identity = _w10_topology_identity(active_before)
        if active_identity[2] != "directory":
            raise TerminalPolicyError("invalid W10 canonical identity")
        active_after = os.lstat(active_path)
        logical_after = os.lstat(logical_root)
        logical_target_after = (
            os.readlink(logical_root) if stat.S_ISLNK(logical_after.st_mode) else None
        )
    except OSError as exc:
        raise TerminalPolicyError("W10 canonical topology is unavailable") from exc
    if logical_identity != _w10_topology_identity(
        logical_after, logical_target_after
    ) or active_identity != _w10_topology_identity(active_after):
        raise TerminalPolicyError("W10 canonical topology changed")
    return {"logical_root": logical_identity, "active_generation": active_identity}


def _parse_w10_canonical_identity(value: Any) -> dict[str, list[Any]]:
    if not isinstance(value, dict) or set(value) != {"logical_root", "active_generation"}:
        raise TerminalPolicyError("invalid W10 canonical identity")

    def parse(identity: Any, allow_symlink: bool) -> list[Any]:
        if not isinstance(identity, list) or len(identity) not in {3, 4}:
            raise TerminalPolicyError("invalid W10 canonical identity")
        dev, ino, kind = identity[:3]
        if (
            type(dev) is not int
            or type(ino) is not int
            or dev < 0
            or ino < 0
            or kind not in {"directory", "symlink"}
        ):
            raise TerminalPolicyError("invalid W10 canonical identity")
        if kind == "directory":
            if len(identity) != 3:
                raise TerminalPolicyError("invalid W10 canonical identity")
            return [dev, ino, kind]
        if not allow_symlink or len(identity) != 4 or not isinstance(identity[3], str):
            raise TerminalPolicyError("invalid W10 canonical identity")
        target = PurePosixPath(identity[3])
        if (
            not identity[3]
            or target.is_absolute()
            or any(part in {"", ".", ".."} for part in target.parts)
        ):
            raise TerminalPolicyError("invalid W10 canonical identity")
        return [dev, ino, kind, identity[3]]

    logical = parse(value["logical_root"], True)
    active = parse(value["active_generation"], False)
    if active[2] != "directory":
        raise TerminalPolicyError("invalid W10 canonical identity")
    return {"logical_root": logical, "active_generation": active}


def _w10_sibling_snapshot(
    bundle_path: Path, receipts: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], list[tuple[int, Path, os.stat_result]]]:
    siblings: dict[str, dict[str, Any]] = {}
    retained: list[tuple[int, Path, os.stat_result]] = []
    try:
        for leaf, digest in receipts.items():
            if validate_sha256(digest, f"W10 receipt {leaf}") != digest:
                raise TerminalPolicyError("invalid W10 transaction receipt map")
            path = bundle_path.parent / leaf
            try:
                named = os.lstat(path)
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            except OSError as exc:
                raise TerminalPolicyError("W10 transaction receipt is unavailable") from exc
            try:
                opened = os.fstat(fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or not _same_stat(named, opened)
                    or opened.st_size > 64 * 1024 * 1024
                ):
                    raise TerminalPolicyError("invalid W10 transaction receipt")
                raw = bytearray()
                while len(raw) < opened.st_size:
                    chunk = os.read(fd, min(1 << 20, opened.st_size - len(raw)))
                    if not chunk:
                        raise TerminalPolicyError("W10 transaction receipt was truncated")
                    raw.extend(chunk)
                after = os.fstat(fd)
                if not _same_stat(opened, after):
                    raise TerminalPolicyError("W10 transaction receipt changed")
                receipt = parse_canonical_object(bytes(raw), f"W10 receipt {leaf}")
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise TerminalPolicyError("W10 transaction receipt changed")
                siblings[leaf] = receipt
                retained.append((fd, path, opened))
            except BaseException:
                os.close(fd)
                raise
        return siblings, retained
    except BaseException:
        for fd, _path, _info in retained:
            os.close(fd)
        raise


def _recheck_w10_sibling_snapshot(retained: list[tuple[int, Path, os.stat_result]]) -> None:
    try:
        for fd, path, before in retained:
            if (
                not _same_stat(before, os.fstat(fd))
                or not _same_stat(before, os.lstat(path))
                or os.fstat(fd).st_nlink != 1
            ):
                raise TerminalPolicyError("W10 transaction receipt changed")
    except OSError as exc:
        raise TerminalPolicyError("W10 transaction receipt changed") from exc


_W10_FINALIZE_BUNDLE_LEAF = "canonical-finalize-bundle.json"
_W10_FINALIZE_BUNDLE_DOMAIN = b"luminaquant.alpha_max.w10_canonical_finalize_bundle.v2\0"
_W10_FINALIZE_RECEIPTS = (
    "decision-localization.json",
    "window-decision-intent.json",
    "W8_FINALIZING.json",
    "predecessor-cleanup-manifest.json",
    "predecessor-quarantined.json",
    "predecessor-cleanup-fsynced.json",
    "completed.json",
    "finalized.json",
)


def validate_w10_canonical_finalize_bundle(
    path: Path | str,
    *,
    authority_public_key_b64: str,
    run_id: str | None = None,
    acquisition_request_id: str | None = None,
    approval_sha256: str | None = None,
    canonical_identity: dict[str, Any] | None = None,
    finalize_authorization_sha256: str | None = None,
    finalize_context_sha256: str | None = None,
    transaction_receipt_sha256s: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Authenticate the fixed, authority-signed W10 transaction-root bundle."""
    bundle_path = Path(validate_lexical_control_path(path))
    if bundle_path.name != _W10_FINALIZE_BUNDLE_LEAF:
        raise TerminalPolicyError("W10 canonical finalize bundle leaf mismatch")
    value = _canonical_object(bundle_path)
    _exact(
        value,
        {"schema", "authority_key_id", "message", "signature"},
        "W10 canonical finalize bundle",
    )
    try:
        authority_raw = base64.b64decode(authority_public_key_b64, validate=True)
        signature = base64.b64decode(value["signature"], validate=True)
    except (TypeError, ValueError) as exc:
        raise TerminalPolicyError("invalid W10 canonical finalize bundle signature") from exc
    if (
        len(authority_raw) != 32
        or value["authority_key_id"] != hashlib.sha256(authority_raw).hexdigest()
    ):
        raise TerminalPolicyError("W10 canonical finalize bundle authority mismatch")
    message = _exact(
        value["message"],
        {
            "run_id",
            "acquisition_request_id",
            "transaction_root",
            "approval_sha256",
            "canonical_identity",
            "state",
            "finalize_authorization_sha256",
            "finalize_context_sha256",
            "transaction_receipt_sha256s",
        },
        "W10 canonical finalize message",
    )
    unsigned = {field: value[field] for field in ("schema", "authority_key_id", "message")}
    try:
        Ed25519PublicKey.from_public_bytes(authority_raw).verify(
            signature, _W10_FINALIZE_BUNDLE_DOMAIN + canonical_bytes(unsigned)
        )
    except InvalidSignature as exc:
        raise TerminalPolicyError("W10 canonical finalize bundle signature is invalid") from exc
    if (
        value["schema"] != "alpha_max_canonical_finalize_bundle.v2"
        or message["state"] != "W10_FINALIZED"
    ):
        raise TerminalPolicyError("W10 canonical finalize bundle binding mismatch")
    if message["transaction_root"] != str(bundle_path.parent):
        raise TerminalPolicyError("W10 canonical finalize bundle path mismatch")
    for field in (
        "run_id",
        "acquisition_request_id",
        "approval_sha256",
        "finalize_authorization_sha256",
        "finalize_context_sha256",
    ):
        validate_sha256(message[field], f"W10 {field}")
    signed_canonical_identity = _parse_w10_canonical_identity(message["canonical_identity"])
    live_canonical_identity = _sample_w10_canonical_identity(bundle_path)
    if signed_canonical_identity != live_canonical_identity:
        raise TerminalPolicyError("W10 canonical topology changed")
    if canonical_identity is not None and canonical_identity != live_canonical_identity:
        raise TerminalPolicyError("W10 canonical finalize bundle binding mismatch")
    receipts = message["transaction_receipt_sha256s"]
    if not isinstance(receipts, dict) or set(receipts) != set(_W10_FINALIZE_RECEIPTS):
        raise TerminalPolicyError("invalid W10 transaction receipt map")
    siblings, retained_siblings = _w10_sibling_snapshot(bundle_path, receipts)
    intent = siblings["window-decision-intent.json"]
    journal = siblings["W8_FINALIZING.json"]
    completed = siblings["completed.json"]
    finalized = siblings["finalized.json"]
    localization = siblings["decision-localization.json"]
    manifest = siblings["predecessor-cleanup-manifest.json"]
    quarantined = siblings["predecessor-quarantined.json"]
    cleanup = siblings["predecessor-cleanup-fsynced.json"]
    if (
        intent.get("action") != "finalize"
        or intent.get("authorization_sha256") != message["finalize_authorization_sha256"]
        or intent.get("context_sha256") != message["finalize_context_sha256"]
        or localization.get("action") != "finalize"
        or localization.get("authorization_sha256") != message["finalize_authorization_sha256"]
        or manifest.get("authorization_sha256") != message["finalize_authorization_sha256"]
        or quarantined.get("authorization_sha256") != message["finalize_authorization_sha256"]
        or cleanup
        != {
            "schema": "alpha_max_predecessor_cleanup.v1",
            "authorization_sha256": message["finalize_authorization_sha256"],
            "phase": "removal-fsynced",
        }
        or journal
        != {
            "schema": "alpha_max_window_action_journal.v1",
            "action": "finalize",
            "authorization_sha256": message["finalize_authorization_sha256"],
            "context_sha256": message["finalize_context_sha256"],
            "phase": "intent-fsynced",
        }
        or completed.get("schema") != "alpha_max_publication_completed.v1"
        or completed.get("authorization_sha256") != message["finalize_authorization_sha256"]
        or completed.get("context_sha256") != message["finalize_context_sha256"]
        or finalized
        != {
            "schema": "alpha_max_window_terminal.v2",
            "action": "finalize",
            "state": "W10_FINALIZED",
            "authorization_sha256": message["finalize_authorization_sha256"],
            "context_sha256": message["finalize_context_sha256"],
        }
    ):
        for fd, _path, _before in retained_siblings:
            os.close(fd)
        raise TerminalPolicyError("W10 transaction receipt cross-link mismatch")
    expected = {
        "run_id": run_id,
        "acquisition_request_id": acquisition_request_id,
        "approval_sha256": approval_sha256,
        "canonical_identity": canonical_identity,
        "finalize_authorization_sha256": finalize_authorization_sha256,
        "finalize_context_sha256": finalize_context_sha256,
        "transaction_receipt_sha256s": transaction_receipt_sha256s,
    }
    if any(
        expected_value is not None and message[field] != expected_value
        for field, expected_value in expected.items()
    ):
        for fd, _path, _before in retained_siblings:
            os.close(fd)
        raise TerminalPolicyError("W10 canonical finalize bundle binding mismatch")
    try:
        _recheck_w10_sibling_snapshot(retained_siblings)
    finally:
        for fd, _path, _before in retained_siblings:
            os.close(fd)
    return message


def load_policy(path: Path | str) -> TerminalPolicy:
    value = _canonical_object(Path(validate_lexical_control_path(path)))
    _exact(
        value,
        {"schema", "accepted_alpha_commit", "baseline_ancestor", "pins", "scope_order"},
        "policy",
    )
    required = {
        "runbook_sha256",
        "uv_lock_sha256",
        "alignment_receipt_sha256",
        "portfolio_sha256",
        "contract_sha256",
        "availability_sha256",
        "preparer_sha256",
        "prelock_sha256",
        "historical_sha256",
        "process_boundary_sha256",
        "acquirer_sha256",
        "phase_wrapper_sha256",
    }
    if (
        value["schema"] != _POLICY_SCHEMA
        or not isinstance(value["pins"], dict)
        or not isinstance(value["scope_order"], list)
        or tuple(value["scope_order"]) != _SCOPES
    ):
        raise TerminalPolicyError("invalid policy")
    _exact(value["pins"], required, "policy pins")
    return TerminalPolicy(
        value["schema"],
        _commit(value["accepted_alpha_commit"], "accepted commit"),
        _commit(value["baseline_ancestor"], "baseline ancestor"),
        {key: validate_sha256(item, key) for key, item in value["pins"].items()},
        _SCOPES,
        hashlib.sha256(canonical_bytes(value)).hexdigest(),
    )


def load_checkpoint(path: Path | str, policy: TerminalPolicy) -> CheckpointPin:
    source_path = validate_lexical_control_path(path)
    if str(Path(source_path)) != source_path:
        raise TerminalPolicyError("checkpoint path is not normalized")
    info, source_sha256, source_byte_count, payload = _regular_file(
        source_path, "checkpoint", capture=True
    )
    if payload is None:
        raise AssertionError("captured checkpoint has no payload")
    value = parse_canonical_object(payload, source_path)
    _exact(
        value,
        set(CheckpointPin.__dataclass_fields__) - {"source_path", "source_identity"},
        "checkpoint",
    )
    source_identity = FileIdentity(
        source_path,
        source_sha256,
        source_byte_count,
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
    )
    pin = CheckpointPin(**value, source_path=source_path, source_identity=source_identity)
    if (
        pin.schema != _CHECKPOINT_SCHEMA
        or pin.accepted_alpha_commit != policy.accepted_alpha_commit
        or pin.baseline_ancestor != policy.baseline_ancestor
    ):
        raise TerminalPolicyError("checkpoint policy mismatch")
    for name in policy.pins:
        if validate_sha256(getattr(pin, name), name) != policy.pins[name]:
            raise TerminalPolicyError(f"checkpoint {name} mismatch")
    validate_sha256(pin.authority_manifest_sha256, "authority manifest sha256")
    return pin


def load_envelope(
    path: Path | str, policy: TerminalPolicy, checkpoint: CheckpointPin
) -> LaunchEnvelope:
    value = _canonical_object(Path(validate_lexical_control_path(path)))
    _exact(value, set(LaunchEnvelope.__dataclass_fields__) - {"sha256"}, "envelope")
    if (
        value["schema"] != _ENVELOPE_SCHEMA
        or value["accepted_alpha_commit"] != policy.accepted_alpha_commit
        or value["baseline_ancestor"] != policy.baseline_ancestor
        or tuple(value["scope_order"]) != _SCOPES
        or validate_sha256(value["policy_sha256"], "policy sha256") != policy.source_sha256
    ):
        raise TerminalPolicyError("envelope policy mismatch")
    if (
        validate_sha256(checkpoint.authority_manifest_sha256, "authority manifest sha256")
        != hashlib.sha256(canonical_bytes(value)).hexdigest()
    ):
        raise TerminalPolicyError("checkpoint authority manifest mismatch")
    current_head = _commit(value["current_head"], "current head")
    repositories = tuple(
        RepositoryBinding(
            item["role"],
            _directory(item["root"]),
            _commit(item["head"], "repository head"),
            _file(item["clean_receipt"]),
        )
        for item in value["repositories"]
        if isinstance(item, dict)
    )
    if (
        not isinstance(value["repositories"], list)
        or len(repositories) != len(value["repositories"])
        or tuple(item.role for item in repositories)
        != ("current_repository", "accepted_alpha_repository")
        or any(
            set(item) != {"role", "root", "head", "clean_receipt"} for item in value["repositories"]
        )
    ):
        raise TerminalPolicyError("invalid repositories")
    _nonoverlap(tuple(item.root.path for item in repositories), "repository roots")
    if (
        repositories[0].head != current_head
        or current_head == policy.accepted_alpha_commit
        or repositories[1].head != policy.accepted_alpha_commit
    ):
        raise TerminalPolicyError("repository head binding mismatch")
    files = tuple(
        FileBinding(item["role"], _file(item["file"]))
        for item in value["files"]
        if isinstance(item, dict)
    )
    if (
        not isinstance(value["files"], list)
        or tuple(item.role for item in files) != _FILE_ROLES
        or any(set(item) != {"role", "file"} for item in value["files"])
    ):
        raise TerminalPolicyError("invalid envelope file roles")
    pin_roles = {
        "runbook": "runbook_sha256",
        "alpha_uv_lock": "uv_lock_sha256",
        "alignment_receipt": "alignment_receipt_sha256",
        "portfolio": "portfolio_sha256",
        "contract_manifest": "contract_sha256",
        "availability_evidence": "availability_sha256",
        "preparer": "preparer_sha256",
        "prelock_script": "prelock_sha256",
        "historical_script": "historical_sha256",
        "process_boundary": "process_boundary_sha256",
        "acquirer": "acquirer_sha256",
        "phase_wrapper": "phase_wrapper_sha256",
    }
    if any(
        item.file.sha256 != policy.pins[pin_roles[item.role]]
        for item in files
        if item.role in pin_roles
    ):
        raise TerminalPolicyError("envelope file pin mismatch")
    interpreters = tuple(
        InterpreterBinding(item["role"], _file(item["file"]), _file(item["package_freeze"]))
        for item in value["interpreters"]
        if isinstance(item, dict)
    )
    if (
        not isinstance(value["interpreters"], list)
        or tuple(item.role for item in interpreters) != ("current_python", "accepted_alpha_python")
        or any(set(item) != {"role", "file", "package_freeze"} for item in value["interpreters"])
    ):
        raise TerminalPolicyError("invalid interpreters")
    authority_key = _key(value["authority_key"])
    if not isinstance(value["observer_keys"], list) or len(value["observer_keys"]) != len(_SCOPES):
        raise TerminalPolicyError("invalid observer keys")
    observers = tuple(
        _key(item, scope=scope) for item, scope in zip(value["observer_keys"], _SCOPES)
    )
    if any(not isinstance(item, ObserverKeyBinding) for item in observers):
        raise TerminalPolicyError("invalid observer keys")
    if len({authority_key.key_id, *(item.key_id for item in observers)}) != len(_SCOPES) + 1:
        raise TerminalPolicyError("duplicate authority or observer key id")
    forbidden = (
        tuple(_absolute(item, "forbidden root") for item in value["forbidden_roots"])
        if isinstance(value["forbidden_roots"], list)
        else ()
    )
    if forbidden != _FORBIDDEN_ROOTS:
        raise TerminalPolicyError("invalid forbidden roots")
    protected_paths = (
        tuple(item.root.path for item in repositories)
        + tuple(item.clean_receipt.path for item in repositories)
        + tuple(item.file.path for item in files)
        + tuple(item.file.path for item in interpreters)
        + tuple(item.package_freeze.path for item in interpreters)
    )
    if len(set(protected_paths)) != len(protected_paths) or any(
        _under_root(path, root) for path in protected_paths for root in forbidden
    ):
        raise TerminalPolicyError("duplicate or forbidden envelope path")
    return LaunchEnvelope(
        value["schema"],
        value["policy_sha256"],
        current_head,
        value["accepted_alpha_commit"],
        value["baseline_ancestor"],
        repositories,
        files,
        interpreters,
        authority_key,
        observers,
        forbidden,
        _SCOPES,
    )


def _parse_publication(value: Any, commands: int) -> PublicationPaths:
    value = _exact(value, {"claim", "journal", "stdout", "stderr", "receipt"}, "publication")
    leaves = {
        "claim": "prelaunch.claim.json",
        "journal": "terminal-observer.journal.jsonl",
        "receipt": "terminal-authority.receipt.json",
    }
    if (
        any(value.get(key) != leaf for key, leaf in leaves.items())
        or not isinstance(value["stdout"], list)
        or not isinstance(value["stderr"], list)
        or len(value["stdout"]) != commands
        or len(value["stderr"]) != commands
    ):
        raise TerminalPolicyError("invalid publication")
    names = tuple(value["stdout"]) + tuple(value["stderr"])
    if len(set(names)) != len(names) or any(
        not isinstance(name, str)
        or "/" in name
        or not re.fullmatch(r"child-[0-9]+\.(stdout|stderr)\.log", name)
        for name in names
    ):
        raise TerminalPolicyError("invalid publication child logs")
    for index, name in enumerate(value["stdout"]):
        if name != f"child-{index}.stdout.log":
            raise TerminalPolicyError("stdout publication order mismatch")
    for index, name in enumerate(value["stderr"]):
        if name != f"child-{index}.stderr.log":
            raise TerminalPolicyError("stderr publication order mismatch")
    return PublicationPaths(
        leaves["claim"],
        leaves["journal"],
        tuple(value["stdout"]),
        tuple(value["stderr"]),
        leaves["receipt"],
    )


def load_request(
    path: Path | str,
    *,
    scope: str,
    policy: TerminalPolicy,
    checkpoint: CheckpointPin,
    envelope: LaunchEnvelope,
) -> ScopeRequest:
    if scope not in _SCOPES:
        raise TerminalPolicyError("unknown scope")
    value = _canonical_object(Path(validate_lexical_control_path(path)))
    common = {
        "schema",
        "request_id",
        "scope",
        "checkpoint_pin_sha256",
        "interpreter",
        "repository_root",
        "evidence_root",
        "authority_socket",
        "environment",
        "forbidden_roots",
        "publication",
        "prerequisites",
    }
    record_types: dict[str, tuple[set[str], type[ScopeRecords]]] = {
        "acquisition": (
            {
                "acquirer",
                "contract_manifest",
                "availability_evidence",
                "source_root",
                "report_root",
            },
            AcquisitionRecords,
        ),
        "phase_preparation": (
            {
                "phase_wrapper",
                "acquirer",
                "source_root",
                "source_report",
                "contract_manifest",
                "availability_evidence",
                "preparer",
                "phase_output",
            },
            PhaseRecords,
        ),
        "one_touch": (
            {
                "portfolio",
                "contract_manifest",
                "prelock_script",
                "historical_script",
                "phase_output",
                "prelock_output",
                "historical_output",
            },
            OneTouchRecords,
        ),
    }
    extras, record_type = record_types[scope]
    _exact(value, common | extras, "request")
    if (
        value["schema"] != f"alpha_max_terminal_request.{scope}.v1"
        or value["scope"] != scope
        or not isinstance(value["request_id"], str)
        or not _REQUEST_ID.fullmatch(value["request_id"])
        or validate_sha256(value["checkpoint_pin_sha256"], "request checkpoint")
        != checkpoint.sha256
    ):
        raise TerminalPolicyError("request mismatch")
    interpreter = _file(value["interpreter"])
    repository_root, evidence_root = (
        _directory(value["repository_root"]),
        _directory(value["evidence_root"]),
    )
    role_index = 1 if scope == "one_touch" else 0
    if (
        interpreter != envelope.interpreters[role_index].file
        or repository_root != envelope.repositories[role_index].root
    ):
        raise TerminalPolicyError("request execution role is not envelope-bound")
    if (
        evidence_root.st_uid != os.getuid()
        or evidence_root.st_gid != os.getgid()
        or evidence_root.mode != 0o700
    ):
        raise TerminalPolicyError("evidence root must be private and leader-owned")
    forbidden = (
        tuple(_absolute(item, "request forbidden root") for item in value["forbidden_roots"])
        if isinstance(value["forbidden_roots"], list)
        else ()
    )
    if forbidden != envelope.forbidden_roots:
        raise TerminalPolicyError("request forbidden roots mismatch")
    _nonoverlap((repository_root.path, evidence_root.path, *forbidden), "request roots")
    authority_socket = _absolute(value["authority_socket"], "authority socket")
    if authority_socket != str(Path(evidence_root.path) / "terminal-authority.sock"):
        raise TerminalPolicyError("authority socket path mismatch")
    commands = 2 if scope == "acquisition" else 1 if scope == "phase_preparation" else 2
    publication = _parse_publication(value["publication"], commands)
    publication_paths = tuple(
        Path(evidence_root.path) / item
        for item in (publication.claim, publication.journal, publication.receipt)
    )
    if len(set(publication_paths)) != len(publication_paths) or any(
        path.parent != Path(evidence_root.path) for path in publication_paths
    ):
        raise TerminalPolicyError("publication path is outside evidence root")
    prereq_kinds, _result_contract = scope_contract(scope)
    if not isinstance(value["prerequisites"], list) or len(value["prerequisites"]) != len(
        prereq_kinds
    ):
        raise TerminalPolicyError("invalid prerequisites")
    prereqs = tuple(
        PrerequisiteRecord(
            **_exact(
                item,
                {"kind", "path", "sha256", "byte_count", "st_dev", "st_ino", "mode", "nlink"},
                "prerequisite",
            )
        )
        for item in value["prerequisites"]
    )
    if tuple(item.kind for item in prereqs) != prereq_kinds:
        raise TerminalPolicyError("prerequisite kind or order mismatch")
    for item in prereqs:
        _absolute(item.path, "prerequisite path")
        validate_sha256(item.sha256, "prerequisite sha256")
        for field in (item.byte_count, item.st_dev, item.st_ino, item.mode, item.nlink):
            _integer(field, "prerequisite identity")
    if any(
        item.path == root or item.path.startswith(root.rstrip("/") + "/")
        for item in prereqs
        for root in forbidden
    ):
        raise TerminalPolicyError("forbidden prerequisite path")
    parsers: dict[str, Any] = dict.fromkeys(extras, _file)
    if scope == "acquisition":
        parsers.update(source_root=_absent, report_root=_absent)
    elif scope == "phase_preparation":
        parsers.update(source_root=_directory, source_report=_directory, phase_output=_absent)
    else:
        parsers.update(phase_output=_directory, prelock_output=_absent, historical_output=_absent)
    records = record_type(**{name: parsers[name](value[name]) for name in extras})
    record_paths = tuple(getattr(records, name).path for name in extras)
    if len(set(record_paths)) != len(record_paths) or any(
        path.startswith(root.rstrip("/") + "/") or path == root
        for path in record_paths
        for root in forbidden
    ):
        raise TerminalPolicyError("duplicate or forbidden record path")
    root_paths = tuple(
        getattr(records, name).path
        for name in (
            "source_root",
            "report_root",
            "source_report",
            "phase_output",
            "prelock_output",
            "historical_output",
        )
        if hasattr(records, name)
    )
    _nonoverlap(
        (repository_root.path, evidence_root.path, *root_paths, *forbidden),
        "request authority roots",
    )
    expected_prerequisite_paths: dict[str, str] = {
        "checkpoint_pin": checkpoint.source_path,
        "alignment_receipt": envelope.file("alignment_receipt").path,
    }
    expected_prerequisite_hashes = {
        "checkpoint_pin": checkpoint.source_identity.sha256,
        "alignment_receipt": policy.pins["alignment_receipt_sha256"],
    }
    if isinstance(records, PhaseRecords):
        source_report = Path(records.source_report.path)
        expected_prerequisite_paths.update(
            source_eligible_receipt=str(source_report / "source_eligible_receipt.json"),
            source_manifest=str(source_report / "source_manifest.json"),
            source_journal=str(source_report / "acquisition.journal.jsonl"),
        )
    elif isinstance(records, OneTouchRecords):
        phase_output = Path(records.phase_output.path)
        expected_prerequisite_paths.update(
            phase_handoff_receipt=str(
                phase_output.parent
                / f".{phase_output.name}.alpha_max_phase_preparation.handoff.json"
            ),
            preparation_manifest=str(phase_output / "preparation_manifest.json"),
        )
    for item in prereqs:
        expected_path = expected_prerequisite_paths.get(item.kind)
        expected_hash = expected_prerequisite_hashes.get(item.kind)
        if (expected_path is not None and item.path != expected_path) or (
            expected_hash is not None and item.sha256 != expected_hash
        ):
            raise TerminalPolicyError(f"prerequisite {item.kind} binding mismatch")
    if scope == "phase_preparation":
        finalize_receipt = next(
            item for item in prereqs if item.kind == "canonical_finalize_receipt"
        )
        validate_w10_canonical_finalize_bundle(
            finalize_receipt.path,
            authority_public_key_b64=envelope.authority_key.public_key_b64,
        )
    checkpoint_record = next(item for item in prereqs if item.kind == "checkpoint_pin")
    checkpoint_identity = checkpoint.source_identity
    if (
        checkpoint_record.path,
        checkpoint_record.sha256,
        checkpoint_record.byte_count,
        checkpoint_record.st_dev,
        checkpoint_record.st_ino,
        checkpoint_record.mode,
        checkpoint_record.nlink,
    ) != (
        checkpoint_identity.path,
        checkpoint_identity.sha256,
        checkpoint_identity.byte_count,
        checkpoint_identity.st_dev,
        checkpoint_identity.st_ino,
        checkpoint_identity.mode,
        checkpoint_identity.nlink,
    ):
        raise TerminalPolicyError("checkpoint prerequisite identity mismatch")
    alignment = envelope.file("alignment_receipt")
    alignment_record = next(item for item in prereqs if item.kind == "alignment_receipt")
    if (
        alignment_record.path,
        alignment_record.sha256,
        alignment_record.byte_count,
        alignment_record.st_dev,
        alignment_record.st_ino,
        alignment_record.mode,
        alignment_record.nlink,
    ) != (
        alignment.path,
        alignment.sha256,
        alignment.byte_count,
        alignment.st_dev,
        alignment.st_ino,
        alignment.mode,
        alignment.nlink,
    ):
        raise TerminalPolicyError("alignment prerequisite identity mismatch")
    return ScopeRequest(
        value["schema"],
        value["request_id"],
        scope,
        value["checkpoint_pin_sha256"],
        interpreter,
        repository_root,
        evidence_root,
        authority_socket,
        _environment(value["environment"], evidence_root=evidence_root.path),
        forbidden,
        publication,
        prereqs,
        records,
    )


def derive_scope_commands(
    envelope: LaunchEnvelope, request: ScopeRequest
) -> tuple[tuple[str, ...], ...]:
    if request.scope not in envelope.scope_order:
        raise TerminalPolicyError("scope not authorized")
    record = request.records
    python = request.interpreter.path
    forbidden = request.forbidden_roots
    if request.scope == "acquisition" and isinstance(record, AcquisitionRecords):
        common = (
            python,
            record.acquirer.path,
            "--contract-manifest",
            record.contract_manifest.path,
            "--availability-evidence",
            record.availability_evidence.path,
            "--output-root",
            record.source_root.path,
            "--report-dir",
            record.report_root.path,
            "--forbidden-root",
            forbidden[0],
            "--forbidden-root",
            forbidden[1],
        )
        return (
            (*common, "--execute", "--validate-complete"),
            (*common, "--verify-eligible"),
        )
    if request.scope == "phase_preparation" and isinstance(record, PhaseRecords):
        return (
            (
                python,
                record.phase_wrapper.path,
                "--acquirer",
                record.acquirer.path,
                "--source-root",
                record.source_root.path,
                "--source-report",
                record.source_report.path,
                "--forbidden-root",
                forbidden[0],
                "--forbidden-root",
                forbidden[1],
                "--contract-manifest",
                record.contract_manifest.path,
                "--availability-evidence",
                record.availability_evidence.path,
                "--preparer",
                record.preparer.path,
                "--output-root",
                record.phase_output.path,
            ),
        )
    if request.scope == "one_touch" and isinstance(record, OneTouchRecords):
        roots: list[str] = []
        for phase in _PHASES[:-1]:
            roots.extend(
                (
                    f"--{phase}-raw-root",
                    f"{record.phase_output.path}/{phase}/raw",
                    f"--{phase}-feature-root",
                    f"{record.phase_output.path}/{phase}/feature",
                )
            )
        prelock = (
            python,
            record.prelock_script.path,
            "--config",
            record.portfolio.path,
            "--contract-manifest",
            record.contract_manifest.path,
            "--exchange",
            "binance",
            "--output-root",
            record.prelock_output.path,
            *roots,
        )
        historical = (
            python,
            record.historical_script.path,
            "--sealed-prelock-directory",
            record.prelock_output.path,
            "--embargo-feature-root",
            f"{record.phase_output.path}/embargo/feature",
            "--historical-evaluation-raw-root",
            f"{record.phase_output.path}/historical_exposed_evaluation/raw",
            "--historical-evaluation-feature-root",
            f"{record.phase_output.path}/historical_exposed_evaluation/feature",
            "--exchange",
            "binance",
            "--output-root",
            record.historical_output.path,
        )
        return (prelock, historical)
    raise TerminalPolicyError("scope records do not match scope")


def _same_file(left: FileIdentity, right: FileIdentity) -> bool:
    return left == right


def _open_absolute(path: Path | str, flags: int, label: str) -> int:
    """Open an absolute path without resolving a symlinked component."""
    try:
        nofollow = os.O_NOFOLLOW
        cloexec = os.O_CLOEXEC
    except AttributeError as exc:
        raise TerminalPolicyError("required secure open flags unavailable") from exc
    if type(nofollow) is not int or nofollow <= 0 or type(cloexec) is not int or cloexec <= 0:
        raise TerminalPolicyError("required secure open flags unavailable")

    def open_fd(name: str, open_flags: int, *, dir_fd: int | None = None) -> int:
        if dir_fd is None:
            descriptor = os.open(name, open_flags)
        else:
            descriptor = os.open(name, open_flags, dir_fd=dir_fd)
        try:
            os.set_inheritable(descriptor, False)
        except OSError:
            os.close(descriptor)
            raise
        return descriptor

    target = validate_lexical_control_path(path)
    parts = tuple(part for part in target.split("/") if part)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow | cloexec
    try:
        descriptor = open_fd("/", directory_flags)
    except OSError as exc:
        raise TerminalPolicyError(f"cannot open {label}") from exc
    try:
        for part in parts[:-1]:
            child = open_fd(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if not parts:
            return descriptor
        child = open_fd(parts[-1], flags | nofollow | cloexec, dir_fd=descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise TerminalPolicyError(f"cannot open {label}") from exc
    os.close(descriptor)
    return child


def open_directory_fd(path: Path | str, label: str) -> int:
    """Open an owned directory descriptor without following any path component."""
    return _open_absolute(path, os.O_RDONLY | os.O_DIRECTORY, label)


def _regular_file(
    path: Path | str,
    label: str,
    *,
    capture: bool = False,
    max_capture_bytes: int = 64 * 1024 * 1024,
) -> tuple[os.stat_result, str, int, bytes | None]:
    flags = os.O_RDONLY
    descriptor = _open_absolute(path, flags, label)
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TerminalPolicyError(f"{label} is not a private regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if capture and byte_count > max_capture_bytes:
                raise TerminalPolicyError(f"{label} exceeds the control-file limit")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_uid",
        "st_gid",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or byte_count != after.st_size
    ):
        raise TerminalPolicyError(f"{label} changed while it was read")
    return after, digest.hexdigest(), byte_count, b"".join(chunks) if chunks is not None else None


def _relative_parts(relative: str, label: str) -> tuple[str, ...]:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise TerminalPolicyError(f"{label} relative path is invalid")
    return tuple(relative.split("/"))


def _open_child_fd(parent_fd: int, name: str, flags: int, label: str) -> int:
    try:
        descriptor = os.open(
            name,
            flags | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise TerminalPolicyError(f"cannot open {label}") from exc
    try:
        os.set_inheritable(descriptor, False)
    except OSError as exc:
        os.close(descriptor)
        raise TerminalPolicyError(f"cannot open {label}") from exc
    return descriptor


def _open_relative_directory(root_fd: int, relative: str, label: str) -> int:
    descriptor = os.dup(root_fd)
    os.set_inheritable(descriptor, False)
    try:
        for part in _relative_parts(relative, label):
            child = _open_child_fd(descriptor, part, os.O_RDONLY | os.O_DIRECTORY, label)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise TerminalPolicyError(f"{label} is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _regular_file_at(
    root_fd: int,
    relative: str,
    label: str,
    *,
    capture: bool = False,
    max_capture_bytes: int = 64 * 1024 * 1024,
) -> tuple[os.stat_result, str, int, bytes | None]:
    parts = _relative_parts(relative, label)
    parent_fd = os.dup(root_fd)
    os.set_inheritable(parent_fd, False)
    try:
        for part in parts[:-1]:
            child = _open_child_fd(parent_fd, part, os.O_RDONLY | os.O_DIRECTORY, label)
            os.close(parent_fd)
            parent_fd = child
        descriptor = _open_child_fd(parent_fd, parts[-1], os.O_RDONLY, label)
    finally:
        os.close(parent_fd)
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TerminalPolicyError(f"{label} is not a private regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if capture and byte_count > max_capture_bytes:
                raise TerminalPolicyError(f"{label} exceeds the control-file limit")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_uid",
        "st_gid",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or byte_count != after.st_size
    ):
        raise TerminalPolicyError(f"{label} changed while it was read")
    return after, digest.hexdigest(), byte_count, b"".join(chunks) if chunks is not None else None


def _canonical_object_at(root_fd: int, relative: str, label: str) -> Any:
    _info, _digest_value, _size, payload = _regular_file_at(root_fd, relative, label, capture=True)
    try:
        value = json.loads(payload)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalPolicyError(f"{label} is not canonical JSON") from exc
    if canonical_bytes(value) != payload:
        raise TerminalPolicyError(f"{label} is not canonical JSON")
    return value


def _validate_directory_fd(
    fd: int, identity: DirectoryIdentity | None, label: str
) -> os.stat_result:
    """Validate a borrowed directory descriptor without taking ownership."""
    try:
        info = os.fstat(fd)
    except OSError as exc:
        raise TerminalPolicyError(f"cannot stat {label}") from exc
    actual = (info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
    if not stat.S_ISDIR(info.st_mode) or (
        identity is not None
        and actual
        != (identity.st_dev, identity.st_ino, identity.st_uid, identity.st_gid, identity.mode)
    ):
        raise TerminalPolicyError(f"{label} identity drift")
    return info


def _open_registered_directory(descriptors: ExitStack, path: str | Path, label: str) -> int:
    descriptor = open_directory_fd(path, label)
    descriptors.callback(os.close, descriptor)
    return descriptor


def _open_registered_child(
    descriptors: ExitStack, parent_fd: int, leaf: str, flags: int, label: str
) -> int:
    descriptor = _open_child_fd(parent_fd, leaf, flags, label)
    descriptors.callback(os.close, descriptor)
    return descriptor


def _snapshot_identity(value: Any, label: str) -> dict[str, int]:
    value = _exact(value, {"st_dev", "st_ino"}, label)
    if any(type(item) is not int or item < 0 for item in value.values()):
        raise TerminalPolicyError(f"{label} mismatch")
    return value


def _open_output_child(descriptors: ExitStack, output: AbsentOutput, label: str) -> tuple[int, int]:
    output_path = Path(validate_lexical_control_path(output.path))
    if output_path.parent != Path(output.parent.path) or output_path.name != output.leaf:
        raise TerminalPolicyError("output parent binding mismatch")
    parent_fd = _open_registered_directory(descriptors, output.parent.path, f"{label} parent")
    _validate_directory_fd(parent_fd, output.parent, f"{label} parent")
    output_fd = _open_registered_child(
        descriptors, parent_fd, output.leaf, os.O_RDONLY | os.O_DIRECTORY, label
    )
    _validate_directory_fd(output_fd, None, label)
    return parent_fd, output_fd


def _walk_tree_at(root_fd: int, label: str) -> list[tuple[str, os.stat_result]]:
    root_info = _validate_directory_fd(root_fd, None, f"{label} root")
    rows: list[tuple[str, os.stat_result]] = [(".", root_info)]

    def walk(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd)):
            if name in {".", ".."}:
                raise TerminalPolicyError(f"{label} tree contains invalid entry")
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise TerminalPolicyError(f"{label} tree entry is unavailable") from exc
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISLNK(info.st_mode):
                raise TerminalPolicyError(f"{label} tree contains a symlink")
            rows.append((relative, info))
            if stat.S_ISDIR(info.st_mode):
                child_fd = _open_child_fd(directory_fd, name, os.O_RDONLY | os.O_DIRECTORY, label)
                try:
                    opened = _validate_directory_fd(child_fd, None, f"{label} tree child")
                    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                        raise TerminalPolicyError(f"{label} tree identity changed")
                    walk(child_fd, relative)
                    after = _validate_directory_fd(child_fd, None, f"{label} tree child")
                    if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
                        raise TerminalPolicyError(f"{label} tree identity changed")
                finally:
                    os.close(child_fd)

    walk(root_fd, "")
    after = _validate_directory_fd(root_fd, None, f"{label} root")
    if (after.st_dev, after.st_ino) != (root_info.st_dev, root_info.st_ino):
        raise TerminalPolicyError(f"{label} root identity changed")
    return rows


def _walk_tree(root: Path, label: str) -> list[tuple[str, os.stat_result]]:
    root_fd = open_directory_fd(root, label)
    try:
        return _walk_tree_at(root_fd, label)
    finally:
        os.close(root_fd)


def _read_identity(identity: FileIdentity, label: str) -> None:
    info, digest, byte_count, _payload = _regular_file(identity.path, label)
    actual = (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
        byte_count,
        digest,
    )
    expected = (
        identity.st_dev,
        identity.st_ino,
        identity.st_uid,
        identity.st_gid,
        identity.mode,
        identity.nlink,
        identity.byte_count,
        identity.sha256,
    )
    if identity.nlink != 1 or actual != expected:
        raise TerminalPolicyError(f"{label} identity drift")


def _read_directory(identity: DirectoryIdentity, label: str) -> None:
    descriptor = open_directory_fd(identity.path, label)
    try:
        info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    actual = (info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
    expected = (identity.st_dev, identity.st_ino, identity.st_uid, identity.st_gid, identity.mode)
    if not stat.S_ISDIR(info.st_mode) or actual != expected:
        raise TerminalPolicyError(f"{label} identity drift")


def _verify_absent(output: AbsentOutput) -> None:
    if (
        Path(output.path).parent != Path(output.parent.path)
        or Path(output.path).name != output.leaf
    ):
        raise TerminalPolicyError("output parent binding mismatch")
    parent_fd = open_directory_fd(output.parent.path, "output parent")
    try:
        _validate_directory_fd(parent_fd, output.parent, f"output parent {output.path}")
        try:
            os.stat(output.leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise TerminalPolicyError("cannot inspect output leaf") from exc
    finally:
        os.close(parent_fd)
    raise TerminalPolicyError("output is not absent")


def validate_prelaunch(
    envelope: LaunchEnvelope,
    request: ScopeRequest,
    *,
    require_outputs_absent: bool = True,
) -> PrelaunchSnapshot:
    """Read-only authority and filesystem validation before launching a target."""
    bound_files = {item.role: item.file for item in envelope.files}
    bound_files.update({item.role: item.file for item in envelope.interpreters})
    expected_roles = {
        "acquisition": {
            "acquirer": "acquirer",
            "contract_manifest": "contract_manifest",
            "availability_evidence": "availability_evidence",
        },
        "phase_preparation": {
            "phase_wrapper": "phase_wrapper",
            "acquirer": "acquirer",
            "contract_manifest": "contract_manifest",
            "availability_evidence": "availability_evidence",
            "preparer": "preparer",
        },
        "one_touch": {
            "portfolio": "portfolio",
            "contract_manifest": "contract_manifest",
            "prelock_script": "prelock_script",
            "historical_script": "historical_script",
        },
    }[request.scope]
    requested_files: list[FileIdentity] = [request.interpreter]
    if request.interpreter not in bound_files.values():
        raise TerminalPolicyError("request interpreter identity is not envelope-bound")
    for field, role in expected_roles.items():
        item = getattr(request.records, field)
        if not _same_file(item, bound_files[role]):
            raise TerminalPolicyError(f"request {field} is not envelope-bound")
        requested_files.append(item)
    directories = [request.repository_root, request.evidence_root]
    for repository in envelope.repositories:
        _read_directory(repository.root, f"repository {repository.role}")
        _read_identity(repository.clean_receipt, f"repository receipt {repository.role}")
    if request.repository_root not in tuple(item.root for item in envelope.repositories):
        raise TerminalPolicyError("request repository root identity is not envelope-bound")
    for item in directories:
        _read_directory(item, "request directory")
    for binding in envelope.files:
        _read_identity(binding.file, f"envelope file {binding.role}")
    for binding in envelope.interpreters:
        _read_identity(binding.file, f"envelope interpreter {binding.role}")
        _read_identity(binding.package_freeze, f"package freeze {binding.role}")
    for item in requested_files:
        _read_identity(item, "request file")
    for item in request.prerequisites:
        info, digest, byte_count, _payload = _regular_file(
            item.path,
            f"prerequisite {item.kind}",
        )
        if (
            (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode), info.st_nlink)
            != (item.st_dev, item.st_ino, item.mode, item.nlink)
            or byte_count != item.byte_count
            or digest != item.sha256
        ):
            raise TerminalPolicyError(f"prerequisite {item.kind} identity drift")
    if request.scope == "phase_preparation":
        finalize_receipt = next(
            item for item in request.prerequisites if item.kind == "canonical_finalize_receipt"
        )
        validate_w10_canonical_finalize_bundle(
            finalize_receipt.path,
            authority_public_key_b64=envelope.authority_key.public_key_b64,
        )
    record = request.records
    outputs: list[AbsentOutput] = []
    for field in (
        "source_root",
        "report_root",
        "phase_output",
        "prelock_output",
        "historical_output",
    ):
        item = getattr(record, field, None)
        if isinstance(item, AbsentOutput):
            if require_outputs_absent:
                _verify_absent(item)
            else:
                _read_directory(item.parent, f"output parent {item.path}")
            outputs.append(item)
    for field in ("source_root", "source_report", "phase_output"):
        item = getattr(record, field, None)
        if isinstance(item, DirectoryIdentity):
            _read_directory(item, field)
            directories.append(item)
    if request.scope == "one_touch":
        _validate_preparation_manifest(request)
    all_paths = tuple(item.path for item in requested_files) + tuple(item.path for item in outputs)
    if len(set(all_paths)) != len(all_paths):
        raise TerminalPolicyError("overlapping prelaunch paths")
    return PrelaunchSnapshot(tuple(requested_files), tuple(directories), tuple(outputs))


def command_bundle_sha256(
    request: ScopeRequest,
    commands: tuple[tuple[str, ...], ...],
) -> str:
    """Hash the exact scoped command order and immutable launch environment."""
    return hashlib.sha256(
        canonical_bytes(
            {
                "schema": "alpha_max_terminal_command_bundle.v1",
                "scope": request.scope,
                "commands": commands,
                "environment": request.environment,
            }
        )
    ).hexdigest()


def derive_command_preflight(
    envelope: LaunchEnvelope, request: ScopeRequest
) -> tuple[CommandPreflight, ...]:
    environment_sha256 = hashlib.sha256(canonical_bytes(request.environment)).hexdigest()
    return tuple(
        CommandPreflight(
            index, argv, hashlib.sha256(canonical_bytes(argv)).hexdigest(), environment_sha256
        )
        for index, argv in enumerate(derive_scope_commands(envelope, request))
    )


def validate_command_semantics(
    envelope: LaunchEnvelope,
    request: ScopeRequest,
    command_index: int,
    argv: tuple[str, ...],
    environment: Environment,
) -> CommandPreflight:
    """Validate one child command without launching it."""
    preflight = derive_command_preflight(envelope, request)
    if type(command_index) is not int or not 0 <= command_index < len(preflight):
        raise TerminalPolicyError("invalid command index")
    expected = preflight[command_index]
    if argv != expected.argv or environment != request.environment:
        raise TerminalPolicyError("command semantics mismatch")
    return expected


def _parse_validated(value: Any) -> ValidatedArtifact:
    value = _exact(
        value,
        {"kind", "path", "sha256", "byte_count", "st_dev", "st_ino", "mode", "nlink"},
        "validated artifact",
    )
    if not isinstance(value["kind"], str) or not value["kind"]:
        raise TerminalPolicyError("invalid artifact kind")
    artifact = ValidatedArtifact(
        value["kind"],
        _absolute(value["path"], "artifact path"),
        validate_sha256(value["sha256"], "artifact sha256"),
        *(
            _integer(value[key], f"artifact {key}")
            for key in ("byte_count", "st_dev", "st_ino", "mode", "nlink")
        ),
    )
    if artifact.st_ino <= 0 or artifact.mode > 0o777 or artifact.nlink != 1:
        raise TerminalPolicyError("invalid artifact identity")
    return artifact


def _parse_sealed(value: Any) -> SealedArtifact:
    base = _parse_validated(
        {
            key: value[key]
            for key in ("kind", "path", "sha256", "byte_count", "st_dev", "st_ino", "mode", "nlink")
        }
        if isinstance(value, dict)
        else value
    )
    _exact(
        value,
        set(ValidatedArtifact.__dataclass_fields__)
        | {"sealed_payload_sha256", "canonical_inventory_sha256", "readback_sha256"},
        "sealed artifact",
    )
    return SealedArtifact(
        *asdict(base).values(),
        validate_sha256(value["sealed_payload_sha256"], "sealed payload"),
        validate_sha256(value["canonical_inventory_sha256"], "inventory"),
        validate_sha256(value["readback_sha256"], "readback"),
    )


def _revalidate_artifact(artifact: ValidatedArtifact, label: str) -> None:
    info, digest, byte_count, _payload = _regular_file(artifact.path, label)
    actual = (
        digest,
        byte_count,
        info.st_dev,
        info.st_ino,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
    )
    expected = (
        artifact.sha256,
        artifact.byte_count,
        artifact.st_dev,
        artifact.st_ino,
        artifact.mode,
        artifact.nlink,
    )
    if actual != expected or info.st_uid != os.getuid() or info.st_gid != os.getgid():
        raise TerminalPolicyError(f"{label} identity drift")


def _revalidate_artifact_at(
    root_fd: int, root_path: str, artifact: ValidatedArtifact, label: str
) -> None:
    if not _under_root(artifact.path, root_path) or artifact.path == root_path:
        raise TerminalPolicyError(f"{label} is outside its retained root")
    relative = artifact.path.removeprefix(root_path + "/")
    info, digest, byte_count, _payload = _regular_file_at(root_fd, relative, label)
    actual = (
        digest,
        byte_count,
        info.st_dev,
        info.st_ino,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
    )
    expected = (
        artifact.sha256,
        artifact.byte_count,
        artifact.st_dev,
        artifact.st_ino,
        artifact.mode,
        artifact.nlink,
    )
    if actual != expected or info.st_uid != os.getuid() or info.st_gid != os.getgid():
        raise TerminalPolicyError(f"{label} identity drift")


def _regular_file_at_enumerated(
    root_fd: int,
    relative: str,
    label: str,
    enumerated: dict[str, os.stat_result],
    *,
    capture: bool = False,
) -> tuple[os.stat_result, str, int, bytes | None]:
    known = enumerated.get(relative)
    if known is None or not stat.S_ISREG(known.st_mode) or known.st_nlink != 1:
        raise TerminalPolicyError(f"{label} was not safely enumerated")
    info, digest, byte_count, payload = _regular_file_at(root_fd, relative, label, capture=capture)
    if (info.st_dev, info.st_ino) != (known.st_dev, known.st_ino):
        raise TerminalPolicyError(f"{label} tree identity changed")
    return info, digest, byte_count, payload


def _canonical_object_at_enumerated(
    root_fd: int, relative: str, label: str, enumerated: dict[str, os.stat_result]
) -> Any:
    _info, _digest_value, _size, payload = _regular_file_at_enumerated(
        root_fd, relative, label, enumerated, capture=True
    )
    try:
        value = json.loads(payload)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalPolicyError(f"{label} is not canonical JSON") from exc
    if canonical_bytes(value) != payload:
        raise TerminalPolicyError(f"{label} is not canonical JSON")
    return value


def _artifact_from_at_enumerated(
    root_fd: int,
    root_path: str,
    relative: str,
    kind: str,
    enumerated: dict[str, os.stat_result],
) -> ValidatedArtifact:
    info, digest, byte_count, _payload = _regular_file_at_enumerated(
        root_fd, relative, f"{kind} artifact", enumerated
    )
    if info.st_uid != os.getuid() or info.st_gid != os.getgid():
        raise TerminalPolicyError(f"{kind} artifact owner mismatch")
    return ValidatedArtifact(
        kind,
        str(Path(root_path) / relative),
        digest,
        byte_count,
        info.st_dev,
        info.st_ino,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
    )


def verify_signed_receipt(path: Path | str, public_key_path: Path | str) -> VerifiedTerminalReceipt:
    receipt_value = validate_lexical_control_path(path)
    receipt_path = Path(receipt_value)
    if receipt_path.parent == receipt_path:
        raise TerminalPolicyError("terminal receipt path must have a parent")
    _relative_parts(receipt_path.name, "terminal receipt leaf")
    evidence_root_path = str(receipt_path.parent)
    evidence_root_fd = open_directory_fd(evidence_root_path, "terminal evidence root")
    try:
        return _verify_signed_receipt_at(receipt_path, evidence_root_fd, public_key_path)
    finally:
        os.close(evidence_root_fd)


def _verify_signed_receipt_at(
    receipt_path: Path, evidence_root_fd: int, public_key_path: Path | str
) -> VerifiedTerminalReceipt:
    evidence_root_path = str(receipt_path.parent)
    receipt = _canonical_object_at(evidence_root_fd, receipt_path.name, "terminal receipt")
    _exact(receipt, set(TERMINAL_RECEIPT_FIELDS), "terminal receipt")
    if receipt["schema"] != WIRE_SCHEMA or receipt["type"] != "terminal_receipt":
        raise TerminalPolicyError("invalid terminal receipt")

    key_info, _key_digest, key_size, raw = _regular_file(
        public_key_path,
        "authority public key",
        capture=True,
        max_capture_bytes=32,
    )
    if (
        raw is None
        or key_size != 32
        or key_info.st_uid != os.getuid()
        or stat.S_IMODE(key_info.st_mode) not in {0o400, 0o444}
    ):
        raise TerminalPolicyError("unsafe authority public key")
    key_id = hashlib.sha256(raw).hexdigest()
    if receipt["authority_key_id"] != key_id:
        raise TerminalPolicyError("terminal receipt authority key mismatch")
    authority = Ed25519PublicKey.from_public_bytes(raw)
    verify_message("terminal_receipt", receipt, authority)

    for name in (
        "authority_key_id",
        "authorization_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "claim_sha256",
        "observer_key_id",
        "command_bundle_sha256",
        "events_sha256",
        "journal_sha256",
    ):
        validate_sha256(receipt[name], name)
    if (
        receipt["scope"] not in _SCOPES
        or not isinstance(receipt["request_id"], str)
        or not _REQUEST_ID.fullmatch(receipt["request_id"])
        or type(receipt["observer_pid"]) is not int
        or receipt["observer_pid"] <= 0
        or type(receipt["observer_start_ticks"]) is not int
        or receipt["observer_start_ticks"] <= 0
    ):
        raise TerminalPolicyError("terminal receipt identity mismatch")
    parse_utc_second(receipt["created_utc"], "receipt timestamp")
    command_count = 1 if receipt["scope"] == "phase_preparation" else 2
    publication = _parse_publication(receipt["publication"], command_count)
    if receipt_path.name != publication.receipt:
        raise TerminalPolicyError("terminal receipt publication mismatch")
    _terminal_state(receipt["terminal_state"])

    prerequisite_kinds = _RECEIPT_PREREQUISITE_KINDS[receipt["scope"]]
    if (
        not isinstance(receipt["prerequisites"], list)
        or tuple(
            item.get("kind") if isinstance(item, dict) else None
            for item in receipt["prerequisites"]
        )
        != prerequisite_kinds
        or not isinstance(receipt["target_results"], list)
    ):
        raise TerminalPolicyError("invalid terminal receipt collections")
    prerequisite_fields = {
        "kind",
        "path",
        "sha256",
        "byte_count",
        "st_dev",
        "st_ino",
        "mode",
        "nlink",
    }
    for item in receipt["prerequisites"]:
        _exact(item, prerequisite_fields, "terminal prerequisite")
        artifact = _parse_validated(item)
        if _under_root(artifact.path, evidence_root_path):
            _revalidate_artifact_at(
                evidence_root_fd,
                evidence_root_path,
                artifact,
                f"terminal prerequisite {artifact.kind}",
            )
        else:
            _revalidate_artifact(artifact, f"terminal prerequisite {artifact.kind}")
    results = tuple(_parse_target_result(item) for item in receipt["target_results"])
    if tuple(item.command_index for item in results) != tuple(range(len(results))):
        raise TerminalPolicyError("terminal result order mismatch")

    terminal = receipt["terminal_state"]
    kind = terminal["kind"]
    if kind == "SUCCEEDED":
        if len(results) != command_count or any(item.return_code != 0 for item in results):
            raise TerminalPolicyError("successful receipt has incomplete results")
    elif kind == "FAILED":
        failed = terminal["failed_command_index"]
        if (
            failed >= command_count
            or len(results) != failed + 1
            or results[-1].return_code == 0
            or any(item.return_code != 0 for item in results[:-1])
        ):
            raise TerminalPolicyError("failed receipt result mismatch")
    elif kind in {"START_FAILED", "START_UNKNOWN", "OBSERVER_LOST"}:
        index = terminal["command_index"]
        if (
            index >= command_count
            or len(results) != index
            or any(item.return_code != 0 for item in results)
        ):
            raise TerminalPolicyError("incomplete receipt result mismatch")
    elif any(item.return_code != 0 for item in results):
        raise TerminalPolicyError("unauthenticated receipt result mismatch")

    expected_kinds = _RESULT_ARTIFACT_KINDS
    for result in results:
        if (
            Path(result.stdout.path)
            != receipt_path.parent / publication.stdout[result.command_index]
            or Path(result.stderr.path)
            != receipt_path.parent / publication.stderr[result.command_index]
        ):
            raise TerminalPolicyError("terminal result log path mismatch")
        for artifact in (
            result.stdout,
            result.stderr,
            *result.validated_artifacts,
            *result.sealed_artifacts,
        ):
            if _under_root(artifact.path, evidence_root_path):
                _revalidate_artifact_at(
                    evidence_root_fd,
                    evidence_root_path,
                    artifact,
                    f"terminal result {artifact.kind}",
                )
            else:
                _revalidate_artifact(artifact, f"terminal result {artifact.kind}")
        if result.stdout.mode != 0o600 or result.stderr.mode != 0o600:
            raise TerminalPolicyError("terminal result log mode mismatch")
        if result.return_code == 0:
            validated_kinds, sealed_kinds = expected_kinds[receipt["scope"]][result.command_index]
            if (
                tuple(item.kind for item in result.validated_artifacts) != validated_kinds
                or tuple(item.kind for item in result.sealed_artifacts) != sealed_kinds
            ):
                raise TerminalPolicyError("terminal result artifact contract mismatch")
        elif result.validated_artifacts or result.sealed_artifacts:
            raise TerminalPolicyError("failed result claims validated artifacts")

    claim_info, claim_digest, _claim_size, claim_bytes = _regular_file_at(
        evidence_root_fd,
        publication.claim,
        "terminal claim",
        capture=True,
    )
    if (
        claim_bytes is None
        or claim_digest != receipt["claim_sha256"]
        or claim_info.st_uid != os.getuid()
        or claim_info.st_gid != os.getgid()
        or stat.S_IMODE(claim_info.st_mode) != 0o600
    ):
        raise TerminalPolicyError("terminal claim identity mismatch")
    claim = parse_canonical_object(claim_bytes, "terminal claim")
    _exact(claim, set(CLAIM_FIELDS), "terminal claim")
    evidence_root = _directory(claim["evidence_root"])
    if (
        claim["schema"] != CLAIM_SCHEMA
        or claim["request_id"] != receipt["request_id"]
        or claim["scope"] != receipt["scope"]
        or claim["checkpoint_pin_sha256"] != receipt["checkpoint_pin_sha256"]
        or claim["observer_pid"] != receipt["observer_pid"]
        or claim["observer_start_ticks"] != receipt["observer_start_ticks"]
        or claim["observer_uid"] != os.getuid()
        or evidence_root.path != str(receipt_path.parent)
    ):
        raise TerminalPolicyError("terminal claim binding mismatch")
    parse_utc_second(claim["created_utc"], "claim timestamp")
    _validate_directory_fd(evidence_root_fd, evidence_root, "terminal evidence root")
    _journal_info, journal_digest, _journal_size, journal_bytes = _regular_file_at(
        evidence_root_fd,
        publication.journal,
        "terminal journal",
        capture=True,
    )
    if journal_bytes is None or journal_digest != receipt["journal_sha256"]:
        raise TerminalPolicyError("terminal journal hash mismatch")
    records = [
        parse_canonical_object(line, "journal record")
        for line in journal_bytes.splitlines(keepends=True)
    ]
    if not records:
        raise TerminalPolicyError("terminal journal is empty")
    authorization = records[0]
    _exact(authorization, set(AUTHORIZATION_FIELDS), "journal authorization")
    verify_message("authorization", authorization, authority)
    if authorization["schema"] != WIRE_SCHEMA or authorization["type"] != "authorization":
        raise TerminalPolicyError("invalid journal authorization")
    authorization_epoch_window(authorization)
    for key in (
        "authority_key_id",
        "authorization_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "claim_sha256",
        "observer_key_id",
        "observer_pid",
        "observer_start_ticks",
    ):
        if authorization[key] != receipt[key]:
            raise TerminalPolicyError("journal authorization binding mismatch")
    if authorization["observer_uid"] != claim["observer_uid"]:
        raise TerminalPolicyError("journal authorization claim mismatch")
    validate_sha256(authorization["observer_source_sha256"], "authorization observer source sha256")
    events = records[1:]
    for sequence, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or event.get("schema") != WIRE_SCHEMA
            or event.get("type") != "process_event"
            or event.get("sequence") != sequence
            or event.get("authorization_id") != receipt["authorization_id"]
            or "observer_signature_b64" not in event
        ):
            raise TerminalPolicyError("terminal event journal mismatch")
    if hashlib.sha256(canonical_bytes(events)).hexdigest() != receipt["events_sha256"]:
        raise TerminalPolicyError("terminal event journal mismatch")
    return VerifiedTerminalReceipt(
        receipt,
        key_id,
        authorization,
        tuple(events),
        hashlib.sha256(canonical_bytes(receipt)).hexdigest(),
    )


def _parse_target_result(value: Any) -> TargetResult:
    value = _exact(
        value,
        {
            "command_index",
            "argv_sha256",
            "environment_sha256",
            "return_code",
            "stdout",
            "stderr",
            "validated_artifacts",
            "sealed_artifacts",
            "completed_utc",
        },
        "target result",
    )
    if (
        type(value["command_index"]) is not int
        or value["command_index"] < 0
        or type(value["return_code"]) is not int
        or not isinstance(value["validated_artifacts"], list)
        or not isinstance(value["sealed_artifacts"], list)
    ):
        raise TerminalPolicyError("invalid target result")
    completed_utc = value["completed_utc"]
    parse_utc_second(completed_utc, "target result timestamp")
    stdout = _parse_validated(value["stdout"])
    stderr = _parse_validated(value["stderr"])
    validated = tuple(_parse_validated(item) for item in value["validated_artifacts"])
    sealed = tuple(_parse_sealed(item) for item in value["sealed_artifacts"])
    if (
        stdout.kind != "stdout"
        or stderr.kind != "stderr"
        or len({item.kind for item in validated}) != len(validated)
        or len({item.kind for item in sealed}) != len(sealed)
        or {item.kind for item in validated} & {item.kind for item in sealed}
    ):
        raise TerminalPolicyError("invalid target artifact contract")
    return TargetResult(
        value["command_index"],
        validate_sha256(value["argv_sha256"], "argv sha256"),
        validate_sha256(value["environment_sha256"], "environment sha256"),
        value["return_code"],
        stdout,
        stderr,
        validated,
        sealed,
        completed_utc,
    )


def _snapshot_digest_at(
    root_fd: int, label: str, tree: list[tuple[str, os.stat_result]] | None = None
) -> str:
    rows: list[list[Any]] = []
    for relative, info in _walk_tree_at(root_fd, label) if tree is None else tree:
        if stat.S_ISDIR(info.st_mode):
            rows.append(
                [
                    relative,
                    "directory",
                    int(info.st_dev),
                    int(info.st_ino),
                    int(info.st_mtime_ns),
                    int(info.st_ctime_ns),
                ]
            )
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TerminalPolicyError("snapshot contains unsafe entry")
        stable, digest, byte_count, _payload = _regular_file_at(
            root_fd, relative, f"{label} file {relative}"
        )
        if (stable.st_dev, stable.st_ino) != (info.st_dev, info.st_ino):
            raise TerminalPolicyError("snapshot tree identity changed")
        rows.append(
            [
                relative,
                "file",
                int(stable.st_dev),
                int(stable.st_ino),
                byte_count,
                int(stable.st_mtime_ns),
                int(stable.st_ctime_ns),
                digest,
            ]
        )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _snapshot_digest(root: Path) -> str:
    root_fd = open_directory_fd(root, "snapshot root")
    try:
        return _snapshot_digest_at(root_fd, "snapshot")
    finally:
        os.close(root_fd)


def _inventory_at(
    root_fd: int, entries: Any, label: str, tree: list[tuple[str, os.stat_result]] | None = None
) -> dict[str, tuple[str, int]]:
    if not isinstance(entries, list):
        raise TerminalPolicyError(f"{label} inventory is invalid")
    declared: dict[str, tuple[str, int]] = {}
    for entry in entries:
        _exact(entry, {"byte_count", "relative_path", "sha256"}, f"{label} inventory entry")
        relative = entry["relative_path"]
        try:
            _relative_parts(relative, f"{label} inventory")
        except TerminalPolicyError as exc:
            raise TerminalPolicyError(f"{label} inventory path is invalid") from exc
        if relative in declared:
            raise TerminalPolicyError(f"{label} inventory path is invalid")
        declared[relative] = (
            validate_sha256(entry["sha256"], f"{label} inventory sha256"),
            _integer(entry["byte_count"], f"{label} inventory byte count"),
        )
    if list(declared) != sorted(declared):
        raise TerminalPolicyError(f"{label} inventory is not canonically sorted")
    tree = _walk_tree_at(root_fd, label) if tree is None else tree
    actual = {
        relative
        for relative, info in tree
        if relative != "."
        and not stat.S_ISDIR(info.st_mode)
        and stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and relative != "SEALED.json"
    }
    if any(
        relative != "."
        and not stat.S_ISDIR(info.st_mode)
        and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)
        for relative, info in tree
    ):
        raise TerminalPolicyError(f"{label} inventory contains an unsafe object")
    if actual != set(declared):
        raise TerminalPolicyError(f"{label} inventory does not cover tree")
    enumerated = dict(tree)
    for relative, (digest, size) in declared.items():
        info, actual_digest, byte_count, _payload = _regular_file_at(
            root_fd, relative, f"{label} inventory file {relative}"
        )
        known = enumerated[relative]
        if (
            (info.st_dev, info.st_ino) != (known.st_dev, known.st_ino)
            or stat.S_IMODE(info.st_mode) != 0o444
            or byte_count != size
            or actual_digest != digest
        ):
            raise TerminalPolicyError(f"{label} inventory identity drift")
    return declared


def _inventory(root: Path, entries: Any, label: str) -> dict[str, tuple[str, int]]:
    root_fd = open_directory_fd(root, label)
    try:
        return _inventory_at(root_fd, entries, label)
    finally:
        os.close(root_fd)


def _safe_tree_files_at(
    root_fd: int, label: str, tree: list[tuple[str, os.stat_result]] | None = None
) -> set[str]:
    tree = _walk_tree_at(root_fd, label) if tree is None else tree
    root_info = tree[0][1]
    if root_info.st_uid != os.getuid() or root_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise TerminalPolicyError(f"{label} root is unsafe")
    files: set[str] = set()
    for relative, info in tree[1:]:
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TerminalPolicyError(f"{label} tree contains an unsafe object")
        files.add(relative)
    return files


def _safe_tree_files(root: Path, label: str) -> set[str]:
    root_fd = open_directory_fd(root, label)
    try:
        return _safe_tree_files_at(root_fd, label)
    finally:
        os.close(root_fd)


@dataclass(frozen=True)
class _AcquisitionContract:
    symbol: str
    raw_start: int
    raw_end: int
    feature_start: int
    feature_end: int
    raw_start_utc: str
    raw_end_utc: str
    feature_start_utc: str
    feature_end_utc: str


def _acquisition_utc_ms(value: Any, label: str) -> int:
    if type(value) is not str or not value.endswith("Z"):
        raise TerminalPolicyError(f"source acquisition {label} mismatch")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise TerminalPolicyError(f"source acquisition {label} mismatch") from exc
    if parsed.tzinfo != UTC or parsed.isoformat().replace("+00:00", "Z") != value:
        raise TerminalPolicyError(f"source acquisition {label} mismatch")
    return int(parsed.timestamp() * 1000)


def _acquisition_contract(
    records: AcquisitionRecords | PhaseRecords | OneTouchRecords,
) -> tuple[_AcquisitionContract, ...]:
    info, digest, size, payload = _regular_file(
        records.contract_manifest.path, "source contract", capture=True
    )
    identity = records.contract_manifest
    if (digest, size, info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode), info.st_nlink) != (
        identity.sha256,
        identity.byte_count,
        identity.st_dev,
        identity.st_ino,
        identity.mode,
        identity.nlink,
    ):
        raise TerminalPolicyError("source contract identity drift")
    if payload is None:
        raise AssertionError("captured source contract is missing")
    try:
        value = parse_canonical_object(payload, "source acquisition contract")
    except TerminalPolicyError as exc:
        raise TerminalPolicyError("source acquisition contract mismatch") from exc
    if (
        value.get("schema_version") != "alpha_max_contract_manifest.v2"
        or value.get("exchange") != "binance"
    ):
        raise TerminalPolicyError("source acquisition contract mismatch")
    rows = value.get("records")
    if not isinstance(rows, list) or [x.get("symbol") for x in rows if isinstance(x, dict)] != list(
        _SYMBOLS
    ):
        raise TerminalPolicyError("source acquisition contract mismatch")
    required = {
        "market_type": "perpetual",
        "linear": True,
        "inverse": False,
        "quote_asset": "USDT",
        "margin_asset": "USDT",
        "settle_asset": "USDT",
        "volume_unit": "base_asset",
        "contract_multiplier": 1.0,
    }
    result = []
    for row in rows:
        if not isinstance(row, dict) or any(
            row.get(key) != expected for key, expected in required.items()
        ):
            raise TerminalPolicyError("source acquisition contract mismatch")
        contract = _AcquisitionContract(
            row["symbol"],
            _acquisition_utc_ms(row.get("raw_availability_start_utc"), "contract"),
            _acquisition_utc_ms(row.get("raw_availability_end_utc"), "contract"),
            _acquisition_utc_ms(row.get("feature_availability_start_utc"), "contract"),
            _acquisition_utc_ms(row.get("feature_availability_end_utc"), "contract"),
            row["raw_availability_start_utc"],
            row["raw_availability_end_utc"],
            row["feature_availability_start_utc"],
            row["feature_availability_end_utc"],
        )
        if contract.raw_end <= contract.raw_start or contract.feature_end <= contract.feature_start:
            raise TerminalPolicyError("source acquisition contract mismatch")
        result.append(contract)
    return tuple(result)


def _acquisition_months(start: int, end: int) -> list[datetime]:
    current = datetime.fromtimestamp(start / 1000, UTC).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    result = []
    while int(current.timestamp() * 1000) < end:
        result.append(current)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result


def _acquisition_receipt(
    report_fd: int,
    enumerated: dict[str, os.stat_result],
    path: str,
    requested_url: str,
    query: dict[str, str] | None = None,
) -> str:
    _info, payload_sha, payload_size, _payload = _regular_file_at_enumerated(
        report_fd, path, "official payload", enumerated
    )
    receipt = _canonical_object_at_enumerated(
        report_fd, path + ".receipt.json", "official request receipt", enumerated
    )
    expected_query = query or {}
    expected = {
        "schema": "official_request_receipt.v1",
        "requested_url": requested_url,
        "final_url": requested_url,
        "final_host": urllib.parse.urlsplit(requested_url).hostname,
        "query": expected_query,
        "retrieved_at_utc": receipt.get("retrieved_at_utc"),
        "byte_count": payload_size,
        "sha256": payload_sha,
    }
    retrieved = expected["retrieved_at_utc"]
    if (
        urllib.parse.urlsplit(requested_url).scheme != "https"
        or urllib.parse.urlsplit(requested_url).hostname
        not in {"fapi.binance.com", "data.binance.vision"}
        or not isinstance(retrieved, str)
    ):
        raise TerminalPolicyError("source official report coverage mismatch")
    _acquisition_utc_ms(retrieved, "retrieval time")
    if receipt != expected or any(type(value) is not str for value in expected_query.values()):
        raise TerminalPolicyError("source official report coverage mismatch")
    return payload_sha


def _acquisition_json_value(
    report_fd: int,
    enumerated: dict[str, os.stat_result],
    path: str,
) -> Any:
    known = enumerated.get(path)
    if known is None or known.st_size > 64 << 20:
        raise TerminalPolicyError("source official report coverage mismatch")
    _info, _digest_value, _byte_count, payload = _regular_file_at_enumerated(
        report_fd,
        path,
        "official JSON payload",
        enumerated,
        capture=True,
    )
    if payload is None:
        raise AssertionError("captured official payload is missing")

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON constant")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise TerminalPolicyError("source official report coverage mismatch") from exc


def _acquisition_partition_path(relative: str) -> str:
    return "partitions/" + hashlib.sha256(relative.encode("utf-8")).hexdigest() + ".json"


def _acquisition_partition(
    report_fd: int,
    enumerated: dict[str, os.stat_result],
    relative: str,
    source_sha: str,
    output_sha: str,
    rows: int,
    start: int,
    end: int,
    input_carry: float | None,
    output_carry: float | None,
    code_sha: str,
    page_hashes: list[str],
) -> None:
    receipt = _canonical_object_at_enumerated(
        report_fd,
        _acquisition_partition_path(relative),
        "source partition receipt",
        enumerated,
    )
    expected = {
        "schema": "alpha_max_partition_receipt.v2",
        "path": relative,
        "source_sha256": source_sha,
        "output_sha256": output_sha,
        "rows": rows,
        "start_ms": start,
        "end_ms": end,
        "input_carry_close": input_carry,
        "output_carry_close": output_carry,
        "derivation_version": "alpha-max-binance-ohlcv-v4",
        "code_sha256": code_sha,
        "page_hashes": page_hashes,
    }
    if receipt != expected:
        if receipt.get("derivation_version") != "alpha-max-binance-ohlcv-v4":
            raise TerminalPolicyError("source derivation version mismatch")
        raise TerminalPolicyError("source partition receipt mismatch")


def _acquisition_raw_partition(
    report_fd: int,
    enumerated: dict[str, os.stat_result],
    relative: str,
    source_sha: str,
    output_sha: str,
    rows: int,
    start: int,
    end: int,
    input_carry: float | None,
    code_sha: str,
    page_hashes: list[str],
) -> float:
    receipt = _canonical_object_at_enumerated(
        report_fd,
        _acquisition_partition_path(relative),
        "source partition receipt",
        enumerated,
    )
    output_carry = receipt.get("output_carry_close")
    if type(output_carry) not in (int, float) or not math.isfinite(output_carry):
        raise TerminalPolicyError("source partition receipt mismatch")
    _acquisition_partition(
        report_fd,
        enumerated,
        relative,
        source_sha,
        output_sha,
        rows,
        start,
        end,
        input_carry,
        output_carry,
        code_sha,
        page_hashes,
    )
    return float(output_carry)


def _expected_tree_directories(files: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative in files:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _observed_tree_directories(
    enumerated: dict[str, os.stat_result],
) -> set[str]:
    return {
        relative
        for relative, info in enumerated.items()
        if relative != "." and stat.S_ISDIR(info.st_mode)
    }


_STORAGE_CONTRACT = {
    "host_reserve_path": "/mnt/c",
    "host_reserve_bytes": 21_474_836_480,
    "max_live_archives": 1,
    "archive_retention": "retired_after_double_derivation",
}


def _acquisition_detached_archive_receipt(
    report_fd: int,
    enumerated: dict[str, os.stat_result],
    path: str,
    requested_url: str,
) -> tuple[dict[str, Any], str]:
    _info, receipt_sha, _size, payload = _regular_file_at_enumerated(
        report_fd, path, "retired archive request receipt", enumerated, capture=True
    )
    if payload is None:
        raise AssertionError("captured archive receipt is missing")
    receipt = _canonical_object_at_enumerated(
        report_fd, path, "retired archive request receipt", enumerated
    )
    expected = {
        "schema": "official_request_receipt.v1",
        "requested_url": requested_url,
        "final_url": requested_url,
        "final_host": "data.binance.vision",
        "query": {},
        "retrieved_at_utc": receipt.get("retrieved_at_utc"),
        "byte_count": receipt.get("byte_count"),
        "sha256": receipt.get("sha256"),
    }
    if (
        payload != canonical_bytes(receipt)
        or receipt != expected
        or not isinstance(expected["retrieved_at_utc"], str)
        or type(expected["byte_count"]) is not int
        or expected["byte_count"] < 0
    ):
        raise TerminalPolicyError("source retired archive receipt mismatch")
    _acquisition_utc_ms(expected["retrieved_at_utc"], "retrieval time")
    validate_sha256(expected["sha256"], "retired archive sha256")
    return receipt, receipt_sha


def _acquisition_archive_evidence(
    report_fd: int,
    enumerated: dict[str, os.stat_result],
    symbol: str,
    label: str,
    archive: str,
    archive_url: str,
    relative: str,
    output_sha: str,
    output_size: int,
    rows: int,
    start: int,
    end: int,
    input_carry: float | None,
    output_carry: float,
    archive_receipt: dict[str, Any],
    archive_receipt_sha: str,
    checksum_sha: str,
    partition: dict[str, Any],
    prior_derivation: str | None,
    code_sha: str,
) -> str:
    prefix = f"provenance/archive-evidence/{symbol}/{label}"
    derivation_path = prefix + ".derivation.json"
    intent_path = prefix + ".retirement-intent.json"
    deletion_path = prefix + ".deletion.json"
    derivation = _canonical_object_at_enumerated(
        report_fd, derivation_path, "archive derivation receipt", enumerated
    )
    expected_derivation = {
        "schema": "alpha_max_archive_derivation_receipt.v1",
        "output_path": relative,
        "output_sha256": output_sha,
        "output_byte_count": output_size,
        "rows": rows,
        "start_ms": start,
        "end_ms": end,
        "input_carry_close": input_carry,
        "output_carry_close": output_carry,
        "archive_url": archive_url,
        "archive_member": archive.rsplit("/", 1)[-1].removesuffix(".zip") + ".csv",
        "archive_sha256": archive_receipt["sha256"],
        "archive_byte_count": archive_receipt["byte_count"],
        "archive_request_receipt_sha256": archive_receipt_sha,
        "checksum_payload_sha256": partition["source_sha256"],
        "checksum_request_receipt_sha256": checksum_sha,
        "partition_receipt_sha256": hashlib.sha256(canonical_bytes(partition)).hexdigest(),
        "prior_derivation_receipt_sha256": prior_derivation,
        "derivation_version": "alpha-max-binance-ohlcv-v4",
        "code_sha256": code_sha,
    }
    if derivation != expected_derivation:
        raise TerminalPolicyError("source archive derivation receipt mismatch")
    derivation_sha = hashlib.sha256(canonical_bytes(derivation)).hexdigest()
    intent = _canonical_object_at_enumerated(
        report_fd, intent_path, "archive retirement intent", enumerated
    )
    expected_intent = {
        "schema": "alpha_max_archive_retirement_intent.v1",
        "derivation_receipt_sha256": derivation_sha,
        "partition_receipt_sha256": expected_derivation["partition_receipt_sha256"],
        "archive_request_receipt_sha256": archive_receipt_sha,
        "archive_relative_path": archive,
        "archive_sha256": archive_receipt["sha256"],
        "archive_byte_count": archive_receipt["byte_count"],
        "output_path": relative,
        "output_sha256": output_sha,
    }
    if intent != expected_intent:
        raise TerminalPolicyError("source archive retirement intent mismatch")
    deletion = _canonical_object_at_enumerated(
        report_fd, deletion_path, "archive deletion receipt", enumerated
    )
    expected_deletion = {
        "schema": "alpha_max_archive_deletion_receipt.v1",
        "retirement_intent_sha256": hashlib.sha256(canonical_bytes(intent)).hexdigest(),
        "derivation_receipt_sha256": derivation_sha,
        "archive_relative_path": archive,
        "archive_sha256": archive_receipt["sha256"],
        "archive_byte_count": archive_receipt["byte_count"],
        "archive_absent": True,
    }
    if deletion != expected_deletion:
        raise TerminalPolicyError("source archive deletion receipt mismatch")
    return derivation_sha


def _acquisition_oracle(
    records: AcquisitionRecords | PhaseRecords,
    source_fd: int,
    report_fd: int,
    source_files: set[str],
    report_files: set[str],
    source_enumerated: dict[str, os.stat_result],
    report_enumerated: dict[str, os.stat_result],
) -> tuple[list[str], list[str], int, int, str]:
    required_output: set[str] = set()
    required_report: set[str] = {
        "provenance/contract_manifest.json",
        "provenance/availability_evidence.json",
        "provenance/exchangeInfo.json",
        "provenance/exchangeInfo.json.receipt.json",
        "acquisition.journal.jsonl",
    }
    for relative, identity in (
        ("provenance/contract_manifest.json", records.contract_manifest),
        ("provenance/availability_evidence.json", records.availability_evidence),
    ):
        _info, digest, size, _payload = _regular_file_at_enumerated(
            report_fd, relative, "source provenance", report_enumerated
        )
        if (digest, size) != (identity.sha256, identity.byte_count):
            raise TerminalPolicyError("source official report coverage mismatch")
    contracts = _acquisition_contract(records)
    exchange_sha = _acquisition_receipt(
        report_fd,
        report_enumerated,
        "provenance/exchangeInfo.json",
        "https://fapi.binance.com/fapi/v1/exchangeInfo",
    )
    exchange = _acquisition_json_value(
        report_fd,
        report_enumerated,
        "provenance/exchangeInfo.json",
    )
    if not isinstance(exchange, dict) or not isinstance(exchange.get("symbols"), list):
        raise TerminalPolicyError("source official report coverage mismatch")
    current_symbols = {item.get("symbol") for item in exchange["symbols"] if isinstance(item, dict)}
    if any(
        contract.symbol != "TONUSDT" and contract.symbol not in current_symbols
        for contract in contracts
    ):
        raise TerminalPolicyError("source official report coverage mismatch")
    raw_total = funding_total = 0
    for contract in contracts:
        carry: float | None = None
        prior_derivation: str | None = None
        for month in _acquisition_months(contract.raw_start, contract.raw_end):
            label = month.strftime("%Y-%m")
            nominal_start = int(month.timestamp() * 1000)
            nominal_end = int(
                ((month.replace(day=28) + timedelta(days=4)).replace(day=1)).timestamp() * 1000
            )
            start, end = max(nominal_start, contract.raw_start), min(nominal_end, contract.raw_end)
            relative = f"market_ohlcv_1s/binance/{contract.symbol}/{label}.parquet"
            filename = f"{contract.symbol}-aggTrades-{label}.zip"
            archive = f"provenance/archives/{contract.symbol}/{filename}"
            archive_receipt_path = archive + ".receipt.json"
            checksum = archive + ".CHECKSUM"
            base = f"https://data.binance.vision/data/futures/um/monthly/aggTrades/{contract.symbol}/{filename}"
            checksum_payload_sha = _acquisition_receipt(
                report_fd, report_enumerated, checksum, base + ".CHECKSUM"
            )
            _info, checksum_receipt_sha, _size, _payload = _regular_file_at_enumerated(
                report_fd,
                checksum + ".receipt.json",
                "archive checksum receipt",
                report_enumerated,
            )
            archive_receipt, archive_receipt_sha = _acquisition_detached_archive_receipt(
                report_fd, report_enumerated, archive_receipt_path, base
            )
            _info, _digest, _size, checksum_payload = _regular_file_at_enumerated(
                report_fd, checksum, "archive checksum", report_enumerated, capture=True
            )
            try:
                fields = checksum_payload.decode("utf-8", "strict").strip().split()
            except UnicodeDecodeError as exc:
                raise TerminalPolicyError("source official report coverage mismatch") from exc
            if (
                len(fields) not in (1, 2)
                or fields[0].lower() != archive_receipt["sha256"]
                or (len(fields) == 2 and fields[1].lstrip("*") != filename)
            ):
                raise TerminalPolicyError("source official report coverage mismatch")
            _info, output_sha, output_size, _payload = _regular_file_at_enumerated(
                source_fd, relative, "raw output", source_enumerated
            )
            rows = (end - start) // 1000
            output_carry = _acquisition_raw_partition(
                report_fd,
                report_enumerated,
                relative,
                fields[0].lower(),
                output_sha,
                rows,
                start,
                end,
                carry,
                records.acquirer.sha256,
                [checksum_payload_sha, archive_receipt["sha256"]],
            )
            partition = _canonical_object_at_enumerated(
                report_fd,
                _acquisition_partition_path(relative),
                "source partition receipt",
                report_enumerated,
            )
            prior_derivation = _acquisition_archive_evidence(
                report_fd,
                report_enumerated,
                contract.symbol,
                label,
                archive,
                base,
                relative,
                output_sha,
                output_size,
                rows,
                start,
                end,
                carry,
                output_carry,
                archive_receipt,
                archive_receipt_sha,
                checksum_receipt_sha,
                partition,
                prior_derivation,
                records.acquirer.sha256,
            )
            carry = output_carry
            required_output.add(relative)
            evidence_prefix = f"provenance/archive-evidence/{contract.symbol}/{label}"
            required_report.update(
                {
                    archive_receipt_path,
                    checksum,
                    checksum + ".receipt.json",
                    _acquisition_partition_path(relative),
                    evidence_prefix + ".derivation.json",
                    evidence_prefix + ".retirement-intent.json",
                    evidence_prefix + ".deletion.json",
                }
            )
            raw_total += rows
        interval = 14_400_000 if contract.symbol == "TONUSDT" else 28_800_000
        cursor = (
            max(0, contract.feature_start - 2 * interval)
            if contract.symbol == "TONUSDT"
            else contract.feature_start
        )
        page_rows: list[dict[str, Any]] = []
        page_hashes: list[str] = []
        number = 0
        while cursor < contract.feature_end:
            number += 1
            page = f"provenance/funding_pages/{contract.symbol}/{number:06d}.json"
            query = {
                "symbol": contract.symbol,
                "startTime": str(cursor),
                "endTime": str(contract.feature_end - 1),
                "limit": "1000",
            }
            page_hashes.append(
                _acquisition_receipt(
                    report_fd,
                    report_enumerated,
                    page,
                    "https://fapi.binance.com/fapi/v1/fundingRate?" + urllib.parse.urlencode(query),
                    query,
                )
            )
            value = _acquisition_json_value(
                report_fd,
                report_enumerated,
                page,
            )
            if not isinstance(value, list):
                raise TerminalPolicyError("source official report coverage mismatch")
            times = [
                row["fundingTime"]
                for row in value
                if isinstance(row, dict)
                and row.get("symbol") == contract.symbol
                and type(row.get("fundingTime")) is int
            ]
            if (
                len(value) > 1000
                or len(times) != len(value)
                or any(b <= a for a, b in itertools.pairwise(times))
                or any(time < cursor or time >= contract.feature_end for time in times)
            ):
                raise TerminalPolicyError("source official report coverage mismatch")
            page_rows.extend(value)
            required_report.update({page, page + ".receipt.json"})
            if not value or len(value) < 1000:
                break
            cursor = times[-1] + 1
        prefix = f"provenance/funding_pages/{contract.symbol}/"
        if {path for path in report_files if path.startswith(prefix)} != {
            path for path in required_report if path.startswith(prefix)
        }:
            raise TerminalPolicyError("source official report coverage mismatch")
        values: dict[int, dict[str, Any]] = {}
        previous = -1
        proof = (
            contract.feature_start // interval * interval - 2 * interval
            if contract.symbol == "TONUSDT"
            else None
        )
        for row in page_rows:
            source = row["fundingTime"]
            try:
                rate = float(row.get("fundingRate"))
            except (TypeError, ValueError) as exc:
                raise TerminalPolicyError("source official report coverage mismatch") from exc
            settlement = source // interval * interval
            if (
                source <= previous
                or source - settlement not in range(1001)
                or not math.isfinite(rate)
                or settlement in values
            ):
                raise TerminalPolicyError("source official report coverage mismatch")
            previous = source
            values[settlement] = {
                "timestamp_ms": source,
                "source_timestamp_ms": source,
                "exchange": "binance",
                "symbol": contract.symbol,
                "funding_rate": rate,
            }
        if proof is not None:
            if proof not in values or proof + interval in values:
                raise TerminalPolicyError("source official report coverage mismatch")
            del values[proof]
        first = ((contract.feature_start + interval - 1) // interval) * interval
        owned_keys = list(range(first, contract.feature_end, interval))
        if sorted(values) != owned_keys:
            raise TerminalPolicyError("source official report coverage mismatch")
        for day_ms in range(
            (contract.feature_start // 86_400_000) * 86_400_000, contract.feature_end, 86_400_000
        ):
            owned = [values[key] for key in owned_keys if day_ms <= key < day_ms + 86_400_000]
            if not owned:
                continue
            day = datetime.fromtimestamp(day_ms / 1000, UTC).strftime("%Y-%m-%d")
            relative = f"feature_points/exchange=binance/symbol={contract.symbol}/date={day}/funding.parquet"
            _info, output_sha, _size, _payload = _regular_file_at_enumerated(
                source_fd, relative, "funding output", source_enumerated
            )
            _acquisition_partition(
                report_fd,
                report_enumerated,
                relative,
                hashlib.sha256(canonical_bytes(owned)).hexdigest(),
                output_sha,
                len(owned),
                day_ms,
                day_ms + 86_400_000,
                None,
                None,
                records.acquirer.sha256,
                page_hashes,
            )
            required_output.add(relative)
            required_report.add(_acquisition_partition_path(relative))
            funding_total += len(owned)
    expected_source_files = required_output | {".alpha_max_owner.json"}
    if source_files != expected_source_files or _observed_tree_directories(
        source_enumerated
    ) != _expected_tree_directories(expected_source_files):
        raise TerminalPolicyError("source acquisition coverage mismatch")
    return sorted(required_output), sorted(required_report), raw_total, funding_total, exchange_sha


def _validate_acquisition(
    records: AcquisitionRecords, *, include_snapshots: bool = False
) -> tuple[ValidatedArtifact, ...] | tuple[tuple[ValidatedArtifact, ...], tuple[str, str]]:
    with ExitStack() as descriptors:
        return _validate_acquired_source_report_at(
            records, descriptors, include_snapshots=include_snapshots
        )


def _validate_acquired_source_report_at(
    records: AcquisitionRecords | PhaseRecords,
    descriptors: ExitStack,
    *,
    source_parent_fd: int | None = None,
    report_parent_fd: int | None = None,
    source_fd: int | None = None,
    report_fd: int | None = None,
    include_snapshots: bool = False,
) -> tuple[ValidatedArtifact, ...] | tuple[tuple[ValidatedArtifact, ...], tuple[str, str]]:
    source_root_path = validate_lexical_control_path(records.source_root.path)
    report_root_path = validate_lexical_control_path(
        records.report_root.path
        if isinstance(records, AcquisitionRecords)
        else records.source_report.path
    )
    source_root = Path(source_root_path)
    report_root = Path(report_root_path)
    if isinstance(records, AcquisitionRecords):
        if (
            str(source_root.parent) != records.source_root.parent.path
            or source_root.name != records.source_root.leaf
            or str(report_root.parent) != records.report_root.parent.path
            or report_root.name != records.report_root.leaf
        ):
            raise TerminalPolicyError("source root parent binding mismatch")
        source_parent_fd = _open_registered_directory(
            descriptors, records.source_root.parent.path, "source parent"
        )
        report_parent_fd = _open_registered_directory(
            descriptors, records.report_root.parent.path, "source report parent"
        )
        _validate_directory_fd(source_parent_fd, records.source_root.parent, "source parent")
        _validate_directory_fd(report_parent_fd, records.report_root.parent, "source report parent")
        source_fd = _open_registered_child(
            descriptors,
            source_parent_fd,
            records.source_root.leaf,
            os.O_RDONLY | os.O_DIRECTORY,
            "source",
        )
        report_fd = _open_registered_child(
            descriptors,
            report_parent_fd,
            records.report_root.leaf,
            os.O_RDONLY | os.O_DIRECTORY,
            "source report",
        )
    if (
        source_parent_fd is None
        or report_parent_fd is None
        or source_fd is None
        or report_fd is None
    ):
        raise TerminalPolicyError("acquired source descriptors are required")
    source_tree = _walk_tree_at(source_fd, "source")
    report_tree = _walk_tree_at(report_fd, "source report")
    source_files = _safe_tree_files_at(source_fd, "source", source_tree)
    report_files = _safe_tree_files_at(report_fd, "source report", report_tree)
    source_enumerated = dict(source_tree)
    report_enumerated = dict(report_tree)

    plan = _canonical_object_at_enumerated(report_fd, "plan.json", "source plan", report_enumerated)
    expected_plan = {
        "schema": "alpha_max_official_acquisition_plan.v4",
        "source_eligible": False,
        "symbols": list(_SYMBOLS),
        "months": [],
        "contract_sha256": records.contract_manifest.sha256,
        "availability_evidence_sha256": records.availability_evidence.sha256,
        "storage_contract": _STORAGE_CONTRACT,
    }
    if plan != expected_plan:
        raise TerminalPolicyError("source acquisition plan mismatch")
    run_id = hashlib.sha256(canonical_bytes(plan)).hexdigest()

    owner_fields = {
        "schema",
        "run_id",
        "output_path",
        "report_path",
        "output_parent_identity",
        "report_parent_identity",
        "output_identity",
        "report_identity",
        "uid",
        "contract_sha256",
        "availability_evidence_sha256",
        "derivation_version",
        "code_sha256",
    }
    source_owner = _canonical_object_at_enumerated(
        source_fd, ".alpha_max_owner.json", "source owner", source_enumerated
    )
    report_owner = _canonical_object_at_enumerated(
        report_fd, ".alpha_max_owner.json", "source report owner", report_enumerated
    )
    _exact(source_owner, owner_fields, "source owner")
    _exact(report_owner, owner_fields, "source report owner")
    source_info = _validate_directory_fd(source_fd, None, "source")
    report_info = _validate_directory_fd(report_fd, None, "source report")
    source_parent_info = _validate_directory_fd(
        source_parent_fd,
        records.source_root.parent if isinstance(records, AcquisitionRecords) else None,
        "source parent",
    )
    report_parent_info = _validate_directory_fd(
        report_parent_fd,
        records.report_root.parent if isinstance(records, AcquisitionRecords) else None,
        "source report parent",
    )
    owner_expected = {
        "schema": "alpha_max_owned_roots.v2",
        "run_id": run_id,
        "output_path": str(source_root),
        "report_path": str(report_root),
        "output_parent_identity": [source_parent_info.st_dev, source_parent_info.st_ino],
        "report_parent_identity": [report_parent_info.st_dev, report_parent_info.st_ino],
        "output_identity": [source_info.st_dev, source_info.st_ino],
        "report_identity": [report_info.st_dev, report_info.st_ino],
        "uid": os.getuid(),
        "contract_sha256": records.contract_manifest.sha256,
        "availability_evidence_sha256": records.availability_evidence.sha256,
        "code_sha256": records.acquirer.sha256,
    }
    if source_owner.get("derivation_version") != "alpha-max-binance-ohlcv-v4":
        raise TerminalPolicyError("source derivation version mismatch")
    if source_owner != report_owner or any(
        source_owner.get(key) != expected for key, expected in owner_expected.items()
    ):
        raise TerminalPolicyError("source ownership binding mismatch")

    output_inventory, required_report, raw_total, funding_total, exchange_sha = _acquisition_oracle(
        records,
        source_fd,
        report_fd,
        source_files,
        report_files,
        source_enumerated,
        report_enumerated,
    )
    manifest = _canonical_object_at_enumerated(
        report_fd, "source_manifest.json", "source manifest", report_enumerated
    )
    _exact(
        manifest,
        {
            "schema",
            "contract_sha256",
            "availability_evidence_sha256",
            "derivation_version",
            "storage_contract",
            "archive_evidence_sha256",
            "artifacts",
        },
        "source manifest",
    )
    if manifest.get("derivation_version") != "alpha-max-binance-ohlcv-v4":
        raise TerminalPolicyError("source derivation version mismatch")
    expected_artifacts = []
    for prefix, files, fd, enumerated in (
        ("output", output_inventory, source_fd, source_enumerated),
        ("report", sorted(required_report), report_fd, report_enumerated),
    ):
        for relative in files:
            _info, digest, _size, _payload = _regular_file_at_enumerated(
                fd, relative, f"source manifest artifact {prefix}/{relative}", enumerated
            )
            expected_artifacts.append({"path": f"{prefix}/{relative}", "sha256": digest})
    evidence_artifacts = sorted(
        (
            artifact
            for artifact in expected_artifacts
            if artifact["path"].startswith("report/provenance/archive-evidence/")
        ),
        key=lambda artifact: artifact["path"],
    )
    expected_manifest = {
        "schema": "alpha_max_official_source_manifest.v5",
        "contract_sha256": records.contract_manifest.sha256,
        "availability_evidence_sha256": records.availability_evidence.sha256,
        "derivation_version": "alpha-max-binance-ohlcv-v4",
        "storage_contract": _STORAGE_CONTRACT,
        "archive_evidence_sha256": hashlib.sha256(canonical_bytes(evidence_artifacts)).hexdigest(),
        "artifacts": expected_artifacts,
    }
    if manifest != expected_manifest:
        raise TerminalPolicyError("source official report coverage mismatch")
    expected_report_files = set(required_report) | {
        ".alpha_max_owner.json",
        "plan.json",
        "source_manifest.json",
        "source_eligible_receipt.json",
    }
    if report_files != expected_report_files or _observed_tree_directories(
        report_enumerated
    ) != _expected_tree_directories(expected_report_files):
        raise TerminalPolicyError("source official report coverage mismatch")
    receipt = _canonical_object_at_enumerated(
        report_fd, "source_eligible_receipt.json", "source receipt", report_enumerated
    )
    _info, manifest_sha, _size, _payload = _regular_file_at_enumerated(
        report_fd, "source_manifest.json", "source manifest", report_enumerated
    )
    _info, journal_sha, _size, _payload = _regular_file_at_enumerated(
        report_fd, "acquisition.journal.jsonl", "source journal", report_enumerated
    )
    expected_receipt = {
        "schema": "alpha_max_official_source_receipt.v4",
        "source_eligible": True,
        "raw_rows": raw_total,
        "funding_rows": funding_total,
        "contract_sha256": records.contract_manifest.sha256,
        "availability_evidence_sha256": records.availability_evidence.sha256,
        "derivation_version": "alpha-max-binance-ohlcv-v4",
        "code_sha256": records.acquirer.sha256,
        "storage_contract": _STORAGE_CONTRACT,
        "archive_evidence_sha256": expected_manifest["archive_evidence_sha256"],
        "exchange_info_sha256": exchange_sha,
        "inventory_sha256": hashlib.sha256(canonical_bytes(output_inventory)).hexdigest(),
        "source_manifest_sha256": manifest_sha,
        "acquisition_journal_sha256": journal_sha,
    }
    if receipt.get("derivation_version") != "alpha-max-binance-ohlcv-v4":
        raise TerminalPolicyError("source derivation version mismatch")
    if receipt != expected_receipt:
        raise TerminalPolicyError("source acquisition coverage mismatch")
    artifacts = tuple(
        _artifact_from_at_enumerated(report_fd, report_root_path, name, kind, report_enumerated)
        for name, kind in (
            ("source_eligible_receipt.json", "source_eligible_receipt"),
            ("source_manifest.json", "source_manifest"),
            ("acquisition.journal.jsonl", "source_journal"),
        )
    )
    root_snapshots = (
        _snapshot_digest_at(source_fd, "source", source_tree),
        _snapshot_digest_at(report_fd, "source report", report_tree),
    )
    if include_snapshots:
        return artifacts, root_snapshots
    return artifacts


def _require_current_prerequisite(
    request: ScopeRequest, artifact: ValidatedArtifact, kind: str
) -> None:
    try:
        prerequisite = next(item for item in request.prerequisites if item.kind == kind)
    except StopIteration as exc:
        raise TerminalPolicyError(f"missing {kind} prerequisite") from exc
    if (
        artifact.kind != kind
        or artifact.path != prerequisite.path
        or (
            artifact.sha256,
            artifact.byte_count,
            artifact.st_dev,
            artifact.st_ino,
            artifact.mode,
            artifact.nlink,
        )
        != (
            prerequisite.sha256,
            prerequisite.byte_count,
            prerequisite.st_dev,
            prerequisite.st_ino,
            prerequisite.mode,
            prerequisite.nlink,
        )
    ):
        raise TerminalPolicyError(f"{kind} prerequisite drift")


def validate_completed_command(
    envelope: LaunchEnvelope,
    request: ScopeRequest,
    index: int,
    prior_evidence: CommandEvidence | None = None,
) -> CommandEvidence:
    """Validate filesystem evidence, never a child exit status, before progression."""
    commands = derive_scope_commands(envelope, request)
    if type(index) is not int or not 0 <= index < len(commands):
        raise TerminalPolicyError("command evidence index is invalid")
    if index and (
        prior_evidence is None
        or prior_evidence.command_index != index - 1
        or prior_evidence.state != "verified"
    ):
        raise TerminalPolicyError("command evidence chain is invalid")
    validated: tuple[ValidatedArtifact, ...] = ()
    sealed: tuple[SealedArtifact, ...] = ()
    if request.scope == "acquisition":
        record = request.records
        acquisition = _validate_acquisition(record, include_snapshots=True)
        validated, root_snapshots = acquisition
    elif request.scope == "phase_preparation":
        validated, root_snapshots = _validate_preparation_manifest(request)
    else:
        record = request.records
        if not isinstance(record, OneTouchRecords):
            raise TerminalPolicyError("one-touch records are required")
        _phase_validated, phase_snapshots = _validate_preparation_manifest(request)
        if index == 0:
            with ExitStack() as descriptors:
                _prelock_parent_fd, prelock_fd = _open_output_child(
                    descriptors, record.prelock_output, "prelock bundle root"
                )
                prelock, prelock_tree, _prelock_seal = _sealed_tree_at(
                    prelock_fd, record.prelock_output.path, "prelock_bundle", "artifacts"
                )
                prelock_enumerated = dict(prelock_tree)
                sealed = (prelock,)
                validated = tuple(
                    _artifact_from_at_enumerated(
                        prelock_fd, record.prelock_output.path, relative, kind, prelock_enumerated
                    )
                    for relative, kind in (
                        ("run/prelock_result.json", "prelock_readback"),
                        (
                            "diagnostics/validation/trend_liquidity_falsifier.json",
                            "prelock_observability",
                        ),
                        ("SEALED.json", "prelock_inventory_before"),
                        ("admission/train_liquidity_buckets.json", "input_inventory_before"),
                    )
                )
                bundle_snapshots = (
                    _snapshot_digest_at(prelock_fd, "prelock bundle", prelock_tree),
                )
        else:
            validated, sealed, bundle_snapshots = _one_touch_second_command_artifacts(
                request, index, prior_evidence
            )
        root_snapshots = phase_snapshots + bundle_snapshots
    if (
        prior_evidence is not None
        and request.scope == "acquisition"
        and (
            root_snapshots != prior_evidence.root_snapshot_sha256s
            or validated != prior_evidence.validated_artifacts
        )
    ):
        raise TerminalPolicyError("offline acquisition verification changed authenticated evidence")
    if (
        prior_evidence is not None
        and request.scope == "one_touch"
        and root_snapshots[:-1] != prior_evidence.root_snapshot_sha256s
    ):
        raise TerminalPolicyError("historical execution changed authenticated phase inputs")
    digest = hashlib.sha256(
        canonical_bytes(
            {
                "scope": request.scope,
                "command_index": index,
                "roots": root_snapshots,
                "prior": prior_evidence.snapshot_sha256 if prior_evidence else None,
            }
        )
    ).hexdigest()
    return CommandEvidence(index, "verified", digest, root_snapshots, validated, sealed)


def _terminal_state(value: Any) -> None:
    if not isinstance(value, dict) or value.get("kind") not in {
        "SUCCEEDED",
        "FAILED",
        "START_FAILED",
        "START_UNKNOWN",
        "OBSERVER_LOST",
        "UNAUTHENTICATED_TERMINAL",
    }:
        raise TerminalPolicyError("invalid terminal state")
    fields = {
        "SUCCEEDED": {"kind"},
        "FAILED": {"kind", "failed_command_index"},
        "START_FAILED": {"kind", "command_index", "errno"},
        "START_UNKNOWN": {"kind", "command_index"},
        "OBSERVER_LOST": {"kind", "command_index", "child_pid", "child_start_ticks"},
        "UNAUTHENTICATED_TERMINAL": {"kind", "last_authenticated_sequence"},
    }[value["kind"]]
    _exact(value, fields, "terminal state")
    positive = {
        "errno",
        "child_pid",
        "child_start_ticks",
    }
    for key in fields - {"kind"}:
        _integer(
            value[key],
            f"terminal state {key}",
            positive=key in positive,
        )


def validate_terminal_state(value: Any) -> None:
    """Validate one exact terminal-state record."""
    _terminal_state(value)


def parse_utc_second(value: Any, label: str) -> datetime:
    """Parse one canonical UTC timestamp with exact-second precision."""
    if type(value) is not str or len(value) != 20:
        raise TerminalPolicyError(f"invalid {label}")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise TerminalPolicyError(f"invalid {label}") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise TerminalPolicyError(f"invalid {label}")
    return parsed


def authorization_epoch_window(value: Mapping[str, Any]) -> tuple[int, int]:
    """Validate an authorization window without making a freshness decision."""
    if not isinstance(value, Mapping):
        raise TerminalPolicyError("invalid authorization window")
    not_before = parse_utc_second(value.get("not_before_utc"), "authorization not-before")
    expires = parse_utc_second(value.get("expires_utc"), "authorization expiry")
    if not not_before < expires or (expires - not_before).total_seconds() > 300:
        raise TerminalPolicyError("invalid authorization window")
    return int(not_before.timestamp()), int(expires.timestamp())


def _stream_identity(value: Any, label: str, expected_path: str) -> dict[str, Any]:
    value = _exact(
        value,
        {
            "path",
            "st_dev",
            "st_ino",
            "st_uid",
            "st_gid",
            "mode",
            "nlink",
            "isatty",
        },
        label,
    )
    if (
        value["path"] != expected_path
        or value["mode"] != 0o600
        or value["nlink"] != 1
        or value["isatty"] is not False
        or value["st_uid"] != os.getuid()
        or value["st_gid"] != os.getgid()
        or any(
            type(value[key]) is not int or value[key] < 0
            for key in ("st_dev", "st_ino", "st_uid", "st_gid")
        )
    ):
        raise TerminalPolicyError(f"invalid {label}")
    return value


def _validate_clearance(
    value: Any,
    *,
    envelope: LaunchEnvelope,
    request: ScopeRequest,
    authorization_id: str,
    completed_index: int,
) -> dict[str, Any]:
    value = _exact(value, set(COMMAND_CLEARANCE_FIELDS), "command clearance")
    authority = public_key_from_b64(envelope.authority_key.public_key_b64)
    verify_message("command_clearance", value, authority)
    if (
        value["schema"] != WIRE_SCHEMA
        or value["type"] != "command_clearance"
        or value["authority_key_id"] != envelope.authority_key.key_id
        or value["authorization_id"] != authorization_id
        or value["scope"] != request.scope
        or value["request_id"] != request.request_id
        or value["completed_command_index"] != completed_index
        or value["next_command_index"] != completed_index + 1
    ):
        raise TerminalPolicyError("command clearance binding mismatch")
    validate_sha256(
        value["validated_artifact_snapshot_sha256"],
        "validated artifact snapshot sha256",
    )
    parse_utc_second(value["issued_utc"], "clearance timestamp")
    return value


def validate_scope_artifacts(
    envelope: LaunchEnvelope,
    request: ScopeRequest,
    events: Any,
    evidence: tuple[CommandEvidence, ...] | None = None,
    *,
    allow_incomplete: bool = False,
) -> tuple[TargetResult, ...]:
    with ExitStack() as descriptors:
        evidence_root_fd = _open_registered_directory(
            descriptors, request.evidence_root.path, "scope evidence root"
        )
        _validate_directory_fd(evidence_root_fd, request.evidence_root, "scope evidence root")
        return _validate_scope_artifacts_at(
            envelope,
            request,
            events,
            evidence,
            evidence_root_fd,
            allow_incomplete=allow_incomplete,
        )


def _validate_scope_artifacts_at(
    envelope: LaunchEnvelope,
    request: ScopeRequest,
    events: Any,
    evidence: tuple[CommandEvidence, ...] | None,
    evidence_root_fd: int,
    *,
    allow_incomplete: bool = False,
) -> tuple[TargetResult, ...]:
    commands = derive_scope_commands(envelope, request)
    if not isinstance(events, (tuple, list)) or (not events and not allow_incomplete):
        raise TerminalPolicyError("authenticated events are required")
    observer = envelope.observer_key(request.scope)
    public = public_key_from_b64(observer.public_key_b64)
    base = {
        "schema",
        "type",
        "event",
        "authorization_id",
        "sequence",
        "command_index",
        "argv_sha256",
        "environment_sha256",
        "prior_clearance",
        "observed_utc",
        "observer_signature_b64",
    }
    child = {"child_pid", "child_start_ticks", "stdin_identity", "stdout", "stderr"}
    shapes = {
        "launch_intent": base,
        "child_started": base | child,
        "child_exited": base
        | child
        | {
            "return_code",
            "stdout_sha256",
            "stdout_byte_count",
            "stderr_sha256",
            "stderr_byte_count",
        },
        "start_failed": base | {"errno", "error_name"},
    }
    states = dict.fromkeys(range(len(commands)), "new")
    started: dict[int, dict[str, Any]] = {}
    exits: dict[int, dict[str, Any]] = {}
    intents: dict[int, dict[str, Any]] = {}
    authorization_id: str | None = None
    next_command = 0
    terminal_failure = False
    for sequence, event in enumerate(events):
        if terminal_failure:
            raise TerminalPolicyError("event appears after terminal failure")
        if not isinstance(event, dict) or event.get("event") not in shapes:
            raise TerminalPolicyError("invalid authenticated event")
        _exact(event, shapes[event["event"]], "authenticated event")
        verify_message("process_event", event, public)
        index = event["command_index"]
        if authorization_id is None:
            authorization_id = validate_sha256(event["authorization_id"], "authorization id")
        if (
            event["schema"] != WIRE_SCHEMA
            or event["type"] != "process_event"
            or event["authorization_id"] != authorization_id
            or event["sequence"] != sequence
            or type(index) is not int
            or not 0 <= index < len(commands)
            or event["argv_sha256"] != hashlib.sha256(canonical_bytes(commands[index])).hexdigest()
            or event["environment_sha256"]
            != hashlib.sha256(canonical_bytes(request.environment)).hexdigest()
        ):
            raise TerminalPolicyError("event command binding mismatch")
        parse_utc_second(event["observed_utc"], "event timestamp")
        state = states[index]
        kind = event["event"]
        if kind == "launch_intent":
            if state != "new" or index != next_command:
                raise TerminalPolicyError("replayed or out-of-order launch intent")
            if index == 0:
                if event["prior_clearance"] is not None:
                    raise TerminalPolicyError("first command has unexpected clearance")
            else:
                _validate_clearance(
                    event["prior_clearance"],
                    envelope=envelope,
                    request=request,
                    authorization_id=authorization_id,
                    completed_index=index - 1,
                )
            intents[index] = event
            states[index] = "intended"
            continue
        intent = intents.get(index)
        if intent is None or event["prior_clearance"] != intent["prior_clearance"]:
            raise TerminalPolicyError("command clearance changed during child lifecycle")
        if kind == "child_started":
            if (
                state != "intended"
                or type(event["child_pid"]) is not int
                or event["child_pid"] <= 0
                or type(event["child_start_ticks"]) is not int
                or event["child_start_ticks"] <= 0
                or event["stdin_identity"] != {"kind": "DEVNULL", "isatty": False}
            ):
                raise TerminalPolicyError("invalid child start")
            _stream_identity(
                event["stdout"],
                "child stdout identity",
                request.publication.stdout[index],
            )
            _stream_identity(
                event["stderr"],
                "child stderr identity",
                request.publication.stderr[index],
            )
            started[index] = event
            states[index] = "started"
            continue
        if kind == "child_exited":
            previous = started.get(index)
            if (
                state != "started"
                or previous is None
                or type(event["return_code"]) is not int
                or any(
                    event[key] != previous[key]
                    for key in (
                        "child_pid",
                        "child_start_ticks",
                        "stdin_identity",
                        "stdout",
                        "stderr",
                    )
                )
            ):
                raise TerminalPolicyError("invalid child exit")
            for digest_key in ("stdout_sha256", "stderr_sha256"):
                validate_sha256(event[digest_key], digest_key)
            for count_key in ("stdout_byte_count", "stderr_byte_count"):
                _integer(event[count_key], count_key)
            exits[index] = event
            if event["return_code"] == 0:
                states[index] = "exited"
                next_command += 1
            else:
                states[index] = "failed"
                terminal_failure = True
            continue
        if kind == "start_failed":
            if (
                state != "intended"
                or type(event["errno"]) is not int
                or event["errno"] <= 0
                or not isinstance(event["error_name"], str)
                or not event["error_name"]
            ):
                raise TerminalPolicyError("invalid child start failure")
            states[index] = "failed"
            terminal_failure = True
            continue
        raise TerminalPolicyError("unknown event transition")

    if not allow_incomplete and (
        len(exits) != len(commands)
        or any(event["return_code"] != 0 for event in exits.values())
        or any(state != "exited" for state in states.values())
    ):
        raise TerminalPolicyError("scope did not complete successfully")

    command_evidence: list[CommandEvidence] = []
    results: list[TargetResult] = []
    for index, event in sorted(exits.items()):
        stdout = _artifact_from_at(
            evidence_root_fd,
            request.evidence_root.path,
            request.publication.stdout[index],
            "stdout",
        )
        stderr = _artifact_from_at(
            evidence_root_fd,
            request.evidence_root.path,
            request.publication.stderr[index],
            "stderr",
        )
        for identity, artifact in (
            (event["stdout"], stdout),
            (event["stderr"], stderr),
        ):
            if identity["path"] != Path(artifact.path).name or (
                identity["st_dev"],
                identity["st_ino"],
                identity["mode"],
                identity["nlink"],
            ) != (
                artifact.st_dev,
                artifact.st_ino,
                artifact.mode,
                artifact.nlink,
            ):
                raise TerminalPolicyError("observer stream identity mismatch")
        if (
            event["stdout_sha256"] != stdout.sha256
            or event["stdout_byte_count"] != stdout.byte_count
            or event["stderr_sha256"] != stderr.sha256
            or event["stderr_byte_count"] != stderr.byte_count
        ):
            raise TerminalPolicyError("observer output identity mismatch")
        validated: tuple[ValidatedArtifact, ...] = ()
        sealed: tuple[SealedArtifact, ...] = ()
        if event["return_code"] == 0:
            verified = validate_completed_command(
                envelope,
                request,
                index,
                command_evidence[-1] if command_evidence else None,
            )
            if index > 0 and (
                intents[index]["prior_clearance"]["validated_artifact_snapshot_sha256"]
                != command_evidence[-1].snapshot_sha256
            ):
                raise TerminalPolicyError("clearance evidence mismatch")
            if evidence is not None and (len(evidence) <= index or evidence[index] != verified):
                raise TerminalPolicyError("command evidence mismatch")
            command_evidence.append(verified)
            validated = verified.validated_artifacts
            sealed = verified.sealed_artifacts
        results.append(
            TargetResult(
                index,
                event["argv_sha256"],
                event["environment_sha256"],
                event["return_code"],
                stdout,
                stderr,
                validated,
                sealed,
                event["observed_utc"],
            )
        )
        if event["return_code"] != 0:
            break
    if evidence is not None and len(evidence) != len(command_evidence):
        raise TerminalPolicyError("command evidence count mismatch")
    return tuple(results)


def _artifact_from_at(root_fd: int, root_path: str, relative: str, kind: str) -> ValidatedArtifact:
    info, digest, byte_count, _payload = _regular_file_at(root_fd, relative, f"{kind} artifact")
    if info.st_uid != os.getuid() or info.st_gid != os.getgid():
        raise TerminalPolicyError(f"{kind} artifact owner mismatch")
    return ValidatedArtifact(
        kind,
        str(Path(root_path) / relative),
        digest,
        byte_count,
        info.st_dev,
        info.st_ino,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
    )


def _one_touch_second_command_artifacts(
    request: ScopeRequest, index: int, prior_evidence: CommandEvidence
) -> tuple[tuple[ValidatedArtifact, ...], tuple[SealedArtifact, ...], tuple[str, ...]]:
    if not isinstance(request.records, OneTouchRecords):
        raise TerminalPolicyError("one-touch records are required")
    if type(index) is not int or index != 1:
        raise TerminalPolicyError("one-touch second command index is invalid")
    record = request.records
    with ExitStack() as descriptors:
        _prelock_parent_fd, prelock_fd = _open_output_child(
            descriptors, record.prelock_output, "prelock bundle root"
        )
        _historical_parent_fd, historical_fd = _open_output_child(
            descriptors, record.historical_output, "historical bundle root"
        )
        try:
            prior_prelock = next(
                item for item in prior_evidence.sealed_artifacts if item.kind == "prelock_bundle"
            )
        except StopIteration as exc:
            raise TerminalPolicyError("prior prelock evidence is missing") from exc
        prelock, prelock_tree, _prelock_seal = _sealed_tree_at(
            prelock_fd, record.prelock_output.path, "prelock_bundle", "artifacts"
        )
        if prelock != prior_prelock:
            raise TerminalPolicyError("prelock sealed receipt changed")
        historical, historical_tree, historical_seal = _sealed_tree_at(
            historical_fd,
            record.historical_output.path,
            "historical_bundle",
            "historical_artifacts",
        )
        prelock_enumerated = dict(prelock_tree)
        historical_enumerated = dict(historical_tree)
        prelock_snapshot = _snapshot_digest_at(prelock_fd, "prelock bundle", prelock_tree)
        if (
            historical_seal["prelock_seal_sha256"] != prelock.sha256
            or historical_seal["prelock_snapshot_sha256"] != prelock_snapshot
        ):
            raise TerminalPolicyError("historical prelock binding mismatch")
        before_seal = _regular_file_at_enumerated(
            prelock_fd, "SEALED.json", "prelock seal before historical", prelock_enumerated
        )
        after_seal = _regular_file_at_enumerated(
            historical_fd,
            "binding/prelock_seal.json",
            "prelock seal after historical",
            historical_enumerated,
        )
        before_inputs = _regular_file_at_enumerated(
            prelock_fd,
            "admission/train_liquidity_buckets.json",
            "prelock inputs before historical",
            prelock_enumerated,
        )
        after_inputs = _regular_file_at_enumerated(
            historical_fd,
            "admission/train_liquidity_buckets.json",
            "prelock inputs after historical",
            historical_enumerated,
        )
        if before_seal[1:3] != after_seal[1:3] or before_inputs[1:3] != after_inputs[1:3]:
            raise TerminalPolicyError("historical immutable input comparison failed")
        validated = tuple(
            _artifact_from_at_enumerated(
                historical_fd, record.historical_output.path, relative, kind, historical_enumerated
            )
            for relative, kind in (
                ("report/historical_result.json", "historical_readback"),
                (
                    "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json",
                    "historical_observability",
                ),
                ("binding/prelock_seal.json", "prelock_inventory_after"),
                ("admission/train_liquidity_buckets.json", "input_inventory_after"),
            )
        )
        snapshots = (
            prelock_snapshot,
            _snapshot_digest_at(historical_fd, "historical bundle", historical_tree),
        )
        return validated, (historical,), snapshots


def _phase_utc(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, UTC).isoformat().replace("+00:00", "Z")


def _phase_source_map_at(
    source_fd: int,
    source_enumerated: dict[str, os.stat_result],
) -> dict[str, tuple[str, int, int, int]]:
    result: dict[str, tuple[str, int, int, int]] = {}
    for relative in _safe_tree_files_at(
        source_fd,
        "phase source",
        list(source_enumerated.items()),
    ):
        if not (relative.startswith("market_ohlcv_1s/") or relative.startswith("feature_points/")):
            continue
        if not relative.endswith(".parquet"):
            raise TerminalPolicyError("preparation manifest entry set mismatch")
        info, digest, byte_count, _payload = _regular_file_at_enumerated(
            source_fd,
            relative,
            f"phase source {relative}",
            source_enumerated,
        )
        result[relative] = (digest, byte_count, info.st_dev, info.st_ino)
    return result


def _phase_expected_layout(
    records: PhaseRecords | OneTouchRecords,
    source_map: dict[str, tuple[str, int, int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    availability = {
        kind: {
            "availability_start_by_symbol": {},
            "availability_end_by_symbol": {},
        }
        for kind in ("raw", "feature")
    }
    expected: list[dict[str, Any]] = []
    for contract in _acquisition_contract(records):
        for root_kind, start, end, start_text, end_text, monthly in (
            (
                "raw",
                contract.raw_start,
                contract.raw_end,
                contract.raw_start_utc,
                contract.raw_end_utc,
                True,
            ),
            (
                "feature",
                contract.feature_start,
                contract.feature_end,
                contract.feature_start_utc,
                contract.feature_end_utc,
                False,
            ),
        ):
            availability[root_kind]["availability_start_by_symbol"][contract.symbol] = start_text
            availability[root_kind]["availability_end_by_symbol"][contract.symbol] = end_text
            for phase_id, phase_start_text, phase_end_text in _PHASE_INTERVALS:
                phase_start = _acquisition_utc_ms(phase_start_text, "phase")
                phase_end = _acquisition_utc_ms(phase_end_text, "phase")
                effective_start = max(start, phase_start)
                effective_end = min(end, phase_end)
                if effective_start >= effective_end:
                    continue
                cursor = datetime.fromtimestamp(effective_start / 1000, UTC)
                cursor = (
                    cursor.replace(
                        day=1,
                        hour=0,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )
                    if monthly
                    else cursor.replace(hour=0, minute=0, second=0, microsecond=0)
                )
                while int(cursor.timestamp() * 1000) < effective_end:
                    following = (
                        (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
                        if monthly
                        else cursor + timedelta(days=1)
                    )
                    cursor_ms = int(cursor.timestamp() * 1000)
                    following_ms = int(following.timestamp() * 1000)
                    owned_start = max(effective_start, cursor_ms)
                    owned_end = min(effective_end, following_ms)
                    if monthly:
                        source_relative = (
                            f"market_ohlcv_1s/binance/{contract.symbol}/{cursor:%Y-%m}.parquet"
                        )
                        output_relative = f"{phase_id}/raw/{source_relative}"
                    else:
                        prefix = (
                            "feature_points/exchange=binance/"
                            f"symbol={contract.symbol}/date={cursor:%Y-%m-%d}/"
                        )
                        matches = sorted(
                            relative
                            for relative in source_map
                            if relative.startswith(prefix) and relative.endswith(".parquet")
                        )
                        if len(matches) != 1:
                            raise TerminalPolicyError("preparation manifest entry set mismatch")
                        source_relative = matches[0]
                        output_relative = f"{phase_id}/feature/{prefix}part-0.parquet"
                    source = source_map.get(source_relative)
                    if source is None:
                        raise TerminalPolicyError("preparation manifest entry set mismatch")
                    expected.append(
                        {
                            "phase_id": phase_id,
                            "root_kind": root_kind,
                            "symbol": contract.symbol,
                            "owned_start_utc": _phase_utc(owned_start),
                            "owned_end_utc": _phase_utc(owned_end),
                            "output_relative_path": output_relative,
                            "source_relative_path": source_relative,
                            "source_sha256": source[0],
                            "source_byte_count": source[1],
                        }
                    )
                    cursor = following
    if {entry["source_relative_path"] for entry in expected} != set(source_map):
        raise TerminalPolicyError("preparation manifest entry set mismatch")
    expected.sort(key=lambda entry: entry["output_relative_path"])
    if len({entry["output_relative_path"] for entry in expected}) != len(expected):
        raise TerminalPolicyError("preparation manifest entry set mismatch")
    return expected, availability


def _phase_snapshot_source_map_at(
    descriptors: ExitStack,
    output_parent_fd: int,
    output: Path,
) -> tuple[
    int,
    dict[str, os.stat_result],
    dict[str, tuple[str, int, int, int]],
    os.stat_result,
]:
    snapshot_leaf = f".{output.name}.alpha_max_phase_preparation.source-snapshot"
    snapshot_fd = _open_registered_child(
        descriptors,
        output_parent_fd,
        snapshot_leaf,
        os.O_RDONLY | os.O_DIRECTORY,
        "phase source snapshot",
    )
    snapshot_info = _validate_directory_fd(
        snapshot_fd,
        None,
        "phase source snapshot",
    )
    snapshot_tree = _walk_tree_at(snapshot_fd, "phase source snapshot")
    snapshot_enumerated = dict(snapshot_tree)
    snapshot_manifest = _canonical_object_at_enumerated(
        snapshot_fd,
        "snapshot-manifest.json",
        "phase snapshot manifest",
        snapshot_enumerated,
    )
    _exact(
        snapshot_manifest,
        {"schema", "descriptor_sha256", "source_manifest_sha256", "entries"},
        "phase snapshot manifest",
    )
    if snapshot_manifest[
        "schema"
    ] != "alpha_max_phase_preparation_source_snapshot.v1" or not isinstance(
        snapshot_manifest["entries"], list
    ):
        raise TerminalPolicyError("phase snapshot inventory mismatch")
    validate_sha256(
        snapshot_manifest["descriptor_sha256"],
        "phase snapshot descriptor sha256",
    )
    validate_sha256(
        snapshot_manifest["source_manifest_sha256"],
        "phase snapshot source manifest sha256",
    )
    source_map: dict[str, tuple[str, int, int, int]] = {}
    declared_paths: list[str] = []
    for entry in snapshot_manifest["entries"]:
        _exact(
            entry,
            {"source_relative_path", "sha256", "byte_count"},
            "phase snapshot entry",
        )
        relative = entry["source_relative_path"]
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not relative.endswith(".parquet")
            or not (
                relative.startswith("market_ohlcv_1s/") or relative.startswith("feature_points/")
            )
        ):
            raise TerminalPolicyError("phase snapshot inventory mismatch")
        expected_sha = validate_sha256(
            entry["sha256"],
            "phase snapshot entry sha256",
        )
        expected_size = _integer(
            entry["byte_count"],
            "phase snapshot entry byte count",
        )
        info, digest, byte_count, _payload = _regular_file_at_enumerated(
            snapshot_fd,
            relative,
            f"phase snapshot clone {relative}",
            snapshot_enumerated,
        )
        if (
            stat.S_IMODE(info.st_mode) != 0o444
            or digest != expected_sha
            or byte_count != expected_size
        ):
            raise TerminalPolicyError("phase snapshot clone mismatch")
        source_map[relative] = (digest, byte_count, info.st_dev, info.st_ino)
        declared_paths.append(relative)
    if declared_paths != sorted(source_map) or len(declared_paths) != len(source_map):
        raise TerminalPolicyError("phase snapshot inventory mismatch")
    completion = _canonical_object_at_enumerated(
        snapshot_fd,
        ".complete.json",
        "phase snapshot completion marker",
        snapshot_enumerated,
    )
    _info, manifest_sha, _size, _payload = _regular_file_at_enumerated(
        snapshot_fd,
        "snapshot-manifest.json",
        "phase snapshot manifest",
        snapshot_enumerated,
    )
    if completion != {
        "schema": "alpha_max_phase_preparation_source_snapshot.v1",
        "snapshot_manifest_sha256": manifest_sha,
    }:
        raise TerminalPolicyError("phase snapshot completion marker mismatch")
    expected_files = set(source_map) | {"snapshot-manifest.json", ".complete.json"}
    if (
        _safe_tree_files_at(snapshot_fd, "phase source snapshot", snapshot_tree) != expected_files
        or _observed_tree_directories(snapshot_enumerated)
        != _expected_tree_directories(expected_files)
        or any(
            stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) != 0o555
            for _relative, info in snapshot_tree
        )
    ):
        raise TerminalPolicyError("phase snapshot inventory mismatch")
    return snapshot_fd, snapshot_enumerated, source_map, snapshot_info


def _validate_phase_handoff_at(
    request: ScopeRequest,
    output_parent_fd: int,
    output_fd: int,
    output: Path,
    output_info: os.stat_result,
    manifest_digest: str,
    file_count: int,
) -> ValidatedArtifact:
    handoff_leaf = f".{output.name}.alpha_max_phase_preparation.handoff.json"
    receipt = _canonical_object_at(output_parent_fd, handoff_leaf, "phase handoff receipt")
    _exact(
        receipt,
        {
            "schema",
            "invocation_descriptor_sha256",
            "source_eligibility_snapshot",
            "verifier_argv_sha256",
            "preparer_argv_sha256",
            "preparer_result",
            "output_root_identity",
            "source_snapshot_manifest_sha256",
            "source_snapshot_identity",
            "output_manifest_sha256",
        },
        "phase handoff receipt",
    )
    preparer_result = receipt["preparer_result"]
    _exact(
        preparer_result,
        {"file_count", "output_root", "preparation_manifest_sha256"},
        "phase preparer result",
    )
    source_snapshot = receipt["source_eligibility_snapshot"]
    _exact(
        source_snapshot,
        {
            "source_root_identity",
            "source_report_identity",
            "source_eligible_receipt_sha256",
            "source_manifest_sha256",
            "acquisition_journal_sha256",
            "plan_sha256",
            "source_owner_sha256",
            "report_owner_sha256",
            "source_manifest_artifact_map_sha256",
        },
        "phase source eligibility snapshot",
    )
    for key in (
        "invocation_descriptor_sha256",
        "verifier_argv_sha256",
        "preparer_argv_sha256",
        "source_snapshot_manifest_sha256",
        "output_manifest_sha256",
    ):
        validate_sha256(receipt[key], f"phase handoff {key}")
    snapshot_identity = _snapshot_identity(
        receipt["source_snapshot_identity"],
        "phase handoff source snapshot identity",
    )
    snapshot_leaf = f".{output.name}.alpha_max_phase_preparation.source-snapshot"
    try:
        snapshot_fd = _open_child_fd(
            output_parent_fd,
            snapshot_leaf,
            os.O_RDONLY | os.O_DIRECTORY,
            "phase handoff source snapshot",
        )
    except TerminalPolicyError as exc:
        raise TerminalPolicyError("phase handoff source snapshot identity mismatch") from exc
    try:
        snapshot_info = _validate_directory_fd(
            snapshot_fd,
            None,
            "phase handoff source snapshot",
        )
        _info, snapshot_manifest_sha, _size, _payload = _regular_file_at(
            snapshot_fd,
            "snapshot-manifest.json",
            "phase handoff source snapshot manifest",
        )
    finally:
        os.close(snapshot_fd)
    if snapshot_identity != {
        "st_dev": snapshot_info.st_dev,
        "st_ino": snapshot_info.st_ino,
    }:
        raise TerminalPolicyError("phase handoff source snapshot identity mismatch")
    if receipt["source_snapshot_manifest_sha256"] != snapshot_manifest_sha:
        raise TerminalPolicyError("phase handoff receipt mismatch")
    for key in (
        "source_eligible_receipt_sha256",
        "source_manifest_sha256",
        "acquisition_journal_sha256",
        "plan_sha256",
        "source_owner_sha256",
        "report_owner_sha256",
        "source_manifest_artifact_map_sha256",
    ):
        validate_sha256(source_snapshot[key], f"phase source eligibility {key}")
    if (
        receipt["schema"] != "alpha_max_phase_preparation_eligible_source_receipt.v2"
        or receipt["output_manifest_sha256"] != manifest_digest
        or preparer_result["file_count"] != file_count
        or preparer_result["output_root"] != str(output)
        or preparer_result["preparation_manifest_sha256"] != manifest_digest
        or receipt["output_root_identity"]
        != {"st_dev": output_info.st_dev, "st_ino": output_info.st_ino}
        or not all(
            isinstance(source_snapshot[key], dict)
            and set(source_snapshot[key]) == {"st_dev", "st_ino"}
            and all(type(value) is int and value >= 0 for value in source_snapshot[key].values())
            for key in ("source_root_identity", "source_report_identity")
        )
    ):
        raise TerminalPolicyError("phase handoff receipt mismatch")
    return _artifact_from_at(
        output_parent_fd, str(output.parent), handoff_leaf, "phase_handoff_receipt"
    )


def _validate_preparation_manifest(
    request: ScopeRequest,
) -> tuple[tuple[ValidatedArtifact, ...], tuple[str, ...]]:
    if not isinstance(request.records, (PhaseRecords, OneTouchRecords)):
        raise TerminalPolicyError("phase preparation records are required")
    records = request.records
    output = Path(records.phase_output.path)
    with ExitStack() as descriptors:
        if isinstance(records, OneTouchRecords):
            output_parent_fd = open_directory_fd(output.parent, "one-touch phase output parent")
            descriptors.callback(os.close, output_parent_fd)
            output_fd = _open_child_fd(
                output_parent_fd,
                output.name,
                os.O_RDONLY | os.O_DIRECTORY,
                "one-touch phase output",
            )
            descriptors.callback(os.close, output_fd)
            output_info = _validate_directory_fd(
                output_fd, records.phase_output, "one-touch phase output"
            )
            parent_path = str(output.parent)
        else:
            output_parent_fd, output_fd = _open_output_child(
                descriptors, records.phase_output, "phase output"
            )
            output_info = _validate_directory_fd(output_fd, None, "phase output")
            parent_path = str(output.parent)

        output_tree = _walk_tree_at(output_fd, "phase output")
        output_enumerated = dict(output_tree)
        manifest = _canonical_object_at_enumerated(
            output_fd, "preparation_manifest.json", "preparation manifest", output_enumerated
        )
        _exact(
            manifest,
            {
                "availability",
                "availability_sha256_by_root_kind",
                "contract_manifest_schema_version",
                "contract_manifest_sha256",
                "exchange",
                "file_count",
                "files",
                "phase_intervals",
                "schema_version",
                "symbols",
            },
            "preparation manifest",
        )
        expected_intervals = [
            {"phase_id": phase, "start_utc": start, "end_utc": end}
            for phase, start, end in _PHASE_INTERVALS
        ]
        if (
            manifest["schema_version"] != "alpha_max_phase_root_preparation_manifest.v1"
            or manifest["exchange"] != "binance"
            or manifest["contract_manifest_sha256"] != records.contract_manifest.sha256
            or manifest["contract_manifest_schema_version"] != "alpha_max_contract_manifest.v2"
            or manifest["symbols"] != list(_SYMBOLS)
            or manifest["phase_intervals"] != expected_intervals
        ):
            raise TerminalPolicyError("preparation manifest semantic mismatch")
        availability = manifest["availability"]
        availability_hashes = manifest["availability_sha256_by_root_kind"]
        _exact(availability, {"raw", "feature"}, "preparation availability")
        _exact(availability_hashes, {"raw", "feature"}, "preparation availability hashes")
        for root_kind in ("raw", "feature"):
            section = availability[root_kind]
            _exact(
                section,
                {"availability_start_by_symbol", "availability_end_by_symbol"},
                f"{root_kind} availability",
            )
            starts = section["availability_start_by_symbol"]
            ends = section["availability_end_by_symbol"]
            if (
                not isinstance(starts, dict)
                or not isinstance(ends, dict)
                or tuple(starts) != _SYMBOLS
                or tuple(ends) != _SYMBOLS
                or any(
                    not isinstance(starts[symbol], str)
                    or not isinstance(ends[symbol], str)
                    or starts[symbol] >= ends[symbol]
                    for symbol in _SYMBOLS
                )
                or availability_hashes[root_kind]
                != hashlib.sha256(canonical_bytes(section)).hexdigest()
            ):
                raise TerminalPolicyError("preparation availability mismatch")

        files = manifest["files"]
        if (
            type(manifest["file_count"]) is not int
            or not isinstance(files, list)
            or manifest["file_count"] <= 0
            or manifest["file_count"] != len(files)
        ):
            raise TerminalPolicyError("preparation manifest entry set mismatch")
        declared: list[str] = []
        source_file_facts: dict[str, tuple[str, int, int, int]] = {}

        source_fd: int
        report_fd: int | None = None
        source_enumerated: dict[str, os.stat_result]
        report_enumerated: dict[str, os.stat_result] = {}
        if isinstance(records, PhaseRecords):
            source_parent = Path(records.source_root.path).parent
            source_parent_fd = open_directory_fd(source_parent, "phase source parent")
            descriptors.callback(os.close, source_parent_fd)
            source_fd = _open_child_fd(
                source_parent_fd,
                Path(records.source_root.path).name,
                os.O_RDONLY | os.O_DIRECTORY,
                "phase source root",
            )
            descriptors.callback(os.close, source_fd)
            _validate_directory_fd(source_fd, records.source_root, "phase source root")
            report_parent = Path(records.source_report.path).parent
            report_parent_fd = open_directory_fd(report_parent, "phase source report parent")
            descriptors.callback(os.close, report_parent_fd)
            report_fd = _open_child_fd(
                report_parent_fd,
                Path(records.source_report.path).name,
                os.O_RDONLY | os.O_DIRECTORY,
                "phase source report",
            )
            descriptors.callback(os.close, report_fd)
            _validate_directory_fd(report_fd, records.source_report, "phase source report")
            acquired_artifacts, _acquired_snapshots = _validate_acquired_source_report_at(
                records,
                descriptors,
                source_parent_fd=source_parent_fd,
                report_parent_fd=report_parent_fd,
                source_fd=source_fd,
                report_fd=report_fd,
                include_snapshots=True,
            )
            for artifact in acquired_artifacts:
                _require_current_prerequisite(request, artifact, artifact.kind)
            source_enumerated = dict(_walk_tree_at(source_fd, "phase source root"))
            report_enumerated = dict(_walk_tree_at(report_fd, "phase source report"))
            source_map = _phase_source_map_at(source_fd, source_enumerated)
            (
                _snapshot_fd,
                _snapshot_enumerated,
                snapshot_source_map,
                _snapshot_info,
            ) = _phase_snapshot_source_map_at(
                descriptors,
                output_parent_fd,
                output,
            )
            if {relative: facts[:2] for relative, facts in snapshot_source_map.items()} != {
                relative: facts[:2] for relative, facts in source_map.items()
            }:
                raise TerminalPolicyError("phase snapshot inventory mismatch")
        else:
            (
                source_fd,
                source_enumerated,
                source_map,
                _snapshot_info,
            ) = _phase_snapshot_source_map_at(
                descriptors,
                output_parent_fd,
                output,
            )

        expected_layout, expected_availability = _phase_expected_layout(
            records,
            source_map,
        )
        expected_by_path = {entry["output_relative_path"]: entry for entry in expected_layout}
        declared_paths = [
            entry.get("output_relative_path") if isinstance(entry, dict) else None
            for entry in files
        ]
        if (
            availability != expected_availability
            or availability_hashes
            != {
                kind: hashlib.sha256(canonical_bytes(expected_availability[kind])).hexdigest()
                for kind in ("raw", "feature")
            }
            or declared_paths != sorted(expected_by_path)
            or len(files) != len(expected_layout)
        ):
            raise TerminalPolicyError("preparation manifest entry set mismatch")
        for entry in files:
            _exact(
                entry,
                {
                    "output_byte_count",
                    "output_relative_path",
                    "output_row_count",
                    "output_sha256",
                    "owned_end_utc",
                    "owned_start_utc",
                    "phase_id",
                    "root_kind",
                    "source_byte_count",
                    "source_relative_path",
                    "source_sha256",
                    "symbol",
                },
                "preparation manifest entry",
            )
            relative = entry["output_relative_path"]
            source_relative = entry["source_relative_path"]
            phase = entry["phase_id"]
            root_kind = entry["root_kind"]
            symbol = entry["symbol"]
            expected = expected_by_path.get(relative)
            if (
                not isinstance(relative, str)
                or not isinstance(source_relative, str)
                or not relative
                or not source_relative
                or relative.startswith("/")
                or source_relative.startswith("/")
                or ".." in Path(relative).parts
                or ".." in Path(source_relative).parts
                or root_kind not in {"raw", "feature"}
                or symbol not in _SYMBOLS
                or not relative.startswith(f"{phase}/{root_kind}/")
                or expected is None
                or any(
                    entry[field] != expected[field]
                    for field in (
                        "phase_id",
                        "root_kind",
                        "symbol",
                        "owned_start_utc",
                        "owned_end_utc",
                        "source_relative_path",
                        "source_sha256",
                        "source_byte_count",
                    )
                )
            ):
                raise TerminalPolicyError("preparation manifest entry set mismatch")
            output_info_file, output_digest, output_size, _ = _regular_file_at_enumerated(
                output_fd, relative, f"preparation output {relative}", output_enumerated
            )
            if (
                stat.S_IMODE(output_info_file.st_mode) != 0o444
                or output_size
                != _integer(entry["output_byte_count"], "preparation output byte count")
                or output_digest
                != validate_sha256(entry["output_sha256"], "preparation output sha256")
            ):
                raise TerminalPolicyError("preparation output identity drift")
            source_info, source_digest, source_size, _ = _regular_file_at_enumerated(
                source_fd,
                source_relative,
                f"preparation source {source_relative}",
                source_enumerated,
            )
            source_file_facts[source_relative] = (
                source_digest,
                source_size,
                source_info.st_dev,
                source_info.st_ino,
            )
            if source_size != _integer(
                entry["source_byte_count"], "preparation source byte count"
            ) or source_digest != validate_sha256(
                entry["source_sha256"], "preparation source sha256"
            ):
                raise TerminalPolicyError("preparation source identity drift")
            _integer(entry["output_row_count"], "preparation output row count", positive=True)
            declared.append(relative)
        if declared != sorted(set(declared)):
            raise TerminalPolicyError("preparation manifest paths are not canonical")
        actual_files = _safe_tree_files_at(output_fd, "phase output", output_tree)
        expected_output_files = set(declared) | {"preparation_manifest.json"}
        expected_output_directories = _expected_tree_directories(expected_output_files) | set(
            _PHASES
        )
        if (
            actual_files != expected_output_files
            or _observed_tree_directories(output_enumerated) != expected_output_directories
        ):
            raise TerminalPolicyError("preparation manifest entry set mismatch")
        if any(
            stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) != 0o555
            for _relative, info in output_tree
        ):
            raise TerminalPolicyError("phase output directory is not immutable")

        manifest_info, manifest_digest, _manifest_size, _ = _regular_file_at_enumerated(
            output_fd, "preparation_manifest.json", "preparation manifest", output_enumerated
        )
        if isinstance(records, OneTouchRecords):
            if stat.S_IMODE(manifest_info.st_mode) != 0o444:
                raise TerminalPolicyError("phase receipts are not immutable")
            _read_identity(records.contract_manifest, "one-touch contract manifest")
            artifact = _artifact_from_at_enumerated(
                output_fd,
                str(output),
                "preparation_manifest.json",
                "preparation_manifest",
                output_enumerated,
            )
            _require_current_prerequisite(request, artifact, "preparation_manifest")
            handoff = _validate_phase_handoff_at(
                request,
                output_parent_fd,
                output_fd,
                output,
                output_info,
                manifest_digest,
                len(files),
            )
            _require_current_prerequisite(request, handoff, "phase_handoff_receipt")
            return (handoff, artifact), (
                _snapshot_digest_at(output_fd, "phase output", output_tree),
            )

        _validate_phase_handoff_at(
            request,
            output_parent_fd,
            output_fd,
            output,
            output_info,
            manifest_digest,
            len(files),
        )
        assert output_parent_fd is not None and source_fd is not None and report_fd is not None
        prefix = f".{output.name}.alpha_max_phase_preparation"
        handoff_leaf = f"{prefix}.handoff.json"
        receipt = _canonical_object_at(output_parent_fd, handoff_leaf, "phase handoff receipt")
        _exact(
            receipt,
            {
                "schema",
                "invocation_descriptor_sha256",
                "source_eligibility_snapshot",
                "verifier_argv_sha256",
                "preparer_argv_sha256",
                "preparer_result",
                "output_root_identity",
                "source_snapshot_manifest_sha256",
                "source_snapshot_identity",
                "output_manifest_sha256",
            },
            "phase handoff receipt",
        )
        preparer_result = receipt["preparer_result"]
        _exact(
            preparer_result,
            {"file_count", "output_root", "preparation_manifest_sha256"},
            "phase preparer result",
        )
        if (
            receipt["schema"] != "alpha_max_phase_preparation_eligible_source_receipt.v2"
            or receipt["output_manifest_sha256"] != manifest_digest
            or preparer_result["file_count"] != len(files)
            or preparer_result["output_root"] != str(output)
            or preparer_result["preparation_manifest_sha256"] != receipt["output_manifest_sha256"]
            or receipt["output_root_identity"]
            != {"st_dev": output_info.st_dev, "st_ino": output_info.st_ino}
        ):
            raise TerminalPolicyError("phase handoff receipt mismatch")
        for key in (
            "invocation_descriptor_sha256",
            "verifier_argv_sha256",
            "preparer_argv_sha256",
            "source_snapshot_manifest_sha256",
        ):
            validate_sha256(receipt[key], f"phase handoff {key}")

        source_manifest = _canonical_object_at_enumerated(
            report_fd, "source_manifest.json", "phase source manifest", report_enumerated
        )
        _exact(
            source_manifest,
            {
                "schema",
                "contract_sha256",
                "availability_evidence_sha256",
                "derivation_version",
                "storage_contract",
                "archive_evidence_sha256",
                "artifacts",
            },
            "phase source manifest",
        )
        if (
            source_manifest["schema"] != "alpha_max_official_source_manifest.v5"
            or source_manifest["contract_sha256"] != records.contract_manifest.sha256
            or source_manifest["availability_evidence_sha256"]
            != records.availability_evidence.sha256
            or not isinstance(source_manifest["artifacts"], list)
        ):
            raise TerminalPolicyError("phase source manifest mismatch")
        artifact_map: dict[str, str] = {}
        for entry in source_manifest["artifacts"]:
            _exact(entry, {"path", "sha256"}, "phase source artifact")
            relative = entry["path"]
            if (
                not isinstance(relative, str)
                or relative in artifact_map
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or not (relative.startswith("output/") or relative.startswith("report/"))
            ):
                raise TerminalPolicyError("phase source artifact path is invalid")
            artifact_map[relative] = validate_sha256(
                entry["sha256"], "phase source artifact sha256"
            )
        if list(artifact_map) != sorted(artifact_map):
            raise TerminalPolicyError("phase source artifact map is not canonical")
        source_snapshot = receipt["source_eligibility_snapshot"]
        _exact(
            source_snapshot,
            {
                "source_root_identity",
                "source_report_identity",
                "source_eligible_receipt_sha256",
                "source_manifest_sha256",
                "acquisition_journal_sha256",
                "plan_sha256",
                "source_owner_sha256",
                "report_owner_sha256",
                "source_manifest_artifact_map_sha256",
            },
            "phase source eligibility snapshot",
        )
        expected_source_snapshot = {
            "source_root_identity": {
                "st_dev": records.source_root.st_dev,
                "st_ino": records.source_root.st_ino,
            },
            "source_report_identity": {
                "st_dev": records.source_report.st_dev,
                "st_ino": records.source_report.st_ino,
            },
            "source_eligible_receipt_sha256": _regular_file_at_enumerated(
                report_fd,
                "source_eligible_receipt.json",
                "phase source eligible receipt",
                report_enumerated,
            )[1],
            "source_manifest_sha256": _regular_file_at_enumerated(
                report_fd, "source_manifest.json", "phase source manifest", report_enumerated
            )[1],
            "acquisition_journal_sha256": _regular_file_at_enumerated(
                report_fd, "acquisition.journal.jsonl", "phase source journal", report_enumerated
            )[1],
            "plan_sha256": _regular_file_at_enumerated(
                report_fd, "plan.json", "phase source plan", report_enumerated
            )[1],
            "source_owner_sha256": _regular_file_at_enumerated(
                source_fd, ".alpha_max_owner.json", "phase source owner", source_enumerated
            )[1],
            "report_owner_sha256": _regular_file_at_enumerated(
                report_fd, ".alpha_max_owner.json", "phase source report owner", report_enumerated
            )[1],
            "source_manifest_artifact_map_sha256": hashlib.sha256(
                canonical_bytes(artifact_map)
            ).hexdigest(),
        }
        if source_snapshot != expected_source_snapshot:
            raise TerminalPolicyError("phase source eligibility snapshot mismatch")

        descriptor_leaf = f"{prefix}.invocation.json"
        lock_leaf = f"{prefix}.lock"
        snapshot_leaf = f"{prefix}.source-snapshot"
        descriptor = _canonical_object_at(
            output_parent_fd, descriptor_leaf, "phase invocation descriptor"
        )
        _exact(
            descriptor,
            {
                "schema",
                "paths",
                "forbidden_roots",
                "frozen_sha256",
                "invocation_lock_identity",
                "source_eligibility_snapshot",
                "verifier_argv",
                "preparer_argv",
            },
            "phase invocation descriptor",
        )
        snapshot_root = Path(parent_path) / snapshot_leaf
        descriptor_path = Path(parent_path) / descriptor_leaf
        invocation_inputs = Path(parent_path) / f"{prefix}.invocation-inputs"
        expected_paths = {
            "acquirer": records.acquirer.path,
            "source_root": records.source_root.path,
            "source_report": records.source_report.path,
            "contract_manifest": records.contract_manifest.path,
            "availability_evidence": records.availability_evidence.path,
            "preparer": records.preparer.path,
            "output_root": records.phase_output.path,
            "raw_root": str(snapshot_root / "market_ohlcv_1s"),
            "feature_root": str(snapshot_root / "feature_points"),
            "invocation_descriptor": str(descriptor_path),
            "invocation_descriptor_stage": str(
                Path(parent_path) / f"{prefix}.invocation.stage.json"
            ),
            "handoff_receipt": str(Path(parent_path) / handoff_leaf),
            "handoff_receipt_stage": str(Path(parent_path) / f"{prefix}.handoff.stage.json"),
            "source_snapshot": str(snapshot_root),
            "source_snapshot_manifest": str(snapshot_root / "snapshot-manifest.json"),
            "source_snapshot_complete": str(snapshot_root / ".complete.json"),
            "invocation_inputs": str(invocation_inputs),
            "invocation_input_acquirer": str(invocation_inputs / "acquirer.py"),
            "invocation_input_contract_manifest": str(invocation_inputs / "contract_manifest.json"),
            "invocation_input_availability_evidence": str(
                invocation_inputs / "availability_evidence.json"
            ),
            "invocation_input_preparer": str(invocation_inputs / "preparer.py"),
            "invocation_lock": str(Path(parent_path) / lock_leaf),
        }
        if descriptor["paths"] != expected_paths:
            raise TerminalPolicyError("phase invocation paths mismatch")
        expected_frozen = {
            "acquirer": records.acquirer.sha256,
            "contract_manifest": records.contract_manifest.sha256,
            "availability_evidence": records.availability_evidence.sha256,
            "preparer": records.preparer.sha256,
            "wrapper": records.phase_wrapper.sha256,
        }
        if (
            descriptor["schema"] != "alpha_max_phase_preparation_invocation.v1"
            or descriptor["forbidden_roots"] != list(request.forbidden_roots)
            or descriptor["frozen_sha256"] != expected_frozen
            or descriptor["source_eligibility_snapshot"] != source_snapshot
        ):
            raise TerminalPolicyError("phase invocation binding mismatch")
        verifier_argv = [
            request.interpreter.path,
            records.acquirer.path,
            "--contract-manifest",
            records.contract_manifest.path,
            "--availability-evidence",
            records.availability_evidence.path,
            "--output-root",
            records.source_root.path,
            "--report-dir",
            records.source_report.path,
        ]
        for forbidden_root in request.forbidden_roots:
            verifier_argv.extend(("--forbidden-root", forbidden_root))
        verifier_argv.append("--verify-eligible")
        preparer_argv = [
            request.interpreter.path,
            records.preparer.path,
            "--raw-root",
            str(snapshot_root / "market_ohlcv_1s"),
            "--feature-root",
            str(snapshot_root / "feature_points"),
            "--contract-manifest",
            records.contract_manifest.path,
            "--output-root",
            records.phase_output.path,
        ]
        if (
            descriptor["verifier_argv"] != verifier_argv
            or descriptor["preparer_argv"] != preparer_argv
            or receipt["verifier_argv_sha256"]
            != hashlib.sha256(canonical_bytes(verifier_argv)).hexdigest()
            or receipt["preparer_argv_sha256"]
            != hashlib.sha256(canonical_bytes(preparer_argv)).hexdigest()
            or receipt["invocation_descriptor_sha256"]
            != _regular_file_at(output_parent_fd, descriptor_leaf, "phase invocation descriptor")[1]
        ):
            raise TerminalPolicyError("phase invocation argv mismatch")
        lock_info, _lock_digest, lock_size, _ = _regular_file_at(
            output_parent_fd, lock_leaf, "phase invocation lock"
        )
        expected_lock_identity = [
            lock_info.st_dev,
            lock_info.st_ino,
            stat.S_IFMT(lock_info.st_mode),
            lock_info.st_nlink,
            lock_size,
            lock_info.st_mtime_ns,
            lock_info.st_ctime_ns,
        ]
        if (
            descriptor["invocation_lock_identity"] != expected_lock_identity
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            raise TerminalPolicyError("phase invocation lock identity mismatch")

        snapshot_fd = _open_child_fd(
            output_parent_fd, snapshot_leaf, os.O_RDONLY | os.O_DIRECTORY, "phase source snapshot"
        )
        descriptors.callback(os.close, snapshot_fd)
        snapshot_tree = _walk_tree_at(snapshot_fd, "phase source snapshot")
        snapshot_enumerated = dict(snapshot_tree)
        snapshot_info = _validate_directory_fd(snapshot_fd, None, "phase source snapshot")
        expected_snapshot_entries: list[dict[str, Any]] = []
        for relative, digest in artifact_map.items():
            path = Path(relative)
            if path.parts[:2] not in {
                ("output", "market_ohlcv_1s"),
                ("output", "feature_points"),
            }:
                continue
            source_relative = Path(*path.parts[1:]).as_posix()
            facts = source_file_facts.get(source_relative)
            if facts is None:
                source_info, source_digest, source_size, _ = _regular_file_at_enumerated(
                    source_fd,
                    source_relative,
                    f"phase snapshot source {source_relative}",
                    source_enumerated,
                )
                facts = (source_digest, source_size, source_info.st_dev, source_info.st_ino)
                source_file_facts[source_relative] = facts
            if facts[0] != digest:
                raise TerminalPolicyError("phase snapshot source digest mismatch")
            expected_snapshot_entries.append(
                {"source_relative_path": source_relative, "sha256": digest, "byte_count": facts[1]}
            )
        expected_snapshot_entries.sort(key=lambda item: item["source_relative_path"])
        snapshot_manifest = _canonical_object_at_enumerated(
            snapshot_fd, "snapshot-manifest.json", "phase snapshot manifest", snapshot_enumerated
        )
        expected_snapshot_manifest = {
            "schema": "alpha_max_phase_preparation_source_snapshot.v1",
            "descriptor_sha256": receipt["invocation_descriptor_sha256"],
            "source_manifest_sha256": expected_source_snapshot["source_manifest_sha256"],
            "entries": expected_snapshot_entries,
        }
        if (
            snapshot_manifest != expected_snapshot_manifest
            or receipt["source_snapshot_manifest_sha256"]
            != _regular_file_at_enumerated(
                snapshot_fd,
                "snapshot-manifest.json",
                "phase snapshot manifest",
                snapshot_enumerated,
            )[1]
        ):
            raise TerminalPolicyError("phase snapshot manifest mismatch")
        if _snapshot_identity(
            receipt["source_snapshot_identity"], "phase handoff source snapshot identity"
        ) != {"st_dev": snapshot_info.st_dev, "st_ino": snapshot_info.st_ino}:
            raise TerminalPolicyError("phase handoff source snapshot identity mismatch")
        expected_complete = {
            "schema": "alpha_max_phase_preparation_source_snapshot.v1",
            "snapshot_manifest_sha256": receipt["source_snapshot_manifest_sha256"],
        }
        if (
            _canonical_object_at_enumerated(
                snapshot_fd,
                ".complete.json",
                "phase snapshot completion marker",
                snapshot_enumerated,
            )
            != expected_complete
        ):
            raise TerminalPolicyError("phase snapshot completion marker mismatch")
        snapshot_files = _safe_tree_files_at(snapshot_fd, "phase source snapshot", snapshot_tree)
        expected_snapshot_files = {
            item["source_relative_path"] for item in expected_snapshot_entries
        } | {"snapshot-manifest.json", ".complete.json"}
        if snapshot_files != expected_snapshot_files:
            raise TerminalPolicyError("phase snapshot inventory mismatch")
        if any(
            stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) != 0o555
            for _relative, info in snapshot_tree
        ):
            raise TerminalPolicyError("phase snapshot directory is not immutable")
        for entry in expected_snapshot_entries:
            relative = entry["source_relative_path"]
            info, digest, byte_count, _ = _regular_file_at_enumerated(
                snapshot_fd, relative, f"phase snapshot clone {relative}", snapshot_enumerated
            )
            source_facts = source_file_facts[relative]
            if (
                stat.S_IMODE(info.st_mode) != 0o444
                or (info.st_dev, info.st_ino) == (source_facts[2], source_facts[3])
                or digest != entry["sha256"]
                or byte_count != entry["byte_count"]
            ):
                raise TerminalPolicyError("phase snapshot clone mismatch")
        handoff_info, _handoff_digest, _handoff_size, _ = _regular_file_at(
            output_parent_fd, handoff_leaf, "phase handoff receipt"
        )
        if (
            stat.S_IMODE(handoff_info.st_mode) != 0o444
            or stat.S_IMODE(manifest_info.st_mode) != 0o444
        ):
            raise TerminalPolicyError("phase receipts are not immutable")
        artifacts = (
            _artifact_from_at(
                output_parent_fd,
                parent_path,
                handoff_leaf,
                "phase_handoff_receipt",
            ),
            _artifact_from_at(
                output_fd,
                str(output),
                "preparation_manifest.json",
                "preparation_manifest",
            ),
        )
        return artifacts, (_snapshot_digest_at(output_fd, "phase output", output_tree),)


def _selection_payload(
    root_fd: int,
    relative: str,
    *,
    role: str,
    enumerated: dict[str, os.stat_result],
) -> dict[str, Any]:
    value = _canonical_object_at_enumerated(root_fd, relative, f"{role} selection", enumerated)
    _exact(
        value,
        {
            "artifact_kind",
            "decisions",
            "historical_evaluation_leader",
            "prelock_champion",
            "ranked_candidate_ids",
            "role",
            "scaling_attributions",
            "selected_candidate_id",
        },
        f"{role} selection",
    )
    expected_kind = (
        "alpha_max_prelock_selection.v2"
        if role == "prelock_selection"
        else "alpha_max_historical_report_ranking.v2"
    )
    ranked = value["ranked_candidate_ids"]
    champion = value["prelock_champion"]
    if (
        value["artifact_kind"] != expected_kind
        or value["role"] != role
        or not isinstance(value["decisions"], list)
        or len(value["decisions"]) != 17
        or not isinstance(ranked, list)
        or any(not isinstance(item, str) or not item for item in ranked)
        or len(ranked) != len(set(ranked))
        or not isinstance(value["scaling_attributions"], list)
        or len(value["scaling_attributions"]) != 2
        or (champion is not None and (not isinstance(champion, str) or not champion))
        or value["selected_candidate_id"] != champion
        or (champion is None and ranked)
        or (champion is not None and (not ranked or ranked[0] != champion))
    ):
        raise TerminalPolicyError(f"{role} selection mismatch")
    return value


def _terminal_payload(
    root_fd: int, relative: str, label: str, enumerated: dict[str, os.stat_result]
) -> dict[str, Any]:
    value = _canonical_object_at_enumerated(root_fd, relative, label, enumerated)
    _exact(
        value,
        {
            "confirmation_status",
            "historical_evaluation_leader",
            "historical_exposure_status",
            "incumbent_comparison_status",
            "leader_differs_from_prelock_champion",
            "prelock_champion",
            "requires_fresh_confirmation",
            "selected_candidate_id",
            "terminal_outcome",
        },
        label,
    )
    if (
        value["selected_candidate_id"] != value["prelock_champion"]
        or type(value["leader_differs_from_prelock_champion"]) is not bool
        or type(value["requires_fresh_confirmation"]) is not bool
        or not isinstance(value["terminal_outcome"], str)
        or not value["terminal_outcome"]
    ):
        raise TerminalPolicyError(f"{label} mismatch")
    return value


def _matrix_and_observability(
    root_fd: int,
    *,
    domain: str,
    physical_fold_run_count: int,
    diagnostic_relative: str,
    enumerated: dict[str, os.stat_result],
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _canonical_object_at_enumerated(
        root_fd, "status/matrix.json", f"{domain} matrix", enumerated
    )
    _exact(
        matrix,
        {
            "artifact_kind",
            "domain",
            "engine_cell_count",
            "physical_fold_run_count",
            "status_count",
            "statuses",
        },
        f"{domain} matrix",
    )
    if (
        matrix["artifact_kind"] != "alpha_max_matrix_statuses.v1"
        or matrix["domain"] != domain
        or matrix["engine_cell_count"] != 68
        or matrix["physical_fold_run_count"] != physical_fold_run_count
        or matrix["status_count"] != 84
        or not isinstance(matrix["statuses"], list)
        or len(matrix["statuses"]) != 84
        or sum(
            item.get("engine_constructed") is True
            for item in matrix["statuses"]
            if isinstance(item, dict)
        )
        != 68
    ):
        raise TerminalPolicyError(f"{domain} matrix mismatch")
    diagnostic = _canonical_object_at_enumerated(
        root_fd, diagnostic_relative, f"{domain} observability", enumerated
    )
    _exact(
        diagnostic,
        {
            "artifact_kind",
            "bucket_contribution_usdt",
            "domain",
            "fold_run_sha256s",
            "nominal_cost_bps",
            "rejection_reasons",
            "report_only",
            "row_id",
            "selection_influence",
            "status",
            "symbol_contribution_usdt",
            "total_contribution_usdt",
            "train_liquidity_buckets",
            "train_liquidity_buckets_sha256",
        },
        f"{domain} observability",
    )
    expected_fold_count = 12 if domain == "validation" else 10
    bucket_digest = _regular_file_at_enumerated(
        root_fd, "admission/train_liquidity_buckets.json", f"{domain} liquidity buckets", enumerated
    )[1]
    if isinstance(diagnostic["fold_run_sha256s"], list):
        for index, item in enumerate(diagnostic["fold_run_sha256s"]):
            validate_sha256(item, f"{domain} fold run {index}")
    if (
        diagnostic["artifact_kind"] != "alpha_max_trend_liquidity_falsifier.v1"
        or diagnostic["domain"] != domain
        or diagnostic["nominal_cost_bps"] != 30
        or diagnostic["report_only"] is not True
        or diagnostic["selection_influence"] is not False
        or diagnostic["row_id"] != "component_trend_1x"
        or not isinstance(diagnostic["fold_run_sha256s"], list)
        or len(diagnostic["fold_run_sha256s"]) != expected_fold_count
        or diagnostic["train_liquidity_buckets_sha256"] != bucket_digest
    ):
        raise TerminalPolicyError(f"{domain} observability mismatch")
    return matrix, diagnostic


def _semantic_bundle_readback(
    root_fd: int,
    root_path: str,
    kind: str,
    seal: dict[str, Any],
    enumerated: dict[str, os.stat_result],
) -> str:
    if kind == "prelock_bundle":
        run = _canonical_object_at_enumerated(
            root_fd, "run/prelock_result.json", "prelock result", enumerated
        )
        _exact(
            run,
            {
                "artifact_kind",
                "engine_cell_count",
                "failure_reasons",
                "physical_fold_run_count",
                "prelock_champion",
                "selected_candidate_id",
                "status",
                "terminal_outcome",
            },
            "prelock result",
        )
        selection = _selection_payload(
            root_fd, "selection/prelock.json", role="prelock_selection", enumerated=enumerated
        )
        terminal = _terminal_payload(
            root_fd, "terminal/prelock.json", "prelock terminal", enumerated
        )
        matrix, diagnostic = _matrix_and_observability(
            root_fd,
            domain="validation",
            physical_fold_run_count=816,
            diagnostic_relative="diagnostics/validation/trend_liquidity_falsifier.json",
            enumerated=enumerated,
        )
        champion = seal["prelock_champion"]
        if (
            run["artifact_kind"] != "alpha_max_prelock_process_result.v1"
            or run["engine_cell_count"] != 68
            or run["physical_fold_run_count"] != 816
            or run["failure_reasons"] != []
            or run["status"] != "complete"
            or run["prelock_champion"] != champion
            or run["selected_candidate_id"] != champion
            or selection["prelock_champion"] != champion
            or terminal["prelock_champion"] != champion
            or run["terminal_outcome"] != terminal["terminal_outcome"]
        ):
            raise TerminalPolicyError("prelock outcome/readback mismatch")
        readback = {
            "run": run,
            "selection": selection,
            "terminal": terminal,
            "matrix": matrix,
            "observability": diagnostic,
        }
    else:
        report = _canonical_object_at_enumerated(
            root_fd, "report/historical_result.json", "historical result", enumerated
        )
        _exact(
            report,
            {
                "artifact_kind",
                "confirmation_status",
                "engine_cell_count",
                "failure_reasons",
                "historical_evaluation_leader",
                "historical_exposure_status",
                "physical_fold_run_count",
                "prelock_champion",
                "requires_fresh_confirmation",
                "selected_candidate_id",
                "status",
                "terminal_outcome",
            },
            "historical result",
        )
        ranking = _selection_payload(
            root_fd,
            "selection/historical_ranking.json",
            role="historical_report",
            enumerated=enumerated,
        )
        terminal = _terminal_payload(
            root_fd, "terminal/historical.json", "historical terminal", enumerated
        )
        matrix, diagnostic = _matrix_and_observability(
            root_fd,
            domain="historical_exposed_evaluation",
            physical_fold_run_count=680,
            diagnostic_relative="diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json",
            enumerated=enumerated,
        )
        if (
            report["artifact_kind"] != "alpha_max_historical_process_result.v1"
            or report["engine_cell_count"] != 68
            or report["physical_fold_run_count"] != 680
            or report["failure_reasons"] != []
            or report["status"] != "complete_report_only"
            or report["selected_candidate_id"] != report["prelock_champion"]
            or ranking["prelock_champion"] != report["prelock_champion"]
            or terminal["prelock_champion"] != report["prelock_champion"]
            or ranking["historical_evaluation_leader"] != report["historical_evaluation_leader"]
            or terminal["historical_evaluation_leader"] != report["historical_evaluation_leader"]
            or terminal["terminal_outcome"] != report["terminal_outcome"]
            or terminal["confirmation_status"] != report["confirmation_status"]
            or terminal["historical_exposure_status"] != report["historical_exposure_status"]
            or terminal["requires_fresh_confirmation"] != report["requires_fresh_confirmation"]
        ):
            raise TerminalPolicyError("historical outcome/readback mismatch")
        readback = {
            "report": report,
            "ranking": ranking,
            "terminal": terminal,
            "matrix": matrix,
            "observability": diagnostic,
        }
    return hashlib.sha256(canonical_bytes(readback)).hexdigest()


def _sealed_tree_at(
    root_fd: int, root_path: str, kind: str, inventory_key: str
) -> tuple[SealedArtifact, list[tuple[str, os.stat_result]], dict[str, Any]]:
    tree = _walk_tree_at(root_fd, kind)
    enumerated = dict(tree)
    value = _canonical_object_at_enumerated(root_fd, "SEALED.json", "sealed bundle", enumerated)
    fields = (
        {
            "artifact_count",
            "artifact_kind",
            "artifacts",
            "historical_evaluation_inputs_included",
            "immutable",
            "inventory_sha256",
            "prelock_champion",
            "selected_candidate_id",
        }
        if kind == "prelock_bundle"
        else {
            "artifact_kind",
            "completion_id",
            "historical_artifacts",
            "immutable",
            "prelock_seal_sha256",
            "prelock_snapshot_sha256",
        }
    )
    _exact(value, fields, "sealed bundle")
    if value["immutable"] is not True:
        raise TerminalPolicyError("sealed bundle is mutable")
    entries = value[inventory_key]
    declared = _inventory_at(root_fd, entries, kind, tree)
    if kind == "prelock_bundle":
        if (
            value["artifact_kind"] != "alpha_max_immutable_prelock_seal.v1"
            or value["artifact_count"] != len(entries)
            or value["historical_evaluation_inputs_included"] is not False
            or value["prelock_champion"] != value["selected_candidate_id"]
        ):
            raise TerminalPolicyError("prelock seal binding mismatch")
    else:
        if (
            value["completion_id"] != "historical_exposed_evaluation"
            or value["artifact_kind"] != "alpha_max_append_only_historical_package.v1"
        ):
            raise TerminalPolicyError("historical seal binding mismatch")
        validate_sha256(value["prelock_seal_sha256"], "historical prelock seal sha256")
        validate_sha256(value["prelock_snapshot_sha256"], "historical prelock snapshot sha256")
    required_counts = (
        (
            (("manifests", "validation_train_fit"), 3, 17),
            (("manifests", "prelock_final_refit"), 3, 17),
            (("capsules", "validation_train_fit"), 4, 204),
            (("capsules", "prelock_final_refit"), 4, 17),
            (("evidence", "validation", "cells"), 5, 68),
            (("evidence", "validation", "rows"), 4, 816),
        )
        if kind == "prelock_bundle"
        else (
            (("capsules", "prelock_final_refit"), 4, 153),
            (("evidence", "historical_exposed_evaluation", "cells"), 5, 68),
            (("evidence", "historical_exposed_evaluation", "rows"), 4, 680),
        )
    )
    for prefix, depth, expected_count in required_counts:
        count = sum(
            len(parts) == depth and parts[: len(prefix)] == prefix and parts[-1].endswith(".json")
            for relative in declared
            for parts in (tuple(relative.split("/")),)
        )
        if count != expected_count:
            raise TerminalPolicyError("sealed artifact cardinality mismatch")
    cardinality_roots = {prefix[0] for prefix, _depth, _count in required_counts}
    if any(
        parts[0] in cardinality_roots
        and not any(
            len(parts) == depth and parts[: len(prefix)] == prefix and parts[-1].endswith(".json")
            for prefix, depth, _count in required_counts
        )
        for relative in declared
        for parts in (tuple(relative.split("/")),)
    ):
        raise TerminalPolicyError("sealed artifact cardinality mismatch")
    required = (
        (
            "admission/train.json",
            "admission/train_computation.json",
            "admission/train_liquidity_buckets.json",
            "allocation/train_fit.json",
            "allocation/train_validation_refit.json",
            "diagnostics/validation/trend_liquidity_falsifier.json",
            "inputs/config.json",
            "inputs/contract_manifest.json",
            "inputs/prior_trial_inventory.json",
            "run/prelock_result.json",
            "selection/prelock.json",
            "status/matrix.json",
            "terminal/prelock.json",
            "trial/ledger.json",
        )
        if kind == "prelock_bundle"
        else (
            "admission/train_liquidity_buckets.json",
            "binding/prelock_seal.json",
            "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json",
            "report/historical_result.json",
            "selection/historical_ranking.json",
            "status/matrix.json",
            "terminal/historical.json",
        )
    )
    if not set(required) <= set(declared):
        raise TerminalPolicyError("required sealed artifact is absent")
    if any(
        stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) != 0o555
        for _relative, info in tree
    ):
        raise TerminalPolicyError("unsafe sealed tree identity")
    inventory_sha256 = hashlib.sha256(canonical_bytes(entries)).hexdigest()
    if kind == "prelock_bundle" and value["inventory_sha256"] != inventory_sha256:
        raise TerminalPolicyError("prelock inventory hash mismatch")
    readback_sha256 = _semantic_bundle_readback(root_fd, root_path, kind, value, enumerated)
    info, digest, byte_count, _payload = _regular_file_at_enumerated(
        root_fd, "SEALED.json", "sealed receipt", enumerated
    )
    if stat.S_IMODE(info.st_mode) != 0o444:
        raise TerminalPolicyError("sealed receipt identity drift")
    return (
        SealedArtifact(
            kind,
            str(Path(root_path) / "SEALED.json"),
            digest,
            byte_count,
            info.st_dev,
            info.st_ino,
            stat.S_IMODE(info.st_mode),
            info.st_nlink,
            digest,
            inventory_sha256,
            readback_sha256,
        ),
        tree,
        value,
    )


def _sealed_tree(root: str, kind: str, inventory_key: str) -> SealedArtifact:
    root_fd = open_directory_fd(root, f"{kind} root")
    try:
        sealed, _tree, _seal = _sealed_tree_at(root_fd, root, kind, inventory_key)
        return sealed
    finally:
        os.close(root_fd)


MAX_PACKET_BYTES = 1_048_576
WIRE_SCHEMA = "alpha-max-terminal-authority/v3"
CLAIM_SCHEMA = "alpha_max_terminal_claim.v1"
CHALLENGE_FIELDS = frozenset(
    {
        "schema",
        "type",
        "authority_key_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "nonce_b64",
        "issued_utc",
        "authority_signature_b64",
    }
)
OBSERVER_PROOF_FIELDS = frozenset(
    {
        "schema",
        "type",
        "authority_key_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "nonce_b64",
        "observer_key_id",
        "observer_pid",
        "observer_uid",
        "observer_start_ticks",
        "observer_source_sha256",
        "claim_sha256",
        "observer_signature_b64",
    }
)
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "type",
        "authority_key_id",
        "authorization_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "claim_sha256",
        "observer_key_id",
        "observer_pid",
        "observer_uid",
        "observer_start_ticks",
        "observer_source_sha256",
        "not_before_utc",
        "expires_utc",
        "authority_signature_b64",
    }
)
COMMAND_CLEARANCE_FIELDS = frozenset(
    {
        "schema",
        "type",
        "authority_key_id",
        "authorization_id",
        "scope",
        "request_id",
        "completed_command_index",
        "next_command_index",
        "validated_artifact_snapshot_sha256",
        "issued_utc",
        "authority_signature_b64",
    }
)
TERMINAL_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "type",
        "authority_key_id",
        "authorization_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "claim_sha256",
        "observer_key_id",
        "observer_pid",
        "observer_start_ticks",
        "command_bundle_sha256",
        "events_sha256",
        "journal_sha256",
        "prerequisites",
        "target_results",
        "terminal_state",
        "publication",
        "created_utc",
        "authority_signature_b64",
    }
)
CLAIM_FIELDS = frozenset(
    {
        "schema",
        "request_id",
        "scope",
        "checkpoint_pin_sha256",
        "evidence_root",
        "observer_pid",
        "observer_uid",
        "observer_start_ticks",
        "created_utc",
    }
)
MESSAGE_SIGNATURE_FIELDS = {
    "challenge": "authority_signature_b64",
    "observer_proof": "observer_signature_b64",
    "publication_observer_ready": "observer_signature_b64",
    "authorization": "authority_signature_b64",
    "command_clearance": "authority_signature_b64",
    "process_event": "observer_signature_b64",
    "terminal_receipt": "authority_signature_b64",
}


def parse_canonical_object(data: bytes, label: str = "JSON") -> dict[str, Any]:
    """Parse one canonical LF-terminated JSON object without permissive JSON quirks."""

    def reject_constant(value: str) -> None:
        raise TerminalPolicyError(f"{label} contains non-finite number")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TerminalPolicyError(f"{label} contains duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(data, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    except TerminalPolicyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalPolicyError(f"invalid {label}") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != data:
        raise TerminalPolicyError(f"non-canonical {label}")
    return value


def _canonical_object(path: Path) -> dict[str, Any]:
    _info, _digest_value, _byte_count, payload = _regular_file(
        path,
        str(path),
        capture=True,
    )
    if payload is None:
        raise AssertionError("captured control file has no payload")
    return parse_canonical_object(payload, str(path))


def packet_bytes(message: Mapping[str, Any]) -> bytes:
    body = canonical_bytes(message)
    if len(body) > MAX_PACKET_BYTES:
        raise TerminalPolicyError("packet exceeds maximum size")
    return struct.pack("!I", len(body)) + body


def send_packet(connection: socket.socket, message: Mapping[str, Any]) -> None:
    packet = packet_bytes(message)
    if connection.send(packet) != len(packet):
        raise TerminalPolicyError("short packet write")


def receive_packet(connection: socket.socket) -> dict[str, Any]:
    packet, ancillary, flags, _address = connection.recvmsg(MAX_PACKET_BYTES + 5)
    if ancillary or flags or len(packet) < 4:
        raise TerminalPolicyError("invalid packet framing")
    declared = struct.unpack("!I", packet[:4])[0]
    body = packet[4:]
    if declared > MAX_PACKET_BYTES or declared != len(body):
        raise TerminalPolicyError("invalid packet framing")
    return parse_canonical_object(body, "packet")


def public_key_from_b64(encoded: str) -> Ed25519PublicKey:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise TerminalPolicyError("invalid public key encoding") from exc
    if len(raw) != 32:
        raise TerminalPolicyError("invalid public key length")
    return Ed25519PublicKey.from_public_bytes(raw)


def public_key_id(public_key: Ed25519PublicKey) -> str:
    return hashlib.sha256(
        public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ).hexdigest()


def secure_private_key(path: Path | str, owner: int | None = None) -> Ed25519PrivateKey:
    """Open an exact raw private key without following links."""
    expected_owner = os.getuid() if owner is None else owner
    info, _digest_value, byte_count, payload = _regular_file(
        path,
        "private key",
        capture=True,
        max_capture_bytes=32,
    )
    if (
        info.st_uid != expected_owner
        or stat.S_IMODE(info.st_mode) != 0o400
        or byte_count != 32
        or payload is None
    ):
        raise TerminalPolicyError("unsafe private key")
    return Ed25519PrivateKey.from_private_bytes(payload)


def sign_message(
    message_type: str, unsigned: Mapping[str, Any], private_key: Ed25519PrivateKey
) -> dict[str, Any]:
    signature_field = MESSAGE_SIGNATURE_FIELDS.get(message_type)
    if signature_field is None:
        raise TerminalPolicyError("unknown signed message type")
    result = dict(unsigned)
    if signature_field in result:
        raise TerminalPolicyError("signature present in unsigned message")
    result[signature_field] = base64.b64encode(
        private_key.sign(signing_preimage(message_type, result))
    ).decode("ascii")
    return result


def verify_message(
    message_type: str, message: Mapping[str, Any], public_key: Ed25519PublicKey
) -> dict[str, Any]:
    signature_field = MESSAGE_SIGNATURE_FIELDS.get(message_type)
    if signature_field is None or signature_field not in message:
        raise TerminalPolicyError("missing message signature")
    unsigned = {key: value for key, value in message.items() if key != signature_field}
    try:
        signature = base64.b64decode(message[signature_field], validate=True)
        public_key.verify(signature, signing_preimage(message_type, unsigned))
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise TerminalPolicyError(f"invalid {message_type} signature") from exc
    return unsigned
