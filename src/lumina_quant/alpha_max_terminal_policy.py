"""Fail-closed schema and argv authority for Alpha-Max terminal scopes."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import stat
import struct
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
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


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
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
        _digest(value["sha256"], "file sha256"),
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
    _digest(value["key_id"], "key id")
    _digest(value["public_key_sha256"], "public key sha256")
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
        {key: _digest(item, key) for key, item in value["pins"].items()},
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
        if _digest(getattr(pin, name), name) != policy.pins[name]:
            raise TerminalPolicyError(f"checkpoint {name} mismatch")
    _digest(pin.authority_manifest_sha256, "authority manifest sha256")
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
        or _digest(value["policy_sha256"], "policy sha256") != policy.source_sha256
    ):
        raise TerminalPolicyError("envelope policy mismatch")
    if (
        _digest(checkpoint.authority_manifest_sha256, "authority manifest sha256")
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
        or _digest(value["checkpoint_pin_sha256"], "request checkpoint") != checkpoint.sha256
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
    prereq_kinds = {
        "acquisition": ("checkpoint_pin", "alignment_receipt"),
        "phase_preparation": (
            "checkpoint_pin",
            "alignment_receipt",
            "source_eligible_receipt",
            "source_manifest",
            "source_journal",
        ),
        "one_touch": (
            "checkpoint_pin",
            "alignment_receipt",
            "phase_handoff_receipt",
            "preparation_manifest",
        ),
    }[scope]
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
        _digest(item.sha256, "prerequisite sha256")
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
    target = validate_lexical_control_path(path)
    parts = tuple(part for part in target.split("/") if part)
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open("/", directory_flags)
    except OSError as exc:
        raise TerminalPolicyError(f"cannot open {label}") from exc
    try:
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if not parts:
            return descriptor
        child = os.open(
            parts[-1],
            flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
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
    _read_directory(output.parent, f"output parent {output.path}")
    if (
        Path(output.path).parent != Path(output.parent.path)
        or Path(output.path).name != output.leaf
    ):
        raise TerminalPolicyError("output parent binding mismatch")
    parent_fd = open_directory_fd(output.parent.path, "output parent")
    try:
        try:
            os.stat(output.leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
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
        _digest(value["sha256"], "artifact sha256"),
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
        _digest(value["sealed_payload_sha256"], "sealed payload"),
        _digest(value["canonical_inventory_sha256"], "inventory"),
        _digest(value["readback_sha256"], "readback"),
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


def verify_signed_receipt(path: Path | str, public_key_path: Path | str) -> VerifiedTerminalReceipt:
    receipt_path = Path(path)
    if not receipt_path.is_absolute():
        raise TerminalPolicyError("terminal receipt path must be absolute")
    receipt = _canonical_object(receipt_path)
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
        _digest(receipt[name], name)
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
    _utc_value(receipt["created_utc"], "receipt timestamp")
    command_count = 1 if receipt["scope"] == "phase_preparation" else 2
    publication = _parse_publication(receipt["publication"], command_count)
    if receipt_path.name != publication.receipt:
        raise TerminalPolicyError("terminal receipt publication mismatch")
    _terminal_state(receipt["terminal_state"])

    prerequisite_kinds = {
        "acquisition": ("checkpoint_pin", "alignment_receipt"),
        "phase_preparation": (
            "checkpoint_pin",
            "alignment_receipt",
            "source_eligible_receipt",
            "source_manifest",
            "source_journal",
        ),
        "one_touch": (
            "checkpoint_pin",
            "alignment_receipt",
            "phase_handoff_receipt",
            "preparation_manifest",
        ),
    }[receipt["scope"]]
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

    expected_kinds: dict[str, tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]] = {
        "acquisition": (
            (
                ("source_eligible_receipt", "source_manifest", "source_journal"),
                (),
            ),
            (
                ("source_eligible_receipt", "source_manifest", "source_journal"),
                (),
            ),
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

    claim_path = receipt_path.parent / publication.claim
    claim_info, claim_digest, _claim_size, claim_bytes = _regular_file(
        claim_path,
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
    _utc_value(claim["created_utc"], "claim timestamp")
    _read_directory(evidence_root, "terminal evidence root")
    journal_path = receipt_path.parent / publication.journal
    _journal_info, journal_digest, _journal_size, journal_bytes = _regular_file(
        journal_path,
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
    not_before = _utc_value(authorization["not_before_utc"], "authorization not-before")
    expires = _utc_value(authorization["expires_utc"], "authorization expiry")
    if not not_before < expires or (expires - not_before).total_seconds() > 300:
        raise TerminalPolicyError("invalid authorization window")
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
    _digest(authorization["observer_source_sha256"], "authorization observer source sha256")
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
    return VerifiedTerminalReceipt(receipt, key_id)


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
    _utc_value(completed_utc, "target result timestamp")
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
        _digest(value["argv_sha256"], "argv sha256"),
        _digest(value["environment_sha256"], "environment sha256"),
        value["return_code"],
        stdout,
        stderr,
        validated,
        sealed,
        completed_utc,
    )


def _snapshot_digest(root: Path) -> str:
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise TerminalPolicyError("snapshot root is missing") from exc
    if resolved != root or root.is_symlink():
        raise TerminalPolicyError("snapshot root identity is invalid")
    rows: list[list[Any]] = []
    paths = (root, *sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))))
    for path in paths:
        info = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise TerminalPolicyError("snapshot contains symlink")
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
        stable, digest, byte_count, _payload = _regular_file(path, f"snapshot file {relative}")
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


def _inventory(root: Path, entries: Any, label: str) -> dict[str, tuple[str, int]]:
    if not isinstance(entries, list):
        raise TerminalPolicyError(f"{label} inventory is invalid")
    declared: dict[str, tuple[str, int]] = {}
    for entry in entries:
        _exact(entry, {"byte_count", "relative_path", "sha256"}, f"{label} inventory entry")
        relative = entry["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in declared
        ):
            raise TerminalPolicyError(f"{label} inventory path is invalid")
        declared[relative] = (
            _digest(entry["sha256"], f"{label} inventory sha256"),
            _integer(entry["byte_count"], f"{label} inventory byte count"),
        )
    if list(declared) != sorted(declared):
        raise TerminalPolicyError(f"{label} inventory is not canonically sorted")
    actual: set[str] = set()
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise TerminalPolicyError(f"{label} inventory contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TerminalPolicyError(f"{label} inventory contains an unsafe object")
        if path.name != "SEALED.json":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(declared):
        raise TerminalPolicyError(f"{label} inventory does not cover tree")
    for relative, (digest, size) in declared.items():
        path = root / relative
        info, actual_digest, byte_count, _payload = _regular_file(
            path,
            f"{label} inventory file {relative}",
        )
        if stat.S_IMODE(info.st_mode) != 0o444 or byte_count != size or actual_digest != digest:
            raise TerminalPolicyError(f"{label} inventory identity drift")
    return declared


def _safe_tree_files(root: Path, label: str) -> set[str]:
    try:
        if root.resolve(strict=True) != root:
            raise TerminalPolicyError(f"{label} root identity is invalid")
        root_info = root.lstat()
    except OSError as exc:
        raise TerminalPolicyError(f"{label} root is missing") from exc
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_ISLNK(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or root_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise TerminalPolicyError(f"{label} root is unsafe")
    files: set[str] = set()
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise TerminalPolicyError(f"{label} tree contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TerminalPolicyError(f"{label} tree contains an unsafe object")
        files.add(path.relative_to(root).as_posix())
    return files


def _validate_acquisition(records: AcquisitionRecords) -> tuple[ValidatedArtifact, ...]:
    report_root = Path(records.report_root.path)
    source_root = Path(records.source_root.path)
    source_files = _safe_tree_files(source_root, "source")
    report_files = _safe_tree_files(report_root, "source report")

    plan = _canonical_object(report_root / "plan.json")
    expected_plan = {
        "schema": "alpha_max_official_acquisition_plan.v3",
        "source_eligible": False,
        "symbols": list(_SYMBOLS),
        "months": [],
        "contract_sha256": records.contract_manifest.sha256,
        "availability_evidence_sha256": records.availability_evidence.sha256,
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
    source_owner = _canonical_object(source_root / ".alpha_max_owner.json")
    report_owner = _canonical_object(report_root / ".alpha_max_owner.json")
    _exact(source_owner, owner_fields, "source owner")
    _exact(report_owner, owner_fields, "source report owner")
    source_info = source_root.lstat()
    report_info = report_root.lstat()
    source_parent_info = source_root.parent.lstat()
    report_parent_info = report_root.parent.lstat()
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
    if (
        source_owner != report_owner
        or any(source_owner.get(key) != expected for key, expected in owner_expected.items())
        or not isinstance(source_owner["derivation_version"], str)
        or not source_owner["derivation_version"]
    ):
        raise TerminalPolicyError("source ownership binding mismatch")

    receipt = _canonical_object(report_root / "source_eligible_receipt.json")
    fields = {
        "schema",
        "source_eligible",
        "raw_rows",
        "funding_rows",
        "contract_sha256",
        "availability_evidence_sha256",
        "derivation_version",
        "code_sha256",
        "exchange_info_sha256",
        "inventory_sha256",
        "source_manifest_sha256",
        "acquisition_journal_sha256",
    }
    _exact(receipt, fields, "source receipt")
    if (
        receipt["schema"] != "alpha_max_official_source_receipt.v3"
        or receipt["source_eligible"] is not True
        or receipt["raw_rows"] != 1_066_681_730
        or receipt["funding_rows"] != 39_569
        or receipt["contract_sha256"] != records.contract_manifest.sha256
        or receipt["availability_evidence_sha256"] != records.availability_evidence.sha256
        or receipt["code_sha256"] != records.acquirer.sha256
        or receipt["derivation_version"] != source_owner["derivation_version"]
    ):
        raise TerminalPolicyError("source receipt binding mismatch")

    manifest_path = report_root / "source_manifest.json"
    journal_path = report_root / "acquisition.journal.jsonl"
    manifest = _canonical_object(manifest_path)
    _exact(
        manifest,
        {
            "schema",
            "contract_sha256",
            "availability_evidence_sha256",
            "derivation_version",
            "artifacts",
        },
        "source manifest",
    )
    if (
        manifest["schema"] != "alpha_max_official_source_manifest.v4"
        or manifest["contract_sha256"] != records.contract_manifest.sha256
        or manifest["availability_evidence_sha256"] != records.availability_evidence.sha256
        or manifest["derivation_version"] != receipt["derivation_version"]
        or not isinstance(manifest["artifacts"], list)
    ):
        raise TerminalPolicyError("source manifest binding mismatch")

    seen: set[str] = set()
    output_inventory: list[str] = []
    report_inventory: list[str] = []
    artifact_paths: list[str] = []
    exchange_info_sha256: str | None = None
    for entry in manifest["artifacts"]:
        _exact(entry, {"path", "sha256"}, "source manifest artifact")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or relative in seen
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not (relative.startswith("output/") or relative.startswith("report/"))
        ):
            raise TerminalPolicyError("source manifest path is invalid")
        seen.add(relative)
        artifact_paths.append(relative)
        prefix, leaf = relative.split("/", 1)
        target_root = source_root if prefix == "output" else report_root
        _info, digest, _byte_count, _payload = _regular_file(
            target_root / leaf,
            f"source manifest artifact {relative}",
        )
        if digest != _digest(entry["sha256"], "source artifact sha256"):
            raise TerminalPolicyError("source manifest artifact drift")
        if prefix == "output":
            output_inventory.append(leaf)
        else:
            report_inventory.append(leaf)
            if leaf == "provenance/exchangeInfo.json":
                exchange_info_sha256 = digest
    if artifact_paths != sorted(artifact_paths):
        raise TerminalPolicyError("source manifest is not canonically sorted")

    expected_source_files = set(output_inventory) | {".alpha_max_owner.json"}
    expected_report_files = set(report_inventory) | {
        ".alpha_max_owner.json",
        "plan.json",
        "source_manifest.json",
        "source_eligible_receipt.json",
    }
    if source_files != expected_source_files or report_files != expected_report_files:
        raise TerminalPolicyError("source manifest does not cover both trees")
    expected_inventory_sha256 = hashlib.sha256(
        canonical_bytes(sorted(output_inventory))
    ).hexdigest()
    if receipt["inventory_sha256"] != expected_inventory_sha256:
        raise TerminalPolicyError("source inventory hash mismatch")
    for key, path in (
        ("source_manifest_sha256", manifest_path),
        ("acquisition_journal_sha256", journal_path),
    ):
        _info, digest, _byte_count, _payload = _regular_file(path, key)
        if receipt[key] != digest:
            raise TerminalPolicyError("source receipt cross-hash mismatch")
    if exchange_info_sha256 is None or receipt["exchange_info_sha256"] != exchange_info_sha256:
        raise TerminalPolicyError("source exchange-info binding mismatch")
    return tuple(
        _artifact_from_path(str(report_root), name, kind)
        for name, kind in (
            ("source_eligible_receipt.json", "source_eligible_receipt"),
            ("source_manifest.json", "source_manifest"),
            ("acquisition.journal.jsonl", "source_journal"),
        )
    )


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
    roots: tuple[Path, ...] = ()
    if request.scope == "acquisition":
        record = request.records
        roots = (Path(record.source_root.path), Path(record.report_root.path))
        validated = _validate_acquisition(record)
    elif request.scope == "phase_preparation":
        record = request.records
        roots = (Path(record.phase_output.path),)
        _validate_preparation_manifest(request)
        validated = (
            _artifact_from_path(
                str(roots[0].parent),
                f".{roots[0].name}.alpha_max_phase_preparation.handoff.json",
                "phase_handoff_receipt",
            ),
            _artifact_from_path(str(roots[0]), "preparation_manifest.json", "preparation_manifest"),
        )
    else:
        record = request.records
        if not isinstance(record, OneTouchRecords):
            raise TerminalPolicyError("one-touch records are required")
        _validate_preparation_manifest(request)
        if index == 0:
            sealed = (_sealed_tree(record.prelock_output.path, "prelock_bundle", "artifacts"),)
            validated = (
                _artifact_from_path(
                    record.prelock_output.path,
                    "run/prelock_result.json",
                    "prelock_readback",
                ),
                _artifact_from_path(
                    record.prelock_output.path,
                    "diagnostics/validation/trend_liquidity_falsifier.json",
                    "prelock_observability",
                ),
                _artifact_from_path(
                    record.prelock_output.path,
                    "SEALED.json",
                    "prelock_inventory_before",
                ),
                _artifact_from_path(
                    record.prelock_output.path,
                    "admission/train_liquidity_buckets.json",
                    "input_inventory_before",
                ),
            )
            roots = (Path(record.phase_output.path), Path(record.prelock_output.path))
        else:
            validated, sealed = _one_touch_second_command_artifacts(request, index)
            roots = (
                Path(record.phase_output.path),
                Path(record.prelock_output.path),
                Path(record.historical_output.path),
            )
    root_snapshots = tuple(_snapshot_digest(root) for root in roots)
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
        and root_snapshots[:2] != prior_evidence.root_snapshot_sha256s[:2]
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


def _utc_value(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise TerminalPolicyError(f"invalid {label}")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise TerminalPolicyError(f"invalid {label}") from exc
    return parsed


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
    _digest(
        value["validated_artifact_snapshot_sha256"],
        "validated artifact snapshot sha256",
    )
    _utc_value(value["issued_utc"], "clearance timestamp")
    return value


def validate_scope_artifacts(
    envelope: LaunchEnvelope,
    request: ScopeRequest,
    events: Any,
    evidence: tuple[CommandEvidence, ...] | None = None,
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
            authorization_id = _digest(event["authorization_id"], "authorization id")
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
        _utc_value(event["observed_utc"], "event timestamp")
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
                _digest(event[digest_key], digest_key)
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
        stdout = _artifact_from_path(
            request.evidence_root.path,
            request.publication.stdout[index],
            "stdout",
        )
        stderr = _artifact_from_path(
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


def _artifact_from_path(parent: str, leaf: str, kind: str) -> ValidatedArtifact:
    path = Path(parent) / leaf
    info, digest, byte_count, _payload = _regular_file(path, f"{kind} artifact")
    if info.st_uid != os.getuid() or info.st_gid != os.getgid():
        raise TerminalPolicyError(f"{kind} artifact owner mismatch")
    return ValidatedArtifact(
        kind,
        str(path),
        digest,
        byte_count,
        info.st_dev,
        info.st_ino,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
    )


def _one_touch_second_command_artifacts(
    request: ScopeRequest, index: int
) -> tuple[tuple[ValidatedArtifact, ...], tuple[SealedArtifact, ...]]:
    if not isinstance(request.records, OneTouchRecords):
        raise TerminalPolicyError("one-touch records are required")
    if type(index) is not int or index != 1:
        raise TerminalPolicyError("one-touch second command index is invalid")
    record = request.records
    prelock = _sealed_tree(record.prelock_output.path, "prelock_bundle", "artifacts")
    historical = _sealed_tree(
        record.historical_output.path,
        "historical_bundle",
        "historical_artifacts",
    )
    historical_root = Path(record.historical_output.path)
    historical_seal = _canonical_object(historical_root / "SEALED.json")
    if historical_seal["prelock_seal_sha256"] != prelock.sha256 or historical_seal[
        "prelock_snapshot_sha256"
    ] != _snapshot_digest(Path(record.prelock_output.path)):
        raise TerminalPolicyError("historical prelock binding mismatch")
    before_seal = _regular_file(
        Path(record.prelock_output.path) / "SEALED.json",
        "prelock seal before historical",
    )
    after_seal = _regular_file(
        historical_root / "binding/prelock_seal.json",
        "prelock seal after historical",
    )
    before_inputs = _regular_file(
        Path(record.prelock_output.path) / "admission/train_liquidity_buckets.json",
        "prelock inputs before historical",
    )
    after_inputs = _regular_file(
        historical_root / "admission/train_liquidity_buckets.json",
        "prelock inputs after historical",
    )
    if before_seal[1:3] != after_seal[1:3] or before_inputs[1:3] != after_inputs[1:3]:
        raise TerminalPolicyError("historical immutable input comparison failed")
    validated = (
        _artifact_from_path(
            record.historical_output.path,
            "report/historical_result.json",
            "historical_readback",
        ),
        _artifact_from_path(
            record.historical_output.path,
            "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json",
            "historical_observability",
        ),
        _artifact_from_path(
            record.historical_output.path,
            "binding/prelock_seal.json",
            "prelock_inventory_after",
        ),
        _artifact_from_path(
            record.historical_output.path,
            "admission/train_liquidity_buckets.json",
            "input_inventory_after",
        ),
    )
    return validated, (historical,)


def _validate_preparation_manifest(request: ScopeRequest) -> None:
    if not isinstance(request.records, (PhaseRecords, OneTouchRecords)):
        raise TerminalPolicyError("phase preparation records are required")
    records = request.records
    output = Path(records.phase_output.path)
    if isinstance(records, OneTouchRecords):
        _read_directory(records.phase_output, "one-touch phase output")
    manifest = output / "preparation_manifest.json"
    value = _canonical_object(manifest)
    _exact(
        value,
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
        value["schema_version"] != "alpha_max_phase_root_preparation_manifest.v1"
        or value["exchange"] != "binance"
        or value["contract_manifest_sha256"] != records.contract_manifest.sha256
        or value["contract_manifest_schema_version"] != "alpha_max_contract_manifest.v2"
        or value["symbols"] != list(_SYMBOLS)
        or value["phase_intervals"] != expected_intervals
    ):
        raise TerminalPolicyError("preparation manifest semantic mismatch")

    availability = value["availability"]
    availability_hashes = value["availability_sha256_by_root_kind"]
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

    files = value["files"]
    if type(value["file_count"]) is not int or not isinstance(files, list):
        raise TerminalPolicyError("preparation file count is invalid")
    if value["file_count"] != len(files):
        raise TerminalPolicyError("preparation file count mismatch")
    phase_bounds = {phase: (start, end) for phase, start, end in _PHASE_INTERVALS}
    declared: list[str] = []
    source_file_facts: dict[str, tuple[str, int, int, int]] = {}
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
        if (
            not isinstance(relative, str)
            or not isinstance(source_relative, str)
            or not relative
            or not source_relative
            or relative.startswith("/")
            or source_relative.startswith("/")
            or ".." in Path(relative).parts
            or ".." in Path(source_relative).parts
            or phase not in phase_bounds
            or root_kind not in {"raw", "feature"}
            or symbol not in _SYMBOLS
            or not relative.startswith(f"{phase}/{root_kind}/")
        ):
            raise TerminalPolicyError("preparation manifest entry is invalid")
        phase_start, phase_end = phase_bounds[phase]
        available_start = availability[root_kind]["availability_start_by_symbol"][symbol]
        available_end = availability[root_kind]["availability_end_by_symbol"][symbol]
        expected_start = max(phase_start, available_start)
        expected_end = min(phase_end, available_end)
        if (
            expected_start >= expected_end
            or entry["owned_start_utc"] != expected_start
            or entry["owned_end_utc"] != expected_end
        ):
            raise TerminalPolicyError("preparation manifest ownership mismatch")
        output_path = output / relative
        output_info, output_digest, output_size, _output_payload = _regular_file(
            output_path,
            f"preparation output {relative}",
        )
        if (
            stat.S_IMODE(output_info.st_mode) != 0o444
            or output_size != _integer(entry["output_byte_count"], "preparation output byte count")
            or output_digest != _digest(entry["output_sha256"], "preparation output sha256")
        ):
            raise TerminalPolicyError("preparation output identity drift")
        if isinstance(records, PhaseRecords):
            source_path = Path(records.source_root.path) / source_relative
            source_info, source_digest, source_size, _source_payload = _regular_file(
                source_path,
                f"preparation source {source_relative}",
            )
            source_file_facts[source_relative] = (
                source_digest,
                source_size,
                source_info.st_dev,
                source_info.st_ino,
            )
            if source_size != _integer(
                entry["source_byte_count"], "preparation source byte count"
            ) or source_digest != _digest(entry["source_sha256"], "preparation source sha256"):
                raise TerminalPolicyError("preparation source identity drift")
        else:
            _integer(entry["source_byte_count"], "preparation source byte count")
            _digest(entry["source_sha256"], "preparation source sha256")
        _integer(entry["output_row_count"], "preparation output row count", positive=True)
        declared.append(relative)
    if declared != sorted(set(declared)):
        raise TerminalPolicyError("preparation manifest paths are not canonical")
    actual_files = _safe_tree_files(output, "phase output")
    top_level_directories = {
        path.name for path in output.iterdir() if stat.S_ISDIR(path.lstat().st_mode)
    }
    if top_level_directories != set(_PHASES):
        raise TerminalPolicyError("preparation phase tree mismatch")
    for path in (
        output,
        *(item for item in output.rglob("*") if stat.S_ISDIR(item.lstat().st_mode)),
    ):
        if stat.S_IMODE(path.lstat().st_mode) != 0o555:
            raise TerminalPolicyError("phase output directory is not immutable")
    if actual_files != set(declared) | {"preparation_manifest.json"}:
        raise TerminalPolicyError("preparation manifest does not cover output tree")

    handoff = output.parent / f".{output.name}.alpha_max_phase_preparation.handoff.json"
    receipt = _canonical_object(handoff)
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
    output_info = output.lstat()
    manifest_digest = _regular_file(manifest, "preparation manifest")[1]
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
        _digest(receipt[key], f"phase handoff {key}")
    if isinstance(records, OneTouchRecords):
        handoff_info = _regular_file(handoff, "phase handoff receipt")[0]
        manifest_info = _regular_file(manifest, "preparation manifest")[0]
        if (
            stat.S_IMODE(handoff_info.st_mode) != 0o444
            or stat.S_IMODE(manifest_info.st_mode) != 0o444
        ):
            raise TerminalPolicyError("phase receipts are not immutable")
        _read_identity(records.contract_manifest, "one-touch contract manifest")
        return
    source_root = Path(records.source_root.path)

    source_report = Path(records.source_report.path)
    source_manifest_path = source_report / "source_manifest.json"
    source_manifest = _canonical_object(source_manifest_path)
    _exact(
        source_manifest,
        {
            "schema",
            "contract_sha256",
            "availability_evidence_sha256",
            "derivation_version",
            "artifacts",
        },
        "phase source manifest",
    )
    if (
        source_manifest["schema"] != "alpha_max_official_source_manifest.v4"
        or source_manifest["contract_sha256"] != records.contract_manifest.sha256
        or source_manifest["availability_evidence_sha256"] != records.availability_evidence.sha256
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
        artifact_map[relative] = _digest(entry["sha256"], "phase source artifact sha256")
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
        "source_eligible_receipt_sha256": _regular_file(
            source_report / "source_eligible_receipt.json",
            "phase source eligible receipt",
        )[1],
        "source_manifest_sha256": _regular_file(
            source_manifest_path,
            "phase source manifest",
        )[1],
        "acquisition_journal_sha256": _regular_file(
            source_report / "acquisition.journal.jsonl",
            "phase source journal",
        )[1],
        "plan_sha256": _regular_file(
            source_report / "plan.json",
            "phase source plan",
        )[1],
        "source_owner_sha256": _regular_file(
            source_root / ".alpha_max_owner.json",
            "phase source owner",
        )[1],
        "report_owner_sha256": _regular_file(
            source_report / ".alpha_max_owner.json",
            "phase source report owner",
        )[1],
        "source_manifest_artifact_map_sha256": hashlib.sha256(
            canonical_bytes(artifact_map)
        ).hexdigest(),
    }
    if source_snapshot != expected_source_snapshot:
        raise TerminalPolicyError("phase source eligibility snapshot mismatch")

    base = output.parent / f".{output.name}.alpha_max_phase_preparation"
    descriptor_path = Path(f"{base}.invocation.json")
    descriptor_stage = Path(f"{base}.invocation.stage.json")
    handoff_stage = Path(f"{base}.handoff.stage.json")
    snapshot_root = Path(f"{base}.source-snapshot")
    invocation_inputs = Path(f"{base}.invocation-inputs")
    invocation_lock = Path(f"{base}.lock")
    snapshot_manifest_path = snapshot_root / "snapshot-manifest.json"
    snapshot_complete_path = snapshot_root / ".complete.json"
    descriptor = _canonical_object(descriptor_path)
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
    paths = descriptor["paths"]
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
        "invocation_descriptor_stage": str(descriptor_stage),
        "handoff_receipt": str(handoff),
        "handoff_receipt_stage": str(handoff_stage),
        "source_snapshot": str(snapshot_root),
        "source_snapshot_manifest": str(snapshot_manifest_path),
        "source_snapshot_complete": str(snapshot_complete_path),
        "invocation_inputs": str(invocation_inputs),
        "invocation_input_acquirer": str(invocation_inputs / "acquirer.py"),
        "invocation_input_contract_manifest": str(invocation_inputs / "contract_manifest.json"),
        "invocation_input_availability_evidence": str(
            invocation_inputs / "availability_evidence.json"
        ),
        "invocation_input_preparer": str(invocation_inputs / "preparer.py"),
        "invocation_lock": str(invocation_lock),
    }
    if paths != expected_paths:
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
        != _regular_file(descriptor_path, "phase invocation descriptor")[1]
    ):
        raise TerminalPolicyError("phase invocation argv mismatch")
    lock_info, _lock_digest, lock_size, _lock_payload = _regular_file(
        invocation_lock,
        "phase invocation lock",
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
            source_info, source_digest, source_size, _source_payload = _regular_file(
                source_root / source_relative,
                f"phase snapshot source {source_relative}",
            )
            facts = (
                source_digest,
                source_size,
                source_info.st_dev,
                source_info.st_ino,
            )
            source_file_facts[source_relative] = facts
        if facts[0] != digest:
            raise TerminalPolicyError("phase snapshot source digest mismatch")
        expected_snapshot_entries.append(
            {
                "source_relative_path": source_relative,
                "sha256": digest,
                "byte_count": facts[1],
            }
        )
    expected_snapshot_entries.sort(key=lambda item: item["source_relative_path"])
    snapshot_manifest = _canonical_object(snapshot_manifest_path)
    expected_snapshot_manifest = {
        "schema": "alpha_max_phase_preparation_source_snapshot.v1",
        "descriptor_sha256": receipt["invocation_descriptor_sha256"],
        "source_manifest_sha256": expected_source_snapshot["source_manifest_sha256"],
        "entries": expected_snapshot_entries,
    }
    if (
        snapshot_manifest != expected_snapshot_manifest
        or receipt["source_snapshot_manifest_sha256"]
        != _regular_file(snapshot_manifest_path, "phase snapshot manifest")[1]
    ):
        raise TerminalPolicyError("phase snapshot manifest mismatch")
    snapshot_info = snapshot_root.lstat()
    if receipt["source_snapshot_identity"] != {
        "st_dev": snapshot_info.st_dev,
        "st_ino": snapshot_info.st_ino,
    }:
        raise TerminalPolicyError("phase snapshot root identity mismatch")
    expected_complete = {
        "schema": "alpha_max_phase_preparation_source_snapshot.v1",
        "snapshot_manifest_sha256": receipt["source_snapshot_manifest_sha256"],
    }
    if _canonical_object(snapshot_complete_path) != expected_complete:
        raise TerminalPolicyError("phase snapshot completion marker mismatch")
    snapshot_files = _safe_tree_files(snapshot_root, "phase source snapshot")
    expected_snapshot_files = {
        item["source_relative_path"] for item in expected_snapshot_entries
    } | {"snapshot-manifest.json", ".complete.json"}
    if snapshot_files != expected_snapshot_files:
        raise TerminalPolicyError("phase snapshot inventory mismatch")
    for path in (
        snapshot_root,
        *(item for item in snapshot_root.rglob("*") if stat.S_ISDIR(item.lstat().st_mode)),
    ):
        if stat.S_IMODE(path.lstat().st_mode) != 0o555:
            raise TerminalPolicyError("phase snapshot directory is not immutable")
    for entry in expected_snapshot_entries:
        relative = entry["source_relative_path"]
        info, digest, byte_count, _payload = _regular_file(
            snapshot_root / relative,
            f"phase snapshot clone {relative}",
        )
        source_facts = source_file_facts[relative]
        if (
            stat.S_IMODE(info.st_mode) != 0o444
            or (info.st_dev, info.st_ino) == (source_facts[2], source_facts[3])
            or digest != entry["sha256"]
            or byte_count != entry["byte_count"]
        ):
            raise TerminalPolicyError("phase snapshot clone mismatch")
    handoff_info = _regular_file(handoff, "phase handoff receipt")[0]
    if (
        stat.S_IMODE(handoff_info.st_mode) != 0o444
        or stat.S_IMODE(_regular_file(manifest, "preparation manifest")[0].st_mode) != 0o444
    ):
        raise TerminalPolicyError("phase receipts are not immutable")


def _selection_payload(directory: Path, relative: str, *, role: str) -> dict[str, Any]:
    value = _canonical_object(directory / relative)
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


def _terminal_payload(directory: Path, relative: str, label: str) -> dict[str, Any]:
    value = _canonical_object(directory / relative)
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
    directory: Path,
    *,
    domain: str,
    physical_fold_run_count: int,
    diagnostic_relative: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _canonical_object(directory / "status/matrix.json")
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
    diagnostic = _canonical_object(directory / diagnostic_relative)
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
    bucket_digest = _regular_file(
        directory / "admission/train_liquidity_buckets.json",
        f"{domain} liquidity buckets",
    )[1]
    if (
        diagnostic["artifact_kind"] != "alpha_max_trend_liquidity_falsifier.v1"
        or diagnostic["domain"] != domain
        or diagnostic["nominal_cost_bps"] != 30
        or diagnostic["report_only"] is not True
        or diagnostic["selection_influence"] is not False
        or diagnostic["row_id"] != "component_trend_1x"
        or not isinstance(diagnostic["fold_run_sha256s"], list)
        or len(diagnostic["fold_run_sha256s"]) != expected_fold_count
        or any(
            not isinstance(item, str) or not _DIGEST.fullmatch(item)
            for item in diagnostic["fold_run_sha256s"]
        )
        or diagnostic["train_liquidity_buckets_sha256"] != bucket_digest
    ):
        raise TerminalPolicyError(f"{domain} observability mismatch")
    return matrix, diagnostic


def _semantic_bundle_readback(directory: Path, kind: str, seal: dict[str, Any]) -> str:
    if kind == "prelock_bundle":
        run = _canonical_object(directory / "run/prelock_result.json")
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
            directory,
            "selection/prelock.json",
            role="prelock_selection",
        )
        terminal = _terminal_payload(directory, "terminal/prelock.json", "prelock terminal")
        matrix, diagnostic = _matrix_and_observability(
            directory,
            domain="validation",
            physical_fold_run_count=816,
            diagnostic_relative="diagnostics/validation/trend_liquidity_falsifier.json",
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
        report = _canonical_object(directory / "report/historical_result.json")
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
            directory,
            "selection/historical_ranking.json",
            role="historical_report",
        )
        terminal = _terminal_payload(
            directory,
            "terminal/historical.json",
            "historical terminal",
        )
        matrix, diagnostic = _matrix_and_observability(
            directory,
            domain="historical_exposed_evaluation",
            physical_fold_run_count=680,
            diagnostic_relative=(
                "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json"
            ),
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


def _sealed_tree(root: str, kind: str, inventory_key: str) -> SealedArtifact:
    directory = Path(root)
    sealed_path = directory / "SEALED.json"
    value = _canonical_object(sealed_path)
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
    declared = _inventory(directory, entries, kind)
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
        _digest(value["prelock_seal_sha256"], "historical prelock seal sha256")
        _digest(value["prelock_snapshot_sha256"], "historical prelock snapshot sha256")
    required_counts = (
        (
            ("manifests/validation_train_fit/*.json", 17),
            ("manifests/prelock_final_refit/*.json", 17),
            ("capsules/validation_train_fit/*/*.json", 204),
            ("capsules/prelock_final_refit/*/*.json", 17),
            ("evidence/validation/cells/*/*.json", 68),
            ("evidence/validation/rows/*.json", 816),
        )
        if kind == "prelock_bundle"
        else (
            ("capsules/prelock_final_refit/*/*.json", 153),
            ("evidence/historical_exposed_evaluation/cells/*/*.json", 68),
            ("evidence/historical_exposed_evaluation/rows/*.json", 680),
        )
    )
    if any(len(list(directory.glob(pattern))) != count for pattern, count in required_counts):
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
    for path in (directory, *directory.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) != 0o555
        ):
            raise TerminalPolicyError("unsafe sealed tree identity")
    inventory_sha256 = hashlib.sha256(canonical_bytes(entries)).hexdigest()
    if kind == "prelock_bundle" and value["inventory_sha256"] != inventory_sha256:
        raise TerminalPolicyError("prelock inventory hash mismatch")
    readback_sha256 = _semantic_bundle_readback(directory, kind, value)
    info, digest, byte_count, _payload = _regular_file(sealed_path, "sealed receipt")
    if stat.S_IMODE(info.st_mode) != 0o444:
        raise TerminalPolicyError("sealed receipt identity drift")
    return SealedArtifact(
        kind,
        str(sealed_path),
        digest,
        byte_count,
        info.st_dev,
        info.st_ino,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
        digest,
        inventory_sha256,
        readback_sha256,
    )


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
