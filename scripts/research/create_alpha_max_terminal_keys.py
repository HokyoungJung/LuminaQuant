#!/usr/bin/env python3
"""Create the terminal Ed25519 keys in a securely opened key directory."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lumina_quant import alpha_max_terminal_policy as policy

KEY_NAMES = (
    "authority",
    "acquisition",
    "phase_preparation",
    "one_touch",
)
FILE_NAMES = tuple(f"{key_name}.{kind}" for key_name in KEY_NAMES for kind in ("private", "public"))
ROOT_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("key root must be absolute")
    try:
        return Path(policy.validate_lexical_control_path(path))
    except policy.TerminalPolicyError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _raise_with_cleanup(
    primary: BaseException, label: str, cleanup_errors: list[BaseException]
) -> None:
    if cleanup_errors:
        cleanup = BaseExceptionGroup(f"{label} cleanup failed", cleanup_errors)
        raise BaseExceptionGroup(
            f"{label} failed and cleanup failed", [primary, cleanup]
        ) from primary
    raise primary


def _open_secure_root(root: Path) -> int:
    root_fd = policy.open_directory_fd(root, "key root")
    try:
        info = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_nlink != 2
        ):
            raise ValueError(
                "key root must be a leader-owned 0700 directory with no subdirectories"
            )
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(root_fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, "key root open", cleanup_errors)
    return root_fd


def _validate_key_file(info: os.stat_result) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise ValueError("new key file is unsafe")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        wrote = os.write(fd, view)
        if wrote <= 0:
            raise OSError("short key write")
        view = view[wrote:]


def _preflight(root_fd: int) -> None:
    existing = os.listdir(root_fd)
    if existing:
        raise FileExistsError("key root must be empty before key creation")


def _cleanup(root_fd: int, created: list[tuple[str, int, int]]) -> None:
    cleanup_errors: list[BaseException] = []
    for name, device, inode in reversed(created):
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != (device, inode):
                raise ValueError(f"created key file identity changed during cleanup: {name}")
            os.unlink(name, dir_fd=root_fd)
        except BaseException as error:
            cleanup_errors.append(error)
    try:
        os.fsync(root_fd)
    except BaseException as error:
        cleanup_errors.append(error)
    if cleanup_errors:
        raise BaseExceptionGroup("key cleanup failed", cleanup_errors)


def _key_material() -> tuple[tuple[str, bytes, bytes], ...]:
    material = []
    for key_name in KEY_NAMES:
        private = Ed25519PrivateKey.generate()
        secret = private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        public = private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        if len(secret) != 32 or len(public) != 32:
            raise ValueError("Ed25519 raw key length is invalid")
        material.append((key_name, secret, public))
    return tuple(material)


def _create_file(root_fd: int, name: str, data: bytes, created: list[tuple[str, int, int]]) -> None:
    fd = os.open(name, FILE_FLAGS, 0o400, dir_fd=root_fd)
    identity: tuple[int, int] | None = None
    try:
        info = os.fstat(fd)
        identity = (info.st_dev, info.st_ino)
        created.append((name, *identity))
        _validate_key_file(info)
        _write_all(fd, data)
        os.fsync(fd)
        _validate_key_file(os.fstat(fd))
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if identity is None:
            try:
                info = os.fstat(fd)
                identity = (info.st_dev, info.st_ino)
                created.append((name, *identity))
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                cleanup_errors.append(
                    ValueError(f"cannot verify created key file identity: {name}")
                )
        try:
            os.close(fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, f"key file creation: {name}", cleanup_errors)
    else:
        os.close(fd)


def create_keys(root: Path) -> dict[str, dict[str, str]]:
    root_fd = _open_secure_root(root)
    created: list[tuple[str, int, int]] = []
    result: dict[str, dict[str, str]] | None = None
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        _preflight(root_fd)
        material = _key_material()
        for key_name, secret, public in material:
            _create_file(root_fd, f"{key_name}.private", secret, created)
            _create_file(root_fd, f"{key_name}.public", public, created)
        os.fsync(root_fd)
        result = {
            key_name: {
                "key_id": hashlib.sha256(public).hexdigest(),
                "public_key_b64": base64.b64encode(public).decode("ascii"),
                "public_key_sha256": hashlib.sha256(public).hexdigest(),
            }
            for key_name, _secret, public in material
        }
    except BaseException as creation_error:
        primary = creation_error
        try:
            _cleanup(root_fd, created)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
    try:
        os.close(root_fd)
    except BaseException as cleanup_error:
        cleanup_errors.append(cleanup_error)
    if primary is not None:
        _raise_with_cleanup(primary, "key creation", cleanup_errors)
    if cleanup_errors:
        raise BaseExceptionGroup("key creation cleanup failed", cleanup_errors)
    if result is None:
        raise AssertionError("key creation produced no result")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-root", required=True, type=_absolute)
    args = parser.parse_args(argv)
    summary = create_keys(args.key_root)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
