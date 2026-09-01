from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DescriptorIdentity = tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class ArtifactReadReceipt:
    artifact_id: str
    requested_path: str
    canonical_path: str
    sha256: str
    byte_count: int
    pre_fstat_identity: DescriptorIdentity
    post_fstat_identity: DescriptorIdentity


def _descriptor_identity(st: os.stat_result) -> DescriptorIdentity:
    return (
        int(st.st_dev),
        int(st.st_ino),
        int(stat.S_IFMT(st.st_mode)),
        int(st.st_nlink),
        int(st.st_size),
        int(st.st_mtime_ns),
        int(st.st_ctime_ns),
    )


def _normalized_requested_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.fspath(Path(path).expanduser()))


def read_artifact_bytes(
    path: str | os.PathLike[str], *, artifact_id: str
) -> tuple[ArtifactReadReceipt, bytes]:
    """Read one regular artifact through a single descriptor and return its receipt.

    The helper intentionally hashes the exact bytes read from the opened file
    descriptor.  It rejects symlinks, non-regular files, hard-linked files, and
    any descriptor identity drift between the pre-read and post-read ``fstat``.
    """
    artifact_token = str(artifact_id or "").strip()
    if not artifact_token:
        raise ValueError("artifact_id is required")
    requested_path = _normalized_requested_path(path)
    canonical_path = str(Path(requested_path).resolve(strict=True))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)

    try:
        fd = os.open(requested_path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"artifact_symlink_rejected:{requested_path}") from exc
        raise

    try:
        pre_stat = os.fstat(fd)
        pre_identity = _descriptor_identity(pre_stat)
        if not stat.S_ISREG(pre_stat.st_mode):
            raise ValueError(f"artifact_not_regular:{requested_path}")
        if int(pre_stat.st_nlink) != 1:
            raise ValueError(f"artifact_link_count_rejected:{requested_path}")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)

        post_stat = os.fstat(fd)
        post_identity = _descriptor_identity(post_stat)
        if pre_identity != post_identity:
            raise ValueError(f"artifact_descriptor_identity_changed:{requested_path}")
    finally:
        os.close(fd)

    receipt = ArtifactReadReceipt(
        artifact_id=artifact_token,
        requested_path=requested_path,
        canonical_path=canonical_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        pre_fstat_identity=pre_identity,
        post_fstat_identity=post_identity,
    )
    return receipt, payload


def read_artifact_json(
    path: str | os.PathLike[str], *, artifact_id: str
) -> tuple[ArtifactReadReceipt, dict[str, Any]]:
    receipt, payload = read_artifact_bytes(path, artifact_id=artifact_id)
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object in {receipt.requested_path}")
    return receipt, parsed
