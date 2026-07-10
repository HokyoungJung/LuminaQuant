from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from lumina_quant.utils import artifact_read_receipt as receipts
from lumina_quant.utils.artifact_read_receipt import read_artifact_bytes, read_artifact_json


def test_read_artifact_bytes_returns_descriptor_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"ok": true}\n')
    original_open = receipts.os.open
    original_read = receipts.os.read
    original_fstat = receipts.os.fstat
    original_close = receipts.os.close
    calls: dict[str, list[int | str]] = {"open": [], "read": [], "fstat": [], "close": []}

    def _open(path, flags):
        fd = original_open(path, flags)
        calls["open"].append(fd)
        return fd

    def _read(fd, size):
        calls["read"].append(fd)
        return original_read(fd, size)

    def _fstat(fd):
        calls["fstat"].append(fd)
        return original_fstat(fd)

    def _close(fd):
        calls["close"].append(fd)
        return original_close(fd)

    monkeypatch.setattr(receipts.os, "open", _open)
    monkeypatch.setattr(receipts.os, "read", _read)
    monkeypatch.setattr(receipts.os, "fstat", _fstat)
    monkeypatch.setattr(receipts.os, "close", _close)

    receipt, payload = read_artifact_bytes(artifact, artifact_id="artifact-a")

    assert payload == b'{"ok": true}\n'
    assert calls["open"]
    fd = calls["open"][0]
    assert calls["open"] == [fd]
    assert calls["fstat"] == [fd, fd]
    assert calls["read"] and all(read_fd == fd for read_fd in calls["read"])
    assert calls["close"] == [fd]
    assert receipt.artifact_id == "artifact-a"
    assert receipt.requested_path == str(artifact.resolve())
    assert receipt.canonical_path == str(artifact.resolve())
    assert receipt.byte_count == len(payload)
    assert receipt.pre_fstat_identity == receipt.post_fstat_identity
    assert stat.S_ISREG(receipt.pre_fstat_identity[2])


def test_read_artifact_json_parses_only_descriptor_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"payload": "ok"}), encoding="utf-8")

    receipt, parsed = read_artifact_json(artifact, artifact_id="artifact-json")

    assert parsed == {"payload": "ok"}
    assert receipt.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_read_artifact_rejects_final_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(artifact)

    with pytest.raises(ValueError, match="artifact_symlink_rejected"):
        read_artifact_bytes(link, artifact_id="artifact-link")


def test_read_artifact_rejects_multilink_target(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(artifact)

    with pytest.raises(ValueError, match="artifact_link_count_rejected"):
        read_artifact_bytes(artifact, artifact_id="artifact-hardlink")


def test_read_artifact_rejects_descriptor_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    original = receipts._descriptor_identity
    calls = 0

    def _changing_identity(st):
        nonlocal calls
        calls += 1
        identity = original(st)
        if calls == 2:
            return (*identity[:-1], identity[-1] + 1)
        return identity

    monkeypatch.setattr(receipts, "_descriptor_identity", _changing_identity)

    with pytest.raises(ValueError, match="artifact_descriptor_identity_changed"):
        read_artifact_bytes(artifact, artifact_id="artifact-changing")
