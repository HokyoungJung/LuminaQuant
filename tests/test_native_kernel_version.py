from __future__ import annotations

import logging

import lumina_quant._native_kernel_version as native_kernel_version
from lumina_quant._native_kernel_version import (
    SRC_HASH_ENV_VAR,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_STALE,
    check_native_kernel_version,
    compare_kernel_source_hash,
    compare_native_version,
    compute_src_hash,
)


def test_compare_native_version_matching_reports_ok():
    assert compare_native_version("0.1.1", "0.1.1") == STATUS_OK


def test_compare_native_version_mismatch_reports_stale():
    assert compare_native_version("0.1.1", "0.1.0") == STATUS_STALE


def test_compare_native_version_no_reported_version_is_missing():
    assert compare_native_version("0.1.1", None) == STATUS_MISSING
    assert compare_native_version("0.1.1", "") == STATUS_MISSING


def test_compare_native_version_no_expected_version_is_ok():
    # Nothing to compare against (e.g. expected version could not be
    # determined) -- treat as fine rather than warn.
    assert compare_native_version(None, "0.1.1") == STATUS_OK
    assert compare_native_version("", "0.1.1") == STATUS_OK


def test_compare_native_version_tolerates_surrounding_whitespace():
    assert compare_native_version(" 0.1.1 ", "0.1.1\n") == STATUS_OK


def test_check_native_kernel_version_never_raises(monkeypatch, caplog):
    monkeypatch.setattr(native_kernel_version, "_checked", False)

    class _StaleModule:
        @staticmethod
        def build_info() -> str:
            return "0.0.1-definitely-not-the-crate-version"

    # Should not raise even if the reported version disagrees with the crate.
    check_native_kernel_version(_StaleModule())


def test_check_native_kernel_version_is_idempotent(monkeypatch):
    monkeypatch.setattr(native_kernel_version, "_checked", False)

    calls = []

    class _TrackedModule:
        @staticmethod
        def build_info() -> str:
            calls.append(1)
            return "0.1.1"

    check_native_kernel_version(_TrackedModule())
    check_native_kernel_version(_TrackedModule())
    assert len(calls) <= 1


def test_check_native_kernel_version_tolerates_missing_build_info(monkeypatch):
    monkeypatch.setattr(native_kernel_version, "_checked", False)

    class _LegacyModule:
        pass

    # No build_info() attribute at all -- must not raise.
    check_native_kernel_version(_LegacyModule())


# --------------------------------------------------------------------------- #
# Source-hash handshake (finding N3)
# --------------------------------------------------------------------------- #


def test_src_hash_env_var_name_is_stable():
    # The build script and the runtime must agree on this exact name.
    assert SRC_HASH_ENV_VAR == "LUMINA_KERNEL_SRC_HASH"


def test_compute_src_hash_is_deterministic_and_content_sensitive(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text("fn main() {}\n", encoding="utf-8")
    (src / "helpers.rs").write_text("// helpers\n", encoding="utf-8")

    first = compute_src_hash(src)
    assert first is not None
    assert first == compute_src_hash(src)  # deterministic

    # Editing any .rs file changes the hash (this is the N3 scenario).
    (src / "lib.rs").write_text("fn main() { let _x = 1; }\n", encoding="utf-8")
    assert compute_src_hash(src) != first

    # Adding a new .rs file also changes the hash.
    after_edit = compute_src_hash(src)
    (src / "extra.rs").write_text("// extra\n", encoding="utf-8")
    assert compute_src_hash(src) != after_edit


def test_compute_src_hash_missing_dir_returns_none(tmp_path):
    assert compute_src_hash(tmp_path / "nope") is None


def test_compute_src_hash_empty_dir_returns_none(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert compute_src_hash(empty) is None  # no *.rs files


def test_compute_src_hash_real_crate_tree_is_hex16():
    # The crate source is checked out in this repo, so the default path hashes.
    digest = compute_src_hash()
    assert digest is not None
    assert len(digest) == 16
    int(digest, 16)  # valid hex


def test_compare_kernel_source_hash_matching_ok():
    assert compare_kernel_source_hash("abc123", "abc123") == STATUS_OK
    assert compare_kernel_source_hash(" abc ", "abc\n") == STATUS_OK


def test_compare_kernel_source_hash_mismatch_stale():
    assert compare_kernel_source_hash("abc123", "def456") == STATUS_STALE


def test_compare_kernel_source_hash_absent_side_is_ok():
    # An absent embedded hash (older .so / plain maturin) is not a failure.
    assert compare_kernel_source_hash("abc", None) == STATUS_OK
    assert compare_kernel_source_hash("abc", "") == STATUS_OK
    assert compare_kernel_source_hash(None, "abc") == STATUS_OK


def test_run_check_flags_stale_source_when_hash_drifts(monkeypatch, caplog):
    monkeypatch.setattr(native_kernel_version, "_checked", False)
    monkeypatch.setattr(native_kernel_version, "_read_expected_version", lambda: "0.1.1")
    monkeypatch.setattr(
        native_kernel_version, "compute_src_hash", lambda src_dir=None: "newhash000000001"
    )

    class _EditedNotRebuilt:
        @staticmethod
        def build_info() -> str:
            return "0.1.1"  # version unchanged -- version compare passes

        @staticmethod
        def kernel_src_hash() -> str:
            return "oldhash000000000"  # embedded hash from the previous build

    with caplog.at_level(logging.WARNING, logger="lumina_quant._native_kernel_version"):
        check_native_kernel_version(_EditedNotRebuilt())

    stale = [r for r in caplog.records if "stale native kernel" in r.getMessage()]
    assert stale, "expected a stale-source warning"
    assert any("source hash" in r.getMessage() for r in stale)


def test_run_check_no_warning_when_hash_matches(monkeypatch, caplog):
    monkeypatch.setattr(native_kernel_version, "_checked", False)
    monkeypatch.setattr(native_kernel_version, "_read_expected_version", lambda: "0.1.1")
    monkeypatch.setattr(
        native_kernel_version, "compute_src_hash", lambda src_dir=None: "matching00000000"
    )

    class _FreshlyBuilt:
        @staticmethod
        def build_info() -> str:
            return "0.1.1"

        @staticmethod
        def kernel_src_hash() -> str:
            return "matching00000000"

    with caplog.at_level(logging.WARNING, logger="lumina_quant._native_kernel_version"):
        check_native_kernel_version(_FreshlyBuilt())

    assert not [r for r in caplog.records if "stale" in r.getMessage()]


def test_run_check_degrades_silently_without_embedded_hash(monkeypatch, caplog):
    monkeypatch.setattr(native_kernel_version, "_checked", False)
    monkeypatch.setattr(native_kernel_version, "_read_expected_version", lambda: "0.1.1")
    # Source has drifted, but the loaded .so predates kernel_src_hash().
    monkeypatch.setattr(
        native_kernel_version, "compute_src_hash", lambda src_dir=None: "newhash000000001"
    )

    class _OlderSo:
        @staticmethod
        def build_info() -> str:
            return "0.1.1"

        # No kernel_src_hash attribute at all.

    with caplog.at_level(logging.WARNING, logger="lumina_quant._native_kernel_version"):
        check_native_kernel_version(_OlderSo())

    # Nothing to compare -> no false stale warning (graceful degrade).
    assert not [r for r in caplog.records if "stale" in r.getMessage()]


def test_run_check_degrades_silently_on_empty_embedded_hash(monkeypatch, caplog):
    monkeypatch.setattr(native_kernel_version, "_checked", False)
    monkeypatch.setattr(native_kernel_version, "_read_expected_version", lambda: "0.1.1")
    monkeypatch.setattr(
        native_kernel_version, "compute_src_hash", lambda src_dir=None: "newhash000000001"
    )

    class _PlainMaturinBuild:
        @staticmethod
        def build_info() -> str:
            return "0.1.1"

        @staticmethod
        def kernel_src_hash() -> str:
            return ""  # built without scripts/build_native_backends.py

    with caplog.at_level(logging.WARNING, logger="lumina_quant._native_kernel_version"):
        check_native_kernel_version(_PlainMaturinBuild())

    assert not [r for r in caplog.records if "stale" in r.getMessage()]


def test_run_check_version_stale_short_circuits_hash_check(monkeypatch, caplog):
    monkeypatch.setattr(native_kernel_version, "_checked", False)
    monkeypatch.setattr(native_kernel_version, "_read_expected_version", lambda: "0.2.0")

    def _boom(src_dir=None):
        raise AssertionError("hash check must not run when the version is stale")

    monkeypatch.setattr(native_kernel_version, "compute_src_hash", _boom)

    class _WrongVersion:
        @staticmethod
        def build_info() -> str:
            return "0.1.1"

        @staticmethod
        def kernel_src_hash() -> str:
            return "whatever00000000"

    with caplog.at_level(logging.WARNING, logger="lumina_quant._native_kernel_version"):
        check_native_kernel_version(_WrongVersion())

    assert [r for r in caplog.records if "Cargo.toml is at 0.2.0" in r.getMessage()]


def test_reported_src_hash_tolerates_raising_getter():
    class _Raises:
        @staticmethod
        def kernel_src_hash() -> str:
            raise RuntimeError("boom")

    assert native_kernel_version._reported_src_hash(_Raises()) is None


def test_reported_src_hash_none_when_absent_or_empty():
    class _Absent:
        pass

    class _Empty:
        @staticmethod
        def kernel_src_hash() -> str:
            return ""

    assert native_kernel_version._reported_src_hash(_Absent()) is None
    assert native_kernel_version._reported_src_hash(_Empty()) is None
