from __future__ import annotations

import lumina_quant._native_kernel_version as native_kernel_version
from lumina_quant._native_kernel_version import (
    STATUS_MISSING,
    STATUS_OK,
    STATUS_STALE,
    check_native_kernel_version,
    compare_native_version,
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
