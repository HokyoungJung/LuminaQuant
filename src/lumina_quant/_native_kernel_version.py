"""Version handshake between native/lumina_compute crate source and the loaded
lumina_quant._compute extension module.

Without this, a stale .so (crate source edited/rebuilt but the extension not
reloaded, or vice versa) drifts silently: the extension just keeps running
whatever kernel it was compiled with. This module compares the crate version
declared in the checked-out native/lumina_compute/Cargo.toml against the
version the loaded extension reports via build_info(), and logs one warning
on a mismatch. It never raises -- the native path stays fully optional, and
installed-wheel deployments (no checked-out crate source) skip the check
silently.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_MISSING = "missing"

_CARGO_TOML_PATH = Path(__file__).resolve().parents[2] / "native" / "lumina_compute" / "Cargo.toml"
_VERSION_RE = re.compile(r'(?m)^\s*version\s*=\s*"([^"]+)"')

_checked = False


def compare_native_version(expected: str | None, reported: str | None) -> str:
    """Pure comparison: crate-source version vs. the extension's build_info().

    Returns STATUS_MISSING when the extension reported no version (an older
    .so that predates this handshake), STATUS_OK when there is nothing to
    compare against or the versions agree, and STATUS_STALE when they
    disagree.
    """
    if not reported:
        return STATUS_MISSING
    if not expected:
        return STATUS_OK
    return STATUS_OK if reported.strip() == expected.strip() else STATUS_STALE


def _read_expected_version() -> str | None:
    try:
        text = _CARGO_TOML_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def _run_check(compute_module: Any) -> None:
    expected = _read_expected_version()
    if expected is None:
        # No checked-out crate source (e.g. installed wheel) -- nothing to verify.
        return

    if compute_module is None:
        from lumina_quant import _compute as compute_module  # type: ignore[no-redef]

    build_info = getattr(compute_module, "build_info", None)
    reported = build_info() if callable(build_info) else None

    status = compare_native_version(expected, reported)
    if status == STATUS_STALE:
        logger.warning(
            "stale native kernel: lumina_quant._compute reports build %s but "
            "native/lumina_compute/Cargo.toml is at %s; rebuild with "
            "'maturin develop --release' (or scripts/build_native_backends.py) "
            "in native/lumina_compute.",
            reported,
            expected,
        )
    elif status == STATUS_MISSING:
        logger.warning(
            "lumina_quant._compute has no build_info(): the loaded kernel "
            "predates the native version handshake (native/lumina_compute/"
            "Cargo.toml is at %s); rebuild with 'maturin develop --release' "
            "to pick up the handshake.",
            expected,
        )


def check_native_kernel_version(compute_module: Any = None) -> None:
    """Warn at most once per process if the loaded _compute kernel looks stale.

    Safe to call from every native_*_backend module that successfully imports
    lumina_quant._compute -- the check itself is idempotent and never raises.
    """
    global _checked
    if _checked:
        return
    _checked = True
    try:
        _run_check(compute_module)
    except Exception:
        logger.debug("native kernel version handshake failed to run", exc_info=True)


__all__ = [
    "STATUS_MISSING",
    "STATUS_OK",
    "STATUS_STALE",
    "check_native_kernel_version",
    "compare_native_version",
]
