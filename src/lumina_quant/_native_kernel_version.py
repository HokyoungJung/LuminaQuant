"""Version + source-hash handshake between the native/lumina_compute crate
source and the loaded lumina_quant._compute extension module.

Without this, a stale .so (crate source edited/rebuilt but the extension not
reloaded, or vice versa) drifts silently: the extension just keeps running
whatever kernel it was compiled with. This module runs two checks and logs at
most one warning on a mismatch. It never raises -- the native path stays fully
optional, and installed-wheel deployments (no checked-out crate source) skip
the check silently.

1. Version compare -- the crate version declared in the checked-out
   native/lumina_compute/Cargo.toml vs. the version the loaded extension
   reports via build_info().

2. Source-hash handshake (finding N3) -- a deterministic content hash of the
   checked-out native/lumina_compute/src/*.rs vs. the hash embedded into the
   extension at build time by scripts/build_native_backends.py (exposed via
   the optional kernel_src_hash() pyfunction). This catches the scenario the
   version compare cannot: lib.rs edited, CARGO_PKG_VERSION *not* bumped, and
   the .so not rebuilt -- build_info() still equals Cargo.toml, but the source
   hash has drifted. The embedded hash is optional: an older .so, or one built
   with a plain ``maturin develop`` (which does not set LUMINA_KERNEL_SRC_HASH),
   exposes no kernel_src_hash() or an empty one -- in that case we degrade
   silently to the version-only compare above.

Convention: because the embedded hash is only present when the build runs
through scripts/build_native_backends.py, kernel changes should ALSO bump the
crate version in native/lumina_compute/Cargo.toml so the version compare stays
a second line of defence for plain-maturin builds.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_MISSING = "missing"

_CRATE_DIR = Path(__file__).resolve().parents[2] / "native" / "lumina_compute"
_CARGO_TOML_PATH = _CRATE_DIR / "Cargo.toml"
_SRC_DIR = _CRATE_DIR / "src"
_VERSION_RE = re.compile(r'(?m)^\s*version\s*=\s*"([^"]+)"')

# Environment variable that scripts/build_native_backends.py sets so the crate
# can embed the source hash at compile time via option_env!. Kept here as the
# single source of truth shared between the build script and this runtime side.
SRC_HASH_ENV_VAR = "LUMINA_KERNEL_SRC_HASH"

_checked = False


def compute_src_hash(src_dir: Path | None = None) -> str | None:
    """Deterministic content hash of the crate's Rust source tree.

    Returns a short hex digest that changes whenever any ``*.rs`` file under
    ``src_dir`` is added, removed, or edited. Returns ``None`` when the source
    tree is absent (an installed wheel with no checked-out crate) or unreadable.

    This exact function is imported by scripts/build_native_backends.py to embed
    the hash into the compiled extension, guaranteeing the runtime side computes
    it identically -- there is no second implementation to drift.
    """
    root = src_dir if src_dir is not None else _SRC_DIR
    try:
        files = sorted(p for p in root.rglob("*.rs") if p.is_file())
    except OSError:
        return None
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        rel = path.relative_to(root).as_posix()
        # Length-prefix path + content so no file boundary is ambiguous.
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()[:16]


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


def compare_kernel_source_hash(expected: str | None, reported: str | None) -> str:
    """Pure comparison: checked-out crate source hash vs. the hash embedded in
    the loaded extension (via kernel_src_hash()).

    Unlike compare_native_version, an *absent* embedded hash is NOT a problem:
    an older .so or a plain ``maturin develop`` build carries no source hash, so
    there is simply nothing to compare -- returns STATUS_OK. STATUS_STALE is
    returned only when both hashes are present and disagree.
    """
    if not reported or not expected:
        return STATUS_OK
    return STATUS_OK if reported.strip() == expected.strip() else STATUS_STALE


def _read_expected_version() -> str | None:
    try:
        text = _CARGO_TOML_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def _reported_src_hash(compute_module: Any) -> str | None:
    """Source hash embedded in the loaded extension, or None when unavailable.

    ``kernel_src_hash()`` is optional -- older extensions do not define it, and
    those built without scripts/build_native_backends.py return an empty string.
    Both cases collapse to None so the caller can degrade to the version check.
    """
    getter = getattr(compute_module, "kernel_src_hash", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except Exception:
        return None
    if not value:
        return None
    return str(value)


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
        return
    if status == STATUS_MISSING:
        logger.warning(
            "lumina_quant._compute has no build_info(): the loaded kernel "
            "predates the native version handshake (native/lumina_compute/"
            "Cargo.toml is at %s); rebuild with 'maturin develop --release' "
            "to pick up the handshake.",
            expected,
        )
        return

    # Version agrees. Now the source-hash handshake (N3): catches a kernel whose
    # lib.rs was edited without a version bump and without a rebuild. Degrades
    # silently when the loaded .so exposes no embedded hash (older build, or a
    # plain `maturin develop`) -- there is nothing to compare, not a failure.
    expected_hash = compute_src_hash()
    reported_hash = _reported_src_hash(compute_module)
    if compare_kernel_source_hash(expected_hash, reported_hash) == STATUS_STALE:
        logger.warning(
            "stale native kernel: lumina_quant._compute was built from crate "
            "source hash %s but native/lumina_compute/src is now at %s (crate "
            "version %s unchanged); rebuild via scripts/build_native_backends.py "
            "to re-embed the current source hash.",
            reported_hash,
            expected_hash,
            expected,
        )
    else:
        logger.debug(
            "native kernel source-hash handshake ok (reported=%s expected=%s)",
            reported_hash,
            expected_hash,
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
    "SRC_HASH_ENV_VAR",
    "STATUS_MISSING",
    "STATUS_OK",
    "STATUS_STALE",
    "check_native_kernel_version",
    "compare_kernel_source_hash",
    "compare_native_version",
    "compute_src_hash",
]
