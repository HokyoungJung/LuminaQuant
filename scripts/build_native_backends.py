"""Build the lumina_compute pyo3 island via maturin.

Replaces the old per-crate ctypes build script. There is now a single
cdylib — native/lumina_compute — that bundles all kernel functions.

Usage (development):
    python scripts/build_native_backends.py           # maturin develop --release
    python scripts/build_native_backends.py --wheel   # maturin build --release

The script honours VIRTUAL_ENV if set; otherwise it locates the project
.venv automatically.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_maturin(venv_bin: Path) -> str:
    """Return path to maturin executable, preferring the venv copy."""
    venv_maturin = venv_bin / "maturin"
    if venv_maturin.exists():
        return str(venv_maturin)
    system = shutil.which("maturin")
    if system:
        return system
    return ""


def _load_src_hash(root: Path, src_dir: Path) -> tuple[str, str | None]:
    """Compute the crate source hash using the shared runtime helper.

    Importing lumina_quant._native_kernel_version.compute_src_hash (rather than
    reimplementing the hash) guarantees the value embedded into the extension
    matches what the runtime handshake recomputes -- there is no second
    algorithm to drift. Returns (env_var_name, hash_or_None); the hash is None
    when the helper is unavailable or the source tree cannot be read, in which
    case the build simply proceeds without embedding a hash.
    """
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from lumina_quant._native_kernel_version import (
            SRC_HASH_ENV_VAR,
            compute_src_hash,
        )
    except Exception:
        return "LUMINA_KERNEL_SRC_HASH", None
    try:
        return SRC_HASH_ENV_VAR, compute_src_hash(src_dir)
    except Exception:
        return SRC_HASH_ENV_VAR, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lumina_compute pyo3 island (maturin)")
    parser.add_argument(
        "--wheel",
        action="store_true",
        help="Build a wheel instead of an editable install (maturin build --release)",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        default=True,
        help="Build in release mode (default: true)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    crate_dir = root / "native" / "lumina_compute"

    if not crate_dir.exists():
        print(f"[ERROR] crate directory not found: {crate_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve the virtual environment
    venv_path_str = os.environ.get("VIRTUAL_ENV", "").strip()
    venv = Path(venv_path_str) if venv_path_str else root / ".venv"
    venv_bin = venv / "bin"

    maturin_exec = _find_maturin(venv_bin)
    if not maturin_exec:
        print(
            "[ERROR] maturin not found. Install it with: uv add --dev maturin",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.wheel:
        cmd = [maturin_exec, "build", "--release"]
        action = "build (wheel)"
    else:
        cmd = [maturin_exec, "develop", "--release"]
        action = "develop (editable install)"

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv)

    # Embed a hash of the crate's Rust source so the runtime handshake can flag
    # an edited-but-not-rebuilt kernel even when the crate version is unchanged
    # (see lumina_quant._native_kernel_version). cargo tracks env vars read via
    # option_env!, so changing this hash forces a rebuild of the affected unit.
    hash_env_var, src_hash = _load_src_hash(root, crate_dir / "src")
    if src_hash:
        env[hash_env_var] = src_hash

    print(f"[maturin] {action}")
    print(f"  crate : {crate_dir}")
    print(f"  venv  : {venv}")
    print(f"  srchash: {src_hash or '(unavailable)'}")
    print(f"  cmd   : {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(crate_dir), env=env)
    rc = int(result.returncode)
    print(f"[maturin] rc={rc}")
    if rc != 0:
        sys.exit(rc)

    print("[maturin] lumina_compute built successfully")


if __name__ == "__main__":
    main()
