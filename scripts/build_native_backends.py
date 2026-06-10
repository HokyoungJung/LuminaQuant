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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build lumina_compute pyo3 island (maturin)"
    )
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

    print(f"[maturin] {action}")
    print(f"  crate : {crate_dir}")
    print(f"  venv  : {venv}")
    print(f"  cmd   : {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(crate_dir), env=env)
    rc = int(result.returncode)
    print(f"[maturin] rc={rc}")
    if rc != 0:
        sys.exit(rc)

    print("[maturin] lumina_compute built successfully")


if __name__ == "__main__":
    main()
