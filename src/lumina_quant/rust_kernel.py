"""Optional Rust backtest-kernel bridge for source-checkout usage."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def rust_kernel_available() -> bool:
    return shutil.which("cargo") is not None and _manifest_path().exists()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_path() -> Path:
    return _repo_root() / "native" / "rust_backtest_kernel" / "Cargo.toml"


def run_rust_backtest(
    data_path: str,
    *,
    fast_window: int = 3,
    slow_window: int = 8,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> dict[str, object]:
    """Run the optional Rust kernel and return its JSON summary."""
    cargo = shutil.which("cargo")
    manifest = _manifest_path()
    if cargo is None:
        raise RuntimeError("cargo is required for the Rust kernel")
    if not manifest.exists():
        raise RuntimeError(f"Rust kernel manifest not found: {manifest}")
    command = [
        cargo,
        "run",
        "--quiet",
        "--manifest-path",
        str(manifest),
        "--",
        str(Path(data_path)),
        str(int(fast_window)),
        str(int(slow_window)),
        str(float(initial_cash)),
        str(float(fee_bps)),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)
