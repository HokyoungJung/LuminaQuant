from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_PRIORITY_SYMBOLS = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT,TRX/USDT"
DEFAULT_MEMORY_BUDGET_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_SOFT_RSS_BYTES = int(7.2 * 1024 * 1024 * 1024)
DEFAULT_PARALLEL_RESERVE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_PARALLEL_PER_WORKER_BYTES = int(1.5 * 1024 * 1024 * 1024)
_THREAD_ENV_NAMES = (
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_script_path() -> Path:
    return _repo_root() / "scripts" / "build_native_backends.py"


def _refresh_script_path() -> Path:
    return _repo_root() / "scripts" / "research" / "refresh_final_portfolio_validation_data.py"


def _native_rawfirst_library_path() -> Path:
    suffix = (
        ".dll"
        if sys.platform.startswith("win")
        else ".dylib"
        if sys.platform == "darwin"
        else ".so"
    )
    prefix = "" if sys.platform.startswith("win") else "lib"
    return (
        _repo_root()
        / "native"
        / "rust_rawfirst"
        / "target"
        / "release"
        / f"{prefix}lumina_rawfirst{suffix}"
    )


def build_refresh_command(extra_args: list[str]) -> list[str]:
    tokens = list(extra_args or [])
    if tokens[:1] == ["--"]:
        tokens = tokens[1:]
    if "--priority-symbols" not in tokens:
        tokens.extend(["--priority-symbols", DEFAULT_PRIORITY_SYMBOLS])
    if "--max-workers" not in tokens:
        tokens.extend(["--max-workers", "0"])
    if "--memory-budget-bytes" not in tokens:
        tokens.extend(["--memory-budget-bytes", str(DEFAULT_MEMORY_BUDGET_BYTES)])
    if "--soft-rss-bytes" not in tokens:
        tokens.extend(["--soft-rss-bytes", str(DEFAULT_SOFT_RSS_BYTES)])
    if "--parallel-reserve-bytes" not in tokens:
        tokens.extend(["--parallel-reserve-bytes", str(DEFAULT_PARALLEL_RESERVE_BYTES)])
    if "--parallel-per-worker-bytes" not in tokens:
        tokens.extend(["--parallel-per-worker-bytes", str(DEFAULT_PARALLEL_PER_WORKER_BYTES)])
    return [sys.executable, str(_refresh_script_path()), *tokens]


def build_refresh_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in _THREAD_ENV_NAMES:
        env.setdefault(name, "1")
    env.setdefault("LQ_RAW_FIRST_BACKEND", "auto")
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the optimized final-portfolio validation refresh with safe defaults.",
    )
    parser.add_argument(
        "--build-native-if-missing",
        action="store_true",
        help="Build the Rust raw-first backend before launching if the library is missing.",
    )
    args, refresh_args = parser.parse_known_args(argv)

    env = build_refresh_env()
    if args.build_native_if_missing and not _native_rawfirst_library_path().exists():
        subprocess.run(
            [sys.executable, str(_build_script_path()), "--backend", "rust-rawfirst"],
            cwd=str(_repo_root()),
            env=env,
            check=True,
        )

    command = build_refresh_command(list(refresh_args or []))
    completed = subprocess.run(command, cwd=str(_repo_root()), env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
