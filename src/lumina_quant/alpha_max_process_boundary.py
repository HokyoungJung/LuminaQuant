"""Dependency-free process boundary for the frozen Alpha-Max commands."""

from __future__ import annotations

import os
import hashlib
import platform
import sys
from pathlib import Path

__all__ = [
    "AlphaMaxRuntimeContractError",
    "AmbientLQEnvironmentError",
    "alpha_max_bootstrap_implementation_inventory",
    "reject_ambient_lq_environment",
    "require_alpha_max_fresh_process_runtime",
]


class AlphaMaxRuntimeContractError(ValueError):
    """The sealed runtime contract or one of its construction inputs is invalid."""


class AmbientLQEnvironmentError(AlphaMaxRuntimeContractError):
    """An ambient ``LQ_*`` environment key would make the replay non-hermetic."""


def reject_ambient_lq_environment() -> None:
    """Reject every ambient key beginning with ``LQ_`` without reading its value."""
    offending = tuple(sorted(key for key in os.environ if key.startswith("LQ_")))
    if offending:
        raise AmbientLQEnvironmentError(f"ambient_lq_environment:{','.join(offending)}")


def require_alpha_max_fresh_process_runtime() -> None:
    """Require the CPython runtime supported by the spawn-only boundary."""
    if platform.python_implementation() != "CPython" or sys.version_info[:2] < (3, 14):
        raise AlphaMaxRuntimeContractError("alpha_max_fresh_process_runtime_invalid")


def alpha_max_bootstrap_implementation_inventory() -> list[dict[str, object]]:
    """Hash every executable Alpha-Max source before importing the runtime graph."""
    repository = Path(__file__).resolve().parents[2]
    paths = sorted((repository / "src" / "lumina_quant").rglob("*.py"))
    paths.extend(
        sorted(
            path
            for path in (repository / "scripts" / "research").glob("run_alpha_max_*.py")
            if path.is_file()
        )
    )
    native_compute = repository / "native" / "lumina_compute"
    paths.extend(sorted((native_compute / "src").rglob("*.rs")))
    for name in ("Cargo.toml", "Cargo.lock", "build.rs"):
        path = native_compute / name
        if path.is_file():
            paths.append(path)
    for relative in ("pyproject.toml", "scripts/build_native_backends.py"):
        path = repository / relative
        if path.is_file():
            paths.append(path)
    lock = repository / "uv.lock"
    if lock.is_file():
        paths.append(lock)
    inventory: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in paths:
        relative = path.relative_to(repository).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        payload = path.read_bytes()
        inventory.append(
            {
                "byte_count": len(payload),
                "relative_path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return inventory
