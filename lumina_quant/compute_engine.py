"""Compute-engine selection with deterministic GPU auto/CPU fallback semantics.

Environment controls:
- LQ_GPU_MODE: auto|cpu|gpu|forced-gpu
- LQ_GPU_DEVICE: integer id or cuda:<id>/gpu:<id>
- LQ_GPU_VERBOSE: truthy values enable selection/fallback logging
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

GPUMode = Literal["auto", "cpu", "gpu", "forced-gpu"]
ResolvedEngine = Literal["cpu", "gpu"]


class GPUNotAvailableError(RuntimeError):
    """Raised when forced GPU execution is requested but unavailable."""


@dataclass(frozen=True, slots=True)
class ComputeEngine:
    """Resolved compute-engine plan for Polars lazy collection."""

    requested_mode: GPUMode
    resolved_engine: ResolvedEngine
    device: int | None
    verbose: bool = False
    reason: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.resolved_engine == "gpu"

    def collect(self, lazy_frame: pl.LazyFrame) -> pl.DataFrame:
        """Collect a lazy frame with the resolved execution backend."""
        if self.resolved_engine == "gpu":
            engine = _build_gpu_engine(device=self.device)
            if engine is None:
                raise GPUNotAvailableError(
                    "GPU engine resolved but unavailable at collect-time. "
                    "Set LQ_GPU_MODE=cpu or install/configure Polars GPU runtime."
                )
            return lazy_frame.collect(engine=engine)
        return lazy_frame.collect(engine="streaming")


def _normalize_gpu_mode(value: str | None) -> GPUMode:
    token = str(value or "auto").strip().lower().replace("_", "-")
    if token in {"", "auto"}:
        return "auto"
    if token in {"cpu"}:
        return "cpu"
    if token in {"gpu"}:
        return "gpu"
    if token in {"forced-gpu", "force-gpu", "forcegpu", "forcedgpu"}:
        return "forced-gpu"
    raise ValueError(
        "Invalid GPU mode. Expected one of auto|cpu|gpu|forced-gpu "
        f"but received: {value!r}."
    )


def _parse_gpu_device(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    token = str(value).strip().lower()
    if not token:
        return None
    if token.startswith(("cuda:", "gpu:")):
        token = token.split(":", maxsplit=1)[1].strip()
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            "Invalid GPU device selector. Expected integer or cuda:<int>/gpu:<int> "
            f"but received: {value!r}."
        ) from exc


def _parse_verbose(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on", "y"}


def detect_nvidia_gpu(*, timeout_seconds: float = 1.5) -> tuple[bool, str]:
    """Best-effort NVIDIA GPU detection via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found"

    try:
        proc = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return False, f"nvidia-smi probe failed: {exc}"

    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or output or f"exit={proc.returncode}"
        return False, f"nvidia-smi failed: {err}"

    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return False, "nvidia-smi returned no GPUs"
    return True, f"nvidia-smi detected {len(lines)} GPU(s)"


def _build_gpu_engine(*, device: int | None) -> Any | None:
    gpu_engine = getattr(pl, "GPUEngine", None)
    if gpu_engine is None:
        return None

    # Polars versions may accept GPUEngine(), GPUEngine(device=...), or GPUEngine(int)
    if device is None:
        return gpu_engine()

    try:
        return gpu_engine(device=device)
    except TypeError:
        return gpu_engine(device)


def _run_polars_gpu_smoke(engine: Any) -> tuple[bool, str]:
    try:
        smoke = pl.DataFrame({"x": [1, 2, 3]}).lazy().select(pl.col("x").sum().alias("x_sum"))
        smoke.collect(engine=engine)
        return True, "gpu smoke test passed"
    except Exception as exc:
        return False, f"gpu smoke test failed: {exc}"


def polars_gpu_available(*, device: int | None, smoke_test: bool = True) -> tuple[bool, str]:
    """Validate Polars GPU availability using GPUEngine and an optional smoke test."""
    nvidia_ok, nvidia_reason = detect_nvidia_gpu()

    try:
        engine = _build_gpu_engine(device=device)
    except Exception as exc:
        return False, f"failed to build GPUEngine: {exc}"

    if engine is None:
        if nvidia_ok:
            return False, f"polars.GPUEngine is not available ({nvidia_reason})"
        return False, f"{nvidia_reason}; polars.GPUEngine is not available"

    if not smoke_test:
        if nvidia_ok:
            return True, f"polars.GPUEngine available ({nvidia_reason})"
        return True, f"polars.GPUEngine available ({nvidia_reason}; proceeding without strict check)"

    smoke_ok, smoke_reason = _run_polars_gpu_smoke(engine)
    if smoke_ok:
        if nvidia_ok:
            return True, f"{smoke_reason} ({nvidia_reason})"
        return True, f"{smoke_reason} ({nvidia_reason}; proceeding with GPUEngine)"

    if nvidia_ok:
        return False, smoke_reason
    return False, f"{smoke_reason}; {nvidia_reason}"


def _probe_gpu_engine(*, device: int | None) -> tuple[bool, str]:
    """Backward-compatible internal probe name."""
    return polars_gpu_available(device=device, smoke_test=True)


def select_engine(
    *,
    mode: str | None = None,
    device: str | int | None = None,
    verbose: str | bool | None = None,
) -> ComputeEngine:
    """Resolve CPU/GPU mode from args/environment with deterministic fallback."""
    requested_mode = _normalize_gpu_mode(mode or os.getenv("LQ_GPU_MODE", "auto"))
    resolved_device = _parse_gpu_device(device if device is not None else os.getenv("LQ_GPU_DEVICE"))
    resolved_verbose = _parse_verbose(
        verbose if verbose is not None else os.getenv("LQ_GPU_VERBOSE", "0")
    )

    if requested_mode == "cpu":
        engine = ComputeEngine(
            requested_mode=requested_mode,
            resolved_engine="cpu",
            device=resolved_device,
            verbose=resolved_verbose,
            reason="LQ_GPU_MODE=cpu",
        )
        _maybe_log(engine)
        return engine

    gpu_ok, probe_reason = polars_gpu_available(device=resolved_device, smoke_test=True)

    if requested_mode in {"gpu", "forced-gpu"}:
        if not gpu_ok:
            raise GPUNotAvailableError(
                f"LQ_GPU_MODE={requested_mode} requires GPU but GPU path is unavailable: "
                f"{probe_reason}. Set LQ_GPU_MODE=cpu or fix GPU runtime configuration."
            )
        engine = ComputeEngine(
            requested_mode=requested_mode,
            resolved_engine="gpu",
            device=resolved_device,
            verbose=resolved_verbose,
            reason=probe_reason,
        )
        _maybe_log(engine)
        return engine

    if gpu_ok:
        engine = ComputeEngine(
            requested_mode=requested_mode,
            resolved_engine="gpu",
            device=resolved_device,
            verbose=resolved_verbose,
            reason=probe_reason,
        )
        _maybe_log(engine)
        return engine

    engine = ComputeEngine(
        requested_mode=requested_mode,
        resolved_engine="cpu",
        device=resolved_device,
        verbose=resolved_verbose,
        reason=f"auto fallback to CPU: {probe_reason}",
    )
    _maybe_log(engine)
    return engine


def resolve_compute_engine(
    *,
    mode: str | None = None,
    device: str | int | None = None,
    verbose: str | bool | None = None,
) -> ComputeEngine:
    """Backward-compatible alias for select_engine."""
    return select_engine(mode=mode, device=device, verbose=verbose)


def collect(
    lazy_frame: pl.LazyFrame,
    *,
    mode: str | None = None,
    device: str | int | None = None,
    verbose: str | bool | None = None,
    engine: ComputeEngine | None = None,
) -> pl.DataFrame:
    """Collect a lazy frame using explicit or auto-selected compute engine."""
    resolved_engine = engine or select_engine(mode=mode, device=device, verbose=verbose)
    return resolved_engine.collect(lazy_frame)


def collect_with_compute_engine(
    lazy_frame: pl.LazyFrame,
    *,
    mode: str | None = None,
    device: str | int | None = None,
    verbose: str | bool | None = None,
) -> pl.DataFrame:
    """Backward-compatible one-shot collection helper."""
    return collect(lazy_frame, mode=mode, device=device, verbose=verbose)


def _maybe_log(engine: ComputeEngine) -> None:
    if not engine.verbose:
        return
    print(
        "[compute_engine] "
        f"requested={engine.requested_mode} resolved={engine.resolved_engine} "
        f"device={engine.device} reason={engine.reason}"
    )
