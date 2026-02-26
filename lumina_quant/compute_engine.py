"""Compute-engine selection with deterministic GPU auto/CPU fallback semantics.

Environment controls:
- LQ_GPU_MODE: auto|cpu|gpu|forced-gpu
- LQ_GPU_DEVICE: integer id or cuda:<id>/gpu:<id>
- LQ_GPU_VRAM_GB: required minimum VRAM in GB for GPU eligibility
- LQ_GPU_VERBOSE: truthy values enable selection/fallback logging
"""

from __future__ import annotations

import math
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
    required_vram_gb: float = 0.0
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
            try:
                return lazy_frame.collect(engine=engine)
            except Exception as exc:
                if self.requested_mode != "forced-gpu" and _is_gpu_oom_error(exc):
                    if self.verbose:
                        print(
                            "[compute_engine] gpu OOM detected; falling back to CPU streaming "
                            f"(reason={exc})"
                        )
                    return lazy_frame.collect(engine="streaming")
                raise
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


def _parse_gpu_vram_gb(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid GPU VRAM selector. Expected float GB, got: {value!r}") from exc
    if parsed < 0.0:
        raise ValueError("GPU VRAM selector must be >= 0.")
    return parsed


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


def detect_nvidia_total_memory_gb(
    *,
    timeout_seconds: float = 1.5,
) -> tuple[list[float], str]:
    """Best-effort NVIDIA VRAM detection in GB via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return [], "nvidia-smi not found"

    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return [], f"nvidia-smi memory probe failed: {exc}"

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or f"exit={proc.returncode}"
        return [], f"nvidia-smi memory probe failed: {err}"

    values: list[float] = []
    for raw in (proc.stdout or "").splitlines():
        token = raw.strip()
        if not token:
            continue
        try:
            values.append(float(token) / 1024.0)
        except ValueError:
            continue
    if not values:
        return [], "nvidia-smi memory query returned no parseable GPUs"
    return values, f"nvidia-smi memory query detected {len(values)} GPU(s)"


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


def polars_gpu_available(
    *,
    device: int | None,
    smoke_test: bool = True,
    min_vram_gb: float = 0.0,
) -> tuple[bool, str]:
    """Validate Polars GPU availability using GPUEngine and an optional smoke test."""
    nvidia_ok, nvidia_reason = detect_nvidia_gpu()
    min_vram = max(0.0, float(min_vram_gb))

    try:
        engine = _build_gpu_engine(device=device)
    except Exception as exc:
        return False, f"failed to build GPUEngine: {exc}"

    if engine is None:
        if nvidia_ok:
            return False, f"polars.GPUEngine is not available ({nvidia_reason})"
        return False, f"{nvidia_reason}; polars.GPUEngine is not available"

    if min_vram > 0.0:
        memory_gb, memory_reason = detect_nvidia_total_memory_gb()
        if not memory_gb:
            return False, f"gpu compatibility check failed ({memory_reason}); requires >= {min_vram} GB"
        target_idx = int(device or 0)
        if target_idx < 0 or target_idx >= len(memory_gb):
            return False, f"gpu device index {target_idx} is unavailable ({memory_reason})"
        available = float(memory_gb[target_idx])
        if available + 1e-12 < min_vram:
            return False, f"gpu vram {available:.2f} GB < required {min_vram:.2f} GB"

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


def _probe_gpu_engine(*, device: int | None, min_vram_gb: float = 0.0) -> tuple[bool, str]:
    """Backward-compatible internal probe name."""
    try:
        return polars_gpu_available(
            device=device,
            smoke_test=True,
            min_vram_gb=float(min_vram_gb),
        )
    except TypeError:
        # Allows legacy monkeypatched test shims with older signature.
        return polars_gpu_available(device=device, smoke_test=True)


def _is_gpu_oom_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    if not message:
        return False
    return any(
        token in message
        for token in (
            "out of memory",
            "cuda out of memory",
            "cuda error: out of memory",
            "memory allocation",
            "insufficient memory",
        )
    )


def select_engine(
    *,
    mode: str | None = None,
    device: str | int | None = None,
    vram_gb: str | float | int | None = None,
    verbose: str | bool | None = None,
) -> ComputeEngine:
    """Resolve CPU/GPU mode from args/environment with deterministic fallback."""
    requested_mode = _normalize_gpu_mode(mode or os.getenv("LQ_GPU_MODE", "auto"))
    resolved_device = _parse_gpu_device(device if device is not None else os.getenv("LQ_GPU_DEVICE"))
    resolved_vram_gb = _parse_gpu_vram_gb(
        vram_gb if vram_gb is not None else os.getenv("LQ_GPU_VRAM_GB", "0")
    )
    resolved_verbose = _parse_verbose(
        verbose if verbose is not None else os.getenv("LQ_GPU_VERBOSE", "0")
    )

    if requested_mode == "cpu":
        engine = ComputeEngine(
            requested_mode=requested_mode,
            resolved_engine="cpu",
            device=resolved_device,
            required_vram_gb=resolved_vram_gb,
            verbose=resolved_verbose,
            reason="LQ_GPU_MODE=cpu",
        )
        _maybe_log(engine)
        return engine

    gpu_ok, probe_reason = _probe_gpu_engine(
        device=resolved_device,
        min_vram_gb=resolved_vram_gb,
    )

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
            required_vram_gb=resolved_vram_gb,
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
            required_vram_gb=resolved_vram_gb,
            verbose=resolved_verbose,
            reason=probe_reason,
        )
        _maybe_log(engine)
        return engine

    engine = ComputeEngine(
        requested_mode=requested_mode,
        resolved_engine="cpu",
        device=resolved_device,
        required_vram_gb=resolved_vram_gb,
        verbose=resolved_verbose,
        reason=f"auto fallback to CPU: {probe_reason}",
    )
    _maybe_log(engine)
    return engine


def resolve_compute_engine(
    *,
    mode: str | None = None,
    device: str | int | None = None,
    vram_gb: str | float | int | None = None,
    verbose: str | bool | None = None,
) -> ComputeEngine:
    """Backward-compatible alias for select_engine."""
    return select_engine(mode=mode, device=device, vram_gb=vram_gb, verbose=verbose)


def collect(
    lazy_frame: pl.LazyFrame,
    *,
    mode: str | None = None,
    device: str | int | None = None,
    vram_gb: str | float | int | None = None,
    verbose: str | bool | None = None,
    engine: ComputeEngine | None = None,
) -> pl.DataFrame:
    """Collect a lazy frame using explicit or auto-selected compute engine."""
    resolved_engine = engine or select_engine(
        mode=mode,
        device=device,
        vram_gb=vram_gb,
        verbose=verbose,
    )
    return resolved_engine.collect(lazy_frame)


def collect_with_compute_engine(
    lazy_frame: pl.LazyFrame,
    *,
    mode: str | None = None,
    device: str | int | None = None,
    vram_gb: str | float | int | None = None,
    verbose: str | bool | None = None,
) -> pl.DataFrame:
    """Backward-compatible one-shot collection helper."""
    return collect(
        lazy_frame,
        mode=mode,
        device=device,
        vram_gb=vram_gb,
        verbose=verbose,
    )


def _resolve_numeric_columns(frame: pl.DataFrame) -> list[str]:
    numeric: list[str] = []
    for column, dtype in zip(frame.columns, frame.dtypes, strict=False):
        is_numeric = False
        try:
            is_numeric = bool(dtype.is_numeric())
        except Exception:
            is_numeric = dtype in {
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            }
        if is_numeric:
            numeric.append(str(column))
    return numeric


def compare_cpu_gpu_determinism(
    lazy_frame: pl.LazyFrame,
    *,
    tolerance: float = 1e-9,
    device: str | int | None = None,
    vram_gb: str | float | int | None = None,
    numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Collect with CPU/GPU engines and report deterministic tolerance deltas."""
    cpu_engine = select_engine(mode="cpu", device=device, vram_gb=vram_gb, verbose=False)
    gpu_engine = select_engine(mode="gpu", device=device, vram_gb=vram_gb, verbose=False)
    cpu = cpu_engine.collect(lazy_frame)
    gpu = gpu_engine.collect(lazy_frame)

    columns = list(numeric_columns or _resolve_numeric_columns(cpu))
    max_abs_diff: dict[str, float] = {}
    within = cpu.height == gpu.height and cpu.width == gpu.width
    eps = max(0.0, float(tolerance))

    if within:
        for column in columns:
            cpu_vals = cpu[column].to_list()
            gpu_vals = gpu[column].to_list()
            if len(cpu_vals) != len(gpu_vals):
                within = False
                max_abs_diff[column] = math.inf
                continue
            col_max = 0.0
            for lhs, rhs in zip(cpu_vals, gpu_vals, strict=False):
                diff = abs(float(lhs) - float(rhs))
                if diff > col_max:
                    col_max = diff
            max_abs_diff[column] = col_max
            if col_max > eps:
                within = False

    return {
        "within_tolerance": within,
        "tolerance": eps,
        "columns_checked": columns,
        "max_abs_diff": max_abs_diff,
        "cpu_rows": int(cpu.height),
        "gpu_rows": int(gpu.height),
    }


def _maybe_log(engine: ComputeEngine) -> None:
    if not engine.verbose:
        return
    print(
        "[compute_engine] "
        f"requested={engine.requested_mode} resolved={engine.resolved_engine} "
        f"device={engine.device} min_vram_gb={engine.required_vram_gb} reason={engine.reason}"
    )
