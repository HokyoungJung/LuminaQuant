"""Validate Polars GPU runtime behavior for CI.

This script has two modes:
1. contract mode (default): verify GPU extras/import surface and skip cleanly when
   no NVIDIA GPU runner is present.
2. strict mode (--require-gpu): require actual NVIDIA GPU execution and fail if the
   query would fall back from GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.compute_engine import (
    ComputeEngine,
    detect_nvidia_gpu,
    polars_gpu_available,
    resolve_compute_engine,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Polars GPU runtime behavior.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless actual NVIDIA GPU execution succeeds.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "gpu", "forced-gpu"),
        help="Compute-engine mode to resolve during the check.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional GPU device selector (e.g. 0, cuda:0).",
    )
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=0.0,
        help="Optional minimum GPU VRAM requirement in GiB.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4096,
        help="Synthetic row count for the strict runtime smoke query.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path for the result payload.",
    )
    return parser


def _normalize_device(raw: str) -> int | None:
    token = str(raw or "").strip().lower()
    if not token:
        return None
    if token.startswith(("cuda:", "gpu:")):
        token = token.split(":", maxsplit=1)[1].strip()
    return int(token)


def _build_strict_gpu_engine(*, device: int | None) -> Any:
    gpu_engine = getattr(pl, "GPUEngine", None)
    if gpu_engine is None:
        msg = "polars.GPUEngine is unavailable"
        raise RuntimeError(msg)

    kwarg_attempts: list[dict[str, Any]] = []
    if device is None:
        kwarg_attempts.extend(
            [
                {"raise_on_fail": True},
                {},
            ]
        )
    else:
        kwarg_attempts.extend(
            [
                {"device": device, "raise_on_fail": True},
                {"device": device},
            ]
        )

    for kwargs in kwarg_attempts:
        try:
            return gpu_engine(**kwargs)
        except TypeError:
            continue

    if device is not None:
        try:
            return gpu_engine(device)
        except TypeError:
            pass
        try:
            return gpu_engine(device, True)
        except TypeError:
            pass

    try:
        return gpu_engine()
    except TypeError as exc:
        msg = f"failed to construct strict GPUEngine: {exc}"
        raise RuntimeError(msg) from exc


def _gpu_runtime_lazy_frame(*, rows: int) -> pl.LazyFrame:
    count = max(64, int(rows))
    frame = pl.DataFrame(
        {
            "id": list(range(count)),
            "x": [float((idx % 17) + 1) for idx in range(count)],
            "y": [float((idx % 7) + 3) for idx in range(count)],
        }
    )
    return (
        frame.lazy()
        .with_columns(
            [
                (pl.col("id") % 32).alias("bucket"),
                (pl.col("x") * pl.col("y")).alias("xy"),
            ]
        )
        .group_by("bucket")
        .agg(
            [
                pl.col("xy").sum().alias("xy_sum"),
                pl.col("x").mean().alias("x_mean"),
                pl.len().alias("rows"),
            ]
        )
        .sort("bucket")
    )


def _run_strict_gpu_query(*, device: int | None, rows: int) -> dict[str, Any]:
    lazy = _gpu_runtime_lazy_frame(rows=rows)
    engine = _build_strict_gpu_engine(device=device)
    gpu = lazy.collect(engine=engine)
    cpu = lazy.collect(engine="streaming")
    matches_cpu = gpu.to_dicts() == cpu.to_dicts()
    return {
        "gpu_rows": int(gpu.height),
        "cpu_rows": int(cpu.height),
        "matches_cpu": bool(matches_cpu),
        "buckets": int(gpu.height),
    }


def run_check(
    *,
    require_gpu: bool,
    mode: str,
    device: int | None,
    min_vram_gb: float,
    rows: int,
) -> tuple[int, dict[str, Any]]:
    nvidia_ok, nvidia_reason = detect_nvidia_gpu()
    gpuextra_ok = hasattr(pl, "GPUEngine")
    payload: dict[str, Any] = {
        "require_gpu": bool(require_gpu),
        "requested_mode": str(mode),
        "device": device,
        "min_vram_gb": float(min_vram_gb),
        "nvidia_ok": bool(nvidia_ok),
        "nvidia_reason": str(nvidia_reason),
        "gpu_engine_attr": bool(gpuextra_ok),
    }

    if not gpuextra_ok:
        payload["status"] = "failed"
        payload["reason"] = "polars.GPUEngine is unavailable"
        return 1, payload

    if not nvidia_ok and not require_gpu:
        payload["status"] = "skipped"
        payload["reason"] = "No NVIDIA GPU runner detected; contract check passed without hardware smoke."
        return 0, payload

    gpu_ok, gpu_reason = polars_gpu_available(
        device=device,
        smoke_test=True,
        min_vram_gb=float(min_vram_gb),
    )
    payload["gpu_probe_ok"] = bool(gpu_ok)
    payload["gpu_probe_reason"] = str(gpu_reason)
    if not gpu_ok:
        payload["status"] = "failed"
        payload["reason"] = str(gpu_reason)
        return 1, payload

    selection: ComputeEngine = resolve_compute_engine(
        mode=str(mode),
        device=device,
        vram_gb=float(min_vram_gb),
        verbose=False,
    )
    payload["resolved_engine"] = str(selection.resolved_engine)
    payload["resolved_reason"] = str(selection.reason)
    if selection.resolved_engine != "gpu":
        payload["status"] = "failed"
        payload["reason"] = f"expected gpu resolution, got {selection.resolved_engine}"
        return 1, payload

    strict = _run_strict_gpu_query(device=device, rows=rows)
    payload["strict_query"] = strict
    if not bool(strict["matches_cpu"]):
        payload["status"] = "failed"
        payload["reason"] = "strict GPU query diverged from CPU result"
        return 1, payload

    payload["status"] = "passed"
    payload["reason"] = "strict GPU runtime smoke passed"
    return 0, payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, payload = run_check(
        require_gpu=bool(args.require_gpu),
        mode=str(args.mode),
        device=_normalize_device(str(args.device)),
        min_vram_gb=float(args.min_vram_gb),
        rows=int(args.rows),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)

    output_path = str(args.output_json or "").strip()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")

    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
