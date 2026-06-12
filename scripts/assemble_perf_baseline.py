"""Assemble baseline/perf-baseline.json from individual benchmark outputs.

Reads:
  baseline/bench_optimizer_kernel.txt  -> Axis 1 MICRO
  baseline/bench_walkforward_e2e.json  -> Axis 1 E2E  (optional, TBD after worker-2)
  baseline/bench_backtest.json         -> Axis 2
  baseline/bench_live_signal.json      -> Axis 3a (batch signal kernels)
  baseline/bench_tick_latency.json     -> Axis 3b/c/d (per-tick + order + replay)
  baseline/8gb_gate.json               -> Axis 4 (RSS gate result)
  baseline/bench_rss_time.log          -> Axis 4 RSS source

Writes:
  baseline/perf-baseline.json
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "baseline"

RSS_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)", re.IGNORECASE)
OPTIMIZER_ELAPSED_PATTERN = re.compile(r"elapsed_sec=([0-9.]+)")
OPTIMIZER_EPS_PATTERN = re.compile(r"evals_per_sec=([0-9.]+)")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _parse_optimizer_txt(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    elapsed = OPTIMIZER_ELAPSED_PATTERN.search(text)
    eps = OPTIMIZER_EPS_PATTERN.search(text)
    return (
        float(elapsed.group(1)) if elapsed else None,
        float(eps.group(1)) if eps else None,
    )


def _parse_rss_log(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = [float(m) for m in RSS_PATTERN.findall(text)]
    if not matches:
        return None
    return max(matches) / 1024.0  # kbytes -> MB


def _gpu_info() -> dict:
    try:
        # name, driver_version, memory.total, compute_cap
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            # Also get CUDA UMD version from header line
            smi_result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            cuda_umd = None
            for line in (smi_result.stdout or "").splitlines():
                m = re.search(r"CUDA UMD Version:\s*([0-9.]+)", line)
                if m:
                    cuda_umd = m.group(1)
                    break
            return {
                "gpu_present": True,
                "gpu_model": parts[0] if len(parts) > 0 else None,
                "nvidia_driver_version": parts[1] if len(parts) > 1 else None,
                "gpu_memory_mib": int(parts[2].replace("MiB", "").strip())
                if len(parts) > 2
                else None,
                "gpu_compute_cap": parts[3] if len(parts) > 3 else None,
                "cuda_umd_version": cuda_umd,
            }
    except Exception:
        pass
    return {"gpu_present": False}


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _ram_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except OSError:
        pass
    return None


def assemble() -> dict:
    # ------------------------------------------------------------------ #
    # Hardware fingerprint
    # ------------------------------------------------------------------ #
    gpu = _gpu_info()
    hardware = {
        "cpu_model": _cpu_model(),
        "ram_mb": _ram_mb(),
        **gpu,
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "platform_kernel": platform.platform(),
    }

    # ------------------------------------------------------------------ #
    # Axis 1 MICRO — optimizer kernel
    # ------------------------------------------------------------------ #
    kernel_elapsed, kernel_eps = _parse_optimizer_txt(BASELINE / "bench_optimizer_kernel.txt")

    # ------------------------------------------------------------------ #
    # Axis 1 E2E — walk-forward (optional: pending worker-2 fixtures)
    # ------------------------------------------------------------------ #
    wf_e2e = _load_json(BASELINE / "bench_walkforward_e2e.json")
    wf_e2e_elapsed = wf_e2e.get("walk_forward_e2e_elapsed_s")

    # ------------------------------------------------------------------ #
    # Axis 2 — backtest throughput
    # ------------------------------------------------------------------ #
    backtest = _load_json(BASELINE / "bench_backtest.json")
    bars_per_sec = backtest.get("median_bars_per_sec")
    backtest_median_s = backtest.get("median_seconds")

    # ------------------------------------------------------------------ #
    # Axis 3a — live signal batch benchmark (benchmark_live_signal_backend)
    # ------------------------------------------------------------------ #
    live_signal = _load_json(BASELINE / "bench_live_signal.json")
    rust_timing = live_signal.get("rust") or {}
    python_timing = live_signal.get("python") or {}
    live_backend = (live_signal.get("diagnostics") or {}).get("resolved_backend", "unknown")

    # ------------------------------------------------------------------ #
    # Axis 3b/c/d — per-tick + order path + replay proxy
    # ------------------------------------------------------------------ #
    tick = _load_json(BASELINE / "bench_tick_latency.json")
    live_per_tick = tick.get("live_signal_per_tick") or {}
    combined_tick = live_per_tick.get("combined_debounced_plus_trailing") or {}
    paper_order = tick.get("paper_order_path") or {}
    order_chain = paper_order.get("full_transition_chain") or {}
    replay_proxy = tick.get("replay_path_proxy") or {}

    # ------------------------------------------------------------------ #
    # Axis 4 — RSS
    # ------------------------------------------------------------------ #
    rss_from_log = _parse_rss_log(BASELINE / "bench_rss_time.log")
    gate = _load_json(BASELINE / "8gb_gate.json")
    gate_rss_source = (gate.get("checks") or {}).get("rss_budget") or {}
    gate_peak_str = gate_rss_source.get("detail", "")
    # Try to extract from gate detail string e.g. "peak_rss=129.59MiB < ..."
    gate_rss_match = re.search(r"peak_rss=([0-9.]+)MiB", gate_peak_str)
    peak_rss_mb = float(gate_rss_match.group(1)) if gate_rss_match else rss_from_log

    # ------------------------------------------------------------------ #
    # Assemble
    # ------------------------------------------------------------------ #
    axes: dict = {
        # Axis 1 MICRO
        "walk_forward_kernel_elapsed_s": kernel_elapsed,
        "walk_forward_kernel_evals_per_sec": kernel_eps,
        # Axis 1 E2E (TBD if fixtures not yet available)
        "walk_forward_e2e_elapsed_s": wf_e2e_elapsed,
        "walk_forward_e2e_status": "complete" if wf_e2e_elapsed else "pending_fixtures",
        # Axis 2
        "backtest_median_bars_per_sec": bars_per_sec,
        "backtest_median_s": backtest_median_s,
        # Axis 3a: batch signal kernels (per-50k-row eval)
        "live_signal_backend": live_backend,
        "live_signal_batch_rust_debounced_s_per_eval": rust_timing.get(
            "debounced_seconds_per_eval"
        ),
        "live_signal_batch_rust_trailing_s_per_eval": rust_timing.get("trailing_seconds_per_eval"),
        "live_signal_batch_rust_total_s_per_eval": rust_timing.get("total_seconds_per_eval"),
        "live_signal_batch_python_total_s_per_eval": python_timing.get("total_seconds_per_eval"),
        "live_signal_batch_speedup_rust_vs_python": live_signal.get("speedup_total"),
        "live_signal_batch_parity_exact": (live_signal.get("parity") or {}).get("passed"),
        # Axis 3b: per-single-tick signal latency (10k iters)
        "live_signal_kernel_latency_median_ms": combined_tick.get("median_ms"),
        "live_signal_kernel_latency_p95_ms": combined_tick.get("p95_ms"),
        "live_signal_kernel_latency_p99_ms": combined_tick.get("p99_ms"),
        # Axis 3c: paper order path (OrderStateMachine, no I/O)
        "paper_order_path_latency_median_ms": order_chain.get("median_ms"),
        "paper_order_path_latency_p95_ms": order_chain.get("p95_ms"),
        "paper_order_path_latency_p99_ms": order_chain.get("p99_ms"),
        "paper_order_path_note": paper_order.get("note"),
        # Axis 3d: replay path proxy (ShadowLiveRunner)
        "replay_path_proxy_latency_median_ms": replay_proxy.get("median_ms"),
        "replay_path_proxy_note": replay_proxy.get("note"),
        # Axis 4
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_source": "bench_rss_time.log (/usr/bin/time -v)",
        "peak_rss_gate_status": (gate.get("checks") or {}).get("rss_budget", {}).get("status"),
    }

    source_files = {
        "walk_forward_kernel": "baseline/bench_optimizer_kernel.txt",
        "walk_forward_e2e": "baseline/bench_walkforward_e2e.json",
        "backtest": "baseline/bench_backtest.json",
        "live_signal_batch": "baseline/bench_live_signal.json",
        "tick_latency": "baseline/bench_tick_latency.json",
        "rss_time_log": "baseline/bench_rss_time.log",
        "memory_gate": "baseline/8gb_gate.json",
    }

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hardware": hardware,
        "axes": axes,
        "source_files": source_files,
    }


def main() -> None:
    out = BASELINE / "perf-baseline.json"
    payload = assemble()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {out}")

    # Summary
    axes = payload["axes"]
    print("\n=== Axis Summary ===")
    print(f"  Axis 1 MICRO  elapsed_s      : {axes.get('walk_forward_kernel_elapsed_s')}")
    print(f"  Axis 1 MICRO  evals/sec       : {axes.get('walk_forward_kernel_evals_per_sec')}")
    print(
        f"  Axis 1 E2E    elapsed_s       : {axes.get('walk_forward_e2e_elapsed_s')} "
        f"[{axes.get('walk_forward_e2e_status')}]"
    )
    print(f"  Axis 2        bars/sec        : {axes.get('backtest_median_bars_per_sec')}")
    print(f"  Axis 3b       sig median_ms   : {axes.get('live_signal_kernel_latency_median_ms')}")
    print(f"  Axis 3c       order median_ms : {axes.get('paper_order_path_latency_median_ms')}")
    print(f"  Axis 4        peak_rss_mb     : {axes.get('peak_rss_mb')}")


if __name__ == "__main__":
    main()
