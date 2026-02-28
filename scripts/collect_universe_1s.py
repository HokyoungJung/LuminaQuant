"""Chunked/resumable Binance universe 1s collector with GPU/VRAM telemetry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.compute_engine import GPUNotAvailableError, select_engine
from lumina_quant.config import BaseConfig, LiveConfig
from lumina_quant.data_collector import auto_collect_market_data
from lumina_quant.data_sync import parse_timestamp_input

MS_PER_DAY = 86_400_000


def _dt_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC)


def _format_ms(value: int) -> str:
    return _dt_from_ms(value).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _chunk_symbols(symbols: list[str], per_batch: int) -> list[list[str]]:
    size = max(1, int(per_batch))
    return [symbols[idx : idx + size] for idx in range(0, len(symbols), size)]


def _chunk_windows(since_ms: int, until_ms: int, chunk_days: int) -> list[tuple[int, int]]:
    if until_ms < since_ms:
        until_ms = since_ms
    span_ms = max(1, int(chunk_days)) * MS_PER_DAY
    windows: list[tuple[int, int]] = []
    cursor = int(since_ms)
    while cursor <= int(until_ms):
        end_ms = min(int(until_ms), cursor + span_ms - 1000)
        windows.append((cursor, end_ms))
        cursor = end_ms + 1000
    return windows


def _gpu_vram_snapshot() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
    except Exception as exc:
        return f"unavailable ({exc})"

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"exit={proc.returncode}"
        return f"unavailable ({err})"

    rows = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not rows:
        return "unavailable (no gpu rows)"
    return " | ".join(rows)


def _print_compute_engine_status() -> None:
    mode = str(os.getenv("LQ_GPU_MODE", str(BaseConfig.GPU_MODE))).strip()
    vram_gb = str(os.getenv("LQ_GPU_VRAM_GB", str(BaseConfig.GPU_VRAM_GB))).strip()
    device = str(os.getenv("LQ_GPU_DEVICE", "")).strip() or None
    try:
        engine = select_engine(
            mode=mode,
            device=device,
            vram_gb=vram_gb,
            verbose=True,
        )
        print(
            "[GPU] "
            f"requested_mode={engine.requested_mode} resolved_engine={engine.resolved_engine} "
            f"device={engine.device} reason={engine.reason}"
        )
    except GPUNotAvailableError as exc:
        print(f"[GPU] unavailable for requested mode '{mode}': {exc}")
    except Exception as exc:
        print(f"[GPU] probe failed: {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect 1s Binance futures data in resumable chunks to avoid OOM/terminal shutdown."
    )
    parser.add_argument("--symbols", nargs="+", default=list(BaseConfig.SYMBOLS))
    parser.add_argument("--timeframe", default="1s")
    parser.add_argument("--db-path", default=BaseConfig.MARKET_DATA_PARQUET_PATH)
    parser.add_argument("--exchange-id", default=BaseConfig.MARKET_DATA_EXCHANGE)
    parser.add_argument("--market-type", default=LiveConfig.MARKET_TYPE, choices=["spot", "future"])
    parser.add_argument("--backend", default=BaseConfig.STORAGE_BACKEND)
    parser.add_argument("--since", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--until", default="")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--symbols-per-batch", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=100000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--base-wait-sec", type=float, default=0.5)
    parser.add_argument("--sleep-between-batches", type=float, default=0.2)
    parser.add_argument("--state-file", default="logs/collect_universe_1s_state.json")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing state and start from first chunk (does not delete state file).",
    )
    parser.add_argument(
        "--no-gpu-snapshot",
        action="store_true",
        help="Disable per-batch nvidia-smi snapshot logs.",
    )
    return parser


def _parse_since_until(args: argparse.Namespace, *, existing_state: dict[str, Any] | None) -> tuple[int, int]:
    since_ms = parse_timestamp_input(args.since)
    if since_ms is None:
        raise ValueError("--since must be a valid ISO8601 or unix timestamp.")

    until_text = str(args.until or "").strip()
    if until_text:
        until_ms = parse_timestamp_input(until_text)
        if until_ms is None:
            raise ValueError("--until must be a valid ISO8601 or unix timestamp.")
        return int(since_ms), int(until_ms)

    if existing_state is not None:
        saved_until = ((existing_state.get("plan") or {}).get("until_ms"))
        if saved_until is not None:
            return int(since_ms), int(saved_until)

    return int(since_ms), int(datetime.now(UTC).timestamp() * 1000)


def _plan_signature(
    *,
    args: argparse.Namespace,
    since_ms: int,
    until_ms: int,
    window_count: int,
    batch_count: int,
) -> dict[str, Any]:
    return {
        "symbols": [str(item) for item in list(args.symbols)],
        "timeframe": str(args.timeframe),
        "db_path": str(args.db_path),
        "exchange_id": str(args.exchange_id),
        "market_type": str(args.market_type),
        "backend": str(args.backend),
        "since_ms": int(since_ms),
        "until_ms": int(until_ms),
        "chunk_days": int(args.chunk_days),
        "symbols_per_batch": int(args.symbols_per_batch),
        "window_count": int(window_count),
        "batch_count": int(batch_count),
    }


def main() -> int:
    args = _build_parser().parse_args()
    state_path = Path(str(args.state_file)).resolve()
    resume_enabled = not bool(args.no_resume)

    if args.reset_state and state_path.exists():
        state_path.unlink()

    existing_state = _read_json(state_path) if (resume_enabled and state_path.exists()) else None
    since_ms, until_ms = _parse_since_until(args, existing_state=existing_state)

    symbol_batches = _chunk_symbols([str(item) for item in list(args.symbols)], args.symbols_per_batch)
    windows = _chunk_windows(since_ms, until_ms, args.chunk_days)
    total_tasks = len(symbol_batches) * len(windows)
    plan = _plan_signature(
        args=args,
        since_ms=since_ms,
        until_ms=until_ms,
        window_count=len(windows),
        batch_count=len(symbol_batches),
    )

    if existing_state is not None:
        saved_plan = dict(existing_state.get("plan") or {})
        if saved_plan != plan:
            raise RuntimeError(
                "State file plan does not match current arguments. "
                "Use --reset-state or pass matching --since/--until/chunk options."
            )
        state = dict(existing_state)
    else:
        state = {
            "version": 1,
            "plan": plan,
            "next_window_index": 0,
            "next_batch_index": 0,
            "completed_tasks": 0,
            "total_tasks": int(total_tasks),
            "fetched_rows_total": 0,
            "upserted_rows_total": 0,
            "last_error": "",
            "started_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_json(state_path, state)

    print(
        "[PLAN] "
        f"symbols={len(plan['symbols'])} "
        f"symbol_batches={plan['batch_count']} "
        f"time_windows={plan['window_count']} "
        f"tasks={state['total_tasks']} "
        f"range={_format_ms(since_ms)}..{_format_ms(until_ms)}"
    )
    print(f"[STATE] {state_path}")
    _print_compute_engine_status()

    if bool(args.dry_run):
        print("[DRY-RUN] planned only, no data collection executed.")
        return 0

    for window_idx in range(int(state.get("next_window_index", 0)), len(windows)):
        batch_start_idx = (
            int(state.get("next_batch_index", 0))
            if window_idx == int(state.get("next_window_index", 0))
            else 0
        )
        range_start_ms, range_end_ms = windows[window_idx]

        for batch_idx in range(batch_start_idx, len(symbol_batches)):
            symbol_batch = symbol_batches[batch_idx]
            task_no = int(state.get("completed_tasks", 0)) + 1
            print(
                f"[TASK {task_no}/{state['total_tasks']}] "
                f"window={window_idx + 1}/{len(windows)} "
                f"batch={batch_idx + 1}/{len(symbol_batches)} "
                f"symbols={','.join(symbol_batch)} "
                f"since={_format_ms(range_start_ms)} until={_format_ms(range_end_ms)}"
            )
            if not bool(args.no_gpu_snapshot):
                print(f"[VRAM] {_gpu_vram_snapshot()}")

            try:
                stats = auto_collect_market_data(
                    symbol_list=list(symbol_batch),
                    timeframe=str(args.timeframe),
                    db_path=str(args.db_path),
                    exchange_id=str(args.exchange_id),
                    market_type=str(args.market_type),
                    since_dt=_dt_from_ms(range_start_ms),
                    until_dt=_dt_from_ms(range_end_ms),
                    api_key=str(LiveConfig.BINANCE_API_KEY or ""),
                    secret_key=str(LiveConfig.BINANCE_SECRET_KEY or ""),
                    testnet=bool(args.testnet),
                    limit=max(1, int(args.limit)),
                    max_batches=max(1, int(args.max_batches)),
                    retries=max(0, int(args.retries)),
                    base_wait_sec=max(0.0, float(args.base_wait_sec)),
                    backend=str(args.backend),
                )
            except Exception as exc:
                state["last_error"] = (
                    f"{datetime.now(UTC).isoformat()} | window={window_idx} batch={batch_idx} | {exc}"
                )
                state["updated_at"] = datetime.now(UTC).isoformat()
                _write_json(state_path, state)
                raise

            fetched_rows = sum(int(item.get("fetched_rows", 0) or 0) for item in stats)
            upserted_rows = sum(int(item.get("upserted_rows", 0) or 0) for item in stats)
            state["fetched_rows_total"] = int(state.get("fetched_rows_total", 0)) + int(fetched_rows)
            state["upserted_rows_total"] = int(state.get("upserted_rows_total", 0)) + int(upserted_rows)
            state["completed_tasks"] = int(state.get("completed_tasks", 0)) + 1

            next_window = window_idx
            next_batch = batch_idx + 1
            if next_batch >= len(symbol_batches):
                next_window += 1
                next_batch = 0
            state["next_window_index"] = int(next_window)
            state["next_batch_index"] = int(next_batch)
            state["last_error"] = ""
            state["updated_at"] = datetime.now(UTC).isoformat()
            _write_json(state_path, state)

            print(
                f"[DONE] fetched={fetched_rows} upserted={upserted_rows} "
                f"accum_fetched={state['fetched_rows_total']} "
                f"accum_upserted={state['upserted_rows_total']}"
            )
            sleep_s = max(0.0, float(args.sleep_between_batches))
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    state["completed_at"] = datetime.now(UTC).isoformat()
    state["updated_at"] = state["completed_at"]
    _write_json(state_path, state)
    print(
        "[COMPLETE] "
        f"tasks={state['completed_tasks']} fetched={state['fetched_rows_total']} "
        f"upserted={state['upserted_rows_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
