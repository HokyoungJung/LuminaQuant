#!/usr/bin/env python3
"""Persistent round2 OOS search runner with resume/checkpoint support.

This script repeatedly executes strict 1s-fidelity strategy research runs,
checkpointing progress so it can resume from the last finished attempt.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
ROUND2_DIR = REPORTS_DIR / "round2"
LOGS_DIR = ROOT / "logs"
LOCKS_DIR = ROOT / ".omx" / "locks"
BOOTSTRAP_ENV_PATH = ROOT / "data" / "influxdb2" / "bootstrap.env"

TOPCAP_ALL = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "XAU/USDT:USDT",
    "XAG/USDT:USDT",
]

TOPCAP_CRYPTO = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT",
]


@dataclass(frozen=True)
class Profile:
    name: str
    strategy_set: str
    selection_mode: str
    timeframes: list[str]
    topcap_symbols: list[str]
    topcap_iters: int
    pair_iters: int
    ensemble_iters: int
    max_selected: int
    max_per_family: int
    max_per_timeframe: int
    max_runs: int
    manifest_candidates: list[Path]


PROFILES: list[Profile] = [
    Profile(
        name="all_robust",
        strategy_set="all",
        selection_mode="robust",
        timeframes=["1m", "5m", "15m", "1h", "4h"],
        topcap_symbols=TOPCAP_ALL,
        topcap_iters=180,
        pair_iters=140,
        ensemble_iters=3000,
        max_selected=48,
        max_per_family=16,
        max_per_timeframe=10,
        max_runs=20,
        manifest_candidates=[
            ROUND2_DIR / "worker1_factory_candidates_20260222T102257Z.json",
        ],
    ),
    Profile(
        name="crypto_robust",
        strategy_set="crypto-only",
        selection_mode="robust",
        timeframes=["1m", "5m", "15m", "1h"],
        topcap_symbols=TOPCAP_CRYPTO,
        topcap_iters=220,
        pair_iters=180,
        ensemble_iters=3500,
        max_selected=56,
        max_per_family=18,
        max_per_timeframe=12,
        max_runs=16,
        manifest_candidates=[
            ROUND2_DIR / "worker2_crypto_candidates_20260222T110542Z.json",
        ],
    ),
    Profile(
        name="crypto_val",
        strategy_set="crypto-only",
        selection_mode="val",
        timeframes=["1m", "5m", "15m", "1h"],
        topcap_symbols=TOPCAP_CRYPTO,
        topcap_iters=220,
        pair_iters=180,
        ensemble_iters=3500,
        max_selected=56,
        max_per_family=18,
        max_per_timeframe=12,
        max_runs=16,
        manifest_candidates=[
            ROUND2_DIR / "worker2_crypto_candidates_20260222T110542Z.json",
        ],
    ),
]


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    scale: float
    data_cache_max_entries: int
    data_cache_max_rows: int
    data_cache_scope: str
    ensemble_mode: str
    ensemble_max_candidates: int
    influx_query_chunk_days: int
    influx_agg_chunk_days: int


def _mem_available_gib() -> float:
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as file:
            for line in file:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    break
                kib = float(parts[1])
                return kib / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _scaled(value: int, scale: float, *, minimum: int) -> int:
    return max(int(minimum), round(float(value) * float(scale)))


def resolve_resource_profile(mode: str) -> ResourceProfile:
    token = str(mode or "auto").strip().lower()
    if token == "conservative":
        return ResourceProfile(
            name="conservative",
            scale=0.20,
            data_cache_max_entries=1,
            data_cache_max_rows=600_000,
            data_cache_scope="strategy",
            ensemble_mode="off",
            ensemble_max_candidates=0,
            influx_query_chunk_days=1,
            influx_agg_chunk_days=2,
        )
    if token == "balanced":
        return ResourceProfile(
            name="balanced",
            scale=0.60,
            data_cache_max_entries=2,
            data_cache_max_rows=1_800_000,
            data_cache_scope="strategy",
            ensemble_mode="off",
            ensemble_max_candidates=0,
            influx_query_chunk_days=1,
            influx_agg_chunk_days=3,
        )
    if token == "aggressive":
        return ResourceProfile(
            name="aggressive",
            scale=1.00,
            data_cache_max_entries=3,
            data_cache_max_rows=0,
            data_cache_scope="global",
            ensemble_mode="auto",
            ensemble_max_candidates=4,
            influx_query_chunk_days=2,
            influx_agg_chunk_days=5,
        )

    mem_available_gib = _mem_available_gib()
    if mem_available_gib <= 0.0:
        return ResourceProfile(
            name="auto-fallback",
            scale=0.50,
            data_cache_max_entries=2,
            data_cache_max_rows=1_500_000,
            data_cache_scope="strategy",
            ensemble_mode="off",
            ensemble_max_candidates=0,
            influx_query_chunk_days=1,
            influx_agg_chunk_days=3,
        )
    if mem_available_gib < 6.0:
        return ResourceProfile(
            name=f"auto-low-{mem_available_gib:.1f}GiB",
            scale=0.30,
            data_cache_max_entries=1,
            data_cache_max_rows=750_000,
            data_cache_scope="strategy",
            ensemble_mode="off",
            ensemble_max_candidates=0,
            influx_query_chunk_days=1,
            influx_agg_chunk_days=2,
        )
    if mem_available_gib < 10.0:
        return ResourceProfile(
            name=f"auto-medium-{mem_available_gib:.1f}GiB",
            scale=0.50,
            data_cache_max_entries=2,
            data_cache_max_rows=1_500_000,
            data_cache_scope="strategy",
            ensemble_mode="off",
            ensemble_max_candidates=0,
            influx_query_chunk_days=1,
            influx_agg_chunk_days=3,
        )
    if mem_available_gib < 14.0:
        return ResourceProfile(
            name=f"auto-balanced-{mem_available_gib:.1f}GiB",
            scale=0.70,
            data_cache_max_entries=2,
            data_cache_max_rows=2_200_000,
            data_cache_scope="strategy",
            ensemble_mode="auto",
            ensemble_max_candidates=3,
            influx_query_chunk_days=2,
            influx_agg_chunk_days=4,
        )
    return ResourceProfile(
        name=f"auto-high-{mem_available_gib:.1f}GiB",
        scale=1.00,
        data_cache_max_entries=3,
        data_cache_max_rows=0,
        data_cache_scope="global",
        ensemble_mode="auto",
        ensemble_max_candidates=4,
        influx_query_chunk_days=2,
        influx_agg_chunk_days=5,
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def ts_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs() -> None:
    ROUND2_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "success": False,
        "attempt": 0,
        "last_profile": None,
        "last_seed": None,
        "history": [],
        "best": None,
        "last_report": None,
        "message": "initialized",
    }


def sanitize_state_for_resume(state: dict[str, Any]) -> dict[str, Any]:
    """Convert stale in-progress history rows into interrupted rows on restart."""
    history = list(state.get("history") or [])
    touched = False
    now = utc_now()
    for row in history:
        if row.get("status") == "running":
            row["status"] = "interrupted"
            row.setdefault("ended_at", now)
            row.setdefault("error", "interrupted_before_completion")
            row.setdefault("return_code", None)
            touched = True
    if touched:
        state["history"] = history
        state["message"] = "resumed_from_interrupted_state"
        state["updated_at"] = now
    return state


def pick_manifest(profile: Profile) -> Path | None:
    for path in profile.manifest_candidates:
        if path.exists():
            return path
    return None


def list_reports_after(start_epoch: float) -> list[Path]:
    candidates = sorted(REPORTS_DIR.glob("strategy_team_research_oos_*.json"), key=lambda p: p.stat().st_mtime)
    return [path for path in candidates if path.stat().st_mtime >= start_epoch]


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = list(report.get("selected_team") or [])

    best_by_return: dict[str, Any] | None = None
    best_by_excess: dict[str, Any] | None = None
    pass_count = 0

    for row in selected:
        name = row.get("name")
        timeframe = row.get("strategy_timeframe")
        symbols = row.get("symbols") or []
        oos = row.get("oos") or {}
        hurdle_oos = (row.get("hurdle_fields") or {}).get("oos") or {}

        oos_return = _as_float(oos.get("return"))
        excess = _as_float(hurdle_oos.get("excess_return"))
        passed = bool(hurdle_oos.get("pass") is True)

        item = {
            "name": name,
            "strategy_timeframe": timeframe,
            "base_timeframe": row.get("base_timeframe") or report.get("base_timeframe"),
            "symbols": symbols,
            "oos_return": oos_return,
            "oos_trades": oos.get("trades"),
            "oos_sharpe": oos.get("sharpe"),
            "hurdle_return": _as_float(hurdle_oos.get("hurdle_return")),
            "benchmark_return": _as_float(hurdle_oos.get("benchmark_return")),
            "floor_return": _as_float(hurdle_oos.get("floor_return")),
            "excess_return": excess,
            "pass": passed,
            "report_path": str(path),
        }

        if passed:
            pass_count += 1

        if oos_return is not None and (
            best_by_return is None or (best_by_return.get("oos_return") or -1e18) < oos_return
        ):
            best_by_return = item
        if excess is not None and (
            best_by_excess is None or (best_by_excess.get("excess_return") or -1e18) < excess
        ):
            best_by_excess = item

    return {
        "report_path": str(path),
        "generated_at": report.get("generated_at"),
        "base_timeframe": report.get("base_timeframe"),
        "timeframes": report.get("timeframes"),
        "selected_team_count": int(report.get("selected_team_count") or len(selected)),
        "pass_count": pass_count,
        "best_by_oos_return": best_by_return,
        "best_by_excess": best_by_excess,
    }


def start_influx(influx_log_path: Path, env: dict[str, str]) -> subprocess.Popen[str] | None:
    influx_url = str(
        env.get("INFLUX_URL")
        or env.get("LQ__STORAGE__INFLUX_URL")
        or "http://127.0.0.1:8086"
    ).strip()
    health_url = f"{influx_url.rstrip('/')}/health"

    def _health_ok() -> bool:
        try:
            req = urllib.request.Request(url=health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:  # nosec B310
                return int(getattr(resp, "status", 200)) < 500
        except Exception:
            return False

    if _health_ok():
        return None

    cmd = [
        str(ROOT / ".local" / "influx" / "influxd"),
        "--bolt-path",
        "data/influxdb2/meta/influxd.bolt",
        "--engine-path",
        "data/influxdb2/engine",
        "--sqlite-path",
        "data/influxdb2/meta/influxd.sqlite",
        "--http-bind-address",
        ":8086",
        "--log-level",
        "error",
        "--query-concurrency",
        "2",
        "--query-queue-size",
        "16",
        "--query-initial-memory-bytes",
        "134217728",
        "--query-memory-bytes",
        "536870912",
        "--query-max-memory-bytes",
        "1610612736",
    ]
    proc_env = os.environ.copy()
    proc_env.update(env)
    proc_env["GOMEMLIMIT"] = "3GiB"
    proc_env["GOGC"] = "50"

    influx_log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = influx_log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=proc_env,
        stdout=fd,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Keep handle attached to proc so caller can close after wait.
    proc._round2_log_fd = fd  # type: ignore[attr-defined]
    for _ in range(20):
        if proc.poll() is not None:
            break
        if _health_ok():
            return proc
        time.sleep(0.5)
    if proc.poll() is not None:
        fd.close()
        raise RuntimeError(f"influxd exited during startup (rc={proc.returncode})")
    return proc


def stop_influx(proc: subprocess.Popen[str] | None, timeout_sec: int = 20) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        fd = getattr(proc, "_round2_log_fd", None)
        if fd:
            try:
                fd.close()
            except Exception:
                pass


def _process_tree_rss_kib(root_pid: int) -> int:
    """Return RSS(KiB) sum for a process tree rooted at root_pid."""
    if int(root_pid) <= 0:
        return 0
    try:
        text = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,rss="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0

    children: dict[int, list[int]] = {}
    rss_map: dict[int, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kib = int(parts[2])
        except Exception:
            continue
        children.setdefault(ppid, []).append(pid)
        rss_map[pid] = max(0, rss_kib)

    total = 0
    stack = [int(root_pid)]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += int(rss_map.get(pid, 0))
        stack.extend(children.get(pid, []))
    return int(total)


def _terminate_process_tree(root_pid: int, *, grace_sec: float = 5.0) -> None:
    if int(root_pid) <= 0:
        return
    try:
        os.killpg(int(root_pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(int(root_pid), signal.SIGTERM)
        except Exception:
            return
    time.sleep(max(0.0, float(grace_sec)))
    try:
        os.killpg(int(root_pid), signal.SIGKILL)
    except Exception:
        try:
            os.kill(int(root_pid), signal.SIGKILL)
        except Exception:
            pass


def build_research_command(
    profile: Profile,
    seed: int,
    manifest: Path | None,
    influx_env: dict[str, str],
    resource: ResourceProfile,
) -> list[str]:
    strategy_set = profile.strategy_set
    timeframes = list(profile.timeframes)
    if resource.scale <= 0.35:
        strategy_set = "crypto-only"
        timeframes = [tf for tf in timeframes if tf in {"1m"}] or timeframes[:1]

    topcap_iters = _scaled(profile.topcap_iters, resource.scale, minimum=40)
    pair_iters = _scaled(profile.pair_iters, resource.scale, minimum=24)
    ensemble_iters = _scaled(profile.ensemble_iters, resource.scale, minimum=300)
    max_selected = _scaled(profile.max_selected, resource.scale, minimum=8)
    max_per_family = _scaled(profile.max_per_family, resource.scale, minimum=3)
    max_per_timeframe = _scaled(profile.max_per_timeframe, resource.scale, minimum=2)
    max_runs = _scaled(profile.max_runs, resource.scale, minimum=2)
    if resource.scale <= 0.35:
        topcap_cap = 1
        topcap_iters = min(topcap_iters, 14)
        pair_iters = min(pair_iters, 12)
        ensemble_iters = min(ensemble_iters, 200)
        max_runs = min(max_runs, 1)
    elif resource.scale <= 0.60:
        topcap_cap = 5
    else:
        topcap_cap = len(profile.topcap_symbols)
    topcap_symbols = list(profile.topcap_symbols[:topcap_cap])

    prefetch_flag = (
        "--prefetch-strategy-data"
        if float(resource.scale) > 0.35
        else "--no-prefetch-strategy-data"
    )

    cmd = [
        sys.executable,
        "scripts/run_strategy_team_research.py",
        "--db-path",
        "data/lq_market.sqlite3",
        "--backend",
        "influxdb",
        "--exchange",
        "binance",
        "--market-type",
        "future",
        "--mode",
        "oos",
        "--strategy-set",
        strategy_set,
        "--base-timeframe",
        "1s",
        "--base-timeframes",
        "1s",
        "--timeframes",
        *timeframes,
        "--seeds",
        str(seed),
        "--train-days",
        "365",
        "--val-days",
        "30",
        "--oos-days",
        "30",
        "--min-insample-days",
        "365",
        "--annual-return-floor",
        "0.26824",
        "--benchmark-symbol",
        "BTC/USDT",
        "--topcap-iters",
        str(topcap_iters),
        "--pair-iters",
        str(pair_iters),
        "--ensemble-iters",
        str(ensemble_iters),
        "--search-engine",
        "random",
        "--selection-mode",
        profile.selection_mode,
        "--topcap-min-coverage-days",
        "30",
        "--topcap-min-row-ratio",
        "0.25",
        "--topcap-min-symbols",
        "1" if len(topcap_symbols) <= 1 else ("2" if len(topcap_symbols) <= 3 else "4"),
        "--ensemble-min-bars",
        "20",
        "--ensemble-min-oos-trades",
        "1",
        "--xau-xag-ensemble-min-overlap-days",
        "120",
        "--xau-xag-ensemble-min-oos-trades",
        "2",
        "--max-selected",
        str(max_selected),
        "--max-per-family",
        str(max_per_family),
        "--max-per-timeframe",
        str(max_per_timeframe),
        "--max-runs",
        str(max_runs),
        "--data-cache-max-entries",
        str(resource.data_cache_max_entries),
        "--data-cache-max-rows",
        str(resource.data_cache_max_rows),
        "--data-cache-scope",
        resource.data_cache_scope,
        "--ensemble-mode",
        resource.ensemble_mode,
        "--ensemble-max-candidates",
        str(resource.ensemble_max_candidates),
        prefetch_flag,
        "--influx-1s-query-chunk-days",
        str(resource.influx_query_chunk_days),
        "--influx-1s-agg-chunk-days",
        str(resource.influx_agg_chunk_days),
        "--child-log-dir",
        "logs/team_research",
        "--influx-token-env",
        "INFLUXDB_TOKEN",
        "--topcap-symbols",
        *topcap_symbols,
    ]

    influx_url = influx_env.get("INFLUX_URL", "").strip()
    influx_org = influx_env.get("INFLUX_ORG", "").strip()
    influx_bucket = influx_env.get("INFLUX_BUCKET", "").strip()
    if influx_url:
        cmd.extend(["--influx-url", influx_url])
    if influx_org:
        cmd.extend(["--influx-org", influx_org])
    if influx_bucket:
        cmd.extend(["--influx-bucket", influx_bucket])

    if manifest is not None:
        cmd.extend(["--candidate-manifest", str(manifest)])
    return cmd


def run_attempt(
    *,
    attempt_no: int,
    profile: Profile,
    seed: int,
    state: dict[str, Any],
    state_path: Path,
    attempt_timeout_sec: int,
    influx_env: dict[str, str],
    resource: ResourceProfile,
    max_attempt_rss_kib: int,
) -> dict[str, Any]:
    ts = ts_compact()
    attempt_prefix = f"round2_attempt{attempt_no:04d}_{profile.name}_{seed}_{ts}"
    run_log = LOGS_DIR / f"{attempt_prefix}.log"
    influx_log = LOGS_DIR / f"{attempt_prefix}_influxd.log"

    manifest = pick_manifest(profile)
    cmd = build_research_command(profile, seed, manifest, influx_env, resource)

    started_at = utc_now()
    start_epoch = time.time() - 1.0

    attempt_info: dict[str, Any] = {
        "attempt": attempt_no,
        "profile": profile.name,
        "seed": seed,
        "started_at": started_at,
        "status": "running",
        "run_log": str(run_log),
        "influx_log": str(influx_log),
        "manifest": str(manifest) if manifest else None,
        "resource_profile": {
            "name": resource.name,
            "scale": resource.scale,
            "data_cache_max_entries": resource.data_cache_max_entries,
            "data_cache_max_rows": resource.data_cache_max_rows,
            "data_cache_scope": resource.data_cache_scope,
            "ensemble_mode": resource.ensemble_mode,
            "ensemble_max_candidates": resource.ensemble_max_candidates,
            "influx_query_chunk_days": resource.influx_query_chunk_days,
            "influx_agg_chunk_days": resource.influx_agg_chunk_days,
        },
        "command": cmd,
        "max_attempt_rss_kib": int(max_attempt_rss_kib),
    }

    state["attempt"] = attempt_no
    state["last_profile"] = profile.name
    state["last_seed"] = seed
    state["updated_at"] = utc_now()
    state["message"] = f"attempt {attempt_no} running"
    state.setdefault("history", []).append(attempt_info)
    atomic_write_json(state_path, state)

    proc_env = os.environ.copy()
    proc_env.update(influx_env)
    proc_env["LQ__STORAGE__BACKEND"] = "influxdb"
    proc_env["LQ__STORAGE__INFLUX_TIMEOUT_SEC"] = "900"
    proc_env["LQ__STORAGE__INFLUX_1S_QUERY_CHUNK_DAYS"] = str(resource.influx_query_chunk_days)
    proc_env["LQ__STORAGE__INFLUX_1S_AGG_CHUNK_DAYS"] = str(resource.influx_agg_chunk_days)
    if resource.scale <= 0.35:
        proc_env["LQ__STORAGE__INFLUX_1S_QUERY_CHUNK_HOURS"] = "2"
        proc_env["LQ__STORAGE__INFLUX_1S_AGG_CHUNK_HOURS"] = "6"
    else:
        proc_env.pop("LQ__STORAGE__INFLUX_1S_QUERY_CHUNK_HOURS", None)
        proc_env.pop("LQ__STORAGE__INFLUX_1S_AGG_CHUNK_HOURS", None)

    influx_proc: subprocess.Popen[str] | None = None
    rc = None
    error_text = None
    peak_rss_kib = 0
    try:
        influx_proc = start_influx(influx_log, proc_env)
        if influx_proc is None:
            attempt_info["influx_startup"] = "reused_existing"
        else:
            attempt_info["influx_startup"] = "started_local"
            time.sleep(3)

        with run_log.open("a", encoding="utf-8") as log_fd:
            log_fd.write(f"[{utc_now()}] START attempt={attempt_no} profile={profile.name} seed={seed}\n")
            log_fd.write(" ".join(cmd) + "\n")
            log_fd.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=proc_env,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            started_monotonic = time.monotonic()
            last_probe_monotonic = started_monotonic
            while True:
                rc_now = proc.poll()
                if rc_now is not None:
                    rc = int(rc_now)
                    break

                elapsed_sec = int(max(0.0, time.monotonic() - started_monotonic))
                tree_rss_kib = int(_process_tree_rss_kib(int(proc.pid)))
                peak_rss_kib = max(int(peak_rss_kib), int(tree_rss_kib))

                now_monotonic = time.monotonic()
                if now_monotonic - last_probe_monotonic >= 20.0:
                    log_fd.write(
                        f"[{utc_now()}] MONITOR elapsed={elapsed_sec}s rss_kib={tree_rss_kib} peak_rss_kib={peak_rss_kib}\n"
                    )
                    log_fd.flush()
                    last_probe_monotonic = now_monotonic

                if int(max_attempt_rss_kib) > 0 and tree_rss_kib > int(max_attempt_rss_kib):
                    error_text = (
                        f"rss_limit_exceeded>{int(max_attempt_rss_kib)}kib current={int(tree_rss_kib)}kib"
                    )
                    log_fd.write(f"[{utc_now()}] ABORT {error_text}\n")
                    log_fd.flush()
                    _terminate_process_tree(int(proc.pid))
                    rc = 137
                    break

                if elapsed_sec > int(attempt_timeout_sec):
                    error_text = f"timeout>{attempt_timeout_sec}s"
                    log_fd.write(f"[{utc_now()}] ABORT {error_text}\n")
                    log_fd.flush()
                    _terminate_process_tree(int(proc.pid))
                    rc = 124
                    break

                time.sleep(2.0)

            if proc.poll() is None:
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
            log_fd.write(f"[{utc_now()}] END return_code={rc}\n")
            log_fd.flush()
    except Exception as exc:
        rc = 1
        error_text = f"exception: {type(exc).__name__}: {exc}"
    finally:
        stop_influx(influx_proc)

    report_candidates = list_reports_after(start_epoch)
    report_path = report_candidates[-1] if report_candidates else None
    summary: dict[str, Any] | None = None
    if report_path is not None:
        try:
            summary = summarize_report(report_path)
        except Exception as exc:
            summary = {
                "report_path": str(report_path),
                "error": f"failed_to_parse_report: {type(exc).__name__}: {exc}",
            }

    ended_at = utc_now()
    attempt_info.update(
        {
            "ended_at": ended_at,
            "status": "completed" if rc == 0 else "failed",
            "return_code": rc,
            "error": error_text,
            "peak_rss_kib": int(peak_rss_kib),
            "report_path": str(report_path) if report_path else None,
            "report_summary": summary,
        }
    )

    if summary is not None:
        state["last_report"] = summary.get("report_path")

        current_best = state.get("best")
        candidate_best = summary.get("best_by_excess") if isinstance(summary, dict) else None
        if isinstance(candidate_best, dict):
            candidate_excess = _as_float(candidate_best.get("excess_return"))
            current_excess = _as_float((current_best or {}).get("excess_return"))
            if current_best is None or (
                candidate_excess is not None
                and (current_excess is None or candidate_excess > current_excess)
            ):
                state["best"] = candidate_best

        pass_count = int(summary.get("pass_count") or 0) if isinstance(summary, dict) else 0
        if pass_count > 0 and isinstance(summary, dict):
            state["success"] = True
            state["status"] = "success"
            state["message"] = f"strict hurdle pass found on attempt {attempt_no}"
    else:
        state["message"] = f"attempt {attempt_no} finished without new report"

    state["updated_at"] = utc_now()
    atomic_write_json(state_path, state)
    return attempt_info


def should_stop(state: dict[str, Any], max_attempts: int) -> bool:
    if bool(state.get("success")):
        return True
    if max_attempts > 0 and int(state.get("attempt") or 0) >= max_attempts:
        state["status"] = "max_attempts_reached"
        state["message"] = "stopped at max attempts"
        return True
    return False


def write_markdown_summary(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# Round2 Resume Runner Summary",
        "",
        f"- updated_at: {state.get('updated_at')}",
        f"- status: {state.get('status')}",
        f"- success: {state.get('success')}",
        f"- attempt: {state.get('attempt')}",
        f"- message: {state.get('message')}",
        f"- last_report: {state.get('last_report')}",
        "",
        "## Best Candidate (by excess)",
        "",
        "```json",
        json.dumps(state.get("best"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Last 10 Attempts",
        "",
    ]

    history = list(state.get("history") or [])
    for row in history[-10:]:
        command_tokens = row.get("command") if isinstance(row.get("command"), list) else []
        command_text = " ".join(str(token) for token in command_tokens)
        lines.append(
            "- "
            f"attempt={row.get('attempt')} "
            f"profile={row.get('profile')} "
            f"seed={row.get('seed')} "
            f"status={row.get('status')} "
            f"rc={row.get('return_code')} "
            f"report={row.get('report_path')}"
        )
        lines.append(f"  - run_log={row.get('run_log')}")
        lines.append(f"  - influx_log={row.get('influx_log')}")
        if command_text:
            lines.append(f"  - command=`{command_text}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent/resumable strict OOS runner.")
    parser.add_argument("--state-path", default=str(ROUND2_DIR / "round2_resume_state.json"))
    parser.add_argument("--summary-path", default=str(ROUND2_DIR / "round2_resume_summary.md"))
    parser.add_argument("--lock-path", default=str(LOCKS_DIR / "influx_round2_resume.lock"))
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means infinite loop until success.")
    parser.add_argument("--sleep-seconds", type=int, default=15)
    parser.add_argument("--seed-base", type=int, default=20260222)
    parser.add_argument("--attempt-timeout-sec", type=int, default=7200)
    parser.add_argument(
        "--max-attempt-rss-kib",
        type=int,
        default=3_500_000,
        help="Abort a running attempt when process-tree RSS exceeds this KiB threshold (0 disables).",
    )
    parser.add_argument(
        "--resource-mode",
        choices=["auto", "conservative", "balanced", "aggressive"],
        default="auto",
        help="Auto scales iterations/cache/chunking to fit available memory.",
    )
    args = parser.parse_args()

    ensure_dirs()

    state_path = Path(args.state_path)
    summary_path = Path(args.summary_path)
    lock_path = Path(args.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    influx_env = load_env_file(BOOTSTRAP_ENV_PATH)
    if not influx_env:
        print(f"[WARN] missing bootstrap env: {BOOTSTRAP_ENV_PATH}", file=sys.stderr)
    resource = resolve_resource_profile(str(args.resource_mode))
    print(
        "[RESOURCE] "
        f"name={resource.name} scale={resource.scale:.2f} "
        f"cache={resource.data_cache_max_entries}/{resource.data_cache_max_rows} "
        f"ensemble={resource.ensemble_mode} "
        f"chunks={resource.influx_query_chunk_days}/{resource.influx_agg_chunk_days}"
    )

    stop_requested = {"value": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        stop_requested["value"] = True
        print(f"[INFO] signal {signum} received, stopping after current attempt.")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    state = sanitize_state_for_resume(load_state(state_path))
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_write_json(state_path, state)
    write_markdown_summary(state, summary_path)

    with lock_path.open("w", encoding="utf-8") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

        while True:
            if stop_requested["value"]:
                state["status"] = "stopped_by_signal"
                state["message"] = "received stop signal"
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                write_markdown_summary(state, summary_path)
                return 0

            if should_stop(state, args.max_attempts):
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                write_markdown_summary(state, summary_path)
                return 0

            next_attempt = int(state.get("attempt") or 0) + 1
            profile = PROFILES[(next_attempt - 1) % len(PROFILES)]
            seed = int(args.seed_base) + next_attempt

            run_attempt(
                attempt_no=next_attempt,
                profile=profile,
                seed=seed,
                state=state,
                state_path=state_path,
                attempt_timeout_sec=int(args.attempt_timeout_sec),
                influx_env=influx_env,
                resource=resource,
                max_attempt_rss_kib=int(args.max_attempt_rss_kib),
            )

            write_markdown_summary(state, summary_path)

            if should_stop(state, args.max_attempts):
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                write_markdown_summary(state, summary_path)
                return 0

            sleep_sec = max(0, int(args.sleep_seconds))
            if sleep_sec > 0:
                time.sleep(sleep_sec)


if __name__ == "__main__":
    raise SystemExit(main())
