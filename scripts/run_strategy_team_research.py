"""High-throughput strategy-team research orchestrator.

Runs many OOS searches across timeframes/seeds, aggregates candidate sleeves,
then selects a diversified strategy team for portfolio construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_TOP10_PLUS_METALS: tuple[str, ...] = (
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
)

_TF_SECONDS: dict[str, int] = {
    "1s": 1,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_saved_report_path(output: str) -> Path | None:
    marker = "Saved report:"
    for line in output.splitlines()[::-1]:
        if marker in line:
            text = line.split(marker, 1)[1].strip()
            if text:
                return Path(text)
    return None


def _read_log_tail(path: Path, *, max_bytes: int = 65536) -> str:
    if not path.exists():
        return ""
    data = b""
    with path.open("rb") as file:
        file.seek(0, 2)
        size = file.tell()
        if size <= 0:
            return ""
        file.seek(max(0, size - int(max_bytes)))
        data = file.read()
    return data.decode("utf-8", errors="ignore")


def _extract_saved_report_path_from_log(path: Path) -> Path | None:
    tail = _read_log_tail(path, max_bytes=131072)
    return _extract_saved_report_path(tail)


def _normalize_timeframe(value: str) -> str:
    return str(value).strip().lower()


def _timeframe_seconds(value: str) -> int:
    token = _normalize_timeframe(value)
    return int(_TF_SECONDS.get(token, 0))


def _enforce_1s_base_timeframe(value: str) -> str:
    token = _normalize_timeframe(value)
    if token != "1s":
        print(f"[WARN] base-timeframe '{token}' overridden to '1s' for all backtests.")
    return "1s"


def _eligible_base_timeframes(strategy_timeframe: str, base_candidates: list[str]) -> list[str]:
    strategy_seconds = _timeframe_seconds(strategy_timeframe)
    if strategy_seconds <= 0:
        return []
    out: list[str] = []
    for raw in base_candidates:
        token = _normalize_timeframe(raw)
        base_seconds = _timeframe_seconds(token)
        if base_seconds <= 0:
            continue
        if base_seconds <= strategy_seconds and token not in out:
            out.append(token)
    return out


def _candidate_score(candidate: dict, mode: str = "oos") -> float:
    hurdle_key = "val" if str(mode).strip().lower() == "live" else "oos"
    fields = ((candidate.get("hurdle_fields") or {}).get(hurdle_key)) or {}
    score = _safe_float(fields.get("score"), -1_000_000.0)
    excess = _safe_float(fields.get("excess_return"), -1_000_000.0)
    passed = bool(fields.get("pass", False))
    if passed:
        return score
    return -1_000_000.0 + excess


def _family_name(candidate_name: str) -> str:
    token = str(candidate_name).strip().lower()
    if token.startswith(("pair_", "lag_convergence", "mean_reversion_std", "vwap_reversion")):
        return "alpha_market_neutral"
    if token.startswith(("topcap_tsmom", "rolling_breakout", "rsi_", "moving_average")):
        return "trend_overlay"
    return "other"


def _candidate_identity(candidate: dict, timeframe: str) -> str:
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    payload = {
        "name": str(candidate.get("name", "")),
        "timeframe": str(timeframe),
        "symbols": list(candidate.get("symbols") or []),
        "params": params,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _select_diversified_team(
    candidates: list[dict],
    *,
    mode: str,
    max_total: int,
    max_per_family: int,
    max_per_timeframe: int,
) -> list[dict]:
    ranked = sorted(candidates, key=lambda row: _candidate_score(row, mode), reverse=True)
    selected: list[dict] = []
    family_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    seen_ids: set[str] = set()

    for row in ranked:
        if len(selected) >= int(max_total):
            break
        timeframe = str(row.get("strategy_timeframe", "")).strip().lower()
        ident = str(row.get("identity", "")).strip()
        if ident and ident in seen_ids:
            continue

        family = _family_name(str(row.get("name", "")))
        if family_counts.get(family, 0) >= int(max_per_family):
            continue
        if timeframe_counts.get(timeframe, 0) >= int(max_per_timeframe):
            continue

        selected.append(row)
        if ident:
            seen_ids.add(ident)
        family_counts[family] = family_counts.get(family, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1

    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a large strategy-team research report.")
    parser.add_argument("--db-path", default="data/lq_market.sqlite3")
    parser.add_argument("--backend", default="influxdb", help="Storage backend override (sqlite|influxdb).")
    parser.add_argument("--influx-url", default="")
    parser.add_argument("--influx-org", default="")
    parser.add_argument("--influx-bucket", default="")
    parser.add_argument("--influx-token", default="")
    parser.add_argument("--influx-token-env", default="INFLUXDB_TOKEN")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--base-timeframe", default="1s")
    parser.add_argument(
        "--base-timeframes",
        nargs="+",
        default=["1s"],
        help="Fallback base timeframe candidates; must be <= strategy timeframe.",
    )
    parser.add_argument("--market-type", choices=["spot", "future"], default="future")
    parser.add_argument("--mode", choices=["oos", "live"], default="oos")
    parser.add_argument(
        "--strategy-set", choices=["all", "crypto-only", "xau-xag-only"], default="all"
    )

    parser.add_argument(
        "--timeframes", nargs="+", default=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260220, 20260221, 20260222])
    parser.add_argument("--topcap-symbols", nargs="+", default=list(DEFAULT_TOP10_PLUS_METALS))

    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--oos-days", type=int, default=30)
    parser.add_argument("--min-insample-days", type=int, default=365)
    parser.add_argument("--annual-return-floor", type=float, default=0.10)
    parser.add_argument("--benchmark-symbol", default="BTC/USDT")

    parser.add_argument("--topcap-iters", type=int, default=320)
    parser.add_argument("--pair-iters", type=int, default=260)
    parser.add_argument("--ensemble-iters", type=int, default=5000)
    parser.add_argument("--search-engine", choices=["optuna", "random"], default="random")
    parser.add_argument("--optuna-jobs", type=int, default=1)
    parser.add_argument("--optuna-topk", type=int, default=24)
    parser.add_argument("--selection-mode", choices=["val", "robust"], default="robust")
    parser.add_argument("--topcap-min-coverage-days", type=float, default=30.0)
    parser.add_argument("--topcap-min-row-ratio", type=float, default=0.25)
    parser.add_argument("--topcap-min-symbols", type=int, default=2)
    parser.add_argument("--ensemble-min-bars", type=int, default=20)
    parser.add_argument("--ensemble-min-oos-trades", type=int, default=1)
    parser.add_argument("--xau-xag-ensemble-min-overlap-days", type=float, default=120.0)
    parser.add_argument("--xau-xag-ensemble-min-oos-trades", type=int, default=2)

    parser.add_argument("--max-selected", type=int, default=32)
    parser.add_argument("--max-per-family", type=int, default=20)
    parser.add_argument("--max-per-timeframe", type=int, default=8)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap for timeframe x seed runs (0 means all).",
    )
    parser.add_argument(
        "--candidate-manifest",
        default="",
        help="Optional path to candidate manifest JSON (forwarded to OOS search).",
    )
    parser.add_argument("--data-cache-max-entries", type=int, default=3)
    parser.add_argument("--data-cache-max-rows", type=int, default=0)
    parser.add_argument(
        "--data-cache-scope",
        choices=["global", "strategy"],
        default="strategy",
    )
    parser.add_argument("--ensemble-mode", choices=["auto", "off"], default="auto")
    parser.add_argument("--ensemble-max-candidates", type=int, default=4)
    parser.add_argument(
        "--prefetch-strategy-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward prefetch behavior to oos_guarded_multistrategy_search.py",
    )
    parser.add_argument("--influx-1s-query-chunk-days", type=int, default=0)
    parser.add_argument("--influx-1s-agg-chunk-days", type=int, default=0)
    parser.add_argument("--child-log-dir", default="logs/team_research")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.base_timeframe = _enforce_1s_base_timeframe(args.base_timeframe)
    args.base_timeframes = ["1s"]
    candidate_manifest_arg = str(args.candidate_manifest or "").strip()
    backend_arg = str(args.backend or "").strip()
    influx_url_arg = str(args.influx_url or "").strip()
    influx_org_arg = str(args.influx_org or "").strip()
    influx_bucket_arg = str(args.influx_bucket or "").strip()
    influx_token_arg = str(args.influx_token or "").strip()
    influx_token_env_arg = str(args.influx_token_env or "INFLUXDB_TOKEN").strip() or "INFLUXDB_TOKEN"
    use_influx_options = bool(
        backend_arg
        or influx_url_arg
        or influx_org_arg
        or influx_bucket_arg
        or influx_token_arg
        or influx_token_env_arg != "INFLUXDB_TOKEN"
    )
    child_log_dir = Path(str(args.child_log_dir or "logs/team_research"))
    child_log_dir.mkdir(parents=True, exist_ok=True)

    run_plan: list[tuple[str, int]] = []
    for timeframe in list(args.timeframes):
        for seed in list(args.seeds):
            run_plan.append((str(timeframe), int(seed)))
    if int(args.max_runs) > 0:
        run_plan = run_plan[: int(args.max_runs)]

    all_candidates: list[dict] = []
    run_rows: list[dict] = []
    base_timeframe_candidates: list[str] = []
    preferred_base = _normalize_timeframe(str(args.base_timeframe))
    if _timeframe_seconds(preferred_base) > 0:
        base_timeframe_candidates.append(preferred_base)
    for raw in list(args.base_timeframes):
        token = _normalize_timeframe(raw)
        if _timeframe_seconds(token) <= 0:
            continue
        if token in base_timeframe_candidates:
            continue
        base_timeframe_candidates.append(token)

    for timeframe, seed in run_plan:
        print(f"\n[TEAM-RUN] timeframe={timeframe} seed={seed}")
        eligible_bases = _eligible_base_timeframes(
            strategy_timeframe=str(timeframe),
            base_candidates=base_timeframe_candidates,
        )
        if not eligible_bases:
            run_rows.append(
                {
                    "timeframe": timeframe,
                    "seed": int(seed),
                    "status": "failed",
                    "reason": "no_eligible_base_timeframe",
                }
            )
            continue

        attempts: list[dict] = []
        if args.dry_run:
            cmd = [
                sys.executable,
                "scripts/oos_guarded_multistrategy_search.py",
                "--db-path",
                str(args.db_path),
                "--exchange",
                str(args.exchange),
                "--timeframe",
                str(timeframe),
                "--base-timeframe",
                str(eligible_bases[0]),
                "--market-type",
                str(args.market_type),
                "--mode",
                str(args.mode),
                "--strategy-set",
                str(args.strategy_set),
                "--seed",
                str(int(seed)),
                "--train-days",
                str(int(args.train_days)),
                "--val-days",
                str(int(args.val_days)),
                "--oos-days",
                str(int(args.oos_days)),
                "--min-insample-days",
                str(int(args.min_insample_days)),
                "--annual-return-floor",
                str(float(args.annual_return_floor)),
                "--benchmark-symbol",
                str(args.benchmark_symbol),
                "--topcap-iters",
                str(int(args.topcap_iters)),
                "--pair-iters",
                str(int(args.pair_iters)),
                "--ensemble-iters",
                str(int(args.ensemble_iters)),
                "--search-engine",
                str(args.search_engine),
                "--optuna-jobs",
                str(int(args.optuna_jobs)),
                "--optuna-topk",
                str(int(args.optuna_topk)),
                "--selection-mode",
                str(args.selection_mode),
                "--topcap-min-coverage-days",
                str(float(args.topcap_min_coverage_days)),
                "--topcap-min-row-ratio",
                str(float(args.topcap_min_row_ratio)),
                "--topcap-min-symbols",
                str(int(args.topcap_min_symbols)),
                "--ensemble-min-bars",
                str(int(args.ensemble_min_bars)),
                "--ensemble-min-oos-trades",
                str(int(args.ensemble_min_oos_trades)),
                "--xau-xag-ensemble-min-overlap-days",
                str(float(args.xau_xag_ensemble_min_overlap_days)),
                "--xau-xag-ensemble-min-oos-trades",
                str(int(args.xau_xag_ensemble_min_oos_trades)),
                "--data-cache-max-entries",
                str(int(args.data_cache_max_entries)),
                "--data-cache-max-rows",
                str(int(args.data_cache_max_rows)),
                "--data-cache-scope",
                str(args.data_cache_scope),
                "--ensemble-mode",
                str(args.ensemble_mode),
                "--ensemble-max-candidates",
                str(int(args.ensemble_max_candidates)),
                "--prefetch-strategy-data"
                if bool(args.prefetch_strategy_data)
                else "--no-prefetch-strategy-data",
                "--influx-1s-query-chunk-days",
                str(int(args.influx_1s_query_chunk_days)),
                "--influx-1s-agg-chunk-days",
                str(int(args.influx_1s_agg_chunk_days)),
                "--topcap-symbols",
                *[str(token) for token in args.topcap_symbols],
            ]
            if candidate_manifest_arg:
                cmd.extend(["--candidate-manifest", candidate_manifest_arg])
            if backend_arg:
                cmd.extend(["--backend", backend_arg])
            if influx_url_arg:
                cmd.extend(["--influx-url", influx_url_arg])
            if influx_org_arg:
                cmd.extend(["--influx-org", influx_org_arg])
            if influx_bucket_arg:
                cmd.extend(["--influx-bucket", influx_bucket_arg])
            if influx_token_arg:
                cmd.extend(["--influx-token", influx_token_arg])
            if use_influx_options and influx_token_env_arg:
                cmd.extend(["--influx-token-env", influx_token_env_arg])
            command_text = " ".join(cmd)
            print(f"[TEAM-RUN-CMD] {command_text}")
            run_rows.append(
                {
                    "timeframe": timeframe,
                    "seed": int(seed),
                    "status": "dry_run",
                    "command": cmd,
                    "eligible_base_timeframes": eligible_bases,
                }
            )
            continue

        report_path = None
        for base_timeframe in eligible_bases:
            cmd = [
                sys.executable,
                "scripts/oos_guarded_multistrategy_search.py",
                "--db-path",
                str(args.db_path),
                "--exchange",
                str(args.exchange),
                "--timeframe",
                str(timeframe),
                "--base-timeframe",
                str(base_timeframe),
                "--market-type",
                str(args.market_type),
                "--mode",
                str(args.mode),
                "--strategy-set",
                str(args.strategy_set),
                "--seed",
                str(int(seed)),
                "--train-days",
                str(int(args.train_days)),
                "--val-days",
                str(int(args.val_days)),
                "--oos-days",
                str(int(args.oos_days)),
                "--min-insample-days",
                str(int(args.min_insample_days)),
                "--annual-return-floor",
                str(float(args.annual_return_floor)),
                "--benchmark-symbol",
                str(args.benchmark_symbol),
                "--topcap-iters",
                str(int(args.topcap_iters)),
                "--pair-iters",
                str(int(args.pair_iters)),
                "--ensemble-iters",
                str(int(args.ensemble_iters)),
                "--search-engine",
                str(args.search_engine),
                "--optuna-jobs",
                str(int(args.optuna_jobs)),
                "--optuna-topk",
                str(int(args.optuna_topk)),
                "--selection-mode",
                str(args.selection_mode),
                "--topcap-min-coverage-days",
                str(float(args.topcap_min_coverage_days)),
                "--topcap-min-row-ratio",
                str(float(args.topcap_min_row_ratio)),
                "--topcap-min-symbols",
                str(int(args.topcap_min_symbols)),
                "--ensemble-min-bars",
                str(int(args.ensemble_min_bars)),
                "--ensemble-min-oos-trades",
                str(int(args.ensemble_min_oos_trades)),
                "--xau-xag-ensemble-min-overlap-days",
                str(float(args.xau_xag_ensemble_min_overlap_days)),
                "--xau-xag-ensemble-min-oos-trades",
                str(int(args.xau_xag_ensemble_min_oos_trades)),
                "--data-cache-max-entries",
                str(int(args.data_cache_max_entries)),
                "--data-cache-max-rows",
                str(int(args.data_cache_max_rows)),
                "--data-cache-scope",
                str(args.data_cache_scope),
                "--ensemble-mode",
                str(args.ensemble_mode),
                "--ensemble-max-candidates",
                str(int(args.ensemble_max_candidates)),
                "--prefetch-strategy-data"
                if bool(args.prefetch_strategy_data)
                else "--no-prefetch-strategy-data",
                "--influx-1s-query-chunk-days",
                str(int(args.influx_1s_query_chunk_days)),
                "--influx-1s-agg-chunk-days",
                str(int(args.influx_1s_agg_chunk_days)),
                "--topcap-symbols",
                *[str(token) for token in args.topcap_symbols],
            ]
            if candidate_manifest_arg:
                cmd.extend(["--candidate-manifest", candidate_manifest_arg])
            if backend_arg:
                cmd.extend(["--backend", backend_arg])
            if influx_url_arg:
                cmd.extend(["--influx-url", influx_url_arg])
            if influx_org_arg:
                cmd.extend(["--influx-org", influx_org_arg])
            if influx_bucket_arg:
                cmd.extend(["--influx-bucket", influx_bucket_arg])
            if influx_token_arg:
                cmd.extend(["--influx-token", influx_token_arg])
            if use_influx_options and influx_token_env_arg:
                cmd.extend(["--influx-token-env", influx_token_env_arg])
            command_text = " ".join(cmd)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            run_log = child_log_dir / (
                f"team_run_{str(timeframe).lower()}_seed{int(seed)}_base{str(base_timeframe).lower()}_{stamp}.log"
            )
            print(f"[TEAM-RUN-CMD] {command_text}")
            print(f"[TEAM-RUN-LOG] {run_log}")
            with run_log.open("w", encoding="utf-8") as log_fd:
                log_fd.write(f"$ {command_text}\n")
                log_fd.flush()
                proc = subprocess.run(
                    cmd,
                    stdout=log_fd,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            log_tail = _read_log_tail(run_log, max_bytes=8192)
            attempts.append(
                {
                    "base_timeframe": str(base_timeframe),
                    "return_code": int(proc.returncode),
                    "log_path": str(run_log),
                    "command": cmd,
                    "log_tail": log_tail[-1200:],
                }
            )
            if int(proc.returncode) != 0:
                continue
            report_path = _extract_saved_report_path_from_log(run_log)
            if report_path is not None and report_path.exists():
                break

        if report_path is None or not report_path.exists():
            run_rows.append(
                {
                    "timeframe": timeframe,
                    "seed": int(seed),
                    "status": "failed",
                    "reason": "report_path_not_found",
                    "attempts": attempts,
                }
            )
            continue

        with report_path.open(encoding="utf-8") as file:
            report = json.load(file)
        candidates = list(report.get("candidates") or [])
        for row in candidates:
            enriched = dict(row)
            enriched["strategy_timeframe"] = str(timeframe)
            enriched["seed"] = int(seed)
            enriched["report_path"] = str(report_path)
            enriched["identity"] = _candidate_identity(enriched, timeframe)
            all_candidates.append(enriched)

        run_rows.append(
            {
                "timeframe": timeframe,
                "seed": int(seed),
                "status": "ok",
                "report_path": str(report_path),
                "candidate_count": len(candidates),
                "attempts": attempts,
            }
        )

    selected = _select_diversified_team(
        all_candidates,
        mode=str(args.mode),
        max_total=int(args.max_selected),
        max_per_family=int(args.max_per_family),
        max_per_timeframe=int(args.max_per_timeframe),
    )

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": str(args.mode),
        "exchange": str(args.exchange),
        "market_type": str(args.market_type),
        "base_timeframe": str(args.base_timeframe),
        "base_timeframes": base_timeframe_candidates,
        "timeframes": list(args.timeframes),
        "seeds": list(args.seeds),
        "strategy_set": str(args.strategy_set),
        "topcap_symbols": list(args.topcap_symbols),
        "backend": backend_arg,
        "influx_url": influx_url_arg,
        "influx_org": influx_org_arg,
        "influx_bucket": influx_bucket_arg,
        "influx_token_env": influx_token_env_arg,
        "candidate_manifest": candidate_manifest_arg,
        "data_cache_max_entries": int(args.data_cache_max_entries),
        "data_cache_max_rows": int(args.data_cache_max_rows),
        "data_cache_scope": str(args.data_cache_scope),
        "ensemble_mode": str(args.ensemble_mode),
        "ensemble_max_candidates": int(args.ensemble_max_candidates),
        "prefetch_strategy_data": bool(args.prefetch_strategy_data),
        "influx_1s_query_chunk_days": int(args.influx_1s_query_chunk_days),
        "influx_1s_agg_chunk_days": int(args.influx_1s_agg_chunk_days),
        "child_log_dir": str(child_log_dir),
        "run_rows": run_rows,
        "all_candidates_count": len(all_candidates),
        "selected_team_count": len(selected),
        "selected_team": selected,
    }

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"strategy_team_research_{args.mode}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n=== Strategy Team Research Done ===")
    print(f"all_candidates={len(all_candidates)} selected_team={len(selected)}")
    print(f"Saved: {out_path}")
    if (not bool(args.dry_run)) and len(all_candidates) <= 0:
        failed_runs = sum(1 for row in run_rows if str(row.get("status")) != "ok")
        total_runs = len(run_rows)
        print(
            (
                "[ERROR] No valid candidates were produced. "
                f"failed_runs={failed_runs}/{total_runs}. "
                "Inspect per-run logs under child_log_dir."
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
