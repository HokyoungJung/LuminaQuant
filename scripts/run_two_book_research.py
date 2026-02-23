"""Two-book research runner for Binance USDT-M.

This script runs the existing timeframe sweep workflow, then derives a
two-book selection artifact:

- Alpha book: market-neutral sleeves (pair/lag/vwap/mean-reversion)
- Trend book: directional overlay sleeves (topcap/breakout)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_sweep_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/timeframe_sweep_oos.py",
        "--db-path",
        str(args.db_path),
        "--backend",
        str(args.backend),
        "--exchange",
        str(args.exchange),
        "--base-timeframe",
        str(args.base_timeframe),
        "--market-type",
        str(args.market_type),
        "--mode",
        str(args.mode),
        "--strategy-set",
        str(args.strategy_set),
        "--train-days",
        str(int(args.train_days)),
        "--val-days",
        str(int(args.val_days)),
        "--oos-days",
        str(int(args.oos_days)),
        "--min-insample-days",
        str(int(args.min_insample_days)),
        "--seed",
        str(int(args.seed)),
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
        "--topcap-count",
        str(int(args.topcap_count)),
        "--topcap-candidate-count",
        str(int(args.topcap_candidate_count)),
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
        "--timeframes",
        *[str(token) for token in args.timeframes],
    ]
    if str(args.influx_url).strip():
        cmd.extend(["--influx-url", str(args.influx_url).strip()])
    if str(args.influx_org).strip():
        cmd.extend(["--influx-org", str(args.influx_org).strip()])
    if str(args.influx_bucket).strip():
        cmd.extend(["--influx-bucket", str(args.influx_bucket).strip()])
    if str(args.influx_token).strip():
        cmd.extend(["--influx-token", str(args.influx_token).strip()])
    if str(args.influx_token_env).strip():
        cmd.extend(["--influx-token-env", str(args.influx_token_env).strip()])
    if args.topcap_symbols:
        cmd.extend(["--topcap-symbols", *[str(symbol) for symbol in args.topcap_symbols]])
    return cmd


def _enforce_1s_base_timeframe(value: str) -> str:
    token = str(value or "").strip().lower() or "1s"
    if token != "1s":
        print(f"[WARN] base-timeframe '{token}' overridden to '1s' for all backtests.")
    return "1s"


def _latest_sweep_report(mode: str) -> Path:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        raise FileNotFoundError("reports directory not found")
    pattern = f"timeframe_sweep_{mode}_*.json"
    files = sorted(reports_dir.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No sweep report found with pattern: {pattern}")
    return files[-1]


def _candidate_hurdle_score(candidate: dict, hurdle_key: str) -> tuple[bool, float, float]:
    fields = ((candidate.get("hurdle_fields") or {}).get(hurdle_key)) or {}
    passed = bool(fields.get("pass", False))
    score = _safe_float(fields.get("score"), -1_000_000.0)
    excess = _safe_float(fields.get("excess_return"), -1_000_000.0)
    return passed, score, excess


def _pick_best_candidate(candidates: list[dict], hurdle_key: str) -> dict | None:
    if not candidates:
        return None
    passing = [row for row in candidates if _candidate_hurdle_score(row, hurdle_key)[0]]
    if passing:
        return max(passing, key=lambda row: _candidate_hurdle_score(row, hurdle_key)[1])
    return max(candidates, key=lambda row: _candidate_hurdle_score(row, hurdle_key)[2])


def _prefixed(candidates: list[dict], prefixes: list[str]) -> list[dict]:
    tokens = [str(prefix).strip().lower() for prefix in prefixes if str(prefix).strip()]
    if not tokens:
        return []
    selected: list[dict] = []
    for row in candidates:
        name = str(row.get("name", "")).strip().lower()
        if any(name.startswith(token) for token in tokens):
            selected.append(row)
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run two-book quant research workflow.")
    parser.add_argument("--db-path", default="data/lq_market.sqlite3")
    parser.add_argument("--backend", default="influxdb", help="Storage backend override (sqlite|influxdb).")
    parser.add_argument("--influx-url", default="")
    parser.add_argument("--influx-org", default="")
    parser.add_argument("--influx-bucket", default="")
    parser.add_argument("--influx-token", default="")
    parser.add_argument("--influx-token-env", default="INFLUXDB_TOKEN")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", choices=["spot", "future"], default="future")
    parser.add_argument("--base-timeframe", default="1s")
    parser.add_argument("--mode", choices=["oos", "live"], default="oos")
    parser.add_argument(
        "--strategy-set", choices=["all", "crypto-only", "xau-xag-only"], default="crypto-only"
    )
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"])
    parser.add_argument("--topcap-symbols", nargs="+", default=[])

    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--oos-days", type=int, default=30)
    parser.add_argument("--min-insample-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument("--annual-return-floor", type=float, default=0.10)
    parser.add_argument("--benchmark-symbol", default="BTC/USDT")

    parser.add_argument("--topcap-iters", type=int, default=240)
    parser.add_argument("--pair-iters", type=int, default=180)
    parser.add_argument("--ensemble-iters", type=int, default=3000)
    parser.add_argument("--search-engine", choices=["optuna", "random"], default="optuna")
    parser.add_argument("--optuna-jobs", type=int, default=1)
    parser.add_argument("--optuna-topk", type=int, default=20)
    parser.add_argument("--selection-mode", choices=["val", "robust"], default="robust")

    parser.add_argument("--topcap-count", type=int, default=10)
    parser.add_argument("--topcap-candidate-count", type=int, default=120)
    parser.add_argument("--topcap-min-coverage-days", type=float, default=30.0)
    parser.add_argument("--topcap-min-row-ratio", type=float, default=0.25)
    parser.add_argument("--topcap-min-symbols", type=int, default=2)
    parser.add_argument("--ensemble-min-bars", type=int, default=20)
    parser.add_argument("--ensemble-min-oos-trades", type=int, default=1)
    parser.add_argument("--xau-xag-ensemble-min-overlap-days", type=float, default=120.0)
    parser.add_argument("--xau-xag-ensemble-min-oos-trades", type=int, default=2)

    parser.add_argument(
        "--alpha-prefixes",
        nargs="+",
        default=["pair_", "lag_convergence_", "mean_reversion_std_", "vwap_reversion_"],
        help="Candidate-name prefixes for market-neutral alpha book.",
    )
    parser.add_argument(
        "--trend-prefixes",
        nargs="+",
        default=["topcap_tsmom", "rolling_breakout_"],
        help="Candidate-name prefixes for directional trend overlay book.",
    )
    parser.add_argument("--alpha-risk-budget", type=float, default=0.8)
    parser.add_argument("--trend-risk-budget", type=float, default=0.2)
    parser.add_argument("--sweep-report", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _normalize_risk_share(alpha_budget: float, trend_budget: float) -> tuple[float, float]:
    alpha = max(0.0, _safe_float(alpha_budget, 0.8))
    trend = max(0.0, _safe_float(trend_budget, 0.2))
    total = alpha + trend
    if total <= 0.0:
        return 0.8, 0.2
    return alpha / total, trend / total


def _select_two_book_candidates(
    candidates: list[dict],
    hurdle_key: str,
    alpha_prefixes: list[str],
    trend_prefixes: list[str],
) -> tuple[dict | None, dict | None]:
    alpha_pool = _prefixed(candidates, alpha_prefixes)
    trend_pool = _prefixed(candidates, trend_prefixes)
    return _pick_best_candidate(alpha_pool, hurdle_key), _pick_best_candidate(
        trend_pool, hurdle_key
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.base_timeframe = _enforce_1s_base_timeframe(args.base_timeframe)

    backend_arg = str(args.backend or "").strip()
    influx_url_arg = str(args.influx_url or "").strip()
    influx_org_arg = str(args.influx_org or "").strip()
    influx_bucket_arg = str(args.influx_bucket or "").strip()
    influx_token_arg = str(args.influx_token or "").strip()
    influx_token_env_arg = str(args.influx_token_env or "INFLUXDB_TOKEN").strip() or "INFLUXDB_TOKEN"
    if backend_arg:
        os.environ["LQ__STORAGE__BACKEND"] = "influxdb" if backend_arg.lower() in {"influx", "influxdb"} else "sqlite"
    if influx_url_arg:
        os.environ["LQ__STORAGE__INFLUX_URL"] = influx_url_arg
    if influx_org_arg:
        os.environ["LQ__STORAGE__INFLUX_ORG"] = influx_org_arg
    if influx_bucket_arg:
        os.environ["LQ__STORAGE__INFLUX_BUCKET"] = influx_bucket_arg
    if influx_token_env_arg:
        os.environ["LQ__STORAGE__INFLUX_TOKEN_ENV"] = influx_token_env_arg
    if influx_token_arg:
        os.environ[influx_token_env_arg] = influx_token_arg

    sweep_cmd = _build_sweep_command(args)
    print("=== Two-Book Sweep Command ===")
    print(" ".join(sweep_cmd))
    if args.dry_run and not str(args.sweep_report).strip():
        return

    if not args.dry_run:
        proc = subprocess.run(sweep_cmd, check=False)
        if int(proc.returncode) != 0:
            raise SystemExit(int(proc.returncode))

    override_sweep = str(args.sweep_report).strip()
    sweep_path = Path(override_sweep) if override_sweep else _latest_sweep_report(str(args.mode))
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep report not found: {sweep_path}")
    with sweep_path.open(encoding="utf-8") as file:
        sweep = json.load(file)

    ranked = list(sweep.get("ranked") or [])
    if not ranked:
        raise RuntimeError("Sweep completed but no ranked entries were found.")

    best = ranked[0]
    report_path = Path(str(best.get("report_path", "")))
    if not report_path.exists():
        raise FileNotFoundError(f"Best report path not found: {report_path}")
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    candidates = list(report.get("candidates") or [])
    hurdle_key = "val" if str(args.mode).strip().lower() == "live" else "oos"

    alpha_pick, trend_pick = _select_two_book_candidates(
        candidates=candidates,
        hurdle_key=hurdle_key,
        alpha_prefixes=list(args.alpha_prefixes),
        trend_prefixes=list(args.trend_prefixes),
    )

    alpha_share, trend_share = _normalize_risk_share(
        alpha_budget=float(args.alpha_risk_budget),
        trend_budget=float(args.trend_risk_budget),
    )

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": str(args.mode),
        "sweep_report": str(sweep_path),
        "best_timeframe": best.get("timeframe"),
        "selected_report": str(report_path),
        "book_policy": {
            "alpha_prefixes": list(args.alpha_prefixes),
            "trend_prefixes": list(args.trend_prefixes),
            "hurdle_key": hurdle_key,
            "alpha_risk_share": float(alpha_share),
            "trend_risk_share": float(trend_share),
        },
        "alpha_book": alpha_pick,
        "trend_book": trend_pick,
        "ensemble": report.get("ensemble"),
        "benchmark": report.get("benchmark"),
        "split": report.get("split"),
    }

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"two_book_selection_{args.mode}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("=== Two-Book Selection ===")
    print(f"Alpha: {None if alpha_pick is None else alpha_pick.get('name')}")
    print(f"Trend: {None if trend_pick is None else trend_pick.get('name')}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
