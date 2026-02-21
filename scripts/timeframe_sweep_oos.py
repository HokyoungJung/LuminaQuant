"""Run multi-timeframe searches and rank final performance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _safe_float(value, default=0.0):
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


def _score_from_report(report: dict, mode: str) -> tuple[float, dict]:
    candidates = list(report.get("candidates", []) or [])
    if not candidates:
        return -1_000_000.0, {"source": "none", "value": -1_000_000.0}

    mode_token = str(mode).strip().lower()
    ensemble = report.get("ensemble", {}) or {}
    hurdle_key = "val" if mode_token == "live" else "oos"
    ensemble_hurdle = ((ensemble.get("hurdle_fields") or {}).get(hurdle_key)) or {}
    if bool(ensemble_hurdle.get("pass", False)):
        score = _safe_float(ensemble_hurdle.get("score"), -1_000_000.0)
        return score, {
            "source": f"ensemble_{hurdle_key}_hurdle_score",
            "value": score,
            "excess": _safe_float(ensemble_hurdle.get("excess_return"), 0.0),
        }

    scored_candidates = []
    for row in candidates:
        hurdle = ((row.get("hurdle_fields") or {}).get(hurdle_key)) or {}
        pass_flag = bool(hurdle.get("pass", False))
        score = _safe_float(hurdle.get("score"), -1_000_000.0)
        excess = _safe_float(hurdle.get("excess_return"), -1_000_000.0)
        scored_candidates.append((pass_flag, score, excess, row))

    passing = [entry for entry in scored_candidates if entry[0]]
    if passing:
        _, score, excess, row = max(passing, key=lambda item: item[1])
        return score, {
            "source": f"candidate_{hurdle_key}_hurdle_score",
            "name": row.get("name"),
            "value": score,
            "excess": excess,
        }

    if scored_candidates:
        _, _, excess, row = max(scored_candidates, key=lambda item: item[2])
        return -1_000_000.0 + excess, {
            "source": "hurdle_not_met",
            "name": row.get("name"),
            "value": -1_000_000.0 + excess,
            "best_excess": excess,
        }

    ensemble_metrics = ensemble.get("oos_metrics")
    if isinstance(ensemble_metrics, dict):
        ret = _safe_float(ensemble_metrics.get("return"), -1_000_000.0)
        return ret, {"source": "ensemble_return_fallback", "value": ret}

    if mode_token == "live":
        best = max(
            candidates, key=lambda row: _safe_float((row.get("val") or {}).get("return"), -1e9)
        )
        ret = _safe_float((best.get("val") or {}).get("return"), -1_000_000.0)
        return ret, {"source": "candidate_val_fallback", "name": best.get("name"), "value": ret}

    best = max(candidates, key=lambda row: _safe_float((row.get("oos") or {}).get("return"), -1e9))
    ret = _safe_float((best.get("oos") or {}).get("return"), -1_000_000.0)
    return ret, {"source": "candidate_oos_fallback", "name": best.get("name"), "value": ret}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run timeframe sweep for OOS/live search.")
    parser.add_argument("--db-path", default="data/lq_market.sqlite3")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--base-timeframe", default="1s")
    parser.add_argument("--market-type", choices=["spot", "future"], default="future")
    parser.add_argument("--mode", choices=["oos", "live"], default="oos")
    parser.add_argument(
        "--strategy-set",
        choices=["all", "crypto-only", "xau-xag-only"],
        default="all",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    )
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--oos-days", type=int, default=30)
    parser.add_argument("--min-insample-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--annual-return-floor", type=float, default=0.10)
    parser.add_argument("--benchmark-symbol", default="BTC/USDT")
    parser.add_argument("--topcap-iters", type=int, default=320)
    parser.add_argument("--pair-iters", type=int, default=260)
    parser.add_argument("--ensemble-iters", type=int, default=5000)
    parser.add_argument("--search-engine", choices=["optuna", "random"], default="optuna")
    parser.add_argument("--optuna-jobs", type=int, default=1)
    parser.add_argument("--optuna-topk", type=int, default=24)
    parser.add_argument("--selection-mode", choices=["val", "robust"], default="val")
    parser.add_argument("--topcap-count", type=int, default=10)
    parser.add_argument("--topcap-candidate-count", type=int, default=120)
    parser.add_argument("--topcap-symbols", nargs="+", default=[])
    parser.add_argument("--topcap-min-coverage-days", type=float, default=360.0)
    parser.add_argument("--topcap-min-row-ratio", type=float, default=0.85)
    parser.add_argument("--topcap-min-symbols", type=int, default=8)
    parser.add_argument("--ensemble-min-bars", type=int, default=20)
    parser.add_argument("--ensemble-min-oos-trades", type=int, default=1)
    parser.add_argument("--xau-xag-ensemble-min-overlap-days", type=float, default=120.0)
    parser.add_argument("--xau-xag-ensemble-min-oos-trades", type=int, default=2)
    args = parser.parse_args()

    reports: list[dict] = []
    for timeframe in list(args.timeframes):
        cmd = [
            sys.executable,
            "scripts/oos_guarded_multistrategy_search.py",
            "--db-path",
            str(args.db_path),
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
            "--timeframe",
            str(timeframe),
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
        ]
        if args.topcap_symbols:
            cmd.append("--topcap-symbols")
            cmd.extend([str(symbol) for symbol in args.topcap_symbols])

        print(f"\n[SWEEP] timeframe={timeframe}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if proc.returncode != 0:
            print(f"  failed rc={proc.returncode}")
            if stderr:
                print(stderr.strip())
            reports.append(
                {
                    "timeframe": timeframe,
                    "status": "failed",
                    "return_code": int(proc.returncode),
                    "stderr": stderr[-4000:],
                }
            )
            continue

        report_path = _extract_saved_report_path(stdout)
        if report_path is None or not report_path.exists():
            reports.append(
                {
                    "timeframe": timeframe,
                    "status": "failed",
                    "reason": "report_path_not_found",
                }
            )
            continue

        with report_path.open(encoding="utf-8") as fp:
            report = json.load(fp)
        score, score_meta = _score_from_report(report, str(args.mode))
        reports.append(
            {
                "timeframe": timeframe,
                "status": "ok",
                "report_path": str(report_path),
                "score": float(score),
                "score_meta": score_meta,
            }
        )
        print(f"  score={score:.6f} via {score_meta}")

    ranked = sorted(
        [row for row in reports if row.get("status") == "ok"],
        key=lambda row: _safe_float(row.get("score"), -1e9),
        reverse=True,
    )

    output = {
        "mode": str(args.mode),
        "strategy_set": str(args.strategy_set),
        "exchange": str(args.exchange),
        "base_timeframe": str(args.base_timeframe),
        "timeframes": list(args.timeframes),
        "runs": reports,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"timeframe_sweep_{args.mode}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print("\n=== Sweep Done ===")
    if ranked:
        print(f"best_timeframe={ranked[0]['timeframe']} score={ranked[0]['score']:.6f}")
    else:
        print("No successful timeframe runs.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
