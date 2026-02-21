"""One-command phase-1 research starter for Binance USDT-M.

This wrapper does two things with stable defaults:
1) Optionally sync base-timeframe OHLCV for a liquid USDT symbol set.
2) Run timeframe sweep on leakage-safe OOS search with explicit topcap symbols.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from lumina_quant.data_sync import create_binance_exchange, parse_timestamp_input, sync_market_data

DEFAULT_PHASE1_SYMBOLS: tuple[str, ...] = (
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
)


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        token = str(raw).strip().upper().replace("_", "/").replace("-", "/")
        if not token:
            continue
        if "/" not in token and token.endswith("USDT") and len(token) > 4:
            token = f"{token[:-4]}/USDT"
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _build_sweep_command(args: argparse.Namespace, symbols: Sequence[str]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/timeframe_sweep_oos.py",
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
    if symbols:
        cmd.extend(["--topcap-symbols", *symbols])
    return cmd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase-1 Binance USDT-M research workflow.")
    parser.add_argument("--db-path", default="data/lq_market.sqlite3")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", choices=["spot", "future"], default="future")
    parser.add_argument("--base-timeframe", default="1m")
    parser.add_argument("--mode", choices=["oos", "live"], default="oos")
    parser.add_argument(
        "--strategy-set",
        choices=["all", "crypto-only", "xau-xag-only"],
        default="crypto-only",
    )
    parser.add_argument("--timeframes", nargs="+", default=["5m", "15m", "1h", "4h"])

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

    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_PHASE1_SYMBOLS))
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--sync-since", default="2021-01-01T00:00:00+00:00")
    parser.add_argument("--sync-limit", type=int, default=1000)
    parser.add_argument("--sync-max-batches", type=int, default=100000)
    parser.add_argument("--sync-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    symbols = _normalize_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No valid symbols resolved from --symbols")

    if not args.skip_sync:
        since_ms = parse_timestamp_input(args.sync_since)
        exchange = create_binance_exchange(market_type=str(args.market_type), testnet=False)
        try:
            stats = sync_market_data(
                exchange=exchange,
                db_path=str(args.db_path),
                exchange_id=str(args.exchange),
                symbol_list=symbols,
                timeframe=str(args.base_timeframe),
                since_ms=since_ms,
                until_ms=None,
                force_full=False,
                limit=max(1, int(args.sync_limit)),
                max_batches=max(1, int(args.sync_max_batches)),
                retries=max(0, int(args.sync_retries)),
                export_csv_dir="data",
            )
        finally:
            close_fn = getattr(exchange, "close", None)
            if callable(close_fn):
                close_fn()

        print("\n=== Phase-1 Sync Summary ===")
        for item in stats:
            print(
                f"- {item.symbol}: fetched={item.fetched_rows} upserted={item.upserted_rows} "
                f"first_ts={item.first_timestamp_ms} last_ts={item.last_timestamp_ms}"
            )

    sweep_cmd = _build_sweep_command(args, symbols)
    print("\n=== Phase-1 Sweep Command ===")
    print(" ".join(sweep_cmd))
    if args.dry_run:
        return

    project_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(sweep_cmd, cwd=str(project_root), check=False)
    if int(proc.returncode) != 0:
        raise SystemExit(int(proc.returncode))


if __name__ == "__main__":
    main()
