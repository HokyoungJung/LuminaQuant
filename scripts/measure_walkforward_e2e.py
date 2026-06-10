"""End-to-end walk-forward timing on the canonical 9-combo × 3-fold spec.

Canonical spec (shared with task #2 golden run):
  Strategy : MovingAverageCrossStrategy, allow_short=True
  Symbols  : BTCUSDT + ETHUSDT (1000 daily bars from 2022-01-01, seed=42)
  Fixtures : baseline/golden/BTCUSDT_1000d_seed42.parquet
             baseline/golden/ETHUSDT_1000d_seed42.parquet
             (SHA-256 verified against baseline/golden/PROVENANCE.json)
  Grid     : short_window=[10,20,30] × long_window=[40,80,120] → 9 combos
             allow_short=True fixed
  Folds    : 3; train=6mo, val=3mo, test=3mo, step=3mo, start=2022-01-01
  Seed     : 42

Output key: walk_forward_e2e_elapsed_s (wall-clock for full 9-combo × 3-fold run)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROVENANCE_PATH = PROJECT_ROOT / "baseline" / "golden" / "PROVENANCE.json"
FIXTURE_BTCUSDT = PROJECT_ROOT / "tests" / "fixtures" / "ohlcv" / "BTCUSDT_seed42_1000d.parquet"
FIXTURE_ETHUSDT = PROJECT_ROOT / "tests" / "fixtures" / "ohlcv" / "ETHUSDT_seed42_1000d.parquet"

# Canonical walk-forward spec
SHORT_WINDOWS = [10, 20, 30]
LONG_WINDOWS = [40, 80, 120]
ALLOW_SHORT = True
N_FOLDS = 3
TRAIN_MONTHS = 6
VAL_MONTHS = 3
TEST_MONTHS = 3
STEP_MONTHS = 3
START_DATE_STR = "2022-01-01"
SEED = 42


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_provenance() -> dict:
    """Verify fixture SHA-256 against PROVENANCE.json; raise if mismatch.

    PROVENANCE structure (worker-2 format):
      input_data.fixtures.{SYMBOL}.sha256
      input_data.fixtures.{SYMBOL}.path
    """
    if not PROVENANCE_PATH.exists():
        raise FileNotFoundError(
            f"PROVENANCE.json not found at {PROVENANCE_PATH}. "
            "Run task #2 (worker-2) first to generate fixtures."
        )
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    fixtures = (provenance.get("input_data") or {}).get("fixtures", {})

    for path, symbol in [
        (FIXTURE_BTCUSDT, "BTCUSDT"),
        (FIXTURE_ETHUSDT, "ETHUSDT"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Fixture missing: {path}. Run task #2 (worker-2) first.")
        expected = (fixtures.get(symbol) or {}).get("sha256")
        if expected:
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {symbol}: expected={expected} actual={actual}"
                )
            print(f"  [OK] SHA-256 verified: {symbol} ({path.name})")
        else:
            print(f"  [WARN] No SHA-256 in PROVENANCE for {symbol}; skipping check.")

    return provenance


def _run_e2e_walkforward() -> float:
    """Run canonical 9-combo × 3-fold walk-forward; return wall-clock seconds."""
    import numpy as np
    import polars as pl

    from lumina_quant.backtesting.backtest import Backtest
    from lumina_quant.backtesting.data import HistoricCSVDataHandler
    from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
    from lumina_quant.backtesting.portfolio_backtest import Portfolio
    from lumina_quant.optimization.fast_eval import evaluate_metrics_numba
    from lumina_quant.optimization.walkers import build_walk_forward_splits
    from lumina_quant.strategies.moving_average import MovingAverageCrossStrategy

    # Load fixtures
    btc_df = pl.read_parquet(FIXTURE_BTCUSDT)
    eth_df = pl.read_parquet(FIXTURE_ETHUSDT)

    # Build walk-forward splits
    from datetime import datetime

    base_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    splits = build_walk_forward_splits(
        base_start=base_start,
        folds=N_FOLDS,
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )

    # Build param grid: 9 combos
    param_grid = [
        {"short_window": sw, "long_window": lw, "allow_short": ALLOW_SHORT}
        for sw, lw in product(SHORT_WINDOWS, LONG_WINDOWS)
    ]

    np.random.seed(SEED)

    # Write fixtures to temp CSV for HistoricCSVDataHandler compatibility
    import tempfile, os

    tmp_dir = tempfile.mkdtemp(prefix="lq_wf_bench_")
    try:
        btc_csv = os.path.join(tmp_dir, "BTCUSDT.csv")
        eth_csv = os.path.join(tmp_dir, "ETHUSDT.csv")
        btc_df.write_csv(btc_csv)
        eth_df.write_csv(eth_csv)

        t0 = perf_counter()

        for fold in splits:
            train_start = fold["train_start"]
            test_end = fold["test_end"]

            for params in param_grid:
                backtest = Backtest(
                    csv_dir=tmp_dir,
                    symbol_list=["BTC/USDT", "ETH/USDT"],
                    start_date=train_start,
                    data_handler_cls=HistoricCSVDataHandler,
                    execution_handler_cls=SimulatedExecutionHandler,
                    portfolio_cls=Portfolio,
                    strategy_cls=MovingAverageCrossStrategy,
                    strategy_params=params,
                    record_history=False,
                    track_metrics=False,
                    record_trades=False,
                )
                backtest.simulate_trading(output=False)

        elapsed = perf_counter() - t0
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    return elapsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time canonical 9-combo × 3-fold walk-forward run (Axis 1 E2E)."
    )
    parser.add_argument(
        "--output",
        default="docs/perf/data/bench_walkforward_e2e.json",
        help=(
            "Output JSON path.  Defaults to docs/perf/data/ — NOT baseline/, which "
            "is a frozen Phase 0 artifact set.  Pass an explicit path or use "
            "--output /tmp/... for ad-hoc runs."
        ),
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip SHA-256 fixture verification (for testing only).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    print("Verifying fixtures against PROVENANCE.json...")
    if args.skip_sha256:
        print("  [SKIP] SHA-256 verification skipped (--skip-sha256).")
        provenance: dict = {}
    else:
        provenance = _verify_provenance()

    n_combos = len(SHORT_WINDOWS) * len(LONG_WINDOWS)
    total_runs = N_FOLDS * n_combos
    print(
        f"Running canonical walk-forward: {N_FOLDS} folds × {n_combos} combos "
        f"= {total_runs} backtests..."
    )

    elapsed = _run_e2e_walkforward()

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "spec": {
            "strategy": "MovingAverageCrossStrategy",
            "allow_short": ALLOW_SHORT,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "fixture_btcusdt": str(FIXTURE_BTCUSDT),
            "fixture_ethusdt": str(FIXTURE_ETHUSDT),
            "short_windows": SHORT_WINDOWS,
            "long_windows": LONG_WINDOWS,
            "n_combos": n_combos,
            "n_folds": N_FOLDS,
            "train_months": TRAIN_MONTHS,
            "val_months": VAL_MONTHS,
            "test_months": TEST_MONTHS,
            "step_months": STEP_MONTHS,
            "start_date": START_DATE_STR,
            "seed": SEED,
            "total_backtest_runs": total_runs,
        },
        "walk_forward_e2e_elapsed_s": elapsed,
        "walk_forward_e2e_runs_per_sec": float(total_runs) / elapsed if elapsed > 0 else 0.0,
        "provenance_path": str(PROVENANCE_PATH),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {out}")
    print(
        f"walk_forward_e2e_elapsed_s = {elapsed:.3f}s  "
        f"({total_runs} runs, {elapsed / total_runs * 1000:.1f}ms/run)"
    )


if __name__ == "__main__":
    main()
