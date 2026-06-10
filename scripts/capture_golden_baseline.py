"""Golden baseline capture for lumina_quant Phase 0b.

This script freezes the current numerical behaviour as a versioned oracle that
all later refactoring phases compare against (rtol 1e-8 contract).

Usage:
    uv run python scripts/capture_golden_baseline.py           # regenerate goldens
    uv run python scripts/capture_golden_baseline.py --dry-run # verify only, no writes

The aggTrades fixture MUST be committed before running the default path.
Fetch it once with:
    uv run python scripts/fetch_aggtrades_fixture.py

NEVER modify this script to fetch from the network in the default path —
doing so would produce a different window on a fresh machine and break
acceptance criterion #2 (bit-exact reproduction across machines).

Regeneration procedure (for baseline/README.md):
    1. uv run python scripts/fetch_aggtrades_fixture.py   # once only; commit result
    2. uv run python scripts/capture_golden_baseline.py   # any time; deterministic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import time

import numpy as np
import polars as pl

# ── Seed: set here, before ANY random call in this process ───────────────────
NUMPY_SEED = 42
np.random.seed(NUMPY_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

GOLDEN_DIR = REPO_ROOT / "baseline" / "golden"
CONFIGS_DIR = GOLDEN_DIR / "configs"
OHLCV_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "ohlcv"
AGGTRADES_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "aggtrades"

# ── Fixture identifiers ───────────────────────────────────────────────────────
AGGTRADES_SYMBOL = "BTCUSDT"
AGGTRADES_START_MS = 1781046000000
AGGTRADES_END_MS = 1781049600000
AGGTRADES_FIXTURE_NAME = f"{AGGTRADES_SYMBOL}_1h_{AGGTRADES_START_MS}_{AGGTRADES_END_MS}.parquet"

# ── Backtest parameters ───────────────────────────────────────────────────────
OHLCV_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
OHLCV_DAYS = 1000
OHLCV_START = datetime(2022, 1, 1)
ANNUAL_PERIODS = 252

# ── Walk-forward grid (canonical spec — shared with worker-3 E2E timing axis) ─
# short_window=[10,20,30] x long_window=[40,80,120] = 9 combos
# 3 folds; train=6mo, val=3mo, test=3mo, step=3mo, start=2022-01-01
WF_SHORT_WINDOWS = [10, 20, 30]
WF_LONG_WINDOWS = [40, 80, 120]
WF_FOLDS = 3
WF_TRAIN_MONTHS = 6
WF_VAL_MONTHS = 3
WF_TEST_MONTHS = 3
WF_STEP_MONTHS = 3


# ── Helpers ───────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def _write_parquet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"  wrote {path.relative_to(REPO_ROOT)} ({len(df)} rows)")


# ── Step 1: Generate synthetic OHLCV fixtures ─────────────────────────────────


def generate_ohlcv(symbol: str) -> pl.DataFrame:
    """Deterministic OHLCV generation — caller must have set np.random.seed."""
    dates = [OHLCV_START + timedelta(days=i) for i in range(OHLCV_DAYS)]
    n = OHLCV_DAYS
    returns = np.random.randn(n) * 0.02
    price_path = 100.0 * np.cumprod(1.0 + returns)
    opens = price_path.copy()
    highs = opens * (1.0 + np.abs(np.random.randn(n) * 0.01))
    lows = opens * (1.0 - np.abs(np.random.randn(n) * 0.01))
    closes = lows + (highs - lows) * np.random.rand(n)
    volumes = np.random.randint(100, 1000, size=n).astype(np.float64)
    return pl.DataFrame(
        {
            "datetime": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime))


def step_generate_fixtures(dry_run: bool) -> dict[str, dict]:
    """Return {symbol: {"path": ..., "sha256": ..., "df": ...}}"""
    print("\n[1/7] Generating synthetic OHLCV fixtures (seed=42)...")
    result = {}
    for sym in OHLCV_SYMBOLS:
        df = generate_ohlcv(sym)
        path = OHLCV_FIXTURE_DIR / f"{sym}_seed{NUMPY_SEED}_{OHLCV_DAYS}d.parquet"
        if not dry_run:
            _write_parquet(path, df)
            sha = sha256_file(path)
        else:
            sha = "<dry-run>"
        result[sym] = {"path": path, "sha256": sha, "df": df}
        print(f"  {sym}: {len(df)} rows, close[0]={df['close'][0]:.4f}")
    return result


# ── Step 2: Freeze configs ────────────────────────────────────────────────────


def step_freeze_configs(dry_run: bool) -> dict[str, str]:
    """Copy config files to baseline/golden/configs/ and return SHA-256 map."""
    print("\n[2/7] Freezing configs...")
    import shutil

    srcs = {
        "config_frozen.yaml": REPO_ROOT / "config.yaml",
        "research_frozen.yaml": REPO_ROOT / "configs" / "profiles" / "research.yaml",
    }
    shas: dict[str, str] = {}
    for dst_name, src in srcs.items():
        dst = CONFIGS_DIR / dst_name
        if not dry_run:
            CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            shas[dst_name] = sha256_file(dst)
        else:
            shas[dst_name] = sha256_file(src)
        print(f"  {dst_name}: sha256={shas[dst_name][:16]}...")
    return shas


# ── Step 3: Backtest goldens ──────────────────────────────────────────────────


def _run_ma_cross_backtest(data_dict: dict[str, pl.DataFrame]) -> dict:
    """Run MovingAverageCrossStrategy on both symbols; return captured artefacts."""
    from lumina_quant.backtesting.backtest import Backtest
    from lumina_quant.backtesting.data import HistoricCSVDataHandler
    from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
    from lumina_quant.backtesting.portfolio_backtest import Portfolio
    from lumina_quant.strategies.moving_average import MovingAverageCrossStrategy

    bt = Backtest(
        csv_dir="",
        symbol_list=list(data_dict.keys()),
        start_date=None,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=MovingAverageCrossStrategy,
        strategy_params={"short_window": 10, "long_window": 30, "allow_short": True},
        data_dict=data_dict,
        record_history=True,
        track_metrics=True,
        record_trades=True,
    )
    stats = bt.simulate_trading(output=True, persist_output=False, verbose=False)

    equity = getattr(bt.portfolio, "equity_curve", None)
    trades = getattr(bt.portfolio, "trades", [])
    positions = getattr(bt.portfolio, "all_positions", [])
    return {
        "stats": stats,
        "equity_curve": equity,
        "trades": trades,
        "positions": positions,
    }


def _run_buyholdbacktest(data_dict: dict[str, pl.DataFrame]) -> dict:
    """Run BitcoinBuyHoldStrategy on BTCUSDT as sanity golden."""
    from lumina_quant.backtesting.backtest import Backtest
    from lumina_quant.backtesting.data import HistoricCSVDataHandler
    from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
    from lumina_quant.backtesting.portfolio_backtest import Portfolio
    from lumina_quant.strategies.bitcoin_buy_hold import BitcoinBuyHoldStrategy

    bt = Backtest(
        csv_dir="",
        symbol_list=["BTCUSDT"],
        start_date=None,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=BitcoinBuyHoldStrategy,
        strategy_params={"symbol": "BTCUSDT", "strength": 1.0},
        data_dict={"BTCUSDT": data_dict["BTCUSDT"]},
        record_history=True,
        track_metrics=True,
        record_trades=True,
    )
    stats = bt.simulate_trading(output=True, persist_output=False, verbose=False)
    equity = getattr(bt.portfolio, "equity_curve", None)
    trades = getattr(bt.portfolio, "trades", [])
    return {"stats": stats, "equity_curve": equity, "trades": trades}


def step_backtest_goldens(fixtures: dict[str, dict], dry_run: bool) -> dict:
    print("\n[3/7] Running backtest goldens...")
    data_dict = {sym: v["df"] for sym, v in fixtures.items()}

    ma_result = _run_ma_cross_backtest(data_dict)
    bh_result = _run_buyholdbacktest(data_dict)

    captured: dict[str, object] = {}

    # MA Cross: equity curve, stats, trades
    if ma_result["equity_curve"] is not None:
        ec = ma_result["equity_curve"]
        path = GOLDEN_DIR / "ma_cross_equity_curve.parquet"
        if not dry_run:
            _write_parquet(path, ec)
        captured["ma_cross_equity_curve"] = str(path.relative_to(REPO_ROOT))
        print(f"  MA cross equity rows={len(ec)}, stats={ma_result['stats']}")
    else:
        print("  WARNING: MA cross equity_curve is None")

    if not dry_run:
        _write_json(GOLDEN_DIR / "ma_cross_stats.json", ma_result["stats"] or {})
        _write_json(GOLDEN_DIR / "ma_cross_trades.json", ma_result["trades"])
    captured["ma_cross_stats"] = str((GOLDEN_DIR / "ma_cross_stats.json").relative_to(REPO_ROOT))
    captured["ma_cross_trades"] = str((GOLDEN_DIR / "ma_cross_trades.json").relative_to(REPO_ROOT))

    # Positions history (first 100 rows for size)
    pos_sample = (ma_result["positions"] or [])[:100]
    if not dry_run:
        _write_json(GOLDEN_DIR / "ma_cross_positions_sample.json", pos_sample)
    captured["ma_cross_positions_sample"] = str(
        (GOLDEN_DIR / "ma_cross_positions_sample.json").relative_to(REPO_ROOT)
    )

    # BuyHold sanity
    if bh_result["equity_curve"] is not None:
        path = GOLDEN_DIR / "buyholdstrategy_equity_curve.parquet"
        if not dry_run:
            _write_parquet(path, bh_result["equity_curve"])
        captured["buyholdstrategy_equity_curve"] = str(path.relative_to(REPO_ROOT))
        print(f"  BuyHold equity rows={len(bh_result['equity_curve'])}, stats={bh_result['stats']}")
    if not dry_run:
        _write_json(GOLDEN_DIR / "buyholdstrategy_stats.json", bh_result["stats"] or {})
    captured["buyholdstrategy_stats"] = str(
        (GOLDEN_DIR / "buyholdstrategy_stats.json").relative_to(REPO_ROOT)
    )

    return captured


# ── Step 4: Native backend goldens ────────────────────────────────────────────


def _call_pyo3_metrics(close_arr: np.ndarray) -> dict:
    """Call lumina_compute.evaluate_metrics via pyo3 binding."""
    from lumina_quant._compute import evaluate_metrics  # type: ignore[attr-defined]

    arr = np.ascontiguousarray(close_arr, dtype=np.float64)
    sharpe, cagr, mdd = evaluate_metrics(arr, ANNUAL_PERIODS)
    return {"sharpe": float(sharpe), "cagr": float(cagr), "max_dd": float(mdd)}


def _call_alpha_fold(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict:
    """Exercise rust_alpha_fold via its Python wrapper."""
    os.environ["LQ_ALPHA_FOLD_BACKEND"] = "rust"
    from lumina_quant.alpha_zoo.native_alpha_fold_backend import simulate_symbol_arrays

    signal = np.where(close > np.roll(close, 5), 1.0, -1.0)
    signal[:5] = 0.0
    returns, liquidation, wipeout = simulate_symbol_arrays(
        close,
        high,
        low,
        signal,
        integer_leverage=2,
        allocation_fraction=0.5,
        round_trip_cost_bps=10.0,
    )
    return {
        "backend": "rust",
        "n": len(returns),
        "total_return": float(np.sum(returns)),
        "liquidation_count": int(np.sum(liquidation)),
        "wipeout_count": int(np.sum(wipeout)),
        "returns_first5": returns[:5].tolist(),
        "returns_last5": returns[-5:].tolist(),
    }


def _call_hybrid_optuna(close: np.ndarray) -> dict:
    """Exercise rust_hybrid_optuna via its Python wrapper."""
    os.environ["LQ_HYBRID_OPTUNA_BACKEND"] = "rust"
    from lumina_quant.alpha_zoo.native_hybrid_optuna_backend import evaluate_hybrid_portfolio_native

    # 3 synthetic strategy return streams (columns) over the close series.
    n = len(close)
    returns_2d = np.zeros((n, 3), dtype=np.float64)
    r = np.diff(close) / close[:-1]
    returns_2d[1:, 0] = r
    returns_2d[1:, 1] = -r  # inverse
    returns_2d[1:, 2] = r * 0.5  # half momentum
    # Add base capital (10000) so values are finite equity-like series.
    equity_2d = 10000.0 * np.cumprod(1.0 + np.clip(returns_2d, -0.5, 0.5), axis=0)

    result = evaluate_hybrid_portfolio_native(
        equity_2d,
        version="v3_5",
        start_idx=20,
        mape_window=20,
        bias_window=20,
        short_vol_window=10,
        bias_correction_alpha=0.1,
        bias_combine_ratio=0.5,
        max_single_weight=0.6,
        default_idx=0,
        high_vol_best_idx=1,
        default_weight_ratio=0.5,
        high_vol_threshold=0.02,
        high_vol_weight_boost=0.2,
    )
    if result is None:
        return {
            "backend": "unavailable",
            "note": "hybrid_optuna returned None (fallback to Python)",
        }
    portfolio, weights_exp, _weights_raw, _d0, _d1, _d2 = result
    return {
        "backend": "rust",
        "n": len(portfolio),
        "portfolio_final": float(portfolio[-1]) if len(portfolio) else None,
        "weights_exp_final": weights_exp[-1].tolist() if len(weights_exp) else None,
    }


def _call_live_signals(close: np.ndarray) -> dict:
    """Exercise rust_live_signals debounced + trailing via Python wrapper."""
    os.environ["LQ_LIVE_SIGNAL_BACKEND"] = "rust"
    from lumina_quant.alpha_zoo.native_live_signal_backend import (
        evaluate_debounced_state_native,
        evaluate_trailing_state_native,
    )

    sma5 = np.convolve(close, np.ones(5) / 5, mode="same")
    long_entry = (close > sma5).astype(np.uint8)
    long_exit = (close < sma5).astype(np.uint8)
    short_entry = long_exit.copy()
    short_exit = long_entry.copy()

    deb = evaluate_debounced_state_native(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side="both",
        min_hold_bars=3,
        cooldown_bars=2,
    )

    atr = np.abs(np.diff(close, prepend=close[0])) * 2.0
    trail = evaluate_trailing_state_native(
        close,
        long_entry,
        short_entry,
        long_exit,
        short_exit,
        atr,
        side="both",
        min_hold_bars=3,
        cooldown_bars=2,
        trail_atr_mult=2.0,
    )

    return {
        "debounced": {
            "available": deb is not None,
            "n": len(deb) if deb is not None else 0,
            "signal_changes": int(np.sum(np.diff(deb) != 0)) if deb is not None else 0,
            "first5": deb[:5].tolist() if deb is not None else [],
        },
        "trailing": {
            "available": trail is not None,
            "n": len(trail) if trail is not None else 0,
            "signal_changes": int(np.sum(np.diff(trail) != 0)) if trail is not None else 0,
            "first5": trail[:5].tolist() if trail is not None else [],
        },
    }


def _call_rawfirst(aggtrades_df: pl.DataFrame) -> dict:
    """Exercise rust_rawfirst aggregate_raw_aggtrades_to_1s via Python wrapper."""
    os.environ["LQ_RAW_FIRST_BACKEND"] = "rust"
    from lumina_quant.data.native_raw_first_backend import aggregate_raw_aggtrades_to_1s_native

    complete_through = int(aggtrades_df["timestamp_ms"][-1]) - 1000
    result = aggregate_raw_aggtrades_to_1s_native(
        aggtrades_df,
        range_start_ms=AGGTRADES_START_MS,
        range_end_ms=AGGTRADES_END_MS,
        previous_close=None,
        complete_through_ms=complete_through,
    )
    if result is None or result.is_empty():
        return {"backend": "unavailable", "n": 0}
    return {
        "backend": "rust",
        "n": len(result),
        "columns": result.columns,
        "first_row": result.row(0, named=True) if len(result) else {},
        "last_row": result.row(-1, named=True) if len(result) else {},
    }


def step_native_backend_goldens(
    fixtures: dict[str, dict],
    aggtrades_df: pl.DataFrame,
    dry_run: bool,
) -> dict:
    print("\n[4/7] Exercising all 5 native kernels (pyo3)...")
    btc_df = fixtures["BTCUSDT"]["df"]
    close = btc_df["close"].to_numpy()
    high = btc_df["high"].to_numpy()
    low = btc_df["low"].to_numpy()

    results: dict[str, dict] = {}

    print("  rust_metrics (pyo3)...")
    results["rust_metrics"] = _call_pyo3_metrics(close)

    print("  c_metrics (pyo3 — consolidated into lumina_compute)...")
    results["c_metrics"] = _call_pyo3_metrics(close)

    print("  rust_alpha_fold...")
    results["rust_alpha_fold"] = _call_alpha_fold(close, high, low)

    print("  rust_hybrid_optuna...")
    results["rust_hybrid_optuna"] = _call_hybrid_optuna(close)

    print("  rust_live_signals...")
    results["rust_live_signals"] = _call_live_signals(close)

    print("  rust_rawfirst (aggTrades -> 1s OHLCV)...")
    results["rust_rawfirst"] = _call_rawfirst(aggtrades_df)

    for name, r in results.items():
        print(f"    {name}: {r}")

    if not dry_run:
        _write_json(GOLDEN_DIR / "native_backends.json", results)

    return results


# ── Step 5: Walk-forward golden ───────────────────────────────────────────────


def _add_months(dt: datetime, months: int) -> datetime:
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    max_day = 29 if month == 2 and leap else days_in_month[month - 1]
    return dt.replace(year=year, month=month, day=min(dt.day, max_day))


def _ma_cross_equity(
    close: np.ndarray,
    short_w: int,
    long_w: int,
    context: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised MA cross equity curve — no event machinery needed for grid search.

    If *context* is supplied it is prepended so both MAs are fully warmed-up
    at the first bar of *close*.  The returned equity covers *close* only,
    re-normalised to start at 10 000.  This is variant (a) of the canonical
    walk-forward spec: feed each eval window the preceding max(long_window)
    bars as warmup context so indicators are defined from bar 0 of the window.
    """
    if context is not None and len(context) > 0:
        full = np.concatenate([context, close])
        n_full = len(full)
        n_ctx = len(context)
        short_ma = np.convolve(full, np.ones(short_w) / short_w, mode="full")[:n_full]
        long_ma = np.convolve(full, np.ones(long_w) / long_w, mode="full")[:n_full]
        signal_full = np.where(short_ma > long_ma, 1.0, -1.0)
        signal_full[:long_w] = 0.0  # zero-out initial warmup in the context portion
        daily_ret_full = np.diff(full, prepend=full[0]) / np.where(full == 0.0, 1.0, full)
        equity_full = 10000.0 * np.cumprod(1.0 + signal_full * daily_ret_full)
        window_eq = equity_full[n_ctx:]
        if len(window_eq) == 0:
            return np.full(len(close), 10000.0)
        # Re-normalise window equity to start at 10 000 for independent metrics
        start_val = window_eq[0]
        if start_val != 0.0:
            window_eq = window_eq * (10000.0 / start_val)
        return window_eq

    n = len(close)
    if n < long_w + 1:
        return np.full(n, 10000.0)
    short_ma = np.convolve(close, np.ones(short_w) / short_w, mode="full")[:n]
    long_ma = np.convolve(close, np.ones(long_w) / long_w, mode="full")[:n]
    # Signal: +1 when short > long, -1 otherwise (after warmup)
    signal = np.where(short_ma > long_ma, 1.0, -1.0)
    signal[:long_w] = 0.0
    daily_ret = np.diff(close, prepend=close[0]) / np.where(close == 0.0, 1.0, close)
    equity = 10000.0 * np.cumprod(1.0 + signal * daily_ret)
    return equity


def step_walk_forward(fixtures: dict[str, dict], dry_run: bool) -> dict:
    """Produce two golden variants in one pass.

    Variant A  walk_forward_results.json          — preservation oracle.
               Val/test evaluated WITHOUT warmup context; emits the legacy
               -999 sentinel where MA cannot warm up (lw=120 in 90-day
               windows).  Freezes current code behaviour exactly.

    Variant B  walk_forward_results_warmup.json   — richer regression target.
               Val/test evaluated WITH 120-bar warmup context prepended so
               MAs are fully defined from window bar 0; all folds produce
               real metrics.  Phase 4 will be compared against this file.
    """
    n_combos = len(WF_SHORT_WINDOWS) * len(WF_LONG_WINDOWS)
    warmup_bars = max(WF_LONG_WINDOWS)  # = 120
    print(
        f"\n[5/7] Running walk-forward grid "
        f"({WF_FOLDS} folds x {n_combos} combos) — variants A (no-ctx) + B ({warmup_bars}-bar ctx)..."
    )
    from lumina_quant.optimization.walkers import build_walk_forward_splits
    from lumina_quant.optimization.native_backend import evaluate_metrics_backend

    btc_df = fixtures["BTCUSDT"]["df"]
    close_full = btc_df["close"].to_numpy()
    dates_full = btc_df["datetime"].to_list()

    def _slice(start: datetime, end: datetime) -> np.ndarray:
        idxs = [i for i, d in enumerate(dates_full) if start <= d < end]
        return close_full[idxs] if idxs else np.array([10000.0, 10000.0])

    def _context_before(window_start: datetime) -> np.ndarray:
        """Return up to warmup_bars immediately preceding window_start."""
        idxs = [i for i, d in enumerate(dates_full) if d < window_start]
        return close_full[idxs[-warmup_bars:]] if idxs else np.array([])

    splits = build_walk_forward_splits(
        base_start=OHLCV_START,
        folds=WF_FOLDS,
        train_months=WF_TRAIN_MONTHS,
        val_months=WF_VAL_MONTHS,
        test_months=WF_TEST_MONTHS,
        step_months=WF_STEP_MONTHS,
    )

    # Timer wraps the full evaluation work only (split-building above is trivial)
    wf_start_t = time.perf_counter()

    folds_a: list[dict] = []  # variant A — no warmup context
    folds_b: list[dict] = []  # variant B — with warmup context

    for split in splits:
        fold = split["fold"]
        train_start, train_end = split["train_start"], split["train_end"]
        val_start, val_end = split["val_start"], split["val_end"]
        test_start, test_end = split["test_start"], split["test_end"]

        train_close = _slice(train_start, train_end)
        val_close = _slice(val_start, val_end)
        test_close = _slice(test_start, test_end)
        val_ctx = _context_before(val_start)
        test_ctx = _context_before(test_start)

        # ── Grid search on train (identical for both variants) ────────────────
        best_sharpe = -999.0
        best_params: dict = {}
        grid_rows: list[dict] = []

        for sw in WF_SHORT_WINDOWS:
            for lw in WF_LONG_WINDOWS:
                if lw <= sw:
                    continue
                eq = _ma_cross_equity(train_close, sw, lw)
                sharpe, cagr, mdd = evaluate_metrics_backend(eq, ANNUAL_PERIODS)
                grid_rows.append(
                    {
                        "short_window": sw,
                        "long_window": lw,
                        "train_sharpe": sharpe,
                        "train_cagr": cagr,
                        "train_max_dd": mdd,
                    }
                )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {"short_window": sw, "long_window": lw}

        sw_b, lw_b = best_params["short_window"], best_params["long_window"]

        # ── Variant A: no context (legacy behaviour, may emit -999) ──────────
        vm_a = evaluate_metrics_backend(_ma_cross_equity(val_close, sw_b, lw_b), ANNUAL_PERIODS)
        tm_a = evaluate_metrics_backend(_ma_cross_equity(test_close, sw_b, lw_b), ANNUAL_PERIODS)

        # ── Variant B: with warmup context (real metrics for every fold) ──────
        vm_b = evaluate_metrics_backend(
            _ma_cross_equity(val_close, sw_b, lw_b, context=val_ctx), ANNUAL_PERIODS
        )
        tm_b = evaluate_metrics_backend(
            _ma_cross_equity(test_close, sw_b, lw_b, context=test_ctx), ANNUAL_PERIODS
        )

        base = {
            "fold": fold,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "val_start": val_start.isoformat(),
            "val_end": val_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "best_params": best_params,
            "train_grid": grid_rows,
        }
        folds_a.append(
            {
                **base,
                "val_metrics": {"sharpe": vm_a[0], "cagr": vm_a[1], "max_dd": vm_a[2]},
                "test_metrics": {"sharpe": tm_a[0], "cagr": tm_a[1], "max_dd": tm_a[2]},
            }
        )
        folds_b.append(
            {
                **base,
                "val_metrics": {"sharpe": vm_b[0], "cagr": vm_b[1], "max_dd": vm_b[2]},
                "test_metrics": {"sharpe": tm_b[0], "cagr": tm_b[1], "max_dd": tm_b[2]},
            }
        )

        print(
            f"  fold {fold}: best={best_params} | "
            f"A val={vm_a[0]:.4f}/test={tm_a[0]:.4f} | "
            f"B val={vm_b[0]:.4f}/test={tm_b[0]:.4f}"
        )

    # split_build_elapsed_s is the numpy-path eval time — NOT the event-driven
    # E2E framework time.  Authoritative E2E timing lives in worker-3's
    # perf-baseline.json (170.71 s for the identical 27-eval workload).
    split_build_elapsed_s = round(time.perf_counter() - wf_start_t, 6)
    print(
        f"  numpy-path elapsed: {split_build_elapsed_s}s (authoritative E2E: worker-3 perf-baseline.json)"
    )

    grid_params = {"short_windows": WF_SHORT_WINDOWS, "long_windows": WF_LONG_WINDOWS}

    # split_build_elapsed_s is wall-clock only — excluded from the golden artifact
    # so byte-level determinism holds across runs.  Timing is recorded to stdout.
    payload_a = {
        "variant": "A",
        "description": "preservation oracle — current behaviour incl. -999 sentinels",
        "splits": folds_a,
        "grid_params": grid_params,
    }
    payload_b = {
        "variant": "B",
        "description": "warmup-context oracle — real metrics for all folds",
        "splits": folds_b,
        "grid_params": grid_params,
        "warmup_context_bars": warmup_bars,
    }

    if not dry_run:
        _write_json(GOLDEN_DIR / "walk_forward_results.json", payload_a)
        _write_json(GOLDEN_DIR / "walk_forward_results_warmup.json", payload_b)

    return {"variant_a": payload_a, "variant_b": payload_b}


# ── Step 6: Verify aggTrades fixture ─────────────────────────────────────────


def step_verify_aggtrades(dry_run: bool) -> tuple[pl.DataFrame, dict]:
    print("\n[6/7] Verifying aggTrades fixture...")
    fixture_path = AGGTRADES_FIXTURE_DIR / AGGTRADES_FIXTURE_NAME
    prov_path = GOLDEN_DIR / "PROVENANCE.json"

    if not fixture_path.exists():
        print(f"  ERROR: fixture not found: {fixture_path}")
        print("  Run: uv run python scripts/fetch_aggtrades_fixture.py")
        sys.exit(1)

    actual_sha = sha256_file(fixture_path)

    # If PROVENANCE.json already has the hash, verify against it.
    if prov_path.exists() and not dry_run:
        try:
            prov = json.loads(prov_path.read_text())
            expected = prov.get("aggtrades_fixture", {}).get("sha256", "")
            if expected and expected != actual_sha:
                print("  FATAL: aggTrades fixture SHA-256 mismatch!")
                print(f"    expected: {expected}")
                print(f"    actual  : {actual_sha}")
                print("  Fixture has changed. Re-fetch and get new baseline approval.")
                sys.exit(1)
        except Exception:
            pass  # First run — no existing PROVENANCE yet.

    df = pl.read_parquet(fixture_path)
    print(f"  fixture: {fixture_path.name}, rows={len(df)}, sha256={actual_sha[:16]}...")

    meta = {
        "status": "captured",
        "path": f"tests/fixtures/aggtrades/{AGGTRADES_FIXTURE_NAME}",
        "symbol": AGGTRADES_SYMBOL,
        "start_ms": AGGTRADES_START_MS,
        "end_ms": AGGTRADES_END_MS,
        "record_count": len(df),
        "sha256": actual_sha,
    }
    return df, meta


# ── Step 7: Write PROVENANCE.json ─────────────────────────────────────────────


def step_write_provenance(
    fixtures: dict[str, dict],
    config_shas: dict[str, str],
    backtest_goldens: dict,
    native_goldens: dict,
    wf_result: dict,
    aggtrades_meta: dict,
    dry_run: bool,
) -> None:
    print("\n[7/7] Writing PROVENANCE.json...")
    import platform
    import subprocess

    env_lock_path = REPO_ROOT / "baseline" / "env.lock"
    env_lock_sha = sha256_file(env_lock_path) if env_lock_path.exists() else "MISSING"

    try:
        rustc = subprocess.check_output(["rustc", "--version"], text=True).strip()
        cargo = subprocess.check_output(["cargo", "--version"], text=True).strip()
    except Exception:
        rustc = cargo = "unavailable"

    provenance = {
        "schema_version": "1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "numpy_seed": NUMPY_SEED,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "rustc": rustc,
            "cargo": cargo,
            "env_lock_path": "baseline/env.lock",
            "env_lock_sha256": env_lock_sha,
        },
        "config": {
            "config_path": "baseline/golden/configs/config_frozen.yaml",
            "config_sha256": config_shas.get("config_frozen.yaml", ""),
            "research_profile_path": "baseline/golden/configs/research_frozen.yaml",
            "research_profile_sha256": config_shas.get("research_frozen.yaml", ""),
        },
        "input_data": {
            "type": "synthetic",
            "generator": "generate_data.py (logic embedded in capture script)",
            "numpy_seed": NUMPY_SEED,
            "symbols": OHLCV_SYMBOLS,
            "days": OHLCV_DAYS,
            "start_date": OHLCV_START.isoformat(),
            "fixtures": {
                sym: {
                    "path": f"tests/fixtures/ohlcv/{sym}_seed{NUMPY_SEED}_{OHLCV_DAYS}d.parquet",
                    "sha256": v["sha256"],
                }
                for sym, v in fixtures.items()
            },
        },
        "aggtrades_fixture": aggtrades_meta,
        "goldens": {
            **backtest_goldens,
            "native_backends": "baseline/golden/native_backends.json",
            "walk_forward_variant_a": "baseline/golden/walk_forward_results.json",
            "walk_forward_variant_b": "baseline/golden/walk_forward_results_warmup.json",
        },
        "backends": {
            "rust_metrics": {
                "crate": "native/rust_metrics",
                "lib": "native/rust_metrics/target/release/liblumina_metrics.so",
                "exported_fns": ["evaluate_metrics"],
                "golden_result": native_goldens.get("rust_metrics", {}),
            },
            "c_metrics": {
                "crate": "native/c_metrics",
                "lib": "native/c_metrics/build/liblumina_metrics.so",
                "exported_fns": ["evaluate_metrics"],
                "golden_result": native_goldens.get("c_metrics", {}),
            },
            "rust_alpha_fold": {
                "crate": "native/rust_alpha_fold",
                "lib": "native/rust_alpha_fold/target/release/liblumina_alpha_fold.so",
                "exported_fns": ["simulate_symbol_fold_native"],
                "golden_result": native_goldens.get("rust_alpha_fold", {}),
            },
            "rust_hybrid_optuna": {
                "crate": "native/rust_hybrid_optuna",
                "lib": "native/rust_hybrid_optuna/target/release/liblumina_hybrid_optuna.so",
                "exported_fns": ["evaluate_hybrid_optuna_portfolio"],
                "golden_result": native_goldens.get("rust_hybrid_optuna", {}),
            },
            "rust_live_signals": {
                "crate": "native/rust_live_signals",
                "lib": "native/rust_live_signals/target/release/liblumina_live_signals.so",
                "exported_fns": ["debounced_state_signal_native", "trailing_state_signal_native"],
                "golden_result": native_goldens.get("rust_live_signals", {}),
            },
            "rust_rawfirst": {
                "crate": "native/rust_rawfirst",
                "lib": "native/rust_rawfirst/target/release/liblumina_rawfirst.so",
                "exported_fns": ["aggregate_raw_aggtrades_to_1s", "append_ohlcv_1s_wal"],
                "golden_result": native_goldens.get("rust_rawfirst", {}),
            },
        },
        "tolerance": {
            "rtol": 1e-8,
            "determinism_contract": (
                "same backend + same seed -> bit-exact reproduction across machines; "
                "all divergences require docs/divergences/<artifact>.md"
            ),
        },
        "walk_forward": {
            "folds": WF_FOLDS,
            "train_months": WF_TRAIN_MONTHS,
            "val_months": WF_VAL_MONTHS,
            "test_months": WF_TEST_MONTHS,
            "step_months": WF_STEP_MONTHS,
            "start_date": OHLCV_START.isoformat(),
            "param_grid": {
                "short_window": WF_SHORT_WINDOWS,
                "long_window": WF_LONG_WINDOWS,
            },
            "n_combos": len(WF_SHORT_WINDOWS) * len(WF_LONG_WINDOWS),
            "variant_a": {
                "file": "baseline/golden/walk_forward_results.json",
                "description": "preservation oracle — current behaviour incl. -999 sentinels",
                "note": "Phase 4 improvement will NOT reproduce -999; divergence documented per procedure",
            },
            "variant_b": {
                "file": "baseline/golden/walk_forward_results_warmup.json",
                "description": "warmup-context oracle — real metrics for all folds",
                "warmup_context_bars": wf_result.get("variant_b", {}).get("warmup_context_bars"),
            },
            "split_build_elapsed_s": wf_result.get("variant_a", {}).get("split_build_elapsed_s"),
            "authoritative_e2e_timing": "baseline/perf-baseline.json (worker-3, 170.71s)",
        },
    }

    if not dry_run:
        _write_json(GOLDEN_DIR / "PROVENANCE.json", provenance)
        print(f"\n  PROVENANCE.json written to {GOLDEN_DIR / 'PROVENANCE.json'}")
    else:
        print("  [dry-run] PROVENANCE.json not written")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture golden baselines.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify and print without writing any files."
    )
    args = parser.parse_args()
    dry_run: bool = args.dry_run

    # Re-seed here too — defensive against any import-time randomness.
    np.random.seed(NUMPY_SEED)

    print("=" * 60)
    print(f"lumina_quant golden baseline capture (seed={NUMPY_SEED})")
    print(f"dry_run={dry_run}")
    print("=" * 60)

    fixtures = step_generate_fixtures(dry_run)
    config_shas = step_freeze_configs(dry_run)
    backtest_goldens = step_backtest_goldens(fixtures, dry_run)
    aggtrades_df, aggtrades_meta = step_verify_aggtrades(dry_run)
    native_goldens = step_native_backend_goldens(fixtures, aggtrades_df, dry_run)
    wf_result = step_walk_forward(fixtures, dry_run)
    step_write_provenance(
        fixtures,
        config_shas,
        backtest_goldens,
        native_goldens,
        wf_result,
        aggtrades_meta,
        dry_run,
    )

    print("\n✓ Golden baseline capture complete.")


if __name__ == "__main__":
    main()
