"""Phase 4.3 + 4.5 integration test: walk-forward golden comparison.

Runs the MA-cross walk-forward grid WITH warmup context (Variant B style),
then compares every numeric field to ``baseline/golden/walk_forward_results_warmup.json``
at rtol ``1e-8`` (``validation.golden_rtol``).

Correctness gates
-----------------
* Every fold must produce a real Sharpe (not ``-999``).
* Results must match Variant B golden within rtol ``1e-8``.

Divergence from Variant A folds 1 & 3 (``val_metrics.sharpe == -999``) is a
documented improvement — see ``docs/divergences/walk_forward_no_sentinel.md``.

Invariants preserved
--------------------
* Same fixture: ``tests/fixtures/ohlcv/BTCUSDT_seed42_1000d.parquet``
* Same constants as ``scripts/capture_golden_baseline.py``
* Warmup context = 120 bars preceding each eval window (max long window)
* ``evaluate_metrics_backend`` is used so the pyo3 / numba / python backend
  selection is exercised identically to how goldens were captured.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lumina_quant.backtesting.golden_comparator import compare_to_golden
from lumina_quant.optimization.native_backend import evaluate_metrics_backend
from lumina_quant.optimization.walkers import (
    InsufficientWarmupError,
    build_walk_forward_splits,
    check_warmup_sufficient,
    ma_cross_equity,
)

# ── Constants (must match scripts/capture_golden_baseline.py) ─────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "ohlcv" / "BTCUSDT_seed42_1000d.parquet"
_GOLDEN_B = _REPO_ROOT / "baseline" / "golden" / "walk_forward_results_warmup.json"

ANNUAL_PERIODS = 252
WF_SHORT_WINDOWS = [10, 20, 30]
WF_LONG_WINDOWS = [40, 80, 120]
WF_FOLDS = 3
WF_TRAIN_MONTHS = 6
WF_VAL_MONTHS = 3
WF_TEST_MONTHS = 3
WF_STEP_MONTHS = 3
OHLCV_START = datetime(2022, 1, 1)
WARMUP_BARS = max(WF_LONG_WINDOWS)  # 120


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def btc_close_and_dates():
    """Load BTCUSDT synthetic fixture once per module."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not found: {_FIXTURE_PATH}")
    df = pl.read_parquet(_FIXTURE_PATH)
    close = df["close"].to_numpy()
    dates = df["datetime"].to_list()
    return close, dates


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slice_close(close_full, dates_full, start: datetime, end: datetime) -> np.ndarray:
    idxs = [i for i, d in enumerate(dates_full) if start <= d < end]
    return close_full[idxs] if idxs else np.array([10_000.0, 10_000.0])


def _context_before(close_full, dates_full, window_start: datetime, n_bars: int) -> np.ndarray:
    """Return up to *n_bars* prices immediately preceding *window_start*."""
    idxs = [i for i, d in enumerate(dates_full) if d < window_start]
    return close_full[idxs[-n_bars:]] if idxs else np.array([])


def _best_train_params(train_grid: list[dict]) -> tuple[int, int]:
    """Select (short_w, long_w) with highest train_sharpe."""
    best = max(train_grid, key=lambda r: float(r["train_sharpe"]))
    return int(best["short_window"]), int(best["long_window"])


# ── Main walk-forward computation ─────────────────────────────────────────────

def _run_walk_forward_b(close_full, dates_full) -> dict:
    """Reproduce Variant B: walk-forward WITH warmup context."""
    splits = build_walk_forward_splits(
        base_start=OHLCV_START,
        folds=WF_FOLDS,
        train_months=WF_TRAIN_MONTHS,
        val_months=WF_VAL_MONTHS,
        test_months=WF_TEST_MONTHS,
        step_months=WF_STEP_MONTHS,
    )

    folds_b: list[dict] = []

    for split in splits:
        fold = split["fold"]
        train_start, train_end = split["train_start"], split["train_end"]
        val_start, val_end = split["val_start"], split["val_end"]
        test_start, test_end = split["test_start"], split["test_end"]

        train_close = _slice_close(close_full, dates_full, train_start, train_end)
        val_close = _slice_close(close_full, dates_full, val_start, val_end)
        test_close = _slice_close(close_full, dates_full, test_start, test_end)

        # Grid search over training window (no warmup needed — full train data)
        train_grid: list[dict] = []
        best_sharpe = float("-inf")
        best_params = (WF_SHORT_WINDOWS[0], WF_LONG_WINDOWS[-1])

        for sw in WF_SHORT_WINDOWS:
            for lw in WF_LONG_WINDOWS:
                eq = ma_cross_equity(train_close, sw, lw)
                sharpe, cagr, mdd = evaluate_metrics_backend(eq, ANNUAL_PERIODS)
                train_grid.append(
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
                    best_params = (sw, lw)

        sw_b, lw_b = best_params

        # Variant B: warmup context for val and test
        val_ctx = _context_before(close_full, dates_full, val_start, WARMUP_BARS)
        test_ctx = _context_before(close_full, dates_full, test_start, WARMUP_BARS)

        # Phase 4.3 guard: raise InsufficientWarmupError if context is missing
        check_warmup_sufficient(val_close, lw_b, fold=fold, context=val_ctx)
        check_warmup_sufficient(test_close, lw_b, fold=fold, context=test_ctx)

        vm_b = evaluate_metrics_backend(
            ma_cross_equity(val_close, sw_b, lw_b, context=val_ctx), ANNUAL_PERIODS
        )
        tm_b = evaluate_metrics_backend(
            ma_cross_equity(test_close, sw_b, lw_b, context=test_ctx), ANNUAL_PERIODS
        )

        folds_b.append(
            {
                "fold": fold,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "val_start": val_start.isoformat(),
                "val_end": val_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "best_params": {"short_window": sw_b, "long_window": lw_b},
                "train_grid": train_grid,
                "val_metrics": {"sharpe": vm_b[0], "cagr": vm_b[1], "max_dd": vm_b[2]},
                "test_metrics": {"sharpe": tm_b[0], "cagr": tm_b[1], "max_dd": tm_b[2]},
            }
        )

    return {
        "variant": "B",
        "description": "warmup-context oracle — real metrics for all folds",
        "splits": folds_b,
        "grid_params": {"short_windows": WF_SHORT_WINDOWS, "long_windows": WF_LONG_WINDOWS},
        "warmup_context_bars": WARMUP_BARS,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_walk_forward_variant_b_matches_golden(btc_close_and_dates):
    """Phase 4.5 gate: all folds match Variant B golden within rtol 1e-8."""
    close_full, dates_full = btc_close_and_dates
    actual = _run_walk_forward_b(close_full, dates_full)

    # Every fold must have a real Sharpe (Phase 4.3: no -999 sentinels)
    for fold_result in actual["splits"]:
        fold_n = fold_result["fold"]
        val_sharpe = fold_result["val_metrics"]["sharpe"]
        assert val_sharpe != -999.0, (
            f"Fold {fold_n}: val_metrics.sharpe == -999 — warmup context was not provided. "
            f"This is the Phase 4.3 InsufficientWarmupError bug."
        )
        assert val_sharpe > -10.0, (
            f"Fold {fold_n}: val_sharpe={val_sharpe!r} looks degenerate."
        )

    # Phase 4.5: compare against Variant B golden at rtol 1e-8
    if not _GOLDEN_B.exists():
        pytest.skip(f"Variant B golden not found: {_GOLDEN_B}")

    compare_to_golden(
        actual,
        _GOLDEN_B,
        label="walk_forward_variant_b",
    )


def test_insufficient_warmup_error_raised_without_context():
    """InsufficientWarmupError is raised when context is missing and window is short."""
    short_close = np.full(50, 10_000.0)  # only 50 bars, long_w=120 → need 121 bars
    with pytest.raises(InsufficientWarmupError) as exc_info:
        check_warmup_sufficient(short_close, required_bars=120, fold=1, context=None)
    assert exc_info.value.window_bars == 50
    assert exc_info.value.required_bars == 120
    assert exc_info.value.fold == 1


def test_insufficient_warmup_error_satisfied_by_context():
    """check_warmup_sufficient passes when context brings total bars above threshold."""
    short_close = np.full(50, 10_000.0)
    context = np.full(80, 10_000.0)  # 50 + 80 = 130 > 120 + 1
    check_warmup_sufficient(short_close, required_bars=120, fold=1, context=context)  # no raise


def test_ma_cross_equity_context_mode_differs_from_no_context():
    """ma_cross_equity with context yields different (non-flat) equity vs no context."""
    rng = np.random.default_rng(42)
    close = 10_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=90))
    ctx = 10_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=120))

    eq_no_ctx = ma_cross_equity(close, 10, 40)   # flat (MA can't warm up in 90 bars)
    eq_with_ctx = ma_cross_equity(close, 10, 40, context=ctx)

    # Without context (90 < 40+1 is False but signal will be zeroed initially) —
    # with context the strategy trades from bar 0.
    # The key assertion: the two paths produce DIFFERENT equity curves.
    assert not np.allclose(eq_no_ctx, eq_with_ctx, atol=1.0), (
        "ma_cross_equity with context should yield different equity vs no context"
    )
