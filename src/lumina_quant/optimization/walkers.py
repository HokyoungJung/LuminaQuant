"""Walk-forward split helpers and warmup-guard error (Phase 4).

Phase 4 loud-failure contract
------------------------------
``evaluate_metrics_backend`` silently returned ``-999.0`` (Sharpe) when the
evaluation window was too short for the indicator to warm up (flat equity curve,
zero-variance returns).  Phase 4 replaces that with ``InsufficientWarmupError``,
raised *before* evaluation when the caller has not provided warmup context.

Walk-forward evaluators MUST always supply warmup context (Variant B style).
Under normal operation ``InsufficientWarmupError`` should never be reached.

Divergence from Variant A folds 1 & 3 (the ``-999`` sentinels) is a documented
improvement — see ``docs/divergences/walk_forward_no_sentinel.md``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from lumina_quant.optimization.constants import MONTH_MAX_DAYS_COMMON_YEAR


def add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime while preserving a valid day."""
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    is_leap_year = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    max_day = 29 if month == 2 and is_leap_year else MONTH_MAX_DAYS_COMMON_YEAR[month - 1]
    day = min(dt.day, max_day)
    return datetime(year, month, day)


def build_walk_forward_splits(
    base_start: datetime,
    folds: int,
    train_months: int = 12,
    val_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
) -> list[dict]:
    """Build rolling walk-forward splits."""
    splits = []
    cursor = base_start
    for i in range(folds):
        train_start = cursor
        train_end = add_months(train_start, train_months)
        val_start = train_end
        val_end = add_months(val_start, val_months)
        test_start = val_end
        test_end = add_months(test_start, test_months)
        splits.append(
            {
                "fold": i + 1,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = add_months(cursor, step_months)
    return splits


# ── Phase 4 loud-failure guard ─────────────────────────────────────────────────


class InsufficientWarmupError(ValueError):
    """Raised when an eval window is too short to warm up the requested indicator.

    Phase 4 replacement for the silent ``-999.0`` Sharpe sentinel.  Callers must
    either supply warmup context (Variant B) or skip the fold with a structured
    error result — ``-999.0`` must never be propagated into golden output.

    Parameters
    ----------
    window_bars:
        Number of bars in the evaluation window supplied.
    required_bars:
        Minimum bars needed by the slowest indicator for full warmup.
    fold:
        Walk-forward fold index (1-based), for diagnostics.
    """

    def __init__(self, *, window_bars: int, required_bars: int, fold: int = 0) -> None:
        super().__init__(
            f"Walk-forward fold {fold}: window has {window_bars} bars but indicator "
            f"requires {required_bars} warmup bars — cannot produce a valid metric. "
            f"Supply warmup context (Variant B) or exclude this fold from selection."
        )
        self.window_bars = int(window_bars)
        self.required_bars = int(required_bars)
        self.fold = int(fold)


def check_warmup_sufficient(
    close: Any,
    required_bars: int,
    *,
    fold: int = 0,
    context: Any = None,
) -> None:
    """Raise ``InsufficientWarmupError`` when warmup cannot be satisfied.

    Call this *before* running an evaluation to prevent silent ``-999`` output.

    Parameters
    ----------
    close:
        Evaluation-window close prices (array-like).
    required_bars:
        Minimum bars the slowest indicator needs for warmup (e.g. long_window).
    fold:
        Walk-forward fold index, for diagnostics.
    context:
        Optional warmup-context array prepended before ``close``.  When supplied,
        its length counts toward satisfying ``required_bars``.
    """
    close_arr = np.asarray(close, dtype=np.float64)
    ctx_len = len(np.asarray(context, dtype=np.float64)) if context is not None else 0
    effective_bars = len(close_arr) + ctx_len
    if effective_bars < required_bars + 1:
        raise InsufficientWarmupError(
            window_bars=effective_bars,
            required_bars=required_bars,
            fold=fold,
        )


def ma_cross_equity(
    close: Any,
    short_w: int,
    long_w: int,
    *,
    context: Any = None,
) -> np.ndarray:
    """Vectorised MA-cross equity curve — shared reference implementation.

    Used by ``scripts/capture_golden_baseline.py`` *and* the Phase 4.3
    walk-forward golden integration test to guarantee identical computation.

    If *context* is provided it is prepended so both MAs are fully warm at
    bar 0 of *close*.  The returned equity covers *close* only, re-normalised
    to start at 10 000.

    Parameters
    ----------
    close:
        Evaluation-window close prices.
    short_w:
        Short MA window length.
    long_w:
        Long MA window length.  ``check_warmup_sufficient`` should be called
        before this function to guard against a flat (``-999``) result.
    context:
        Optional warmup-context prices prepended before *close*.
    """
    close_arr = np.asarray(close, dtype=np.float64)

    if context is not None and len(context) > 0:
        ctx = np.asarray(context, dtype=np.float64)
        full = np.concatenate([ctx, close_arr])
        n_full = len(full)
        n_ctx = len(ctx)
        short_ma = np.convolve(full, np.ones(short_w) / short_w, mode="full")[:n_full]
        long_ma = np.convolve(full, np.ones(long_w) / long_w, mode="full")[:n_full]
        signal_full = np.where(short_ma > long_ma, 1.0, -1.0)
        signal_full[:long_w] = 0.0
        daily_ret_full = np.diff(full, prepend=full[0]) / np.where(full == 0.0, 1.0, full)
        equity_full = 10_000.0 * np.cumprod(1.0 + signal_full * daily_ret_full)
        window_eq = equity_full[n_ctx:]
        if len(window_eq) == 0:
            return np.full(len(close_arr), 10_000.0)
        start_val = window_eq[0]
        if start_val != 0.0:
            window_eq = window_eq * (10_000.0 / start_val)
        return window_eq

    n = len(close_arr)
    if n < long_w + 1:
        return np.full(n, 10_000.0)
    short_ma = np.convolve(close_arr, np.ones(short_w) / short_w, mode="full")[:n]
    long_ma = np.convolve(close_arr, np.ones(long_w) / long_w, mode="full")[:n]
    signal = np.where(short_ma > long_ma, 1.0, -1.0)
    signal[:long_w] = 0.0
    daily_ret = np.diff(close_arr, prepend=close_arr[0]) / np.where(
        close_arr == 0.0, 1.0, close_arr
    )
    equity = 10_000.0 * np.cumprod(1.0 + signal * daily_ret)
    return equity


__all__ = [
    "InsufficientWarmupError",
    "add_months",
    "build_walk_forward_splits",
    "check_warmup_sufficient",
    "ma_cross_equity",
]
