"""Deterministic batch factor IC / IC-IR / turnover / decay evaluator.

This module is additive research tooling.  It evaluates a *panel* of one or more
factors against forward returns and produces, per factor, cross-sectional
rank-IC statistics (mean, std, IC-IR, positive ratio, t-stat), signal turnover,
and a factor-rank decay (autocorrelation) profile.

Determinism discipline (the whole point of this module)
-------------------------------------------------------
The bit-for-bit reproducibility rules below are enforced by structure, not by
convention:

1.  Polars is used *only* for element-wise cleaning (cast to Float64, map
    non-finite values to null).  No ``group_by().agg()`` and no cross-sectional
    reduction ever runs inside Polars, because Polars parallel-partitioned
    float reductions are not guaranteed bit-identical across thread counts or
    input row order.

2.  Every final float reduction happens in NumPy, in a single canonical order.
    The reduction routine re-derives its own canonical ordering from the data
    (integer symbol/timestamp codes from ``np.unique``), so the result is
    invariant to the input row order it is handed.  This is the SOLE reduction
    path: both the Polars-fed entry point and the pure-NumPy reference call the
    exact same reducer, so they are bit-identical by construction. Internally,
    ``reduce_factor_ic`` dispatches to a vectorized, batched-over-timestamps
    implementation; a per-(factor, timestamp) Python-loop reference is kept
    (private) as the permanent parity oracle the batched path is pinned
    against -- see the "Batched (2-D) primitives" section below for how
    vectorizing preserves bit-identical accumulation order.

3.  Correlations are computed as Pearson-on-average-ranks using explicit,
    fixed-order NumPy sums (no BLAS ``corrcoef``/``cov`` calls whose accumulation
    order is opaque), so thread count cannot perturb the low bits.

Because the reducer canonically re-sorts, the three determinism properties fall
out directly: (a) bit-identity with an independent NumPy reference, (b)
invariance to ``POLARS_MAX_THREADS``, and (c) invariance to input row shuffling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:  # Polars is a hard dependency of the package but keep the import defensive.
    import polars as pl
except Exception:  # pragma: no cover - polars is always installed in this repo
    pl = None  # type: ignore[assignment]

# Module-level constants (parameters live here, never in configuration/schema.py).
DEFAULT_MIN_CROSS_SECTION = 3
DEFAULT_TOP_QUANTILE = 0.8
DEFAULT_BOTTOM_QUANTILE = 0.2
DEFAULT_MAX_DECAY_LAG = 5
_HUGE = 1_000_000.0
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class FactorICResult:
    """Deterministic per-factor evaluation summary."""

    factor: str
    n_periods: int
    n_observations: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_ratio: float
    t_stat: float
    quantile_spread_mean: float
    turnover_mean: float
    rank_autocorr: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rank_autocorr"] = list(self.rank_autocorr)
        return payload


@dataclass(frozen=True, slots=True)
class BatchFactorICResult:
    """Container for a batch of per-factor results (sorted by factor name)."""

    results: tuple[FactorICResult, ...]
    n_rows: int
    n_symbols: int
    n_timestamps: int
    min_cross_section: int
    max_decay_lag: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "batch_factor_ic",
            "n_rows": int(self.n_rows),
            "n_symbols": int(self.n_symbols),
            "n_timestamps": int(self.n_timestamps),
            "min_cross_section": int(self.min_cross_section),
            "max_decay_lag": int(self.max_decay_lag),
            "factors": {res.factor: res.to_dict() for res in self.results},
        }

    def as_mapping(self) -> dict[str, FactorICResult]:
        return {res.factor: res for res in self.results}


# ---------------------------------------------------------------------------
# Deterministic NumPy primitives (fixed-order, no BLAS reductions).
# ---------------------------------------------------------------------------


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks with tie handling, deterministically.

    Equivalent to ``scipy.stats.rankdata(values, method='average')`` but pure
    NumPy so accumulation order is fixed.
    """
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    # Vectorized tie grouping (2026-07-03 audit perf fix): the previous Python
    # while-loop dominated reduce_factor_ic (called per factor per timestamp).
    # For a tie group starting at i with count c the average 1-based rank is
    # i + (c - 1)/2 + 1 == (i + j - 1)/2 + 1 with j = i + c -- bit-identical to
    # the loop for any n < 2**52.
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(sorted_vals[1:], sorted_vals[:-1], out=new_group[1:])
    group_ids = np.cumsum(new_group) - 1
    group_starts = np.flatnonzero(new_group)
    counts = np.diff(np.append(group_starts, n))
    group_avg = group_starts + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_avg[group_ids]
    return ranks


def _pearson_fixed_order(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation via explicit fixed-order sums (no BLAS)."""
    n = x.shape[0]
    if n < 2:
        return math.nan
    mean_x = float(np.sum(x)) / n
    mean_y = float(np.sum(y)) / n
    dx = x - mean_x
    dy = y - mean_y
    cov = float(np.sum(dx * dy))
    var_x = float(np.sum(dx * dx))
    var_y = float(np.sum(dy * dy))
    if var_x <= _EPS or var_y <= _EPS:
        return math.nan
    corr = cov / math.sqrt(var_x * var_y)
    if not math.isfinite(corr):
        return math.nan
    return max(-1.0, min(1.0, corr))


def _spearman(a: np.ndarray, b: np.ndarray, min_n: int) -> float:
    """Cross-sectional Spearman rank correlation over finite pairs."""
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) < min_n:
        return math.nan
    xa = a[mask]
    xb = b[mask]
    if xa.shape[0] < 3:
        return math.nan
    ra = _average_rank(xa)
    rb = _average_rank(xb)
    return _pearson_fixed_order(ra, rb)


def _quantile_spread(
    factor_vals: np.ndarray,
    labels: np.ndarray,
    top_q: float,
    bottom_q: float,
) -> float:
    mask = np.isfinite(factor_vals) & np.isfinite(labels)
    if int(np.count_nonzero(mask)) < 5:
        return math.nan
    fv = factor_vals[mask]
    lb = labels[mask]
    ranks_pct = _average_rank(fv) / float(fv.shape[0])
    top = lb[ranks_pct >= top_q]
    bottom = lb[ranks_pct <= bottom_q]
    if top.shape[0] == 0 or bottom.shape[0] == 0:
        return math.nan
    spread = float(np.sum(top)) / top.shape[0] - float(np.sum(bottom)) / bottom.shape[0]
    return spread if math.isfinite(spread) else math.nan


def _signed_weights(factor_vals: np.ndarray) -> np.ndarray:
    """Cross-sectional signed weights from centered ranks, L1-normalized.

    Returns a vector aligned with ``factor_vals`` (non-finite entries -> 0).
    """
    out = np.zeros(factor_vals.shape[0], dtype=np.float64)
    mask = np.isfinite(factor_vals)
    m = int(np.count_nonzero(mask))
    if m < 2:
        return out
    ranks = _average_rank(factor_vals[mask])
    centered = ranks - float(np.sum(ranks)) / m
    l1 = float(np.sum(np.abs(centered)))
    if l1 <= _EPS:
        return out
    out[mask] = centered / l1
    return out


def _safe_float(value: float, default: float = 0.0) -> float:
    return value if isinstance(value, float) and math.isfinite(value) else default


# ---------------------------------------------------------------------------
# Batched (2-D) primitives for the vectorized reducer.
#
# These operate on dense ``(n_timestamps, n_symbols)`` panels (one row per
# timestamp, one column per global symbol code) and reproduce the 1-D
# primitives above bit-for-bit, row by row, despite the padding/absent
# entries every dense row necessarily carries. Two distinct techniques make
# that possible, and both were verified empirically (not just reasoned
# about) before relying on them:
#
# 1. ``_average_rank_2d`` never reduces (sums) across the row -- its final
#    formula (``group_start + (count - 1) / 2.0 + 1.0``) is evaluated
#    elementwise from integer group bookkeeping (argsort/cumsum/accumulate),
#    so padding elsewhere in the row cannot perturb a valid entry's rank.
#
# 2. Everywhere the original code reduces with ``np.sum`` over a *masked,
#    variable-length* selection (Pearson's mean/cov/var, quantile-spread's
#    top/bottom means), NumPy's pairwise summation algorithm is provably
#    length-dependent: padding a shorter row out to a common width changes
#    the low bits of ``np.sum`` almost always (confirmed by direct
#    experiment). So those sums go through ``_grouped_row_sum``, which
#    groups rows by their exact valid-count and reduces each equal-length
#    group with a single ``np.sum`` call -- bit-identical to reducing each
#    row alone, because NumPy's per-row reduction along an axis only depends
#    on that row's own length.
#
# The one place the original code sums *without* ``np.sum`` -- turnover's
# ``l1 += abs(...)`` Python loop over sorted symbol keys -- is reproduced
# with ``np.cumsum(...)[..., -1]`` instead, because a cumulative sum must
# stay a strict left-to-right scan (each partial sum has to be exact), which
# empirically reproduces the Python loop bit-for-bit and, unlike
# ``np.sum``, is insensitive to trailing zero padding.
# ---------------------------------------------------------------------------


def _average_rank_2d(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Row-wise average rank (tie-aware), bit-identical to ``_average_rank``.

    Equivalent, row by row, to calling ``_average_rank(values[row][valid[row]])``
    -- entries where ``valid`` is False (padding or non-finite) are excluded
    from ranking and their output value is unspecified (callers must not read
    it). Safe regardless of how many invalid columns a row carries: see the
    module-level note above.
    """
    r, c = values.shape
    if c == 0:
        return np.empty((r, 0), dtype=np.float64)
    vals = np.where(valid, values, np.inf)
    order = np.argsort(vals, axis=1, kind="stable")
    sorted_vals = np.take_along_axis(vals, order, axis=1)
    new_group = np.empty((r, c), dtype=bool)
    new_group[:, 0] = True
    if c > 1:
        np.not_equal(sorted_vals[:, 1:], sorted_vals[:, :-1], out=new_group[:, 1:])
    col_idx = np.broadcast_to(np.arange(c, dtype=np.int64), (r, c))
    group_start = np.maximum.accumulate(np.where(new_group, col_idx, -1), axis=1)
    is_last = np.empty((r, c), dtype=bool)
    if c > 1:
        is_last[:, :-1] = new_group[:, 1:]
    is_last[:, -1] = True
    end_marker = np.where(is_last, col_idx, c)
    group_end = np.flip(np.minimum.accumulate(np.flip(end_marker, axis=1), axis=1), axis=1)
    group_count = group_end - group_start + 1
    avg_rank_sorted = group_start + (group_count - 1) / 2.0 + 1.0
    ranks = np.empty((r, c), dtype=np.float64)
    np.put_along_axis(ranks, order, avg_rank_sorted, axis=1)
    return ranks


def _compact_order(valid: np.ndarray) -> np.ndarray:
    """Column permutation packing each row's ``valid`` entries to the front.

    Order-preserving: matches boolean fancy-indexing (``a[mask]``) exactly,
    row by row (verified empirically), so taking the first ``valid.sum(axis=1)``
    columns of a compacted array reproduces ``values[row][valid[row]]``.
    """
    return np.argsort(~valid, axis=1, kind="stable")


def _apply_compact(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, order, axis=1)


def _grouped_row_sum(compact: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Per-row ``sum(compact[row, :lengths[row]])``, bit-identical to a
    standalone ``np.sum`` on each row's own ``lengths[row]``-element array.

    NumPy's pairwise summation is length-dependent, so a single
    ``np.sum(compact, axis=1)`` over padded rows would NOT reproduce that
    (confirmed empirically). Rows are grouped by their exact length and each
    equal-length group is reduced with one ``np.sum`` call, which IS
    bit-identical to the independent per-row reduction (also confirmed
    empirically): NumPy's axis reduction treats each row independently.
    """
    r = compact.shape[0]
    out = np.zeros(r, dtype=np.float64)
    if r == 0:
        return out
    order = np.argsort(lengths, kind="stable")
    sorted_lengths = lengths[order]
    change = np.nonzero(np.diff(sorted_lengths) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [r]))
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        length = int(sorted_lengths[start])
        idx = order[start:end]
        if length == 0:
            continue
        out[idx] = np.sum(compact[idx, :length], axis=1)
    return out


def _spearman_2d(a_dense: np.ndarray, b_dense: np.ndarray, min_n: int) -> np.ndarray:
    """Row-wise cross-sectional Spearman correlation, batched over many rows.

    Bit-identical, row by row, to calling ``_spearman(a_dense[row],
    b_dense[row], min_n)``. Used both for the factor/forward-return IC and
    (with rank inputs) for the rank-decay autocorrelation, since both are
    "masked Pearson-on-average-ranks" reductions.
    """
    valid = np.isfinite(a_dense) & np.isfinite(b_dense)
    counts = valid.sum(axis=1)
    ra_dense = _average_rank_2d(a_dense, valid)
    rb_dense = _average_rank_2d(b_dense, valid)
    order = _compact_order(valid)
    compact_ra = _apply_compact(ra_dense, order)
    compact_rb = _apply_compact(rb_dense, order)
    sum_ra = _grouped_row_sum(compact_ra, counts)
    sum_rb = _grouped_row_sum(compact_rb, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ra = sum_ra / counts
        mean_rb = sum_rb / counts
    compact_dx = compact_ra - mean_ra[:, None]
    compact_dy = compact_rb - mean_rb[:, None]
    cov = _grouped_row_sum(compact_dx * compact_dy, counts)
    var_x = _grouped_row_sum(compact_dx * compact_dx, counts)
    var_y = _grouped_row_sum(compact_dy * compact_dy, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(var_x * var_y)
        corr = cov / denom
    degenerate = (var_x <= _EPS) | (var_y <= _EPS)
    corr = np.where(degenerate | ~np.isfinite(corr), np.nan, np.clip(corr, -1.0, 1.0))
    return np.where(counts >= min_n, corr, np.nan)


def _quantile_spread_2d(
    a_dense: np.ndarray,
    b_dense: np.ndarray,
    top_q: float,
    bottom_q: float,
) -> np.ndarray:
    """Row-wise quantile spread, bit-identical to ``_quantile_spread`` per row."""
    valid = np.isfinite(a_dense) & np.isfinite(b_dense)
    counts = valid.sum(axis=1)
    ranks_dense = _average_rank_2d(a_dense, valid)
    order = _compact_order(valid)
    compact_ranks = _apply_compact(ranks_dense, order)
    compact_lb = _apply_compact(b_dense, order)
    with np.errstate(invalid="ignore", divide="ignore"):
        ranks_pct = compact_ranks / counts[:, None]
    c = a_dense.shape[1]
    col_idx = np.broadcast_to(np.arange(c, dtype=np.int64), compact_ranks.shape)
    within = col_idx < counts[:, None]
    top_mask = within & (ranks_pct >= top_q)
    bottom_mask = within & (ranks_pct <= bottom_q)
    top_counts = top_mask.sum(axis=1)
    bottom_counts = bottom_mask.sum(axis=1)
    compact_top_lb = _apply_compact(compact_lb, _compact_order(top_mask))
    compact_bottom_lb = _apply_compact(compact_lb, _compact_order(bottom_mask))
    sum_top = _grouped_row_sum(compact_top_lb, top_counts)
    sum_bottom = _grouped_row_sum(compact_bottom_lb, bottom_counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = sum_top / top_counts - sum_bottom / bottom_counts
    gate = (counts >= 5) & (top_counts > 0) & (bottom_counts > 0) & np.isfinite(spread)
    return np.where(gate, spread, np.nan)


def _signed_weights_from_ranks(ranks_all: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Row-wise signed, L1-normalized centered ranks; bit-identical to
    ``_signed_weights`` applied per row (given that row's own average ranks).
    """
    counts = valid.sum(axis=1)
    order = _compact_order(valid)
    compact_ranks = _apply_compact(ranks_all, order)
    sum_ranks = _grouped_row_sum(compact_ranks, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ranks = sum_ranks / counts
    centered_dense = ranks_all - mean_ranks[:, None]
    compact_centered = _apply_compact(centered_dense, order)
    l1 = _grouped_row_sum(np.abs(compact_centered), counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = centered_dense / np.where(l1 > _EPS, l1, 1.0)[:, None]
    gate = valid & (counts[:, None] >= 2) & (l1[:, None] > _EPS)
    return np.where(gate, weights, 0.0)


def _rank_decay_profile_dense(rank_panel: np.ndarray, max_lag: int, min_n: int) -> list[float]:
    """Dense-panel equivalent of ``_rank_decay_profile``: mean cross-sectional
    Spearman of factor ranks between t and t+lag, batched over all t at once.
    """
    n_ts = rank_panel.shape[0]
    profile: list[float] = []
    for lag in range(1, max_lag + 1):
        if n_ts - lag <= 0:
            profile.append(0.0)
            continue
        corr = _spearman_2d(rank_panel[: n_ts - lag], rank_panel[lag:], min_n)
        arr = corr[np.isfinite(corr)]
        profile.append(_safe_float(float(np.sum(arr)) / arr.shape[0]) if arr.shape[0] else 0.0)
    return profile


# ---------------------------------------------------------------------------
# Sole reduction path (pure NumPy, canonically re-sorted -> shuffle-invariant).
#
# Two implementations compute the exact same reduction: ``_reduce_factor_ic_loop``
# (the original per-(factor, timestamp) Python-loop reference) and
# ``_reduce_factor_ic_batched`` (the vectorized production path, batched over
# timestamps per factor). ``reduce_factor_ic`` -- the public entry point both
# ``evaluate_factor_ic`` and ``evaluate_factor_ic_numpy`` call -- always uses
# the batched path; the loop path is retained solely so the parity test suite
# can pin batched == loop, bit-for-bit, permanently.
# ---------------------------------------------------------------------------


def _reduce_factor_ic_loop(
    symbols: np.ndarray,
    timestamps: np.ndarray,
    factor_matrix: np.ndarray,
    forward_returns: np.ndarray,
    factor_names: Sequence[str],
    *,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
    top_quantile: float = DEFAULT_TOP_QUANTILE,
    bottom_quantile: float = DEFAULT_BOTTOM_QUANTILE,
) -> BatchFactorICResult:
    """Reference reduction: per-(factor, timestamp) Python loop.

    Pure NumPy, canonical order, no BLAS reductions -- kept only as the
    permanent parity oracle for :func:`_reduce_factor_ic_batched`, which is
    the reduction ``reduce_factor_ic`` actually dispatches to. See that
    function's docstring for the parameter contract; it is identical here.
    """
    names = tuple(str(name) for name in factor_names)
    n_rows = int(symbols.shape[0])
    factor_matrix = np.ascontiguousarray(factor_matrix, dtype=np.float64)
    forward_returns = np.ascontiguousarray(forward_returns, dtype=np.float64)
    if factor_matrix.ndim != 2 or factor_matrix.shape[0] != n_rows:
        raise ValueError("factor_ic_factor_matrix_shape_mismatch")
    if len(names) != factor_matrix.shape[1]:
        raise ValueError("factor_ic_factor_names_length_mismatch")
    if forward_returns.shape[0] != n_rows:
        raise ValueError("factor_ic_forward_returns_length_mismatch")

    min_n = max(3, int(min_cross_section))
    max_lag = max(0, int(max_decay_lag))

    # Deterministic integer codes; np.unique returns sorted unique values.
    sym_unique, sym_codes = np.unique(symbols, return_inverse=True)
    ts_unique, ts_codes = np.unique(timestamps, return_inverse=True)
    sym_codes = np.asarray(sym_codes, dtype=np.int64).reshape(-1)
    ts_codes = np.asarray(ts_codes, dtype=np.int64).reshape(-1)
    n_symbols = int(sym_unique.shape[0])
    n_timestamps = int(ts_unique.shape[0])

    # Canonical order: primary timestamp, secondary symbol (timestamps grouped).
    order = np.lexsort((sym_codes, ts_codes))
    ts_sorted = ts_codes[order]
    sym_sorted = sym_codes[order]
    fmat_sorted = factor_matrix[order]
    fwd_sorted = forward_returns[order]

    # Group boundaries per timestamp (contiguous after the canonical sort).
    if n_rows == 0:
        boundaries: list[tuple[int, int]] = []
    else:
        change = np.nonzero(np.diff(ts_sorted) != 0)[0] + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [n_rows]))
        boundaries = list(zip(starts.tolist(), ends.tolist(), strict=True))

    k = len(names)
    results: list[FactorICResult] = []
    for col in range(k):
        fcol = fmat_sorted[:, col]
        ic_values: list[float] = []
        spread_values: list[float] = []
        n_obs = 0
        # Per-timestamp cross-sectional IC + quantile spread.
        prev_weights: dict[int, float] | None = None
        turnover_values: list[float] = []
        # For decay we keep per-timestamp ranked factor keyed by symbol code.
        ts_symbol_ranks: list[dict[int, float]] = []
        for start, end in boundaries:
            fv = fcol[start:end]
            lb = fwd_sorted[start:end]
            syms = sym_sorted[start:end]
            ic = _spearman(fv, lb, min_n)
            if math.isfinite(ic):
                ic_values.append(ic)
                n_obs += int(np.count_nonzero(np.isfinite(fv) & np.isfinite(lb)))
                spread = _quantile_spread(fv, lb, top_quantile, bottom_quantile)
                if math.isfinite(spread):
                    spread_values.append(spread)

            # Turnover: signed weights aligned by symbol between consecutive ts.
            weights = _signed_weights(fv)
            wmap = {
                int(sym): float(w)
                for sym, w in zip(syms.tolist(), weights.tolist(), strict=True)
                if w != 0.0
            }
            if prev_weights is not None:
                keys = sorted(set(prev_weights) | set(wmap))
                l1 = 0.0
                for key in keys:
                    l1 += abs(wmap.get(key, 0.0) - prev_weights.get(key, 0.0))
                turnover_values.append(0.5 * l1)
            prev_weights = wmap

            # Decay: retain average ranks keyed by symbol code for this ts.
            mask = np.isfinite(fv)
            if int(np.count_nonzero(mask)) >= 2:
                ranks = _average_rank(fv[mask])
                rmap = {
                    int(sym): float(r)
                    for sym, r in zip(syms[mask].tolist(), ranks.tolist(), strict=True)
                }
            else:
                rmap = {}
            ts_symbol_ranks.append(rmap)

        ic_arr = np.asarray(ic_values, dtype=np.float64)
        summary = _summarize(ic_arr)
        turnover_mean = (
            float(np.sum(np.asarray(turnover_values, dtype=np.float64))) / len(turnover_values)
            if turnover_values
            else 0.0
        )
        spread_mean = (
            float(np.sum(np.asarray(spread_values, dtype=np.float64))) / len(spread_values)
            if spread_values
            else 0.0
        )
        autocorr = _rank_decay_profile(ts_symbol_ranks, max_lag, min_n)

        results.append(
            FactorICResult(
                factor=names[col],
                n_periods=int(ic_arr.shape[0]),
                n_observations=int(n_obs),
                ic_mean=_safe_float(summary[0]),
                ic_std=_safe_float(summary[1]),
                ic_ir=_safe_float(summary[2]),
                ic_positive_ratio=_safe_float(summary[3]),
                t_stat=_safe_float(summary[4]),
                quantile_spread_mean=_safe_float(spread_mean),
                turnover_mean=_safe_float(turnover_mean),
                rank_autocorr=tuple(autocorr),
            )
        )

    results.sort(key=lambda r: r.factor)
    return BatchFactorICResult(
        results=tuple(results),
        n_rows=n_rows,
        n_symbols=n_symbols,
        n_timestamps=n_timestamps,
        min_cross_section=min_n,
        max_decay_lag=max_lag,
    )


def _reduce_factor_ic_batched(
    symbols: np.ndarray,
    timestamps: np.ndarray,
    factor_matrix: np.ndarray,
    forward_returns: np.ndarray,
    factor_names: Sequence[str],
    *,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
    top_quantile: float = DEFAULT_TOP_QUANTILE,
    bottom_quantile: float = DEFAULT_BOTTOM_QUANTILE,
) -> BatchFactorICResult:
    """The production reduction: batched over timestamps, per factor.

    Proven bit-identical to :func:`_reduce_factor_ic_loop` by
    ``tests/research/test_factor_ic_batched_parity.py`` -- see the module-level
    note above the batched primitives for why each step is safe to vectorize.
    Scatters the long panel into one dense ``(n_timestamps, n_symbols)`` array
    per factor (NaN where a (symbol, timestamp) pair has no row), so the
    per-(factor, timestamp) Python loop the reference path pays for collapses
    into a handful of whole-panel NumPy calls per factor.

    Parameters
    ----------
    symbols, timestamps:
        1-D arrays of length ``N`` (row-aligned).  Any dtype that ``np.unique``
        can order deterministically (ints, datetime64, strings) is accepted.
    factor_matrix:
        ``(N, K)`` float array of factor values.
    forward_returns:
        1-D float array of length ``N``.
    factor_names:
        Length-``K`` names for the factor columns.
    """
    names = tuple(str(name) for name in factor_names)
    n_rows = int(symbols.shape[0])
    factor_matrix = np.ascontiguousarray(factor_matrix, dtype=np.float64)
    forward_returns = np.ascontiguousarray(forward_returns, dtype=np.float64)
    if factor_matrix.ndim != 2 or factor_matrix.shape[0] != n_rows:
        raise ValueError("factor_ic_factor_matrix_shape_mismatch")
    if len(names) != factor_matrix.shape[1]:
        raise ValueError("factor_ic_factor_names_length_mismatch")
    if forward_returns.shape[0] != n_rows:
        raise ValueError("factor_ic_forward_returns_length_mismatch")

    min_n = max(3, int(min_cross_section))
    max_lag = max(0, int(max_decay_lag))

    sym_unique, sym_codes = np.unique(symbols, return_inverse=True)
    ts_unique, ts_codes = np.unique(timestamps, return_inverse=True)
    sym_codes = np.asarray(sym_codes, dtype=np.int64).reshape(-1)
    ts_codes = np.asarray(ts_codes, dtype=np.int64).reshape(-1)
    n_symbols = int(sym_unique.shape[0])
    n_timestamps = int(ts_unique.shape[0])

    fwd_dense = np.full((n_timestamps, n_symbols), np.nan, dtype=np.float64)
    if n_rows > 0:
        fwd_dense[ts_codes, sym_codes] = forward_returns

    k = len(names)
    results: list[FactorICResult] = []
    for col in range(k):
        factor_dense = np.full((n_timestamps, n_symbols), np.nan, dtype=np.float64)
        if n_rows > 0:
            factor_dense[ts_codes, sym_codes] = factor_matrix[:, col]

        # IC + quantile spread: jointly masked on (factor, forward return).
        ic_per_row = _spearman_2d(factor_dense, fwd_dense, min_n)
        n_obs_per_row = (np.isfinite(factor_dense) & np.isfinite(fwd_dense)).sum(axis=1)
        spread_per_row = _quantile_spread_2d(factor_dense, fwd_dense, top_quantile, bottom_quantile)

        ic_finite = np.isfinite(ic_per_row)
        ic_arr = ic_per_row[ic_finite]
        n_obs = int(np.sum(n_obs_per_row[ic_finite]))
        spread_finite = ic_finite & np.isfinite(spread_per_row)
        spread_arr = spread_per_row[spread_finite]

        summary = _summarize(ic_arr)
        spread_mean = (
            float(np.sum(spread_arr)) / spread_arr.shape[0] if spread_arr.shape[0] else 0.0
        )

        # Turnover + decay: masked on factor value alone (labels not involved).
        valid_factor = np.isfinite(factor_dense)
        ranks_all = _average_rank_2d(factor_dense, valid_factor)
        weights_dense = _signed_weights_from_ranks(ranks_all, valid_factor)
        if n_timestamps >= 2:
            diff = np.abs(weights_dense[1:] - weights_dense[:-1])
            turnover_per_pair = 0.5 * np.cumsum(diff, axis=1)[:, -1]
            turnover_mean = float(np.sum(turnover_per_pair)) / turnover_per_pair.shape[0]
        else:
            turnover_mean = 0.0

        rank_panel = np.where(valid_factor, ranks_all, np.nan)
        autocorr = _rank_decay_profile_dense(rank_panel, max_lag, min_n)

        results.append(
            FactorICResult(
                factor=names[col],
                n_periods=int(ic_arr.shape[0]),
                n_observations=int(n_obs),
                ic_mean=_safe_float(summary[0]),
                ic_std=_safe_float(summary[1]),
                ic_ir=_safe_float(summary[2]),
                ic_positive_ratio=_safe_float(summary[3]),
                t_stat=_safe_float(summary[4]),
                quantile_spread_mean=_safe_float(spread_mean),
                turnover_mean=_safe_float(turnover_mean),
                rank_autocorr=tuple(autocorr),
            )
        )

    results.sort(key=lambda r: r.factor)
    return BatchFactorICResult(
        results=tuple(results),
        n_rows=n_rows,
        n_symbols=n_symbols,
        n_timestamps=n_timestamps,
        min_cross_section=min_n,
        max_decay_lag=max_lag,
    )


def reduce_factor_ic(
    symbols: np.ndarray,
    timestamps: np.ndarray,
    factor_matrix: np.ndarray,
    forward_returns: np.ndarray,
    factor_names: Sequence[str],
    *,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
    top_quantile: float = DEFAULT_TOP_QUANTILE,
    bottom_quantile: float = DEFAULT_BOTTOM_QUANTILE,
) -> BatchFactorICResult:
    """The SOLE reduction entry point: pure NumPy, canonical order, no BLAS reductions.

    Parameters
    ----------
    symbols, timestamps:
        1-D arrays of length ``N`` (row-aligned).  Any dtype that ``np.unique``
        can order deterministically (ints, datetime64, strings) is accepted.
    factor_matrix:
        ``(N, K)`` float array of factor values.
    forward_returns:
        1-D float array of length ``N``.
    factor_names:
        Length-``K`` names for the factor columns.

    The routine derives its own canonical ordering from integer symbol/timestamp
    codes, so the returned statistics do not depend on the row order of the
    inputs. Dispatches to :func:`_reduce_factor_ic_batched` (the vectorized
    production path); :func:`_reduce_factor_ic_loop` is kept importable as the
    permanent parity oracle for that path, not for production use.
    """
    return _reduce_factor_ic_batched(
        symbols,
        timestamps,
        factor_matrix,
        forward_returns,
        factor_names,
        min_cross_section=min_cross_section,
        max_decay_lag=max_decay_lag,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )


def _summarize(ic_arr: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (mean, std, ir, positive_ratio, t_stat) from an IC series."""
    n = int(ic_arr.shape[0])
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    mean = float(np.sum(ic_arr)) / n
    if n >= 2:
        dev = ic_arr - mean
        var = float(np.sum(dev * dev)) / (n - 1)
        std = math.sqrt(var) if var > 0.0 else 0.0
    else:
        std = 0.0
    if std <= _EPS:
        if abs(mean) <= _EPS:
            ir = 0.0
            t_stat = 0.0
        else:
            ir = math.copysign(_HUGE, mean)
            t_stat = math.copysign(_HUGE, mean)
    else:
        ir = mean / std
        t_stat = mean / (std / math.sqrt(float(n)))
    positive_ratio = float(np.count_nonzero(ic_arr > 0.0)) / n
    return (mean, std, ir, positive_ratio, t_stat)


def _rank_decay_profile(
    ts_symbol_ranks: Sequence[Mapping[int, float]],
    max_lag: int,
    min_n: int,
) -> list[float]:
    """Mean cross-sectional Spearman of factor ranks between t and t+lag."""
    profile: list[float] = []
    n_ts = len(ts_symbol_ranks)
    for lag in range(1, max_lag + 1):
        corrs: list[float] = []
        for t in range(n_ts - lag):
            cur = ts_symbol_ranks[t]
            nxt = ts_symbol_ranks[t + lag]
            common = sorted(set(cur) & set(nxt))
            if len(common) < min_n:
                continue
            a = np.asarray([cur[s] for s in common], dtype=np.float64)
            b = np.asarray([nxt[s] for s in common], dtype=np.float64)
            # Ranks-of-ranks == Spearman on the underlying factor for the common set.
            corr = _pearson_fixed_order(_average_rank(a), _average_rank(b))
            if math.isfinite(corr):
                corrs.append(corr)
        if corrs:
            arr = np.asarray(corrs, dtype=np.float64)
            profile.append(_safe_float(float(np.sum(arr)) / arr.shape[0]))
        else:
            profile.append(0.0)
    return profile


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def _extract_arrays(
    frame: Any,
    factor_columns: Sequence[str],
    label_column: str,
    time_column: str,
    symbol_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Element-wise clean via Polars, then extract canonical NumPy arrays.

    Only element-wise transforms run in Polars (cast + non-finite -> null).  All
    reductions are deferred to :func:`reduce_factor_ic`.
    """
    names = list(factor_columns)
    if pl is not None and isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
        lazy = frame.lazy()
        required = [symbol_column, time_column, label_column, *names]
        # Element-wise only: cast numeric columns and null out non-finite values.
        exprs = []
        for col in (label_column, *names):
            casted = pl.col(col).cast(pl.Float64, strict=False)
            exprs.append(pl.when(casted.is_finite()).then(casted).otherwise(None).alias(col))
        lazy = lazy.select(required).with_columns(exprs)
        collected = lazy.collect()
        symbols = collected.get_column(symbol_column).to_numpy()
        timestamps = collected.get_column(time_column).to_numpy()
        fwd = collected.get_column(label_column).to_numpy().astype(np.float64)
        fmat = np.column_stack(
            [collected.get_column(col).to_numpy().astype(np.float64) for col in names]
        )
        return symbols, timestamps, fmat, fwd
    # Fallback: pandas / mapping-like.
    import pandas as pd

    pdf = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    symbols = pdf[symbol_column].to_numpy()
    timestamps = pdf[time_column].to_numpy()
    fwd = pd.to_numeric(pdf[label_column], errors="coerce").to_numpy(dtype=np.float64)
    fwd[~np.isfinite(fwd)] = np.nan
    cols = []
    for col in names:
        arr = pd.to_numeric(pdf[col], errors="coerce").to_numpy(dtype=np.float64)
        arr[~np.isfinite(arr)] = np.nan
        cols.append(arr)
    fmat = np.column_stack(cols) if cols else np.empty((len(pdf), 0), dtype=np.float64)
    return symbols, timestamps, fmat, fwd


def evaluate_factor_ic(
    frame: Any,
    *,
    factor_columns: Sequence[str],
    label_column: str = "forward_return",
    time_column: str = "timestamp",
    symbol_column: str = "symbol",
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
    top_quantile: float = DEFAULT_TOP_QUANTILE,
    bottom_quantile: float = DEFAULT_BOTTOM_QUANTILE,
) -> BatchFactorICResult:
    """Evaluate a batch of factors from a Polars/pandas long panel.

    The panel must contain ``symbol_column``, ``time_column``, ``label_column``
    and every entry of ``factor_columns``.  Polars handles element-wise cleaning
    only; :func:`reduce_factor_ic` performs the sole (deterministic) reduction.
    """
    names = list(factor_columns)
    if not names:
        raise ValueError("factor_ic_no_factor_columns")
    symbols, timestamps, fmat, fwd = _extract_arrays(
        frame, names, label_column, time_column, symbol_column
    )
    return reduce_factor_ic(
        symbols,
        timestamps,
        fmat,
        fwd,
        names,
        min_cross_section=min_cross_section,
        max_decay_lag=max_decay_lag,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )


def evaluate_factor_ic_numpy(
    symbols: np.ndarray,
    timestamps: np.ndarray,
    factor_matrix: np.ndarray,
    forward_returns: np.ndarray,
    factor_names: Sequence[str],
    *,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    max_decay_lag: int = DEFAULT_MAX_DECAY_LAG,
    top_quantile: float = DEFAULT_TOP_QUANTILE,
    bottom_quantile: float = DEFAULT_BOTTOM_QUANTILE,
) -> BatchFactorICResult:
    """Pure-NumPy entry point (no Polars); shares the sole reducer."""
    return reduce_factor_ic(
        np.asarray(symbols),
        np.asarray(timestamps),
        np.asarray(factor_matrix, dtype=np.float64),
        np.asarray(forward_returns, dtype=np.float64),
        factor_names,
        min_cross_section=min_cross_section,
        max_decay_lag=max_decay_lag,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )


__all__ = [
    "DEFAULT_BOTTOM_QUANTILE",
    "DEFAULT_MAX_DECAY_LAG",
    "DEFAULT_MIN_CROSS_SECTION",
    "DEFAULT_TOP_QUANTILE",
    "BatchFactorICResult",
    "FactorICResult",
    "evaluate_factor_ic",
    "evaluate_factor_ic_numpy",
    "reduce_factor_ic",
]
