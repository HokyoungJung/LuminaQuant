"""Determinism proofs for the batch factor-IC evaluator.

Covers the three properties that matter for a determinism-critical reducer:

(a) bit-identity against an independent pure-NumPy reference,
(b) invariance across ``POLARS_MAX_THREADS`` settings, and
(c) invariance to input row shuffling (would break if a reduction leaked into
    Polars parallel partitions).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from lumina_quant.research.factor_ic import (
    evaluate_factor_ic,
    evaluate_factor_ic_numpy,
    reduce_factor_ic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_panel(
    *,
    n_symbols: int = 12,
    n_timestamps: int = 40,
    n_factors: int = 3,
    seed: int = 20260701,
) -> dict[str, np.ndarray]:
    """Deterministic long-format panel with a few embedded non-finite values."""
    rng = np.random.default_rng(seed)
    symbols = np.array([f"SYM{idx:02d}" for idx in range(n_symbols)])
    timestamps = np.arange(n_timestamps, dtype=np.int64) * 60_000
    sym_col = np.repeat(symbols, n_timestamps)
    ts_col = np.tile(timestamps, n_symbols)
    n_rows = n_symbols * n_timestamps
    factors = rng.standard_normal((n_rows, n_factors))
    # A signal-bearing forward return: correlated with factor 0 plus noise.
    fwd = 0.35 * factors[:, 0] + 0.15 * factors[:, 1] + rng.standard_normal(n_rows) * 0.9
    # Inject non-finite values to exercise cleaning paths.
    factors[3, 0] = np.inf
    factors[7, 1] = np.nan
    fwd[11] = -np.inf
    names = [f"f{idx}" for idx in range(n_factors)]
    return {
        "symbol": sym_col,
        "timestamp": ts_col,
        "factors": factors,
        "forward_return": fwd,
        "factor_names": names,
    }


def _to_polars(panel: dict[str, np.ndarray]) -> pl.DataFrame:
    data = {
        "symbol": panel["symbol"],
        "timestamp": panel["timestamp"],
        "forward_return": panel["forward_return"],
    }
    for idx, name in enumerate(panel["factor_names"]):
        data[name] = panel["factors"][:, idx]
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Independent pure-NumPy reference (transcribed separately from the module).
# ---------------------------------------------------------------------------


def _ref_avg_rank(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        # Mean of 1-based positions i+1..j.
        avg = float((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _ref_pearson(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    if n < 2:
        return np.nan
    mx = float(np.sum(x)) / n
    my = float(np.sum(y)) / n
    dx = x - mx
    dy = y - my
    cov = float(np.sum(dx * dy))
    vx = float(np.sum(dx * dx))
    vy = float(np.sum(dy * dy))
    if vx <= 1e-12 or vy <= 1e-12:
        return np.nan
    corr = cov / np.sqrt(vx * vy)
    if not np.isfinite(corr):
        return np.nan
    return float(max(-1.0, min(1.0, corr)))


def _ref_spearman(a: np.ndarray, b: np.ndarray, min_n: int) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) < min_n:
        return np.nan
    xa, xb = a[mask], b[mask]
    if xa.shape[0] < 3:
        return np.nan
    return _ref_pearson(_ref_avg_rank(xa), _ref_avg_rank(xb))


def _reference_factor_ic(panel: dict[str, np.ndarray], min_n: int = 3, max_lag: int = 5):
    symbols = panel["symbol"]
    timestamps = panel["timestamp"]
    fmat = panel["factors"].astype(np.float64).copy()
    fwd = panel["forward_return"].astype(np.float64).copy()
    fmat[~np.isfinite(fmat)] = np.nan
    fwd[~np.isfinite(fwd)] = np.nan
    names = panel["factor_names"]

    _sym_u, sym_c = np.unique(symbols, return_inverse=True)
    _ts_u, ts_c = np.unique(timestamps, return_inverse=True)
    sym_c = sym_c.reshape(-1).astype(np.int64)
    ts_c = ts_c.reshape(-1).astype(np.int64)
    order = np.lexsort((sym_c, ts_c))
    ts_s = ts_c[order]
    sym_s = sym_c[order]
    fmat_s = fmat[order]
    fwd_s = fwd[order]
    uniq_ts = np.unique(ts_s)

    out: dict[str, dict[str, float]] = {}
    for col, name in enumerate(names):
        ics: list[float] = []
        turnovers: list[float] = []
        rank_maps: list[dict[int, float]] = []
        prev_w: dict[int, float] | None = None
        for t in uniq_ts.tolist():
            sel = ts_s == t
            fv = fmat_s[sel, col]
            lb = fwd_s[sel]
            syms = sym_s[sel]
            ic = _ref_spearman(fv, lb, min_n)
            if np.isfinite(ic):
                ics.append(ic)
            # weights
            wvec = np.zeros(fv.shape[0])
            m = np.isfinite(fv)
            mm = int(np.count_nonzero(m))
            if mm >= 2:
                r = _ref_avg_rank(fv[m])
                centered = r - float(np.sum(r)) / mm
                l1 = float(np.sum(np.abs(centered)))
                if l1 > 1e-12:
                    wvec[m] = centered / l1
            wmap = {int(s): float(w) for s, w in zip(syms.tolist(), wvec.tolist()) if w != 0.0}
            if prev_w is not None:
                keys = sorted(set(prev_w) | set(wmap))
                l1 = 0.0
                for key in keys:
                    l1 += abs(wmap.get(key, 0.0) - prev_w.get(key, 0.0))
                turnovers.append(0.5 * l1)
            prev_w = wmap
            if mm >= 2:
                r = _ref_avg_rank(fv[m])
                rank_maps.append({int(s): float(rr) for s, rr in zip(syms[m].tolist(), r.tolist())})
            else:
                rank_maps.append({})

        ic_arr = np.asarray(ics, dtype=np.float64)
        n = ic_arr.shape[0]
        mean = float(np.sum(ic_arr)) / n if n else 0.0
        if n >= 2:
            dev = ic_arr - mean
            var = float(np.sum(dev * dev)) / (n - 1)
            std = np.sqrt(var) if var > 0.0 else 0.0
        else:
            std = 0.0
        turn = float(np.sum(np.asarray(turnovers))) / len(turnovers) if turnovers else 0.0
        # decay
        autoc: list[float] = []
        for lag in range(1, max_lag + 1):
            cs: list[float] = []
            for i in range(len(rank_maps) - lag):
                cur, nxt = rank_maps[i], rank_maps[i + lag]
                common = sorted(set(cur) & set(nxt))
                if len(common) < min_n:
                    continue
                a = np.asarray([cur[s] for s in common])
                b = np.asarray([nxt[s] for s in common])
                c = _ref_pearson(_ref_avg_rank(a), _ref_avg_rank(b))
                if np.isfinite(c):
                    cs.append(c)
            autoc.append(float(np.sum(np.asarray(cs))) / len(cs) if cs else 0.0)
        out[name] = {
            "ic_mean": mean,
            "ic_std": std,
            "turnover_mean": turn,
            "autocorr": autoc,
        }
    return out


# ---------------------------------------------------------------------------
# (a) bit-identity vs independent reference.
# ---------------------------------------------------------------------------


def test_bit_identity_against_independent_numpy_reference():
    panel = _synthetic_panel()
    frame = _to_polars(panel)
    result = evaluate_factor_ic(
        frame,
        factor_columns=panel["factor_names"],
        max_decay_lag=5,
    ).as_mapping()
    reference = _reference_factor_ic(panel, min_n=3, max_lag=5)

    assert set(result) == set(reference)
    for name, ref in reference.items():
        got = result[name]
        # Exact bit-for-bit equality (no tolerance).
        assert got.ic_mean == ref["ic_mean"], name
        assert got.ic_std == ref["ic_std"], name
        assert got.turnover_mean == ref["turnover_mean"], name
        assert list(got.rank_autocorr) == ref["autocorr"], name


def test_numpy_entry_matches_polars_entry_bitwise():
    panel = _synthetic_panel()
    frame = _to_polars(panel)
    via_polars = evaluate_factor_ic(frame, factor_columns=panel["factor_names"]).to_dict()
    via_numpy = evaluate_factor_ic_numpy(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    ).to_dict()
    assert via_polars == via_numpy


# ---------------------------------------------------------------------------
# (c) shuffle-invariance of the input row order.
# ---------------------------------------------------------------------------


def test_row_shuffle_invariance():
    panel = _synthetic_panel()
    frame = _to_polars(panel)
    base = evaluate_factor_ic(frame, factor_columns=panel["factor_names"]).to_dict()

    rng = np.random.default_rng(999)
    perm = rng.permutation(frame.height)
    shuffled = frame[perm.tolist()]
    got = evaluate_factor_ic(shuffled, factor_columns=panel["factor_names"]).to_dict()
    assert got == base


def test_numpy_reducer_shuffle_invariance():
    panel = _synthetic_panel()
    base = evaluate_factor_ic_numpy(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    ).to_dict()
    rng = np.random.default_rng(7)
    perm = rng.permutation(panel["symbol"].shape[0])
    got = reduce_factor_ic(
        panel["symbol"][perm],
        panel["timestamp"][perm],
        panel["factors"][perm],
        panel["forward_return"][perm],
        panel["factor_names"],
    ).to_dict()
    assert got == base


# ---------------------------------------------------------------------------
# (b) thread-count invariance (Polars max threads).
# ---------------------------------------------------------------------------

_THREAD_SNIPPET = """
import json, sys
import numpy as np
import polars as pl
from lumina_quant.research.factor_ic import evaluate_factor_ic

rng = np.random.default_rng(20260701)
n_symbols, n_ts, n_factors = 12, 40, 3
symbols = np.array([f"SYM{i:02d}" for i in range(n_symbols)])
ts = np.arange(n_ts, dtype=np.int64) * 60000
sym_col = np.repeat(symbols, n_ts)
ts_col = np.tile(ts, n_symbols)
n = n_symbols * n_ts
factors = rng.standard_normal((n, n_factors))
fwd = 0.35 * factors[:, 0] + 0.15 * factors[:, 1] + rng.standard_normal(n) * 0.9
factors[3, 0] = np.inf
factors[7, 1] = np.nan
fwd[11] = -np.inf
data = {"symbol": sym_col, "timestamp": ts_col, "forward_return": fwd}
names = [f"f{i}" for i in range(n_factors)]
for i, nm in enumerate(names):
    data[nm] = factors[:, i]
frame = pl.DataFrame(data)
res = evaluate_factor_ic(frame, factor_columns=names).to_dict()
sys.stdout.write(json.dumps(res, sort_keys=True))
"""


def _run_with_threads(threads: int) -> str:
    env = dict(os.environ)
    env["POLARS_MAX_THREADS"] = str(threads)
    proc = subprocess.run(
        [sys.executable, "-c", _THREAD_SNIPPET],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise AssertionError(f"threads={threads} failed: {proc.stderr}")
    return proc.stdout


def test_polars_thread_count_invariance():
    out_1 = _run_with_threads(1)
    out_2 = _run_with_threads(2)
    out_4 = _run_with_threads(4)
    assert out_1 == out_2
    assert out_1 == out_4


# ---------------------------------------------------------------------------
# Sanity: signal-bearing factor 0 has the highest |IC| of the three.
# ---------------------------------------------------------------------------


def test_signal_factor_has_strongest_ic():
    panel = _synthetic_panel()
    res = evaluate_factor_ic_numpy(
        panel["symbol"],
        panel["timestamp"],
        panel["factors"],
        panel["forward_return"],
        panel["factor_names"],
    ).as_mapping()
    assert abs(res["f0"].ic_mean) > abs(res["f2"].ic_mean)
    # f0 IC should be clearly positive by construction.
    assert res["f0"].ic_mean > 0.05
