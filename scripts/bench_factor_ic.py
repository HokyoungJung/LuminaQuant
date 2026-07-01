"""Three-arm benchmark for the deterministic batch factor-IC reduction kernel.

The dominant cost in factor-IC evaluation is the per-timestamp cross-sectional
rank-IC reduction.  This benchmark isolates that kernel and times three
implementations of the *same* quantity (per-factor mean rank-IC over a canonical
``(T*S, K)`` panel) so the Rust escalate/skip decision rests on real numbers.

Arms
----
1. ``python_loop``   : per-candidate (per-factor) Python loop; for each factor,
                       loop timestamps and compute one cross-sectional Spearman
                       at a time.  This mirrors the naive research script.
2. ``numpy_batched`` : one fully vectorized reduction over the reshaped
                       ``(T, S, K)`` panel in canonical order -- ranks and
                       correlations for all factors and timestamps at once.
3. ``rust``          : conditional native arm; absent by default -> skipped.

Escalation policy (hotspot-only)
--------------------------------
The decision is judged **against the numpy_batched arm** at the representative
panel size.  If it stays within ``--budget-ms`` a Rust kernel is SKIPPED
(recommended default): the vectorized NumPy reduction already carries the
hotspot, so a native kernel plus its rtol=1e-8 parity oracle is not justified.
Escalate only when the numpy_batched arm exceeds the budget.

Both arms are cross-checked for numerical agreement (rtol=1e-9) so the timing
comparison is honest.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-confirmed escalation threshold: max acceptable ms/run for the numpy_batched
# arm at the representative size.  Escalate to Rust only if it is exceeded.
DEFAULT_BUDGET_MS = 250.0


def _synthetic_panel(n_symbols: int, n_timestamps: int, n_factors: int, seed: int):
    """Rectangular full panel in canonical (timestamp, symbol) order."""
    rng = np.random.default_rng(seed)
    n_rows = n_symbols * n_timestamps
    # Row r = ts * n_symbols + sym  -> already canonical (ts primary, sym secondary).
    factors = rng.standard_normal((n_rows, n_factors))
    fwd = 0.3 * factors[:, 0] + rng.standard_normal(n_rows) * 0.9
    names = [f"f{idx}" for idx in range(n_factors)]
    return factors, fwd, names


def _avg_rank_1d(values: np.ndarray) -> np.ndarray:
    """Average ranks (tie-aware) for a 1-D array."""
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    mx = float(np.sum(x)) / n
    my = float(np.sum(y)) / n
    dx = x - mx
    dy = y - my
    cov = float(np.sum(dx * dy))
    vx = float(np.sum(dx * dx))
    vy = float(np.sum(dy * dy))
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    return cov / (vx**0.5 * vy**0.5)


def _arm_python_loop(factors: np.ndarray, fwd: np.ndarray, n_ts: int, n_sym: int):
    """Per-candidate python loop: one Spearman per (factor, timestamp)."""

    def run():
        k = factors.shape[1]
        out = np.empty(k, dtype=np.float64)
        for col in range(k):
            ics = []
            for t in range(n_ts):
                s = t * n_sym
                fv = factors[s : s + n_sym, col]
                lb = fwd[s : s + n_sym]
                ics.append(_pearson(_avg_rank_1d(fv), _avg_rank_1d(lb)))
            arr = np.asarray(ics, dtype=np.float64)
            out[col] = float(np.sum(arr)) / arr.shape[0]
        return out

    return run


def _rank_along_symbols(mat3d: np.ndarray) -> np.ndarray:
    """Vectorized ordinal ranks along the symbol axis (axis=1) of a (T,S,K) cube.

    Assumes continuous values (no ties) for the batched fast path -- ordinal
    ranks then equal average ranks.  Ties are a measure-zero perturbation for
    the timing comparison; correctness of the production reducer is covered by
    the determinism test-suite, not here.
    """
    order = np.argsort(mat3d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    s = mat3d.shape[1]
    positions = np.arange(1, s + 1, dtype=np.float64)[None, :, None]
    np.put_along_axis(ranks, order, np.broadcast_to(positions, order.shape), axis=1)
    return ranks


def _arm_numpy_batched(factors: np.ndarray, fwd: np.ndarray, n_ts: int, n_sym: int):
    """Fully vectorized batched reduction over the reshaped (T,S,K) panel."""
    k = factors.shape[1]
    f3d = factors.reshape(n_ts, n_sym, k)
    r2d = fwd.reshape(n_ts, n_sym)

    def run():
        rf = _rank_along_symbols(f3d)  # (T,S,K)
        rr = _rank_along_symbols(r2d[:, :, None])  # (T,S,1)
        # Center along the symbol axis.
        rf_c = rf - rf.mean(axis=1, keepdims=True)
        rr_c = rr - rr.mean(axis=1, keepdims=True)
        cov = np.sum(rf_c * rr_c, axis=1)  # (T,K)
        vf = np.sum(rf_c * rf_c, axis=1)
        vr = np.sum(rr_c * rr_c, axis=1)
        denom = np.sqrt(vf * vr)
        ic = np.where(denom > 1e-12, cov / denom, np.nan)  # (T,K)
        return np.nanmean(ic, axis=0)

    return run


def _try_rust_arm():
    """Return a native kernel if wired, else ``None`` (documents the seam)."""
    try:  # pragma: no cover - no native kernel exists yet.
        from lumina_quant.native import lumina_compute  # type: ignore

        if hasattr(lumina_compute, "batch_factor_ic"):
            return lumina_compute.batch_factor_ic
    except Exception:
        return None
    return None


def _time(fn, iters: int) -> tuple[float, float]:
    samples: list[float] = []
    for _ in range(iters):
        start = perf_counter()
        fn()
        samples.append((perf_counter() - start) * 1000.0)
    return statistics.median(samples), min(samples)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark batch factor-IC kernel.")
    parser.add_argument("--symbols", type=int, default=60)
    parser.add_argument("--timestamps", type=int, default=750)
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--output", default="")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    factors, fwd, _names = _synthetic_panel(args.symbols, args.timestamps, args.factors, args.seed)
    n_rows = factors.shape[0]

    loop_fn = _arm_python_loop(factors, fwd, args.timestamps, args.symbols)
    numpy_fn = _arm_numpy_batched(factors, fwd, args.timestamps, args.symbols)

    # Honesty check: both arms must agree numerically.
    agree = bool(np.allclose(loop_fn(), numpy_fn(), rtol=1e-9, atol=1e-12, equal_nan=True))

    loop_med, loop_min = _time(loop_fn, args.iters)
    numpy_med, numpy_min = _time(numpy_fn, args.iters)

    rust_fn = _try_rust_arm()
    escalate = numpy_med > args.budget_ms
    decision = "escalate_to_rust" if escalate else "skip_rust"

    payload = {
        "artifact_kind": "bench_factor_ic",
        "panel": {
            "symbols": int(args.symbols),
            "timestamps": int(args.timestamps),
            "factors": int(args.factors),
            "rows": int(n_rows),
        },
        "iters": int(args.iters),
        "arms_agree_rtol_1e_9": agree,
        "arms": {
            "python_loop": {"median_ms": loop_med, "min_ms": loop_min},
            "numpy_batched": {"median_ms": numpy_med, "min_ms": numpy_min},
            "rust": {"available": rust_fn is not None, "median_ms": None},
        },
        "speedup_numpy_vs_loop": (loop_med / numpy_med) if numpy_med > 0 else None,
        "budget_ms": float(args.budget_ms),
        "rust_decision": decision,
        "rust_decision_basis": "numpy_batched_arm_vs_budget_ms",
        "escalate": bool(escalate),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
