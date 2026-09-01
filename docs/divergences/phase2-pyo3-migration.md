# Golden Divergence: Phase 2 pyo3 Migration

**Generated:** 2026-06-10
**Branch:** `refactor/overhaul`
**Approved by:** team-lead (divergence review after full capture re-run)
**Command:** `uv run python scripts/capture_golden_baseline.py`
**Backend change:** ctypes cdylib loading → pyo3 cdylib (`lumina_quant._compute`)

---

## Result: APPROVED MIGRATION DIVERGENCE

Two categories of change observed after re-capturing all golden artifacts
against the pyo3 backend. Both are approved.

---

## Category 1 — Schema Change: `status` fields removed

**Affected files:** `baseline/golden/native_backends.json` (2 fields),
`baseline/golden/PROVENANCE.json` (2 fields)

**Root cause:** The old ctypes callers (`_call_rust_metrics`, `_call_c_metrics`)
returned the C ABI integer status code (`fn.restype = ctypes.c_int`) in the
captured dict:

```json
"rust_metrics": { "status": 0, "sharpe": ..., "cagr": ..., "max_dd": ... }
```

The pyo3 binding raises a Python exception on failure instead of returning a
status integer. The replacement `_call_pyo3_metrics` therefore omits the
`"status"` key entirely, returning only the three numeric metrics.

**Verdict:** IMPROVEMENT — exception-based error reporting is strictly more
informative than a silent integer status code. The numeric payload (sharpe,
cagr, max_dd) is **bit-exact** across the ctypes and pyo3 paths because the
underlying Rust algorithm is identical.

### Before (ctypes path)

```json
"rust_metrics": { "status": 0, "sharpe": 0.2993542540374392, ... }
"c_metrics":    { "status": 0, "sharpe": 0.2993542540374392, ... }
```

### After (pyo3 path)

```json
"rust_metrics": { "sharpe": 0.2993542540374392, ... }
"c_metrics":    { "sharpe": 0.2993542540374392, ... }
```

---

## Category 2 — ULP-level numeric drift in walk-forward train_sharpe

**Affected files:** `baseline/golden/walk_forward_results.json`,
`baseline/golden/walk_forward_results_warmup.json`

**Root cause:** The pyo3 `evaluate_metrics` kernel reorders floating-point
accumulation slightly differently from the Python/numba path used to compute
`train_sharpe` during walk-forward grid search. IEEE 754 floating-point is not
associative; different summation orders produce last-bit differences.

**Max observed relative drift:**

| Location | Old value | New value | Rel diff |
|----------|-----------|-----------|----------|
| `walk_forward_results.json` `splits[0].train_grid[0].train_sharpe` | `−0.14215801318300003` | `−0.14215801318300036` | **2.342934456735753e-15** |

All other drifting values are at the same or smaller magnitude.

**Tolerance gate:** rtol `1e-8` — all observed drifts are **7 orders of
magnitude inside the tolerance**. Gate: **PASS**.

### Summary table (all affected train_sharpe values)

| Location | Max rel diff | Gate |
|----------|-------------|------|
| `walk_forward_results.json` — all train_sharpe | 2.34e-15 | PASS |
| `walk_forward_results_warmup.json` — all train_sharpe | ≤ 2.34e-15 | PASS |

No changes to `val_sharpe`, `test_sharpe`, `best_params`, equity curves,
backtest stats, or rawfirst / alpha-fold / live-signal / hybrid-optuna outputs.

---

## New pyo3 Goldens Are Now Canonical

The four regenerated golden files committed in the "Phase 2 gate" commit are the
new canonical baseline for all Phase 3–9 tolerance gates:

- `baseline/golden/native_backends.json`
- `baseline/golden/PROVENANCE.json`
- `baseline/golden/walk_forward_results.json`
- `baseline/golden/walk_forward_results_warmup.json`

All future comparisons use these pyo3-captured values. The ctypes-era goldens
(with `"status": 0` fields and the slightly different train_sharpe values) are
superseded.

---

## Approval

Team-lead re-ran the full golden capture against the pyo3 backend and diffed
all artifacts vs the committed Phase 0b/0.5 goldens. Both categories of
divergence were reviewed and approved as documented migration divergences.
The numeric divergence (Category 2) passes the project tolerance at rtol `1e-8`
with 7 orders of margin.
