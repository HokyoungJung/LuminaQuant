# Golden Re-baseline: Phase 0 (3.11) → Phase 0.5 Stack Lift (3.14)

**Generated:** 2026-06-10
**Branch:** `refactor/overhaul`
**Command:** `uv run python scripts/capture_golden_baseline.py`
**Python:** 3.11.15 (old) → 3.14.4 (new)
**Seed:** 42 (unchanged)

---

## Result: ZERO NUMERICAL DIVERGENCES

All golden artifacts produced on Python 3.14.4 are **bit-exact identical** to
the Phase 0 (Python 3.11.15) baseline. Same backend + same seed + same inputs
= deterministic outputs across Python versions, confirming the numerical
contract (rtol `1e-8`; spec acceptance criterion #2).

---

## Artifact-by-Artifact Comparison

### Backtest Goldens

| Artifact | Phase 0 (3.11) | Phase 0.5 (3.14) | Divergence |
|----------|----------------|------------------|-----------|
| MA Cross Sharpe | −0.9177 | −0.9177 | **none** |
| MA Cross MaxDD | 7.65% | 7.65% | **none** |
| MA Cross Total Return | −3.61% | −3.61% | **none** |
| BuyHold Sharpe | −1.6746 | −1.6746 | **none** |
| BuyHold MaxDD | 5.63% | 5.63% | **none** |
| BuyHold Total Return | −3.30% | −3.30% | **none** |
| Equity curve rows | 1001 | 1001 | **none** |

### Native Backend Goldens (all 6 crates)

| Backend | Key metric | Phase 0 | Phase 0.5 | Divergence |
|---------|------------|---------|-----------|-----------|
| rust_metrics | sharpe | 0.2993542540374392 | 0.2993542540374392 | **none** |
| rust_metrics | cagr | 0.04491008636884075 | 0.04491008636884075 | **none** |
| rust_metrics | max_dd | 0.5085791489826454 | 0.5085791489826454 | **none** |
| c_metrics | sharpe | 0.2993542540374392 | 0.2993542540374392 | **none** |
| rust_alpha_fold | total_return | −1.335412446167978 | −1.335412446167978 | **none** |
| rust_alpha_fold | n | 1000 | 1000 | **none** |
| rust_hybrid_optuna | portfolio_final | 9824.906231479908 | 9824.906231479908 | **none** |
| rust_live_signals | debounced n | 1000 | 1000 | **none** |
| rust_live_signals | trailing n | 1000 | 1000 | **none** |
| rust_rawfirst | n (1s bars) | 3598 | 3598 | **none** |
| rust_rawfirst | first close | 61687.9 | 61687.9 | **none** |
| rust_rawfirst | last close | 61694.9 | 61694.9 | **none** |

### aggTrades Fixture

SHA-256 verified: `51426ca3fe8493488d637b1bd66f4ba2253c863c9abb32195a8ac8ecef59e0d7`
(unchanged — frozen fixture, not regenerated).

### Configs

Both `config_frozen.yaml` and `research_frozen.yaml` SHA-256s unchanged.

### Walk-Forward (variant A — authoritative perf denominator)

| Fold | Phase 0 best_params | Phase 0.5 best_params | Divergence |
|------|---------------------|-----------------------|-----------|
| 1 | {short:20, long:120} | {short:20, long:120} | **none** |
| 2 | {short:30, long:80} | {short:30, long:80} | **none** |
| 3 | {short:10, long:120} | {short:10, long:120} | **none** |

### Walk-Forward (variant B — warmup context, new in Phase 0.5)

`baseline/golden/walk_forward_results_warmup.json` added in this re-capture.
Worker-2 updated `capture_golden_baseline.py` to produce both variants A and B
in the same run. Variant B feeds 120-bar warmup context so all MA windows yield
real metrics (no `-999` sentinels). This is a **new artifact** (not present in
Phase 0 baseline), not a divergence — it's an additive oracle expansion.

| Fold | B val_sharpe | B test_sharpe |
|------|-------------|--------------|
| 1 | −0.2350 | −0.5597 |
| 2 | −1.2458 | −0.9687 |
| 3 | 0.6152 | 2.7989 |

---

## New-Stack Goldens Are Now Canonical

Per plan §3 Phase 0.5.3: "The re-baselined new-stack goldens become the
canonical reference; all Phase 1–6 tolerance gates compare against new-stack
goldens, never the old-stack ones."

The Phase 0.5 goldens are bit-exact identical to Phase 0 goldens (Python
version had no numerical effect), so **all Phase 1–6 gates compare against
the existing `baseline/golden/` files** — no numerical threshold relaxation
needed and no `docs/divergences/<artifact>.md` approval is required beyond
this document.

---

## Why No Divergence?

The native kernels are compiled Rust/C shared libraries (`.so` files). They
are Python-ABI-independent — the same `.so` is loaded via `ctypes.CDLL`
regardless of whether CPython 3.11 or 3.14 is running. Python-level
backtest/strategy arithmetic uses the same numpy operations with the same
seed, so determinism holds across interpreter versions.
