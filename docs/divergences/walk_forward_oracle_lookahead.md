# Divergence: Walk-forward oracle 1-bar lookahead correction

**Date:** 2026-06-10
**Branch:** `refactor/overhaul`
**Classification:** Oracle CORRECTION (lookahead removal) — not a regression
**Artifacts:**
- `baseline/golden/walk_forward_results.json` (Variant A)
- `baseline/golden/walk_forward_results_warmup.json` (Variant B)

---

## What changed

The shared MA-cross reference implementation (`ma_cross_equity` in
`src/lumina_quant/optimization/walkers.py`, mirrored by `_ma_cross_equity` in
`scripts/capture_golden_baseline.py`) carried a **1-bar signal lookahead** plus a
**wrong return denominator**:

```python
# BEFORE (lookahead):
signal = np.where(short_ma > long_ma, 1.0, -1.0)  # MA at bar i INCLUDES close[i]
signal[:long_w] = 0.0
daily_ret = np.diff(close, prepend=close[0]) / close  # denominator = close[i] (wrong)
equity = 10_000.0 * np.cumprod(1.0 + signal * daily_ret)  # signal[i] earns daily_ret[i]
```

Two defects:

1. **Lookahead** — `signal[i]` is computed from a causal MA that includes
   `close[i]`, then applied to the return `daily_ret[i]` that *ends* at
   `close[i]`. The position earning the `(i-1, i]` return is therefore chosen
   using the very close that defines that return.
2. **Wrong denominator** — the simple return used `close[i]` as the denominator;
   the correct simple return is `(close[i] - close[i-1]) / close[i-1]`.

```python
# AFTER (causal, no lookahead):
ret[i] = (close[i] - close[i - 1]) / close[i - 1]  # prior-close denominator
equity = 10_000.0 * np.cumprod(1.0 + signal[i - 1] * ret[i])  # prior-bar signal
```

Implemented via a single shared helper `_lagged_strategy_returns(prices, signal)`
that lags the signal one bar (`signal[i-1]`) and uses the prior-close
denominator. The helper is duplicated **bit-identically** in `walkers.py` and the
capture script so the rtol-1e-8 golden integration test
(`tests/integration/test_walk_forward_golden.py`) still holds.

## Why this is allowed

This is an **oracle correction**: the pre-correction goldens were lookahead-
inflated artifacts (the engine never used this path — only golden capture and the
golden test import it, so live backtests were unaffected). Per the determinism
contract, an oracle correction is documented here with old→new values and the
goldens are regenerated; the rtol-1e-8 integration test then passes against the
**new** goldens. The tolerance was never loosened.

## Impact (committed fixture `BTCUSDT_seed42_1000d.parquet`)

Single-combo sanity (full 1000-bar series), sw=10 / lw=40:

| | Sharpe |
|---|---|
| before (lookahead) | **+0.3398** |
| after (lagged)     | **−0.1175** |

A sign flip — the lookahead manufactured positive performance.

### Variant A (`walk_forward_results.json`) — best params + metrics

| Fold | best_params (old → new) | val sharpe (old → new) | test sharpe (old → new) |
|------|--------------------------|------------------------|-------------------------|
| 1 | {20,120} → {20,120} | −999.0 → −999.0 (sentinel¹) | −999.0 → −999.0 (sentinel¹) |
| 2 | **{30,80} → {20,80}** | 0.7141 → 1.2838 | −0.7039 → −1.0166 |
| 3 | {10,120} → {10,120} | −999.0 → −999.0 (sentinel¹) | −999.0 → −999.0 (sentinel¹) |

### Variant B (`walk_forward_results_warmup.json`) — warmup-context oracle

| Fold | best_params (old → new) | val sharpe (old → new) | test sharpe (old → new) |
|------|--------------------------|------------------------|-------------------------|
| 1 | {20,120} → {20,120} | −0.2350 → 0.1413 | −0.5597 → −1.7303 |
| 2 | **{30,80} → {20,80}** | −1.2458 → −0.7335 | −0.9687 → 0.4652 |
| 3 | {10,120} → {10,120} | 0.6152 → 0.6749 | 2.7989 → 2.4464 |

Fold 2's selected parameters changed (`{30,80}` → `{20,80}`): removing the
lookahead reorders the train-grid Sharpe ranking, so the corrected oracle selects
a different best combo. This is expected — the previous selection was driven by
lookahead-inflated train metrics.

¹ The −999.0 sentinels in Variant A folds 1 & 3 are unchanged (the lw=120 window
is shorter than the indicator warmup → flat equity → −999). See
`docs/divergences/walk_forward_no_sentinel.md`; Variant B is the Phase 4 oracle.

## Regeneration

```bash
uv run python scripts/capture_golden_baseline.py   # deterministic; seed=42
```

(Only the two `walk_forward_*` goldens move; the backtest/native goldens are
unaffected — the event-driven engine never used this numpy path.)

## References

- `src/lumina_quant/optimization/walkers.py`: `_lagged_strategy_returns`, `ma_cross_equity`
- `scripts/capture_golden_baseline.py`: `_lagged_strategy_returns`, `_ma_cross_equity`
- `tests/integration/test_walk_forward_golden.py`: rtol-1e-8 golden gate
- `docs/divergences/walk_forward_no_sentinel.md`: companion −999 sentinel divergence
