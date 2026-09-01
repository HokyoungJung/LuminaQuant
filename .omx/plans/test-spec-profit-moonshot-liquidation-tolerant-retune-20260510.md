# Test spec — profit moonshot liquidation-tolerant retune 2026-05-10

## Behavior locks added/updated
1. Intrabar adverse high/low liquidation events still trigger as before.
2. Split summaries record liquidation count, min margin buffer, min margin ratio, max liquidation event drawdown, and max event equity-loss fraction.
3. Tiny liquidation tolerance permits promotion only when:
   - total liquidations <= configured allowance,
   - per-split liquidations <= configured allowance,
   - event drawdown/equity-loss fraction <= configured caps,
   - every split minimum margin buffer > 0.
4. Excess liquidation count or margin-buffer failure blocks promotion.
5. Train/validation leverage/candidate selection remains train/validation-only and does not use locked-OOS; locked-OOS is report-only/gate-only.
6. Mission validator still rejects unsafe liquidation evidence, but can recognize explicit tolerance metadata when buffers and event-impact caps are safe.

## Required verification
- `pytest tests/test_profit_moonshot_liquidation_aware_validation.py -q`
- liquidation-tolerant retune replay with max RSS evidence under 8 GiB
- full `pytest`
- `ruff check .`
- `python -m compileall -q src scripts tests`
- `git diff --check`
- push Lore commit to `private/main`
- confirm GitHub Actions `ci` and `private-ci` green
