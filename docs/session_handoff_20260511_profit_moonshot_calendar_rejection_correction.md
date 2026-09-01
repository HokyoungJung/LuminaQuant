# Session handoff — profit moonshot calendar rejection correction — 2026-05-11

## Corrected conclusion

The current-base profit-moonshot tuple must **not** be treated as deployable. Its strong 4x/5x/6x liquidation-aware performance is dominated by fixed month/asset calendar-primary sleeves. That violates the strategy-validity rule from 2026-05-10: `calendar_primary_alpha_unsupported`, `calendar_fixed_month_alpha`, and `fixed_asset_calendar_target` are hard live-promotion rejects.

Correct live recommendation: **no live promotion** until a non-calendar/state-signal strategy passes train/validation selection, liquidation-aware replay, locked-OOS report/gate checks, and strategy-validity gates.

## What changed

- Added a strategy-validity gate directly to `scripts/research/run_profit_moonshot_liquidation_aware_validation.py`.
- Calendar-primary sleeves are now fail-closed before `deployable_success` can be true, regardless of liquidation tolerance, OOS return, MDD, or margin buffer.
- Retune seeds from integer-audit and candidate CSV sources now skip calendar-primary sleeves by default.
- `highest_zero_liquidation_integer` now requires strategy-validity as well as zero-liquidation safety, preventing invalid calendar rows from being presented as clean alternatives.
- Added regression tests proving the calendar current-base tuple is strategy-invalid and cannot become deployable even when metrics/liquidation gates pass.

## New strategy-valid liquidation replay

Artifact root:
`var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_strategy_valid_20260511/`

Latest JSON:
`liquidation_aware_current_base_latest.json`

Run command:

```bash
/usr/bin/time -v uv run --extra dev python scripts/research/run_profit_moonshot_liquidation_aware_validation.py \
  --retune-audit-limit 50 \
  --retune-csv-limit 80 \
  --retune-report-limit 40 \
  --output-dir var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_strategy_valid_20260511
```

Memory evidence: max RSS `263368 KiB`, under 8 GiB.

## New artifact decision

- `outcome=no_live_promotion_strategy_validity_failed`
- `deployable_improvement=false`
- `current_base_strategy_valid=false`
- `forced_5x_deployable=false`
- `selected_integer_deployable=false`
- `reselected_deployable=false`
- summary: calendar-primary current-base tuple rejected by strategy-validity gates; no live promotion.

Current-base tuple strategy rejection reasons:

- `calendar_primary_alpha_unsupported`
- `calendar_fixed_month_alpha`
- `fixed_asset_calendar_target`

## Non-calendar retune evidence

The retune pass evaluated non-calendar/state-signal seeds only:

- candidate seed count: `6`
- evaluated integer results: `30`
- deployable candidates: `0`
- best train/validation retune row: `fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus_liquidation_aware_1x`
  - train return/MDD/Sharpe: `+0.0423%` / `0.2462%` / `0.1840`
  - validation return/MDD/Sharpe: `+0.0646%` / `0.0885%` / `1.3936`
  - OOS return/MDD/Sharpe: `+0.0241%` / `0.0810%` / `0.6158`
  - liquidations train/validation/OOS: `0/0/0`
  - strategy-valid: `true`
  - deployable: `false` due weak train/validation and OOS performance.

## Important invalid-but-informative numbers

These are research-only, not live candidates:

- current-base 5x: train `+60.5997%`, validation `+45.6166%`, OOS `+14.0578%`, validation liquidation `1`, strategy-valid `false`.
- current-base 6x: train `+74.2590%`, validation `+55.6648%`, OOS `+17.0656%`, validation liquidations `2`, strategy-valid `false`.

They are invalid because the tuple contains fixed month/asset calendar-primary sleeves.

## Next work

Do not resurrect calendar-primary current-base rows. Future profit-moonshot work should build from non-calendar causal/state variables only: residual spreads, cross-sectional momentum/reversal, funding/OI/flow states, volatility compression/expansion, or other live-observable features. Locked-OOS remains report-only/gate-only.
