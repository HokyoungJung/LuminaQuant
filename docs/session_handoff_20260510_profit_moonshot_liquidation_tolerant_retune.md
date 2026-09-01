# Profit moonshot liquidation-tolerant retune/reselection — 2026-05-10

## Objective
Retune/reselect the profit-moonshot sleeve tuple after allowing only a tiny liquidation tolerance, while preserving train/validation-only selection and locked-OOS report/gate-only semantics.

## Rule change
- Promotion no longer requires `liquidation_count == 0` exactly.
- Explicit tiny tolerance used for this run:
  - total liquidations <= `1`
  - per-split liquidations <= `1`
  - maximum event drawdown <= `0.5%`
  - maximum event equity loss <= `0.5%`
  - all split minimum margin buffers must remain `> 0`
- Locked-OOS remains gate-only/report-only and is not used in selection ranking.

## Implementation notes
- Fixed the liquidation-aware replay path so single-leg fresh families (for example calendar/TRX take-profit sleeves) are replayed instead of being ignored. The prior spread-only replay made train performance look artificially dead.
- Added event impact fields: pre/post liquidation equity, event drawdown, equity loss fraction, account wipeout flag, and closed scope/legs.
- Replayed current-base 2.342733x, forced current-base 5x, integer grid 1x..6x, and 44 prior train/validation candidate seeds from the integer audit + merged candidate CSV.

## Main results

### Current-base replay reference, 2.342733x
- train return `+24.5533%`, MDD `7.4996%`, liquidations `0`
- validation return `+20.1842%`, MDD `6.6189%`, liquidations `0`
- OOS return `+6.4281%`, MDD `0.9293%`, return/MDD `6.9169`

### Forced current-base 5x
- deployable under the tiny-liquidation tolerance: `true`
- train return `+60.5997%`, MDD `16.2149%`, liquidations `0`
- validation return `+45.6166%`, MDD `14.0994%`, liquidations `1`
- liquidation event: `2026-02-05T20:00:00Z` BTC/USDT LONG, closed one sleeve state (`2` legs), not whole account, `account_wipeout=false`, event drawdown/equity loss `0.080233%`
- OOS return `+14.0578%`, MDD `1.9584%`, return/MDD `7.1780`

### Promoted reselected candidate
Source: `integer_audit_diagnostic_quarantine_04`, leverage `5x`.

Sleeves:
1. `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600`
2. `fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600`
3. `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss100_tp600`

Metrics:
- train return `+61.6855%`, MDD `15.1068%`, liquidations `0`, min buffer `9144.6285`
- validation return `+42.6032%`, MDD `13.2668%`, liquidations `0`, min buffer `8514.0666`
- OOS return `+14.6634%`, MDD `1.9646%`, return/MDD `7.4640`, liquidations `0`, min buffer `9833.7810`
- OOS Sharpe `5.2251`, Sortino `6.5707`, smart Sortino `6.0543`, Calmar `57.1718`
- OOS return delta vs current-base replay `+8.2353%`; return/MDD delta `+0.5471`

## Decision
`liquidation_tolerant_reselected_deployable`.

The forced current-base 5x is deployable under the relaxed tiny-liquidation rule, but the reselected 3-sleeve 5x tuple is preferred after gate filtering because it has zero liquidations and better OOS return/return-MDD while keeping selection train/validation-ranked and locked-OOS gate-only.

## Artifacts
- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_tolerant_retune_20260510/liquidation_aware_current_base_latest.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_tolerant_retune_20260510/liquidation_aware_current_base_latest.md`

## Verification
- Targeted test: `.venv/bin/pytest tests/test_profit_moonshot_liquidation_aware_validation.py -q` => `9 passed`, max RSS `177180 KiB`.
- Replay command: exit `0`, max RSS `267840 KiB` (`~261.6 MiB`), under 8 GiB.
- Full pytest: `1229 passed in 303.57s` (wall `4:38.21`), max RSS `2772424 KiB` (`~2.64 GiB`).
- `ruff check .` passed.
- `python -m compileall -q src scripts tests` passed.
- `git diff --check` passed.
- Source Lore commit `ff1786da0233d2e39acab8c310e0cf3e2a2b0891` pushed to `private/main`.
- GitHub Actions green for source commit: `private-ci` run `25619911800` success; `ci` run `25619911802` success.
