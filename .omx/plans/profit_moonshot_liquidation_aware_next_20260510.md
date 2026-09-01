# Profit moonshot liquidation-aware next plan — 2026-05-10

## Stored conclusion
The integer-leverage audit should not be read as "leverage is bad." The actual conclusion is narrower:

- If leverage is only making a weak row look better and train/validation quality or current-base gates fail, do not promote it.
- If a levered portfolio truly improves return/risk and **cannot liquidate** under realistic futures margin assumptions, it can be a valid improvement even if raw/unlevered train return is modest.
- Under that interpretation, the best next candidate is the exact current-base sleeve tuple at **integer `5x` leverage**, pending liquidation-aware validation.

## Current conditional best candidate
Candidate source: forced current-base integer row from the top40 integer-leverage audit.

- Source artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_top40_20260509/fresh_portfolio_tuning_latest.json`
- Audit summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_audit_20260509.json`
- Sleeve tuple:
  1. `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600`
  2. `fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600`
  3. `fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all`
  4. `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600`
- Mode: `train_val_monthly_return_budget`
- Integer leverage: `5x`

## Metrics versus current base
| Candidate | OOS return | OOS MDD | OOS return/MDD | OOS monthly | Sharpe | Sortino | Smart Sortino | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current base `2.3427x` | `+6.8582%` | `0.8198%` | `8.3659` | `+3.0883%` | `5.6537` | `7.3961` | `7.1536` | `53.7350` |
| Forced base `5x` | `+14.6371%` | `1.6919%` | `8.6514` | `+6.4641%` | `5.7215` | `7.4828` | `6.9764` | `66.2284` |

Interpretation: `5x` improves OOS return, return/MDD, Sharpe, Sortino, and Calmar. Smart Sortino is slightly lower but remains high. It was not promoted only because the previous raw-train/train-Sortino audit was conservative and did not model liquidation directly.

## Critical caveat
Current portfolio tuning combines per-sleeve equity curves with a linear leverage transform. It does **not** yet prove exchange-level safety:

- no per-position liquidation price,
- no intrabar high/low liquidation check,
- no explicit initial/maintenance margin path,
- no cross-vs-isolated margin model,
- no portfolio-level margin buffer / margin ratio time series,
- no funding/fee/slippage stress beyond what the base sleeve replay already applies.

Therefore `5x` is a **conditional best candidate**, not a deployable promotion, until liquidation-aware replay confirms zero liquidation events.

## Required next work
1. Implement a liquidation-aware validation path for the current-base sleeve tuple at integer leverages, starting with `5x` and optionally searching safer/stronger integer levels.
2. Lock behavior with tests before heavy replay:
   - liquidation event triggers when intrabar adverse price breaches liquidation threshold,
   - liquidation count is reported per split,
   - margin buffer / margin ratio minima are recorded,
   - locked-OOS is not used for selection,
   - `5x` is not promoted if any split has liquidation or non-positive margin buffer.
3. Use conservative futures assumptions unless exact exchange details are available:
   - Binance USDT perpetual style accounting,
   - explicit cross/isolated assumption recorded in artifact,
   - maintenance margin rate table or conservative scalar fallback,
   - fees/slippage/funding included or stress-adjusted,
   - intrabar high/low worst-case checks.
4. Evaluate train/validation/OOS for current-base `5x` and any fallback integer leverage.
5. Promotion condition for a levered candidate:
   - train/validation selection only,
   - locked-OOS report-only/gate-only,
   - liquidation count `0` in train/validation/OOS,
   - positive minimum margin buffer in all splits,
   - OOS MDD `<=25%`,
   - OOS return and return/MDD beat current base,
   - risk-adjusted metrics remain high,
   - memory remains `<8 GiB`.
6. If `5x` fails liquidation safety, search the highest/safest integer leverage that passes the liquidation gate and compare against the current base.
7. Run targeted tests, full pytest, ruff, compileall, git diff --check, commit with Lore protocol, push to `private/main`, and verify GitHub Actions `ci` and `private-ci` green.

## Recommended artifacts to produce next session
- `.omx/plans/profit_moonshot_liquidation_aware_5x_YYYYMMDD.md`
- `docs/session_handoff_YYYYMMDD_profit_moonshot_liquidation_aware_5x.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_5x_YYYYMMDD/*`
- updated `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json` only after validator/CI evidence is final.
