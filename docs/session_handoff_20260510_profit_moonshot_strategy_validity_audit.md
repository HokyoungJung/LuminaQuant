# Profit moonshot strategy-validity audit handoff — 2026-05-10

## Final conclusion

**Do not promote any current profit-moonshot candidate to live.** After adding the source-aware strategy-validity gate, every row that looked deployable under the previous integer/leverage/liquidation/performance gates is rejected because its active sleeves are calendar-primary month/asset rules (mostly TRX/ETH calendar take-profit/rotation). That is a theoretical/live-deployability defect, not just a metric issue.

The updated final-selection artifact is fail-closed: `recommendation=no_live_promotion`, `winner=null`, and locked-OOS remains report-only/gate-only.

## Scope and data

- Workdir: `/home/hoky/Quants-agent/LuminaQuant`
- Baseline preserved: performance baseline `02f4520cf906f48089b8852c2651a0f1e4bd0c1c`; prior pushed green handoff `77f10d54174628c24f1a6bbba34a74505a2a40b5` remains historical reference.
- Universe: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, TRX/USDT.
- Fresh data cutoff: latest complete OOS date `2026-05-09` minimum OHLCV max UTC `2026-05-10T05:38:38Z`; no required symbols missing.
- Split windows from current candidate/hybrid artifacts: train `2025-01-01..2025-12-31`, validation `2026-01-01..2026-02-28`, locked-OOS `2026-03-01..2026-05-09`.
- Timeframe/source: profit-moonshot fresh candidates use 1s OHLCV derived from Binance raw aggregate trades; candidate hybrid reports compounded 1h-to-1d portfolio rows where applicable. Locked-OOS is never used for selection.

## Audit summary

| Item | Value |
|---|---:|
| Final-selection rows audited | 33 |
| Rows with required strategy-validity metadata missing | 0 |
| Strategy-invalid final rows | 31 |
| Deployable valid rows after all gates | 0 |
| Deployable invalid rows that escaped | 0 |
| Source-pool rows scanned | 8821 |
| Source-pool calendar-primary invalid rows | 4392 |
| Source-pool strategy-valid rows | 4429 |
| Original source-pool success candidates | 300 |
| Strategy-valid source-pool success candidates | 0 |

Interpretation: the prior `success_candidate` pool was entirely calendar-primary after this audit (`strategy_valid_success_candidate_count=0`). The ranking is therefore materially changed: previous apparent winners are invalid for live deployment rather than merely lower-ranked.

## Rows that previously looked deployable but are now rejected

| Kind | Name | TV score | Lev | OOS return | OOS MDD | OOS R/MDD | Liq | Min buffer | Strategy reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr_liquidation_aware_5x` | 19.882952 | 5.000000 | 14.6634% | 1.9646% | 7.463950 | 0 | 8514.066649 | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fr_liquidation_aware_5x` | 19.875591 | 5.000000 | 14.5038% | 1.9435% | 7.462877 | 0 | 8514.066649 | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `current_base_tuple_liquidation_aware_5x` | 19.244936 | 5.000000 | 14.0578% | 1.9584% | 7.178039 | 1 | 8415.811075 | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_01_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | 1.713430 | 1.000000 | 14.2041% | 1.9447% | 7.303859 | 1 | 9174.874641 | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |

## Source-pool dynamic candidates

There are strategy-valid/dynamic rows in the merged source CSV, but none survives the existing `success_candidate` gate, and none has the full integer-leverage + liquidation-aware + performance promotion package. Top dynamic rows before success/liquidation promotion gates are report-only and must not be promoted from this handoff.

| Name | Family | Success | Train return | Val return | OOS return | Train/Val screen |
|---|---|---:|---:|---:|---:|---:|
| `fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus` | `residual_pair_reversion_spread` | `False` | 0.4366% | 0.0046% | -0.1101% | 0.002083 |
| `fresh_resid_flow_rev_fl12_lb48_z175_imb10` | `residual_reversion_flow_confirmed` | `False` | 0.0000% | 0.0000% | -0.0072% | 0.000000 |
| `fresh_resid_flow_rev_fl12_lb48_z15_imb10` | `residual_reversion_flow_confirmed` | `False` | 0.0000% | 0.0000% | -0.0014% | 0.000000 |
| `fresh_resid_flow_rev_fl12_lb48_z125_imb10` | `residual_reversion_flow_confirmed` | `False` | 0.0000% | 0.0000% | -0.0014% | 0.000000 |
| `fresh_resid_flow_rev_fl12_lb24_z175_imb10` | `residual_reversion_flow_confirmed` | `False` | 0.0000% | 0.0000% | 0.0000% | 0.000000 |
| `fresh_resid_flow_rev_fl12_lb24_z15_imb10` | `residual_reversion_flow_confirmed` | `False` | 0.0000% | 0.0000% | 0.0000% | 0.000000 |
| `fresh_resid_flow_rev_fl12_lb24_z125_imb10` | `residual_reversion_flow_confirmed` | `False` | 0.0000% | 0.0000% | -0.0019% | 0.000000 |
| `fresh_funding_oi_fade_lb6_z175_f50_oi0` | `funding_oi_carry_fade` | `False` | 0.0000% | 0.0000% | -0.2507% | 0.000000 |
| `fresh_funding_oi_fade_lb6_z175_f100_oi0` | `funding_oi_carry_fade` | `False` | 0.0000% | 0.0000% | -0.0921% | 0.000000 |
| `fresh_funding_oi_fade_lb6_z15_f50_oi0` | `funding_oi_carry_fade` | `False` | 0.0000% | 0.0000% | -0.2507% | 0.000000 |

## Artifacts written

- Final decision JSON: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/profit_moonshot_live_final_selection_latest.json`
- Final decision MD: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/profit_moonshot_live_final_selection_latest.md`
- Strategy-validity audit JSON: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/strategy_validity_audit/strategy_validity_audit_latest.json`
- Strategy-validity audit MD: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/strategy_validity_audit/strategy_validity_audit_latest.md`
- Runner evidence: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/final_decision_strategy_validity_*`, `strategy_validity_audit_*`

## Verification status

Completed:
- Targeted strategy tests: `uv run pytest -q tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_strategy_validity_audit.py tests/test_profit_moonshot_candidate_hybrid.py tests/test_profit_moonshot_liquidation_aware_validation.py` → 27 passed, max RSS 168.98 MiB.
- Full pytest: `uv run pytest` → 1253 passed in 255.94s, max RSS 2658.05 MiB.
- Ruff: `uv run ruff check .` → All checks passed, max RSS 34.63 MiB.
- Compileall: `python3 -m compileall -q scripts tests` → passed, max RSS 20.97 MiB.
- Git whitespace: `git diff --check` → passed, max RSS 7.06 MiB.
- Final-selection rebuild under strategy gate → status `no_live_promotion`, max RSS 27.56 MiB.
- Strategy-validity audit artifact → status `pass_with_rejections`, max RSS 28.03 MiB.

Still required before final delivery: squash/amend to Lore commit, push `private/main`, verify GitHub Actions `ci/private-ci` green.

## Recommendation

Use **no live profit-moonshot candidate** from the current candidate/hybrid set. The correct safe action is to block promotion and open a follow-up research lane for genuinely dynamic/state-based alpha with liquidation-aware replay from the start. Do not revive the invalid calendar-primary sleeves as live rules without a separate, pre-registered seasonal thesis and out-of-sample validation protocol.
