# Profit moonshot candidate-derived hybrid

- generated_at_utc: `2026-05-10T08:40:07.511746Z`
- oos_end_date: `2026-05-09`
- selection basis: train/validation-only candidate hybrid tuning
- locked-OOS: report-only / gate-only
- live leverage policy: integer leverage source rows only
- liquidation replay: dynamic candidate weights + source leverage + conservative Binance-style margin model
- discarded non-integer/missing leverage sources: `0`

## Selected candidate hybrid

- name: `candidate_hybrid_online_rank_01_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_`
- TV score: `1.7134`
- allocator train/validation/OOS return: `+32.3689%` / `+27.4536%` / `+14.3709%`
- replay train return/MDD: `+20.2319%` / `+15.8562%`
- replay validation return/MDD: `+22.3058%` / `+13.4881%`
- replay OOS return/MDD: `+14.2041%` / `+1.9447%`
- replay OOS return/MDD ratio: `7.3039`
- replay OOS Sharpe/Sortino/Calmar: `5.1698` / `6.7407` / `55.3698`
- liquidation counts train/validation/OOS: `0 / 1 / 0`
- minimum margin buffer train/validation/OOS: `9174.8746` / `9764.4797` / `9834.0897`
- minimum margin ratio train/validation/OOS: `36.8360` / `47.6152` / `88.8763`

## Final allocation

- date: `2026-05-06`
- cash_weight: `+0.0000%`
- `candidate_hybrid_input_17_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+23.4351%`
- `candidate_hybrid_input_03_current_base_tuple_liquidation_aware_5x`: `+20.2497%`
- `candidate_hybrid_input_18_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+16.8755%`
- `candidate_hybrid_input_19_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+14.3272%`
- `candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+4.4743%`
- `candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+3.2754%`
- `candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+3.2754%`
- `candidate_hybrid_input_20_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+3.2754%`
- `candidate_hybrid_input_09_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+2.8037%`
- `candidate_hybrid_input_10_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+2.8037%`
- `candidate_hybrid_input_21_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+2.8037%`
- `candidate_hybrid_input_11_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+2.4007%`

## Top tuned candidate-hybrid rows

| rank | name | TV score | train | val | OOS | OOS MDD | OOS R/MDD | Sharpe | Sortino | SmartSort | Calmar |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `candidate_hybrid_online_rank_01_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | 1.7134 | +32.3689% | +27.4536% | +14.3709% | +0.9362% | 15.3499 | 6.4228 | 16.6617 | 16.0377 | 115.1662 |
| 2 | `candidate_hybrid_online_rank_02_candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_` | 30.2688 | +32.3689% | +27.4536% | +14.3709% | +0.9362% | 15.3499 | 6.4228 | 16.6617 | 16.0377 | 115.1662 |
| 3 | `candidate_hybrid_online_rank_03_candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_` | 30.2688 | +32.3689% | +27.4536% | +14.3709% | +0.9362% | 15.3499 | 6.4228 | 16.6617 | 16.0377 | 115.1662 |
| 4 | `candidate_hybrid_online_rank_04_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | 30.2550 | +32.2968% | +27.4218% | +14.4356% | +0.9365% | 15.4137 | 6.4143 | 16.7329 | 16.1060 | 115.8115 |
| 5 | `candidate_hybrid_online_rank_05_candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_` | 30.2550 | +32.2968% | +27.4218% | +14.4356% | +0.9365% | 15.4137 | 6.4143 | 16.7329 | 16.1060 | 115.8115 |
| 6 | `candidate_hybrid_online_rank_06_candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_` | 30.2550 | +32.2968% | +27.4218% | +14.4356% | +0.9365% | 15.4137 | 6.4143 | 16.7329 | 16.1060 | 115.8115 |
| 7 | `candidate_hybrid_online_rank_07_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | 30.2522 | +32.1327% | +27.3852% | +14.4690% | +0.9369% | 15.4432 | 6.3917 | 16.7207 | 16.0941 | 116.1193 |
| 8 | `candidate_hybrid_online_rank_08_candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_` | 30.2522 | +32.1327% | +27.3852% | +14.4690% | +0.9369% | 15.4432 | 6.3917 | 16.7207 | 16.0941 | 116.1193 |
| 9 | `candidate_hybrid_online_rank_09_candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_` | 30.2522 | +32.1327% | +27.3852% | +14.4690% | +0.9369% | 15.4432 | 6.3917 | 16.7207 | 16.0941 | 116.1193 |
| 10 | `candidate_hybrid_online_rank_10_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | 30.2492 | +32.0573% | +27.4536% | +14.3716% | +0.9362% | 15.3507 | 6.4227 | 16.6625 | 16.0385 | 115.1741 |
| 11 | `candidate_hybrid_online_rank_11_candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_` | 30.2492 | +32.0573% | +27.4536% | +14.3716% | +0.9362% | 15.3507 | 6.4227 | 16.6625 | 16.0385 | 115.1741 |
| 12 | `candidate_hybrid_online_rank_12_candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_` | 30.2492 | +32.0573% | +27.4536% | +14.3716% | +0.9362% | 15.3507 | 6.4227 | 16.6625 | 16.0385 | 115.1741 |
