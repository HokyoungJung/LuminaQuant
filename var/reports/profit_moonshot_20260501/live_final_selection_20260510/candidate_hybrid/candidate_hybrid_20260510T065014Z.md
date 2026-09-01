# Profit moonshot candidate-derived hybrid

- generated_at_utc: `2026-05-10T06:50:14.291085Z`
- oos_end_date: `2026-05-09`
- selection basis: train/validation-only candidate hybrid tuning
- locked-OOS: report-only / gate-only
- promotion note: no dynamic-weight liquidation replay is claimed, so this is comparison evidence unless a dedicated margin replay is added.

## Selected candidate hybrid

- name: `candidate_hybrid_online_rank_01_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_`
- TV score: `24.4342`
- train return/MDD: `+33.5524%` / `+7.0280%`
- validation return/MDD: `+19.1320%` / `+3.5874%`
- OOS return/MDD: `+7.3573%` / `+2.6858%`
- OOS return/MDD ratio: `2.7393`
- OOS Sharpe/Sortino/smart Sortino/Calmar: `3.5505` / `4.9816` / `4.4464` / `17.5808`

## Final allocation

- date: `2026-05-06`
- cash_weight: `+0.0000%`
- `candidate_hybrid_input_12_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+35.2474%`
- `candidate_hybrid_input_13_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+26.9254%`
- `candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+8.8923%`
- `candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+6.5096%`
- `candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+6.5096%`
- `candidate_hybrid_input_09_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+5.5722%`
- `candidate_hybrid_input_10_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+5.5722%`
- `candidate_hybrid_input_11_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_se`: `+4.7712%`

## Top tuned candidate-hybrid rows

| rank | name | TV score | train | val | OOS | OOS MDD | OOS R/MDD | Sharpe | Sortino | SmartSort | Calmar |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `candidate_hybrid_online_rank_01_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_` | 24.4342 | +33.5524% | +19.1320% | +7.3573% | +2.6858% | 2.7393 | 3.5505 | 4.9816 | 4.4464 | 17.5808 |
| 2 | `candidate_hybrid_online_rank_02_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_` | 24.4342 | +33.5523% | +19.1320% | +7.3589% | +2.6843% | 2.7414 | 3.5513 | 4.9827 | 4.4477 | 17.5949 |
| 3 | `candidate_hybrid_online_rank_03_candidate_hybrid_input_04_fresh_portfolio_train_val_monthly_` | 24.3467 | +33.4877% | +19.1320% | +7.3573% | +2.6858% | 2.7393 | 3.5505 | 4.9816 | 4.4464 | 17.5808 |
| 4 | `candidate_hybrid_online_rank_04_candidate_hybrid_input_04_fresh_portfolio_train_val_monthly_` | 24.3467 | +33.4876% | +19.1320% | +7.3589% | +2.6843% | 2.7414 | 3.5513 | 4.9827 | 4.4477 | 17.5949 |
| 5 | `candidate_hybrid_online_rank_05_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_` | 24.0187 | +33.0743% | +19.0961% | +7.4299% | +2.5947% | 2.8635 | 3.6070 | 5.1611 | 4.6254 | 18.4076 |
| 6 | `candidate_hybrid_online_rank_06_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_` | 24.0187 | +33.0743% | +19.0961% | +7.4313% | +2.5934% | 2.8655 | 3.6077 | 5.1620 | 4.6266 | 18.4209 |
| 7 | `candidate_hybrid_online_rank_07_candidate_hybrid_input_04_fresh_portfolio_train_val_monthly_` | 23.9317 | +33.0099% | +19.0961% | +7.4299% | +2.5947% | 2.8635 | 3.6070 | 5.1611 | 4.6254 | 18.4076 |
| 8 | `candidate_hybrid_online_rank_08_candidate_hybrid_input_04_fresh_portfolio_train_val_monthly_` | 23.9317 | +33.0098% | +19.0961% | +7.4313% | +2.5934% | 2.8655 | 3.6077 | 5.1620 | 4.6266 | 18.4209 |
| 9 | `candidate_hybrid_online_rank_09_candidate_hybrid_input_02_fresh_portfolio_train_val_monthly_` | 23.8556 | +33.4230% | +19.1320% | +7.3573% | +2.6858% | 2.7393 | 3.5505 | 4.9816 | 4.4464 | 17.5808 |
| 10 | `candidate_hybrid_online_rank_10_candidate_hybrid_input_02_fresh_portfolio_train_val_monthly_` | 23.8556 | +33.4229% | +19.1320% | +7.3589% | +2.6843% | 2.7414 | 3.5513 | 4.9827 | 4.4477 | 17.5949 |
| 11 | `candidate_hybrid_online_rank_11_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_` | 23.6549 | +32.5803% | +19.0612% | +7.5017% | +2.5156% | 2.9820 | 3.6611 | 5.2962 | 4.7633 | 19.2001 |
| 12 | `candidate_hybrid_online_rank_12_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_` | 23.6549 | +32.5803% | +19.0612% | +7.5030% | +2.5145% | 2.9839 | 3.6617 | 5.2971 | 4.7643 | 19.2128 |
