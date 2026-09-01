# Profit moonshot live final selection handoff — 2026-05-10

## Final recommendation

Use **`fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr_liquidation_aware_5x`** as the live candidate, subject to normal live sizing/ops controls. It is the only selected top candidate that simultaneously has refreshed cutoff parity, zero liquidation events across train/validation/OOS, positive margin buffer on every split, OOS MDD below 25%, and improved OOS return plus return/MDD versus current-base 2.3427x.

## Decision evidence

- Latest complete OOS end date: `2026-05-09`; artifact freshness gate: `True`.
- Selection firewall: train/validation ranking only; locked-OOS is report-only/gate-only; `uses_locked_oos_for_selection=false`.
- Memory gate: `True`; largest observed RSS is data refresh `5069.789 MiB`, under 8 GiB.
- Final status: `promote_candidate` / recommendation `promote`.

## Winner: liquidation-aware candidate 5x

- Train: return 61.6855%, MDD 15.1068%, return/MDD 4.083293, Sharpe 1.673420, Sortino 1.282831, smart Sortino n/a, Calmar 4.083880, liq 0, min buffer 9144.628466, min ratio 41.121304
- Validation: return 42.6032%, MDD 13.2668%, return/MDD 3.211274, Sharpe 3.886332, Sortino 4.440037, smart Sortino n/a, Calmar 60.292162, liq 0, min buffer 8514.066649, min ratio 41.788522
- OOS: return 14.6634%, MDD 1.9646%, return/MDD 7.463950, Sharpe 5.225099, Sortino 6.570666, smart Sortino 6.054326, Calmar 57.171779, liq 0, min buffer 9833.780971, min ratio 93.884781
- Train/validation score: `19.882952`.

## Baselines / alternatives

### Current-base 2.3427x
- Train: return 24.5533%, MDD 7.4996%, return/MDD 3.273947, Sharpe 1.500942, Sortino 1.448381, smart Sortino n/a, Calmar 3.274363, liq 0, min buffer 9605.222052, min ratio 86.074547
- Validation: return 20.1842%, MDD 6.6189%, return/MDD 3.049502, Sharpe 3.879259, Sortino 4.714085, smart Sortino n/a, Calmar 32.047719, liq 0, min buffer 9256.942427, min ratio 85.823937
- OOS: return 6.4281%, MDD 0.9293%, return/MDD 6.916878, Sharpe 5.202362, Sortino 6.795743, smart Sortino 6.543121, Calmar 43.998321, liq 0, min buffer 9924.143568, min ratio 187.204436

### Forced current-base 5x
- Train: return 60.5997%, MDD 16.2149%, return/MDD 3.737287, Sharpe 1.590031, Sortino 1.543343, smart Sortino n/a, Calmar 3.737823, liq 0, min buffer 9053.886123, min ratio 38.407976
- Validation: return 45.6166%, MDD 14.0994%, return/MDD 3.235358, Sharpe 3.888693, Sortino 4.721708, smart Sortino n/a, Calmar 65.552710, liq 1, min buffer 8415.811075, min ratio 37.185131
- OOS: return 14.0578%, MDD 1.9584%, return/MDD 7.178039, Sharpe 5.218432, Sortino 6.839753, smart Sortino 6.303942, Calmar 54.236926, liq 0, min buffer 9837.883542, min ratio 88.906120
- Note: validation has 1 tiny liquidation event; max event drawdown/equity loss `0.080233%`; this is not an account wipeout and margin buffer remains positive, but the zero-liquidation candidate is preferred.

### Candidate-portfolio tuple (without liquidation replay)
- OOS: return 8.2550%, MDD 1.8301%, return/MDD 4.510591, Sharpe 3.918301, Sortino 4.862806, smart Sortino 4.506822, Calmar 29.900159, liq 0, min buffer n/a, min ratio n/a
- Not promoted because liquidation evidence is missing at portfolio-composite level and it is diagnostic/not-promoted in the final gate.

### Tuned hybrid benchmark
- OOS: return 0.1618%, MDD 0.4897%, return/MDD 0.330422, Sharpe 0.776317, Sortino 0.899862, smart Sortino n/a, Calmar 3.575247, liq 0, min buffer n/a, min ratio n/a
- Not promoted: benchmark-only, not candidate-derived, does not beat current base or direct candidate.

## Verification

- Targeted tests: `54 passed`.
- Full pytest: `1236 passed`.
- Ruff: `All checks passed`.
- Compileall: `exit 0`.
- `git diff --check`: `exit 0`.
- CI/private-ci: pending push-time verification.

## Key artifacts

- Final decision JSON: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/profit_moonshot_live_final_selection_latest.json`
- Final decision MD: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/profit_moonshot_live_final_selection_latest.md`
- Data refresh: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/data_refresh/data_refresh_latest.json`
- Candidate tuning: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_portfolio/fresh_portfolio_tuning_latest.json`
- Liquidation validation: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/liquidation_validation/liquidation_aware_current_base_latest.json`
- Hybrid tuning/final: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/hybrid_tuning/hybrid_online_tuning_latest.json`, `var/reports/profit_moonshot_20260501/live_final_selection_20260510/hybrid_final/hybrid_online_portfolio_latest.json`

## Remaining risks / directives

- Do not promote candidate-portfolio composite rows unless they receive liquidation-aware replay evidence equivalent to the direct candidate replay.
- Keep locked-OOS gate-only/report-only; never optimize selection using OOS.
- Forced current-base 5x is acceptable only under the relaxed tiny-liquidation rule, but the recommended live candidate has stricter zero-liquidation evidence.
- Use a staged live rollout with ordinary operational kill-switches; this repository run is backtest/replay evidence, not a live order instruction.

## Candidate-derived hybrid addendum

After user review, the missing non-legacy candidate-hybrid lane was implemented and rerun. This is **not** the legacy hybrid benchmark: it reconstructs profit-moonshot candidate portfolio return streams from candidate rows, tunes an online allocator using train/validation only, and keeps locked-OOS report-only/gate-only.

- Artifact: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_hybrid/candidate_hybrid_latest.json`
- Report: `var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_hybrid/candidate_hybrid_latest.md`
- Final decision artifact now includes `candidate_hybrid` rows (`10` rows in the final comparison surface).
- Candidate-hybrid selected row: `candidate_hybrid_online_rank_01_candidate_hybrid_input_05_fresh_portfolio_train_val_monthly_`
- Train: return 33.5524%, MDD 7.0280%, return/MDD 4.7741, Sharpe 1.5649, Sortino 1.7402, smart Sortino 1.2510, Calmar 4.7741.
- Validation: return 19.1320%, MDD 3.5874%, return/MDD 5.3331, Sharpe 4.2026, Sortino 7.9196, smart Sortino 6.7832, Calmar 54.4563.
- OOS: return 7.3573%, MDD 2.6858%, return/MDD 2.7393, Sharpe 3.5505, Sortino 4.9816, smart Sortino 4.4464, Calmar 17.5808.
- Final allocation cash weight: approximately 0%; selected weights are spread over candidate-derived input portfolios, not the legacy hybrid sleeves.
- Memory: candidate-hybrid time log max RSS 327,856 KiB / artifact peak RSS 320.172 MiB, below 8 GiB.

Decision impact: candidate-hybrid is useful comparison evidence, but it is **not live-promoted** because (1) it does not beat current-base return/MDD (`2.7393` vs current-base `6.9169`), and (2) it does not yet have a dedicated dynamic-weight liquidation replay/margin-buffer proof. The live recommendation therefore remains the zero-liquidation 5x direct candidate.

## Risk-handling addendum — candidate-derived hybrid dynamic liquidation replay

- Implemented `dynamic_weight_candidate_hybrid_margin_replay_v1` in `scripts/research/run_profit_moonshot_candidate_hybrid.py` so candidate-derived hybrid rows are no longer promoted from allocator-only returns.
- Replay uses dynamic daily candidate weights, source-candidate leverage, conservative Binance-style maintenance/fee/slippage/funding/stress/liquidation buffers, and intrabar high/low liquidation checks.
- Initial candidate-hybrid replay exposed train liquidation risk in `candidate_hybrid_input_06/12/13`; follow-up risk pruning removed the offending train/validation liquidation source sleeves:
  - `fresh_calendar_rot_lethusdt_sweakest_lb168_thr200_h120_sc80_st0`
  - `fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus`
- Risk-pruned candidate hybrid is now deployable as a secondary/contingency row under the tiny-liquidation tolerance:
  - train/validation/OOS liquidation count: `0 / 1 / 0`
  - validation liquidation impact: max event drawdown `0.0000998172`, equity loss fraction `0.0000080897`, no account wipeout
  - min margin buffer train/validation/OOS: `9174.8746 / 9764.4797 / 9834.0897`
  - replay OOS return/MDD/R-MDD: `14.2041% / 1.9447% / 7.3039`
  - replay OOS Sharpe/Sortino/smart Sortino/Calmar: `5.1698 / 6.7407 / 6.2163 / 55.3698`
- Final primary recommendation remains the zero-liquidation direct 5x candidate because it has no liquidation events and slightly better OOS return/R-MDD: `14.6634% / 7.4640`.

## Integer-leverage live hardening addendum

- Live implementation cannot use fractional leverage, so final selection now includes a hard `live_integer_leverage` gate. Fractional rows are retained only as benchmark/report rows and are rejected from `deployable_candidate`.
- Candidate-derived hybrid was rebuilt with source-row integer leverage filtering before active hybrid construction. Current rebuilt hybrid had no fractional active sources: `22/22` considered source rows integer, `0` discarded; every active source input is `5x`.
- Final decision JSON/MD were regenerated with `live_integer_leverage_required=true`; the selected winner remains the zero-liquidation direct `5x` candidate.
- Verification after hardening: targeted tests `20 passed`; full pytest `1240 passed`; `ruff check .`, `python -m compileall -q .`, and `git diff --check` passed. Push/CI verification pending at this handoff line.
