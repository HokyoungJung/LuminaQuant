# Session handoff — Profit moonshot state-distilled market-state next pass

Date: 2026-05-12 KST
Scope: continue 2026-05-11 state-distilled leadership/unwind work without calendar/month/day/hour entry rules.

## Baseline/validity constraints

- Preserved pushed green handoff head: `private/main 7e451311757a1ce0e43bebaec0a24b3746dbcb65`.
- Preserved performance baseline/reference: `02f4520cf906f48089b8852c2651a0f1e4bd0c1c` and current-base reference OOS return `+6.4281%`, return/MDD `6.9169`.
- Current-base/calendar tuple remains `hypothesis_reference_only`; it is not a selector or live candidate.
- Locked-OOS remains `gate_only_report_only_after_train_validation_freeze`.

## Code changes

- Added/strengthened non-calendar market-state families in `scripts/research/replay_profit_moonshot_fresh_start.py`:
  - `state_distilled_crowded_unwind_v2`
  - `funding_oi_exhaustion_reversal`
  - `beta_residual_reversion`
  - `dispersion_compression_breakout_unwind`
  - `vol_regime_margin_scaled_momentum`
- Added factory/test guardrails so valid non-calendar families fail if they use `calendar_long_months`, `calendar_short_months`, `entry_days_of_month`, or `entry_hours`.
- Added explicit selection provenance and separated strict deploy vs diagnostic nonfatal liquidation lanes in `scripts/research/run_profit_moonshot_liquidation_aware_validation.py`.
- Fixed selection leakage: OOS-discovered diagnostic seeds (`integer_audit_diagnostic_best_oos` / quarantine) are report-only and cannot become train/validation selection targets.

## Research runs

### New market-state replay

Artifact directory: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_market_state_next_20260512/`

- Allowlist: `state_distilled_crowded_unwind_v2`, `funding_oi_exhaustion_reversal`, `beta_residual_reversion`, `dispersion_compression_breakout_unwind`, `vol_regime_margin_scaled_momentum`.
- Specs: `184`.
- Train/validation-positive candidates: `0`.
- Replay survivors: `0`.
- Success candidates: `0`.
- Peak RSS: `257.844 MiB` in script payload; memory guard peak `250.164 MiB`.

### Train/validation-only portfolio tuning diagnostic

Artifact directory: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_market_state_next_20260512/portfolio_tuning_leadership_unwind_top18/`

- Input: prior valid non-calendar `state_distilled_leadership_unwind_20260511` candidate CSV.
- `--no-current-base`; no calendar/current-base anchor.
- Candidate sleeves: `18`.
- Portfolio specs: `56,203`.
- Success candidates: `0`.
- Selected by train/validation stability (diagnostic, not deployable): OOS `+4.6759%`, OOS MDD `6.9705%`, OOS Sharpe `1.2568`.
- Diagnostic best-OOS portfolio (OOS-discovered, report-only, not selectable): OOS `+8.2808%`, OOS MDD `7.1067%`, Sharpe `1.8702`, but return/MDD still below reference and OOS was not a valid selector.
- Peak RSS: `954.422 MiB`.
- Full portfolio candidate CSV was compressed to `fresh_portfolio_tuning_candidates.csv.gz` (top-200 mirror: `fresh_portfolio_tuning_candidates_top200.csv`) to keep git artifacts below hosting limits.

### Liquidation-aware validation

Artifact directory: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_market_state_next_20260512/`

- Candidate seeds: `62`; evaluated integer results: `366`; deployable candidates: `0`.
- Decision: `no_live_promotion_strategy_validity_failed`; `deployable_improvement=false`.
- Top train/validation-selected retune seed: `integer_audit_selected_by_train_val_stability`, selected at `2x`.
  - Train `+43.5818%`, MDD `18.6069%`, Sharpe `1.5922`, liquidation `0`, min buffer `9468.5897`.
  - Validation `+18.4067%`, MDD `4.6310%`, Sharpe `5.9590`, liquidation `0`, min buffer `9918.1941`.
  - Locked-OOS report: `+2.1440%`, MDD `5.0839%`, Sharpe `0.8590`, Sortino `1.1255`, Calmar `2.4354`, liquidation `0`, min buffer `9715.8844`.
  - Strict-liquidation safe, but failed OOS return/return-MDD/Sharpe/Sortino/smart-Sortino gates vs baseline.
- Highest strict zero-liquidation selection-target leverage: `4x` on `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600`.
  - Train `+32.9431%`, MDD `9.4768%`, Sharpe `1.9463`, liquidation `0`.
  - Validation `+11.6925%`, MDD `3.1028%`, Sharpe `4.9606`, liquidation `0`.
  - Locked-OOS report `+2.4722%`, MDD `2.5328%`, Sharpe `1.5131`, Sortino `1.8815`, Calmar `5.6787`, liquidation `0`.
  - Not deployable: OOS return `+2.4722%` and return/MDD `0.9761` do not beat current-base reference `+6.4281%` and `6.9169`.
- Diagnostic 5x/6x high-leverage lane is non-promotional:
  - Train/validation-selected portfolio at `5x`: total liquidation `10`, no wipeout, OOS `+5.0099%`, OOS MDD `12.3856%`.
  - Same at `6x`: total liquidation `17`, no wipeout, OOS `+5.8698%`, OOS MDD `14.7349%`.
- Validation memory guard peak `268.637 MiB`, under 8 GiB.

## Conclusion

Strategy validity and provenance improved, but strict deployable improvement is still not achieved. The new crowded/funding/OI/residual/dispersion/scaled market-state families did not produce train/validation-positive single-spec candidates in the narrow grid. The best strict zero-liquidation non-calendar candidate remains the prior state-distilled leadership/unwind row at `4x`, and it does not beat the invalid current-base reference economics.

## Next recommended work

1. Do not widen grids blindly. First diagnose why the new five families are mostly inactive or negative on train/validation.
2. Add mechanism diagnostics to replay output: trade counts by split, signal-side reason histograms, feature threshold hit rates, and per-family train/validation activation coverage.
3. Explore non-calendar combinations using only train/validation stability, but keep OOS-discovered `diagnostic_best_oos` rows permanently report-only.
4. Consider a future fresh holdout before any live-promotion claim, because the locked-OOS window has now been repeatedly observed.
