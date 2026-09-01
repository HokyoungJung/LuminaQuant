# Profit moonshot return-quality contract — 2026-05-09

## User correction
The previous alpha-v2 "pass" is not sufficient. MDD can be tolerated up to roughly `25%`, but return must be stable and materially higher: target is about `2%` average monthly return with high Sharpe, Sortino, smart Sortino, and Calmar.

## New promotion contract
A new candidate may be called `improved` only when all conditions hold:
- Selection uses train/validation only; locked-OOS remains report-only/gate-only.
- Train monthlyized return `>= +2.0%`.
- Validation monthlyized return `>= +2.0%`.
- Locked-OOS monthlyized return `>= +2.0%`.
- Locked-OOS total return beats the current champion `+1.2181%` and return/risk beats the champion.
- Locked-OOS MDD is within the relaxed budget: `<= 25%`.
- Locked-OOS Sharpe `>= 2.0`, Sortino `>= 3.0`, smart Sortino `>= 3.0`, Calmar `>= 1.0`.
- Locked-OOS participation is not starved.

## Superseded alpha-v2 result
The old target-budget calendar candidate is now historical evidence only, not a pass:
- Train total return `+5.5019%`, monthlyized `+0.4474%`.
- Validation total return `+4.0000%`, monthlyized `+2.0440%`.
- Locked-OOS total return `+1.3397%`, monthlyized `+0.6121%`.
- Locked-OOS MDD `+0.1774%`, Sharpe `5.4774`, Sortino `6.7769`, Calmar `42.8252`, smart Sortino `2.0594`.
- Failed new gates: `train_monthly_return_gte_2pct`, `oos_monthly_return_gte_2pct`, `oos_smart_sortino_high`.

## Re-evaluation evidence
Return-quality replay artifact:
`var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_return_quality_v1/fresh_portfolio_tuning_latest.json`

Run parameters: top-n `30`, family-quota `5`, calendar-neighborhood-reps `10`, max-sleeves `4`, max-combos-per-size `6000`, cluster-cap `0.40`, sleeve-cap `0.25`.

Results:
- Portfolio specs evaluated: `62,970`.
- Success candidates under new contract: `0`.
- Peak RSS payload: `812.921875 MiB`; `/usr/bin/time` max RSS `832432 KB`; below 8 GiB.
- Selected-by-validation diagnostic: train monthly `+0.9286%`, validation monthly `+6.7768%`, locked-OOS monthly `+0.9745%`; failed train/OOS monthly and smart-Sortino/return-risk gates.
- Diagnostic best locked-OOS: train monthly `+0.9583%`, validation monthly `+5.7605%`, locked-OOS monthly `+1.8056%`; failed train/OOS monthly gates.

## Code-state directives
- `scripts/research/tune_profit_moonshot_fresh_portfolio.py` now computes monthlyized return and local smart Sortino and requires the return-quality gates before promotion.
- `scripts/research/validate_profit_moonshot_pass_under_8gb.py` now rejects stale mission pass labels that lack the 2% monthly/quality evidence.
- `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json` and `passing_candidate_latest.json` are marked superseded, not passed.

## Next research lane
Continue alpha discovery, but do not lower the return floor. The current evidence says calendar sleeves are high quality but underpowered on train/OOS monthly return. Next lanes should search for additional stable return drivers that are selected on train/validation only, then combine with calendar sleeves under the same locked-OOS gates.
