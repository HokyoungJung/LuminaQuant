# Integer-Leverage Hybrid Decision Verification Summary

Generated on 2026-05-24 KST/UTC session for `alpha_zoo_integer_leverage_hybrid_decision_20260524`.

## Result

The runner reconstructed the three 10bps-costed integer-leverage profile PnL streams and selected `hybrid_mdd20_three_profile_blend` using train+validation evidence only:

- weights: `balanced_mdd12_gross5=0.15`, `growth_mdd20_gross8=0.70`, `aggressive_mdd30_gross10_shadow=0.15`
- gross notional: `3.615x`
- train/validation/locked-OOS report-only return: `+262.3642% / +72.5692% / +21.2977%`
- validation/OOS MDD: `19.9330% / 9.0718%`
- RPT proxy train/validation/OOS: `33.78 / 39.96 / 22.97bps`
- trade events train/validation/OOS: `3363 / 789 / 362`
- liquidation/account-wipeout: `0/0` across train, validation, and locked-OOS report-only

The hybrid is paper/testnet-only and relaxed, not strict 12% validation-MDD promotion. All outputs keep `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

## Method checks

- Source profiles are the three paper/testnet candidates from the frozen integer-leverage artifact.
- Hybrid weights are selected on a 5% grid with each source profile weight >=10%.
- Selection score and selection gates use train+validation only; locked-OOS is attached after freeze as gate/report-only evidence.
- Primary cost remains `10bps` all-in round-trip backtest friction proxy.
- RPT gate remains `avg BBO spread 2bps * 5 = 10bps`.
- No calendar/date feature or real-money execution path is introduced.

## Correlation note

The source profile PnL streams are not independent enough for a large diversification miracle. Train+validation PnL correlations are approximately:

- balanced vs growth: `0.5790`
- balanced vs aggressive: `0.8493`
- growth vs aggressive: `0.8680`

The hybrid therefore mainly produces a risk-return compromise around the MDD~20 profile, not a fundamentally new alpha sleeve.

## Verification evidence

Log: `local_verification_hybrid_decision_20260524T132911Z.log`

- Artifact invariant: passed (`ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`, 10bps cost, 10bps RPT threshold, locked-OOS selection=false, 4 comparison rows).
- Targeted tests: `tests/test_alpha_zoo_integer_leverage_hybrid_decision.py` + `tests/test_alpha_zoo_corr_integer_leverage_portfolio.py` -> `9 passed`.
- `ruff check .`: passed.
- `python -m compileall -q src scripts tests`: passed.
- `python scripts/audit_hardcoded_params.py`: `total=567 new=0 baselined=567`.
- `git diff --check` and `git diff --cached --check`: passed.
- Full pytest: `1440 passed in 68.46s`.
- Runner max RSS: `6,505,524 KiB` (<8 GiB).
- Full pytest max RSS: `2,771,296 KiB` (<8 GiB).
