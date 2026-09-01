# Optuna Hybrid Verification Summary

Generated: `2026-05-24T14:28:06.729552Z`

- Artifact invariant: passed (`ready_for_real=false`, `real_money_execution=false`, locked-OOS selection/objective/pruning/fitting/discovery flags false, 10bps cost/RPT threshold).
- Runner: `--n-trials 240 --seed 20260524`, max RSS `6,357,368 KiB` (<8 GiB), elapsed `32:37.97`.
- Targeted tests: `13 passed` (`test_alpha_zoo_integer_leverage_optuna_hybrid_decision`, hybrid grid baseline, integer leverage).
- Ruff: `ruff check .` passed.
- Compileall: `python -m compileall -q src scripts tests` passed.
- Hardcoded audit: `total=567 new=0 baselined=567`.
- Diff checks: `git diff --check` and `git diff --cached --check` passed.
- Full pytest: `1444 passed in 76.49s`; max RSS `2,723,060 KiB` (<8 GiB).

Selected Optuna result: `hybrid_v3_5_optuna_three_profile_blend` train/validation/OOS `611.5025%` / `138.3170%` / `20.8319%`, validation MDD `18.9796%`, RPT `83.39/79.17/25.29` bps.
