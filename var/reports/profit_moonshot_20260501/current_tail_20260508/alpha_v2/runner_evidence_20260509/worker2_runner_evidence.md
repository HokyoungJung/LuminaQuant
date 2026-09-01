# Worker 2 runner evidence — task 3 — 2026-05-09

## Policy
- Heavy replay authorized: `false`.
- Heavy replay started: `false`.
- Pre-run heavy-job check: no matching heavy runner processes before the bounded smoke run.
- Delegation compliance: Subagent skip reason: serial runner/evidence work was safer because the task forbids overlapping heavy jobs and the required checks were bounded artifact/smoke reads.

## Bounded smoke
- Command: `uv run python scripts/minimum_viable_run.py --days 30`
- Exit status: `0`
- Wall time: `0:01.06`
- Max RSS: `134272 KB`
- Output: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/runner_evidence_20260509/worker2_minimum_viable_run_output.txt`
- Time log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/runner_evidence_20260509/worker2_minimum_viable_run_time.txt`

## Candidate metrics
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`
- Mode/leverage: `train_val_monthly_return_budget` / `2.3427334297703024`
- Promotion: `improved_success_candidate`; success candidate: `True`; failed gates: `[]`
- Selection basis: `train_val_only`; uses locked-OOS for selection: `False`
- Train: return `26.8207%`, monthly `2.0000%`, MDD `6.9060%`, Sharpe `1.7213`, Sortino `1.5151`, Calmar `3.8842`
- Validation: return `19.9713%`, monthly `9.8490%`, MDD `6.4935%`, Sharpe `4.0964`, Sortino `4.8859`, Calmar `32.1417`
- Locked-OOS: return `6.8582%`, monthly `3.0883%`, MDD `0.8198%`, Sharpe `5.6537`, Sortino `7.3961`, smart Sortino `7.1536`, Calmar `53.7350`, round trips `33`
- Champion OOS return: `1.2181%`; excess: `5.6401%`

## Memory evidence
- Portfolio specs: `73465`; success candidates: `8`
- Payload peak RSS: `920.51171875 MiB`
- Memory guard peak RSS: `911.58984375 MiB` (`955871232` bytes); under 8 GiB: `True`
- Soft/hard memory triggers: `0` / `0`
- Summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Artifact validator
- Status: `passed`; passed: `True`
- Validation JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/runner_evidence_20260509/worker2_validation_latest.json`
