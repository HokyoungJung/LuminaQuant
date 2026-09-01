# Performance Evaluator: profit-moonshot-return-quality

## Objective
Find and verify a profit moonshot candidate with train/validation/OOS monthlyized return >=2%, OOS MDD <=25%, high Sharpe/Sortino/smart Sortino/Calmar, train/validation-only selection, locked-OOS report/gate-only, and <8GiB memory

## Evaluator Command
```sh
uv run --extra dev python scripts/research/validate_profit_moonshot_pass_under_8gb.py --result-path .omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json --output-path .omx/specs/autoresearch-profit-moonshot-alpha-v2/validation_latest.json --markdown-path var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/return_quality_validation_20260509.md
```

## Pass/Fail Contract
PASS only when validator passes and the candidate_return_quality_contract proves train/validation/OOS monthlyized return >=2%, OOS MDD <=25%, OOS Sharpe >=2, Sortino >=3, smart Sortino >=3, Calmar >=1, current champion return/risk improvement, locked-OOS report/gate-only, and <8GiB memory evidence.

This evaluator must exist and produce concrete pass/fail evidence before the performance goal can be completed.
