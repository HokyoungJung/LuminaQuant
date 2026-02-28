# Cost-Aware Framework Config Spec

## Required top-level keys
- `experiment_id`
- `universe.assets`
- `timeframes`
- `strategies[]`
- `cost_global`
- `run_controls`

## Strategy block
Each strategy supports:
- `name`, `enabled`
- `signal_params`
- `rebalance_rule`
- `portfolio_construction`
- `execution_model`
- `cost_model`

## Execution constraints
- `execution_model.fill_basis`: `next_open` or `mid`
- `execution_model.max_participation`: hard cap per bar
- `execution_model.unfilled_policy`: `carry` or `drop`

## Runner artifacts
`run_cost_aware_framework.py` writes:
- `reports/<run_id>/summary.json`
- `reports/<run_id>/tables.csv`
- `reports/<run_id>/config_resolved.yaml`

## Post-cost ranking objective
`run_controls` supports:
- `ranking_objective`: `composite` (default), `sharpe`, `total_return`, `drawdown`
- `ranking_weights` (used when `composite`):
  - `sharpe`
  - `total_return`
  - `drawdown`
  - `stability`
