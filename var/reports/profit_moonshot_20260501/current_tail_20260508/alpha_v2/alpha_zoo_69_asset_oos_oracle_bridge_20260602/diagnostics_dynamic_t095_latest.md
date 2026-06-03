# 69-Asset Monthly Refit Diagnostics

- candidate: `dynamic_conviction_switch:t0.95_risk_capped_fallback`
- aggregate recompute: `PASS`
- fold schedule: `PASS`
- latest detail match: `None`

## Selected candidate extended metrics

- OOS comp: `53.33%`
- hit rate: `50.00%`
- monthly Sharpe approx: `2.07`
- monthly Sortino approx: `13.82`
- 5% monthly VaR: `-2.37%`
- 25% monthly CVaR: `-1.80%`
- avg gain / avg loss: `10.34%` / `-1.38%`
- gain/loss ratio: `7.48`
- equity max DD: `4.62%`
- max loss streak: `2`

## Diagnostic challenger selector

- name: `diagnostic_individual_calmar_vcap20_vmdd8_with_relaxed_fallback`
- clean promotion allowed: `False`
- warning: This rule was surfaced after inspecting historical OOS; use only as a forward shadow challenger unless it passes future months without further tuning.
- OOS comp: `-24.37%`
- hit rate: `30.00%`
- min monthly OOS: `-14.79%`
- max fold MDD: `17.65%`

This report distinguishes verified accounting from diagnostic OOS-ranked ideas.
