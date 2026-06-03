# 69-Asset Monthly Refit Diagnostics

- candidate: `dynamic_conviction_switch:t0.90_risk_capped_fallback`
- aggregate recompute: `PASS`
- fold schedule: `PASS`
- latest detail match: `None`

## Selected candidate extended metrics

- OOS comp: `53.38%`
- hit rate: `50.00%`
- monthly Sharpe approx: `2.07`
- monthly Sortino approx: `15.31`
- 5% monthly VaR: `-2.37%`
- 25% monthly CVaR: `-1.80%`
- avg gain / avg loss: `10.49%` / `-1.25%`
- gain/loss ratio: `8.40`
- equity max DD: `4.62%`
- max loss streak: `2`

## Diagnostic challenger selector

- name: `diagnostic_individual_calmar_vcap20_vmdd8_with_relaxed_fallback`
- clean promotion allowed: `False`
- warning: This rule was surfaced after inspecting historical OOS; use only as a forward shadow challenger unless it passes future months without further tuning.
- OOS comp: `-15.22%`
- hit rate: `30.00%`
- min monthly OOS: `-5.24%`
- max fold MDD: `12.96%`

This report distinguishes verified accounting from diagnostic OOS-ranked ideas.
