# 69-Asset Monthly Refit Diagnostics

- candidate: `individual_robust:hybrid_v3_5`
- aggregate recompute: `PASS`
- fold schedule: `PASS`
- latest detail match: `True`

## Selected candidate extended metrics

- OOS comp: `17.78%`
- hit rate: `50.00%`
- monthly Sharpe approx: `0.75`
- monthly Sortino approx: `2.31`
- 5% monthly VaR: `-7.25%`
- 25% monthly CVaR: `-6.59%`
- avg gain / avg loss: `8.64%` / `-4.62%`
- gain/loss ratio: `1.87`
- equity max DD: `14.99%`
- max loss streak: `3`

## Diagnostic challenger selector

- name: `diagnostic_individual_calmar_vcap20_vmdd8_with_relaxed_fallback`
- clean promotion allowed: `False`
- warning: This rule was surfaced after inspecting historical OOS; use only as a forward shadow challenger unless it passes future months without further tuning.
- OOS comp: `35.36%`
- hit rate: `60.00%`
- min monthly OOS: `-7.20%`
- max fold MDD: `13.85%`

This report distinguishes verified accounting from diagnostic OOS-ranked ideas.
