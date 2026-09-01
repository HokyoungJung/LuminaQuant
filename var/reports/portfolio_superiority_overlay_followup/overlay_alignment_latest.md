# Portfolio Superiority Overlay Alignment

- generated_at: `2026-04-22T10:47:06.920658+00:00`
- train: `2025-01-01` -> `2025-12-31`
- val: `2026-01-01` -> `2026-02-28`
- oos: `2026-03-01` -> `2026-04-03`
- wave2_original_oos: `2026-02-01T00:00:00Z` -> `2026-04-20T13:26:24Z`

## Key realignment result
- wave2 aligned pair OOS return: `+0.1181%`
- wave2 aligned pair OOS Sharpe: `0.2376`
- wave2 aligned pair OOS MaxDD: `1.4112%`

## Baseline snapshot
- production_guarded OOS: `+0.3654%` / Sharpe `1.6080` / MaxDD `0.3184%`
- hybrid_guarded OOS: `+0.1618%` / Sharpe `0.7763` / MaxDD `0.4897%`
- soft_three_way OOS: `+0.0887%` / Sharpe `0.7423` / MaxDD `0.5734%`
- old pair tactical OOS: `+0.2892%` / Sharpe `3.2765` / MaxDD `0.0000%`

> Conclusion: overlay tests must use the aligned common baseline window; the headline wave2 +8.18% OOS sleeve metric is not directly comparable to the current March-window baseline artifacts.
