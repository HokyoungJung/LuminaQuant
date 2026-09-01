# Performance-first threshold sensitivity — 2026-04-16

- basis artifact: `current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.json`
- method: lightweight analytical sweep only (no additional heavy reruns)
- current live default after override: `hybrid_guarded_mode`

## Current reboot-validation metrics
- hybrid OOS return: `+0.6868%`
- hybrid OOS sharpe: `3.2370`
- hybrid OOS max DD: `0.2573%`
- hybrid val return: `+6.5372%`
- hybrid val sharpe: `3.2857`
- balanced OOS return: `+0.1091%`
- balanced OOS sharpe: `0.4828`
- balanced OOS max DD: `0.5162%`
- balanced val return: `+8.3078%`
- balanced val sharpe: `4.1120`
- OOS return edge: `+0.5777%`
- OOS sharpe edge: `+2.7542`
- OOS drawdown ratio (hybrid / balanced): `0.498`

## Current performance-first override thresholds
- min OOS return edge: `+0.4000%`
- min OOS sharpe edge: `2.0000`
- min hybrid val return: `+5.0000%`
- min hybrid val sharpe: `3.0000`

## Sweep grid
- return edge grid: `[0.001, 0.002, 0.003, 0.004, 0.005, 0.006]`
- sharpe edge grid: `[0.75, 1.0, 1.5, 2.0, 2.5, 3.0]`
- val return grid: `[0.03, 0.04, 0.05, 0.06, 0.07]`
- val sharpe grid: `[2.0, 2.5, 3.0, 3.5, 4.0]`
- total combinations: `900`
- passing combinations: `300`

## Largest thresholds that still keep hybrid promoted
- max return-edge threshold: `+0.5000%`
- max sharpe-edge threshold: `2.5000`
- max min-val-return threshold: `+6.0000%`
- max min-val-sharpe threshold: `3.0000`

## Interpretation
- The current override is moderately robust on the lightweight grid (`300` / `900` combinations pass).
- The most fragile dimension is validation strength: promotion stops once required hybrid val return exceeds `+6.0000%` or required hybrid val sharpe exceeds `3.0000` on this grid.
- The OOS edge remains strong enough to tolerate raising the required return edge up to `+0.5000%` and the sharpe edge up to `2.5000` on this grid.
- Pair tactical remains excluded from default promotion because the policy still treats it as tactical-only despite its raw strength.
