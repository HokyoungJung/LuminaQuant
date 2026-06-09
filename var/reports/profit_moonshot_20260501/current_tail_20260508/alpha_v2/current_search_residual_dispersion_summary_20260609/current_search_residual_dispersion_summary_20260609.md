# Current-search residual/dispersion alpha summary — 2026-06-09

## Decision

- **No live/shadow promotion. Real-money allocation stays `0%`.**
- New `cross_sectional_residual_reversal` standalone failed: negative/flat locked-OOS.
- New `cross_sectional_dispersion_gated_momentum` standalone was slightly positive but too weak; when included in the current overlay it displaced better train/validation winners and degraded locked-OOS.
- Best current 3fold overlay remains the family subset **excluding dispersion-gated momentum**: `+4.16%` compounded / `+17.73%` annualized approx / `3/3` positive folds at 10bps embedded round-trip cost.
- Existing-candidate reuse diagnostic improved with `robust_quality_v1_top1`: `24.55%` comp / `30.14%` annualized / `7/10` positive, versus prior robust_top1 `22.14%` comp. This remains **post-failure research**, not clean promotion.

## Evidence table

| Run | OOS comp | Annualized approx | Positive folds | Max OOS MDD |
| --- | ---: | ---: | ---: | ---: |
| `current_search_xs_residual_reversal_smoke_20260609` | -0.25% | -1.49% | 0/2 | 0.98% |
| `current_search_residual_reversal_overlay_20260609` | 4.16% | 17.73% | 3/3 | 1.45% |
| `current_search_xs_dispersion_gated_momentum_smoke_20260609` | 0.69% | 2.79% | 2/3 | 1.56% |
| `current_search_dispersion_residual_overlay_20260609` | 0.75% | 3.04% | 2/3 | 1.56% |
| `current_search_residual_only_overlay_v2_20260609` | 4.16% | 17.73% | 3/3 | 1.45% |

## Theory anchors used

- Zhang & Makgolo, “Cross-Sectional Dispersion and the State Dependence of Cryptocurrency Momentum” — dispersion-gated momentum hypothesis: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6648082
- Dobrynskaya, “Cryptocurrency Momentum and Reversal” — crypto momentum/reversal plausibility: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263
- Avellaneda & Lee, “Statistical Arbitrage in the U.S. Equities Market” — residual mean-reversion/stat-arb framing: https://docslib.org/doc/10726218/statistical-arbitrage-in-the-u-s-equities-market
- Moskowitz, Ooi & Pedersen, “Time Series Momentum” — trend/momentum baseline: https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf

## Gate notes

- Locked OOS was report/gate-only after train+validation freeze in all clean-discovery runs.
- `robust_train_validation_v1` remains a post-failure research selector and requires fresh-forward before promotion.
- Do **not** tune thresholds or family inclusion against these locked-OOS smoke results for live deployment.
- Next real gate: freeze a candidate family subset before a genuinely new unseen slice, then test 10/15/20bps costs or paper fill telemetry, turnover/RPT, BBO spread/slippage, partial/cancel/reject, and reconciliation.
