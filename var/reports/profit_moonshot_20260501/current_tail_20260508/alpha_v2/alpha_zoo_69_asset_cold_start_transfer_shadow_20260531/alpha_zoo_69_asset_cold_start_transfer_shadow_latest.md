# 69-Asset Cold-Start Transfer Shadow Report

This artifact is report-only. It does not promote validation-only assets to live, paper, or real-money execution.

## Safety Contract

- `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`.
- Donor selection uses donor train/validation quality, static domain similarity, and target coverage only.
- Target validation PnL is not used for the primary donor-frozen selection.
- Target train metrics are not synthesized from donor performance.
- The validation-oracle lane is an upper-bound diagnostic only and is not promotable.

## Portfolio Results

| lane | sleeves | gross | train return | validation return | validation MDD | validation RPT | promotable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| donor-frozen primary | 18 | 2.00 | 0.00% | 31.68% | 10.24% | 83.87bps | no |
| validation-oracle diagnostic | 16 | 1.78 | 0.00% | 44.47% | 8.18% | 178.67bps | no |

## Strict Live Reference

Corrected strict live handoff remains the reference: train 119.38%, validation 79.71%, validation MDD 7.48%, gross 2.20.

## Primary Shadow Sleeves

| symbol | donor | family | tf | weighted notional | val return | val MDD | val RPT |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| MUUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 68.79% | 9.71% | 280.78bps |
| SNDKUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 42.14% | 15.72% | 162.71bps |
| AMDUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 15.62% | 7.82% | 159.43bps |
| DRAMUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 14.93% | 5.51% | 304.73bps |
| QCOMUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 14.68% | 11.71% | 123.37bps |
| SOXLUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 11.73% | 13.50% | 186.11bps |
| QQQUSDT | EWJUSDT | volatility_adjusted_trend_persistence | 30m | 0.111 | 4.45% | 1.16% | 22.82bps |
| SPYUSDT | EWJUSDT | volatility_adjusted_trend_persistence | 30m | 0.111 | 3.35% | 0.63% | 17.73bps |
| MRVLUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 6.76% | 8.15% | 107.26bps |
| ARMUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 6.05% | 6.59% | 288.24bps |
| AVGOUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 5.31% | 4.92% | 36.11bps |
| TSMUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | 2.74% | 7.46% | 13.52bps |
| QNTXUSDT | PLTRUSDT | cross_sectional_momentum_rank | 30m | 0.111 | 1.40% | 2.01% | 116.83bps |
| WDCUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | -0.32% | 5.63% | -22.97bps |
| OPENAIUSDT | PLTRUSDT | cross_sectional_momentum_rank | 30m | 0.111 | -1.63% | 3.68% | -30.27bps |
| SPCXUSDT | PLTRUSDT | cross_sectional_momentum_rank | 4h | 0.111 | -2.92% | 4.81% | -292.32bps |
| BABAUSDT | AMZNUSDT | volatility_adjusted_trend_persistence | 30m | 0.111 | -0.79% | 1.77% | -24.62bps |
| COHRUSDT | INTCUSDT | cross_sectional_momentum_rank | 1h | 0.111 | -4.92% | 7.18% | -702.27bps |

## Conclusion

The donor-frozen cold-start transfer lane was positive on validation, but it remains shadow-only because every selected target has zero train-window rows.
The validation-oracle upper bound is better than the donor-frozen lane, which means future real train data may unlock useful sleeves, but current validation-only selection would be data leakage.
