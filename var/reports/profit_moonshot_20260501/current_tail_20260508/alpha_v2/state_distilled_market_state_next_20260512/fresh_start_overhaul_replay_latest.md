# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-12T11:18:53.160357Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, non-calendar state-distilled leadership/unwind, non-calendar crowded-leadership unwind v2, non-calendar TRX state-momentum proxy, non-calendar TRX/ETH state-relative-strength spread, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `184`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `257.844 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_dispersion_compression_state_lb24_thr60_comp75_fl6` | `dispersion_compression_breakout_unwind` | False | False | -0.0531% | +0.0000% | +0.0000% | +0.0000% | 0.000000 | 0 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 2 | `fresh_dispersion_compression_state_lb24_thr120_comp75_fl3` | `dispersion_compression_breakout_unwind` | False | False | -0.1302% | +0.0000% | +0.0000% | +0.0000% | 0.000000 | 0 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 3 | `fresh_dispersion_compression_state_lb24_thr120_comp75_fl6` | `dispersion_compression_breakout_unwind` | False | False | -0.0221% | +0.0000% | +0.0000% | +0.0000% | 0.000000 | 0 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 4 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret120_h72_fund_gap20` | `state_distilled_crowded_unwind_v2` | False | False | -0.4119% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 5 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret120_h72_fund_gap40` | `state_distilled_crowded_unwind_v2` | False | False | -0.3883% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 6 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret120_h96_fund_gap20` | `state_distilled_crowded_unwind_v2` | False | False | -0.3404% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 7 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret120_h96_fund_gap40` | `state_distilled_crowded_unwind_v2` | False | False | -0.3136% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 8 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret200_h72_fund_gap20` | `state_distilled_crowded_unwind_v2` | False | False | -0.4119% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 9 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret200_h72_fund_gap40` | `state_distilled_crowded_unwind_v2` | False | False | -0.3883% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 10 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret200_h96_fund_gap20` | `state_distilled_crowded_unwind_v2` | False | False | -0.3404% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 11 | `fresh_state_crowded_unwind_v2_lb168_fast72_z175_ret200_h96_fund_gap40` | `state_distilled_crowded_unwind_v2` | False | False | -0.3136% | -0.0359% | -0.0077% | +0.0077% | -3.318425 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 12 | `fresh_funding_oi_exhaustion_rev_lb72_fast24_z075_f120_oi12_fl3` | `funding_oi_exhaustion_reversal` | False | False | +0.0000% | +0.0000% | -0.0162% | +0.0162% | -2.839431 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 13 | `fresh_funding_oi_exhaustion_rev_lb72_fast24_z075_f120_oi12_fl6` | `funding_oi_exhaustion_reversal` | False | False | +0.0000% | +0.0000% | -0.0162% | +0.0162% | -2.839431 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 14 | `fresh_funding_oi_exhaustion_rev_lb72_fast24_z075_f120_oi20_fl3` | `funding_oi_exhaustion_reversal` | False | False | +0.0000% | +0.0000% | -0.0162% | +0.0162% | -2.839431 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 15 | `fresh_funding_oi_exhaustion_rev_lb72_fast24_z075_f120_oi20_fl6` | `funding_oi_exhaustion_reversal` | False | False | +0.0000% | +0.0000% | -0.0162% | +0.0162% | -2.839431 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
