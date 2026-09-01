# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-08T12:28:37.129014Z`
OOS end date: `2026-05-08`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, calendar rotation, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+0.8284%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `6805`
- Replay survivors: `300`
- Success candidates: `300`
- Peak RSS: `289.465 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls620_ss80_tp180` | `calendar_rotation` | True | True | +1.2877% | +0.3008% | +1.0872% | +0.1093% | 9.001618 | 16 | `` |
| 2 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls620_ss100_tp180` | `calendar_rotation` | True | True | +1.4572% | +0.3755% | +1.0872% | +0.1093% | 9.001618 | 16 | `` |
| 3 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls620_ss120_tp180` | `calendar_rotation` | True | True | +1.6267% | +0.4491% | +1.0872% | +0.1093% | 9.001618 | 16 | `` |
| 4 | `fresh_calendar_trx_takeprofit_sweakest_thr150_h168_ls620_ss80_tp180` | `calendar_rotation` | True | True | +1.3024% | +0.2784% | +1.0606% | +0.0845% | 9.140637 | 14 | `` |
| 5 | `fresh_calendar_trx_takeprofit_sweakest_thr150_h168_ls620_ss100_tp180` | `calendar_rotation` | True | True | +1.4719% | +0.3467% | +1.0606% | +0.0845% | 9.140637 | 14 | `` |
| 6 | `fresh_calendar_trx_takeprofit_sweakest_thr150_h168_ls620_ss120_tp180` | `calendar_rotation` | True | True | +1.6415% | +0.4149% | +1.0606% | +0.0845% | 9.140637 | 14 | `` |
| 7 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss80_tp180` | `calendar_rotation` | True | True | +2.4001% | +0.1884% | +1.0606% | +0.0845% | 9.140637 | 14 | `` |
| 8 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss100_tp180` | `calendar_rotation` | True | True | +2.7426% | +0.2345% | +1.0606% | +0.0845% | 9.140637 | 14 | `` |
| 9 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp180` | `calendar_rotation` | True | True | +3.0858% | +0.2795% | +1.0606% | +0.0845% | 9.140637 | 14 | `` |
| 10 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls600_ss80_tp180` | `calendar_rotation` | True | True | +1.2680% | +0.3008% | +1.0520% | +0.1057% | 9.001613 | 16 | `` |
| 11 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls600_ss100_tp180` | `calendar_rotation` | True | True | +1.4374% | +0.3755% | +1.0520% | +0.1057% | 9.001613 | 16 | `` |
| 12 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls600_ss120_tp180` | `calendar_rotation` | True | True | +1.6069% | +0.4491% | +1.0520% | +0.1057% | 9.001613 | 16 | `` |
| 13 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls590_ss80_tp180` | `calendar_rotation` | True | True | +1.2581% | +0.3008% | +1.0343% | +0.1040% | 9.001610 | 16 | `` |
| 14 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls590_ss100_tp180` | `calendar_rotation` | True | True | +1.4275% | +0.3755% | +1.0343% | +0.1040% | 9.001610 | 16 | `` |
| 15 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h168_ls590_ss120_tp180` | `calendar_rotation` | True | True | +1.5970% | +0.4491% | +1.0343% | +0.1040% | 9.001610 | 16 | `` |

## Decision

- At least one fresh-start replay candidate earned a one-at-a-time full live-equivalent backtest slot.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
