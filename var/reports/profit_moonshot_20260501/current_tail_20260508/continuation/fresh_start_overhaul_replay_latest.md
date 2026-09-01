# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-08T11:07:22.129356Z`
OOS end date: `2026-05-08`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, calendar rotation, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+0.8284%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `4941`
- Replay survivors: `63`
- Success candidates: `63`
- Peak RSS: `2129.160 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss80_tp180` | `calendar_rotation` | True | True | +1.9449% | +0.5464% | +1.0074% | +0.1532% | 7.949855 | 15 | `` |
| 2 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss100_tp180` | `calendar_rotation` | True | True | +2.2558% | +0.6824% | +1.0074% | +0.1532% | 7.949855 | 15 | `` |
| 3 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp180` | `calendar_rotation` | True | True | +2.5676% | +0.8184% | +1.0074% | +0.1532% | 7.949855 | 15 | `` |
| 4 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss80_tp180` | `calendar_rotation` | True | True | +1.9216% | +0.5464% | +0.9747% | +0.1483% | 7.949850 | 15 | `` |
| 5 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss100_tp180` | `calendar_rotation` | True | True | +2.2324% | +0.6824% | +0.9747% | +0.1483% | 7.949850 | 15 | `` |
| 6 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls600_ss120_tp180` | `calendar_rotation` | True | True | +2.5441% | +0.8184% | +0.9747% | +0.1483% | 7.949850 | 15 | `` |
| 7 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss80_tp180` | `calendar_rotation` | True | True | +1.8983% | +0.5464% | +0.9421% | +0.1433% | 7.949844 | 15 | `` |
| 8 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss100_tp180` | `calendar_rotation` | True | True | +2.2091% | +0.6824% | +0.9421% | +0.1433% | 7.949844 | 15 | `` |
| 9 | `fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls580_ss120_tp180` | `calendar_rotation` | True | True | +2.5207% | +0.8184% | +0.9421% | +0.1433% | 7.949844 | 15 | `` |
| 10 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss80_tp180` | `calendar_rotation` | True | True | +1.7799% | +0.1007% | +0.9402% | +0.1540% | 6.849368 | 17 | `` |
| 11 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss100_tp180` | `calendar_rotation` | True | True | +2.0462% | +0.1217% | +0.9402% | +0.1540% | 6.849368 | 17 | `` |
| 12 | `fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp180` | `calendar_rotation` | True | True | +2.3128% | +0.1513% | +0.9402% | +0.1540% | 6.849368 | 17 | `` |
| 13 | `fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss80_tp180` | `calendar_rotation` | True | True | +1.9537% | +0.1666% | +0.9402% | +0.1540% | 6.849368 | 17 | `` |
| 14 | `fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss100_tp180` | `calendar_rotation` | True | True | +2.2643% | +0.2070% | +0.9402% | +0.1540% | 6.849368 | 17 | `` |
| 15 | `fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp180` | `calendar_rotation` | True | True | +2.5767% | +0.2476% | +0.9402% | +0.1540% | 6.849368 | 17 | `` |

## Decision

- At least one fresh-start replay candidate earned a one-at-a-time full live-equivalent backtest slot.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
