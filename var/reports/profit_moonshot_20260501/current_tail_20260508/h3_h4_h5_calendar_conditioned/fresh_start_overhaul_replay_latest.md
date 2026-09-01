# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-09T04:45:47.012341Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `80`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `250.707 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_calendar_trx_veto_rz10_sweakest_thr180_h168` | `calendar_rotation` | False | False | +2.0672% | +1.9644% | +0.9737% | +0.1512% | 5.390546 | 6 | `oos_return_beats_incumbent` |
| 2 | `fresh_calendar_trx_veto_rz10_sethusdt_thr180_h168` | `calendar_rotation` | False | False | +3.9663% | +2.9389% | +0.9737% | +0.1512% | 5.390546 | 6 | `oos_return_beats_incumbent` |
| 3 | `fresh_calendar_trx_veto_mkt24_sweakest_thr180_h168` | `calendar_rotation` | False | False | +0.4739% | +0.3636% | +0.9556% | +0.1512% | 5.317699 | 6 | `oos_return_beats_incumbent` |
| 4 | `fresh_calendar_trx_veto_mkt24_sethusdt_thr180_h168` | `calendar_rotation` | False | False | +3.6371% | +2.2376% | +0.9556% | +0.1512% | 5.317699 | 6 | `oos_return_beats_incumbent` |
| 5 | `fresh_calendar_trx_veto_fund100_sweakest_thr150_h168` | `calendar_rotation` | False | False | +2.0624% | +1.9769% | +0.9553% | +0.1509% | 5.281609 | 6 | `oos_return_beats_incumbent` |
| 6 | `fresh_calendar_trx_veto_fund100_sethusdt_thr150_h168` | `calendar_rotation` | False | False | +3.6035% | +2.5521% | +0.9553% | +0.1509% | 5.281609 | 6 | `oos_return_beats_incumbent` |
| 7 | `fresh_calendar_trx_veto_rz15_sweakest_thr180_h168` | `calendar_rotation` | False | False | +2.3842% | +1.9644% | +0.9245% | +0.1512% | 5.143585 | 6 | `oos_return_beats_incumbent` |
| 8 | `fresh_calendar_trx_veto_flow6_sweakest_thr180_h168` | `calendar_rotation` | False | False | +2.4371% | +1.9644% | +0.9245% | +0.1512% | 5.143585 | 6 | `oos_return_beats_incumbent` |
| 9 | `fresh_calendar_trx_veto_rz15_sethusdt_thr180_h168` | `calendar_rotation` | False | False | +4.2892% | +2.9389% | +0.9245% | +0.1512% | 5.143585 | 6 | `oos_return_beats_incumbent` |
| 10 | `fresh_calendar_trx_veto_flow6_sethusdt_thr180_h168` | `calendar_rotation` | False | False | +4.3431% | +2.9389% | +0.9245% | +0.1512% | 5.143585 | 6 | `oos_return_beats_incumbent` |
| 11 | `fresh_calendar_trx_veto_rz15_sweakest_thr150_h120` | `calendar_rotation` | False | False | +1.7561% | +2.7639% | +0.8337% | +0.2245% | 4.569037 | 8 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 12 | `fresh_calendar_trx_veto_rz15_sethusdt_thr150_h120` | `calendar_rotation` | False | False | +2.3201% | +3.4108% | +0.8337% | +0.2245% | 4.569037 | 8 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 13 | `fresh_calendar_trx_veto_rz10_sweakest_thr150_h168` | `calendar_rotation` | False | False | -1.1854% | +1.9769% | +0.7813% | +0.2710% | 4.188878 | 7 | `train_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 14 | `fresh_calendar_trx_veto_rz10_sethusdt_thr150_h168` | `calendar_rotation` | False | False | +4.2112% | +2.9521% | +0.7813% | +0.2710% | 4.188878 | 7 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 15 | `fresh_calendar_trx_veto_flow6_sweakest_thr150_h120` | `calendar_rotation` | False | False | +1.7519% | +2.7643% | +0.7792% | +0.2656% | 4.220519 | 9 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
