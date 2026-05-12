# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-12T13:27:25.399052Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, non-calendar state-distilled leadership/unwind, non-calendar crowded-leadership unwind v2, non-calendar TRX state-momentum proxy, non-calendar TRX/ETH state-relative-strength spread, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `1728`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `280.348 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret60_h120_tp750_fl0_xr200` | `state_distilled_external_risk_filter` | False | False | +0.0567% | -0.2313% | +1.4421% | +0.3708% | 4.444125 | 6 | `val_positive,oos_mdd_beats_shadow` |
| 2 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret120_h120_tp750_fl0_xr200` | `state_distilled_external_risk_filter` | False | False | +0.0130% | -0.2313% | +1.4284% | +0.3682% | 4.585932 | 6 | `val_positive,oos_mdd_beats_shadow` |
| 3 | `fresh_state_distilled_ext_both_lb336_fast168_z050_ret180_h120_tp750_fl0_xr200` | `state_distilled_external_risk_filter` | False | False | +3.2248% | +1.8146% | +1.4128% | +0.5320% | 3.669888 | 8 | `oos_mdd_beats_shadow` |
| 4 | `fresh_state_distilled_ext_both_lb336_fast168_z050_ret180_h120_tp750_fl0_xr125` | `state_distilled_external_risk_filter` | False | False | +2.2610% | +2.0088% | +1.4112% | +0.5320% | 3.867471 | 7 | `oos_mdd_beats_shadow` |
| 5 | `fresh_state_distilled_ext_both_lb336_fast168_z050_ret180_h120_tp750_fl0_xr150` | `state_distilled_external_risk_filter` | False | False | +2.9881% | +1.7598% | +1.4112% | +0.5320% | 3.867471 | 7 | `oos_mdd_beats_shadow` |
| 6 | `fresh_state_distilled_ext_both_lb336_fast168_z050_ret180_h120_tp750_fl0_xr175` | `state_distilled_external_risk_filter` | False | False | +3.5352% | +1.8146% | +1.4112% | +0.5320% | 3.867471 | 7 | `oos_mdd_beats_shadow` |
| 7 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret180_h120_tp750_fl0_xr200` | `state_distilled_external_risk_filter` | False | False | -0.0951% | -0.2313% | +1.4086% | +0.3713% | 4.526432 | 6 | `train_positive,val_positive,oos_mdd_beats_shadow` |
| 8 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret120_h120_tp750_fl0_xr125` | `state_distilled_external_risk_filter` | False | False | +0.1338% | -0.0313% | +1.4069% | +0.2826% | 4.969030 | 5 | `val_positive,oos_mdd_beats_shadow` |
| 9 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret120_h120_tp750_fl0_xr150` | `state_distilled_external_risk_filter` | False | False | +0.1338% | -0.2878% | +1.4069% | +0.2826% | 4.969030 | 5 | `val_positive,oos_mdd_beats_shadow` |
| 10 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret120_h120_tp750_fl0_xr175` | `state_distilled_external_risk_filter` | False | False | +0.0130% | -0.2313% | +1.4069% | +0.2826% | 4.969030 | 5 | `val_positive,oos_mdd_beats_shadow` |
| 11 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret180_h120_tp750_fl0_xr125` | `state_distilled_external_risk_filter` | False | False | +0.0258% | -0.0313% | +1.4069% | +0.2826% | 4.969030 | 5 | `val_positive,oos_mdd_beats_shadow` |
| 12 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret180_h120_tp750_fl0_xr150` | `state_distilled_external_risk_filter` | False | False | +0.0258% | -0.2878% | +1.4069% | +0.2826% | 4.969030 | 5 | `val_positive,oos_mdd_beats_shadow` |
| 13 | `fresh_state_distilled_ext_longonly_lb336_fast168_z075_ret180_h120_tp750_fl0_xr175` | `state_distilled_external_risk_filter` | False | False | -0.0951% | -0.2313% | +1.4069% | +0.2826% | 4.969030 | 5 | `train_positive,val_positive,oos_mdd_beats_shadow` |
| 14 | `fresh_state_distilled_ext_both_lb336_fast168_z050_ret180_h120_tp600_fl0_xr200` | `state_distilled_external_risk_filter` | False | False | +1.8139% | +1.5551% | +1.3989% | +0.5327% | 3.794908 | 8 | `oos_mdd_beats_shadow` |
| 15 | `fresh_state_distilled_ext_longonly_lb336_fast168_z050_ret120_h120_tp600_fl0_xr200` | `state_distilled_external_risk_filter` | False | False | +1.8943% | -0.2390% | +1.3932% | +0.3021% | 4.750538 | 6 | `val_positive,oos_mdd_beats_shadow` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
