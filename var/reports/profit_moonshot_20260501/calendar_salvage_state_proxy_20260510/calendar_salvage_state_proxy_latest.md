# Profit moonshot calendar salvage via non-calendar state proxies — 2026-05-10

## Bottom line

- Calendar-primary 전략은 계속 **폐기**한다. 고정 월 규칙(Mar-Apr-May TRX long, Jan-Feb ETH short)은 live-causal signal이 아니다.
- 하지만 “아예 버리기 아쉬운” 성질은 일부 살렸다: **TRX state long-only proxy**가 calendar OOS 수익률과 유사한 OOS를 냈다.
- 최종 live 승격은 아직 금지: current baseline OOS return `+1.2181%`와 strict shadow MDD `0.1778%` 게이트를 넘지 못했고, 이 replay는 liquidation-aware final이 아니다.

## Invalid calendar references used as inspiration

| source | invalid rule | train ret/MDD/Sh | val ret/MDD/Sh | OOS ret/MDD/Sh | decision |
|---|---|---:|---:|---:|---|
| `var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion/calendar_optuna_latest.json` | long `TRXUSDT` in `[3, 4, 5]`, short `ETHUSDT` in `[1, 2]` | +2.69% / 1.59% / 1.26 | +0.23% / 1.08% / 0.43 | +1.01% / 0.08% / 9.14 | reject calendar-primary |
| `var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/calendar_optuna_latest.json` | long `TRXUSDT` in `[3, 4, 5]`, short `ETHUSDT` in `[1, 2]` | +2.26% / 0.98% / 1.30 | +0.68% / 1.25% / 1.40 | +1.01% / 0.15% / 7.95 | reject calendar-primary |

## Non-calendar proxy families added

- `state_momentum_proxy`: 현재 bar의 수익률 + residual z-score + optional taker-flow로 TRX long/ETH short를 결정한다. 월/일 캘린더 입력 없음.
- `fresh_state_trx_longonly_*`: calendar OOS가 주로 TRX long exposure에서 나온 점을 causal TRX momentum/residual 조건으로만 대체한다.
- `fresh_state_trx_dual_mom_*`: fast momentum에 slower regime confirmation을 추가해 train 안정성을 보강한다.
- `state_relative_strength_spread`: TRX/ETH spread return, residual gap, optional flow gap으로 two-leg spread를 결정한다.

## Combined replay evidence

- Combined artifact: `var/reports/profit_moonshot_20260501/calendar_salvage_state_proxy_20260510/combined_replay/fresh_start_overhaul_replay_latest.json`
- CSV: `var/reports/profit_moonshot_20260501/calendar_salvage_state_proxy_20260510/combined_replay/fresh_start_overhaul_replay_candidates.csv`
- Specs evaluated: `7092`; replay survivors: `0`; success candidates: `0`
- Peak RSS: `321.875 MiB` (< 8 GiB)
- Split windows: train `2025-01-01..2025-12-31`, val `2026-01-01..2026-02-28`, OOS `2026-03-01..2026-05-09`

## Family summary

| family bucket | specs | top OOS candidate | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | failed gates | best train+val-positive OOS |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `state_trx_dual_mom` | 1728 | `fresh_state_trx_dual_mom_fast72_reg168_z075_rz000_ret120_h168_ls620_ss60_tp450` | -0.22% | +0.30% | +0.69% | 0.67% | 2.35 | `train_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` | `fresh_state_trx_dual_mom_fast72_reg168_z100_rz075_ret120_h96_ls620_ss100_tp450` (+0.61%, fail `oos_return_beats_incumbent,oos_mdd_beats_shadow`) |
| `state_trx_eth_spread` | 1296 | `fresh_state_trx_eth_spread_lb168_thr180_zg050_h168_hr50_sc40_tp240` | -1.83% | -0.02% | +0.05% | 0.38% | 0.38 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` | none |
| `state_trx_longonly` | 1152 | `fresh_state_trx_longonly_lb168_z075_ret60_h168_ls800_tp600` | -1.30% | -0.24% | +1.25% | 0.24% | 5.92 | `train_positive,val_positive,oos_mdd_beats_shadow` | `fresh_state_trx_longonly_lb72_z050_ret60_h168_ls800_tp450` (+1.09%, fail `oos_return_beats_incumbent,oos_mdd_beats_shadow`) |
| `state_trx_mom_rotation` | 2916 | `fresh_state_trx_mom_lb168_z075_ret120_h96_ls620_ss100_tp240` | -1.03% | +0.75% | +1.33% | 0.44% | 4.87 | `train_positive,oos_mdd_beats_shadow` | `fresh_state_trx_mom_lb168_z100_ret120_h96_ls620_ss100_tp450` (+0.71%, fail `oos_return_beats_incumbent,oos_mdd_beats_shadow`) |

## Recommended salvage candidate (research-only)

`fresh_state_trx_longonly_lb72_z050_ret60_h168_ls800_tp450`

| split | return | MDD | Sharpe | Sortino | Volatility | round trips | liquidations |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | +0.50% | 1.26% | 0.28 | 0.26 | 1.85% | 35 | 0 |
| val | +0.20% | 0.61% | 1.15 | 0.88 | 1.08% | 4 | 0 |
| oos | +1.09% | 0.22% | 4.94 | 6.14 | 1.21% | 7 | 0 |

Why this is the best salvage rather than a live pick:

- It has no calendar/month/day condition and trades only when TRX 72h return and residual z-score are positive enough.
- It keeps train and validation positive and produces OOS `+1.09%`, similar to the invalid calendar best OOS around `+1.01%`.
- It still fails current-base improvement and strict MDD-vs-shadow gates; therefore it is **not deployable** without further liquidation-aware validation and improvement.

## Decision

- Keep `fresh_state_trx_longonly_lb72_z050_ret60_h168_ls800_tp450` as a **research salvage seed**.
- Do not promote any calendar-primary or state-proxy candidate to live yet.
- Next viable path: use this as one sleeve candidate in train/validation-only portfolio construction, then run liquidation-aware live-equivalent replay against current base.
