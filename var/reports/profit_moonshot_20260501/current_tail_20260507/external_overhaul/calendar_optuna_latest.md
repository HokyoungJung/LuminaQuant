# Profit moonshot calendar Optuna tuning

- Generated: `2026-05-07T13:58:28.742454Z`
- Trials: `64`
- Success candidates: `5`
- Peak RSS: `252.613 MiB`
- Objective policy: `train_val_only_locked_oos_report`

| rank | trial | name | success | objective | train | val | locked OOS | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | `optuna_calendar_trx_sethusdt_t0_thr150_h120_ls620_ss100_tp180` | True | 6.401653 | +2.2558% | +0.6824% | +1.0074% | 0.1532% | 7.9499 | 15 | `` |
| 2 | 48 | `optuna_calendar_trx_sethusdt_t48_thr180_h120_ls640_ss120_tp180` | True | 7.449943 | +2.6664% | +0.9401% | +0.9766% | 0.1581% | 7.1866 | 14 | `` |
| 3 | 1 | `optuna_calendar_trx_sweakest_t1_thr120_h120_ls540_ss120_tp450` | True | 18.283503 | +0.6978% | +3.0370% | +0.8871% | 0.1776% | 5.6177 | 10 | `` |
| 4 | 40 | `optuna_calendar_trx_sweakest_t40_thr150_h120_ls540_ss120_tp240` | True | 10.571003 | +3.3924% | +1.3859% | +0.8353% | 0.1645% | 6.5893 | 13 | `` |
| 5 | 38 | `optuna_calendar_trx_sweakest_t38_thr180_h168_ls540_ss80_tp600` | True | 11.376954 | +1.6480% | +1.3071% | +0.8318% | 0.1361% | 5.1431 | 6 | `` |
| 6 | 4 | `optuna_calendar_trx_sethusdt_t4_thr120_h120_ls630_ss100_tp600` | False | 17.265050 | +2.2551% | +2.1260% | +1.0982% | 0.2357% | 5.5932 | 9 | `oos_mdd_beats_shadow` |
| 7 | 8 | `optuna_calendar_trx_sethusdt_t8_thr120_h120_ls600_ss100_tp600` | False | 17.258868 | +2.2416% | +2.1260% | +1.0456% | 0.2245% | 5.5930 | 9 | `oos_mdd_beats_shadow` |
| 8 | 30 | `optuna_calendar_trx_sweakest_t30_thr180_h120_ls580_ss120_tp180` | False | -76.387586 | +2.8374% | -0.0720% | +0.8847% | 0.1433% | 7.1866 | 14 | `val_positive` |
| 9 | 29 | `optuna_calendar_trx_sweakest_t29_thr150_h120_ls540_ss80_tp180` | False | -76.875048 | +1.6781% | -0.1290% | +0.8768% | 0.1334% | 7.9498 | 15 | `val_positive` |
| 10 | 47 | `optuna_calendar_trx_sweakest_t47_thr120_h168_ls610_ss80_tp300` | False | 8.859977 | +1.7597% | +0.9576% | +0.8079% | 0.1879% | 4.9968 | 10 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 11 | 59 | `optuna_calendar_trx_sethusdt_t59_thr120_h120_ls500_ss120_tp240` | False | 11.454347 | +2.5005% | +1.5182% | +0.7668% | 0.1526% | 6.5169 | 13 | `oos_return_beats_incumbent` |
| 12 | 39 | `optuna_calendar_trx_sethusdt_t39_thr100_h144_ls509_ss100_tp450` | False | 10.196038 | +3.2694% | +1.2685% | +0.7631% | 0.1212% | 5.2608 | 10 | `oos_return_beats_incumbent` |
| 13 | 2 | `optuna_calendar_trx_sethusdt_t2_thr100_h96_ls640_ss80_tp600` | False | 16.655489 | +1.8175% | +1.7959% | +0.7550% | 0.2339% | 3.8095 | 12 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 14 | 34 | `optuna_calendar_trx_sweakest_t34_thr100_h120_ls500_ss120_tp450` | False | 18.322637 | +0.7394% | +3.0370% | +0.7466% | 0.1644% | 5.1001 | 10 | `oos_return_beats_incumbent` |
| 15 | 49 | `optuna_calendar_trx_sweakest_t49_thr100_h120_ls500_ss120_tp450` | False | 18.322637 | +0.7394% | +3.0370% | +0.7466% | 0.1644% | 5.1001 | 10 | `oos_return_beats_incumbent` |
| 16 | 28 | `optuna_calendar_trx_sweakest_t28_thr180_h168_ls520_ss120_tp350` | False | 6.198238 | +1.7165% | +1.0961% | +0.7312% | 0.1648% | 5.0870 | 9 | `oos_return_beats_incumbent` |
| 17 | 6 | `optuna_calendar_trx_sethusdt_t6_thr150_h168_ls570_ss80_tp600` | False | 16.902555 | +3.1609% | +1.6982% | +0.6901% | 0.2575% | 3.9032 | 7 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 18 | 7 | `optuna_calendar_trx_sethusdt_t7_thr100_h96_ls620_ss100_tp240` | False | 2.910716 | +1.5727% | +0.4930% | +0.6855% | 0.2037% | 4.2482 | 14 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 19 | 5 | `optuna_calendar_trx_sweakest_t5_thr200_h168_ls600_ss100_tp300` | False | 9.223021 | +2.2229% | +1.1585% | +0.6793% | 0.1816% | 4.3897 | 9 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 20 | 36 | `optuna_calendar_trx_sweakest_t36_thr150_h96_ls530_ss100_tp300` | False | 6.580927 | +1.7578% | +0.9337% | +0.6739% | 0.1305% | 4.8941 | 12 | `oos_return_beats_incumbent` |
