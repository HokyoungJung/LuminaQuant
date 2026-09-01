# Profit moonshot calendar Optuna tuning

- Generated: `2026-05-08T11:08:33.459470Z`
- Trials: `24`
- Success candidates: `2`
- Peak RSS: `243.586 MiB`
- Objective policy: `train_val_only_locked_oos_report`

| rank | trial | name | success | objective | train | val | locked OOS | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | `optuna_calendar_trx_sethusdt_t0_thr150_h120_ls620_ss100_tp180` | True | 6.401653 | +2.2558% | +0.6824% | +1.0074% | 0.1532% | 7.9499 | 15 | `` |
| 2 | 1 | `optuna_calendar_trx_sweakest_t1_thr120_h120_ls540_ss120_tp450` | True | 18.283503 | +0.6978% | +3.0370% | +0.8871% | 0.1776% | 5.6177 | 10 | `` |
| 3 | 18 | `optuna_calendar_trx_sweakest_t18_thr120_h120_ls590_ss120_tp600` | False | 11.283508 | +1.6582% | +2.2603% | +1.0282% | 0.2207% | 5.5930 | 9 | `oos_mdd_beats_shadow` |
| 4 | 9 | `optuna_calendar_trx_sweakest_t9_thr120_h168_ls570_ss100_tp240` | False | -45.815007 | -0.2354% | +0.8389% | +0.9206% | 0.1725% | 6.9330 | 13 | `train_positive` |
| 5 | 14 | `optuna_calendar_trx_sweakest_t14_thr120_h120_ls500_ss120_tp180` | False | -0.292884 | +2.1709% | +0.1513% | +0.7576% | 0.1242% | 6.8493 | 17 | `oos_return_beats_incumbent` |
| 6 | 2 | `optuna_calendar_trx_sethusdt_t2_thr180_h144_ls620_ss120_tp600` | False | 12.631506 | +1.8326% | +2.2609% | +0.7521% | 0.1552% | 4.0909 | 7 | `oos_return_beats_incumbent` |
| 7 | 7 | `optuna_calendar_trx_sethusdt_t7_thr150_h96_ls560_ss80_tp300` | False | 3.560557 | +2.4204% | +0.3638% | +0.7122% | 0.1379% | 4.8942 | 12 | `oos_return_beats_incumbent` |
| 8 | 10 | `optuna_calendar_trx_sethusdt_t10_thr180_h144_ls600_ss80_tp300` | False | 14.630933 | +0.7763% | +1.4329% | +0.6013% | 0.2212% | 3.5899 | 10 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 9 | 16 | `optuna_calendar_trx_sweakest_t16_thr120_h120_ls540_ss80_tp350` | False | 3.570743 | +0.8308% | +0.5867% | +0.5984% | 0.2043% | 3.8526 | 11 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 10 | 19 | `optuna_calendar_trx_sweakest_t19_thr200_h168_ls550_ss80_tp450` | False | 10.146683 | +0.5753% | +1.2423% | +0.5919% | 0.1982% | 3.6859 | 7 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 11 | 12 | `optuna_calendar_trx_sweakest_t12_thr120_h120_ls520_ss120_tp350` | False | 2.655592 | +0.9048% | +0.8759% | +0.5762% | 0.1967% | 3.8525 | 11 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 12 | 5 | `optuna_calendar_trx_sethusdt_t5_thr150_h144_ls600_ss100_tp450` | False | 9.935039 | +3.1753% | +1.2485% | +0.5680% | 0.2932% | 3.1586 | 7 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 13 | 8 | `optuna_calendar_trx_sweakest_t8_thr200_h144_ls570_ss80_tp600` | False | -37.466797 | -0.0559% | +1.6456% | +0.5248% | 0.2468% | 3.1280 | 7 | `train_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 14 | 11 | `optuna_calendar_trx_sethusdt_t11_thr200_h96_ls640_ss80_tp600` | False | 10.416579 | +0.8326% | +1.2793% | +0.5199% | 0.2032% | 2.7501 | 9 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 15 | 4 | `optuna_calendar_trx_sweakest_t4_thr150_h96_ls560_ss80_tp600` | False | 13.165370 | +0.3337% | +1.6332% | +0.5114% | 0.2023% | 3.0448 | 10 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 16 | 20 | `optuna_calendar_trx_sweakest_t20_thr100_h120_ls590_ss120_tp240` | False | 11.715168 | +3.5031% | +1.5046% | +0.4969% | 0.2232% | 3.1602 | 12 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 17 | 15 | `optuna_calendar_trx_sweakest_t15_thr100_h120_ls540_ss120_tp240` | False | 11.676205 | +3.4499% | +1.5046% | +0.4548% | 0.2043% | 3.1600 | 12 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 18 | 3 | `optuna_calendar_trx_sweakest_t3_thr100_h168_ls570_ss100_tp450` | False | -41.190376 | -0.7715% | +1.5971% | +0.4492% | 0.2058% | 2.6727 | 8 | `train_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 19 | 6 | `optuna_calendar_trx_sweakest_t6_thr200_h120_ls560_ss80_tp600` | False | 17.576308 | +2.1690% | +1.9157% | +0.3615% | 0.2052% | 2.1069 | 8 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 20 | 13 | `optuna_calendar_trx_sweakest_t13_thr200_h120_ls530_ss120_tp450` | False | 15.681064 | +2.0157% | +2.5468% | +0.3285% | 0.1942% | 2.0401 | 8 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
