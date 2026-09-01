# Profit moonshot calendar Optuna tuning

- Generated: `2026-05-08T12:22:46.885347Z`
- Trials: `128`
- Success candidates: `8`
- Peak RSS: `242.758 MiB`
- Objective policy: `train_val_only_locked_oos_report`

| rank | trial | name | success | objective | train | val | locked OOS | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 47 | `optuna_calendar_trx_sethusdt_t47_thr150_h168_ls590_ss100_tp180` | True | 2.325917 | +2.6910% | +0.2345% | +1.0091% | 0.0804% | 9.1406 | 14 | `` |
| 2 | 0 | `optuna_calendar_trx_sethusdt_t0_thr150_h120_ls620_ss100_tp180` | True | 6.401653 | +2.2558% | +0.6824% | +1.0074% | 0.1532% | 7.9499 | 15 | `` |
| 3 | 39 | `optuna_calendar_trx_sethusdt_t39_thr150_h120_ls620_ss100_tp180` | True | 6.401653 | +2.2558% | +0.6824% | +1.0074% | 0.1532% | 7.9499 | 15 | `` |
| 4 | 85 | `optuna_calendar_trx_sethusdt_t85_thr180_h120_ls590_ss100_tp180` | True | 7.527503 | +2.2901% | +0.7848% | +0.9000% | 0.1458% | 7.1866 | 14 | `` |
| 5 | 55 | `optuna_calendar_trx_sethusdt_t55_thr150_h120_ls580_ss100_tp240` | True | 11.845920 | +1.9890% | +1.3364% | +0.8974% | 0.1766% | 6.5894 | 13 | `` |
| 6 | 80 | `optuna_calendar_trx_sethusdt_t80_thr180_h168_ls580_ss80_tp600` | True | 20.696408 | +3.1280% | +1.9512% | +0.8936% | 0.1462% | 5.1434 | 6 | `` |
| 7 | 1 | `optuna_calendar_trx_sweakest_t1_thr120_h120_ls540_ss120_tp450` | True | 18.283503 | +0.6978% | +3.0370% | +0.8871% | 0.1776% | 5.6177 | 10 | `` |
| 8 | 59 | `optuna_calendar_trx_sethusdt_t59_thr180_h120_ls570_ss80_tp180` | True | 7.603401 | +1.9535% | +0.6261% | +0.8694% | 0.1408% | 7.1866 | 14 | `` |
| 9 | 43 | `optuna_calendar_trx_sethusdt_t43_thr100_h120_ls610_ss100_tp600` | False | 17.578408 | +2.5804% | +2.1257% | +1.0507% | 0.2267% | 5.6713 | 9 | `oos_mdd_beats_shadow` |
| 10 | 94 | `optuna_calendar_trx_sethusdt_t94_thr120_h120_ls600_ss100_tp600` | False | 17.258868 | +2.2416% | +2.1260% | +1.0456% | 0.2245% | 5.5930 | 9 | `oos_mdd_beats_shadow` |
| 11 | 15 | `optuna_calendar_trx_sweakest_t15_thr120_h120_ls540_ss120_tp600` | False | 11.267769 | +1.6357% | +2.2603% | +0.9407% | 0.2021% | 5.5926 | 9 | `oos_mdd_beats_shadow` |
| 12 | 25 | `optuna_calendar_trx_sweakest_t25_thr120_h120_ls530_ss120_tp600` | False | 11.264572 | +1.6312% | +2.2603% | +0.9232% | 0.1983% | 5.5925 | 9 | `oos_mdd_beats_shadow` |
| 13 | 6 | `optuna_calendar_trx_sethusdt_t6_thr120_h144_ls580_ss120_tp180` | False | -76.421138 | +3.4173% | -0.1716% | +0.8972% | 0.1807% | 7.0904 | 16 | `val_positive,oos_mdd_beats_shadow` |
| 14 | 34 | `optuna_calendar_trx_sethusdt_t34_thr150_h120_ls640_ss100_tp600` | False | 22.305192 | +1.9348% | +2.8383% | +0.8313% | 0.2832% | 4.2207 | 9 | `oos_mdd_beats_shadow` |
| 15 | 42 | `optuna_calendar_trx_sethusdt_t42_thr150_h120_ls630_ss100_tp600` | False | 22.306478 | +1.9341% | +2.8383% | +0.8182% | 0.2788% | 4.2207 | 9 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 16 | 16 | `optuna_calendar_trx_sweakest_t16_thr120_h120_ls530_ss120_tp240` | False | 11.419660 | +3.1920% | +1.5046% | +0.8129% | 0.1617% | 6.5169 | 13 | `oos_return_beats_incumbent` |
| 17 | 116 | `optuna_calendar_trx_sethusdt_t116_thr180_h120_ls620_ss120_tp240` | False | 13.229644 | +2.4182% | +1.7276% | +0.8087% | 0.1877% | 5.2864 | 12 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 18 | 35 | `optuna_calendar_trx_sethusdt_t35_thr150_h120_ls620_ss100_tp600` | False | 22.307748 | +1.9334% | +2.8383% | +0.8052% | 0.2744% | 4.2206 | 9 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 19 | 31 | `optuna_calendar_trx_sethusdt_t31_thr150_h120_ls610_ss100_tp600` | False | 22.309001 | +1.9327% | +2.8383% | +0.7922% | 0.2700% | 4.2206 | 9 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 20 | 32 | `optuna_calendar_trx_sethusdt_t32_thr150_h120_ls610_ss100_tp600` | False | 22.309001 | +1.9327% | +2.8383% | +0.7922% | 0.2700% | 4.2206 | 9 | `oos_return_beats_incumbent,oos_mdd_beats_shadow` |
