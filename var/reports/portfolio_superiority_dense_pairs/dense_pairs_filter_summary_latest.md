# Portfolio Superiority Dense Pairs Filter Summary

- Generated: 2026-04-22T12:24:13.097527+00:00
- Input candidates: 30
- Kept candidates: 4

## Rules
- val total_return > 0
- val sharpe > 0
- oos total_return > 0
- oos sharpe > 0
- oos max_drawdown <= 15%
- val active_days >= 5
- oos active_days >= 5
- oos active_day_ratio >= 15%

## Kept candidates
- pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.2_0.55 | val_ret=0.0956 | val_sharpe=3.7317 | oos_ret=0.0126 | oos_sharpe=3.1892 | oos_mdd=0.0040 | oos_active=8/34
- pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | val_ret=0.0939 | val_sharpe=3.6691 | oos_ret=0.0012 | oos_sharpe=0.2376 | oos_mdd=0.0141 | oos_active=8/34
- pair_spread_1h_exec_takeprofit_bnbusdt_trxusdt_2.2_0.55 | val_ret=0.0939 | val_sharpe=3.6691 | oos_ret=0.0012 | oos_sharpe=0.2376 | oos_mdd=0.0141 | oos_active=8/34
- pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.2_0.55 | val_ret=0.0939 | val_sharpe=3.6691 | oos_ret=0.0012 | oos_sharpe=0.2376 | oos_mdd=0.0141 | oos_active=8/34

## Top 10 by daily OOS Sharpe
- pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70 | keep=False | val_ret=+10.0099% | val_sharpe=3.9878 | oos_ret=+3.2652% | oos_sharpe=5.9241 | oos_mdd=0.0648%
- pair_spread_1h_core_btcusdt_xagusdt_2.6_0.70 | keep=False | val_ret=-0.7197% | val_sharpe=-2.2878 | oos_ret=+5.7251% | oos_sharpe=4.7275 | oos_mdd=2.0305%
- pair_spread_1h_core_bnbusdt_xauusdt_2.2_0.55 | keep=False | val_ret=-1.8056% | val_sharpe=-2.4195 | oos_ret=+2.6860% | oos_sharpe=3.4808 | oos_mdd=1.0630%
- pair_spread_1h_state_vwap_bnbusdt_xauusdt_2.2_0.55 | keep=False | val_ret=-2.6299% | val_sharpe=-2.7427 | oos_ret=+2.6860% | oos_sharpe=3.4808 | oos_mdd=1.0630%
- pair_spread_1h_exec_takeprofit_bnbusdt_xauusdt_2.2_0.55 | keep=False | val_ret=-1.8056% | val_sharpe=-2.4195 | oos_ret=+2.6860% | oos_sharpe=3.4808 | oos_mdd=1.0630%
- pair_spread_1h_exec_tightstop_tp_bnbusdt_xauusdt_2.2_0.55 | keep=False | val_ret=-1.9119% | val_sharpe=-2.5202 | oos_ret=+2.6860% | oos_sharpe=3.4808 | oos_mdd=1.0630%
- pair_spread_1h_core_xauusdt_xagusdt_2.2_0.55 | keep=False | val_ret=+1.7126% | val_sharpe=1.8029 | oos_ret=+2.1750% | oos_sharpe=3.3608 | oos_mdd=0.5496%
- pair_spread_1h_state_vwap_xauusdt_xagusdt_2.2_0.55 | keep=False | val_ret=+1.7126% | val_sharpe=1.8029 | oos_ret=+2.1750% | oos_sharpe=3.3608 | oos_mdd=0.5496%
- pair_spread_1h_exec_takeprofit_xauusdt_xagusdt_2.2_0.55 | keep=False | val_ret=+1.7126% | val_sharpe=1.8029 | oos_ret=+2.1750% | oos_sharpe=3.3608 | oos_mdd=0.5496%
- pair_spread_1h_state_vwap_btcusdt_xagusdt_2.2_0.55 | keep=False | val_ret=-1.0049% | val_sharpe=-4.5746 | oos_ret=+2.3770% | oos_sharpe=3.2876 | oos_mdd=1.3339%
