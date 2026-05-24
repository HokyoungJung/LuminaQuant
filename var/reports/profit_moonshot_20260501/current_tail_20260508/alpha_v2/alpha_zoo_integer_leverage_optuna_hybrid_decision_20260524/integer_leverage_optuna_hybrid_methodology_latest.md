# Integer-Leverage Optuna Hybrid Methodology

- Corrects the previous coarse-grid hybrid: Optuna/TPESampler is the optimizer for the selected hybrid.
- Source: three paper/testnet integer-leverage profiles from the frozen corr/integer artifact.
- PnL streams: reconstructed from the same fixed position-state rules and integer asset leverage maps.
- Cost: 10bps all-in round-trip backtest friction proxy is embedded before hybriding.
- v3.5: warmup-learned default profile, rolling return/error weights, high-vol boost, max-weight cap, bias/exposure dampening.
- v3.6: v3.5 mechanics plus online adaptive default-profile refresh from rolling score evidence.
- Optuna space mirrors run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py: bias alpha/combine, max weight, mape window, bias window, short-vol window.
- Objective/learning/selection: train+validation only. locked-OOS is not read for discovery, pruning, objective, fitting, or selection.
- Real money: blocked. Paper/testnet only; monitoring must record BBO spread, all-in fee/slippage, liquidation-inclusive MDD, account wipeout, and replay/live notional parity.
