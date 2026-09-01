# Integer-Leverage Optuna Hybrid Methodology

- Optuna/TPESampler is the optimizer for the selected hybrid.
- Source: three paper/testnet integer-leverage profiles from the frozen corr/integer artifact.
- PnL streams: reconstructed from the same fixed position-state rules and integer asset leverage maps.
- Cost: 10bps all-in round-trip backtest friction proxy is embedded before hybriding.
- v3.5: warmup-learned default profile, rolling return/error weights, high-vol boost, max-weight cap, bias/exposure dampening.
- v3.6: v3.5 mechanics plus online adaptive default-profile refresh from rolling score evidence.
- Optuna space now covers every exposed HybridParams field, including warmup ratio, boost bounds/shape, high-vol quantile, and default-weight ratio candidate range.
- Standard live refit: tune/learn on train only, score on train+recent validation, then final-refit learned state on train+validation after selection.
- Real money: blocked. Paper/testnet only; monitoring must record BBO spread, all-in fee/slippage, liquidation-inclusive MDD, account wipeout, and replay/live notional parity.
