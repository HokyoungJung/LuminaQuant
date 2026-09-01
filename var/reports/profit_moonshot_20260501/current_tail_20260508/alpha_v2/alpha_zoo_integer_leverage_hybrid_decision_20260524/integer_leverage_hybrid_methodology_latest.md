# Integer-Leverage Hybrid Methodology

- Source: three paper/testnet profiles from the frozen integer-leverage artifact.
- PnL streams: reconstructed from the same fixed candidate position states and integer asset leverage maps.
- Cost: 10bps all-in round-trip backtest friction proxy is already embedded in each stream.
- Weight selection: train+validation only, 5% grid, each source profile weight >=10%, validation MDD target <=20%.
- locked-OOS: not used for discovery, objective, pruning, parameter fitting, or weight selection; report/gate only after freeze.
- Real money: blocked. Paper/testnet monitoring must record BBO spread, all-in fee/slippage, liquidation-inclusive MDD, account wipeout, and replay/live notional parity.
