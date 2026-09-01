# Profit Moonshot Useful Alpha Execution Report — 2026-05-06

Generated: `2026-05-06T10:50:00Z`; updated: `2026-05-06T13:26:37Z`
Decision: **no new successful alpha promoted**.

## Stop condition / gate

A mode is successful only if train/val/OOS raw-first evidence is separated, OOS return exceeds `+0.8284%`, OOS MDD improves on the funding-guard shadow `0.1778%`, and OOS Sharpe is above `1.0`. Sub-1 Sharpe candidates are shadow/rejected only.

Current incumbent references remain unchanged:

- OOS-return best: `profit_moonshot_hourly_shock_reversion_eth_12h_mode` — OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`.
- Sharpe/MDD shadow: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` — OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`.
- XAU/XAG/XPT/XPD direct alpha remains rejected: raw-first train/val/OOS coverage is unavailable and legacy OOS was negative.

## Process / resource controls

- Duplicate process check before execution: no existing `revalidate_live_equivalent_candidates`, `backtest`, `refresh`, or `replay_eth_shock_filters` process was running.
- Full raw-first backtests were run one mode at a time.
- Replay peak RSS: `6689.238 MiB` (< 8192 MiB).
- Full backtest peak RSS: taker-flow guard `681172 KB`; SOL regime guard `784208 KB`; funding+taker-flow guard `1979660 KB` (< 8192 MiB).

## Raw-first coverage

Default current-tail OOS through `2026-05-05` was incomplete for ETH/BTC/SOL by one day, so full backtests used an explicit complete OOS end date: `2026-05-04`.

- `ETH/USDT`: train 365/365 complete; val 59/59 complete; oos 65/66 missing 2026-05-05
- `BTC/USDT`: train 365/365 complete; val 59/59 complete; oos 65/66 missing 2026-05-05
- `SOL/USDT`: train 365/365 complete; val 59/59 complete; oos 65/66 missing 2026-05-05
- `XAU/USDT`: train 0/365 missing 2025-01-01,2025-01-02,2025-01-03; val 0/59 missing 2026-01-01,2026-01-02,2026-01-03; oos 0/66 missing 2026-03-01,2026-03-02,2026-03-03
- `XAG/USDT`: train 0/365 missing 2025-01-01,2025-01-02,2025-01-03; val 0/59 missing 2026-01-01,2026-01-02,2026-01-03; oos 0/66 missing 2026-03-01,2026-03-02,2026-03-03
- `XPT/USDT`: train 0/365 missing 2025-01-01,2025-01-02,2025-01-03; val 0/59 missing 2026-01-01,2026-01-02,2026-01-03; oos 0/66 missing 2026-03-01,2026-03-02,2026-03-03
- `XPD/USDT`: train 0/365 missing 2025-01-01,2025-01-02,2025-01-03; val 0/59 missing 2026-01-01,2026-01-02,2026-01-03; oos 0/66 missing 2026-03-01,2026-03-02,2026-03-03

## Feature support inventory

Funding/OI/taker-flow are present for BTC/ETH/SOL; liquidation confirmation is not usable because all inspected symbols have `0` liquidation rows. Taker-flow tails end at `2026-05-03T00:00:00+00:00`, so late OOS taker confirmations are not assumed.

- `BTCUSDT`: funding `2707`, OI `16755`, taker `2101668` ending `2026-05-03T00:00:00+00:00`, liquidation `0`.
- `ETHUSDT`: funding `1468`, OI `17043`, taker `2102583` ending `2026-05-03T00:00:00+00:00`, liquidation `0`.
- `SOLUSDT`: funding `1468`, OI `17043`, taker `2099436` ending `2026-05-03T00:00:00+00:00`, liquidation `0`.
- `XAUUSDT`: funding `523`, OI `8000`, taker `0` ending `None`, liquidation `0`.
- `XAGUSDT`: funding `361`, OI `8201`, taker `0` ending `None`, liquidation `0`.
- `XPTUSDT`: funding `220`, OI `2191`, taker `0` ending `None`, liquidation `0`.
- `XPDUSDT`: funding `220`, OI `2176`, taker `0` ending `None`, liquidation `0`.

## Stateful replay screen

Replay artifact: `var/reports/profit_moonshot_20260501/current_tail_20260506/eth_shock_filter_replay/eth_shock_filter_replay_latest.json`.

- Evaluated `130` filter specs over ETH 12h shock reversion with one-position, fee, fill, cooldown, and max-hold realism.
- Replay-relative survivors: `8`.
- Absolute final-gate survivors: `0`. No replay candidate met the +0.8284% / MDD / Sharpe>1 success definition.

| replay survivor | train ret | val ret | OOS ret | OOS Sharpe | OOS MDD | filters |
|---|---:|---:|---:|---:|---:|---|
| `replay_base_taker_flow_1h_10pct` | +1.0395% | +0.1820% | +0.4329% | 1.022035 | 0.0879% | `{"cooldown_bars": 0, "excluded_hours": [], "flow_imbalance_min": 0.1, "flow_lookback_bars": 1, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.0, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": [], "regime_threshold": 0.0, "return_threshold": 0.01, "vol_lookback_bars": 0}` |
| `replay_base_sol_any_regime_350bp` | +0.1584% | +0.1933% | +0.3747% | 0.822076 | 0.1195% | `{"cooldown_bars": 0, "excluded_hours": [], "flow_imbalance_min": 0.0, "flow_lookback_bars": 0, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.0, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": ["SOL/USDT"], "regime_threshold": 0.035, "return_threshold": 0.01, "vol_lookback_bars": 0}` |
| `replay_funding_guard_taker_flow_1h_10pct` | +0.0627% | +0.2877% | +0.3408% | 0.808203 | 0.1087% | `{"cooldown_bars": 0, "excluded_hours": [0, 1, 8, 9, 16, 17], "flow_imbalance_min": 0.1, "flow_lookback_bars": 1, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.0, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": [], "regime_threshold": 0.0, "return_threshold": 0.008, "vol_lookback_bars": 0}` |
| `replay_base_btc_sol_regime_150bp_rv48_q70` | +0.6916% | +0.1014% | +0.3097% | 0.830850 | 0.0676% | `{"cooldown_bars": 0, "excluded_hours": [], "flow_imbalance_min": 0.0, "flow_lookback_bars": 0, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.007843252694217028, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": ["BTC/USDT", "SOL/USDT"], "regime_threshold": 0.015, "return_threshold": 0.01, "vol_lookback_bars": 48}` |
| `replay_base_btc_sol_any_regime_350bp` | +0.3467% | +0.1933% | +0.2832% | 0.628635 | 0.1550% | `{"cooldown_bars": 0, "excluded_hours": [], "flow_imbalance_min": 0.0, "flow_lookback_bars": 0, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.0, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": ["BTC/USDT", "SOL/USDT"], "regime_threshold": 0.035, "return_threshold": 0.01, "vol_lookback_bars": 0}` |
| `replay_base_btc_sol_regime_150bp_rv48_q55` | +0.8725% | +0.0493% | +0.2129% | 0.596526 | 0.0813% | `{"cooldown_bars": 0, "excluded_hours": [], "flow_imbalance_min": 0.0, "flow_lookback_bars": 0, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.006911474175370662, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": ["BTC/USDT", "SOL/USDT"], "regime_threshold": 0.015, "return_threshold": 0.01, "vol_lookback_bars": 48}` |
| `replay_base_sol_any_regime_150bp` | +0.6434% | +0.0123% | +0.2049% | 0.487939 | 0.1272% | `{"cooldown_bars": 0, "excluded_hours": [], "flow_imbalance_min": 0.0, "flow_lookback_bars": 0, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.0, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": ["SOL/USDT"], "regime_threshold": 0.015, "return_threshold": 0.01, "vol_lookback_bars": 0}` |
| `replay_funding_guard_sol_any_regime_350bp` | +0.2435% | +0.1471% | +0.1969% | 0.438235 | 0.1472% | `{"cooldown_bars": 0, "excluded_hours": [0, 1, 8, 9, 16, 17], "flow_imbalance_min": 0.0, "flow_lookback_bars": 0, "funding_abs_cap": 0.0, "funding_sign_guard": false, "liquidation_lookback_bars": 0, "liquidation_z_min": 0.0, "max_realized_vol": 0.0, "oi_mode": "", "oi_z_min": 0.0, "regime_policy": "any", "regime_symbols": ["SOL/USDT"], "regime_threshold": 0.035, "return_threshold": 0.008, "vol_lookback_bars": 0}` |

## Live-equivalent raw-first backtests

Only replay survivors were promoted to full engine tests, one mode at a time. Both failed the OOS useful-alpha gate.

| mode | train ret / Sharpe / MDD / trades | val ret / Sharpe / MDD / trades | OOS ret / Sharpe / MDD / trades | gate failure |
|---|---:|---:|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode` | +0.0255% / 0.000518 / 3.8315% / 236 | +0.8373% / 0.109701 / 0.2026% / 37 | +0.5871% / 0.070688 / 0.3203% / 35 | `oos_return_not_above_0.8284pct_incumbent;oos_mdd_not_below_funding_guard_shadow;oos_sharpe_not_above_1.0_success_target` |
| `profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode` | +1.3156% / 0.003544 / 4.5264% / 232 | +0.0905% / 0.002686 / 2.2562% / 39 | +0.3221% / 0.014160 / 0.9275% / 32 | `oos_return_not_above_0.8284pct_incumbent;oos_mdd_not_below_funding_guard_shadow;oos_sharpe_not_above_1.0_success_target` |
| `profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode` | -0.2259% / -0.003174 / 3.0258% / 228 | +0.7242% / 0.078468 / 0.3234% / 38 | +0.1759% / 0.043476 / 0.2204% / 34 | `train_total_return_not_positive;train_liquidation_observed;oos_return_not_above_0.8284pct_incumbent;oos_mdd_not_below_funding_guard_shadow;oos_sharpe_not_above_1.0_success_target` |

### Failure conclusions

- `profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode`: replay looked best, but live-equivalent OOS return fell to `+0.5871%`, MDD worsened to `0.3203%`, and Sharpe stayed `0.070688`; reject. The first engine attempt also exposed a portfolio-wrapper context bug that withheld `feature_lookup` from child strategies; that integration issue was fixed before the accepted evidence run.
- `profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode`: SOL 24h counter-regime filtering reduced OOS return to `+0.3221%`, MDD worsened to `0.9275%`, and Sharpe stayed `0.014160`; reject.
- `profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode`: selected after discarding the cadence bankruptcy artifact; full raw-first replay had negative train return (`-0.2259%`), one train liquidation, OOS `+0.1759%`, MDD `0.2204%`, and Sharpe `0.043476`; reject.
- Remaining replay survivors had lower replay OOS return than the already full-tested candidates; liquidation-specific confirmation variants are also untestable until liquidation rows exist. Do not spend another full backtest on them without a new replay edge.

## Code/report changes worth keeping

- `FeaturePointLookup.sum_between(...)` now supports raw, non-forward-filled feature sums for confirmation filters.
- `ArtifactPortfolioModeStrategy` now propagates child `required_features` and context/`feature_lookup`, preventing false zero-trade feature-filter runs.
- `HourlyShockReversionStrategy` supports cooldown state and taker-flow confirmation filters without changing existing default behavior.
- `revalidate_live_equivalent_candidates.py` supports explicit `--oos-end-date`, allows shadow artifact modes for targeted revalidation, and applies the useful-alpha OOS gate.
- Shadow-only artifact modes remain resolvable for audit/replay reproduction, but are **not** listed in `SUPPORTED_LIVE_PORTFOLIO_MODES`.

## Final decision

No new profit-moonshot alpha qualifies for success. Keep `profit_moonshot_hourly_shock_reversion_eth_12h_mode` as OOS-return best and `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` as Sharpe/MDD shadow. Do not promote metals, taker-flow guard, SOL regime guard, funding+taker-flow guard, or cadence-screen artifacts.

## Leverage / rebalancing cadence sweep addendum — 2026-05-06

User-requested leveraged-strategy cadence tuning was run across current profit-moonshot / reboot portfolio modes without increasing gross exposure, target allocation, or max-order caps.

- Cadence artifact: `var/reports/profit_moonshot_20260501/current_tail_20260506/cadence_sweep/profit_moonshot_cadence_sweep_latest.json`.
- Exact validation screen: `174` cadence variants across `28` eligible modes; grid included frequent `1` bar through low-frequency `1440` bars plus native cadences.
- Top one-day validation screen was dominated by aggressive `1b` adaptive-momentum variants, but the best screen survivor failed full train/val/OOS replay.
- Full live-equivalent raw-first replay: `profit_moonshot_adaptive_momentum_boost_mode__cadence_1b`.
  - train: report-capped `-100.0000%` return / MDD `100.0000%` because equity breached non-positive territory; raw arithmetic was `-111.5894%` return / `175.7880%` MDD; trades `2606` / liquidations `5`.
  - val: `+12.5743%` return / Sharpe `0.031136` / MDD `25.8855%` / trades `661` / liquidations `0`.
  - OOS: `-0.9619%` return / Sharpe `0.012010` / MDD `33.0023%` / trades `374` / liquidations `0`.
- Gate decision: **discard / reject / no promotion** — `train_total_return_not_positive;oos_return_not_above_0.8284pct_incumbent;oos_mdd_not_below_funding_guard_shadow;oos_sharpe_not_above_1.0_success_target;train_equity_breach_observed;train_liquidation_observed`. The `-111%`/`175%` figures were raw negative-equity arithmetic, not valid investment-performance metrics.
- Resource/optimization evidence: max RSS `4693.50 MiB`; checkpointed replay wall `1943.70s`; Rust raw-first backend resolved to `rust:/home/hoky/Quants-agent/LuminaQuant/native/rust_rawfirst/target/release/liblumina_rawfirst.so`; exact row cache and epoch-ms prefrozen rows preserve the same event-driven fill/fee/partial-fill path.
- Bottleneck conclusion: structural exactness cost remains in Python strategy/portfolio/event state, especially when leveraged positions stay open and liquidation/funding/active-order checks prevent safe skip-ahead. The safe hot-loop work completed here removes redundant raw-first load/freeze/refreeze work and keeps RSS under 8GB; it does **not** mark the failed cadence candidate as useful alpha.
