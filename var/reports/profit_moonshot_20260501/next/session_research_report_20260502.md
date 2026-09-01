# Profit Moonshot Continuation — 2026-05-02

Generated: `2026-05-02T06:29:00Z`
Repo branch: `private-main`
Primary stop condition reached: **yes** — `profit_moonshot_adaptive_momentum_120_mode` beats baseline validation return with materially safer train metrics than boost.

## External source-backed ideas used

Accessed date for all sources: `2026-05-02`.

| source | URL | core idea | repo implementation | failure mode addressed |
|---|---|---|---|---|
| Volatility-managed portfolios, Moreira & Muir / NBER WP 22208 | https://www.nber.org/papers/w22208 | Scale or suppress exposure when realized volatility/risk is elevated; volatility timing can improve risk-adjusted performance. | `profit_moonshot_adaptive_momentum_governed_mode` adds realized-volatility cap (`max_realized_vol=0.0035`), breadth floor (`broad_threshold=0.0015`), and slower rebalance cadence to the adaptive momentum sleeve. | Addresses boost fragility and 2025 circuit-breaker sensitivity without increasing gross exposure. |
| Time-series momentum / crypto trend-following research | https://www.sciencedirect.com/science/article/pii/S1062940821000590 | Crypto trend/momentum can be useful, but regime and volatility controls are necessary because return distributions are unstable. | Same governed mode keeps the existing MARKET_WINDOW-compatible momentum signal but filters high-volatility / weak-breadth regimes. | Avoids treating raw momentum exposure as deployment-ready. |
| Pure Momentum in Cryptocurrency Markets, SSRN | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4138685 | 24/7 crypto markets can exhibit pure momentum effects; timing and horizons matter. | Robustness ladder tests the same adaptive momentum signal at 120/130 sizing and compares validation return versus train fragility. | Isolates signal quality from simple leverage increases. |
| Binance USDⓈ-M Futures funding-rate API docs | https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History | Funding-rate history is available as derivatives positioning/carry support data. | `PerpCrowdingCarryStrategy` was upgraded with MARKET_WINDOW support, explicit sizing metadata, and evaluation cadence. | Explores funding/crowding carry as a future alpha family rather than only price momentum. |
| Binance USDⓈ-M Futures open-interest statistics docs | https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics | Open interest is a derivatives crowding proxy; local inventory showed OI coverage starts too late for train/val. | Funding/OI sleeve was not promoted as a portfolio mode after live-equivalent smoke produced repeated zero-limit pending fills before a checkpoint. | Prevents claiming success from a support-data alpha that cannot pass the live-equivalent gate. |
| Binance force-order / liquidation docs | https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/All-Force-Orders | Liquidations/force orders can proxy forced-flow shocks but endpoint coverage can be sparse. | Local support inventory currently has zero liquidation rows for the train/val major-symbol set; liquidation alpha is deferred. | Avoids overfitting a shock-fade idea without live-equivalent support data. |
| Reconciling Open Interest with Traded Volume in Perpetual Swaps, arXiv | https://arxiv.org/abs/2310.14973 | Perp OI/liquidation data quality can vary across exchanges; open interest may be misreported or delayed. | Report treats derivatives crowding as research-only until execution/min-notional and data-quality gates are fixed. | Explains why raw OI/liquidation indicators were not accepted as success. |

## Live-equivalent candidates tested this session

Gate reference:
- baseline `profit_moonshot_adaptive_momentum_mode` val return: `+0.264933%`
- boost `profit_moonshot_adaptive_momentum_boost_mode` val return: `+0.509082%`, train return `-2.994796%`, train MDD `18.021085%`

| mode | role | train return | train MDD | train trades | val return | val MDD | val Sharpe / Sortino | val trades | liq train/val | max RSS | outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `profit_moonshot_adaptive_momentum_120_mode` | robustness ladder | `-2.233917%` | `14.457370%` | `359` | `+0.329164%` | `0.925556%` | `0.012557 / 0.012179` | `52` | `0/0` | `4,567,432 KB` | **Best robust session candidate**: beats baseline and satisfies strict train return/MDD/RSS gates; safer than boost. |
| `profit_moonshot_adaptive_momentum_130_mode` | robustness ladder | `-2.406386%` | `15.676295%` | `360` | `+0.332099%` | `1.062192%` | `0.011467 / 0.011165` | `53` | `0/0` | `5,787,492 KB` | Beats baseline but breaches strict train MDD `<15%`; research-only. |
| `profit_moonshot_adaptive_momentum_governed_mode` | source-backed volatility/breadth governor | `+1.927971%` | `14.847094%` | `278` | `+0.169461%` | `1.160625%` | `0.006029 / 0.005898` | `51` | `0/0` | `5,799,048 KB` | Improves train return/MDD dramatically, but fails baseline validation-return target; research-only. |

`profit_moonshot_adaptive_momentum_140_mode` was implemented but not run: 130 already breached the strict train-MDD gate, and 140 would be a pure exposure step. Compute was instead allocated to source-backed governor validation to avoid an exposure-only session.

## Failed-candidate diagnosis retained

| candidate/family | observed failure | remediation attempted or recommended |
|---|---|---|
| `profit_moonshot_balanced_mode` | validation return/Sharpe/Sortino negative | Do not blend weak breakout/reversion legs until each leg has positive live-equivalent validation evidence. |
| `profit_moonshot_reversion_mode` | shock fade loses on validation | Only revisit with forced-flow/liquidation or volatility-regime filters once support-data coverage and execution gating are reliable. |
| `profit_reboot_compression_breakout_mode` | validation `-1.0252%`, large loss | Redesign with volatility expansion confirmation/trailing exit; not rerun this session because robustness ladder already found a safer candidate. |
| `profit_moonshot_breakout_mode` | validation trades `0` | Breakout needs looser cadence/threshold or different Donchian/range definition before live-equivalent retest. |
| funding/OI/liquidation carry sleeve | initial portfolio-mode smoke produced repeated zero-limit pending fills before a train checkpoint | Strategy support was made MARKET_WINDOW-compatible, but the mode was not retained as live-selectable. Next step: fix execution min-notional/pending remainder behavior or build a vector prefilter before another full live-equivalent run. |

## Decisions

- **Best by conservative summary score:** existing `profit_moonshot_adaptive_momentum_mode` remains top because the summary scoring heavily rewards lower validation MDD.
- **Best by validation return:** existing `profit_moonshot_adaptive_momentum_boost_mode` remains highest at `+0.509082%`, but it is still **research-candidate only** due train return/MDD fragility.
- **Best robust live-equivalent session candidate:** `profit_moonshot_adaptive_momentum_120_mode` — beats baseline, passes strict train/MDD/RSS gates, and is safer than boost.
- **Deployment readiness:** no automatic deployment. `120_mode` is a safer research candidate requiring review; boost is not deployment-ready; governed mode and 130 are research-only.

## Next alpha family

Prioritize a non-exposure-only breakout/reversion redesign:
1. Donchian/range breakout with volatility expansion and volume/liquidity confirmation to fix 0-trade breakout.
2. Liquidation/forced-flow reversal only after support-data coverage and zero-limit pending-fill behavior are fixed.
3. Compression breakout with dynamic trailing stop and avoid-trade filter around volatility spikes.
