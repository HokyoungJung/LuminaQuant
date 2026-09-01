# Profit moonshot continuation — latest-data/source-backed alpha round 2

Generated: `2026-05-02T08:53:58.338635+00:00`

## Bottom line

- Latest data refresh completed to `2026-05-02T06:46:33Z` for BTC/ETH/BNB/SOL/TRX; peak RSS `3071.0 MiB`.
- Best raw validation return remains `profit_moonshot_adaptive_momentum_boost_mode` at `+0.5091%`, but it is not robust under this session gate (`train_return=-2.9948%`, `train_MDD=+18.0211%`).
- Best strict/preferred validation candidate (`val >= +0.4000%` and all robustness gates) is `profit_moonshot_adaptive_momentum_vol_target_132_mode`: val `+0.4176%`, train return `-2.1161%`, train MDD `+14.0900%`, val MDD `+1.1911%`, liquidations `0/0`.
- Best conservative-score minimum-gate passer is still the baseline `profit_moonshot_adaptive_momentum_mode` (score `10.6293`, val `+0.2649%`). Among improved-over-baseline candidates, the safer conservative choice is `profit_moonshot_adaptive_momentum_vol_target_mode` (score `10.5204`, val `+0.3964%`, train MDD `+13.9448%`); the stronger strict/preferred-return choice is `vol_target_132`.
- Classification: `vol_target_132` is a live-equivalent train/val-passed research candidate, not deployment-ready until OOS/current-tail live-equivalent feature coverage and shadow/live checks are added.

## Latest data refresh evidence

| symbol | before OHLCV max UTC | after OHLCV max UTC | archive raw rows | live raw rows | derived 1s rows |
|---|---:|---:|---:|---:|---:|
| `BTC/USDT` | `2026-04-23T11:41:18Z` | `2026-05-02T06:46:32Z` | 9677216 | 151770 | 759907 |
| `ETH/USDT` | `2026-04-23T11:41:18Z` | `2026-05-02T06:46:32Z` | 7978362 | 103460 | 759907 |
| `BNB/USDT` | `2026-04-23T11:41:18Z` | `2026-05-02T06:46:32Z` | 1438541 | 35438 | 759907 |
| `SOL/USDT` | `2026-04-23T11:41:18Z` | `2026-05-02T06:46:32Z` | 1965154 | 43573 | 759907 |
| `TRX/USDT` | `2026-04-23T11:41:18Z` | `2026-05-02T06:46:32Z` | 736074 | 18047 | 759908 |

## Live-equivalent mode results

| mode | train ret | train MDD | train Sharpe | train Sortino | trades train | val ret | val MDD | val Sharpe | val Sortino | trades val | max RSS MiB | score | minimum gate | strict +0.40% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `profit_moonshot_adaptive_momentum_boost_mode` | -2.9948% | +18.0211% | 0.007081 | 0.007648 | 361 | +0.5091% | +1.3583% | 0.014751 | 0.014527 | 56 | n/a | 10.057457 | FAIL | FAIL |
| `profit_moonshot_adaptive_momentum_vol_target_132_mode` | -2.1161% | +14.0900% | 0.004969 | 0.005250 | 361 | +0.4176% | +1.1911% | 0.013068 | 0.012661 | 54 | 5727.1 | 10.494910 | PASS | PASS |
| `profit_moonshot_adaptive_momentum_vol_target_mode` | -2.0229% | +13.9448% | 0.004873 | 0.005143 | 362 | +0.3964% | +1.1323% | 0.012988 | 0.012585 | 54 | 5688.5 | 10.520393 | PASS | FAIL |
| `profit_moonshot_adaptive_momentum_130_mode` | -2.4064% | +15.6763% | 0.006028 | 0.006435 | 360 | +0.3321% | +1.0622% | 0.011467 | 0.011165 | 53 | n/a | 10.191932 | FAIL | FAIL |
| `profit_moonshot_adaptive_momentum_120_mode` | -2.2339% | +14.4574% | 0.005466 | 0.005801 | 359 | +0.3292% | +0.9256% | 0.012557 | 0.012179 | 52 | n/a | 10.388915 | PASS | FAIL |
| `profit_moonshot_adaptive_momentum_mode` | -1.8628% | +12.0686% | 0.004374 | 0.004593 | 356 | +0.2649% | +0.7544% | 0.012417 | 0.012036 | 52 | n/a | 10.629285 | PASS | FAIL |
| `profit_moonshot_adaptive_momentum_governed_mode` | +1.9280% | +14.8471% | 0.004441 | 0.004619 | 278 | +0.1695% | +1.1606% | 0.006029 | 0.005898 | 51 | n/a | 11.185079 | FAIL | FAIL |
| `profit_moonshot_trend_mode` | -0.3549% | +1.0497% | -0.003730 | -0.001015 | 228 | +0.0090% | +0.0267% | 0.016720 | 0.003521 | 6 | n/a | 11.747074 | FAIL | FAIL |
| `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | -0.4211% | +1.0642% | -0.008484 | -0.001959 | 681 | +0.0000% | +0.0672% | 0.000025 | 0.000010 | 102 | 4511.6 | 11.558509 | FAIL | FAIL |
| `profit_moonshot_balanced_mode` | -0.2385% | +0.4264% | -0.007091 | -0.003482 | 777 | -0.0055% | +0.0428% | -0.010205 | -0.006049 | 112 | n/a | 11.576922 | FAIL | FAIL |
| `profit_moonshot_reversion_mode` | -0.4697% | +0.7352% | -0.011718 | -0.005017 | 567 | -0.0277% | +0.1294% | -0.016622 | -0.009552 | 106 | n/a | 11.354638 | FAIL | FAIL |
| `profit_moonshot_adaptive_momentum_volume_guard_mode` | -0.5108% | +1.1696% | -0.010376 | -0.002096 | 825 | -0.0485% | +0.0909% | -0.055409 | -0.014281 | 122 | 4552.2 | 10.997267 | FAIL | FAIL |
| `profit_reboot_compression_breakout_mode` | -0.1904% | +1.0968% | -0.003915 | -0.001345 | 144 | -1.0252% | +1.0400% | -0.114203 | -0.050488 | 38 | n/a | 8.615547 | FAIL | FAIL |
| `profit_moonshot_breakout_mode` | +0.0015% | +0.0269% | 0.001020 | 0.000093 | 6 | +0.0000% | +0.0000% | 0.000000 | 0.000000 | 0 | n/a | 11.799922 | FAIL | FAIL |

## External-source-backed idea ledger

### https://www.nber.org/papers/w22208
- Accessed: `2026-05-02`
- Core idea: Volatility-managed portfolios reduce risk when realized volatility is high; risk timing can improve Sharpe because expected returns do not rise one-for-one with volatility.
- Repo implementation/variant: Added volatility_target_per_bar, min/max_volatility_exposure_multiplier to AdaptiveRegimeMomentumStrategy; vol_target_132 keeps boost signal but scales target allocation by target/realized_vol, clipped at 0.55..1.0.
- Failure addressed: Targets boost fragility and October-2025 circuit-breaker drawdown without simply increasing or lowering fixed exposure.

### https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389
- Accessed: `2026-05-02`
- Core idea: Crypto time-series momentum can be strengthened with volume-weighted market returns.
- Repo implementation/variant: Implemented quote-volume history, volume-weighted broad regime, and volume-z entry gate; tested asym_dynamic and volume_guard modes.
- Failure addressed: Investigated whether volume-aware breadth could fix breakout/reversion/liquidity failures; live-equivalent val rejected the attempted variants.

### https://www.sciencedirect.com/science/article/pii/S1062940822000833
- Accessed: `2026-05-02`
- Core idea: Crypto intraday returns show both momentum and reversal; predictability changes around jumps and liquidity states.
- Repo implementation/variant: Kept live-equivalent minute decision cadence and avoided unconditional shock fade; tested volume/liquidity guarded variants rather than always-fade reversion.
- Failure addressed: Explains why previous reversion/shock-fade candidates lost in validation; filters were attempted and rejected when validation stayed weak.

### https://arxiv.org/abs/2602.11708
- Accessed: `2026-05-02`
- Core idea: Recent AdaptiveTrend paper combines high-frequency trend following with volatility-calibrated trailing stops and asymmetric long/short allocation.
- Repo implementation/variant: Implemented dynamic trailing and long/short multipliers in AdaptiveRegimeMomentumStrategy; live-equivalent asym_dynamic/volume_guard rejected these settings, while volatility sizing survived.
- Failure addressed: Documents why asymmetric/dynamic trailing was not promoted despite source support: it killed validation alpha.

### https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History
- Accessed: `2026-05-02`
- Core idea: Funding-rate history is available via Binance USDⓈ-M futures REST data and can support carry/crowding research.
- Repo implementation/variant: Not promoted in this patch because the live-equivalent strategy path currently validates OHLCV MARKET_WINDOW behavior; derivatives features remain next-family work.
- Failure addressed: Prevents claiming funding/open-interest alpha before the live-equivalent engine can consume those features consistently.

### https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics
- Accessed: `2026-05-02`
- Core idea: Open-interest statistics endpoint exposes futures crowding context but only latest-one-month history on that endpoint.
- Repo implementation/variant: Recorded as next-family data-engineering requirement, not as validated alpha.
- Failure addressed: Avoids mixing current-only derivatives endpoints into train/val without reproducible historical coverage.

## Failure handling

- `profit_moonshot_adaptive_momentum_asym_dynamic_mode`: External AdaptiveTrend-style asymmetry + dynamic trailing made train safe but over-constrained validation; val return was effectively flat, so not selectable.
- `profit_moonshot_adaptive_momentum_boost_mode`: Still best raw validation return, but fails this session user gate: train return is below -2.5% and train MDD exceeds 15%.
- `profit_moonshot_adaptive_momentum_governed_mode`: Broad-threshold/vol cap materially improved train return but cut validation alpha below baseline.
- `profit_moonshot_adaptive_momentum_volume_guard_mode`: Volume-z guard and long/short asymmetry removed too much profitable validation exposure and turned val negative.

## Next alpha family

- Do not promote derivatives/funding/OI claims until feature-point data is wired into the live-equivalent MARKET_WINDOW strategy path with historical train/val coverage.
- Next useful research branch: perp crowding carry/contrarian using funding + open interest + liquidation imbalance, but only after engine-level feature replay support is validated.
- If optimizing current candidate further, vary `volatility_target_per_bar` narrowly around `0.00125–0.00132`; do not use simple gross-exposure increases unless the volatility governor remains active and passes the train MDD gate.
