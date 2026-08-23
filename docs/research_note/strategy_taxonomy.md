# Strategy taxonomy and research priorities

Observed repository facts and evaluation results are separated from hypotheses. The structured source is `strategy_evidence_index.json`; typed relations are in `strategy_relationships.json`; status semantics are in `evaluation_contract.md`.

## Binding current decision

G003 selection-v11 completed 24 candidate runs with 16 allowed exclusions and 20 active return panels across 14 suite-family labels. `crypto_turtle_20_10_atr_v1` was the only positive-quality survivor; six were required. No allocator portfolio was emitted and locked OOS was not launched. Order routing remained false. The post-activation 19/19 and 127/143 passes are two-day execution coverage, not promotion evidence.

The earlier 144-strategy snapshot and later 143-row registry smoke remain an explicit commit/scope mismatch. They are not silently normalized.

## Families

| Family | Implemented examples | Observed evidence | Main assumptions and risks | Falsifiable improvement or combination hypothesis |
|---|---|---|---|---|
| `benchmark` | Bitcoin buy/hold; cash/no-trade | Common diagnostic BTC buy/hold `+0.74%/+0.08%`; cadence unverified. | Single-symbol beta is not alpha; first-symbol benchmarks do not match multi-asset baskets. | Freeze equal-weight and initial-weight matched baskets for each portfolio-like strategy. |
| `breakout` | Rolling/Donchian/Turtle; compression and squeeze breakout | G003 Turtle: 28 observations, Sharpe `0.1604078053`, Calmar `0.8298548399`, hit `0.3571428571`, turnover `0.01376170885`; sole survivor, suite rejected. | Sparse trades, whipsaw, OHLC stop ordering, short/funding costs. | One preregistered equal-gross crash/correlation guard ablation on unchanged Turtle. |
| `cross_sectional` | raw/residual/echo momentum; near-high/low; downside beta, tail, coskewness, CLV | Common diagnostic echo `+0.29%/+0.16%`, Sharpe `0.81/1.65`; residual momentum `+0.05%/+0.10%`, Sharpe `0.48/2.87`. G003 residual momentum had zero active returns/trades. | Point-in-time membership, survivorship, small cross-section, BTC beta, clone multiplicity. | Residualize echo/anchoring against raw momentum and liquidity; test asset-age cohorts only with point-in-time membership. |
| `derivatives_directional_crowding` | funding momentum/harvest; OI/taker pressure; squeeze | G003 funding-sign Sharpe `-0.2885407186`, Calmar `-1.2287728802`, hit `0.3214`, turnover `0.1883`; not a survivor. | Funding direction is not cash-and-carry; release time, staleness, basis, liquidation and coverage dominate. | Causally lagged funding+OI+taker ablation only after complete feature coverage. |
| `ensemble_regime_router` | disagreement ensembles; bull/bear rotation; confirmed routers | Disagreement gate common full `+2.07%`, nested recent `-0.27%`; no promoted portfolio. | Router overfit, correlated leaves, t-1 observability, excess cash. | Compare a preregistered router only after leaf survival against a static equal-risk blend. |
| `event_alpha` | abnormal continuation; panic rebound; liquidation reversal | Two-day abnormal-return smoke lost about `70.7%`; this is short smoke evidence, not a rank. Liquidation features were unavailable. | Timestamp look-ahead, tail concentration, gaps, turnover, post-event liquidity. | Predeclare event buckets and continuation/reversal arms before observing outcomes. |
| `formulaic_alpha` | Alpha101; alpha-zoo hybrids | Alpha101 was resource/feature constrained and breached equity in a two-day smoke; no admissible promotion evidence. | Formula/horizon multiplicity, shared baskets, non-finite metrics, opaque state. | Freeze a small formula family and count every formula/horizon/stop as a trial. |
| `intermarket` | lead-lag; Kalman/pair/PCA stat-arb; metals rotation | G003 Kalman Sharpe: ETH/BTC `-3.6105`, NVDA/AMD `-4.7378`, QQQ/SPY `-7.2713`; other rows excluded for coverage. | Synchronization, stale leaders, hedge-ratio drift, stationarity, exchange hours, two-leg fills. | Test beta-diversifying relative value only after synchronized point-in-time coverage and leg costs. |
| `mean_reversion_relative_value` | IBS/VWAP/z-score; pair/PCA residual; RSI divergence | G003 IBS `-3.0369`, PCA residual `-7.7403`, RSI divergence `-28.8109`; none survived. Exact strict residual-reversion/off-session variants are genuine nulls. | Stationarity decay, synchronized execution, adverse selection, shorts, sign-flip mining. | Preregister a t-1 stationarity switch between residual momentum and reversion; never choose sign from target OOS. |
| `microstructure_intraday` | order-book/VPIN/taker; session scalp; previous-day box; VB noise | G003 session scalp `-16.6648`; previous-day box `-0.3797` crypto and `-12.7529` TradFi; VB `-0.8192/-3.9597`. | OHLCV is not BBO/order flow; latency, queue, impact, sessions and coverage dominate. | Collect causal BBO/depth/fills before testing one frozen exhaustion-versus-continuation arm. |
| `rebalancing_diversification` | rebalancing premium; HRP/ERC allocation | Common raw `+5.98%/+1.78%`, Sharpe `3.14/5.76`, suppressed because the identical-basket no-rebalance control is absent; G003 monthly had zero trades. | Universe drift, beta mistaken for premium, timing, weight mismatch. | Same assets/weights/dates/costs: no-rebalance, periodic equal-weight, threshold-band, diversity weighting. |
| `seasonality` | calendar, intraday/overnight, off-session | Calendar-primary premise rejected. State-distilled successor observed OOS `2.4722%`, return/MDD `0.9761`, below base references `6.4281%` and `6.9169`. | Calendar mining, timezone drift, partial folds, changing microstructure. | Decompose dates into observable state; use shifted calendars as placebos. |
| `trend_momentum` | MA/composite/multiframe trend; price-volume; TSMOM | Price-volume common diagnostic `+1.07%/+0.59%`, Sharpe `4.69/4.63`, 10/9 trades; unsealed and nested. G003 top-cap TSMOM did not survive. | Momentum crash, correlated clones, overlay truncation, sparse observations, funding. | Fresh sealed price-volume and echo validation, then incremental correlation to Turtle. |
| `volatility_risk_overlay` | vol-managed; crash scaling; correlation guard; kill switch | No independent alpha claim. G003 MA-score vol-target Sharpe `-2.6412`; kill-switch/Turtle had zero active return. | Equal-gross mismatch, estimator/leverage choice, same-window tuning, path dependence. | Accept a wrapper only if equal-gross MDD/Calmar improves without violating a frozen net-return floor. |

## Priorities

### P0 — repair evidence before searching

1. Reconcile 143 versus 144 by source commit and dedicated-runner scope.
2. Preserve the G003 suite rejection; do not launch its locked OOS retrospectively.
3. Close point-in-time history and causal feature-coverage gaps.
4. Install matched controls for rebalancing, overlays, routers, and ensembles.

### P1 — highest-value tests

1. Turtle versus one equal-gross crash/correlation guard.
2. Unchanged PriceVolumeCorrContinuation on a fresh non-overlapping sealed window.
3. Echo and residual momentum independently, then their incremental contribution to Turtle.
4. Matched rebalancing controls.
5. Frozen stationarity regime switch between residual momentum and reversion.

### P2 — combine only after independent survival

Breakout plus slower cross-sectional momentum; directional trend plus synchronized intermarket relative value; true microstructure exhaustion plus trend after causal depth/fill evidence; static leaves versus confirmed router.

### P3 — unexplored variants

Point-in-time asset-age cohorts; cross-family residualization against momentum/liquidity; bounded causal feature-age sensitivity; no-trade bands for fast strategies; calendar and round-number placebos; fold-level complementarity on aligned net returns.
