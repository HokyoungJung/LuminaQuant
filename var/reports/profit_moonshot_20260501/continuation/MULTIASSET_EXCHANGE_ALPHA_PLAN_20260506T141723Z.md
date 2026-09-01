# Profit Moonshot — Hyperliquid/Tickmill multiasset alpha expansion plan

Generated: `2026-05-06T14:17:23Z`
Base commit: `6656328bea21d5eaec74e86776c4a2c845c6ef92`
Primary handoff copy: `var/reports/profit_moonshot_20260501/continuation/MULTIASSET_EXCHANGE_ALPHA_PLAN_20260506T141723Z.md`

## 0. Stop condition / success definition

This plan is not a license to mark another weak shadow as success. Stop only when either:

1. A candidate has separated train/val/OOS raw-first or explicitly listing-aware live-equivalent evidence, OOS return `> +0.8284%`, OOS MDD `< 0.1778%`, OOS Sharpe `> 1.0`, liquidations `0`, RSS `< 8GB`, and no gross-exposure increase; or
2. All planned read-only/replay candidates fail and the failure reasons are saved in reports.

Current ground truth remains:

- Incumbent/OOS-return best: `profit_moonshot_hourly_shock_reversion_eth_12h_mode` — OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`.
- Sharpe/MDD shadow: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` — OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`.
- Previously tried metals direct trading is rejected: raw-first coverage and OOS failed; do not count it as success.

## 1. Requirements summary

Goal: test whether expanding data/execution context to Hyperliquid and Tickmill/MT5 can create genuinely useful multiasset alpha for profit moonshot without reintroducing vector-screen artifacts.

Hard requirements:

- Start every session from `private/main` and clean process state.
- Use the existing live-equivalent discipline: `ArtifactPortfolioModeStrategy` definitions, raw-first data, stateful one-position replay, fees/fills/cooldown/max-hold, then one full backtest mode at a time.
- Add external venues as **read-only feature sources first**. No direct Hyperliquid/Tickmill execution until read-only data, cost model, and replay evidence prove value.
- Keep Python as the public surface. Rust is acceptable only for exact, parity-tested hot loops such as raw ingestion/aggregation and replay primitives; do not replace stateful execution semantics with vector approximations.
- Record failed candidates and why they failed.
- Final execution session must Lore-commit and push with `git push private private-main:main` after verification.

## 2. Evidence anchors

### Repo anchors

- Existing exchange factory supports `binance_futures`/`binance_native`, `mt5`, and `polymarket`; there is no Hyperliquid driver yet (`src/lumina_quant/exchanges/__init__.py:6-25`).
- For read-only multiasset onboarding, the existing portable seam is the `external` data route: external path resolution/loaders (`src/lumina_quant/market_data.py:293-355`), live source routing to `ExternalWindowDataHandler` (`src/lumina_quant/live/_source_routing.py:6-28`; `src/lumina_quant/live/data_external.py:19-49`), and backtest `--data-source external` / `--external-data-root` (`src/lumina_quant/cli/backtest.py:954-977`). Prefer this before adding live execution drivers.
- MT5 already has a bridge-capable exchange class and WSL/Linux bridge path (`src/lumina_quant/exchanges/mt5_exchange.py:37-64`) plus `fetch_ohlcv` and order surfaces (`src/lumina_quant/exchanges/mt5_exchange.py:219-352`). Use it for Tickmill read-only first; do not enable direct orders in the first phase.
- Config validation currently allows live market data sources `committed`, `binance_futures`, `external`, and `polymarket_live`, and exchange drivers `binance_futures`, `binance_native`, `mt5`, and `polymarket` (`src/lumina_quant/configuration/validate.py:103-128`). A new Hyperliquid driver requires schema/validation work; an external read-only feed does not.
- The MT5 worker returns OHLCV from `copy_rates_from_pos` with timestamp/open/high/low/close/tick-volume rows (`scripts/mt5_bridge_worker.py:88-119`). Tickmill spread/swap/session modeling still needs explicit capture; OHLCV alone is insufficient.
- Feature columns already include funding, mark/index price, OI, taker buy/sell volume, and liquidation fields (`src/lumina_quant/market_data.py:38-54`; `src/lumina_quant/data/feature_points.py:15-31`). Hyperliquid should map into this schema where semantics match, with exchange-specific provenance.
- `FeaturePointLookup` is timestamp-safe and uses latest/sum-before-or-at semantics, which is suitable for completed-feature confirmation filters (`src/lumina_quant/data/feature_points.py:41-184`).
- The hourly shock strategy already enforces completed bars (`src/lumina_quant/strategies/hourly_shock_reversion.py:205-213`), entry hour exclusions, realized-vol filter, BTC/SOL-style regime counterguard, and taker-flow confirmation hooks (`src/lumina_quant/strategies/hourly_shock_reversion.py:406-479`).
- The ETH shock family includes the incumbent, funding guard, taker-flow guard, funding+taker-flow guard, and SOL-regime guard modes (`src/lumina_quant/strategies/artifact_portfolio_mode.py:1198-1313`). Treat recent rejected variants as evidence, not wins.
- Live-equivalent revalidation split windows are train `2025-01-01..2025-12-31`, val `2026-01-01..2026-02-28`, OOS `2026-03-01..latest complete day` (`scripts/research/revalidate_live_equivalent_candidates.py:143-151`).
- Alpha gate logic rejects candidates not exceeding `+0.8284%` OOS return, not below funding-guard MDD, Sharpe not above `1.0`, zero OOS trades, or any liquidation (`scripts/research/revalidate_live_equivalent_candidates.py:520-531`).
- Full live-equivalent replay loads raw-first data and runs the event-driven chunked backtest path (`scripts/research/revalidate_live_equivalent_candidates.py:628-702`).
- Cadence sweep already covered frequent-to-slow rebalancing intervals without increasing exposure (`scripts/research/run_profit_moonshot_cadence_sweep.py:1-7`, `scripts/research/run_profit_moonshot_cadence_sweep.py:61-66`). Do not redo broad cadence grids unless a new liquidation-free train pre-gate survives.

### External documentation anchors checked on 2026-05-06

- Hyperliquid official docs: perps `/info` endpoint exposes metadata (`meta`), asset contexts with mark/current funding/OI (`metaAndAssetCtxs`), funding history, predicted funding, and related perp context endpoints: <https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals>.
- Hyperliquid official fees: base perp tier currently lists taker `0.045%` and maker `0.015%`; use current official fee docs at run time, not hardcoded assumptions: <https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees>.
- Tickmill official instruments page: broad CFD coverage across FX, commodities, indices, stocks/ETFs, plus MT4/MT5 availability: <https://www.tickmill.com/trading-instruments/>.
- Tickmill official spreads/swaps page: spread, swap, trading-hour, MT4/MT5 terminal-property, and Wednesday triple-swap details must be modeled before direct trading evidence: <https://www.tickmill.com/conditions/spreads-swaps>.

## 3. Architecture decision record

Decision: expand multiasset research in four strict phases: **Hyperliquid read-only → Tickmill/MT5 read-only → stateful replay → one-mode live-equivalent raw-first backtest**.

Drivers:

1. The current best live-equivalent alpha is still ETH shock reversion; external venues should improve regime/confirmation quality before they add tradeable legs.
2. Hyperliquid can add crypto-perp funding/OI/mark/context diversity with low integration friction but limited macro diversification.
3. Tickmill/MT5 can add true macro/risk diversification via FX/indices/metals/commodities, but its CFD spread/swap/session semantics make direct trading unsafe until modeled.
4. The last major failure mode was screen-to-engine decay; replay must precede full backtests.

Alternatives considered:

- Directly trade Hyperliquid/Tickmill legs immediately: rejected because raw-first coverage, fee/spread/swap/fill, and session semantics are not proven.
- Continue adding indicators to Binance-only ETH shock: rejected as first step because the user asked to evaluate venue/multiasset expansion and because feature-source breadth may be more useful than another local indicator.
- Rewrite the whole backend in Rust first: rejected because correctness depends on stateful event semantics; Rust should target exact hot-loop components only after parity tests.

Consequences:

- First useful artifact may be a filter/regime feature, not a new execution venue.
- Hyperliquid and Tickmill data can be valuable even if neither becomes a live execution driver.
- The plan prioritizes falsifiability: every phase has a pre-gate and failure report.

## 4. Phased execution plan

### Phase A — session hygiene and coverage inventory

1. Pull/update base:
   - `git fetch private main`
   - `git checkout private-main`
   - `git reset --hard private/main` only if local branch has no unpushed intended work.
2. Confirm no duplicate heavy processes:
   - `pgrep -af 'backtest|revalidate_live_equivalent|run_profit_moonshot|collect|refresh'`
   - Kill only stale processes that are clearly from old sessions; record any action in the report.
3. Read current handoffs/reports:
   - `var/reports/profit_moonshot_20260501/continuation/latest.md`
   - `var/reports/profit_moonshot_20260501/current_tail_20260506/useful_alpha_execution_report_20260506.md`
   - this plan file.
4. Inventory raw-first coverage for ETH/BTC/SOL feature points and current metals. Do not run any full backtest until coverage says which features can be replayed.

Acceptance for Phase A:

- A coverage JSON/MD is saved under `var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion/`.
- The report states which symbols/features are usable for train/val/OOS, which are listing-aware/non-standard, and which are blocked.

### Phase B — Hyperliquid read-only collector and normalizer

Implement read-only Hyperliquid support before any trading surface. Default to a user-managed external-feed contract first; add a full exchange driver only if execution/live-source integration is later justified:

1. Add a small client/collector for public endpoints only:
   - `metaAndAssetCtxs` for mark/current funding/OI context.
   - `fundingHistory` for historical funding.
   - optional candle/trade data only if official endpoint coverage is sufficient and rate limits are respected.
2. Normalize into the existing feature-point schema where exact:
   - `mark_price`, `funding_rate`, `open_interest`, `index_price` if available.
   - do not invent taker-flow/liquidation fields unless a source provides exact semantics.
3. Store under exchange provenance `hyperliquid`, separate from `binance`, so `FeaturePointLookup(exchange=...)` can compare sources without contamination.
4. Add unit tests for response parsing, null handling, timestamp ordering, and no-lookahead writes.
5. Create a coverage report for BTC/ETH/SOL first; expand only if those pass.

Initial Hyperliquid hypotheses to replay, in order:

- Funding divergence filter: block ETH shock entries when Hyperliquid funding disagrees with Binance funding or absolute funding is overheated.
- OI confirmation filter: accept shock fades only when Hyperliquid OI delta/z-score supports exhaustion rather than trend continuation.
- Cross-venue stress filter: block entries when Hyperliquid mark/index basis or OI cap/stress metadata suggests abnormal venue risk.

Acceptance for Phase B:

- Collector is read-only and test-covered.
- No full backtest is run from vector-only Hyperliquid signals.
- At least one stateful replay candidate survives train/val pre-gates before live-equivalent backtest consideration.

### Phase C — Tickmill/MT5 read-only macro filter lane

Use the existing MT5 bridge path and/or external-feed root first; Tickmill is a broker/provider configuration over MT5, not a new direct-execution alpha in phase C. Treat MT5 bridge output as raw input to the same external/canonical OHLCV path until spread/swap/session capture is tested.

1. Extend the MT5 bridge/read-only tooling to capture or snapshot:
   - OHLCV/tick-volume for FX majors, USD index proxy if available, XAUUSD/XAGUSD, major indices, oil/natural gas where available.
   - symbol properties needed for spreads, contract size, min lot, point/pip value, trading sessions, and swap long/short.
2. Normalize Tickmill/MT5 symbols to repo canonical symbols without mixing them with Binance/Hyperliquid symbols.
3. Build macro/risk filters first:
   - USD/rates proxy filter for ETH shock entries.
   - XAU/XAG risk-stress filter, but **not** direct metals trading unless raw coverage and OOS improve.
   - indices/oil risk-on/off filter for crypto shock entries.
4. Explicitly model spread and swap in replay before any direct Tickmill strategy is eligible.

Acceptance for Phase C:

- Tickmill/MT5 reports include spread/swap/session assumptions and provenance.
- Direct Tickmill trading remains blocked unless replay includes spread, swap, session gaps, and lot sizing.
- Macro filter replay improves train/val without starving trades before a full live-equivalent backtest.

### Phase D — stateful replay gate

Before creating any new portfolio mode, build/extend a fast replay harness that approximates engine constraints and writes split evidence:

- completed bars/features only; no lookahead.
- one position at a time.
- fee/spread/slippage/fill assumptions.
- cooldown and max-hold.
- stop/take-profit/trailing exit if used by the strategy.
- split-aware warmup; measure only train/val/OOS windows.
- output return/MDD/Sharpe/trades/turnover/liquidation/equity-breach and rejection reason.

Candidate acceptance for full backtest:

- train return positive or at least above current train floor, zero liquidation/equity breach.
- val return positive, Sharpe positive, trade count not starved.
- OOS replay is report-only but must not show obvious collapse.
- No candidate advances because of gross-exposure increase.

### Phase E — one-mode live-equivalent raw-first backtest

Only after Phase D survivor evidence:

1. Add exactly one shadow portfolio mode under `src/lumina_quant/strategies/artifact_portfolio_mode.py` with unchanged risk caps.
2. Add unit tests in `tests/unit/test_artifact_portfolio_mode.py` and strategy-specific tests if new filters are added.
3. Run exactly one full raw-first revalidation mode at a time through `scripts/research/revalidate_live_equivalent_candidates.py`.
4. Keep RSS under 8GB; if not, reduce chunk size/cache footprint rather than parallelizing full modes.
5. Promote nothing unless the strict gates pass. Sub-1 Sharpe remains shadow.

### Phase F — optimization lane, only when profiling identifies a bottleneck

Allowed optimizations:

- Batch/parquet scan reduction, chunk cache, prefrozen rows, feature lookup memoization.
- Rust hot loops for deterministic raw parse/aggregate/replay primitives if Python tests prove byte/metric parity.
- Keep the Python public API stable.

Disallowed shortcuts:

- Vector-only replacement of live-equivalent state.
- Rust rewrites that do not model fills, fees, funding, liquidation, cooldown, max-hold, and split carry.
- Increasing gross exposure to manufacture return.

## 5. First candidate queue

Run these sequentially; stop early if pre-gates fail.

1. `hl_funding_divergence_eth_12h_guard`
   - Base: incumbent ETH 12h shock.
   - Feature: Hyperliquid vs Binance funding sign/absolute cap.
   - Expected benefit: avoid funding-crowded shock fades.
2. `hl_oi_exhaustion_eth_12h_guard`
   - Base: incumbent ETH 12h shock.
   - Feature: Hyperliquid OI delta/z-score around shock bar.
   - Expected benefit: distinguish exhaustion from continuation.
3. `tickmill_macro_risk_eth_12h_guard`
   - Base: incumbent ETH 12h shock.
   - Feature: FX/metals/indices risk filter from MT5/Tickmill read-only data.
   - Expected benefit: avoid crypto shock entries during macro stress/trend continuation.
4. Combined survivor only:
   - Combine Hyperliquid and Tickmill filters only if each survives alone in replay and trade count is not starved.

## 6. Verification checklist

Before final commit/push in the execution session:

- `uv run ruff check <changed files>`
- `uv run python -m py_compile <changed python files>`
- targeted pytest for collectors, feature lookup, strategy filters, replay harness, and artifact modes.
- JSON sanity check: separated train/val/OOS metrics, liquidations, equity breach flags, RSS, failure reasons.
- Read the final report and ensure it does not label sub-1 Sharpe as success.
- Lore commit with `Constraint`, `Rejected`, `Confidence`, `Scope-risk`, `Directive`, `Tested`, and `Not-tested` trailers.
- `git push private private-main:main`.

## 7. Report artifacts to create during execution

Create/update these under `var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion/`:

- `coverage_inventory_latest.{json,md}`
- `hyperliquid_readonly_collection_latest.{json,md}`
- `tickmill_mt5_readonly_collection_latest.{json,md}`
- `stateful_replay_candidates_latest.{json,md,csv}`
- per-mode live-equivalent folders only for replay survivors
- `multiasset_exchange_alpha_execution_report_20260506.{json,md}`

Also update:

- `var/reports/profit_moonshot_20260501/continuation/latest.md`
- `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.md`

## 8. New-session command prompt

Paste this into the next session:

```text
cd /home/hoky/Quants-agent/LuminaQuant

private/main 최신 상태를 기준으로 시작해. 먼저 아래 multiasset exchange expansion plan을 읽고 그대로 이어서 진행해:
var/reports/profit_moonshot_20260501/continuation/MULTIASSET_EXCHANGE_ALPHA_PLAN_20260506T141723Z.md

목표는 profit moonshot에서 Hyperliquid와 Tickmill/MT5를 multiasset alpha 개선에 쓸 수 있는지 검증하는 것이다. 순서는 반드시 Hyperliquid read-only → Tickmill/MT5 read-only → stateful replay → replay survivor 1개씩 live-equivalent raw-first backtest다. 외부 venue는 처음엔 feature/regime source로만 쓰고, direct trading은 spread/swap/funding/fill/session/lot-size 모델과 raw-first evidence가 없으면 금지해라.

현재 incumbent는 `profit_moonshot_hourly_shock_reversion_eth_12h_mode`이며 OOS return `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`이다. Sharpe/MDD shadow는 `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`이며 OOS return `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`다. 성공은 OOS return `+0.8284%` 초과, MDD `0.1778%` 미만, Sharpe `>1.0`, liquidations `0`, train/val/OOS raw-first evidence 분리, RSS 8GB 미만일 때만 인정해라. sub-1 Sharpe는 shadow로만 저장하고 성공이라고 하지 마라.

먼저 중복 backtest/refresh/collector 프로세스를 확인하고 raw-first/feature coverage inventory를 만들어라. 그 다음 Hyperliquid funding/OI/mark context와 Tickmill MT5 FX/metals/indices macro filter를 기존 ETH 12h shock reversion에 confirmation/regime filter로 stateful replay하라. vector screen만 좋은 후보는 버리고, one-position/fee/spread/fill/cooldown/max-hold를 반영한 replay survivor만 한 번에 한 mode씩 live-equivalent raw-first backtest해라. 병목이 발견되면 정확성 parity test를 먼저 만들고 Python-facing API를 유지한 채 chunk/cache/Rust hot-loop 최적화를 진행해도 된다.

실패 후보도 왜 실패했는지 보고서에 저장하고, 최종 상태를 Lore commit으로 커밋 후 `git push private private-main:main`까지 완료해라.
```
