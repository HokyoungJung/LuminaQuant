# Profit Moonshot — useful-alpha execution plan for next session

Generated: `2026-05-05T13:26:34Z`
Base commit at handoff start: `c09084c8f1d1d2e992aa3cab19754f959ef49c42`

## One-line diagnosis

The problem is no longer “find another indicator.” The bottleneck is **screen-to-engine decay**: vector screens can look good, but live-equivalent execution with raw-first coverage, one-position state, fees, partial fills, cooldowns, and train/val/OOS separation destroys most apparent edge. Useful alpha will come from improving the current ETH shock family with real feature filters, or from building a stateful feature-replay pipeline before spending full backtests.

## Current ground truth

- Current OOS-return best: `profit_moonshot_hourly_shock_reversion_eth_12h_mode`
  - OOS return `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`, liquidations `0`.
- Risk-adjusted shadow: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`
  - OOS return `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`, liquidations `0`.
- Conservative legacy candidate: `profit_moonshot_momentum_hybrid_safe_mode`; not deployment-ready.
- Latest metals attempt: `profit_moonshot_precious_metal_pair_aggressive_mode`
  - Implemented XAU/XAG + XPT/XPD mode, but rejected.
  - raw-first live-equivalent status: `blocked_missing_raw_first_market_data` for all four metals train/val.
  - legacy-windowed split: train `-0.0570%`, val `+0.1914%`, OOS `-0.0478%`, OOS Sharpe `-0.164066`.
  - Report: `var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive/precious_metal_pair_aggressive_report_20260505.md`.

## Definition of “useful alpha”

Do **not** mark success unless all gates below pass:

1. **Raw-first / live-equivalent**: same `ArtifactPortfolioModeStrategy` path used by live and backtest; standard train/val/OOS raw-first evidence present, or a clearly documented listing-aware gate for newly listed assets.
2. **OOS improvement**: must beat `profit_moonshot_hourly_shock_reversion_eth_12h_mode` on OOS return (`+0.8284%`) **and** improve risk-adjusted quality versus the funding-guard shadow (`Sharpe 0.111225`, MDD `0.1778%`).
3. **Sharpe objective**: user explicitly rejected sub-1 Sharpe. Treat `Sharpe > 1.0` as the success target. Intermediate candidates with lower Sharpe can be saved as shadows only, not wins.
4. **Risk**: liquidations `0`; no naked gross-exposure bump; max RSS <8GB; one full mode backtest at a time.
5. **Evidence split**: train / val / OOS metrics must be recorded separately, including failures and why they failed.

## Highest-probability path to a useful alpha

### Path A — improve the current ETH shock reversion alpha first

Reason: it is the only family currently proving positive OOS in the live-equivalent engine. Do not abandon it for unvalidated screens.

Implement a **filtered ETH shock reversion family**, not larger size:

- Base: `profit_moonshot_hourly_shock_reversion_eth_12h_mode` / funding-guard variant.
- Add entry filters one at a time, then combine only survivors:
  1. funding settlement guard already exists; retune excluded hours and adjacent windows.
  2. realized-volatility cap / volatility regime filter.
  3. BTC and SOL regime counterguard: block ETH longs during broad downtrend and shorts during broad uptrend.
  4. true taker-flow feature confirmation: use completed raw feature points only, no overlapping-event screen.
  5. OI/funding confirmation: funding sign/absolute cap + OI delta z-score.
  6. liquidation exhaustion confirmation: only fade shocks when opposite-side liquidation spike supports exhaustion.
  7. time-of-day/session filter: UTC sessions that historically survive OOS, not just train/val.

Acceptance target for the first useful variant:

- train > 0, val > 0, OOS > `+0.8284%`.
- OOS Sharpe > `0.25` for an interim shadow; OOS Sharpe > `1.0` for success.
- OOS MDD <= `0.1778%` if claiming risk-adjusted improvement, or <= `0.2819%` if claiming return-first improvement.
- liquidations `0`; trade count not starved.

### Path B — build stateful feature replay before any more feature alpha full backtests

Do not run full live-equivalent backtests on vector-screen winners until a fast stateful replay says they survive.

Build a replay tool that approximates engine constraints:

- completed-bar only; no lookahead.
- one position at a time per component.
- fees and slippage.
- cooldown and max-hold.
- stop / take-profit / trailing exit.
- optional current-volume guard and min notional.
- split-aware warmup: warm from prior history, measure only split window.
- output train/val/OOS return, MDD, Sharpe, trades, turnover, and reason for rejection.

Use it for:

- ETH shock filters above.
- derivatives feature alpha: taker-flow, OI, funding, liquidation exhaustion.
- metals only after raw-first coverage is solved.

### Path C — use metals as regime filters before trading metals directly

The XAU/XAG/XPT/XPD trading mode failed because coverage and fills are weak. A more useful near-term metals use is **macro/risk filter**, not direct metals legs:

- XAU/XAG ratio trend or z-score as “industrial risk-on/off” filter for crypto shock entries.
- XPT/XPD stress ratio as a risk/liquidity stress proxy; only use if coverage is fresh and aligned.
- Do not require metals raw-first 2025 train if they were not listed; if metals are used, make a listing-aware evidence gate and label it non-standard.

Only retry direct metals trading after:

- XAU/XAG/XPT/XPD raw aggtrades + materialized OHLCV are complete for the intended split.
- partial-fill/zero-volume bars are explicitly handled.
- XAU/XAG alone passes before adding XPT/XPD.

## Recommended next-session sequence

1. `git pull private main` / ensure `private-main` equals `private/main`.
2. Confirm no duplicate backtest/refresh processes.
3. Read this plan and the latest report files.
4. Run a quick raw-first coverage inventory for ETH/BTC/SOL derivatives features and metals.
5. Build/extend the stateful replay tool for ETH shock filters.
6. Screen **only ETH shock filter variants** first; choose max 2–3 modes for full live-equivalent backtest.
7. Run one mode at a time through `revalidate_live_equivalent_candidates.py`.
8. Promote nothing unless gates pass; save failure evidence if not.
9. Update summary/continuation report; Lore commit and `git push private private-main:main`.

## Concrete first candidate ideas

Start with narrow variants, not wide grids:

1. `profit_moonshot_hourly_shock_reversion_eth_12h_regime_guard_mode`
   - base ETH 12h shock.
   - exclude funding hours.
   - block longs if BTC 24h return < negative threshold.
   - block shorts if BTC 24h return > positive threshold.

2. `profit_moonshot_hourly_shock_reversion_eth_12h_vol_flow_guard_mode`
   - base ETH 12h shock.
   - funding hour guard.
   - realized vol cap.
   - taker-flow exhaustion confirmation from completed raw feature points.

3. `profit_moonshot_hourly_shock_reversion_eth_12h_liq_exhaustion_mode`
   - base ETH 12h shock.
   - only fade positive shocks when short-side/long-side liquidation pattern indicates exhaustion, and vice versa.
   - strict cooldown to avoid clustered overtrading.

## Files to inspect first in next session

- `var/reports/profit_moonshot_20260501/continuation/latest.md`
- `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive/precious_metal_pair_aggressive_report_20260505.md`
- `src/lumina_quant/strategies/hourly_shock_reversion.py`
- `src/lumina_quant/strategies/taker_flow_exhaustion_reversal.py`
- `src/lumina_quant/strategies/artifact_portfolio_mode.py`
- `scripts/research/revalidate_live_equivalent_candidates.py`
- `src/lumina_quant/market_data.py`

## New-session prompt to paste

```text
cd /home/hoky/Quants-agent/LuminaQuant

private/main 최신 상태를 기준으로 시작해. 먼저 아래 handoff/plan을 읽고 그대로 이어서 진행해:
var/reports/profit_moonshot_20260501/continuation/USEFUL_ALPHA_EXECUTION_PLAN_20260505T132634Z.md

목표는 profit moonshot에서 진짜 쓸모 있는 alpha를 만드는 것이다. 현재 OOS-return best는 `profit_moonshot_hourly_shock_reversion_eth_12h_mode`이고, Sharpe/MDD shadow는 `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`다. XAU/XAG/XPT/XPD 직접 거래 alpha는 구현됐지만 raw-first gate와 OOS에서 실패했으므로 성공으로 치지 마라.

이번 세션에서는 무작정 새 indicator를 추가하지 말고, 먼저 중복 backtest/refresh 프로세스를 확인하고 raw-first coverage를 점검해라. 그 다음 ETH 12h shock reversion alpha를 기준으로 funding-hour, BTC/SOL regime, realized-vol, taker-flow/OI/funding/liquidation feature confirmation filter를 stateful replay로 먼저 검증해라. vector screen만 좋은 후보는 버리고, one-position/fee/fill/cooldown/max-hold를 반영한 replay survivor만 한 번에 한 mode씩 live-equivalent raw-first backtest해라.

성공 기준: train/val/OOS raw-first evidence가 분리되어 있고, OOS return은 `+0.8284%`를 넘고, MDD/Sharpe는 funding-guard shadow보다 개선되어야 한다. 최종 성공은 Sharpe > 1.0을 목표로 하며, sub-1 Sharpe는 shadow로만 저장하고 성공이라고 하지 마라. 단순 gross exposure 증가는 금지하고, RSS 8GB 미만 및 한 번에 한 full backtest mode만 유지해라. 실패 후보도 왜 실패했는지 보고서에 저장하고, 최종 상태를 Lore commit으로 커밋 후 `git push private private-main:main`까지 완료해라.
```
