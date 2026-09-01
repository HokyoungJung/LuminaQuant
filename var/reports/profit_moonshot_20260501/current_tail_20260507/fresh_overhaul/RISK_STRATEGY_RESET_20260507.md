# Profit moonshot risk + strategy reset (2026-05-07)

Generated: `2026-05-07T11:02:49.092856Z`

## 결론

- **promote 없음**: fresh-start 후보 `1219`개 중 replay survivor/success 모두 `0`.
- OOS 수익률 1위: `fresh_resid_rev_lb24_z175_h72_asia_us` / OOS `+0.2021%`, Sharpe `0.908931`, train `-0.5135%`, val `-0.0113%` → gate 실패.
- train+val 양수 중 OOS 1위: `fresh_resid_rev_rvcap3156_lb12_z175` / OOS `+0.1040%`, Sharpe `0.869259` → incumbent OOS `+0.8284%`와 Sharpe `>1` 미달.
- 기존 incumbent/context-wrapper를 더 비틀어 쓰는 방향은 중단. 기존 산출물은 **baseline/gate**로만 사용.

## 데이터/메모리 증거

- data refresh status: `completed`, cutoff: `2026-05-06T23:59:59Z`, source: `binance_raw_aggtrades`
- data refresh peak RSS: `4793.78515625` MiB, workers: `1`
- fresh replay peak RSS: `2547.137` MiB, specs: `1219`
- 수정 사항: fresh replay 로더가 1h materialized 47시간만 보던 문제를 고쳐, monthly/daily raw-first 1s를 1h로 chunk 집계한다.

## 리스크 정리

- **context_leakage** (high): Treat handoffs and old incumbent reports as baselines only; every new candidate must be generated from current raw-first data and self-contained manifests.
- **invalid_data_window** (high): The crashed fresh replay used only 47 joined 1h rows; loader is now fixed to aggregate full 1s monthly/daily raw-first coverage before scoring.
- **OOM_session_failure** (high): One heavy lane only, selected_workers=1, process RSS evidence: data refresh 4793.8 MiB, fresh replay 2547.1 MiB, both below 8 GiB.
- **alpha_weakness** (high): 1219 fresh candidates, including taker-flow families, produced zero promotion candidates; do not trade them. Broaden data/features before more tuning.
- **feature_coverage_asymmetry** (medium): Taker flow covers train/val/OOS for BTC/ETH/SOL, but open interest is mostly OOS; OI must not drive train/val selection until backfilled.
- **selection_leakage** (high): Validation remains primary for selection; OOS is report-only and cannot tune thresholds.
- **live_execution_gap** (medium): Replay enforces one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stops/takes/max-hold before full backtest.

## 새 전략 원칙

1. **Data-first reset**: refresh/inventory 결과 없이는 전략 생성 금지.
2. **Feature contract 먼저**: train/val/OOS 모두 있는 feature만 selection에 사용. OI처럼 OOS 위주 feature는 backfill 전 selection 금지.
3. **Small survivor funnel**: vector score → stateful replay → live-equivalent raw-first backtest 순서. survivor 없으면 full backtest 금지.
4. **No stale context**: 이전 handoff는 목표/게이트만 읽고, 후보 생성은 현재 data artifact에서 재시작.
5. **8GB hard discipline**: 한 번에 heavy process 하나, workers=1, `/usr/bin/time -v`와 RSS log를 산출물에 남김.

## 다음 실행 순서

1. Backfill open-interest history for train/val or remove OI from any selectable alpha path.
2. Stop adding indicator-only variants; first run feature predictive diagnostics by split for taker-flow, funding, volatility, and cross-sectional residuals.
3. If diagnostics show edge, generate a small candidate set and require train+val-positive before OOS report.
4. Only after a replay survivor exists, run one live-equivalent raw-first full backtest mode under the same 8 GiB guard.

## 재개 명령

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run python scripts/research/replay_profit_moonshot_fresh_start.py --oos-end-date 2026-05-06 --output-dir var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul
uv run pytest tests/test_profit_moonshot_fresh_start_replay.py
```
