# Session handoff — 2026-04-12 pair-spread robustness next session

## Current repo state

- Repo: `/home/hoky/Quants-agent/LuminaQuant`
- Branch baseline at handoff: `private-main...private/main [ahead 1]`
- Latest committed HEAD: `e378dc3 Preserve article-pipeline research expansion and reboot handoff state`
- Working tree includes uncommitted changes for:
  - `src/lumina_quant/storage/parquet/ohlcv_repo.py`
  - `src/lumina_quant/compute/ops.py`
  - `tests/test_wal_binary.py`
  - `tests/test_parquet_market_data.py`
  - `tests/test_compute_ops.py`
  - `scripts/research/build_pair_spread_robustness_followup.py`
  - handoff docs under `docs/`
- End-of-session process state: **no active LuminaQuant python/uv research process**

## What is already done

1. **WAL startup bottleneck fixed**
   - Read-only parquet/WAL loads now avoid eager repair on every open.
   - Large 1s WALs for the problematic symbols were compacted to monthly parquet.

2. **Alpha101 compute bottleneck fixed**
   - `rolling_rank_series` no longer uses pandas `rolling.apply` callback.
   - Replaced with vectorized/chunked NumPy implementation to stay OOM-safe.

3. **All article batches completed**
   - `batch_01~44` are all done.
   - Do **not** rerun the article batches.

4. **Pair-spread follow-up also completed**
   - Focused BNB/TRX 1h robustness follow-up finished.
   - Trade-count / midpoint singleton follow-up finished.

## Read these first in the new session

1. `docs/session_handoff_20260411_article_pipeline_complete_and_pair_followup.md`
2. `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.md`
3. `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_followup_current/pair_spread_robustness_followup_summary_latest.md`

## Best current candidates

### Best real-trade article candidate
- `batch_28`
- `liquidity_shock_reversion_15m_thin_lo_64_0.015`
- train `+4.5655%`, val `+4.9364%`, oos `-1.7393%`
- Not promotable because OOS broke.

### Best overall near-miss lane
BNB/TRX 1h pair-spread family:
- `batch_15` → `pair_spread_1h_exec_takeprofit_bnbusdt_trxusdt_2.6_0.70`
- `batch_14` → `pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70`
- `batch_18` → `pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.6_0.70`

These show **train/val/oos all positive**, but each hard-rejects on **PBO**.

## True remaining blocker

The blocker is **not runtime anymore**.
The blocker is **statistical robustness / overfitting control**.

Observed trade-off on BNB/TRX 1h pair-spread:
- profitable settings (`entry_z ~= 2.5~2.6`) keep OOS positive but PBO remains too high (`0.5~0.625`)
- lower-entry settings (`2.2`, `2.0`) reduce PBO, but train quality and/or OOS quality degrades too much

So the next session should focus on **robustness tightening**, not more broad batch brute-force.

## Constraints for the next session

- Total session memory must stay **below 8GB**.
- **No parallel heavy runs.**
- Always use **sequential execution** only.
- If you run new research, keep the same low-memory envelope:
  - `POLARS_MAX_THREADS=1`
  - `LQ_BACKTEST_LOW_MEMORY=1`
  - `LQ_AUTO_COLLECT_DB=0`
  - `PYTHONUNBUFFERED=1`
- Do **not** rerun `batch_01~44`.
- Keep treating `train total_return=0` with `train trade_count=0` as **no-trade train**, strongly penalized.

## Practical next priorities

### Priority 1 — Pair-spread robustness tightening
Work specifically on the BNB/TRX 1h lane.

Good directions:
- reduce parameter degrees of freedom
- add walk-forward / stability-style constraints
- add stronger candidate-family robustness priors
- examine whether the current PBO approximation is too brittle for sparse-trade but positive-split pair candidates, but only if you can justify it rigorously

### Priority 2 — Broader redesign
If robustness tightening still fails, move to:
- new candidate families, or
- redesigned acceptance / search logic with stronger robustness priors from the start

## Exact prompt for the next Codex session

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) git log --oneline -1 확인
3) docs/session_handoff_20260412_pair_spread_robustness_next.md 읽기
4) docs/session_handoff_20260411_article_pipeline_complete_and_pair_followup.md 읽기
5) var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_followup_current/pair_spread_robustness_followup_summary_latest.md 읽기
6) 현재 실행 중인 python/uv 프로세스가 없는지 확인

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential 실행만 허용
- article batch는 이미 전부 완료됨; batch_01~44는 재실행하지 말 것
- train total_return=0 이면서 train trade_count=0이면 no-trade train으로 간주해서 robust 후보에서 강하게 감점

현재 목표:
- BNB/TRX 1h pair-spread lane의 robustness(PBO) 리스크를 낮추는 방향으로 계속 진행
- 남은 문제는 runtime이 아니라 PBO/overfitting
- broad batch search가 아니라 pair-spread robustness tightening부터 진행
- 필요한 경우에만 새로운 focused follow-up manifest를 만들고, 같은 low-memory/sequential 원칙으로 검증
```
