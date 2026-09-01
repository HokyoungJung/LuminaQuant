# Session handoff — 2026-04-11 article pipeline post-timeout wave

## Current repo/process state

- Repo: `/home/hoky/Quants-agent/LuminaQuant`
- Branch status during this wave: `private-main...private/main [ahead 1]`
- HEAD during this wave: `e378dc3 Preserve article-pipeline research expansion and reboot handoff state`
- End-of-wave process state: **no active LuminaQuant python/uv research process**

## Hard constraints still in force

- Total session memory must stay **below 8GB**.
- **No parallel heavy runs.**
- Always run **one batch at a time**.
- Use **only** `scripts/research/run_article_pipeline_research_batches.py` for article batch work.
- Treat `train total_return == 0` with `train trade_count == 0` as **no-trade train** and penalize heavily.
- If Alpha101 is slow again, stop it and fall back rather than letting it monopolize the session.

## New work completed in this wave

### Startup checks completed
- `git status --short --branch` → `private-main...private/main [ahead 1]`
- `git log --oneline -1` → `e378dc3 Preserve article-pipeline research expansion and reboot handoff state`
- Read `docs/session_handoff_20260411_article_pipeline_continue.md`
- Verified there was **no active LuminaQuant article-pipeline python/uv process** before restarting work

### Newly attempted batches

1. `batch_37` — `VwapReversionStrategy` 15m, 2 candidates
   - Command ran under the standard low-memory 30m envelope
   - Result: **timeout / no final artifact**
   - Elapsed: `29:38.66`
   - Max RSS: `1,457,900 KiB`
   - `batch.log` ended in the same `parquet -> WAL` load path seen before, specifically `ohlcv_repo._load_wal_frame -> BinaryWAL.iter_range -> decode_record`
   - Interpretation: **infra-blocked**, not strategy-rejected

2. `batch_40` — `CompositeTrendStrategy` 30m, 6 candidates
   - Also run under the same standard low-memory 30m envelope
   - Result: **timeout / no final artifact**
   - Elapsed: `30:00.22`
   - Max RSS: `1,475,836 KiB`
   - `batch.log` again died in the same `parquet -> WAL` decode/load path
   - Interpretation: **infra-blocked**, not strategy-rejected

3. `batch_09` — `Alpha101FormulaStrategy` 4h, 2 candidates
   - Tried because it became the only non-sweep fallback outside the growing timeout set
   - Result: **manually stopped as too slow**
   - Elapsed at stop: `20:39.71`
   - Max RSS: `1,480,212 KiB`
   - `batch.log` still showed `BinaryWAL.repair -> scan_valid_length -> decode_record` inside parquet/WAL auto-repair when interrupted
   - Interpretation: even the Alpha101 fallback lane is currently **infra-constrained**, not just strategy-slow

## Canonical summary artifacts refreshed

Updated:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.json`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.md`

These now reflect:
- attempted incomplete batches = `08, 09, 31, 37, 40, 43`
- the new blocked/slow status for `batch_37`, `batch_40`, `batch_09`
- the updated next-priority queue

## Best candidates are still unchanged

Still **no train/val/oos all-positive robust candidate**.

Best current real-trade candidate remains:
- `batch_28` — `liquidity_shock_reversion_15m_thin_lo_64_0.015`
  - train `+4.5655%`, trades `150`
  - val `+4.9364%`, trades `106`
  - oos `-1.7393%`, trades `28`
  - Verdict: train+val positive, but OOS broke

Best no-trade-train near-miss still remains:
- `batch_41` — `regime_breakout_1h_trend_guarded_36_0.65`
  - train `0.0%`, trades `0`
  - val `+0.6528%`
  - oos `+6.1051%`
  - Verdict: good val/oos, but **train no-trade** so heavily penalized

## Updated interpretation after this wave

The problem is now less about "which strategy family is next" and more about **which remaining lanes can even start research inside the 30m sequential budget**.

Confirmed/par confirmed blocked under the current envelope:
- `batch_31`
- `batch_37`
- `batch_40`
- `batch_43`
- `batch_09` (manual slow abort, same underlying WAL-repair bottleneck)

High-risk next probes under the same envelope:
- `batch_35`
- `batch_38`

Reason:
- they are still-unrun smaller probes,
- but they remain in the same mixed-universe zone where the current bottleneck is `parquet -> WAL auto_repair / decode`,
- so they may consume time without ever reaching a final candidate report.

## Recommended next order from here

1. **Do not rerun completed final-output batches**
   - `batch_01~07, 10, 11, 13, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 39, 41, 42, 44`

2. If you insist on continuing article search **without infra work**, the only remaining small live probes are:
   - `batch_35`
   - `batch_38`
   but they are now **high-risk/no-result** candidates under the current envelope

3. Otherwise the more rational next step is:
   - **WAL-load infra fix or a consciously longer retry budget** for blocked batches

4. Only after that should you reconsider:
   - `batch_12` (low expected value after batch_11 collapse)
   - `batch_14~23` (large market-neutral sweeps; heavy)

## Exact restart prompt for next session

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) git log --oneline -1 확인
3) docs/session_handoff_20260411_article_pipeline_post_timeout_wave.md 읽기
4) 현재 실행 중인 python/uv 프로세스가 없는지 확인

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential batch만 실행
- batch runner는 scripts/research/run_article_pipeline_research_batches.py만 사용
- completed batch_01~07, 10, 11, 13, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 39, 41, 42, 44는 재실행하지 말고 결과만 읽기
- batch_31, 37, 40, 43은 현재 30m budget에서 parquet->WAL 경로 timeout으로 본다
- batch_09는 4h Alpha101 fallback이었지만 20m+ slow라 중단했다
- train total_return=0 이면서 train trade_count=0이면 no-trade train으로 간주해서 robust 후보에서 강하게 감점

목표:
- train/val/oos가 동시에 덜 깨지는 robust 후보 찾기
- 현재는 결과 요약 + 다음 우선순위 업데이트가 우선이며, 추가 live run은 infra 제약을 고려해서 신중히 선택
```
