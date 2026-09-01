# Session handoff — 2026-04-11 article pipeline continuation checkpoint

## Current repo state

- Repo: `/home/hoky/Quants-agent/LuminaQuant`
- Branch status at checkpoint: `private-main...private/main [ahead 1]`
- HEAD during this wave: `e378dc3 Preserve article-pipeline research expansion and reboot handoff state`
- At checkpoint end: **no active LuminaQuant article-pipeline python/uv process**

## Hard operating constraints (keep these)

- Total session memory must stay **below 8GB**.
- **No parallel heavy runs.**
- Always run **one batch at a time**.
- Use **only** this runner for article work:
  - `scripts/research/run_article_pipeline_research_batches.py`
- Treat `train total_return == 0` together with `train trade_count == 0` as **no-trade train** and penalize heavily.
- If Alpha101 is slow again, stop it and go back to lighter families.

## Batches now considered completed (do not rerun)

Completed final-output batches now include:
- `batch_01~07`
- `batch_10`
- `batch_11`
- `batch_13`
- `batch_24`
- `batch_25`
- `batch_26`
- `batch_27`
- `batch_28`
- `batch_29`
- `batch_30`
- `batch_32`
- `batch_33`
- `batch_34`
- `batch_36`
- `batch_39`
- `batch_41`
- `batch_42`
- `batch_44`

Still incomplete / blocked:
- `batch_08` — Alpha101 1h too slow, aborted
- `batch_31` — 30m timeout before final artifact
- `batch_43` — 30m timeout before final artifact

## Best current article candidates

Still **no train/val/oos all-positive robust candidate**.

### Best real-trade post-resume result
1. `batch_28`
   - Candidate: `liquidity_shock_reversion_15m_thin_lo_64_0.015`
   - train: `+4.5655%` (`trade_count=150`)
   - val: `+4.9364%` (`trade_count=106`)
   - oos: `-1.7393%` (`trade_count=28`)
   - oos Sharpe: `-3.6103`
   - Verdict: **train+val positive with real trades, but OOS broke**

### Best no-trade-train near-misses
2. `batch_41`
   - Candidate: `regime_breakout_1h_trend_guarded_36_0.65`
   - train: `0.0%`, `trade_count=0`
   - val: `+0.6528%`
   - oos: `+6.1051%`
   - oos Sharpe: `2.4023`
   - Verdict: **best breakout near-miss, but train no-trade**

3. `batch_10`
   - Candidate: `lag_convergence_4h_metals_core_xauusdt_xagusdt_2_0.018`
   - train: `0.0%`, `trade_count=0`
   - val: `+14.9810%`
   - oos: `+1.5279%`
   - Verdict: **good val/oos, but train no-trade**

## Key findings from this wave

### Breakout lane
- `batch_44` completed successfully.
- Best candidate: `rolling_breakout_30m_guarded_ls_64_0.002`
- train: `0.0%` / `trade_count=0`
- val: `+21.4363%`
- oos: `-3.2764%`
- Verdict: **breakout follow-up did not beat batch_41; OOS still failed**

### Mean-reversion / event-driven lane
These recent probes were weak or dead:
- `batch_26`: dead zero
- `batch_27`: dead zero
- `batch_29`: train negative, OOS negative
- `batch_30`: val and OOS strongly negative
- `batch_32`: negative across train/val/oos
- `batch_33`: negative across train/val/oos
- `batch_34`: val and OOS negative
- `batch_36`: dead zero

Interpretation:
- The recent small reversion/event-driven batches did **not** produce a new robust candidate.
- `batch_28` is the only recent one that produced a meaningful train+val-positive result, but it still failed OOS.

### Timeout / infra pattern
- `batch_31` and `batch_43` both timed out before a final artifact.
- Both failed in the same general path: `parquet -> WAL auto_repair`.
- Important nuance: this is **not universal** to every mixed crypto+metal batch.
  - Mixed-universe batches `39`, `41`, `42`, `44`, `30`, `34` all completed under the 30m budget.
- So `31` and `43` are currently **infra-blocked**, but not automatically strategy-rejected.

## Canonical summary artifacts to read first

Read these before choosing the next batch:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.json`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.md`

## Recommended next TODO order

### Immediate next action
1. **Run `batch_37` first**
   - Rationale: remaining small reversion-family probe at 15m before deciding whether the small-reversion lane is fully exhausted.

### After that
2. If `batch_37` is weak, try `batch_38`
3. Keep `batch_31` / `batch_43` blocked unless you explicitly decide a longer dedicated timeout is worth it
4. Keep `batch_09` (Alpha101 4h) as fallback only

## Exact restart prompt for a new session

Use this prompt in a fresh Codex session:

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) git log --oneline -1 확인
3) docs/session_handoff_20260411_article_pipeline_continue.md 읽기
4) 현재 실행 중인 python/uv 프로세스가 없는지 확인

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential batch만 실행
- batch runner는 scripts/research/run_article_pipeline_research_batches.py만 사용
- completed batch_01~07, 10, 11, 13, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 39, 41, 42, 44는 재실행하지 말고 결과만 읽기
- batch_37부터 재개
- batch_08, 31, 43은 blocked/slow 이력 참고
- train total_return=0 이면서 train trade_count=0이면 no-trade train으로 간주해서 robust 후보에서 강하게 감점
- Alpha101가 느리면 중단하고 lighter batch 유지

목표:
- train/val/oos가 동시에 덜 깨지는 robust 후보 찾기
- 완료된 batch 결과를 요약하고 다음 batch 우선순위를 계속 업데이트
```

## Exact next command

```bash
cd /home/hoky/Quants-agent/LuminaQuant
export POLARS_MAX_THREADS=1
export LQ_BACKTEST_LOW_MEMORY=1
export LQ_AUTO_COLLECT_DB=0
export PYTHONUNBUFFERED=1
/usr/bin/time -v timeout --signal=INT 30m \
  uv run python scripts/research/run_article_pipeline_research_batches.py \
  --batch-ids batch_37 \
  --max-batches 1 \
  --stop-after-errors 1
```

## If batch_37 is weak, next command

```bash
/usr/bin/time -v timeout --signal=INT 30m \
  uv run python scripts/research/run_article_pipeline_research_batches.py \
  --batch-ids batch_38 \
  --max-batches 1 \
  --stop-after-errors 1
```
