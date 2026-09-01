# Session handoff — 2026-04-09 article pipeline reboot checkpoint

## Current repo state

- Repo: `/home/hoky/Quants-agent/LuminaQuant`
- Branch status at checkpoint: `private-main...private/main [ahead 1]`
- HEAD at session start: `e378dc3 Preserve article-pipeline research expansion and reboot handoff state`

## Hard operating constraints (keep these)

- Total session memory must stay **below 8GB**.
- **No parallel heavy runs.**
- Always run **one batch at a time**.
- Use **only** this runner for article work:
  - `scripts/research/run_article_pipeline_research_batches.py`
- Do **not** rerun completed `batch_01~07`; only read their outputs.
- Alpha101 (`batch_08`, `batch_09`) is allowed only as a secondary follow-up. If it is too slow, abort and switch back to lighter families.
- Treat `train total_return == 0` together with `train trade_count == 0`, `turnover == 0`, `exposure == 0` as **no-trade in train**, not as a meaningful flat result.

## Completed batches already on disk before this session

Completed with final outputs:
- `batch_01` ~ `batch_07`

Key conclusion from completed `batch_01~07`:
- No candidate had train/val/oos all positive.
- Best earlier near-miss remained:
  - `batch_03`
  - `carry_trend_factor_rotation_4h_carry_guarded_ls_16_4_0.150`
  - train `+4.7230%`, val `+3.3663%`, oos `-12.0380%`

## Work completed in this session

### 1) batch_08 (Alpha101 1h) — aborted as too slow

- Command family: `batch_08`
- Outcome: **aborted manually**
- Elapsed: `31:02`
- Max RSS: `1,470,044 KiB`
- Final report artifact: **none**
- Log:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/batch_runs/batch_08/batch.log`
- Relevant observation:
  - Stalled deep in Alpha101 rolling rank path (`compute/ops.py` rolling apply rank)
- Decision:
  - Deprioritize Alpha101 and move to lighter batches.

### 2) batch_10 — completed

- Elapsed: `5:39`
- Max RSS: `1,335,104 KiB`
- Output dir:
  - `.../article_inspired_research_current/batch_runs/batch_10`
- Best candidate:
  - `lag_convergence_4h_metals_core_xauusdt_xagusdt_2_0.018`
- Metrics:
  - train `0.0000%`
  - val `+14.9810%`
  - oos `+1.5279%`
  - oos Sharpe `0.6199`
  - oos maxDD `3.4909%`
- Important interpretation:
  - `train=0` here means **no trades in train** (`trade_count=0`, `turnover=0`, `exposure=0`).
- Verdict:
  - One of the two best current near-misses.

### 3) batch_11 — completed

- Elapsed: `28:46`
- Max RSS: `1,464,936 KiB`
- Family:
  - `leadlag_spillover_15m`
- Verdict:
  - Strongly negative across splits; treat this family as currently unattractive.
- Example top row:
  - train `-81.6415%`, val `-47.6029%`, oos `-37.6054%`
- Decision:
  - Deprioritize `batch_12` (same family, 5m variant).

### 4) batch_13 — completed

- Elapsed: `17:57`
- Max RSS: `1,449,572 KiB`
- Output dir:
  - `.../batch_runs/batch_13`
- Best candidate:
  - `pair_spread_15m_core_bnbusdt_trxusdt_2.6_0.70`
- Metrics:
  - train `-1.6612%`
  - val `-1.2716%`
  - oos `-1.6559%`
- Verdict:
  - Negative across splits; this small 15m market-neutral probe did not justify escalating to large pair-spread sweeps.

### 5) batch_25 — completed

- Elapsed: `23:33`
- Max RSS: `1,465,824 KiB`
- Candidate:
  - `basis_snapback_reversion_30m_balanced_ls_96_1.8`
- Metrics:
  - train `0`, val `0`, oos `0`
  - all trade/exposure fields also `0`
- Verdict:
  - Completely dead / no-trade candidate.

### 6) batch_39 — completed

- Elapsed: `29:02`
- Max RSS: `1,665,708 KiB`
- Candidate:
  - `composite_trend_stable_1h_stable_lo_core_lo_0.60_0.75_0.25_0.80`
- Metrics:
  - train `0.0000%`
  - val `-0.2835%`
  - oos `-0.5663%`
- Verdict:
  - Slightly negative; not compelling.

### 7) batch_41 — completed

- Elapsed: `28:59`
- Max RSS: `1,473,368 KiB`
- Output dir:
  - `.../batch_runs/batch_41`
- Best candidate:
  - `regime_breakout_1h_trend_guarded_36_0.65`
- Metrics:
  - train `0.0000%`
  - val `+0.6528%`
  - oos `+6.1051%`
  - oos Sharpe `2.4023`
  - oos maxDD `4.0460%`
- Important interpretation:
  - `train=0` again means **no trades in train**.
- Verdict:
  - The best recent trend-family near-miss. Not robust enough yet, but materially better than most prior article batches.

### 8) batch_42 — completed

- Elapsed: `29:01`
- Max RSS from batch.log: `1,446,608 KiB`
- Output dir:
  - `.../batch_runs/batch_42`
- Best candidate:
  - `regime_breakout_30m_trend_guarded_48_0.68`
- Metrics:
  - train `0.0000%`
  - val `-1.8112%`
  - oos `+2.3966%`
  - oos Sharpe `0.9068`
  - oos maxDD `4.2281%`
- Verdict:
  - OOS positive, but weaker than `batch_41` because validation is negative.

### 9) batch_43 — interrupted for reboot handoff

- Batch: `RollingBreakoutStrategy`, `1h`, `2` candidates
- Current status: **aborted intentionally for reboot handoff**
- Elapsed before interrupt: `17:31`
- Max RSS: `1,464,612 KiB`
- Final report artifact: **none saved yet**
- Existing files:
  - `.../batch_runs/batch_43/candidate_manifest.json`
  - `.../batch_runs/batch_43/batch.log`
- Log shows interruption during WAL/parquet load path:
  - stack reached `src/lumina_quant/storage/wal/binary.py` / `decode_record` / `crc32`
- This is **not a failure verdict** on the strategy; it is simply unfinished and should be rerun cleanly in the next session.

## Current overall ranking (article candidates only)

Still no train/val/oos all-positive robust candidate.

Current best near-misses:

1. `batch_10`
   - `lag_convergence_4h_metals_core_xauusdt_xagusdt_2_0.018`
   - train `0.0`, val `+14.98%`, oos `+1.53%`

2. `batch_41`
   - `regime_breakout_1h_trend_guarded_36_0.65`
   - train `0.0`, val `+0.65%`, oos `+6.11%`, Sharpe `2.40`

Important common issue:
- These are **val/oos-positive but train-no-trade** candidates.
- Going forward, treat `train trade_count == 0` as a major downgrade / near-disqualifier.

## Recommended next TODO order

### Immediate next action
1. **Rerun `batch_43` first**
   - because it is already half-probed, small (2 candidates), same promising breakout neighborhood as `batch_41`, and has no final artifact yet.

### After that
2. Re-rank `batch_41`, `batch_42`, `batch_43` together
3. Then choose between:
   - `batch_24` (small 4h market-neutral stability probe)
   - `batch_09` (Alpha101 4h, only if you accept slower runtime)

### Still deprioritized
- `batch_12` — leadlag family looked very bad in `batch_11`
- `batch_14+` large market-neutral sweeps — too heavy before smaller probes justify them
- `batch_08` Alpha101 1h — too slow for its value

## Exact restart commands for a new session

Use this prompt in a fresh Codex session:

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) git log --oneline -1 확인
3) docs/session_handoff_20260409_article_pipeline_reboot.md 읽기
4) 현재 실행 중인 python/uv 프로세스가 없는지 확인

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential batch만 실행
- batch runner는 scripts/research/run_article_pipeline_research_batches.py만 사용
- completed batch_01~07, 10, 11, 13, 25, 39, 41, 42는 재실행하지 말고 결과만 읽기
- batch_43부터 재개
- train total_return=0 이면서 train trade_count=0이면 no-trade train으로 간주해서 robust 후보에서 강하게 감점
- Alpha101가 느리면 중단하고 lighter batch 유지

목표:
- train/val/oos가 동시에 덜 깨지는 robust 후보 찾기
- 완료된 batch 결과를 요약하고 다음 batch 우선순위를 계속 업데이트
```

Then run exactly this batch first:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
export POLARS_MAX_THREADS=1
export LQ_BACKTEST_LOW_MEMORY=1
export LQ_AUTO_COLLECT_DB=0
export PYTHONUNBUFFERED=1
/usr/bin/time -v timeout --signal=INT 30m \
  uv run python scripts/research/run_article_pipeline_research_batches.py \
  --batch-ids batch_43 \
  --max-batches 1 \
  --stop-after-errors 1
```

If `batch_43` completes and is not clearly strong, preferred next candidates:

```bash
# small 4h market-neutral stability probe
/usr/bin/time -v timeout --signal=INT 30m \
  uv run python scripts/research/run_article_pipeline_research_batches.py \
  --batch-ids batch_24 \
  --max-batches 1 \
  --stop-after-errors 1
```

If you explicitly want to test slower Alpha101 4h afterward:

```bash
/usr/bin/time -v timeout --signal=INT 30m \
  uv run python scripts/research/run_article_pipeline_research_batches.py \
  --batch-ids batch_09 \
  --max-batches 1 \
  --stop-after-errors 1
```

## Files to inspect quickly in the next session

- Handoff doc:
  - `docs/session_handoff_20260409_article_pipeline_reboot.md`
- Candidate manifest:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_candidate_manifest_latest.json`
- Batch plan:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_research_batches_latest.json`
- Best current near-miss outputs:
  - `.../batch_runs/batch_10/candidate_research_latest.json`
  - `.../batch_runs/batch_41/candidate_research_latest.json`
- Interrupted run log:
  - `.../batch_runs/batch_43/batch.log`

