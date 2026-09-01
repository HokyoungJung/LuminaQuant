# Session handoff — 2026-04-08 article pipeline / strict retune

## What was completed

- Strict leverage rerun sweep completed and summarized:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/strict_blend_76_24_leverage_sweep_rerun_current/summary_latest.json`
  - Best combo by OOS ranking: incumbent `1x`, autoresearch `4x`
  - Caveat: train/validation robustness remained weak
- Added a new article-inspired strategy:
  - `CarryTrendFactorRotationStrategy`
- Added article-pipeline manifest tooling:
  - `scripts/research/build_article_pipeline_manifest.py`
  - `scripts/research/run_article_pipeline_research_batches.py`
- Built full article-pipeline candidate manifest:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_candidate_manifest_latest.json`
  - `.../article_pipeline_research_batches_latest.json`
- OOM-conscious sequential batch research executed successfully for batches `01` through `07`

## Batch progress

Completed final reports exist for:

- `batch_01`
- `batch_02`
- `batch_03`
- `batch_04`
- `batch_05`
- `batch_06`
- `batch_07`

Each completed batch has:

- `candidate_research_latest.json`
- `candidate_research_latest.csv`
- `strategy_factory_report_latest.json`
- `strategy_factory_shortlist_*.md`

Partial-only directories exist for:

- `batch_08` — started, no final report saved
- `batch_10` — started, no final report saved

No research processes should be left running after the handoff cleanup.

## Quick findings so far

- Newly added `CarryTrendFactorRotationStrategy` showed some train/validation promise in isolated batches, but no robust winner yet.
- Completed batches `01`–`07` did **not** produce a clearly train/val/oos-robust candidate.
- The OOM-conscious batch runner kept per-batch RSS around ~1.8GB or lower on completed runs, which stayed comfortably below the 8GB total-session ceiling when run sequentially.

## Key files changed in code

- `src/lumina_quant/strategy_factory/research_runner.py`
- `src/lumina_quant/strategy_factory/candidate_library.py`
- `src/lumina_quant/strategy_factory/__init__.py`
- `tests/test_research_runner_feature_support.py`
- `tests/test_strategy_factory_library.py`
- `scripts/research/build_article_pipeline_manifest.py`
- `scripts/research/run_article_pipeline_research_batches.py`

Also preserved existing local strict-validation work already present in the tree:

- `scripts/research/run_grouped_allocator_strict_leverage_validation.py`
- `tests/unit/test_grouped_allocator_strict_leverage_validation.py`

## Recommended restart commands

Rebuild the article manifest only if needed:

```bash
uv run python scripts/research/build_article_pipeline_manifest.py \
  --symbols BTC/USDT ETH/USDT BNB/USDT SOL/USDT TRX/USDT XAU/USDT XAG/USDT \
  --timeframes 5m 15m 30m 1h 4h
```

Resume sequential article research from the next batches:

```bash
uv run python scripts/research/run_article_pipeline_research_batches.py \
  --batch-ids batch_08,batch_09,batch_10,batch_11,batch_12 \
  --stop-after-errors 1
```

If Alpha101 continues to be too slow, skip it and continue with lighter families:

```bash
uv run python scripts/research/run_article_pipeline_research_batches.py \
  --batch-ids batch_10,batch_11,batch_12,batch_13,batch_14 \
  --stop-after-errors 1
```

## Suggested prompt for a new session

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) docs/session_handoff_20260408_article_pipeline.md 읽기
3) 현재 실행 중인 python/uv 프로세스가 없는지 확인

그 다음 article pipeline research를 이어서 진행해.

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential batch만 실행
- batch runner는 scripts/research/run_article_pipeline_research_batches.py만 사용
- 이미 완료된 batch_01~07은 재실행하지 말고 결과만 읽어
- batch_08, batch_09 이후부터 재개하되, Alpha101가 지나치게 오래 걸리면 중단하고 lighter batch로 우회

참고 아티팩트:
- strict rerun summary:
  var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/strict_blend_76_24_leverage_sweep_rerun_current/summary_latest.json
- article candidate manifest:
  var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_candidate_manifest_latest.json
- article batch plan:
  var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_research_batches_latest.json

목표:
- train/val/oos가 동시에 덜 깨지는 robust 후보 찾기
- 중간중간 완료된 batch 결과를 요약하고, 다음 배치 우선순위를 업데이트
```
