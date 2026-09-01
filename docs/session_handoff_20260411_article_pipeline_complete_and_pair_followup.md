# Session handoff — 2026-04-11 article pipeline complete + pair-spread follow-up

## Final state

- Repo: `/home/hoky/Quants-agent/LuminaQuant`
- Branch during this wave: `private-main...private/main [ahead 1]`
- End-of-wave process state: **no active LuminaQuant python/uv research process**

## What is now fully done

1. **WAL startup bottleneck fixed**
   - read-only parquet/WAL paths no longer force eager repair on every open
   - key large 1s WALs were compacted to monthly parquet

2. **Alpha101 compute bottleneck fixed**
   - `rolling_rank_series` no longer uses pandas `rolling.apply` callback
   - replaced with vectorized/chunked NumPy path

3. **All 44 article batches completed**
   - article sweep is done end-to-end under the shared sequential `<8GB` envelope

4. **Pair-spread robustness follow-up completed**
   - focused BNB/TRX 1h follow-up ran successfully using the same batch runner
   - additional trade-count curve follow-up also completed

## Canonical article result summary

Read these first for the full completed article sweep:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.json`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.md`

Key conclusions from the completed 44-batch sweep:
- Best real-trade article candidate is still `batch_28`, but it fails OOS
- Most interesting overall near-misses are now market-neutral pair-spread candidates:
  - `batch_15`
  - `batch_14`
  - `batch_18`
- These show **train/val/oos all positive**, but fail hard on **PBO**

## Pair-spread follow-up artifacts

Read these next:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_followup_current/pair_spread_robustness_followup_summary_latest.json`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/pair_spread_robustness_followup_current/pair_spread_robustness_followup_summary_latest.md`

### Pair-spread conclusion
The BNB/TRX 1h lane shows a clear trade-off:
- profitable settings (`2.5~2.6 entry`) keep OOS positive, but PBO stays too high (`0.5~0.625`)
- lower-entry settings (`2.0`) reduce PBO, but OOS turns negative

So the remaining blocker is no longer runtime, memory, or batch orchestration.
It is **statistical robustness / overfitting control**.

Additional singleton midpoint follow-up confirmed the same trade-off:
- `2.2/0.55` settings lowered train/oos PBO materially (`train pbo=0.125`, `oos pbo=0.5`)
- but OOS Sharpe collapsed to `0.297` and train return turned negative


## Best current non-promotable candidates

### Most interesting pair-spread near-misses
1. `pair_spread_1h_exec_takeprofit_bnbusdt_trxusdt_2.6_0.70` (`batch_15`)
   - train `+0.6551%`
   - val `+5.8769%`
   - oos `+4.6274%`
   - oos Sharpe `5.5142`
   - hard reject: `PBO=0.5`

2. `pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70` (`batch_14`)
   - train `+0.6294%`
   - val `+6.1098%`
   - oos `+4.6274%`
   - oos Sharpe `5.5142`
   - hard reject: `PBO=0.5`

3. `pair_spread_1h_state_vwap_bnbusdt_trxusdt_2.6_0.70` (`batch_18`)
   - train `+0.9529%`
   - val `+5.3183%`
   - oos `+4.2738%`
   - oos Sharpe `5.1324`
   - hard reject: `PBO=0.625`

## Practical next priorities

### Priority 1
**Pair-spread robustness tightening, not more article batch execution**

Possible directions:
- reduce parameter degrees of freedom for the BNB/TRX 1h lane
- add explicit walk-forward / stability constraints rather than only entry-threshold tweaks
- revise acceptance / evaluation logic only if you are confident it is scientifically justified
- test whether PBO should be computed on a different stability basis only with strong evidence

### Priority 2
**Broader search redesign**

If pair-spread robustness cannot be improved convincingly, the next serious step is:
- new candidate families, or
- a redesigned search process with stronger robustness priors from the start

## What not to do next

- Do **not** rerun article `batch_01~44`
- Do **not** spend more time on the light/fallback article lanes; they are already exhausted
- Do **not** re-open the WAL/Alpha101 timeout diagnosis unless new evidence appears; those runtime blockers are already fixed

## Exact restart prompt

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) git log --oneline -1 확인
3) docs/session_handoff_20260411_article_pipeline_complete_and_pair_followup.md 읽기
4) 현재 실행 중인 python/uv 프로세스가 없는지 확인

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential 실행만 허용
- article batch는 이미 전부 완료됨; batch_01~44는 재실행하지 말 것
- train total_return=0 이면서 train trade_count=0이면 no-trade train으로 간주해서 robust 후보에서 강하게 감점

현재 목표:
- pair-spread robustness tightening 또는 broader redesign
- 남은 문제는 runtime이 아니라 PBO/robustness
```
