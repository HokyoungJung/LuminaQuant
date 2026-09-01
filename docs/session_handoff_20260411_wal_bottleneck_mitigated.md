# Session handoff — 2026-04-11 WAL bottleneck mitigated

## What changed

This session switched from article-batch retries to the underlying startup bottleneck.

### Code fix
Updated read-only parquet/WAL access so large WAL files are **not eagerly repaired on every open**:
- `src/lumina_quant/storage/parquet/ohlcv_repo.py`
  - `_load_wal_frame(...)` now opens `BinaryWAL(..., auto_repair=False)`
  - min/max metadata scan path now also uses `auto_repair=False`

Intent:
- read paths only need the valid prefix
- eager repair was re-scanning large healthy WALs before immediately decoding them again
- this was consuming most of the 30m article-batch budget

### Regression coverage added
- `tests/test_wal_binary.py`
  - read-only open ignores partial tail without truncating
  - read-only open stops at invalid CRC tail without truncating
- `tests/test_parquet_market_data.py`
  - repo load still reads the valid prefix correctly even when WAL has a broken tail

### Verification
- `uv run pytest tests/test_wal_binary.py tests/test_parquet_market_data.py -q`
  - **14 passed**
- `uv run ruff check src/lumina_quant/storage/parquet/ohlcv_repo.py tests/test_wal_binary.py tests/test_parquet_market_data.py`
  - **passed**

## Operational mitigation applied

To remove the existing hot-path debt from the current market store, I also ran the canonical compaction script on the symbols shared by the blocked mixed-universe article batches:

```bash
uv run python scripts/compact_wal_to_monthly_parquet.py \
  --root-path data/market_parquet \
  --exchange binance \
  --symbols BNB/USDT,BTC/USDT,ETH/USDT,SOL/USDT,TRX/USDT,XAG/USDT,XAU/USDT
```

Observed result:
- completed successfully in **1:15.69**
- max RSS **2,234,600 KiB**
- `wal.bin` size is now **0** for:
  - `BNBUSDT`
  - `BTCUSDT`
  - `ETHUSDT`
  - `SOLUSDT`
  - `TRXUSDT`
  - `XAGUSDT`
  - `XAUUSDT`

This removed the historical WAL load debt for the exact symbol set that had been timing out.

## Proof that the bottleneck is actually mitigated

Previously blocked / slow batches were rerun **under the same sequential 30m / <8GB envelope** and all completed:

1. `batch_37` — VwapReversionStrategy 15m
   - **before:** timeout at `29:38.66`
   - **after:** completed in `0:50.42`
   - max RSS `1,122,580 KiB`
   - result: negative across splits

2. `batch_40` — CompositeTrendStrategy 30m
   - **before:** timeout at `30:00.22`
   - **after:** completed in `1:11.00`
   - max RSS `1,608,364 KiB`
   - result: **val +4.0762%, oos +0.9129%, but train no-trade**

3. `batch_43` — RollingBreakoutStrategy 1h
   - **before:** timeout at `30:20.54`
   - **after:** completed in `0:51.04`
   - max RSS `1,138,960 KiB`
   - result: val strong, oos negative

4. `batch_31` — MeanReversionStdStrategy 30m
   - **before:** timeout at `29:39.81`
   - **after:** completed in `0:37.88`
   - max RSS `1,117,720 KiB`
   - result: negative across splits

5. `batch_09` — Alpha101FormulaStrategy 4h
   - **before:** manual stop at `20:39.71` while stuck in WAL auto-repair path
   - **after:** completed in `2:34.53`
   - max RSS `1,103,120 KiB`
   - result: mixed / non-robust (`train` and `val` negative, `oos` positive)

6. `batch_35` — VolCompressionVWAPReversionStrategy 5m
   - completed in `0:49.39`
   - max RSS `1,119,860 KiB`
   - result: **negative across splits**

7. `batch_38` — VwapReversionStrategy 5m
   - completed in `0:36.11`
   - max RSS `1,126,156 KiB`
   - result: **negative across splits**

8. `batch_08` — Alpha101FormulaStrategy 1h
   - after the WAL fix and Alpha101 compute fix, it completed in `5:40.35`
   - max RSS `1,135,272 KiB`
   - result: **negative across splits**

9. `batch_12` — LeadLagSpilloverStrategy 5m
   - completed in `1:01.24`
   - max RSS `1,368,476 KiB`
   - result: **catastrophically negative across splits**

10. `batch_14` — PairSpreadZScoreStrategy 1h
   - completed in `1:27.64`
   - max RSS `1,188,456 KiB`
   - result: **train/val/oos all positive**, but hard-reject on `PBO=0.5`

11. `batch_15` — PairSpreadZScoreStrategy 1h
   - completed in `1:22.26`
   - max RSS `1,183,832 KiB`
   - result: **train/val/oos all positive**, but hard-reject on `PBO=0.5`

12. `batch_16` — PairSpreadZScoreStrategy 1h
   - completed in `1:20.25`
   - max RSS `1,134,424 KiB`
   - result: mixed; high OOS snapshot but train no-trade / val break

13. `batch_17` — PairSpreadZScoreStrategy 1h
   - completed in `1:36.37`
   - max RSS `1,134,812 KiB`
   - result: mixed; high OOS snapshot but train no-trade / val break

14. `batch_18` — PairSpreadZScoreStrategy 1h
   - completed in `1:53.54`
   - max RSS `1,149,216 KiB`
   - result: **train/val/oos all positive**, but hard-reject on `PBO=0.625`

15. `batch_19` — PairSpreadZScoreStrategy 1h
   - completed in `0:59.10`
   - max RSS `1,152,544 KiB`
   - result: mixed / non-robust

16. `batch_20` — PairSpreadZScoreStrategy 30m
   - completed in `1:20.13`
   - max RSS `1,122,568 KiB`
   - result: negative across splits

17. `batch_21` — PairSpreadZScoreStrategy 4h
   - completed in `0:26.91`
   - max RSS `1,130,756 KiB`
   - result: weak OOS-only / no-trade train

18. `batch_22` — PairSpreadZScoreStrategy 4h
   - completed in `0:26.75`
   - max RSS `1,104,832 KiB`
   - result: dead zero

19. `batch_23` — PairSpreadZScoreStrategy 4h
   - completed in `0:27.24`
   - max RSS `1,113,244 KiB`
   - result: strong OOS-only near-miss, but no-trade train/val


## Current article summary state

Canonical summary artifacts were refreshed:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.json`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/article_inspired_research_current/article_pipeline_resume_summary_latest.md`

Important updated conclusions:
- bottleneck-dominated blocked set is cleared, and the Alpha101 compute blocker is also cleared
- best overall real-trade candidate is **still `batch_28`** (train/val positive, oos negative)
- best no-trade-train near-miss is still **`batch_41`**
- new weaker near-miss added:
  - `batch_40` (`train no-trade`, `val +4.08%`, `oos +0.91%`)
- `batch_35` and `batch_38` both completed after the fix and both failed decisively, so the remaining light mixed-universe reversion family is effectively exhausted
- `batch_14`, `batch_15`, and `batch_18` are the strongest post-fix family members and all show **train/val/oos all positive**, but each still hard-rejects on PBO, so the market-neutral story is now about robustness rather than raw return

## Recommended next order

1. `pair-spread robustness tightening`
   - the full article batch sweep is now complete, and the best remaining candidates are pair-spread 1h variants with excellent split returns but unacceptable PBO
2. `broader article search redesign`
   - since all 44 article batches are done, additional progress now requires either new families or revised acceptance/robustness logic rather than more brute-force batch execution

## Restart prompt for next session

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) git log --oneline -1 확인
3) docs/session_handoff_20260411_wal_bottleneck_mitigated.md 읽기
4) 현재 실행 중인 python/uv 프로세스가 없는지 확인

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential batch만 실행
- batch runner는 scripts/research/run_article_pipeline_research_batches.py만 사용
- completed batch_01~44는 재실행하지 말고 결과만 읽기
- train total_return=0 이면서 train trade_count=0이면 no-trade train으로 간주해서 robust 후보에서 강하게 감점

목표:
- train/val/oos가 동시에 덜 깨지는 robust 후보 찾기
- article batch는 전부 끝났고, 다음 우선순위는 pair-spread robustness tightening 또는 search redesign이다
```
