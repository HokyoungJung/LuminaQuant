# Profit Moonshot / Hybrid Alpha — next-session continuation plan

Generated: `2026-05-02T14:50:37Z` (`2026-05-02 23:50:37 KST`)
Repo: `/home/hoky/Quants-agent/LuminaQuant`
Branch/remote: local `private-main`, pushed to `private/main`
Latest pushed commit at handoff: `3040ca1 Stabilize moonshot momentum with survivable hybrids`
Previous derivative-flow commit: `94318dd` pushed before hybrid work

## 0. One-line state

Current best **conservative live-equivalent train/val candidate** is
`profit_moonshot_momentum_hybrid_safe_mode`.
It beats the baseline validation return and materially reduces `boost` train fragility,
but it is **not full deployment-ready** because OOS/raw-first coverage is still incomplete.

## 1. What happened so far

### A. Existing baseline and boost target

Baseline gate from the original handoff:

- `profit_moonshot_adaptive_momentum_mode`
- validation return: `+0.264933%`

Raw validation leader before this handoff:

- `profit_moonshot_adaptive_momentum_boost_mode`
- validation return: `+0.509082%`
- train return: `-2.9948%`
- train MDD: `18.0211%`
- Status: **research-candidate only**, not robust enough for deployment.

### B. Derivatives Flow Squeeze / Exhaustion attempt

Implemented and pushed in commit `94318dd`:

- `src/lumina_quant/strategies/derivatives_flow_squeeze.py`
- `derivatives_flow_squeeze_mode`
- tests and feature-field plumbing for taker-flow support

Live-equivalent evidence:

- train return: `-9.099%`
- validation return: `-0.557%`
- Status: **failed / not deployment-ready**

Root issue:

- true historical funding/OI/taker/liquidation alignment is not complete enough for a valid flow alpha claim.
- Do **not** claim DFSE success until feature-point replay coverage is complete.

### C. Hybrid attempt after user asked whether promising candidates can be blended

Added three recursive portfolio modes in commit `3040ca1`:

1. `profit_moonshot_momentum_hybrid_return_mode`
   - 60% boost
   - 25% vol_target_132
   - 15% governed
2. `profit_moonshot_momentum_hybrid_safe_mode`
   - 35% boost
   - 35% vol_target_132
   - 20% governed
   - 10% asym_dynamic
3. `profit_moonshot_momentum_hybrid_core_mode`
   - 40% boost
   - 40% vol_target_132
   - 15% governed
   - 5% asym_dynamic

Implementation files:

- `src/lumina_quant/strategies/artifact_portfolio_mode.py`
- `src/lumina_quant/live_selection.py`
- `tests/unit/test_artifact_portfolio_mode.py`

Hybrid report artifacts:

- `var/reports/profit_moonshot_20260501/hybrid/momentum_hybrid_report.md`
- `var/reports/profit_moonshot_20260501/hybrid/momentum_hybrid_report.json`
- per-mode live-equivalent JSON/MD/checkpoints under:
  - `var/reports/profit_moonshot_20260501/hybrid/profit_moonshot_momentum_hybrid_return_mode/`
  - `var/reports/profit_moonshot_20260501/hybrid/profit_moonshot_momentum_hybrid_safe_mode/`
  - `var/reports/profit_moonshot_20260501/hybrid/profit_moonshot_momentum_hybrid_core_mode/`

## 2. Current results table

| Mode | Train return | Train MDD | Val return | Val MDD | Val Sharpe | Val Sortino | Trades train/val | Liquidations | Strict user gate | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `profit_moonshot_adaptive_momentum_mode` | `-1.8628%` | `12.0686%` | `+0.2649%` | `0.7544%` | `0.012417` | `0.012036` | `356/52` | `0` | baseline | incumbent baseline |
| `profit_moonshot_adaptive_momentum_boost_mode` | `-2.9948%` | `18.0211%` | `+0.5091%` | `1.3583%` | `0.014751` | `0.014527` | `361/56` | `0` | FAIL | best raw val, fragile train |
| `profit_moonshot_momentum_hybrid_return_mode` | `-1.7990%` | `16.0694%` | `+0.2687%` | `1.0144%` | `0.009768` | `0.009451` | `909/134` | `0` | FAIL | val barely beats baseline; train MDD > 15% |
| `profit_moonshot_momentum_hybrid_safe_mode` | `-1.3551%` | `12.3695%` | `+0.2837%` | `1.0438%` | `0.010168` | `0.009833` | `1185/183` | `0` | PASS | best conservative hybrid |
| `profit_moonshot_momentum_hybrid_core_mode` | `-1.4765%` | `14.1011%` | `+0.2550%` | `1.0143%` | `0.009413` | `0.009093` | `932/138` | `0` | FAIL | train OK, val below baseline |

Resource evidence:

- return hybrid max RSS: `4,803,196 KB`, wall clock `24:10.45`
- safe hybrid max RSS: `4,804,884 KB`, wall clock `24:39.81`
- core hybrid max RSS: `4,807,492 KB`, wall clock `23:06.02`

All stayed below the 8GB RSS cap.

## 3. Current decision

- **Best by validation return overall:** `profit_moonshot_adaptive_momentum_boost_mode`, but do **not** promote because train loss/MDD are fragile.
- **Best conservative live-equivalent candidate:** `profit_moonshot_momentum_hybrid_safe_mode`.
- **Deployment classification:** `hybrid_safe` is a **research/live-equivalent candidate**, not full deployment-ready, because OOS split remains `skipped_oos_data_incomplete`.

Important nuance:

- `hybrid_safe` does **not** solve the user's desire for much higher return.
- It only solves part of the problem: it reduces train fragility while staying above baseline validation return.
- More adaptive-momentum recombination is probably low-yield now.

## 4. Latest data status

A latest-data refresh already completed earlier in this workstream:

- Report: `var/reports/profit_moonshot_20260501/latest/data_refresh_20260502_latest.md`
- collection cutoff: `2026-05-02T06:46:33Z`
- symbols: `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`
- peak RSS: `3071 MiB`

However, if starting a new session later, refresh again before claiming anything current.

Suggested refresh command:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv run lq refresh-data-fast \
  --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,TRX/USDT \
  --priority-symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,TRX/USDT \
  --max-workers 1
```

For the report-style refresh bundle used by prior final-validation flows, inspect and use:

```bash
uv run python scripts/research/refresh_final_portfolio_validation_data.py --help
```

## 5. What to do next

### Priority 1 — make OOS/raw-first validation real

Before promoting any candidate, fix or run the raw-first OOS/latest-tail path so the current best can be tested beyond train/val.

Targets:

- `profit_moonshot_momentum_hybrid_safe_mode`
- `profit_moonshot_adaptive_momentum_vol_target_132_mode`
- `profit_moonshot_adaptive_momentum_boost_mode` only as raw-return comparator, not promotion default

Stop condition:

- OOS no longer says `skipped_oos_data_incomplete`.
- Candidate report explicitly separates train/val and OOS/current-tail evidence.

### Priority 2 — stop over-optimizing adaptive momentum blends

Do not keep trying endless weight blends unless there is a clear new reason.
Evidence from current blends:

- more boost -> validation can rise, train MDD gets fragile.
- more dampers/governors -> train improves, validation edge collapses.
- `hybrid_safe` is already near the useful compromise.

### Priority 3 — new alpha family, but only after feature coverage

The next real uplift likely requires a non-price alpha family:

- derivatives flow/crowding
- funding carry with OI/liquidation filters
- taker-flow imbalance continuation
- liquidation exhaustion/reversal with reclaim confirmation
- cross-sectional momentum/regime breadth with actual OOS coverage

But do **not** rerun DFSE-style claims until the live-equivalent engine can replay timestamp-aligned historical feature points for:

- funding
- open interest
- taker buy/sell flow
- liquidation long/short notional

Relevant current code/artifacts:

- `src/lumina_quant/strategies/derivatives_flow_squeeze.py`
- `tests/unit/test_derivatives_flow_squeeze_strategy.py`
- `tests/test_historic_data_feature_support.py`
- `src/lumina_quant/data_collector.py`
- `var/reports/profit_moonshot_20260501/latest/session_research_report_20260502_round2.md`
- `var/reports/profit_moonshot_20260501/next/session_research_report_20260502.md`

### Priority 4 — if user insists on immediate improvement

Use `hybrid_safe` as the conservative research candidate and try **one** clearly justified new sleeve, not another exposure bump.
Possible local candidate directions:

1. Revisit `vol_target_132` as a standalone/current-tail candidate because it has better validation (`+0.4176%`) and train MDD below 15%.
2. Add a drawdown governor around boost that specifically suppresses the 2025-10 circuit-breaker interval, but verify it does not just overfit one date.
3. Add a breadth/regime filter to boost/vol-target sleeves rather than changing gross exposure.

## 6. Mandatory duplicate-process check before any backtest

Run this first in every new session:

```bash
ps -eo pid,ppid,etimes,rss,cmd | rg 'revalidate_live_equivalent|execute-backtests|profit_moonshot' | rg -v 'rg ' || true
```

If anything is running, do not start duplicate heavy backtests. Inspect logs/checkpoints first.

## 7. Single-mode backtest template

Use one mode at a time; keep RSS below 8GB.

```bash
mode=<MODE_NAME>
out=var/reports/profit_moonshot_20260501/next/${mode}
/usr/bin/time -v uv run python scripts/research/revalidate_live_equivalent_candidates.py \
  --output-dir "$out" \
  --backtest-checkpoint-path "${out}_checkpoint.json" \
  --portfolio-modes "$mode" \
  --execute-backtests --chunk-days 7 --no-live-decision-update --fail-fast-alpha-gate
```

After each candidate:

```bash
uv run python scripts/research/profit_moonshot_research.py \
  --input-dir var/reports/profit_moonshot_20260501 \
  --output-dir var/reports/profit_moonshot_20260501
uv run python scripts/research/validate_profit_moonshot_continuation.py || true
```

## 8. Gates to preserve

Minimum gates from the prior instruction set:

- `val_total_return > 0.0026493262` baseline, preferably `>= 0.0040`
- if possible, `val_total_return > 0.0050908199` boost
- `train_total_return > -0.025`
- `train_max_drawdown < 0.15`
- `val_max_drawdown <= 0.02`
- `val_sharpe > 0`
- `val_sortino > 0`
- `train_trades >= 20`
- `val_trades >= 3`
- `liquidations == 0`
- max RSS `< 8GB`

For full deployment readiness, require OOS/current-tail evidence too; train/val alone is not enough.

## 9. Verification commands

Run before committing:

```bash
uv run ruff check
uv run pytest \
  tests/unit/test_artifact_portfolio_mode.py \
  tests/unit/test_profit_moonshot_research.py \
  tests/test_live_selection_infer.py \
  tests/unit/test_profit_moonshot_strategies.py \
  tests/test_strategy_factory_library.py \
  tests/unit/test_adaptive_regime_momentum.py \
  tests/unit/test_live_equivalent_revalidation.py \
  tests/unit/test_derivatives_flow_squeeze_strategy.py \
  tests/test_historic_data_feature_support.py -q
```

Last run in this session:

- `uv run ruff check` -> pass
- pytest command above -> `61 passed`

## 10. Exact command/prompt for a new session

Paste this into the next Codex/OMX session:

```text
cd /home/hoky/Quants-agent/LuminaQuant

private/main 최신 상태를 기준으로 시작해. 먼저 아래 handoff를 읽고 그대로 이어서 진행해:
var/reports/profit_moonshot_20260501/continuation/NEXT_SESSION_PLAN_20260502T145037Z.md

목표는 profit moonshot 성능을 더 끌어올리는 것이다. 현재 conservative best는 `profit_moonshot_momentum_hybrid_safe_mode`지만 deployment-ready는 아니다. 먼저 중복 backtest 프로세스를 확인하고, 최신 데이터 tail을 refresh한 뒤, train/val/OOS raw-first evidence를 분리해서 검증해라. 단순 gross exposure 증가는 금지. `hybrid_safe`를 보수 후보로 유지하면서, OOS coverage를 복구하거나 funding/OI/taker-flow/liquidation feature replay가 준비된 새 alpha family를 구현/검증해라. live-equivalent/OOS gate를 통과하지 못한 것은 성공으로 간주하지 마라. 한 번에 한 mode만 backtest하고 RSS 8GB 미만을 유지해라. 실패 후보도 왜 실패했는지 보고서에 저장하고, 최종 상태를 Lore commit으로 커밋 후 `git push private private-main:main`까지 완료해라.
```

## 11. Current tracked report files to read first

1. `var/reports/profit_moonshot_20260501/continuation/NEXT_SESSION_PLAN_20260502T145037Z.md`
2. `var/reports/profit_moonshot_20260501/hybrid/momentum_hybrid_report.md`
3. `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.md`
4. `var/reports/profit_moonshot_20260501/latest/session_research_report_20260502_round2.md`
5. `var/reports/profit_moonshot_20260501/next/session_research_report_20260502.md`
6. `var/reports/profit_moonshot_20260501/latest/data_refresh_20260502_latest.md`

## 12. Commit/push status at this handoff

Already committed and pushed:

```bash
git log -1 --oneline
# 3040ca1 Stabilize moonshot momentum with survivable hybrids

git push private private-main:main
# completed: 94318dd..3040ca1 private-main -> main
```

This handoff file itself should be committed/pushed by the session that creates it.
