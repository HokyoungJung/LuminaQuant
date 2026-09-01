# Session handoff — 2026-04-13 risk-off switch + bearish follow-up

## Current repo state

- Repo: `/home/hoky/Quants-agent/LuminaQuant`
- Branch at handoff: `private-main...private/main [ahead 1]`
- Latest committed HEAD at session start: `e378dc3 Preserve article-pipeline research expansion and reboot handoff state`
- New untracked source files added this session:
  - `scripts/research/write_portfolio_operating_switch.py`
  - `scripts/research/write_portfolio_operating_playbook.py`
  - `tests/unit/test_portfolio_operating_switch.py`
- End-of-session process state: **no active LuminaQuant python/uv research process**

## What this session completed

### 1) Current-tail data refresh
- Refresh command completed under the sequential low-memory envelope.
- Canonical artifact:
  - `var/reports/exact_window_backtests/followup_status/final_portfolio_validation_data_refresh_latest.json`
- Key facts:
  - `collection_cutoff_utc`: `2026-04-13T11:25:16Z`
  - feature tails updated through `2026-04-13T11:25:00Z`
  - mode: sequential (`selected_workers=1`)
  - max RSS during refresh: `4,370,196 KiB`
  - article `batch_01~44` was **not** rerun

### 2) Risk-off-capable operating switch
Added a lightweight operational switch layer that now:
- recomputes the **current** market snapshot from the latest repaired `feature_points`
- uses **raw aggTrades dollar volume** for BNB/TRX pair liquidity instead of relying on stale/missing materialized volume
- blocks `aggressive_realized_mode` when the hard allocator is unhealthy
- supports a new explicit `risk_off_mode = {"cash": 1.0}`

Source files:
- `scripts/research/write_portfolio_operating_switch.py`
- `tests/unit/test_portfolio_operating_switch.py`

Verification:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/unit/test_portfolio_operating_switch.py` -> **7 passed**
- `uv run ruff check scripts/research/write_portfolio_operating_switch.py tests/unit/test_portfolio_operating_switch.py` -> **pass**

### 3) Refreshed current-switch validation
A full refreshed latest-anchored validation chain was run and saved under:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/`

Important outputs:
- refreshed incumbent portfolio:
  - `.../refreshed_current_one_shot_incumbent_portfolio_latest.json`
- refreshed 55/45 autoresearch portfolio:
  - `.../refreshed_autoresearch_pair_55_45_portfolio_latest.json`
- refreshed static blend:
  - `.../refreshed_grouped_static_blend_latest.json`
- refreshed market regime judgement:
  - `.../refreshed_market_regime_judgement_current/group_market_regime_judgement_latest.md`
- refreshed soft allocator:
  - `.../refreshed_soft_three_way_allocator_current/soft_three_way_market_regime_allocator_latest.md`
- refreshed hard allocator:
  - `.../refreshed_three_way_allocator_current/three_way_market_regime_allocator_latest.md`
- refreshed pair sleeve:
  - `.../refreshed_pair_fast_exit_candidate_latest.json`
- refreshed strategy1 (= balanced 80/20):
  - `.../refreshed_balanced_overlay_strategy_latest.json`
- refreshed switch recommendation:
  - `.../refreshed_operating_switch_current/portfolio_operating_switch_latest.md`
- refreshed switch vs strategy1 comparison:
  - `.../refreshed_switch_vs_strategy1_validation_latest.md`

### 4) Current bearish-strategy scan summary
Rather than trusting unverified directional-short ideas, the session summarized only already-refreshed/validated current-regime choices:
- `.../current_regime_bearish_strategy_scan_latest.md`

Current ranking (refreshed OOS):
1. `pair_spread_1h_exec_tightstop_tp_fastexit_bnbusdt_trxusdt_2.5_0.75`
   - OOS `+2.2211%`, Sharpe `1.7881`, max DD `1.9721%`
2. `risk_off_cash`
   - OOS `0.0000%`, Sharpe `0.0000`, max DD `0.0000%`
3. `balanced_overlay_80_20`
   - OOS `-0.8991%`, Sharpe `-2.0816`
4. `soft_three_way_regime`
   - OOS `-1.6702%`, Sharpe `-5.1342`
5. `three_way_regime`
   - OOS `-5.4140%`, Sharpe `-9.7822`

Interpretation:
- There is still **no clean pure directional-short winner**.
- The only active strategy still positive in the refreshed hostile regime is the **pair fast-exit sleeve**.
- But because current pair liquidity is weak and every active multi-sleeve allocator is negative, the switch now prefers **cash**.

## Final current recommendation

### Current market state (refreshed)
From `.../refreshed_operating_switch_current/portfolio_operating_switch_latest.md`:
- favored_group: `autoresearch`
- confidence: `1.0000`
- trend_state: `bearish`
- breadth_state: `mixed`
- volatility_state: `calm`
- pair_liquidity_state: `weak`
  - BNB volume ratio: `0.3727`
  - TRX volume ratio: `0.6167`

### Recommended mode now
- mode: `risk_off_mode`
- allocation: `{"cash": 1.0}`

Reason:
- balanced health is negative
- soft allocator health is negative
- hard allocator health is negative
- regime is bearish/defensive enough that active sleeves should be disabled

## Switch vs strategy1 verdict

`strategy1` here means the prior default recommendation:
- `balanced_overlay_mode`
- `soft_three_way_regime 80% + pair_fast_exit 20%`

From `.../refreshed_switch_vs_strategy1_validation_latest.md`:
- risk_off_cash: return `+0.0000%` | sharpe `0.0000` | max DD `0.0000%`
- strategy1 balanced 80/20: return `-0.8991%` | sharpe `-2.0816` | max DD `1.6058%`
- three_way_regime: return `-5.4140%` | sharpe `-9.7822` | max DD `5.4140%`

Therefore:
- `risk_off_mode` is currently **better than strategy1** on return, sharpe, and drawdown
- strategy1 is still the **least-bad active** multi-sleeve option
- if active exposure must be kept despite the hostile regime, the tactical choice is:
  - `pair_fast_exit` single sleeve (with strict operator gating), not the allocators

## Operator-facing playbook

Canonical operator playbook:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_candidate_overlay_review_current/portfolio_operating_playbook_latest.md`

Important modes in the playbook now:
- `risk_off_mode`: `cash 100%`
- `core_mode`: `soft_three_way_regime 100%`
- `balanced_overlay_mode`: `soft_three_way_regime 80% + pair_fast_exit 20%`
- `defensive_overlay_mode`: `soft_three_way_regime 70% + pair_fast_exit 30%`
- `aggressive_realized_mode`: `three_way_regime 100%`
- `pair_tactical_mode`: `pair_fast_exit 100%`

Single-sleeve pair rules:
- `pair_fast_exit` remains **overlay_or_tactical_only**
- preferred overlay weight: `20%`
- max weight inside multi-sleeve portfolio: `30%`
- disable when pair liquidity is weak/stale or when the sleeve turns negative on refreshed validation

## What not to trust blindly

- Do **not** assume `autoresearch` favored_group means `three_way_regime` should go live.
- The refreshed validation showed the hard allocator is unhealthy even though the refreshed market regime judgement favored `autoresearch`.
- The switch now explicitly blocks that escalation.

## Recommended next-session workflow

Start with this exact checklist:

1. `cd /home/hoky/Quants-agent/LuminaQuant`
2. `git status --short --branch`
3. `git log --oneline -1`
4. Read these first:
   - `docs/session_handoff_20260413_risk_off_switch_and_bearish_followup.md`
   - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md`
   - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_switch_vs_strategy1_validation_latest.md`
   - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_candidate_overlay_review_current/portfolio_operating_playbook_latest.md`
   - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/current_regime_bearish_strategy_scan_latest.md`
5. Confirm no active heavy process:
   - `ps -eo pid,ppid,rss,etimes,cmd | awk 'tolower($0) ~ /python|[[:space:]]uv([[:space:]]|$)/ && $0 !~ /awk / {print}'`

## Exact commands to rerun the current pipeline in a new session

### 0) Environment guardrails
Always use:
```bash
export POLARS_MAX_THREADS=1
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LQ_BACKTEST_LOW_MEMORY=1
export LQ_AUTO_COLLECT_DB=0
export PYTHONUNBUFFERED=1
```

### 1) Refresh latest data tail
```bash
uv run lq refresh-data-fast \
  --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,TRX/USDT \
  --priority-symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,TRX/USDT \
  --max-workers 1
```

### 2) Rebuild the refreshed current-switch validation bundle
This uses the saved helper script generated in the artifact directory:
```bash
/usr/bin/time -v uv run python \
  var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/run_current_switch_validation.py
```

### 3) Rebuild the refreshed operating switch (risk-off aware)
```bash
uv run python scripts/research/write_portfolio_operating_switch.py \
  --market-judgement-path var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_market_regime_judgement_current/group_market_regime_judgement_latest.json \
  --soft-allocator-path var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_soft_three_way_allocator_current/soft_three_way_market_regime_allocator_latest.json \
  --three-way-allocator-path var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_three_way_allocator_current/three_way_market_regime_allocator_latest.json \
  --balanced-strategy-path var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_balanced_overlay_strategy_latest.json \
  --output-dir var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_operating_switch_current
```

### 4) Rebuild the operator playbook
```bash
uv run python scripts/research/write_portfolio_operating_playbook.py
```

## If the next session focuses on bearish ideas

Use the refreshed evidence first before launching new searches:
- if `risk_off_mode` remains recommended, do not rush into active short strategies
- treat `pair_fast_exit` as the only currently validated tactical active sleeve
- any new bearish search should be **focused, regime-aware, sequential, and low-memory**
- avoid broad brute-force across all candidate families

## Important constraints to preserve next session

- total session memory **must remain below 8GB**
- **no parallel heavy runs**
- **sequential execution only**
- do **not** rerun article `batch_01~44`
- keep the no-trade-train penalty rule:
  - `train total_return = 0` and `train trade_count = 0` => strong demotion / reject

## One-line resume prompt for the next session

```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status 확인
2) git log --oneline -1 확인
3) docs/session_handoff_20260413_risk_off_switch_and_bearish_followup.md 읽기
4) current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md 읽기
5) current_switch_validation_current/refreshed_switch_vs_strategy1_validation_latest.md 읽기
6) portfolio_candidate_overlay_review_current/portfolio_operating_playbook_latest.md 읽기
7) 현재 실행 중인 python/uv 프로세스가 없는지 확인

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential 실행만 허용
- article batch_01~44 재실행 금지
- train total_return=0 && train trade_count=0 은 no-trade train으로 간주해서 강하게 감점

현재 목표:
- risk_off / balanced / pair_tactical 중 무엇이 맞는지 최신 데이터 기준으로 유지/갱신
- 필요하면 bearish 후보를 아주 좁게 follow-up 하되 regime-aware하게 진행
- broad batch search 말고 현재 hostile regime 대응력 위주로 검증
```
