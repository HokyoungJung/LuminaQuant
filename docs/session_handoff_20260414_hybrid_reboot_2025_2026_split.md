# Session handoff — hybrid reboot plan for 2025/2026 split

## Goal for the next session
Re-evaluate and re-tune the hybrid online portfolio governor using the new calendar split:
- **train:** `2025-01-01` ~ `2025-12-31`
- **val:** `2026-01-01` ~ `2026-02-28`
- **oos:** `2026-03-01` ~ latest available tail

Do **not** reuse the old saved/refreshed split assumptions for the final decision after reboot.

## Warm-up policy (user-specified)
Warm-up is **60% of the full pre-OOS period**.

- pre-OOS period = `2025-01-01` ~ `2026-02-28`
- total pre-OOS days = `424`
- warmup_days = `ceil(424 * 0.60) = 255`
- online scoring should therefore start on **`2025-09-13`**

Interpretation:
- `2025-01-01` ~ `2025-09-12`: warm-up only
- `2025-09-13` onward: online adaptive scoring active
- train/val/oos reporting should still use the requested calendar split above; warm-up just controls when the hybrid starts adaptive switching

## Current repo state at handoff
- branch: `private-main`
- latest pushed commit: `26a0c7f Preserve adaptive portfolio governance with guarded live routing`
- after that push, the hybrid runner was updated to:
  - separate `warmup_days` from `lookback_days`
  - set default `warmup_days=182` temporarily
- current local modification still pending at handoff:
  - `scripts/research/run_hybrid_online_portfolio.py`
  - `tests/test_hybrid_online_portfolio.py`
- next session should decide whether to commit/push the warm-up refactor before or together with the new 2025/2026 split work

## What already exists and should be reused
### Hybrid / tuning code
- `scripts/research/run_hybrid_online_portfolio.py`
- `scripts/research/tune_hybrid_online_portfolio.py`
- `scripts/research/optuna_tune_hybrid_online_portfolio.py`
- `scripts/research/run_hybrid_online_portfolio_major.py`
- `tests/test_hybrid_online_portfolio.py`

### Switch / playbook integration
- `scripts/research/write_portfolio_operating_switch.py`
- `scripts/research/write_portfolio_operating_playbook.py`
- `tests/unit/test_portfolio_operating_switch.py`
- `tests/unit/test_portfolio_operating_playbook.py`

### Current generated artifacts
- hybrid current: `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json`
- curated tuning: `.../portfolio_hybrid_online_tuning_current/hybrid_online_tuning_latest.json`
- optuna tuning: `.../portfolio_hybrid_online_optuna_current/hybrid_online_optuna_latest.json`
- major-universe experiment: `.../portfolio_hybrid_online_major_current/hybrid_online_major_latest.json`
- switch: `.../current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md`
- playbook: `.../portfolio_candidate_overlay_review_current/portfolio_operating_playbook_latest.md`
- runbook: `.../final_master_scoreboard_current/hybrid_guarded_mode_runbook_latest.md`

## Recommended plan for the next session
### Phase 1 — repo hygiene + grounding
1. `git status --short --branch`
2. `git log --oneline -3`
3. Read this handoff doc
4. Read:
   - `final_master_scoreboard_current/portfolio_master_scoreboard_latest.md`
   - `final_master_scoreboard_current/hybrid_guarded_mode_runbook_latest.md`
   - `portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json`
5. Confirm no active python/uv heavy process

### Phase 2 — make the hybrid split-aware
Adjust the hybrid surfaces so they do **not** rely on the old saved/refreshed split assumptions:
- allow explicit split windows:
  - `train_start`, `train_end`
  - `val_start`, `val_end`
  - `oos_start`, `oos_end/latest`
- allow explicit `warmup_days`
- keep `lookback_days` as a separate tuned parameter
- ensure tuning/evaluation artifacts are clearly labeled with the 2025/2026 split

### Phase 3 — re-tune all hybrid internal params
Re-run:
- curated tuning
- Optuna tuning (`live_guarded` and `train_aware_guarded` if both still make sense)

But now under the **new split** and with:
- `warmup_days=255`
- full parameter retuning for the hybrid internal knobs

### Phase 4 — re-evaluate switch/playbook
Only after the 2025/2026 split tuning is complete:
- rebuild hybrid artifact
- rebuild switch
- rebuild playbook
- rebuild scoreboard / one-pager if recommendation changes

## Constraints to preserve
- total session/process memory **< 8 GiB**
- heavy runs **strictly sequential only**
- do **not** rerun article `batch_01~44`
- pair remains capped / sparse penalties remain active
- if `omx team` is needed, use a clean review worktree rather than the dirty source worktree

## Exact shell prep for the new split
```bash
cd /home/hoky/Quants-agent/LuminaQuant

export POLARS_MAX_THREADS=1
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LQ_BACKTEST_LOW_MEMORY=1
export LQ_AUTO_COLLECT_DB=0
export PYTHONUNBUFFERED=1

python - <<'PY'
from datetime import date, timedelta
import math
start = date(2025, 1, 1)
oos_start = date(2026, 3, 1)
pre_oos_days = (oos_start - start).days
warmup_days = math.ceil(pre_oos_days * 0.60)
print({'pre_oos_days': pre_oos_days, 'warmup_days': warmup_days, 'online_start': str(start + timedelta(days=warmup_days))})
PY
```

Expected output:
- `pre_oos_days: 424`
- `warmup_days: 255`
- `online_start: 2025-09-13`

## Suggested new-session prompt
```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

목표:
- hybrid online portfolio를 새 split으로 전체 재평가/재튜닝
- train: 2025-01-01 ~ 2025-12-31
- val: 2026-01-01 ~ 2026-02-28
- oos: 2026-03-01 ~ latest
- warmup_days는 OOS 제외 전체 기간의 60%로 고정 (424일 기준 255일, online start=2025-09-13)
- lookback_days와 warmup_days는 분리 유지
- hybrid 내부 파라미터 전부 다시 튜닝
- heavy run은 항상 sequential only, 8GB 절대 초과 금지
- article batch_01~44 재실행 금지

먼저:
1) git status / git log 확인
2) docs/session_handoff_20260414_hybrid_reboot_2025_2026_split.md 읽기
3) final_master_scoreboard_current/portfolio_master_scoreboard_latest.md 읽기
4) final_master_scoreboard_current/hybrid_guarded_mode_runbook_latest.md 읽기
5) portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json 확인
6) 실행 중인 python/uv process 없는지 확인

그 다음:
- hybrid runner / tuners를 새 split + warmup_days=255 기준으로 parameterize
- curated + optuna tuning 다시 실행
- 결과 비교 후 switch/playbook/scoreboard 갱신
```
