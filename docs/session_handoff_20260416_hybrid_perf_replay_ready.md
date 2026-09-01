# Session handoff — hybrid performance-first + replay status

## Current checkpoint
- repo: `/home/hoky/Quants-agent/LuminaQuant`
- branch: `private-main`
- latest pushed commit before next session: `379e2b1`
- current date context: `2026-04-16`

## What is already done
### Reboot split contract
- train: `2025-01-01` ~ `2025-12-31`
- val: `2026-01-01` ~ `2026-02-28`
- oos: `2026-03-01` ~ latest
- warmup_ratio / warmup_days / online_start: `0.60` / `255` / `2025-09-13`

### Live switch / policy state
- current reboot-validation switch artifact:
  - `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md`
- current default live mode:
  - **`hybrid_guarded_mode`**
- pair remains:
  - **tactical-only**
- balanced remains:
  - **small-overlay backup**

### Current key performance snapshot
- hybrid_guarded_mode:
  - OOS return `+0.6868%`
  - OOS Sharpe `3.2370`
  - OOS max DD `0.2573%`
- balanced_overlay_mode:
  - OOS return `+0.1091%`
  - OOS Sharpe `0.4828`
  - OOS max DD `0.5162%`
- pair_tactical_mode:
  - OOS return `+0.2892%`
  - OOS Sharpe `3.2765`
  - but still tactical-only

## Replay / threshold analysis state
### Threshold frontier artifact
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/performance_first_threshold_frontier_current/performance_first_threshold_frontier_latest.md`

Key result:
- current performance-first override is valid
- largest thresholds that still keep hybrid promoted on the lightweight frontier:
  - max return-edge threshold `+0.5000%`
  - max sharpe-edge threshold `2.5000`
  - max min-val-return threshold `+6.0000%`
  - max min-val-sharpe threshold `3.0000`

### Switch replay artifact
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/performance_first_switch_replay_current/performance_first_switch_replay_latest.md`

Key result:
- replay now shows **both strict baseline and coverage-adjusted replay**
- market judgement coverage is fixed:
  - `market_judgement_days = 34 / 34`
- pair liquidity still has a strong coverage issue:
  - strict counts: `{"normal": 4, "strong": 5, "weak": 25}`
  - coverage-adjusted counts: `{"normal": 22, "strong": 5, "weak": 7}`
  - coverage-gap days: `18`

### Replay interpretation
- strict replay:
  - OOS return `+0.3400%`
  - Sharpe `1.6644`
  - mode_counts:
    - balanced `5`
    - core `20`
    - defensive `2`
    - hybrid `2`
    - risk_off `5`
- coverage-adjusted replay:
  - OOS return `+0.6839%`
  - Sharpe `3.4091`
  - mode_counts:
    - balanced `14`
    - core `9`
    - defensive `3`
    - hybrid `5`
    - risk_off `3`

Meaning:
- the replay is no longer broken by missing market-state history
- the main distortion now is **pair liquidity raw-data coverage before 2026-03-19**

## Important constraints to preserve
- total session memory must stay below **8 GiB**
- heavy runs must remain **strictly sequential**
- do **not** rerun article `batch_01~44`
- pair tactical remains **override-only**

## Files to read first next session
1. `docs/session_handoff_20260416_reboot_validation_rerun.md`
2. `docs/session_handoff_20260416_performance_first_switch_override.md`
3. `docs/performance_first_threshold_sensitivity_20260416.md`
4. `docs/session_handoff_20260416_hybrid_perf_replay_ready.md`
5. `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md`
6. `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/final_master_scoreboard_current/portfolio_master_scoreboard_latest.md`
7. `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/performance_first_threshold_frontier_current/performance_first_threshold_frontier_latest.md`
8. `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/performance_first_switch_replay_current/performance_first_switch_replay_latest.md`
9. `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json`

## Recommended next goal
Best next task:
- **re-tune the performance-first override thresholds using the coverage-adjusted replay as the primary path metric**

Recommended direction:
- keep pair tactical tactical-only
- do not change live policy directly from replay without checking one-shot switch and frontier together
- prefer threshold changes that:
  - preserve hybrid default on current one-shot state
  - do not materially degrade coverage-adjusted replay return / Sharpe
  - avoid overreacting to early OOS liquidity coverage gaps

## Exact prompt for the next session
```text
LuminaQuant 작업 재개. /home/hoky/Quants-agent/LuminaQuant 에서 시작.

먼저:
1) git status / git log 확인
2) docs/session_handoff_20260416_reboot_validation_rerun.md 읽기
3) docs/session_handoff_20260416_performance_first_switch_override.md 읽기
4) docs/performance_first_threshold_sensitivity_20260416.md 읽기
5) docs/session_handoff_20260416_hybrid_perf_replay_ready.md 읽기
6) current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md 읽기
7) final_master_scoreboard_current/portfolio_master_scoreboard_latest.md 읽기
8) current_switch_validation_current/performance_first_threshold_frontier_current/performance_first_threshold_frontier_latest.md 읽기
9) current_switch_validation_current/performance_first_switch_replay_current/performance_first_switch_replay_latest.md 읽기
10) portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json 읽기
11) 실행 중인 python/uv process 없는지 확인

현재 상태 요약:
- reboot split은 train 2025 / val 2026-01~02 / oos 2026-03~latest
- warmup_ratio=0.60 -> warmup_days=255
- current one-shot live default는 hybrid_guarded_mode
- pair는 tactical-only 유지
- frontier 상으로는 hybrid 승격 threshold가 성립
- replay는 market-state coverage는 정상화됐고, pair liquidity coverage gap이 주요 왜곡 요인
- strict replay보다 coverage-adjusted replay가 hybrid thesis를 더 잘 지지

이번 목표:
- performance-first override threshold를 coverage-adjusted replay 기준으로 재튜닝
- 단, pair tactical은 계속 tactical-only 유지
- one-shot switch / frontier / replay 세 관점을 동시에 보고 threshold를 조정

중요 제약:
- 세션 총합 memory 8GB 절대 초과 금지
- heavy run 병렬 금지
- 항상 sequential 실행만 허용
- article batch_01~44 재실행 금지
```
