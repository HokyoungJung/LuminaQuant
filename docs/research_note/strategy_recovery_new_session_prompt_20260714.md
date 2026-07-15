# New-session bootstrap prompt

Paste the following as the first message in the new coding session:

```text
/skill:ultragoal

이전 Strategy Recovery durable Ultragoal을 정확히 이어서 재개해. 새 plan/create-goals를 만들거나 완료된 작업을 반복하지 마.

저장소는 `/home/hoky/Quants-agent/LuminaQuant`, branch는 `recovery/strategy-plan-20260714`이다. G004 구현 snapshot commit `f8ba7f1d`를 포함해야 하며 reset/rebase/amend하지 마. 먼저 아래 파일을 전부 읽고 binding contract로 사용해:

- `docs/research_note/strategy_recovery_session_handoff_20260714.md`
- `docs/research_note/strategy_recovery_resume_state_20260714.json`
- `docs/research_note/strategy_recovery_master_plan_20260713.md`
- `docs/research_note/data_pc_strategy_recovery_runbook_20260713.md`
- `docs/audits/strategy_reality_audit_20260713.md`
- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json`
- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl`

기존 durable session id는 `019f603a-0e73-7000-88a7-c94f42950c09`이다. 모든 native command에 `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09`를 설정해. `gjc ultragoal status --json`으로 다음 상태를 확인해:

- G001, G012, G017 complete
- G002/G003/G008–G011/G013–G016 superseded
- G004 review_blocked
- G018 active review-blocker story
- G005–G007 pending

inline aggregate goal이 세션 이관 때문에 pause되어 있으면 `goal resume`으로 재개해. objective는 다음 stable aggregate objective 하나뿐이다:
`Complete the durable ultragoal plan in .gjc/ultragoal/goals.json, including later accepted/appended stories, under the original brief constraints; use .gjc/ultragoal/ledger.jsonl as the audit trail.`

G018은 durable state에서 이미 active다. `complete-goals`를 다시 실행하지 말고 paused inline aggregate goal만 resume한 뒤 G018을 계속해. G005를 먼저 시작하지 마.

G004 snapshot은 일부 구현이지만 proof completion이 아니다. 마지막 snapshot gate는 다음과 같다:

`uv run pytest -q tests/research/test_cost_proof.py tests/research/test_router_replay.py tests/test_strategy_signal_dispatch.py tests/test_strategy_signal_dispatch_routing.py`
→ 72 passed

동일 파일의 Ruff check도 passed. 편집 전에 이 gate를 재실행해 checkout을 검증해.

G018 구현 순서는 handoff note의 “Mandatory G018 blocker plan”을 그대로 따른다:

A. strict dispatch end-to-end:
- mapped pair handler의 simulator 실패가 proxy exposure로 대체되지 않게 strict error/cause로 전파
- final portfolio_ret/turnover/exposure shape·finite 검증
- production numpy.datetime64[ms] cadence 계산
- malformed strict params를 typed StrategySignalDispatchError로 변환
- public call-chain/adversarial tests

B. router replay authenticity/determinism:
- prior-fold immutable evidence로 frozen branch/label/leaf decision 재계산
- 실제 receipt bytes 및 transitive engine/data/window/params identity 검증
- shared fallback MDD evidence로 R1 MDD30/cap3, R2 MDD20/cap2 scale 재계산
- commit/freeze provenance를 외부 authoritative root에 bind
- profile recursive finite/exact types, overflow STOP

C. cost proof independent reconciliation:
- router receipt commitment와 cost tape slice exact binding
- market/funding bytes 또는 authenticated row proof와 data/commit receipt semantic binding
- exact frozen combined profile digest pinning
- authenticated volume/ADV
- fill→cash/inventory→realized/unrealized PnL→equity reconciliation
- exact validation/purge/embargo/locked segment 및 explicit flattening
- profile-derived stop과 entry-bar event-aware stop/liquidation, OHLC ambiguity fail-close
- complete authenticated attempted/skipped/failed trial ledger
- whole-family SPA/max-statistic
- 각 exploit에 대한 STOP regression test

중요 contract decision:
- master plan R2의 selection rule은 각 후보가 binding gates를 통과한 후 둘 다 통과할 때 validation 20bp Calmar, 동률이면 낮은 MDD다. reviewer 제안만으로 이 규칙을 바꾸지 마.
- R1/R2는 정확히 두 historical candidates이고 candidate SHA는 `ddc8996136e70d3847e8270f6165a26992ec8def8439ba6f56e3bcdbdee239b9`다.
- `new_grid_search=false`, `recompute_from_json=false`, `post_oos_augment=false`, current-fold OOS input=0, generic fallback=0을 유지해.
- stronger proof에서 기존 Router가 실패하면 G005에서 scientific KILL한다. gate를 완화하거나 새 변형을 만들지 마.

안전 경계:
- G018 동안 strategy performance, grid search, data download/append, network, order, paper/testnet/live, capital 명령을 실행하지 마.
- 원본 data root는 read-only다.
- quarantined Alpha source `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source`를 읽거나 사용하지 마.
- Alpha blocker-v2는 `/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-phase-preparation-blocker-v2.json`이며 phase preparation/prelock/historical은 계속 금지다.
- 명시적 지시 없이 push하지 마.

G018 구현 후 focused/combined/full tests, Ruff/format/diff, AI-slop cleanup, evidence receipts, fresh architect 3-lane review, executor adversarial QA를 수행해. CLEAR/CLEAR/CLEAR + APPROVE와 executor PASS 전에는 G018/G004 complete checkpoint를 하지 마. 모든 진행과 blocker를 기존 ledger와 `/home/hoky/quants-recovery-runs/20260714T105113Z`에 남겨.
```
