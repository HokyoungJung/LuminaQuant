# New-session bootstrap prompt

Paste the following as the first message in the new coding session:

```text
/skill:ultragoal

이전 Strategy Recovery durable Ultragoal을 정확히 이어서 재개해. 새 plan/create-goals를 만들거나 완료된 작업을 반복하지 마. 저장소는 `/home/hoky/Quants-agent/LuminaQuant`, branch는 `recovery/strategy-plan-20260714`이고, 최소한 implementation checkpoint `512d2b804ab05bc1cf023ac4cbea81e0506a8736` (`Checkpoint G019 proof contract repairs`)와 이 prompt를 포함한 최신 handoff commit을 포함해야 한다. reset/rebase/amend/push하지 마.

가장 먼저 아래 파일을 전부 읽고 서로 함께 binding plan/contract로 사용해:

- `docs/research_note/strategy_recovery_session_handoff_20260714.md`
- `docs/research_note/strategy_recovery_resume_state_20260714.json`
- `docs/research_note/strategy_recovery_master_plan_20260713.md`
- `docs/research_note/data_pc_strategy_recovery_runbook_20260713.md`
- `docs/audits/strategy_reality_audit_20260713.md`
- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/brief.md`
- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json`
- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl`

기존 durable session id는 `019f603a-0e73-7000-88a7-c94f42950c09`다. 모든 native command에 `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09`를 설정하고 `gjc ultragoal status --json`으로 실제 상태를 확인해. 기대 상태는 다음과 같다:

- G001, G012, G017 complete
- G002/G003/G008–G011/G013–G016 superseded
- G004 review_blocked
- G018 review_blocked
- G019 active review-blocker replacement story
- G005–G007 pending

inline aggregate goal은 세션 이관 때문에 `paused` 상태다. `goal({"op":"get"})`으로 확인하고 `goal({"op":"resume"})`으로 재개해. objective는 다음 stable aggregate objective 하나뿐이다:

`Complete the durable ultragoal plan in .gjc/ultragoal/goals.json, including later accepted/appended stories, under the original brief constraints; use .gjc/ultragoal/ledger.jsonl as the audit trail.`

Durable G019는 이미 active다. `complete-goals`를 실행하지 말고 paused inline aggregate goal만 resume한 뒤 G019를 계속해. G005를 먼저 시작하지 마. 이전 세션의 Cost architect task 97은 verdict 없이 pause되었으므로 결과로 인정하지 말고 새 architect review를 다시 실행해.

현재 checkpoint의 구현 내용:

1. Strict dispatch
- mapped actual-engine 실패를 proxy exposure로 숨기지 않고 typed cause로 전파
- final portfolio return/turnover/exposure shape와 finite 검증
- NumPy datetime64[ms] cadence 처리
- malformed strict params의 StrategySignalDispatchError 변환
- 38 focused tests passed, dispatch architect CLEAR/CLEAR/CLEAR APPROVE

2. Router replay
- source-first deterministic selection, prior-fold chronology, receipt bytes와 transitive engine/data/window identity, out-of-band source/commit roots, exact profile/type/overflow, artifact closure, Router cost ownership
- authenticated shared returns로 R1/R2 fallback scale 재계산
- base PPM domain은 동결하고 derived position/return만 최대 3x signed range 허용
- MDD가 pre-period equity에서 시작
- authenticated negative-scaled PASS와 initial-loss stale-scale STOP regression
- 57 tests passed after formatting
- final Router architect task 95: CLEAR/CLEAR/CLEAR APPROVE

3. Cost proof/G019
- Router receipt/tape ownership, source/profile/market/funding/trial roots, exact ordered rows
- volume/ADV/tick/step/sqrt impact/funding/cash-inventory-fill-price accounting/segment continuity/stop/liquidation reconciliation
- liquidation causality는 bar extreme이 아니라 authenticated immediate event-state mark로 계산
- post-event breach가 남으면 다음 action은 liquidation만 허용; residual breach 동안 consecutive liquidation만 허용
- valid full breached terminal liquidation은 zero-position explicit exit로 인정하고 economic REJECT; healthy liquidation은 STOP
- oversized native number는 `_num`과 `evaluate_cost_proof_file` ArithmeticError boundary에서 STOP
- whole-family Hansen-style SPA: shared deterministic 2,000 circular-block draws, add-one, original zero-scale member p=1, nondegenerate member의 positive degenerate bootstrap sample은 +inf로 conservative exceedance
- fully rooted public file/real CLI overflow STOP/exit 2
- both candidates × 10/15/20/30bp full-liquidation REJECT fixture
- 1% partial liquidation 후 99% residual breach에서 non-liquidation action STOP fixture; 최종 strengthening은 carried-breach guard line 2264까지 trace됨
- healthy liquidation은 causality guard line 2323까지 trace됨
- full Cost suite는 최종 residual-fixture strengthening 직전 61 passed; 최종 test-only edit 후 전체 suite/format은 새 세션에서 재실행해야 함

재개 직후 순서:

A. Checkout와 현재 상태 검증
- `git branch --show-current`, `git rev-parse HEAD`, `git status --short`, `git log -3 --oneline` 확인
- branch가 checkpoint `512d2b80`와 handoff commit을 포함하고 worktree가 clean인지 확인
- 예상 밖 변경은 user work로 취급하고 되돌리지 마

B. G019 current-state verification
- 다음 9개 파일에 Ruff format/check와 `git diff --check` 실행:
  - `src/lumina_quant/cli/research.py`
  - `src/lumina_quant/research/cost_proof.py`
  - `src/lumina_quant/research/router_replay.py`
  - `src/lumina_quant/strategy_factory/research_runner.py`
  - `src/lumina_quant/strategy_factory/strategy_signal_dispatch.py`
  - `tests/research/test_cost_proof.py`
  - `tests/research/test_router_replay.py`
  - `tests/test_strategy_signal_dispatch.py`
  - `tests/test_strategy_signal_dispatch_routing.py`
- `uv run pytest -q tests/research/test_cost_proof.py` 실행; 현재 예상 61 passed
- `uv run pytest -q tests/research/test_router_replay.py` 실행; 현재 예상 57 passed
- `uv run pytest -q tests/test_strategy_signal_dispatch.py tests/test_strategy_signal_dispatch_routing.py` 실행; 현재 예상 38 passed
- `uv run pytest -q tests/test_research_profile_activation.py tests/test_research_selection_flags_config.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py -k 'not test_shipped_config_yaml_full_load_is_byte_identical_to_head'` 실행; 현재 예상 79 passed, 1 deselected
- 제외된 config test는 local user/environment-managed `config.yaml`이 HEAD와 다른 별도 상태 때문이다. config를 수정하거나 값을 출력하지 마.

C. Cost final certification
- 새 architect에게 certification 92의 잔여 항목 7/10/12/13을 정확히 재검토시켜:
  - immediate liquidation causality와 carried breach ordering
  - valid full liquidation REJECT 및 healthy liquidation STOP
  - full exit로 liquidation을 허용해도 zero final position/non-entry/exact count/ownership/accounting/cause가 유지되는지
  - oversized market/funding number가 in-memory와 actual file/CLI boundary에서 STOP인지
  - positive degenerate bootstrap resample의 conservative SPA와 exact deterministic reference
  - 테스트가 count/hash mismatch 같은 더 이른 invariant에서 우연히 STOP하지 않는지
- CLEAR/CLEAR/CLEAR + APPROVE가 아니면 checkpoint하지 말고 `gjc ultragoal record-review-blockers`로 기록하고 bounded executor로 고친 뒤 full loop를 재실행해.

D. Final G018/G019 gate
- dispatch/Router/Cost/profile/selection/monthly-refit combined regression 실행
- Ultragoal internal AI-slop cleaner를 정확히 위 9개 changed files에 실행; blocking finding은 executor로 고치고 zero blockers까지 rerun
- cleaned change set을 freeze한 뒤 verification 재실행
- complete dispatch/Router/Cost contract에 fresh 3-lane architect review
- executor adversarial QA/red-team. Parent가 실제 API/package test-report와 CLI-surface artifact를 만들어야 하며 inline text만으로 증명하지 마.
- strict quality gate는 architect CLEAR/CLEAR/CLEAR, APPROVE, executor QA passed, e2e/red-team passed, fullRerun=true, blockers=[], 실제 artifact refs를 요구한다.
- 그 뒤에만 G019를 `checkpoint --status complete --quality-gate-json ...`으로 완료해. Later goals가 남아 있으므로 inline `goal complete`는 호출하지 마.
- G019 receipt가 생긴 후 G018을 G019 replacement evidence로 supersede하고, G004를 completed G018/G019 chain evidence로 supersede해. blocked history/ledger는 삭제하지 마.

E. 이어지는 frozen durable plan
- Phase D가 clean하게 끝난 뒤에만 G005를 계속해: bounded owned-data repair와 R-04/A-03 one-touch scientific decisions. source read-only/no-retuning/no-substitution/no-order 규칙 유지.
- R-04와 A-03가 terminal인 뒤 G006: preregistered bounded follow-up alpha/volatility cycle, complete trial ledger, locked-OOS report-only.
- 통과한 champion에 한해서 G007: zero orders/zero capital fresh-forward, 30-day checkpoint, 60-day terminal PASS/KILL.
- 모든 required durable goal과 fresh final aggregate receipt가 존재하기 전에는 aggregate goal을 complete하지 마.

고정 contract:
- R1/R2 exact candidate SHA: `ddc8996136e70d3847e8270f6165a26992ec8def8439ba6f56e3bcdbdee239b9`
- 각 후보를 독립적으로 gate한 뒤 passer만 validation 20bp Calmar, lower validation MDD, frozen candidate order로 선택
- locked OOS는 report/gate only, selection/retune 금지
- `post_oos_research_variant=true`, `post_oos_augment=false`, augmentation/current-fold-OOS/grid/recompute zero/false, generic fallback=0

안전 경계:
- G019/G018 certification 중 strategy performance, grid search, data download/append, network, order, paper/testnet/live, capital/scientific execution 명령 금지
- 원본 data root read-only
- quarantined Alpha source `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source` 사용 금지
- Alpha blocker-v2 `/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-phase-preparation-blocker-v2.json`가 STOP인 동안 phase preparation/prelock/historical 금지
- 사용자 명시 지시 없이 reset/rebase/amend/push 금지
- 진행, blocker, review, checkpoint evidence는 기존 session ledger에 append하고 runtime goals/ledger를 수동 편집하지 마.
```
