/skill:ultragoal

`/home/hoky/Quants-agent/LuminaQuant`의 중단된 Alpha-Max recovery를 durable Ultragoal session `019fad7d-536a-7000-b794-52ccaa961746`, goal `G001`에서 재개하고 전체 작업을 끝까지 완료하라.

이 프롬프트는 새롭고 완전히 분리된 공식-network acquisition run을 생성·실행하고, 성공한 staging의 감사, WAL 반영, 충돌 승인, 원자적 canonical publication, 기존 loader/feature/funding/backtest 통합, 전체 회귀검증, 정리, 문서·ledger·Git 완료까지 수행하라는 명시적 실행 승인이다. 중간 실패나 monitor timeout만으로 작업을 멈추거나 추가 승인을 요구하지 마라. 실패한 산출물을 재사용하지 말고 원인을 고친 뒤 새 ID/key/root로 안전한 fresh run을 진행하라.

모든 native Ultragoal 명령에 다음을 설정하라.

```bash
export GJC_SESSION_ID=019fad7d-536a-7000-b794-52ccaa961746
```

새 plan이나 goal을 만들지 마라. `create-goals`와 `complete-goals`를 실행하지 마라. 기존 G001과 ledger가 유일한 durable 작업이다.

작업 전에 다음 파일을 전부 읽어라.

1. `docs/research_note/alpha_max_recovery_session_handoff_20260802.md`
2. `docs/research_note/alpha_max_recovery_resume_plan_20260802.md`
3. `docs/research_note/alpha_max_recovery_new_session_prompt_20260802.md`
4. `docs/research_note/alpha_max_recovery_resume_state_20260802.json`
5. `.gjc/_session-019fad7d-536a-7000-b794-52ccaa961746/ultragoal/brief.md`
6. `.gjc/_session-019fad7d-536a-7000-b794-52ccaa961746/ultragoal/goals.json`
7. `.gjc/_session-019fad7d-536a-7000-b794-52ccaa961746/ultragoal/ledger.jsonl`
8. `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/user-interrupted-acquisition-47eeac483e70-v1.json`
9. `/home/hoky/quants-recovery-runs/g056v8-acquisition-evidence-47eeac483e70d6af0784b873895024c8a2d01793a2447d3d3fbaa776d63bd2ad/terminal-authority.receipt.json`

필수 초기 검증:

1. branch가 `recovery/strategy-plan-20260714`인지 확인한다.
2. `HEAD == git rev-parse recovery-session-handoff-20260802^{commit}`인지 확인한다.
3. HEAD가 product/recovery checkpoint `f518dc7bed5cb416dba27886b2337f4a31ea7650`의 descendant인지 확인한다.
4. worktree가 clean인지 확인한다. reset/stash/revert로 사용자 작업을 없애지 마라.
5. resume state에 기록된 파일 SHA-256, byte count, line count를 확인한다.
6. `GJC_SESSION_ID=019fad7d-536a-7000-b794-52ccaa961746 gjc ultragoal status --json`으로 G001이 paused/active incomplete이며 마지막 ledger가 이 handoff pause와 결속됐는지 확인한다.
7. `goal get` 후 paused라면 `goal resume`을 호출한다. 새 thread에 inline goal이 전혀 없을 때만 아래 정확한 objective로 `goal create`를 호출한다.

`Complete the durable ultragoal plan in .gjc/ultragoal/goals.json, including later accepted/appended stories, under the original brief constraints; use .gjc/ultragoal/ledger.jsonl as the audit trail.`

8. acquisition/authority/observer/telemetry/capacity/publication monitor와 관련 child process가 모두 중지돼 있는지 확인한다.
9. Windows sleep policy가 AC 900초/DC 600초인지 확인한다.
10. canonical root identity가 `[2096,195868,493,7]`이고 user interruption 이후 generation/publication mutation이 없었는지 확인한다.

중요한 상태:

- 승인된 publication/evidence repair는 Git ancestry의 `f518dc7b`에 통합돼 있다.
- 최종 review는 `agent://44-FinalEvidenceProtocolReview`, `agent://45-FinalRecoveryControlsReview`이며 둘 다 CLEAR/APPROVE이다.
- `47eeac...` run은 사용자 요청으로 SIGTERM 종료됐고 signed state `FAILED`, return code `-15`이다.
- 중단 시 raw `260/415`, funding `7668/12347`이었으나 partial source와 live ZIP은 cleanup receipt 이후 삭제됐다.
- `47eeac...`와 `6fefca...`의 데이터는 publication이나 새 run에 절대 재사용하지 않는다.
- canonical publication, WAL compaction, conflict authorization은 아직 수행되지 않았다.
- 기존 canonical은 그대로이며 root identity는 `[2096,195868,493,7]`, 측정 크기는 `26,909,667,462` bytes이다.
- ten-symbol WAL planning total은 `3,967,207` records이다.
- 계획 참조 conflict는 raw 56 partitions/649,585 rows, funding conflict 0이지만 fresh success 이후 반드시 재계산한다.
- Windows 20GiB reserve와 cgroup/OOM guard를 유지한다.
- ignored local `uv.lock`을 frozen-lock test를 통과시키려고 수정하거나 force-add하지 않는다.

실행 순서는 resume plan을 그대로 따른다.

1. final HEAD에 recovery control commit pins를 기계적으로 갱신하고 focused verification을 다시 통과시킨다.
2. 새 random run/request IDs, signing keys, credentials, controls, source/report/evidence/telemetry roots를 만든다.
3. capacity gate 후 Windows sleep을 임시로 비활성화하고 authority → telemetry → observer 순서로 fresh official acquisition을 실행한다.
4. 정확히 raw 415 / 1,066,681,730 rows, funding 12,347 partitions / 39,569 rows, total 12,762 parquet contract를 signed `SUCCEEDED`로 완주한다.
5. fresh audit PASS receipt를 봉인한다.
6. WAL rehearsal과 execute를 수행하고 열 개 WAL이 모두 empty인지 검증한다.
7. fresh conflict authorization과 exact atomic-merge capacity audit를 생성한다.
8. capacity guard와 publisher를 실행해 canonical을 원자 교체하고 동일 입력 replay의 멱등성을 검증한다.
9. post verifier로 loader/downsampling/funding/feature/chunk/panel/legacy backtest를 검증한다.
10. focused 및 broad regression, Ruff, diff check를 실행한다.
11. 임시 unit/socket/private key/scratch/non-reusable source만 receipt와 함께 정리한다.
12. Windows sleep을 AC 900/DC 600초로 복구한다.
13. research note, resume/completion state, durable ledger를 갱신하고 commit/clean status를 확인한 뒤에만 G001과 aggregate goal을 완료한다.

`47eeac...`의 signed interruption receipt SHA-256은 `c8cda120cb0bd41da6da706281bfa4213fa0ad620773211423712effe0a543cc`, handoff receipt SHA-256은 `73a4c20623bc29e2af7a3853c9d5071f0bb5ea7e6d07e88e4ffcb35df55d0c43`이다. 이 identity가 다르면 실행하지 말고 현재 상태를 규명하라.

작업을 축소하거나 중간 보고만 하고 끝내지 마라. 실제 완료 증거가 생길 때까지 계속 실행하라.
