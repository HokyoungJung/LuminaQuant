# New-session bootstrap prompt

Paste the following as the first message in the new coding session:

```text
/skill:ultragoal

이전 세션의 Strategy Recovery Ultragoal 실행을 정확히 이어서 재개해. 새 plan을 만들거나 완료된 작업을 반복하지 마.

작업 저장소는 /home/hoky/Quants-agent/LuminaQuant 이고 branch는 recovery/strategy-plan-20260714 이다. 구현 commit 66c85d5da2edbe42c8e9f359ea59582dd814f997 과 그 뒤의 handoff 문서 commit들을 보존하고, branch를 reset하지 마. 먼저 다음 파일을 전부 읽고 binding contract로 사용해:
- docs/research_note/strategy_recovery_session_handoff_20260714.md
- docs/research_note/strategy_recovery_resume_state_20260714.json
- docs/research_note/strategy_recovery_master_plan_20260713.md
- docs/research_note/data_pc_strategy_recovery_runbook_20260713.md
- docs/audits/strategy_reality_audit_20260713.md
- .gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json
- .gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl

기존 durable Ultragoal session id는 019f603a-0e73-7000-88a7-c94f42950c09 이다. 모든 native `gjc ultragoal ...` 명령에 환경변수 `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09`를 전달해 이 session state를 사용해. `create-goals`를 실행하지 마. 먼저 해당 session id로 `gjc ultragoal status --json`을 실행해서 다음 상태를 확인해:
- G001 complete
- G002, G008, G009, G010, G011 superseded
- G012 complete
- G003 active
- G004, G005, G006, G007 pending

inline aggregate goal은 사용자 요청에 따른 세션 이관 때문에 pause된 상태다. goal tool에서 paused로 보이면 resume하고, 새 세션이라 active goal이 없으면 아래 stable aggregate objective로 create해. 다른 objective가 active면 새 goal을 만들지 말고 충돌을 먼저 해소해:
`Complete the durable ultragoal plan in .gjc/ultragoal/goals.json, including later accepted/appended stories, under the original brief constraints; use .gjc/ultragoal/ledger.jsonl as the audit trail.`

재개 지점은 G003뿐이다. G001/G012 구현이나 검증을 반복하지 마. 별도 Alpha worktree는 /home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc, branch recovery/alpha-max-rev515-alignment-20260714, HEAD 629d91e5d4aac26911af65a4a5e15ebdcbded30f 이며 handoff 당시 clean하고 alignment edit은 하나도 없다.

G003에서 먼저 bounded executor에게 아래 두 파일만 맡겨 Rev5.15 runbook 정합화를 구현해:
1. docs/research_note/alpha_max_data_pc_runbook_20260711.md
2. docs/research_note/alpha_max_final_sha256_20260711.txt

정확한 변경 계약은 handoff note의 “G003 implementation still required” 절을 그대로 따른다. 핵심은 Rev5.15 identity/baseline, listing-aware config/contract/provenance/preparer와 네 SHA-256, runtime/config hashes, immutable six phase dates, TONUSDT official raw/feature intervals와 admission rejection, GRAM/synthetic/date/funding substitution 금지, listing-aware prelock command paths, 68/816/17/680 structural counts, final SHA manifest 갱신이다. 리더가 diff를 통합·검증하고 Alpha branch에 commit하되 push는 하지 마.

중요 안전 경계:
- /home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source 는 incomplete unauthorized partial로 명시적으로 quarantine되었다.
- blocker receipt는 /home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-phase-preparation-blocker.json 이다.
- 이 source를 consume/materialize/backfill하지 말고 phase-root preparation, prelock, historical evaluation을 실행하지 마.
- synthetic data, symbol substitution, pre-listing fill, missing funding proxy, locked-OOS reselection, paper/testnet/live order, capital allocation은 계속 금지다.
- 원본 데이터 root는 전부 read-only다.
- 어떤 branch도 명시적 사용자 지시 없이 push하지 마.

Alignment 후 exact hash/date/path 검사, focused Alpha config/preparer/runtime tests, cleanup, 적절한 full worktree gate, architect review, executor QA를 수행하고 기존 session id로 G003를 checkpoint해. G003 완료 후 durable 순서대로 G004 → G005 → G006 → G007을 계속 실행하되, Alpha A-03 데이터 경로는 authorized complete canonical source가 생기기 전 external blocker로 유지하고 독립 Router 작업은 계속해.

도구 사용 전 두 worktree의 branch/HEAD/clean status와 resume-state hash를 확인하고, 예상 밖 변경은 사용자 작업으로 보존해. 모든 진행/결정/외부 blocker는 기존 Ultragoal ledger와 /home/hoky/quants-recovery-runs/20260714T105113Z evidence root에 기록해.
```
