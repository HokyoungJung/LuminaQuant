작업 저장소: /home/hoky/Quants-agent/LuminaQuant

이 메시지는 분석이나 계획 승인 요청이 아니라, 데이터 수집·결함 수정·검증·공용 DB 반영·기존 프레임워크 통합까지 완료하라는 명시적 실행 승인이다. 안전 불변식이 깨질 때만 fail-closed로 멈춰라. 이전 G073의 no-retry 경계는 이 새 작업에 한해 해제하지만, 실패한 G073 run을 재개하거나 재사용하지 말고 완전히 새로운 단일 recovery story/run을 만들어라. goal을 반복 생성하지 말고 하나의 durable ultragoal에서 끝까지 관리하라. 기존 G073 blocked 기록은 변경하지 않는다.

# 기준 상태

- 저장소: /home/hoky/Quants-agent/LuminaQuant
- 브랜치: recovery/strategy-plan-20260714
- 마지막 확인 HEAD: 46799fe3f1f6181153ecc0f9a429a11cd3fc2e2e
- 핵심 구현 commit: 3b0cc6be253e29ae3ca11fbb92a456bafa0eee8a
- 마지막 문서 commit: 46799fe3f1f6181153ecc0f9a429a11cd3fc2e2e
- 기존 공용 DB: data/market_parquet, 약 27.2GB
- 기존 대상 OHLCV 행: 467,877,392
- 목표 OHLCV 행: 1,066,681,730
- 부족 OHLCV 행: 598,804,338
- 부족 범위: 233/415 symbol-month
- funding 최종 목표: 39,569행
- 부족한 최종 압축 데이터 예상량: OHLCV 약 5.06GB, funding 포함 약 5.4GB
- fresh source-eligible staging 예상량: 약 11GB + 동시에 처리 중인 ZIP 1개 + scratch
- 마지막 Windows C: 실제 여유 공간: 37,861,703,680 bytes
- 절대 보존 reserve: 21,474,836,480 bytes(20GiB)
- WSL에서 보이는 약 710GB 여유는 동적 VHD 상한이므로 용량 판단에 사용하지 않는다.
- 기존 작업의 fresh 수집 행과 canonical DB write는 모두 0건이었다.
- alpha-max/G073 수집 unit, observer, telemetry, monitor는 모두 중지된 상태였다.
- OOM guard는 의도적으로 유지 중이다: MemoryHigh=5G, MemoryMax=7G, SwapMax=2G.

시작 즉시 실제 HEAD, branch, git status, active goal, systemd unit/job 상태와 Windows 호스트 여유 공간을 다시 확인하라. 상태가 달라도 reset, stash, revert, 삭제로 사용자 작업을 없애지 말고 현재 변경을 먼저 규명하라.

# 반드시 읽을 근거

1. docs/research_note/g070_v8_and_canonical_db_cross_session_handoff_20260726.md
2. /home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g073-v8-terminal-launch-failure-85be5b266630-v1.json
   - SHA-256: 3d9de04c55efb3a413daa5446925113f243ac5fee6f85e43ec17ff4861244d22
3. 같은 증거 디렉터리의 g073-late-agent-reconciliation-v1.json
   - SHA-256: e5cde281878da682b2b65156cbf5a9224babbb27e5bfa76f3d463882cee36c7a
4. 같은 증거 디렉터리의 g073-data-capacity-audit-v3.json
   - SHA-256: ed4d5d59e061d700321ae6b3b8d1de0a981cd253c0f5a337e38da6b059c6e971
5. 필요한 경우 다음 비밀 없는 영수증만 읽어 사실관계를 확인한다.
   - g073-v8-control-readback-85be5b266630-v2.json
   - g073-v8-prelaunch-85be5b266630-v1.json
   - g073-replacement-acquisition-authorization-v1.json
   - g073-v8-renderer-failure-cd0f446dd0e7-v1.json

v6/v7 private control, credential, 수집 데이터 또는 실패한 d999/cd0f/85be 실행 root의 control/data를 읽거나 재사용하지 말라. 위에 지정한 비밀 없는 문서와 영수증은 읽어도 된다. 새 run/request ID, signing material, credential, control, source/report/staging root를 암호학적으로 새로 생성한다. 예전 approval이나 성공 판정을 현재 HEAD에 재사용하지 않는다.

# 완료 목표

공식 원본에서 부족한 10개 target symbol/month와 funding을 수집하고 검증한다. 월별 원본 ZIP은 한 번에 하나만 보유하고 durable provenance receipt를 남긴 뒤 제거한다. 검증된 데이터는 별도의 영구 canonical DB를 만들지 말고 기존 data/market_parquet에 generation 단위로 원자 반영한다. 기존 loader, downsampling, feature, funding, strategy/backtest 프레임워크를 새 canonical generation에 연결하고 관련 회귀를 통과시킨다. 1초봉을 합성하거나 더 낮은 해상도의 데이터에서 재구성하지 않는다.

# 실행 순서

## 1. 단일 durable 작업과 안전 경계 설정

- 기존 paused aggregate goal/ledger를 검사한다.
- 기존 G073 blocked story는 그대로 보존하고, 새 replacement story/run 하나만 추가한다.
- 추가 goal을 계속 생성하지 않는다.
- 실제 Windows 호스트 여유 공간을 직접 측정하는 지속 monitor를 설정한다.
- 20GiB reserve를 침범하거나 projected peak가 reserve를 보장하지 못하면 다운로드와 publication을 시작하지 않는다.
- OOM guard와 cgroup 제한을 유지한다.
- 공용 DB에는 수집 완료 및 publication 승인 전까지 쓰지 않는다.

## 2. systemd credential launch 결함을 먼저 수정

이전 두 실행은 네트워크 전에 실패했다. 첫 번째는 `%d` credential specifier escape 문제였고, 두 번째는 systemd credential setup 단계의 243/CREDENTIALS, ENOENT였다.

- builder가 정적 systemd unit 파일을 생성하고 그 정확한 bytes/digest를 manifest에 결속하도록 한다.
- unit 계약에 최소한 다음을 포함한다.
  - UMask=0077
  - systemd가 해석하는 literal `%d`
  - 승인된 정확히 하나의 LoadCredential source/target 계약
  - 기존 sandbox, isolation, cgroup 및 resource 제한
- credential source의 존재, regular-file 여부, owner/group, mode, nlink, digest와 unit namespace에서의 target path를 검증한다.
- systemd-analyze verify만으로 성공으로 간주하지 않는다.
- 실제 production unit과 동일한 renderer/unit/credential/ExecStart 경로를 사용하는 비네트워크 synthetic end-to-end probe를 추가하고 실행한다.
- probe는 credential 전달과 실제 ExecStart 진입을 입증해야 한다.
- probe가 실패하면 fresh source root나 네트워크를 열지 말고 terminal evidence를 남긴다.

## 3. atomic publisher의 HIGH blocker 수정

다음 문제를 공통 저장소/loader 경로에서 해결한다. 호출부별 임시 우회나 빈 fallback을 만들지 않는다.

1. prepared resume에서 정상적인 CoW/reflink 전환으로 nlink가 변하는 경우를 안전하게 판정하되 source identity, digest, inode 독립성 검증을 약화하지 않는다.
2. merge reserve는 실제 temporary file, merged output, fsync, exchange 및 rollback/staging 보존 공간의 peak를 보수적으로 계산한다.
3. reader는 shared generation lock 아래 논리 generation을 pin하거나 재해석하고, 교환 중 empty/mixed generation을 반환하지 않는다.
4. raw, WAL, upsert, compaction, funding, feature writer가 모두 같은 global generation lock 계약을 사용한다.
5. listing/metadata도 데이터와 같은 generation에 포함해 old-or-new로 전환한다.
6. staging은 canonical과 독립 inode여야 하고 publication 뒤 provenance로 보존한다.
7. 최상위 generation의 renameat2(RENAME_EXCHANGE)만 canonical activation 지점으로 사용한다.
8. 실패나 crash에서 기존 canonical은 그대로 읽을 수 있어야 하고 partial generation은 활성화되지 않아야 한다.

필수 테스트:

- concurrent reader가 오직 complete old 또는 complete new generation만 관찰
- exchange 직전/직후 crash와 prepared resume
- 준비 중 source mutation
- WAL/upsert/feature/funding/compaction 동시 쓰기
- bootstrap 및 symlink generation
- listing/metadata와 data의 동일 generation 가시성
- staging 보존과 독립 inode
- reserve 부족 시 fail-closed 및 canonical 무변경
- publication failure 후 loader가 빈 결과로 fallback하지 않음

focused test, 관련 storage/market-data 회귀, lint, formatter check, git diff --check를 실행한다. 수정사항을 커밋하고 clean HEAD에 결속된 fresh approval/control을 만든다. 독립 review는 한 번, 필요시 repair pass도 최대 한 번만 허용한다. HIGH blocker가 남으면 수집이나 publication으로 진행하지 않는다.

## 4. 완전히 새로운 control과 실행 root 생성

- 새 random run ID, request ID, signing key/credential, source/report/staging root를 생성한다.
- control COMPLETE/readback, manifest, HEAD binding, inventory, key binding, permission, byte count와 digest를 독립 검증한다.
- authority만 signing credential을 볼 수 있어야 하며 telemetry/observer/log에 secret이 노출되면 안 된다.
- failed run의 root, socket, credential, signed receipt, phase state를 복사하지 않는다.
- authority → telemetry → observer 순서로만 기동한다.

## 5. 공식 데이터 순차 수집

- 기존 frozen inventory에서 정확한 10개 symbol과 요구 month를 읽고 임의로 범위를 추정하지 않는다.
- 공식 source에서만 다운로드한다.
- 원본 ZIP은 동시에 정확히 하나만 유지한다.
- 첫 archive를 pilot으로 다음 전체 경로를 검증한다.
  1. 공식 원본 다운로드
  2. source URL/HTTP metadata/size/digest 기록
  3. archive 구조와 entry 검증
  4. production parser로 streaming 변환
  5. schema, symbol, timestamp, row count, duplicate/order 검증
  6. staging Parquet durable write와 fsync
  7. signed/durable provenance receipt 생성 및 readback
  8. receipt와 staging이 durable함을 확인한 뒤 원본 ZIP 삭제
- pilot에서 canonical DB write가 0건이고 credential, memory, scratch peak, host reserve, receipt가 모두 정상인지 확인한다.
- 객관적 gate가 통과하면 별도의 human approval을 기다리지 말고 같은 새 story/run에서 나머지 부족 archive를 하나씩 순차 처리한다.
- 월별 ZIP을 영구 누적하지 않는다.
- 각 archive 완료 후 Windows 호스트 여유, WSL memory/swap, scratch와 staging 사용량을 다시 확인한다.
- 20GiB reserve 침범, OOM pressure, manifest/source/digest/schema 불일치가 발생하면 즉시 fail-closed로 종료하고 unit/socket/credential만 격리 정리한다.
- 자동 retry loop를 만들거나 실패한 archive/run을 성공으로 재사용하지 않는다.

## 6. publication 전 완전성 검증

공용 DB를 건드리기 전에 staging과 기존 canonical의 합집합이 다음을 모두 만족해야 한다.

- OHLCV 정확히 1,066,681,730행
- funding 정확히 39,569행
- target 10개 symbol의 요구 415 symbol-month 전체 완성
- 중복, 누락, 역전 timestamp 없음
- schema, dtype, timezone, partition, 가격/수량 invariant 충족
- 각 source archive의 공식 출처, size, digest, parser/tool version 및 결과 행 통계 receipt 존재
- signed terminal success receipt와 독립 readback 성공
- 기존 canonical DB는 이 시점까지 mutation 0건

하나라도 충족하지 못하면 publication하지 않는다. 숫자를 맞추려고 synthetic row, silent dedup, 빈 partition 또는 기존 데이터 삭제를 사용하지 않는다.

## 7. 기존 data/market_parquet에 원자 통합

- 영구적인 두 번째 canonical DB를 만들지 않는다.
- global generation lock 아래 기존 canonical과 검증된 staging을 merge한다.
- reflink/CoW 가능 여부와 실제 peak 공간을 측정하고, reserve audit가 통과하지 않으면 교환을 시작하지 않는다.
- 새 generation 전체를 검증하고 file/directory durability를 확보한 뒤 단 한 번의 최상위 RENAME_EXCHANGE로 활성화한다.
- reader는 교환 전후에 complete old 또는 complete new만 관찰해야 한다.
- old generation과 staging/provenance의 보존/정리 방식은 승인된 atomic protocol과 reserve 계산을 따라야 한다.
- 실패하면 old canonical을 계속 서비스하고 partial new generation을 활성화하지 않는다.

## 8. 기존 프레임워크 통합

새 저장소 추상화나 별도 데이터 경로를 만들지 말고 기존 API와 패턴을 재사용한다.

- 기존 market-data loader가 활성 canonical generation을 읽도록 보장한다.
- 기존 downsampling 경로를 사용한다.
- funding, feature, WAL/upsert, compaction 호출부를 generation/lock 계약에 맞춘다.
- 직접 영향받는 strategy/backtest/research callsite를 갱신한다.
- 1초봉 source-of-truth, timezone, partition와 public API 호환성을 유지한다.
- 병렬 reader, feature 계산, funding 조회, downsampling, representative backtest로 실제 통합을 검증한다.

## 9. 최종 검증, 증거, 종료

publication 후 다음을 독립 재계산한다.

- OHLCV/funding 최종 행 수와 symbol/month coverage
- timestamp/schema/partition invariant
- publication 전후 generation digest
- loader/downsampling/feature/funding/backtest 결과
- 동시 reader/writer old-or-new 가시성
- Windows host disk low-water mark와 최종 여유 공간

관련 focused 및 broader regression, lint, format check, git diff --check를 통과시킨다. handoff 문서와 durable ultragoal ledger에 실제 결과와 evidence path를 기록한다. 코드/문서 변경을 커밋하고 clean status를 확인한다. 임시 unit, monitor, socket, credential, scratch만 정리하고 canonical, 필요한 staging/provenance, terminal evidence와 OOM guard는 보존한다. 사용자 작업이나 기존 복구 증거를 삭제하지 않는다.

# 완료 보고 형식

반드시 다음을 실제 관측값으로 보고한다.

- 최종 branch와 HEAD
- 새 run/request ID
- 수집·통합한 symbol/month 목록과 행 수
- 최종 OHLCV 및 funding 행 수
- publication 전후 generation digest
- canonical mutation이 atomic complete인지 여부
- Windows 호스트 최소/최종 여유 공간
- peak memory/swap
- 실행한 테스트와 정확한 결과
- signed receipt, provenance, capacity audit, handoff 경로와 SHA-256
- 남은 blocker가 있는지 여부
- 최종 git status

실패한 경우 재시도를 임의 반복하지 말고 정확한 실패 단계, 원인, terminal evidence, canonical mutation이 0건인지 또는 부분 변경인지 보고한다. 정상 gate를 통과하는 동안에는 중간 상태 보고만 하고 멈추지 말고 수집부터 기존 프레임워크 통합과 최종 검증까지 완료한다.
