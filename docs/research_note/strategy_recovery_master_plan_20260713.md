# Strategy Recovery Master Plan

작성일: 2026-07-13
상태: 실행 준비 계획, 실자본 배분 0%
근거 감사: [`docs/audits/strategy_reality_audit_20260713.md`](../audits/strategy_reality_audit_20260713.md)

## 1. 목적과 종료 조건

이 계획은 새 전략을 더 만드는 계획이 아니다. 다음 세 트랙을 실제 데이터와 수정된 평가 경로로 한 번만 다시 측정하고, 통과하지 못하면 같은 표본에서 추가 튜닝하지 않고 종료하는 계획이다.

1. 고-CAGR router R1: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`
2. 고-CAGR router R2: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2`
3. Alpha-Max Revision 5.15 listing-aware 프로토콜

완료 조건은 데이터 PC가 재현 가능한 입력 묶음과 실행 기록을 만들고, 각 트랙이 과학적 통과 또는 종료 판정을 받은 뒤 최대 두 후보만 fresh-forward로 넘기는 것이다. 이 계획의 완료 자체는 paper, testnet, live 또는 실자본 승인을 뜻하지 않는다.

## 2. 현재 판단

### 2.1 고-CAGR 전략은 포함한다

기존 headline을 버리는 것이 아니라 가장 강했던 두 router만 보존한다.

| 후보 | 과거 진단 headline | 현재 지위 |
|---|---:|---|
| R1 exact-unscaled | 10-fold comp `+159.83%`, ann approx `+214.51%`, MDD `27.69%`, hit `4/10`; 별도 exposed replay는 comp `+197.37%` | post-OOS 연구 진단, 미검증 |
| R2 fallback-mdd20-cap2 | comp `+138.11%`, ann approx `+183.23%`, MDD `23.58%`, hit `4/10` | 위험절감 연구 진단, 미검증 |
| 최신 11-fold R1 계열 | comp `+63.36%`, ann approx `+70.81%` | `clean_promotion_eligible=false`, fresh-forward 필수 |

개선 목표는 과거 CAGR을 더 키우는 것이 아니다. 실제 전략 라우팅, point-in-time 유니버스, 실제 funding, 비용, 노출 정규화와 fold 안정성을 고쳐 그 CAGR 중 재현 가능한 부분이 있는지 확인한다.

### 2.2 날짜 정책

- Router는 기존 runner의 원래 경계인 train start `2025-01-01`, first OOS `2025-09-01`을 유지한다. Router 때문에 2024년부터 전량 백필하지 않는다.
- Alpha-Max는 원래 기간을 유지한다. warmup이나 train 시작을 뒤로 미루지 않는다.
- 데이터가 더 오래 존재하면 보존하지만, 존재한다는 이유만으로 discovery 기간을 넓히지 않는다.

### 2.3 백필의 정의

이 계획에서 백필은 **인벤토리로 확인된 owned interval의 실제 결손을 공식 원천으로 채우는 것**이다. 다음은 백필이 아니다.

- 현재 스냅샷 110개 전체를 무조건 오래 수집하기
- 없는 상장 전 구간을 0, forward-fill 또는 합성 가격으로 만들기
- Alpha-Max의 날짜를 바꾸거나 TONUSDT를 다른 symbol로 대체하기
- 기존 funding row가 없는데 파생 열만 계산해 실제 funding을 대체하기

## 3. 불변조건

1. 신규 indicator, ML, overlay, router grid와 인접 후보 대체를 추가하지 않는다.
2. 모든 단계의 실자본 배분은 0%다.
3. synthetic `data/BTCUSDT.csv`, `data/ETHUSDT.csv`는 성과 실험에서 제외한다.
4. operational STOP과 scientific KILL을 구분한다.
   - STOP: 디스크, 네트워크, checksum, 프로세스 실패처럼 입력을 고치고 새 output ID로 재실행할 수 있는 문제
   - KILL: 유효한 실행이 binding gate를 실패하거나 원본 전략을 유일하게 복원할 수 없는 문제
5. exposed historical 결과는 diagnostic일 뿐 selection이나 배분에 사용하지 않는다.
6. result가 좋아도 같은 과거 구간에서 파라미터, universe, risk cap을 다시 고르지 않는다.

## 4. 알려진 선행 결함

| ID | 결함 | 영향 | 필요한 조치 |
|---|---|---|---|
| G-01 | `LQ_CONFIG_PATH`는 root replacement이며 profile merge가 아님 | 비용 profile만 쓰면 registry routing flag가 사라질 수 있음 | cost-realistic 기준의 단일 strict combined profile과 hash 생성 |
| G-02 | 고-CAGR CLI는 family 단위만 선택하고 R1/R2 exact ID replay 입력이 없음 | full rerun이 다시 grid search가 될 수 있음 | exact 두 candidate만 받는 frozen manifest replay seam 추가 |
| G-03 | 과거 router artifact에 faithful actual-engine leaf manifest가 없음 | headline을 실제 전략 실행으로 유일하게 복원하지 못할 수 있음 | fold별 class/params/symbol/timeframe/weights/gross/decision-clock 복원; 불가능하면 router KILL |
| G-04 | point-in-time lifecycle registry가 없음 | survivorship와 listing-season 혼입 | symbol lifecycle registry와 fold별 membership manifest 구현 |
| G-05 | monthly router의 비용 stress는 선형 proxy 10/15/20bp | final cost proof가 아님 | frozen signal에 actual funding, sqrt impact, 10/15/20/30bp 적용 |
| G-06 | main의 `scripts/materialize_market_windows.py --help`가 config view 오류로 실패 | raw-first final proof 준비 불가 | CLI 회귀 테스트 후 최소 수정 |
| G-07 | Alpha-Max branch의 공식 data-PC runbook은 Rev5.14 파일과 hash를 지시 | 최신 Rev5.15를 그대로 실행할 수 없음 | runbook을 Rev5.15에 맞춰 수정·검증·push하기 전 prelock 실행 금지 |
| G-08 | 현재 coverage/inventory CLI는 row count와 first/last 중심이며 duplicate, nonmonotone, nonfinite, expected-grid gap, funding settlement gap을 fail-close하지 않음 | 단순 coverage JSON을 데이터 무결성 증명으로 오인할 수 있음 | 기존 `validate_ohlcv_frame`을 재사용하고 expected 1m grid와 실제 funding cadence를 추가한 단일 JSON validation receipt CLI 구현 |

## 5. 실행 단계

### Phase D0 — 데이터 PC 인벤토리와 원본 복구

**입력**

- main의 정확한 실행 commit
- Alpha-Max `feat/alpha-max-20260710` commit `629d91e5d4aac26911af65a4a5e15ebdcbded30f`
- 기존 parquet, raw aggTrades/1s, feature/funding root
- 거래소 listing/delivery와 source provenance

**작업**

1. 기존 데이터 root를 먼저 찾고 read-only 원본으로 동결한다.
2. 물리 파일 인벤토리와 실제 repository loader coverage를 각각 만든다.
3. coverage/inventory 결과는 triage로만 기록하고, D-01A validation receipt로 symbol/timeframe별 duplicate, nonmonotone, nonfinite, expected-grid gap과 funding settlement gap을 검증한다.
4. 현재 static universe와 별개로 onboard/delivery/delist provenance gap을 기록한다.
5. 복구 가능한 기존 백업을 새 canonical root에 복사하고 원본은 수정하지 않는다.

**산출물**

- `physical-coverage.json`
- `repository-coverage.json`
- `strategy-support-inventory.json`과 CSV
- `data-gap-ledger.md`
- source path, Git SHA, 명령, 환경, 파일 inventory를 포함한 run record

**Acceptance**

- 실제 repository loader가 필요한 timeframe을 읽는다.
- synthetic source가 0건이다.
- 원본 root는 read-only이며 run output과 분리된다.
- 각 fold에서 membership과 data availability를 판정할 수 있다.
- D-01A validator가 pre-append에서 interior/prefix gap 0과 tail-only 여부를 JSON으로 판정한다.

**STOP**: source hash drift, symlink/multilink, 부분 복구, 디스크 부족.
**KILL**: 없음. 이 단계는 결손을 기록한다.

### Phase D1 — 결손 기반 최소 백필

순서는 다음으로 고정한다.

1. Router frozen symbol의 공식 1m OHLCV를 original window에 맞춰 복구한다.
2. 실제 funding settlement와 listing interval을 복구한다.
3. 4h/1d는 1m에서 결정적으로 파생하고 직접 서로 다른 원천을 섞지 않는다.
4. bar-level gate를 통과한 finalist만 raw aggTrades/1s와 체결 현실성 데이터를 준비한다.
5. Alpha-Max는 별도의 canonical 1s와 funding source에서 공식 owned interval만 phase root로 materialize한다.

초기 단계에서 OI, L2, liquidation history 또는 110-symbol 전체 1s를 수집하지 않는다. `collect_all_strategy_support_data.py`는 1s layout만 OHLCV 경계로 인식하므로 1m-only root에는 사용하지 않는다. `backfill_funding_fee_features.py`는 기존 funding의 파생 열 도구이지 funding downloader가 아니다.

**Acceptance**

- D-01A post-append receipt에서 required owned interval의 expected-grid gap, duplicate, nonmonotone, nonfinite가 0이다.
- D-01A receipt에서 funding은 실제 settlement timestamp, expected cadence gap 0, source receipt를 가진다.
- append 전후 coverage와 collector report가 보존된다.
- point-in-time membership이 없는 full-universe 실험은 시작하지 않는다.

**STOP**: API rate limit/ban, 중단된 temporary output, checksum mismatch.
**KILL**: 공식 owned-interval 데이터가 존재하지 않거나 lifecycle provenance를 복구할 수 없는 해당 트랙.

### Phase R1 — 고-CAGR router 두 개의 실제 전략 복구

허용 후보는 R1과 R2 정확히 두 개다. `--recompute-from-json`, post-OOS selector augment, 새로운 grid search는 성과 증명에 사용하지 않는다.

각 fold에 다음을 가진 immutable leaf manifest를 만든다.

- strategy class와 params
- symbol tuple과 point-in-time membership receipt
- native timeframe
- weights와 gross exposure
- decision timestamp와 입력 cutoff
- evaluation mode와 source artifact hash

R2는 R1과 동일한 leaf 선택을 가져야 하고 strict fallback의 MDD20/cap2 차이만 허용한다.

**Acceptance**

- candidate 수와 hash가 정확히 2다.
- 모든 leaf의 `evaluation_mode`가 `handler` 또는 `registry_simulator`다.
- `generic_fallback_proxy=0`이다.
- current-fold OOS 입력이 0건이다.
- 원본 label을 유일한 actual-engine manifest로 복원한다.

**KILL**: proxy/inferred semantics가 필요하거나 leaf identity가 비유일적이면 router 전체 종료. 근사 후보로 교체하지 않는다.

### Phase R2 — strict + cost-realistic proof

최종 proof는 cost-realistic profile을 기준으로 strict research gate와 `research.route_unmapped_registered_strategies: true`를 포함하는 **하나의 frozen replacement profile**을 사용한다. profile 두 개를 순서대로 적용하지 않는다.

같은 frozen signal과 position에 비용만 바꾼 10/15/20/30bp 네 셀을 실행한다. 실제 funding, UTC settlement, sqrt impact, protective stop, exposure normalization과 purge/embargo를 적용한다.

**Binding gate**

- DSR `>= 0.90`
- SPA p-value `<= 0.05`
- PBO `<= 0.50`
- 20bp all-in net return `> 0`
- MDD `<= 30%`
- liquidation/ruin `0`
- leave-best-fold-out net return `> 0`
- complete funding와 point-in-time membership
- 한 fold의 이익이 전체 결론을 지배하지 않음

둘 다 통과하면 validation 20bp Calmar가 높은 후보를 선택하고, 동률이면 MDD가 낮은 후보 하나만 forward로 동결한다. 어느 binding gate든 실패하면 해당 후보를 종료한다. 둘 다 실패하면 router 트랙을 끝내며 새 변형을 만들지 않는다.

### Phase A1 — Alpha-Max Revision 5.15

Alpha-Max는 main과 별도의 detached worktree와 별도 데이터 root에서 실행한다.

**고정 입력**

- branch/commit: `feat/alpha-max-20260710` / `629d91e5d4aac26911af65a4a5e15ebdcbded30f`
- config: `configs/research/alpha_max_portfolio_20260711_listing_aware.json`
- contract: `configs/research/alpha_max_contract_manifest_20260711_listing_aware.json`
- provenance: `configs/research/alpha_max_official_availability_evidence_20260711.json`
- phase-root preparer: `scripts/research/prepare_alpha_max_phase_roots.py`

각 SHA-256은 순서대로 `2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c`, `ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220`, `214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719`, `ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac`다.

**원래 기간 — 변경 금지**

| Phase | Start inclusive | End exclusive |
|---|---:|---:|
| warmup | 2022-12-31 | 2024-01-01 |
| train | 2024-01-01 | 2025-06-01 |
| purge | 2025-06-01 | 2025-06-08 |
| validation | 2025-06-08 | 2025-08-31 |
| embargo | 2025-08-31 | 2025-09-07 |
| exposed historical | 2025-09-07 | 2026-07-01 |

TONUSDT는 공식 raw `[2024-03-01T12:31:10Z, 2026-06-23T09:00:00Z)`, feature `[2024-03-01T16:00:00Z, 2026-06-23T09:00:00Z)`만 소유한다. chronology는 그대로이며 원 warmup/train gate에서 탈락해야 한다. GRAMUSDT 대체, synthetic warmup, 시작일 이동은 금지한다.

현재 branch의 `docs/research_note/alpha_max_data_pc_runbook_20260711.md`는 Rev5.14 지시이므로 **Rev5.15 정합화 commit이 push되기 전에는 phase-root preparation까지만 허용하고 prelock/historical 실행은 BLOCKED**다.

**Structural acceptance**

- prelock 68 actual-engine cells, 816 physical fold runs
- phase별 17 manifests
- historical 680 physical runs
- sealed bundle과 before/after prelock byte identity 통과
- TON availability-aware admission rejection 허용

유효 실행의 `no_demonstrated_alpha` 또는 historical robustness 실패는 Alpha-Max KILL이다. 날짜나 후보를 다시 조정하지 않는다.

### Phase F1 — fresh-forward

대상은 router champion 최대 1개와 Alpha-Max champion 최대 1개다. 서로 독립된 frozen commit/config/manifest/universe/risk/cost로 shadow-only 실행한다.

- 30일 checkpoint, 목표 60일
- 시작점은 final manifest freeze 이후이면서 이미 열어 본 마지막 bar보다 늦은 첫 complete bar다.
- parameter, universe, risk scaling 변경 시 forward clock을 0일부터 다시 시작
- 같은 forward interval을 사용해 새 후보를 추가 선택하지 않음
- actual funding, hypothetical order/fill, spread/slippage/impact와 position reconciliation을 일별 저장

**60일 acceptance**

- after-cost net return `> 0`
- coverage/reconciliation error `0`
- liquidation/kill-switch event `0`
- frozen MDD ceiling 준수
- implementation/config/data hash drift `0`

통과는 별도 canary risk review 자격만 뜻한다. 자동 자본 배분은 없다.

## 6. 복구 트랙 종료 후의 후속 연구

R1/R2와 Alpha-Max가 모두 scientific KILL이면 같은 표본에서 변형을 더 만들지 않는다. 별도 run ID와 새 preregistration으로 아래 경제 가설만 다시 열 수 있다.

1. core 6~10의 저회전 1d/4h time-series trend. Alpha-Max daily trend 결과를 먼저 증거로 사용한다.
2. 실제 spot long + perpetual short 양 leg의 funding/basis carry. futures-only funding slope를 진짜 carry로 부르지 않는다.
3. point-in-time crypto-only universe의 BTC beta-residual 횡단면 모멘텀. lifecycle registry가 없으면 시작하지 않는다.

각 패밀리는 기존 class와 utility를 우선 재사용하고 사전등록 variant를 최소화한다. 이 후속 연구는 현재 data-PC runbook의 실행 범위가 아니며, R1/R2 실패를 본 뒤 과거 OOS에 맞춰 설계해서는 안 된다.

## 7. 작업 목록과 의존성

| Task | 작업 | 선행 | 완료 증거 |
|---|---|---|---|
| D-01 | data PC root 발견, read-only inventory | 없음 | coverage와 gap ledger |
| D-01A | fail-closed research data contract validator 구현 | 없음 | `validate_ohlcv_frame` 재사용, expected-grid/actual-funding-cadence tests, JSON receipt |
| D-02 | core/router 1m tail 복구; interior gap은 별도 bounded repair | D-01,D-01A | pre/post validation receipt, collector receipt, coverage diff |
| D-03 | funding와 support inventory 복구 | D-01,D-01A | settlement validation receipt |
| D-04 | symbol lifecycle registry 구현 | D-01 | fold membership manifests와 tests |
| D-05 | main materializer CLI 회귀 수정 | 없음 | `--help`와 targeted tests 통과 |
| R-01 | exact R1/R2 frozen manifest replay seam | D-04 | exactly-two replay test |
| R-02 | actual-routing/evaluation-mode fail-close | R-01 | fallback 0 receipt |
| R-03 | single combined strict/cost profile | R-02 | resolved config와 SHA |
| R-04 | 10/15/20/30bp proof와 gate report | D-02,D-03,R-03 | immutable decision bundle |
| A-01 | Rev5.15 Alpha-Max runbook 정합화와 push | 없음 | exact hashes/commands review |
| A-02 | canonical source에서 phase roots 생성 | D-01 | preparation manifest |
| A-03 | prelock/historical one-touch 실행 | A-01,A-02 | sealed bundles |
| F-01 | 최대 두 champion fresh-forward | R-04,A-03 | 60-day final report |

Critical path는 `D-01A -> D-02/D-03 -> R-04 -> F-01`, `D-01 -> D-04 -> R-01 -> R-02 -> R-03 -> R-04 -> F-01`, `D-01 -> A-02`, `A-01 -> A-03 -> F-01`이다. 데이터 inventory와 validator가 준비되기 전에 전략 코드를 늘리지 않는다.

## 8. 최종 체크리스트

- [ ] 정확한 Git commit과 clean worktree 기록
- [ ] synthetic source 0건
- [ ] source provenance와 immutable inventory
- [ ] 연속 tail만 일반 collector로 복구; interior gap은 bounded repair
- [ ] point-in-time lifecycle와 actual funding 완전
- [ ] router 후보 정확히 2개
- [ ] faithful actual-engine leaf manifest
- [ ] `generic_fallback_proxy=0`
- [ ] combined strict/cost profile 하나와 hash
- [ ] 같은 signal의 10/15/20/30bp 결과
- [ ] DSR/SPA/PBO, MDD, leave-best-fold-out gate
- [ ] Alpha-Max Rev5.15 runbook 갱신 후 실행
- [ ] Alpha-Max 원래 날짜 유지
- [ ] seals, readback, before/after identity
- [ ] 60일 forward와 변경 시 clock reset
- [ ] 모든 단계 실자본 0%
