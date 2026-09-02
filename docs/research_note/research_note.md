# Research Note
## 2026-09-02 KST — Alpha-Max를 단일 백테스트 파이프라인으로 재정렬하고 TradFi canonical snapshot 갱신

반복된 `prelock-vNN` supervisor 재기동은 연구 진척이 아니었다. 1,588-unit
precompute/parity, 816 validation folds, 680 historical report-only folds는
각각 별도 goal이 아니라 **같은 Alpha-Max 백테스트 파이프라인의 내부 단계**다.
Observability는 추가 백테스트가 아니라 완료된 두 평가 단계의 acceptance
projection이다. 따라서 실행 순서를
`통합 → 공식 데이터/TradFi 증거 갱신 → immutable input snapshot → precompute/parity
→ prelock 816 folds → historical 680 folds → observability → 연구노트/graph/Obsidian`
으로 고정했다. wrapper receipt 문제를 새 버전 번호로 반복 재시도하는 경로는
폐기하고 shared integration boundary를 먼저 수정한다.

Binance USD-M `/fapi/v1/exchangeInfo`를 2026-09-02T10:09:40Z에 다시 읽어
`TRADIFI_PERPETUAL + USDT + TRADING`을 엄격 적용했다. 현재 canonical
research-only snapshot은 기존 100개에서 **182개**로 늘었고, static snapshot과
공식 응답의 집합 차이는 0이다. `SPCXUSD1`은 trading TradFi 계약이지만 quote가
`USD1`이므로 USDT universe에서 명시적으로 제외했다. 앞선 임시 v2 audit의
185개 수치는 `ALLUSDT`, `BTCDOMUSDT`, `SPCXUSD1`을 잘못 포함한 값이라
superseded다. 정정 증거:
`var/g003-data-refresh/TRADFI_UNIVERSE_REFRESH_AUDIT_20260902.json`,
tracked canonical:
`docs/research_note/tradfi_universe_refresh_audit_20260902.json`,
runtime mirror:
`var/reports/tradfi_universe_refresh/TRADFI_UNIVERSE_REFRESH_AUDIT.json`,
source SHA-256:
`78780e56d0a49a88ab8d326d68cd4198fc9ef9ffd3ef0278310993c0173a13aa`.
이 갱신은 연구 universe/coverage 기본값만 바꾸며 주문 라우팅과
paper/testnet/live/real-money 승인을 열지 않는다.

통합 선행 감사도 실제 canonical DB에서 통과했다. Deep contract audit는
raw `1,066,681,730/1,066,681,730` rows, `415/415` partitions,
funding `39,569/39,569` rows, missing/extra/duplicate/jitter/error 전부 0을
재계산했다. Public loader/downsampling/feature/funding 검증도 `complete`다.
증거:
`var/g003-data-refresh/alpha_max_canonical_contract_audit_20260902.json`,
`var/g003-data-refresh/alpha_max_canonical_pipeline_verification_20260902.json`.

전략 카탈로그도 현재 registry로 재생성했다(`163` registry, `161` in-scope,
`2` excluded). 관계 그래프에서 누락돼 Obsidian publish를 막던 신규 전략
15개와 `derivatives_carry` family를 canonical graph에 편입한 뒤 generated
namespace를 원자 교체했다(`215` notes, `2,294` links). vault 전체의 다른
프로젝트 노트가 LuminaQuant ID gate를 오염시키지 않도록 audit authority를
`LuminaQuant/` namespace로 한정하되 wikilink 해석은 vault 전체를 유지했다.
실제 vault readback audit는 `408` authoritative notes, `4,136` wikilinks,
broken/ambiguous/duplicate/missing ID 모두 0으로 통과했다. 영수증:
`docs/research_note/obsidian_strategy_graph_audit_20260902.json`
(SHA-256
`b9df4b33133fc5b15a4704f0885b8ead76e63f5a46f8855c0789eb5d59689711`).

판정: **integration_first / research_only_no_execution /
order_routing_enabled=false**.

## 2026-08-20 KST — 적대 리뷰 라운드: 25건 발견 → 23건 확정 → 전건 수정·회귀봉인

배치 전체 diff에 대해 5-차원 헌터 → 발견별 회의론자 반증 검증(런타임 재현 의무) 라운드를 돌렸다. 25건 발견, **23건 CONFIRMED / 2건 REFUTED**, 확정 전건을 당일 수정하고 회귀 테스트로 봉인했다(전체 스위트 **4946 passed**, 골든·manifest 스냅샷 바이트 불변, ruff/format/architecture/docs 그린). 핵심 수정:

- **L-C 민홀드 드랍→지연(deferral) 재설계(크리티컬 F1)**: 원샷-전이 전략의 bare EXIT를 차단만 하면 전략-북 영구 탈동기+포지션 누수+재진입 스태킹이 발생함을 런타임 재현으로 확정. 오버레이가 컴포넌트 스코프 원장(`component|symbol`)으로 차단 EXIT를 보존했다가 만기 시 합성 EXIT(`overlay_reason=min_hold_released`)로 방출하고, exit-pending 중 동방향 재진입도 차단. `exit_reason` 우회는 truthy가 아닌 **보호성 화이트리스트**로 교체(F4: 'rebalance' 라벨이 게이트를 무력화하던 결함), 컴포넌트 북 분리(F6), 라이브 네임스페이스 패리티(F5)까지 봉인.
- **L-D 앵커 결함(F2/W1)**: 결정봉 타임스탬프 기준이라 실제 스트래들을 놓치고(12.5%) 깨끗한 진입을 오차단(12.5%) — MKT는 다음 봉 시가 체결이므로 **체결시각(+1 timeframe) 앵커**로 수정.
- **warmup 훅 청크 연속성(F3)**: `had_warmup` 출처를 엔진 상태로 지속해 첫 라이브 이벤트가 warmup_bars=0 연속 청크에 떨어져도 정확히 1회 발화. **신규 슬리브 5종 전부 `on_warmup_end()` 구현**(유령 북 리셋, 데이터 deque 보존).
- **슬리브 공통 결함**: 죽은 피드 심볼의 동결값이 크로스섹션을 영구 오염(entry↔exit 무한 churn 포함) → 전 XS 슬리브에 신선도 게이트; offsession 72h 만기를 패널-와이드로(정지 심볼도 만기 청산); 유니버스 전면 결측 1회가 성숙 북을 rank_lapsed로 플러시하던 경로 차단; taker-flow가 자기 계약을 어기고 엔진 인트라바 브래킷을 달던 결함 제거(종가확정 스톱만); basis-gap set_state 적대 입력 내성; salience/prospect 형성 패널 캘린더 정렬(내부 갭 심볼 제외, 위치 정렬은 시장수익률 왜곡 ~25% 재현); OI 민홀드 단위 재고정(주간 5→2, 등록 지평 1-4주 내로); V-DIAG har_lags 비정형 스펙 fail-closed; 행동가치 지표 파라미터 never-raise; VR z-stat 손계산 골든; **배치 배선 테스트 신설**(광역 유니버스 23행 전부 레지스트리 경유 생성+스키마 포함성 — 미지 파라미터 무성 드랍 차단).
- W3 판정에 따라 cross_sectional_anomaly는 **원본 plain-sum skewness 사본을 의도적으로 유지**(fsum 정준화는 ~10% 윈도우에서 1-ULP 표류 — 등록 전략 기본 수치의 무게이트 변경 금지 원칙).

REFUTED 2건: W4(basis-gap family="carry"는 빌더 헤더가 명시한 의도적 예외 — funding-정산 캐리는 진짜 캐리), W5(기본 매니페스트 행수 21이 정답 — guarded_ls 셀은 min_symbols=12라 기본 10심볼 유니버스에서 비물질화. 광역 유니버스 기준 23행). 판정 불변: **do_not_promote / research_only_no_execution / 실배분 0%**.

## 2026-08-20 KST — 신규 알파 슬리브 7클래스 + V-DIAG + 지표 5모듈 저작 완료 (3종 재고정 포함)

같은 날 오전의 레버 커밋에 이어, 적대 심판 3인을 통과한 신규 저작물을 원자 통합했다. 전부 `research_only`이고 기본 후보 매니페스트에 21개 row로 편입되며(전부 cross_sectional 바스켓, `allow_multi_asset` 핸드오프 경로 전용), 2-symbol 스냅샷 픽스처에는 하나도 물질화되지 않아 **manifest sha는 불변**이다. tier-guard(BATCH 7종 추가)·hardcoded-baseline 재고정 완료, 전체 스위트 **4914 passed / 21 skipped / 3 xfailed**(베이스라인 4712 대비 +202 전원 신규 테스트), ruff/format/compileall/check_architecture/verify_docs/골든 핀 전부 그린.

- **신규 슬리브(각 모듈 독스트링에 EXPECTED NULL + 단일 반증 측정 + 최근접 그레이브야드/인컴번트 사전등록)**: ① `CrossSectionalResidualTakerFlow`(~5일 taker 순공격흐름/턴오버를 수익률 z에 잔차화, 주간 L/S 퀸타일+1주 민홀드 — aggTrades 백필로 그레이브야드 #7 커버리지 사인 해제) ② `BasisFundingGapConvergence`(mark-index basis − funding-내재 basis 스프레드를 8h 정산 경계에서만 |z|>2 페이드, 민홀드 2인터벌 — 인컴번트가 계산하지 않는 funding-기대 오차) ③ `OffSessionBasisDislocation`(TradFi 퍼프의 오프세션 스테일-앵커 괴리를 현물 재개장 직전 페이드, 36h/72h) ④/⑤ `SalienceTheoryValue`/`ProspectTheoryValue`(BGS 2012/TK 1992 정준 상수 동결, 주간 ISO, 모멘텀+MAX 잔차화; ST-PT 중복 토너먼트 사전등록) ⑥ `OpenInterestGrowthPressure`(Hong-Yogo ΔOI/달러볼륨 연속, 모멘텀 잔차화, 부호 양(+) 사전등록·플립 금지).
- **V-DIAG 구현**(마스터플랜 §6.3 최초 코드화): `research/vol_spillover_diagnostic.py` + 러너 — HAR-RV 기준 vs 리더 lagged-RV 블록, QLIKE(Patton)+블록 부트스트랩+BH-FDR, 승인선 사전등록(중앙 QLIKE 개선 ≥5%·폴드 ≥60%·p≤0.05). 실패 시 vol-스필오버 전략·신규 다변량-vol 코드 전면 KILL. 연기된 leader-RV 사이징 오버레이 설계는 모듈 독스트링에 동결 수록(빌드 게이트=BTC→alt 1쌍 이상 승인).
- **지표**: `har_rv`·`variance_ratio`(cusum_varratio에서 패리티-락 추출, 편향 추정기 기본 유지)·`funding_structure`(funding_momentum 추출+텀구조 스프레드+basis-funding 갭; 심판 지시로 단일 모듈 병합)·`behavioral_value`·`rolling_stats`의 skew/kurt(전략 사본 2개 dedup, fsum 정준화·패리티 골든). funding-carry 슬리브에 config-gated `require_term_structure_agreement`(기본 False, 진입-스킵 전용).
- **커버리지 감사 도구**: `scripts/research/audit_liquidation_feature_coverage.py`(OI ≥90%/청산 ≥80% 게이트, 무데이터 시 insufficient_data 폐쇄).
- 실행서: [`alpha_sleeves_levers_20260820_handoff.md`](alpha_sleeves_levers_20260820_handoff.md) — 백필/감사 선행 → 표준 walkforward(21 row, evaluation_mode 감사) → V-DIAG → L-C/L-D A/B(사전등록 고정값: min_hold 1일 상당, band 8bps, guard ON) → 넷팅 발생률 측정 순.

판정 불변: **do_not_promote / research_only_no_execution / 실배분 0%**. 이 배치는 falsification 프로토콜이며 대부분의 리프의 EXPECTED NULL은 reject다 — 죽음을 보고하는 것이 산출물이다.

## 2026-08-20 KST — 사전등록 엔진 레버 4종 구현 (L-C/L-D/warmup-hook/H1), 전부 기본 OFF

`performance_lever_measurement.md`에 사전등록만 되어 있던 엔진 레버를 실제 엔진 seam에 구현했다. 설계는 8-lens 매핑 + 4-lens 제안 패널 + 3-adversarial-judge 라운드로 검증했고(그레이브야드/비용·데이터/신규성 심판 만장일치 PASS), 전부 config-gated OFF·기본 경로 바이트 동일이다. 골든 핀(`tests/integration/test_engine_golden.py`, `test_walk_forward_golden.py`) 그린, 신규·기존 타깃 스위트 85 passed.

- **L-C no-trade band + hard min-hold (실엔진 seam)**: `StrategyQualityConfig.min_hold_bars`(기본 0)·`no_trade_band_bps`(기본 0.0). min-hold는 `StrategyQualityOverlay`에서 overlay `enabled`와 독립으로 동작 — 보유기간 내 **사유 없는(bare) 전략 EXIT와 역방향 재진입만** 차단하고, `risk_exit`/`exit_reason`/`overlay_reason` 마커가 있는 보호성 EXIT와 엔진 레벨 스톱/청산 체결(check_open_orders 경로)은 절대 지연시키지 않는다. 밴드는 `Portfolio.generate_order_from_signal`에서 진입·부분청산 주문 노셔널이 자본 대비 band bps 미만이면 드랍(전량 청산은 위생상 면제). 이전에 죽어 있던 `cost_aware_constructor.py` 밴드의 엔진(b) 재타게팅이며, `strategy_signal_dispatch` 삽입안은 문서대로 배제했다. 측정은 data-PC A/B(flag OFF vs ON, 사전등록 파라미터, 10/15/20/30bps 그리드)로만 한다.
- **L-D funding-entry guard**: `ExecutionConfig.funding_entry_guard`(기본 False). 신호 metadata가 `intended_hold_seconds`(또는 `intended_hold_bars`×config TIMEFRAME)를 선언한 경우에 한해, 의도 보유가 펀딩 인터벌(8h)보다 짧으면서 다음 00/08/16 UTC 정산 경계를 걸치면 진입을 스킵한다. 선언 없는 신호·EXIT는 절대 차단하지 않으며 부호 무관(튜너블 없음) 고정 규칙이다.
- **Warmup-end 훅**: 엔진이 warmup→live 전이 시 전략의 선택적 `on_warmup_end()`를 정확히 1회 호출(첫 라이브 바 처리 전). 유령 포지션(warmup 중 신호 억제로 전략 내부 상태만 진입) 탈동기화를 전략이 스스로 리셋할 수 있는 계약. `get_engine_state`/`set_engine_state`에 `warmup_end_hook_fired` 보존으로 청크 경계에서 중복/누락 없음. 훅 미정의 전략은 바이트 동일(현재 레포에 정의 클래스 0개, grep 검증).
- **H1 tier fail-safe**: `registry.get_strategy_metadata`의 미지 이름 폴백을 `live_default`→`research_only`로 전환. 68개 pre-contract 레거시 클래스는 레지스트리에 동결 스냅샷(`_LEGACY_UNHINTED_LIVE_DEFAULT`)으로 명시해 기존 tier 해석 전부 불변(신규 테스트가 68개 전원 live_default 유지 + 레지스트리/가드 스냅샷 일치 + 미지 이름 research_only를 고정). 심판 라운드에서 "폴백 한 줄만 뒤집으면 레거시 68개가 라이브 맵에서 탈락한다"는 결함이 독립적으로 검출됐고, 구현은 이를 선반영했다.

판정 관례 유지: **do_not_promote / research_only_no_execution / 실배분 0%**. 레버의 성과 주장 없음 — 효과는 data-PC A/B가 결정한다. 신규 알파 슬리브 6종(taker-flow 잔차 XS·basis-funding 갭 컨버전스·off-session basis 괴리·salience/prospect 행동가치 XS·OI 증가압력 XS)과 V-DIAG(HAR-RV QLIKE 스필오버 승인 진단) + 지표 4모듈은 병렬 저작 진행 중이며 다음 엔트리에서 3종 재고정(manifest+baseline+tier)과 함께 기록한다. 심판 판정으로 **연기**된 항목: leader-RV 사이징 오버레이(V-DIAG가 BTC→alt 페어를 승인해야 저작), 청산 XS 슬리브(liquidation 컬럼은 백필 경로가 없어 커버리지 감사 선행), 동일봉 MKT 넷팅(발생률 측정 우선 — 엔진 개입 없이 주문 로그 분석으로 측정).
## 2026-08-23 — G003 integrated alpha-research decision and restartable native replay

Current canonical navigation moved to `README.md`, `strategy_taxonomy.md`,
`strategy_relationships.json`, `strategy_evidence_index.json`, and
`evaluation_contract.md`. This diary remains chronological provenance rather
than the current status authority.

G003 selection-v11 completed 24 candidate runs with 16 allowed data exclusions
and 20 active return panels. Only `crypto_turtle_20_10_atr_v1` survived its
positive-quality gate; the preregistered floor was six. The suite was rejected,
no allocator portfolio was emitted, and locked OOS was not launched.
Post-activation `19/19` and registry `127 pass + 16 exclusions` are execution
smokes, not promotion evidence. The earlier 144-strategy snapshot and later
143-row registry report remain explicitly unreconciled by commit/scope.

The checkpointless parity-v10 oracle was stopped at modeled `21.97%` progress
and sealed non-reusable under receipt
`f1793acc435a626f2e42fdbad739a12decfa8f8c33f1a39c9fed57e86161dd24`.
Its separately sealed exact-native candidate remains execution provenance, not
official acceptance or performance evidence. The integrated replacement moves
only authenticated canonical 1-second OHLCV folding into Rust/PyO3; Python
retains release grouping, strategy/event ordering, final handoff,
finalization, capsule serialization, and economic semantics. Whole UTC days
are the parity restart unit and whole row/cost cells are the prelock/historical
restart unit.

Research priorities are now explicit: fresh sealed price-volume continuation;
Turtle versus one equal-gross crash/correlation guard; echo/residual momentum
orthogonality; matched rebalancing controls; and a preregistered
stationarity-gated residual momentum/reversion switch. These are hypotheses,
not observed performance. Official parity, prelock, historical, and
observability acceptance remain pending. `order_routing_enabled=false`.

## 2026-07-10 KST — 최신 upstream task 전수 실행: evaluability v3 / defect v5 / eq-flow / dashboard

`private/main`을 `49bdd52a`까지 fast-forward하고, 새 task 원문 `alpha_pool_evaluability_v3_handoff.md`, `systematic_defect_fix_v5_handoff.md`, `eqflow_complements_handoff.md`와 dashboard UI/UX fix를 현재 data-PC에서 검증했다. 이번 추가 구현은 strict profile을 실제 candidate CLI까지 전달하고, cost override를 score JSON보다 우선시키며, 등록된 research strategy를 shared generic momentum proxy 대신 registry simulator로 라우팅하고, validation/whole-search evidence가 없으면 fail-closed하도록 만든다. locked-OOS는 보고 전용으로 남겼고 post-OOS variant의 paper/live/real alias와 실배분 경로는 모두 닫았다.

Coverage-first 결과는 로컬 128-symbol store의 384 symbol-timeframe pair 중 기본 360-bar 기준 72 pair가 충분했다(1h 35, 4h 27, 1d 10). 이 교집합으로 만든 raw manifest는 1h 828 + 4h 730 + 1d 468 = 비용 셀당 2,026 rows다. `offsession_tugofwar_1h_tow_42d_fade`와 `stationarity_gated_residual_reversion_4h_strict_adf`는 handoff의 regression-locked `do_not_rerun_as_is` 대상이다. 기계적으로 확장된 raw grid에 포함된 사실을 숨기지 않고 모든 비용의 8 rows를 보존했지만, authoritative gate와 allocator에서는 미리 정해진 known-null quarantine으로 제외했다. 따라서 protocol gate 분모는 1h 827 + 4h 729 + 1d 468 = 2,024 rows다.

`configs/profiles/backtest_cost_realistic.yaml`, validation selection, DSR `0.90` / SPA `0.05` / PBO `0.50`, cost override `10/15/20/30bps`로 12개 timeframe-cost cell을 모두 실행했다. Symbol/timeframe preflight에서는 protocol manifest `2,024/2,024` rows가 evaluable이었지만, 실제 split/window 요구까지 적용한 runtime에서는 네 비용마다 실제 평가 `510`, `insufficient_data` `1,514`, hard reject `2,024`, pass `0`, strict shortlist `0`으로 동일하게 닫혔다. raw reports는 비용마다 2,026/2,026 rows와 각 row의 `metadata.missing_symbols`를 보존한다. analyzer 총계도 8,096 rows에서 `PASS=0`, `NEAR-MISS=0`, `DEAD=2,040`, `INSUFFICIENT=6,056`이다. 20bps routing audit에서는 새 29개 strategy class가 모두 존재했고, 데이터로 평가된 102 rows 전부 `registry_simulator`, `generic_fallback_proxy=0`이었다.

110-symbol monthly-refit은 11 folds, raw 1,921 candidate-fold rows / 199 labels를 보존했다. whole-search DSR/SPA/PBO evidence가 없는 상태에서 strict gate를 추정값으로 우회하지 않아 admitted fold/aggregate는 둘 다 0이다. Eq-flow 진단은 C1 correlation guard가 한 fold도 engage하지 않아 기대한 July trim을 재현하지 못했고 밴드를 현장에서 튜닝하지 않았다. B2와 A3a는 각각 16/16 variant가 raw compounded return을 개선했고 8/16이 max MDD를 줄였으며, A3a의 raw/effective selection·scale 결정값을 fold마다 직렬화했다. A3b bar weighting은 199 labels 중 197개의 순위를 바꿨다. D4는 March/July를 cash로 gate해 raw compounded return을 `-13.35%`에서 `+17.32%`, max MDD를 `25.45%`에서 `9.76%`로 바꿨지만 사전 기대 Sep/Dec/Apr shape와 달라 승격하지 않았다.

F1은 current 110-symbol locked-OOS daily NET stream, tradfi open-impulse stream, state-VWAP dense-pair stream을 정확한 34 common dates로 materialize했다. 월별 두 sleeve의 turnover evidence가 unknown이고 두 source artifact도 promotion-ready가 아니다. native quality에서 양수 sleeve가 하나뿐이라 `min_sleeves=2`를 못 채워 children `0`, cash `100%`로 fail-closed했다. 별도 alpha-pool 3-arm은 20bps 4h train+validation의 165 streams / 11 families를 base M2, MR1 turnover+Ledoit-Wolf, family meta-momentum으로 동일 입력 재생했다. native quality children은 각 27개였지만 upstream strict admission이 0이므로 세 arm의 effective children은 전부 0, cash 100%다. locked-OOS report-only raw return도 base `-0.18%`, MR1 `-0.27%`, family tilt `-0.27%`로 개선 근거가 없다.

D5 preregistered universe control은 동일 코드/profile/cost/fold calendar에서 roster만 85와 110으로 바꿨고 모든 control equality가 참이다. 주 label raw compounded locked-OOS는 85-symbol `+3.23%`에서 110-symbol `-4.11%`로 바뀌었다. June은 `-14.45%` 대 `+2.57%`로 sign flip, July는 `+0.57%` 대 `-9.41%`다. 결론은 `universe_sensitive_do_not_promote`다.

Dashboard upstream fix는 current tree에서 frontend 6 files / 86 tests, ESLint, TypeScript, Next production build 16/16 pages를 다시 통과했다. Python clean isolated full suite는 `4,365 passed, 36 skipped, 3 xfailed`, 집중 research suite는 `110 passed`, Ruff check/format과 diff check도 통과했다. 주요 근거는 `var/reports/data_pc_tasks_20260710/{coverage.json,manifest_coverage.json,candidate_grid_verified_summary.json,candidate_grid_analysis.json,monthly_refit_walkforward_110_verified.json,eqflow_analysis.json,eqflow_f1_evaluation.json,allocator_3arm.json,d5_universe_comparison.json,research_artifact_validation.json}`이다.

최종 판정: **do_not_promote / research_only_no_execution / 실배분 0%**. fresh-forward shadow와 완전한 whole-search/turnover provenance 전에는 paper/testnet/live/real-money/order 경로를 열지 않는다.
## 2026-08-12~15 KST — ≥1분 전략 144개 전수 인벤토리·공통 exact-1m 재평가·문헌 분류

현재 전략 레지스트리 `144`개를 기준으로 cadence/timeframe과 실행 인터페이스를 코드에서 다시 추출했다. `MicroRangeExpansion1sStrategy` 하나는 `1s` 의존이라 사용자 범위에서 제외했다. 명시 cadence/timeframe이 ≥1분인 클래스는 `119`; cadence metadata가 없어 시간 범위를 검증할 수 없는 `24`개는 완전한 레지스트리 회계를 위해 카탈로그에 남기되 `scope_unverified`로 분리했다. 따라서 카탈로그 전략 행은 `143`개지만 “검증된 ≥1분 전략”은 `119`개다. `DacapogoDailySourceStrategy`는 명시적 `1d`, `polars_batch`, 전용 연구 러너, `research_only`, live 미지원으로 등록됐고 공통 event-driven 화면에는 억지로 끼우지 않아 성과가 `not_available`이다. 이름 substring으로 family를 추정하지 않고 candidate-library 의미와 구현을 읽어 override했으며, 최종 family 미해결은 `0`이다. 후보 생성기에는 있지만 레지스트리에 없는 클래스 `12`개는 별도 `candidate_library_definition_not_registered` 목록으로 보존했고 성과 전략 수에 섞지 않았다.

성과 화면은 exact 1m 공통기간 `2026-07-01T00:00:00Z`~`2026-08-12T00:00:00Z`(42일)와 그 안의 cold-start recent `2026-08-01T00:00:00Z`~`2026-08-12T00:00:00Z`(11일)이다. gap-fill/interpolation/synthetic rows는 금지하고, 초기자본 10,000, 종목별 기본 명목 10%, maker 2bp / aggressive taker 4bp + 2.5–7.5bp slippage + 1bp half-spread, bar-volume 10% cap으로 실행했다. 이 화면은 selection provenance가 봉인되지 않았고 recent가 full에 중첩되므로 **독립 OOS 또는 승격 증거가 아니다**. 현재 144개 레지스트리 중 전용 일봉 Dacapogo를 제외한 `143`개가 이 공통화면에 있고, 상태는 full `pass 114 / fail 15 / excluded 12 / resource_excluded 2`, recent `pass 115 / excluded 27 / resource_excluded 1`; raw 거래·순위 적격은 full `89`, recent `86`, 양쪽 비교 가능 `82`, 양쪽 양수 `5`다. 이 raw 공통화면 진단 `5/82`에는 범위 밖 1s 전략과 matched-control 없는 rebalancing 행이 포함된다. 명시적 sub-minute를 제외하고 rebalancing raw 성과를 억제했지만 cadence 미확인 24개와 Dacapogo `not_available` 행을 보존한 catalog-controlled 진단은 `4/80`이다. `scope_status=verified_in_scope`만 남긴 **최종 검증된 ≥1분 controlled scorecard는 `3/67`**이며 Dacapogo의 NA를 0으로 바꾸지 않았다. full raw 적격 중앙 return은 `-18.78%`, recent는 `-5.50%`; 같은 기간 20종 동일가중 시장은 `+7.58%`였으므로 양수 headline도 자동으로 alpha가 아니다.

양쪽 양수 진단값:

| Strategy | Family | Scorecard status | Full | Recent nested | Full/Recent Sharpe | Trades |
|---|---|---|---:|---:|---:|---:|
| RebalancingPremiumHarvestStrategy | rebalancing_diversification | raw-only; cadence unverified; matched control missing | raw `+5.98%` | raw `+1.78%` | `3.14 / 5.76` | `45 / 29` |
| PriceVolumeCorrContinuationStrategy | trend_momentum | verified ≥1m controlled | `+1.07%` | `+0.59%` | `4.69 / 4.63` | `10 / 9` |
| BitcoinBuyHoldStrategy | benchmark | catalog diagnostic only; cadence unverified | `+0.74%` | `+0.08%` | `1.88 / 1.18` | `1 / 1` |
| CrossSectionalIntermediateEchoMomentumStrategy | cross_sectional | verified ≥1m controlled | `+0.29%` | `+0.16%` | `0.81 / 1.65` | `35 / 13` |
| TrendGatedResidualMomentumStrategy | cross_sectional | verified ≥1m controlled | `+0.05%` | `+0.10%` | `0.48 / 2.87` | `5 / 2` |

`RebalancingPremiumHarvestStrategy`는 raw return 1위지만 동일 초기 바스켓 buy-and-hold 대조군이 없다. 참고용 14종 동일가중 시장 바스켓은 같은 full 구간 `+8.68%`로 전략 raw `+5.98%`보다 높았다. 초기 weight까지 동일한 matched control이 아니므로 premium을 수치화할 수 없고, 카탈로그/Obsidian 성과에서는 return·Sharpe·MDD를 `matched_control_missing`/NA로 억제하고 raw 값만 진단 필드에 보존했다. `DisagreementGatedEnsembleStrategy`는 full `+2.07%`였지만 recent `-0.27%`라 양쪽 양수가 아니다. resource exclusion은 full의 `AbnormalReturnContinuationStrategy`(12 GiB 초과), `OvernightSessionReturnRiderStrategy`(반복 30분 초과), recent의 `Alpha101FormulaStrategy`(동시 작업 안전 RSS 상한)이며 NA를 0으로 바꾸지 않았다.

온라인/논문/인증 소스 근거 `38`건은 `docs/research_note/strategy_evidence_20260812.json`에 제목·저자·출판처·URL·지지 범위·한계를 구조화했다. Dacapogo private source snapshot은 인증 접근이 필요한 grade C source-code 근거로만 사용했고 외부 알파·배포 성능으로 취급하지 않았다. 핵심 결론은 다음과 같다.

- Crypto trend/cross-sectional momentum, near-high anchoring, pairs/reversal, order-flow/lead-lag, funding/carry에는 외부 prior가 있지만 paper 성과는 저장소 성과가 아니다. wall-clock horizon, point-in-time universe, borrow/funding, BBO/latency/capacity와 실제 비용을 별도 검증해야 한다.
- 단일 perp funding/OI/basis/liquidation 방향성 신호는 delta-neutral carry/arbitrage가 아니므로 `derivatives_directional_crowding`으로 분리했다.
- volatility scaling은 unscaled control을 이긴다고 가정하지 않는다. rebalancing은 동일 바스켓 무리밸런싱 대조군 없이는 premium을 주장하지 않는다.
- 기존 레지스트리가 정당화 가능한 전략군을 이미 포괄한다. 새 전략 클래스를 더 쓰는 대신 기존 클래스의 paper-aligned ablation `E1`~`E9`를 preregister하고 전체 trial ledger, purged validation, genuine family-wide DSR/CSCV/Hansen SPA, 비용·capacity·locked report-only OOS를 요구한다.

연구 선택 경로도 fail-closed로 수정했다. validation 선택이 locked OOS를 읽지 않게 분리하고, binding cost stress를 validation에 적용하며, OOS는 report-only stream으로 보존했다. non-finite metric과 누락 PBO는 거부하고, shortlist는 모든 후보의 upstream pass/hard-reject를 강제한다. weight cap이 불가능하면 cap을 몰래 완화하지 않고 현금을 명시한다. 현재 row-wise PBO/SPA-like 값은 genuine CSCV/Hansen SPA가 아니므로 선택 artifact에 `promotion_eligible=false`, `selected_team=[]`를 고정했다. 한-bar embargo는 최소값일 뿐 보유기간 전체 leakage proof가 아니다.

산출물:
- 공통 화면: `var/reports/common_period_reval_20260812/common_period_report.md`, `common_period_summary.json`, `strategy_comparison.csv`
- 카탈로그/성과: `var/reports/strategy_research_20260812/strategy_catalog.json|csv`, `strategy_scorecards.md`, `family_scorecards.md|csv`, hash manifest
- Obsidian: Windows vault `MyDB/LuminaQuant/Strategy Research Generated`에 현재 stage를 적용하고 설치본을 다시 읽어 동일성을 확인했다. 전략 `143`, family `14`, evidence `38`, index `1` = note `196`; link `2011`, broken link `0`; catalog/exporter/stage/installed manifest hash가 receipt와 일치한다. 생성 namespace 외 사용자 note는 건드리지 않았다.
- CodeGraph: `codegraph sync .` 완료, files `1281`, nodes `34693`, edges `111413`, pending `0`, reindex 권고 없음. 구조화 receipt는 `var/reports/strategy_research_20260812/codegraph_receipt.json`에 CLI hash와 전체 status를 저장했다.
- 최종 snapshot receipt: `var/reports/strategy_research_20260812/strategy_research_delivery_manifest.json`에 HEAD/branch, dirty-status hash, 핵심 코드·테스트·연구·성과·Obsidian receipt SHA-256, CodeGraph 상태, no-promotion 결정을 묶었다.

Dacapogo 일봉 별도 lane도 승격하지 않았다. entry-last 해석은 full `+9.06%`지만 같은 OHLC bar에서 stop-first는 `-8.61%`로 intrabar path ambiguity가 뒤집고, v2 ML은 full `+0.37%`, recent `+0.56%`이나 gate false / locked action `cash`다.

최종 판정은 **do_not_promote / research_only_no_execution / live 배포 불가**다. literal 100% 무결함·무회귀는 유한 테스트로 증명할 수 없다. 더구나 독립 OOS, genuine search-wide multiplicity matrix, matched controls, BBO/funding/capacity, shadow reconciliation이 없으므로 현재 live-ready라고 주장하는 것이 오류다. 최종 변경 범위 focused gate는 `318 passed`, `ruff check .`와 대상 Python `ruff format --check`는 green이다. 별도 560-file/8-shard broad baseline은 `6035 passed / 20 skipped / 41 subtests passed / 3 xfailed / 130 failed`를 그대로 보존했다. 실패는 숨기지 않았다: WSL `/mnt/c` EIO·host-reserve 직접 실패 `109`, 그 선행 실패로 가려진 acquirer 오류 `13`(로컬 reserve 재실행에서는 `167 passed / 7 failed`, 이 중 scratch-cleanup 불변식 6개와 의도적 `/mnt/c` contract 1개), 사용자 수정 acquirer hash `d5f7…` 대 signed policy `d3c6…` 불일치 `7`, 로컬 materialized 1s/1h close 불일치 `1`이다. 이 Alpha-Max·외부 mount·로컬 데이터 gate를 되돌리거나 pass로 바꾸지 않았고, 세부 command·시간·failure nodeid·output hash는 `var/reports/strategy_research_20260812/strategy_research_verification_receipt.json`에 봉인했다.

## 2026-07-09 KST — 알파 풀 v2c data-PC cost-grid 재측정: 전원 리젝트

`private/main` 최신 `28a06e02`(alpha-pool-expansion-v2c)까지 가져온 뒤 추가 핸드오프 `alpha_pool_expansion_v2c_handoff.md`를 같은 research-only/data-PC 경계로 실행했다. v2c 전략 후보는 기본 candidate library에서 9개 `strategy_class`만 필터링해 구성했다: 54개 후보 row. 10번째 family meta-momentum lane은 `@register` 전략 row가 아니라 `quality_gated_allocation.py`의 오프라인 allocator manifest route라 별도 3-arm 측정으로 기록했다. 실행 경계는 유지했다: real-money/paper/testnet/live/order execution 전부 0회, 신규 배분 0%.

측정 조건:
- Profile: `configs/profiles/backtest_cost_realistic.yaml`
- Strict research gate: `use_lockbox_split=true`, `purge_embargo_bars=1`, HAC DSR, `enforce_selection_reject_gate=true`, DSR floor `0.90`, SPA ceiling `0.05`, PBO ceiling `0.50`
- Cost grid: `cost_rate_bps_override = 10 / 15 / 20 / 30`
- Candidate report: no-survivorship — v2c 54/54 row를 모두 보고하고, 데이터 부족 row도 리젝트로 보존
- Local artifacts: `/tmp/lq_alpha_pool_v2c_manifest.json`, `/tmp/lq_alpha_pool_v2c_eval/combined_summary.json`, `/tmp/lq_alpha_pool_v2c_allocator_3arm.json`

결과는 v2/v2b와 똑같이 전원 리젝트다. 네 비용 셀 모두 `reported_candidate_count=54`, `pass_count=0`, `hard_reject_count=54`, `shortlist_count_after_strict_merge_gate=0`이다. 비용별로 52개 row는 현재 로컬 parquet coverage에서 `insufficient_data`로 닫혔고, 실제 데이터로 평가된 2개 row는 둘 다 `CrossSectionalOffSessionTugOfWarStrategy`이며 `oos_sharpe`, `max_drawdown`, `deflated_sharpe`, `spa_pvalue`, `stress_x2_sharpe`, `stress_x3_sharpe`에서 실패했다. 상대적으로 덜 나쁜 1위 row는 `offsession_tugofwar_1h_tow_42d_fade`였지만 20bps 기준 OOS return `-63.19%`, OOS Sharpe `-33.95`, MDD `63.85%`, DSR `0.0`, SPA p `1.0`이라 승격 근거가 아니다.

Allocator lane 10도 promotion 신호가 없다. 20bps candidate report의 train+validation return stream을 입력으로 base M2 HRP, MR1 turnover-tilt+shrinkage, family meta-momentum tilt 3-arm을 동일 window에서 재생했고, 비어 있지 않은 stream은 2개뿐이며 세 arm 모두 survivor `0`, children `0`, `cash_fail_closed_no_promotion`으로 닫혔다. 그래서 v2/v2b 121개 row + v2c 54개 row + allocator route까지 이번 최신 배치 판정은 **do_not_promote / research_only_no_execution / 실배분 0% 유지**다. strict gate를 통과한 후보가 없으므로 paper/testnet/live/real-money/order 경로로 이어지는 변경은 없다.

## 2026-07-09 KST — 알파 풀 v2/v2b data-PC cost-grid 재측정: 전원 리젝트

최신 `private/main`을 `204461ee`까지 fast-forward한 뒤 v2/v2b 핸드오프(`alpha_pool_expansion_v2_handoff.md`, `alpha_pool_expansion_v2b_handoff.md`)의 research-only 후보를 data-PC 모드로 재측정했다. 매니페스트는 기본 candidate library에서 정확한 wave-1/v2b `strategy_class`만 필터링해 구성했다: 20개 전략 클래스, 121개 후보 row. MR1/E1은 후보 row가 아닌 allocation/engine 배선이고 X1은 기존 `VolManagedRiskOverlayStrategy`의 config-gated 파라미터라 이 row 매니페스트에 넣지 않았다. 실행 경계는 유지했다: real-money/paper/testnet/live/order execution 전부 0회, 신규 배분 0%.

측정 조건:
- Profile: `configs/profiles/backtest_cost_realistic.yaml`
- Strict research gate: `use_lockbox_split=true`, `purge_embargo_bars=1`, HAC DSR, `enforce_selection_reject_gate=true`, DSR floor `0.90`, SPA ceiling `0.05`, PBO ceiling `0.50`
- Cost grid: `cost_rate_bps_override = 10 / 15 / 20 / 30`
- Candidate report: no-survivorship — 121/121 row를 모두 보고하고, 데이터 부족 row도 리젝트로 보존
- Local artifacts: `/tmp/lq_alpha_pool_v2_v2b_manifest.json`, `/tmp/lq_alpha_pool_v2_v2b_eval/combined_summary.json`

결과는 깔끔하게 죽었다. 네 비용 셀 모두 `reported_candidate_count=121`, `pass_count=0`, `shortlist_count_after_strict_merge_gate=0`이다. 비용별 hard reject는 전부 121/121이고, 실제 데이터로 평가된 row는 4개뿐이며 나머지 117개는 현재 로컬 parquet coverage에서 `insufficient_data`로 닫혔다. 평가된 4개도 전부 `oos_sharpe`, `deflated_sharpe`, `spa_pvalue`, `stress_x2_sharpe`, `stress_x3_sharpe`에서 실패했다. 20/30bps에서는 4/4가 `max_drawdown`까지 같이 실패했다. 상대적으로 덜 나쁜 1위 row는 모든 비용 셀에서 `stationarity_gated_residual_reversion_4h_strict_adf`였지만 20bps 기준 OOS return `-39.26%`, OOS Sharpe `-18.41`, MDD `40.52%`, DSR `0.0`, SPA p `1.0`이라 승격 근거가 아니다.

판정: **do_not_promote / research_only_no_execution / 실배분 0% 유지**. 이번 v2/v2b 배치는 기대했던 null을 확인한 falsification 결과이며, strict gate를 통과한 후보가 없으므로 paper/testnet/live/real-money/order 경로로 이어지는 변경은 없다.

## 2026-07-09 KST — 알파 풀 확장 v2 배치(9개 레인) 저작 완료 + data-PC 2-tier 게이트 핸드오프

ralplan 컨센서스(`.omc/plans/alpha-pool-expansion-consensus.md`, Planner→Architect R1/R2→Critic 2라운드→APPROVE)를 team으로 실행해 v2 알파-헌트 배치를 저작했다. 원칙은 이전 메타-스파인 배치와 동일하다: **이 PC는 코드+테스트+CI만, 백테스트/데이터 수집 없음(build-not-measure)**, 모든 수익/OOS/비용 수치는 data-PC 몫, 신규 슬리브는 W3 원자 통합 전까지 `@register` 미적용(미등록=완전 비활성), 실배분 0% 유지.

정직한 배너를 먼저 박는다: **이번 사이클은 A1에 EV가 집중된 falsification 프로토콜이지 엣지 약속이 아니다.** A1(52주 고점 근접 앵커링)만 강한 외부 crypto-specific OOS prior(Jia et al. 2025 JBF, long-short ≈ +130bps/주 GROSS, 비용 미검증)을 갖고, L1/L2/A2/N4는 각자 명시적 EXPECTED-NULL을 지닌 저-prior 직교 falsification 프로브다(L1=쇠퇴 팩터, L2=BTC 상관에 희석되는 분산수익, A2=조상처럼 죽을 확률 높음(graveyard #6), N4=비용 취약). 모든 후보는 CORE 포함 **2-key 게이트**를 통과한다: (key1) 저작 시점 결정론 divergence BUILD 게이트(레인 내부에서 keep/drop 해소 완료), (key2) data-PC 한계 orthogonal `factor_ic` 어드미션 게이트. "CORE"는 high-prior일 뿐 게이트 면제가 아니다.

저작 결과(9 레인, 전부 `research_only`, live 맵에 없음):
- **W1 CORE 3종** — A1 `CrossSectionalNearHighAnchoringStrategy`(`0f99e48`, family=cross_sectional; B2 하드 빌드 게이트+breadth-fallback 테스트: `<52wk` young alt는 `max_available` 폴백으로 admit, 최소 이력 미달만 skip), L1 `LowTurnoverTrendPersistenceStrategy`(`bcae576`, family=time_series_momentum 단일자산; MTTE 직교성 테스트+min-hold 레스큐 테스트), L2 `RebalancingPremiumHarvestStrategy`(`6e08e6f`, family=cross_sectional; forecast-free 속성 테스트).
- **W2 신규-파일 조건부/변형 3종** — A2 `SlowCrossSectionalLeadLagStrategy`(`7408f0e`, 3-인컴번트 divergence 게이트 `CrossCryptoSlowDiffusion`/`SemisLeadLagRotation`/`IntermarketLeadLagContinuation` + RPT 설계 속성, first-to-cut이나 통과), N4 `StationarityGatedResidualReversionStrategy`(`a99c568`, `EquityBenchmarkResidualReversal` 대비 ADF 게이트 격리 2-케이스 테스트), MR2 `RegimeAdaptiveDisagreementEnsembleStrategy`(`f2cb076`, 독립 변형 클래스, verdict 8 반영해 Sharpe-lift 아닌 variance-reduction/MDD-Calmar 논지; divergence 통과).
- **W2 config-gated-OFF 골든-패리티 배리어 3종** — MR1 M2 할당기 턴오버-페널티(`net_sharpe_20bps − λ·turnover`)+손계산 Ledoit-Wolf 수축(`42752f0`, 오프라인, `@register` 없음), X1 기존 `VolManagedRiskOverlayStrategy`에 `vol_estimator` 파라미터 추가(`49ed97d`, 기본 `close_to_close`=바이트 동일, opt-in parkinson/garman_klass/yang_zhang), E1 alpha-zoo WF 러너 `_candidate_overfit_stats`가 `spa_pvalue`(`spa_like_pvalue`)+`approx_pbo`를 validation 블록에 방출(`1eeb1f7`, `emit_candidate_overfit_stats` armed일 때만, 기본 OFF 바이트 동일).
- **W3 원자 통합**(`b8edf44`) — 신규 6개 클래스(A1/L1/L2/A2/N4/MR2)를 한 커밋에 `@register`+`_STRATEGY_TIER_HINTS[...]="research_only"`+thin 빌더(`_build_*_candidates`)로 등록, manifest 스냅샷 재고정(sha `fa49ae95fa502b6a9f053a701d9ff623e28a111f7475d3a0f6e5d7973ea5dfaf`), hardcoded-params 베이스라인 재생성(new=0), tier-guard 그린, 6종 모두 `live_default` 아님 확인. MR1/X1/E1은 신규 `@register` 없이 flag-OFF 바이트 동일 재검증. 이 시점 full suite 3725 passed.

E1은 whole-search DSR을 방출하지 **않는다**: 러너의 `_period_metrics`는 후보 1개 슬라이스만 보므로 검색-전역 `num_trials=candidate_count` 기반 deflated_sharpe를 구조적으로 계산할 수 없다. 그래서 코드는 후보별 `spa_pvalue`+`approx_pbo`만 방출하고, whole-search DSR 활성화는 data-PC 워크드-예제 문서로 넘긴다(`research_runner.py:6649`의 aggregation-layer `num_trials=candidate_count` 선례; `selection.py:347-353`가 `num_trials=1` DSR을 금지). 상세 배선은 `docs/research_note/overfit_selection_gate_integration.md` §10에 추가했다.

핸드오프 요지(전문은 [`alpha_pool_expansion_v2_handoff.md`](alpha_pool_expansion_v2_handoff.md)): **2-tier 게이트** — 풀 어드미션(완화: net-20bps 엣지>0 AND N_eff 후 DSR>0 AND 기존 슬리브 대비 한계 orthogonal factor_ic>0; default 세트 이길 필요 없음) vs 라이브 승격(불변 strict (a)-(f)). **NO-SURVIVORSHIP 의무(B6b)**: data-PC 리포트는 리젝트 포함 저작된 모든 후보를 각자의 prior-of-death·실측 falsifying measurement·EXPECTED NULL 대비 결과와 함께 보고해야 한다(조용히 빠진 후보=생존자 편향 결함). 운영: 정확한 family+strategy_class 문자열, clean-WF(train/validation-only 선택, locked-OOS report-only monthly, `no_nested_oos_mining=true`), **비용 그리드 10/15/20/30bps + `backtest_cost_realistic.yaml` 어드미션 필수**(리서치 스코어러의 선형/무펀딩 비용은 낙관적), cross_sectional 슬리브(A1/A2/L2/N4)에 `allow_multi_asset=True` 필수(가짜 carry 태그 금지; L1은 단일자산이라 미부여), **L1/A2는 split당 RPT≥10bps 필수**(#14가 실패한 바로 그 게이트), **A1 horizon factor_ic 스윕 10/20/30/52wk vs NearHighMomentum**, MR1 M2 manifest-provenance 체크리스트, per-fold admit+최종 merge 양쪽을 `selection.apply_selection_reject_and_dedup`(strict robust_score_params)로 라우팅, `emit_candidate_overfit_stats=ON`(research.yaml), G005 완주+G006 cost-stress+G007 decision 권고.

정직한 기대: 리프의 EXPECTED NULL은 대부분 `reject`다 — A1은 published GROSS 효과가 주간 회전 비용을 못 넘김, L1은 저회전에서 비용은 넘어도 인컴번트 추세 앙상블을 못 이김, L2는 BTC 상관이 분산수익을 비용 밑으로 붕괴, A2는 조상(#6)처럼 20-30bps에서 사망(높은 확률), N4는 회전이 엣지를 갉아먹음. 죽음을 보고하는 것이 산출물이다. 따라서 최종 판정은 유지: **do_not_promote / research_only_no_execution / 실배분 0%**, data-PC 2-tier 게이트가 실제로 돌기 전까지 paper/testnet/live/real-money/orders 전부 금지. 다음 단계는 브랜치 push→GitHub CI 그린→PR→main 머지, 그 뒤 data-PC 측정.

## 2026-07-06 KST — Alpha scoreboard persisted + historical champion 재확인

사용자 지시로 2026-07-04 전수 walk-forward 결과에 새 `alpha_scoreboard` persistent runner를 적용해 전체 성적표를 저장했다. 입력 snapshot은 `var/reports/alpha_scoreboard/full_walkforward_score_input_snapshot_latest.json`, validation scoreboard는 `alpha_scoreboard_full_walkforward_validation_latest.json|md`, locked-OOS report-only scoreboard는 `alpha_scoreboard_full_walkforward_oos_report_only_latest.json|md`, 종합 리뷰는 `full_walkforward_score_review_latest.json|md`다. `min_trades=5`, `max_mdd=0.3` gate를 적용했고 source rows에는 measured `liquidation_count`가 없어 liquidation gate는 일부러 켜지 않았다.

전체 회계: unique candidates `1400`, candidate-fold rows `5600`, source reports `16`. 단일 fold라도 pass한 후보는 `28`, 모든 available folds를 통과한 후보는 `0`이다. Fold별 pass율은 `WF202604 2/1400 (0.14%)`, `WF202605 4/1400 (0.29%)`, `WF202606 19/1400 (1.36%)`, `WF202607_PARTIAL 4/1400 (0.29%)`다. 따라서 scoreboard를 persist한 뒤에도 최종 판정은 **do_not_promote / research_only_no_execution**이다.

Validation composite 상위는 BTC/TRX 1h pair-spread 계열이다. `1523ae68946b002a` `pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70`, `c01a3510bb5eab15` tightstop/tp, `f54d2780eb950ba8` takeprofit가 모두 validation `+9.77%`, Sharpe `2.48`, MDD `3.38%`, trades `22`였지만 PBO `0.75`, DSR `0.0832`, pass `0/4`라 실행 후보가 아니다. OOS report-only composite 1위는 `1f1fd241c12f0bc2` `pair_spread_4h_balanced_btcusdt_bnbusdt_1.6_0.35`로 OOS report-only `+10.15%`, Sharpe `4.74`, MDD `1.30%`, trades `9`, PBO `0.75`, DSR `0.0585`, pass `0/4`다. Pair-spread lane은 watchlist로 챙기되, 현재 파라미터를 승자로 보거나 추가 튜닝하는 것은 금지한다.

Historical champion도 다시 명시한다. 최고 raw historical/proxy 성적은 `var/reports/major_candidate_improvement_20260705/historical_router_guard_proxy_latest.json`의 `expanded_110_latest_tail_full` / `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`다. 10 monthly OOS folds에서 compounded OOS `+159.83%`, max OOS MDD proxy `27.69%`, monthly Sharpe proxy `1.98`, positive folds `4/10`, PF proxy `23.63`이다. Historical 85 preregistered 기준 최대는 compounded OOS `+125.13%`, max OOS MDD proxy `30.47%`, monthly Sharpe proxy `2.64`, positive folds `5/10`이며, 보수형 `post_mdd12_half_exposure`는 return `+85.84%`, MDD proxy `20.26%`, monthly Sharpe proxy `2.77`, positive folds `5/10`다.

우선순위는 `historical router / lagged leaf router` lane을 1순위 watchlist, BTC/BNB·BTC/TRX market-neutral pair-spread를 2순위 watchlist로 둔다. 단 historical router artifact는 exact bar-level rerun이 아니라 fold-level proxy이고, 기존 OOS row 기반 post-OOS research variant이며, PF 20+와 outlier-month 의존이 강해 cost/funding/fill realism 전에는 과대평가 가능성이 크다. 다음 검증은 frozen-rule fresh-forward + cost-realistic 10/15/20bps + funding/BBO/slippage/reconciliation telemetry가 전제이며, 그 전까지 paper/testnet/live/real-money/orders는 계속 금지다.

## 2026-07-05 KST — Alpha101 안정화 후보 research freeze + 즉시 fresh probe

사용자 지시로 개선 실험에서 살아남은 Alpha101 안정화 후보 2개를 research freeze 했다. Freeze manifest는 `var/reports/major_candidate_improvement_20260705/alpha101_research_freeze/alpha101_research_freeze_manifest_latest.json`, exact stream/covariance evidence는 `alpha101_research_freeze_exact_stream_covariance_latest.json`, 요약은 `alpha101_research_freeze_latest.md`다. Frozen candidates는 `4aee63220c8221ec` `alpha101_formula_4h_a011_flow_swing_tight_stop_z14`와 `3fff476361bd1f80` `alpha101_formula_4h_a011_flow_swing_strict_z14`이며, freeze 뒤에는 formula/threshold/membership/weights 변경 금지로 박았다.

동일가중 exact-stream evidence: validation aggregate stream 1430 bars에서 portfolio return `+9.38%`, annualized 4h Sharpe proxy `1.042`, MDD `7.69%`; locked-OOS diagnostic aggregate stream 552 bars에서 return `+10.66%`, Sharpe proxy `2.869`, MDD `6.56%`다. 이 evidence는 exact return streams/covariance/correlation을 갖췄지만 과거 validation/diagnostic 기반이므로 승격 근거가 아니라 freeze 근거다.

가용 데이터로 즉시 micro fresh probe도 실행했다. `2026-07-04T00:00:00Z`~`2026-07-04T07:15:59.999Z` 구간은 기존 partial OOS 종료 뒤 남은 데이터라 probe를 돌렸지만 4h 전략 기준 OOS return stream length가 두 후보 모두 `0`, trades `0`이었다. 산출물 `alpha101_research_freeze_micro_fresh_probe_summary.json|md`의 verdict는 `insufficient_fresh_forward_no_4h_oos_return_streams`다. 따라서 결론은 여전히 **do_not_promote / research_only_no_execution**이며, 다음 유효 검증은 새 4h bars가 충분히 쌓인 genuinely unseen fresh-forward/shadow 구간에서만 가능하다.

## 2026-07-05 KST — 주요 후보 개선 실험 전수 진행

사용자 지시로 주요 후보 개선을 모두 진행했다. 산출물은 `var/reports/major_candidate_improvement_20260705/major_candidate_improvement_walkforward_summary_latest.md|json`, 검증은 `major_candidate_improvement_verification.json`(`status=passed`)이다. 2개 신규 후보 lane(`pairspread_btcbnb_robustness` 10개, `alpha101_flow_stability` 9개)을 4개 fold(`WF202604`~`WF202607_PARTIAL`)에 재평가했고, P2/H35 manifest-level `TONUSDT` 제거본과 historical router guard proxy 분석도 생성했다. 총 8/8 lane-fold reports 존재, failure 0, P2/H35 sanitized manifests의 remaining `TONUSDT` occurrences 0이다.

개선 결과:
- `pairspread_btcbnb_robustness`: 실패. 보수형/ATR/VWAP/take-profit/low-turnover 변형 모두 promotion 관점에서 lane을 살리지 못했다. 상위 변형도 validation Sharpe가 음수이거나 OOS Sharpe가 음수이고, 전부 `hard_reject_count=4/4`다. 예: `pair_spread_4h_btcbnb_balanced_vwap_volume_1.8_0.45`는 validation `+0.17% / -0.895`, OOS diagnostic `+0.09% / -0.833`; `pair_spread_4h_btcbnb_fast_robust_corr25_1.8_0.45`는 validation `+2.07% / 0.408`, OOS diagnostic `+1.00% / -6.688`.
- `alpha101_flow_stability`: 의미 있는 research 개선이 나왔다. 원본 `19d07d85cab54789`는 validation `+10.68% / 2.944`지만 OOS diagnostic Sharpe가 `-2.576`이었다. 변형 중 `4aee63220c8221ec` `alpha101_formula_4h_a011_flow_swing_tight_stop_z14`는 validation `+2.66% / 1.073`, OOS diagnostic `+2.49% / 3.023`, OOS return+Sharpe 양수 fold `3/4`, pass/hard `2/2`; `3fff476361bd1f80` `alpha101_formula_4h_a011_flow_swing_strict_z14`는 validation `+2.08% / 0.791`, OOS diagnostic `+2.70% / 3.142`, OOS return+Sharpe 양수 fold `4/4`, pass/hard `2/2`. 즉 원본의 validation headline은 낮아졌지만 OOS diagnostic 안정성은 개선됐다.
- P2/H35: `p2_corr_core_tonusdt_excluded_research_manifest_latest.json`와 `h35_return_overlay_80_20_tonusdt_excluded_research_manifest_latest.json`를 생성했고 manifest-level `TONUSDT`는 0으로 제거됐다. 단, 이것은 manifest sanitize일 뿐 exact fresh-forward portfolio streams/covariance 검증이 아니므로 계속 research-only다.
- historical router: fold-level proxy guard를 시험했지만 대부분 base/no-overlay가 return-to-MDD에서 그대로 우세했다. historical_85 fallback만 `post_mdd12_half_exposure`가 MDD를 낮추지만 return도 크게 포기한다. 이 분석은 기존 OOS fold row 기반 proxy라 fresh-forward 전 승격 근거가 아니다.

최신 결론: 개선 실험 후에도 **실행 승격 없음**. 다만 다음 research freeze 후보는 기존 rank 1 원본이 아니라 `alpha101_formula_4h_a011_flow_swing_tight_stop_z14`와 `alpha101_formula_4h_a011_flow_swing_strict_z14` 두 Alpha101 안정화 변형이다. 둘 다 post-hoc 개선 후보이므로 신규 unseen fresh-forward/shadow 검증 전에는 paper/testnet/live/real-money/order 사용 금지다.

## 2026-07-05 KST — 전체 전략 관점 최종 결론 업데이트

사용자 요청에 따라 `G005` 후보군만 보지 않고, 기존 historical/current-top 계열과 2026-07-04 최신 전수 walk-forward 정정판을 함께 놓고 결론을 재정리했다. 최종 판정은 유지한다: **현재 실행 승격할 전략/포트폴리오는 없다.** 모든 후보는 research/shadow-only이며 live/paper/testnet/real-money/orders는 계속 disabled, `TONUSDT`는 최신 안전 기준에서 excluded다.

기간 구분:
- 최신 전수 monthly walk-forward 정정판: Binance fapi 1m universe 128 symbols, `2025-01-01T00:00:00Z` → `2026-07-04T07:15:59.999Z`. Fold는 `WF202604`~`WF202607_PARTIAL`이며 선별은 expanding train + 직전 2개월 validation만 사용하고 locked OOS는 report-only diagnostic이다.
- 기존 최고 성적 historical/current-top 계열: 10 monthly OOS folds `2025-09`~`2026-06`. 최신 2026-06 fold는 train `2025-01-01`~`2026-03-31`, validation `2026-04-01`~`2026-05-31`, OOS `2026-06-01`~`2026-06-28T09:30` 기준으로 재삽입/재계산된 산출물이다.

기존 최고 성적 전략들의 상태:
- `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`는 110-symbol 최신 재계산 기준 historical headline winner다. 10-fold compounded OOS `+159.83%`, annualized approx `+214.51%`, monthly Sharpe `1.881`, max OOS MDD `27.69%`, hit `4/10`. 성과는 가장 크지만 hit가 낮고 drawdown/최신 fold 의존이 커서 실행 승격 근거가 아니다.
- risk-trimmed `...fallback_mdd20_cap2`는 compounded OOS `+138.11%`, monthly Sharpe `1.783`, max OOS MDD `23.58%`, hit `4/10`로 drawdown은 낮추지만 역시 shadow/research 성격이다.
- 기존 85-symbol preregistered exact는 최신 2026-06 fold 재실행 뒤 compounded OOS `+125.13%`, monthly Sharpe `2.506`, hit `5/10`, latest June return `+24.76%`, latest June MDD `30.47%`로 내려갔다. 살아는 있지만 최신 전체 기준 winner는 아니다.
- `P2_corr_core`는 long OOS `+53.85%`, Sharpe `1.958`, MDD `3.43%`, fresh 10bps `+3.79%`로 balanced shadow 후보이고, `H35_return_overlay_80_20`은 long OOS `+73.49%`, Sharpe `1.941`, hit `6/10`, fresh 10bps `+3.91%`인 return-seeking shadow 후보다. 둘 다 fresh-forward 기간이 짧고 BBO/fill/slippage/funding/reconciliation telemetry가 없으며, 오래된 manifest 계열에는 `TON/USDT`가 포함되어 최신 fail-closed 기준으로 그대로 승격할 수 없다.

최신 full-pool/G004-G007 관점:
- G004 frozen universe는 1466 candidates. G005는 지원 가능한 30m/1h/4h/1d 1404개를 평가했고, 1s/5m/15m lower-latency 62개는 별도 durable goal 전까지 deferred다.
- G006의 best metric-proxy portfolio는 `914ff36cba1555ea`, `632d6e864ee01bd8`, `1f1fd241c12f0bc2` 3개 4h BTC/BNB pair-spread 조합(각 0.1 weight, gross 0.3)이며 validation objective proxy `0.08065`로 incumbent proxy보다 높았다. 그러나 exact portfolio return streams/covariance가 없어 `blocked_fail_closed_metric_proxy_only`다.
- G007 최종 결정은 `do_not_promote` / `blocked_fail_closed`. 실행 weights는 발행하지 않았다.

최신 2026-07-04 전수 walk-forward 정정판의 research shortlist:
- `19d07d85cab54789` `alpha101_formula_4h_a011_a011_flow_swing_dir`: rank 1, selected folds 3, validation `+10.68% / 2.944`, locked-OOS diagnostic `+3.07% / -2.576`.
- `1f1fd241c12f0bc2` `pair_spread_4h_balanced_btcusdt_bnbusdt_1.6_0.35`: rank 4, selected folds 2, validation `+9.72% / 5.409`, locked-OOS diagnostic `+0.28% / 0.720`; 상위권 중 평균 OOS return/sharpe가 둘 다 양수인 유일한 후보다.
- `632d6e864ee01bd8`, `a27ef46e91376b7c`, `914ff36cba1555ea`는 validation Sharpe가 높지만 평균 locked-OOS diagnostic이 나빠서 research-only다.

통합 결론: historical headline winner는 `codex_lagged_leaf_router...exact_unscaled`, 최신 monthly walk-forward rank 1은 `19d07d85cab54789`, OOS diagnostic까지 같이 보면 가장 덜 나쁜 단일 후보는 `1f1fd241c12f0bc2`, portfolio proxy best는 `914/632/1f1` 조합이다. 하지만 이 네 관점 모두 실행 승격 조건을 충족하지 못한다. 신규 승격 작업은 exact portfolio return streams/covariance, fresh-forward/shadow 검증, no-OOS-use review gate, 명시적 실행 승인 없이는 시작하지 않는다.

## 2026-07-04 KST — 최신 데이터 전수 walk-forward 정정 리포트

사용자 지적으로 이전 fixed-split/최종 3개 알파 리포트는 폐기 수준으로 낮추고, G005 후보군 전체를 월별 walk-forward 방식으로 다시 평가했다. 데이터 refresh 기준은 Binance fapi 1m universe 128 symbols, `2025-01-01T00:00:00Z` → `2026-07-04T07:15:59.999Z`까지이며, 선별 방식은 expanding train + 직전 2개월 validation만 사용한다. 다음 1개월 locked OOS는 fold별 선별이 고정된 뒤 붙이는 diagnostic/report-only 값이며, rank/status/research-selected 여부에는 쓰지 않는다.

평가 fold:
- `WF202604`: train `2025-01-01`~`2026-01-31T23:59:59`, validation `2026-02-01`~`2026-03-31T23:59:59`, locked OOS `2026-04-01`~`2026-04-30T23:59:59`.
- `WF202605`: train `2025-01-01`~`2026-02-28T23:59:59`, validation `2026-03-01`~`2026-04-30T23:59:59`, locked OOS `2026-05-01`~`2026-05-31T23:59:59`.
- `WF202606`: train `2025-01-01`~`2026-03-31T23:59:59`, validation `2026-04-01`~`2026-05-31T23:59:59`, locked OOS `2026-06-01`~`2026-06-30T23:59:59`.
- `WF202607_PARTIAL`: train `2025-01-01`~`2026-04-30T23:59:59`, validation `2026-05-01`~`2026-06-30T23:59:59`, locked OOS `2026-07-01`~`2026-07-04T00:00:00`.

범위/회계: 30m 305개 + 4h 360개 + 1d 299개 + 1h 436개 evaluated = fold당 1400개를 실행했고, native runner stall을 일으킨 1h `Alpha101FormulaStrategy` 4개(`6afebe39638237ca`, `c49978799aff2168`, `82a31b3aa1d93bb0`, `b730cdad557b46e3`)는 timeout-filter fail-closed로 회계에 포함해 fold당 accounted 1404개다. 4 folds × 4 consolidated shards = 16/16 reports present, missing 0, failure 0.

결과: train/validation repeated-selection 기준 research-selected 후보는 22개다. Research-selected 22개의 평균 validation return/sharpe는 `+10.24% / 2.621`; locked-OOS diagnostic 평균 return/sharpe는 `-1.98% / -4.289`이며, 평균 OOS return과 sharpe가 둘 다 양수인 후보는 1/22뿐이다. 따라서 이번 리포트의 올바른 결론은 `research_selected_candidates`이지만 배포 판정은 `no_execution_promotion` / `research_only_no_execution`이다. live/paper/testnet/real-money/orders는 전부 disabled이고 `TONUSDT`는 계속 excluded다.

상위 research-selected 후보(선별/랭킹은 train+validation only):
1. `19d07d85cab54789` `alpha101_formula_4h_a011_a011_flow_swing_dir` — selected folds 3, validation `+10.68% / 2.944`, OOS diagnostic `+3.07% / -2.576`.
2. `632d6e864ee01bd8` `pair_spread_4h_fast_cycle_btcusdt_bnbusdt_1.6_0.35` — selected folds 2, validation `+10.06% / 6.158`, OOS diagnostic `-0.24% / -20.896`.
3. `a27ef46e91376b7c` `pair_spread_4h_fast_cycle_btcusdt_bnbusdt_1.8_0.45` — selected folds 2, validation `+9.61% / 5.322`, OOS diagnostic `-0.14% / -22.303`.
4. `1f1fd241c12f0bc2` `pair_spread_4h_balanced_btcusdt_bnbusdt_1.6_0.35` — selected folds 2, validation `+9.72% / 5.409`, OOS diagnostic `+0.28% / 0.720`.
5. `914ff36cba1555ea` `pair_spread_4h_balanced_btcusdt_bnbusdt_2.0_0.50` — selected folds 2, validation `+9.02% / 5.039`, OOS diagnostic `-0.54% / -23.974`.

산출물:
- Final report: `var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_latest.md|json|csv`.
- Verification: `var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_verification.json` (`status=passed`, blockers empty).
- Cleanup: `var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_cleanup.json` (`status=passed`, blocking_findings empty).
- Run summary: `var/reports/latest_alpha_refresh_20260704_full_walkforward/full_walkforward_consolidated_run_summary.json`.
- Repro scripts preserved under `var/reports/latest_alpha_refresh_20260704_full_walkforward/repro_scripts/`.

## 2026-07-03 KST — full-pool Ultragoal G004 checkpoint + G005 stop/handoff

Full-pool Ultragoal durable run was resumed from `/tmp/.gjc/_session-019f22a1-90f7-7000-ab18-d0fd7010803b/ultragoal` and advanced through `G004`. G004 is now checkpointed complete with frozen search-budget artifacts under `var/reports/ultragoal_full_pool_strategy/`: `g004_search_budget_manifest.json`, `g004_frozen_candidate_manifest.json`, `g004_verification_test_report.json`, and `g004_ai_slop_cleanup_report.json`. Frozen budget: 1466 candidates after fail-closed `TONUSDT` quarantine exclusion, 23328 portfolio-grid combinations, total effective-trials denominator 24794, locked-OOS selection/tuning/tie-break/weight use disabled, and live/paper/testnet/real-money execution disabled.

`G005` remains active and not complete. Full single-process evaluation was too slow, so completed-bar candidates were split into 30m/1h/4h/1d manifests. 30m, 4h, and 1d shard final artifacts exist; 1h was still incomplete at user-requested stop and was further split into eleven 40-candidate chunk manifests (`g005_walkforward_candidate_manifest_1h_chunk_01.json` … `_11.json`, index `g005_1h_chunk_index.json`). Running monitors/processes were stopped for new-session resume. Resume from `docs/research_note/full_pool_ultragoal_resume_20260703.md` and `var/reports/ultragoal_full_pool_strategy/g005_session_stop_handoff_20260703.json`; do not checkpoint G005 complete until the 1h chunks are rerun/merged and the mandatory cleanup/review gate is clean.

## 2026-07-03 KST — Alpha-hunt 메타-스파인 배치: disagreement 앙상블 + 오프라인 품질-게이트 할당기 + flow-share 로테이션 (+확인형 레짐 라우터) — data-PC 핸드오프

"현 전략 세트를 이길 알파" 요청에 대한 ralplan 컨센서스(Planner→Architect→Critic 2라운드, APPROVE) 실행 결과. 핵심 결정: **단일 leaf 알파는 locked-OOS에서 반복 붕괴해 왔으므로(2026-06-08 -8.77%, deep-research leaf 비용열화 등) 스파인은 결합(meta)+할당(allocation)이고, 신규 leaf는 정확히 1개만** — 77개 live_default 슬리브 대비 직교성이 이 PC에서는 측정 불가하므로, 2번째 leaf부터는 data-PC의 한계 orthogonal factor_ic 게이트 뒤로 이연(N4 stationarity residual reversion 등). 백테스트/데이터 다운로드 없음, 실배분 0% 유지.

구현(5 커밋: `02694c8`/`d12a8b2`/`cacf12d`/`ad01d7c`/`14030b0`):
- **M1 `DisagreementGatedEnsembleStrategy`** (family=trend, per-symbol single-asset): 4개 인과 OHLCV 컴포넌트(EMA-slope TSMOM, rolling-z 리버전, Donchian 위치, 효율비-부호 추세)를 직전 바 예측 vs 실현수익으로 채점(`direction_hit_rate`)→`inverse_error_weights` 적응 가중, **`disagreement_coefficient`(CV) 게이트가 컴포넌트 불일치 시 진입 자체를 차단**(합의 시에만 ±entry_band 진입, 이후 ATR 트레일링 라이드). `ensemble_weights` 모듈 첫 전략 소비자. 이론: Bates-Granger 1969, Timmermann 2006, Krogh-Vedelsby 1995.
- **M2 오프라인 품질-게이트 슬리브 할당기** (`portfolio/quality_gated_allocation.py` + `scripts/research/build_quality_gated_allocation.py`, **live 표면 없음**): 슬리브별 정적 품질점수(20bps `CostRegime`으로 `apply_cost_drag`→net Sharpe/Calmar via `optimizer_core.metrics`, hit-rate 자체계산)→net_sharpe≤0 탈락→ERC/HRP 가중→`ArtifactPortfolioModeStrategy`가 fail-close 없이 수용하는 **provenance-완전 manifest** 방출. 실소비자 왕복 happy-path + **fail-closed 사유 14종 파라미터라이즈** + byte-golden(sha256 `27438660…`) 테스트. 이론: Maillard-Roncalli-Teïletche 2010 ERC, López de Prado 2016 HRP.
- **N1 `CrossSectionalFlowShareRotationStrategy`** (family=cross_sectional 바스켓, `flow_share` 모듈 첫 소비자): 심볼별 달러볼륨의 유니버스 점유율→롤링 share-z + `cdf_extremeness`; 점유율 상승+수익 확인 롱 / 점유율 붕괴·블로우오프(극단 점유율+음수 수익) 숏; |z| 상위 n, 역변동성 사이징. **정직 어드미션**: carry 태그 미부착→기본 숏리스트에서 의도적으로 제외되며, 평가는 `select_diversified_shortlist(..., allow_multi_asset=True)`로만(테스트가 양쪽 모두 고정). 이론: Gervais-Kaniel-Mingelgrin 2001, Barber-Odean 2008, Amihud 2002.
- **I2 `RegimeRouterConfirmedRotationStrategy`** (family=cross_sectional, BullBear 부모와 동일 어드미션): 부모의 breadth+BTC 기반 투표에 **GARCH 조건부변동성 확인**(상승 vol만 bear 확정; GARCH 불가 시 spectral phase 폴백)과 3-상태 스티키 히스테리시스를 추가. **비중복 게이트 통과**: 동일 chop 픽스처에서 부모는 bear-short 진입(기반 투표 충족을 별도 단언), I2는 확인 불충족으로 CHOP 유지 — 진짜 하락+상승 변동성에서는 정상 진입(게이트 생존 증명).
- **라이브 안전(이번 배치의 인프라 기여)**: 저작 단계는 `@register` 미적용(미등록 모듈은 레지스트리에 완전 비활성임을 검증), 통합 커밋에서 등록+`research_only` 힌트를 **원자적으로** 적용. 신설 `tests/test_strategy_tier_guard.py`가 하드 게이트: 등록된 모든 클래스는 힌트/레거시맵/**동결 68종 legacy 스냅샷**(append 금지) 중 하나여야 하며, 힌트 누락 시 CI 실패(멤버십 단언 — tier 값으로는 누락 힌트를 구별 불가). manifest 스냅샷 145→149, hardcoded-param 베이스라인 1004 시그니처 재고정.

검증: 독립 verifier PASS — full suite **3010 passed / 21 skipped**, ruff 클린, 3클래스 research_only+live 세트 제외 확인, 가드 실효성(힌트 1개 제거 시뮬레이션→정확히 그 클래스 검출), M1/I2 인과성 스팟체크(직전 바 점수를 당 바 실현수익으로 채점; GARCH 예측은 t−1까지의 수익만 사용), golden 안전(additive-only diff 확인).

### Data-PC 핸드오프 — 승격 결정규칙 (verbatim)
평가 대상: `DisagreementGatedEnsembleStrategy`(disagreement_ensemble_* 후보 28개@7심볼), `CrossSectionalFlowShareRotationStrategy`(flow_share_rotation_* 8개, **allow_multi_asset=True 필수**), `RegimeRouterConfirmedRotationStrategy`(regime_router_confirmed_* 4개), M2 할당기는 슬리브 수익스트림 확보 후 manifest 생성 경로로 평가. Clean-WF 규칙: train/validation-only 선택, locked-OOS report-only monthly WF, `no_nested_oos_mining=true`, 10/15/20/30bps 비용 그리드.
- (a) 20bps 기준비용에서 신규 결합 북의 net-of-cost DSR-조정 Sharpe/IR > 현 default 세트 — **동일 WF 윈도·동일 비용모델**로 비교.
- (b) 30bps에서 default 세트 미만으로 열화 없음.
- (c) `evaluate_survivorship_gate`/`effective_number_of_trials` 기준 N_eff 페널티 후 DSR > 0.
- (d) leaf별 한계 테스트: 기존 슬리브 대비 incremental orthogonal factor_ic > 0 (N1 게이트; 이연된 N4/2nd leaf도 동일).
- (e) 최신-OOS coverage-gate 통과.
- (f) 사인오프 전까지 실배분 0%.
M2 manifest 체크리스트: real_money 키 전부 false(톱레벨+자식), oos 클린, optimizer/correlation provenance(source+selection_inputs, ready=True), source-artifact id/path/sha/freshness, 자식별 no_current_fold_oos + train_validation provenance. 후속(비차단): 미힌트 default를 research_only로 뒤집는 fail-safe 강화(H1), N1이 성과를 입증하면 selection.py allowlist 편입, `_restore_deque`의 truthy 비반복자 TypeError(공유 헬퍼) 수정.

사용자 요청으로 `HokyoungJung/DeepLearning`(로컬 `/home/hoky/DeepLearning`), `D:\PythonProjects\precious_metal`, `HokyoungJung/Reference-Price` 세 레포에서 지표/알파로 이식 가능한 방법론만 추출해 재구현했다. **데이터는 일절 복사하지 않았고 백테스트도 실행하지 않았다**(data-bearing PC 몫). 이미 흡수된 부분과의 중복을 먼저 확인했다: DeepLearning 신경망 계열은 2026-06-20 artifact-only forecast bridge로, precious-metal pair z-score는 `TimeframePairZScoreReversionStrategy`(+2026-05-05 metals alpha 실패 기록)로, lead-lag은 `cross_asset_lead_lag_momentum`으로 이미 커버되어 있어 제외하고, **main 레지스트리에 전혀 없던 기법**(periodogram 주기 추출, ADF 정상성 검정, GARCH 조건부변동성, 거래대금 점유율 지수, 앙상블 결합 가중)만 선정했다.

구현 (전부 순수 Python — scipy/statsmodels/numpy 불사용, 결정론):
- `indicators/spectral_cycle.py`: precious_metal `rfft_periodogram.py`의 demean→rFFT→지배주기 추출을 **인과적으로 재설계**(trailing window + 선형 detrend + 후보주기별 직접 Fourier projection). 지배주기/진폭/최신바 위상/스펙트럼 순도(purity) 반환. 순수 톤 purity≈0.38 vs 백색잡음≈0.07 (누설 때문에 1.0이 아님 — sleeve 기본 임계 0.18~0.20은 이를 반영).
- `indicators/stationarity.py`: precious_metal VECM 노트북의 from-scratch `python_adf_test`를 이식(가우스 소거 OLS, 상수항 ADF, 점근 임계값 1%/5%/10%) + AR(1) half-life. 원 레포 docs/08이 "정량적 pair 스크리닝 부재"를 1순위 과제로 자인한 부분의 구현.
- `indicators/garch.py`: R fGarch 의존을 제거한 GARCH(1,1) — 분산 타게팅(`omega = s²(1−α−β)`) + (α,β) 300콤보 결정론 그리드 QMLE. 동일 입력→비트 단위 동일 파라미터.
- `indicators/flow_share.py`: Reference-Price `amt_ratio_index` 계열 — 거래대금 점유율, dense rank, CDF 극단도 `2|Φ(z)−0.5|`, top-N 점유율 가중 합성수익률.
- `indicators/ensemble_weights.py`: DeepLearning ensemble_strategies/metric — 역오차·softmax·prior blend 가중, disagreement CV 게이트(`std/|mean|`), deadband 방향 적중률. 향후 DL bridge 헬스게이트/시그널 컴바이너 입력용.
- Rider sleeves 3종(kalman/cusum 선례: `_ReturnRiderBase` 상속, per-symbol single-asset, ≥30m, candidate-library 전용, tier hint 없음): `SpectralCycleRiderStrategy`(순도 게이트 + trough/crest 직후 위상 밴드 진입), `AdfGatedReversionRiderStrategy`(ADF가 unit root를 기각하고 half-life가 보유 예산 이내일 때만 log-price z-극단 fade), `GarchInnovationRiderStrategy`(전 바 σ 예측으로 표준화한 `z=r/σ`가 band_k 도달 시 continuation, trailing-quantile jump-소진 게이트로 blow-off 배제). Candidate manifest 133→145 재고정.

적대적 검증(28-agent 워크플로: lookahead/수학/컨벤션/견고성 4렌즈 × finding당 skeptic 2): raw 12건 중 **7건 확정 → 전부 수정**. 주요: `(x−c)**2`의 huge-finite OverflowError(critical, `set_state` 경유 재현됨 → `d*d`로 inf→None 가드 경로 복원), `set_state`의 bare `int()` 크래시(→`_safe_non_negative_int`), spectral 튜닝 상한의 병리 비용(측정 0.85s/bar → 상한 512/128 축소 + (symbol, time_key) 메모이즈), GARCH `refit_every=1×window=8192` 조합 차단, entry metadata의 σ가 표준화에 쓴 σ와 불일치(→ `z·σ == r` 보장 + 회귀 테스트). 반박된 5건(demean 표준화, 소표본 임계값 등)은 미적용.

검증·전달:
- Full suite `2946 passed / 21 skipped`, ruff format/check clean. 신규 결정론 테스트(LCG 시드): 기지 주기/위상 복원, AR(0.5) vs random walk ADF 판별, GARCH 결정론+충격 반응, 게이트 차단 케이스(결정론 추세는 ADF 게이트가 항상 차단; 5% 유의수준의 랜덤워크 오검출 ~0.8%는 통계적 본성이라 테스트는 residual-free 추세 사용), 적대적 `set_state`, candidate wiring, manifest byte-pin.
- CI 1차 실패: quality job **Hardcoded parameter audit** 신규 9건(전부 cadence 1800/클램프 하한/len==3 구조 상수) → 선례(4b2b8e0)대로 `--write-baseline` 갱신으로 통과. **신규 sleeve 추가 시 manifest 스냅샷 + hardcoded-param 베이스라인 2종 재고정이 필수 체크리스트.**
- **PR #38 main 머지 완료**(merge `e0a563b`), feature 브랜치 및 머지 완료된 feat/vibe-adoption 브랜치 로컬·원격 삭제.

한계/다음 단계: 이 sleeve들은 이론적 타당성 + 합성데이터 단위검증까지만 확보한 **search 입력 후보**다. real/shadow 승격 근거 아님(`0%` 유지). 다음은 data-bearing PC에서 clean walk-forward + 10/15/20bps cost stress로 후보군 평가하고, `min_purity`/`min_amplitude_frac`/`band_k`/`max_half_life_bars` 노브를 실데이터로 조정하는 것. flow_share/ensemble_weights는 아직 소비자가 없는 indicator-layer 자산이므로 factor-IC 파이프라인 또는 DL bridge 헬스게이트에 물리는 후속 작업이 자연스럽다.

## 2026-06-29 KST — BullBearRegimeRotationStrategy research sleeve

최신 winner의 headline OOS는 높지만 monthly hit `4/10`이고 상승/하락 추세장을 명시적으로 나눠 먹는 sleeve가 약하다는 점을 겨냥해, 새 OHLCV-only basket strategy를 추가했다. 목적은 기존 TopCap/leaf router를 즉시 대체하는 것이 아니라, 다음 data-bearing walk-forward에서 **bull long / bear short / chop flat** 후보를 검증할 수 있게 만드는 것이다.

구현:
- Strategy: `src/lumina_quant/strategies/bull_bear_regime_rotation.py`.
- Registry tier: `BullBearRegimeRotationStrategy = research_only`.
- Candidate builder: `build_binance_futures_candidates()`에서 crypto basket 5종목 이상일 때 `30m/1h/4h/1d` 후보를 생성한다.
- Rule: cross-sectional momentum breadth와 BTC benchmark return이 상승을 확인하면 strongest names LONG, 하락을 확인하면 weakest names SHORT, 중립/chop에서는 stale beta를 EXIT한다.

상태:
- Backtest/shadow promotion은 아직 없음.
- `ready_for_real=false` 유지.
- 다음 검증은 10/15/20bps 비용, bull/bear/chop bucket, hit-rate 개선, outlier-month 의존도 제거 기준으로 incumbent와 비교한다.

검증:
- `uv run pytest tests/test_bull_bear_regime_rotation.py` → `7 passed`.
- `uv run pytest tests/test_bull_bear_regime_rotation.py tests/test_candidate_manifest_snapshot.py tests/test_strategy_registry_defaults.py tests/test_strategy_factory_library.py -q` → `45 passed`.
- `uv run pytest tests/test_bull_bear_regime_rotation.py tests/test_candidate_manifest_snapshot.py tests/test_strategy_registry_defaults.py tests/test_strategy_factory_library.py tests/test_param_registry.py tests/test_scan_param_registry_script.py -q` → `52 passed`.
- `uv run ruff check src/lumina_quant/strategies/bull_bear_regime_rotation.py src/lumina_quant/strategies/registry.py src/lumina_quant/strategy_factory/candidate_library.py tests/test_bull_bear_regime_rotation.py` → pass.

## 2026-06-28 KST — 최신 데이터 포함 전체 WF 재평가 정정

사용자 지적대로 "데이터가 있는데 2026-06-21까지만 본" 것은 데이터 부재가 아니라 검증 오류였다. `date=` partition/이전 runner 상태를 최신성 근거로 착각하지 않고, 이번에는 실제 1m parquet `datetime.max()` 기준으로 커버리지를 재확인했다.

데이터 커버리지:
- historical 85 universe: 85/85 symbols, missing 0, direct 1m parquet 최신 timestamp `2026-06-28T10:09:00Z`.
- expanded 110 universe: 110/110 symbols, missing 0, direct 1m parquet 최신 timestamp `2026-06-28T10:09:00Z`.
- 월별 walk-forward runner는 30m 완료 bar 기준이라 최신 OOS 산출 끝점은 `2026-06-28T09:30:00Z`.

평가 방식:
- 기존 historical best artifact의 닫힌 OOS folds `2025-09`~`2026-05`는 그대로 유지했다.
- 최신 데이터 영향을 받는 `2026-06` fold만 repo runner로 재실행한 뒤, 기존 10-fold 전체 성적표에 삽입해 comp/ann/Sharpe/PF/MDD/hit을 재계산했다.
- 최신 fold train: `2025-01-01T00:00:00`~`2026-03-31T23:30:00`, validation: `2026-04-01T00:00:00`~`2026-05-31T23:30:00`, OOS: `2026-06-01T00:00:00`~`2026-06-28T09:30:00`.
- June selected leaf는 `relaxed_efficiency:aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna`로 유지했다.

최신 반영 결과:
- 최종 1등: `expanded_110_latest_tail_full` / `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`.
- 전체 OOS comp `+159.83%`, annualized approx `+214.51%`, monthly Sharpe `1.881`, PF `23.635`, max intra-fold OOS MDD `27.69%`, monthly equity MDD `5.09%`, hit `4/10`.
- risk-trimmed `fallback_mdd20_cap2`는 comp `+138.11%`, annualized approx `+183.23%`, Sharpe `1.783`, PF `21.454`, max intra-fold OOS MDD `23.58%`, hit `4/10`.
- 기존 85-symbol preregistered exact는 최신 June fold에서 return `+24.76%`, MDD `30.47%`가 되어 전체 comp `+125.13%`, Sharpe `2.506`, hit `5/10`로 내려갔다.

산출물:
- Summary: `var/reports/manual_reval_20260628_overall_latest_included_rerun/manual_overall_latest_included_summary_latest.md|json`.
- June 85 rerun: `var/reports/manual_reval_20260628_monthly_refit_june_latest_rerun85/june_latest_85_relaxed_efficiency_wf_latest.md|json`.
- June 110 rerun: `var/reports/manual_reval_20260628_monthly_refit_june_latest_rerun110/june_latest_110_relaxed_efficiency_wf_latest.md|json`.

판단:
- "전체 성적에 최신 데이터까지 반영" 기준의 raw return champion은 expanded 110 exact-unscaled로 바뀌었다.
- 단, 이전과 동일하게 backtest/shadow-paper evidence일 뿐이며 `ready_for_real=false` 판단은 유지한다. 실거래 승격에는 fresh-forward shadow, fill/slippage/funding/reconciliation telemetry가 필요하다.

## 2026-06-21 KST — H35 executable shadow/testnet adoption checkpoint

H35를 현재 공격형 운영 후보의 기준 포트폴리오로 채택했다. 기존 `H35_return_overlay_80_20`은 nested/report-only였으나 leaf-level executable manifest로 분해했고, live/backtest 공통 `manifest:` portfolio-mode resolver가 이 manifest를 읽어 `ArtifactPortfolioModeStrategy`로 실행할 수 있다. 기준 manifest는 `var/reports/current_top_models/executable_shadow_manifests/h35_return_overlay_80_20_leaf_manifest_latest.json`이다.

H35 구성:
- 56% `p2_latest_relaxed_aggressive_leaf` 계열 relaxed aggressive leaf.
- 20% `relaxed_efficiency:hybrid_v3_5` overlay.
- 24% cash.
- manifest gross cap `2.25`, child/source readiness, source sha/freshness, optimizer/correlation provenance, OOS contamination veto, real-money flag veto, malformed manifest cash fail-closed가 적용된다.

성적:
- 장기 OOS: `+73.49%`, monthly equity MDD `5.71%`, Sharpe `1.94`, 최신 fold `+7.01%`.
- fresh-forward window `2026-06-14T00:00:00` → `2026-06-21T11:01:00`: 10bps `+3.91%`, 15bps `+3.16%`, 20bps `+2.41%`, MDD `8.14%`, max gross `3.04x`.
- 비교 기준 P2: 10bps `+3.79%`, 20bps `+2.39%`, MDD `7.68%`, max gross `2.85x`.
- 장기 상위 raw aggressive proxy는 fresh `+5.24%`지만 MDD `10.85%`, gross `4.07x`라 core 채택은 보류하고 H35처럼 capped/cash-buffered 형태로만 운용한다.

운영 세팅:
- committed decision template: `configs/live/h35_shadow_testnet_decision.json`.
- live/testnet 실행 reference: `manifest:var/reports/current_top_models/executable_shadow_manifests/h35_return_overlay_80_20_leaf_manifest_latest.json`.
- paper/testnet/shadow 준비 상태는 `ready_for_shadow=true`; real-money는 preflight/fill/slippage/funding/reconciliation telemetry 전까지 계속 fail-closed다.
- 실전 테스트는 limit-first, market-order disabled, kill-switch/stop-file enabled, testnet/shadow parity and reconciliation collection을 전제로 한다. real endpoint canary/full은 별도 preflight가 통과하고 `LUMINA_ENABLE_LIVE_REAL=true`가 명시될 때만 허용된다.

검증:
- Manifest resolver smoke: H35/P2 `manifest:` resolution 통과.
- Fresh-forward performance report: `var/reports/current_top_models/fresh_forward_shadow_eval/fresh_forward_shadow_performance_latest.md`.
- Long-top3 fresh mapping report: `var/reports/current_top_models/fresh_forward_shadow_eval/long_top3_fresh_forward_mapping_latest.md`.
- Focused tests: `uv run pytest tests/test_alpha_zoo_69_asset_efficiency_live_adapter.py tests/unit/test_artifact_portfolio_mode.py` → `51 passed`.

## 2026-06-20 KST — DeepLearning forecast bridge + strategy-quality overlay broad reassessment

DeepLearning repo의 FITS/CycleNet/CMamba/PatchTST를 LuminaQuant에 직접 import하지 않고 **artifact-only forecast bridge**로 받는 경로를 만들었다. 학습은 실행하지 않았다. `DeepLearningForecastStore`는 CSV/JSON/JSONL/Parquet 예측 산출물을 읽고, `DeepLearningForecastGateStrategy`는 모델별 예측 return/dispersion/agreement/confidence를 합쳐 LONG/SHORT/EXIT 신호를 낸다. 별도 pipeline manifest/CLI는 export → optional HPO plan → train/predict command plan → artifact validation → strategy profile materialization → backtest/shadow gate 순서만 생성하며, 실제 학습 실행은 금지했다.

구현 범위:
- Forecast artifact bridge: `src/lumina_quant/data/deep_learning_forecasts.py`.
- Forecast strategy: `src/lumina_quant/strategies/deep_learning_forecast_gate.py`.
- Pipeline/CLI/config: `src/lumina_quant/workflows/deep_learning_pipeline.py`, `src/lumina_quant/cli/deep_learning.py`, `deep_learning` config schema/loader/example.
- 기존 전략 개선 overlay: `src/lumina_quant/portfolio/strategy_quality.py` 및 backtest portfolio 통합. Edge gate, regime router, vol sizing, turnover budget, ATR exit overlay, health cooldown, ProfitMoonshot conflict guard, pair-correlation guard를 공통 portfolio layer에 추가했다.

초기 BTC 단일/5전략 평가는 신뢰 가능한 최종평가가 아니라 smoke check였으므로 폐기 수준으로 낮춰 봐야 한다. 사용자 지적에 따라 즉시 pool/전략/기간을 넓혀 재측정했다.

Broad 평가:
- 전체 등록 전략 36개 × 3 pools(`BTC`, `ETH`, `BTC+ETH`) × `2024-01-01~2025-12-31`: raw backtest 216회, 비교 108개. 통과 97개, 실제 거래 비교 66개.
- 실제 거래 비교 기준 최종자산 개선 39 / 악화 24 / 변화 없음 3. MDD 개선 49 / 악화 14.
- 좋은 쪽: `RsiStrategy`, `RegimeBreakoutCandidateStrategy`, `ProfitMoonshotBreakoutStrategy`, `AdaptiveRegimeMomentumStrategy`, `ProfitMoonshotTrendStrategy`.
- 나쁜 쪽: `LagConvergenceStrategy`, `Alpha101FormulaStrategy`, `RollingBreakoutStrategy`, `CompressionBreakoutContinuationStrategy`, 일부 vol-compression reversion aliases.

Live-default 다기간 평가:
- live_default 9개 전략 × 3 pools × 3 periods(`2024`, `2025`, `2024~2025`): raw backtest 162회, 비교 81개. 통과 69개, 실제 거래 비교 60개.
- 실제 거래 비교 기준 최종자산 개선 27 / 악화 24. MDD 개선 37 / 악화 14.
- `RsiStrategy`: 8/9 최종자산 개선, 9/9 MDD 개선, 평균 delta equity `+415.58`.
- `PairTradingZScoreStrategy`: 3/3 개선.
- `VwapReversionStrategy`: 평균 `+76.12`지만 pool/기간 편차 큼.
- `MeanReversionStdStrategy`: 혼합/거의 중립.
- `MovingAverageCrossStrategy`: MDD는 대부분 개선하지만 최종자산 평균은 `-10.97`.
- `RollingBreakoutStrategy`: MDD는 9/9 개선하지만 평균 delta equity `-126.46`; 추세 수익을 잘라먹는다.
- `LagConvergenceStrategy`: 평균 delta equity `-809.91`, MDD도 3/3 악화. 현 overlay와 궁합이 나쁘다.
- `BitcoinBuyHoldStrategy`: 영향 없음.

운영 판단:
- 공통 overlay를 모든 전략에 일괄 적용하면 안 된다.
- 적용 후보: `RsiStrategy`, `PairTradingZScoreStrategy`, 일부 reversion 계열.
- 조건부/개별 튜닝: `VwapReversionStrategy`, `MeanReversionStdStrategy`, `MovingAverageCrossStrategy`.
- 제외 또는 family-specific 재설계: `LagConvergenceStrategy`, `Alpha101FormulaStrategy`, `RollingBreakoutStrategy`, `CompressionBreakoutContinuationStrategy`.
- 특히 trend/breakout 계열은 exit overlay와 regime/vol sizing이 upside를 너무 빨리 잘라서 별도 완화 파라미터가 필요하다.

신뢰도 한계:
- 이번 평가는 로컬 sample CSV(`BTCUSDT.csv`, `ETHUSDT.csv`) 기반이므로 실전 승격 근거가 아니다.
- 더 믿을 수 있는 판정에는 raw-first 실데이터, multi-asset parquet, 비용/슬리피지 재보정, walk-forward, fresh-forward shadow, live fill telemetry reconciliation이 필요하다.
- 현재 결론은 “전략군별 overlay 궁합을 가르는 연구 노트”로만 사용한다. real-money/paper 승격 근거 아님.

검증:
- Focused tests: `uv run pytest tests/test_strategy_quality_overlay.py tests/test_deep_learning_pipeline.py tests/test_deep_learning_forecast_gate_strategy.py tests/test_runtime_config_loader.py` → `27 passed`.
- Ruff targeted check 통과.
- Config example load 및 `lq deep-learning plan --json` 확인.
- Broad eval artifacts: `/tmp/lq_strategy_quality_all_strategies_full_pools.{json,csv}`, `/tmp/lq_strategy_quality_live_default_period_pools.{json,csv}`.

## 2026-06-19 KST — post-OOS 갱신 규칙 walk-forward 박제 + 주요 shadow 후보 신뢰도 재분류

사용자 정정에 따라 "post-OOS를 봤다면 같은 전략을 다음 달에 갱신하는 방식으로 walk-forward 평가할 수 있느냐"를 별도 규칙으로 분리해 코드/테스트/리포트로 박제했다. 기존 전역 monthly switching 평가는 이 질문의 답으로는 부정확하므로, 새 평가는 **같은 normalized strategy stem 내부에서만** 변형을 고르고 fold `t`의 OOS는 fold `t` 선택/가중치에 쓰지 않는다. fold `t` 선택은 완료된 과거 folds `<t`의 post-OOS Calmar만 사용하고, 히스토리가 없는 첫 fold만 train/validation Calmar로 bootstrap한다.

구현/테스트:
- `scripts/research/run_same_stem_post_oos_update_walkforward.py`: `hybrid_v3_5`, `hybrid_v3_6`, `fixed_relaxed_dynamic_blend` stem에 대해 same-stem lagged post-OOS update replay를 생성한다. 산출 row는 `post_oos_research_variant=true`, `requires_fresh_forward_shadow=true`, `clean_promotion_eligible=false`, `ready_for_real=false`, `real_money_execution=false`, `paper_order_execution=false`로 fail-closed된다.
- `tests/test_same_stem_post_oos_update_walkforward.py`: H3.5/H3.6/fixed-blend stem 정규화, "현재 fold OOS 승자"가 아니라 "이전 완료 OOS 승자"를 고르는지, promotion/real-money gate가 닫혀 있고 JSON/Markdown 산출물이 round-trip 되는지를 검증한다.

산출물:
- Same-stem WF replay: `var/reports/strategy_research/same_stem_post_oos_update_walkforward_20260619/same_stem_post_oos_update_walkforward_latest.md|json`.
- 선행 same-strategy reassessment: `var/reports/strategy_research/same_strategy_post_oos_update_reassessment_20260619/same_strategy_post_oos_update_reassessment_latest.md|json`.
- 상관 기반 포트폴리오 진단: `var/reports/strategy_research/corr_aware_hybrid_portfolio_20260619/corr_aware_hybrid_portfolio_latest.md|json`.

결과:
- `same_stem_post_oos_update:fixed_relaxed_dynamic_blend_lagged_top1_calmar`: comp `+98.66%`, monthly equity MDD `15.43%`, Sharpe `1.84`, hit `5/10`, MDD<=30 pass.
- `same_stem_post_oos_update:hybrid_v3_5_lagged_top1_calmar`: comp `+89.93%`, monthly equity MDD `15.66%`, Sharpe `1.65`, hit `5/10`, MDD<=30 pass. H3.5는 validation-only selector(`+19.62%`/`+25.87%`)보다 post-OOS 갱신 방식에서 크게 좋아진다.
- `same_stem_post_oos_update:hybrid_v3_6_lagged_top1_calmar`: comp `-6.18%`, monthly equity MDD `25.88%`, Sharpe `-0.10`, hit `3/10`. H3.6은 이 기준에서 주력 후보가 아니다.
- Corr-aware portfolio 진단상 실행 가능한 shadow blueprint는 `P2_corr_core`(70% risk-trimmed lagged router + 30% clean MDD20 dynamic switch): comp `+53.85%`, monthly equity MDD `3.43%`, Sharpe `1.96`. `H35_return_overlay_80_20`은 comp `+73.49%`지만 nested-hybrid/report-only라 leaf-decompose + fresh-forward 전에는 executable 후보가 아니다.

신뢰도 판단:
- **믿을 수 있는 범위**: 동일한 frozen rule을 재실행했을 때 재현되는 research/shadow ranking으로는 신뢰도가 올라갔다. 특히 same-stem WF는 current-fold OOS selection/weighting, same-month self-feeding, real-money/paper execution 카운트가 모두 `0`임을 리포트와 테스트가 확인한다.
- **믿을 수 없는 범위**: promotion/real-money 신뢰도는 아직 없다. 이 규칙 자체가 과거 locked/post-OOS를 본 뒤 만든 post-OOS research variant이므로 `clean_promotion_eligible=false`가 맞다. 실전 승격에는 leaf-level executable manifest 재구축, cost/fill stress, fresh-forward monthly shadow, reconciliation, MDD/liquidation gates가 별도로 필요하다.
- 운영 결론: 현 주력 shadow 후보는 (1) 안정형 `P2_corr_core`, (2) 연구용 return 후보 `fixed_relaxed_dynamic_blend` same-stem update, (3) 연구용 H3.5 same-stem update다. H3.6은 제외한다. 모두 real-money 제외다.

검증:
- `uv run ruff check scripts/research/run_same_stem_post_oos_update_walkforward.py tests/test_same_stem_post_oos_update_walkforward.py` → pass.
- `uv run pytest -q tests/test_same_stem_post_oos_update_walkforward.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_alpha_zoo_existing_strategy_reassessment.py` → `58 passed`.

## 2026-06-18 KST — Alpha/strategy improvement execution checkpoint: promotion gates, survivor manifests, artifact-portfolio fail-closed mode

Deep Interview/Ralplan 승인안(`.gjc/specs/deep-interview-alpha-strategy-improvement.md`, `.gjc/plans/ralplan/2026-06-07-0457-b89b/pending-approval.md`) 기반으로 Ultragoal 실행을 G004까지 완료했다. 핵심 제약은 그대로 유지했다: locked-OOS는 selection/objective/tie-break/correlation/sizing에 금지, weak-data TradFi는 shadow-only, MDD cap 30%, liquidation/wipeout 불허, shadow/clean-paper two-tier benchmark, real-money 제외.

완료된 변경:
- `run_alpha_zoo_clean_new_alpha_discovery.py`: candidate-level promotion gate/report schema 추가. 후보별 family/source/theory, data sufficiency, locked-OOS usage flags, freeze hash, benchmark gate, MDD/liquidation/cost rejection reasons, tried universe, promotion summary를 JSON/Markdown에 노출한다. row-level full-WF advancement는 이제 `selected_by_train_validation_freeze=True`가 필수이고, freeze selection/hash 입력은 `locked_oos_*` report fields를 제거한 train/validation view만 사용한다.
- survivor manifest workflow: `clean_new_alpha_survivor_manifest_latest.json` 별도 산출. manifest는 train/validation-frozen survivors만 포함하고 `locked_oos` 키를 scrub하며, report-only OOS metric 변경으로 hash가 바뀌지 않는다. unselected-but-eligible 후보와 holdout-contaminated 후보는 full-WF retest candidate가 될 수 없다.
- `write_alpha_zoo_existing_strategy_reassessment.py`: 기존 registry 전략을 smoke/audit용으로 열거하고, current top evidence는 context로만 붙인다. survivor/full-WF promotion list는 엄격 게이트 통과 전까지 빈 리스트, real-money false.
- `artifact_portfolio_mode.py` + `live_selection.py`: `artifact_manifest_mode` 및 `manifest:<path>` 지원. manifest source artifact는 regular file + sha256 + freshness + ready/portfolio_ready가 필수이고, child readiness/provenance/schema/correlation/gross/netting cap이 깨지면 100% cash로 fail-closed한다. malformed collections/children, non-file source, OOS contamination, source mismatch, gross cap breach 모두 live component를 만들지 않는다.

증거/산출물:
- Current top control evidence: `var/reports/current_top_models/current_top_models_20260618.md`, `var/reports/current_top_models/top_strategy_correlation_portfolio_20260618.md`. Shadow risk-trim benchmark는 OOS comp `64.42%` 또는 return/MDD `3.49`, clean/paper benchmark는 OOS comp `34.39%`.
- Existing strategy reassessment probe: `var/reports/strategy_research/existing_strategy_reassessment_g002_probe_20260618.md|json` — 3개 전략 audit, full-WF/real-money promotion 없음.
- Clean alpha bounded probe: `var/reports/strategy_research/g005_clean_alpha_probe_20260618/clean_new_alpha_discovery_latest.md|json` 및 `clean_new_alpha_survivor_manifest_latest.json` — 1 family/1 fold smoke에서 no-promotion/watchlist evidence, real-money false.

검증:
- `uv run pytest tests/test_alpha_zoo_clean_new_alpha_discovery.py tests/test_alpha_zoo_existing_strategy_reassessment.py tests/unit/test_artifact_portfolio_mode.py tests/unit/test_backtest_live_portfolio_mode_resolution.py tests/test_strategy_registry_defaults.py` → `88 passed`.
- G001~G004 Ultragoal checkpoint 완료. 다음 active story는 G005(`Run focused verification and produce execution evidence`)이며, 이미 focused tests/probe 일부는 수행됐지만 최종 checkpoint와 aggregate goal completion은 다음 세션에서 이어가야 한다.

운영 판단: 이번 변경은 신규 전략을 실전 승격한 것이 아니라 **승격/포트폴리오 구성 경로를 fail-closed로 만드는 안전장치**다. 실제 채용 후보는 여전히 paper/shadow only이며, fresh-forward monthly folds와 cost/fill telemetry가 쌓이기 전까지 real-money는 금지다.


## 2026-06-17 KST (o) — 과거 선택 전체 전략 최고성적 모델 재개선: lagged leaf router risk-trim

사용자 정정에 따라 TopCap이 아니라 `var/reports` 전체 historical selected/high-water 후보를 다시 audit했다. 최고 headline은 `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`였고, 기존 deep-research의 `clean_input_meta_selector`(+85.91%)보다 높다: 85-symbol pre-registered replay에서 OOS comp `+197.37%`, ann approx `+269.80%`, max OOS MDD `27.69%`, monthly MDD `4.50%`, Sharpe `2.12`, PF `30.04`, hit `5/10`. expanded 110 latest-tail artifact에서도 같은 frozen rule은 `+79.42%`, Sharpe `1.96`, max OOS MDD `27.69%`로 여전히 110-asset report rank 1이다.

개선은 headline 수익 뻥튀기가 아니라 위험효율 개선으로 잡았다. 기존 최고 모델의 큰 bar-MDD는 warmup/no-history 구간 strict-core fallback이 validation-MDD30/cap3까지 스케일되는 데서 나온다. 새 후보 `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2`를 추가했다. lagged leaf 선택식(완료된 prior OOS + validation score, current-fold OOS 미사용)은 그대로 두고, strict-core fallback만 validation-MDD20/cap2로 낮춘다. 따라서 warmup 후 leaf choice는 동일하고, 위험 컷은 2025-11 fallback 같은 구간에만 작동한다.

Report: `var/reports/best_historical_strategy_improvement/best_historical_strategy_improvement_latest.md|json`.

결과:
- Historical 85 replay baseline: comp `+197.37%`, max OOS MDD `27.69%`, return/MDD `7.13`.
- Historical 85 risk-trim: comp `+172.51%`, max OOS MDD `20.26%`, return/MDD `8.52`.
- Expanded 110 latest-tail baseline: comp `+79.42%`, max OOS MDD `27.69%`, return/MDD `2.87`, Sharpe `1.96`.
- Expanded 110 risk-trim: comp `+64.42%`, max OOS MDD `18.46%`, return/MDD `3.49`, Sharpe `2.11`.
- Frozen `clean_input_meta_selector`를 expanded 110 latest-tail에 그대로 적용하면 comp `+9.75%`, Sharpe `0.50`뿐이라, 개선 대상은 clean-input meta selector가 아니라 이 lagged leaf router가 맞다.

판단: raw return champion은 기존 exact-unscaled baseline이고, 채용 가능성이 더 나은 건 신규 `fallback_mdd20_cap2` risk-trim shadow variant다. 둘 다 여전히 `post_oos_research_variant` / `requires_fresh_forward_shadow` / `ready_for_real=false`다. 실전 승격은 금지이고, paper/shadow에서 같은 frozen rule로 1~2개월 이상 forward 확인 + 10/15/20bps cost/fill telemetry가 필요하다.

검증: lagged/preregistered focused tests `2 passed`; ruff check/format targeted 통과. Fast replay report는 existing fold rows 기반이며 risk-trim replay는 strict-core fallback aggregate metrics에 proportional scale approximation flag를 단다. Full WF path에는 exact return-stream scaling 후보도 추가되어 다음 full rerun에서는 exact metrics로 나온다.

운영 원칙 업데이트: clean OOS 후보도 동일하게 **expanded universe 유지**가 기본이다. 단기 성적이 흔들려도 universe를 축소해서 숫자를 예쁘게 만들지 않는다. 대신 110개 이상으로 계속 넓힌 뒤 selector/lagged-router/clean-OOS gate가 그 안에서 스스로 걸러내게 두고, 신규 자산은 충분한 train+validation history가 생기기 전까지 promotion evidence가 아니라 monitored feature support로만 다룬다.

## 2026-06-17 KST (n) — 기존 최고성적 TopCap 개선: target-pool selector 전처리

사용자 확인 요청에 따라 universe/selection 경로부터 재검증했다. `config.yaml`은 `trading.symbols`를 의도적으로 생략하고 `TradingConfig.symbols` 기본값이 `BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED`를 쓰므로 runtime/default research/candidate universe가 모두 **110개**(crypto 10 + tradfi/commodity/ETF/equity/premarket 100)로 확장되어 있다. 후보 3504개가 110개 전부를 커버했고, 로컬 Binance 1m parquet도 2026-05-15~06-13 창에서 110개 모두 200 bars 이상 로드됐다. selector 자체도 110개 입력에서 정상 동작해 pool 12개(`TON/CL/BZ/DOGE/AVAX/AMAT/SPCX/ADA/XAU/STXX/BTC/TRX`)를 골랐다.

수정 1 — selection-aware cross-sectional 후보가 실제로는 `ctx.crypto_symbols`(=metals 제외 106개)를 써서 XAU/XAG/XPT/XPD를 selector universe에서 빼고 있었다. 이를 `ctx.normalized_symbols`로 바꿔 **SelectionGatedMomentum/Reversion도 full 110개**를 스크리닝하게 했다(커밋 `3fdcfbda`). 검증: 생성 후보 6개 모두 `symbols=110`, `symbol_scope=expanded_research_universe`.

수정 2 — 기존 최고 성과 축인 **`TopCapTimeSeriesMomentumStrategy`** 자체를 개선했다. 기본 동작은 완전 보존(`selector_enabled=False`)하되, 옵션으로 target-pool selector를 먼저 돌린 뒤 그 pool 안에서 기존 TopCap momentum long/short를 고르게 했다. 새 파라미터: `selector_enabled`, `selector_pool_size`, `selector_history_window`, `selector_min_history_bars`, selector factor weights, `selector_vwap_sign`. 이를 위해 TopCap이 close만 저장하던 구조를 OHLCV deques로 확장하고 state roundtrip도 high/low/volume까지 복원하도록 했다. candidate slice에는 1h selector TopCap 3개를 추가: `selector_exec_tightstop_20`, `selector_exec_tightstop_12`, `selector_resid_btc_20`; selector variant는 **full 110개** 입력에서 TopCap을 실행한다.

**실측(`var/reports/topcap_selector_improvement/`, strict local Binance 1m parquet, CSV/synthetic fallback 없음, 2026-05-15~06-13).** 기존 expanded TopCap `topcap_tsmom_1h_exec_tightstop_16_4_0.015`(106 symbols)는 +5.26%, Sharpe 3.44, MDD 2.88%, 3109 fills. 개선 후보:
- `selector_exec_tightstop_12`: **+8.22%, Sharpe 5.13, MDD 2.82%, fills 1039** — return +2.96%p, Sharpe +1.69, MDD 소폭 개선, fill 수 67% 감소.
- `selector_exec_tightstop_20`: +5.70%, Sharpe 3.81, MDD 2.58%, fills 1322 — return/Sharpe/MDD 모두 개선.
- `selector_resid_btc_20`: +4.72%, Sharpe 4.74, MDD 1.78%, fills 1127 — return은 낮지만 방어형 Sharpe/MDD 개선 후보.

결론: 기존 최고성적 전략 개선의 1차 답은 **TopCap + full-universe target-pool selector**다. 채용 후보는 `selector_exec_tightstop_12`(수익 개선)와 `selector_resid_btc_20`(방어형). 기존 TopCap을 바로 대체하지 말고 paper/shadow에서 5~10거래일 forward 확인 후, `selector_exec_tightstop_12`를 primary replacement 후보로 본다.

검증: TopCap/strategy-factory/selection-aware focused tests 44개 통과, ruff check/format, `check_architecture.py`, hardcoded-params audit(768/new=0), `verify_docs.py`, compileall, full pytest 2137개 통과.

## 2026-06-17 KST (m) — Seasonal Micro Breakout Rider: 1s/1m 미시 확인 + 30m/1h 거래

새 전략 **`SeasonalMicroBreakoutRiderStrategy`** 추가(`micro_signal_alpha_sleeves.py`, `_ReturnRiderBase` 상속·per-symbol 단일자산·crypto-only·>=30m). 사용자 제약을 그대로 반영: **알파는 마지막 market-window의 1s tape/VWAP를 확인용으로 볼 수 있지만, 실제 signal/order 결정은 `decision_cadence_seconds=1800` 이상**. 후보 라이브러리는 30m/1h/4h만 발행하고 1d는 제외(일중 slot 구조 희석).

- **신호 구조.** 결정바 close-to-close 수익률을 UTC `slot_minutes` 버킷별 decay-weighted 평균/분산으로 **one-bar-deferred** 업데이트(현재 바는 자기 슬롯 통계에 미포함 → 룩어헤드 방지). 슬롯 t-like signal이 `slot_t_threshold`를 넘고, 직전 `breakout_window` Donchian high/low + ATR buffer 돌파와 `trend_lookback` ROC 방향이 정렬될 때만 진입. 마지막 1s window는 tick 부호 일치율(`tick_agree_frac`)과 micro VWAP edge를 fresh confirmation으로만 사용하며, 이벤트 백테스트의 `MARKET` fallback에서는 micro가 없으면 neutral로 처리한다(위조/합성 없음).
- **차별점.** 기존 `IntradaySeasonalMomentumRider`는 slot drift × trend 조건만 보고, `VolatilityBreakoutRider`는 rolling breakout만 본다. 신규는 **시간대 slot drift + trailing breakout + ROC trend + micro tape/VWAP 확인**이 동시에 맞아야 발화한다. `TakerFlow*` 류의 60s/flow fade와도 반대: 본 전략은 30m/1h decision bar에서 continuation ride.
- **후보/selector 반영.** `candidate_library.py`에 `seasonal_micro_breakout_rider_{30m,1h,4h}` 슬라이스 추가. `ctx.crypto_only_symbols`만 사용해 tradfi/equity perp 누수 없음. 생성 확인: 총 30개(30m 10, 1h 10, 4h 10).

**실측(엄격 로컬 Binance 1m parquet, CSV/synthetic fallback 없음, 2026-05-15~2026-06-13, `var/reports/seasonal_micro_breakout_eval/`).** 단독 상위: BNB 1h +4.86%, Sharpe 3.37, MDD 3.32~3.43%, 40 trades; TON 1h +4.00%, Sharpe 2.87, MDD 3.20~3.31%, 27 trades; BTC 1h +3.96%, Sharpe 2.95, MDD 3.78~3.93%, 35 trades; BTC 30m +2.55%, Sharpe 1.82; TON 30m +2.12%, Sharpe 1.24. 반대로 ETH/SOL/XRP/DOGE/ADA/AVAX는 대체로 음수이고 4h는 1개월 window에서 slot history 부족으로 0거래. 결론: **범용 승격은 불가**, 1h BNB/TON/BTC만 paper/shadow 후보.

**기존 TopCap에 추가 가능성.** 동일 창에서 incumbent `topcap_tsmom_1h_exec_tightstop_16_4_0.015`(BTC/ETH/SOL/BNB/TRX)는 +2.37%, Sharpe 3.09, MDD 1.62%. 1h sampled blend(`seasonal_micro_topcap_blend_check_latest.md`)는 BNB sleeve 5/10/20/30% overlay가 각각 +2.50/+2.62/+2.87/+3.12%, Sharpe 3.27/3.43/3.66/3.79, MDD 1.67/1.73/1.89/2.09%. TON/BTC overlay도 return과 Sharpe는 개선, MDD는 소폭 증가. 운영 판단: **TopCap 대체가 아니라 5~20% shadow overlay 후보**(30%는 MDD 상승이 명확).

**검증.** 신규/관련 테스트 `tests/test_micro_signal_alpha_sleeves.py`, `tests/test_strategy_factory_library.py` 43개 통과. 로컬 CI: ruff check/format, native maturin build, live-data/market-window/native-Binance architecture gates, `check_architecture.py`, hardcoded-params audit(768/new=0; baseline 갱신), `verify_docs.py`, compileall, full pytest 2136개, dashboard lint/test/typecheck/build, GPU contract smoke, benchmark smoke+8GB gate 모두 통과.

## 2026-06-17 KST (l) — CUSUM 변화점 트렌드 라이더 + 분산비(Variance-Ratio) 트렌드 라이더

레지스트리 99종 — 새 추정자 2종(`cusum_varratio_alpha_sleeves.py`, 둘 다 `_ReturnRiderBase` 상속·per-symbol 단일자산·>=30m). 사전 확인: **CUSUM/change-point·variance-ratio 전략·인디케이터 전무** → 둘 다 distinct. *(빌드 Workflow 529 불안정 → 세션 내 직접 작성 + 독립 critic 적대 리뷰.)*

- **CusumChangePointTrendRiderStrategy** (OHLCV-only). 이론: Page(1954) 누적합 변화점 검출. 수익을 롤링 변동성으로 표준화한 뒤 양측 CUSUM control chart: `S_hi=max(0,S_hi+z-k)`, `S_lo=min(0,S_lo+z+k)`. `S_hi>=cusum_h`면 상방 드리프트 레짐 전환 선언→LONG 진입+`S_hi` 리셋, `S_lo<=-cusum_h`면 하방→SHORT. **검출 방향 자체가 진입**, ATR 트레일링 라이드. **차별점**: KalmanTrend(연속 상태공간 slope)·AdaptiveTrend(KAMA)·MA/채널 트렌드와 달리 **이산 순차 변화점 검출기** — 타 슬리브 미사용.
- **VarianceRatioTrendRiderStrategy** (OHLCV-only). 이론: Lo-MacKinlay(1988) 분산비 랜덤워크 검정. `VR(k)=Var(k기간수익)/(k·Var(1기간수익))`은 랜덤워크에서 ~1, 양의 자기상관(지속/추세)에서 >1, 평균회귀에서 <1. `VR(k) >= 1+vr_threshold`(랜덤워크가 지속성 쪽으로 기각)이고 추세 확인 시에만 라이드. **차별점**: HurstRegimeGated(R/S Hurst)·PermutationEntropy(ordinal 엔트로피)·TrendEfficiency(Kaufman ER)와 달리 게이트가 **Lo-MacKinlay 분산비 통계량** — 타 슬리브 미계산.

유니버스 둘 다 `ctx.crypto_only_symbols`, 30m/1h/4h/1d, per-TF cadence `_RIDER_TF_CADENCE_SECONDS`(>=1800). 검증: `.venv/bin/python`(3.14) ruff·format·py_compile·audit(baseline 764/new=0)·check_architecture·verify_docs·신규 11테스트(VR 헬퍼 단위검증[지속>1·평균회귀<1]·CUSUM 상/하 검출·VR 진입/억제 대조 포함)·full pytest 통과 + 독립 critic 적대 리뷰(특히 CUSUM 시그널 래치/리셋·VR 통계량). 백테스트는 데이터 PC.

## 2026-06-17 KST (k) — 순열 엔트로피 레짐 트렌드 라이더 + Amihud 비유동성 프리미엄 모멘텀 라이더

레지스트리 97종 — 새 신호원 2종(`entropy_amihud_alpha_sleeves.py`, 둘 다 `_ReturnRiderBase` 상속·per-symbol 단일자산·>=30m). 사전 확인: **순열-엔트로피/ordinal-pattern 전략 전무**, `amihud_illiquidity` **인디케이터는 존재하나 사용 전략 전무**(재사용) → 둘 다 distinct. *(빌드 Workflow 529 불안정 → 세션 내 직접 작성 + 독립 critic 적대 리뷰.)*

- **PermutationEntropyTrendRiderStrategy** (OHLCV-only). 이론: Bandt-Pompe 순열 엔트로피(연속 `pe_dim`개 값의 ordinal-pattern 분포의 Shannon 엔트로피를 `log(dim!)`로 정규화 → [0,1], 0=구조적 1=랜덤). 정규화 엔트로피가 `pe_threshold` 미만(예측가능/구조적 레짐)이고 추세가 확인될 때만 라이드, 고엔트로피(랜덤)면 진입 억제. **차별점**: HurstRegimeGated(R/S Hurst)·TrendEfficiency(Kaufman ER)·AdaptiveTrend(KAMA)·KalmanTrend(상태공간 slope)과 달리 레짐 게이트가 **정보이론 ordinal-pattern 엔트로피** — 타 슬리브 미계산. 인과적(현재 종가까지로 계산).
- **AmihudIlliquidityMomentumRiderStrategy** (OHLCV-only). 이론: Amihud(2002) 비유동성 프리미엄(단위 달러거래량당 가격충격↑ = 비유동 → 모멘텀/수익 강함). 기존 `amihud_illiquidity` 프록시(`mean |logret|/dollar-vol`) 재사용, volume을 병렬 deque로 캡처. 현재 Amihud가 자기 **롤링-median 대비 elevated**(`>= illiquidity_rel*median`, 비유동 프리미엄 레짐)이고 추세 확인 시에만 라이드. **차별점**: LiquidityShock/OrderBookImbalance(미시구조 반전)·순수 트렌드라이더(유동성 비조건)와 달리 게이트가 **Amihud 비유동성 프리미엄 레짐**.

유니버스 둘 다 `ctx.crypto_only_symbols`, 30m/1h/4h/1d, per-TF cadence `_RIDER_TF_CADENCE_SECONDS`(>=1800). 검증: `.venv/bin/python`(3.14) ruff·format·py_compile·audit(baseline 760/new=0)·check_architecture·verify_docs·신규 10테스트(순열엔트로피 헬퍼 단위검증·entry/suppress 대조 포함)·full pytest 통과 + 독립 critic 적대 리뷰(특히 Amihud closes/volumes 정렬·룩어헤드). 백테스트는 데이터 PC.

## 2026-06-17 KST (j) — Kalman 상태공간 트렌드 라이더 + 실현 세미분산 비대칭 트렌드 라이더

레지스트리 95종 — 신규 신호원 2종(`kalman_semivar_alpha_sleeves.py`, 둘 다 `_ReturnRiderBase` 상속·per-symbol 단일자산·>=30m). 사전 확인: 코드베이스에 **Kalman 추정자 전무**, **실현 세미분산/good-bad-vol/signed-jump 전략 전무**(downside_volatility 헬퍼만 존재) → 둘 다 명확히 distinct. *(빌드 Workflow가 529로 불안정해 이번도 세션 내 직접 작성 + 독립 critic 적대 리뷰로 검증.)*

- **KalmanTrendRiderStrategy** (OHLCV-only). 이론: 상태공간/Kalman 트렌드 추정. **로그 종가**에 local-linear-trend Kalman 필터(상태 [level, slope], 2×2 공분산)를 재귀 적용 → 관측/프로세스 노이즈 비율로 반응성을 적응시키는 **저지연·불확실성-인지** slope 추정(스케일프리 per-bar 드리프트). 필터 slope가 통계적으로 유의(`slope/sqrt(slope_var) >= slope_t`)하고 경제적으로 유의(`|slope| >= min_slope_frac`)할 때 라이드. 현재 종가에서 posterior slope로 결정 = **인과적**(룩어헤드 아님). **차별점**: AdaptiveTrendRider(KAMA 효율비)·TrendEfficiency·DonchianAtrTrend·MovingAverageCross와 달리 트렌드 **추정자가 재귀 베이지안 상태공간 slope**(이동평균/채널 아님) — 레지스트리에 Kalman 없음.
- **RealizedSemivarianceTrendRiderStrategy** (OHLCV-only). 이론: Barndorff-Nielsen 실현 세미분산; Patton-Sheppard "Good/Bad Volatility"(상/하방 실현분산은 다른 정보를 담고 signed jump variation이 수익을 예측). 결정바 로그수익 윈도우에서 상방 `RS+ = Σr²|r>0`·하방 `RS- = Σr²|r<0` 누적, **부호 비대칭 `SJ=(RS+-RS-)/(RS++RS-)∈[-1,1]`**. `|SJ| >= semivar_threshold`(good vol가 bad vol를 확실히 압도/역) **그리고** 추세 정렬일 때만 컨티뉴에이션 라이드(knife-catch 아님). **차별점**: LotterySkewness/RareEvent(교차섹션 lottery/tail)·vol-managed(vol로 사이징만)와 달리 진입 트리거가 **상/하방 실현 세미분산 비대칭** — 타 슬리브 미사용.

유니버스 둘 다 `ctx.crypto_only_symbols`. Kalman 30m/1h/4h/1d, semivar 30m/1h/4h, per-TF cadence `_RIDER_TF_CADENCE_SECONDS`(>=1800). 검증: `.venv/bin/python`(3.14) ruff·format·py_compile·audit(baseline 755/new=0)·check_architecture·신규 11테스트·full pytest 통과 + 독립 critic 적대 리뷰(특히 Kalman 공분산 재귀·룩어헤드 검증). 백테스트는 데이터 PC.

## 2026-06-17 KST (i) — intraday 시간대 시즈널-모멘텀 라이더 + overnight 세션 리턴 틸트 라이더

레지스트리가 93종으로 포화 — 리드랙(6슬리브)·트렌드·돌파·반전·carry·vol-레짐·day-level 시즈널은 이미 커버. **미커버 niche인 sub-day 시간구조** 2종(`intraday_overnight_alpha_sleeves.py`, 둘 다 `_ReturnRiderBase` 상속·per-symbol 단일자산·>=30m). *(주: 이번 배치는 build Workflow가 API 529로 2회 죽어 세션 내 직접 작성→독립 critic 적대 리뷰로 검증.)*

- **IntradaySeasonalMomentumRiderStrategy** (OHLCV-only). 이론: 일중 주기성/시간대 드리프트(Aleti-Bollerslev intraday periodicity; crypto hour-of-day 효과). 각 결정바를 UTC 시간대 **슬롯**(`_event_datetime_utc`, slot_minutes 폭)으로 버킷팅하고, 슬롯별 **decay-weighted (mean,var,n)** 을 **one-bar-deferred**로 갱신(결정 중인 바는 자기 슬롯 통계에 미포함 → 룩어헤드 없음). 슬롯이 (a)`min_slot_observations` 이상 (b)유사 t-통계(`|mean|/std*sqrt(eff_n)`, eff_n은 `1/decay`로 캡) `>=slot_t_threshold` (c)추세 부호와 **정렬**일 때만 그 방향 라이드, 아니면 진입 억제. **차별점**: CalendarSeasonalityOverlay는 day-of-week/month 정적 틸트(sub-day 해상도·온라인 드리프트·추세 조건화 없음); 순수 트렌드라이더와 달리 **불리/이력부족 슬롯이 진입을 억제**(동일 상승추세에서 plain 라이더는 진입, 본 슬리브는 차단 — 테스트로 증명).
- **OvernightSessionReturnRiderStrategy** (OHLCV-only). 이론: overnight-vs-active 세션 리턴 아노말리(Cooper-Cliff-Gulen overnight returns; Lou-Polk-Skouras "A Tug of War"). UTC 일을 "overnight" 창(`[start,end)` UTC시, 자정 wrap 처리)과 "active"로 분할, 세션별 decay-weighted 평균 리턴을 동일 deferred 방식으로 추적. 현재 세션의 과거 평균이 **유의 양(+)**이면 구조적으로 LONG 틸트(유의 음이면 `allow_short` 시 SHORT), 그 외 진입 억제 — **추세 비조건** 세션-리턴 하베스터(intraday 슬리브와의 분리점). 청산은 상속된 ATR 트레일링/`max_hold`로 관리(force-flat 훅 없음 → 음의 세션은 진입 억제로 표현).

유니버스 둘 다 `ctx.crypto_only_symbols`, 30m/1h/4h(1d는 sub-day 구조 없어 제외). per-TF cadence `_RIDER_TF_CADENCE_SECONDS`(>=1800). 독립 critic 적대 리뷰 verdict **clean** — 두 슬리브 모두 진짜 비중복(KEEP, 상호 간에도 구분: 추세-조건 슬롯 라이드 vs 무조건 세션 틸트), **룩어헤드 경험적 반증**(97결정 0불일치), suppression 실재. minor 2건: ①절대 유의 임계값 약함(t=1.0≈1σ) → **선제 반영해 기본값 1.0→1.5 상향**(과적합 저항↑), ②`_symbol_for` O(N)은 N=1 단일자산이라 무해(보류). 검증: `.venv/bin/python`(3.14) ruff·format·py_compile·audit(baseline 749/new=0)·check_architecture·신규 11테스트·full pytest 통과. 백테스트는 데이터 PC.

## 2026-06-17 KST (h) — 세션 오프닝-레인지 돌파 라이더 + 오픈인터레스트 확정 추세 라이더

고확신·비중복 2종(`orb_oi_trend_alpha_sleeves.py`, 둘 다 `_ReturnRiderBase` 상속·per-symbol 단일자산·방향성·>=30m).

- **OpeningRangeBreakoutRiderStrategy** (OHLCV-only, crypto_only). 이론: 세션 오프닝-레인지 돌파(Toby Crabel ORB; Zarattini-Aziz 2023 ORB 연구 — 강한 위험조정 수익). 메커니즘: 바 타임스탬프를 **UTC 캘린더 일자로 버킷팅**(`_session_key`/`_event_datetime_utc` 재사용), 각 새 UTC 일의 첫 `opening_range_bars` 결정바 동안 오프닝-레인지 high/low를 누적(무거래), 확립 후 **one-shot** 돌파 무장 — close가 레인지 상단(+ATR 버퍼) 돌파 시 LONG / 하단 돌파 시 SHORT, 세션마다 1회(`_session_fired` 래치). 오프닝-레인지는 매 UTC 일 **리셋**. ATR 트레일링+피라미딩 라이드. **차별점**: VolatilityBreakoutRider는 롤링 N바 Donchian 채널(세션 앵커·리셋 없음), VolatilitySqueezeBreakoutRider는 변동성 수축-전제 필요 — ORB는 **세션 앵커 오프닝-레인지(일일 리셋, 그날 첫 k바 극값)**가 트리거. 1d는 의도적 제외(일봉=UTC일당 단일바라 일중 앵커 무의미). 룩어헤드 없음(돌파바 high/low는 누적 종료 후라 레인지에서 제외).
- **OpenInterestTrendConfirmationRiderStrategy** (crypto-perp only, `ctx.perp_support_data_available` 게이트). 이론: 선물 미시구조 OI-가격 관계 — 상승추세가 **OI 증가**로 받쳐지면 신규 자금 유입(건강·지속), OI 감소 상 상승은 숏커버링(약함, 페이드). **엔트리 = 추세 부호 ∧ OI가 `oi_lookback`에 걸쳐 상승**일 때만(롱: 추세↑∧OI↑ = 신규 롱; `allow_short`면 추세↓∧OI↑ = 신규 숏); OI 평탄/하락이면 진입 억제. `open_interest` 피처를 FundingHarvestCarry와 동일하게 None-safe로 읽고, OI 없으면 graceful no-entry. ATR 트레일링 라이드. **차별점**: CarryTrendConfluence/FundingHarvest는 **funding** 부호로 게이팅·거래, 순수 트렌드라이더는 포지셔닝 데이터 미사용 — 신규는 **OI 변화(OI/가격 사분면)**가 추세 확정 게이트(다른 피처·다른 게이트 의미). 검증: 평탄 OI 깨끗한 상승추세 = 진입 0(순수 트렌드라이더면 발화했을 것).

유니버스 둘 다 `ctx.crypto_only_symbols`(tradfi 누수 없음). ORB 30m/1h/4h, OI 30m/1h/4h/1d, per-TF cadence `_RIDER_TF_CADENCE_SECONDS`(>=1800). 적대적 리뷰 verdict **clean** — 두 슬리브 모두 진짜 비중복(KEEP; 기존 비-라이더 `OpeningRangeContinuationStrategy`[60s cadence·고정 TP/stop]와도 구분). minor 2건은 선택적 향후 리팩토링(perp-feature 공유 mixin DRY·`_symbol_for` O(n)) — PR 범위 밖, 보류. 검증: `.venv/bin/python`(3.14) ruff·format·py_compile·audit(baseline 740/new=0)·check_architecture·verify_docs·compileall·신규 14테스트·full pytest 통과. 백테스트는 데이터 PC.

## 2026-06-17 KST (g) — carry×trend 컨플루언스 라이더 + 변동성 스퀴즈 브레이크아웃 라이더

고확신·비중복 2종(`carry_trend_squeeze_alpha_sleeves.py`, 둘 다 `_ReturnRiderBase` 상속·per-symbol 단일자산·방향성·>=30m).

- **CarryTrendConfluenceRiderStrategy** (crypto-perp only). 이론: Koijen-Moskowitz-Pedersen-Vrij "Carry" + Asness-Moskowitz-Pedersen "Value and Momentum Everywhere" — carry와 momentum은 상보적 수익원. **엔트리 = 추세 부호 ∧ carry 부호 합치(both-agree)**: 추세 상승 ∧ 평균 funding <= `long_carry_funding`(롱 perp이 carry를 받음/적어도 안 깎임)일 때만 LONG; `allow_short`면 추세 하락 ∧ funding >= `short_carry_funding`일 때 SHORT. ATR 트레일링+피라미딩으로 라이드. **차별점**: FundingHarvestCarry는 funding 부호만으로 진입(추세를 거스를 수 있고 ROC는 약한 veto뿐)·|funding| 사이징; 순수 트렌드라이더는 funding 비용 무시(크라우디드 롱에서 음의 carry로 출혈). 신규는 추세 라이드를 **carry 순풍으로 필터**하는 both-agree 게이트가 novelty. funding 데이터 없으면 graceful no-entry.
- **VolatilitySqueezeBreakoutRiderStrategy** (OHLCV-only, crypto_only). 이론: 변동성 클러스터링·수축→확장 사이클(Bollinger squeeze; John Carter TTM squeeze). **저변동성 스퀴즈(Bollinger bandwidth 다중바 퍼센타일 최저)를 필수 전제**로 latch한 뒤, 확장(bandwidth >= `expansion_mult`×스퀴즈 baseline) + 직전 N바 Donchian 레인지 돌파 시 **돌파 방향으로 진입**·라이드. 선택적으로 `require_bb_in_kc`(BB가 Keltner 채널 안 = 정통 TTM 스퀴즈). **차별점**: VWAPCompressionReversion은 같은 수축 레짐을 **반전(페이드)**으로 — 정반대 폴라리티; VolatilityBreakoutRider는 Donchian 채널 돌파지만 수축-전제 없음(아무 변동성 레짐에서나 발화). 신규는 **수축-레짐 전제**가 돌파 컨티뉴에이션을 게이팅하는 게 edge.

유니버스는 둘 다 `ctx.crypto_only_symbols`(tradfi perp 누수 없음; carry 빌더는 추가로 `ctx.perp_support_data_available` 게이트). 30m/1h/4h/1d, per-TF cadence `_RIDER_TF_CADENCE_SECONDS`(>=1800). 적대적 리뷰: **두 슬리브 모두 진짜 비중복(KEEP)**. major 1건 수정 — 스퀴즈의 Keltner 게이트(`_bb_inside_kc`)가 latch 경로에 미배선이라 `require_bb_in_kc`/`keltner_*` 4개 파라미터가 무효였음 → `_refresh_squeeze` latch에 Keltner 조건 배선 + dead `_is_squeeze` 제거 + 게이트 동작변경 회귀 테스트 추가. 검증: `.venv/bin/python`(3.14) ruff·format·py_compile·audit(baseline 736/new=0)·check_architecture·verify_docs·compileall·신규 15테스트·full pytest 통과. 백테스트는 데이터 PC.

## 2026-06-17 KST (f) — 실현변동성 term-structure 쇼크-라이드 + 교차섹션 breadth 레짐 타이머

고확신·비중복 2종(`vol_term_breadth_alpha_sleeves.py`). 둘 다 방향성·리턴지향, >=30m.

- **RealizedVolTermStructureStrategy** (per-symbol 단일자산, `_ReturnRiderBase` 상속). 엔트리 트리거가 **실현변동성 term-structure 비율 `RV_s/RV_l`** 자체 — 단기창 close-to-close RV_s와 장기창 RV_l. 비율 >= `shock_ratio`(변동성 쇼크/백워데이션)면 V-회복을 **롱 라이드**(또는 `fade_upside_short`+`allow_short`이면 상단 블로우오프를 숏 페이드), ATR 트레일링 스톱+피라미딩으로 라이딩, 변동성 정상화(`ratio<=exit_ratio`)·트레일·max_hold에서 청산. 단기 레그는 마지막 윈도우의 Parkinson 단일바 추정 `log(high/low)/(2*sqrt(ln2))`을 max()로 블렌딩해 micro 정보를 장기 레그와 **동일 per-bar 단위**로 정렬. **차별점**: PanicRebound(가격 드로다운+거래량 z-쇼크+VWAP 리클레임), HourlyShock(완성바 가격쇼크 페이드), vol-managed(기존 모멘텀을 사이징만)와 달리 vol-비율 자체가 진입 신호.
- **BreadthRegimeTrendTimerStrategy** (multi-symbol basket, cross_sectional). 톱다운 **breadth**(바스켓 중 자기추세 위 비중)가 **총 net-long 그로스**를 게이팅·스케일: `gross = target_allocation*max_gross*breadth`, risk_off/on breadth 사이 히스테리시스. risk-on은 상위 업트렌더 롱 바스켓을 breadth-스케일로, risk-off는 flat. **차별점**: CrossAssetDiversifiedTrend(per-symbol inverse-vol 리스크패리티+포트폴리오 vol-target, 총노출 breadth 게이트 없음), DualMomentum(단일상품 절대/상대 로테이션), TopCap TSMOM(breadth 게이트 없는 모멘텀 랭킹)와 달리 **교차섹션 breadth가 net long/flat 레짐 스위치 + 그로스 스칼라를 구동**.

유니버스는 둘 다 `ctx.crypto_only_symbols`(주식/ETF 누수 없음). RV-term 30m/1h/4h/1d per-symbol, breadth 1h/4h/1d basket(>=min_symbols 가드), per-TF cadence는 `_RIDER_TF_CADENCE_SECONDS`(>=1800). 검증: `.venv/bin/python`(3.14) ruff·format·py_compile·audit(baseline 731/new=0)·check_architecture·신규 8테스트·verify_docs·compileall·full pytest 통과. 적대적 리뷰 verdict **clean**(minor 2건 — 중립밴드 docstring 과대표현·미사용 `window` 인자 — 둘 다 반영 수정). 백테스트는 데이터 PC.

## 2026-06-17 KST (e) — micro-signal / >=30m-cadence 배치 (intrabar 시그널, 30m 거래)

사용자 지시: "알파 시그널은 micro 타임프레임을 봐도 되지만 거래빈도는 >=30m." 메커니즘(엔진 매핑): **Pattern A** — `decision_cadence_seconds=1800` + `preferred_contract="market_window"`. 엔진은 1s 데이터를 매 윈도우 ingest하고 스톱/트레일링은 1s 해상도로 평가하되(core/engine.py:448-472), 결정 콜백은 30m 버킷마다 1회만 발화(engine.py:339-363). **트랩**: 게이트된 콜백은 마지막 윈도우의 bars_1s만 봄 → 코어 시그널은 **누적 30m 결정-바**로 계산하고, 마지막 윈도우 bars_1s는 결정 순간의 **fresh micro 리딩/확인**으로만 사용.

신규 3종(`micro_signal_alpha_sleeves.py`, 전부 decision_cadence_seconds=1800):
- **IntradayFlowPressureRider** — 30m 결정-바 누적 `order_flow_imbalance`(taker quote vol, OHLCV up/down-vol 폴백) z-score로 방향성 진입, 마지막 윈도우 1s 틱 부호 일치 + 윈도우 VWAP 체결로 micro 확인, ATR 트레일링 라이드. TakerFlow*(60s fade)와 반대(1800s continuation).
- **VolOfVolRegimeTrendGate** — 추세 방향(pv_trend_score 게이트) × **GK vs close-to-close 변동성 발산 사이즈 거버너**. 주의: GK는 구조적으로 close-to-close의 ~2~2.5배라 절대 임계값이면 정상 바에서도 veto → 무거래(리뷰가 잡음). 그래서 **자기 자신의 롤링-median 대비 상대값**으로 판정(상대 스트레스), 워밍업엔 neutral. KER 게이트로 깨끗한 추세만 up-size. OHLCV-only.
- **VWAPCompressionReversion** — `volcomp_vwap_pressure`(처음 live 활성화) 게이트로 vol-압축 레짐에서만 VWAP 이격 z-score 반전, 마지막 윈도우 1s VWAP로 이격 anchor 정밀화, +-6 zero-sigma 가드. 비상관 diversifier.

**dead-primitive 5종 처음/재활성화**: pv_trend_score, garman_klass_volatility, kaufman_efficiency_ratio(라이브 게이트), volcomp_vwap_pressure, order_flow_imbalance. 적대적 리뷰가 MAJOR(GK/rv 거버너 무거래 오보정)를 잡아 상대-median으로 교정 + 디폴트 파라미터로 LONG 진입하는 no-trade 회귀 테스트 추가; minor 2건(symbol 값-동등 매칭→명시 전달, #1 required_features 제거로 OHLCV 폴백 도달)도 수정. 검증: ruff·audit(729/new=0)·architecture·신규 10테스트 통과. 백테스트는 데이터 PC.

## 2026-06-17 KST (c) — 상품/매크로 추세 라이더 + 주식 52주 신고가 라이더(라이더 재사용·신규 유니버스)

검증된 리턴-라이더 클래스를 **신규 방향성 유니버스로 라우팅**(신규 전략 클래스 없음): (1) **상품 managed-futures 추세 라이더** — `_COMMODITY_TREND_UNIVERSE`(=BINANCE_TRADFI_COMMODITY_SYMBOLS: CL/BZ/COPPER/NATGAS + XAU/XAG/XPT/XPD) 8종에 AdaptiveTrend/Breakout/Acceleration 라이더를 per-symbol·**롱숏**(상품은 양방향 추세)·4h+1d로; (2) **주식 52주 신고가 브레이크아웃 라이더**(George-Hwang) — `VolatilityBreakoutRiderStrategy`를 `_EQUITY_FACTOR_UNIVERSE`에 donchian_window=252(=52주 일봉)·**롱온리**로. 전부 `_intersect_universe(상수, normalized_symbols)`로 라우팅(crypto_symbols 미사용), per-TF cadence, ATR 트레일링 라이딩. de-leak로 crypto 라이더는 crypto-only라 상품/주식과 중복 없음. 검증: ruff·audit(711/new=0, 슬라이스는 데이터 dict)·architecture·신규 7테스트 + 기존 라이더 19통과, 적대적 리뷰 clean(잔여 minor: 4h 52w는 빠른 확인 변형·상품 가속 게이트 = 스키마 기본값, 둘 다 no-change). 백테스트는 데이터 PC(상품/주식 장기 일봉 필요).

## 2026-06-17 KST (b) — tradfi/equity 고수익 방향성 배치(리서치 기반) + crypto 누수 de-leak

**문헌 리서치(웹).** 고수익·이론근거 directional 주식/ETF 전략을 정리: TSMOM on equity-index ETFs(Moskowitz-Ooi-Pedersen; 12개월 모멘텀, 약세 시 현금, Sharpe~1.3), **Dual Momentum GEM**(Antonacci; 절대+상대 모멘텀, 약세 시 방어자산 로테이션, 17.4% CAGR / maxDD 22.7% vs buy-hold 60%), **레버리지-ETF 추세타이밍**(TQQQ/SOXL+200일 SMA: 26.7% CAGR vs 10.9%, 디케이는 추세필터로 회피), 섹터 상대강도 로테이션, 52주 신고가 모멘텀(George-Hwang), residual momentum(Blitz), **vol-managed momentum**(Barroso-Santa-Clara; inverse-vol 스케일로 Sharpe 0.53→0.97). 핵심 통찰: 교차섹션 마켓뉴트럴은 절대수익이 낮음 → 고수익엔 **방향성(롱바이어스)·추세 라이딩·컴파운딩**이 맞다.

**유니버스 매핑에서 발견한 버그.** candidate context의 `ctx.crypto_symbols`는 "`_METALS`가 아닌 모든 심볼"이라 **equity/ETF perp이 crypto 라이더 슬리브로 누수**됐다(crypto 라이더가 주식을 잘못 거래). 주식 슬리브는 반드시 `_intersect_universe(research_universe 상수, ctx.normalized_symbols)`로 타겟해야 한다.

**구현(이번, main에 반영 예정).** 디자인 랭크 백로그 중 S-EQ1/EQ2/EQ3 + 누수 수정:
- **S-EQ1 단일종목 주식 추세 라이더** — `AdaptiveTrendRiderStrategy`를 그대로 재사용, `_EQUITY_FACTOR_UNIVERSE`(76 단일종목)에 per-symbol 단일자산, 롱온리(allow_short=False), ATR 트레일링으로 NVDA/TSLA형 추세를 끝까지 + 피라미딩, 1d(+4h). 코드 재사용이라 구현 리스크 최소.
- **S-EQ2 `LeveragedTrendTimingRiderStrategy`(신규)** — SOXL/URNM, `close>SMA200` AND 골든크로스(SMA50>SMA200)+confirm 버퍼에서만 롱; `_vol_scaled_allocation`에 **디케이 페널티**(실현변동성↑→사이징↓); 추세필터 이탈 시 flat. LETF 디케이는 추세 밖에서 발생하므로 필터가 곧 엣지.
- **S-EQ3 `DualMomentumDefensiveRotationStrategy`(신규)** — 절대 모멘텀 게이트(약세면 방어), 강세면 ETF 상대강도 top-N(>200SMA), 약세면 **방어자산(XAU/XLE/UVXY 중 모멘텀 최강)으로 100% 롱 로테이션**(S4는 flat만 함 → 이게 차별점). 1d.
- **de-leak**: `ctx.crypto_only_symbols`(= crypto_symbols − tradfi perp) 추가, 기존 crypto 라이더 3종을 crypto-only로 전환 → 주식 누수 제거 + S-EQ1과 중복 방지. crypto-only 입력 유니버스 테스트엔 무영향(no-op)이라 골든 깨짐 없음.

**검증.** `.venv/bin/python`(3.14): ruff·hardcoded-params audit(baseline 711/new=0)·check_architecture·신규 13테스트 + 관련 531테스트 통과. 적대적 리뷰 verdict clean(잔여 minor는 의도된 설계: 약세에서 방어자산도 음수면 flat, 200SMA 워밍업 ≥252d 필요). S-EQ4(상대강도 로테이션, S-EQ3와 중복)·S-EQ5(vol-managed overlay, 레버리지 경로 리스크)는 보류.

**데이터/평가 주의.** 전부 OHLCV·≥30m(1d 주). 주식 perp 장기 일봉(≥252d)+200SMA 워밍업이 데이터 PC에 있어야 1d 트랜치가 발화. LETF는 실제 3x perp 시계열로 백테스트(3×기초자산 합성 금지). survivorship: 유니버스는 2026-06-13 스냅샷(megacap/AI 편중)이라 모멘텀 백테스트는 다소 낙관적 → 롤링/OOS 평가 권장.

## 2026-06-17 KST — 성적 피드백 반영: return-우선 피벗(>=30m), 리턴-라이더 2배치 + CI 정상화

**성적 평가(`var/reports/new_strategy_eval_20260616/REPORT.md`) 해석.** crypto5(BTC/ETH/BNB/SOL/TRX), 2026-05-16~06-13, 1h/4h 직접 백테스트:
- 신규 단일 알파 6종 중 **`TrendEfficiencyMomentum`(1h)만 양수**: +0.1495%, Sharpe 11.9, 55 trades, TopCap과 **per-step 상관 0.25**, 5~10% 블렌딩 시 포트 MDD 1.60%→1.44%. = 비상관 DD-감소 diversifier(수익 enhancer는 아님).
- 교차섹션/selection 6종은 음수/무거래. 4h·1d·긴 윈도우 후보는 1개월 창에서 대거 **무거래**. `IdiosyncraticVolatility`/`LotterySkewness`는 0거래.
- 인컴번트 `topcap_tsmom_1h_exec_tightstop_tp_16_4_0.015` = +2.40%, Sharpe 24.8, 157 trades — 절대수익 기준 여전히 최강.
- 핵심 제약: (1) **장기·다심볼 평가 불가** — full research runner가 106심볼/기본 리소스 로딩에서 900s, 상위10 크립토에서도 1200s 타임아웃. (2) tradfi/equity perp는 로컬 parquet 장기 커버리지 부족(>=252d 0개). 따라서 교차섹션/주식 알파의 진짜 성과는 데이터 PC에서만 채점 가능.

**사용자 지시.** "return·compound return이 너무 낮다 + >=30m 전략만 사용." → 설계 목표를 **DD-감소에서 수익 극대화로 전환**.

**조치 1 — 유니버스 기본값 동적 확장(main `e1b34c2`).** `config.trading.symbols`와 리서치/후보 기본 유니버스가 단일 상수 `BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED`(110: 크립토+금속+tradfi perp)에서 파생되도록 변경(`configuration/schema.py` `TradingConfig.symbols` 기본값 + `runtime_settings._DEFAULT_SYMBOL_FALLBACK`; `config.yaml`는 `trading.symbols` 생략→상속). 상수 하나만 키우면 라이브·리서치가 자동 확장.

**조치 2 — target-pool selector(main `12451fd`).** `strategy_factory/universe_selection.py`: 가격위치(Donchian)·거래량(달러볼륨 랭크+서지)·변동성(ATR% inverted-U)·VWAP 다팩터 복합점수로 유니버스를 target pool로 스크리닝(opt-in, 기본 off). `selection_aware_alpha_sleeves.py`로 selector를 알파에 조합.

**조치 3 — 리턴-라이더 2배치(>=30m, 수익 라이딩).** 전부 per-symbol 단일자산, ATR 트레일링으로 추세 끝까지 타기(고정 TP 없음)+피라미딩+vol-타게팅, TopCap 스케일 사이징(target_allocation 0.30, max_order_value 5000), TF별 cadence(30m/1h/4h/1d→1800/3600/14400/86400):
- 배치 A(`ea71cdb`): `AdaptiveTrendRider`(KAMA/효율), `VolatilityBreakoutRider`(Donchian/ATR확장), `AccelerationRider`(가속 모멘텀).
- 배치 B(이번): `MultiTimeframeTrendEnsemble`(단/중/장 호라이즌 정렬 시에만 진입, conviction 사이징), `PullbackTrendContinuation`(추세 내 눌림목 진입=더 좋은 엔트리), `FundingHarvestCarry`(funding 부호 지속성 carry 수확, perp 데이터 게이트). FundingDislocationTrendCarry(디슬로케이션)·PerpCrowdingCarry(크라우딩 페이드)와 메커니즘 구분.

**조치 4 — CI 정상화(main `a972e55`).** 팀의 `07539f3 "Demote weak validation Alpha Zoo exposure before paper trials"`로 페이퍼 readiness가 의도적으로 False로 강등됐는데 `test_alpha_zoo_7x_paper_forward_preflight`/`test_alpha_zoo_validation_first_discovery`가 여전히 `ready_for_paper is True`를 단언해 main ci가 red였음. readiness 데이터/게이트는 그대로 두고 **stale 단언만 현실(False)에 정렬**. 재승급은 train+validation 리튠이 locked-OOS 게이트를 통과할 때.

**검증.** 모든 배치는 `.venv/bin/python`(3.14) 기준 ruff check/format, hardcoded-params audit(baseline 갱신), check_architecture, verify_docs, compileall, 전체 pytest 통과 + 적대적 리뷰. 단 **수익 실측은 데이터 PC 백테스트 필요** — 라이더는 "수익을 끝까지 타도록" 설계됐을 뿐, crypto5에서 TopCap(+2.4%)을 능가하는지는 코드만으로 보장 불가. 다음: (1) 리턴-라이더 백테스트로 엑싯/사이징 튜닝, (2) 110-유니버스 리소스 타임아웃 해소로 교차섹션/tradfi 알파 채점 경로 개방.

## 2026-06-15 KST — latest merge + 신규 decorrelated sleeves 실측: 신규 단일 알파는 아직 탈락

최신 `private/main`(`aa0582b3`)을 `private-main`에 병합했고, 환경은 `uv sync --group dev --extra dev --extra gpu` 및 native backend rebuild까지 다시 맞췄다. 현재 검증 환경은 Python `3.14.5`, `pytest 9.0.2`, `ruff 0.15.1`, `cudf-polars-cu12 26.6.0`, `nvidia-nvjitlink-cu12 12.9.86`, `maturin 1.13.3`이고 `_compute` native kernels 및 numba/pyo3 metric backend load를 확인했다.

수정/검증 내용:
- `StrategyFactory` lazy plugin discovery 후 새 전략 이름은 보이지만 param schema/default가 비는 registry bug를 수정했다. `get_strategy_map()` discovery 결과를 `ParamRegistry`에 동기화하도록 `src/lumina_quant/strategies/registry.py`를 고쳤다.
- 신규 test `tests/test_new_decorrelated_alpha_sleeves.py`는 rolling primitive, lazy-discovered strategy schema/default, candidate-library wiring/admission tags를 검증한다.
- `tests/test_strategy_factory_library.py`의 carry tag assertion은 obsolete rule을 제거하고 `family == "carry"` 행만 data-dependent carry로 강제하도록 정정했다.

Asset 확장 상태:
- Candidate build는 broadened universe 기준 총 `729` rows, 신규 sleeve rows `59`개까지 나온다: `ConfidenceGatedTrendStrategy 40`, `HurstRegimeGatedStrategy 3`, `MetalsRelativeValueBasketStrategy 3`, `LiquidationCascadeReversionStrategy 3`, `OrderBookImbalanceReversionStrategy 1`, `CrossSectionalEquityMomentumStrategy 2`, `ResidualEquityMomentumStrategy 1`, `BettingAgainstBetaStrategy 1`, `SemisLeadLagRotationStrategy 2`, `DualMomentumIndexRotationStrategy 1`, `CalendarSeasonalityOverlayStrategy 2`.
- 하지만 “asset을 넓혔다”와 “장기 성능평가가 가능하다”는 다르다. Local Binance parquet coverage는 crypto core `5/5`가 `>=252d`, metals `4/4`가 `>=120d`지만 `>=252d`는 0, tradfi equity `76/76` 중 `>=120d`는 8·`>=252d`는 0, tradfi ETF/index `12/12`는 `>=120d`와 `>=252d` 모두 0, semis required `9/9`도 `>=120d`와 `>=252d` 모두 0이다. 200/252-bar daily lookback 계열은 현 로컬 데이터만으로 clean 장기 점수를 낼 수 없다.
- 주의할 source-level issue: candidate context의 `crypto_symbols`가 “metals가 아닌 모든 normalized symbol”로 잡혀 equity/ETF가 crypto sleeve 쪽으로 섞일 수 있다. 이번 실측 scorer는 crypto/metals/equity/semis/index universe를 명시 override해서 proxy/fill 없이 평가했다.

신규 sleeve 실측 score:
- Artifact: `var/reports/update_validation/new_alpha_real_score/new_alpha_real_score_latest.json|md`.
- 조건: local Binance `1m` parquet only, synthetic/proxy/fill 없음, 기간 `2026-05-16T00:00:00Z` → `2026-06-13T09:25:00Z`, deterministic representative 11 strategies.
- Summary: pass/traded/excluded/fail = `4 / 3 / 1 / 6`.
- 실제 거래된 3개는 모두 손실:
  - `HurstRegimeGatedStrategy`: return `-81.77%`, MDD `109.22%`, trades `761`, signals `726`.
  - `OrderBookImbalanceReversionStrategy`: return `-0.45%`, Sharpe `-55.513`, MDD `0.45%`, trades `417`.
  - `ConfidenceGatedTrendStrategy`: return `-0.05%`, Sharpe `-32.830`, MDD `0.09%`, trades `5`.
- `CalendarSeasonalityOverlayStrategy`는 pass지만 no-trade `0.00%`라 성능 후보가 아니다.
- `LiquidationCascadeReversionStrategy`는 required feature data unavailable로 제외했다. `MetalsRelativeValueBasket`, cross-sectional equity, residual equity, BAB, semis lead-lag, dual-momentum index 계열은 warmup/history 부족으로 fail-closed했다.

기존 성능 우수 후보 baseline:
- `dynamic_conviction_switch:t0.90_risk_capped_fallback`: OOS comp `+53.38%`, Sharpe `2.07`, max OOS MDD `18.80%`, hit `5/10`; highest paper/shadow challenger.
- `cross_candidate_hybrid:hybrid_v3_5`: OOS comp `+27.01%`, Sharpe `1.24`, max OOS MDD `13.72%`, hit `5/10`; robust full-run conservative default.
- `robust_balanced_v1_top1`: OOS comp `+27.03%`, annualized approx `+33.26%`, monthly MDD `2.72%`, hit `7/10`; existing-candidate reuse selector, fresh-forward 전 승격 금지.
- `robust_quality_v1_top1`: OOS comp `+24.55%`, monthly MDD `3.10%`, hit `7/10`; existing-candidate reuse selector, fresh-forward 전 승격 금지.
- `profile_optuna:selected_train_validation_legal`: paper candidate, OOS comp `+18.32%`, Sharpe `0.80`, max OOS MDD `18.80%`, hit `5/10`.
- `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna`: defensive fallback, OOS comp `+6.97%`, Sharpe `0.66`, max OOS MDD `7.32%`, hit `4/10`.
- 결론: 현재 강한 후보는 composite/selector 계열이고, 이번 신규 decorrelated single-alpha sleeves는 현 로컬 실측으로는 기존 우수 후보를 대체하지 못한다. Real-money 승격 후보는 여전히 0개다.

Verification:
- `uv run pytest tests/test_new_decorrelated_alpha_sleeves.py -q` → `3 passed`.
- `uv run pytest tests/test_indicators_core.py tests/test_strategy_factory_library.py tests/test_exact_window_pair_focus_profiles.py tests/test_symbol_canonicalization_pipeline.py tests/test_new_decorrelated_alpha_sleeves.py -q` → `44 passed`.
- `uv run ruff check src/lumina_quant/strategies/registry.py tests/test_strategy_factory_library.py tests/test_new_decorrelated_alpha_sleeves.py` → pass.
- `uv run ruff format --check src/lumina_quant/strategies/registry.py tests/test_strategy_factory_library.py tests/test_new_decorrelated_alpha_sleeves.py` → pass.
- `uv run pytest -q` → `1936 passed in 165.80s`.
- Backtest perf benchmark: median `0.043093s`, median `29424.42 bars/sec`, old baseline 대비 speedup `1311.07x`.
- RSS probe: max RSS `242.22MiB`; 8GB gate PASS (`242.22MiB < 7372.80MiB`, disk snapshot `20.051GiB <= 30GiB`).

## 2026-06-14 KST — defensive row-level selector salvage: 소폭 플러스, 실전은 계속 금지

TradFi/external-alpha 110-asset WF artifact를 빠르게 재계산해 row-level leaf selector 계열을 살렸다. 새 후보는 `row_level_leaf_selector:defensive_validation_utility_mdd20`이고, validation 수익만 쫓던 기존 fast selector보다 validation MDD, train MDD, train 대비 과한 validation spike를 강하게 패널티한다. 선택 입력은 train/validation row metric뿐이며 current-fold OOS는 선택에 쓰지 않는다.

결과:
- Recompute artifact: `/tmp/lumina_defensive_selector_wf.json|md`.
- 신규 defensive selector: OOS comp `+6.72%`, max OOS MDD `14.37%`, positive folds `3/10`.
- 기존 fast row-level selector 대비 개선:
  - `row_level_leaf_selector:validation_calmar_mdd20`: OOS comp `-18.87%`, max OOS MDD `26.62%`, positive folds `4/10`.
  - `row_level_leaf_selector:validation_return_mdd25`: OOS comp `-20.01%`, max OOS MDD `26.62%`, positive folds `4/10`.
  - `row_level_leaf_selector:high_conviction_mdd30`: OOS comp `-31.34%`, max OOS MDD `26.62%`, positive folds `3/10`.
  - `row_level_leaf_selector:stability_utility_mdd25`: OOS comp `-50.99%`, max OOS MDD `26.62%`, positive folds `2/10`.

판단:
- 이건 “살린 후보”지만 clean promotion 후보가 아니다. historical locked-OOS를 본 뒤 추가한 post-OOS research selector이므로 `post_oos_research_variant=true`, `requires_fresh_forward_shadow=true`, `clean_promotion_eligible=false`가 맞다.
- 실전/페이퍼 투입은 계속 금지. frozen rule로 새 forward slice에서 selector 변경 없이 살아남는지 봐야 한다.
- 백테스팅 로직은 focused parity/contract/chunked/live-vs-backtest tests로 재검증했다.

Verification:
- `.venv/bin/python -m pytest tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py -q` → `51 passed`.
- `.venv/bin/python -m pytest tests/test_windowed_backtest_parity.py tests/test_chunked_backtest.py tests/test_run_backtest_data_mode_contract.py tests/test_market_window_emission_parity_live_vs_backtest.py -q` → `17 passed`.
- `.venv/bin/python -m py_compile scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` → pass.
- `.venv/bin/ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` → pass.

## 2026-06-13 KST — 30m+ 후보 재검토: 최고 수익은 dynamic switch, 실거래는 아직 금지

사용자 요청에 따라 최신 `1m` scoreboards가 아니라 며칠 전까지 작업했던 `30m+` 연구만 다시 보았다. 세부 handoff는 `docs/session_handoff_20260613_30m_plus_live_candidate_review.md`에 남겼다. 포함 timeframe은 `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`이고, 최신 저봉/1m 후보는 제외했다.

결론:
- **실거래(real-money) 승인 후보는 0개**다. `var/reports` JSON audit 기준 `ready_for_real=true`, `real_money_execution=true`, `real_execution_allowed=true`가 모두 0건이었다.
- Paper/testnet/live-shadow 후보까지 포함하면 최고 수익은 `dynamic_conviction_switch:t0.90_risk_capped_fallback`: OOS comp `+53.38%`, Sharpe `2.07`, Sortino `15.31`, hit `5/10`, max OOS MDD `18.80%`, `ready_for_paper_folds=10`. 단, risk-capped fallback은 해당 research iteration 이후 도입된 규칙이므로 forward-shadow challenger일 뿐 real-money approval이 아니다.
- 보수적/운영 default는 `cross_candidate_hybrid:hybrid_v3_5`: OOS comp `+27.01%`, Sharpe `1.24`, hit `5/10`, max OOS MDD `13.72%`, `ready_for_paper_folds=4`. Final recommendation도 robust full-run default를 이 후보로 유지한다.
- 기존 후보 재활용 selector의 `robust_balanced_v1_top1`도 `30m/1h/4h`만 고른 비교 후보로 OOS comp `+27.03%`, annualized approx `+33.26%`, monthly MDD `2.72%`, hit `7/10`, PF `7.92`를 냈지만 post-failure diagnostic이므로 fresh-forward 전 승격 금지다.
- 30m+ strategy factory strict pass는 `pair_spread_4h_participation_btcusdt_bnbusdt_2.0_0.50`가 OOS `+4.48%`, Sharpe `2.409`로 가장 깨끗한 strict-pass 아이디어지만 최고 수익 후보는 아니다.

운영 판단: `dynamic_conviction_switch:t0.90_risk_capped_fallback`을 aggressive paper/live-shadow challenger로 freeze하고, `cross_candidate_hybrid:hybrid_v3_5`를 conservative paper control/default로 둔다. 새 unseen slice에서 selector 변경 없이 forward-shadow를 돌리고, 10/15/20bps 비용 stress, BBO/spread/fill, reject/cancel/partial-fill/reconciliation telemetry가 통과하기 전까지 real-money execution은 금지한다.

## 2026-06-09 KST — 기존 후보 재활용 selector 한계 및 microstructure alpha 전환

이번 세션 결론은 기존 후보 pool 안에서 selector를 계속 깎는 방식은 한계가 있다는 것이다. 기존 full candidate artifact `indicator_kalman_ml_robust_selector_full_universe_20260609/clean_new_alpha_discovery_latest.json`는 10 folds / 100,000 rows이고, 이를 train+validation-only 방식으로 재활용하면 기존 clean default 대비 수치는 개선되지만 fresh-forward 전 실전 승격 근거는 아니다.

기존 후보 재활용 결과:
- 새 reuse runner: `scripts/research/run_alpha_zoo_existing_candidate_reuse_selector.py`.
- Report artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/existing_candidate_reuse_selector_20260609/existing_candidate_reuse_selector_latest.json|md`.
- `robust_top1`: OOS comp `22.14%`, ann `27.12%`, monthly MDD `3.10%`, hit `6/10`, PF `4.95`.
- `robust_quality_v1_top1`: OOS comp `24.55%`, ann `30.14%`, monthly MDD `3.10%`, hit `7/10`, PF `6.30`.
- `robust_balanced_v1_top1`: OOS comp `27.03%`, ann `33.26%`, monthly MDD `2.72%`, hit `7/10`, PF `7.92`.
- 판단: 개선은 됐지만 selector variants 자체가 historical locked-OOS 리뷰 후 설계된 post-failure research이므로 `real_money_execution=false`, `fresh_forward_required=true`.

새 후보 확장:
- Clean new-alpha runner에 추가/정리된 family:
  - `btc_beta_residual_momentum`
  - `indicator_kalman_residual_reversion`
  - `feature_microstructure_squeeze_reversal`
- 특히 `feature_microstructure_squeeze_reversal`는 taker-flow, liquidation imbalance, BBO spread, 1pct book-depth imbalance를 동시에 요구하는 microstructure squeeze mean-reversion leaf다. 이 방향이 OHLCV/Kalman/후보줍기보다 ceiling을 깰 가능성이 높다.
- Isolated real-data smoke (`ETHUSDT`, `1h`, last 2 folds, microstructure family only)는 candidate rows `0`이었다. 원인은 alpha 공식보다 feature coverage gate다: BBO+depth+liquidation+flow를 동시에 요구하면 current WF train/validation coverage가 충분하지 않아 fail-closed된다.

운영 결론:
- 같은 locked-OOS window에서 selector mining으로 headline 수치를 더 키우는 작업은 중단한다. 그건 점점 OOS-fitting이다.
- 다음 수익 개선 경로는 microstructure feature coverage를 train/validation 전체로 채운 뒤, frozen selector/family로 fresh-forward 평가하는 것이다.
- 현 단계 real-money/paper promotion은 여전히 blocked. 필요한 gate는 fresh-forward slice, 10/15/20bps cost stress, paper fill/slippage/reject/reconcile telemetry다.
- Verification: `uv run ruff check scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py tests/test_alpha_zoo_clean_new_alpha_discovery.py`, `PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_clean_new_alpha_discovery.py` (`27 passed`), `uv run ruff check scripts/research/run_alpha_zoo_existing_candidate_reuse_selector.py tests/test_alpha_zoo_existing_candidate_reuse_selector.py`, `PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_existing_candidate_reuse_selector.py` (`5 passed`).

## 2026-06-07 KST — Deep-research 최종 결론: 최고 수익 후보는 freeze/shadow, real-money는 아직 금지

최종 리포트: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_best_strategy_clean_oos_20260607/deep_research_best_strategy_clean_oos_20260607.md|json`.

Hard gates는 사용자 답변 기준 `no_nested_oos_mining`, `execution_cost_gate`, `theory_plausibility_gate`다. 이 기준을 동시에 적용하면 **현재 real-money 실전 투입 승인 후보는 없다**. 수익 극대화 관점에서 가장 좋은 운영안은 `clean_input_meta_selector`를 freeze/shadow로 고정하고, `strict_no_leak_best_single`을 paper-control로 관찰하는 2-track이다.

| Track | Evidence class | 주요 성과 | 결론 |
| --- | --- | --- | --- |
| `clean_input_meta_selector` | `shadow-freeze-only` | OOS comp `85.91%`, ann `110.46%`, max OOS MDD `19.29%`, monthly MDD `6.32%`, Sharpe `1.28`, PF `6.26`, hit `5/10`, latest partial month `64.80%` | 최고 수익 후보지만 selector-grid ranking이 historical locked-OOS를 사용했으므로 fresh-forward 전 paper/real 금지 |
| `strict_no_leak_best_single_10bps` | paper-only / no-real-sleeve | 10bps total return `54.56%`, MDD `30.63%`, Sharpe `1.26`, PF `1.21`, hit `6/10`; 20bps stress return `27.10%`, MDD `43.63%` | clean/theory plausible control. Drawdown·cost stress·10-symbol concentration 때문에 real-sleeve 차단 |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | paper baseline / monitor | OOS comp `34.39%`, ann `42.57%`, max bar MDD `27.69%`, hit `3/10`, train mean `12.68%`, val mean `10.34%` | 85-symbol clean mechanics baseline이나 sparse hit-rate와 MDD 때문에 실전 아님 |
| `lagged_shadow_leaf_router cap150` | shadow-only | OOS comp `61.40%`, ann `77.62%`, max bar MDD `29.13%`, Sharpe `1.61`, hit `4/10` | current-fold OOS-free/non-nested이나 post-OOS family라 fresh-forward 전 승격 금지 |
| `clean_new_alpha_discovery_full` | reject / diagnostic-only | 5-family diagnostic OOS comp `+2.51%`, ann `+3.01%`, max OOS MDD `8.77%`, monthly MDD `10.28%`, Sharpe `0.24`, PF `1.20`, hit `5/10`; search hash `4a6fee0f540f5d9ce15158beaf6b7c91ad89600cb5d76e1c4bfa0e33008b81b7` | 새 정보-flow/feature-flow family가 손실은 줄였지만 수익이 너무 낮고 `continuous_position_state_across_split_boundaries` blocker가 있어 `clean_promotion_eligible=false`. 같은 OOS에서 selector 튜닝 금지 |

운영 규칙:
- Freeze manifest: `alpha_zoo_clean_meta_selector_research_20260607/clean_meta_selector_freeze_manifest_latest.json`, sha256 `bd26dcd5116337647d9c6f1ce20ed4710a387184f0f64d0cffce02cb6c21c43a`.
- 같은 historical OOS window에서 selector/grid를 다시 돌려 수치를 키우면 OOS mining으로 간주한다.
- Optuna/hybrid는 train+validation 내부 objective와 fold-local material에만 허용한다. Nested hybrid material, post-hoc selector, OOS tie-breaker는 demote한다.
- Paper fill telemetry는 mean all-in round-trip `<=10bps`, p95 `<=15bps`, reject/reconcile gap 0을 요구한다. 15bps stress 양수와 20bps diagnostic tail 안정성이 없으면 real sleeve 논의 금지.
- TradFi/commodity/stock perps는 계속 85-symbol monitor/backfill로 유지하되, train + 2M validation history가 충분해질 때만 fold-local feature support에 편입한다. 최신/validation-only 편입은 금지한다.

## 2026-06-07 KST — Post-OOS 불신 반영: pre-registered clean new-alpha 1차 결과

사용자 지적대로 post-OOS selector/meta-grid 결과는 clean evidence로 믿지 않는다. Ralplan consensus `2026-06-07-0457-b89b`에서 확정한 원칙에 따라 `+85.91%` meta-selector와 `+61.40%` lagged-shadow는 모두 fresh-forward shadow/hypothesis로만 격리했다.

실행 변경:
- 새 runner `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py` 추가.
- 새 test `tests/test_alpha_zoo_clean_new_alpha_discovery.py` 추가.
- search space는 실행 전 고정/해시화: latest `4a6fee0f540f5d9ce15158beaf6b7c91ad89600cb5d76e1c4bfa0e33008b81b7` after adding pre-registered cross-asset lead-lag and feature-flow crowding reversal diagnostics.
- families: `volatility_squeeze_breakout`, `volume_absorption_reversal`, `range_reclaim_continuation`, `cross_asset_lead_lag_momentum`, `feature_flow_crowding_reversal`.
- fold 선택은 train+validation score only; locked-OOS는 freeze 이후 report/gate only; post-OOS selector trusted false; real-money false.
- split metric policy는 `continuous_full_period_signal_slice_report_only`다. 포지션 state가 split boundary를 넘어갈 수 있으므로 이 artifact는 `clean_promotion_eligible=false`이고, promotion evidence가 아니라 future fresh-forward hypothesis source로만 사용한다.

Core-crypto full 10-fold artifact:
`var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607/clean_new_alpha_discovery_latest.json|md`

결과:
- Latest full OOS comp `+2.51%`, annualized approx `+3.01%`; earlier OHLCV-only diagnostic before lead-lag/feature-flow was `-13.17%`.
- monthly equity MDD `10.28%`, max bar OOS MDD `8.77%`.
- positive folds `5/10`, Sharpe approx `0.24`.
- 결론: cross-asset lead-lag/feature-flow가 손실은 크게 줄였지만 절대 성과가 낮고 continuous carry-state blocker가 있다. 이 1차 genuinely-new alpha set은 diagnostic-only reject다.
- Smoke run(5 core symbols, 3 latest folds)은 `+1.92%`, hit `1/3`였지만 full 10-fold 기준으로만 판단한다.

Performance/verification:
- Optuna hybrid warmup hot path는 이전 병목 `42.650s → 0.934s`로 개선했고, small monthly profile 전체는 `80.633s → 38.525s`.
- Focused verification: `ruff format --check`, `ruff check`, report/artifact assertions, docs verification, py_compile passed; focused pytest `18 passed`.
- 기존 post-OOS meta-selector artifact는 계속 `clean_promotion_eligible=false`, `requires_fresh_forward_shadow=true`로만 유지한다.
- Feature-point extension follow-up: added six Binance feature-backed families — `feature_flow_crowding_reversal`, `feature_liquidation_imbalance_reversal`, `feature_flow_oi_trend_continuation`, `funding_oi_taker_crowding_continuation`, `perp_crowding_score_reversion`, and `feature_taker_flow_exhaustion_reversal` — using local `funding_rate`, `open_interest`, `taker_buy_sell_imbalance`, liquidation notional imbalance, Binance-only crowding score, and price-extension exhaustion with train/validation coverage requirements (`>=60%`) plus fail-closed handling for legacy parquet days missing taker-flow/liquidation columns.
- Full feature-backed 10-fold rerun completed with `--feature-root data/market_parquet/feature_points/exchange=binance`: aggregate still remained `+2.51%` OOS comp / `10.28%` monthly-equity MDD / Sharpe `0.24` / hit `5/10`, identical to the lead-lag-led full rerun.
- Bounded feature-backed reruns up to candidate cap `320` kept the same result: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_clean_new_alpha_discovery_20260607_feature_bounded/clean_new_alpha_discovery_latest.json|md` stayed at OOS comp `-0.24%`, monthly equity MDD `8.72%`, Sharpe `0.04`, hit `3/5`.
- 중요: full과 bounded 둘 다 selected rows는 끝까지 `cross_asset_lead_lag_momentum`이었다. 새 feature-backed 여섯 family 모두 우승 family가 되지 못했다. 즉 현재 Binance feature coverage만으로는 funding/OI/taker-flow/liquidation/crowding-score/price-exhaustion 기반 continuation/reversal family가 우세한 clean alpha를 만들지 못했다.
- Binance BBO groundwork landed: added BBO feature columns (`best_bid_price`, `best_bid_quantity`, `best_ask_price`, `best_ask_quantity`, `bbo_mid_price`, `bbo_spread_bps`) to parquet feature-point schema, support inventory, and the forward-only collector `scripts/collect_binance_book_ticker_feature_points.py`.
- Collector hardening: fixed websocket import compatibility (`websockets.sync.client`) and added `--summary` mode so long-running monitors no longer fail on missing shell `python` or broken pipe parsing.
- Smoke capture succeeded for `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, then expanded to `BNBUSDT` and `TRXUSDT`; rows persist under `data/market_parquet/feature_points/exchange=binance`. Current support-inventory snapshot shows `BTC/ETH/SOL` each with `16` BBO rows, `BNBUSDT` with `19`, and `TRXUSDT` with `10`.
- BBO-aware bounded re-evaluation (`max-folds 5`, candidate cap `360`) still stayed at OOS comp `-0.24%`, monthly equity MDD `8.72%`, Sharpe `0.04`, hit `3/5`. The new `feature_bbo_flow_exhaustion_reversal` family is wired, but current BBO history is still only fresh-forward/current-day sidecar data and cannot satisfy historical train/validation coverage strongly enough to beat `cross_asset_lead_lag_momentum`.
- Continued BBO accumulation check: latest support-inventory snapshot now shows `BTCUSDT 628`, `ETHUSDT 642`, `SOLUSDT 638`, `BNBUSDT 603`, `TRXUSDT 587` BBO rows through roughly `2026-06-07T10:18Z`.
- Re-ran the BBO-aware bounded clean search at candidate cap `450`; result was unchanged (`-0.24%` OOS comp / `8.72%` monthly equity MDD / Sharpe `0.04` / hit `3/5`). Current-day BBO buildup still does not change fold winners.
- Final BBO accumulation freeze: collector loop stopped after rows reached `BTCUSDT 741`, `ETHUSDT 755`, `SOLUSDT 746`, `BNBUSDT 716`, `TRXUSDT 700` through `2026-06-07T10:31:00+00:00`.
- Re-ran the BBO-aware bounded clean search at candidate cap `500`; result still remained unchanged (`-0.24%` OOS comp / `8.72%` monthly equity MDD / Sharpe `0.04` / hit `3/5`). Current-day BBO buildup still does not change fold winners.
- After additional healthy `bg_3` BBO monitor cycles (`rows=60`, `buckets=60`, `errors=0` per cycle), Binance support-inventory now shows `BTCUSDT 705`, `ETHUSDT 719`, `SOLUSDT 710`, `BNBUSDT 680`, `TRXUSDT 664` BBO rows through `2026-06-07T10:26:45Z`.
- Re-ran the bounded BBO-aware clean discovery again at candidate cap `500`; the aggregate remained unchanged at OOS comp `-0.24%`, monthly equity MDD `8.72%`, Sharpe `0.04`, and hit `3/5`.
- Correction to the BBO source gap: official Binance USDⓈ-M real-time `bookTicker` docs are WebSocket-only, but Binance's public historical archive (`data.binance.vision`, documented by `binance/binance-public-data`) also exposes daily/monthly futures `bookTicker` ZIPs under `data/futures/um/{daily,monthly}/bookTicker/{SYMBOL}/`. The prior "official historical bookTicker unavailable" blocker is therefore narrowed to an ingestion/scale problem, not a source-existence problem.
- Follow-up implementation: added `scripts/import_binance_book_ticker_history.py` for generic approved BBO files and `scripts/backfill_binance_public_book_ticker_history.py` for official Binance public-data daily `bookTicker` ZIPs. The official backfill supports cadence sampling (`--cadence-seconds`) and safety caps (`--max-rows-per-archive`) because raw bookTicker archives can contain hundreds of thousands to millions of rows per symbol-day.
- Real official-archive smoke backfill succeeded for core Binance symbols on `2024-03-30`: raw archive sizes were `BTC 7,398,592`, `ETH 5,996,993`, `SOL 5,458,186`, `BNB 2,273,856`, and `TRX 722,198` rows; each was cadence-sampled to `288` five-minute rows and persisted to the local feature-point store. Support inventory now shows `BTCUSDT 1029`, `ETHUSDT 1043`, `SOLUSDT 1034`, `BNBUSDT 1004`, and `TRXUSDT 988` BBO rows, with first BBO timestamps on `2024-03-30T00:00:00Z` and latest forward sidecar timestamps at `2026-06-07T10:31:00Z`.
- Re-ran the cap-500 bounded BBO-aware clean discovery after the official one-day backfill; the result still remained unchanged at OOS comp `-0.24%`, monthly equity MDD `8.72%`, Sharpe `0.04`, and hit `3/5`. One day of historical BBO proves the source/ingestion path but is not enough fold coverage to change selection.
- This still does **not** permit real-money deployment. The next required step is bounded official-archive backfill across enough train/validation dates for the core symbols/folds, then a fresh train/validation-first BBO-aware clean rerun.
- Attempted `2025-12-01..2025-12-07` official daily `bookTicker` backfill for the same core symbols; all 35 requested symbol-days returned missing archive entries. Current bounded folds require train/validation coverage from `2025-01-01` through `2026-05-31`, while the observed official daily `bookTicker` listing for `BTCUSDT` currently ends at `2024-03-30`. Therefore the official Binance public-data archive proves the ingestion path but still cannot satisfy the current fold windows without either a newer historical source, a separate data vendor/source approval, or a deliberately earlier BBO-specific research window.
- Found a better post-2024 official Binance microstructure source: `data/futures/um/daily/bookDepth/{SYMBOL}/` is available through 2026 and files are small (~500KB/day for BTC). Added `scripts/backfill_binance_public_book_depth_history.py`, which derives separate book-depth features (`book_depth_bid_notional_1pct`, `book_depth_ask_notional_1pct`, `book_depth_imbalance_1pct`) rather than pretending this is top-of-book BBO.
- Backfilled official `bookDepth` for core symbols over `2026-05-01..2026-05-31` at hourly cadence: `155` archives imported, `3,720` rows persisted, `744` rows per core symbol. Added pre-registered clean family `feature_book_depth_imbalance_reversal` using this official Binance depth imbalance source.
- Re-ran the bounded depth-aware clean discovery after May 2026 depth import; aggregate still stayed unchanged at OOS comp `-0.24%`, monthly equity MDD `8.72%`, Sharpe `0.04`, hit `3/5`. May-only depth covers late validation/OOS context but still lacks enough train+validation coverage across the five bounded folds, so it does not change fold winners yet.
- Resume checkpoint saved after abort during BNB import: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/bookdepth_resume_state_20260607.json`. Current hourly `bookDepth` coverage for `2025-01-01..2026-05-31`: `BTCUSDT` complete (`12,384` rows), `ETHUSDT` complete (`12,360` rows), `SOLUSDT` complete (`12,360` rows), `BNBUSDT` partial (`2,376` rows; resume from `2025-03-12..2026-04-30`), `TRXUSDT` partial (`744` rows; resume `2025-01-01..2026-04-30`). No `.tmp.parquet` files were present after the latest abort audit.
- Why the coverage step exists: feature-backed families (`feature_bbo_flow_exhaustion_reversal`, `feature_book_depth_imbalance_reversal`, funding/OI/taker/liquidation families) are fail-closed unless feature coverage is at least `60%` in both train and validation. This prevents sparse historical features from being selected only where data happens to exist. For the bounded five-fold run, the relevant train/validation span is `2025-01-01..2026-05-31`; May-only depth data is not enough.
- Backfill throughput issue: current `bookDepth` ingestion streams hourly samples from official ZIPs, but still performs one durable upsert per symbol-day. That is safe but slow over hundreds of days. Resume plan now includes a backfill optimization task: add manifest skip/resume, avoid re-downloading complete symbol-days, batch sampled rows per symbol-month before parquet merge, and only then optionally use bounded symbol-level parallelism.
- Resumed the pending BNB/TRX `bookDepth` coverage imports after batching symbol-level upserts. `BNBUSDT` imported `415` missing archives (`9,960` rows) plus two targeted missing days, and now has complete hourly depth coverage for `2025-01-01..2026-05-31` (`12,384` rows / `516` days). `TRXUSDT` imported `484` archives (`11,616` rows); the official archive is still missing `2026-01-14`, so TRX remains at `515/516` days. Follow-up targeted repairs also completed `ETHUSDT` `2026-03-19`; `SOLUSDT` remains missing official archive `2026-01-14`.
- Re-ran the depth-aware clean discovery after the BNB/TRX coverage repair with `--max-folds 5 --max-candidates-per-fold 500`; aggregate remained no-promotion at OOS comp `-0.24%`, monthly equity MDD `8.72%`, Sharpe `0.04`, hit `3/5`. The improved BNB coverage did not change fold winners, and the remaining `2026-01-14` SOL/TRX official-archive gap is recorded as a source gap, not a selection input.
- Investigated remaining `SOLUSDT`/`TRXUSDT` `2026-01-14` depth gap directly against Binance public-data listings. Daily listing jumps from `2026-01-13` to `2026-01-15` for `SOLUSDT`, and the same date is absent for `TRXUSDT`; monthly `data/futures/um/monthly/bookDepth/{SYMBOL}/{SYMBOL}-bookDepth-2026-01.zip` also returns `404`. Therefore there is no official Binance public archive to import for that day. No synthetic fill was added, because that would turn an audited source gap into an invented feature point.
- Re-ran the same depth-aware clean discovery after confirming the SOL/TRX gap is unrepairable from official Binance public data; result was unchanged: OOS comp `-0.24%`, monthly equity MDD `8.72%`, Sharpe `0.04`, hit `3/5`. Current status remains no-promotion, with `BTC/ETH/BNB` complete and `SOL/TRX` at `515/516` audited official-source days.
- Correction: the preceding depth-aware reruns that passed `--feature-root data/market_parquet` did **not** load parquet feature rows, because the clean runner expects `feature_root/symbol=...` and the correct Binance root is `data/market_parquet/feature_points/exchange=binance`. Re-ran with the corrected feature root and feature-backed candidates were generated (`442` rows: `294` taker-flow exhaustion, `148` book-depth imbalance). Aggregate improved to OOS comp `+7.69%`, monthly equity MDD `5.28%`, Sharpe `1.06`, hit `3/5`, but remains no-promotion because selected rows still carry validation sample/spike rejection reasons and fresh-forward is still required before paper/testnet.
- Corrected-run selections: `2026-04` and `2026-05` selected `feature_taker_flow_exhaustion_reversal` on `ETHUSDT`; other folds selected cross-asset lead-lag. `feature_book_depth_imbalance_reversal` did enter the candidate set with full BNB train/validation/locked-OOS coverage in `2026-02`, but did not become a final fold winner. This means the BNB/SOL/TRX coverage work is now functionally wired into selection, not merely stored.
- Performance complaint follow-up: using single-fold winners alone is weak; the already-built Optuna hybrid path is the right tool for portfolio construction. Ran `run_alpha_zoo_69_asset_optuna_hybrid_refit.py` on the 10-symbol core universe with `1h,4h`, `32` max streams, `160` TPE trials, seed `20260608`. The optimized hybrid is much stronger on train/validation: train return `+19.50%`, train MDD `4.90%`, validation return `+18.55%`, validation MDD `1.51%`, validation Sharpe `9.07`, validation turnover proxy `237.95bps`, `359` validation trade events. It is `ready_for_paper=true` but `ready_for_real=false`; real-money remains blocked until forward fill/BBO/slippage/reconciliation telemetry.
- The Optuna hybrid selected diversified weights across the core universe (`effective_symbol_count 4.99`, top symbol `TONUSDT 33.18%`) and families (`cross_sectional_momentum_rank 52.28%`, `volatility_adjusted_trend_persistence 35.32%`, `trend_pullback_reclaim 12.40%`). This is the current best performance lane; the clean locked-OOS discovery artifact remains a no-promotion research diagnostic, while Optuna is the portfolio construction path to carry into paper/testnet telemetry.

다음 연구 방향:
- 이번 failed families를 그대로 튜닝하지 말고 새 pre-registered family를 추가한다.
- 특히 OHLCV만으로 약한 absorption/squeeze를 반복하지 말고, funding/open-interest/BBO/fill telemetry 또는 cross-exchange/cross-asset lead-lag처럼 OOS 리뷰 후 selector가 아닌 새 정보원을 붙여야 한다.
- fresh-forward 데이터가 생기기 전까지 historical OOS window에서 selector/grid를 더 돌려 숫자를 키우는 작업은 금지한다.

## 2026-06-06 KST — 최신 85-symbol non-nested 결과 문제점/리스크 분석

이번 업데이트는 새 전략 rerun이 아니라, 직전 최신 artifact를 기준으로 “왜 아직 실전 승격이 어렵고 어디가 취약한가”를 명시한 리스크 리뷰다. 근거 artifact는 `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.json|md`이며, source full rerun은 `alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606`다.

### Evidence snapshot

- Protocol: 월 1일 refit, expanding train + 직전 2개월 validation, 다음 1개월 locked OOS, `30m,1h,2h,4h,6h,8h,12h,1d`, 10bps round-trip cost proxy.
- OOS window: `2025-09-01T00:00:00` → `2026-06-06T08:30:00` UTC. `2026-06` fold는 부분 월이다.
- Universe/data: 85/85 symbols loaded, missing 0, 하지만 fold-local train-eligible feature support는 29 symbols이고 신규/짧은 TradFi 56 symbols는 대부분 monitor/backfill 상태다.
- Audit: `metric_reconciliation.metrics_reconciled=true`, candidate 152, clean 123, demoted 29, locked-OOS selection row 0, nested row 0, dynamic self-feed violation 0, online lagged-weight violation 0.
- Runtime: full exact rerun `31:37.96`, peak RSS 약 `1.52 GiB`; row-level selector 재집계는 `0:08.45`, peak RSS 약 `205 MiB`.

### 최신 성과와 판단

| Track | Candidate | Status | OOS comp | Ann approx | Max bar MDD | Sharpe | PF/Omega | Hit | 핵심 판단 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best raw/shadow | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap150` | non-clean shadow | `61.40%` | `77.62%` | `29.13%` | `1.61` | `8.55/8.55` | `4/10` | 현재-fold OOS를 선택에 쓰지는 않았지만 post-OOS 설계라 fresh-forward 필요 |
| Lower-MDD shadow | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap140` | non-clean shadow | `59.99%` | `75.76%` | `27.69%` | `1.60` | `8.70/8.70` | `4/10` | cap150보다 MDD가 낮지만 같은 fresh-forward 제약 |
| Best clean mechanics | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | clean, hard-stop false | `34.39%` | `42.57%` | `27.69%` | `1.12` | `∞/∞` sample | `3/10` | 기계적으로 clean이나 실전 승격 성능/안정성 부족 |
| Lower-MDD clean leaf | `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | clean, partial fold count | `32.74%` | `45.88%` | `14.77%` | `0.84` | `2.95/2.95` | `4/9` | MDD는 낮지만 9 folds, min month `-14.22%`, 단독 robust 부족 |

### Ranked problem synthesis

1. **Clean 후보의 절대 성능이 아직 실전 기준에 못 미친다.** Best clean은 `+34.39%` comp지만 max bar MDD가 `27.69%`이고 hard-stop은 false다. 현재 hard-stop 비교 기준은 historical challenger `+53.38%` comp / `18.8%` MDD, robust-default `+27.01%` comp / `15%` MDD limit이므로, return은 challenger보다 낮고 risk는 robust-default보다 높다.
2. **가장 좋아 보이는 `+61.40%` 성과는 clean promotion 성과가 아니다.** Lagged shadow router는 source leaf만 쓰고 current-fold OOS를 선택에 쓰지 않는 구조지만, 이 family 자체가 기존 OOS 리뷰 이후 설계된 `post_oos_research_variant` / `requires_fresh_forward_shadow`다. 같은 historical OOS window에서 바로 real 후보로 승격하면 OOS-mining 리스크가 크다.
3. **월별 분포가 넓게 안정적이지 않다.** Shadow router는 `4/10` positive folds이고 median OOS가 `0%`다. Best clean dynamic은 `3/10` positive folds이며 7개 fold는 cash/no-position에 가깝다. 즉 headline comp는 일부 강한 월에 크게 의존하고, “항상 골고루 이기는” 형태는 아니다. `∞` Sortino/PF도 손실 월이 없는 작은 표본/cash-gate의 산물이지 무한 edge라는 뜻이 아니다.
4. **85-symbol 확장은 아직 성과 원천이라기보다 모니터링 레이어다.** 모든 symbol bar는 loaded됐지만 train-eligible은 29개뿐이다. 새 TradFi/프리마켓 symbol 다수는 2026년 4~6월 상장/지원 시작이라 train+validation history가 부족하며, 지금 강제로 넣으면 look-ahead 또는 validation-only sleeve 문제가 재발할 수 있다.
5. **June 2026 성과는 부분 월이다.** 최신 OOS는 `2026-06-06T08:30:00`까지만 반영된다. 특히 shadow top의 latest OOS `-3.34%`는 월말 확정치가 아니므로, 좋은 쪽/나쁜 쪽 모두 과해석하면 안 된다.
6. **실전 비용/체결 리스크는 아직 proxy 수준이다.** Backtest는 10bps fixed round-trip cost를 강제하고, symbol simulation은 진입/청산 transition에 각각 half cost를 부과한다. 그러나 실제 Binance futures/TradFi perp의 funding, spread, partial fill, latency, reject/reconcile, session-liquidity 차이는 아직 이 artifact가 증명하지 않는다. Paper fill telemetry와 BBO/slippage guard 통과 전 real-money는 계속 blocked다.
7. **MDD 30% 허용만으로는 clean하지 않다.** Shadow top은 max bar MDD `29.13%`로 한계에 가까우며, bar-level MDD가 live liquidation/margin/portfolio-level intraday gap risk를 완전히 대변하지 않는다. Gross/leverage, per-symbol concentration, stop/fail-closed execution guard가 별도로 필요하다.
8. **평가 속도는 개선됐지만 full search는 여전히 무겁다.** Full exact rerun은 약 31분이 걸렸고, row-level replay는 8초다. 병목은 새 후보를 전부 다시 만드는 full evaluation이며, 향후 월간 운영에서는 per-symbol/timeframe candidate return cache, fold-level immutable row store, incremental latest-fold replay, family별 Optuna stage cache가 필요하다.
9. **Nested/calendar 문제는 현재 통과했지만 유지보수 리스크가 있다.** 최신 artifact에서는 nested row와 locked-OOS selection row가 0이고 calendar-primary 계열도 promotion path에서 제외됐다. 다만 hybrid-as-hybrid를 다시 넣거나 post-hoc selector를 clean으로 오표기하면 같은 문제가 재발하므로, 관련 tests와 demotion flag를 절대 완화하면 안 된다.

### Operating recommendation

- **Real-money 승격:** 아직 금지. 현재 증거는 paper/shadow까지만 충분하다.
- **Paper baseline:** clean mechanics 후보 `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled`를 conservative baseline으로 유지하되, “cash가 많은 방어형 selector”로 해석한다.
- **Return shadow:** `lagged_shadow_leaf_router` cap140/cap150은 non-nested이고 current-fold OOS-free이므로 shadow monitoring에는 가치가 있다. 하지만 2026-06-06 이후 fresh-forward 월별 결과가 쌓이기 전에는 deployable clean으로 부르지 않는다.
- **Fresh-forward gate:** 최소 1~2개 신규 월, 가능하면 4개 월간 refit을 같은 frozen rule로 관찰한다. 새 월에서 OOS comp/MDD/Sharpe가 현 historical shadow와 크게 괴리되면 router를 demote한다.
- **TradFi 확장:** 85 symbols는 계속 수집/모니터링한다. 신규 symbol은 train+2M validation이 충분해질 때까지 feature support에는 자동 편입하되, validation-only 편입은 금지한다.
- **Execution gate:** paper/testnet에서 realized all-in round-trip cost mean ≤10bps, p95 ≤15bps, BBO spread/slippage guard pass, no unexplained reconciliation gap을 먼저 확인한다.
- **Optimization gate:** 앞으로 성능 개선은 같은 OOS window에서 무제한 튜닝하지 말고, train/validation objective 또는 fresh-forward shadow objective로만 판단한다.


## 2026-06-05 KST — 85-symbol clean dynamic v5, CI 실제 경로 검증, ranking/reporting repair

사용자 피드백에 따라 “CI 툴 검증”을 실제 GitHub Actions 기준으로 재확인했다. 직전 push `cf25437e`에서 private-ci는 성공했지만 public `ci` run `27019925528`이 `uv run ruff format --check .` 단계에서 실패했다. 실패 파일은 `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`와 `scripts/research/run_alpha_zoo_69_asset_optuna_hybrid_refit.py`였고, repo-wide ruff format으로 수정했다.

전략 쪽에서는 두 가지 문제가 있었다.

1. `dynamic_conviction_switch:*_val_ret02_calmar80_gate_*_scaled` 후보가 validation-strength gate 실패 fold에서 cash row를 내지 않아 일부 월이 누락될 수 있었다. 이제 scaled gate 후보도 gate 실패 시 `cash_validation_strength_guard` row를 명시적으로 내므로 `fold_count=10`으로 평가된다.
2. aggregate/clean ranking이 comp보다 positive-fold 수를 먼저 정렬해서, 실제 최고 comp 후보가 clean top 표 밖으로 밀리는 리포팅/선정 문제가 있었다. 정렬을 `compounded_oos_return` 우선으로 바꿨고, no-loss 후보의 Sortino/PF/Omega는 0이 아니라 `unbounded` 플래그와 `∞` 표시로 보고한다.

Clean OOS 재평가 artifacts:

- v5 JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v5_20260605.json`
- v5 Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v5_20260605.md`
- OOS schedule: 10 folds, `2025-09-01T00:00:00` → `2026-06-05T12:00:00` UTC, monthly day-1 refit, 2M validation, 10bps, 85/85 loaded symbols, 30m~1D.
- Selection discipline: train + validation only; OOS month is report-only; `nested_hybrid_dependency=false`; `uses_locked_oos_for_selection=false`; final weights are strict-efficiency leaf or cash, not hybrid-as-hybrid material.

Final clean comp ranking after repair:

| Rank | Candidate | OOS comp | Max bar MDD | Monthly eq MDD | Sharpe | Sortino/PF | Hit | Min OOS | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | `34.39%` | `27.69%` | `0.00%` | `1.12` | `∞/∞` | `3/10` | `0.00%` | highest clean comp, trades only strong validation months |
| 5 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd20_scaled` | `29.65%` | `23.59%` | `0.00%` | `1.13` | `∞/∞` | `3/10` | `0.00%` | lower MDD, still selective/no losing OOS month |
| 9 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd30_scaled` | `27.92%` | `27.69%` | `5.75%` | `0.94` | `6.24/5.92` | `5/10` | `-3.18%` | more active but accepts small losing months |
| 13 | `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` | `23.41%` | `23.59%` | `5.75%` | `0.91` | `5.24/5.12` | `5/10` | `-3.18%` | safer active variant |

Full-family/higher-trial v4 was intentionally tested and **not selected**: best clean only `12.45%` comp with `16.97%` max MDD, because the larger candidate pool chased validation strength into OOS tail losses. The correct conclusion is not “add every family”; the better clean result is the stable v3/v5 family set plus validation-only MDD30 sizing and validation-strength cash gate.

Hard-stop status remains false versus the historical challenger (`+53.38%` comp / `18.8%` MDD) and robust-default 15% MDD hurdle. Therefore this is paper/shadow research evidence, not real-money approval. If using it for paper, rank 1 is the return-seeking selective candidate; rank 5 is the lower-MDD selective candidate; rank 9/13 are more active but accept losses.

CI/local verification recorded for this pass:

- `uv run ruff format --check .` — passed.
- `uv run ruff check .` — passed.
- Raw-first periodic preload CI subset — `78 passed`.
- Rust native checks for `rust_metrics`, `rust_rawfirst`, `rust_hybrid_optuna`, `rust_live_signals` — passed.
- Architecture gates: live data, market-window parity, native Binance — passed.
- `scripts/check_architecture.py`, `scripts/audit_hardcoded_params.py`, `scripts/verify_docs.py` — passed (`119 markdown files checked`, hardcoded audit `new=0`).
- Dashboard CI path: `npm install`, `npm run lint`, `npm run test`, `npm run typecheck`, `npm run build` — passed.
- GPU contract tests and auto runtime smoke — `24 passed`, strict Polars GPU smoke passed on detected NVIDIA GPU.
- Full pytest — `1619 passed in 87.51s`.
- Benchmark smoke + 8GB baseline — passed, peak RSS `186.61 MiB` < `7.2 GiB` budget.

Remote CI follow-up: after the first repair push, public `ci` passed ruff/dashboard/GPU but failed full pytest because `DEFAULT_BRIDGE_PROTOCOL_MANIFEST` pointed at ignored runtime state under `.omx/plans/`. The frozen bridge protocol manifest is now tracked at `configs/research/bridge-protocol-manifest-oos-oracle-hybrid-v1-20260602.json`, so clean checkouts and CI can reproduce the bridge tests without local OMX state. Previous public CI format failure root cause was the ruff format drift fixed above.

## 2026-06-04 KST — Fresh full clean non-nested monthly-refit rerun and TradFi auto-expansion monitor

현재 코드 기준으로 69-asset monthly-refit walk-forward를 다시 full rerun했다. 기준은 `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`, round-trip slippage `10bps`, 매월 1일 refit, 직전 2개월 validation, 다음 1개월 locked OOS다. 평가 OOS는 `2025-09-01T00:00:00`부터 최신 보유 데이터 `2026-06-01T06:30:00`까지이며, `2026-06`은 부분 월이다. 총 10개 OOS fold를 사용했다.

이번 rerun은 기존 row recompute가 아니라 현행 no-nested/material guard가 들어간 runner로 새로 실행한 결과다. 검증상 `fold_candidate_rows=625`, `aggregate_rankings=70`, `clean_promotion_rankings=54`, `demoted_nested_or_historical_rankings=16`, 모든 timeframe coverage `69/69`, metric reconciliation `true`, clean contamination violation `0`이었다. Runtime은 `31:54.04`, peak RSS는 `1158.04 MiB`였다.

Artifacts:

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final/clean_non_nested_monthly_refit_full_20260604_final.md`
- SHA256: `83094ae79f946d81b95f1a1a3422f5b2934b82550d9862fc7241e6dc3a93909a`

Clean-promotion ranking 상위:

| Rank | Candidate | OOS Comp | Max OOS MDD | Sharpe | Sortino | Hit | Min OOS | Latest OOS | 판단 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `profile_optuna:selected_optuna` | `10.05%` | `19.20%` | `0.50` | `1.16` | `5/10` | `-9.30%` | `-0.73%` | clean ranker 1위지만 hard-stop promotion 실패 |
| 2 | `profile_optuna:hybrid_v3_5` | `9.06%` | `19.20%` | `0.47` | `1.04` | `5/10` | `-9.30%` | `-0.23%` | non-nested top-level hybrid output, paper/shadow only |
| 3 | `dynamic_conviction_switch:t0.85_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | return은 낮지만 drawdown/tail이 더 안정적 |
| 4 | `dynamic_conviction_switch:t0.90_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | 위와 동일 family/threshold |
| 5 | `dynamic_conviction_switch:t0.95_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | 위와 동일 family/threshold |
| 6 | `dynamic_conviction_switch:t1.00_risk_capped_fallback` | `8.61%` | `10.08%` | `0.80` | `2.65` | `5/9` | `-2.65%` | `0.00%` | 위와 동일 family/threshold |

참고로 comp만 보면 `dynamic_conviction_switch:*_strict_fallback`은 `10.47%` / MDD `10.62%`로 높지만 hit가 `4/9`라 conservative ranker에서 11~12위권으로 내려간다. 어떤 후보도 challenger/robust hard-stop 기준은 통과하지 못했으므로 **real-money 승격은 계속 금지**한다. 실전 후보 해석은 `profile_optuna:selected_optuna`/`hybrid_v3_5`를 수익형 shadow, `dynamic_conviction_switch:*_risk_capped_fallback`을 방어형 shadow로 나누고, fresh-forward paper telemetry가 쌓이기 전까지 capital allocation은 하지 않는 쪽이 맞다.

Nested-hybrid 정책은 유지한다: top-level hybrid output은 분석/후보로 남을 수 있지만, hybrid/blend/selector/gate/portfolio row를 다른 hybrid/portfolio의 **재료**로 다시 넣는 것은 금지한다. 이번 패치로 hidden `final_weights`/`weights` 참조까지 검사해 disguised nested material도 downstream source에서 제외한다. Calendar/month-fixed primary-alpha material도 raw params와 validity/rejection metadata 기준으로 clean material에서 제외한다.

TradFi 확장 모니터링도 추가했다. `lumina_quant.research_universe`는 side-effect 없는 static 69 snapshot을 유지하되, `binance_tradfi_perp_symbols_from_exchange_info()`와 `binance_extended_research_symbols_from_exchange_info()` helper로 Binance `/fapi/v1/exchangeInfo`의 현재 `TRADIFI_PERPETUAL`/USDT trading 심볼을 합칠 수 있게 했다. `scripts/collect_binance_1m_research_universe.py`의 기본 `--universe-source`는 `static-plus-fapi-tradfi`로 바뀌었다. 따라서 명시 `--symbols`가 없으면 기존 69개를 유지하면서, 새 TradFi 지원 심볼은 자동으로 1m data-vision/FAPI fetch plan과 report의 `universe_discovery`에 들어간다. 완전 고정 재현이 필요할 때만 `--universe-source static`을 사용한다.

Verification:

```text
uv run ruff format src/lumina_quant/research_universe.py scripts/collect_binance_1m_research_universe.py tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
uv run ruff check src/lumina_quant/research_universe.py scripts/collect_binance_1m_research_universe.py tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
uv run pytest -q tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
# 9 passed

uv run --extra optimize python scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py ... --output-json ...20260604_final.json
# Exit status 0; elapsed 31:54.04; peak RSS 1185828 KB

uv run ruff format --check .
uv run ruff check .
uv run pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_profit_moonshot_candidate_hybrid.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_strategy_validity_audit.py tests/test_research_universe.py tests/test_collect_binance_1m_research_universe.py
# 64 passed

uv run python scripts/verify_docs.py
# 119 markdown files checked
```

## 2026-06-04 KST — No-nested clean recompute, stale deep-report reconciliation, and report checkpoint optimization

`C:\Users\hoky1\Desktop\deep-research-report.md`를 확인했다. 해당 보고서는 2026-06-03 당시의 `fixed_relaxed_dynamic_blend:*` / `dynamic_aware_hybrid:*` 후보를 중심으로 평가했기 때문에, **전략 랭킹 부분은 현재 no-nested 정책하에서 stale**이다. 다만 selection-bias, execution realism, slippage telemetry, governance/PBO/DSR 같은 주의사항은 여전히 유효하다.

Ralplan consensus 후 실행한 추가 정리:

- 월별 refit runner가 `raw aggregate`, `clean_promotion_rankings`, `demoted_nested_or_historical_rankings`를 분리해 출력하도록 변경했다. 최종 추천/해석은 clean-only ranking에서만 한다.
- recompute artifact에 `recompute_provenance`를 추가했다: source JSON path, source sha256, output paths, `recomputed_from_existing_rows=true`, `fresh_optuna_rerun=false`를 명시한다.
- nested/material detector acceptance를 넓혔다: `cross_candidate_hybrid`, `meta_portfolio`, `dynamic_conviction_switch`, `validation_selector`, `mdd30_*`, `*:selected_optuna`, `*:selected_train_validation_legal`, `*:static_guarded`, `*:hybrid_*` 등은 downstream hybrid/portfolio 재료로 금지된다.
- full rerun 중 매 fold마다 커지는 Markdown을 렌더링하던 비용을 줄이기 위해 `--checkpoint-markdown-interval`을 추가했다. 기본값 `0`은 최종 Markdown만 렌더링한다. JSON checkpoint는 `--checkpoint-interval` 기본값 `1`로 fold-level recovery를 유지한다.

새 no-nested clean recompute artifact:

- Source: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/exact_blend_full_tuning_walkforward_latest.json`
- Source sha256: `563aff7f59174a7ebb6b53f9164eb1feb0cf67881e7f203aecb06987024fa58f`
- Output JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.json`
- Output Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_no_nested_clean_recompute_20260604/no_nested_clean_recompute_latest.md`
- 해석: 기존 row의 governance/ranking repair이며, fresh no-nested Optuna rerun은 아니다.

Clean-only 상위 결과:

| Rank | Candidate | OOS Comp | Max OOS MDD | Sharpe | Sortino | Hit | Min OOS | Latest OOS | 판단 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `relaxed_efficiency:hybrid_v3_5` | `156.03%` | `19.75%` | `1.69` | `10.48` | `5/10` | `-8.41%` | `-0.09%` | clean recompute 기준 1위, 하지만 hit 5/10과 MDD 19.75%라 shadow 우선 |
| 2 | `relaxed_efficiency:selected_optuna` | `60.18%` | `24.27%` | `1.12` | `2.32` | `5/10` | `-22.55%` | `-0.09%` | MDD/left-tail이 커서 보조 후보 |
| 3 | `strict_efficiency:static_guarded` | `40.37%` | `29.16%` | `0.84` | `1.57` | `5/10` | `-26.40%` | `-0.53%` | drawdown 관점에서 실전 후보 약함 |
| 4 | `relaxed_efficiency:selected_train_validation_legal` | `33.82%` | `25.79%` | `0.81` | `1.57` | `5/10` | `-22.55%` | `-0.09%` | 보조/진단 후보 |
| 5 | `strict_efficiency:hybrid_v3_6` | `32.13%` | `15.89%` | `1.03` | `5.48` | `5/10` | `-5.89%` | `-0.17%` | return은 낮지만 MDD가 상대적으로 안정적 |

Demoted raw 상위 후보:

- `fixed_relaxed_dynamic_blend:relaxed70_dynamic30`: raw OOS comp `122.36%`, but `nested_hybrid_dependency`, `post_oos_research_variant`, `requires_fresh_forward_shadow`.
- `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`: raw OOS comp `111.75%`, same demotion reasons.
- `cross_candidate_hybrid:*`, `dynamic_aware_hybrid:*`, `mdd30_high_vol_gate:*`는 non-leaf/nested material 또는 historical research layer로 demoted.

검증:

```text
uv run python -m py_compile scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py

uv run python -m pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# 27 passed in 0.25s

uv run python -m ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# All checks passed

uv run python scripts/research/benchmark_monthly_refit_eval_hotpath.py --min-speedup 1.5
# speedup=6.375x, checksum identical, PASS
```

## 2026-06-04 KST — No-nested-hybrid cleanup and monthly-refit evaluation hotpath optimization

정책을 변경했다: **hybrid/blend/selector/gate/bridge 등 포트폴리오 레이어 후보는 다른 hybrid/portfolio의 재료로 사용할 수 없다.** hybrid는 이미 내부 sleeve를 합친 portfolio이므로, hybrid를 다시 hybrid 재료로 넣으면 분산처럼 보이지만 실제로는 동일 sleeve/asset/factor exposure를 중복 매수할 수 있다.

이번 코드 정리 결과:

- 새 leaf-only 필터 `_leaf_strategy_material_candidate`를 추가해 `cross_candidate_hybrid`, `dynamic_aware_hybrid`, `hybrid_oracle_bridge`, `meta_portfolio`, `validation_selector`, `risk_enhanced_blend`, `fixed_relaxed_dynamic_blend`, `mdd30_*`, `dynamic_conviction_switch` 계열을 downstream hybrid/portfolio 재료에서 제외한다.
- `dynamic_conviction_switch`는 더 이상 `cross_candidate_hybrid:*`, `profile_optuna:hybrid_*`, `selected_optuna` 같은 non-leaf 후보를 고르지 않고, profile/strict/relaxed/individual leaf 후보만 선택한다.
- `dynamic_aware_hybrid`, `risk_enhanced_blend`, `fixed_relaxed_dynamic_blend`는 기존 구현이 non-leaf를 다시 섞는 구조라 명시적으로 no-op 처리했다. 특히 이전 최종 shadow 후보였던 `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` 및 `relaxed70_dynamic30`은 새 정책하에서 promotion/selection 후보가 아니라 historical deprecated artifact로만 본다.
- `validation_selector`, bridge eligible pool, MDD30 risk/gate family도 leaf-only 입력만 받도록 정리했다. MDD30 연구 family는 기존 dynamic/hybrid source 대신 profile/strict/relaxed leaf source만 scale/blend한다.
- 기존 artifact를 재계산하는 fast repair path는 non-leaf reference를 `nested_hybrid_dependency=true`로 표시하고 clean promotion에서 제외한다.

평가 성능도 개선했다. 월별 refit runner의 반복 병목은 같은 candidate return stream에 대해 train/validation/OOS `_period_metrics`를 여러 selector/hybrid/report 단계에서 반복 계산하는 부분이었다. `_period_metrics`에 bounded LRU-style 캐시를 추가해 return series 정렬/Datetime mask 생성은 1회만 수행하고, 이후 window는 int64 timestamp `searchsorted`와 metric-result cache를 사용한다. 캐시는 `LQ_MONTHLY_REFIT_PERIOD_METRICS_CACHE_SIZE`와 `LQ_MONTHLY_REFIT_PREPARED_RETURNS_CACHE_SIZE`로 조정 가능하다.

성능/회귀 검증:

```text
uv run python scripts/research/benchmark_monthly_refit_eval_hotpath.py --min-speedup 1.5
# latest rerun speedup=6.375x, checksum identical, PASS

uv run python -m pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# 27 passed in 0.25s

uv run python -m ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/benchmark_monthly_refit_eval_hotpath.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py
# All checks passed
```

운용 해석: 2026-06-03 note의 `fixed_relaxed_dynamic_blend:*`, `dynamic_aware_hybrid:*`, `risk_enhanced_blend:*`, `hybrid_oracle_bridge:*` 계열 성과는 nested-hybrid exposure risk 때문에 더 이상 최종 선택 근거로 쓰면 안 된다. 새 기준의 최종 후보는 leaf-only 재료로 walk-forward를 다시 돌린 결과에서만 선정해야 한다.

## 2026-06-03 KST — 69-asset monthly clean-OOS walk-forward final selection

최종 69-asset 월별 refit walk-forward 연구를 최신 데이터까지 재평가했다. 기준은 `30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d`, round-trip slippage `10bps`, train expanding from `2025-01-01T00:00:00`, 매월 1일 refit, 직전 2개월 validation, 다음 1개월 locked OOS이다. 평가 OOS는 `2025-09-01T00:00:00`부터 최신 보유 데이터 `2026-06-01T06:30:00`까지이며, `2026-06` fold는 부분 월이다. 총 10개 OOS fold를 사용했다.

Clean/no-leakage contract:

- 각 fold의 현재 OOS는 해당 fold의 파라미터, bridge/dynamic weight, selector label, portfolio weighting에 사용하지 않았다.
- dynamic-aware lane은 same-month dynamic self-feeding 금지 검사를 통과했다.
- bridge/online metric reconciliation은 `metrics_reconciled=true`, mismatches `[]`로 확인했다.
- 단, `fixed_relaxed_dynamic_blend:*` 후보는 OOS 리뷰 후 추가한 exact bar-level 조합이므로 동일 창에서는 clean promotion이 아니라 `fresh_forward_shadow_required_before_promotion`이다.

최종 후보 성과 요약:

| 후보 | Clean | OOS Comp | Ann approx | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF | Hit | Worst | Latest | 판단 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `relaxed_efficiency:hybrid_v3_5` | Y | `156.03%` | `209.00%` | `19.75%` | `15.66%` | `1.69` | `10.48` | `7.04` | `5/10` | `-8.41%` | `-0.09%` | clean-only 최고 comp, paper/testnet 후보 |
| `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` | N | `122.36%` | `160.90%` | `16.66%` | `14.66%` | `1.74` | `8.97` | `6.33` | `6/10` | `-8.42%` | `-0.17%` | shadow 고수익형 |
| `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` | N | `111.75%` | `146.03%` | `16.19%` | `14.35%` | `1.75` | `8.43` | `6.00` | `6/10` | `-8.61%` | `-0.20%` | 최종 균형 shadow 후보 |
| `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | Y | `54.76%` | `68.89%` | `16.75%` | `12.76%` | `1.53` | `4.43` | `3.87` | `5/10` | `-9.97%` | `-0.38%` | clean 방어 sleeve |
| `cross_candidate_hybrid:hybrid_v3_6` | Y | `61.61%` | `77.89%` | `27.57%` | `14.83%` | `1.55` | `4.28` | `4.47` | `7/10` | `-9.48%` | `-0.57%` | 별도 leverage run 내 clean 비교 상위, MDD 큼 |
| `asset_timeframe_leverage:hybrid_v3_6` | Y | `21.78%` | `26.67%` | `21.97%` | `21.19%` | `0.72` | `1.67` | `1.79` | `4/10` | `-15.87%` | `-0.57%` | leverage 진단/monitor only |

월별 OOS 분포에서 최종 균형 shadow 후보 `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`은 `2025-09 +0.59%`, `2025-10 +0.07%`, `2025-11 +45.85%`, `2025-12 -2.90%`, `2026-01 +19.80%`, `2026-02 +30.64%`, `2026-03 -6.28%`, `2026-04 -8.61%`, `2026-05 +11.04%`, `2026-06 -0.20%`였다. `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit`은 같은 기간 `+2.65%, -0.47%, +22.76%, -3.48%, +24.60%, +7.43%, -3.10%, -9.97%, +9.88%, -0.38%`로 수익이 더 낮지만 smoother 방어 sleeve 역할을 한다.

Exposure/rebalancing/leverage 결론:

- 최종 exact blend 60/40은 fold 전체에서 unique assets `29`, 평균 active assets/fold `13.7`, active range `10~29`, 평균 gross `1.92x`, gross range `1.35~2.58x`였다. 후보풀은 69개 전체를 계속 모니터링하며, allocation은 train/validation evidence와 fold별 gate를 통과한 자산만 사용한다.
- 기존 source sleeve 단계에서 `symbol x timeframe x integer_leverage`는 이미 train/validation 기반으로 튜닝된다. 이번에는 그 위에 asset/timeframe post-allocation multiplier와 gross cap을 추가로 clean 검증했다. 하지만 최고 `asset_timeframe_leverage:hybrid_v3_6`도 OOS comp `21.78%`, MDD `21.97%`, Sharpe `0.72`로 final core를 대체하지 못했다.
- 따라서 portfolio-level rebalance/leverage 확장은 즉시 실전 코어 편입이 아니라 monitor/diagnostic 보조축으로 유지한다. 리밸런싱 cadence 자체는 monthly day-1 refit protocol로 고정했고, intramonth에는 signal-level position update로 해석한다.

최종 선택:

1. **실전 균형 shadow/paper 후보:** `fixed_relaxed_dynamic_blend:relaxed60_dynamic40`. OOS comp `111.75%`, max OOS MDD `16.19%`, Sharpe `1.75`, Hit `6/10`로 risk/return 균형이 가장 낫다. 다만 post-OOS research variant라 fresh-forward shadow가 쌓이기 전에는 clean promotion 금지.
2. **고수익 shadow 후보:** `fixed_relaxed_dynamic_blend:relaxed70_dynamic30`. OOS comp `122.36%`로 더 높지만 relaxed sleeve 의존도가 더 크다.
3. **clean-only 기준 최고:** `relaxed_efficiency:hybrid_v3_5`. 같은 OOS 창에서는 clean comp `156.03%`로 최고이나 max OOS MDD `19.75%`, hit `5/10`이고 relaxed repair risk가 있으므로 paper/testnet challenger로 다룬다.
4. **real-money 상태:** 모든 후보는 아직 `ready_for_real=false`로 유지한다. 실전 전환은 fresh-forward shadow, paper/testnet fill/BBO/slippage/reconciliation telemetry, 월별 hard-stop review를 통과해야 한다.

Artifacts:

- Exact blend full-tuning walk-forward: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/exact_blend_full_tuning_walkforward_latest.json` and `.md`. Selection report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/exact_blend_selection_report_ko.md`. Runtime `2:35:12`, peak RSS `1121.3 MiB`.
- Asset/timeframe leverage clean test: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_asset_tf_leverage_20260603/asset_tf_leverage_walkforward_latest.json` and `.md`. Selection report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_asset_tf_leverage_20260603/asset_tf_leverage_selection_report_ko.md`. Runtime `1:40:52`, peak RSS `1085.0 MiB`.
- Supporting earlier OOS/selector artifacts: `alpha_zoo_69_asset_best_strategy_factory_20260601`, `alpha_zoo_69_asset_monthly_refit_diagnostics_20260601`, `alpha_zoo_69_asset_clean_full_tuning_20260602`, `alpha_zoo_69_asset_dynamic_aware_hybrid_20260602`, `alpha_zoo_69_asset_oos_oracle_bridge_20260602`, `alpha_zoo_69_asset_risk_enhanced_20260602`, and `alpha_zoo_69_asset_mdd30_high_vol_20260602`.

Verification after final patches and report generation: `py_compile` and `ruff check` passed for all changed/untracked Python research files; targeted Alpha Zoo 69-asset pytest passed `66 passed in 1.31s`; docs verification passed (`118 markdown files checked`); `git diff --check` passed.


## 2026-05-31 KST — Relaxed repair interpretation: trust, liquidation, 69→19 selection, and future refit of train-ineligible assets

Follow-up interpretation for the MDD-guarded relaxed 69-asset efficiency-repair artifact. The relaxed artifact is **not** a real-money/live-ready result. It is a high-return, high-MDD **paper/testnet challenger** that should be compared against the stricter lower-MDD v3.6 live-handoff baseline before any promotion decision.

Trust assessment:

- Trustworthy parts: the corrected train-eligibility guard is preserved; train-ineligible symbol/timeframe rows are not used for parameter fitting, repair source rows, sleeve allocation, hybrid selection, or live promotion. The primary `10bps` round-trip RPT gate remains strict. Train and validation are both strongly positive and selected aggressive profile train return remains above validation return.
- Discounted parts: no locked test/OOS is active in this final-refit-style pass; paper/testnet fill telemetry is not yet measured; relaxed policy admits material-positive train<validation and low-sample TradFi rows under MDD/RPT guards; selected gross is high at `7.2541x`; MDD is material. Therefore these numbers must not be read as live expected returns.
- Current operating conclusion: keep the strict corrected v3.6 result as the safer baseline while running the relaxed artifact as paper/testnet challenger evidence.

Liquidation / wipeout interpretation:

| candidate | train liquidation / wipeout | validation liquidation / wipeout | train MDD | validation MDD | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | `0 / 0` | `0 / 0` | `27.7065%` | `21.2478%` | liquidation-free in replay, but high drawdown |
| relaxed selected `hybrid_v3_5_optuna_three_profile_blend` | `0 / 0` | `0 / 0` | `23.9019%` | `16.3433%` | lower MDD than aggressive, but validation exceeds train under relaxed rule |

The liquidation fields are clean in train/validation replay, but locked-OOS/test-set liquidation evidence is not available for this final-refit artifact. The correct reading is: no simulated liquidation/wipeout occurred under the replay assumptions, but MDD and live fill/slippage risk remain the binding safety concerns.

Why 19 symbols instead of all 69:

1. Research/monitoring universe is `69` symbols.
2. `37` symbols had no train-split rows and remain watch/shadow-only under the corrected no-validation-only-leakage policy.
3. `32` symbols are train-eligible.
4. `32 × 3` profile rows produce `96` candidate rows.
5. Relaxed gate passes `30` rows across `19` unique symbols; all `30` relaxed gate-ok rows are selected as sleeves in the profile set.

Selected relaxed symbols are `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `COINUSDT`, `COPPERUSDT`, `CRCLUSDT`, `DOGEUSDT`, `ETHUSDT`, `GOOGLUSDT`, `INTCUSDT`, `METAUSDT`, `MSTRUSDT`, `PLTRUSDT`, `SOLUSDT`, `TONUSDT`, `XAGUSDT`, `XPDUSDT`, and `XPTUSDT`. Eligible symbols that still have no relaxed gate-ok row are `AMZNUSDT`, `BZUSDT`, `CLUSDT`, `EWJUSDT`, `EWYUSDT`, `HOODUSDT`, `NATGASUSDT`, `NVDAUSDT`, `PAYPUSDT`, `TRXUSDT`, `TSLAUSDT`, `XAUUSDT`, and `XRPUSDT`. Main rejection causes remain insufficient validation return, train/validation RPT not above `10bps`, train<validation without the material-positive/MDD exception, and profile trade-count minima.

Forced inclusion of all 69 assets is intentionally rejected. The standard rule is: monitor all 69; allocate only to symbols with train evidence and passing RPT/MDD/sample/concentration gates; keep the rest as shadow/watchlist until they earn train+validation evidence.

Future refit expectation for the 37 train-ineligible assets:

- Re-running the same split does not make the 37 assets eligible; they still have no train rows in that split.
- A train+validation final refit can technically fit them, but it would leave no independent validation for those assets and should not promote them by itself.
- A later rolling refit, after enough new bars make today’s validation/cold-start period part of train and reserve a fresh latest validation window, can admit a meaningful subset.
- Based on the cold-start donor-frozen shadow artifact, roughly `15–20` of the 37 could plausibly enter the candidate pool when real train history exists, but optimizer selection will likely keep fewer nonzero exposures.

Cold-start shadow evidence for the 37 symbols remains report-only: donor-frozen primary shadow selected `18` sleeves with validation `+31.6832%`, validation MDD `10.2407%`, validation RPT `83.87bps`, gross `2.0x`, and no promotion because target train rows are absent. Stronger watchlist names include `MUUSDT`, `SNDKUSDT`, `AMDUSDT`, `DRAMUSDT`, `QCOMUSDT`, `SOXLUSDT`, `QQQUSDT`, `SPYUSDT`, `MRVLUSDT`, `ARMUSDT`, `AVGOUSDT`, and `TSMUSDT`. Weak/negative cold-start names such as `OPENAIUSDT`, `SPCXUSDT`, `BABAUSDT`, `WDCUSDT`, and `COHRUSDT` need fresh train evidence before any inclusion.


## 2026-05-31 KST — MDD-guarded relaxed 69-asset efficiency repair

Applied the operator's relaxed gate interpretation to the corrected 69-asset efficiency-repair artifact without changing the 10bps execution-efficiency requirement. New runner/test: `scripts/research/run_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna.py` and `tests/test_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna.py`. Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_20260531/alpha_zoo_69_asset_relaxed_efficiency_repair_optuna_latest.json` plus Markdown/CSV siblings.

Relaxed policy `material_positive_tradfi_low_sample_mdd_guard_20260531`:

- `train < validation` is no longer a hard rejection when both train and validation returns are at least `2%` and the MDD guard passes.
- TradFi low-sample rows can be admitted as warnings when MDD and the strict `>10bps` train/validation RPT gates pass.
- Material-positive non-TradFi low-sample rows are also reportable/admissible under the same MDD/RPT guard, but the optimizer still penalizes relaxed/low-sample notional share.
- Gross/concentration pressure is optimized and penalized rather than hard rejected while validation MDD remains under the relaxed guard; a `12x` hard gross cap remains.
- Train-ineligible symbol/timeframe rows are still excluded from parameter fitting, repair source rows, sleeve allocation, hybrid selection, and live promotion.
- No locked test/OOS is used for selection. All artifacts remain `paper_testnet_only=true`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

Candidate pool impact versus the strict corrected pass: strict gate-ok rows `18`; relaxed gate-ok rows `30`; newly admitted rows `16`; relaxed unique gate-ok symbols `19`. Newly admitted symbols: `COINUSDT`, `COPPERUSDT`, `CRCLUSDT`, `ETHUSDT`, `GOOGLUSDT`, `INTCUSDT`, `METAUSDT`, `MSTRUSDT`, `PLTRUSDT`, `TONUSDT`, and `XPDUSDT`.

Best train/validation-legal relaxed portfolio is the aggressive profile:

| portfolio | train | validation | train MDD | validation MDD | RPT bps train/val | 20bps stress train/val | gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | `+555.8771%` | `+518.4857%` | `27.7065%` | `21.2478%` | `96.70 / 286.89` | `+498.3937% / +500.4133%` | `7.2541x` | `true` | `false` |

Selected relaxed Optuna hybrid comparison is `hybrid_v3_5_optuna_three_profile_blend`: train `+284.5998%`, validation `+373.8607%`, train/validation MDD `23.9019%/16.3433%`, RPT `58.07/250.46bps`, gross `6.8713x`, `selection_reasons=[]`, paper/testnet only. This hybrid uses the relaxed dominance rule because both train and validation are materially positive and MDD remains below the relaxed hybrid guard.

Strict corrected reference remains available and safer/lower-MDD: selected balanced profile train/validation `+119.3799%/+79.7120%`, MDD `16.6872%/7.4789%`, RPT `108.53/157.53bps`, gross `2.2x`; strict live handoff v3.6 train/validation `+96.5913%/+68.1871%`, MDD `12.5785%/7.8678%`, RPT `42.30/149.64bps`, gross `2.5042x`. The relaxed artifact is a higher-return, higher-MDD paper/testnet challenger, not a real-money enablement.

Selected relaxed sleeve set has `30` sleeves across `19` train-eligible symbols. Concentration for the aggressive selected profile: top symbol `PLTRUSDT` `13.78%`, top group `tradfi_equity` `53.59%`, effective symbol count `9.95`; group mix `crypto_core 38.05%`, `tradfi_equity 53.59%`, `tradfi_commodity 8.36%`. All sleeve integer leverages remain integers (`2x` to `12x`). Runner evidence: wall `1:49.86`, max RSS `914,748 KiB` / artifact `893.31 MiB`, below the 8GB cap.

## 2026-05-31 KST — Cold-start donor transfer shadow for validation-only assets

Added the report-only cold-start transfer pass for the `37` train-ineligible 69-asset symbols. The new runner `scripts/research/run_alpha_zoo_69_asset_cold_start_transfer_shadow.py` and tests `tests/test_alpha_zoo_69_asset_cold_start_transfer_shadow.py` evaluate whether recently listed TradFi/premarket symbols can be initialized from similar train-eligible donor profiles without leaking target validation PnL into donor choice.

Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_cold_start_transfer_shadow_20260531/alpha_zoo_69_asset_cold_start_transfer_shadow_latest.json` plus Markdown/CSV siblings. Source remains the corrected 69-asset profile artifact and the strict live reference remains `alpha_zoo_69_asset_efficiency_repair_optuna_20260530`; the live handoff was not changed.

Safety contract in the artifact:

- Donor selection uses donor train/validation quality, static domain similarity, and target bar coverage only. Target validation PnL is not used in the primary donor-frozen lane.
- No donor OHLCV, donor PnL, donor trade counts, or synthetic target train metrics are substituted into target train performance.
- Validation-oracle selection is emitted only as a diagnostic upper bound and is explicitly non-promotable.
- `paper_testnet_only=true`, `shadow_report_only=true`, `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` throughout. The primary cost/RPT assumption remains `10bps` round trip.

Results under the latest 8-week validation window (`2026-04-04T04:00:00` through `2026-05-30T03:00:00`):

| lane | sleeves | gross | train | validation | validation MDD | validation RPT | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| donor-frozen primary cold-start shadow | `18` | `2.00x` | `0.00%` | `+31.6832%` | `10.2407%` | `83.87bps` | report-only, not promotable |
| validation-oracle diagnostic upper bound | `16` | `1.7778x` | `0.00%` | `+44.4677%` | `8.1768%` | `178.67bps` | leakage diagnostic, not promotable |

Primary donor-frozen selected symbols were `TSMUSDT`, `MUUSDT`, `SNDKUSDT`, `AVGOUSDT`, `AMDUSDT`, `QCOMUSDT`, `MRVLUSDT`, `DRAMUSDT`, `WDCUSDT`, `ARMUSDT`, `COHRUSDT`, `BABAUSDT`, `SPYUSDT`, `SOXLUSDT`, `QQQUSDT`, `SPCXUSDT`, `OPENAIUSDT`, and `QNTXUSDT`. Positive contributors included `MUUSDT`, `SNDKUSDT`, `AMDUSDT`, `DRAMUSDT`, `QCOMUSDT`, `SOXLUSDT`, `QQQUSDT`, and `SPYUSDT`; negative contributors included `COHRUSDT`, `SPCXUSDT`, `OPENAIUSDT`, `BABAUSDT`, and `WDCUSDT`. The primary lane stays useful as a monitored cold-start watchlist but cannot enter live/paper allocation until those symbols have physical train-window data.

Runner evidence: wall `0:14.80`, max RSS `852,388 KiB`; artifact `runner_peak_rss_mib=832.41`, below the 8GB cap. Local targeted verification passed: new cold-start tests `5 passed`; related Alpha Zoo targeted suite `16 passed`; Ruff check and compileall passed for the new runner/tests.

## 2026-05-31 KST — 69-asset train-eligibility correction and live handoff refresh

Corrected the 69-symbol efficiency-repair pipeline after finding that several newly listed TradFi/premarket symbols had no train-split rows before `2026-04-04T03:00:00`, yet the earlier allocation layer could still admit validation-only sleeves. The superseded headline `+295.9880%` train / `+172.7926%` validation was therefore contaminated by validation-only assets and is no longer the live handoff evidence.

Implemented train-eligibility gating in the shared 69-asset refit utilities and repair runner:

- Build a symbol/timeframe train-eligibility report from the actual train/validation windows.
- Exclude any symbol/timeframe with zero train rows from per-asset parameter fitting, repair source rows, sleeve allocation, hybrid selection, and live promotion.
- Keep those symbols in the watch/data universe only; they become eligible only after a future refit has real train-split history.
- Apply `warmup_ratio` to train-split bars only, even when final refit fits train+validation parameters.
- Filter rejected/diagnostic repair streams out of allocation so rows with `efficiency_repair_reasons` cannot re-enter through portfolio weights.

Train eligibility in the corrected artifact: `32` eligible symbols and `37` train-ineligible symbols. The train-ineligible list is `QQQUSDT`, `SPYUSDT`, `SOXLUSDT`, `AAPLUSDT`, `TSMUSDT`, `MUUSDT`, `SNDKUSDT`, `MSFTUSDT`, `AVGOUSDT`, `BABAUSDT`, `AMDUSDT`, `QCOMUSDT`, `USARUSDT`, `LITEUSDT`, `ORCLUSDT`, `DISUSDT`, `UBERUSDT`, `CSCOUSDT`, `HDUSDT`, `MRVLUSDT`, `CRWVUSDT`, `WMTUSDT`, `JPMUSDT`, `VUSDT`, `BRKBUSDT`, `FLNCUSDT`, `DRAMUSDT`, `RKLBUSDT`, `CBRSUSDT`, `NBISUSDT`, `WDCUSDT`, `ARMUSDT`, `BEUSDT`, `COHRUSDT`, `SPCXUSDT`, `OPENAIUSDT`, and `QNTXUSDT`.

Corrected artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_efficiency_repair_optuna_20260530/alpha_zoo_69_asset_efficiency_repair_optuna_latest.json` generated at `2026-05-30T16:17:37Z`. Rerun evidence: wall `1:55.26`, max RSS `891,500 KiB` / artifact `870.61 MiB`, below the 8GB budget. No locked test set is used in this final-refit style pass; train and latest 8-week validation are the only selection inputs.

Corrected selected train/validation-legal portfolio:

| portfolio | train | validation | train MDD | validation MDD | RPT bps train/val | 20bps stress train/val | gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | `+119.3799%` | `+79.7120%` | `16.6872%` | `7.4789%` | `108.53 / 157.53` | `+108.3799% / +74.6520%` | `2.2000x` | `true` | `false` |

Corrected paper/testnet live handoff now uses artifact-selected `hybrid_v3_6_optuna_three_profile_blend`:

| hybrid | train | validation | train MDD | validation MDD | RPT bps train/val | historical gross | live final gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hybrid_v3_6_optuna_three_profile_blend` | `+96.5913%` | `+68.1871%` | `12.5785%` | `7.8678%` | `42.30 / 149.64` | `2.5042x` | `2.3389x` | `true` | `false` |

The corrected v3.5 comparison remains paper-eligible but is not the selected live handoff: train `+153.0941%`, validation `+57.2165%`, train/validation MDD `14.0862%/8.4343%`, RPT `49.29/91.23bps`, gross `3.0697x`.

The live adapter now reconstructs `18` selected source sleeves, watches all `69` symbols, and has nonzero selected source exposure only in `13` train-eligible symbols: `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `COPPERUSDT`, `CRCLUSDT`, `DOGEUSDT`, `SOLUSDT`, `TONUSDT`, `XAGUSDT`, `XAUUSDT`, `XPTUSDT`, and `XRPUSDT`. No train-ineligible asset has nonzero selected live gross after the fix.

Limit/no-fill/slippage policy is unchanged and remains paper/testnet only: `LMT` default, `one_tick_worse`, market fallback disabled, max chase attempts `0`, missing/high-spread/high-slippage BBO guard skips without market conversion, and realized all-in round-trip cost must stay within the `10bps` replay/gate assumption before any real-money review. Safety flags remain `paper_testnet_only=true`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

Verification after the correction passed: targeted eligibility/live adapter suites `23 passed`; `ruff format --check .`; `ruff check .`; `compileall`; docs verification `117` markdown files; architecture check; hardcoded-parameter audit `new=0`; `git diff --check`; artifact invariant script confirmed no train-ineligible selected live gross; and full `pytest -q` `1539 passed` with max RSS `2,756,628 KiB` (<8GB). static dependency review was refreshed and reported high impact because the research/live artifacts and default live profile changed; the change is intentional and covered by the verification above.

## 2026-05-30 KST — 69-asset per-profile Optuna rebuild after broad-blend audit

Follow-up to the first broad 69-symbol blend: the earlier pass optimized a diversified stream blend and did **not** fully rebuild the original hybrid source-profile logic per asset. Added `scripts/research/run_alpha_zoo_69_asset_profile_optuna_hybrid_refit.py` and `tests/test_alpha_zoo_69_asset_profile_optuna_hybrid_refit.py` to run the corrected expansion:

- Treat the three source profiles as risk/selection templates, not as three final assets: balanced/growth/aggressive each tunes all 69 symbols independently.
- Each symbol/profile pair is Optuna-tuned over family, timeframe, side, entry/exit, min-hold, cooldown, and integer leverage; no grid selection is used for the tunable search.
- Domain anchors are tracked beyond BTC: BTC, ETH, SOL, SPY, QQQ, XAU, XAG, and crude proxy anchors. Candidate/profile objectives penalize single-anchor clones, top-symbol concentration, top-asset-group concentration, and top-anchor concentration.
- Each rebuilt source profile performs an Optuna sleeve-allocation pass, then the three rebuilt multi-asset profile streams are passed to the v3.5/v3.6 Optuna hybrid engine plus a train-dominance guarded static Optuna blend.
- Locked test/OOS remains disabled for live-final-refit style research; latest 8 complete weeks are validation/report evidence. Real-money flags remain false.

Final artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_profile_optuna_hybrid_refit_20260530/alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json`.

OOM evidence: full 69-symbol per-profile run completed in `7:05.78` wall time with max RSS `1,043,232 KiB` / artifact `runner_peak_rss_mib=1018.78125`, safely below 8GB.

Selected train/validation-legal portfolio is the guarded static blend selecting the rebuilt balanced multi-asset profile:

| portfolio | train | validation | train MDD | validation MDD | RPT bps train/val | gross | paper | real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hybrid_static_train_dominance_guarded_three_profile_blend` | `+160.3316%` | `+150.0726%` | `15.3371%` | `7.5634%` | `69.40 / 152.18` | `5.00x` | `true` | `false` |

Rebuilt source profile audit:

| profile | sleeves | train | validation | validation MDD | RPT bps train/val | top symbol | top anchor | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| balanced | `22` | `+160.3316%` | `+150.0726%` | `7.5634%` | `69.40 / 152.18` | `XPTUSDT 10.73%` | `energy_crude_beta 21.77%` | train/validation legal |
| growth | `30` | `+174.8699%` | `+1433.8154%` | `17.630%` | `38.3 / 594.9` | `AAPLUSDT 9.56%` | `crypto_beta_btc 20.60%` | rejected as validation spike |
| aggressive | `45` | `+218.5251%` | `+1497.3979%` | `18.2%` | `43.7 / 642.9` | `MUUSDT 7.93%` | `crypto_beta_btc 29.81%` | rejected as validation spike + gross > 8x |

The adaptive v3.5/v3.6 hybrid result is deliberately not promoted despite high validation because it violates the train-dominance rule (`train +301.56% < validation +715.70%`). This is recorded as a validation-spike rejection, not a live candidate. The legal candidate is therefore the balanced multi-asset profile/guarded static blend. It is still paper/testnet-only and requires exchange-connected paper telemetry before any real review.

Canonical path: `docs/research_note/research_note.md`.

This file is the current cumulative research note / research journal. Keep the filename stable as `research_note.md`; strategy, project, and date labels belong inside entries, not in the path.

Current live/paper-testnet identity:

- Runtime strategy name: `AlphaZooOptunaHybridLiveStrategy`.
- Selected frozen profile/artifact family for the corrected 69-asset efficiency-repair handoff: `hybrid_v3_6_optuna_three_profile_blend`. Older standard/integer-leverage artifacts using `hybrid_v3_5_optuna_three_profile_blend` remain historical baselines.
- `profit_moonshot_alpha_zoo` is retained only as a historical artifact namespace, not as the strategy name.
- Real-money execution remains prohibited until separate paper/testnet fill telemetry gates pass.

## Research journal — latest first

Prepend new research diary entries below this heading. The legacy historical entries were reordered newest-first during the 2026-05-28 naming cleanup.

## 2026-05-30 KST — 69-asset Optuna hybrid refit and concentration audit

Expanded the Alpha Zoo broad research pass from the prior ETH/SOL/TRX-heavy incumbent universe to the full `69`-symbol Binance research universe (`10` core crypto + `59` Binance USD-M `TRADIFI_PERPETUAL` proxies) using direct stored `1m` OHLCV resampled into `30m`, `1h`, `2h`, and `4h` bars. The new runner is `scripts/research/run_alpha_zoo_69_asset_optuna_hybrid_refit.py`; outputs are under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_optuna_hybrid_refit_20260530/`. The standard live-facing split rule is preserved: latest `8` complete weeks are validation, no locked-OOS/test set is used in this final-refit mode, and every real-money flag remains hard false. Candidate and hybrid search uses Optuna TPE through the shared `lumina_quant.optimization.search_policy` path, not grid search.

OOM-safe implementation choices: direct 1m reads are symbol/timeframe chunked, in-memory hybrid streams are capped (`MAX_CACHED_STREAMS_PER_SYMBOL=8`, run cap `--max-hybrid-streams 48`, `--max-streams-per-symbol 2`), and the selected-stream matrix is built only after ranking/freeze. Full run evidence: `/usr/bin/time` wall `1:00.25`, max RSS `838,416 KiB` and payload `runner_peak_rss_mib=818.77`, well below the 8 GiB session budget.

Discovery result after embedded `10bps` round-trip friction proxy:

- `45,120` candidates evaluated; no single component passed full paper promotion by itself. `17` rows passed the sample gate but failed execution-efficiency proxy, so they remain shadow rows (`AVAXUSDT` 10, `HOODUSDT` 4, `COINUSDT` 2, `XAGUSDT` 1).
- The Optuna-selected aggregate hybrid passed the strict backtest paper gate only as a diversified hybrid: train `+9.0960%`, validation `+8.5640%`, train MDD `2.5726%`, validation MDD `0.6934%`, train RPT proxy `10.9603bps`, validation RPT proxy `31.8371bps`, component trade events train/validation `9,613 / 3,789`, liquidation/account wipeout `0 / 0`. Backtest gate says `ready_for_paper=true`, but `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`.
- Concentration audit improved materially versus the incumbent: top symbol `TONUSDT` share `16.27%`, effective symbol count `11.59`, top asset group `crypto_core` `49.91%`, TradFi equity `32.64%`, ETF/index `10.21%`, commodities `7.24%`; no concentration flags. Validation long/short exposure when active is approximately `52.06% / 47.94%`, so the hybrid is not only one direction despite every rule being long/short-capable. Timeframe mix: `30m` `41.94%`, `1h` `23.93%`, `2h` `15.29%`, `4h` `18.84%`. Family mix: volatility-adjusted trend persistence `55.45%`, cross-sectional momentum rank `29.20%`, pullback reclaim `15.34%`.

Interpretation: the broad 69-asset hybrid is a lower-return, lower-MDD, much more diversified challenger/shadow paper candidate. It is not a replacement for the current high-return standard live refit yet: the prior `hybrid_v3_5_optuna_three_profile_blend` standard refit remains far higher returning (`validation +38.0717%`, validation MDD `7.4789%`, gross `4.6902x`) but is more concentrated and train-dominant. The 69-asset result is therefore best treated as a paper/testnet challenger whose value is diversification and concentration reduction, not immediate performance superiority.

Real-transition decision: do **not** enable real-money. The path from this artifact to real requires a separate live adapter/handoff if this 69-asset rule set is to be executed, then at least `2-4` weeks of paper/testnet forward telemetry for limit-first fills, realized fee/spread/slippage/all-in round-trip cost against the `10bps` replay assumption, intended-vs-actual notional parity, partial/cancel/timeout rates, BBO/depth/fill-latency quality, protective-order attach/reconciliation, liquidation-distance/margin buffers, and continued asset/position concentration monitoring.

Local verification for this pass: repo-wide `ruff format --check .` and `ruff check .`; targeted tests `23 passed`; `compileall`; docs verification `117` markdown files; architecture check; hardcoded-parameter audit `new=0`; `git diff --check`; and full `pytest -q` `1519 passed in 104.14s` with max RSS `2,728,592 KiB` (<8 GiB). Commit, push, and CI follow in the same work session.

---

## 2026-05-30 KST — Binance 69-symbol direct 1m research-bar backfill

Updated the expanded Binance research universe to `69` compact USDT symbols after the 2026-05-29 addition of `QNTXUSDT`: `10` core crypto plus `59` active Binance USD-M `TRADIFI_PERPETUAL` symbols. `QNTXUSDT` is classified under the premarket/not-yet-standardized equity proxy group. This universe is a research/shadow-monitoring input only; it does not expand the selected paper/testnet live strategy universe and does not approve real-money trading. Hard safety flags remain `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

A direct `1m` OHLCV collector was added at `scripts/collect_binance_1m_research_universe.py` with two historical sources: official Binance public `data.binance.vision` USD-M kline archives for broad backfills and throttled `/fapi/v1/klines` for targeted current tails. The first REST attempt intentionally used small chunks but no global request throttle and triggered Binance HTTP `429/418`; the IP ban response stated `banned until 2026-05-30T04:51:48.940Z`. The collector was therefore hardened with a global request throttle and explicit 429/418 ban backoff, and the historical backfill was rerun through official `data.binance.vision` to avoid further REST-weight pressure.

Historical collection result: `var/reports/data_collection/binance_1m_research_universe/binance_1m_research_universe_collection_latest.json`, source mode `data-vision`, range `2025-01-01T00:00:00Z` through `2026-05-28T23:59:59.999Z` (latest public daily file confirmed during the run). The run processed `69` planned symbols with `0` errors: `53` fetched in this run, `15` already up-to-date from the earlier partial REST write/resume state, and `1` initially empty (`QNTXUSDT`, because it listed after the currently published 2026-05-28 data-vision daily boundary). It fetched/upserted `10,071,511` rows from `1,661` source files, made `2,326` archive requests, and had `665` expected pre-listing missing files. Runtime evidence: `/usr/bin/time` wall `2:23.92`, max RSS `235,036 KiB`, below the 8 GiB memory budget.

After the Binance REST ban cleared, a separate throttled FAPI tail job filled the unpublished current tail and `QNTXUSDT`: `var/reports/data_collection/binance_1m_research_universe_fapi_tail/binance_1m_research_universe_collection_latest.json`, source mode `fapi`, range `2026-05-29T00:00:00Z` through `2026-05-30T04:40:59.999Z`, `69/69` symbols ok, `0` errors, `92,679` fetched/upserted rows, `123` requests, max RSS `133,636 KiB`. `QNTXUSDT` now has `1,226` 1m rows from `2026-05-29T08:15:00Z` through `2026-05-30T04:40:00Z`. Final coverage artifact: `var/reports/data_collection/binance_1m_research_universe_coverage/binance_1m_research_universe_coverage_latest.json`; direct stored 1m coverage now has `69/69` non-empty symbols and `11,559,900` rows under `data/market_parquet/exchange=binance/symbol=*/timeframe=1m`, with every symbol covered through at least `2026-05-30T04:40:00Z`. Full-year crypto symbols have `740,441` rows each from 2025-01-01 through 2026-05-30 04:40 UTC. Storage footprint for the direct Binance 1m tree is about `245M`.

Operational decision: use official `data.binance.vision` monthly/daily 1m archives for broad historical research-bar coverage, then use throttled `/fapi/v1/klines` only for small targeted tails such as newly listed symbols and the most recent unpublished daily gap. Direct 1m bars are sufficient for broad alpha discovery/refit screening at 30m+ source timeframes, but promotion to execution-quality live evidence still requires the existing raw-first/1s or paper/testnet fill/BBO/slippage telemetry gates. Real-money remains blocked.

## 2026-05-28 KST — Binance TradFi perpetual universe added for future monitoring/refits

Recorded the current Binance USD-M `TRADIFI_PERPETUAL` research universe without running any data collection. Source check: `Binance USD-M Futures /fapi/v1/exchangeInfo` at `2026-05-28T13:40:47Z`. The standard expanded research universe is now `68` compact USDT symbols: `10` core crypto plus `58` TradFi perpetual symbols. Future refresh/monitoring jobs should keep these symbols current and make them available to strategy discovery, but every new asset remains shadow/research-only until it passes the standard latest-8-week validation, final-refit, paper/testnet handoff, and live telemetry gates. Real-money flags remain hard-false.

Core crypto watch set (`10`): `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `TRXUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `TONUSDT`.

TradFi groups (`58` total):

- Commodity / metal / energy proxies (`8`): `XAUUSDT`, `XAGUSDT`, `XPTUSDT`, `XPDUSDT`, `COPPERUSDT`, `CLUSDT`, `BZUSDT`, `NATGASUSDT`.
- ETF / index-linked proxies (`5`): `QQQUSDT`, `SPYUSDT`, `EWYUSDT`, `EWJUSDT`, `SOXLUSDT`.
- Equity-linked proxies (`43`): `TSLAUSDT`, `INTCUSDT`, `HOODUSDT`, `MSTRUSDT`, `AMZNUSDT`, `CRCLUSDT`, `COINUSDT`, `PLTRUSDT`, `PAYPUSDT`, `METAUSDT`, `NVDAUSDT`, `GOOGLUSDT`, `AAPLUSDT`, `TSMUSDT`, `MUUSDT`, `SNDKUSDT`, `MSFTUSDT`, `AVGOUSDT`, `BABAUSDT`, `AMDUSDT`, `QCOMUSDT`, `USARUSDT`, `LITEUSDT`, `ORCLUSDT`, `DISUSDT`, `UBERUSDT`, `CSCOUSDT`, `HDUSDT`, `MRVLUSDT`, `CRWVUSDT`, `WMTUSDT`, `JPMUSDT`, `VUSDT`, `BRKBUSDT`, `FLNCUSDT`, `DRAMUSDT`, `RKLBUSDT`, `CBRSUSDT`, `NBISUSDT`, `WDCUSDT`, `ARMUSDT`, `BEUSDT`, `COHRUSDT`.
- Premarket / not-yet-standardized equity proxies (`2`): `SPCXUSDT`, `OPENAIUSDT`.

Implementation notes: added the side-effect-free canonical list in `src/lumina_quant/research_universe.py`; wired `scripts/research/build_multiasset_exchange_coverage_inventory.py` defaults to that expanded slashed universe; and expanded `scripts/research/run_alpha_zoo_multi_asset_monitoring_slate.py` asset-group fallbacks so future artifacts can classify all TradFi candidates. This did **not** modify the frozen selected live strategy universe or execute a backfill. `AlphaZooOptunaHybridLiveStrategy` remains the current paper/testnet strategy identity; the `profit_moonshot_alpha_zoo` path remains only a historical artifact namespace.

Estimated data-refresh effort if all 2025-to-current data is refreshed later under the 8GB memory budget: existing local storage already has `14` symbols with roughly `429M` 1s rows. A naive full-calendar `68`-symbol upper bound would be about `3.0B` 1s rows, but current Binance `onboardDate` evidence means the 58 TradFi perps only have about `241M` possible 1s rows from launch through `2026-05-28T13:40:47Z`. The practical full available universe is therefore roughly `684M` rows, with incremental missing coverage from the current local store around `250-300M` rows before retries and compaction overhead. For an update from the current repo data state, expect about `8-18h` staged under 8GB (`~2-6h` existing crypto/metals tails plus `~6-12h` TradFi batches if archives/API are cooperative). A cold full rebuild of all available 2025-to-current coverage should be scheduled as `18-36h`; `2-3d` is the realistic worst-case if missing archives, API backfill, retries, or low-parallelism constraints dominate.

No data collection was run for this documentation/code universe update.

---

## 2026-05-28 KST — Standard live-refit rule: latest 8-week validation + Optuna full-parameter final refit

Established the new system standard for live-facing Alpha Zoo refits: update local raw-first/committed market data first, reserve the latest **8 complete weeks** as validation, tune all exposed strategy-internal hybrid parameters with Optuna, select using train+validation evidence while fitting/learning on train only, then run a final refit on train+validation for the frozen paper/testnet runtime artifact. There is intentionally no locked-OOS/test set in this live final-refit mode; the final live artifact is frozen after refit and does not self-train online. Warmup remains a ratio inside the train window and is now part of the Optuna search space. Real-money flags remain hard-false.

Data refresh and split:

- Watch universe refreshed/compacted for `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, and `TRXUSDT` through `2026-05-28T10:59:59Z` in `data/market_parquet/market_ohlcv_1s/binance`.
- Standard split from latest complete 1h bars: train `2025-01-01T00:00:00Z` → `2026-04-02T10:00:00Z`; validation `2026-04-02T11:00:00Z` → `2026-05-28T10:00:00Z`; locked-OOS disabled for live final refit.
- Refresh/compaction evidence: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/standard_live_refit_20260528/`. Peak RSS stayed below the 8 GiB session budget: data refresh about `5.82 GiB` artifact peak / `6,045,100 KiB` `/usr/bin/time`; WAL compaction `1,565,284 KiB`.

Implementation/artifact updates:

- Added `src/lumina_quant/alpha_zoo/live_training_policy.py` to make the latest-8-week validation/final-refit split deterministic and reusable.
- Added `scripts/research/run_alpha_zoo_standard_live_refit.py` as the standard wrapper around the Optuna hybrid decision runner.
- Updated `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py` so Optuna covers every exposed `HybridParams` field: bias alpha/combine ratio/window, MAPE/short-vol windows, warmup ratio, max single weight, volatility-regime threshold/boost shape, min/max boost, and default-weight-ratio range/steps. Grid remains comparison-only.
- Removed the old hard-coded data end from the 30m+/HTF alpha source loaders so refreshed committed data can be used without editing dates.
- New standard-refit outputs: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_standard_live_refit_20260528/alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json|md`.

Result after embedded `10bps` round-trip friction proxy:

| Profile / run | Train | Validation | Validation MDD | RPT bps T/V | Trades T/V | Gross | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prior selected 2026-05-24 Optuna v3.5 | `+611.5025%` | `+138.3170%` | `18.9796%` | `83.39 / 79.17` | `3363 / 789` | `4.3646x` | Old split with locked-OOS report-only. |
| Standard live refit Optuna v3.5 selection/final-refit | `+3447.4699%` | `+38.0717%` | `7.4789%` | `368.22 / 36.77` | `4167 / 447` | `4.6902x` | Latest-8-week validation; no live test set. |

The selected profile remains `hybrid_v3_5_optuna_three_profile_blend`, now final-refit on train+validation. Final refit active weights are aggressive `91.4204%`, balanced `4.1881%`, growth `4.3915%`; asset gross notional fractions are approximately ETH `0.9573x`, SOL `2.0035x`, and TRX `1.7295x`. The recent validation is much lower than the old validation split but remains positive, above the 2% gate, below the 12% validation-MDD cap, and above the 10bps return-per-turnover threshold. The lower validation drawdown is an improvement; the very large train-vs-validation gap and aggressive sleeve concentration remain the main overfit/concentration risks to monitor in paper/testnet.

Verification for the standard-refit patch passed locally: changed-file Ruff pass; targeted live-training/hybrid tests `12 passed`; `compileall` over `src scripts tests`; repo-wide `ruff format --check .` and `ruff check .`; architecture check; docs verification; hardcoded-parameter audit `total=567 new=0 baselined=567`; `git diff --check`; and full `pytest -q` `1506 passed in 89.76s` with max RSS `2,736,260 KiB` (<8 GiB). The standard-refit runner itself completed with `/usr/bin/time` max RSS `6,324,184 KiB` and artifact peak `6175.96 MiB` (<8 GiB).

Safety conclusion: paper/testnet-only candidate evidence was refreshed, but real-money remains blocked. The latest artifacts still keep `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`; live execution must use the frozen artifact and collect realized BBO/fill/fee/slippage/reconciliation telemetry before any future real-money discussion.

---

## 2026-05-28 KST — Research-note path canonicalization

Renamed the confusing strategy/date-specific research-note paths into the stable `docs/research_note/` directory:

- Current cumulative note: `docs/research_note/research_note.md`.
- Global source/history ledger: `docs/research_note/research_history.md`.
- Archived state-distilled predecessor note: `docs/research_note/state_distilled.md`.

Future sessions should prepend new research diary entries here in latest-first order and keep strategy/project names inside the entry body instead of the filename. Large generated evidence remains under `var/reports/`; session checkpoints remain in `.omx/notepad.md`.

---

## 2026-05-27 KST — Live MARKET_WINDOW hot-path optimization after Rust conversion review

After converting the Alpha Zoo live state-signal state machines to optional Rust, the next live/paper/testnet bottleneck candidate was reviewed without opening real-money execution. User-stream order-state projection measured about `518k events/sec`, so it was not the priority. The live rolling `MARKET_WINDOW` path slowed as the window length grew because each internal event repeatedly re-normalized already-canonical Python rows and revalidated schema rows.

The fix keeps the public/runtime API in Python and does **not** weaken safety gates. `build_market_window_event` now has an internal trusted fast path (`bars_1s_already_normalized=True`) used only by canonical producers (`RollingWindowAggregator` and committed materialized snapshots). External payloads still go through full `normalize_bars_1s` validation. Binance live push now extends already-normalized rows directly, and rolling history pruning only scans the touched symbol, preserving stale-symbol carry-forward behavior while reducing per-tick overhead. A Rust FFI rewrite was not selected for this boundary because the final `MarketWindowEvent` remains a Python tuple/dict payload and the measured bottleneck was Python object validation, not numeric compute.

Benchmark artifact: `var/reports/native_acceleration_20260527/market_window_contract_benchmark_latest.json`. Local evidence: generic builder `0.0005508s/eval`, trusted builder `0.000001006s/eval` (`~547x` builder speedup), 5-symbol/300s rolling aggregation `16.5k ticks/sec`, max RSS `42.5MB`. This is a live data-path optimization only; `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` remain unchanged until paper/testnet fill/BBO/slippage/reconciliation telemetry proves the live assumptions.

---

## 2026-05-27 KST — Python-wrapped Rust live state-signal acceleration

After the operator clarified that useful Rust conversions should still be exposed through Python, the live Alpha Zoo state-machine kernels were moved behind an optional Rust backend without changing the public API. New backend: `native/rust_live_signals`; Python wrapper: `lumina_quant.alpha_zoo.native_live_signal_backend`; runtime control: `LQ_LIVE_SIGNAL_BACKEND=auto|python|rust` and `LQ_LIVE_SIGNAL_BACKEND_DLL`.

The accelerated functions are `debounced_state_signal` and `trailing_state_signal` in `lumina_quant.alpha_zoo.optuna_hybrid_signals`, which are pure deterministic loops used by `AlphaZooOptunaHybridLiveStrategy`. The Python wrapper remains the final interface and falls back to the original Python implementation when the Rust release library is unavailable. Explicit `rust` mode fails fast for diagnostics.

Local benchmark evidence is stored at `var/reports/native_acceleration_20260527/live_signal_backend_benchmark_latest.json`: 50,000 rows, 20 evaluations, Python `0.1349s/eval`, Rust `0.000544s/eval`, total speedup `247.80x`, exact debounced/trailing state-array parity (`max_abs_diff=0.0`). The benchmark max RSS was `97,200 KiB`, below the 8 GiB budget.

Scope boundary: exchange order submission, Binance/MT5/Polymarket HTTP/WebSocket clients, and raw network data collection were not moved in this pass because they are protocol/I/O-bound and safety/venue semantics dominate latency. Existing raw aggTrades→1s OHLCV and 1s WAL append acceleration remains under `native/rust_rawfirst`. Real-money safety is unchanged: paper/testnet-only artifacts still keep `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

---

## 2026-05-27 KST — Repo-wide format/Rust hygiene baseline

Follow-up cleanup normalized the whole tracked Python/Rust/code-hygiene surface after the live adapter split. This is a behavior-preserving cleanup pass: the selected `hybrid_v3_5_optuna_three_profile_blend` artifacts, paper/testnet-only live contract, limit-first order defaults, 10bps replay assumption, and hard real-money veto remain unchanged.

Changes made:

- Applied repo-wide `ruff format` to remove the previously known formatting drift outside the live-adapter patch.
- Applied `cargo fmt` to native Rust crates and added CI checks for `cargo fmt --check`, `cargo check`, and `cargo test` on `native/rust_metrics` and `native/rust_rawfirst`.
- Added `.gitattributes` to lock repository text hygiene to LF by default while preserving CRLF for Windows launchers (`*.bat`, `*.cmd`, `*.ps1`).
- Updated the hardcoded-parameter audit baseline after formatting moved source coordinates; audit result remains `total=567 new=0 baselined=567`.
- Added CI `ruff format --check .` so future pushes fail on format drift instead of relying on local-only checks.

Validation evidence for this pass: full pytest `1480 passed` with max RSS `2,746,164 KiB` (<8 GiB); Ruff format/check pass; compileall pass; docs verification pass; architecture check pass; hardcoded-parameter audit `new=0`; `uv lock --check` pass; `git diff --check` pass; native Rust format/check/tests pass; GPU auto and forced runtime smokes pass with CPU/GPU row parity.

---

## 2026-05-27 KST — Optuna hybrid live adapter cleanup/reproducibility lock

Follow-up cleanup refactored the selected `hybrid_v3_5_optuna_three_profile_blend` live path without changing the frozen research result or live safety posture. The previously monolithic `src/lumina_quant/alpha_zoo/optuna_hybrid_live_strategy.py` was split into:

- `src/lumina_quant/alpha_zoo/optuna_hybrid_config.py` for frozen Optuna/integer-leverage artifact loading, paper/testnet governance validation, selected profile/source sleeve reconstruction, and v3.5 allocator metadata.
- `src/lumina_quant/alpha_zoo/optuna_hybrid_signals.py` for completed-bar handling, debounced/trailing state rules, intrabar risk-frame helpers, and source-family signal math.
- `src/lumina_quant/alpha_zoo/optuna_hybrid_live_strategy.py` for runtime state, component intrabar guards, notional-fraction sizing metadata, and `SignalEvent` emission.

Behavior was locked before the split. `tests/test_alpha_zoo_optuna_hybrid_live_strategy.py` now asserts the selected artifact metrics and live decision contract directly: selected profile `hybrid_v3_5_optuna_three_profile_blend`, Optuna TPE v3.5, train `+611.5025%`, validation `+138.3170%`, locked-OOS report-only `+20.8319%`, validation MDD `18.9796%`, locked-OOS MDD `10.5735%`, RPT proxy `83.39/79.17/25.29bps`, trades `3363/789/362`, gross notional fraction `4.3645889x`, final profile weights aggressive `57.2699%`, balanced `7.9831%`, growth `8.0672%`, and train+validation average weights aggressive `78.1094%`, balanced `10.9140%`, growth `10.9767%`. The same tests lock the latest `paper_testnet_live_decision_latest.json` limit-first contract: `default_order_type=LMT`, `allow_market_orders=false`, `limit_price_mode=one_tick_worse`, one-tick offset, and side-aware limit entry/exit policies.

Governance is unchanged. The live adapter remains paper/testnet-only: `ready_for_paper=true`, `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`; locked-OOS is still gate/report-only and the 10bps round-trip cost/RPT threshold remains the primary assumption. This cleanup is a structure/reproducibility pass, not a new discovery or performance rerun.

---

## 2026-05-26 KST — Limit-first live execution hardening

Implemented the operator requirement that market orders remain optional and that live/paper execution defaults to limit orders. `live.default_order_type` now defaults to `LMT`, `live.allow_market_orders=false`, and `live.limit_price_mode=one_tick_worse`: BUY limits are priced one exchange tick above the reference and SELL limits one tick below it for fast bounded execution. `same_price` and `one_tick_better` remain configurable alternatives. The Portfolio signal path now applies this policy to entries, shorts, and reduce-only exits; `LiveTrader` risk-flatten orders use the same limit policy. `LiveExecutionHandler` rejects market parent orders when the live market-order guard is present and disabled.

Paper/testnet Binance USD-M protective algo orders were moved from default market-style `STOP_MARKET` / `TAKE_PROFIT_MARKET` to default conditional limit `STOP` / `TAKE_PROFIT`, with `triggerPrice`, side-aware one-tick-worse `price`, and `GTC` time-in-force. Market-style parent or protective orders remain possible only by explicit config opt-in (`live.allow_market_orders=true`, plus the relevant order-style setting) and still do not approve real-money. Refreshed `paper_testnet_live_decision_latest.json|md` records `limit_order_contract`, `protective_order_style=limit`, and hard-false real-money flags.

Research interpretation is unchanged: backtest results still embed the 10bps round-trip cost proxy, but one-tick-worse limits are marketable limits and may still incur taker fees; paper/testnet must record realized `fee_bps`, BBO/spread, limit reference/price, fill latency, partial fills, cancels, and all-in costs before any real-money review. `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` remain invariant.

---

## 2026-05-25 KST — Paper/testnet exchange-side protective orders and asset applicability

One additional live-readiness gap was closed for paper/testnet: after an entry order is confirmed filled, `LiveExecutionHandler` can submit Binance USD-M Futures conditional algo protective orders through the request gateway. The current supported paper/testnet protective types are conditional limit `STOP` and `TAKE_PROFIT`, routed through the repo-native Binance Futures adapter to `POST /fapi/v1/algoOrder` with side-aware one-tick-worse prices. The order uses the same parent component quantity and the signal/component `position_side`; it intentionally remains paper/testnet-only. Market-style `STOP_MARKET` / `TAKE_PROFIT_MARKET` is retained only as an explicit market opt-in path. Real mode still fails closed unless a separate artifact explicitly approves exchange-side protective order handling and measured paper/testnet telemetry.

Asset applicability was checked beyond the dominant SOL sleeve. The intrabar guard/protective-order logic is symbol-generic and now has tests for the selected frozen source assets `ETHUSDT`, `SOLUSDT`, and `TRXUSDT`, including both long and short protective directions. This does not add new alpha assets or change the frozen hybrid portfolio; it verifies that the live protective machinery is not hard-coded to one asset family.

Remaining live limitations after this pass: no actual exchange paper/testnet fill sample has been collected yet, exact queue priority is still unknowable and proxy-only, and real-money remains vetoed until at least two weeks of fill/slippage/reconciliation telemetry confirms the 10bps cost and notional-parity assumptions.

Local verification for this follow-up passed: `ruff check .`, architecture check, `compileall`, hardcoded-parameter audit (`new=0`), `git diff --check`, and full `pytest -q`; full pytest was `1460 passed` with max RSS `2,752,152 KiB`, below the 8 GiB limit.

Payload-hardening update: the Binance conditional algo order adapter now uses a documented-field allowlist for `/fapi/v1/algoOrder`; internal parent/protection telemetry fields are not forwarded to the exchange request payload. Targeted exchange/state-machine/live-strategy regression tests passed (`25 passed`).

Final validation after payload hardening: `ruff check .`, architecture check, `compileall`, hardcoded-parameter audit (`new=0`), `git diff --check`, and full `pytest -q` passed; full pytest was `1460 passed` with max RSS `2,724,568 KiB`, below the 8 GiB limit.

---

## 2026-05-25 KST — Intrabar protective guard added for Alpha Zoo Optuna hybrid paper/testnet

The prior live-readiness caveat about intrabar exits and microstructure was narrowed. The adapter still uses completed `1h/2h/4h` bars for alpha decisions, but it now attaches a paper/local-simulation intrabar protective contract to entry signals. Each entry signal includes `component_id`, `stop_loss`, optional chandelier-style `trailing_percent`, `intrabar_protection`, and `microstructure_telemetry_required` metadata. The strategy maintains component-level intrabar guards and can emit a component `EXIT` signal when a `MARKET` event breaches the guard. The risk frame prefers `1m`, then `5m`, then source-timeframe fallback; stops are ATR/cost-floor based and are intended as a conservative paper/testnet risk overlay, not a train+validation optimizer surface.

This does **not** change the real-money conclusion. Real exchange-side protective orders remain unapproved until exchange-specific STOP/TAKE_PROFIT order support is wired and observed in paper/testnet. Exact queue priority remains impossible to know from exchange APIs; monitoring must use BBO/depth/fill-latency/partial-fill proxies. The refreshed `paper_testnet_live_decision_latest.json|md` records the `intrabar_protection_contract` and `microstructure_telemetry_contract`. Real-money flags remain false and the artifact veto stays active.

Local verification after the guard update: `ruff check .`, architecture check, `compileall`, hardcoded-parameter audit (`new=0`), `git diff --check`, and full `pytest -q` all passed; full pytest was `1456 passed` with max RSS `2,933,144 KiB`, below the 8 GiB session limit.

---

## 2026-05-25 KST — Live adapter limitation audit and paper/testnet blockers

After the paper/testnet live adapter was implemented, the live-readiness interpretation was tightened: the adapter is live-compatible for paper/testnet monitoring, but real-money startup remains intentionally blocked. The refreshed `paper_testnet_live_decision_latest.json|md` now records explicit `real_money_blockers`, `known_limitations`, and `paper_testnet_validation_requirements` alongside the existing `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` flags.

Current real-money blockers:

- The governing artifacts are still paper/testnet-only: `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`; readiness policy should keep `artifact_real_money_veto=true` for real mode.
- There is no exchange paper/testnet fill telemetry yet. Realized BBO spread, fees, slippage, rejects, timeouts, cancels, partial fills, position reconciliation drift, and stale-data recovery must be observed before any real-money review.
- The `10bps` assumption is a replay/gate round-trip friction proxy, not a live measured all-in cost. Paper/testnet monitoring must show realized all-in costs remain compatible with that assumption.
- The decision artifact deliberately keeps global `target_allocation=0.0` fail-closed. Live sizing depends on `SignalEvent.metadata.target_allocation`; paper/testnet must verify signal metadata to submitted-order notional parity.

Known disadvantages and research risks:

- The selected `hybrid_v3_5_optuna_three_profile_blend` is dominated by the aggressive source profile. It is a risk-managed allocator over correlated source profiles, not a clearly independent new alpha sleeve.
- Validation MDD `18.9796%` is near the relaxed 20% label and above the strict 12% promotion cap; locked-OOS report-only MDD is `10.5735%`.
- Train return `+611.5025%` versus validation `+138.3170%` indicates strong train dominance and potential optimizer overfit despite positive validation and locked-OOS report-only results.
- Source universe breadth is still limited in the promoted live adapter: the frozen sleeves are concentrated in SOL/ETH/TRX with BTC/BNB mainly as reference/watch symbols; broader assets remain shadow/data-extension work until OOS coverage and live telemetry exist.
- The live adapter evaluates completed `1h/2h/4h` bars only, so it does not model intrabar exits, queue priority, funding/fee drift, liquidation-engine edge cases, or exchange microstructure timing.
- Frozen-artifact replay avoids online learning and locked-OOS leakage, but regime drift or stale artifacts require a new train+validation-first research cycle rather than live self-tuning.

Paper/testnet evidence required next: realized BBO/all-in round-trip cost by symbol and timeframe, reject/timeout/cancel/partial-fill rates, signal-metadata-to-order-notional parity, position reconciliation drift, stale-data block/recovery behavior, liquidation-inclusive MDD/account wipeout telemetry, and at least two continuous weeks of observation before any future real-money discussion. Until then the correct conclusion remains: paper/testnet launch is allowed for evidence gathering; real-money launch is prohibited.

---

## 2026-05-25 KST — Optuna hybrid paper/testnet live adapter implemented

Implemented a live-ready but real-money-vetoed adapter for the selected `hybrid_v3_5_optuna_three_profile_blend` from `alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524`. The runtime class is `AlphaZooOptunaHybridLiveStrategy`, registered as `live_opt_in`; its implementation lives in `src/lumina_quant/alpha_zoo/optuna_hybrid_live_strategy.py` with a thin strategy-registry wrapper at `src/lumina_quant/strategies/alpha_zoo_optuna_hybrid_live.py`.

Key live-readiness choices:

- Source universe is reconstructed from the frozen Optuna and integer-leverage artifacts, not a hand-picked allowlist. The exact six sleeves are SOL 1h debounced short-only, SOL 1h debounced long/short, SOL 1h shorter-hold debounced short-only, SOL 2h relative-strength chandelier breakout, ETH/BTC 2h residual reclaim, and TRX 4h volatility-adjusted trend.
- Runtime evaluates completed bars only and drops the active working bar. Required timeframes are `1h`, `2h`, and `4h`; watch symbols are `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, and `TRX/USDT`.
- The v3.5 allocator is frozen from the artifact; there is no Optuna runtime dependency and no locked-OOS learning. locked-OOS remains gate/report-only.
- Live sizing uses `notional_fraction` and each signal emits `metadata.target_allocation = source_allocation_fraction * sum(final_profile_weight * integer_leverage)`. The decision artifact keeps global `target_allocation=0.0` so missing signal metadata fails closed. Generated risk caps are equity-scaled: max order notional `1.247444x`, max symbol exposure `1.427227x`, and total notional/margin `3.520744x`.
- The adapter and preflight policy keep the real-money veto: `paper_testnet_only=true`, `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`. Real-mode preflight is blocked by artifact veto while paper/testnet can pass when other operational checks are healthy.
- Strategy logic has a no-calendar/date-rule regression check; the paper/testnet handoff records the same `10bps` round-trip cost and RPT threshold assumption as the research artifacts.

New handoff writer: `scripts/ops/write_alpha_zoo_optuna_hybrid_live_decision.py`. Latest outputs:

- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/paper_testnet_live_decision_latest.md`

Operator runbook updated at `docs/live-readiness/04-paper-trading-runbook.md` with the `--check-only` and writer commands plus the sizing/real-veto invariants. Verification on 2026-05-25 KST passed: targeted live adapter/readiness/start-live tests `32 passed`; post-architecture-fix full pytest `1455 passed in 102.70s` with max RSS `2,946,288 KiB` (<8 GiB); `ruff check .`; `uv run python scripts/check_architecture.py`; `python3 -m compileall -q src scripts tests`; hardcoded-parameter audit `total=567 new=0 baselined=567`; `git diff --check`; and `git diff --cached --check`. Real-money execution remains prohibited.

---

## 2026-05-24 KST — Optuna v3.5/v3.6 correction for the integer-leverage hybrid

The previous three-profile hybrid decision used a coarse `5%` grid and therefore was **not** equivalent to the repository's hybrid v3.5/v3.6 Optuna workflow. This was corrected with `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py` and `tests/test_alpha_zoo_integer_leverage_optuna_hybrid_decision.py`.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524/`. Latest outputs include `alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json|md`, comparison CSV, top-trials CSV, methodology note, paper/testnet handoff, hardcoded audit report, and `verification_summary_latest.md`.

Method correction:

- Source universe remains the same three paper/testnet integer-leverage profiles: `balanced_mdd12_gross5`, `growth_mdd20_gross8`, and `aggressive_mdd30_gross10_shadow`.
- The prior grid hybrid is retained only as a comparison row, not the optimizer-selected decision surface.
- Optuna uses `TPESampler`, `240` trials per version, and the same parameter family as `run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`: bias alpha/combine ratio, max single weight, MAPE/rolling score window, bias window, and short-volatility window.
- v3.5 mapping: warmup-learned default profile + rolling return/error weights + high-volatility boost + bias/exposure dampening.
- v3.6 mapping: v3.5 mechanics plus online adaptive default-profile refresh from rolling score evidence.
- Objective/learning/selection use train+validation only. locked-OOS is attached after frozen Optuna params and remains gate/report-only; all locked-OOS discovery/selection/objective/pruning/parameter-fitting flags are false.

Result after embedded `10bps` round-trip friction proxy:

| Profile | Optimizer | Train | Validation | locked-OOS report-only | Validation MDD | OOS MDD | RPT bps T/V/OOS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced_mdd12_gross5` | source | `+74.6685%` | `+33.2153%` | `+5.5300%` | `11.6134%` | `7.2003%` | `30.91/57.02/22.21` |
| `growth_mdd20_gross8` | source | `+262.3353%` | `+71.6291%` | `+23.3695%` | `19.9983%` | `9.2371%` | `32.16/37.72/23.35` |
| `aggressive_mdd30_gross10_shadow` | source | `+438.4462%` | `+117.4976%` | `+27.5772%` | `29.4044%` | `12.3630%` | `38.81/44.16/21.87` |
| `hybrid_mdd20_three_profile_blend` | old grid baseline | `+262.3642%` | `+72.5692%` | `+21.2977%` | `19.9330%` | `9.0718%` | `33.78/39.96/22.97` |
| `hybrid_v3_5_optuna_three_profile_blend` | Optuna v3.5 | `+611.5025%` | `+138.3170%` | `+20.8319%` | `18.9796%` | `10.5735%` | `83.39/79.17/25.29` |
| `hybrid_v3_6_optuna_three_profile_blend` | Optuna v3.6 | `+296.4869%` | `+85.9099%` | `+11.5273%` | `15.7399%` | `8.4448%` | `51.05/62.78/17.91` |

Selected Optuna hybrid: `hybrid_v3_5_optuna_three_profile_blend`. Average train+validation profile weights are approximately aggressive `78.11%`, balanced `10.91%`, and growth `10.98%`; final active weights are lower because the v3.5 exposure-dampening rule can hold cash. It passes the paper/testnet candidate gates, but remains non-real-money: `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`. The paper/testnet handoff requires realized BBO/fill/all-in cost telemetry, replay/live notional parity, liquidation-inclusive MDD, account wipeout, and margin-buffer monitoring before any future review.

Verification passed locally: artifact invariant check; runner max RSS `6,357,368 KiB` (<8 GiB), elapsed `32:37.97`; targeted tests `13 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded audit `total=567 new=0`; diff checks; full pytest `1444 passed in 76.49s` with max RSS `2,723,060 KiB` (<8 GiB).

---

## 2026-05-24 KST — Three-profile integer-leverage hybrid decision

Added `scripts/research/run_alpha_zoo_integer_leverage_hybrid_decision.py` and `tests/test_alpha_zoo_integer_leverage_hybrid_decision.py` to answer whether the three current integer-leverage paper/testnet profiles can be blended into a hybrid candidate. The runner consumes the frozen `alpha_zoo_corr_integer_leverage_portfolio_latest.json`, reconstructs each source profile's 10bps-costed PnL stream from fixed strategy position states, and searches hybrid weights on a 5% grid with each source profile weight at least 10%. The hybrid selection objective and gates use train+validation only; locked-OOS is attached only after weights are frozen as report/gate evidence. All outputs remain paper/testnet-only with `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`.

Selected hybrid: `hybrid_mdd20_three_profile_blend`, weights `balanced_mdd12_gross5=15%`, `growth_mdd20_gross8=70%`, `aggressive_mdd30_gross10_shadow=15%`. It is a relaxed paper/testnet candidate, not a strict 12% validation-MDD promotion. Metrics after the embedded 10bps round-trip friction proxy:

| Profile | Gross | Train | Validation | locked-OOS report-only | Validation MDD | OOS MDD | RPT bps T/V/OOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced_mdd12_gross5` | `1.00x` | `+74.6685%` | `+33.2153%` | `+5.5300%` | `11.6134%` | `7.2003%` | `30.91/57.02/22.21` |
| `growth_mdd20_gross8` | `3.90x` | `+262.3353%` | `+71.6291%` | `+23.3695%` | `19.9983%` | `9.2371%` | `32.16/37.72/23.35` |
| `aggressive_mdd30_gross10_shadow` | `4.90x` | `+438.4462%` | `+117.4976%` | `+27.5772%` | `29.4044%` | `12.3630%` | `38.81/44.16/21.87` |
| `hybrid_mdd20_three_profile_blend` | `3.615x` | `+262.3642%` | `+72.5692%` | `+21.2977%` | `19.9330%` | `9.0718%` | `33.78/39.96/22.97` |

Interpretation: the hybrid slightly improves validation return and validation MDD versus the growth profile while keeping the MDD~20 risk label and preserving positive locked-OOS/RPT/liquidation gates. It does not dominate growth on locked-OOS return (`+21.30%` vs `+23.37%`) and it does not match aggressive return, because the aggressive profile's drawdown is deliberately diluted. Train+validation PnL correlations show why: balanced-growth is moderately correlated (`0.5790`), but balanced-aggressive (`0.8493`) and growth-aggressive (`0.8680`) are high, so the hybrid is a risk-return compromise rather than a new independent alpha sleeve.

Verification passed locally. Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_integer_leverage_hybrid_decision_20260524/`; latest JSON/Markdown, timestamped JSON, comparison CSV, weight-candidate CSV, methodology note, runner log, and `verification_summary_latest.md` were written. Verification log `local_verification_hybrid_decision_20260524T132911Z.log`: artifact invariant pass, targeted tests `9 passed`, `ruff check .`, `compileall`, hardcoded audit `total=567 new=0`, diff checks, and full pytest `1440 passed in 68.46s`. Runner max RSS `6,505,524 KiB` and full pytest max RSS `2,771,296 KiB`, both below 8 GiB.

---

## 2026-05-24 KST — Integer-leverage paper candidate set and strategy-integrity review

Operator feedback corrected the previous framing: the `growth_mdd20_gross8` profile with validation MDD near `19.9983%` is sufficiently strong for paper/testnet candidate observation, and the aggressive profile should also remain in the candidate set. The integer-leverage portfolio runner now distinguishes strict promotion from paper/testnet candidacy:

- `balanced_mdd12_gross5` — strict paper/testnet promotion candidate. Integer asset leverage map `SOLUSDT=2`, `TRXUSDT=1`, gross notional `1.00x`; train/validation/locked-OOS report-only returns `+74.6685%/+33.2153%/+5.5300%`; validation MDD `11.6134%`; locked-OOS MDD `7.2003%`; trade events `945/229/100`; return-per-turnover proxy `30.91/57.02/22.21bps`; liquidation/account-wipeout `0/0`.
- `growth_mdd20_gross8` — relaxed paper/testnet candidate, not a strict 12% MDD promotion. Integer map `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `3.90x`; train/validation/locked-OOS `+262.3353%/+71.6291%/+23.3695%`; validation MDD `19.9983%`; locked-OOS MDD `9.2371%`; trade events `879/200/104`; RPT proxy `32.16/37.72/23.35bps`; liquidation/account-wipeout `0/0`.
- `aggressive_mdd30_gross10_shadow` — relaxed paper/testnet candidate, still explicitly not strict promotion. Same integer asset map `ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `4.90x`; train/validation/locked-OOS `+438.4462%/+117.4976%/+27.5772%`; validation MDD `29.4044%`; locked-OOS MDD `12.3630%`; trade events `1539/360/158`; RPT proxy `38.81/44.16/21.87bps`; liquidation/account-wipeout `0/0`.

Theoretical/implementation review was added as `strategy_integrity_review_latest.json|md` in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_corr_integer_leverage_portfolio_20260524/`. Review status is `pass`. The active sleeves are momentum/trend/residual state rules, not calendar/date rules: SOL debounced momentum hysteresis, SOL relative-strength chandelier breakout, ETH/BTC relative residual reclaim, and TRX volatility-adjusted trend persistence. This is aligned with established momentum/trend-following evidence (Moskowitz, Ooi & Pedersen, 2012, *Journal of Financial Economics*, `https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf`), crypto trend-following/regime-volatility research context (`https://arxiv.org/abs/2602.11708`), volatility-scaled momentum risk-control literature (`https://www.sciencedirect.com/science/article/abs/pii/S0275531917308322`), and Wilder-style ADX/trend-strength filtering (`https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx`). These references justify the hypothesis class only; promotion still depends on repository backtests, paper/testnet fills, and gates.

Hard anti-calendar/hardcode checks were recorded:

- Candidate model IDs for the review are derived from the frozen PnL-correlation decision artifact and the profile rows, not from a hardcoded model allowlist.
- Metadata-level calendar/date token check found no forbidden hits among active paper candidate sleeves.
- Source-code calendar-feature grep across `run_alpha_zoo_debounced_efficiency_repair_discovery.py`, `run_alpha_zoo_30m_plus_alpha_booster_discovery.py`, `run_alpha_zoo_30m_plus_alpha_feedback_discovery.py`, and `run_alpha_zoo_asset_diverse_strategy_discovery.py` found no `dt.day`, `dt.weekday`, `dt.dayofweek`, `dt.month`, `dt.hour`, `day_of_week`, `weekday`, `month_end`, `hour_of_day`, or `time_of_day` strategy features.
- Integer-leverage runner hardcoded-ID grep found none of the selected model IDs embedded in the code.

Cost/slippage interpretation is now explicit. The runner imports and enforces `PRIMARY_ROUND_TRIP_COST_BPS = 10.0`; per transition, it charges `round_trip_cost_bps / 2` per side, so an entry+exit round trip is modeled as `10bps` all-in backtest friction. The promotion/monitoring RPT proxy threshold is also `avg_bbo_spread_bps_assumption 2.0 * multiplier 5.0 = 10.0bps`. This confirms the 10bps assumption is baked into the replay and gates, but it is **not** a live fill-derived slippage measurement; paper/testnet monitoring must still record realized BBO spread, fee/slippage/all-in round-trip cost, timeout/cancel/partial-fill hygiene, replay/live notional parity, liquidation-inclusive MDD, and account wipeout before any future real-money review.

Governance remains unchanged: all outputs keep `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`; locked-OOS is gate/report-only after train+validation freeze and all locked-OOS discovery/selection/objective/pruning/parameter-fitting flags are false. The existing four `quality_single_pair` baseline lanes remain preserved separately.

Verification passed after this correction. Artifact regeneration wrote timestamped JSON `alpha_zoo_corr_integer_leverage_portfolio_20260524T124248Z.json`, latest JSON/Markdown, profile CSV, handoff/preflight JSON/Markdown, `strategy_integrity_review_latest.json|md`, and `verification_summary_latest.md`. Local verification log `local_verification_integer_leverage_20260524T124342Z.log` records: artifact invariant pass; calendar-feature grep pass; hardcoded model-ID grep pass; targeted tests `14 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `total=567 new=0`; diff checks; full pytest `1436 passed in 67.59s`. Runner max RSS was `6,396,380 KiB`; full pytest max RSS was `2,745,132 KiB`, both below the 8 GiB cap.

---

## 2026-05-24 KST — Corr-diversified integer leverage portfolio improves strict paper/testnet return

Following the PnL-correlation matrix decision, a new integer-leverage portfolio pass was added in `scripts/research/run_alpha_zoo_corr_integer_leverage_portfolio.py`. The method starts from the frozen 11-row corr-diversified slate, not the full duplicate 136-row book, and searches train+validation-only greedy subsets with integer per-asset leverage maps. locked-OOS remains post-freeze gate/report-only and is not used for discovery, objective, pruning, parameter fitting, ranking, subset selection, or leverage selection.

The strict promotion profile `balanced_mdd12_gross5` passes all hard paper/testnet criteria with active leverage map `SOLUSDT=2`, `TRXUSDT=1` and gross notional `1.00x`. Selected sleeves: SOL 1h debounced short-only, SOL 1h debounced long/short, SOL 2h relative-strength chandelier breakout, and TRX 4h volatility-adjusted trend. Metrics: train `+74.6685%`, validation `+33.2153%`, locked-OOS report-only `+5.5300%`; train/validation/OOS trade events `945/229/100`; validation MDD `11.6134%`; locked-OOS MDD `7.2003%`; RPT proxy `30.91/57.02/22.21bps`; liquidation/account-wipeout `0/0`. This is a material improvement over the previous equal-weight corr slate validation `+8.9654%` / locked-OOS `+2.3304%` while retaining the strict validation-MDD and 10bps efficiency gates.

For context only, relaxed shadow profiles show much higher returns but are not strict promotions: growth shadow (`ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `3.90x`) has train/validation/OOS `+262.3353%/+71.6291%/+23.3695%` with validation MDD `19.9983%`; aggressive shadow (`ETHUSDT=8`, `SOLUSDT=4`, `TRXUSDT=12`, gross `4.90x`) has `+438.4462%/+117.4976%/+27.5772%` with validation MDD `29.4044%`. They are paper/testnet shadow-review only under the existing strict `12%` validation-MDD promotion rule.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_corr_integer_leverage_portfolio_20260524/`; includes latest JSON/MD, decision CSV, paper/testnet handoff, preflight JSON/MD, runner log, and verification summary. Governance remains unchanged: `ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`, paper/testnet-only decision/preflight/monitoring handoff, replay/live notional parity recorded, primary cost/RPT threshold `10bps`, and no calendar/date hack. Verification passed locally: artifact invariant check, targeted tests `13 passed`, `ruff`, `compileall`, hardcoded audit `new=0`, diff checks, and full pytest `1435 passed`.

---

## 2026-05-24 KST — PnL-correlation matrix decision: reject all-in, keep corr-diversified paper slate

Added `scripts/research/run_alpha_zoo_pnl_correlation_decision.py` and `tests/test_alpha_zoo_pnl_correlation_decision.py` to evaluate the current multi-asset paper/testnet monitoring book by actual replayed strategy PnL correlation rather than headline aggregate returns alone. The runner consumes `alpha_zoo_multi_asset_monitoring_slate_latest.json`, replays the already-discovered candidate definitions to capture per-bar PnL return streams, and writes correlation matrices plus a correlation-aware decision record under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_pnl_correlation_decision_20260524/`.

Decision method recorded in `pnl_correlation_decision_methodology_latest.md`:

- Candidate universe for selection: `136` rows with `monitoring_status=paper_testnet_monitor` from the multi-asset slate.
- Additional report-only context: top `30` strict shadow representatives; these do not become promotable through this correlation pass.
- PnL stream: per-bar fractional strategy return after the primary `10bps` round-trip cost and each candidate's native `notional_fraction`.
- Correlation matrix: Pearson correlation of PnL streams, aligned on datetime across mixed `30m/1h/2h/4h/6h` bars; missing bars are filled with zero PnL before correlation.
- Selection surface: train+validation PnL correlation only, with validation-only correlation as a guard. locked-OOS correlation is saved only as report/monitoring evidence after the train+validation freeze.
- Greedy rule: sort by `monitoring_score_train_validation_only`, then accept a candidate only if its max absolute train+validation corr to already selected candidates is `<=0.70` and its max absolute validation corr is `<=0.75`.
- High-correlation clusters are treated as one alpha sleeve; duplicate parameter/sizing rows are rejected as correlated duplicates, not as bad standalone alphas.

The all-paper book is not suitable as an unscaled portfolio. The replay captured all `136/136` paper candidates, and the train+validation matrix has `9,180` pairs with mean absolute corr `0.3074`, max corr `1.0`, `1,285` pairs with `|corr|>=0.85`, `14` high-corr clusters, and largest cluster size `41`. Unscaled adoption would sum to `49.80x` gross notional; the unscaled portfolio replay is unstable (`-100%` validation return and `-35.0490%` locked-OOS report-only return), so the decision is `reject_unscaled_all_in_due_to_duplicate_pnl_clusters_and_excess_gross_notional`.

The corr-diversified paper/testnet-only subset has `11` candidates and `4.50x` unscaled gross notional. Equal-weight comparison (for research normalization, not execution sizing) is validation `+8.9654%` and locked-OOS report-only `+2.3304%`, versus all-paper equal-weight validation `+8.4824%` and locked-OOS `+2.5290%`. The selected slate keeps the main independent sleeves: SOL 1h debounced short-only, SOL 2h relative-strength chandelier, SOL 1h debounced long/short, a lower-correlation SOL 1h debounced short-only variant, ETH 1h/2h debounced variants, SOL 6h volatility-adjusted trend, ETH/BTC residual reclaim, SOL 4h debounced, TRX 4h volatility-adjusted trend, and an ETH 1h short-only debounced variant. Full IDs and corr matrix are in `selected_corr_diversified_candidates_latest.csv` and `selected_pnl_corr_train_validation_latest.csv`.

Governance remains unchanged: this is paper/testnet-only research; every artifact keeps `ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false`. locked-OOS is not used for discovery, selection, objective, pruning, parameter fitting, or correlation acceptance; it is report-only after the train+validation decision. The four existing `quality_single_pair` paper/testnet baseline lanes remain preserved separately.

Runner verification evidence: artifact generation succeeded with max RSS `6,481,872 KiB` (<8 GiB), captured paper PnL count `136`, selected count `11`, missing PnL `0`, and locked-OOS selection flag false. Targeted tests for the new runner passed (`4 passed`) before full verification.

Final verification for the PnL-correlation decision passed on 2026-05-24 KST. Runner max RSS was `6,481,872 KiB` (<8 GiB). Artifact invariants passed (`136/136` paper PnL streams captured, selected `11`, missing `0`, all real-money flags false, all locked-OOS selection/discovery/objective/pruning/fitting flags false). Targeted tests passed (`11 passed`), `ruff check .` passed, `python -m compileall -q src scripts tests` passed, hardcoded-parameter audit reported `total=567 new=0 baselined=567`, diff checks passed, and full pytest passed with `1431 passed in 74.52s` and max RSS `2,816,620 KiB` (<8 GiB). Verification summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_pnl_correlation_decision_20260524/verification_summary_latest.md`.

A time-sensitive test fixture was corrected during verification: frozen paper-forward 10bps artifacts used a `10,000` minute stale threshold against the 2026-05-17 refresh snapshot, which became stale on 2026-05-24. `scripts/research/run_alpha_zoo_7x_paper_forward_preflight.py` now uses a named 30-day stale override for historical paper/testnet handoff generation while still forcing `ready_for_real=false` and `real_money_execution=false`; this keeps research artifact tests deterministic without opening real-money readiness.

---

## 2026-05-24 KST — Multi-asset monitoring slate combines all recent paper/shadow Alpha Zoo lanes

Added `scripts/research/run_alpha_zoo_multi_asset_monitoring_slate.py` and `tests/test_alpha_zoo_multi_asset_monitoring_slate.py` to turn the recent frozen discovery artifacts into one paper/testnet-only monitoring book instead of watching only one or two headline candidates. The slate reads the debounced efficiency repair, 30m+ feedback, 30m+ booster, and asset-diverse strategy artifacts and normalizes paper candidates, top rows, and shadow shortlists by source artifact, symbol, asset group, timeframe, and family.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_multi_asset_monitoring_slate_20260524/`. Latest outputs include `multi_asset_monitoring_slate_latest.json|md`, timestamped JSON, `multi_asset_monitoring_rows_latest.csv`, `asset_monitoring_matrix_latest.csv`, `paper_monitoring_handoff_latest.json|md`, `no_real_money_guard_latest.json`, `artifact_generation_validation_latest.log`, `runner_time_latest.log`, `targeted_pytest_time_latest.log`, `full_pytest_time_latest.log`, and `verification_summary_latest.md`.

Result: `2488` normalized candidate rows and a `14`-symbol monitoring matrix. The paper/testnet monitoring book now contains `136` strict paper candidates across `ETHUSDT` (`15`), `SOLUSDT` (`119`), and `TRXUSDT` (`2`). It also preserves non-promotional monitoring coverage for the rest of the source universe: `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `DOGEUSDT`, `TONUSDT`, `XAGUSDT`, `XAUUSDT`, `XPDUSDT`, `XPTUSDT`, and `XRPUSDT`. `AVAXUSDT`, `DOGEUSDT`, `XAUUSDT`, and `XRPUSDT` are coverage-blocked shadows because locked-OOS coverage/trade evidence is insufficient; `BNBUSDT` and `BTCUSDT` remain shadow-watchlist only; source-coverage-only symbols stay in the matrix so future data extension is visible.

Best paper/testnet monitor rows by train+validation-only priority remain: SOLUSDT 1h debounced momentum hysteresis efficiency repair (`+28.2198%` train, `+16.9294%` validation, `+2.4704%` locked-OOS report-only, RPT `24.69/59.72/21.11bps`), SOLUSDT 2h relative-strength chandelier breakout (`+37.4602%`, `+16.0919%`, `+4.2373%`, RPT `30.96/60.72/31.39bps`), ETHUSDT debounced/residual paper candidates, and TRXUSDT 4h volatility-adjusted trend candidates. The slate is a monitoring synthesis, not a new discovery selection: locked-OOS is not used for discovery, selection, objective, pruning, parameter fitting, or monitoring-score priority; it is gate/report-only after the source artifacts' train+validation freezes.

Governance remains unchanged: real-money execution is prohibited (`ready_for_real=false`, `real_money_execution=false`, `real_execution_allowed=false`), paper/testnet-only monitoring is allowed only after preflight, primary cost/RPT threshold remains `10bps`, replay/live notional parity must be recorded, and every monitored candidate must record realized BBO spread, fee/slippage/all-in round-trip cost, liquidation-inclusive MDD, and account wipeout. The four existing `quality_single_pair` baseline lanes remain preserved unchanged.

Verification passed locally on 2026-05-24 KST: runner max RSS `139092 KiB`; artifact invariant check passed; targeted monitoring/debounced/feedback/booster/asset-diverse tests `27 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; diff checks; full pytest `1427 passed in 65.82s` with max RSS `2759596 KiB`, all below the 8 GiB session cap.

---

## 2026-05-23 KST — Asset-diverse strategy discovery expands beyond one asset lane

A follow-up asset-diverse Alpha Zoo pass was added via `scripts/research/run_alpha_zoo_asset_diverse_strategy_discovery.py` with tests in `tests/test_alpha_zoo_asset_diverse_strategy_discovery.py`. The goal was to stop concentrating only on the latest SOL/ETH lane and search cross-asset-conditioned, single-symbol strategies across the locally available Binance universe while preserving the same strict paper/testnet-only controls.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_asset_diverse_strategy_discovery_20260523/`. Latest files include `alpha_zoo_asset_diverse_strategy_discovery_latest.json`, latest Markdown, timestamped JSON, candidate/decision/shadow CSVs, `paper_testnet_handoff_latest.json|md`, `no_promotion_shadow_shortlist_latest.json`, `runner_time_latest.log`, `full_pytest_time_latest.log`, and `verification_summary_latest.md`.

The run evaluated `97,560` candidates across three strategy families: cross-asset rank chandelier breakout, relative residual reclaim, and breadth-regime pullback/reclaim. Promotion-eligible locked-OOS universe was `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, and `TRXUSDT`; shadow-only probes widened coverage to `ADAUSDT`, `AVAXUSDT`, `DOGEUSDT`, `TONUSDT`, `XRPUSDT`, and precious-metal proxy symbols `XAUUSDT`, `XAGUSDT`, `XPTUSDT`, `XPDUSDT`. Shadow symbols are explicitly non-promotable until they have the required locked-OOS coverage.

Result: `4` strict paper/testnet-only candidates, all ETHUSDT 2h `relative_residual_reclaim` vs BTCUSDT. Best row `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5`: train `+16.8301%`, validation `+4.7367%`, locked-OOS `+4.8120%`, validation MDD `4.89%`, trades `184/32/26`, return-per-turnover proxy `18.29/29.60/37.02bps`, locked-OOS liquidation/account-wipeout `0/0`. It is weaker than the prior SOL 2h booster headline return, but diversifies the live paper/testnet research book into an ETH/BTC relative-value state rule with positive locked-OOS and stronger OOS RPT.

Best non-promotable shadows came from XRPUSDT 1h cross-asset rank chandelier variants with train roughly `+36%` to `+55%`, validation roughly `+29%` to `+32%`, and train/validation RPT above 10bps. They are rejected because local XRPUSDT has no locked-OOS bars after 2026-03 and because shadow-only symbols cannot promote. This is useful evidence for the next data-extension step but not a paper handoff.

Governance remains unchanged: `ready_for_real=false`, `real_money_execution=false`, 10bps primary cost, return-per-turnover threshold `avg BBO spread * 5 = 10bps`, replay/live notional parity recorded, no calendar/date hack, single-symbol position state rules only, and locked-OOS is gate/report-only after train+validation ranking freeze with all OOS discovery/selection/objective/pruning/fitting flags false. Existing four `quality_single_pair` paper/testnet baseline lanes remain preserved.

Verification passed locally on 2026-05-23 KST: artifact invariant check, targeted tests `15 passed`, `ruff check .`, `compileall`, hardcoded audit `new=0`, diff checks, and full pytest `1422 passed in 61.85s`. Runner max RSS was `2,020,024 KiB`; full pytest max RSS was `2,769,980 KiB`, both below 8 GiB.

---

## 2026-05-23 KST — 30m+ booster pass finds materially stronger SOL 2h paper/testnet candidates

After the first 30m+ feedback pass produced acceptable but modest paper candidates, a stricter booster pass was added via `scripts/research/run_alpha_zoo_30m_plus_alpha_booster_discovery.py` with tests in `tests/test_alpha_zoo_30m_plus_alpha_booster_discovery.py`. The pass keeps native per-file 1s→30m bar construction, default `30m/1h/2h/4h/6h` timeframes, single-symbol promotable strategies, train+validation-only ranking, and locked-OOS as post-freeze gate/report only. No real-money path is enabled: every output keeps `ready_for_real=false` and `real_money_execution=false`.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_30m_plus_alpha_booster_discovery_20260523/`. Latest outputs include `alpha_zoo_30m_plus_alpha_booster_discovery_latest.json`, latest Markdown, candidate/decision/shadow CSVs, `paper_testnet_handoff_latest.json|md`, `no_promotion_shadow_shortlist_latest.json`, and `runner_time_v_latest.log`.

Result: `63,450` candidates evaluated across relative-strength chandelier breakouts, multi-horizon consensus momentum, trend-pullback reclaim, and volatility-squeeze range expansion. Strict paper/testnet-only pass rows increased to `46`; preferred booster target rows remain `0` because the best rows still have a slightly weak validation-half or preferred-MDD diagnostic. The best paper/testnet candidate is `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26`: SOLUSDT `2h` relative-strength chandelier breakout, lookback `12`, ATR breakout `0.25`, rel-strength threshold `0.5%`, trailing stop `3 ATR`, min-hold `18`, equivalent notional `0.50`. Metrics: train `+37.4602%`, validation `+16.0919%`, locked-OOS report-only `+4.2373%`, validation MDD `10.8554%`, trades `242/53/27`, liquidation/account-wipeout `0/0`, return-per-turnover proxy `30.96/60.72/31.39bps`. This is materially stronger than the prior 30m+ feedback leader (`+14.45%/+5.55%/+1.82%`) while staying paper/testnet-only.

Operational decision: this is the current strongest new-alpha paper/testnet candidate, but still not real-money ready. Preferred booster diagnostics require monitoring because validation is not positive in both halves and validation MDD is above the stricter preferred `10%` target, though it remains within the hard `12%` promotion gate. Keep the existing four `quality_single_pair` paper/testnet baseline lanes unchanged and run only paper/testnet observation with realized BBO/fill telemetry before any future review.

---

## 2026-05-23 KST — 30m+ Alpha feedback discovery promotes SOL/TRX paper/testnet-only candidates

Added `scripts/research/run_alpha_zoo_30m_plus_alpha_feedback_discovery.py` and `tests/test_alpha_zoo_30m_plus_alpha_feedback_discovery.py` as a new >=30m Alpha Zoo feedback pass, not a reversal or `quality_single_pair` retune. It constructs native 1s→30m bars with per-file chunk aggregation, derives `30m/1h/2h/4h/6h` candidates, ranks only train+validation, and uses locked-OOS only after ranking freeze as gate/report evidence. Web/reference anchors are persisted in the artifact for AdaptiveTrend-style crypto trend-following, dynamic time-series momentum, and Binance funding/taker-flow docs. Real-money remains prohibited throughout: `ready_for_real=false`, `real_money_execution=false`.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_30m_plus_alpha_feedback_discovery_20260523/`. Latest outputs include `alpha_zoo_30m_plus_alpha_feedback_discovery_latest.json`, timestamped JSON `alpha_zoo_30m_plus_alpha_feedback_discovery_20260523T090125Z.json`, latest Markdown, candidate/decision/shadow CSVs, `paper_testnet_handoff_latest.json|md`, `no_promotion_shadow_shortlist_latest.json`, and `verification_summary_latest.md`.

Result: `18450` candidates evaluated across volatility-adjusted trend persistence, Donchian ATR breakout, MA-slope ADX trend filter, and funding/OI/taker crowding continuation for BTC/ETH/SOL/BNB/TRX. Train-dominant sample-gate pass rows: `23`; execution-efficiency proxy pass rows: `73`; strict paper/testnet-only pass rows: `4`. The promoted rows are simple sparse trend-state rules, not execution permissions: SOLUSDT 6h volatility-adjusted trend persistence (`hold8`, cooldown 2, ADX20+low-vol filter) at equivalent 0.30 notional, plus TRXUSDT 4h volatility-adjusted trend persistence (`hold12`, cooldown 2) variants. Best promoted SOL row: train `14.4540%`, validation `5.5469%`, locked-OOS report-only `1.8230%`, validation MDD `10.7958%`, trades `164/42/23`, liq/wipeout `0/0`, RPT `29.38/44.02/26.42bps`. TRX rows clear the 10bps RPT gate narrowly (`15.07/12.64/10.51bps` and `14.83/12.64/10.51bps`) and should be treated as paper/testnet-only monitoring candidates.

Feedback loop: OMX worker-1 found a first real-data run shape that exceeded the 8GB cap, then validated chunked aggregation. The main runner was repaired to per-file native 1s→30m aggregation before combining shards; final leader real-data run max RSS `1,908,808 KB` (`1864.07 MiB`, <8 GiB), elapsed `3:11.64`. Team mode completed and was shut down after terminal task reconciliation; its worktree merge was skipped due overlapping untracked files, but its memory/test findings were integrated manually in the main worktree.

Verification passed on 2026-05-23 UTC/KST session: artifact postprocess invariants; targeted Alpha Zoo tests `20 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; `git diff --check`; `git diff --cached --check`; full pytest `1414 passed in 70.71s (0:01:10)` with max RSS `2,771,784 KiB` (<8 GiB). Verification summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_30m_plus_alpha_feedback_discovery_20260523/verification_summary_latest.md`. Ignored local time logs are in the artifact dir as `runner_time_v_latest.log` and `full_pytest_time_latest.log`.

---

## 2026-05-23 KST — Debounced efficiency repair discovery produces SOL paper/testnet-only candidates

A focused ETH/SOL debounced momentum hysteresis repair pass was added via `scripts/research/run_alpha_zoo_debounced_efficiency_repair_discovery.py` with tests in `tests/test_alpha_zoo_debounced_efficiency_repair_discovery.py`. The pass continues from the 2026-05-22 diverse train-dominant artifact and deliberately avoids reversal / `quality_single_pair` retuning. It searches simple state-rule repairs only: expanded min-hold, entry/exit threshold redesign, stronger hysteresis/debounce, volatility-regime filtering, ADX-like/trend-strength proxies, and post-exit cooldown over `1h/2h/4h` ETH/SOL bars.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_debounced_efficiency_repair_discovery_20260523/`. Latest files include `alpha_zoo_debounced_efficiency_repair_discovery_latest.json`, timestamped JSON `alpha_zoo_debounced_efficiency_repair_discovery_20260523T025254Z.json`, latest Markdown, `debounced_efficiency_repair_candidates_latest.csv`, `debounced_efficiency_repair_decisions_latest.csv`, `debounced_efficiency_repair_shadow_hypotheses_latest.csv`, `paper_testnet_handoff_latest.json`, `paper_testnet_handoff_latest.md`, `no_promotion_shadow_shortlist_latest.json`, and `artifact_generation_validation_latest.log`.

Governance is unchanged: locked-OOS is attached only after train+validation ranking freeze and all `uses_locked_oos_for_*` discovery/selection/objective/pruning/parameter-fitting flags are `false`; `no_calendar_date_hack=true`; primary round-trip cost remains `10bps`; return-per-turnover proxy threshold remains `avg_bbo_spread * 5 = 10bps`; all outputs keep `ready_for_real=false` and `real_money_execution=false`. Existing four `quality_single_pair` paper/testnet baseline lanes are preserved unchanged: active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, and efficiency reference `4x/0.175`.

Result: `36,000` candidates evaluated (`18,000` ETH, `18,000` SOL), `14,465` rows with train return >= validation return, `274` train-dominant sample-gate rows, `954` execution-efficiency proxy pass rows, and `82` full paper/testnet-only gate pass rows. Best paper/testnet candidate is `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d`: SOLUSDT `1h` short-only, lookback `12`, entry `2%`, exit `0.5%`, min-hold `36`, no cooldown, no extra filter, `3x/0.15` notional parity. Metrics: train `+28.2198%`, validation `+16.9294%`, locked-OOS report-only `+2.4704%`, validation MDD `9.2418%`, trades `254/63/26`, liquidation/account-wipeout `0/0`, return-per-turnover proxy `24.69/59.72/21.11bps` across train/validation/locked-OOS. The previous SOL debounced short-only near-miss had OOS RPT around `3.69bps`; the repaired family exceeds the strict `10bps` proxy gate on all splits by reducing transitions while keeping train-dominant validation evidence.

Paper/testnet handoff is generated but remains real-money blocked. The handoff requires paper/testnet-only mode, replay/live notional parity confirmation, realized fee/slippage/all-in round-trip cost, BBO spread at submit, liquidation-inclusive MDD, and account-wipeout monitoring. No live/real-money order execution was attempted or authorized.

Verification passed on 2026-05-23 UTC: artifact regeneration max RSS `4,932,260 KiB` (<8 GiB); artifact invariant check (`82` paper candidates, locked-OOS flags false, real money disabled); targeted debounced/diverse tests `12 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; `git diff --check`; `git diff --cached --check`; full pytest `1407 passed` with max RSS `2,744,928 KiB` (<8 GiB). Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_debounced_efficiency_repair_discovery_20260523/local_verification_debounced_efficiency_repair_20260523T025328Z.log`.

---

## 2026-05-22 KST — Diverse train-dominant discovery: trust gate tightened to train >= validation

The prior HTF pass surfaced high validation returns, but candidates with `train_return < validation_return` are now treated as untrusted validation spikes rather than promotion leads. A new runner, `scripts/research/run_alpha_zoo_diverse_train_dominant_discovery.py`, was added to enforce `train_return >= validation_return` and train/validation ratio `>=1.0` for any promotion path while broadening the strategy set beyond reversal and quality-single-pair tuning. The runner is still paper/testnet research only: no order execution, `ready_for_real=false`, `real_money_execution=false`, and locked-OOS is attached only after train+validation ranking freeze as gate/report evidence.

Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_diverse_train_dominant_discovery_20260522/`. Required outputs written: `alpha_zoo_diverse_train_dominant_discovery_latest.json`, timestamped JSON `alpha_zoo_diverse_train_dominant_discovery_20260522T135607Z.json`, `alpha_zoo_diverse_train_dominant_discovery_latest.md`, `diverse_train_dominant_candidates_latest.csv`, `diverse_train_dominant_decisions_latest.csv`, `diverse_train_dominant_shadow_hypotheses_latest.csv`, and `artifact_generation_validation_latest.log`.

The diversified search evaluated `22,800` candidates across nine strategy families: stateful momentum hysteresis, debounced momentum hysteresis, volatility-contraction breakout, trend-pullback reentry, range z-score mean reversion, funding-carry momentum, OI expansion/flow trend, cross-sectional market-neutral momentum, and cross-sectional low-vol momentum rotation. Coverage widened to `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `TRXUSDT`, `XRPUSDT`, `DOGEUSDT`, and `ADAUSDT` on `1h/2h/4h/6h/12h` bars. `7,968` rows had train return >= validation return, `50` rows passed the strict train-dominant sample gate, `151` rows passed the execution-efficiency proxy gate, and `0` rows passed both; therefore no new paper/testnet promotion was made.

Best train-dominant sample-gate shadows are materially better than the old 4-lane validation baseline but still fail execution-efficiency gates:

- `divtd_debounced_hysteresis_trend_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold12_3p0x_0p15_5ea41145` (`debounced_momentum_hysteresis`, ETHUSDT 1h long/short, min hold 12 bars): train `+25.0916%`, validation `+23.5473%`, locked-OOS report-only `+1.3122%`, validation MDD `4.6260%`, trades `596/147/55`, liquidation/account-wipeout `0`. Rejected because train return-per-turnover proxy is `9.356bps` (<10bps) and locked-OOS return-per-turnover proxy is `5.302bps` (<10bps).
- `divtd_debounced_hysteresis_trend_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold12_4p0x_0p1_3425bd0a`: train `+22.4142%`, validation `+20.7652%`, locked-OOS report-only `+1.1768%`, validation MDD `4.1161%`, trades `596/147/55`, liquidation/account-wipeout `0`. Rejected for train/locked-OOS return-per-turnover proxy (`9.402bps` / `5.349bps`).
- `divtd_debounced_hysteresis_trend_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold12_3p0x_0p1_bac39c24`: train `+16.9461%`, validation `+15.3254%`, locked-OOS report-only `+0.8981%`, validation MDD `3.1010%`, trades `596/147/55`, liquidation/account-wipeout `0`. Rejected for train/locked-OOS return-per-turnover proxy (`9.478bps` / `5.443bps`).
- `divtd_debounced_hysteresis_trend_1h_solusdt_short_only_lb6_e0p02_x0p005_hold12_3p0x_0p15_08f4d887`: train `+35.5640%`, validation `+9.9201%`, locked-OOS report-only `+0.5645%`, validation MDD `7.5629%`, trades `428/100/34`; rejected only because locked-OOS return-per-turnover proxy is `3.690bps` (<10bps).

Operational decision: no paper/testnet handoff from this pass. Keep the existing four `quality_single_pair` paper/testnet baseline lanes unchanged, but the next research should prioritize execution-efficiency repair on the ETH 1h debounced long/short family and SOL 1h debounced short-only family rather than retuning old reversal. The hard trust rule is now persisted: promotion candidates with `train_return < validation_return` are rejected as validation spikes, regardless of headline validation return.

Local verification passed on 2026-05-22 UTC: artifact invariant check; targeted Alpha Zoo tests `22 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; `git diff --check`; and full pytest `1400 passed` with max RSS `2,763,076 KiB` (<8 GiB). Verification logs: `local_verification_alpha_zoo_diverse_train_dominant_discovery_20260522T135607Z.log` and `local_verification_latest.log` in the artifact directory.

---

## 2026-05-22 KST — New 30m+ HTF momentum/crowding discovery finds higher-validation shadows

A new-alpha pass was added via `scripts/research/run_alpha_zoo_htf_momentum_crowding_discovery.py` rather than retuning the existing reversal or `quality_single_pair` families. The runner uses local Binance 1s OHLCV parquet resampled to `1h`, `2h`, `4h`, and `6h` bars for BTC/ETH/SOL/BNB/TRX, optionally joins local funding/open-interest/taker-flow feature points, and evaluates four genuinely new 30m+ families: HTF trend persistence, Donchian breakout continuation, funding-squeeze continuation, and liquid cross-sectional momentum. The external research/docs anchors for this direction are recorded in the artifact: AdaptiveTrend-style longer-horizon crypto trend-following, Binance funding, open-interest, and taker buy/sell volume documentation.

Artifacts were generated under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_htf_momentum_crowding_discovery_20260522/`: latest JSON/Markdown, timestamped JSON `alpha_zoo_htf_momentum_crowding_discovery_20260522T124629Z.json`, `htf_momentum_crowding_candidates_latest.csv`, `htf_momentum_crowding_decisions_latest.csv`, `htf_momentum_crowding_shadow_hypotheses_latest.csv`, and `artifact_generation_validation_latest.log`. The run evaluated `2,732` candidates: `1,600` HTF trend persistence, `400` Donchian breakout, `540` funding-squeeze continuation, and `192` liquid cross-sectional momentum. Locked-OOS is explicitly report/gate-only after train+validation freeze (`uses_locked_oos_for_discovery=false`, `uses_locked_oos_for_selection=false`, `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`). All artifacts keep `ready_for_real=false` and `real_money_execution=false`.

Result: no full paper/testnet promotion yet (`paper_candidate_gate_pass_count=0`), but the new HTF family materially raised validation-return evidence versus the current 4-lane baseline. The best validation shadow is `htf_trend_4h_trxusdt_lb48_th0p005_5p0x_0p15_3a7d84a7` (`htf_trend_persistence`, TRXUSDT, 4h, lookback 48, threshold 0.5%, 5x/0.15): train `-8.4327%`, validation `+15.0262%`, locked-OOS report-only `+4.4971%`, validation MDD `2.6586%`, trades `234/43/34`, liquidation/account-wipeout `0`; rejected because train return and train/validation ratio are negative and train return-per-turnover proxy is below the spread gate. The strict backtest-sample gate pass is `htf_trend_1h_ethusdt_lb6_th0p02_5p0x_0p15_3a23dcc0` (ETHUSDT, 1h, lookback 6, threshold 2%, 5x/0.15): train `+1.4647%`, validation `+2.1537%`, locked-OOS report-only `+4.1498%`, validation MDD `6.4524%`, trades `570/122/36`, train/validation ratio `0.6801`, liquidation/account-wipeout `0`; it remains `validation_alpha_shadow_until_execution_efficiency` because train and validation return-per-turnover proxies (`0.343bps` and `2.354bps`) fail the `avg_bbo_spread * 5 = 10bps` proxy gate. One lower-sample efficiency proxy row (`htf_trend_4h_ethusdt_lb48_th0p04_3p0x_0p1_2956d92f`) passes return-per-turnover but fails validation/OOS sample counts.

Operational decision: keep the four existing `quality_single_pair` paper/testnet baseline lanes unchanged, do not promote the new HTF candidates to paper/testnet yet, and continue next research from the HTF momentum/crowding direction rather than further tuning the old reversal clue. The immediate next experiments should improve train robustness and turnover efficiency for the TRX 4h/6h validation leaders and the ETH 1h sample-pass shadow without using locked-OOS for selection: e.g., volatility-regime conditioning, fewer-transition holding/exit hysteresis, and actual paper/testnet BBO/fill telemetry once available.

---

## 2026-05-22 KST — Missing fill/BBO telemetry now falls back to a fail-closed backtest cut

The paper fill-efficiency gate now handles absent actual paper/testnet fill/BBO telemetry without ending as an empty stop state. When no fill JSONL is available, it still cuts the current sample-guarded universe using the existing 10bps backtest evidence from `alpha_zoo_sample_guarded_alpha_discovery_latest.json`. This fallback is explicitly paper/testnet-review-only: it never sets `ready_for_real=true` and never permits real-money execution.

The fallback pass rule is intentionally strict: candidate rows must already be `ready_for_paper=true` in the sample-guarded discovery, pass the primary 10bps promotion gate, pass the return-per-turnover-vs-spread proxy gate, satisfy split sample guards, and survive locked-OOS only as a post-freeze report gate with positive return and zero liquidation/account wipeout. Locked-OOS is still not used for discovery, selection, objective scoring, pruning, or parameter fitting.

The regenerated fill-efficiency artifact under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/` records this policy in `gate_policy.backtest_fallback_policy` and writes `backtest_fallback_candidates_latest.csv`. Current result remains no-promotion: actual fill telemetry status is `pending_paper_testnet_fill_telemetry`; the fallback evaluated `976` sample-guarded candidates, emitted the top `60` backtest-cut rows for review, and found `0` fallback pass rows. All outputs keep `ready_for_real=false` and `real_money_execution=false`. Timestamped JSON: `alpha_zoo_paper_fill_efficiency_gate_20260522T112551Z.json`.


Final verification passed on 2026-05-22 UTC for the missing-fill fallback cut: artifact regeneration; targeted fill-efficiency/sample-guarded tests `11 passed`; full pytest `1389 passed` with max RSS `2,715,696 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; `git diff --cached --check`; and artifact invariants for `ready_for_real=false`, `real_money_execution=false`, fill count `0`, and fallback pass count `0`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/local_verification_alpha_zoo_paper_fill_efficiency_gate_backtest_fallback_final_20260522T112551Z.log`.

---

## 2026-05-22 KST — Actual fill/BBO efficiency gate added, pending telemetry

Added a fail-closed actual paper/testnet telemetry gate for the Alpha Zoo sample-guarded path: `scripts/research/run_alpha_zoo_paper_fill_efficiency_gate.py`. It consumes the sample-guarded discovery artifact, the paper-forward monitoring contract, and an optional fill JSONL. This is not an execution runner and it never enables real-money trading.

The gate converts the earlier proxy condition into the actual telemetry contract to use once paper/testnet fills exist:

- Return per turnover: `sum(realized_pnl_quote) * 10000 / sum(abs(notional_quote))`.
- Average BBO spread: notional-weighted average `spread_bps_at_submit`.
- Primary edge gate: `realized_return_per_turnover_bps > avg_bbo_spread_bps * 5`.
- Cost gates: mean all-in cost `<=10bps`, p95 all-in cost `<=15bps`.
- Execution hygiene: timeout/cancel/partial-fill limits, zero liquidation, zero account wipeout.

No paper/testnet fill JSONL is available in the repo yet, so the generated artifact is intentionally fail-closed: status `pending_paper_testnet_fill_telemetry`, `fill_count=0`, `actual_fill_efficiency_gate_pass=false`, `ready_for_real=false`, and `real_money_execution=false`. Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/`; timestamped JSON `alpha_zoo_paper_fill_efficiency_gate_20260522T110432Z.json`, latest JSON/MD, `paper_fill_efficiency_decisions_latest.csv`, and `artifact_generation_validation_latest.log` were written.

Verification passed on 2026-05-22 UTC: artifact regeneration; new fill-efficiency gate tests `4 passed`; sample-guarded regression tests `6 passed`; full pytest `1388 passed` with max RSS `2,732,348 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and `git diff --cached --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_paper_fill_efficiency_gate_20260522/local_verification_alpha_zoo_paper_fill_efficiency_gate_20260522T110431Z.log`.

---

## 2026-05-22 KST — Sample-guarded discovery adds return/turnover-vs-spread proxy gate

The sample-guarded 10bps discovery runner was tuned to incorporate the execution-quality hint `return per turnover > avg BBO spread * 5`. The current frozen expanded-retune metric surface does **not** contain actual average BBO spread or exact turnover fields, so the artifact now records this as an explicit proxy rather than claiming live microstructure evidence:

- Average BBO spread assumption: `2.0bps`.
- Multiplier: `5.0`.
- Return/turnover proxy threshold: `10.0bps`.
- Turnover proxy formula: `trade_event_count * abs(leverage * allocation_fraction)`.
- Return/turnover proxy formula: `total_return * 10000 / turnover_proxy`.
- Train+validation proxy profile: `execution_efficiency_proxy_v1`, with locked-OOS excluded from ranking/objective/pruning/parameter-fitting.
- Locked-OOS proxy role: attached only after train+validation freeze as a report-only promotion gate.

Regenerated artifacts under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/`, including timestamped JSON `alpha_zoo_sample_guarded_alpha_discovery_20260522T105008Z.json`, latest JSON/Markdown/CSVs, and `artifact_generation_validation_latest.log`. The updated run still found `0` new paper candidates across `976` complete 10bps models: `252` thin-sample shadows, `724` reject/quarantine, `20` historical OOS-bucket quarantines, and `0` calendar quarantines. Execution-efficiency proxy pass counts were: train `184`, validation `376`, locked-OOS report gate `57`, and full proxy gate `40`; those rows still failed other sample, locked-OOS, or primary 10bps promotion requirements. The four existing `quality_single_pair` paper/testnet baseline lanes remain unchanged, and all artifacts keep `ready_for_real=false` and `real_money_execution=false`.

Verification passed on 2026-05-22 UTC: artifact regeneration; targeted sample-guarded tests `6 passed`; full pytest `1384 passed` with max RSS `2,729,152 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and `git diff --cached --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/local_verification_sample_guarded_turnover_proxy_20260522T105007Z.log`.

---

## 2026-05-22 KST — Sample-guarded Alpha Zoo discovery rerun hardens no-promotion evidence

The 10bps sample-guarded discovery bundle was regenerated from `private/main` commit `711eeb4dbf83895b94f9fbf22d1afb79c4be1284` under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/`. The refreshed timestamped artifact is `alpha_zoo_sample_guarded_alpha_discovery_20260522T103013Z.json`; `alpha_zoo_sample_guarded_alpha_discovery_latest.json`, `alpha_zoo_sample_guarded_alpha_discovery_latest.md`, `sample_guarded_candidates_latest.csv`, `paper_candidate_decisions_latest.csv`, `shadow_hypotheses_latest.csv`, `cost_sensitivity_latest.csv`, and `artifact_generation_validation_latest.log` were updated in place.

The rerun again found `0` new paper candidates across `976` complete 10bps models. Status counts remain `252` `shadow_only_thin_sample` and `724` `reject_or_quarantine`; `956` rows were selection eligible, `20` historical OOS-bucket rows were quarantined, and `0` calendar rows were selected. Locked-OOS remains gate/report-only after train+validation ranking freeze: discovery/selection/objective/pruning/parameter-fitting/correlation flags are all `false` for locked-OOS use.

No real-money handoff was created. The generated decision status is `no_new_paper_promotion_shadow_shortlist`, with `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, and `paper_execution_allowed=false`. The four existing `quality_single_pair` paper/testnet baseline lanes remain unchanged: active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, and efficiency reference `4x/0.175`. Replay/live notional parity remains recorded for candidates and baseline lanes.

Test hardening added regressions for locked-OOS-insensitive sample-guarded ranking, non-10bps source rejection, exact four-lane baseline preservation, non-empty no-promotion rejection reasons, and output CSV/markdown/generation-log content. Local verification passed on 2026-05-22 UTC: artifact regeneration; targeted sample-guarded/long-only/expanded-shadow/10bps retune tests `26 passed`; full pytest `1382 passed`; `ruff check .`; `compileall` over `src scripts tests`; `git diff --check`; and `git diff --cached --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/local_verification_alpha_zoo_sample_guarded_alpha_discovery_20260522T103012Z.log`.

---

## 2026-05-21 KST — New-factor sample-guarded Alpha Zoo discovery remains shadow-only

A follow-up discovery added opt-in/default-zero factor families to `CryptoFxAlphaZooStateStrategy`: liquidity-sweep reversal, liquidity-sweep continuation, range-expansion breakout, and range-expansion fade. The source grid was expanded via `--include-sample-guarded-new-alpha-grid`, then the 10bps retune used Optuna (`--n-trials 96`) plus the new `--sample-guarded-composite-grid` to widen side/symbol/factor-family/threshold filters without calendar/date rules. Locked-OOS stayed gate/report-only after train+validation freeze; it was not used for discovery, selection, objectives, pruning, or parameter fitting.

Final artifacts are under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_new_alpha_inversion_sample_guarded_discovery_20260521/`, sourced from `alpha_zoo_new_alpha_inversion_source_20260521/` and `alpha_zoo_new_alpha_inversion_10bps_optuna_20260521/`. The retune evaluated `2,585` models (`1,894` fresh trade-filter models), `75,104` train+validation trade-filter variants, and kept `94` live-promotable 10bps rows in the old quality-single-pair family. Strict sample-guarded promotion found `0` new paper candidates across `2,565` selection-eligible rows; `1,694` rows are shadow-only thin-sample and `891` are reject/quarantine. The discovery bundle remains `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, and `paper_execution_allowed=false`.

Best new-family shadows:

- `alpha_zoo_range_sweep_blend`, BNB/USDT abs-score `>=2`, `7x/0.175`: train `+9.6065%` / `66` trades, validation `+13.4943%` / `14` trades, validation MDD `0.8534%`, locked-OOS report-only `-1.7994%` / `10` trades, liquidation/account-wipeout `0`. Rejected for train, validation, and locked-OOS sample counts plus negative locked-OOS and primary 10bps gate failure.
- `alpha_zoo_liquidity_sweep_continuation`, SOL/USDT short abs-score `>=1`, `7x/0.20`: train `+9.2147%` / `58` trades, validation `+10.7779%` / `17` trades, validation MDD `2.4691%`, locked-OOS report-only `-0.8948%` / `3` trades, liquidation/account-wipeout `0`. Rejected for all sample guards, negative locked-OOS, and primary 10bps gate failure.
- `alpha_zoo_range_expansion_breakout`, SOL/USDT short abs-score `>=1.5`, `6x/0.15`: primary 10bps gate passed, train `+30.6004%` / `105` trades, but validation was only `+0.1163%` / `29` trades and locked-OOS report-only was `+0.0631%` / `16` trades. Rejected for validation trades `<30`, locked-OOS trades `<20`, and validation return `<2%`.

Operational decision: do not promote a new alpha to paper/testnet from this pass. Keep the four existing `quality_single_pair` paper lanes unchanged (active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, efficiency reference `4x/0.175`). New factor families stay shadow-only for future sample accumulation or a genuinely broader causal factor search; do not weaken sample gates or use locked-OOS to rescue them.

Required outputs were written: `alpha_zoo_sample_guarded_alpha_discovery_latest.json`, timestamped JSON `alpha_zoo_sample_guarded_alpha_discovery_20260521T132201Z.json`, `alpha_zoo_sample_guarded_alpha_discovery_latest.md`, `sample_guarded_candidates_latest.csv`, `paper_candidate_decisions_latest.csv`, `shadow_hypotheses_latest.csv`, `cost_sensitivity_latest.csv`, `artifact_generation_validation_latest.log`, and `local_verification_latest.log`. Final verification passed on 2026-05-21 UTC, including the post-CI hardcoded-parameter audit fix: `ruff check .`; `uv run python scripts/audit_hardcoded_params.py` (`new=0`); targeted Alpha Zoo tests `17 passed`; `python -m compileall -q src scripts tests`; `git diff --check`; full pytest `1380 passed`; max full-test RSS `2,880,156 KiB` (~2.75 GiB, <8 GiB). Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_new_alpha_inversion_sample_guarded_discovery_20260521/local_verification_new_alpha_sample_guarded_ci_fix_20260521T133109Z.log`.

---

## 2026-05-21 KST — Sample-guarded 10bps Alpha Zoo discovery: no new paper promotion

A new sample-guarded discovery bundle was generated under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/` using the frozen expanded 10bps retune artifact. The new runner `scripts/research/run_alpha_zoo_sample_guarded_alpha_discovery.py` ranks only train+validation evidence across three profiles (`validation_strength_v1`, `train_validation_robustness_v1`, and `cost_efficiency_v1`) and attaches locked-OOS only after the train+validation profile freeze as gate/report-only. The artifact explicitly records `uses_locked_oos_for_discovery=false`, `uses_locked_oos_for_selection=false`, `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, and `uses_locked_oos_for_parameter_fitting=false`.

Strict sample-guarded promotion found `0` new paper candidates across `976` complete models (`956` selection-eligible, `20` historical OOS-bucket rows quarantined). Status counts are `252` `shadow_only_thin_sample` and `724` `reject_or_quarantine`; `ready_for_paper=false`, `ready_for_real=false`, and `real_money_execution=false`. The required 10bps paper promotion guard remains stricter than the high-validation shadow rows: `conservative_exit` variants still show large validation returns but fail locked-OOS/ratio/gate checks, while the prior long-only residual-reversal clue remains shadow-only (`43` rows, `0` paper candidates, max validation trades `18`, max locked-OOS trades `13`). The upstream selected metric surface includes side/factor/threshold filters (`LONG`, `crypto_residual_reversal`, `crypto_residual_momentum`, `volume_vwap_pressure`, abs-score thresholds `1.5/2.5/3.0`); no symbol-filtered variant survived into the frozen selected metric/gate-pass surface, so symbol ideas are reported as not-promoted rather than mined via locked-OOS.

The four existing `quality_single_pair` paper/testnet baseline lanes remain unchanged: active `7x/0.20`, balanced `6x/0.175`, validation leader `5x/0.20`, and efficiency reference `4x/0.175`. Replay/live notional parity is recorded for every sample-guarded row; all new decisions are no-promotion shadow/reject decisions and keep real-money blocked.

Required outputs were written: `alpha_zoo_sample_guarded_alpha_discovery_latest.json`, timestamped JSON `alpha_zoo_sample_guarded_alpha_discovery_20260521T110118Z.json`, `alpha_zoo_sample_guarded_alpha_discovery_latest.md`, `sample_guarded_candidates_latest.csv`, `paper_candidate_decisions_latest.csv`, `shadow_hypotheses_latest.csv`, `cost_sensitivity_latest.csv`, and `artifact_generation_validation_latest.log`. Local verification passed on 2026-05-21 UTC: artifact regeneration; targeted sample/retune/guarded tests `23 passed`; full pytest `1377 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and `git diff --cached --check`. Maximum full-test RSS was `2,831,596 KiB` (<8 GiB) after post-deslop rerun. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_sample_guarded_alpha_discovery_20260520/local_verification_alpha_zoo_sample_guarded_alpha_discovery_20260521T110118Z.log`.

---

## 2026-05-20 KST — Long-only reversal guarded study kept the new family shadow-only

A dedicated guarded study was generated for the expanded-retune discovery under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_long_only_reversal_guarded_study_20260520/`. It focuses on `alpha_zoo_high_confidence_long_only` with `dominant_factor_family=crypto_residual_reversal` and `abs_factor_score_min=1.5`, at the same fixed `10.0bps` all-in round-trip cost assumption.

The study ranks variants using train+validation only, then applies locked-OOS strictly as a post-freeze gate/report field. Locked-OOS is not used for selection/objective/pruning/parameter fitting. The target family has `43` positive train/validation/locked-OOS variants, but `0` pass the strict paper guard: maximum validation sample is only `18` trades versus the `30`-trade guard, maximum locked-OOS sample is only `13` trades versus the `20`-trade report gate, maximum train/validation return ratio is `0.4788` versus the `0.50` train-robustness guard, and the primary 10bps promotion gate pass count is `0`.

The train+validation leader remains `fresh_tv10_filter_family_crypto_residual_reversal_abs_score_ge_1p5_alpha_zoo_high_confidence_long_only_8p0x_0p2alloc`: validation `+14.1732%`, train `+5.0784%`, locked-OOS report-only `+1.8620%`, validation MDD `3.1214%`, locked-OOS MDD `1.8137%`, liquidation/account-wipeout `0`. Despite positive locked-OOS, it is still `ready_for_paper=false`, `ready_for_real=false`, and `real_money_execution=false` because the split sample and train robustness evidence are too thin.

Operational decision: keep the four quality-single-pair paper/testnet lanes running side-by-side, and keep the long-only residual-reversal family as shadow-only research until a future train+validation-only discovery can satisfy stricter split sample guards without using locked-OOS for selection.

Verification passed on 2026-05-20 UTC for the guarded study: artifact regeneration; targeted Alpha Zoo retune/paper-forward tests `29 passed`; full pytest `1375 passed`; peak full-test RSS `2,728,984 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; and `git diff --check`. Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_long_only_reversal_guarded_study_20260520/local_verification_alpha_zoo_long_only_reversal_guarded_study_20260520T133551Z.log`.

---

## 2026-05-20 KST — Expanded filter retune found a new shadow-only validation family

An expanded 10bps train+validation retune was run under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_expanded_filter_retune_20260520/` with `--n-trials 0`, `--hybrid-seed-count 0`, `--trade-filter-top-n 300`, `--min-trade-filter-trades 10`, and `--fresh-candidate-limit 600`. Runtime was `7:34.86`, peak RSS was `410,552 KiB` (`400.9 MiB`, well under 8 GiB). The expanded universe increased from `796` to `976` complete models and from `2,388` to `2,928` primary metric rows, but the live-promotable count stayed `56`; the active and balanced live-gate models remain the same `quality_single_pair` 7x/0.20 and 6x/0.175 lanes.

The new useful discovery is not a paper candidate yet: `alpha_zoo_high_confidence_long_only` filtered to `dominant_factor_family=crypto_residual_reversal` and `abs_factor_score_min=1.5`. The leading row is `fresh_tv10_filter_family_crypto_residual_reversal_abs_score_ge_1p5_alpha_zoo_high_confidence_long_only_8p0x_0p2alloc`: validation return `+14.1732%`, validation MDD `3.1214%`, train return `+5.0784%`, locked-OOS return `+1.8620%`, locked-OOS MDD `1.8137%`, and liquidation `0`. However it has only `18` validation trades and `13` locked-OOS trades, and it fails the primary 10bps promotion gate because train metrics are not above validation/locked-OOS metrics (`train_total_return_not_above_validation` and related train metric asymmetry reasons). Therefore the companion artifact `alpha_zoo_expanded_filter_shadow_selection_20260520/` marks it `ready_for_paper=false`, `ready_for_real=false`, `real_money_execution=false`, `paper_execution_allowed=false`, `shadow_observation_allowed=true`.

Expanded conservative-exit rescue did not produce a positive locked-OOS conservative candidate. The conclusion is now sharper: keep the four quality-single-pair lanes as actual paper/testnet execution candidates, and treat `high_confidence_long_only / crypto_residual_reversal` as the next shadow-only strategy family requiring a dedicated train+validation retune with stricter minimum trade-count guards before any paper execution artifact.

---

## 2026-05-20 KST — Four-lane paper-forward contract and shadow strategy audit

A follow-up four-lane artifact now joins the original active/balanced paper handoff with the validation-first handoff under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_four_lane_shadow_discovery_20260520/`. The contract stays paper/testnet-only: `ready_for_paper=true`, `ready_for_real=false`, `real_money_execution=false`, and the primary research cost is still `10.0bps` all-in round-trip.

Four lanes to run side-by-side in paper/testnet:

| Lane | Model | Notional/equity | Validation return | Validation MDD | Locked-OOS return | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Active | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` | `140%` | `+0.4724%` | `14.9117%` | `+1.8382%` | paper only |
| Balanced | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` | `105%` | `+0.5942%` | `11.3653%` | `+1.5464%` | paper only |
| Validation leader | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` | `100%` | `+0.5986%` | `10.8490%` | `+1.4956%` | paper only |
| Efficiency reference | `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` | `70%` | `+0.5561%` | `7.6999%` | `+1.1432%` | paper only |

The shadow strategy audit found higher validation families but no frozen-universe non-quality paper candidate. Top `conservative_exit` rows reach validation `+27.3844%`, yet locked-OOS is `-4.2507%`; top side/family threshold rows reach validation `+5.9487%`, yet locked-OOS is `-4.5444%`. Both remain shadow-only. The next true strategy-search step is an expanded train+validation-only retune/rescue pass, not real-money promotion.

---

## 2026-05-20 KST — Validation-first 10bps discovery after weak validation check

Follow-up validation-first discovery confirmed that the prior active 7x/0.20 handoff is not the best validation performer inside the frozen 10bps universe. The new artifact is under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_validation_first_discovery_20260520/` and keeps `real_money_execution=false`, `ready_for_real=false`, and locked-OOS gate/report-only after train+validation ranking freeze.

Selected paper/testnet validation-first candidates:

- Validation return leader: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` (`5x`, allocation `0.20`, isolated notional/equity `100%`). Validation return `+0.5986%`, validation MDD `10.8490%`, train return `+34.5152%`, locked-OOS return `+1.4956%`, liquidation `0`; preflight `ready_for_paper=true`, `ready_for_real=false`.
- Validation efficiency reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` (`4x`, allocation `0.175`, isolated notional/equity `70%`). Validation return `+0.5561%`, validation MDD `7.6999%`, train return `+24.8845%`, locked-OOS return `+1.1432%`, liquidation `0`; preflight `ready_for_paper=true`, `ready_for_real=false`.

The frozen 10bps universe has `56` live-gate-passed candidates, but the best live-gate validation return is only `+0.5986%`. A material validation-edge audit found `0` zero-liquidation candidates with validation return `>1%` and positive locked-OOS return. High-validation alternatives do exist, especially `conservative_exit` variants with validation around `+20%` to `+27%`, but they fail promotion gates because locked-OOS returns are negative and train metrics are not above validation; they are shadow-only strategy hypotheses, not paper candidates.

Recommended next experiments are therefore not immediate real-money promotion: run a train+validation-only regime-gated `conservative_exit` rescue, side/symbol-specific `abs_score` thresholds for `quality_single_pair`, and continue paper-forward monitoring of the validation-first 5x/0.20 and 4x/0.175 lanes beside the existing active/balanced lanes. Real fill monitoring must still compare all-in round-trip bps to the 10bps mean / 15bps p95 contract before any real-money discussion.

---

## 2026-05-20 KST — 10bps Alpha Zoo paper/testnet preflight and monitoring handoff

Built the paper/testnet-only handoff bundle for the final 10bps active profile and balanced reference under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/`. This is not a real-money promotion: every decision/preflight artifact keeps `real_money_execution=false` and `ready_for_real=false`.

Side-by-side paper candidates:

- Active: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` (`higher_risk_train_return_tilt_v1`, isolated `7x`, allocation `0.20`, target notional/equity `140%`, paper-equivalent `$10,000 -> $14,000` notional). Preflight status: `ready_for_paper=true`, `ready_for_real=false`.
- Balanced reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` (`balanced_train_validation_v1`, isolated `6x`, allocation `0.175`, target notional/equity `105%`, paper-equivalent `$10,000 -> $10,500` notional). Preflight status: `ready_for_paper=true`, `ready_for_real=false`.

The live strategy now supports the retuned `abs_factor_score_min=1.5` gate directly, preserving the 10bps trade-filter contract in paper/testnet runtime instead of only recording it as offline metadata. The bundle reuses the notional-risk-aligned `isolated_margin_fraction` sizing parity path and carries split/source/profile lineage from the frozen 10bps retune source, low-correlation sidecar, and notional-risk-aligned live artifact. Locked-OOS remains gate/report-only after candidate/profile freeze; selection/profile metadata continues to record `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`, and `uses_locked_oos_for_selection=false`.

Artifacts:

- Bundle JSON/Markdown: `alpha_zoo_7x_paper_forward_preflight_latest.json`, `alpha_zoo_7x_paper_forward_preflight_latest.md`.
- Timestamped bundle: `alpha_zoo_7x_paper_forward_preflight_20260520T112422Z.json`.
- Active decision: `live_alpha_zoo_quality_single_pair_7x_0p20_paper_decision_latest.json`.
- Balanced decision: `live_alpha_zoo_quality_single_pair_6x_0p175_balanced_reference_decision_latest.json`.
- Active preflight: `live_readiness_preflight_alpha_zoo_7x_0p20_paper_latest.json`.
- Balanced preflight: `live_readiness_preflight_alpha_zoo_6x_0p175_balanced_reference_paper_latest.json`.
- Monitoring contract: `paper_forward_monitoring_contract_latest.json` and `paper_forward_monitoring_contract_latest.csv`.
- Verification log: `local_verification_alpha_zoo_7x_paper_forward_preflight_20260520T111631Z.log`.

Monitoring contract status is `pending_paper_forward_fills`; it defines realized `fee_bps`, `slippage_bps`, and `all_in_round_trip_bps` fields, active-vs-balanced grouping keys, maker/taker/partial-fill/missed-signal fields, and liquidation-inclusive equity/MDD/account-wipeout checks. The cost audit keeps the research assumption fixed at `10.0bps` all-in round-trip with pass thresholds `mean <= 10bps` and `p95 <= 15bps`; actual paper/testnet fills must be collected before any real-money discussion.

Verification passed on 2026-05-20 UTC: artifact generation smoke; CSV-LF final verification: artifact regeneration; targeted tests `17 passed`; full pytest `1369 passed`; max RSS `2,877,880 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and staged `git diff --cached --check` after index sync.

---

## 2026-05-19 KST — Next plan: 7x/0.20 paper-forward live preflight

Saved next-session plan: `.omx/plans/plan-alpha-zoo-7x-paper-forward-live-preflight-20260519.md`.

The next step is **not real-money execution**. It is a paper/testnet-only live decision, preflight, and forward-monitoring handoff for the active 10bps profile and the balanced reference:

- Active paper candidate: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` (`higher_risk_train_return_tilt_v1`, isolated `7x`, allocation `0.20`).
- Balanced reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` (`balanced_train_validation_v1`, isolated `6x`, allocation `0.175`).
- Required governance: `real_money_execution=false`, `ready_for_real=false`, locked-OOS gate/report-only, replay/live sizing parity, liquidation-inclusive MDD, and realized round-trip cost monitoring against the 10bps research assumption.
- Recommended artifact dir for the follow-up: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/`.

The plan rejects immediate real-money promotion because the 7x/0.20 validation return is still weak and the low-correlation discovery found `0` independently deployable low-correlation gate-pass streams. The next evidence needed is paper/testnet fill-quality and risk monitoring for active vs balanced side-by-side.

---

## 2026-05-19 KST — Higher-risk 10bps selection profile and low-correlation discovery result

The follow-up risk-selection run finalized the active 10bps model by a predeclared train+validation-only higher-risk profile, not by locked-OOS ranking. The runner now replays all `600` source candidates by default, records the selected profile metadata in the artifact, and emits low-correlation discovery sidecars. Locked-OOS remains gate/report-only after candidate/profile freeze; `real_money_execution=false` throughout.

Final artifact summary:

- Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/`.
- Latest main JSON/Markdown: `alpha_zoo_10bps_full_retune_latest.json`, `alpha_zoo_10bps_full_retune_latest.md`.
- Timestamped final JSON/Markdown: `alpha_zoo_10bps_full_retune_20260519T140542Z.json`, `alpha_zoo_10bps_full_retune_20260519T140542Z.md`.
- Candidate accounting: `796` models / `2388` split metric rows; `600` source candidates selected before locked-OOS gates; `775` low-correlation candidate streams compared against the active reference.
- Memory: artifact peak RSS `385.7539 MiB`; `/usr/bin/time` max RSS `395,012 KiB`, under the 8 GiB session limit.

Selection profiles:

- Active final profile: `higher_risk_train_return_tilt_v1` -> `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc`; profile score `0.5139152402359146`.
- Balanced reference profile: `balanced_train_validation_v1` -> `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`; profile score `0.20188310617174543`.
- Both profile definitions record train+validation as objective/selection/optimization/pruning/parameter-fit/score-formula inputs and all `uses_locked_oos_*` selector flags as `false`.

Active final 10bps profile split metrics (`7x`, allocation `0.20`, non-calendar `abs_score_ge_1.5` quality-single-pair filter):

- Train: return `+45.6916%`, MDD `31.6050%`, Sharpe `0.994753`, Sortino `0.287423`, Calmar `1.445708`, trades `405`, liquidation `0`.
- Validation: return `+0.4724%`, MDD `14.9117%`, Sharpe `0.219005`, Sortino `0.047090`, Calmar `0.129660`, trades `86`, liquidation `0`.
- Locked-OOS gate/report-only: return `+1.8382%`, MDD `14.3237%`, Sharpe `0.556723`, Sortino `0.169307`, Calmar `1.074121`, trades `53`, liquidation `0`.

Balanced reference 10bps split metrics (`6x`, allocation `0.175`, same non-calendar filter): train `+36.0268%`, validation `+0.5942%`, locked-OOS `+1.5464%`; all splits liquidation `0`.

Low-correlation discovery sidecars: `low_correlation_discovery_latest.json` and `low_correlation_discovery_latest.csv`. Correlations are computed from train+validation returns only against the active higher-risk reference; `423` streams are below the `0.35` absolute-correlation threshold, but `0` are deployable 10bps gate passers independent of the reference in this run, so the discovery rows are research-only until a low-correlation stream also clears locked-OOS gate/report checks.

Verification passed on 2026-05-19 UTC: artifact assertion passed (`796` models, `2388` metric rows, `50` low-correlation rows); focused 10bps retune and artifact assertion tests `19 passed`; `ruff` passed on changed runner/assertion/tests; `compileall` passed. Full `n_trials=80` hybrid optimizer was not rerun in this final profile pass; the required deliverable was profile-safe selection plus low-correlation discovery under the 8 GiB guard.

---

## 2026-05-19 KST — Risk-selection and low-correlation verification contract

The follow-up risk-selection plan keeps locked-OOS as post-freeze gate/report-only evidence while predeclaring two train+validation-only selection profiles:

- `balanced_train_validation_v1` remains the balanced reference and must report `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`.
- `higher_risk_train_return_tilt_v1` is the active final profile and may select `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` only after the existing 10bps promotion gates pass.

The artifact validator now requires explicit profile metadata, score-formula inputs, `uses_locked_oos_* = false` flags, balanced-reference preservation, and a separate `low_correlation_discovery_latest.json/csv` surface. Low-correlation discovery rows must compute correlations from train+validation returns only, label deployable 10bps gate passers separately from research-only locked-OOS gate failures, keep `real_money_execution=false`, and preserve the `<8192 MiB` memory guard.

---

## 2026-05-19 KST — Alpha Zoo full 10bps round-trip retune and live-gate repair

Re-ran the Alpha Zoo backtest-to-live candidate family under the latest split and locked promotion cost of round-trip slippage/fee `10bps`. The run covers historical top-bucket / live-ranked Alpha Zoo seed streams, the prior Hybrid v3.5/v3.6 seed-union rows, references, and fresh train+validation-only variants. Locked-OOS stayed strictly post-freeze gate/report-only: the artifact records selection inputs `['train', 'validation']`, `uses_locked_oos_for_selection=false`, and trade-filter locked-OOS role `gate_report_only_after_variant_freeze`. No real-money execution was attempted.

The earlier top-bucket/hybrid rows did not survive as live candidates at `10bps`: `hybrid_v3_5_seed_union` remains a shadow-only historical OOS-bucket row and fails the fresh live gate because locked-OOS return/Sharpe/Sortino/smart Sortino/Calmar are non-positive and because its lineage uses OOS-derived bucket selection. The plain fresh 10bps seed/hybrid retune also found no live-ready model with positive validation and locked-OOS performance, so a non-calendar fixed trade-filter retune lane was added. The lane evaluates only signal-structure filters such as side/symbol/factor-family/absolute factor-score/hold cap over train+validation, then freezes the variant before locked-OOS reporting.

Final full retune summary:

- Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/`.
- Latest main JSON/Markdown: `alpha_zoo_10bps_full_retune_latest.json`, `alpha_zoo_10bps_full_retune_latest.md`.
- Timestamped final JSON/Markdown: `alpha_zoo_10bps_full_retune_20260519T121224Z.json`, `alpha_zoo_10bps_full_retune_20260519T121224Z.md`.
- Split: train `2025-01-01T00:00:00Z..2025-12-31T23:00:00Z`; validation `2026-01-01T00:00:00Z..2026-03-31T23:00:00Z`; locked-OOS `2026-04-01T00:00:00Z..2026-05-17T10:00:00Z`; timestamp-index hash `b973165bc1057f3aaa08ea637b73a45df3e84fdb7d1337b1637233d205696bb0`.
- Candidate accounting: `798` models / `2394` split metric rows; `778` fresh train+validation models; `176` fresh trade-filter models; `20` shadow-only historical rows.
- Search accounting: `600` source candidate rows replayed, `776` fresh 10bps streams evaluated, `30,354` trade-filter variants evaluated, `176` selected variants, `56` passing the final 10bps live gate.
- Memory: runner peak RSS `400.9883 MiB` by artifact memory summary and `/usr/bin/time` max RSS `420,012 KiB`; full pytest post-LF max RSS `2,722,856 KiB`, both under the 8 GiB session limit.

Best live-gate candidate at round-trip `10bps`:

- Model id: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`.
- Strategy family: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_quality_single_pair`.
- Variant: non-calendar `abs_factor_score_min=1.5` (`abs_score_ge_1.5`).
- Sizing: isolated `6x`, allocation fraction `0.175`, `544` filtered trades; liquidation/account-wipeout counts are `0` on all splits and margin buffers stay positive.
- Train: return `+36.0268%`, MDD `24.6028%`, Sharpe `1.002645`, Sortino `0.289764`, smart Sortino `0.232550`, Calmar `1.464340`, trades `405`, min margin buffer `8514.118330`.
- Validation: return `+0.5942%`, MDD `11.3653%`, Sharpe `0.219846`, Sortino `0.047272`, smart Sortino `0.042448`, Calmar `0.214374`, trades `86`, min margin buffer `9251.785896`.
- Locked-OOS gate/report-only: return `+1.5464%`, MDD `10.9211%`, Sharpe `0.554965`, Sortino `0.168704`, smart Sortino `0.152094`, Calmar `1.173217`, trades `53`, min margin buffer `9383.460782`.

The selected live-gate candidate satisfies the requested dominance shape: train return/Sharpe/Sortino/smart Sortino/Calmar are all above validation and locked-OOS; validation and locked-OOS returns are positive; locked-OOS was not used for variant selection, pruning, objective scoring, or parameter fitting. Calendar/date rules remain rejected.

Fresh verification passed on 2026-05-19 UTC: artifact assertion passed (`798` models, `2394` metric rows, `56` promotable); 10bps retune tests `15 passed`; top-seed/hybrid split tests `19 passed`; live/liquidation/state tests `49 passed`; full pytest `1360 passed`; `ruff check .`, `compileall`, and diff checks passed. Research history/source ledger not regenerated: this session reused the already-refreshed 2026-05-17 current-tail data and same Alpha Zoo artifact lineage, adding a same-lineage 10bps retune/validation bundle rather than a new market-data source family or chronology refresh.

---

## 2026-05-19 KST — Alpha Zoo top-seed Hybrid v3.5/v3.6 cost validation

Recomputed the current Alpha Zoo top-bucket plus filtered top-3 seed-union from the latest live-notional/risk-aligned candidate CSV, then generated a separate research-only Hybrid v3.5/v3.6 cost-validation bundle. The deduped seed universe now has `16` rows across `fast_residual`, `quality_single_pair`, `high_confidence_single_pair`, and `high_confidence_long_only`. The bundle reports every individual seed, `hybrid_v3_5_seed_union`, `hybrid_v3_6_seed_union`, `reference_fast_residual_7x_0p15`, and `reference_strict_zero_fast_residual_6x_0p10` at round-trip slippage/fee `5bps` and `10bps` over `train`, `validation`, and `locked_oos` splits: `(16 + 2 + 2) * 2 * 3 = 120` metric rows.

Locked-OOS remains a gate/report split after model freeze for the hybrid objective/pruning/parameter-fitting path: the artifact audit records `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`, and `uses_locked_oos_for_selection=false`. Because the requested seed basket is assembled from current leaderboard buckets, the artifact also labels the seed basket as a post-hoc research basket rather than a deployable live-selection rule. No real-money execution was attempted.

Key cost outcomes:

- `hybrid_v3_5_seed_union`, `5bps`: train `+49.19%` / MDD `30.09%`; validation `+21.16%` / MDD `12.33%`; locked-OOS `+3.29%` / MDD `15.39%`; liquidation `0`; account wipeout `0`; locked-OOS gate `true`.
- `hybrid_v3_6_seed_union`, `5bps`: train `-7.98%`; validation `+8.21%`; locked-OOS `-2.90%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- `hybrid_v3_5_seed_union`, `10bps`: train `+47.75%`; validation `+18.91%`; locked-OOS `-2.82%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- `hybrid_v3_6_seed_union`, `10bps`: train `-9.07%`; validation `-7.11%`; locked-OOS `-6.22%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- References: `fast_residual 7x/0.15` locked-OOS is `+7.63%` at `5bps` and `-12.44%` at `10bps`; strict zero `fast_residual 6x/0.10` locked-OOS is `+4.59%` at `5bps` and `-7.05%` at `10bps`.
- Best locked-OOS individual seed in the bundle is `alpha_zoo_high_confidence_single_pair 7x/0.2`: `+11.03%` / MDD `14.02%` at `5bps`, and `+3.02%` / MDD `16.87%` at `10bps`. Isolated liquidation losses are included in equity/MDD; the bundle's max split liquidation count is `9` on one high-leverage seed train split, and all account-wipeout counts are `0`.

Artifacts:

- Runner: `scripts/research/run_alpha_zoo_top_seed_hybrid_v35_v36_cost_validation.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/alpha_zoo_top_seed_hybrid_cost_validation_latest.json`
- Main Markdown/full metric table: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/alpha_zoo_top_seed_hybrid_cost_validation_latest.md`
- Seed selection CSV/JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/seed_selection_latest.csv`, `seed_selection_latest.json`
- Model metrics CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/model_cost_metrics_latest.csv`
- Hybrid weights CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/hybrid_weights_latest.csv`
- Generation log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_20260519T093343Z.log`
- Final verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_final_20260519T100131Z.log`
- Post-deslop verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_post_deslop_20260519T100507Z.log`

Fresh local verification passed on 2026-05-19 UTC: artifact assertion passed (`seed_count=16`, `metric_rows=120`, max liquidation count `9`, account wipeout max `0`); new Alpha Zoo top-seed tests `5 passed`; hybrid/common split tests `14 passed`; live/liquidation/state tests `46 passed`; full pytest `1345 passed`; ruff, compileall, and `git diff --check` all passed. Post-deslop verification re-ran after narrowing broad exception handling in the new runner: artifact assertion passed; new tests `5 passed`; full pytest `1345 passed`; ruff, compileall, and `git diff --check` passed.

Research history/source ledger not regenerated: this session reused the already-refreshed 2026-05-17 current-tail data and the 2026-05-18 live-notional/risk-aligned Alpha Zoo artifact family; it added a same-lineage research-only cost-validation bundle, not a new market-data source family or global chronology refresh.

---

## 2026-05-18 KST — Plan for Alpha Zoo top-seed hybrid v3.5/v3.6 cost validation

Prepared a next-session plan to test whether the current Alpha Zoo leaderboard can be improved by building Hybrid v3.5/v3.6 portfolios from the top individual candidate streams rather than from the prior fixed-input `A0 + P0 + E0 + S1 + S2 + S3 + S4` universe. The plan is saved at `.omx/plans/plan-alpha-zoo-hybrid-v35-v36-cost-validation-20260518.md`.

Current seed-selection snapshot is based on `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/`: leverage grid `1x..20x`, allocation grid `0.03,0.05,0.075,0.10,0.125,0.15,0.175,0.20`, `1600` rows, `113` live-promotion rows, and `50` rows passing the train-dominant/val-good/OOS-good filter. The next session must recompute the buckets from the latest CSV before running.

Planned seed universe uses top-3 bucket union from live OOS return/Sharpe/Sortino/smart Sortino/Calmar, full compound, and filtered balanced/validation-return/OOS-return/OOS-Calmar lists. The current deduped snapshot has `18` rows spanning `fast_residual`, `quality_single_pair`, `high_confidence_single_pair`, and `high_confidence_long_only` configurations. Known duplicates such as `fast_residual 7x/0.15` and `6x/0.175` are intentional because they have the same notional/equity (`105%`) but different isolated margin semantics.

The required cost validation is explicitly round-trip slippage/fee `5bps = 0.05%` and `10bps = 0.10%`. For every individual seed plus `hybrid_v3_5_seed_union` and `hybrid_v3_6_seed_union`, the next run must report train/validation/locked-OOS total return, MDD, Sharpe, Sortino, smart Sortino, Calmar, trade/event count, liquidation count, account-wipeout count, and minimum margin buffer. Locked-OOS remains gate/report-only and must not enter objective, pruning, parameter fitting, or seed/hybrid selection.

Recommended output directory for the next run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/`. Real-money execution remains prohibited. Research history/source ledger does not need regeneration for this planning-only update because it introduces no new data source family or market-data chronology; the execution session must revisit that decision if it refreshes data or changes source lineage.

---

## 2026-05-18 KST — Live/replay notional-risk aligned Alpha Zoo 7x contract

Implemented the live sizing contract required for the latest high-leverage winner. Existing strategies keep backward-compatible `legacy_notional_cap` behavior by default, while the promoted Alpha Zoo lane now opts into explicit `isolated_margin_fraction`: `target_allocation=0.15` means `15%` isolated margin/equity and, with `leverage=7`, `105%` notional/equity. The legacy fixed-dollar `max_order_value` cap is disabled for this lane (`0.0`) and replaced by equity-scaled notional caps: per-order `110%`, symbol `110%`, total notional `220%`; any positive `max_order_value` remains an explicit emergency ceiling.

Latest-data train+validation retune kept `CryptoFxAlphaZooStateStrategy / alpha_zoo_fast_residual / isolated 7x / allocation 0.15`. The raw grid also found a notional-equivalent `6x/0.175` row with the same train+validation score; a documented incumbent tie-breaker selected the requested `7x/0.15` contract without using locked-OOS for scoring. Locked-OOS remains freeze-after-selection gate/report-only. No real-money execution was attempted.

Selected high-performance lane:

- Sizing mode: `isolated_margin_fraction`.
- Exposure: notional/equity `105.00%`; isolated margin/equity `15.00%`.
- Train: `+1.4941%` return / `59.9354%` MDD.
- Validation: `+44.9483%` return / `13.7796%` MDD.
- Locked-OOS no-cost: `+30.5357%` return / liquidation-inclusive MDD `11.3027%`; Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.083139`, Calmar `2.701628`, trades `391`, locked-OOS liquidation `0`, total account wipeout `0`.
- Paper-equivalent parity fixture: equity `$10,000`, price `$100`, `target_allocation=0.15`, `leverage=7` -> replay expected notional `$10,500`, live quantity `105.0`, live notional `$10,500`, absolute diff `0.0`, risk check `Passed`.
- Preflight: `recommended_action=paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`.

Cost-stressed locked-OOS diagnostics are separate from the no-cost headline. Round-trip slippage/fee: `1bps` `+25.2882%` / MDD `11.9349%`; `3bps` `+15.4160%` / MDD `13.1860%`; `5bps` `+6.3199%` / MDD `15.4731%`; `10bps` `-13.4130%` / MDD `24.2149%`; `20bps` `-42.5899%` / MDD `44.8361%`. Funding drag: `1bps/day` `+29.9911%`; `2bps/day` `+29.4486%`; `5bps/day` `+27.8349%`; `10bps/day` `+25.1897%`; `20bps/day` `+20.0619%`.

Strict zero-liquidation lane is kept separate: same calibrated Alpha Zoo parameters at strict `6x` / `10%` allocation show locked-OOS `+16.7783%` return, MDD `6.5951%`, Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.175137`, Calmar `2.544032`, liquidation `0`, account wipeout `0`, minimum margin buffer `9150.924760`.

Artifacts:

- Main aligned JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.json`
- Main aligned Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.md`
- Live decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_alpha_zoo_notional_risk_aligned_decision_latest.json`
- Preflight artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_readiness_preflight_notional_risk_aligned_latest.json`
- Candidate CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
- Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/local_verification_live_notional_risk_alignment_20260518T113100Z.log`

Fresh local verification passed on 2026-05-18 UTC: required live/Alpha Zoo suite `32 passed`, required moonshot validation suite `74 passed`, full pytest `1340 passed`, ruff, compileall, `git diff --check`, and `git diff --cached --check` all passed.

Research history/source ledger not regenerated: this session used the already-refreshed 2026-05-17 current-tail data and added a live-sizing contract/validation artifact bundle, not a new market-data source family or global chronology refresh.

---

## 2026-05-17 KST — Live notional/risk alignment plan for Alpha Zoo 7x lane

After reviewing the latest `alpha_zoo_fast_residual` 7x isolated live decision, the key remaining live-readiness issue is not the signal hypothesis but the sizing contract. The no-cost research replay models account return as `allocation_fraction * leverage * gross_return`, so the current winner `allocation_fraction=0.15`, `leverage=7` represents approximately `105%` notional exposure with about `15%` isolated margin. The live runtime currently treats `target_allocation` as a notional cap, so the same `0.15` can size closer to `15%` notional exposure. That mismatch must be fixed before expecting live/paper results to match replay performance.

Static `max_order_value=5000.0` is also identified as a legacy fixed-dollar guardrail, not a strategy-derived cap. For this futures lane it must be replaced/subordinated by an equity-scaled and leverage-aware cap, with any absolute dollar cap treated only as an explicit emergency ceiling. Otherwise a $10,000 account targeting the replay-intended `0.15 * 7 = 1.05` notional/equity can be silently truncated.

Planning artifacts created for the next implementation session:

- PRD: `.omx/plans/prd-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Test spec: `.omx/plans/test-spec-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Handoff: `docs/session_handoff_20260517_live_notional_risk_alignment.md`

Next-session acceptance target: add an explicit sizing mode such as `isolated_margin_fraction` vs `notional_fraction`, preserve backward compatibility, retune leverage/allocation using train+validation only, include liquidation losses in equity/MDD for any isolated high-performance lane, keep strict zero-liquidation lane separate, add no-cost and cost-stressed reports, prove paper-equivalent live sizing parity, and avoid real-money execution until a separate credentialed real preflight is green and explicitly authorized.

Research history/source ledger was not regenerated for this planning-only update because it introduces no new data-source family or new market-data chronology artifact. The next implementation session must revisit the global research history/source ledger if it refreshes data beyond the current tail or adds new source families.

---

## 2026-05-17 KST — Latest-data validation-to-March high-leverage Alpha Zoo replay

Refreshed the five-symbol raw-first Binance Futures data tail to cutoff `2026-05-17T10:59:59Z`, compacted the OHLCV WAL to monthly parquet, and rebuilt the joined hourly current-tail panel `var/cache/profit_moonshot_fresh_start/joined_panel_76f825ffea81c04f2fe41fbf.parquet` with actual max timestamp `2026-05-17T10:00:00Z`.

Updated split authority:

- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (`8760` hourly timestamps, `43800` rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-03-31T23:00:00Z` (`2156` hourly timestamps, `10780` rows)
- locked-OOS: `2026-04-01T00:00:00Z` ~ `2026-05-17T10:00:00Z` (`1115` hourly timestamps, `5575` rows)

High-leverage tuning used only train+validation for candidate ranking. Locked-OOS was applied after candidate freeze as gate/report-only. The high-leverage lane assumes isolated per-position margin; if a path breaches liquidation threshold, the trade loss is capped to the configured isolated allocation fraction, and account-wipeout count must remain zero.

Result:

- Top train/validation score was `alpha_zoo_conservative_exit`/carry-forward at `9x` with `12.5%` allocation, but it failed locked-OOS gate (`-0.6029%` OOS return, non-positive Calmar).
- First pre-frozen candidate to pass locked-OOS gate: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_fast_residual` / isolated `7x` / `15%` allocation.
- Promoted high-leverage candidate metrics: train `+1.4941%` / MDD `59.9354%`; validation `+44.9483%` / MDD `13.7796%`; locked-OOS `+30.5357%` / MDD `11.3027%`; locked-OOS Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.083139`, Calmar `2.701628`, trades `391`, liquidation count `0`, account-wipeout count `0`, `live_promotion_possible=true`.
- Strict zero-liquidation integer lane at `10%` allocation for the same `alpha_zoo_fast_residual` params still promotes `6x`: locked-OOS `+16.7783%`, MDD `6.5951%`, liquidation `0`, min buffer `9150.924760`, positive Sharpe/Sortino/smart Sortino/Calmar.
- Runtime leverage validation cap was raised from `6x` to `20x` so the isolated `7x` decision artifact can pass live configuration validation; real execution remains operator/credential gated.
- Peak RSS: data refresh `4467.1172 MiB`, replay `736.1953 MiB`, both under 8 GiB.

Artifacts:

- Runner: `scripts/research/run_alpha_zoo_validation_march_high_leverage.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json`
- Main Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.md`
- Candidate CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
- Live decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json`
- Data refresh report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/data_refresh_latest.json`

Research history/source ledger not regenerated: this is a same-source tail refresh and session-scoped Alpha Zoo replay, not a new global source family or chronology ledger change.

Latest-data March-validation verification passed on 2026-05-17 UTC: live/source validation suite `27 passed`, required Alpha Zoo suite `24 passed`, required moonshot validation suite `74 passed`, full pytest `1329 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/local_verification_validation_march_high_leverage_20260517T120700Z.log`.

Live-readiness preflight for the isolated `7x` decision artifact also passed for paper/testnet mode with a supplied paper Postgres DSN placeholder and freshness threshold override: `recommended_action=paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`. Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_readiness_preflight_7x_latest.json`.

Post-staging CSV LF/runner-lineterminator verification re-ran clean: full pytest `1329 passed in 55.01s`, required Alpha Zoo suite `24 passed`, moonshot suite `74 passed`, live/source targeted suite `27 passed`, ruff/compileall/diff checks passed.

---

## 2026-05-17 Addendum — strict 6x live-wiring final verification

Final hardening synced live decision exchange overrides into both `LiveConfig.EXCHANGE` and derived fields (`EXCHANGE_ID`, `MARKET_TYPE`, `POSITION_MODE`, `MARGIN_MODE`, `LEVERAGE`) before validation/trader construction and made unknown strategy-class live decisions fail closed. The decision artifact preflight reports `paper_run_allowed` and `ready_for_paper=true` while keeping real execution operator/credential gated. Fresh verification passed: targeted live/common-split suite `46 passed`, live-readiness/parity suite `11 passed`, required Alpha Zoo suite `24 passed`, required moonshot validation suite `74 passed`, full pytest `1328 passed`, ruff/compileall/diff checks passed.

---

## 2026-05-17 Addendum — strict 6x Alpha Zoo live wiring check

The common-split #1 (`CryptoFxAlphaZooStateStrategy` / `alpha_zoo_conservative_exit` / strict `6x`) is now explicitly represented as a live decision artifact rather than only as a replay result. The live decision path maps Alpha Zoo references to `CryptoFxAlphaZooStateStrategy`, passes train+validation calibrated edges and selected conservative-exit params to `LiveTrader`, and applies 3600s MARKET_WINDOW/cadence plus isolated `6x` and `target_allocation=0.10` overrides.

Live-equivalent tests were added for selection inference, decision override propagation, live CLI parameter injection, runtime leverage validation (`6x` allowed, `>6x` rejected), and MARKET_WINDOW-vs-MARKET_BATCH strategy parity for hourly Alpha Zoo decisions. The strict 6x replay evidence remains: locked-OOS return `+20.512682%`, MDD `6.788365%`, liquidation `0`, and positive min margin buffer. Real live fills/slippage/funding remain execution-environment risks and are not asserted identical to replay.

Artifacts:

- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.md`

---

## 2026-05-17 KST — Integrated margin replay addendum for fixed-input hybrid v3.5/v3.6

Supersedes the earlier same-day note that marked fixed-input hybrid v3.5/v3.6 non-promotable solely because liquidation count/minimum margin buffer were `not_replayed`. Added a mixed-allocator integrated margin replay for the frozen A0+P0+E0+S1+S2+S3+S4 hybrid return streams. The replay uses post-freeze v3.5/v3.6 allocator weights, maps each fixed stream to its source gross-notional fraction, and evaluates one cross-margin account path. It is not used by Optuna objective, pruning, selection, or tie-break; locked-OOS remains gate/report-only after candidate freeze.

Updated common-split hybrid live-gate result:

| Candidate | Split | Return | MDD | Sharpe | Sortino | Smart Sortino | Calmar | Active hours | Liquidations | Min margin buffer | Deployable success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hybrid_v3_5_optuna | train | +47.7257% | 11.0421% | 2.705685 | 2.932093 | 2.640523 | 4.322148 | 7514 | 0 | 9932.438663 | true |
| hybrid_v3_5_optuna | validation | +13.3102% | 2.2622% | 5.390597 | 7.610005 | 7.441656 | 51.558258 | 1302 | 0 | 14594.054033 | true |
| hybrid_v3_5_optuna | locked-OOS | +8.5233% | 1.7654% | 5.259028 | 7.316663 | 7.189734 | 32.173151 | 1467 | 0 | 16587.499982 | true |
| hybrid_v3_6_optuna | train | +49.5204% | 7.6947% | 2.897597 | 2.999204 | 2.784914 | 6.435678 | 7514 | 0 | 9847.514685 | true |
| hybrid_v3_6_optuna | validation | +12.4946% | 1.5354% | 7.002337 | 8.680826 | 8.549560 | 69.800312 | 1302 | 0 | 14690.924128 | true |
| hybrid_v3_6_optuna | locked-OOS | +7.7916% | 1.7491% | 4.859674 | 5.991026 | 5.888040 | 29.199963 | 1467 | 0 | 16664.270300 | true |

Decision update: fixed-input hybrid v3.5/v3.6 are now live-promotion-capable under the integrated margin gate, but they do **not** beat the common-split Alpha Zoo strict 6x lane on locked-OOS return (`+20.5127%` for Alpha Zoo vs `+8.5233%` v3.5 and `+7.7916%` v3.6). Alpha Zoo strict 6x remains the common-split performance leader; hybrid v3.5/v3.6 become lower-return, lower-MDD deployable alternatives rather than blocked diagnostics. Research history/source ledger still not regenerated: this addendum adds a local validation/replay artifact and no new global source family.

Updated artifacts:

- Runner update: `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- Common report JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- Common report Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- Hybrid stage JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.json`
- Hybrid stage Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.md`

Integrated margin addendum verification passed on 2026-05-17 UTC: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1321 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_integrated_margin_20260517T071937Z.log`.

---

## 2026-05-17 KST — Common-split Alpha Zoo vs fixed-input hybrid v3.5/v3.6 fair comparison

Re-ran the Alpha Zoo best and fixed-input hybrid v3.5/v3.6 candidates on one explicit common split from baseline `private/main@80a557c133930f51748ec20c4e582aa0d6f678de`. The prior Alpha Zoo split is now **historical only** in this comparison; it is not used for common-split selection, promotion, or tie-breaks.

Common split authority:

- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (`8760` hourly timestamps, `43800` rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z` (`1416` hourly timestamps, `7080` rows)
- locked-OOS: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z` (`1593` hourly timestamps, `7965` rows)

Alpha Zoo result on common split:

- Historical old split strict 6x is preserved only as reference: old split periods were train `2025-01-01T00:00:00Z`~`2025-10-22T04:00:00Z`, validation `2025-10-22T05:00:00Z`~`2026-01-28T06:00:00Z`, locked-OOS `2026-01-28T07:00:00Z`~`2026-05-06T23:00:00Z`; old locked-OOS return was `+41.0967%`, but it is not comparable for new selection.
- Common-split carry-forward of the old selected `alpha_zoo_conservative_exit` and common-split reselected grid both select/retain `alpha_zoo_conservative_exit` and produce the same strict 6x replay: train `+114.4617%` / MDD `29.5651%`; validation `+19.9681%` / MDD `13.6667%`; locked-OOS `+20.5127%` / MDD `6.7884%`; locked-OOS Sharpe `1.772136`, Sortino `2.578776`, smart Sortino `2.414847`, Calmar `3.021741`, trades `365`, liquidation `0`, min margin buffer `9643.447509`; `deployable_success=true` under the strict lane.
- Strict integer leverage 1x..6x on the common split keeps `6x` as the highest deployable integer: OOS return `+20.5127%`, MDD `6.7884%`, liquidation `0`, min buffer `9049.125962`. Return/MDD remains diagnostic/report-only.
- Diagnostic nonfatal 5x/6x lane remains separate from live promotion: 5x and 6x both have `promotion_allowed=false` even though their diagnostic replay has zero liquidations.

Fixed-input hybrid v3.5/v3.6 common-split Optuna result:

- Input universe remains exactly `A0 + P0 + E0 + S1 + S2 + S3 + S4`; no literal hybrid/hybrid-online/hybrid-tuning output is used as an input.
- v3.5 uses fixed default + rolling weights/high-vol boost + Optuna; v3.6 is v3.5 core plus online dynamic default-candidate refresh only. No other knob is OOS-adaptive.
- Optuna ran with `n_trials=80`, `seed=42`, and train+validation objective/selection only. Audit found no locked-OOS use for objective, pruning, selection, or tie-break (`violation=false`; calibration locked-OOS records `0`).
- v3.5 common-split locked-OOS: return `+8.5233%`, MDD `1.7654%`, Sharpe `5.259028`, Sortino `7.316663`, smart Sortino `7.189734`, Calmar `32.173151`; `deployable_success=false` because dedicated integrated margin replay is missing.
- v3.6 common-split locked-OOS: return `+7.7916%`, MDD `1.7491%`, Sharpe `4.859674`, Sortino `5.991026`, smart Sortino `5.888040`, Calmar `29.199963`; `deployable_success=false` for the same margin-replay reason.

Decision: on the fair common split, Alpha Zoo strict 6x remains the only live-promotion-capable lane. Hybrid v3.5/v3.6 are useful diagnostics and beat the invalid current-base OOS return reference, but they are not live-promotable until an integrated strict liquidation/margin replay supplies liquidation count and minimum margin buffer. Peak RSS was `769.1015625 MiB` (<8 GiB). Research history/source ledger was not regenerated because this session added a session-scoped comparison artifact only and did not introduce a new global source family or chronology ledger.

Artifacts:

- Runner: `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py`
- Regression tests: `tests/test_common_split_alpha_zoo_hybrid_v35_v36.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- Main Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- Alpha stage artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/alpha_zoo_common_split/`
- Hybrid stage artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/`

Verification for the common-split run passed on 2026-05-17 UTC: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1319 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_20260517T054257Z.log`.
Post-deslop verification re-ran the full required command set and passed again: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1319 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_post_deslop_20260517T054855Z.log`.

---

## 2026-05-17 KST — External Hybrid v3.5/v3.6 method applied to fixed A0+P0+E0+S1+S2+S3+S4 inputs

Operator correction incorporated: `/home/hoky/DeepLearning/ensemble_strategies` defines v3.6 as **v3.5 core plus online dynamic default-model/candidate refresh**, not a new candidate universe and not a hybrid-inside-hybrid stack. Evidence checked directly in the external method source:

- `models/hybrid/v3_5.py`: lines 1-8 describe adaptive weights + Optuna; lines 31-49 define the Optuna-tuned defaults/search candidates; lines 311-328 learn train/warmup parameters; lines 397-420 keep the default model fixed while applying rolling weights/high-vol boost.
- `models/hybrid/v3_6.py`: lines 1-9 state the v3.6 delta: Step A `default_model` is dynamically updated online by rolling MAPE while v3.5 defaults/Optuna results are retained; lines 29-30 reuse v3.5 learning; lines 87-105 learn the same parameters; lines 178-223 dynamically refresh only the default model and otherwise use the v3.5 weight/high-vol/bias structure.
- `scripts/compare_v35_v36.py`: lines 1-5 summarize the same delta; lines 49-55 compare v3.5 fixed default vs v3.6 dynamic default.

Repo adaptation now uses fixed input universe `A0 + P0 + E0 + S1 + S2 + S3 + S4` only. No literal prior hybrid/hybrid-online/hybrid-tuning output is an input; no calendar/month/day/hour entry rule is introduced; Optuna objective/selection uses train+validation only. Locked-OOS remains gate/report-only after candidate freeze.

Split periods for this fixed-input experiment:

- locked_oos: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z` (1593 rows)
- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (8760 rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z` (1416 rows)

Candidate input split metrics from the reconstructed return-stream experiment:

| Input | Split | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades/active hours | Liquidations | Min buffer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A0 | train | +114.46% | +28.27% | 4.049305 | 1.979653 | 1.131499 | 0.882143 | 4.049305 | 1812 | not_replayed | not_replayed |
| A0 | validation | +19.97% | +13.67% | 1.461080 | 2.668444 | 1.654321 | 1.455414 | 15.249889 | 292 | not_replayed | not_replayed |
| A0 | locked_oos | +20.51% | +6.79% | 3.021741 | 3.993463 | 2.711715 | 2.539336 | 26.368611 | 313 | not_replayed | not_replayed |
| P0 | train | +64.11% | +17.75% | 3.610917 | 1.854506 | 1.719569 | 1.460317 | 3.610917 | 7173 | not_replayed | not_replayed |
| P0 | validation | +26.68% | +5.37% | 4.966077 | 6.497669 | 7.032072 | 6.673577 | 61.776012 | 1262 | not_replayed | not_replayed |
| P0 | locked_oos | +4.52% | +6.97% | 0.647757 | 1.217684 | 1.227542 | 1.147552 | 3.943449 | 1429 | not_replayed | not_replayed |
| E0 | train | +32.17% | +7.89% | 4.077794 | 2.121676 | 1.710879 | 1.585758 | 4.077794 | 5314 | not_replayed | not_replayed |
| E0 | validation | +11.62% | +2.79% | 4.162712 | 5.240997 | 4.732774 | 4.604245 | 34.893482 | 838 | not_replayed | not_replayed |
| E0 | locked_oos | +2.90% | +2.36% | 1.226822 | 1.780358 | 1.646460 | 1.608436 | 7.201675 | 1175 | not_replayed | not_replayed |
| S1 | train | +8.04% | +2.31% | 3.485607 | 2.064400 | 1.661117 | 1.623648 | 3.485607 | 5314 | not_replayed | not_replayed |
| S1 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S1 | locked_oos | +0.73% | +0.60% | 1.200019 | 1.748046 | 1.615190 | 1.605489 | 6.707520 | 1175 | not_replayed | not_replayed |
| S2 | train | +7.07% | +2.86% | 2.470000 | 1.818764 | 1.450761 | 1.410400 | 2.470000 | 5309 | not_replayed | not_replayed |
| S2 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S2 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S3 | train | +6.90% | +2.86% | 2.412668 | 1.779218 | 1.417786 | 1.378343 | 2.412668 | 5297 | not_replayed | not_replayed |
| S3 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S3 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S4 | train | +3.51% | +3.27% | 1.072977 | 1.014829 | 0.764783 | 0.740531 | 1.072977 | 4885 | not_replayed | not_replayed |
| S4 | validation | +1.52% | +0.90% | 1.684443 | 3.832281 | 3.193221 | 3.164658 | 10.840367 | 748 | not_replayed | not_replayed |
| S4 | locked_oos | +0.62% | +0.81% | 0.758295 | 1.514867 | 1.299905 | 1.289404 | 4.228257 | 1062 | not_replayed | not_replayed |

Hybrid Optuna outputs using the corrected external concept:

| Candidate | Train/validation score | Train return | Validation return | Locked-OOS return | Locked-OOS MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Liquidations | Min buffer | Deployable success | Rejection reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| hybrid_v3_5_optuna | 70.585 | +47.73% | +13.31% | +8.52% | +1.77% | 4.827936 | 5.259028 | 7.316663 | 7.189734 | 32.173151 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |
| hybrid_v3_6_optuna | 85.548 | +49.52% | +12.49% | +7.79% | +1.75% | 4.454705 | 4.859674 | 5.991026 | 5.888040 | 29.199963 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |

Alpha Zoo strict 6x comparison anchor remains superior for live promotion:

| Candidate | Split | Period start | Period end | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades | Liquidations | Min buffer |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alpha Zoo strict 6x | train | `2025-01-01T00:00:00` | `2025-10-19T13:00:00` | +68.88% | +29.57% | 2.329914 | 1.569139 | 1.919776 | 1.481707 | 2.329914 | 1779 | 0 | 9049.125962 |
| Alpha Zoo strict 6x | validation | `2025-10-22T05:00:00` | `2026-01-28T06:00:00` | +30.12% | +9.56% | 3.150734 | 1.552041 | 2.095744 | 1.912882 | 3.150734 | 524 | 0 | 9527.695928 |
| Alpha Zoo strict 6x | locked_oos | `2026-01-28T07:00:00` | `2026-05-06T23:00:00` | +41.10% | +13.67% | 3.007073 | 2.143209 | 2.841936 | 2.500237 | 3.007073 | 540 | 0 | 9572.449083 |

Decision: the fixed-input v3.5/v3.6 Optuna experiments are useful diagnostics but **not live-promotable** yet because the mixed A0/P0/E0/S-sleeve allocator has no dedicated integrated margin replay, so liquidation count and minimum margin buffer are `not_replayed`. Both fixed-input hybrids satisfy train/validation-only selection and have locked-OOS MDD below 25%, but Alpha Zoo strict 6x still dominates live promotion with locked-OOS return `+41.0967%`, zero liquidations, and positive min buffer. The fixed-input hybrids remain report/reference until a dedicated strict zero-liquidation margin replay is implemented.

Artifacts:

- Script: `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- Regression test: `tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.json`
- Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.md`
- Timestamped latest run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_20260516T172901Z.json`
- Peak RSS: `353.754 MiB` (<8 GiB)

Research history/source ledger was not regenerated: this run reused existing repo-local market/research artifacts and added a session-scoped method-adaptation report; it did not introduce a new global data-source family or chronology ledger.

---

## 2026-05-16 KST — Hybrid v3.5/v3.6 and Optuna comparison against Alpha Zoo strict 6x

Completed a repo-wide inventory and policy audit for hybrid v3.5/v3.6, hybrid Optuna, tuning/optimization, candidate-hybrid, calendar Optuna, and fresh-portfolio optimization artifacts against the preserved private/main baseline `1c6816fced44d277f6c7112934c9dded65ba710f`. Corrections on 2026-05-17 KST: the **comparison core excludes calendar/current-base-derived rows and literal hybrid/hybrid-online/hybrid-tuning rows before ranking**. `portfolio`, `allocator`, `meta`, `static_blend`, and `leverage_sweep` labels are not exclusion triggers by themselves; calendar/current-base and literal-hybrid rows are retained only in quarantine/reference ledgers.

Decision: **only** `CryptoFxAlphaZooStateStrategy / alpha_zoo_conservative_exit / strict 6x` remains live-promotion possible. Alpha Zoo selection is train/validation-only, locked-OOS is gate/report-only after candidate freeze, and strict 6x passes zero-liquidation, positive margin buffer, OOS MDD <=25%, OOS return above the invalid current-base reference, positive Sharpe/Sortino/smart Sortino/Calmar, and memory <8 GiB.

Alpha Zoo strict 6x split evidence:

- train: 2025-01-01T00:00:00 → 2025-10-19T13:00:00, return 68.8842%, MDD 29.5651%, Sharpe 1.569139, Sortino 1.919776, smart Sortino 1.481707, Calmar 2.329914, trades 1779, liq 0, min buffer 9049.125962
- validation: 2025-10-22T05:00:00 → 2026-01-28T06:00:00, return 30.1195%, MDD 9.5595%, Sharpe 1.552041, Sortino 2.095744, smart Sortino 1.912882, Calmar 3.150734, trades 524, liq 0, min buffer 9527.695928
- locked_oos: 2026-01-28T07:00:00 → 2026-05-06T23:00:00, return 41.0967%, MDD 13.6667%, Sharpe 2.143209, Sortino 2.841936, smart Sortino 2.500237, Calmar 3.007073, trades 540, liq 0, min buffer 9572.449083

Hybrid/Optuna conclusions:

- Hybrid v3.5/v3.6 rows are **not strict-core rows** after correction because their own provenance is literal hybrid / hybrid-online / hybrid-final-selection. `portfolio`, `allocator`, `meta`, `static_blend`, and `leverage_sweep` labels are not exclusion triggers by themselves; they are only evidence/context when the top-level row is already literal hybrid.
- Hybrid Optuna `live_guarded` and `train_aware_guarded` are same-family hybrid optimizer outputs and are also live-promotion invalid because those objective profiles consume OOS metrics; good-looking OOS values remain diagnostic/reference only.
- Hybrid/tuning `locked_train_val` policy shape is cleaner (`oos_is_objective_input=false`) but it is still a same-family hybrid-online tuning output, not an atomic-source hybrid candidate, and remains quarantine/reference only.
- State-distilled fresh portfolio tuning is **not literal hybrid** and is restored to the non-calendar comparison core. It remains diagnostic/non-promotable because strict liquidation/margin replay is missing and Alpha Zoo strict 6x dominates the locked-OOS/live-promotion gates.
- Calendar Optuna and calendar/current-base-dependent fresh/candidate-hybrid rows are **not part of the strict core**. They remain in a separate quarantine/reference ledger because calendar month/day/hour rules and current-base tuple dependencies are invalid before any ranking.
- Candidate-hybrid had strong OOS metrics but is excluded due calendar/current-base-source dependency, validation liquidation count `1`, not the hybrid-inside-hybrid rule.

Strict integer recheck 1x..6x was rerun in the comparison directory. Highest strict integer remains `6.0x`: OOS return `41.0967%`, MDD `13.6667%`, return/MDD diagnostic `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, liquidation `0`, min buffer `9049.125962`. The separate diagnostic 5x/6x lane is preserved with `promotion_allowed=false`.

Artifacts:

- Full JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.json`
- Corrected non-calendar JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T000000Z_calendar_quarantine_corrected.json`
- Corrected literal-hybrid quarantine JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T015000KST_hybrid_only_quarantine_corrected.json`
- Full Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.md`
- Comparison-core split performance CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/candidate_split_performance_latest.csv`
- Literal nested-hybrid quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_nested_hybrid_same_family_quarantine_latest.csv`
- Calendar/current-base quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_calendar_current_base_quarantine_latest.csv`
- Inventory JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_inventory_latest.json`
- Prompt checklist audit: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/prompt_checklist_audit_latest.json`
- Strict integer recheck: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.json`

Non-calendar comparison core after corrections has `2` candidates (`crypto_fx_alpha_zoo_state_calibrated` plus the non-hybrid state-distilled portfolio diagnostic row); literal nested-hybrid quarantine has `8` candidates and calendar/current-base quarantine has `5`. Verification after literal-hybrid quarantine passed (`1308` full tests plus ruff/compileall/diff checks). Latest log `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/local_verification_hybrid_only_quarantine_20260517T020000KST.log`. Memory peak across inspected/replayed comparison artifacts: `1239.703125 MiB`, below 8 GiB. Research history/source ledger was **not regenerated** because this session did not add a new global data source family or chronology ledger; it reused existing repo-local artifacts and added a session-scoped comparison report.

---

## 2026-05-14 KST — Paper-forward diagnostics added

Added non-promotional diagnostics requested for the current leading `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_conservative_exit` / strict 6x candidate. The strategy/replay now records dominant factor-family metadata and exit-reason metadata, and the real-data replay summary includes locked-OOS PnL breakdowns plus cost sensitivity.

Locked-OOS diagnostic breakdown at 6x, 10% allocation:

- Regime: `neutral` `+41.0967%` across `540` trades. Direct FX OHLCV trading remains blocked; lagged FRED state remains context only.
- Symbol: SOL/USDT `+19.2672%` (`126`), BNB/USDT `+10.3167%` (`128`), TRX/USDT `+4.9566%` (`139`), ETH/USDT `+1.4843%` (`132`), BTC/USDT `+0.6807%` (`15`).
- Side: SHORT `+26.3040%` (`259`), LONG `+11.7120%` (`281`).
- Dominant factor family: crypto residual momentum `+30.1822%` (`184`), crypto residual reversal `+8.4241%` (`237`), volume/vwap pressure `-0.0370%` (`119`).
- Exit reason: score_exit `+41.0936%` (`526`), take_profit `+16.1976%` (`4`), end_of_sample `-0.0788%` (`2`), stop_loss `-13.8700%` (`8`).
- Slippage sensitivity, round-trip 0/2.5/5/10/20 bps: `+41.0967%`, `+30.1241%`, `+20.0034%`, `+2.0585%`, `-26.1930%`.
- Conservative funding drag sensitivity, 0/1/2/5/10 bps per day: `+41.0967%`, `+40.4210%`, `+39.7486%`, `+37.7505%`, `+34.4835%`.

Promotion policy unchanged: these diagnostics are `diagnostic_only`, `promotion_allowed=false`; the strict lane remains zero-liquidation + positive buffer + OOS return, MDD cap, and positive risk-metric policy with return/MDD report-only. Research history/source ledger was not regenerated because no new global source family was introduced.

---

## 2026-05-14 KST — Return/MDD diagnostic-only policy correction

Applied the latest operator clarification: OOS return/MDD is diagnostic/report-only, not a strict promotion hurdle. The current-base/calendar tuple remains `hypothesis_reference_only`, not a selection or promotion target; the strict deploy lane still requires zero liquidation, positive buffers, OOS MDD <=25%, OOS return beating the current-base reference, and positive risk metrics.

Real current-tail run under `crypto_fx_alpha_zoo_real_data_20260514`:

- Screen: `58,845` rows, `63` factors, `20` selected cards; direct FX OHLCV remains blocked and lagged FRED state is regime context only.
- Candidate outcome ledger: `67,259` rows; train+validation `45,311`; locked-OOS `21,948`.
- Edge calibration: train/validation-only physical filter; locked-OOS calibration records `0`; calibrated edge keys `12`.
- Replay selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only.
- Strict zero-liquidation lane promoted integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`.
- Deployable success: `true`; return/MDD `3.007073` vs current-base `6.916878` is reported as diagnostic-only and does not block promotion.
- Diagnostic 5x/6x lane is separate and non-promotional; 5x/6x both zero liquidation but `promotion_allowed=false` in that diagnostic lane.
- Peak RSS `626.7266 MiB` (<8 GiB).
- Artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/`.

Research history/source ledger was not regenerated because no new global source family was introduced; this pass reused the current-tail crypto cache and 20260512 lagged FRED external-state source.

---

## 2026-05-13 KST — Related state-distilled regime-boost overlay note

A separate research-only `StateDistilledRegimeBoostPortfolio` overlay was tested on the existing state-distilled seeds. This did not change the `CryptoFxAlphaZooStateStrategy` promotion policy or Alpha Zoo factor cards. The overlay reused real current-tail crypto data and lagged FRED external-risk state, kept calendar/current-base as hypothesis reference only, and kept locked-OOS gate/report-only after freeze.

Result: strict zero-liquidation/margin gates passed, but validation and locked-OOS return/risk-quality metrics failed, so no new deployable promotion came from the regime-boost overlay. Artifacts live under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/`.

---

---

## 2026-05-13 — Real-data Alpha Zoo calibrated replay result

- Ran real current-tail Alpha Zoo screen against `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet` with lagged FRED context; direct FX OHLCV trading stayed blocked because current-tail cache contains crypto OHLCV only.
- Factor/card validity passed fail-closed gates: `calendar_primary=false`, `uses_locked_oos_for_selection=false`, strategy validity pass.
- Candidate outcome ledger: `45160` rows; train+validation `30494`; locked-OOS `14666`.
- Edge calibration physically filtered to train/validation: input `45160`, calibration `30494`, locked-OOS calibration `0`, excluded locked-OOS `14666`.
- Replay grid selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only; locked-OOS remained hidden until candidate freeze.
- Strict zero-liquidation lane highest safe integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`.
- 2026-05-14 latest correction: return/MDD is diagnostic/report-only per operator clarification; OOS return must beat the invalid current-base reference, but OOS return/MDD does not block promotion.
- Deployable success is now `true` under the corrected policy: strict 6x has OOS return `41.0967%`, MDD `13.6667%`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, Calmar/return-MDD `3.007073`, zero liquidations, and positive buffers.
- Diagnostic 5x/6x lane remains non-promotional and separate: 5x/6x both zero liquidation in this approximate replay, but that lane is report-only.
- Peak RSS `512.711` MiB (<8 GiB).
- Artifacts: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_real_data_summary_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/edge_calibration_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/candidate_outcome_ledger_latest.jsonl`.
- Research history/source ledger not regenerated: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo artifacts.

---

---

## Static baseline context — 2026-05-12 origin

## Current answer: what should be used next?

Use `CryptoFxAlphaZooStateStrategy` as the next primary research path, but only after wiring it to real current-tail data and outcome calibration.

Do **not** use the high-performing current-base/calendar tuple as a live strategy. It remains useful as a teacher/hypothesis/reference because it explains a profitable market-state pattern, but it is calendar-primary and therefore invalid for live promotion.

## Latest green baseline

- Latest pushed green head: `e4b5f6942fe06ffff262502e40ff7dd4d7005323` on `private/main`.
- Prior implementation head for external-risk teacher pass: `fcc63f6c053c451152b0d780fa84ee91b5512f82`.
- GitHub Actions were green for both `ci` and `private-ci` after the handoff commit.

## What exists now

### 1. Alpha Zoo / outcome-calibration scaffold

Implemented files:

- `src/lumina_quant/alpha_zoo/operators.py`
- `src/lumina_quant/alpha_zoo/crypto_fx_factors.py`
- `src/lumina_quant/alpha_zoo/factor_card.py`
- `src/lumina_quant/research/triple_barrier.py`
- `src/lumina_quant/research/candidate_outcome_ledger.py`
- `src/lumina_quant/research/edge_calibration.py`
- `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`
- `scripts/research/run_crypto_fx_alpha_zoo_screen.py`
- `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
- `scripts/research/calibrate_crypto_fx_edges.py`

Current state: smoke/research scaffold only. It proves factor generation, train/validation-only selection metadata, triple-barrier labels, candidate ledger, edge calibration, and calibrated-entry strategy plumbing. It does **not** yet prove economic profitability on real current-tail data.

### 2. External-risk teacher pass

Implemented `scripts/research/fetch_profit_moonshot_external_state.py` to fetch lagged FRED daily state features from:

- DTWEXBGS
- VIXCLS
- DGS2
- DGS10
- DCOILWTICO

Features are lagged before joining to hourly crypto panels to avoid same-day lookahead.

Added non-calendar replay families:

- `calendar_teacher_state_similarity`
- `calendar_teacher_state_fade`
- `state_distilled_external_risk_filter`

Results:

- `calendar_teacher_state_similarity`: 972 specs, 0 survivor.
- `calendar_teacher_state_fade`: 324 specs, 0 survivor.
- `state_distilled_external_risk_filter`: 1728 specs, 565 train/validation-positive, 0 replay survivor under legacy shadow-MDD gate, peak RSS about 280 MiB.

### 3. Best valid strict state-distilled seed

Train/validation-selected strict candidate:

`fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` at 4x.

Metrics:

- Train: `+30.9030%`, MDD `10.2437%`, Sharpe `1.8484`, liquidation `0`.
- Validation: `+12.4704%`, MDD `2.5167%`, Sharpe `5.7588`, liquidation `0`.
- Locked-OOS: `+2.4852%`, MDD `2.5328%`, Sharpe `1.5096`, liquidation `0`.
- All split min margin buffers are positive.
- Strategy-validity passes.
- `deployable_success=false` because it does not beat the invalid current-base/calendar reference economics.

### 4. Calendar/current-base teacher status

The current-base/calendar tuple remains economically strong:

- Locked-OOS return `+6.4281%`.
- Return/MDD `6.9169`.
- Sharpe about `5.2024` in the liquidation-aware reference report.

But it is calendar-primary/fixed calendar behavior, so it is invalid for live promotion and must remain:

- `hypothesis_reference_only`
- not a selection target
- not a live strategy
- not a promoted candidate

## Next research objective

Convert the Alpha Zoo scaffold from synthetic smoke to real current-tail research:

1. Wire `run_crypto_fx_alpha_zoo_screen.py` to real crypto OHLCV/funding/OI/flow fields where available.
2. Add FX OHLCV regime fields if reliable; otherwise use lagged FRED risk-state as temporary regime context and explicitly record direct-FX trading as blocked.
3. Generate real factor cards with source coverage and `uses_locked_oos_for_selection=false`.
4. Produce triple-barrier candidate outcomes on train/validation.
5. Calibrate edge buckets with shrinkage and lower-confidence edge gating.
6. Replay `CryptoFxAlphaZooStateStrategy` plus state-distilled/residual seeds only if selected by train/validation.
7. Open locked-OOS after freeze only as gate/report.
8. Run strict zero-liquidation integer grid and separate diagnostic nonfatal 5x/6x lane.

## Required research-note practice

Every future profit-moonshot session must update research notes before final handoff:

- Update or supersede this file when Alpha Zoo real-data results change.
- Update `.omx/notepad.md` with concise conclusions and artifact paths.
- Update the active `.omx/plans/*` file with result status.
- Write or update `docs/session_handoff_*` for the session.
- If the work changes the global research inventory/source ledger, regenerate or explicitly update `docs/research_note/research_history.md` and the matching `var/reports/.../research_history/` artifacts, or document why regeneration was not required.

## Failure mode to avoid

Do not repeat the earlier loop of finding a good-looking single rule and then discovering it is calendar-primary or OOS-selected. The next path must be evidence-first:

`real factors → train/validation labels → calibrated edge → stateful replay → locked-OOS gate/report → strict liquidation validation`.

---

## 2026-06-05 — 69-asset clean OOS non-nested teacher-leaf rerun

### Objective

Re-check whether the previously high-performing nested hybrid/teacher structure can be made live-usable without nested hybrid material.  The rerun keeps the current live discipline:

- Universe/timeframes: 69 Binance research symbols, `30m,1h,2h,4h,6h,8h,12h,1d`.
- Cost: 10 bps slippage/cost proxy.
- Refit: monthly, calendar day 1 UTC.
- Selection inputs: expanding train + prior 2 calendar months validation only.
- Locked OOS: next calendar month, report-only after fold params/candidate selection are frozen.
- OOS span: `2025-09-01T00:00:00` → `2026-06-01T06:30:00`; 2026-06 is a partial fold because latest available data is `2026-06-01T06:30:00`.

### Implementation / hygiene changes

- Added `teacher_leaf_blend` as a non-leaf portfolio family that **does not use hybrid rows as material**. It rebuilds the old nested-teacher idea from clean leaf candidates only, with train/validation scoring and train/validation risk scaling.
- Kept `teacher_leaf_blend` out of downstream material by adding it to `_NON_LEAF_PORTFOLIO_FAMILIES`.
- Fixed `dynamic_conviction_switch` fold accounting: when the train/validation pool has no eligible aggressive/fallback branch, it now emits an explicit clean cash/no-position guard instead of silently omitting the fold. This avoids 9/10 fold aggregate distortion.
- Added an optional numba JIT speed path for `_debounced_state_signal`; sample verification matched the Python loop exactly. The full core-family run still took `25:52.79`, so the dominant cost remains per-symbol Optuna + pandas feature work, not just the signal loop.
- Added `--source-symbol-workers` as an experimental per-symbol threaded option. Tiny smoke passed, but `workers=4` did not materially improve the full 69-symbol path, so default remains sequential (`1`).

### Main artifact

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_teacher_leaf_blend_20260605/teacher_leaf_blend_corefamilies_full_v2_20260605.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_teacher_leaf_blend_20260605/teacher_leaf_blend_corefamilies_full_v2_20260605.md`

### Final clean-OOS result table

| Candidate | Clean | OOS comp | Ann approx | Max bar MDD | Monthly eq MDD | Sharpe | Sortino | Hit | Min OOS | Latest OOS | Val min/mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `profile_optuna:selected_optuna` | True | 10.05% | 12.18% | 19.20% | 13.58% | 0.50 | 1.16 | 5/10 | -9.30% | -0.73% | 25.71%/59.74% |
| `profile_optuna:hybrid_v3_5` | True | 9.06% | 10.97% | 19.20% | 13.58% | 0.47 | 1.04 | 5/10 | -9.30% | -0.23% | 25.71%/59.51% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | True | 8.61% | 10.42% | 10.08% | 4.62% | 0.76 | 2.38 | 5/10 | -2.65% | 0.00% | 0.00%/4.62% |
| `dynamic_conviction_switch:t0.85_strict_fallback` | True | 10.47% | 12.69% | 10.62% | 2.11% | 0.67 | 1.13 | 4/10 | -8.56% | -0.17% | 0.00%/9.22% |
| `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | True | 6.97% | 8.42% | 7.32% | 3.70% | 0.66 | 1.84 | 4/10 | -3.58% | -0.17% | 0.01%/6.18% |
| `teacher_leaf_blend:val_balanced_top12_cap20_mdd20` | True | -28.11% | -32.70% | 26.40% | 22.44% | -1.10 | -1.70 | 4/10 | -15.80% | -0.88% | 29.89%/51.65% |
| `teacher_leaf_blend:val_return_top8_cap35_mdd30` | True | -42.49% | -48.52% | 37.72% | 35.31% | -1.33 | -2.10 | 4/10 | -22.86% | -1.07% | 40.46%/85.50% |

### Interpretation

- The de-nested teacher idea **does not survive** full clean OOS.  It looked good in the latest 2-fold smoke (`val_return_top8` about +90.22% OOS comp), but across all 10 folds it is negative.  The failure pattern is classic validation-overfit/high-volatility-chasing: very high validation returns (`val_return_top8` mean validation about +85.50%) paired with large negative next-month OOS tails.
- Therefore, do **not** promote `teacher_leaf_blend` to live.  Keep it as a negative control / lesson: leaf-only plus validation-only mechanics are necessary but not sufficient; validation return must be regularized heavily for tail risk and regime turnover.
- Best practical live/paper shadow candidate is `dynamic_conviction_switch:t0.85_risk_capped_fallback`: lower comp than profile selected, but much better realized risk profile: max bar MDD 10.08%, monthly equity MDD 4.62%, min OOS -2.65%, Sortino 2.38.
- If pure return is prioritized, `profile_optuna:selected_optuna` remains the highest 10-fold clean full candidate in this core-family rerun (+10.05% comp), but its min OOS (-9.30%) and bar MDD (19.20%) make it less attractive for live without an external risk throttle.
- `dynamic_conviction_switch:t0.85_strict_fallback` has slightly higher comp (+10.47%) and low monthly equity MDD (2.11%) but only 4/10 positive OOS folds and a -8.56% worst month, so it is a shadow/challenger rather than the primary risk candidate.

### Recommendation

For live/paper continuation:

1. Primary paper candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback`.
2. Return-seeking shadow: `profile_optuna:selected_optuna`.
3. Additional challenger/shadow: `dynamic_conviction_switch:t0.85_strict_fallback`.
4. Do not use `teacher_leaf_blend` except as a monitored research negative-control family.
5. Next improvement should not chase current OOS.  Use train/validation-only rules to add a risk throttle around `profile_optuna:selected_optuna`, preferably from realized validation drawdown/volatility, external lagged risk-state, and cash/no-position guards.  Then require a fresh-forward shadow window before promotion.

### Validation evidence

- `uv run ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py scripts/research/run_alpha_zoo_69_asset_optuna_hybrid_refit.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` — passed.
- `uv run pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` — `33 passed`.
- Full rerun: `25:52.79`, max RSS `1293384 KB`, exit status `0`.
- Audits in final artifact: metric reconciliation passed, dynamic self-feed audit passed, online weight audit passed.

---

## 2026-06-05 — 85-symbol expanded-universe clean dynamic risk-scaled rerun

### Objective

Address the weak clean OOS return after nested-hybrid removal while reflecting the expanded Binance TradFi universe.  The rerun keeps the same live discipline:

- Universe/timeframes: 85 Binance research symbols = 10 core crypto + 75 `TRADIFI_PERPETUAL`, `30m,1h,2h,4h,6h,8h,12h,1d`.
- Cost: 10 bps slippage/cost proxy.
- Refit: monthly, calendar day 1 UTC.
- Selection inputs: expanding train + prior 2 calendar months validation only.
- Locked OOS: next calendar month, report-only after fold params/candidate selection are frozen.
- OOS span: `2025-09-01T00:00:00` → `2026-06-05T12:00:00`; 2026-06 is partial because that is the latest local Binance bar.

### Implementation / hygiene changes

- Expanded the static Binance research universe snapshot to 85 symbols and backfilled the 16 newly listed/added TradFi symbols locally.  New symbols are monitored/backfilled now, but most are not train-eligible yet because they listed around 2026-06.
- Made the OHLCV loader missing-symbol safe and records requested/loaded/missing symbol counts.  The 2026-06-05 run requested and loaded all 85 symbols.
- Fold-local feature support now excludes symbols without train-window data, preventing newly listed OOS-only symbols from diluting cross-sectional ranks.
- Removed nested-hybrid material from the final path: the new dynamic candidates reference only leaf strategy labels in `final_weights`.
- Added validation-only dynamic position sizing:
  - base dynamic switch still picks a clean leaf from train/validation only;
  - `val_mdd12/15/20_scaled` variants increase exposure only on a coarse grid that remains within the fold's validation MDD budget;
  - `val_ret02_calmar80_gate` can cash-guard weak validation months;
  - cash/no-eligible folds are explicitly emitted for every dynamic label, so aggregate metrics use all 10 folds.
- Removed pandas `pct_change` default-fill behavior in the alpha-zoo loaders to avoid warning-driven implicit forward-fill distortion.

### Main artifacts

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v2_20260605.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_dynamic_scaled_20260605/alpha_zoo_85_asset_dynamic_scaled_full_v2_20260605.md`
- Symbol expansion/backfill report: `var/reports/data_collection/binance_1m_research_universe_20260605_symbol_expansion/binance_1m_research_universe_collection_20260605T123257Z.json`
- ExchangeInfo snapshot: `var/reports/symbol_universe_20260605/binance_fapi_exchangeInfo_20260605.json`

### Final clean-OOS result table

| Candidate | Clean | Nested material | OOS comp | Ann approx | Max bar MDD | Monthly eq MDD | Sharpe | Sortino | Hit | Min OOS | Tail ratio | Profit factor | Max validation MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` | True | False | 23.41% | 28.72% | 23.59% | 5.75% | 0.91 | 5.24 | 5/10 | -3.18% | 5.52 | 5.12 | 18.36% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd15_scaled` | True | False | 18.46% | 22.55% | 19.29% | 5.75% | 0.87 | 4.14 | 5/10 | -3.18% | 4.55 | 4.26 | 14.84% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd12_scaled` | True | False | 13.17% | — | 14.79% | — | 0.80 | 2.96 | 5/10 | -3.18% | — | — | ≤12% target |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate` | True | False | 12.97% | 15.76% | 10.08% | 0.00% | 1.16 | 0.00* | 3/10 | 0.00% | n/a | n/a | 7.58% |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback` | True | False | 8.02% | 9.70% | 10.08% | 5.13% | 0.71 | 2.06 | 5/10 | -2.65% | 2.71 | 2.59 | 7.58% |
| `strict_calm_leaf_selector:val_mdd8_train_val_spike_penalty` | True | False | 3.31% | 3.98% | 10.08% | 5.22% | 0.32 | 0.63 | 3/10 | -5.22% | 1.73 | 1.47 | 7.58% |

`*` Sortino/profit factor are not meaningful for the cash-gated candidate because its monthly loss observations are zero in this sample.

### Fold details for the selected top candidate

Top candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled`.

| Fold | OOS return | OOS MDD | Selected leaf | Leaf weight |
| --- | ---: | ---: | --- | ---: |
| 2025-09 | 0.14% | 0.07% | `strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna` | 1.00 |
| 2025-10 | 0.00% | 0.00% | cash/no eligible signal | 0.00 |
| 2025-11 | 28.71% | 23.59% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 2.50 |
| 2025-12 | -0.13% | 1.63% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |
| 2026-01 | 0.99% | 2.15% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.25 |
| 2026-02 | 0.26% | 8.44% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 2.50 |
| 2026-03 | -2.65% | 2.80% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |
| 2026-04 | -3.18% | 3.45% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.25 |
| 2026-05 | 0.47% | 2.28% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |
| 2026-06 | 0.00% | 0.00% | `strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna` | 1.00 |

### Interpretation

- The 85-symbol expansion is reflected in the monitor/backfill layer, but not yet a direct performance source: the newest TradFi symbols mostly lack train-window data and are intentionally excluded from fold-local feature support until they have enough history.
- The improvement comes from clean validation-only position sizing, not from OOS oracle selection or nested hybrid reuse.  `final_weights` point only to strict-efficiency leaf candidates or cash.
- The best clean candidate improves from the prior unscaled dynamic baseline (+8.02% comp, max bar MDD 10.08%) to +23.41% comp, but it spends a 23.59% bar-MDD budget.  The more conservative `val_mdd15_scaled` variant gives +18.46% comp with 19.29% max bar MDD; `val_mdd12_scaled` gives +13.17% comp with 14.79% max bar MDD.
- Hard-stop promotability is still false: the candidate does not beat the historical challenger (+53.38% comp) or robust-default hurdle (+27.01% comp / 15% MDD limit).  Treat this as paper/shadow, not real-money approval.
- The clean candidate is still concentrated in one strict-efficiency leaf in most folds.  That is preferable to hidden nested duplication, but live risk should treat it as one alpha family, not a diversified portfolio.

### Recommendation

1. Primary paper/shadow return candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd20_scaled` if a ~24% bar-MDD budget is acceptable.
2. More balanced paper candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd15_scaled`.
3. Conservative risk candidate: `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd12_scaled` or the unscaled `dynamic_conviction_switch:t0.85_risk_capped_fallback`.
4. Do not revive nested hybrid-on-hybrid blends; the current improvement already avoids nested material.
5. Keep collecting/monitoring all 85 symbols.  Refit should automatically start admitting newly listed TradFi names once they have train+validation coverage.
6. Before any real-money promotion, require fresh-forward shadow evidence after 2026-06-05 and an execution-layer risk cap that enforces total gross, per-asset concentration, and TradFi session/liquidity guards.

### Validation evidence

- Full rerun: `alpha_zoo_85_asset_dynamic_scaled_full_v2_20260605`, 10 folds, 85/85 symbols loaded, latest data `2026-06-05T12:00:00`, peak RSS `1191.1 MiB`, exit status `0`.
- Audits in final artifact: metric reconciliation passed, dynamic self-feed audit passed, online weight audit passed.
- `uv run ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` — passed during implementation.
- `uv run pytest tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py::test_dynamic_conviction_switch_emits_fallback_when_aggressive_pool_missing -q` — passed during implementation.

---

## 2026-06-06 KST — Non-nested lagged shadow leaf router, 성능 상향 재평가

사용자 피드백은 “nested를 제거한 뒤 OOS comp가 너무 낮다. 그러나 OOS는 clean해야 하고, hybrid 안에 hybrid를 넣지 말라”였다. 이번 패스의 결론은 다음과 같다.

1. 단순히 relaxed leaf를 validation 성과로 더 공격적으로 여는 `regime_opportunity_leaf_switch`는 실패했다. 최고 변형도 `+33.94%` comp에 그쳤고, `2025-09`와 `2025-12`에서 validation trap을 밟아 각각 큰 손실을 냈다. 따라서 이 family는 diagnostic/shadow negative control로만 둔다.
2. 성능을 실제로 끌어올린 후보는 `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12`다. 이 후보는 hybrid를 재료로 쓰지 않는다. 후보 pool은 `strict_efficiency`/`relaxed_efficiency` leaf만 사용하고, 현재 fold OOS를 선택/weighting에 쓰지 않는다.
3. 다만 이 router 자체는 과거 OOS 리뷰 후 설계된 `post_oos_research_variant`이므로, 같은 historical window에서는 clean promotion 대상이 아니다. “기계적으로 current-fold OOS-free”이지만 “즉시 real 승격 가능한 clean protocol”은 아니다. fresh-forward shadow가 필요하다.

### Final rerun artifact

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_lagged_shadow_router_v2_20260606/alpha_zoo_85_asset_lagged_shadow_router_v2_full_20260606.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_lagged_shadow_router_v2_20260606/alpha_zoo_85_asset_lagged_shadow_router_v2_full_20260606.md`
- Universe/timeframes: `85/85` symbols loaded, `30m,1h,2h,4h,6h,8h,12h,1d`.
- Data end: `2026-06-05T12:00:00` UTC.
- OOS schedule: 10 folds, `2025-09-01T00:00:00` → `2026-06-05T12:00:00`, monthly day-1 refit, 2M validation, 10bps.
- Full rerun: `34:42.20`, peak RSS `1474.02 MiB`; metric reconciliation passed with `candidate_count=144` and no mismatches.

### Aggregate comparison

| Candidate | Clean promotion | Why non-clean if any | OOS comp | Ann approx | Max bar MDD | Monthly eq MDD | Sharpe | PF/Omega | Hit | Min OOS | Latest OOS | Nested material | Current OOS used |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12` | False | `post_oos_research_variant`, `requires_fresh_forward_shadow` | `67.16%` | `85.25%` | `27.69%` | `2.51%` | `1.81` | `23.83/23.83` | `4/10` | `-2.51%` | `0.00%` | False | False |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | True | — | `34.39%` | `42.57%` | `27.69%` | `0.00%` | `1.12` | `∞/∞` sample | `3/10` | `0.00%` | `0.00%` | False | False |
| `regime_opportunity_leaf_switch:strict30_relaxed15_cap150` | False | `post_oos_research_variant`, `requires_fresh_forward_shadow` | `33.94%` | `42.00%` | `29.13%` | `14.02%` | `0.90` | `2.25/2.25` | `4/10` | `-17.00%` | `-0.18%` | False | False |

### Lagged router mechanics

- Warmup: at least 4 completed paper/OOS months per leaf.
- Source universe: strict/relaxed leaf rows only; no `hybrid_v3_5`, `hybrid_v3_6`, `dynamic_conviction_switch`, `validation_selector`, `meta_portfolio`, `bridge`, or nested portfolio material.
- Monthly rule after warmup: among leaves that pass current train/validation guards (`validation_return >= 5%`, `validation_mdd <= 12%`, `train_mdd <= 50%`), rank by the average of the last 2 completed shadow/OOS returns. The current month OOS is not available at selection time.
- Fallback before warmup or with no eligible lagged leaf: direct strict-core/cash guard, using train/validation only.
- The runner now records `router_branch`, lagged history tail, completed-fold cutoff, and scale metadata in fold rows so future audit can confirm no same-month self-feed.

### Fold-level result

| Fold | OOS window | Router cutoff | Branch | Selected leaf / cash | Val return | OOS return | OOS MDD |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `2025-09` | `2025-09-01` → `2025-09-30` | — | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` |
| `2025-10` | `2025-10-01` → `2025-10-31` | `2025-09` | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` |
| `2025-11` | `2025-11-01` → `2025-11-30` | `2025-10` | strict-core scaled | strict balanced leaf | `85.56%` | `33.48%` | `27.69%` |
| `2025-12` | `2025-12-01` → `2025-12-31` | `2025-11` | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` |
| `2026-01` | `2026-01-01` → `2026-01-31` | `2025-12` | lagged shadow leaf | strict growth leaf | `11.19%` | `10.55%` | `4.73%` |
| `2026-02` | `2026-02-01` → `2026-02-28` | `2026-01` | lagged shadow leaf | relaxed growth leaf | `15.45%` | `3.28%` | `7.87%` |
| `2026-03` | `2026-03-01` → `2026-03-31` | `2026-02` | lagged shadow leaf | relaxed growth leaf | `10.74%` | `-2.51%` | `5.16%` |
| `2026-04` | `2026-04-01` → `2026-04-30` | `2026-03` | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` |
| `2026-05` | `2026-05-01` → `2026-05-31` | `2026-04` | lagged shadow leaf | relaxed balanced leaf | `45.43%` | `12.51%` | `20.26%` |
| `2026-06` | `2026-06-01` → `2026-06-05` | `2026-05` | lagged shadow leaf | relaxed balanced leaf | `5.64%` | `0.00%` | `0.00%` |

### Recommendation

- 현 시점 real-money 승격 후보는 여전히 없다. Hard-stop은 historical challenger `+53.38%` comp / `18.8%` MDD 대비 `lagged_shadow_leaf_router`가 return은 이기지만 MDD와 fresh-forward 조건을 통과하지 못해 false다.
- 그래도 “성능이 너무 낮다”는 문제에 대한 가장 합리적인 개선 경로는 이 router다. 기존 clean-promotable dynamic은 baseline/paper-safe conservative track으로 유지하고, lagged router는 85-symbol monitor에서 fresh-forward shadow로 굴린다.
- 실제 운용 후보가 되려면 2026-06-05 이후 새 월간 forward에서 최소 1~2개월 이상 같은 선택 규칙으로 관찰하고, 가능하면 4개월 이상 lagged telemetry가 쌓인 뒤에만 소액/테스트넷 승격을 검토한다.
- nested hybrid 재도입은 금지한다. 겉으로 comp가 높아 보여도 같은 sleeve/factor를 두 번 사는 노출 중복과 OOS-review contamination 리스크가 더 크다.

### Validation evidence

- Full exact rerun: exit `0`, `34:42.20`, peak RSS `1474.02 MiB`.
- Artifact audits: `metric_reconciliation.metrics_reconciled=true`, `nested_hybrid_dependency=false`, `uses_locked_oos_for_selection=false` for both top router and current clean dynamic.
- `uv run ruff check scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py` — passed.
- `uv run pytest tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py -q` — `38 passed`.

## 2026-06-06 — 최신 데이터 85-symbol non-nested clean/shadow 재평가

- Data refresh: `scripts/collect_binance_1m_research_universe.py --source fapi --universe-source static-plus-fapi-tradfi`로 85/85 symbols 최신화. 최신 30m 기준 `2026-06-06T08:30:00` UTC, 수집 report `var/reports/data_collection/binance_1m_research_universe_refresh_20260606/binance_1m_research_universe_collection_latest.json`, error 0.
- Full WF artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606/alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606.json`.
- Augmented/report artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606/alpha_zoo_85_asset_non_nested_augmented_selectors_latest_20260606.json` and `.md`.
- Protocol: monthly day-1 refit, expanding train from 2025-01-01, 2M validation, 1M locked OOS, allowed TF `30m,1h,2h,4h,6h,8h,12h,1d`, 10bps round-trip cost. OOS range `2025-09-01T00:00:00` → `2026-06-06T08:30:00` UTC; June is partial.
- Runtime: full exact rerun exit 0, `31:37.96`, peak RSS `1,520.3 MiB`. Augmented row-level replay exit 0, `0:08.45`, peak RSS `209,948 KiB`.
- Evaluation audit: `metric_reconciliation.metrics_reconciled=true`, `candidate_count=152`, `uses_locked_oos_for_selection=0`, `nested_hybrid_dependency=0`, clean+nested rows 0, dynamic self-feed violations 0.
- Slippage audit: runner refuses `--slippage-bps` unless it equals `broad69.PRIMARY_ROUND_TRIP_COST_BPS`; both are `10.0`. `simulate_symbol` charges half cost on entry and half on exit, so 0→1→0 costs one full 10bps round trip.

### Latest aggregate comparison

| Candidate | Clean promotion | Why non-clean if any | OOS comp | Ann approx | Max bar MDD | Monthly eq MDD | Sharpe | Sortino | PF/Omega | Hit | Min OOS | Latest OOS | Nested | Current OOS used |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap150` | False | `post_oos_research_variant`, `requires_fresh_forward_shadow` | `61.40%` | `77.62%` | `29.13%` | `3.86%` | `1.61` | `50.87` | `8.55/8.55` | `4/10` | `-3.86%` | `-3.34%` | False | False |
| `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap140` | False | `post_oos_research_variant`, `requires_fresh_forward_shadow` | `59.99%` | `75.76%` | `27.69%` | `3.59%` | `1.60` | `105.04` | `8.70/8.70` | `4/10` | `-3.59%` | `-3.34%` | False | False |
| `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12` | False | `post_oos_research_variant`, `requires_fresh_forward_shadow` | `56.15%` | `70.71%` | `27.69%` | `6.59%` | `1.53` | `6.09` | `6.58/6.58` | `4/10` | `-6.59%` | `-6.59%` | False | False |
| `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled` | True | — | `34.39%` | `42.57%` | `27.69%` | `0.00%` | `1.12` | `∞` | `∞/∞` | `3/10` | `0.00%` | `0.00%` | False | False |
| `strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna` | True | — | `32.74%` | `45.88%` | `14.77%` | `14.40%` | `0.84` | `2.58` | `2.95/2.95` | `4/9` | `-14.22%` | `49.10%` | False | False |
| `relaxed_efficiency:aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna` | True | — | `31.62%` | `39.05%` | `26.47%` | `22.01%` | `0.70` | `2.89` | `2.13/2.13` | `4/10` | `-13.99%` | `40.86%` | False | False |
| `row_level_leaf_selector:validation_return_mdd25` | False | post-OOS row-level diagnostic | `22.11%` | — | `26.62%` | — | `0.59` | — | — | `5/10` | `-15.35%` | — | False | False |

### Best clean candidate fold detail

Best clean/promotable-mechanics candidate remains `dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled`.

| Fold | Validation return | OOS return | OOS MDD | Selected leaf/cash | Risk scale |
| --- | ---: | ---: | ---: | --- | ---: |
| `2025-09` | `0.00%` | `0.00%` | `0.00%` | cash guard | `0.0` |
| `2025-10` | `0.00%` | `0.00%` | `0.00%` | cash/no eligible | `0.0` |
| `2025-11` | `85.56%` | `33.48%` | `27.69%` | strict balanced leaf | `3.0` |
| `2025-12` | `0.00%` | `0.00%` | `0.00%` | cash guard | `0.0` |
| `2026-01` | `0.00%` | `0.00%` | `0.00%` | cash guard | `0.0` |
| `2026-02` | `15.01%` | `0.21%` | `10.08%` | strict balanced leaf | `3.0` |
| `2026-03` | `0.00%` | `0.00%` | `0.00%` | cash guard | `0.0` |
| `2026-04` | `0.00%` | `0.00%` | `0.00%` | cash guard | `0.0` |
| `2026-05` | `2.86%` | `0.47%` | `2.28%` | strict balanced leaf | `1.0` |
| `2026-06` | `0.00%` | `0.00%` | `0.00%` | cash guard | `0.0` |

### Best shadow candidate fold detail

Best raw/shadow candidate is `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd20_cap150`. It is mechanically current-fold OOS-clean and non-nested, but not promotion-clean because the router/scale variant was introduced after historical OOS review.

| Fold | Branch | Selected leaf/cash | Validation return | OOS return | OOS MDD | Risk scale |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2025-09` | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` | `0.0` |
| `2025-10` | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` | `0.0` |
| `2025-11` | strict-core scaled | strict balanced leaf | `85.56%` | `33.48%` | `27.69%` | `3.0` |
| `2025-12` | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` | `0.0` |
| `2026-01` | lagged shadow leaf | strict growth leaf | `5.77%` | `5.28%` | `2.38%` | `0.5` |
| `2026-02` | lagged shadow leaf | relaxed growth leaf | `23.28%` | `4.39%` | `11.61%` | `1.5` |
| `2026-03` | lagged shadow leaf | relaxed growth leaf | `16.16%` | `-3.86%` | `7.69%` | `1.5` |
| `2026-04` | strict-core cash | cash guard | `0.00%` | `0.00%` | `0.00%` | `0.0` |
| `2026-05` | lagged shadow leaf | relaxed balanced leaf | `72.66%` | `18.40%` | `29.13%` | `1.5` |
| `2026-06` | lagged shadow leaf | relaxed balanced leaf | `2.63%` | `-3.34%` | `3.13%` | `0.5` |

### Conclusion

- Nested hybrid 문제는 이번 latest artifact 기준 제거됐다. Hybrid/selector/gate를 hybrid 재료로 넣지 않고, leaf-only source/reference rule과 row-reference audit이 모두 통과한다.
- Clean-promotion 기준으로 성능 상단은 여전히 dynamic cash-gated sleeve다. +34.39% comp는 이전보다 높지만 historical challenger +53.38%에는 못 미치므로 real 승격은 불가하다.
- Return을 더 올린 후보는 lagged-shadow leaf router scale variants다. 다만 이것은 same-month OOS를 보지 않는 온라인형 아이디어일 뿐, 같은 historical window에서 이미 OOS를 본 뒤 만든 family라 fresh-forward shadow 없이는 clean promotion으로 부르면 안 된다.
- Row-level selector diagnostic은 빠른 재평가 도구로는 유용하지만 성능 개선에는 실패했다. 가장 나은 변형도 +22.11% comp라 dynamic clean보다 낮다.
- 이론적으로 타당한 방향은 “cross-sectional/trend leaf → train+validation gate → cash/no-position guard → coarse validation-MDD position sizing”이다. 반대로 OOS oracle, calendar-primary alpha, nested hybrid-on-hybrid, same-month dynamic self-feed는 계속 금지한다.
- 다음 실전 조건은 2026-06-06 이후 fresh-forward shadow. 최소 다음 1~2개 refit month에서 lagged router가 같은 rule로 positive/controlled-DD를 보여야 하며, 동시에 live/paper fill cost가 replay 10bps 이내인지 별도 검증해야 한다.

## 2026-06-07 — Clean 100%+ live strategy search ultragoal final

- Workflow: `$ralplan` → `$team` → `$ultragoal`; durable evidence under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/` and `.omx/ultragoal/ledger.jsonl`.
- Objective was reframed cleanly: not “mine until 100%+”, but “verify whether a pre-registered clean process yields any 100%+ annualized reporting-label candidate.” The 100% threshold was not used as selector/Optuna objective/promotion gate.
- Result: **no real-money candidate and no small-sleeve candidate**. Historical 100%+ labels exist only as `shadow_freeze_only`/control context.
- `clean_input_meta_selector`: annualized OOS approx `110.46%`, OOS comp `85.91%`, but **not promotable** because post-OOS selector-grid ranking used historical locked-OOS context. Label: `shadow_freeze_only`.
- `strict_no_leak_best_single_10bps`: `54.56%` total return at 10bps and `27.10%` at 20bps stress, but high MDD/tail and missing 15bps/paper-fill telemetry block live. Label: `paper_control`.
- 85-symbol clean baseline: annualized approx around `42.57%`, below 100% and high/sparse OOS profile. Label: `paper_control`.
- Clean new-alpha discovery: full `3.01%` annualized, feature-bounded `-0.57%`; rejected for current promotion.
- Cost gate: requires 10/15/20bps, turnover/RPT, capacity/liquidity proxy, all-in fill telemetry, BBO/spread/slippage/cancel/partial/reject evidence. Current evidence fails real-money and small-sleeve gates.
- TradFi expansion: monitor SPY/QQQ/IWM/TLT/IEF/GLD/USO/DXY proxy/VIX/US10Y as report-only context until a separate data/cost/session manifest exists; do not include TradFi signals in selection before that manifest.
- Final report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/final_report_clean_100pct_live_strategy_search_20260607.json` and `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/final_report_clean_100pct_live_strategy_search_20260607.md`.

## 2026-06-07 — Clean 100%+ live-target audit final label

- Final report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/final_report_clean_100pct_live_strategy_search_20260607.md`; manifest: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/immutable_manifest_clean_100pct_live_strategy_search_20260607.json`.
- 결론: 현재 **실전 투입 가능한 연 100%+ clean 후보 없음**. 100%+ headline은 `clean_input_meta_selector`/historical incumbent에 존재하지만 각각 `shadow_freeze_only`/`paper_control`로 제한한다.
- Hard gates: `no_nested_oos_mining`, `execution_cost_gate`, `theory_plausibility_gate` 유지. Locked OOS는 post-freeze report/gate only; 100% annualized threshold는 post-evaluation reporting label only.
- 실제 투입 금지 사유: fresh-forward shadow와 paper/testnet fill telemetry 부재, 후보 단위 10/15/20bps cost grid·RPT·turnover·tail/CVaR·capacity/liquidity 출력 미완비.
- 허용 다음 단계: 동일 rule로 fresh-forward shadow/paper 관찰 후, manifest-first runner와 fail-closed execution verifier를 통과할 때만 small sleeve 검토.
- Formal ultragoal G007 approval은 원래 독립 code-reviewer/architect evidence와 hidden Codex goal snapshot mismatch 때문에 `review_blocked`로 남겼다. 이후 G008(`resolve-final-independent-review-and`)에서 독립 리뷰를 회수했다: code-reviewer는 `APPROVE`, architect는 `WATCH/no safety FAIL`.
- G008 WATCH 항목은 두 가지다: (1) G007은 final-report construction/provenance, G008은 independent-review/checkpoint reconciliation으로 명시할 것, (2) relaxed-efficiency incumbent는 source artifact 209.00% ann / 156.03% comp와 locked-OOS/cost fixed-blend 160.90% ann / 122.36% comp가 같은 candidate id에 공존하므로 둘 다 paper/control lineage로만 해석할 것. 연구 결론(실전 0% allocation / no small-sleeve)은 변하지 않는다.


## 2026-06-07 — G008 independent review reconciliation

- Recheck result: 작업은 “완료 직전”이 아니었다. `G008-resolve-final-independent-review-and`를 열어 최종 독립 리뷰를 회수했다.
- Code-reviewer lane: `APPROVE`; compileall/Ruff/format/targeted pytest/artifact assertions/git diff --check 통과, CRITICAL/HIGH/MEDIUM 0.
- Architect lane: `WATCH/no safety FAIL`; no nested OOS / locked-OOS report-only / execution-cost gate / theory plausibility / TradFi monitoring-only expansion 구조는 유지되지만, G007/G008 provenance와 relaxed-efficiency metric lineage는 명시 주석이 필요했다.
- 최종 리스크 라벨: 실전 투입 금지 유지. 100%+ headline은 historical/shadow/control일 뿐, current clean real-money 기대수익으로 제시하면 안 된다.

- Reverification after G008 annotation: compileall/Ruff/format/targeted pytest/core+BBO/git diff check/artifact assertions passed. G008 checkpoint is still a formal-state blocker because blocked checkpoint is intentionally non-terminal: ledger records `goal_blocked` while `goals.json` remains `in_progress`, and hidden `get_goal` is the old completed latency objective.

## 2026-06-07 — Official Binance BBO archive backfill plumbing

- 원인 재점검 결과, 작업은 전략 승격 단계에서 멈춘 것이 아니라 `bookTicker` 히스토리 백필 구현이 미커밋/미문서 상태로 남아 있었다.
- Added `scripts/backfill_binance_public_book_ticker_history.py` and `tests/test_backfill_binance_public_book_ticker_history.py`: official Binance USD-M daily `bookTicker` ZIPs from `data.binance.vision` can now be normalized into the feature-point store using the approved adapter path.
- The importer now accepts official archive column aliases (`transaction_time`, `event_time`, `best_bid_price`, `best_bid_qty`, `best_ask_price`, `best_ask_qty`) in addition to generic/JSONL aliases.
- Controls: symbol/date universe must be pre-manifested; missing archives are recorded fail-closed; cadence sampling is explicit; this is **not** a live-promotion approval and does not alter the latest cap=500 BBO clean result (`-0.24%` OOS comp / `-0.57%` annualized).
- Verification: Ruff check pass; Ruff format `4 files already formatted`; `PYTHONPATH=. uv run pytest -q tests/test_import_binance_book_ticker_history.py tests/test_backfill_binance_public_book_ticker_history.py` -> `8 passed in 0.16s`; sample official archive `BTCUSDT-bookTicker-2024-03-30.zip` returned `HTTP/2 200` / `application/zip`.
- Next real research step, if approved as a new manifest: backfill predeclared symbols/dates, then rerun clean train/validation-only selection and locked-OOS report-only walk-forward. Until then deployment remains `0% allocation / paper-control only`.

## 2026-06-07 — Codex independent rematch vs GJC: sparse feature alignment fix

- Context: GJC는 별도 라인으로 유지하고, Codex 독립 라인은 `codex_independent_flow_coverage_research_20260607` 아래에서 실행했다. 목적은 headline 100%를 억지로 찾는 것이 아니라 `no_nested_oos_mining`, `execution_cost_gate`, `theory_plausibility_gate`를 유지한 상태에서 실제 research gap을 닫는 것이다.
- 핵심 버그: `feature_points`는 funding/OI/taker/BBO/depth가 같은 row에 공존하지 않는 sparse-source 구조인데, 기존 runner는 latest whole-row `merge_asof`를 사용했다. 이 때문에 최신 taker row를 붙이면 funding/OI가 NaN으로 남아 feature-backed family coverage가 거의 0이 됐다.
- Fix: `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py`에서 per-column last-observation asof와 per-column age gate를 적용했다. `feature_valid`는 funding+taker flow-only로 분리하고, OI/liquidation/BBO/depth는 각각 `feature_oi_flow_valid`, `feature_liquidation_valid`, `feature_bbo_valid`, `feature_depth_valid`로 fail-closed 분리했다. `.tmp.parquet`도 로더에서 제외한다.
- Test lock: `tests/test_alpha_zoo_clean_new_alpha_discovery.py`에 sparse-source independent alignment, OI 없는 flow-only validity, empty feature validity flags, tmp parquet ignore tests를 추가/보강했다. Verification: Ruff format/check pass; `PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_clean_new_alpha_discovery.py` -> `12 passed`.
- BTC/ETH/SOL pre-fix clean run: `-4.06%` OOS comp / `-9.46%` annualized / PF `0.57`, feature-backed rows 0.
- BTC/ETH/SOL sparse-asof patch run: `+3.75%` OOS comp / `+9.23%` annualized / PF `1.67`, selected folds 5, feature-backed selected 2 (`feature_taker_flow_exhaustion_reversal`, ETHUSDT 4h).
- Core10 sparse-asof patch run: `+7.69%` OOS comp / `+19.45%` annualized / PF `2.32`, selected symbols ETHUSDT/AVAXUSDT, selected families `cross_asset_lead_lag_momentum` 3 and `feature_taker_flow_exhaustion_reversal` 2.
- Deployment label: **실전/소액 투입 금지 유지**. Latest taker-flow source가 2026-05-03 부근에서 끊겨 2026-06 feature-flow live coverage가 0이므로, 현재 결과는 `research_shadow_only_after_data_pipeline_recovery`다. Core10의 19.45% annualized는 100%+ 목표에 미달하고 live feature coverage/cost/fill telemetry gate를 통과하지 못한다.
- Report artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/codex_independent_flow_coverage_research_20260607/codex_vs_gjc_independent_research_20260607.md` and `.json`.
- External basis noted in report: Binance public data archive, Time Series Momentum, Volatility Managed Portfolios, DeepLOB/LOB feature literature, and public bookTicker historical gap evidence. These justify the direction, not a live promotion.

## 2026-06-07 — Correction: Codex sparse-asof line is not the performance leader

- User challenge accepted: `core10 sparse feature asof patch` is overall weaker than existing top candidates. It is `+7.69%` OOS comp / `+19.45%` annualized, versus clean dynamic 85-symbol `+34.39%` comp / `+42.57%` annualized, lagged shadow router `+61.40%` comp / `+77.62%` annualized, clean-input shadow `+85.91%` comp / `+110.46%` annualized, and relaxed historical/control `+156.03%` comp / `+209.00%` annualized.
- Correct interpretation: the sparse-asof work is an infrastructure/research-space fix, not a better strategy. It makes feature-backed candidates observable, but the resulting new-alpha standalone strategy remains too weak and live-blocked.
- Performance path must shift back to the 85-symbol/router/Optuna lineage: use the feature-alignment patch as an input improvement there, then run the same clean train/validation freeze and locked-OOS report-only evaluation. Do not present the core10 new-alpha result as beating GJC or the existing best lines.

## 2026-06-07 — Codex performance re-research: lagged leaf router grid diagnostic

- User challenge accepted: 기존 Codex sparse-asof line은 약했고, 성능 개선은 85-symbol/router/Optuna lineage에서 다시 찾았다. GJC/backfill 프로세스는 별도 유지했고 건드리지 않았다.
- New artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/codex_performance_research_20260607/lagged_leaf_router_grid_diagnostic_20260607.md` and `.json`.
- Grid size: 54,540 variants over 10 locked OOS folds, using only strict/relaxed efficiency leaf rows (`balanced`, `growth`, `aggressive`). Hybrid/selector/router/meta/static-guarded labels were excluded to preserve the no-nested rule.
- Best exact source-metric diagnostic: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`.
  - OOS compounded `197.37%`, annualized approx `269.80%`, max fold OOS MDD `27.69%`, monthly equity MDD `4.50%`, positive folds `5/10`, profit factor `30.04`.
  - This improves the previous canonical lagged best exact line from `61.40%` comp / `77.62%` ann to `197.37%` comp / `269.80%` ann on the same fold set.
- Mechanism: after a 4-month warmup, select a leaf using last-1 completed leaf OOS return plus `0.25 * validation_score`, with train return >= `-2%`, train MDD <= `50%`, validation return >= `0%`, validation MDD <= `25%`; strict-core cash/scaled fallback remains for warmup/no-pool months.
- Interpretation: this is theoretically plausible (trend/cross-sectional momentum + validation/risk gating), not a date/asset hard-code. The large 2026-06 contribution from relaxed aggressive (`+64.80%`) also makes overfit risk obvious.
- Deployment label: **research_shadow_only_requires_fresh_forward_shadow_and_bar_exact_rerun**. It is not live-clean because the grid was selected after reviewing the historical OOS set. Real-money allocation remains `0%` until the rule is pre-registered, bar-exact rerun reproduces it, and fresh-forward shadow/paper fill telemetry passes 10/15/20bps cost, turnover/RPT, spread/slippage, partial/reject/cancel checks.
- Row-metric scaled stress variants reached `377.94%` comp / `553.50%` ann, but those are fold-metric linear sizing approximations, not bar-exact production metrics; use only to prioritize the next bar-exact rerun.

## 2026-06-07 — Pre-registered lagged router replay pass

- Follow-up to the Codex performance re-research: the high-return lagged leaf router hypothesis has now been moved from ad-hoc diagnostic into the monthly walk-forward runner as `PREREGISTERED_LAGGED_LEAF_ROUTER_LABEL`.
- Runner label: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`.
- Code path: `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py` now emits the pre-registered spec during full lagged-router candidate generation and replays it from existing exact fold rows during `--recompute-from-json`.
- Replay artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/codex_preregistered_lagged_router_pass_20260607/preregistered_lagged_router_pass_20260607.md` and `.json`.
- Official replay rank: aggregate rank `1` among the recomputed rows.
- Official metrics: OOS compounded `197.37%`, annualized approx `269.80%`, max fold OOS MDD `27.69%`, monthly equity MDD `4.50%`, positive folds `5/10`, profit factor `30.04`, latest fold `+64.80%`.
- Governance flags passed for the internal replay gate: `uses_locked_oos_for_selection=false`, `nested_hybrid_dependency=false`, current-fold OOS is report-only, and selection uses train/validation + prior completed leaf OOS history.
- Governance flags still blocking live: `post_oos_research_variant=true`, `requires_fresh_forward_shadow=true`, `clean_promotion_eligible=false`. This is now a **pre-registered shadow/paper candidate**, not a real-money approval.
- Tests added: lagged-router full candidate path now checks the pre-registered label, prior-only history tail, validation weight, no locked-OOS selection, and no nested dependency; replay path checks exact source-row selection and copied OOS metrics.
- Verification for this pass: Ruff format/check passed; `PYTHONPATH=. uv run pytest -q tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py -k 'lagged_shadow or preregistered'` -> `2 passed`; `--recompute-from-json` generated the pass artifact and aggregate rank 1.
- Next gate that cannot be faked: freeze this exact spec before the next unseen month, then require fresh-forward shadow/paper with 10/15/20bps costs, turnover/RPT, BBO spread/slippage, partial/reject/cancel telemetry before any small-sleeve review.

## 2026-06-08 — Desktop deep-research report reflection and leaf-alpha implementation

- Source reflected: `C:\Users\hoky1\Desktop\deep-research-report.md` (`/mnt/c/Users/hoky1/Desktop/deep-research-report.md`). Deep-interview scope decision: `new_alpha_implementation`.
- Reconciliation: the desktop report's headline shadow reference (`lagged_shadow_leaf_router` around `+61.40%` OOS comp / `+77.62%` annualized) is older than the repo's current pre-registered replay candidate (`codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`, `+197.37%` OOS comp / approx `+269.80%` annualized). The report's governance recommendations still apply: the +197.37% line remains **shadow/paper only**, `post_oos_research_variant=true`, `requires_fresh_forward_shadow=true`, `clean_promotion_eligible=false`.
- Implemented report-inspired leaf candidates in the alpha factory, all `research_only`, `leaf_only`, `no_nested_oos_mining`, and blocked from live promotion until fresh-forward + cost/fill telemetry gates pass:
  - `FundingDislocationTrendCarryStrategy`: multi-horizon cross-sectional trend + funding/basis carry + crowding penalty using existing perp feature points.
  - `VolManagedMomentumCrashGateStrategy`: volatility-managed cross-sectional momentum with benchmark crash/volatility stress gate and bounded leverage.
  - `FlowImbalanceLiquidationSweepStrategy`: BTC/ETH/SOL/BNB/TRX-style major-asset order-flow/liquidation sweep sleeve using taker imbalance, book/depth or BBO quality, spread, and liquidation flush confirmation.
- Planning artifacts: `.omx/specs/deep-interview-apply-deep-research-report.md`, `.omx/plans/prd-apply-deep-research-report-20260608.md`, `.omx/plans/test-spec-apply-deep-research-report-20260608.md`. Ultragoal steering added pending `G009-apply-deep-research-report-leaf-alph` rather than overwriting the existing G008-blocked plan.
- Deployment label unchanged: **0% real-money allocation**. These new leaf families are search inputs only. Required next validation: clean train/validation-only candidate search, locked-OOS report-only walk-forward, PBO/DSR/PSR family reporting, 10/15/20bps cost grid, RPT/turnover/capacity, and paper/live-shadow execution telemetry (decision/submit/ack timestamps, BBO at submit/fill, fill/slippage vs mid/touch, partial/cancel/reject/reconciliation).
- Initial verification: targeted candidate registration + signal tests passed with `.venv/bin/python -m pytest tests/test_strategy_factory_library.py -k "deep_research_report" tests/test_research_runner_feature_support.py -k "deep_research"` -> `4 passed`.

## 2026-06-08 — Clean WF check for desktop-report leaf alphas: no promotion

- Follow-up to the desktop report reflection: the report-inspired leaf families were moved into the clean new-alpha discovery runner and evaluated with train/validation-only freeze plus locked-OOS report-only monthly walk-forward. Search hash: `d92dce2b046441bcf1a7a7ebfa5499844b418a52d6d2416fefad491458967312`.
- New runner families: `deep_research_funding_dislocation_trend_carry`, `deep_research_vol_managed_momentum_crash_gate`, and `deep_research_flow_imbalance_liquidation_sweep`. All rows are tagged `source_report=desktop-deep-research-report-20260608`, `no_nested_oos_mining=true`, `ready_for_real=false`, `real_money_execution=false`, `clean_promotion_eligible=false`.
- Feature-backed report families correctly failed the coverage gate rather than being force-fit: current feature/OI coverage is only about 5% in train, 54-56% in validation, and 0% in the latest locked-OOS slice; liquidation feature coverage is 0%. Lowering this gate would be data-availability overfit, so it remains blocked.
- Full available monthly WF artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_report_leaf_clean_discovery_20260608_full/clean_new_alpha_discovery_summary_20260608.json` and `clean_new_alpha_discovery_latest.md`.
- Full 10-fold result at the default 10bps round-trip cost: OOS compounded `+1.71%`, annualized approx `+2.06%`, positive folds `5/10`, latest fold `-0.37%`, min fold `-7.16%`, monthly Sharpe approx `0.195`, monthly equity MDD `14.87%`, max fold OOS MDD `11.85%`.
- Selected full-WF families: `deep_research_vol_managed_momentum_crash_gate` 7 folds and existing `cross_asset_lead_lag_momentum` 3 folds. The report vol-managed sleeve alone is negative under exact cost stress; the whole selected set degrades from `+1.71%` comp at 10bps to `+0.46%` at 15bps, `-0.78%` at 20bps, and `-3.21%` at 30bps.
- Cost-stress artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/deep_research_report_leaf_clean_discovery_20260608_full/clean_new_alpha_discovery_cost_stress_20260608.json` and `.md`.
- Decision: **FAIL real-money and shadow-promotion gate**. This line is theoretically plausible as a leaf research input, but it is nowhere near the requested 100%+ annualized clean target, has weak fold stability, and loses edge under realistic cost stress. Real-money allocation remains `0%`.
- Safe next step remains the already pre-registered lagged leaf-router shadow candidate, not these new report leaf alphas: freeze before unseen fresh-forward, collect paper/shadow execution telemetry, and require 10/15/20bps cost, turnover/RPT, BBO/slippage, partial/reject/cancel/reconciliation gates before any small-sleeve review.

## 2026-06-08 — VWAP/ATR/Bollinger/Kalman/train-only ML leaf clean WF: saved, no promotion

- User requested adding VWAP, ATR, volatility, Bollinger, Kalman filter, train-only standardization, and 30m+ ML-style leaves. The clean new-alpha discovery runner now includes `indicator_vwap_atr_bollinger_reversion`, `indicator_kalman_volatility_trend`, and `standardized_indicator_ridge_directional` under search hash `ee6ecd539a6b5a8c078bd0e39f22ef0bb483d10e1ecbf97236c51b9a6fb087e8`.
- Validation policy preserved: train+validation-only selection, locked OOS report/gate only, `no_nested_oos_mining=true`, no real-money execution, and no clean-promotion flag. The ML leaf uses train-only standardization and train-only ridge fitting; locked OOS is not used for fitting/selection.
- Full 10-fold clean WF artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/indicator_kalman_ml_clean_discovery_20260608_full/indicator_kalman_ml_clean_wf_handoff_summary_20260608.md` and `.json`. Raw local candidate JSON remains in the same directory but was intentionally not committed to avoid git bloat.
- Full result at default 10bps round-trip cost: OOS compounded `-8.77%`, annualized approx `-10.43%`, positive folds `4/10`, profit factor `0.88`, monthly equity MDD `28.82%`, max fold OOS MDD `16.70%`.
- Selected folds included standardized ML (`SOLUSDT` 4h in 2025-09/10/11, `DOGEUSDT` 4h in 2026-03, `AVAXUSDT` 4h in 2026-04), Kalman trend (`ADAUSDT` 4h in 2025-12), VWAP/ATR/Bollinger reversion (`AVAXUSDT` 4h in 2026-05), cross-asset lead-lag (`AVAXUSDT` 4h in 2026-01/02), and vol-managed momentum (`TONUSDT` 4h in 2026-06). The apparent train/validation strength did not survive locked OOS.
- New family stored-candidate diagnostics were mixed but not deployable: VWAP/ATR/Bollinger median OOS `-1.06%` with 94/261 positive rows; Kalman trend median OOS `-0.25%` with 575/1282 positive rows; standardized ML median OOS `-0.67%` with 367/967 positive rows. Some individual rows were positive, but selecting on those OOS winners would violate the locked-OOS rule.
- Decision: **FAIL promotion; research-only save**. Real/shadow allocation remains `0%` for this line. Next session should not tune against this locked OOS; if continuing, freeze a new train/validation-only Optuna/search plan, add robustness penalties for train/validation gap, turnover/RPT, leverage, and cost sensitivity, then run fresh clean WF + 10/15/20bps stress. Keep the existing pre-registered lagged leaf-router shadow candidate separate from this failed indicator/ML line.

## 2026-06-09 — Robust train/validation selector v1 for indicator/Kalman/ML leaves

- Follow-up to the failed 2026-06-08 VWAP/ATR/Bollinger/Kalman/train-only ML clean WF. The prior default selector lost `-8.77%` OOS comp / `-10.43%` annualized with only `4/10` positive folds. The next safe branch was to add a train/validation-only robustness selector, not to retune on locked OOS winners.
- Code change: `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py` now supports `--selection-policy robust_train_validation_v1` in addition to the preserved default selector, and adds a pre-registered `btc_beta_residual_momentum` leaf that trades target residual momentum after rolling BTC/ETH beta adjustment with crash/volatility gates. The robust policy keeps locked-OOS flags false, scores only train/validation metrics, and the runner now sorts the retained per-fold candidate cap by the active train/validation selection score: positive train/validation, MDD caps, train/validation return-ratio cap, train-minus-validation cap, minimum validation activity, positive train/validation RPT, validation-Calmar emphasis, turnover/activity bonus, and train/validation gap/MDD penalties.
- Tests: `tests/test_alpha_zoo_clean_new_alpha_discovery.py` now asserts robust score/eligibility ignore locked-OOS report fields, robust selection rejects OOS-winning but train/validation-overfit rows, realism diagnostics block live assumptions, and runner JSON records the robust selection policy. Verification: `.venv/bin/python -m pytest tests/test_alpha_zoo_clean_new_alpha_discovery.py -q` -> `18 passed`; Ruff check passed.
- Preliminary stored top-500 diagnostic (not promotion): `indicator_kalman_ml_robust_selector_research_20260609` improved the same stored candidate rows from `-8.77%` comp to `+14.56%` comp / `+17.72%` annualized / `5/10` positive, but this was only a post-failure diagnostic.
- Generated-candidate robust run (100k retained rows; 10k/fold cap, source hash `ee6ecd539a6b5a8c078bd0e39f22ef0bb483d10e1ecbf97236c51b9a6fb087e8`, before the beta-residual family and active-policy-aware cap sorting changed the current runner state; current code hash is `57121f6a8ade6faeaf1a83b06276728a8f3590d320d5af501ce3115e9b260a82`): `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/indicator_kalman_ml_robust_selector_full_universe_20260609/indicator_kalman_ml_robust_selector_full_universe_summary_20260609.md` and `.json`. Raw local JSON is `369MB` and intentionally not committed.
- Capped generated-candidate result at default 10bps round-trip cost: OOS compounded `+22.14%`, annualized approx `+27.12%`, positive folds `6/10`, monthly equity MDD `3.10%`, max fold OOS MDD `9.89%`, profit factor `4.95`. Selected families were cross-asset lead-lag, standardized ridge ML, Kalman trend, volatility squeeze, feature taker-flow exhaustion, VWAP/ATR/Bollinger reversion, and vol-managed momentum.
- Interpretation: **meaningful research improvement, still no live/shadow promotion**. Because the robust selector was added after reviewing the prior locked-OOS failure, it is `post_failure_research_variant_requires_fresh_forward`; the score itself ignores OOS, but the design iteration is OOS-informed. Real/shadow allocation remains `0%`. Next required gates: freeze this robust selector before a new unseen/fresh-forward slice, rerun fresh-forward candidate generation on the current search space with cap explicitly recorded or uncapped, add exact 10/15/20bps cost stress or paper fill telemetry, and check turnover/RPT, BBO spread/slippage, partial/reject/cancel, and reconciliation telemetry.


## 2026-06-09 — Current-search fast probes and new residual momentum alpha

- User rejected prior results as too weak and asked to keep discovering; stress assumption remains default `10bps` round-trip for now. Implemented fold-test speed controls in `run_alpha_zoo_clean_new_alpha_discovery.py`: active-policy eligible-first heap cap, `--families` subset gate, `--leverages` single-value probe, `--fold-workers`, and `--max-candidate-rows-output` to skip huge candidate JSON during smoke tests.
- Added new pre-registered alpha `cross_sectional_vol_adjusted_momentum`: panel median residual momentum scaled by realized volatility, with market-stress and max-volatility gates. Also retained the already-added `indicator_kalman_residual_reversion` and `btc_beta_residual_momentum` lines.
- Fast latest 3-fold results were still weak. Combined 4-symbol new-alpha smoke: `-3.24%` comp / `0/3` positive in `1:05`. Family smokes: Kalman residual `+0.66%` comp / `1/3`, beta residual no eligible rows in the subset, Kalman vol trend `-1.86%`, standardized ridge `-0.22%`, lead-lag `+2.60%` but only `1/3`, cross-sectional vol-adjusted momentum `-3.67%`. Core 10-symbol momentum probe was `-5.41%` comp / `1/3` and took `3:56`.
- Summary artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/current_search_fast_probe_summary_20260609/current_search_fast_probe_summary_20260609.md`. Decision: **NO PROMOTION; allocation remains 0%**. These latest-fold probes show train/validation winners still decay in locked OOS, so continue with smaller family smoke loops before any expensive 10-fold run.

## 2026-06-09 — Alpha overlay plus Rust fold backend acceleration

- User correction accepted: reflected alpha families must be evaluated inside the existing candidate flow, not only as isolated one-off probes. Added `indicator_vwap_kalman_pullback_continuation` to the default clean new-alpha search space, combining Kalman trend, VWAP/Bollinger pullback, ATR distance, and realized-volatility gates. It is tagged as a theory-plausible 30m+ leaf, not a post-OOS hard-coded rule.
- Fold-loop optimization implemented: `native/rust_alpha_fold` + `lumina_quant.alpha_zoo.native_alpha_fold_backend`, `--simulation-backend auto|python|rust`, cached frame arrays, cached train/validation/locked-OOS split masks, and fast finalize parity tests. Local evaluator passed with Rust resolved: clean discovery tests `24 passed`, simulation speedup `2.55x`, finalize/mask speedup `16.41x`, parity true.
- Existing+new alpha overlay smoke: `current_search_existing_alpha_overlay_fast_20260609` (BTC/ETH/SOL/BNB, 1h, 3 folds, lev2, 8 families, robust train/validation selector) produced OOS comp `+4.16%`, annualized approx `+17.73%`, `3/3` positive folds, max OOS MDD `1.45%`. Selected rows were cross-asset lead-lag and cross-sectional vol-adjusted momentum; the new VWAP/Kalman pullback alpha appeared in retained candidate rows but did not win selection.
- New VWAP/Kalman pullback standalone smoke: `current_search_vwap_kalman_pullback_smoke_20260609` produced OOS comp `+0.61%`, annualized approx `+2.46%`, `2/3` positive folds. It is retained as a search input only; no promotion.
- Existing-candidate reuse selector script added for post-failure research over already-evaluated candidate rows. `robust_top1` reproduces the robust diagnostic at `+22.14%` comp / `+27.12%` annualized / `6/10` positive, but the selector design is still OOS-informed and fresh-forward-required. `robust_top2_equal` is `+20.02%` comp / `+24.48%` annualized / `7/10`; diverse3 is weak.
- Decision: **NO PROMOTION; allocation remains 0%**. The backend/finalize speedups are kept because they preserve semantics and make iteration faster. The latest alpha overlay does not approach the requested 100%+ annualized clean/live target. Continue with smaller pre-registered family sweeps and escalate to expensive 10-fold/full-universe only after current 3-fold smoke survives 10/15/20bps cost, RPT/turnover, and paper/live fill telemetry gates.

## 2026-06-09 — Residual/dispersion alpha search: 후보 추가, 승격 없음

- Added two theory-plausible clean-discovery families under search hash `4dd982a04779707f11d4530059f314ebe965cdee32fbd5a92a87a946ca3c7be7`:
  - `cross_sectional_residual_reversal`: panel-median residual shock mean-reversion with own-trend, realized-vol, and market-stress gates.
  - `cross_sectional_dispersion_gated_momentum`: residual relative-strength momentum gated by lagged cross-sectional return dispersion vs rolling history.
- External theory anchors checked for hypothesis shape: Zhang & Makgolo 2026 dispersion/state-dependent crypto momentum, Dobrynskaya crypto momentum/reversal, Avellaneda-Lee residual/stat-arb mean reversion, and Moskowitz-Ooi-Pedersen time-series momentum.
- Smoke evidence at embedded 10bps round-trip cost, BTC/ETH/SOL/BNB, 1h, 3 recent folds, lev2, Rust backend, robust train/validation selector:
  - `current_search_xs_residual_reversal_smoke_20260609`: OOS comp `-0.25%`, annualized `-1.49%`, `0/2` positive. **Reject standalone**.
  - `current_search_xs_dispersion_gated_momentum_smoke_20260609`: OOS comp `+0.69%`, annualized `+2.79%`, `2/3` positive. Weak standalone only.
  - `current_search_dispersion_residual_overlay_20260609`: adding dispersion-gated momentum to the current overlay degraded OOS to `+0.75%` comp / `+3.04%` annualized / `2/3` positive because it displaced better train/validation winners.
  - `current_search_residual_only_overlay_v2_20260609`: excluding dispersion-gated momentum preserved the prior best current overlay at `+4.16%` comp / `+17.73%` annualized / `3/3` positive, selected lead-lag + XS vol-adjusted momentum.
- Existing-candidate reuse diagnostic added/recorded `robust_quality_v1_top1`: `+24.55%` comp / `+30.14%` annualized / `7/10` positive, improving prior `robust_top1` (`+22.14%` comp). This is still **post-failure research**, not clean/fresh-forward promotion.
- Summary artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/current_search_residual_dispersion_summary_20260609/current_search_residual_dispersion_summary_20260609.md` and `.json`.
- Decision: **NO LIVE/SHADOW PROMOTION; real allocation remains `0%`**. Do not tune thresholds or family inclusion against these locked-OOS smoke results. If using this branch next, freeze family subset before a genuinely unseen/fresh-forward slice, then run 10/15/20bps cost or paper fill telemetry, turnover/RPT, BBO spread/slippage, partial/cancel/reject, and reconciliation gates.

## 2026-07-07/08 — Strategy performance improvement pass: guarded lagged-shadow candidate, completed full-universe measurement, no promotion

- User objective: improve the highest-performing plausible strategy line without nonsensical/OOS-mined promotion, refresh TradFi discovery evidence, keep data writes/backfill safe, optimize compute/memory under `uv`, run full-universe walk-forward if supported by existing data, pass CI, and push to git.
- Primary artifacts:
  - `var/reports/strategy_performance_improvement_20260707/improved_candidate_manifest_latest.json`
  - `var/reports/strategy_performance_improvement_20260707/wf_report_normalized_latest.json` and `.md`
  - `var/reports/strategy_performance_improvement_20260707/full_universe_walkforward/full_universe_walkforward_summary_latest.json` and `.md`
  - `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/tradfi_data_coverage_summary_latest.json` and `.md`
  - `var/reports/strategy_performance_improvement_20260707/final_ci_summary_latest.json` and `.md`
- Implemented candidate: `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd12_cap110_trimmed`.
  - Family: `lagged_shadow_leaf_router`.
  - Status: `research/shadow-only`.
  - `clean_promotion_eligible=false`, `promotion_decision=no_clean_promotion`, `requires_fresh_forward_shadow=true`.
  - Risk controls: no current-fold OOS selection/weighting, lagged max scale `1.1`, lagged target validation MDD `0.12`.
- Governance repair: locked-OOS/report rankings are now diagnostic-only for this workflow. Promotion requires fresh-forward shadow evidence rather than selecting or sizing from the same locked OOS report set.
- Current official full-universe WF result: **completed measurement / no promotion**.
  - `full_universe_claim_status=claimed_loaded_all_requested_symbols_completed_walkforward`.
  - Requested/loaded/missing symbols: `110 / 110 / 0`; latest data timestamp `2026-07-04T06:30:00`.
  - Fold count: `11`; fold candidate rows: `1733`; aggregate rankings: `165`; clean ranking rows: `131`.
  - Top diagnostic/research headline remains Track B only: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled` recorded `+63.36%` compounded OOS / `+70.81%` annualized approximation, but `clean_promotion_eligible=false`, `post_oos_research_variant=true`, `requires_fresh_forward_shadow=true`, and `ranking_usage_policy=locked_oos_diagnostic_only_not_selection`.
  - Integrated candidate `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12_lag_val_mdd12_cap110_trimmed` recorded `+39.14%` compounded OOS / `+43.39%` annualized approximation, max OOS MDD `27.69%`, return/MDD `1.41`, `3/11` positive OOS folds, and remains research/shadow-only.
  - Best clean Track A row recorded only `+7.99%` compounded OOS / `+8.75%` annualized approximation, max OOS MDD `23.59%`, return/MDD `0.34`, and `hard_stop_promotable=false`; it does not beat the strict benchmark/hard-stop boundary.
  - Leak checks pass in the normalized report: zero locked-OOS selection/weighting/sizing/tiebreak use and no same-month self-feed violations.
  - Raw WF row-level `candidate_tier` labels may contain legacy paper/testnet/live wording; artifact-level promotability and this note are authoritative and do not approve paper/testnet/live/real-money execution.
  - Peak RSS for the completed WF run: `1834.566 MiB`; source-symbol workers: `2`; native backend report: `numba`, `pyo3_available=True`.
- Historical proxy context preserved but not promoted:
  - Proxy compounded OOS: `+159.83%`.
  - Proxy annualized approximate: `+214.51%`.
  - Caveat: comparison/reference only; current completed full-universe WF still does not create a deployable or clean-promotion claim because the high Track B rows are post-OOS research/diagnostic-only and the clean rows do not clear hard-stop promotion gates.
- TradFi discovery/coverage lane:
  - Static TradFi snapshot: `100`.
  - Discovered TradFi trading contracts: `118`.
  - Newly discovered since static snapshot: `18` (`ALABUSDT`, `BSPUSDT`, `CATUSDT`, `CIENUSDT`, `FLEXUSDT`, `KLACUSDT`, `KORUUSDT`, `KSTRUSDT`, `LRCXUSDT`, `MVLLUSDT`, `SMCIUSDT`, `SONYUSDT`, `SQQQUSDT`, `STRCUSDT`, `TERUSDT`, `TQQQUSDT`, `TTWOUSDT`, `TXNUSDT`).
  - Selected symbols: `128` (`10` core crypto + `118` discovered TradFi trading).
  - Data-write status: `dry_run_discovery_no_data_write`; fetched rows `0`, upserted rows `0`. No uncontrolled backfill was performed.
- Verification and compute evidence:
  - Full clean-env pytest: `3574 passed, 20 skipped, 3 xfailed`.
  - Dashboard gates: `npm install`, `lint`, `test`, `typecheck`, `build` passed; dashboard tests `60`.
  - Coverage gates: total `79%`, financial core `83% >= 70%`, live/exchanges/core `70% >= 65%`.
  - Native/Rust gates: native backend build and Binance native architecture gate passed; no untested Rust hot-path change was introduced.
  - Benchmark/8GB gate: median `0.05218s`, `12149` bars/sec, peak RSS `254.19 MiB`.
  - Final targeted clean-env tests: `70 passed`.
- Git/CI delivery: Lore commit `5b48fcd943050141294376e4eb8436b5713964a0` was pushed to `private/main` before this research-note follow-up.
- Decision: **implementation, governance cleanup, and full-universe measurement succeeded, but performance improvement is not promotable**. The new lagged-shadow trimmed candidate is retained only as a research/shadow candidate. Do not report the Track B headline rows as live/paper/testnet/real-money candidates; promotion still requires leader-authorized fresh-forward shadow evidence, cost/funding/slippage telemetry, and explicit future approval.

## 2026-07-22 KST — G060/G061 repository-native terminal authority, stopped checkpoint

- G060 was independently blocked for twelve integration defects in the first delegated terminal-authority implementation: incompatible wire schemas, request-selected identities, circular manifest trust, incomplete process states, shallow acquisition/A-02/seal semantics, fake before/after evidence, incompatible recovery, inherited runtime state, and unsafe key provenance. No target or external root was executed.
- G061 was created as the explicit review-blocker replacement and implemented at commit `475f3f2ebe37994f574dc970e1b3fa9563da8009`: closed typed policy/config, exact frozen pins and argv, descriptor-relative no-symlink path opens, lexical quarantine enforcement, secure Ed25519 keys, no-launch authority, sole-launch observer, durable intent/recovery, exact signed receipts, semantic acquisition/A-02/phase/prelock/historical validation, and immutable phase/prelock snapshots.
- Stopped-snapshot verification: Ruff format/check passed, `py_compile` passed, and the focused terminal suite passed `135` tests. No acquisition, phase preparation, one-touch, prelock, historical, exchange order, or capital action occurred.
- This is deliberately incomplete: the final mandatory cleaner rerun, integrated/sanitized full verification, architect review, executor QA/red-team, and strict G061 checkpoint remain pending. G060 remains review-blocked and this implementation is not execution authority.
- The user ordered all work stopped for a cross-session transition. All workers and monitors are terminal; the inline aggregate goal is paused; durable event `7523ed6e-93fb-41f8-a937-18897ac3de8f` records `human_blocked`.
- Full cross-session contract and continuation order: `docs/research_note/g060_g061_terminal_authority_v3_resume_plan_20260722.md`.
- Methodology clarification: `HokyoungJung/Market-Cap-Weighted-Indices` is not directly integrated. Existing TopCap and turnover/flow-share work is adjacent, not equivalent to point-in-time market-cap weighting. A later port must use point-in-time constituent/capitalization evidence, remain research/shadow-only, and pass clean walk-forward plus cost/funding gates. Current pipeline work increases correctness and provenance; it does not tune headline performance.


## 2026-07-11 — Alpha-Max Revision 5.14 local implementation complete; data-PC replay pending

- Completed the repository-side Alpha-Max experiment implementation on `feat/alpha-max-20260710` after integrating the latest fetched `private/main`. The immutable prior-trial blob, embedded 21-node registry, exact incumbent audit, validation-only selection, and report-only historical boundary remain frozen and hash bound; runtime reads no `.omx`/`.omc` research artifact.
- Added strict raw/feature/config/manifest activation, indicator-only warmup and fresh economic scoring, exact native completed/barrier/finalization receipt coverage, causal funding and execution attribution, full-event ruin/liquidation evidence, frozen matrix orchestration, separate prelock/historical CLIs, immutable sealed bundles, and descriptor-bound observability export.
- Fixed the real near-high aggregator lifecycle so all admitted symbols for a completed daily key are collected before an atomic barrier closes or a genuinely expired missing cross-section fails.
- Local integrity and subprocess control-flow verification passed; expensive market replay in the subprocess harness remains deterministic because this machine does not contain the complete frozen phase roots. The actual 816 validation and 680 historical physical fold replays are intentionally delegated to the data PC under `docs/research_note/alpha_max_data_pc_runbook_20260711.md`.
- Decision: **no performance claim and no promotion**. Real-money allocation remains `0%`; paper/testnet/live approval remains absent; a one-touch exposed historical report is still report-only and genuinely fresh confirmation remains mandatory.
- Normative handoff: `docs/research_note/alpha_max_final_delivery_20260711.md`, `docs/research_note/alpha_max_data_pc_runbook_20260711.md`, and `docs/research_note/alpha_max_final_sha256_20260711.txt`.
