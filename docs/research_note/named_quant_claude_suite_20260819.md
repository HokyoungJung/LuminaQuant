# 네임드 공개규칙 스위트 — Claude 레인 (2026-08-19)

> **통합 data-PC 실행 정본:** [`named_quant_full_suite_20260819.md`](named_quant_full_suite_20260819.md). 아래 내용은 이 레인의 25후보·배분기·attribution inventory다. 실행 명령과 locked-OOS 계약은 정본만 따른다.

## 상태

- 실행 명세: `configs/research/named_quant_claude_suite_v1.json`
- 상태: **research-only / 통합 data-PC 백테스트 대기**. 성과 주장·실거래 적격성·원저자 전략 복제 주장은 없다.
- Codex 레인의 기존 엔진은 재사용하고, 겹치는 공개 골격은 별도 파라미터 변형으로 명시했다. Codex 레인은 Donchian 20/10, TSMOM, 잔차 모멘텀, BAB, 금/은 비율 등을 포함하고, 이 레인은 **새 규칙 엔진 9종 + 지표 2모듈 + 배분기 6종**(dendrogram HRP, constrained HRP, HERC, NCO, Wasserstein DRO, graph heuristic)을 추가한다.

## 출처 해석 (요청 이름 → 코드에 반영한 공개 규칙)

| 요청 이름 | 처리 | 반영 위치 |
|---|---|---|
| systrader32 | **검증 불가 → 귀속·매핑 없음**(다른 핸들로 해석하지 않음) | — |
| systrader79 (요청 이름과 별개인 인접 공개 자료로만 인용) | 설계 철학 참고만. 노이즈-K·MA 창·목표변동성 및 기타 정확한 파라미터는 공개된 systrader79 규칙으로 귀속하지 않는 독립 각색 | `NoiseFilteredVolatilityBreakoutStrategy`, `MaScoreVolTargetRotationStrategy` |
| 물탄찬밥 | 공개 자료가 지지하는 범위는 **120일 게이트 + 20일 고가/10일 저가 + ATR 단기 손절/Turtle 골격**. 종가 채널, −3.5% 고정 손절, 사이징·피라미딩 값은 독립 연구 파라미터 | `TurtleUnitPyramidingStrategy` |
| 아마추어퀀트(조성현) | 공개 연구 범위(페어트레이딩·통계적 차익거래·미시구조·포트폴리오)만 참고. 2-state Kalman 헤지비율 페어, PCA 잔차 s-score(Avellaneda–Lee) | `indicators/stat_arb.py`, `KalmanPairsStatArbStrategy`, `PcaResidualStatArbStrategy` |
| 알바트로스(성필규) | 계좌 손실한도·연패 사이징·규칙 준수·킬스위치를 **프록시 자본곡선 기반 오버레이**로 | `EquityCurveKillSwitchOverlayStrategy` (+ 순수함수 `kill_switch_scale`) |
| 부동심 | **공개 재현 규칙 없음 → 전략·원칙 어느 쪽에도 이름 귀속 없음** | — |
| FlightF(플라이트) | 후보의 `hypothesis_refs`는 Flight의 직접 게시물 2개만 사용. dacapogo는 진단 provenance이며 그곳에서 시험한 비교 프록시 계열의 음수 결과를 비재현 경고로 보존 | `RsiDivergenceScaleOutStrategy` |
| 워뇨띠/AOA | 인터뷰는 유동성 높은 메이저 선호·위험관리 맥락만 지지하며 전일 박스는 지지하지 않음. 박스 후보는 AOA 귀속 없는 완전 독립 프록시이고 dacapogo는 provenance만 | `PrevDayBoxQuartileReversionStrategy` |
| 돌파고 | 수식은 비공개/입수 불가. 세션 스캘프는 독립 OHLCV 프록시이며 Dogdrip+dacapogo는 신원·진단 provenance만 | `SessionHighBreakoutScalpStrategy` |

> 모두 "공개 자료에서 영감을 받아 독립 각색한 연구 가설"이며 원저자 성과의 재현·보증이 아니다. 미공개 변수(피벗 폭, 급증 배수, 손절 수치 등)는 전부 이 레인의 선택값이다.

## 신규 구성요소

### 지표 (`src/lumina_quant/indicators/`)
- `breakout_noise.py`: `bar_noise_ratio`, `average_noise_ratio`, `volatility_breakout_levels`, `moving_average_score`, `range_volatility_target_weight`
- `stat_arb.py`: `KalmanHedgeState`, `kalman_hedge_ratio_step/kalman_hedge_ratio/kalman_spread` (2-state 랜덤워크 회귀 칼만, 표준화 혁신 z), `pca_residual_sscores` (numpy `eigh` 기반 A–L s-score; 레포의 다른 횡단면 북은 eigen-free이며 이 레인만 PCA 사용)

### 전략 (`src/lumina_quant/strategies/`, 전부 `research_only`)
체결 계약: 신호는 봉 종가에 생성, 엔진(SimulatedExecutionHandler)은 **다음 봉 시가**에 체결 — 별도 pending 로직 없음. 세션 타임컷/일말 청산은 새 세션 첫 봉에서 발신되므로 체결이 경계 다음 봉 시가(1봉 지연)로 떨어지며, 이는 프록시 한계로 각 전략 독스트링에 명시했다.
| 클래스 | 축 | 대상/봉 |
|---|---|---|
| `NoiseFilteredVolatilityBreakoutStrategy` | 세션 변동성 돌파 + 노이즈 K + 이평스코어 + 볼타겟 + 저노이즈 유니버스 | crypto10 / TradFi, 1h·15m 세션(UTC) |
| `MaScoreVolTargetRotationStrategy` | 이평스코어 × 역변동성 RP × 볼타겟, 롱온리 동적배분 | crypto10, ETF/귀금속 퍼프, 1d |
| `TurtleUnitPyramidingStrategy` | 물탄찬밥 공개 골격(120MA·20일 고가/10일 저가·ATR 손절) + 독립 종가채널/−3.5%/사이징 파라미터 및 Faith(2007) 기반 Turtle 변형 | crypto10 / TradFi, 1d |
| `KalmanPairsStatArbStrategy` | 2-state Kalman 헤지·혁신 z·ADF 게이트·반감기 캡 | 페어(ETH/BTC, SOL/AVAX, XAU/XAG, QQQ/SPY…), 1h·4h |
| `PcaResidualStatArbStrategy` | PCA 잔차 s-score, 달러중립 롱숏 | crypto10(k=1), TradFi 주식(k=3), 1d |
| `EquityCurveKillSwitchOverlayStrategy` | 프록시 자본곡선 DD 사다리·연패 반감·월손실 한도·자본 MA 필터 | 임의 자식 전략 래핑 |
| `RsiDivergenceScaleOutStrategy` | RSI(11) 다이버전스(FlightF, HTF 확인·반대거래량 무효화) | BTC/ETH 10m |
| `PrevDayBoxQuartileReversionStrategy` | 귀속 없는 독립 전일 박스 사분위 프록시 | BTC/ETH 15m |
| `SessionHighBreakoutScalpStrategy` | 비공개 수식을 재현하지 않는 독립 세션 OHLCV 프록시 | 상위 거래대금, 1s/1m |

### 포트폴리오 배분기 (`src/lumina_quant/portfolio/hierarchical.py`)
사용자 우선순위표(NCO★5·HERC★5·Wasserstein DRO★4·Constrained HRP★3·Graph★3·Deep/RL★2)를 반영. Deep/RL은 과적합 관리 부담으로 **의도적으로 보류**. 독립 감사(2026-08-19) 지적 8건은 전부 수정·회귀테스트(`tests/portfolio/test_hierarchical_allocators.py`)로 봉인했다.

| `method` 토큰 | 함수/클래스 | 요지 | 감사 반영 |
|---|---|---|---|
| `hrp_dendrogram` (`hrp_full`) | `hrp_dendrogram_weights` / `HRPDendrogram` | López de Prado 정식 HRP: 상관거리 → 단일/완전/평균/워드 연결(Lance–Williams) → 준대각화 → 재귀 이분(IVP 군집분산) | 손으로 유도한 (0.4,0.4,0.1,0.1) 케이스 4개 연결법 일치 |
| `constrained_hrp` | 같은 함수 + `bounds` | 이분 시 자식 박스제약으로 α 클리핑 후 **박스-심플렉스 정확 투영** | 불가능 박스(Σlo>1, Σhi<1, lo>hi)는 `ValueError`로 **fail-closed**; 감사 반례(diag[.1,.1,.01], lo[.05,.2,.1], hi[.6,.4,.8])에서 하한 준수 확인 |
| `herc` | `herc_weights` / `HERC` | Raffinot HERC: 군집수 = 명시 또는 **실루엣 최적**(싱글톤 0점), 서브트리 위험 합으로 ERC 분할, 군집 내 역변동성 | 원 논문은 Gap statistic(참조분포 샘플링 필요) — 결정론 대체임을 문서·독스트링에 명시; 싱글톤 버그 수정 |
| `nco` | `nco_weights` / `NCO` | López de Prado NCO 롱온리: 군집 내 **롱온리 min-var QP(정확: 능동집합 + PGD 폴백, KKT 잔차 검증)** 또는 max-Sharpe(`μ'y=1` QP 치환) → 축약 공분산 → 합성 | 무제약 해 clip 방식 폐기; brute-force 심플렉스 오라클과 KKT 잔차<1e-8 회귀테스트 |
| `wasserstein_dro` (`dro`) | `wasserstein_dro_weights` / `WassersteinDRO` | **BCZ(2022) 정확식 p=q=2**: ε=√δ, `min √(w'Σw)+ε‖w‖₂` s.t. 심플렉스, (선택) `μ'w−ε‖w‖₂ ≥ target_return`; Armijo PGD + 이중 이분법; 불가능 target은 `ValueError`; Σ는 n-divisor 표본공분산 기본 | δ=0·비구속 target ⇒ 롱온리 min-var와 일치. 반지름은 **제곱수익 단위 민감도 파라미터**일 뿐 신뢰수준·보정값 주장이 아님 |
| `graph_inverse_centrality` (`graph`) | `graph_inverse_centrality_weights` / `GraphInverseCentrality` | **full-graph inverse-centrality graph-risk heuristic**: `A_ij=max(|ρ_ij|,1e-6)`, Perron 벡터 z, `min Σ z_i w_i²` 정확해 `w∝1/z`, 순열 등변 | MST 차수/근접 버전(엣지 가중치 무시·동률 순서 의존) 폐기; NRP/Pozzi 정확 재현 주장 제거; 순열 회귀테스트 |

공통: 공분산이 실질적으로 비-PSD(최소 고유값 < −1e-8×스케일)이면 **fail-closed**(`ValueError`), 수치 잡음 수준의 음수만 0으로 클립. 기존 `erc`/`hrp`(상관-임계 군집)는 바이트 동일. `build_quality_gated_allocation.py` 입력 JSON의 `method` + `allocator_params`로 선택하며 산출 매니페스트에 `allocator_params`가 provenance로 기록된다. 셀 스펙 검증(`--validate-cell-spec`) 통과. `lq registry list`가 포트폴리오 플러그인(옵트인 모듈)을 임포트한 뒤 나열하도록 수정. 비교/변형 실행 스크립트: `scripts/research/compare_hierarchical_allocators.py --input returns.json [--variants]`(effN·분산비·최대비중 + 셀의 `allocator_variants` 전량 실행 — 빌더는 1회 1 method만 발행하므로 변형 비교는 이 스크립트가 담당).

**배분 입력 정책(lockbox 위생)**: 가중치·품질게이트 입력은 **common-date 정렬 train+validation NET 수익만**. locked-OOS 구간은 selection/sizing에 절대 쓰지 않고 사전등록 가중치의 보고용 OOS 평가에만 쓴다(JSON `allocation_input_policy`, 각 sleeve `returns_source.selection_inputs=["train","validation"]`, 테스트로 고정).

**출처 레지스트리**: 후보 `hypothesis_refs`는 전부 JSON `evidence_sources`의 `source_id`로 해석되며 회귀테스트로 고정한다. 행은 Codex 레인과 동일 `source_id`의 verbatim 복사 + 관리자가 1차 출처를 확인해 준 신규 5행(`faith_way_of_turtle_2007`, `avellaneda_lee_2010`, `chan_algorithmic_trading_2013`, `triantafyllopoulos_montana_2011`, `blanchet_chen_zhou_2022_wasserstein_mv`). dacapogo 레포는 pinned commit URL로 **diagnostic/proxy provenance만**(규칙 근거 아님); 전일 박스 프록시는 AOA 인터뷰에 박스/사분위 규칙이 없으므로 인터뷰를 인용하지 않는 독립 프록시로 라벨링; 터틀 유닛/피라미딩은 Faith(2007) 발표 규칙 형태이되 수치는 이 레인의 독립 선택임을 후보 notes에 구분했다.

## HRP 포트폴리오 구상 (data-PC 실행 대상)

1. **슬리브 레벨** — 두 레인의 train+validation 순수익 스트림을 공통 날짜로 정렬한 뒤 `hrp_dendrogram / constrained_hrp / herc / nco / wasserstein_dro / graph_inverse_centrality`을 같은 입력에 돌려 effN·DR·회전율·폴드 안정성을 비교(셀 `named_quant_claude_hier_cell_v1`). locked OOS는 동결된 가중치의 사후 보고에만 사용한다.
2. **자산 레벨** — crypto10 + TradFi 100(귀금속·에너지·ETF·주식 그룹)의 일간 로그수익을 같은 CLI에 넣으면 자산배분이 된다(sleeves=assets). 기대되는 dendrogram 구조: {크립토}, {귀금속}, {에너지}, {미국 지수 ETF+대형주}, {한국·대만·일본 ETF}. NCO는 군집 내 min-var이므로 주식 76종에서 집중 위험 → `constrained_hrp`/`herc`와 비교.
3. **systrader79형 동적배분** — `MaScoreVolTargetRotationStrategy`가 자산 레벨 배분의 타이밍 오버레이 역할(스코어 0인 자산은 현금).
4. 모든 배분은 **월간·주간 리밸런스 + 비용 후** 측정, 회전율 페널티(`turnover_penalty_lambda`)와 상관 축소(`correlation_shrinkage`)는 기존 옵션 재사용.

## Codex 레인과의 접점
- 통합 manifest가 이 레인의 실제 NCO/HERC/Wasserstein-DRO/constrained-HRP/graph allocator와 두 레인의 40후보를 결합한다. 하위 Codex manifest의 `design_only_*` 문구는 legacy metadata이며 현재 구현 상태가 아니다.
- 두 레인 모두 `_STRATEGY_TIER_HINTS`·`indicators/__init__.py`·`.github/hardcoded_params_baseline.json`을 건드린다. 머지 후 `uv run python scripts/audit_hardcoded_params.py --write-baseline`을 한 번 더 돌려 줄 번호 시프트로 생긴 가짜 "new" 위반을 재고정해야 한다.
- 전략 이름·후보 id·family는 서로 겹치지 않는다(이 레인 family: `session_volatility_breakout`, `ma_score_dynamic_allocation`, `turtle_unit_pyramiding`, `kalman_pairs_stat_arb`, `pca_residual_stat_arb`, `equity_curve_kill_switch_overlay`, `rsi_divergence_flight`, `prev_day_box_quartile`, `session_high_breakout_scalp`).

## Universe 계약
Codex 레인과 동일: JSON의 crypto 10·TradFi 100은 정적 스모크 스냅샷이며 data-PC가 폴드별 point-in-time 시총 상위 10 ∩ Binance USD-M `TRADING` 퍼프, `TRADIFI_PERPETUAL` 전체+상장이력·필터·펀딩·마크/인덱스·비용을 채운다. 1s/1m 스캘프(`SessionHighBreakoutScalpStrategy`)는 체결·큐 모델 없는 백테스트를 승격 근거로 쓰지 않는다.

## data-PC 실행

이 레인 단독의 오래된 `run_research_candidates.py` 예시는 폐기했다. 40후보 정본 생성 → PIT 물질화 → 400봉 warmup·비용 현실화 event run → allocator variant 비교 → QGA 동결 → 별도 PIT locked-OOS → frozen evaluator의 정확한 명령은 [통합 실행서](named_quant_full_suite_20260819.md)에 있다.

## 공개 근거
- systrader79 저서/블로그: 『주식투자 ETF로 시작하라』, 『돌파매매 전략』(이레미디어), stock79.tistory.com; 크립토 변동성돌파 4대 개선(노이즈·이평스코어·변동성조절·시간분할) 커뮤니티 재현본(예: coinpick.com/quant_program/39857)
- 물탄찬밥 미리보기: Codex 레인 문서 링크 참조 (뉴지스탁 아카데미 PDF)
- 아마추어퀀트 조성현 프로필: insightcampus.co.kr (페어트레이딩·통계적 차익거래·미시구조)
- 알바트로스(성필규) 도서: 『돈을 이기는 법』(yes24)
- FlightF·워뇨띠/AOA·돌파고: `/home/hoky/dacapogo/docs/korean-trader-strategies.md`, `README.md`, `docs/research.md` (원출처 링크 포함: dcinside 548338/993821, BitMEX AOA 인터뷰, minsuk-sung Q&A 복원본)
- Turtle rules: Curtis Faith, *Way of the Turtle* (McGraw-Hill, 2007, copyrighted publication; public-domain 아님), Chan "Algorithmic Trading" (Kalman pairs), Avellaneda & Lee (2010), López de Prado (2016 HRP, 2019 NCO), Raffinot (2018 HERC), Blanchet–Chen–Zhou (2022 W-DRO), Pfitzinger & Katzke (2019 constrained HRP)
