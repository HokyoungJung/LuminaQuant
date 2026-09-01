# 네임드 공개자료 기반 Crypto·TradFi 레인 (15후보)

> **통합 data-PC 실행 정본:** [`named_quant_full_suite_20260819.md`](named_quant_full_suite_20260819.md). 아래 내용은 이 레인의 후보·근거 inventory다. 실행 명령, PIT 유니버스, 비용·체결, allocator, locked-OOS 계약은 정본만 따른다.

## 상태

- 실행 명세: `configs/research/named_quant_crypto_tradfi_suite_v1.json`
- 상태: **research-only / 통합 data-PC 백테스트 대기**
- 성과 주장, 실거래 적격성, 원저자 전략 복제 주장은 없다.
- 공개 교육자료에서 확인되는 규칙만 독립적으로 각색했고, 학술 근거가 다른 자산군에 그대로 적용된다고 가정하지 않는다.

## 출처 해석

| 요청 이름 | 처리 |
|---|---|
| 물탄찬밥 | 공개 미리보기에서 확인되는 20일 신고가 진입·10일 저가 이탈·ATR 손절, 120일 추세 필터, `IBS < 0.3`·20일 추세·직전 음봉 조건만 후보화 |
| 알바트로스(성필규) | 구체 알파가 아니라 자본관리, 사전 손절, 규칙 준수와 킬스위치 원칙에만 반영 |
| 아마추어퀀트(조성현) | 팩터·페어·미시구조·포트폴리오라는 공개 연구 범위만 참고; 비공개 매매 규칙을 귀속하지 않음 |
| systrader32 | 신뢰할 만한 공개 식별자를 확인하지 못해 귀속 제외. `systrader79`와 동일인으로 간주하지 않음 |
| 부동심 | 퀀트 저자로 확인 가능한 공개 식별자를 찾지 못해 귀속 제외 |
| systrader79 | 사용자 요청 이름과 별개인 인접 공개자료로만 기록. 저회전·유동성·분산 실행이라는 연구 원칙만 참고 |
| 돌파고 | 본인 대회 인증과 `dacapogo@633ba5d`의 원장·반증 연구를 참고. 공개된 완전한 주문식이 없으므로 L2/체결 기반 fresh-surge 가설만 설계 상태로 등록 |
| 워뇨띠/AOA | BitMEX 공식 1인칭 인터뷰에서 확인되는 대형·유동 종목 선호와 포트폴리오 위험 상한만 반영. 박스 진입식은 미공개이므로 독립 regime 가설로만 등록 |
| 플라이트/FlightF | 본인 RSI 다이버전스·매매원칙 글에서 확인되는 10분봉 교육 예시, 분할청산, 거래량 무효화, 상위 시간봉 확인만 가설화 |

> 공개자료에서 영감을 받아 독립적으로 각색한 연구 가설이며, 원저자의 실제 전략·성과 재현이나 보증이 아니다.
> 아마추어퀀트 프로필은 manifest에서도 `provenance_refs`/`evidence_only`로만 기록한다. residual momentum의 규칙 근거는 학술 문헌이고, 금/은 및 금속 상대가치 규칙은 독립 가설이다.

## 사전등록 후보 15개

### Binance USD-M crypto 상위 10

1. Donchian 20/10 + ATR 추세 추종
2. 20일 추세 조건부 IBS 단기 역추세
3. 90일 time-series momentum
4. BTC 공통요인 제거 residual momentum
5. 52주 고점 근접도 횡단면 전략
6. 월간 diversity rebalancing-premium 벤치마크
7. funding 변화/기울기 + 실제 수취 방향 필터
8. 변동성 압축 후 breakout

### Binance `TRADIFI_PERPETUAL`

9. 전 자산군 inverse-vol diversified trend
10. 단일주식 12-1 cross-sectional momentum
11. SPY 잔차 단일주식 momentum
12. 단일주식 betting-against-beta
13. 금/은 비율 평균회귀
14. XAU/XAG/XPT/XPD 상대가치 바스켓
15. 주식 ETF/귀금속 risk-on·off rotation

각 후보는 파라미터 한 세트만 사전등록했다. 대규모 grid를 추가하기 전에 이 단순 후보와 equal-weight·inverse-vol 기준선을 먼저 비교한다.

## `dacapogo`에서 추가한 provenance 가설

`/home/hoky/dacapogo`의 최신 `main`(`633ba5d`)을 확인했다. 이 저장소 역시 세 매매법을 원전략 복제가 아닌 반증 가능한 proxy로 구분한다. 이 레인 manifest의 `supplemental_hypotheses`는 provenance 기록이고, 통합 스위트는 아래 방향을 독립 프록시 후보로 실행한다.

1. 돌파고 방향: L2 호가·체결방향·스프레드·참여율이 갖춰진 경우에만 fresh surge와 가격 수용을 결합한다. OHLCV 급등 프록시를 원전략으로 부르지 않는다.
2. 워뇨띠/AOA 방향: BTC·ETH와 시점별 유동성 상위 종목의 박스 국면에서만 range-percentile 역추세를 연구하고, 원웨이 국면에서는 차단한다. 1.5–2배/30%는 추천값이 아니라 스트레스 상한이다.
3. 플라이트 방향: BTCUSDT 10분 확정 피벗 RSI 다이버전스, 상위 시간봉 regime, 반대 방향 거래량 무효화, next-open과 분할청산을 함께 검증한다. 미공개 피벗·거래량·손절식은 사전등록 독립 가정으로 남긴다.

이 세 방향의 실행 후보는 통합 스위트에서 독립 프록시 전략으로 구현됐다. 공개되지 않은 원 주문식의 재현 주장은 하지 않는다. `dacapogo`의 Wony quartile/wick/volume 및 Flight RSI 커널도 사후 탐색 proxy이며 성과 근거로 사용하지 않는다.

## 지표와 포트폴리오

- 신규 지표: `internal_bar_strength = (close-low)/(high-low)`; 0 range/비유한 값은 신호를 내지 않는다.
- 기존 지표 재사용: Donchian, ATR, SMA, 모멘텀, 실현변동성, funding delta/slope, rolling beta/residual, correlation distance, 거래대금.
- 기존 allocator 재사용: `ERC`, `HRP`; HRP는 알파 예측기가 아니라 동일 날짜로 정렬된 **순수익 스트림의 위험 배분기**다.
- `named_quant_hrp_barbell_v1` 셀은 8개 이질적 sleeve를 미리 고정한다. data-PC가 **train/validation** 순수익과 turnover를 채운 뒤에만 가중치를 계산하고, locked OOS는 가중치 선택·사이징에서 제외한다.
- 이 레인의 legacy HRP 셀은 correlation-threshold 구현이다. 통합 스위트의 primary는 full dendrogram HRP이며, sleeve 상한과 gross cap은 통합 manifest를 따른다.

### 포트폴리오 구현 상태

| 방법 | 현재 상태 | 구현 경계 |
|---|---|---|
| NCO | 구현 | long-only 군집 내·간 최적화; 통합 variant는 min-variance |
| HERC | 구현 | 결정론적 silhouette 군집 선택 변형; 원 논문의 gap statistic 재현은 아님 |
| Wasserstein DRO | 구현 | BCZ p=q=2 mean-variance; radius는 민감도 파라미터 |
| Constrained HRP | 구현 | 명시적 box-simplex 제약을 적용한 dendrogram HRP |
| Graph portfolio | 구현 | full-graph inverse-centrality 위험 heuristic; NRP 정확 재현은 아님 |
| Deep/RL | `deferred_overfit_risk` | 단순 allocator가 locked OOS에서 실패하고 별도 trial budget·환경 누출 검사가 있을 때만 착수 |

구현·실행 경계와 locked-OOS 절차는 통합 실행서를 따른다. 이 manifest의 `evidence_sources`에 남은 `no optimizer`류 문구는 병합 전 legacy source note이며 현재 구현 상태가 아니다.

## Universe 계약

현재 JSON의 crypto 10개와 TradFi 100개는 실행 smoke용 정적 스냅샷이다. Binance `exchangeInfo`는 시가총액 원천이 아니므로 이를 “역사적 시총 상위 10” 증거로 쓰지 않는다.

data-PC는 fold마다 다음을 저장해야 한다.

1. 당시 외부 시총 순위 상위 10과 Binance USD-M `TRADING` perpetual의 교집합
2. TradFi `contractType=TRADIFI_PERPETUAL`, `status=TRADING` 전체 및 자산군 라벨
3. 상장·폐지 이력과 `PRICE_FILTER`, `LOT_SIZE`, `MARKET_LOT_SIZE`, `MIN_NOTIONAL`, `PERCENT_PRICE`
4. OHLCV, mark/index, funding rate와 실제 결제 시각, fee/spread/slippage/impact
5. 다음 bar open 체결과 동일 날짜 교집합으로 정렬된 sleeve 순수익

현재 구성 종목을 과거 전체에 역적용하면 survivorship bias이므로 결과를 승격하지 않는다. point-in-time 펀더멘털이 없으므로 value/quality, term structure가 없으므로 TradFi curve carry, order book이 없으므로 market making은 의도적으로 보류했다.

## data-PC 실행

이 레인 단독의 오래된 `run_research_candidates.py` 예시는 폐기했다. 40후보 정본 생성 → PIT 물질화 → `run_named_quant_suite.py` event run → allocator 비교/QGA 동결 → 별도 PIT locked-OOS → frozen evaluator의 정확한 명령은 [통합 실행서](named_quant_full_suite_20260819.md)에 있다.

## 공개 근거

- 물탄찬밥 코인전략 미리보기: <https://resource.newsystock.com/Admin/Academy/%28Preview%29_MultanChanbab_coin_strategy.pdf>
- 물탄찬밥 뼈대전략 미리보기: <https://resource.newsystock.com/Admin/Academy/%28Preview%29%20Developing%20the%20basic%20strategy%20of%20Multanchanbap%2816%29.pdf>
- systrader79 공개 글: <https://stock79.tistory.com/entry/systrader79-%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%94%A9-%EB%A7%88%EC%8A%A4%ED%84%B0%ED%81%B4%EB%9E%98%EC%8A%A4-%ED%8C%A8%ED%82%A4%EC%A7%80-%EC%98%A4%ED%94%88>
- 알바트로스 도서 정보: <https://www.yes24.com/Product/Goods/141814145>
- 아마추어퀀트 공개 프로필: <https://insightcampus.co.kr/teachers/%EC%A1%B0%EC%84%B1%ED%98%84-%EA%B0%95%EC%82%AC%EB%8B%98/>
- Time-series momentum: <https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf>
- Cross-sectional momentum: <https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>
- Betting against beta: <https://doi.org/10.1016/j.jfineco.2013.10.005>
- Carry: <https://doi.org/10.1016/j.jfineco.2017.11.002>
- Volatility management: <https://doi.org/10.1111/jofi.12513>
- HRP: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>
- Binance USD-M 공식 문서: <https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/Introduction>
- 돌파고 본인 대회 인증: <https://www.dogdrip.net/341084283>
- 보완 연구 저장소 `dacapogo`: <https://github.com/HokyoungJung/dacapogo/tree/633ba5d6bc0c84a20696af6b2bf807cf55d21248>
- 워뇨띠/AOA BitMEX 공식 인터뷰: <https://www.bitmex.com/blog/whale-trader-talks-aoa>
- FlightF 10분봉 RSI 다이버전스 본인 글: <https://gall.dcinside.com/mgallery/board/view/?id=electronicmoney&no=548338>
- FlightF 매매원칙 본인 글: <https://gall.dcinside.com/mgallery/board/view/?id=electronicmoney&no=187860>
- NCO 원 논문: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961>
- HERC 원 논문: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540>
- Wasserstein mean-variance DRO 원 논문: <https://doi.org/10.1287/mnsc.2021.4155>
- Constrained HRP WP14/2019: <https://www.ekon.sun.ac.za/wpapers/2019/wp142019/wp142019.pdf>
- Network Risk Parity: <https://doi.org/10.1057/s41260-023-00347-8>
