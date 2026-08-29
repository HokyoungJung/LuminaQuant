# 네임드 퀀트 통합 40후보 data-PC 실행서 (2026-08-19)

## 범위와 현재 상태

- 정본 생성기: `scripts/research/build_named_quant_full_suite.py`
- 정본 명세: `configs/research/named_quant_full_suite_v1.json`
- 구성: 기존 Crypto·TradFi 15후보 + Claude 레인 25후보 = **40후보**
- 상태: `research_only_pending_data_pc_backtest`; 실거래와 승격은 금지되어 있다.
- 공개 근거로 구현 가능한 네임드 규칙과 독립 프록시는 모두 전략 클래스로 구현돼 있다. 비공개 수식을 원저자 전략으로 가장하지 않는다.
- 배분기는 HRP dendrogram, constrained HRP, HERC, NCO, Wasserstein DRO, graph inverse-centrality까지 구현돼 있다. Deep/RL은 기존 구현도 없고 과적합·다중검정 부담이 커서 의도적으로 보류한다.
- **성과 주장은 없다.** 이 문서는 실행·누출 방지 계약이며 수익성, 재현성, 실거래 적격성을 주장하지 않는다.

## 고정 실행 계약

1. 선택 입력은 train+validation뿐이다. `locked_oos`는 선택, 품질게이트, 상관, 최적화, 임계값, tie-break, 사이징에 쓰지 않는다.
2. 각 구간은 시작 시점 이전 최신 스냅샷으로 point-in-time(PIT) 유니버스를 별도 물질화한다. 선택 유니버스를 locked OOS에 재사용하지 않는다.
3. `universe_binding` 후보는 PIT 목록으로 치환한다. 고정 페어·고정 바스켓은 `universe_constraint` 밖 종목이 하나라도 있으면 `enabled=false`가 되어 runner에서 `skip`된다.
4. runner는 후보가 요구하는 timeframe의 봉을 `MarketDataRepository`에서 읽는다. data-PC는 `1m`, `10m`, `15m`, `1h`, `4h`, `1d` 등 후보 timeframe 봉을 사전에 물질화·검증해 두고, 전략 내부 timeframe aggregator에 의존하지 않는다.
5. 신호는 종가에서 발생하고 기본 백테스트 주문은 `MKT`로 다음 봉 시가에 대기 체결된다. 체결가는 스프레드·슬리피지·sqrt-impact와 봉 거래량 참여 한도를 적용한다.
6. `--warmup-bars 400`은 후보 timeframe 기준으로 선택/평가 시작 전 400봉을 상태 초기화에만 읽는다.
7. `configs/profiles/backtest_cost_realistic.yaml`을 `LQ_CONFIG_PATH`로 지정한다. 이 프로필은 taker/maker fee, 스프레드, 슬리피지, sqrt-impact, funding coverage·UTC 결제경계, 레버리지 청산을 켠다.
8. 네임드 전략의 오래된 이중관리 bracket은 제거됐다. 전략은 stop/target을 metadata에 기록하고 확정 종가에서 내부 상태를 확인해 `EXIT`를 낸다. 엔진 intrabar bracket과 전략 내부 상태를 동시에 운용하지 않는다.
9. runner는 프로필의 `risk.attach_default_protective_stop=true`를 후보 실행마다 **false로 덮어쓴다**. 따라서 임의 기본 stop은 주입되지 않는다.
10. 선택 run만 `allocation_input.json`을 만든다. `--purpose locked_oos` run은 allocator 입력을 절대 만들지 않는다.

## 입력 준비

다음 로컬 입력을 먼저 고정한다.

- `$MARKET_CAP_SNAPSHOTS`: timestamp와 순위를 가진 JSON/JSONL. Binance `exchangeInfo`를 시가총액 자료로 쓰지 않는다.
- `$EXCHANGE_INFO_SNAPSHOTS`: timestamp, `PERPETUAL`/`TRADIFI_PERPETUAL`, `TRADING`, 상장 상태와 주문 필터를 가진 JSON/JSONL.
- `$DATA_ROOT`: 후보 timeframe OHLCV와 mark/index/funding feature가 있는 parquet market-data root.
- 날짜: `SEL_AS_OF <= SEL_START < SEL_END < LOCK_START < LOCK_END`이고 `LOCK_AS_OF <= LOCK_START`다. 실제 embargo 규칙에 맞게 더 엄격히 분리해도 된다.

유니버스 receipt에는 입력 경로·SHA-256, 선택 스냅샷 시각, crypto top-10과 TradFi 종목, 주문 필터, 비활성 후보가 기록된다. 시점별 상장·폐지 이력과 외부 시총 스냅샷이 없으면 실행하지 않는다.

## 정확한 실행 순서

아래 변수 값만 data-PC 경로와 사전등록 날짜로 바꾼다.

```bash
set -euo pipefail

ART=/path/to/named-quant-20260819
DATA_ROOT=/path/to/market_parquet
MARKET_CAP_SNAPSHOTS=/path/to/market_cap_snapshots.jsonl
EXCHANGE_INFO_SNAPSHOTS=/path/to/binance_exchange_info_snapshots.jsonl

SEL_AS_OF=YYYY-MM-DDTHH:MM:SSZ
SEL_START=YYYY-MM-DD
SEL_END=YYYY-MM-DD
LOCK_AS_OF=YYYY-MM-DDTHH:MM:SSZ
LOCK_START=YYYY-MM-DD
LOCK_END=YYYY-MM-DD

mkdir -p "$ART"
```

### 1. 40후보 정본 생성과 구조 검증

```bash
uv run python scripts/research/build_named_quant_full_suite.py \
  --output "$ART/full_suite.json"

uv run python scripts/research/build_quality_gated_allocation.py \
  --validate-cell-spec "$ART/full_suite.json"
```

생성기는 두 레인을 결정적으로 합친다. 현재 정본은 40후보, 33개 사전등록 allocation sleeve, 10개 allocator variant다.

### 2. 선택 구간 PIT 유니버스 물질화

```bash
uv run python scripts/research/materialize_named_quant_universe.py \
  --suite "$ART/full_suite.json" \
  --market-caps "$MARKET_CAP_SNAPSHOTS" \
  --exchange-info "$EXCHANGE_INFO_SNAPSHOTS" \
  --as-of "$SEL_AS_OF" \
  --crypto-top-n 10 \
  --output "$ART/selection_universe.json"
```

crypto는 당시 외부 시총 순위와 Binance USD-M `TRADING` perpetual의 교집합 상위 10개다. TradFi는 당시 `TRADIFI_PERPETUAL`·`TRADING`·USDT 계약 전체다. 고정 페어가 이 목록 밖이면 자동 비활성화된다.

### 3. train+validation 선택 event run

`SEL_START`~`SEL_END`는 locked OOS를 포함하지 않는 사전등록 train+validation 합성 창이다.

```bash
LQ_CONFIG_PATH=configs/profiles/backtest_cost_realistic.yaml \
uv run python scripts/research/run_named_quant_suite.py \
  --manifest "$ART/selection_universe.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$ART/selection" \
  --exchange binance \
  --start "$SEL_START" \
  --end "$SEL_END" \
  --purpose selection \
  --warmup-bars 400
```

`selection/suite_results.json`과 후보별 JSON을 보관한다. 각 통과 행에서 다음 증거를 확인한다.

- `lineage.runtime_config.source.path`/`sha256`, `warmup_bars=400`, `returns_are_net=true`
- `execution_model`: fee, spread, slippage, sqrt-impact, leverage·maintenance-margin 설정
- `commission_paid`, `net_funding_paid`, `liquidation_count`, `turnover`, `trade_count`
- `default_protective_stop_attached=false`

funding data가 없으면 비용 현실화 프로필은 레버리지 run을 fail-closed한다. 수수료·funding·slippage/impact·청산 증거가 없는 후보는 배분 입력으로 사용하지 않는다.

### 4. 같은 입력에서 allocator variant 비교

```bash
uv run python scripts/research/compare_hierarchical_allocators.py \
  --input "$ART/selection/allocation_input.json" \
  --variants \
  --output "$ART/selection/allocator_variants.json"
```

비교 대상은 equal-weight, inverse-vol, ERC, legacy HRP, dendrogram HRP, constrained HRP, HERC, NCO, W-DRO, graph heuristic과 manifest의 사전등록 variant다. 출력의 `eff_n`, `div_ratio`, `max_w`와 가중치는 배분 진단일 뿐 성과나 승격 근거가 아니다. variant 결과를 보고 사후 cherry-pick하지 않는다. 정본의 primary는 single-linkage `hrp_dendrogram`이다.

### 5. QGA 가중치 동결

```bash
mkdir -p "$ART/frozen"

uv run python scripts/research/build_quality_gated_allocation.py \
  --input "$ART/selection/allocation_input.json" \
  --output "$ART/frozen/quality_gated_allocation.json"

sha256sum "$ART/frozen/quality_gated_allocation.json" \
  > "$ART/frozen/quality_gated_allocation.sha256"
```

이 명령은 공통 UTC 날짜 교집합의 비용 후 train+validation 순수익과 turnover만 사용해 품질게이트와 primary allocator를 실행한다. 생성된 `quality_gated_allocation.json`과 SHA-256을 잠그고 이후 수정하지 않는다.

### 6. 별도 locked-OOS PIT 유니버스와 event run

```bash
uv run python scripts/research/materialize_named_quant_universe.py \
  --suite "$ART/full_suite.json" \
  --market-caps "$MARKET_CAP_SNAPSHOTS" \
  --exchange-info "$EXCHANGE_INFO_SNAPSHOTS" \
  --as-of "$LOCK_AS_OF" \
  --crypto-top-n 10 \
  --output "$ART/locked_oos_universe.json"

LQ_CONFIG_PATH=configs/profiles/backtest_cost_realistic.yaml \
uv run python scripts/research/run_named_quant_suite.py \
  --manifest "$ART/locked_oos_universe.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$ART/locked_oos" \
  --exchange binance \
  --start "$LOCK_START" \
  --end "$LOCK_END" \
  --purpose locked_oos \
  --selection-artifact "$ART/selection/suite_results.json" \
  --warmup-bars 400
```

runner는 두 run의 suite id·기본 전략 spec·실효 runtime identity가 같은지 확인하되, selection과 locked-OOS의 PIT receipt가 다른 것은 허용하고 둘 다 lineage에 기록한다. `locked_oos/`에 `allocation_input.json`이 생기면 계약 위반이다. locked 시점에 PIT 유니버스 밖이 된 고정 페어는 skip되며, 그 후보가 동결 가중치의 양수 child라면 최종 evaluator가 fail-closed해야 한다.

### 7. 동결 가중치 전용 evaluator

```bash
uv run python scripts/research/evaluate_named_quant_locked_oos.py \
  --allocation-manifest "$ART/frozen/quality_gated_allocation.json" \
  --suite-results "$ART/locked_oos/suite_results.json" \
  --selection-artifact "$ART/selection/suite_results.json" \
  --output "$ART/locked_oos/frozen_weight_report.json" \
  --rebalance-every-observations 5 \
  --allocation-cost-bps 10 \
  --periods-per-year 252
```

마지막 세 값은 정본 `locked_oos_evaluation`의 사전등록 값이다. evaluator는 selection artifact의 SHA-256·기간·suite/runtime lineage와 동결 manifest의 결합을 먼저 확인한다. 이어 양수 가중치 child의 locked-OOS 순수익을 정확한 UTC timestamp 교집합으로 맞추고, 5개 관측마다 같은 동결 목표비중으로 재조정하면서 one-way turnover에 10 bps를 차감한다. 재최적화, 재스케일, 현금 수익, locked-OOS 선택은 없다. 누출 flag, 누락 child, 실패 child, cash/gross 불일치는 오류로 종료한다.

## 배분기 구현 경계

| 방법 | 현재 구현과 귀속 한계 |
|---|---|
| HRP dendrogram / constrained HRP | 상관거리·연결법·준대각화·재귀 이분; constrained는 명시적 box-simplex 제약을 fail-closed 적용 |
| HERC | 계층 ERC 구현. 군집 수는 원 논문의 gap statistic이 아니라 결정론적 silhouette 변형 |
| NCO | 군집 내·간 long-only 최적화 구현; manifest variant는 min-variance |
| Wasserstein DRO | BCZ p=q=2 mean-variance 식 구현. radius는 수익 단위 민감도이며 신뢰수준·보정값 주장이 아님 |
| Graph | full-graph inverse-centrality 위험 heuristic. Network Risk Parity 논문의 정확 재현 주장이 아님 |
| Deep/RL | 미구현·보류. 단순 배분기가 locked OOS에서 실패하고 별도 trial budget·누출 검사가 생기기 전에는 추가하지 않음 |

## 출처와 attribution 주의

정본 manifest의 `evidence_sources` 27행이 전체 레지스트리다. 핵심 링크는 다음과 같다.

- 물탄찬밥 공개 미리보기: [코인 전략](https://resource.newsystock.com/Admin/Academy/%28Preview%29_MultanChanbab_coin_strategy.pdf), [뼈대 전략](https://resource.newsystock.com/Admin/Academy/%28Preview%29%20Developing%20the%20basic%20strategy%20of%20Multanchanbap%2816%29.pdf)
- 시계열·횡단면 모멘텀: [Moskowitz–Ooi–Pedersen](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf), [Jegadeesh–Titman](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
- HRP/NCO/HERC: [HRP](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678), [NCO](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961), [HERC](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540)
- 강건·제약·graph 배분: [Wasserstein MV-DRO](https://doi.org/10.1287/mnsc.2021.4155), [constrained HRP](https://www.ekon.sun.ac.za/wpapers/2019/wp142019/wp142019.pdf), [Network Risk Parity](https://doi.org/10.1057/s41260-023-00347-8)
- 통계차익거래: [Avellaneda–Lee](https://doi.org/10.1080/14697680903124632), [Chan](https://doi.org/10.1002/9781118676998), [Triantafyllopoulos–Montana](https://doi.org/10.1007/s10287-009-0105-8)
- 한국 트레이더 자료: [systrader79 공개 글](https://stock79.tistory.com/entry/systrader79-%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%94%A9-%EB%A7%88%EC%8A%A4%ED%84%B0%ED%81%B4%EB%9E%98%EC%8A%A4-%ED%8C%A8%ED%82%A4%EC%A7%80-%EC%98%A4%ED%94%88), [AOA/워뇨띠 BitMEX 인터뷰](https://www.bitmex.com/blog/whale-trader-talks-aoa), [FlightF RSI 글](https://gall.dcinside.com/mgallery/board/view/?id=electronicmoney&no=548338), [FlightF 원칙 글](https://gall.dcinside.com/mgallery/board/view/?id=electronicmoney&no=187860), [돌파고 신원 자료](https://www.dogdrip.net/341084283)
- 보완 provenance: [`dacapogo` 고정 커밋](https://github.com/HokyoungJung/dacapogo/tree/633ba5d6bc0c84a20696af6b2bf807cf55d21248), [Binance USD-M 공식 문서](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/Introduction)

해석 한계는 고정한다.

- `systrader32`와 부동심은 신뢰 가능한 공개 재현 규칙을 확인하지 못해 귀속하지 않는다. `systrader79`를 동일인으로 간주하지 않는다.
- 알바트로스 자료는 자본관리·규율 근거이지 구체 alpha 식이 아니다. 아마추어퀀트 프로필은 `provenance_refs`/`evidence_only` 범위 기록일 뿐 규칙·세부 파라미터 근거가 아니다.
- AOA 인터뷰는 유동 메이저 선호와 위험관리만 지지한다. 전일 박스·사분위 규칙은 독립 프록시다.
- 돌파고 자료는 신원·대회 provenance이고 완전 주문식은 없다. `dacapogo`는 진단·proxy provenance일 뿐 규칙 근거가 아니다.
- FlightF 공개 글은 10분 RSI divergence·분할청산 맥락을 지지하지만 pivot, sizing, stop 수치는 독립 선택이다.
- 통합 manifest가 하위 레인의 `evidence_sources`를 verbatim 병합하므로 NCO/HERC/constrained-HRP/graph를 “미구현”으로 적은 일부 legacy notes가 남아 있다. **구현 상태는 현재 `src/lumina_quant/portfolio/hierarchical.py`, allocator tests, 이 실행서를 따른다.** 출처 행은 attribution 범위로만 읽는다.
